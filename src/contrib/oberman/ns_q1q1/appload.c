/*
     Loads the qquadrilateral grid database from a file  and sets up the local 
     data structures. 
*/

#include "appctx.h"

#undef __FUNC__
#define __FUNC__ "AppCxtCreate"
int AppCtxCreate(MPI_Comm comm,AppCtx **appctx)
{
  int    ierr,flag;
  Viewer binary;
  char   filename[256];
  double maxs[2],mins[2],xmin,xmax,ymin,ymax,hx,hy;
  AppEquations *equations;
  double *u_val_ptr;

  (*appctx) = (AppCtx *) PetscMalloc(sizeof(AppCtx));CHKPTRQ(*appctx);
  (*appctx)->comm = comm;

/*  ---------------
      Setup the functions
-------------------*/

  equations = &(*appctx)->equations;
 
  /*-----------------------------------------------------------------------
     Load in the grid database
    ---------------------------------------------------------------------------*/
  ierr = OptionsGetString(0,"-f",filename,256,&flag);CHKERRQ(ierr);
  if (!flag) PetscStrcpy(filename,"gridfile");
  ierr = ViewerFileOpenBinary((*appctx)->comm,filename,BINARY_RDONLY,&binary);CHKERRQ(ierr);
  ierr = AODataLoadBasic(binary,&(*appctx)->aodata); CHKERRQ(ierr);
  ierr = ViewerDestroy(binary); CHKERRQ(ierr);


  /*------------------------------------------------------------------------
      Setup the local data structures 
      ----------------------------------------------------------------------------*/
  /*
      Partition the grid cells
  */
  ierr = AODataKeyPartition((*appctx)->aodata,"cell"); CHKERRA(ierr);  

  /*
      Partition the vertices subservient to the cells
  */ 
  ierr = AODataSegmentPartition((*appctx)->aodata,"cell","vertex"); CHKERRA(ierr);  

  /*
     Generate the local numbering of cells and vertices
  */
  ierr = AppCtxSetLocal(*appctx); CHKERRA(ierr);


  PetscFunctionReturn(0);
}

#undef __FUNC__
#define __FUNC__ "AppCxtGraphics"
int AppCtxGraphics(AppCtx *appctx)
{
  int    ierr,flag;
  Viewer binary;
  char   filename[256];
  double maxs[2],mins[2],xmin,xmax,ymin,ymax,hx,hy;

  /*---------------------------------------------------------------------
     Setup  the graphics windows
     ------------------------------------------------------------------------*/

  ierr = OptionsHasName(PETSC_NULL,"-show_numbers",&appctx->view.shownumbers);CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-show_elements",&appctx->view.showelements);CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-show_vertices",&appctx->view.showvertices);CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-show_boundary",&appctx->view.showboundary);CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-show_boundary_vertices",
         &appctx->view.showboundaryvertices);CHKERRQ(ierr);

  (appctx)->view.showsomething = (appctx)->view.shownumbers + (appctx)->view.showelements + 
                                 (appctx)->view.showvertices + (appctx)->view.showboundary +
                                 (appctx)->view.showboundaryvertices;

  if ((appctx)->view.showsomething) {
    ierr = DrawOpenX(PETSC_COMM_WORLD,PETSC_NULL,"Total Grid",PETSC_DECIDE,PETSC_DECIDE,400,400,
                     &appctx->view.drawglobal); CHKERRQ(ierr);
    ierr = DrawOpenX(PETSC_COMM_WORLD,PETSC_NULL,"Local Grids",PETSC_DECIDE,PETSC_DECIDE,400,400,
                     &appctx->view.drawlocal);CHKERRQ(ierr);
    ierr = DrawSplitViewPort((appctx)->view.drawlocal);CHKERRQ(ierr);

    /*
       Set the window coordinates based on the values in vertices
    */
    ierr = AODataSegmentGetExtrema((appctx)->aodata,"vertex","values",maxs,mins);CHKERRQ(ierr);
    hx = maxs[0] - mins[0]; xmin = mins[0] - .1*hx; xmax = maxs[0] + .1*hx;
    hy = maxs[1] - mins[1]; ymin = mins[1] - .1*hy; ymax = maxs[1] + .1*hy;
    ierr = DrawSetCoordinates((appctx)->view.drawglobal,xmin,ymin,xmax,ymax);CHKERRQ(ierr);
    ierr = DrawSetCoordinates((appctx)->view.drawlocal,xmin,ymin,xmax,ymax);CHKERRQ(ierr);
    /*
       Visualize the grid 
    */
    ierr = DrawZoom((appctx)->view.drawglobal,AppCtxView,appctx); CHKERRA(ierr);
  }
  ierr = OptionsHasName(PETSC_NULL,"-matlab_graphics",&(appctx)->view.matlabgraphics); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNC__
#define __FUNC__ "AppCxtSetLocal"
/*
     AppCtxSetLocal - Sets the local numbering data structures for the grid.

*/
int AppCtxSetLocal(AppCtx *appctx)
{
  AOData                 ao     = appctx->aodata;
  AppGrid                *grid = &appctx->grid;
  BT                     vertex_boundary_flag, inlet_boundary_flag,  outlet_boundary_flag, wall_boundary_flag;
  ISLocalToGlobalMapping ltogcell;
  int                    i, ierr, rstart, rend, rank;

  
  /* arrays to create for AppGrid */
  int *cell_cell;
  int  *cell_vertex;
  double  *vertex_value;
  /* constants for AppGrid */
  int cell_n;
  int vertex_n_ghosted;
/* index sets to create for  AppGrid */
  IS  cell_global, isvertex_global, vertex_boundary, boundary_wall, boundary_inlet, boundary_outlet;
  IS  isvertex_global_blocked, isvertex_boundary_blocked, isvertex_wall_blocked, isvertex_inlet_blocked, isvertex_outlet_blocked;
  /* Local to global mapping to create for AppGrid */
  /*  ltog, dltog - these are created as calls to &grid->ltog etc*/

  /* local variables */
  int    *vertices;
  int  *vertex_indices;
  int  vertex_size, vertex_boundary_size, vertex_wall_size, vertex_inlet_size, vertex_outlet_size;
  int  *vertex_blocked, *vertex_boundary_blocked, *vertex_wall_blocked, *vertex_inlet_blocked, *vertex_outlet_blocked;

  MPI_Comm_rank(appctx->comm,&rank);

  /*      Generate the list of on processor cells */  
  ierr = AODataKeyGetOwnershipRange(ao,"cell",&rstart,&rend);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_WORLD,rend-rstart,rstart,1,&cell_global);CHKERRQ(ierr);

  /*       Get the list of vertices used by those cells  */
  ierr = AODataSegmentGetReducedIS(ao,"cell","vertex",cell_global,&isvertex_global);CHKERRQ(ierr);

  /*      Make local to global mapping of cells and vertices  */
  ierr = ISLocalToGlobalMappingCreateIS(cell_global,&ltogcell);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreateIS(isvertex_global,&grid->ltog);CHKERRQ(ierr);

 /* create a blocked version of the isvertex_global (will be vertex_global)*/
  /* make the extra index set needed for MatZeroRows */
  ierr = ISGetIndices(isvertex_global, &vertex_indices);CHKERRQ(ierr);
  ierr = ISGetSize(isvertex_global, &vertex_size);CHKERRQ(ierr);
  vertex_blocked = (int *) PetscMalloc(((DF*vertex_size)+1)*sizeof(int));CHKPTRQ(vertex_blocked);
#if DF == 3
    for(i=0;i<vertex_size;i++){
      vertex_blocked[DF*i] = DF*vertex_indices[i];
      vertex_blocked[DF*i+1] =  DF*vertex_indices[i] + 1;
      vertex_blocked[DF*i+2] =  DF*vertex_indices[i] + 2;
    }
#elif DF == 2
     for(i=0;i<vertex_size;i++){
      vertex_blocked[DF*i] = DF*vertex_indices[i];
      vertex_blocked[DF*i+1] =  DF*vertex_indices[i] + 1;
     }
#endif
     /* Create A Local to Global mapping for the blocked index set. */
  ierr = ISCreateGeneral(PETSC_COMM_WORLD,DF*vertex_size, vertex_blocked, &isvertex_global_blocked);
  ierr = ISLocalToGlobalMappingCreateIS(isvertex_global_blocked,&grid->dltog);CHKERRQ(ierr);

  ierr = AODataKeySetLocalToGlobalMapping(ao,"cell",ltogcell);CHKERRQ(ierr);
  ierr = AODataKeySetLocalToGlobalMapping(ao,"vertex",grid->ltog);CHKERRQ(ierr);

  ierr = PetscObjectDereference((PetscObject)ltogcell);CHKERRQ(ierr);

  /*      Get the local edge and vertex lists*/
  ierr = AODataSegmentGetLocalIS(ao,"cell","vertex",cell_global,(void **)&cell_vertex);CHKERRQ(ierr);
  ierr = AODataSegmentGetIS(ao,"cell","cell",cell_global,(void **)&cell_cell);CHKERRQ(ierr);

  /*       Get the numerical values of all vertices for local vertices  */
  ierr = AODataSegmentGetIS(ao,"vertex","values",isvertex_global,(void **)&vertex_value);CHKERRQ(ierr);

  /*      Get the size of local objects for plotting  */
  ierr = ISGetSize(cell_global,&cell_n); CHKERRQ(ierr);
  ierr = ISGetSize(isvertex_global,&vertex_n_ghosted); CHKERRQ(ierr);

  /*       Get the bit flag indicating boundary for local vertices */
  ierr = AODataSegmentGetIS(ao,"vertex","boundary",isvertex_global,(void **)&vertex_boundary_flag);CHKERRQ(ierr);
  /*       Get the bit flag indicating boundary for wall  vertices */
 /*  ierr = AODataSegmentGetIS(ao,"vertex","boundary_wall",isvertex_global,(void **)&wall_boundary_flag);CHKERRQ(ierr); */
/*       Get the bit flag indicating boundary for outlet  vertices */
 /*  ierr = AODataSegmentGetIS(ao,"vertex","boundary_outlet",isvertex_global, (void **)&outlet_boundary_flag);CHKERRQ(ierr); */
  /*       Get the bit flag indicating boundary for inlet  vertices */
/*   ierr = AODataSegmentGetIS(ao,"vertex","boundary_inlet",isvertex_global,(void **)&inlet_boundary_flag);CHKERRQ(ierr); */


  /*     Generate a list of local vertices that are on the boundary */
  ierr = ISGetIndices(isvertex_global,&vertices);CHKERRQ(ierr);
  ierr = AODataKeyGetActiveLocal(ao,"vertex","boundary",vertex_n_ghosted,vertices,0,&vertex_boundary);  CHKERRQ(ierr);
 ierr = AODataKeyGetActiveLocal(ao,"vertex","boundary_wall",vertex_n_ghosted,vertices,0,&boundary_wall);  CHKERRQ(ierr);
 ierr = AODataKeyGetActiveLocal(ao,"vertex","boundary_inlet",vertex_n_ghosted,vertices,0,&boundary_inlet);  CHKERRQ(ierr);
  ierr = AODataKeyGetActiveLocal(ao,"vertex","boundary_outlet",vertex_n_ghosted,vertices,0,&boundary_outlet);  CHKERRQ(ierr);
 ierr = ISRestoreIndices(isvertex_global,&vertices);CHKERRQ(ierr);
printf("boundary wall IS");
ISView(boundary_wall, VIEWER_STDOUT_SELF);
printf("boundary inlet IS");
ISView(boundary_inlet,  VIEWER_STDOUT_SELF);
printf("boundaryoutletIS");
ISView(boundary_outlet,  VIEWER_STDOUT_SELF);

  /* Create a blocked IS for the wall boundary, which contains u,v but skips p 
  At a wall boundary conditions  are hard for u,v but soft for p */
  ierr = ISGetIndices(boundary_wall, &vertex_indices);CHKERRQ(ierr);
  ierr = ISGetSize(boundary_wall, &vertex_wall_size); CHKERRQ(ierr); 
  vertex_wall_blocked = (int *) PetscMalloc((2*vertex_wall_size)*sizeof(int));CHKPTRQ(vertex_wall_blocked); 
 for(i=0;i<vertex_wall_size;i++){ 
      vertex_wall_blocked[2*i] = 3*vertex_indices[i]; 
      vertex_wall_blocked[2*i+1] = 3*vertex_indices[i] + 1; 
      /* Skip the pressure */
    }
  ierr = ISCreateGeneral(PETSC_COMM_WORLD,2*vertex_wall_size, vertex_wall_blocked, &isvertex_wall_blocked); 
  ierr = PetscFree(vertex_wall_blocked);CHKERRQ(ierr);
  /* Could reuse the array */
   
  /* Create a blocked IS for the inletl boundary, which contains u,v but skips p 
  At an inlet  boundary conditions  are hard for u,v but soft for p */
  ierr = ISGetIndices(boundary_inlet, &vertex_indices);CHKERRQ(ierr);
  ierr = ISGetSize(boundary_inlet, &vertex_inlet_size); CHKERRQ(ierr); 
  vertex_inlet_blocked = (int *) PetscMalloc((2*vertex_inlet_size)*sizeof(int));CHKPTRQ(vertex_inlet_blocked); 
  for(i=0;i<vertex_inlet_size;i++){ 
      vertex_inlet_blocked[2*i] = 3*vertex_indices[i]; 
      vertex_inlet_blocked[2*i+1] = 3*vertex_indices[i] + 1; 
      /* Skip the pressure */
    }
  ierr = ISCreateGeneral(PETSC_COMM_WORLD,2*vertex_inlet_size, vertex_inlet_blocked, &isvertex_inlet_blocked); 
  ierr = PetscFree(vertex_inlet_blocked);CHKERRQ(ierr);
  /* Could reuse the array */

 /* Create a blocked IS for the outlet boundary, 
At the outlet, we set pressure = 0 , and then du/dn = 0 naturally  */
  ierr = ISGetIndices(boundary_outlet, &vertex_indices);CHKERRQ(ierr);
  ierr = ISGetSize(boundary_outlet, &vertex_outlet_size); CHKERRQ(ierr); 
  vertex_outlet_blocked = (int *) PetscMalloc((vertex_outlet_size)*sizeof(int));CHKPTRQ(vertex_outlet_blocked); 
 for(i=0;i<vertex_outlet_size;i++){ 
      vertex_outlet_blocked[i] = 3*vertex_indices[i] + 2;  /* index of the pressure */
    }
  ierr = ISCreateGeneral(PETSC_COMM_WORLD,vertex_outlet_size, vertex_outlet_blocked, &isvertex_outlet_blocked); 
  ierr = PetscFree(vertex_outlet_blocked);CHKERRQ(ierr);

   /* Make the assignments   */ 
   grid->cell_vertex          = cell_vertex; 
   grid->cell_global          = cell_global;
  grid->vertex_global_blocked       = isvertex_global_blocked; /* my addition */
   grid->vertex_boundary_blocked = isvertex_boundary_blocked; /* my addition */
   grid->vertex_wall_blocked = isvertex_wall_blocked; /* my addition */
   grid->vertex_inlet_blocked = isvertex_inlet_blocked; /* my addition */
 grid->vertex_outlet_blocked = isvertex_outlet_blocked; /* my addition */
/*    ISView(isvertex_wall_blocked, VIEWER_STDOUT_SELF); */
/*   ISView(isvertex_open_blocked, VIEWER_STDOUT_SELF); */
  
   grid->cell_n               = cell_n;
   grid->cell_cell            = cell_cell;

  grid->vertex_global        = isvertex_global;
  grid->vertex_value         = vertex_value;

  grid->vertex_boundary_flag = vertex_boundary_flag;
  grid->inlet_boundary_flag = inlet_boundary_flag;
  grid->outlet_boundary_flag = outlet_boundary_flag;
  grid->wall_boundary_flag = wall_boundary_flag;
  
  grid->vertex_boundary      = vertex_boundary;
  grid->boundary_inlet    = boundary_inlet;
  grid->boundary_outlet    = boundary_outlet;
  grid->boundary_wall      = boundary_wall;


  grid->vertex_n_ghosted     = vertex_n_ghosted;

  ierr = AODataKeyGetInfo(ao,"vertex",PETSC_NULL,&grid->vertex_n,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = AODataSegmentGetInfo(ao,"cell","vertex",&grid->NV,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__
#define __FUNC__ "AppCxtDestroy"
int AppCtxDestroy(AppCtx *appctx)
{
  int        ierr;
  AOData     ao = appctx->aodata;
  AppGrid    *grid = &appctx->grid;
  AppAlgebra *algebra = &appctx->algebra;

  ierr = AODataSegmentRestoreIS(ao,"vertex","values",PETSC_NULL,(void **)&grid->vertex_value);CHKERRQ(ierr);
  ierr = AODataSegmentRestoreLocalIS(ao,"cell","vertex",PETSC_NULL,(void **)&grid->cell_vertex);CHKERRQ(ierr);
  ierr = AODataSegmentRestoreIS(ao,"cell","cell",PETSC_NULL,(void **)&grid->cell_cell);CHKERRQ(ierr);
  ierr = AODataSegmentRestoreIS(ao,"vertex","boundary",PETSC_NULL,(void **)&grid->vertex_boundary_flag);CHKERRQ(ierr);
  ierr = AODataDestroy(ao);CHKERRQ(ierr);

  /*
      Free the algebra 
  */
  ierr = MatDestroy(appctx->algebra.A);CHKERRQ(ierr);
  ierr = MatDestroy(appctx->algebra.J);CHKERRQ(ierr);

  ierr = VecDestroy(appctx->algebra.b);CHKERRQ(ierr);
  ierr = VecDestroy(appctx->algebra.g);CHKERRQ(ierr);
  ierr = VecDestroy(appctx->algebra.f);CHKERRQ(ierr);
  ierr = VecDestroy(appctx->algebra.f_local);CHKERRQ(ierr);
  /* Need to destroy the array of vectors for the solution */
  ierr = VecDestroyVecs(appctx->algebra.solnv, NSTEPS + 1);CHKERRQ(ierr);

  ierr = VecDestroy(appctx->algebra.x);CHKERRQ(ierr);
  ierr = VecDestroy(appctx->algebra.z);CHKERRQ(ierr);
  ierr = VecDestroy(appctx->algebra.w_local);CHKERRQ(ierr);
  ierr = VecDestroy(appctx->algebra.x_local);CHKERRQ(ierr);
  ierr = VecDestroy(appctx->algebra.z_local);CHKERRQ(ierr);
  ierr = VecScatterDestroy(appctx->algebra.gtol);CHKERRQ(ierr);
  ierr = VecScatterDestroy(appctx->algebra.dgtol);CHKERRQ(ierr);

  if (appctx->view.showsomething) {
    ierr = DrawDestroy(appctx->view.drawglobal); CHKERRQ(ierr);
    ierr = DrawDestroy(appctx->view.drawlocal); CHKERRQ(ierr);
  }

  ierr = ISDestroy(appctx->grid.vertex_global);CHKERRQ(ierr);
  ierr = ISDestroy(appctx->grid.vertex_global_blocked);CHKERRQ(ierr);
  ierr = ISDestroy(appctx->grid.vertex_boundary);CHKERRQ(ierr);
  ierr = ISDestroy(appctx->grid.vertex_boundary_blocked);CHKERRQ(ierr);
  ierr = ISDestroy(appctx->grid.cell_global);CHKERRQ(ierr);

  ierr = ISLocalToGlobalMappingDestroy(appctx->grid.ltog);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(appctx->grid.dltog);CHKERRQ(ierr);

  ierr = PetscFree(appctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}





