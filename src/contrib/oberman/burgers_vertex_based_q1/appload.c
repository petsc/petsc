
/*
     Loads the qquadrilateral grid database from a file  and sets up the local 
     data structures. 
*/

#include "appctx.h"

#undef __FUNCT__
#define __FUNCT__ "AppCxtCreate"
PetscErrorCode AppCtxCreate(MPI_Comm comm,AppCtx **appctx)
{
  PetscErrorCode         ierr;
  PetscTruth  flag;
  PetscViewer binary;
  char        filename[PETSC_MAX_PATH_LEN];

  ierr = PetscMalloc(sizeof(AppCtx),appctx);CHKERRQ(ierr);
  (*appctx)->comm = comm;

/*  ---------------
      Setup the functions
-------------------*/


  /*-----------------------------------------------------------------------
     Load in the grid database
    ---------------------------------------------------------------------------*/
  ierr = PetscOptionsGetString(0,"-f",filename,256,&flag);CHKERRQ(ierr);
  if (!flag) PetscStrcpy(filename,"gridfile");
  ierr = PetscViewerBinaryOpen((*appctx)->comm,filename,FILE_MODE_READ,&binary);CHKERRQ(ierr);
  ierr = AODataLoadBasic(binary,&(*appctx)->aodata);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(binary);CHKERRQ(ierr);


  /*------------------------------------------------------------------------
      Setup the local data structures 
      ----------------------------------------------------------------------------*/

  /*
     Generate the local numbering of cells and vertices
  */
  ierr = AppCtxSetLocal(*appctx);CHKERRQ(ierr);


  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "AppCxtSetLocal"
/*
     AppCtxSetLocal - Sets the local numbering data structures for the grid.

*/
PetscErrorCode AppCtxSetLocal(AppCtx *appctx)
{
  AOData                 ao     = appctx->aodata;
  AppGrid                *grid = &appctx->grid;

  PetscErrorCode                    ierr;
  PetscMPIInt rank;

  PetscInt                    *cell_vertex;
  double                 *vertex_coords;
  PetscInt                    cell_n,vertex_n,*cell_neighbors;
  PetscInt  *vertex_indices;
  IS                     iscell,isvertex,vertex_boundary;

  IS  isvertex_global_blocked,isvertex_boundary_blocked;
  PetscInt i,vertex_size,vertex_boundary_size,*vertex_blocked,*vertex_boundary_blocked;

  MPI_Comm_rank(appctx->comm,&rank);


  /* get the local vertices, and the local to global mapping */
  ierr = AODataPartitionAndSetupLocal(ao,"cell","vertex",&iscell,&isvertex,&grid->ltog);CHKERRQ(ierr);

 
 /* create a blocked version of the isvertex (will be vertex_global)*/
  /* make the extra index set needed for MatZeroRows */
  ierr = ISGetIndices(isvertex,&vertex_indices);CHKERRQ(ierr);
  ierr = ISGetLocalSize(isvertex,&vertex_size);CHKERRQ(ierr);
  ierr = PetscMalloc(((2*vertex_size)+1)*sizeof(int),&vertex_blocked);CHKERRQ(ierr);
  for(i=0;i<vertex_size;i++){
    vertex_blocked[2*i] = 2*vertex_indices[i];
    vertex_blocked[2*i+1] =  2*vertex_indices[i] + 1;
  }
  ierr = ISCreateGeneral(PETSC_COMM_WORLD,2*vertex_size,vertex_blocked,&isvertex_global_blocked);
  ierr = PetscFree(vertex_blocked);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreateIS(isvertex_global_blocked,&grid->dltog);CHKERRQ(ierr);


  /*
      Get the local edge and vertex lists
  */
  ierr = AODataSegmentGetLocalIS(ao,"cell","vertex",iscell,(void **)&cell_vertex);CHKERRQ(ierr);
  ierr = AODataSegmentGetIS(ao,"cell","cell",iscell,(void **)&cell_neighbors);CHKERRQ(ierr);

  /* 
      Get the numerical values of all vertices for local vertices
  */
  ierr = AODataSegmentGetIS(ao,"vertex","values",isvertex,(void **)&vertex_coords);CHKERRQ(ierr);

  /*
      Get the size of local objects for plotting
  */
  ierr = ISGetLocalSize(iscell,&cell_n);CHKERRQ(ierr);
  ierr = ISGetLocalSize(isvertex,&vertex_n);CHKERRQ(ierr);

  ierr = AODataSegmentGetIS(ao,"vertex","boundary",isvertex,(void **)&grid->vertex_boundary_flag);CHKERRQ(ierr);

 /*      Generate a list of local vertices that are on the boundary */
  ierr = AODataKeyGetActiveLocalIS(ao,"vertex","boundary",isvertex,0,&vertex_boundary);CHKERRQ(ierr);

  /*  Now create a blocked IS for MatZeroRowsLocal */
  ierr = ISGetIndices(vertex_boundary,&vertex_indices);CHKERRQ(ierr);
  ierr = ISGetLocalSize(vertex_boundary,&vertex_boundary_size);CHKERRQ(ierr); 
  ierr = PetscMalloc((2*vertex_boundary_size)*sizeof(int),&vertex_boundary_blocked);CHKERRQ(ierr); 
  for(i=0;i<vertex_boundary_size;i++){ 
     vertex_boundary_blocked[2*i] = 2*vertex_indices[i]; 
     vertex_boundary_blocked[2*i+1] = 2*vertex_indices[i] + 1; 
   } 
   ierr = ISCreateGeneral(PETSC_COMM_WORLD,2*vertex_boundary_size,vertex_boundary_blocked,&isvertex_boundary_blocked); 
  ierr = PetscFree(vertex_boundary_blocked);CHKERRQ(ierr);

  grid->cell_vertex          = cell_vertex; 
  grid->cell_global          = iscell;
  grid->vertex_global_blocked       = isvertex_global_blocked; /* my addition */
  grid->vertex_boundary_blocked = isvertex_boundary_blocked; /* my addition */
  grid->cell_n               = cell_n;
  grid->cell_neighbors            = cell_neighbors;

  grid->vertex_global        = isvertex;
  grid->vertex_coords         = vertex_coords;

  grid->vertex_boundary      = vertex_boundary;
  grid->vertex_n     = vertex_n;

  ierr = AODataKeyGetInfo(ao,"vertex",PETSC_NULL,&grid->vertex_local_n,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = AODataSegmentGetInfo(ao,"cell","vertex",&grid->NVs,0);CHKERRQ(ierr);

  PetscFunctionReturn(0);

}


#undef __FUNCT__
#define __FUNCT__ "AppCxtGraphics"
PetscErrorCode AppCtxGraphics(AppCtx *appctx)
{
  PetscErrorCode    ierr;
  double maxs[2],mins[2],xmin,xmax,ymin,ymax,hx,hy;

  /*---------------------------------------------------------------------
     Setup  the graphics windows
     ------------------------------------------------------------------------*/

  ierr = PetscOptionsHasName(PETSC_NULL,"-show_grid",&appctx->view.show_grid);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-show_solution",&appctx->view.show_solution);CHKERRQ(ierr);

  if (appctx->view.show_grid || appctx->view.show_solution) {
    ierr = PetscDrawCreate(PETSC_COMM_WORLD,PETSC_NULL,"Total Grid",PETSC_DECIDE,PETSC_DECIDE,400,400,
                     &appctx->view.drawglobal);CHKERRQ(ierr);
    ierr = PetscDrawSetFromOptions(appctx->view.drawglobal);CHKERRQ(ierr);

    ierr = PetscDrawCreate(PETSC_COMM_WORLD,PETSC_NULL,"Local Grids",PETSC_DECIDE,PETSC_DECIDE,400,400,
                     &appctx->view.drawlocal);CHKERRQ(ierr);
    ierr = PetscDrawSetFromOptions(appctx->view.drawlocal);CHKERRQ(ierr);
    ierr = PetscDrawSplitViewPort((appctx)->view.drawlocal);CHKERRQ(ierr);

    /*
       Set the window coordinates based on the values in vertices
    */
    ierr = AODataSegmentGetExtrema((appctx)->aodata,"vertex","values",maxs,mins);CHKERRQ(ierr);
    hx = maxs[0] - mins[0]; xmin = mins[0] - .1*hx; xmax = maxs[0] + .1*hx;
    hy = maxs[1] - mins[1]; ymin = mins[1] - .1*hy; ymax = maxs[1] + .1*hy;
    ierr = PetscDrawSetCoordinates((appctx)->view.drawglobal,xmin,ymin,xmax,ymax);CHKERRQ(ierr);
    ierr = PetscDrawSetCoordinates((appctx)->view.drawlocal,xmin,ymin,xmax,ymax);CHKERRQ(ierr);
    /*
       Visualize the grid 
    */
  }

  if (appctx->view.show_grid) {
    ierr = PetscDrawZoom((appctx)->view.drawglobal,AppCtxView,appctx);CHKERRQ(ierr);
  }
  ierr = PetscOptionsHasName(PETSC_NULL,"-matlab_graphics",&(appctx)->view.matlabgraphics);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "AppCxtDestroy"
PetscErrorCode AppCtxDestroy(AppCtx *appctx)
{
  PetscErrorCode        ierr;
  AOData     ao = appctx->aodata;
  AppGrid    *grid = &appctx->grid;

  ierr = AODataSegmentRestoreIS(ao,"vertex","values",PETSC_NULL,(void **)&grid->vertex_coords);CHKERRQ(ierr);
  ierr = AODataSegmentRestoreLocalIS(ao,"cell","vertex",PETSC_NULL,(void **)&grid->cell_vertex);CHKERRQ(ierr);
  ierr = AODataSegmentRestoreIS(ao,"cell","cell",PETSC_NULL,(void **)&grid->cell_neighbors);CHKERRQ(ierr);
  ierr = AODataSegmentRestoreIS(ao,"vertex","boundary",PETSC_NULL,(void **)&grid->vertex_boundary_flag);CHKERRQ(ierr);
 
  ierr = AODataDestroy(ao);CHKERRQ(ierr);

  /*
      Free the algebra 
  */
  ierr = MatDestroy(appctx->algebra.A);CHKERRQ(ierr);
  ierr = MatDestroy(appctx->algebra.J);CHKERRQ(ierr);
  ierr = VecDestroy(appctx->algebra.b);CHKERRQ(ierr);
  ierr = VecDestroy(appctx->algebra.x);CHKERRQ(ierr);
  ierr = VecDestroy(appctx->algebra.z);CHKERRQ(ierr);
  ierr = VecDestroy(appctx->algebra.f);CHKERRQ(ierr);
  ierr = VecDestroy(appctx->algebra.g);CHKERRQ(ierr);
  ierr = VecDestroy(appctx->algebra.w_local);CHKERRQ(ierr);
  ierr = VecDestroy(appctx->algebra.x_local);CHKERRQ(ierr);
  ierr = VecDestroy(appctx->algebra.z_local);CHKERRQ(ierr);
  ierr = VecDestroy(appctx->algebra.f_local);CHKERRQ(ierr);
  ierr = VecScatterDestroy(appctx->algebra.dgtol);CHKERRQ(ierr);
  ierr = VecScatterDestroy(appctx->algebra.gtol);CHKERRQ(ierr);

  if (appctx->view.show_grid || appctx->view.show_solution) {
    ierr = PetscDrawDestroy(appctx->view.drawglobal);CHKERRQ(ierr);
    ierr = PetscDrawDestroy(appctx->view.drawlocal);CHKERRQ(ierr);
  }

  ierr = ISDestroy(appctx->grid.vertex_global);CHKERRQ(ierr);
  ierr = ISDestroy(appctx->grid.vertex_boundary);CHKERRQ(ierr);
  ierr = ISDestroy(appctx->grid.vertex_boundary_blocked);CHKERRQ(ierr);
  ierr = ISDestroy(appctx->grid.cell_global);CHKERRQ(ierr);
  ierr = ISDestroy(appctx->grid.vertex_global_blocked);CHKERRQ(ierr);

  ierr = ISLocalToGlobalMappingDestroy(appctx->grid.ltog);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(appctx->grid.dltog);CHKERRQ(ierr);

  ierr = PetscFree(appctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}





