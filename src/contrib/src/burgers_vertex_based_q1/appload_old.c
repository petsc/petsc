

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
  u_val_ptr = &(equations->u_val);
  
 ierr = OptionsGetDouble(0,"-u_val", u_val_ptr, &flag); CHKERRQ(ierr);
 if(!flag) *u_val_ptr = 0.5;
 /*
 printf("u_val %f", equations->u_val); 
 printf("u_val %f", ((* appctx)->equations).u_val); 
 */

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

  BT                     vertex_boundary_flag;
  int                    *cell_vertex;
  double                 *vertex_value;
  int                    *vertices,cell_n,vertex_n_ghosted,*cell_cell;
  int  *vertex_indices;
  ISLocalToGlobalMapping ltogcell;

  int                    ierr,rstart,rend,rank;
  IS                     iscell,isvertex,vertex_boundary, isvertex_doubled;
  int i, vertex_size, *vertex_doubled;

  MPI_Comm_rank(appctx->comm,&rank);

  /*
      Generate the list of on processor cells 
  */
  ierr = AODataKeyGetOwnershipRange(ao,"cell",&rstart,&rend);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_WORLD,rend-rstart,rstart,1,&iscell);CHKERRQ(ierr);

  /*
       Get the list of vertices used by those cells 
  */
  ierr = AODataSegmentGetReducedIS(ao,"cell","vertex",iscell,&isvertex);CHKERRQ(ierr);

  /* 
      Make local to global mapping of cells and vertices
  */
  ierr = ISLocalToGlobalMappingCreateIS(iscell,&ltogcell);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreateIS(isvertex,&grid->ltog);CHKERRQ(ierr);

  /* make the extra index set needed for MatZeroRows */
  ierr = ISGetIndices(isvertex, &vertex_indices);CHKERRQ(ierr);
  ierr = ISGetSize(isvertex, &vertex_size);CHKERRQ(ierr);
  vertex_doubled = (int *) PetscMalloc(((2*vertex_size)+1)*sizeof(int));CHKPTRQ(vertex_doubled);
  for(i=0;i<vertex_size;i++){
    vertex_doubled[2*i] = 2*vertex_indices[i];
    vertex_doubled[2*i+1] =  2*vertex_indices[i] + 1;
  }
  ierr = ISCreateGeneral(PETSC_COMM_WORLD,2*vertex_size, vertex_doubled, &isvertex_doubled);
  ierr = ISLocalToGlobalMappingCreateIS(isvertex_doubled,&grid->dltog);CHKERRQ(ierr);

/*(  ierr = AODataKeySetLocalToGlobalMapping(ao,"blockedvertex", grid->dltog);CHKERRQ(ierr);*/

  ierr = AODataKeySetLocalToGlobalMapping(ao,"cell",ltogcell);CHKERRQ(ierr);
  ierr = AODataKeySetLocalToGlobalMapping(ao,"vertex",grid->ltog);CHKERRQ(ierr);

  ierr = PetscObjectDereference((PetscObject)ltogcell);CHKERRQ(ierr);

  /*
      Get the local edge and vertex lists
  */
  ierr = AODataSegmentGetLocalIS(ao,"cell","vertex",iscell,(void **)&cell_vertex);CHKERRQ(ierr);
  ierr = AODataSegmentGetIS(ao,"cell","cell",iscell,(void **)&cell_cell);CHKERRQ(ierr);

  /* 
      Get the numerical values of all vertices for local vertices
  */
  ierr = AODataSegmentGetIS(ao,"vertex","values",isvertex,(void **)&vertex_value);CHKERRQ(ierr);

  /*
      Get the size of local objects for plotting
  */
  ierr = ISGetSize(iscell,&cell_n); CHKERRQ(ierr);
  ierr = ISGetSize(isvertex,&vertex_n_ghosted); CHKERRQ(ierr);

  /* 
      Get the bit flag indicating boundary for local vertices
  */
  ierr = AODataSegmentGetIS(ao,"vertex","boundary",isvertex,(void **)&vertex_boundary_flag);CHKERRQ(ierr);

  /*
      Generate a list of local vertices that are on the boundary
  */
  ierr = ISGetIndices(isvertex,&vertices);CHKERRQ(ierr);
  ierr = AODataKeyGetActiveLocal(ao,"vertex","boundary",vertex_n_ghosted,vertices,0,&vertex_boundary);  CHKERRQ(ierr);
  ierr = ISRestoreIndices(isvertex,&vertices);CHKERRQ(ierr);

  /*  Now create a doubled IS for MatZeroRowsLocal */


  grid->cell_vertex          = cell_vertex; 
  grid->cell_global          = iscell;
  grid->vertex_doubled       = isvertex_doubled; /* my addition */
  grid->cell_n               = cell_n;
  grid->cell_cell            = cell_cell;

  grid->vertex_global        = isvertex;
  grid->vertex_value         = vertex_value;
  grid->vertex_boundary_flag = vertex_boundary_flag;
  grid->vertex_boundary      = vertex_boundary;
  grid->vertex_n_ghosted     = vertex_n_ghosted;

  ierr = AODataKeyGetInfo(ao,"vertex",PETSC_NULL,&grid->vertex_n,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = AODataSegmentGetInfo(ao,"cell","vertex",&grid->NVs,0);CHKERRQ(ierr);

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
  ierr = VecDestroy(appctx->algebra.b);CHKERRQ(ierr);
  ierr = VecDestroy(appctx->algebra.x);CHKERRQ(ierr);
  ierr = VecDestroy(appctx->algebra.z);CHKERRQ(ierr);
  ierr = VecDestroy(appctx->algebra.w_local);CHKERRQ(ierr);
  ierr = VecDestroy(appctx->algebra.x_local);CHKERRQ(ierr);
  ierr = VecDestroy(appctx->algebra.z_local);CHKERRQ(ierr);
  ierr = VecScatterDestroy(appctx->algebra.gtol);CHKERRQ(ierr);

  if (appctx->view.showsomething) {
    ierr = DrawDestroy(appctx->view.drawglobal); CHKERRQ(ierr);
    ierr = DrawDestroy(appctx->view.drawlocal); CHKERRQ(ierr);
  }

  ierr = ISDestroy(appctx->grid.vertex_global);CHKERRQ(ierr);
  ierr = ISDestroy(appctx->grid.vertex_boundary);CHKERRQ(ierr);
  ierr = ISDestroy(appctx->grid.cell_global);CHKERRQ(ierr);

  ierr = ISLocalToGlobalMappingDestroy(appctx->grid.ltog);CHKERRQ(ierr);
 ierr = ISLocalToGlobalMappingDestroy(appctx->grid.dltog);CHKERRQ(ierr);

  PetscFree(appctx);
  PetscFunctionReturn(0);
}





