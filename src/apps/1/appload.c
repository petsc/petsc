

/*
       Demonstrates how to load a database from a file and set up the local 
     data structures. See appctx.h for information on how this is to be used.
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

  (*appctx) = (AppCtx *) PetscMalloc(sizeof(AppCtx));CHKPTRQ(*appctx);
  (*appctx)->comm = comm;

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


  /*---------------------------------------------------------------------
     Setup  the graphics windows
     ------------------------------------------------------------------------*/

  ierr = OptionsHasName(PETSC_NULL,"-show_numbers",&(*appctx)->view.shownumbers);CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-show_elements",&(*appctx)->view.showelements);CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-show_vertices",&(*appctx)->view.showvertices);CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-show_boundary",&(*appctx)->view.showboundary);CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-show_boundary_vertices",
         &(*appctx)->view.showboundaryvertices);CHKERRQ(ierr);

  (*appctx)->view.showsomething = (*appctx)->view.shownumbers + (*appctx)->view.showelements + 
                                 (*appctx)->view.showvertices + (*appctx)->view.showboundary +
                                 (*appctx)->view.showboundaryvertices;

  if ((*appctx)->view.showsomething) {
    ierr = DrawOpenX(PETSC_COMM_WORLD,PETSC_NULL,"Total Grid",PETSC_DECIDE,PETSC_DECIDE,400,400,
                     &(*appctx)->view.drawglobal); CHKERRQ(ierr);
    ierr = DrawOpenX(PETSC_COMM_WORLD,PETSC_NULL,"Local Grids",PETSC_DECIDE,PETSC_DECIDE,400,400,
                     &(*appctx)->view.drawlocal);CHKERRQ(ierr);
    ierr = DrawSplitViewPort((*appctx)->view.drawlocal);CHKERRQ(ierr);

    /*
       Set the window coordinates based on the values in vertices
    */
    ierr = AODataSegmentGetExtrema((*appctx)->aodata,"vertex","values",maxs,mins);CHKERRQ(ierr);
    hx = maxs[0] - mins[0]; xmin = mins[0] - .1*hx; xmax = maxs[0] + .1*hx;
    hy = maxs[1] - mins[1]; ymin = mins[1] - .1*hy; ymax = maxs[1] + .1*hy;
    ierr = DrawSetCoordinates((*appctx)->view.drawglobal,xmin,ymin,xmax,ymax);CHKERRQ(ierr);
    ierr = DrawSetCoordinates((*appctx)->view.drawlocal,xmin,ymin,xmax,ymax);CHKERRQ(ierr);
    /*
       Visualize the grid 
    */
    ierr = DrawZoom((*appctx)->view.drawglobal,AppCtxView,*appctx); CHKERRA(ierr);
  }

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

  ISLocalToGlobalMapping ltogcell;

  int                    ierr,rstart,rend,rank;
  IS                     iscell,isvertex,vertex_boundary;

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
  ierr = AODataKeyGetActiveLocal(ao,"vertex","boundary",vertex_n_ghosted,vertices,0,&vertex_boundary);
  CHKERRQ(ierr);
  ierr = ISRestoreIndices(isvertex,&vertices);CHKERRQ(ierr);

  grid->cell_vertex          = cell_vertex; 
  grid->cell_global          = iscell;
  grid->cell_n               = cell_n;
  grid->cell_cell            = cell_cell;

  grid->vertex_global        = isvertex;
  grid->vertex_value         = vertex_value;
  grid->vertex_boundary_flag = vertex_boundary_flag;
  grid->vertex_boundary      = vertex_boundary;
  grid->vertex_n_ghosted     = vertex_n_ghosted;

  ierr = AODataKeyGetInfo(ao,"vertex",PETSC_NULL,&grid->vertex_n,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = AODataSegmentGetInfo(ao,"cell","vertex",&grid->ncell,0);CHKERRQ(ierr);

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
  ierr = AODataDestroy(ao);CHKERRQ(ierr);

  if (appctx->view.showsomething) {
    ierr = DrawDestroy(appctx->view.drawglobal); CHKERRQ(ierr);
    ierr = DrawDestroy(appctx->view.drawlocal); CHKERRQ(ierr);
  }

  ierr = ISDestroy(appctx->grid.vertex_global);CHKERRQ(ierr);
  ierr = ISDestroy(appctx->grid.cell_global);CHKERRQ(ierr);

  ierr = ISLocalToGlobalMappingDestroy(appctx->grid.ltog);CHKERRQ(ierr);
  ierr = PetscFree(appctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}





