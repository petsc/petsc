
/*
     Loads the quadrilateral grid database from a file  and sets up the local 
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

  /*      Partition the vertices subservient to the cells */ 
  ierr = AODataSegmentPartition((*appctx)->aodata,"cell","vertex"); CHKERRA(ierr);  

  /*     Generate the local numbering of cells and vertices  */
  ierr = AppCtxSetLocal(*appctx); CHKERRA(ierr);

  PetscFunctionReturn(0);
}


#undef __FUNC__
#define __FUNC__ "AppCxtSetLocal"
/*
     AppCtxSetLocal - Sets the local numbering data structures for the grid.
Main Output of AppCCxSetLocal:

    - local IScell
    - globaal ISvertex
    - global ISdf
    - ltog mappings: ltog for vertices (not needed?), dfltog for DF
    - local indices: cell_vertex, cell_DF, cell_cell
    - info associated with vertices: vertex_values, vertex_Df
    - sizes: cell_n, vertex_n_ghosted, df_count, vertex_n
    - boundary info: 
          vertex_boundary, (vertices on boundary)
          boundary_df (df associated with boundary)                     

*/
int AppCtxSetLocal(AppCtx *appctx)
{
  AOData                 ao     = appctx->aodata;
  AppGrid                *grid = &appctx->grid;
  BT                     vertex_boundary_flag;
  ISLocalToGlobalMapping ltogcell;
double *vertex_value, *cell_coords;
  int                    ierr, rstart,rend,rank, *vertices;
int *cell_cell, *cell_vertex, *cell_df, vertex_df;
  IS                     iscell, isvertex, isdf;
  int *indices, nindices, i,j;
  int *vertex_ptr;


  MPI_Comm_rank(appctx->comm,&rank);

/* AODataView(ao, VIEWER_STDOUT_SELF);  */

  /*   Generate the list of on processor cells   */
  /* Need a local numbering so that we can loop over the cells */
  ierr = AODataKeyGetOwnershipRange(ao,"cell",&rstart,&rend);CHKERRQ(ierr);
  /* Creates a local IS, grid->cell_global, which is indexed from rstart to rend */
  ierr = ISCreateStride(PETSC_COMM_WORLD,rend-rstart,rstart,1,&grid->cell_global);CHKERRQ(ierr);

  /*       Get the list of vertices used by those cells   */
  /* Retreives the vertices associated with the local cells (in the global numbering) */
  ierr = AODataSegmentGetReducedIS(ao,"cell","vertex",grid->cell_global,&grid->vertex_global);CHKERRQ(ierr);

 /*       Get the list of Degrees of Freedom associated with those cells  (global numbering) */
 ierr = AODataSegmentGetReducedIS(ao,"cell","df",grid->cell_global,&grid->df_global);CHKERRQ(ierr);
 if( 1 ){  printf("df_global \n");  ISView(grid->df_global, VIEWER_STDOUT_SELF);}

 /*    Get the coords corresponding to each cell */
 ierr = AODataSegmentGetIS(ao, "cell", "coords", grid->cell_global , (void **)&grid->cell_coords);CHKERRQ(ierr);
   if( 1 ){  printf("cell_coords\n");   PetscDoubleView(grid->cell_n*8, grid->cell_coords, VIEWER_STDOUT_SELF);}
  /*      Make local to global mapping of cells and vertices  */
 /* Don't want to carry around table which contains the info for all nodes */
  ierr = ISLocalToGlobalMappingCreateIS(grid->cell_global,&ltogcell);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreateIS(grid->vertex_global,&grid->ltog);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreateIS(grid->df_global,&grid->dfltog);CHKERRQ(ierr);
if(0){ printf("the local to global mapping \n");  ierr = ISLocalToGlobalMappingView(grid->dfltog, VIEWER_STDOUT_SELF);}

  /* Attach the ltog to the database */
  ierr = AODataKeySetLocalToGlobalMapping(ao,"cell",ltogcell);CHKERRQ(ierr);
  ierr = AODataKeySetLocalToGlobalMapping(ao,"vertex",grid->ltog);CHKERRQ(ierr);
  ierr = AODataKeySetLocalToGlobalMapping(ao,"df",grid->dfltog);CHKERRQ(ierr);
  ierr = PetscObjectDereference((PetscObject)ltogcell);CHKERRQ(ierr);

  /*      Get the local DF  and vertex lists */
  /* AODataSegmentGetLocalIS uses the ltog info in the database to return the local values for indices */
  ierr = AODataSegmentGetLocalIS(ao,"cell","vertex",grid->cell_global,(void **)&grid->cell_vertex);CHKERRQ(ierr);
  ierr = AODataSegmentGetLocalIS(ao,"cell","df",grid->cell_global,(void **)&grid->cell_df);CHKERRQ(ierr);
  ierr = AODataSegmentGetIS(ao,"cell","cell",grid->cell_global,(void **)&grid->cell_cell);CHKERRQ(ierr);

 /*      Get the size of local objects   */
  ierr = ISGetSize(grid->cell_global,&grid->cell_n); CHKERRQ(ierr);
  ierr = ISGetSize(grid->vertex_global,&grid->vertex_n_ghosted); CHKERRQ(ierr);
  ierr = ISGetSize(grid->df_global, &grid->df_count); CHKERRQ(ierr);
  if(1){ printf("the number of cells on processor %d: %d\n ", rank, grid->cell_n);}

 if( 1 ){  printf("grid cell_df\n");   PetscIntView(grid->cell_n*8, grid->cell_df, VIEWER_STDOUT_SELF);}
  /*       Get the numerical values of all vertices for local vertices  */
  ierr = AODataSegmentGetIS(ao,"vertex","values",grid->vertex_global,(void **)&grid->vertex_value);CHKERRQ(ierr);
  /* Get Df's corresponding to the vertices */
 ierr = AODataSegmentGetIS(ao,"vertex","df",grid->vertex_global,(void **)&grid->vertex_df);CHKERRQ(ierr);

  /* Get  the number of local vertices (rather than the number of ghosted vertices) */
  ierr = AODataKeyGetInfo(ao,"vertex",PETSC_NULL,&grid->vertex_n,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  /* get the number of local dfs */
 ierr = AODataKeyGetInfo(ao,"df",PETSC_NULL,&grid->df_local_count,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);

  /************************************************************/
  /*   Set up data structures to simplify dealing with boundary values */
 /************************************************************/

  /*       Get the bit flag indicating boundary for local vertices  */
  ierr = AODataSegmentGetIS(ao,"vertex","boundary",grid->vertex_global,(void **)&vertex_boundary_flag);CHKERRQ(ierr);
  /*     Generate a list of local vertices that are on the boundary  */
  ierr = ISGetIndices(grid->vertex_global,&vertices);CHKERRQ(ierr);
  /*  AODataKeyGetActiveLocal dumps the flagged (active) indices into the IS */
  ierr = AODataKeyGetActiveLocal(ao,"vertex","boundary",grid->vertex_n_ghosted, vertices,0,&grid->isvertex_boundary);  CHKERRQ(ierr);
  ierr = ISRestoreIndices(grid->vertex_global,&vertices);CHKERRQ(ierr);

  /* Create some boundary information */
  ierr = ISGetIndices(grid->isvertex_boundary, &grid->vertex_boundary);CHKERRQ(ierr);
  ierr = ISGetSize(grid->isvertex_boundary, &grid->vertex_boundary_count); CHKERRQ(ierr);
  grid->boundary_df = (int*)PetscMalloc(2*grid->vertex_boundary_count*sizeof(int)); CHKPTRQ(grid->boundary_df);
  grid->bvs = (double*)PetscMalloc(2*grid->vertex_boundary_count*sizeof(double)); CHKPTRQ(grid->bvs);
  for( i = 0; i < grid->vertex_boundary_count; i++ ){
   grid->boundary_df[2*i] = grid->vertex_df[2*grid->vertex_boundary[i]];
   grid->boundary_df[2*i+1] = grid->vertex_df[2*grid->vertex_boundary[i]+1];
 }
 ierr = ISCreateGeneral(PETSC_COMM_WORLD, 2*grid->vertex_boundary_count, grid->boundary_df, &grid->isboundary_df);

 if(0){printf("here  comes boundary df\n"); PetscIntView(2*grid->vertex_boundary_count, grid->boundary_df, VIEWER_STDOUT_SELF); }

 /* need a list of x,y coors corresponding to the boundary vertices only */
  grid->bvc = (double*)PetscMalloc(2*grid->vertex_boundary_count*sizeof(double)); CHKPTRQ(grid->bvc);
 for( i = 0; i < grid->vertex_boundary_count; i++ ){
   grid->bvc[2*i] = grid->vertex_value[2*grid->vertex_boundary[i]];
   grid->bvc[2*i+1]  = grid->vertex_value[2*grid->vertex_boundary[i]+1];
 }

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
 /*  ierr = AODataSegmentRestoreIS(ao,"vertex","boundary",PETSC_NULL,(void **)&grid->vertex_boundary_flag);CHKERRQ(ierr); */
  ierr = AODataDestroy(ao);CHKERRQ(ierr);

  /*
      Free the algebra 
  */
  ierr = MatDestroy(appctx->algebra.A);CHKERRQ(ierr);
  ierr = VecDestroy(appctx->algebra.b);CHKERRQ(ierr);
  ierr = VecScatterDestroy(appctx->algebra.dfgtol);CHKERRQ(ierr);

  if (appctx->view.showsomething) {
    ierr = DrawDestroy(appctx->view.drawglobal); CHKERRQ(ierr);
    ierr = DrawDestroy(appctx->view.drawlocal); CHKERRQ(ierr);
  }

  ierr = ISDestroy(appctx->grid.vertex_global);CHKERRQ(ierr);
  ierr = ISDestroy(appctx->grid.isvertex_boundary);CHKERRQ(ierr);
  ierr = ISDestroy(appctx->grid.cell_global);CHKERRQ(ierr);

  ierr = ISLocalToGlobalMappingDestroy(appctx->grid.ltog);CHKERRQ(ierr);
 ierr = ISLocalToGlobalMappingDestroy(appctx->grid.dfltog);CHKERRQ(ierr);

  PetscFree(appctx);
  PetscFunctionReturn(0);
}





