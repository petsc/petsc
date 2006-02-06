/*
     Loads the quadrilateral grid database from a file  and sets up the local 
     data structures. 
*/

#include "appctx.h"

/*-------------------------------------------------------------------*/
/* (Almost identical with laplacian_q1/AppCtxCreate) */

#undef __FUNCT__
#define __FUNCT__ "AppCxtCreate"
PetscErrorCode AppCtxCreate(MPI_Comm comm,AppCtx **appctx)
{
  PetscErrorCode         ierr;
  PetscTruth  flag;
  PetscViewer binary;
  char        filename[PETSC_MAX_PATH_LEN];
  AppView     *view;  /*added by H. */

  ierr = PetscMalloc(sizeof(AppCtx),appctx);CHKERRQ(ierr);
  (*appctx)->comm = comm;
  view    = &(*appctx)->view; /*added by H. */

  /*-----------------------------------------------------------------------
     Load in the grid database
    ---------------------------------------------------------------------------*/
  ierr = PetscOptionsGetString(0,"-f",filename,256,&flag);CHKERRQ(ierr);
  if (!flag) PetscStrcpy(filename,"gridfile");
  ierr = PetscViewerBinaryOpen((*appctx)->comm,filename,FILE_MODE_READ,&binary);CHKERRQ(ierr);
  ierr = AODataLoadBasic(binary,&(*appctx)->aodata);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(binary);CHKERRQ(ierr);

  /*----------------------------------------------------
    setup viewing options (moved from AppCtxGraphics by H)
   --------------------------------------------------------*/
  ierr = PetscOptionsHasName(PETSC_NULL,"-show_grid",&view->show_grid);CHKERRQ(ierr); 
  ierr = PetscOptionsHasName(PETSC_NULL,"-show_solution",&view->show_solution);CHKERRQ(ierr); 
  ierr = PetscOptionsHasName(PETSC_NULL,"-show_griddata",&view->show_griddata);CHKERRQ(ierr);

  /*------------------------------------------------------------------------
      Setup the local data structures 
      ----------------------------------------------------------------------------*/
  /* (moved to AppCtxSetLocal by H)
      Partition the grid cells */
  
  /* ierr = AODataKeyPartition((*appctx)->aodata,"cell");CHKERRQ(ierr); */ 

  /* Partition the vertices subservient to the cells */
  /* ierr = AODataSegmentPartition((*appctx)->aodata,"cell","vertex");CHKERRQ(ierr);  
  */

  /*     Generate the local numbering of cells and vertices  */
  ierr = AppCtxSetLocal(*appctx);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*-------------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "AppCxtSetLocal"
/*
     AppCtxSetLocal - Sets the local numbering data structures for the grid.
Main Output of AppCCxSetLocal:

    - local IScell
    - globaal ISvertex
    - global ISdf
    - ltog mappings: ltog for vertices (not needed?), dfltog for DF
    - local indices: cell_vertex, cell_DF, cell_cell
    - info associated with vertices: vertex_values, vertex_Df
    - sizes: cell_n, vertex_n_ghosted, df_n_ghosted, vertex_n
    - boundary info: 
          vertex_boundary, (vertices on boundary)
          boundary_df (df associated with boundary)                     

*/
PetscErrorCode AppCtxSetLocal(AppCtx *appctx)
{
  AOData                 ao     = appctx->aodata;
  AppGrid                *grid = &appctx->grid;
  PetscBT                vertex_boundary_flag;
  ISLocalToGlobalMapping cell_ltog;
  PetscErrorCode         ierr;
  PetscInt               rstart,rend,*vertices,i;
  PetscMPIInt            rank;

  MPI_Comm_rank(appctx->comm,&rank);

  if (appctx->view.show_griddata) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"The Application Ordering Database (AO)\n:");CHKERRQ(ierr);
    AODataView(ao,PETSC_VIEWER_STDOUT_SELF);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\n-------- end of AO -------\n");CHKERRQ(ierr);  
  }

  /* (moved from AppCtxCreate by H)
     Partition the grid cells */
  
  ierr = AODataKeyPartition(ao,"cell");CHKERRQ(ierr); 

  /* Partition the vertices subservient to the cells */
  ierr = AODataSegmentPartition(ao,"cell","vertex");CHKERRQ(ierr);  

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
 
  if(appctx->view.show_griddata){  
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\n [%d], vertex_global \n",rank);CHKERRQ(ierr);  
    ISView(grid->vertex_global,PETSC_VIEWER_STDOUT_WORLD);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\n [%d], df_global \n",rank);CHKERRQ(ierr);  
    ISView(grid->df_global,PETSC_VIEWER_STDOUT_WORLD);
  }
 
  /*    Get the coords corresponding to each cell */
  ierr = AODataSegmentGetIS(ao,"cell","coords",grid->cell_global,(void **)&grid->cell_coords);CHKERRQ(ierr);

  /*      Make local to global mapping of cells and vertices  */
  /* Don't want to carry around table which contains the info for all nodes */
  ierr = ISLocalToGlobalMappingCreateIS(grid->cell_global,&cell_ltog);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreateIS(grid->vertex_global,&grid->ltog);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreateIS(grid->df_global,&grid->dfltog);CHKERRQ(ierr);
  /*
    if(appctx->view.show_griddata){ 
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"\n [%d],grid->dfltog,the local to global mapping \n",rank);CHKERRQ(ierr);  
    ierr = ISLocalToGlobalMappingView(grid->dfltog,PETSC_VIEWER_STDOUT_SELF);
    }
    */

  /* Attach the ltog to the database */
  ierr = AODataKeySetLocalToGlobalMapping(ao,"cell",cell_ltog);CHKERRQ(ierr);
  ierr = AODataKeySetLocalToGlobalMapping(ao,"vertex",grid->ltog);CHKERRQ(ierr);
  ierr = AODataKeySetLocalToGlobalMapping(ao,"df",grid->dfltog);CHKERRQ(ierr);
  ierr = PetscObjectDereference((PetscObject)cell_ltog);CHKERRQ(ierr);

  /*      Get the local DF  and vertex lists */
  /* AODataSegmentGetLocalIS uses the ltog info in the database to return the local values for indices */
  ierr = AODataSegmentGetLocalIS(ao,"cell","vertex",grid->cell_global,(void **)&grid->cell_vertex);CHKERRQ(ierr);
  ierr = AODataSegmentGetLocalIS(ao,"cell","df",grid->cell_global,(void **)&grid->cell_df);CHKERRQ(ierr);
  ierr = AODataSegmentGetIS(ao,"cell","cell",grid->cell_global,(void **)&grid->cell_cell);CHKERRQ(ierr);

  /*      Get the size of local objects   */
  ierr = ISGetLocalSize(grid->cell_global,&grid->cell_n);CHKERRQ(ierr);
  ierr = ISGetLocalSize(grid->vertex_global,&grid->vertex_n_ghosted);CHKERRQ(ierr);
  ierr = ISGetLocalSize(grid->df_global,&grid->df_n_ghosted);CHKERRQ(ierr);
  
  /*     Get the numerical values/coords of all vertices for local vertices  */
  ierr = AODataSegmentGetIS(ao,"vertex","values",grid->vertex_global,(void **)&grid->vertex_value);CHKERRQ(ierr);
  /* Get Df's corresponding to the vertices */
  ierr = AODataSegmentGetIS(ao,"vertex","df",grid->vertex_global,(void **)&grid->vertex_df);CHKERRQ(ierr);

  if(appctx->view.show_griddata){ 
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"\n [%d], cell_n= %d, vertex_n_ghosted=%d, df_n_ghosted=%d,\n",rank,grid->cell_n,grid->vertex_n_ghosted,grid->df_n_ghosted);CHKERRQ(ierr);
    /* 
       ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"\n [%d], grid->cell_df:\n",rank);CHKERRQ(ierr); 
       PetscIntView(grid->cell_n*8,grid->cell_df,PETSC_VIEWER_STDOUT_SELF); 
       */
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"\n [%d], grid->vertex_df:\n",rank);CHKERRQ(ierr);
    ierr = PetscIntView(2*grid->vertex_n_ghosted,grid->vertex_df,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  } 

  /* Get  the number of local vertices (rather than the number of ghosted vertices) */
  ierr = AODataKeyGetInfo(ao,"vertex",PETSC_NULL,&grid->vertex_n,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  /* get the number of local dfs */
  ierr = AODataKeyGetInfo(ao,"df",PETSC_NULL,&grid->df_local_count,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);

  if(appctx->view.show_griddata){ 
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"\n [%d], vertex_n (num. of unique vertices)=%d, df_local_count=%d,\n ",rank,grid->vertex_n,grid->df_local_count);CHKERRQ(ierr);
  }

  /************************************************************/
  /*   Set up data structures to simplify dealing with boundary values */
  /************************************************************/

  /*       Get the bit flag indicating boundary for local vertices  */
  ierr = AODataSegmentGetIS(ao,"vertex","boundary",grid->vertex_global,(void **)&vertex_boundary_flag);CHKERRQ(ierr);
  /*     Generate a list of local vertices that are on the boundary  */
  ierr = ISGetIndices(grid->vertex_global,&vertices);CHKERRQ(ierr);
  /*  AODataKeyGetActiveLocal dumps the flagged (active) indices into the IS */
  ierr = AODataKeyGetActiveLocal(ao,"vertex","boundary",grid->vertex_n_ghosted,vertices,0,&grid->isvertex_boundary);CHKERRQ(ierr);
  ierr = ISRestoreIndices(grid->vertex_global,&vertices);CHKERRQ(ierr);

  /* Create some boundary information */
  ierr = ISGetIndices(grid->isvertex_boundary,&grid->vertex_boundary);CHKERRQ(ierr);
  ierr = ISGetLocalSize(grid->isvertex_boundary,&grid->vertex_boundary_count);CHKERRQ(ierr);
  ierr = PetscMalloc(2*grid->vertex_boundary_count*sizeof(int),&grid->boundary_df);CHKERRQ(ierr);
  ierr = PetscMalloc(2*grid->vertex_boundary_count*sizeof(double),&grid->bvs);CHKERRQ(ierr);
  for(i = 0; i < grid->vertex_boundary_count; i++){
    grid->boundary_df[2*i] = grid->vertex_df[2*grid->vertex_boundary[i]];
    grid->boundary_df[2*i+1] = grid->vertex_df[2*grid->vertex_boundary[i]+1];
  }
  ierr = ISCreateGeneral(PETSC_COMM_WORLD,2*grid->vertex_boundary_count,grid->boundary_df,&grid->isboundary_df);

  if(0){printf("here  comes boundary df\n"); PetscIntView(2*grid->vertex_boundary_count,grid->boundary_df,PETSC_VIEWER_STDOUT_SELF); }

  /* need a list of x,y coors corresponding to the boundary vertices only */
  ierr = PetscMalloc(2*grid->vertex_boundary_count*sizeof(double),&grid->bvc);CHKERRQ(ierr);
  for(i = 0; i < grid->vertex_boundary_count; i++){
    grid->bvc[2*i] = grid->vertex_value[2*grid->vertex_boundary[i]];
    grid->bvc[2*i+1]  = grid->vertex_value[2*grid->vertex_boundary[i]+1];
  }

  PetscFunctionReturn(0);

}

/*-----------------------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "AppCxtDestroy"
PetscErrorCode AppCtxDestroy(AppCtx *appctx)
{
  PetscErrorCode ierr;
  AOData         ao = appctx->aodata;
  AppGrid        *grid = &appctx->grid;

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

  if (appctx->view.drawglobal) {
    ierr = PetscDrawDestroy(appctx->view.drawglobal);CHKERRQ(ierr);
    ierr = PetscDrawDestroy(appctx->view.drawlocal);CHKERRQ(ierr);
  }

  ierr = ISDestroy(appctx->grid.vertex_global);CHKERRQ(ierr);
  ierr = ISDestroy(appctx->grid.isvertex_boundary);CHKERRQ(ierr);
  ierr = ISDestroy(appctx->grid.cell_global);CHKERRQ(ierr);

  ierr = ISLocalToGlobalMappingDestroy(appctx->grid.ltog);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(appctx->grid.dfltog);CHKERRQ(ierr);

  ierr = PetscFree(appctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}





