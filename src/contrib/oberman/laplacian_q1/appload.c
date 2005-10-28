
#include "appctx.h"

/*
     Loads the quadrilateral grid database from a file  and sets up the local 
     data structures. 
*/

#undef __FUNCT__
#define __FUNCT__ "AppCxtCreate"
/*
    AppCtxCreate - Fills in the data structures using the grid information from 
  a AOData file.
*/
int AppCtxCreate(MPI_Comm comm,AppCtx **appctx)
{
  int        ierr;
  PetscTruth flag;
  PetscViewer     binary;
  char       filename[PETSC_MAX_PATH_LEN];
  AppView    *view;

  PetscFunctionBegin;
  ierr = PetscMalloc(sizeof(AppCtx),appctx);CHKERRQ(ierr);
  (*appctx)->comm = comm;

  /*-----------------------------------------------------------------------
    Load in the grid database
    ---------------------------------------------------------------------------*/
  /* user indicagtes grid database with -f; defaults to "gridfile" if not give */
  ierr = PetscOptionsGetString(0,"-f",filename,256,&flag);CHKERRQ(ierr);
  if (!flag) {ierr = PetscStrcpy(filename,"gridfile");CHKERRQ(ierr);}

  /* Open the database and read in grid (each processor gets a portion of the grid data)*/
  ierr = PetscViewerBinaryOpen((*appctx)->comm,filename,FILE_MODE_READ,&binary);CHKERRQ(ierr);
  ierr = AODataLoadBasic(binary,&(*appctx)->aodata);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(binary);CHKERRQ(ierr);

  /* allows "values" key or segment in data base to be refered to as "coords" */
  ierr = AODataAliasAdd((*appctx)->aodata,"coords","values");CHKERRQ(ierr);

  /*----------------------------------------------------
    setup viewing (graphics) options 
   --------------------------------------------------------*/
  view    = &(*appctx)->view;
  ierr = PetscOptionsHasName(PETSC_NULL,"-show_solution",&view->show_solution);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-show_matrix",&view->show_matrix);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-show_griddata",&view->show_griddata);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-show_grid",&view->show_grid);CHKERRQ(ierr);

  /*------------------------------------------------------------------------
      Setup the local data structures; this generates a local numbering of cells and vertices
   ----------------------------------------------------------------------------*/
  ierr = AppCtxSetLocal(*appctx);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "AppCxtSetLocal"
/*
     AppCtxSetLocal - Sets the local numbering data structures for the grid.

 Here the basic local data structures are set up.  It was found that the code was simplified by 
making all data structures and computations as cell-based as possible.  The one exception was the 
boundary condtions, and so the code for applying boundary conditions  is kept separate for clarity.

  This is by far the most complicated part of the program, it uses the "magic" of the AOData object
to organize all the grid data to make it suitable for a finite element computation.

  Each processor is assigned a certain number of cells. These are numbered 0 to cell_n[0]-1,
cell_n[0] to cell_n[1]-1, cell_n[1] to cell_n[2]-1, etc. So each processor has cell_n[rank] cells.
The vertices are then divided up among the processors.
*/
int AppCtxSetLocal(AppCtx *appctx)
{
  AOData         ao = appctx->aodata;
  AppGrid        *grid = &appctx->grid;
  IS             isvertex;
  PetscReal      *vertex_coords;
  PetscInt       *vertex_ptr,i;
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  PetscTruth     flag;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

 /************************************************************/
  /*   Set up cell related data structures                   */
 /************************************************************/

  /*
      Get the local cells and vertices, and the local to global mapping for vertices 
      - Partitions the cells across the processors, then 
      - Partitions the vertices, subservient to the elements

         grid->iscell - the cells owned by this processor (in global numbering of cells)
         isvertex     - the vertices owned by this processor (in global numbering of vertices)
         grid->ltog   - the mapping from local numbering of vertices to global numbering of vertices
  */
  ierr = AODataPartitionAndSetupLocal(ao,"cell","vertex",&grid->iscell,&isvertex,&grid->ltog);CHKERRQ(ierr);
  if (appctx->view.show_griddata) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"The Application Ordering Database (AOData:\n");CHKERRQ(ierr);
    ierr = AODataView(ao,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\nAODataPartitionAndSetupLocal generates \n grid->iscell:\n");CHKERRQ(ierr);
    ierr = ISView(grid->iscell,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

    ierr = PetscPrintf(PETSC_COMM_WORLD,"\n isvertex:\n");CHKERRQ(ierr);
    ierr = ISView(isvertex,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

    ierr = PetscPrintf(PETSC_COMM_WORLD,"\n grid->ltog:\n");CHKERRQ(ierr); 
    ierr = ISLocalToGlobalMappingView(grid->ltog,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  /*      Get the number of cells on processor */
  ierr = ISGetLocalSize(grid->iscell,&grid->cell_n);CHKERRQ(ierr);
 
  /* the total number of vertices which are belong to some cell on this processor, */
  ierr = ISGetLocalSize(isvertex,&grid->vertex_n);CHKERRQ(ierr);

  /* the number of vertices which were partionned onto (storage  actually belongs on) this processor */
  ierr = AODataKeyGetInfo(ao,"vertex",PETSC_NULL,&grid->vertex_local_n,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);

  /* get the vertex coordinates */
  ierr = AODataSegmentGetIS(ao,"vertex","coords",isvertex,(void **)&vertex_coords);CHKERRQ(ierr);
  if (appctx->view.show_griddata) {
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] grid->cell_n %d grid->vertex_n %d grid->vertex_local_n %d\n",
                                   rank,grid->cell_n,grid->vertex_n,grid->vertex_local_n);CHKERRQ(ierr);
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] AODataSegmentGetIS generated vertex coordinates\n",rank);CHKERRQ(ierr);
    ierr = PetscRealView(2*grid->cell_n,vertex_coords,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  /*   Get the vertex local numbering for each local cell  */
  ierr = AODataSegmentGetLocalIS(ao,"cell","vertex",grid->iscell,(void **)&grid->cell_vertex);CHKERRQ(ierr);
  if (appctx->view.show_griddata) {
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] AODataSegmentGetLocalIS generated vertex local numbering for each cell\n",rank);CHKERRQ(ierr);
    ierr = PetscIntView(4*grid->cell_n,grid->cell_vertex,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  /*  Get the global vertex number for each cell, for graphics only */
  ierr = PetscMalloc(4*grid->cell_n*sizeof(int),&grid->global_cell_vertex);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingApply(grid->ltog,4*grid->cell_n,grid->cell_vertex,grid->global_cell_vertex);CHKERRQ(ierr);

  /*   Get the coordinates of the cell vertices */
  ierr = AODataSegmentExists(ao,"cell","coords",&flag);CHKERRQ(ierr);
  if (flag) {
    ierr = AODataSegmentGetIS(ao,"cell","coords",grid->iscell,(void **)&grid->cell_coords);CHKERRQ(ierr);
  } else {
    ierr = PetscMalloc(2*4*grid->cell_n*sizeof(PetscReal),&grid->cell_coords);CHKERRQ(ierr);
    for (i=0; i<4*grid->cell_n; i++) {
      grid->cell_coords[2*i]   = vertex_coords[2*grid->cell_vertex[i]];
      grid->cell_coords[2*i+1] = vertex_coords[2*grid->cell_vertex[i]+1];
    }
  }

 /************************************************************/
  /*   Set up boundary related data structures               */
 /************************************************************/

 /*      Generate a list of local vertices that are on the boundary */
  ierr = AODataKeyGetActiveLocalIS(ao,"vertex","boundary",isvertex,0,&grid->vertex_boundary);CHKERRQ(ierr);

  /* get the number of local boundary vertices */
  ierr = ISGetLocalSize(grid->vertex_boundary,&grid->boundary_n);CHKERRQ(ierr); 

  /* pre-allocate storage space for the boundary values to set, and the coordinates */
  ierr = PetscMalloc((grid->boundary_n+1)*sizeof(PetscReal),&grid->boundary_values);CHKERRQ(ierr);
  ierr = PetscMalloc(2*(grid->boundary_n+1)*sizeof(PetscReal),&grid->boundary_coords);CHKERRQ(ierr);

  /* now extract the needed vertex cordinates for the boundary ones */
  ierr = ISGetIndices(grid->vertex_boundary,&vertex_ptr);CHKERRQ(ierr);
  for(i = 0;  i<grid->boundary_n; i++){
    grid->boundary_coords[2*i]    = vertex_coords[2*vertex_ptr[i]];
    grid->boundary_coords[2*i+1]  = vertex_coords[2*vertex_ptr[i]+1];
  }   
  ierr = ISRestoreIndices(grid->vertex_boundary,&vertex_ptr);CHKERRQ(ierr);
  ierr = AODataSegmentRestoreIS(ao,"vertex","coords",isvertex,(void **)&vertex_coords);CHKERRQ(ierr);
  ierr = ISDestroy(isvertex);CHKERRQ(ierr);

  if (appctx->view.show_griddata) {
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] AODataKeyGetActiveLocalIS generated grid->vertex_boundary\n",rank);CHKERRQ(ierr);
    ierr = ISView(grid->vertex_boundary,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "AppCxtDestroy"
/*
          Frees the all the data structures in the program
*/
int AppCtxDestroy(AppCtx *appctx)
{
  int        ierr;
  AOData     ao = appctx->aodata;
  AppGrid    *grid = &appctx->grid;

  PetscFunctionBegin;
  /* 
      Free the database, first we must "restore" any data we have accessed from the AODatabase
  */
  ierr = AODataSegmentRestoreLocalIS(ao,"cell","vertex",PETSC_NULL,(void **)&grid->cell_vertex);CHKERRQ(ierr);
  ierr = AODataSegmentRestoreIS(ao,"cell","coords",0,(void **)&grid->cell_coords);CHKERRQ(ierr);
  ierr = AODataDestroy(ao);CHKERRQ(ierr);

  /*
      Free the grid data 
  */
  ierr = ISLocalToGlobalMappingDestroy(appctx->grid.ltog);CHKERRQ(ierr);
  ierr = PetscFree(grid->boundary_values);CHKERRQ(ierr);
  ierr = PetscFree(grid->boundary_coords);CHKERRQ(ierr);
  ierr = ISDestroy(grid->vertex_boundary);CHKERRQ(ierr);
  ierr = PetscFree(grid->global_cell_vertex);CHKERRQ(ierr);
  ierr = ISDestroy(grid->iscell);CHKERRQ(ierr);

  /*
      Free the algebra 
  */
  ierr = MatDestroy(appctx->algebra.A);CHKERRQ(ierr);
  ierr = VecDestroy(appctx->algebra.b);CHKERRQ(ierr);
  ierr = VecDestroy(appctx->algebra.x);CHKERRQ(ierr);
 
  ierr = PetscFree(appctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}





