/*$Id: appload.c,v 1.4 2000/01/06 20:43:19 bsmith Exp bsmith $*/
#include "appctx.h"

/*
     Loads the quadrilateral grid database from a file  and sets up the local 
     data structures. 
*/

#undef __FUNC__
#define __FUNC__ "AppCxtCreate"
/*
    AppCtxCreate - Fills in the data structures using the grid information from 
  a AOData file.
*/
int AppCtxCreate(MPI_Comm comm,AppCtx **appctx)
{
  int        ierr;
  PetscTruth flag;
  Viewer     binary;
  char       filename[256];
  AppView    *view;

  PetscFunctionBegin;
  (*appctx)       = (AppCtx*)PetscMalloc(sizeof(AppCtx));CHKPTRQ(*appctx);
  (*appctx)->comm = comm;

  /*-----------------------------------------------------------------------
    Load in the grid database
    ---------------------------------------------------------------------------*/
  /* user indicagtes grid database with -f; defaults to "gridfile" if not give */
  ierr = OptionsGetString(0,"-f",filename,256,&flag);CHKERRQ(ierr);
  if (!flag) {ierr = PetscStrcpy(filename,"gridfile");CHKERRQ(ierr);}

  /* Open the database and read in grid (each processor gets a portion of the grid data)*/
  ierr = ViewerBinaryOpen((*appctx)->comm,filename,BINARY_RDONLY,&binary);CHKERRQ(ierr);
  ierr = AODataLoadBasic(binary,&(*appctx)->aodata);CHKERRQ(ierr);
  ierr = ViewerDestroy(binary);CHKERRQ(ierr);

  /* allows "values" key or segment in data base to be refered to as "coords" */
  ierr = AODataAliasAdd((*appctx)->aodata,"coords","values");CHKERRQ(ierr);

  /*----------------------------------------------------
    setup viewing (graphics) options 
   --------------------------------------------------------*/
  view    = &(*appctx)->view;
  ierr = OptionsHasName(PETSC_NULL,"-show_solution",&view->show_solution);CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-show_matrix",&view->show_matrix);CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-show_griddata",&view->show_griddata);CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-show_grid",&view->show_grid);CHKERRQ(ierr);

  /*------------------------------------------------------------------------
      Setup the local data structures; this generates a local numbering of cells and vertices
   ----------------------------------------------------------------------------*/
  ierr = AppCtxSetLocal(*appctx);CHKERRA(ierr);

  PetscFunctionReturn(0);
}


#undef __FUNC__
#define __FUNC__ "AppCxtSetLocal"
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
  AOData     ao = appctx->aodata;
  AppGrid    *grid = &appctx->grid;
  IS         isvertex;
  double     *vertex_coords;
  int        *vertex_ptr,i,ierr,rank;
  PetscTruth flag;

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
    ierr = AODataView(ao,VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\nAODataPartitionAndSetupLocal generates \n grid->iscell:\n");CHKERRA(ierr);
    ierr = ISView(grid->iscell,VIEWER_STDOUT_WORLD);CHKERRA(ierr);

    ierr = PetscPrintf(PETSC_COMM_WORLD,"\n isvertex:\n");CHKERRA(ierr);
    ierr = ISView(isvertex,VIEWER_STDOUT_WORLD);CHKERRA(ierr);

    ierr = PetscPrintf(PETSC_COMM_WORLD,"\n grid->ltog:\n");CHKERRA(ierr); 
    ierr = ISLocalToGlobalMappingView(grid->ltog,VIEWER_STDOUT_WORLD);CHKERRA(ierr);
  }

  /*      Get the number of cells on processor */
  ierr = ISGetSize(grid->iscell,&grid->cell_n);CHKERRQ(ierr);
 
  /* the total number of vertices which are belong to some cell on this processor, */
  ierr = ISGetSize(isvertex,&grid->vertex_count);CHKERRQ(ierr);

  /* the number of vertices which were partionned onto (storage  actually belongs on) this processor */
  ierr = AODataKeyGetInfo(ao,"vertex",PETSC_NULL,&grid->vertex_local_count,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);

  /* get the vertex coordinates */
  ierr = AODataSegmentGetIS(ao,"vertex","coords",isvertex,(void **)&vertex_coords);CHKERRQ(ierr);
  if (appctx->view.show_griddata) {
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] grid->cell_n %d grid->vertex_count %d grid->vertex_local_count %d\n",
                                   rank,grid->cell_n,grid->vertex_count,grid->vertex_local_count);CHKERRQ(ierr);
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] AODataSegmentGetIS generated vertex coordinates\n",rank);CHKERRQ(ierr);
    ierr = PetscDoubleView(2*grid->cell_n,vertex_coords,VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  /*   Get the vertex local numbering for each local cell  */
  ierr = AODataSegmentGetLocalIS(ao,"cell","vertex",grid->iscell,(void **)&grid->cell_vertex);CHKERRQ(ierr);
  if (appctx->view.show_griddata) {
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] AODataSegmentGetLocalIS generated vertex local numbering for each cell\n",rank);CHKERRQ(ierr);
    ierr = PetscIntView(4*grid->cell_n,grid->cell_vertex,VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  /*  Get the global vertex number for each cell, for graphics only */
  grid->global_cell_vertex = (int*)PetscMalloc(4*grid->cell_n*sizeof(int));CHKPTRQ(grid->global_cell_vertex);
  ierr = ISLocalToGlobalMappingApply(grid->ltog,4*grid->cell_n,grid->cell_vertex,grid->global_cell_vertex);CHKERRQ(ierr);

  /*   Get the coordinates of the cell vertices */
  ierr = AODataSegmentExists(ao,"cell","coords",&flag);CHKERRQ(ierr);
  if (flag) {
    ierr = AODataSegmentGetIS(ao,"cell","coords",grid->iscell,(void **)&grid->cell_coords);CHKERRQ(ierr);
  } else {
    grid->cell_coords = (double*)PetscMalloc(2*4*grid->cell_n*sizeof(double));CHKERRQ(ierr);
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
  ierr = ISGetSize(grid->vertex_boundary,&grid->boundary_count);CHKERRQ(ierr); 

  /* pre-allocate storage space for the boundary values to set, and the coordinates */
  grid->boundary_values = (double*)PetscMalloc((grid->boundary_count+1)*sizeof(double));CHKPTRQ(grid->boundary_values);
  grid->boundary_coords = (double*)PetscMalloc(2*(grid->boundary_count+1)*sizeof(double));CHKPTRQ(grid->boundary_coords);

  /* now extract the needed vertex cordinates for the boundary ones */
  ierr = ISGetIndices(grid->vertex_boundary,&vertex_ptr);CHKERRQ(ierr);
  for(i = 0;  i<grid->boundary_count; i++){
    grid->boundary_coords[2*i]    = vertex_coords[2*vertex_ptr[i]];
    grid->boundary_coords[2*i+1]  = vertex_coords[2*vertex_ptr[i]+1];
  }   
  ierr = ISRestoreIndices(grid->vertex_boundary,&vertex_ptr);CHKERRQ(ierr);
  ierr = AODataSegmentRestoreIS(ao,"vertex","coords",isvertex,(void **)&vertex_coords);CHKERRQ(ierr);
  ierr = ISDestroy(isvertex);CHKERRQ(ierr);

  if (appctx->view.show_griddata) {
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] AODataKeyGetActiveLocalIS generated grid->vertex_boundary\n",rank);CHKERRQ(ierr);
    ierr = ISView(grid->vertex_boundary,VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNC__
#define __FUNC__ "AppCxtDestroy"
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





