

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
  AppEquations *equations;
  AppView *view;
  AppElement *element;

  (*appctx) = (AppCtx *) PetscMalloc(sizeof(AppCtx));CHKPTRQ(*appctx);
  (*appctx)->comm = comm;
   view = &(*appctx)->view;
  equations =&(*appctx)->equations;
  element = &(*appctx)->element;

  /*-----------------------------------------------------------------------
    Load in the grid database
    ---------------------------------------------------------------------------*/
  ierr = OptionsGetString(0,"-f",filename,256,&flag);CHKERRQ(ierr);
  if (!flag) PetscStrcpy(filename,"gridfile");
  ierr = ViewerFileOpenBinary((*appctx)->comm,filename,BINARY_RDONLY,&binary);CHKERRQ(ierr);

  ierr = AODataLoadBasic(binary,&(*appctx)->aodata); CHKERRQ(ierr);
  ierr = ViewerDestroy(binary); CHKERRQ(ierr);

  /*----------------------------------------------------
 setup viewing options 
--------------------------------------------------------*/

  ierr = OptionsHasName(PETSC_NULL,"-matlab_graphics",&view->matlabgraphics); CHKERRQ(ierr);

  ierr = OptionsHasName(PETSC_NULL,"-show_matrix",&view->show_matrix); CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-show_vector",&view->show_vector); CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-show_is",&view->show_is); CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-show_ao",&view->show_ao); CHKERRQ(ierr);

  /* timestepping */
  equations->Nsteps = 8;
  equations->initial_time = 0.0;
  equations->final_time = 1.0;
  equations->amp = 1;
  equations->offset = 0;
  ierr = OptionsGetInt(0,"-Nsteps", &equations->Nsteps, &flag);CHKERRQ(ierr);
  ierr = OptionsGetDouble(0,"-initial_time", &equations->initial_time, &flag);CHKERRQ(ierr);
  ierr = OptionsGetDouble(0, "-final_time", &equations->final_time, &flag);CHKERRQ(ierr);
  ierr = OptionsGetDouble(0, "-amp", &equations->amp, &flag);CHKERRQ(ierr);
  ierr = OptionsGetDouble(0, "-offset", &equations->offset, &flag);CHKERRQ(ierr);
  /* basic parameters */
  equations->eta = 0.1;
  ierr = OptionsGetDouble(0,"-viscosity", &equations->eta, &flag);CHKERRQ(ierr);


  /*------------------------------------------------------------------------
      Setup the local data structures 
      ----------------------------------------------------------------------------*/

  /*     Generate the local numbering of cells and vertices  */
  ierr = AppCtxSetLocal(*appctx); CHKERRA(ierr);


  PetscFunctionReturn(0);
}


#undef __FUNC__
#define __FUNC__ "AppCxtSetLocal"
/*
     AppCtxSetLocal - Sets the local numbering data structures for the grid.

 Here the basic local data structures are set up.  It was found that the code was simplified by making all data structures and computations as cell-based as possible.  The one exception was the boundary condtions, and so the code for applying boundary conditions  is kept separate for clarity.

 This results in the code for the boundary being more complex.  
 It should be skimmed over at first glance.

*/
int AppCtxSetLocal(AppCtx *appctx)
{
  AOData ao = appctx->aodata;
  AppGrid *grid = &appctx->grid;

  IS  iscell,isvertex;
  double *vertex_coords;
  int *vertex_ptr;
  int i, ierr;
  int rank;

  /* get the local cells and vertices, and the local to global mapping for vertices */
  ierr = AODataPartitionAndSetupLocal(ao, "cell", "vertex", &iscell, &isvertex, &grid->ltog); CHKERRQ(ierr);
 
  /*      Get the local vertex and neighbour lists  */
  ierr = AODataSegmentGetLocalIS(ao,"cell","vertex",iscell,(void **)&grid->cell_vertex);CHKERRQ(ierr);
  ierr = AODataSegmentGetIS(ao,"cell","coords",iscell,(void **)&grid->cell_coords);CHKERRQ(ierr);
 
  /* Get Sizes of local objects  */
  /*      Get the size of local objects  */
  ierr = ISGetSize(iscell,&grid->cell_n); CHKERRQ(ierr);
  /* the total number of vertices which are belong to some cell on this processor, */
  ierr = ISGetSize(isvertex,&grid->vertex_count); CHKERRQ(ierr);
  /* the number of vertices which were partionned onto 
     (ie storage  actually belongs on) this processor */
  ierr = AODataKeyGetInfo(ao,"vertex",PETSC_NULL,&grid->vertex_local_count,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);

 /************************************************************/
  /*   Set up boundary related data structures                                    */
 /************************************************************/

 /*      Generate a list of local vertices that are on the boundary */
  ierr = AODataKeyGetActiveLocalIS(ao,"vertex","boundary",isvertex,0,&grid->vertex_boundary); CHKERRQ(ierr);

 /* get the size */
  ierr = ISGetSize(grid->vertex_boundary, &grid->boundary_count); CHKERRQ(ierr); 
 /* pre-allocate storage space for the boundary values to set, and the coords */
  grid->boundary_values = (double*)PetscMalloc((grid->boundary_count+1)*sizeof(double)); CHKPTRQ(grid->boundary_values);
  grid->boundary_coords = (double*)PetscMalloc(2*(grid->boundary_count+1)*sizeof(double)); CHKPTRQ(grid->boundary_coords);
  /* get the coords */
  /* get the vertex coords */
  ierr = AODataSegmentGetIS(ao,"vertex","coords",isvertex,(void **)&vertex_coords);CHKERRQ(ierr);
  /* now extract the needed ones */
    ierr = ISGetIndices(grid->vertex_boundary, &vertex_ptr); CHKERRQ(ierr);
    for( i = 0;  i < grid->boundary_count; i++ ){
      grid->boundary_coords[2*i] = vertex_coords[2*vertex_ptr[i]];
      grid->boundary_coords[2*i+1]  = vertex_coords[2*vertex_ptr[i]+1];
    }   
    ierr = ISRestoreIndices(grid->vertex_boundary, &vertex_ptr); CHKERRQ(ierr);
/* free up unneeded vertex_coords */
  ierr = PetscFree(vertex_coords);CHKERRQ(ierr);

 /**********view things ****************/
  
  /* get the number of this processor (just for viewing purposes) */
 MPI_Comm_rank(appctx->comm,&rank);

  if(appctx->view.show_ao){
    printf("The Application Ordering Database (AO)\n:"); 
    ierr = AODataView(ao, VIEWER_STDOUT_SELF );CHKERRA(ierr); }

  if(appctx->view.show_is){
    printf("The local index sets and local to global mappings (best with more than one processor):\n");
    printf("The local cell indices\n");
    printf("Processor %d\n", rank);
    ISView(iscell, VIEWER_STDOUT_SELF);
    printf("\nThe local vertex indices\n"); printf("Processor %d\n", rank);
    ISView(isvertex, VIEWER_STDOUT_SELF);
    printf("\nThe local to global mapping of vertices\n"); printf("Processor %d\n", rank);
    ISLocalToGlobalMappingView(grid->ltog, VIEWER_STDOUT_SELF);

    printf("The local cell_vertices\n");printf("Processor %d\n", rank);
    PetscIntView(4*grid->cell_n, grid->cell_vertex, VIEWER_STDOUT_SELF);
  }

  /* for viewing grid */
    grid->cell_global = iscell; 


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

  ierr = AODataSegmentRestoreLocalIS(ao,"cell","vertex",PETSC_NULL,(void **)&grid->cell_vertex);CHKERRQ(ierr);
 
  ierr = AODataDestroy(ao);CHKERRQ(ierr);

  /*
      Free the algebra 
  */
  ierr = MatDestroy(appctx->algebra.A);CHKERRQ(ierr);
  ierr = VecDestroy(appctx->algebra.b);CHKERRQ(ierr);
  ierr = VecDestroy(appctx->algebra.x);CHKERRQ(ierr);
 
  ierr = ISLocalToGlobalMappingDestroy(appctx->grid.ltog);CHKERRQ(ierr);
  ierr = PetscFree(appctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}




/*
This will be a Petsc Utility function, not part of the code.
*/
#undef __FUNC__
#define __FUNC__ "AODataPartitionAndSetupLocal"
/*     AppDataSetupLocal - Sets the local numbering data structures for the grid.
Input:
        AOData ao,  - the AO
        string cell	- the name of the key
        string segment    -the name of the segment 
        IS baseIS          - the local indices of the key

Output:
       IS   issegment     - the local indices in global numbering of the segment
        ISLocalToGlobalMapping ltogsegment - the local to global mapping for the segment
	*/

/*
do I need to do more derefencing???
possible side effect: attaching ltogcell to the database
*/
int AODataPartitionAndSetupLocal(AOData ao, char *keyname,  char *segmentname, IS *iscell, IS *issegment, ISLocalToGlobalMapping *ltog){
  ISLocalToGlobalMapping ltogcell;
  int ierr,rstart,rend,rank;

  PetscFunctionBegin;  

  /*      Partition the grid cells   */
  ierr = AODataKeyPartition(ao,keyname); CHKERRA(ierr);  

  /*      Partition the vertices subservient to the cells  */ 
  ierr = AODataSegmentPartition(ao,keyname,segmentname); CHKERRA(ierr);  

 /*     Generate the list of on processor cells   */
  ierr = AODataKeyGetOwnershipRange(ao,"cell",&rstart,&rend);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_WORLD,rend-rstart,rstart,1,iscell);CHKERRQ(ierr);

 /*       Get the list of vertices used by those cells   */
  ierr = AODataSegmentGetReducedIS(ao,keyname,segmentname,*iscell,issegment);CHKERRQ(ierr);
 /*     Make local to global mapping of cells  */
  ierr = ISLocalToGlobalMappingCreateIS(*iscell,&ltogcell);CHKERRQ(ierr);
  /*       Make local to global mapping of  vertices  */
  ierr = ISLocalToGlobalMappingCreateIS(*issegment,ltog);CHKERRQ(ierr);
  /*        Attach the local to global mapping to the database */
  ierr = AODataKeySetLocalToGlobalMapping(ao,keyname,ltogcell);CHKERRQ(ierr);
  ierr = AODataKeySetLocalToGlobalMapping(ao,segmentname,*ltog);CHKERRQ(ierr);
  /* Dereference the ltogcell */
  ierr = PetscObjectDereference((PetscObject)ltogcell);CHKERRQ(ierr);

 PetscFunctionReturn(0);
}





