

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
#define __FUNC__ "AppCxtSetLocal"
/*
     AppCtxSetLocal - Sets the local numbering data structures for the grid.

*/
int AppCtxSetLocal(AppCtx *appctx)
{
  AOData                 ao     = appctx->aodata;
  AppGrid                *grid = &appctx->grid;

  BT                     vertex_boundary_flag;
  ISLocalToGlobalMapping ltogcell;
  int                    ierr,rstart,rend,rank;

  int                    *cell_vertex;
  double                 *vertex_value;
  int                    *vertices,cell_n,vertex_n_ghosted,*cell_cell;
  int  *vertex_indices;
const int DFS = 2;
  IS                     iscell,isvertex,vertex_boundary;

  IS  isvertex_global_blocked, isvertex_boundary_blocked;
  int i, vertex_size, vertex_boundary_size, *vertex_blocked, *vertex_boundary_blocked;

  MPI_Comm_rank(appctx->comm,&rank);

  /*      Generate the list of on processor cells  */
  ierr = AODataKeyGetOwnershipRange(ao,"cell",&rstart,&rend);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_WORLD,rend-rstart,rstart,1,&iscell);CHKERRQ(ierr);

  /* get the local vertices, and the local to global mapping */
  ierr = AODataSetupLocal(ao, "cell", "vertex", iscell, &isvertex, &grid->ltog); CHKERRQ(ierr);

 
 /* create a blocked version of the isvertex (will be vertex_global)*/
  /* make the extra index set needed for MatZeroRows */
  ierr = ISGetIndices(isvertex, &vertex_indices);CHKERRQ(ierr);
  ierr = ISGetSize(isvertex, &vertex_size);CHKERRQ(ierr);
  vertex_blocked = (int *) PetscMalloc(((2*vertex_size)+1)*sizeof(int));CHKPTRQ(vertex_blocked);
  for(i=0;i<vertex_size;i++){
    vertex_blocked[2*i] = 2*vertex_indices[i];
    vertex_blocked[2*i+1] =  2*vertex_indices[i] + 1;
  }
  ierr = ISCreateGeneral(PETSC_COMM_WORLD,2*vertex_size, vertex_blocked, &isvertex_global_blocked);
  ierr = ISLocalToGlobalMappingCreateIS(isvertex_global_blocked,&grid->dltog);CHKERRQ(ierr);


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

 /*      Generate a list of local vertices that are on the boundary */
  ierr = AODataKeyGetActiveLocalIS(ao,"vertex","boundary",isvertex,0,&vertex_boundary); CHKERRQ(ierr);

  /*  Now create a blocked IS for MatZeroRowsLocal */
  ierr = ISGetIndices(vertex_boundary, &vertex_indices);CHKERRQ(ierr);
  ierr = ISGetSize(vertex_boundary, &vertex_boundary_size); CHKERRQ(ierr); 
  vertex_boundary_blocked = (int *) PetscMalloc((2*vertex_boundary_size)*sizeof(int));CHKPTRQ(vertex_boundary_blocked); 
  for(i=0;i<vertex_boundary_size;i++){ 
     vertex_boundary_blocked[2*i] = 2*vertex_indices[i]; 
     vertex_boundary_blocked[2*i+1] = 2*vertex_indices[i] + 1; 
   } 
   ierr = ISCreateGeneral(PETSC_COMM_WORLD,2*vertex_boundary_size, vertex_boundary_blocked, &isvertex_boundary_blocked); 

  grid->cell_vertex          = cell_vertex; 
  grid->cell_global          = iscell;
  grid->vertex_global_blocked       = isvertex_global_blocked; /* my addition */
  grid->vertex_boundary_blocked = isvertex_boundary_blocked; /* my addition */
  grid->cell_n               = cell_n;
  grid->cell_cell            = cell_cell;

  grid->vertex_global        = isvertex;
  grid->vertex_value         = vertex_value;

  grid->vertex_boundary      = vertex_boundary;
  grid->vertex_n_ghosted     = vertex_n_ghosted;

  ierr = AODataKeyGetInfo(ao,"vertex",PETSC_NULL,&grid->vertex_n,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = AODataSegmentGetInfo(ao,"cell","vertex",&grid->NVs,0);CHKERRQ(ierr);

  PetscFunctionReturn(0);

}


#undef __FUNC__
#define __FUNC__ "AODataSetupLocal"
/*     AppDataSetupLocal - Sets the local numbering data structures for the grid.
Input:
        AOData ao,  - the AO
        string cell	- the name of the key
        string segment    -the name of the segment 
        IS baseIS          - the local indices of the key

Output:
       IS   issegment     - the local indices in global numbering of the segment
        ISLocalToGlobalMapping ltogsegment - the local to global mapping for the segment

do I need to do more derefencing???
possible side effect: attaching ltogcell to the database
*/
int AODataSetupLocal(AOData ao, char *keyname,  char *segmentname, IS iscell, IS *issegment, ISLocalToGlobalMapping *ltog){
  ISLocalToGlobalMapping ltogcell;
  int ierr;

 PetscFunctionBegin;  
 /*       Get the list of vertices used by those cells   */
  ierr = AODataSegmentGetReducedIS(ao,keyname,segmentname,iscell,issegment);CHKERRQ(ierr);
 /*     Make local to global mapping of cells  */
  ierr = ISLocalToGlobalMappingCreateIS(iscell,&ltogcell);CHKERRQ(ierr);
  /*       Make local to global mapping of  vertices  */
  ierr = ISLocalToGlobalMappingCreateIS(*issegment,ltog);CHKERRQ(ierr);
  /*        Attach the local to global mapping to the database */
  ierr = AODataKeySetLocalToGlobalMapping(ao,keyname,ltogcell);CHKERRQ(ierr);
  ierr = AODataKeySetLocalToGlobalMapping(ao,segmentname,*ltog);CHKERRQ(ierr);
  /* Dereference the ltogcell */
  ierr = PetscObjectDereference((PetscObject)ltogcell);CHKERRQ(ierr);

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





