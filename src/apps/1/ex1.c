
static char help[] ="Solves a simple linear PDE on an unstructured grid\n";


/*

*/

#include "ao.h"
#include "mat.h"
#include "sles.h"

/*
        cell_n        - number of cells on this processor 
      
        cell_vertex   - vertices of the cells (in local numbering)
        cell_global   - global number of each cell on this processor

        vertex_n      - number of vertices on this processor
        vertex_global - global number of each vertex on this processor
        vertex_value  - x,y coordinates of vertices on this processor

*/
        
typedef struct {
  int      cell_n;
  int      *cell_vertex;
  IS       cell_global;
  int      vertex_n;
  IS       vertex_global;
  double   *vertex_value;
} LocalGrid;

typedef struct {
  MPI_Comm  comm;
  AOData    aodata;
  LocalGrid lgrid;
  Draw      drawlocal;
  Draw      drawglobal;
  int       number;        /* show cell/vertex numbers on graphics */
} AppCtx;


extern int AppCtxView(Draw,void*);
extern int AppctxDestroy(AppCtx *);
extern int AppCtxSetLocal(AppCtx *);
extern int AppCtxCreate(MPI_Comm,AppCtx **);

int main( int argc, char **argv )
{
  int            ierr;
  AppCtx         *appctx;

  /* ---------------------------------------------------------------------
     Initialize PETSc
     ------------------------------------------------------------------------*/

  PetscInitialize(&argc,&argv,(char *)0,help);

  /* 
      Load the grid database and initialize graphics 
  */
  ierr = AppCtxCreate(PETSC_COMM_WORLD,&appctx);

  /*
      Partition the grid by cells
  */
  ierr = AODataKeyPartition(appctx->aodata,"cell"); CHKERRA(ierr); 

  /*
     Generate the local numbering of cells and vertices
  */
  ierr = AppCtxSetLocal(appctx); CHKERRA(ierr);

  /*
     Visualize the grid 
  */
  ierr = DrawZoom(appctx->drawglobal,AppCtxView,appctx); CHKERRA(ierr);


  ierr = AppctxDestroy(appctx); CHKERRA(ierr);

  PetscFinalize();

  PetscFunctionReturn(0);
}


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

  /*
     Load in the grid database
  */
  ierr = OptionsGetString(0,"-f",filename,256,&flag);CHKERRA(ierr);
  if (!flag) PetscStrcpy(filename,"gridfile");
  ierr = ViewerFileOpenBinary((*appctx)->comm,filename,BINARY_RDONLY,&binary);CHKERRQ(ierr);
  ierr = AODataLoadBasic(binary,&(*appctx)->aodata); CHKERRA(ierr);
  ierr = ViewerDestroy(binary); CHKERRA(ierr);

  /*---------------------------------------------------------------------
     Open the graphics windows
     ------------------------------------------------------------------------*/

  ierr = DrawOpenX(PETSC_COMM_WORLD,PETSC_NULL,"Total Grid",PETSC_DECIDE,PETSC_DECIDE,400,400,
                  &(*appctx)->drawglobal); CHKERRQ(ierr);
  ierr = DrawOpenX(PETSC_COMM_WORLD,PETSC_NULL,"Local Grids",PETSC_DECIDE,PETSC_DECIDE,400,400,
                   &(*appctx)->drawlocal);CHKERRQ(ierr);
  ierr = DrawSplitViewPort((*appctx)->drawlocal);CHKERRQ(ierr);

  ierr = OptionsHasName(PETSC_NULL,"-number",&(*appctx)->number);CHKERRQ(ierr);

  /*
       Set the window coordinates based on the values in vertices
  */
  ierr = AODataSegmentGetExtrema((*appctx)->aodata,"vertex","values",maxs,mins);CHKERRQ(ierr);
  hx = maxs[0] - mins[0]; xmin = mins[0] - .1*hx; xmax = maxs[0] + .1*hx;
  hy = maxs[1] - mins[1]; ymin = mins[1] - .1*hy; ymax = maxs[1] + .1*hy;
  ierr = DrawSetCoordinates((*appctx)->drawglobal,xmin,ymin,xmax,ymax);CHKERRQ(ierr);
  ierr = DrawSetCoordinates((*appctx)->drawlocal,xmin,ymin,xmax,ymax);CHKERRQ(ierr);

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
  LocalGrid              *lgrid = &appctx->lgrid;

  int                    cell_n,vertex_n;
  int                    *cell_vertex;
  double                 *vertex_value;

  ISLocalToGlobalMapping ltogvertex,ltogcell;

  int                    ierr,rstart,rend,rank;
  IS                     iscell,isvertex;

  MPI_Comm_rank(appctx->comm,&rank);


  /*
      Generate the list of on processor cells 
  */
  ierr = AODataKeyGetInfoOwnership(ao,"cell",&rstart,&rend);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_WORLD,rend-rstart,rstart,1,&iscell);CHKERRQ(ierr);

  /*
       Get the list of vertices and edges used by those cells 
  */
  ierr = AODataSegmentGetReducedIS(ao,"cell","vertex",iscell,&isvertex);CHKERRQ(ierr);

  /* 
      Make local to global mapping of cell, vertices, and edges 
  */
  ierr = ISLocalToGlobalMappingCreateIS(iscell,&ltogcell);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreateIS(isvertex,&ltogvertex);CHKERRQ(ierr);
  
  ierr = AODataKeyAddLocalToGlobalMapping(ao,"cell",ltogcell);CHKERRQ(ierr);
  ierr = AODataKeyAddLocalToGlobalMapping(ao,"vertex",ltogvertex);CHKERRQ(ierr);

  ierr = PetscObjectDereference((PetscObject)ltogcell);CHKERRQ(ierr);
  ierr = PetscObjectDereference((PetscObject)ltogvertex);CHKERRQ(ierr);

  /*
      Get the local edge and vertex lists
  */
  ierr = AODataSegmentGetLocalIS(ao,"cell","vertex",iscell,(void **)&cell_vertex);CHKERRQ(ierr);

  /* 
      Get the numerical values of all vertices for local cells 
  */
  ierr = AODataSegmentGetIS(ao,"vertex","values",isvertex,(void **)&vertex_value);CHKERRQ(ierr);

  /*
      Get the size and global indices of local objects for plotting
  */
  ierr = ISGetSize(iscell,&cell_n); CHKERRQ(ierr);
  ierr = ISGetSize(isvertex,&vertex_n); CHKERRQ(ierr);

  lgrid->cell_n        = cell_n; 
  lgrid->cell_vertex   = cell_vertex; 
  lgrid->cell_global   = iscell;

  lgrid->vertex_n      = vertex_n;
  lgrid->vertex_global = isvertex;
  lgrid->vertex_value  = vertex_value;

  PetscFunctionReturn(0);

}

#undef __FUNC__
#define __FUNC__ "AppCxtDestroy"
int AppctxDestroy(AppCtx *appctx)
{
  int        ierr;
  AOData     ao = appctx->aodata;
  LocalGrid  *lgrid = &appctx->lgrid;

  ierr = AODataSegmentRestoreIS(ao,"vertex","values",PETSC_NULL,(void **)&lgrid->vertex_value);CHKERRQ(ierr);
  ierr = AODataSegmentRestoreLocalIS(ao,"cell","vertex",PETSC_NULL,(void **)&lgrid->cell_vertex);CHKERRQ(ierr);
  ierr = AODataDestroy(ao);CHKERRQ(ierr);

  ierr = DrawDestroy(appctx->drawglobal); CHKERRQ(ierr);
  ierr = DrawDestroy(appctx->drawlocal); CHKERRQ(ierr);

  ierr = ISDestroy(appctx->lgrid.vertex_global);CHKERRQ(ierr);
  ierr = ISDestroy(appctx->lgrid.cell_global);CHKERRQ(ierr);

  PetscFree(appctx);
  PetscFunctionReturn(0);
}


#undef __FUNC__
#define __FUNC__ "AppCxtView"
int AppCtxView(Draw idraw,void *iappctx)
{
  AppCtx                 *appctx = (AppCtx *)iappctx;
  LocalGrid              *lgrid = &appctx->lgrid;

  int                    cell_n,vertex_n,ncell;

  /*
        These contain the edge and vertex lists in local numbering
  */ 
  int                    *cell_vertex;

  /* 
        These contain the global numbering for local objects
  */
  int                    *cell_global,*vertex_global;
  
  double                 *vertex_value;


  int                    ierr,i,rank,c,j;
  int                    ij,number = appctx->number;

  Draw                   drawglobal = appctx->drawglobal;
  Draw                   drawlocal = appctx->drawlocal;
  double                 xl,yl,xr,yr,xm,ym,xp,yp;
  char                   num[5];

  /* get number of vertices per cell */
  ierr = AODataSegmentGetInfo(appctx->aodata,"cell","vertex",0,0,&ncell,0);CHKERRQ(ierr);

  ierr = DrawCheckResizedWindow(drawglobal); CHKERRQ(ierr);
  ierr = DrawCheckResizedWindow(drawlocal); CHKERRQ(ierr);

  MPI_Comm_rank(appctx->comm,&rank); c = rank + 2;

  cell_n        = lgrid->cell_n;
  ierr = ISGetIndices(lgrid->cell_global,&cell_global); CHKERRQ(ierr);
  ierr = ISGetIndices(lgrid->vertex_global,&vertex_global); CHKERRQ(ierr);

  cell_vertex   = lgrid->cell_vertex;
  vertex_n      = lgrid->vertex_n;
  vertex_value  = lgrid->vertex_value;

  /*
        Draw edges of local cells and number them
     First do ghost cells
  */
  /* do the local cells */
  for (i=0; i<cell_n; i++ ) {
    xp = 0.0; yp = 0.0;
    xl = vertex_value[2*cell_vertex[ncell*i]]; yl = vertex_value[2*cell_vertex[ncell*i] + 1];
    for (j=0; j<ncell; j++) {
      ij = ncell*i + ((j+1) % ncell);
      xr = vertex_value[2*cell_vertex[ij]]; yr = vertex_value[2*cell_vertex[ij] + 1];
      ierr = DrawLine(drawglobal,xl,yl,xr,yr,c);CHKERRQ(ierr);
      ierr = DrawLine(drawlocal,xl,yl,xr,yr,DRAW_BLUE);CHKERRQ(ierr);
      xp += xl;         yp += yl;
      xl  = xr;         yl =  yr;
    }
    xp /= ncell; yp /= ncell;
    sprintf(num,"%d",i);
    if (number) {ierr = DrawString(drawlocal,xp,yp,DRAW_GREEN,num);CHKERRQ(ierr);}
    sprintf(num,"%d",cell_global[i]);
    if (number) {ierr = DrawString(drawglobal,xp,yp,c,num);CHKERRQ(ierr);}
  }

  /*
      Number vertices
  */
  for (i=0; i<vertex_n; i++ ) {
    xm = vertex_value[2*i]; ym = vertex_value[2*i + 1];
    sprintf(num,"%d",i);
    if (number) {ierr = DrawString(drawlocal,xm,ym,DRAW_BLUE,num);CHKERRQ(ierr);}
    sprintf(num,"%d",vertex_global[i]);
    if (number) {ierr = DrawString(drawglobal,xm,ym,DRAW_BLUE,num);CHKERRQ(ierr);}
  }
  ierr = DrawSynchronizedFlush(drawglobal);CHKERRQ(ierr);
  ierr = DrawSynchronizedFlush(drawlocal);CHKERRQ(ierr);

  ierr = ISRestoreIndices(lgrid->cell_global,&cell_global); CHKERRQ(ierr);
  ierr = ISRestoreIndices(lgrid->vertex_global,&vertex_global); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

