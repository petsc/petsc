
static char help[] ="Reads in parallel a cellrilateral grid and partitions it.\n";


/*

*/

#include "ao.h"
#include "mat.h"
#include "draw.h"

/*
        cell_nlocal   - number of local (nonghosted cells on this processor
        cell_n        - number of cells on this processor including ghosts
      
        cell_edge     - edges of each of the cells (in local numbering)
        cell_vertex   - vertices of the cells (in local numbering)
        cell_global   - global number of each cell on this processor

        edge_n        - number of edges on this processor 
        edge_vertex   - vertices of the edges (in local numbering)
        edge_global   - global number of each edge on this processor

        vertex_n      - number of vertices on this processor
        vertex_global - global number of each vertex on this processor
        vertex_value  - x,y coordinates of vertices on this processor

*/
        
typedef struct {
  int      ncell;                /* number of edges per cell, 3 or 4 */
  int      cell_nlocal, cell_n;   
  int      *cell_edge, *cell_vertex, *cell_global;

  int      edge_n;
  int      *edge_vertex, *edge_global;

  int      vertex_n;
  int      *vertex_global;
  double   *vertex_value;
} LocalGrid;

typedef struct {
  MPI_Comm  comm;
  AOData    aodata;
  LocalGrid lgrid;
  Draw      drawlocal;
  Draw      drawglobal;
} AppCtx;


extern int AppCtxView(AppCtx *);
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

  ierr = AppCtxCreate(PETSC_COMM_WORLD,&appctx);


  /*
     Generate the local numbering of required objects 
  */
  ierr = AppCtxSetLocal(appctx); CHKERRA(ierr);

  /*
     Visualize the grid 
  */
  ierr = AppCtxView(appctx); CHKERRA(ierr);

{
  AO              ao;
  Mat             adj;
  Partitioning    part;
  IS              is,isg;

  ierr = AODataKeyGetAdjacency(appctx->aodata,"cell",&adj);CHKERRA(ierr);
  ierr = PartitioningCreate(appctx->comm,&part);CHKERRA(ierr);
  ierr = PartitioningSetAdjacency(part,adj);CHKERRA(ierr);
  ierr = PartitioningSetFromOptions(part);CHKERRA(ierr);
  ierr = PartitioningApply(part,&is);CHKERRA(ierr);
  ierr = PartitioningDestroy(part); CHKERRA(ierr);
  ierr = ISPartitioningToNumbering(is,&isg);CHKERRA(ierr);

  ierr = AOCreateBasicIS(isg,PETSC_NULL,&ao);CHKERRA(ierr);
  ierr = ISDestroy(is);CHKERRA(ierr);

  ierr = DrawSynchronizedClear(appctx->drawglobal);CHKERRA(ierr);
  ierr = DrawClear(appctx->drawlocal);CHKERRA(ierr);

  ierr = AODataKeyRemap(appctx->aodata,"cell",ao);CHKERRA(ierr);
  ierr = AppCtxSetLocal(appctx); CHKERRA(ierr);
  ierr = AppCtxView(appctx); CHKERRA(ierr);
  ierr = AODestroy(ao);CHKERRA(ierr);
}

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

  ierr = DrawOpenX(PETSC_COMM_WORLD,PETSC_NULL,"Total Grid",PETSC_DECIDE,PETSC_DECIDE,300,300,
                  &(*appctx)->drawglobal); CHKERRQ(ierr);
  ierr = DrawOpenX(PETSC_COMM_WORLD,PETSC_NULL,"Local Grids",PETSC_DECIDE,PETSC_DECIDE,300,300,
                   &(*appctx)->drawlocal);CHKERRQ(ierr);
  ierr = DrawSplitViewPort((*appctx)->drawlocal);CHKERRQ(ierr);


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

  int                    cell_n,edge_n,vertex_n;

  /*
        These contain the edge and vertex lists in local numbering
  */ 
  int                    *cell_edge,*cell_vertex;
  int                    *edge_vertex;

  /* 
        These contain the global numbering for local objects
  */
  int                    *cell_global,*edge_global,*vertex_global;
  
  double                 *vertex_value;

  ISLocalToGlobalMapping ltogvertex,ltogcell,ltogedge;

  int                    ierr,rstart,rend,rank;
  IS                     iscell,isvertex,isedge,istmp;

  MPI_Comm_rank(appctx->comm,&rank);

  /* 
     Determine if cells are triangles or quads
  */
  ierr = AODataSegmentGetInfo(ao,"cell","cell",0,0,&lgrid->ncell,0);CHKERRQ(ierr);

  /*
      Generate the list of on processor cells 
  */
  ierr = AODataKeyGetInfoOwnership(ao,"cell",&rstart,&rend);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_WORLD,rend-rstart,rstart,1,&istmp);CHKERRQ(ierr);
  lgrid->cell_nlocal   = rend - rstart;
  ierr = AODataKeyGetNeighborsIS(ao,"cell",istmp,&iscell);CHKERRQ(ierr);
  ierr = ISDestroy(istmp);CHKERRQ(ierr);

  /*
       Get the list of vertices and edges used by those cells 
  */
  ierr = AODataSegmentGetReducedIS(ao,"cell","edge",iscell,&isedge);CHKERRQ(ierr);
  ierr = AODataSegmentGetReducedIS(ao,"cell","vertex",iscell,&isvertex);CHKERRQ(ierr);

  /* 
      Make local to global mapping of cell, vertices, and edges 
  */
  ierr = ISLocalToGlobalMappingCreateIS(iscell,&ltogcell);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreateIS(isvertex,&ltogvertex);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreateIS(isedge,&ltogedge);CHKERRQ(ierr);
  
  ierr = AODataKeyAddLocalToGlobalMapping(ao,"cell",ltogcell);CHKERRQ(ierr);
  ierr = AODataKeyAddLocalToGlobalMapping(ao,"edge",ltogedge);CHKERRQ(ierr);
  ierr = AODataKeyAddLocalToGlobalMapping(ao,"vertex",ltogvertex);CHKERRQ(ierr);

  /*
      Get the local edge and vertex lists
  */
  ierr = AODataSegmentGetLocalIS(ao,"cell","edge",iscell,(void **)&cell_edge);CHKERRQ(ierr);
  ierr = AODataSegmentGetLocalIS(ao,"cell","vertex",iscell,(void **)&cell_vertex);CHKERRQ(ierr);
  ierr = AODataSegmentGetLocalIS(ao,"edge","vertex",isedge,(void **)&edge_vertex);CHKERRQ(ierr);

  /* 
      Get the numerical values of all vertices for local cells 
  */
  ierr = AODataSegmentGetIS(ao,"vertex","values",isvertex,(void **)&vertex_value);CHKERRQ(ierr);

  /*
      Get the size and global indices of local objects for plotting
  */
  ierr = ISGetSize(iscell,&cell_n); CHKERRQ(ierr);
  ierr = ISGetSize(isedge,&edge_n); CHKERRQ(ierr);
  ierr = ISGetSize(isvertex,&vertex_n); CHKERRQ(ierr);
  ierr = ISGetIndices(iscell,&cell_global); CHKERRQ(ierr);
  ierr = ISGetIndices(isedge,&edge_global); CHKERRQ(ierr);
  ierr = ISGetIndices(isvertex,&vertex_global); CHKERRQ(ierr);

  lgrid->cell_n        = cell_n; 
  lgrid->cell_edge     = cell_edge;
  lgrid->cell_vertex   = cell_vertex; 
  lgrid->cell_global   = cell_global;

  lgrid->edge_n        = edge_n;
  lgrid->edge_vertex   = edge_vertex; 
  lgrid->edge_global   = edge_global;

  lgrid->vertex_n      = vertex_n;
  lgrid->vertex_global = vertex_global;
  lgrid->vertex_value  = vertex_value;

  PetscFunctionReturn(0);

}

#undef __FUNC__
#define __FUNC__ "AppCxtDestroy"
int AppctxDestroy(AppCtx *appctx)
{
  int ierr;
  ierr = AODataDestroy(appctx->aodata);CHKERRQ(ierr);

  ierr = DrawDestroy(appctx->drawglobal); CHKERRQ(ierr);
  ierr = DrawDestroy(appctx->drawlocal); CHKERRQ(ierr);

  PetscFree(appctx);
  PetscFunctionReturn(0);
}


#undef __FUNC__
#define __FUNC__ "AppCxtView_Private"
int AppCtxView_Private(AppCtx *appctx)
{
  LocalGrid              *lgrid = &appctx->lgrid;

  int                    ncell,cell_n,edge_n,vertex_n,cell_nlocal;

  /*
        These contain the edge and vertex lists in local numbering
  */ 
  int                    *cell_edge,*cell_vertex;
  int                    *edge_vertex;

  /* 
        These contain the global numbering for local objects
  */
  int                    *cell_global,*edge_global,*vertex_global;
  
  double                 *vertex_value;


  int                    ierr,i,rank,c,j;
  int                    ij;

  Draw                   drawglobal = appctx->drawglobal;
  Draw                   drawlocal = appctx->drawlocal;
  double                 xl,yl,xr,yr,xm,ym,xp,yp;
  char                   num[5];

  ierr = DrawCheckResizedWindow(drawglobal); CHKERRQ(ierr);
  ierr = DrawCheckResizedWindow(drawlocal); CHKERRQ(ierr);


  MPI_Comm_rank(appctx->comm,&rank); c = rank + 2;

  ncell         = lgrid->ncell;

  cell_nlocal   = lgrid->cell_nlocal;
  cell_n        = lgrid->cell_n;
  cell_edge     = lgrid->cell_edge;
  cell_vertex   = lgrid->cell_vertex;
  cell_global   = lgrid->cell_global;

  edge_n        = lgrid->edge_n;
  edge_vertex   = lgrid->edge_vertex;
  edge_global   = lgrid->edge_global;

  vertex_n      = lgrid->vertex_n;
  vertex_global = lgrid->vertex_global;
  vertex_value  = lgrid->vertex_value;

  /*
        Draw edges of local cells and number them
     First do ghost cells
  */
  for (i=cell_nlocal; i<cell_n; i++ ) {
    xp = 0.0; yp = 0.0;
    xl = vertex_value[2*cell_vertex[ncell*i]]; yl = vertex_value[2*cell_vertex[ncell*i] + 1];
    for (j=0; j<ncell; j++) {
      ij = ncell*i + ((j+1) % ncell);
      xr = vertex_value[2*cell_vertex[ij]]; yr = vertex_value[2*cell_vertex[ij] + 1];
      ierr = DrawLine(drawlocal,xl,yl,xr,yr,DRAW_GRAY);CHKERRQ(ierr);
      xm = .5*(xr+xl); ym = .5*(yr+yl);
      sprintf(num,"%d",cell_edge[ncell*i+j]);
      ierr = DrawString(drawlocal,xm,ym,DRAW_GRAY,num);
      xp += xl;         yp += yl;
      xl  = xr;         yl =  yr;
    }
    xp /= ncell; yp /= ncell;
    sprintf(num,"%d",i);
    ierr = DrawString(drawlocal,xp,yp,DRAW_GRAY,num);
  }
  /* Second do the local cells */
  for (i=0; i<cell_nlocal; i++ ) {
    xp = 0.0; yp = 0.0;
    xl = vertex_value[2*cell_vertex[ncell*i]]; yl = vertex_value[2*cell_vertex[ncell*i] + 1];
    for (j=0; j<ncell; j++) {
      ij = ncell*i + ((j+1) % ncell);
      xr = vertex_value[2*cell_vertex[ij]]; yr = vertex_value[2*cell_vertex[ij] + 1];
      ierr = DrawLine(drawglobal,xl,yl,xr,yr,c);CHKERRQ(ierr);
      ierr = DrawLine(drawlocal,xl,yl,xr,yr,DRAW_BLUE);CHKERRQ(ierr);
      xm = .5*(xr+xl); ym = .5*(yr+yl);
      sprintf(num,"%d",cell_edge[ncell*i+j]);
      ierr = DrawString(drawlocal,xm,ym,DRAW_RED,num);
      sprintf(num,"%d",edge_global[cell_edge[ncell*i+j]]);
      ierr = DrawString(drawglobal,xm,ym,DRAW_RED,num);
      xp += xl;         yp += yl;
      xl  = xr;         yl =  yr;
    }
    xp /= ncell; yp /= ncell;
    sprintf(num,"%d",i);
    ierr = DrawString(drawlocal,xp,yp,DRAW_GREEN,num);
    sprintf(num,"%d",cell_global[i]);
    ierr = DrawString(drawglobal,xp,yp,c,num);
  }

  /*
      Number vertices
  */
  for (i=0; i<vertex_n; i++ ) {
    xm = vertex_value[2*i]; ym = vertex_value[2*i + 1];
    sprintf(num,"%d",i);
    ierr = DrawString(drawlocal,xm,ym,DRAW_BLUE,num);
    sprintf(num,"%d",vertex_global[i]);
    ierr = DrawString(drawglobal,xm,ym,DRAW_BLUE,num);
  }
  ierr = DrawSynchronizedFlush(drawglobal);CHKERRQ(ierr);
  ierr = DrawSynchronizedFlush(drawlocal);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNC__
#define __FUNC__ "AppCxtView"
int AppCtxView(AppCtx *appctx)
{
  int        ierr,pause;
  DrawButton button;
  double     xc,yc,scale,w,h,xr,xl,yr,yl,xmin,xmax,ymin,ymax;
  Draw       draw = appctx->drawglobal,drawlocal = appctx->drawlocal;

  PetscFunctionBegin;
  ierr = DrawSynchronizedClear(draw);CHKERRQ(ierr);
  ierr = DrawSynchronizedClear(drawlocal);CHKERRQ(ierr);
  ierr = AppCtxView_Private(appctx); CHKERRQ(ierr);

  DrawGetPause(draw,&pause);
  if (pause >= 0) { PetscSleep(pause); PetscFunctionReturn(0);}

  ierr = DrawCheckResizedWindow(draw); CHKERRQ(ierr);
  ierr = DrawSynchronizedGetMouseButton(draw,&button,&xc,&yc,0,0); CHKERRQ(ierr); 
  ierr = DrawGetCoordinates(draw,&xl,&yl,&xr,&yr); CHKERRQ(ierr);
  w    = xr - xl; xmin = xl; ymin = yl; xmax = xr; ymax = yr;
  h    = yr - yl;

  while (button != BUTTON_RIGHT) {

    ierr = DrawSynchronizedClear(draw);CHKERRQ(ierr);
    ierr = DrawSynchronizedClear(drawlocal);CHKERRQ(ierr);
    if (button == BUTTON_LEFT)        scale = .5;
    else if (button == BUTTON_CENTER) scale = 2.;
    xl = scale*(xl + w - xc) + xc - w*scale;
    xr = scale*(xr - w - xc) + xc + w*scale;
    yl = scale*(yl + h - yc) + yc - h*scale;
    yr = scale*(yr - h - yc) + yc + h*scale;
    w *= scale; h *= scale;
    ierr = DrawSetCoordinates(draw,xl,yl,xr,yr); CHKERRQ(ierr);
    ierr = DrawSetCoordinates(drawlocal,xl,yl,xr,yr); CHKERRQ(ierr);

    ierr = AppCtxView_Private(appctx); CHKERRQ(ierr);
    ierr = DrawCheckResizedWindow(draw); CHKERRQ(ierr);
    ierr = DrawSynchronizedGetMouseButton(draw,&button,&xc,&yc,0,0);  CHKERRQ(ierr);
  }

  ierr = DrawSetCoordinates(draw,xmin,ymin,xmax,ymax); CHKERRQ(ierr);
  ierr = DrawSetCoordinates(drawlocal,xmin,ymin,xmax,ymax); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
