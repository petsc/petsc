
static char help[] ="Solves a simple linear PDE on an unstructured grid\n";


/*

*/

#include "ao.h"
#include "mat.h"

/*
        cell_n        - number of cells on this processor 
      
        cell_vertex   - vertices of the cells (in local numbering)
        cell_global   - global number of each cell on this processor

        vertex_n      - number of vertices on this processor
        vertex_global - global number of each vertex on this processor
        vertex_value  - x,y coordinates of vertices on this processor
        ltog          - mapping from local numbering of vertices (including ghosts)
                        to global

        vertex_n_ghosted - number of vertices including ghost vertices
*/
        
typedef struct {
  int                    cell_n;
  int                    *cell_vertex;
  IS                     cell_global;
  int                    vertex_n,vertex_n_ghosted;
  IS                     vertex_global;
  double                 *vertex_value;
  BT                     vertex_boundary_flag;
  int                    ncell;
  ISLocalToGlobalMapping ltog;
  IS                     vertex_boundary;
} AppGrid;

/*
    gtol    - global to local vector scatter
              (used to move data from x to w_local for example
    A       - parallel sparse stiffness matrix
    b       - parallel vector contains right hand side
    x       - parallel vector contains solution
    w_local - sequential vector contains local plus ghosted part of load
              (during load assembly stage)
*/
typedef struct {
  Vec                    b,x;
  Vec                    w_local,x_local;  /* local ghosted work vectors */
  VecScatter             gtol;
  Mat                    A;
} AppAlgebra;

/*
    drawlocal    - window where processor local portion is drawn
    drawglobal   - window where entire grid is drawn

    shownumbers  - print the vertex and cell numbers 
    showvertices - draw the vertices as points
    showelements - draw the elements 
    showboundary - draw boundary of domain
    showboundaryvertices - draw points on boundary of domain
*/

typedef struct {
  Draw       drawlocal;
  Draw       drawglobal;
  int        shownumbers,showvertices,showelements,showboundary;
  int        showboundaryvertices;
} AppView;

typedef struct {
  MPI_Comm   comm;
  AOData     aodata;
  AppGrid    grid;
  AppAlgebra algebra;
  AppView    view;
} AppCtx;


extern int AppCtxView(Draw,void*);
extern int AppCtxViewSolution(Draw,void*);
extern int AppctxDestroy(AppCtx *);
extern int AppCtxSetLocal(AppCtx *);
extern int AppCtxCreate(MPI_Comm,AppCtx **);
extern int AppCtxSolve(AppCtx*);

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
  ierr = AppCtxCreate(PETSC_COMM_WORLD,&appctx); CHKERRA(ierr);

  /*
      Partition the grid cells
  */
  ierr = AODataKeyPartition(appctx->aodata,"cell"); CHKERRA(ierr);  

  /*
      Partition the vertices subservient to the cells
  */ 
  ierr = AODataSegmentPartition(appctx->aodata,"cell","vertex"); CHKERRA(ierr);  

  /*
     Generate the local numbering of cells and vertices
  */
  ierr = AppCtxSetLocal(appctx); CHKERRA(ierr);

  /*
     Visualize the grid 
  */
  if (appctx->view.shownumbers || appctx->view.showvertices || appctx->view.showelements ||
      appctx->view.showboundary || appctx->view.showboundaryvertices)  {
    ierr = DrawZoom(appctx->view.drawglobal,AppCtxView,appctx); CHKERRA(ierr);
  }

  ierr = AppCtxSolve(appctx);CHKERRQ(ierr);

  ierr = DrawZoom(appctx->view.drawglobal,AppCtxViewSolution,appctx); CHKERRA(ierr);

  ierr = AppctxDestroy(appctx); CHKERRA(ierr);

  PetscFinalize();

  PetscFunctionReturn(0);
}


#undef __FUNC__
#define __FUNC__ "AppCxtSolve"
int AppCtxSolve(AppCtx* appctx)
{
  AppGrid              *grid = &appctx->grid;
  Vec                    x,b,w_local;
  int                    ierr, i, cell_n = grid->cell_n, ncell = grid->ncell;
  int                    *cell_vertex = grid->cell_vertex;
  Scalar                 *values;
  AppAlgebra             *algebra = &appctx->algebra;
  VecScatter             gtol;

  PetscFunctionBegin;

  /*
        Create vector to contain load
  */
  ierr = VecCreateMPI(appctx->comm,grid->vertex_n,PETSC_DECIDE,&b);CHKERRQ(ierr);
  ierr = VecSetLocalToGlobalMapping(b,grid->ltog);CHKERRQ(ierr);

  ierr = VecDuplicate(b,&x);

  values = (Scalar *) PetscMalloc(ncell*sizeof(Scalar));CHKPTRQ(values);
  for ( i=0; i<ncell; i++ ) values[i] = 1.0;

  for ( i=0; i<cell_n; i++ ) {
    ierr = VecSetValuesLocal(b,ncell,cell_vertex+ncell*i,values,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b);CHKERRQ(ierr);
  
  ierr = VecCreateSeq(PETSC_COMM_SELF,grid->vertex_n_ghosted,&w_local);CHKERRQ(ierr);
  ierr = VecDuplicate(w_local,&algebra->x_local);CHKERRQ(ierr);

  ierr = VecScatterCreate(b,grid->vertex_global,w_local,0,&gtol);CHKERRQ(ierr);

  ierr = VecScatterBegin(b,w_local,INSERT_VALUES,SCATTER_FORWARD,gtol);CHKERRQ(ierr);
  ierr = VecScatterEnd(b,w_local,INSERT_VALUES,SCATTER_FORWARD,gtol);CHKERRQ(ierr);

  /* ierr = VecView(w_local,0); */

  algebra->x       = x;
  algebra->b       = b;
  algebra->w_local = w_local;
  algebra->gtol    = gtol;

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
  ierr = OptionsGetString(0,"-f",filename,256,&flag);CHKERRQ(ierr);
  if (!flag) PetscStrcpy(filename,"gridfile");
  ierr = ViewerFileOpenBinary((*appctx)->comm,filename,BINARY_RDONLY,&binary);CHKERRQ(ierr);
  ierr = AODataLoadBasic(binary,&(*appctx)->aodata); CHKERRQ(ierr);
  ierr = ViewerDestroy(binary); CHKERRQ(ierr);

  /*---------------------------------------------------------------------
     Open the graphics windows
     ------------------------------------------------------------------------*/

  ierr = DrawOpenX(PETSC_COMM_WORLD,PETSC_NULL,"Total Grid",PETSC_DECIDE,PETSC_DECIDE,400,400,
                  &(*appctx)->view.drawglobal); CHKERRQ(ierr);
  ierr = DrawOpenX(PETSC_COMM_WORLD,PETSC_NULL,"Local Grids",PETSC_DECIDE,PETSC_DECIDE,400,400,
                   &(*appctx)->view.drawlocal);CHKERRQ(ierr);
  ierr = DrawSplitViewPort((*appctx)->view.drawlocal);CHKERRQ(ierr);

  ierr = OptionsHasName(PETSC_NULL,"-show_numbers",&(*appctx)->view.shownumbers);CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-show_elements",&(*appctx)->view.showelements);CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-show_vertices",&(*appctx)->view.showvertices);CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-show_boundary",&(*appctx)->view.showboundary);CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-show_boundary_vertices",
         &(*appctx)->view.showboundaryvertices);CHKERRQ(ierr);


  /*
       Set the window coordinates based on the values in vertices
  */
  ierr = AODataSegmentGetExtrema((*appctx)->aodata,"vertex","values",maxs,mins);CHKERRQ(ierr);
  hx = maxs[0] - mins[0]; xmin = mins[0] - .1*hx; xmax = maxs[0] + .1*hx;
  hy = maxs[1] - mins[1]; ymin = mins[1] - .1*hy; ymax = maxs[1] + .1*hy;
  ierr = DrawSetCoordinates((*appctx)->view.drawglobal,xmin,ymin,xmax,ymax);CHKERRQ(ierr);
  ierr = DrawSetCoordinates((*appctx)->view.drawlocal,xmin,ymin,xmax,ymax);CHKERRQ(ierr);

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

  ISLocalToGlobalMapping ltogcell;

  int                    ierr,rstart,rend,rank;
  IS                     iscell,isvertex;

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

  /* 
      Get the numerical values of all vertices for local vertices
  */
  ierr = AODataSegmentGetIS(ao,"vertex","values",isvertex,(void **)&vertex_value);CHKERRQ(ierr);

  /* 
      Get the bit flag indicating boundary for local vertices
  */
  ierr = AODataSegmentGetIS(ao,"vertex","boundary",isvertex,(void **)&vertex_boundary_flag);CHKERRQ(ierr);

  /*
      Get the size of local objects for plotting
  */
  ierr = ISGetSize(iscell,&grid->cell_n); CHKERRQ(ierr);
  ierr = ISGetSize(isvertex,&grid->vertex_n_ghosted); CHKERRQ(ierr);

  grid->cell_vertex          = cell_vertex; 
  grid->cell_global          = iscell;

  grid->vertex_global        = isvertex;
  grid->vertex_value         = vertex_value;
  grid->vertex_boundary_flag = vertex_boundary_flag;

  ierr = AODataKeyGetInfo(ao,"vertex",PETSC_NULL,&grid->vertex_n,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = AODataSegmentGetInfo(ao,"cell","vertex",&grid->ncell,0);CHKERRQ(ierr);

  /*
      Generate a list of local vertices that are on the boundary
  */
  {
  int *vertices;
  ierr = ISGetIndices(isvertex,&vertices);CHKERRQ(ierr);
  ierr = AODataKeyGetActiveLocal(ao,"vertex","boundary",grid->vertex_n_ghosted,vertices,0,&grid->vertex_boundary);
  CHKERRQ(ierr);
  ierr = ISRestoreIndices(isvertex,&vertices);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);

}

#undef __FUNC__
#define __FUNC__ "AppCxtDestroy"
int AppctxDestroy(AppCtx *appctx)
{
  int      ierr;
  AOData   ao = appctx->aodata;
  AppGrid  *grid = &appctx->grid;

  ierr = AODataSegmentRestoreIS(ao,"vertex","values",PETSC_NULL,(void **)&grid->vertex_value);CHKERRQ(ierr);
  ierr = AODataSegmentRestoreLocalIS(ao,"cell","vertex",PETSC_NULL,(void **)&grid->cell_vertex);CHKERRQ(ierr);
  ierr = AODataDestroy(ao);CHKERRQ(ierr);

  ierr = DrawDestroy(appctx->view.drawglobal); CHKERRQ(ierr);
  ierr = DrawDestroy(appctx->view.drawlocal); CHKERRQ(ierr);

  ierr = ISDestroy(appctx->grid.vertex_global);CHKERRQ(ierr);
  ierr = ISDestroy(appctx->grid.cell_global);CHKERRQ(ierr);

  ierr = ISLocalToGlobalMappingDestroy(appctx->grid.ltog);CHKERRQ(ierr);
  PetscFree(appctx);
  PetscFunctionReturn(0);
}


#undef __FUNC__
#define __FUNC__ "AppCxtView"
int AppCtxView(Draw idraw,void *iappctx)
{
  AppCtx                 *appctx = (AppCtx *)iappctx;
  AppGrid                *grid = &appctx->grid;

  int                    cell_n,vertex_n,ncell,*verts,nverts;

  /*
        These contain the edge and vertex lists in local numbering
  */ 
  int                    *cell_vertex;

  /* 
        These contain the global numbering for local objects
  */
  int                    *cell_global,*vertex_global;
  
  double                 *vertex_value;

  BT                     vertex_boundary_flag;

  int                    ierr,i,rank,c,j,ijp;
  int                    ij,shownumbers = appctx->view.shownumbers;
  int                    showelements = appctx->view.showelements;
  int                    showvertices = appctx->view.showvertices;
  int                    showboundary = appctx->view.showboundary;
  int                    showboundaryvertices = appctx->view.showboundaryvertices;

  Draw                   drawglobal = appctx->view.drawglobal;
  Draw                   drawlocal = appctx->view.drawlocal;
  double                 xl,yl,xr,yr,xm,ym,xp,yp;
  char                   num[5];

  /* get number of vertices per cell */
  ierr = AODataSegmentGetInfo(appctx->aodata,"cell","vertex",&ncell,0);CHKERRQ(ierr);

  ierr = DrawCheckResizedWindow(drawglobal); CHKERRQ(ierr);
  ierr = DrawCheckResizedWindow(drawlocal); CHKERRQ(ierr);

  MPI_Comm_rank(appctx->comm,&rank); c = rank + 2;

  cell_n        = grid->cell_n;
  ierr = ISGetIndices(grid->cell_global,&cell_global); CHKERRQ(ierr);
  ierr = ISGetIndices(grid->vertex_global,&vertex_global); CHKERRQ(ierr);

  cell_vertex     = grid->cell_vertex;
  vertex_n        = grid->vertex_n;
  vertex_value    = grid->vertex_value;
  vertex_boundary_flag = grid->vertex_boundary_flag;

  /*
        Draw edges of local cells and number them
     First do ghost cells
  */
  /* do the local cells */
  if (showelements) {
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
      if (shownumbers) {ierr = DrawString(drawlocal,xp,yp,DRAW_GREEN,num);CHKERRQ(ierr);}
      sprintf(num,"%d",cell_global[i]);
      if (shownumbers) {ierr = DrawString(drawglobal,xp,yp,c,num);CHKERRQ(ierr);}
    }
  }

  /*
       Draws only boundary edges 
  */
  if (showboundary) {
    for (i=0; i<cell_n; i++ ) {
      xp = 0.0; yp = 0.0;
      xl  = vertex_value[2*cell_vertex[ncell*i]]; yl = vertex_value[2*cell_vertex[ncell*i] + 1];
      ijp = ncell*i;
      for (j=0; j<ncell; j++) {
        ij = ncell*i + ((j+1) % ncell);
        xr = vertex_value[2*cell_vertex[ij]]; yr = vertex_value[2*cell_vertex[ij] + 1];
        if (BTLookup(vertex_boundary_flag,cell_vertex[ijp]) && BTLookup(vertex_boundary_flag,cell_vertex[ij])) {
          ierr = DrawLine(drawglobal,xl,yl,xr,yr,c);CHKERRQ(ierr);
          ierr = DrawLine(drawlocal,xl,yl,xr,yr,DRAW_BLUE);CHKERRQ(ierr);
        }
        xp += xl;         yp += yl;
        xl  = xr;         yl =  yr;
        ijp = ij;
      }
      xp /= ncell; yp /= ncell;
      sprintf(num,"%d",i);
      if (shownumbers) {ierr = DrawString(drawlocal,xp,yp,DRAW_GREEN,num);CHKERRQ(ierr);}
      sprintf(num,"%d",cell_global[i]);
      if (shownumbers) {ierr = DrawString(drawglobal,xp,yp,c,num);CHKERRQ(ierr);}
    }
  }

  /*
      Number vertices
  */
  if (showvertices) {
    for (i=0; i<vertex_n; i++ ) {
      xm = vertex_value[2*i]; ym = vertex_value[2*i + 1];
      ierr = DrawString(drawglobal,xm,ym,DRAW_BLUE,num);CHKERRQ(ierr);
      ierr = DrawPoint(drawglobal,xm,ym,DRAW_ORANGE);CHKERRQ(ierr);
      ierr = DrawPoint(drawlocal,xm,ym,DRAW_ORANGE);CHKERRQ(ierr);
    }
  }
  if (shownumbers) {
    for (i=0; i<vertex_n; i++ ) {
      xm = vertex_value[2*i]; ym = vertex_value[2*i + 1];
      sprintf(num,"%d",i);
      ierr = DrawString(drawlocal,xm,ym,DRAW_BLUE,num);CHKERRQ(ierr);
      sprintf(num,"%d",vertex_global[i]);
      ierr = DrawString(drawglobal,xm,ym,DRAW_BLUE,num);CHKERRQ(ierr);
    }
  }

  if (showboundaryvertices) {
    ierr = ISGetSize(grid->vertex_boundary,&nverts);CHKERRQ(ierr);
    ierr = ISGetIndices(grid->vertex_boundary,&verts);CHKERRQ(ierr);
    for ( i=0; i<nverts; i++ ) {
      xm = vertex_value[2*verts[i]]; ym = vertex_value[2*verts[i] + 1];
      ierr = DrawPoint(drawglobal,xm,ym,DRAW_RED);CHKERRQ(ierr);
      ierr = DrawPoint(drawlocal,xm,ym,DRAW_RED);CHKERRQ(ierr);
    }
    ierr = ISRestoreIndices(grid->vertex_boundary,&verts);CHKERRQ(ierr);
  }
  ierr = DrawSynchronizedFlush(drawglobal);CHKERRQ(ierr);
  ierr = DrawSynchronizedFlush(drawlocal);CHKERRQ(ierr);

  ierr = ISRestoreIndices(grid->cell_global,&cell_global); CHKERRQ(ierr);
  ierr = ISRestoreIndices(grid->vertex_global,&vertex_global); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNC__
#define __FUNC__ "AppCxtViewSolution"
int AppCtxViewSolution(Draw idraw,void *iappctx)
{
  AppCtx                 *appctx = (AppCtx *)iappctx;
  AppGrid                *grid = &appctx->grid;

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


  int                    ierr,i;

  Draw                   drawglobal = appctx->view.drawglobal;
  Draw                   drawlocal = appctx->view.drawlocal,popup;
  double                 x0,x1,x2,y0,y1,y2,c0,c1,c2,vmin,vmax;

  Scalar                 *values;

  /* get number of vertices per cell */
  ierr = AODataSegmentGetInfo(appctx->aodata,"cell","vertex",&ncell,0);CHKERRQ(ierr);

  ierr = DrawCheckResizedWindow(drawglobal); CHKERRQ(ierr);
  ierr = DrawCheckResizedWindow(drawlocal); CHKERRQ(ierr);


  cell_n        = grid->cell_n;
  ierr = ISGetIndices(grid->cell_global,&cell_global); CHKERRQ(ierr);
  ierr = ISGetIndices(grid->vertex_global,&vertex_global); CHKERRQ(ierr);

  cell_vertex   = grid->cell_vertex;
  vertex_n      = grid->vertex_n;
  vertex_value  = grid->vertex_value;

  ierr = VecMin(appctx->algebra.b,PETSC_NULL,&vmin);CHKERRQ(ierr);
  ierr = VecMax(appctx->algebra.b,PETSC_NULL,&vmax);CHKERRQ(ierr);
  
  ierr = VecCopy(appctx->algebra.w_local,appctx->algebra.x_local);CHKERRQ(ierr);
  ierr = VecContourScale(appctx->algebra.x_local,vmin,vmax);CHKERRQ(ierr);
  ierr = DrawGetPopup(drawglobal,&popup);CHKERRQ(ierr);
  ierr = DrawScalePopup(popup,vmin,vmax);CHKERRQ(ierr);

  ierr = VecGetArray(appctx->algebra.x_local,&values);CHKERRQ(ierr);

  /*
      Handle linear (triangular elements)
  */
  if (ncell == 3) {
    for (i=0; i<cell_n; i++ ) {
      x0 = vertex_value[2*cell_vertex[ncell*i]];   y0 = vertex_value[2*cell_vertex[ncell*i] + 1];
      x1 = vertex_value[2*cell_vertex[ncell*i+1]]; y1 = vertex_value[2*cell_vertex[ncell*i+1] + 1];
      x2 = vertex_value[2*cell_vertex[ncell*i+2]]; y2 = vertex_value[2*cell_vertex[ncell*i+2] + 1];
      c0 = values[cell_vertex[ncell*i]];
      c1 = values[cell_vertex[ncell*i+1]];
      c2 = values[cell_vertex[ncell*i+2]];
      ierr = DrawTriangle(drawglobal,x0,y0,x1,y1,x2,y2,c0,c1,c2);CHKERRQ(ierr);
    }
  /*
      Handle bilinear (quadrilateral elements)
  */
  } else if (ncell == 4) {
    for (i=0; i<cell_n; i++ ) {
      x0 = vertex_value[2*cell_vertex[ncell*i]];   y0 = vertex_value[2*cell_vertex[ncell*i] + 1];
      x1 = vertex_value[2*cell_vertex[ncell*i+1]]; y1 = vertex_value[2*cell_vertex[ncell*i+1] + 1];
      x2 = vertex_value[2*cell_vertex[ncell*i+2]]; y2 = vertex_value[2*cell_vertex[ncell*i+2] + 1];
      c0 = values[cell_vertex[ncell*i]];
      c1 = values[cell_vertex[ncell*i+1]];
      c2 = values[cell_vertex[ncell*i+2]];
      ierr = DrawTriangle(drawglobal,x0,y0,x1,y1,x2,y2,c0,c1,c2);CHKERRQ(ierr);
      x0 = vertex_value[2*cell_vertex[ncell*i]];   y0 = vertex_value[2*cell_vertex[ncell*i] + 1];
      x1 = vertex_value[2*cell_vertex[ncell*i+3]]; y1 = vertex_value[2*cell_vertex[ncell*i+3] + 1];
      x2 = vertex_value[2*cell_vertex[ncell*i+2]]; y2 = vertex_value[2*cell_vertex[ncell*i+2] + 1];
      c0 = values[cell_vertex[ncell*i]];
      c1 = values[cell_vertex[ncell*i+3]];
      c2 = values[cell_vertex[ncell*i+2]];
      ierr = DrawTriangle(drawglobal,x0,y0,x1,y1,x2,y2,c0,c1,c2);CHKERRQ(ierr);
    }
  /*
      Handle quadratic (triangular elements)
  */
  } else if (ncell == 6) {
    for (i=0; i<cell_n; i++ ) {
      x0 = vertex_value[2*cell_vertex[ncell*i]];   y0 = vertex_value[2*cell_vertex[ncell*i] + 1];
      x1 = vertex_value[2*cell_vertex[ncell*i+1]]; y1 = vertex_value[2*cell_vertex[ncell*i+1] + 1];
      x2 = vertex_value[2*cell_vertex[ncell*i+5]]; y2 = vertex_value[2*cell_vertex[ncell*i+5] + 1];
      c0 = values[cell_vertex[ncell*i]];
      c1 = values[cell_vertex[ncell*i+1]];
      c2 = values[cell_vertex[ncell*i+5]];
      ierr = DrawTriangle(drawglobal,x0,y0,x1,y1,x2,y2,c0,c1,c2);CHKERRQ(ierr);
      x0 = vertex_value[2*cell_vertex[ncell*i+1]]; y0 = vertex_value[2*cell_vertex[ncell*i+1] + 1];
      x1 = vertex_value[2*cell_vertex[ncell*i+2]]; y1 = vertex_value[2*cell_vertex[ncell*i+2] + 1];
      x2 = vertex_value[2*cell_vertex[ncell*i+3]]; y2 = vertex_value[2*cell_vertex[ncell*i+3] + 1];
      c0 = values[cell_vertex[ncell*i+1]];
      c1 = values[cell_vertex[ncell*i+2]];
      c2 = values[cell_vertex[ncell*i+3]];
      ierr = DrawTriangle(drawglobal,x0,y0,x1,y1,x2,y2,c0,c1,c2);CHKERRQ(ierr);
      x0 = vertex_value[2*cell_vertex[ncell*i+1]]; y0 = vertex_value[2*cell_vertex[ncell*i+1] + 1];
      x1 = vertex_value[2*cell_vertex[ncell*i+5]]; y1 = vertex_value[2*cell_vertex[ncell*i+5] + 1];
      x2 = vertex_value[2*cell_vertex[ncell*i+3]]; y2 = vertex_value[2*cell_vertex[ncell*i+3] + 1];
      c0 = values[cell_vertex[ncell*i+1]];
      c1 = values[cell_vertex[ncell*i+5]];
      c2 = values[cell_vertex[ncell*i+3]];
      ierr = DrawTriangle(drawglobal,x0,y0,x1,y1,x2,y2,c0,c1,c2);CHKERRQ(ierr);
      x0 = vertex_value[2*cell_vertex[ncell*i+4]]; y0 = vertex_value[2*cell_vertex[ncell*i+4] + 1];
      x1 = vertex_value[2*cell_vertex[ncell*i+5]]; y1 = vertex_value[2*cell_vertex[ncell*i+5] + 1];
      x2 = vertex_value[2*cell_vertex[ncell*i+3]]; y2 = vertex_value[2*cell_vertex[ncell*i+3] + 1];
      c0 = values[cell_vertex[ncell*i+4]];
      c1 = values[cell_vertex[ncell*i+5]];
      c2 = values[cell_vertex[ncell*i+3]];
      ierr = DrawTriangle(drawglobal,x0,y0,x1,y1,x2,y2,c0,c1,c2);CHKERRQ(ierr);
    }
  }   

  ierr = VecRestoreArray(appctx->algebra.x_local,&values);CHKERRQ(ierr);

  ierr = DrawSynchronizedFlush(drawglobal);CHKERRQ(ierr);
  ierr = DrawSynchronizedFlush(drawlocal);CHKERRQ(ierr);

  ierr = ISRestoreIndices(grid->cell_global,&cell_global); CHKERRQ(ierr);
  ierr = ISRestoreIndices(grid->vertex_global,&vertex_global); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

