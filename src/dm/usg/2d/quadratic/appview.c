
/*
    Demonstrates how to draw two-dimensional grids and solutions.
    See appctx.h for information on how this code is to be used.
*/

#include "appctx.h"

/*
    AppCtxViewGrid - Views grid.

    Input Parameters:
    idraw - drawing context
    iappctx - user-defined application contex

    Note:
    This routine is called by DrawZoom().
*/
#undef __FUNC__
#define __FUNC__ "AppCxtViewGrid"
int AppCtxViewGrid(Draw idraw,void *iappctx)
{
  AppCtx                 *appctx = (AppCtx *)iappctx;
  AppGrid                *grid = &appctx->grid;

  int                    cell_n,vertex_n,ncell = 6,*verts,nverts;

  /*
        These contain the  vertex lists in local numbering
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

  ierr = DrawCheckResizedWindow(drawglobal); CHKERRQ(ierr);
  ierr = DrawCheckResizedWindow(drawlocal); CHKERRQ(ierr);

  MPI_Comm_rank(appctx->comm,&rank); c = rank + 2;

  ierr = ISGetIndices(grid->cell_global,&cell_global); CHKERRQ(ierr);
  ierr = ISGetIndices(grid->vertex_global,&vertex_global); CHKERRQ(ierr);

  cell_n               = grid->cell_n;
  cell_vertex          = grid->cell_vertex;
  vertex_n             = grid->vertex_n;
  vertex_value         = grid->vertex_value;
  vertex_boundary_flag = grid->vertex_boundary_flag;

  ierr = ISGetSize(grid->vertex_boundary,&nverts); CHKERRQ(ierr);
  ierr = ISGetIndices(grid->vertex_boundary,&verts); CHKERRQ(ierr);

  /*
        Draw edges of local cells and number them
  */
  if (showelements) {
    for (i=0; i<cell_n; i++ ) {
      xp = 0.0; yp = 0.0;
      xl = vertex_value[2*cell_vertex[ncell*i]]; yl = vertex_value[2*cell_vertex[ncell*i] + 1];
      for (j=0; j<ncell; j++) {
        ij = ncell*i + ((j+1) % ncell);
        xr = vertex_value[2*cell_vertex[ij]]; yr = vertex_value[2*cell_vertex[ij] + 1];
        ierr = DrawLine(drawglobal,xl,yl,xr,yr,c); CHKERRQ(ierr);
        ierr = DrawLine(drawlocal,xl,yl,xr,yr,DRAW_BLUE); CHKERRQ(ierr);
        xp += xl;         yp += yl;
        xl  = xr;         yl =  yr;
      }
      if (shownumbers) {
        xp /= ncell; yp /= ncell;
        sprintf(num,"%d",i);
        ierr = DrawString(drawlocal,xp,yp,DRAW_GREEN,num); CHKERRQ(ierr);
        sprintf(num,"%d",cell_global[i]);
        ierr = DrawString(drawglobal,xp,yp,c,num); CHKERRQ(ierr);
      }
    }
  }

  /*
       Draw only boundary edges 
  */
  if (showboundary) {
    /* loop over all cells on this processor */
    for (i=0; i<cell_n; i++ ) {
      xp = 0.0; yp = 0.0;
      xl  = vertex_value[2*cell_vertex[ncell*i]]; yl = vertex_value[2*cell_vertex[ncell*i] + 1];
      ijp = ncell*i;
      /* loop over all edges of the cell (in this case always 6) */      
      for (j=0; j<ncell; j++) {
        ij = ncell*i + ((j+1) % ncell);
        xr = vertex_value[2*cell_vertex[ij]]; yr = vertex_value[2*cell_vertex[ij] + 1];
        /* are both ends of the edge on the boundary ? Note this can generate incorrect edges */
        if (BTLookup(vertex_boundary_flag,cell_vertex[ijp]) && BTLookup(vertex_boundary_flag,cell_vertex[ij])) {
          ierr = DrawLine(drawglobal,xl,yl,xr,yr,c); CHKERRQ(ierr);
          ierr = DrawLine(drawlocal,xl,yl,xr,yr,DRAW_BLUE); CHKERRQ(ierr);
        }
        xp += xl;         yp += yl;
        xl  = xr;         yl =  yr;
        ijp = ij;
      }
      if (shownumbers) {
        xp /= ncell; yp /= ncell;
        sprintf(num,"%d",i);
        ierr = DrawString(drawlocal,xp,yp,DRAW_GREEN,num); CHKERRQ(ierr);
        sprintf(num,"%d",cell_global[i]);
        ierr = DrawString(drawglobal,xp,yp,c,num); CHKERRQ(ierr);
      }
    }
  }


  /*
      Plot the vertices with little points
  */
  if (showvertices) {
    for (i=0; i<vertex_n; i++ ) {
      xm = vertex_value[2*i]; ym = vertex_value[2*i + 1];
      ierr = DrawString(drawglobal,xm,ym,DRAW_BLUE,num); CHKERRQ(ierr);
      ierr = DrawPoint(drawglobal,xm,ym,DRAW_ORANGE); CHKERRQ(ierr);
      ierr = DrawPoint(drawlocal,xm,ym,DRAW_ORANGE); CHKERRQ(ierr);
      if (shownumbers) {
        sprintf(num,"%d",i);
        ierr = DrawString(drawlocal,xm,ym,DRAW_BLUE,num); CHKERRQ(ierr);
        sprintf(num,"%d",vertex_global[i]);
        ierr = DrawString(drawglobal,xm,ym,DRAW_BLUE,num); CHKERRQ(ierr);
      }
    }
  }

  /*
      Plot the boundary vertices with little points
  */
  if (showboundaryvertices) {
    for ( i=0; i<nverts; i++ ) {
      xm = vertex_value[2*verts[i]]; ym = vertex_value[2*verts[i] + 1];
      ierr = DrawPoint(drawglobal,xm,ym,DRAW_RED); CHKERRQ(ierr);
      ierr = DrawPoint(drawlocal,xm,ym,DRAW_RED); CHKERRQ(ierr);
    }
  }
  ierr = DrawSynchronizedFlush(drawglobal); CHKERRQ(ierr);
  ierr = DrawSynchronizedFlush(drawlocal); CHKERRQ(ierr);

  ierr = ISRestoreIndices(grid->vertex_boundary,&verts); CHKERRQ(ierr);
  ierr = ISRestoreIndices(grid->cell_global,&cell_global); CHKERRQ(ierr);

  ierr = ISRestoreIndices(grid->vertex_global,&vertex_global); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
/* ----------------------------------------------------------------------- */
/*
    AppCtxViewSolution - Views solution.

    Input Parameters:
    idraw - drawing context
    iappctx - user-defined application contex

    Note:
    This routine is called by DrawZoom()
*/
#undef __FUNC__
#define __FUNC__ "AppCxtViewSolution"
int AppCtxViewSolution(Draw idraw,void *iappctx)
{
  AppCtx                 *appctx = (AppCtx *)iappctx;
  AppGrid                *grid = &appctx->grid;
  AppAlgebra             *algebra = &appctx->algebra;
  int                    cell_n,vertex_n,ncell = 6;

  /*
        These contain the vertex lists in local numbering
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

  ierr = DrawCheckResizedWindow(drawglobal); CHKERRQ(ierr);
  ierr = DrawCheckResizedWindow(drawlocal); CHKERRQ(ierr);

  cell_n        = grid->cell_n;
  ierr = ISGetIndices(grid->cell_global,&cell_global); CHKERRQ(ierr);
  ierr = ISGetIndices(grid->vertex_global,&vertex_global); CHKERRQ(ierr);

  cell_vertex   = grid->cell_vertex;
  vertex_n      = grid->vertex_n;
  vertex_value  = grid->vertex_value;

  ierr = VecMin(algebra->x,PETSC_NULL,&vmin);CHKERRQ(ierr);
  ierr = VecMax(algebra->x,PETSC_NULL,&vmax);CHKERRQ(ierr);
  
  ierr = VecCopy(algebra->w_local,appctx->algebra.x_local);CHKERRQ(ierr);
  ierr = VecContourScale(algebra->x_local,vmin,vmax);CHKERRQ(ierr);
  ierr = DrawGetPopup(drawglobal,&popup);CHKERRQ(ierr);
  ierr = DrawScalePopup(popup,vmin,vmax);CHKERRQ(ierr);

  ierr = VecGetArray(algebra->x_local,&values);CHKERRQ(ierr);

  /*
      Plots solution on quadratic element by chopping it into four subtriangles and 
      doing a separate contour plot on each
  */
  for (i=0; i<cell_n; i++ ) {
    x0 = vertex_value[2*cell_vertex[ncell*i]];   y0 = vertex_value[2*cell_vertex[ncell*i] + 1];
    x1 = vertex_value[2*cell_vertex[ncell*i+1]]; y1 = vertex_value[2*cell_vertex[ncell*i+1] + 1];
    x2 = vertex_value[2*cell_vertex[ncell*i+5]]; y2 = vertex_value[2*cell_vertex[ncell*i+5] + 1];
    c0 = PetscReal(values[cell_vertex[ncell*i]]);
    c1 = PetscReal(values[cell_vertex[ncell*i+1]]);
    c2 = PetscReal(values[cell_vertex[ncell*i+5]]);
    ierr = DrawTriangle(drawglobal,x0,y0,x1,y1,x2,y2,c0,c1,c2);CHKERRQ(ierr);
    x0 = vertex_value[2*cell_vertex[ncell*i+1]]; y0 = vertex_value[2*cell_vertex[ncell*i+1] + 1];
    x1 = vertex_value[2*cell_vertex[ncell*i+2]]; y1 = vertex_value[2*cell_vertex[ncell*i+2] + 1];
    x2 = vertex_value[2*cell_vertex[ncell*i+3]]; y2 = vertex_value[2*cell_vertex[ncell*i+3] + 1];
    c0 = PetscReal(values[cell_vertex[ncell*i+1]]);
    c1 = PetscReal(values[cell_vertex[ncell*i+2]]);
    c2 = PetscReal(values[cell_vertex[ncell*i+3]]);
    ierr = DrawTriangle(drawglobal,x0,y0,x1,y1,x2,y2,c0,c1,c2);CHKERRQ(ierr);
    x0 = vertex_value[2*cell_vertex[ncell*i+1]]; y0 = vertex_value[2*cell_vertex[ncell*i+1] + 1];
    x1 = vertex_value[2*cell_vertex[ncell*i+5]]; y1 = vertex_value[2*cell_vertex[ncell*i+5] + 1];
    x2 = vertex_value[2*cell_vertex[ncell*i+3]]; y2 = vertex_value[2*cell_vertex[ncell*i+3] + 1];
    c0 = PetscReal(values[cell_vertex[ncell*i+1]]);
    c1 = PetscReal(values[cell_vertex[ncell*i+5]]);
    c2 = PetscReal(values[cell_vertex[ncell*i+3]]);
    ierr = DrawTriangle(drawglobal,x0,y0,x1,y1,x2,y2,c0,c1,c2);CHKERRQ(ierr);
    x0 = vertex_value[2*cell_vertex[ncell*i+4]]; y0 = vertex_value[2*cell_vertex[ncell*i+4] + 1];
    x1 = vertex_value[2*cell_vertex[ncell*i+5]]; y1 = vertex_value[2*cell_vertex[ncell*i+5] + 1];
    x2 = vertex_value[2*cell_vertex[ncell*i+3]]; y2 = vertex_value[2*cell_vertex[ncell*i+3] + 1];
    c0 = PetscReal(values[cell_vertex[ncell*i+4]]);
    c1 = PetscReal(values[cell_vertex[ncell*i+5]]);
    c2 = PetscReal(values[cell_vertex[ncell*i+3]]);
    ierr = DrawTriangle(drawglobal,x0,y0,x1,y1,x2,y2,c0,c1,c2);CHKERRQ(ierr);
  }   

  ierr = DrawSynchronizedFlush(drawglobal);CHKERRQ(ierr);
  ierr = DrawSynchronizedFlush(drawlocal);CHKERRQ(ierr);

  ierr = VecRestoreArray(algebra->x_local,&values);CHKERRQ(ierr);
  ierr = ISRestoreIndices(grid->cell_global,&cell_global); CHKERRQ(ierr);
  ierr = ISRestoreIndices(grid->vertex_global,&vertex_global); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

