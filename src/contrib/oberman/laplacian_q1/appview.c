/*$Id: milu.c,v 1.18 1999/11/05 14:48:07 bsmith Exp bsmith $*/
#include "appctx.h"


/* ----------------------------------------------------------------------- */
/*
   AppCtxViewMatlab - Views solution using Matlab via socket connections.

   Input Parameter:
   appctx - user-defined application context

   Note:
   See the companion Matlab file mscript.m for usage instructions.

   Only works for one processor

*/
#undef __FUNC__
#define __FUNC__ "AppCxtViewMatlab"
int AppCtxViewMatlab(AppCtx* appctx)
{
  int    ierr;
  Viewer viewer;
  FILE   *fp;

  PetscFunctionBegin;
  /* send the cell_coords */
  ierr = PetscStartMatlab(PETSC_COMM_WORLD,"fire","bscript(0)",&fp);CHKERRQ(ierr);
  
  viewer = VIEWER_SOCKET_WORLD;
  ierr = PetscDoubleView(2*4*appctx->grid.cell_n,appctx->grid.cell_coords,viewer);CHKERRQ(ierr);
  /* send cell_vertices */
  ierr = PetscIntView(4*appctx->grid.cell_n,appctx->grid.global_cell_vertex,viewer);CHKERRQ(ierr);
  /* send the solution */
  ierr = VecView(appctx->algebra.x,viewer);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*
        Displays the grid and its numbering
*/
#undef __FUNC__
#define __FUNC__ "AppCxtGraphics"
int AppCtxGraphics(AppCtx *appctx)
{
  int    ierr;
  double maxs[2],mins[2],xmin,xmax,ymin,ymax,hx,hy;


  PetscFunctionBegin;

  /*---------------------------------------------------------------------
     Setup  the graphics windows
       drawglobal - contains a single picture of the entire grid
       drawlocal - shows each subgrid (one for each processor)
     ------------------------------------------------------------------------*/
  if (appctx->view.show_grid) {
    ierr = DrawCreate(PETSC_COMM_WORLD,PETSC_NULL,"Total Grid",PETSC_DECIDE,PETSC_DECIDE,DRAW_HALF_SIZE,DRAW_HALF_SIZE,
                     &appctx->view.drawglobal);CHKERRQ(ierr);
    ierr = DrawSetFromOptions(appctx->view.drawglobal);CHKERRA(ierr);
    ierr = DrawCreate(PETSC_COMM_WORLD,PETSC_NULL,"Local Grids",PETSC_DECIDE,PETSC_DECIDE,DRAW_HALF_SIZE,DRAW_HALF_SIZE,
                     &appctx->view.drawlocal);CHKERRQ(ierr);
    ierr = DrawSetFromOptions(appctx->view.drawlocal);CHKERRA(ierr);
    ierr = DrawSplitViewPort((appctx)->view.drawlocal);CHKERRQ(ierr);

    /*
       Set the window coordinates based on the values in vertices
    */
    ierr = AODataSegmentGetExtrema((appctx)->aodata,"vertex","coords",maxs,mins);CHKERRQ(ierr);
    hx = maxs[0] - mins[0]; xmin = mins[0] - .1*hx; xmax = maxs[0] + .1*hx;
    hy = maxs[1] - mins[1]; ymin = mins[1] - .1*hy; ymax = maxs[1] + .1*hy;
    ierr = DrawSetCoordinates(appctx->view.drawglobal,xmin,ymin,xmax,ymax);CHKERRQ(ierr);
    ierr = DrawSetCoordinates(appctx->view.drawlocal,xmin,ymin,xmax,ymax);CHKERRQ(ierr);

    /*
      Visualize the grid 
    */
    ierr = DrawZoom((appctx)->view.drawglobal,AppCtxViewGrid,appctx);CHKERRA(ierr);
    ierr = DrawZoom((appctx)->view.drawlocal,AppCtxViewGrid,appctx);CHKERRA(ierr);

    ierr = DrawDestroy((appctx)->view.drawglobal);CHKERRQ(ierr);
    ierr = DrawDestroy((appctx)->view.drawlocal);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

/*
       Actually draw the grid. This routine is called indirectly by the PETSc graphics function
    DrawZoom() allowing the user to zoom in and out on the grid. 

         left mouse button - zoom in 
         center button - zoom out
         right button - continue to next graphic
*/
#undef __FUNC__
#define __FUNC__ "AppCxtViewGrid"
int AppCtxViewGrid(Draw draw,void *iappctx)
{
  AppCtx  *appctx = (AppCtx *)iappctx;  
  AppGrid *grid = &appctx->grid;
  double  xl,yl,xr,yr,xp,yp,w,h;
  int     ierr,i,rank,c,j,jnext,*iscell;
  char    num[5];

  PetscFunctionBegin;
  /* gets the global numbering of each vertex of each local cell */
  ierr = ISGetIndices(grid->iscell,&iscell);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(appctx->comm,&rank);CHKERRQ(ierr); c = rank + 2;

  /*
        Draw edges of local cells and number the cells and vertices
  */
  for (i=0; i<grid->cell_n; i++) {   
    xp = 0.0;
    yp = 0.0;
    for (j=0; j<4; j++) {
      jnext = (j+1) % 4;
      xl   = grid->cell_coords[2*4*i + 2*j];
      yl   = grid->cell_coords[2*4*i + 2*j + 1];
      /* attach number to each vertex */
      if (draw == appctx->view.drawlocal) {
        sprintf(num,"%d",grid->cell_vertex[4*i+j]); /* local number of vertex */
        ierr = DrawString(draw,xl,yl,c+1,num);CHKERRQ(ierr);
      } else {
        sprintf(num,"%d",grid->global_cell_vertex[4*i+j]); /* global number of vertex */
        ierr = DrawString(draw,xl,yl,DRAW_BLACK,num);CHKERRQ(ierr);
      }
      xr   = grid->cell_coords[2*4*i + 2*jnext];
      yr   = grid->cell_coords[2*4*i + 2*jnext +1];
      ierr = DrawLine(draw,xl,yl,xr,yr,c);CHKERRQ(ierr);
      xp   += xl;
      yp   += yl;
    }
    /* attach number to center of each cell */
    xp *= 0.25;
    yp *= 0.25;
    if (draw == appctx->view.drawlocal) {  
      sprintf(num,"%d",i); /* local number of cell */
      ierr = DrawString(draw,xp,yp,c,num);CHKERRQ(ierr);
    } else {
      sprintf(num,"%d",iscell[i]); /* global number of cell */
      ierr = DrawString(draw,xp,yp,DRAW_RED,num);CHKERRQ(ierr);
    }
  }
  ierr = ISRestoreIndices(grid->iscell,&iscell);CHKERRQ(ierr); 

  /*
     Loop over boundary nodes and label them
  */
  if (draw == appctx->view.drawlocal) {
    ierr = DrawStringGetSize(draw,&w,&h);CHKERRQ(ierr);
    for (i=0; i<grid->boundary_count; i++) {
      xp = grid->boundary_coords[2*i] - 4.0*w;
      yp = grid->boundary_coords[2*i+1] - 4.0*h;
      sprintf(num,"%d",i);
      ierr = DrawString(draw,xp,yp,DRAW_BROWN,num);CHKERRQ(ierr);
    }
  }

  PetscFunctionReturn(0);
}

