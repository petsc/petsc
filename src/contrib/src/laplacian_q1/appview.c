#include "appctx.h"


/* ----------------------------------------------------------------------- */
/*
   AppCtxViewMatlab - Views solution using Matlab via socket connections.

   Input Parameter:
   appctx - user-defined application context

   Note:
   See the companion Matlab file mscript.m for usage instructions.
*/
#undef __FUNC__
#define __FUNC__ "AppCxtViewMatlab"
int AppCtxViewMatlab(AppCtx* appctx)
{
  int    ierr;
  Viewer viewer = VIEWER_MATLAB_WORLD;

  PetscFunctionBegin;
  /* now send the cell_coords */
  ierr = PetscDoubleView(2*4*appctx->grid.cell_n, appctx->grid.cell_coords, viewer);
  /* send cell_vertices */
  ierr = PetscIntView(4*appctx->grid.cell_n, appctx->grid.cell_vertex, viewer);CHKERRQ(ierr);
  /* send the solution */
  ierr = VecView(appctx->algebra.x, viewer); CHKERRQ(ierr);

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
    ierr = AODataSegmentGetExtrema((appctx)->aodata,"vertex","coords",maxs,mins);CHKERRQ(ierr);
    hx = maxs[0] - mins[0]; xmin = mins[0] - .1*hx; xmax = maxs[0] + .1*hx;
    hy = maxs[1] - mins[1]; ymin = mins[1] - .1*hy; ymax = maxs[1] + .1*hy;
    ierr = DrawSetCoordinates((appctx)->view.drawglobal,xmin,ymin,xmax,ymax);CHKERRQ(ierr);
    ierr = DrawSetCoordinates((appctx)->view.drawlocal,xmin,ymin,xmax,ymax);CHKERRQ(ierr);
    /*
       Visualize the grid 
    */
    ierr = DrawZoom((appctx)->view.drawglobal,AppCtxViewGrid,appctx); CHKERRA(ierr);
  }

  PetscFunctionReturn(0);
}


#undef __FUNC__
#define __FUNC__ "AppCxtViewGrid"
int AppCtxViewGrid(Draw idraw,void *iappctx)
{
   AppCtx                 *appctx = (AppCtx *)iappctx;  
  AppGrid                *grid = &appctx->grid;
  AppView *view = &appctx->view;

 double                 xl,yl,xr,yr,xm,ym,xp,yp;
 int                    ierr,i,rank,c,j, jnext;
 char                   num[5];
  Draw                   drawglobal = appctx->view.drawglobal;
  Draw                   drawlocal = appctx->view.drawlocal,popup;
  int                    *cell_global,*vertex_global;

  ierr = ISGetIndices(grid->cell_global,&cell_global); CHKERRQ(ierr);
  
  ierr = DrawCheckResizedWindow(drawglobal); CHKERRQ(ierr);
  ierr = DrawCheckResizedWindow(drawlocal); CHKERRQ(ierr);

  MPI_Comm_rank(appctx->comm,&rank); c = rank + 2;

  /*
        Draw edges of local cells and number them
  */
  if (view->showelements) {
   for (i=0; i<grid->cell_n; i++ ) {   
    for (j=0; j<4; j++) {
      jnext = (j+1) % 4;
      xl = grid->cell_coords[2*4*i + 2*j];
      yl = grid->cell_coords[2*4*i + 2*j + 1];
      xr = grid->cell_coords[2*4*i + 2*jnext];
      yr = grid->cell_coords[2*4*i + 2*jnext +1];
      ierr = DrawLine(drawglobal,xl,yl,xr,yr,c);CHKERRQ(ierr);
      ierr = DrawLine(drawlocal,xl,yl,xr,yr,DRAW_BLUE);CHKERRQ(ierr);
    }
    if (view->shownumbers){
      xp = 0.5*( grid->cell_coords[2*4*i ] + grid->cell_coords[2*4*i + 2]); 
      yp = 0.5*( grid->cell_coords[2*4*i + 1] + grid->cell_coords[2*4*i + 4 +1]); 
      sprintf(num,"%d",i);
      ierr = DrawString(drawlocal,xp,yp,DRAW_GREEN,num);CHKERRQ(ierr);
      sprintf(num,"%d",cell_global[i]);
      ierr = DrawString(drawglobal,xp,yp,c,num);CHKERRQ(ierr);
    }  
   }
  }

 
  PetscFunctionReturn(0);
}

















