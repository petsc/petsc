
#include "appctx.h"


/* ----------------------------------------------------------------------- */
/*
   AppCtxViewMatlab - Views solution using Matlab interactively

   Input Parameter:
   appctx - user-defined application context

   Note:
   See the companion Matlab file bscript.m for usage instructions.

*/
#undef __FUNCT__
#define __FUNCT__ "AppCxtViewMatlab"
int AppCtxViewMatlab(AppCtx* appctx)
{
  PetscErrorCode ierr;
  PetscViewer    viewer;
  FILE           *fp;

  PetscFunctionBegin;
  /* start up the companion Matlab processor to receive the data */
  ierr = PetscStartMatlab(PETSC_COMM_WORLD,PETSC_NULL,"bscript(0)",&fp);CHKERRQ(ierr);
#if defined(PETSC_USE_SOCKET_VIEWER)
  viewer = PETSC_VIEWER_SOCKET_WORLD;
#else
  viewer = PETSC_VIEWER_BINARY_WORLD;
#endif
  
  /* send the cell_coordinates */
  ierr = PetscRealView(2*4*appctx->grid.cell_n,appctx->grid.cell_coords,viewer);CHKERRQ(ierr);
  /* send cell_vertices */
  ierr = PetscIntView(4*appctx->grid.cell_n,appctx->grid.global_cell_vertex,viewer);CHKERRQ(ierr);
  /* send the solution */
  ierr = VecView(appctx->algebra.x,viewer);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*
        Displays the grid and its numbering
*/
#undef __FUNCT__
#define __FUNCT__ "AppCxtGraphics"
int AppCtxGraphics(AppCtx *appctx)
{
  PetscErrorCode ierr;
  PetscReal      maxs[2],mins[2],xmin,xmax,ymin,ymax,hx,hy;

  PetscFunctionBegin;
  /*---------------------------------------------------------------------
     Setup  the graphics windows
       drawglobal - contains a single picture of the entire grid
       drawlocal - shows each subgrid (one for each processor)
     ------------------------------------------------------------------------*/
  if (appctx->view.show_grid) {
    ierr = PetscDrawCreate(PETSC_COMM_WORLD,PETSC_NULL,"Total Grid",PETSC_DECIDE,PETSC_DECIDE,PETSC_DRAW_HALF_SIZE,PETSC_DRAW_HALF_SIZE,
                     &appctx->view.drawglobal);CHKERRQ(ierr);
    ierr = PetscDrawSetFromOptions(appctx->view.drawglobal);CHKERRQ(ierr);
    ierr = PetscDrawCreate(PETSC_COMM_WORLD,PETSC_NULL,"Local Grids",PETSC_DECIDE,PETSC_DECIDE,PETSC_DRAW_HALF_SIZE,PETSC_DRAW_HALF_SIZE,
                     &appctx->view.drawlocal);CHKERRQ(ierr);
    ierr = PetscDrawSetFromOptions(appctx->view.drawlocal);CHKERRQ(ierr);
    ierr = PetscDrawSplitViewPort((appctx)->view.drawlocal);CHKERRQ(ierr);

    /*
       Set the window coordinates based on the values in vertices
    */
    ierr = AODataSegmentGetExtrema((appctx)->aodata,"vertex","coords",maxs,mins);CHKERRQ(ierr);
    hx = maxs[0] - mins[0]; xmin = mins[0] - .1*hx; xmax = maxs[0] + .1*hx;
    hy = maxs[1] - mins[1]; ymin = mins[1] - .1*hy; ymax = maxs[1] + .1*hy;
    ierr = PetscDrawSetCoordinates(appctx->view.drawglobal,xmin,ymin,xmax,ymax);CHKERRQ(ierr);
    ierr = PetscDrawSetCoordinates(appctx->view.drawlocal,xmin,ymin,xmax,ymax);CHKERRQ(ierr);

    /*
      Visualize the grid 
    */
    ierr = PetscDrawZoom((appctx)->view.drawglobal,AppCtxViewGrid,appctx);CHKERRQ(ierr);
    ierr = PetscDrawZoom((appctx)->view.drawlocal,AppCtxViewGrid,appctx);CHKERRQ(ierr);

    ierr = PetscDrawDestroy((appctx)->view.drawglobal);CHKERRQ(ierr);
    ierr = PetscDrawDestroy((appctx)->view.drawlocal);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

/*
       Actually draw the grid. This routine is called indirectly by the PETSc graphics function
    PetscDrawZoom() allowing the user to zoom in and out on the grid. 

         left mouse button - zoom in 
         center button - zoom out
         right button - continue to next graphic
*/
#undef __FUNCT__
#define __FUNCT__ "AppCxtViewGrid"
int AppCtxViewGrid(PetscDraw draw,void *iappctx)
{
  AppCtx         *appctx = (AppCtx *)iappctx;  
  AppGrid        *grid = &appctx->grid;
  PetscReal      xl,yl,xr,yr,xp,yp,w,h;
  PetscErrorCode ierr;
  PetscInt       i,c,j,jnext,*iscell;
  PetscMPIInt    rank;
  char           num[5];

  PetscFunctionBegin;
  /* gets the global numbering of each vertex of each local cell */
  ierr = ISGetIndices(grid->iscell,&iscell);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(appctx->comm,&rank);CHKERRQ(ierr); c = rank + 2;

  /*
        PetscDraw edges of local cells and number the cells and vertices
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
        sprintf(num,"%d",(int)grid->cell_vertex[4*i+j]); /* local number of vertex */
        ierr = PetscDrawString(draw,xl,yl,c+1,num);CHKERRQ(ierr);
      } else {
        sprintf(num,"%d",(int)grid->global_cell_vertex[4*i+j]); /* global number of vertex */
        ierr = PetscDrawString(draw,xl,yl,PETSC_DRAW_BLACK,num);CHKERRQ(ierr);
      }
      xr   = grid->cell_coords[2*4*i + 2*jnext];
      yr   = grid->cell_coords[2*4*i + 2*jnext +1];
      ierr = PetscDrawLine(draw,xl,yl,xr,yr,c);CHKERRQ(ierr);
      xp   += xl;
      yp   += yl;
    }
    /* attach number to center of each cell */
    xp *= 0.25;
    yp *= 0.25;
    if (draw == appctx->view.drawlocal) {  
      sprintf(num,"%d",(int)i); /* local number of cell */
      ierr = PetscDrawString(draw,xp,yp,c,num);CHKERRQ(ierr);
    } else {
      sprintf(num,"%d",(int)iscell[i]); /* global number of cell */
      ierr = PetscDrawString(draw,xp,yp,PETSC_DRAW_RED,num);CHKERRQ(ierr);
    }
  }
  ierr = ISRestoreIndices(grid->iscell,&iscell);CHKERRQ(ierr); 

  /*
     Loop over boundary nodes and label them
  */
  if (draw == appctx->view.drawlocal) {
    ierr = PetscDrawStringGetSize(draw,&w,&h);CHKERRQ(ierr);
    for (i=0; i<grid->boundary_n; i++) {
      xp = grid->boundary_coords[2*i] - 4.0*w;
      yp = grid->boundary_coords[2*i+1] - 4.0*h;
      sprintf(num,"%d",(int)i);
      ierr = PetscDrawString(draw,xp,yp,PETSC_DRAW_BROWN,num);CHKERRQ(ierr);
    }
  }

  PetscFunctionReturn(0);
}

