
/*
       Demonstrates how to draw two dimensional grids and solutions. See appctx.h
       for information on how this is to be used.
*/

#include "appctx.h"

#undef __FUNCT__
#define __FUNCT__ "AppCxtView"
PetscErrorCode AppCtxView(PetscDraw idraw,void *iappctx)
{
  AppCtx                 *appctx = (AppCtx *)iappctx;
  AppGrid                *grid = &appctx->grid;

  PetscInt                    cell_n,vertex_n,ncell = 4,*verts,nverts;

  /*
        These contain the  vertex lists in local numbering
  */ 
  PetscInt                    *cell_vertex;

  /* 
        These contain the global numbering for local objects
  */
  PetscInt                    *cell_global,*vertex_global;
  
  double                 *vertex_value;

  PetscErrorCode                    ierr;
  PetscMPIInt rank;
  PetscInt                    ij,i,c,j;

  PetscDraw                   drawlocal = appctx->view.drawlocal;
  double                 xl,yl,xr,yr,xm,ym,xp,yp,w,h;
  char                   num[5];


  MPI_Comm_rank(appctx->comm,&rank); c = rank + 2;

  ierr = ISGetIndices(grid->cell_global,&cell_global);CHKERRQ(ierr);
  ierr = ISGetIndices(grid->vertex_global,&vertex_global);CHKERRQ(ierr);

  cell_n               = grid->cell_n;
  cell_vertex          = grid->cell_vertex;
  vertex_n             = grid->vertex_n_ghosted;
  vertex_value         = grid->vertex_value;

  ierr = ISGetLocalSize(grid->isvertex_boundary,&nverts);CHKERRQ(ierr);
  ierr = ISGetIndices(grid->isvertex_boundary,&verts);CHKERRQ(ierr);

  /*
        PetscDraw edges of local cells and number them
  */
  for (i=0; i<cell_n; i++) {
    xp = 0.0; yp = 0.0;
    xl = vertex_value[2*cell_vertex[ncell*i]]; yl = vertex_value[2*cell_vertex[ncell*i] + 1];
    for (j=0; j<ncell; j++) {
      ij = ncell*i + ((j+1) % ncell);
      xr = vertex_value[2*cell_vertex[ij]]; yr = vertex_value[2*cell_vertex[ij] + 1];
      ierr = PetscDrawLine(idraw,xl,yl,xr,yr,c);CHKERRQ(ierr);
      xp += xl;         yp += yl;
      xl  = xr;         yl =  yr;
    }
    xp /= ncell; yp /= ncell;
    if (idraw == drawlocal) {
      sprintf(num,"%d",(int)i);
      ierr = PetscDrawString(idraw,xp,yp,PETSC_DRAW_GREEN,num);CHKERRQ(ierr);
    } else {
      sprintf(num,"%d",(int)cell_global[i]);
      ierr = PetscDrawString(idraw,xp,yp,c,num);CHKERRQ(ierr);
    }
  }


  /*
      Numbering of vertices
  */
  for (i=0; i<vertex_n; i++) {
    xm = vertex_value[2*i]; ym = vertex_value[2*i + 1];
    if (idraw == drawlocal) {
      sprintf(num,"%d",(int)i);
      ierr = PetscDrawString(idraw,xm,ym,PETSC_DRAW_BLUE,num);CHKERRQ(ierr);
    } else {
      sprintf(num,"%d",(int)vertex_global[i]);
      ierr = PetscDrawString(idraw,xm,ym,PETSC_DRAW_BLUE,num);CHKERRQ(ierr);
    }
  }

  /*
     Print Numbering of boundary nodes 
  */
  ierr = PetscDrawStringGetSize(idraw,&w,&h);CHKERRQ(ierr);
  if (idraw == drawlocal) {
    for (i=0; i<nverts; i++) {
      xm = vertex_value[2*verts[i]] - 4.*w; ym = vertex_value[2*verts[i] + 1] - 4.*h;
      sprintf(num,"%d",(int)i);
      ierr = PetscDrawString(idraw,xm,ym,PETSC_DRAW_BLACK,num);CHKERRQ(ierr);
    }
  }

  ierr = ISRestoreIndices(grid->isvertex_boundary,&verts);CHKERRQ(ierr);
  ierr = ISRestoreIndices(grid->cell_global,&cell_global);CHKERRQ(ierr);
  ierr = ISRestoreIndices(grid->vertex_global,&vertex_global);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "AppCxtViewSolution"
PetscErrorCode AppCtxViewSolution(PetscDraw idraw,void *iappctx)
{
  AppCtx                 *appctx = (AppCtx *)iappctx;
  AppGrid                *grid = &appctx->grid;
  AppAlgebra             *algebra = &appctx->algebra;
  PetscInt                    cell_n,ncell = 4;

  /*
        These contain the vertex lists in local numbering
  */ 
  PetscInt                    *cell_vertex;

  /* 
        These contain the global numbering for local objects
  */
  PetscInt                    *cell_global,*vertex_global;
  
  double                 *vertex_value;


  PetscErrorCode       ierr;
  PetscInt i,c0,c1,c2;

  PetscDraw                   drawglobal = appctx->view.drawglobal;
  PetscDraw                   drawlocal = appctx->view.drawlocal,popup;
  double                 x0,x1,x2,y_0,y_1,y2,vmin,vmax;

  PetscScalar            *values;

  cell_n        = grid->cell_n;
  ierr = ISGetIndices(grid->cell_global,&cell_global);CHKERRQ(ierr);
  ierr = ISGetIndices(grid->vertex_global,&vertex_global);CHKERRQ(ierr);

  cell_vertex   = grid->cell_vertex;
  vertex_value  = grid->vertex_value;

  ierr = VecMin(algebra->g,PETSC_NULL,&vmin);CHKERRQ(ierr);
  ierr = VecMax(algebra->g,PETSC_NULL,&vmax);CHKERRQ(ierr);
  
  ierr = VecContourScale(algebra->f_local,vmin,vmax);CHKERRQ(ierr);
  ierr = PetscDrawGetPopup(drawglobal,&popup);CHKERRQ(ierr);
  if (popup) {ierr = PetscDrawScalePopup(popup,vmin,vmax);CHKERRQ(ierr);}

  ierr = VecGetArray(algebra->f_local,&values);CHKERRQ(ierr);

  for (i=0; i<cell_n; i++) {
    x0 = vertex_value[2*cell_vertex[ncell*i]];   y_0 = vertex_value[2*cell_vertex[ncell*i] + 1];
    x1 = vertex_value[2*cell_vertex[ncell*i+1]]; y_1 = vertex_value[2*cell_vertex[ncell*i+1] + 1];
    x2 = vertex_value[2*cell_vertex[ncell*i+2]]; y2 = vertex_value[2*cell_vertex[ncell*i+2] + 1];
    c0 = (PetscInt)values[cell_vertex[ncell*i]];
    c1 = (PetscInt)values[cell_vertex[ncell*i+1]];
    c2 = (PetscInt)values[cell_vertex[ncell*i+2]];
    ierr = PetscDrawTriangle(drawglobal,x0,y_0,x1,y_1,x2,y2,c0,c1,c2);CHKERRQ(ierr);
    x0 = vertex_value[2*cell_vertex[ncell*i]];   y_0 = vertex_value[2*cell_vertex[ncell*i] + 1];
    x1 = vertex_value[2*cell_vertex[ncell*i+3]]; y_1 = vertex_value[2*cell_vertex[ncell*i+3] + 1];
    x2 = vertex_value[2*cell_vertex[ncell*i+2]]; y2 = vertex_value[2*cell_vertex[ncell*i+2] + 1];
    c0 = (PetscInt)values[cell_vertex[ncell*i]];
    c1 = (PetscInt)values[cell_vertex[ncell*i+3]];
    c2 = (PetscInt)values[cell_vertex[ncell*i+2]];
    ierr = PetscDrawTriangle(drawglobal,x0,y_0,x1,y_1,x2,y2,c0,c1,c2);CHKERRQ(ierr);
  }

  ierr = PetscDrawSynchronizedFlush(drawglobal);CHKERRQ(ierr);
  ierr = PetscDrawSynchronizedFlush(drawlocal);CHKERRQ(ierr);

  ierr = VecRestoreArray(algebra->f_local,&values);CHKERRQ(ierr);
  ierr = ISRestoreIndices(grid->cell_global,&cell_global);CHKERRQ(ierr);
  ierr = ISRestoreIndices(grid->vertex_global,&vertex_global);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------------- */
/*
   AppCtxViewMatlab - Views solution using Matlab via socket connections.

   Input Parameter:
   appctx - user-defined application context

   Note:
   See the companion Matlab file mscript.m for usage instructions.
*/
#undef __FUNCT__
#define __FUNCT__ "AppCxtViewMatlab"
PetscErrorCode AppCtxViewMatlab(AppCtx* appctx)
{
  PetscInt    *cell_vertex,rstart,rend;
  PetscErrorCode ierr;
#if defined(PETSC_USE_SOCKET_VIEWER)
  PetscViewer viewer = PETSC_VIEWER_SOCKET_WORLD;
#else
  PetscViewer viewer = PETSC_VIEWER_BINARY_WORLD;
#endif
  double *vertex_values;
  IS     isvertex;
  PetscFunctionBegin;

  /* First, send solution vector to Matlab */
  ierr = VecView(appctx->algebra.g,viewer);CHKERRQ(ierr);

  /* now send the number of steps coming */
  
 /******** Here need to send multiple solution vectors **********/
/*   for(i=0;i<NSTEPS;i++)*/
  /*   {ierr = VecView(appctx->algebra.solnv[i],viewer);CHKERRQ(ierr);} */


  /* Next, send vertices to Matlab */
  ierr = AODataKeyGetOwnershipRange(appctx->aodata,"vertex",&rstart,&rend);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_WORLD,rend-rstart,rstart,1,&isvertex);CHKERRQ(ierr);
  ierr = AODataSegmentGetIS(appctx->aodata,"vertex","values",isvertex,(void **)&vertex_values);CHKERRQ(ierr);
  ierr = PetscRealView(2*(rend-rstart),vertex_values,viewer);CHKERRQ(ierr);
  ierr = AODataSegmentRestoreIS(appctx->aodata,"vertex","values",PETSC_NULL,(void **)&vertex_values);CHKERRQ(ierr);
  ierr = ISDestroy(isvertex);CHKERRQ(ierr);

  /* 
     Send list of vertices for each cell; these MUST be in the global (not local!) numbering); 
     this cannot use appctx->grid->cell_vertex 
  */
  ierr = AODataSegmentGetIS(appctx->aodata,"cell","vertex",appctx->grid.cell_global,
        (void **)&cell_vertex);CHKERRQ(ierr);
  ierr = PetscIntView(4*appctx->grid.cell_n,cell_vertex,viewer);CHKERRQ(ierr);
  ierr = AODataSegmentRestoreIS(appctx->aodata,"cell","vertex",PETSC_NULL,(void **)&cell_vertex); 
         CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}


/*--------------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "AppCxtGraphics"
PetscErrorCode AppCtxGraphics(AppCtx *appctx)
{
  PetscErrorCode    ierr;
  double maxs[2],mins[2],xmin,xmax,ymin,ymax,hx,hy;

  /*---------------------------------------------------------------------
     Setup  the graphics windows
     ------------------------------------------------------------------------*/

  /* moved to AppCtxCreate -- H
  ierr = PetscOptionsHasName(PETSC_NULL,"-show_grid",&appctx->view.show_grid);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-show_griddata",&appctx->view.show_griddata);CHKERRQ(ierr);
  printf("in AppCtxGraphics, appctx->view.show_griddata= %d\n",appctx->view.show_griddata);
  */

  if ((appctx)->view.show_grid) {
    ierr = PetscDrawCreate(PETSC_COMM_WORLD,PETSC_NULL,"Total Grid",PETSC_DECIDE,PETSC_DECIDE,400,400,
                     &appctx->view.drawglobal);CHKERRQ(ierr);
    ierr = PetscDrawSetFromOptions(appctx->view.drawglobal);CHKERRQ(ierr);
    ierr = PetscDrawCreate(PETSC_COMM_WORLD,PETSC_NULL,"Local Grids",PETSC_DECIDE,PETSC_DECIDE,400,400,
                     &appctx->view.drawlocal);CHKERRQ(ierr);
    ierr = PetscDrawSetFromOptions(appctx->view.drawlocal);CHKERRQ(ierr);
    ierr = PetscDrawSplitViewPort((appctx)->view.drawlocal);CHKERRQ(ierr);

    /*
       Set the window coordinates based on the values in vertices
    */
    ierr = AODataSegmentGetExtrema((appctx)->aodata,"vertex","values",maxs,mins);CHKERRQ(ierr);
    hx = maxs[0] - mins[0]; xmin = mins[0] - .1*hx; xmax = maxs[0] + .1*hx;
    hy = maxs[1] - mins[1]; ymin = mins[1] - .1*hy; ymax = maxs[1] + .1*hy;
    ierr = PetscDrawSetCoordinates((appctx)->view.drawglobal,xmin,ymin,xmax,ymax);CHKERRQ(ierr);
    ierr = PetscDrawSetCoordinates((appctx)->view.drawlocal,xmin,ymin,xmax,ymax);CHKERRQ(ierr);
    /*
       Visualize the grid 
    */
    ierr = PetscDrawZoom((appctx)->view.drawglobal,AppCtxView,appctx);CHKERRQ(ierr);
    ierr = PetscDrawZoom((appctx)->view.drawlocal,AppCtxView,appctx);CHKERRQ(ierr);
  }
  ierr = PetscOptionsHasName(PETSC_NULL,"-matlab_graphics",&(appctx)->view.matlabgraphics);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
