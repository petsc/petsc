
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

  PetscInt                    cell_n,vertex_local_n,ncell = 4,*verts,nverts;

  /*
        These contain the  vertex lists in local numbering
  */ 
  PetscInt                    *cell_vertex;

  /* 
        These contain the global numbering for local objects
  */
  PetscInt                    *cell_global,*vertex_global;
  
  PetscReal                 *vertex_coords;

  PetscBT                vertex_boundary_flag;
  PetscErrorCode   ierr;
  PetscInt                    i,c,j,ijp;
  PetscInt                    ij;
  PetscMPIInt                 rank;
  PetscDraw                   drawglobal = appctx->view.drawglobal;
  PetscDraw                   drawlocal = appctx->view.drawlocal;
  PetscReal                 xl,yl,xr,yr,xm,ym,xp,yp;
  char                   num[5];

  ierr = PetscDrawCheckResizedWindow(drawglobal);CHKERRQ(ierr);
  ierr = PetscDrawCheckResizedWindow(drawlocal);CHKERRQ(ierr);

  MPI_Comm_rank(appctx->comm,&rank); c = rank + 2;

  ierr = ISGetIndices(grid->cell_global,&cell_global);CHKERRQ(ierr);
  ierr = ISGetIndices(grid->vertex_global,&vertex_global);CHKERRQ(ierr);

  cell_n               = grid->cell_n;
  cell_vertex          = grid->cell_vertex;
  vertex_local_n             = grid->vertex_local_n;
  vertex_coords         = grid->vertex_coords;
  vertex_boundary_flag = grid->vertex_boundary_flag;

  ierr = ISGetLocalSize(grid->vertex_boundary,&nverts);CHKERRQ(ierr);
  ierr = ISGetIndices(grid->vertex_boundary,&verts);CHKERRQ(ierr);

  /*
        PetscDraw edges of local cells and number them
  */

    for (i=0; i<cell_n; i++) {
      xp = 0.0; yp = 0.0;
      xl = vertex_coords[2*cell_vertex[ncell*i]]; yl = vertex_coords[2*cell_vertex[ncell*i] + 1];
      for (j=0; j<ncell; j++) {
        ij = ncell*i + ((j+1) % ncell);
        xr = vertex_coords[2*cell_vertex[ij]]; yr = vertex_coords[2*cell_vertex[ij] + 1];
        ierr = PetscDrawLine(drawglobal,xl,yl,xr,yr,c);CHKERRQ(ierr);
        ierr = PetscDrawLine(drawlocal,xl,yl,xr,yr,PETSC_DRAW_BLUE);CHKERRQ(ierr);
        xp += xl;         yp += yl;
        xl  = xr;         yl =  yr;
      }
        xp /= ncell; yp /= ncell;
        sprintf(num,"%d",(int)i);
        ierr = PetscDrawString(drawlocal,xp,yp,PETSC_DRAW_GREEN,num);CHKERRQ(ierr);
        sprintf(num,"%d",(int)cell_global[i]);
        ierr = PetscDrawString(drawglobal,xp,yp,c,num);CHKERRQ(ierr);
    }


  /*
       PetscDraws only boundary edges 
  */

    for (i=0; i<cell_n; i++) {
      xp = 0.0; yp = 0.0;
      xl  = vertex_coords[2*cell_vertex[ncell*i]]; yl = vertex_coords[2*cell_vertex[ncell*i] + 1];
      ijp = ncell*i;
      for (j=0; j<ncell; j++) {
        ij = ncell*i + ((j+1) % ncell);
        xr = vertex_coords[2*cell_vertex[ij]]; yr = vertex_coords[2*cell_vertex[ij] + 1];
        if (PetscBTLookup(vertex_boundary_flag,cell_vertex[ijp]) && PetscBTLookup(vertex_boundary_flag,cell_vertex[ij])) {
          ierr = PetscDrawLine(drawglobal,xl,yl,xr,yr,c);CHKERRQ(ierr);
          ierr = PetscDrawLine(drawlocal,xl,yl,xr,yr,PETSC_DRAW_BLUE);CHKERRQ(ierr);
        }
        xp += xl;         yp += yl;
        xl  = xr;         yl =  yr;
        ijp = ij;
      }

        xp /= ncell; yp /= ncell;
        sprintf(num,"%d",(int)i);
        ierr = PetscDrawString(drawlocal,xp,yp,PETSC_DRAW_GREEN,num);CHKERRQ(ierr);
        sprintf(num,"%d",(int)cell_global[i]);
        ierr = PetscDrawString(drawglobal,xp,yp,c,num);CHKERRQ(ierr);

  }

  /*
      Number vertices
  */

    for (i=0; i<vertex_local_n; i++) {
      xm = vertex_coords[2*i]; ym = vertex_coords[2*i + 1];
      ierr = PetscDrawString(drawglobal,xm,ym,PETSC_DRAW_BLUE,num);CHKERRQ(ierr);
      ierr = PetscDrawPoint(drawglobal,xm,ym,PETSC_DRAW_ORANGE);CHKERRQ(ierr);
      ierr = PetscDrawPoint(drawlocal,xm,ym,PETSC_DRAW_ORANGE);CHKERRQ(ierr);

        sprintf(num,"%d",(int)i);
        ierr = PetscDrawString(drawlocal,xm,ym,PETSC_DRAW_BLUE,num);CHKERRQ(ierr);
        sprintf(num,"%d",(int)vertex_global[i]);
        ierr = PetscDrawString(drawglobal,xm,ym,PETSC_DRAW_BLUE,num);CHKERRQ(ierr);
    }

    for (i=0; i<nverts; i++) {
      xm = vertex_coords[2*verts[i]]; ym = vertex_coords[2*verts[i] + 1];
      ierr = PetscDrawPoint(drawglobal,xm,ym,PETSC_DRAW_RED);CHKERRQ(ierr);
      ierr = PetscDrawPoint(drawlocal,xm,ym,PETSC_DRAW_RED);CHKERRQ(ierr);
    }

  ierr = PetscDrawSynchronizedFlush(drawglobal);CHKERRQ(ierr);
  ierr = PetscDrawSynchronizedFlush(drawlocal);CHKERRQ(ierr);

  ierr = ISRestoreIndices(grid->vertex_boundary,&verts);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  PetscInt    *cell_vertex,rstart,rend;
#if defined(PETSC_USE_SOCKET_VIEWER)
  PetscViewer viewer = PETSC_VIEWER_SOCKET_WORLD;
#else
  PetscViewer viewer = PETSC_VIEWER_BINARY_WORLD;
#endif
  PetscReal *vertex_coords;
  IS     isvertex;

  PetscFunctionBegin;

  /* First, send solution vector to Matlab */
  ierr = VecView(appctx->algebra.g,viewer);CHKERRQ(ierr);

  /* Next, send vertices to Matlab */
  ierr = AODataKeyGetOwnershipRange(appctx->aodata,"vertex",&rstart,&rend);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_WORLD,rend-rstart,rstart,1,&isvertex);CHKERRQ(ierr);
  ierr = AODataSegmentGetIS(appctx->aodata,"vertex","values",isvertex,(void **)&vertex_coords);CHKERRQ(ierr);
  ierr = PetscRealView(2*(rend-rstart),vertex_coords,viewer);CHKERRQ(ierr);
  ierr = AODataSegmentRestoreIS(appctx->aodata,"vertex","values",PETSC_NULL,(void **)&vertex_coords);CHKERRQ(ierr);
  ierr = ISDestroy(isvertex);CHKERRQ(ierr);

  /* 
     Send list of vertices for each cell; these MUST be in the global (not local!) numbering); 
     this cannot use appctx->grid->cell_vertex 
  */
  ierr = AODataSegmentGetIS(appctx->aodata,"cell","vertex",appctx->grid.cell_global,(void **)&cell_vertex);CHKERRQ(ierr);
  ierr = PetscIntView(4*appctx->grid.cell_n,cell_vertex,viewer);CHKERRQ(ierr);
  ierr = AODataSegmentRestoreIS(appctx->aodata,"cell","vertex",PETSC_NULL,(void **)&cell_vertex);CHKERRQ(ierr);
  
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
  
  PetscReal                 *vertex_coords;


  PetscInt                    ierr,i;

  PetscDraw                   drawglobal = appctx->view.drawglobal,popup;
  PetscReal                 x0,x1,x2,y_0,y_1,y2,vmin,vmax;
  PetscInt                    c0,c1,c2;
  PetscScalar            *values;

  ierr = PetscDrawCheckResizedWindow(drawglobal);CHKERRQ(ierr);

  cell_n        = grid->cell_n;
  ierr = ISGetIndices(grid->cell_global,&cell_global);CHKERRQ(ierr);
  ierr = ISGetIndices(grid->vertex_global,&vertex_global);CHKERRQ(ierr);

  cell_vertex   = grid->cell_vertex;
  vertex_coords  = grid->vertex_coords;

  ierr = VecMin(algebra->x,PETSC_NULL,&vmin);CHKERRQ(ierr);
  ierr = VecMax(algebra->x,PETSC_NULL,&vmax);CHKERRQ(ierr);
  
  ierr = VecCopy(algebra->w_local,appctx->algebra.x_local);CHKERRQ(ierr);
  ierr = VecContourScale(algebra->x_local,vmin,vmax);CHKERRQ(ierr);
  ierr = PetscDrawGetPopup(drawglobal,&popup);CHKERRQ(ierr);
  if (popup) {ierr = PetscDrawScalePopup(popup,vmin,vmax);CHKERRQ(ierr);}

  ierr = VecGetArray(algebra->x_local,&values);CHKERRQ(ierr);

  for (i=0; i<cell_n; i++) {
    x0 = vertex_coords[2*cell_vertex[ncell*i]];   y_0 = vertex_coords[2*cell_vertex[ncell*i] + 1];
    x1 = vertex_coords[2*cell_vertex[ncell*i+1]]; y_1 = vertex_coords[2*cell_vertex[ncell*i+1] + 1];
    x2 = vertex_coords[2*cell_vertex[ncell*i+2]]; y2 = vertex_coords[2*cell_vertex[ncell*i+2] + 1];
    c0 = (PetscInt)values[cell_vertex[ncell*i]];
    c1 = (PetscInt)values[cell_vertex[ncell*i+1]];
    c2 = (PetscInt)values[cell_vertex[ncell*i+2]];
    ierr = PetscDrawTriangle(drawglobal,x0,y_0,x1,y_1,x2,y2,c0,c1,c2);CHKERRQ(ierr);
    x0 = vertex_coords[2*cell_vertex[ncell*i]];   y_0 = vertex_coords[2*cell_vertex[ncell*i] + 1];
    x1 = vertex_coords[2*cell_vertex[ncell*i+3]]; y_1 = vertex_coords[2*cell_vertex[ncell*i+3] + 1];
    x2 = vertex_coords[2*cell_vertex[ncell*i+2]]; y2 = vertex_coords[2*cell_vertex[ncell*i+2] + 1];
    c0 = (PetscInt)values[cell_vertex[ncell*i]];
    c1 = (PetscInt)values[cell_vertex[ncell*i+3]];
    c2 = (PetscInt)values[cell_vertex[ncell*i+2]];
    ierr = PetscDrawTriangle(drawglobal,x0,y_0,x1,y_1,x2,y2,c0,c1,c2);CHKERRQ(ierr);
  }

  ierr = PetscDrawSynchronizedFlush(drawglobal);CHKERRQ(ierr);

  ierr = VecRestoreArray(algebra->x_local,&values);CHKERRQ(ierr);
  ierr = ISRestoreIndices(grid->cell_global,&cell_global);CHKERRQ(ierr);
  ierr = ISRestoreIndices(grid->vertex_global,&vertex_global);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


















