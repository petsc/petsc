


/*
       Demonstrates how to draw two dimensional grids and solutions. See appctx.h
       for information on how this is to be used.
*/

#include "appctx.h"
#include "bitarray.h"

#undef __FUNC__
#define __FUNC__ "AppCxtView"
PetscErrorCode AppCtxView(Draw idraw,void *iappctx)
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

  PetscBT                vertex_boundary_flag;

  PetscErrorCode ierr;
  PetscInt              i,c,j,ijp;
  PetscMPIInt rank;
  PetscInt                    ij;

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

  ierr = ISGetSize(grid->isvertex_boundary,&nverts);CHKERRQ(ierr);
  ierr = ISGetIndices(grid->isvertex_boundary,&verts);CHKERRQ(ierr);

  /*
        Draw edges of local cells and number them
  */
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
    ierr = DrawString(drawlocal,xp,yp,DRAW_GREEN,num);CHKERRQ(ierr);
    sprintf(num,"%d",cell_global[i]);
    ierr = DrawString(drawglobal,xp,yp,c,num);CHKERRQ(ierr);
  }

  /*
       Draws only boundary edges 
  */
  for (i=0; i<cell_n; i++ ) {
    xp = 0.0; yp = 0.0;
    xl  = vertex_value[2*cell_vertex[ncell*i]]; yl = vertex_value[2*cell_vertex[ncell*i] + 1];
    ijp = ncell*i;
    for (j=0; j<ncell; j++) {
      ij = ncell*i + ((j+1) % ncell);
      xr = vertex_value[2*cell_vertex[ij]]; yr = vertex_value[2*cell_vertex[ij] + 1];
      if (PetscBTLookup(vertex_boundary_flag,cell_vertex[ijp]) && PetscBTLookup(vertex_boundary_flag,cell_vertex[ij])) {
        ierr = DrawLine(drawglobal,xl,yl,xr,yr,c);CHKERRQ(ierr);
        ierr = DrawLine(drawlocal,xl,yl,xr,yr,DRAW_BLUE);CHKERRQ(ierr);
      }
      xp += xl;         yp += yl;
      xl  = xr;         yl =  yr;
      ijp = ij;
    }
    xp /= ncell; yp /= ncell;
    sprintf(num,"%d",i);
    ierr = DrawString(drawlocal,xp,yp,DRAW_GREEN,num);CHKERRQ(ierr);
    sprintf(num,"%d",cell_global[i]);
    ierr = DrawString(drawglobal,xp,yp,c,num);CHKERRQ(ierr);
  }

  /*
      Number vertices
  */
  for (i=0; i<vertex_n; i++ ) {
    xm = vertex_value[2*i]; ym = vertex_value[2*i + 1];
    ierr = DrawString(drawglobal,xm,ym,DRAW_BLUE,num);CHKERRQ(ierr);
    ierr = DrawPoint(drawglobal,xm,ym,DRAW_ORANGE);CHKERRQ(ierr);
    ierr = DrawPoint(drawlocal,xm,ym,DRAW_ORANGE);CHKERRQ(ierr);
    sprintf(num,"%d",i);
    ierr = DrawString(drawlocal,xm,ym,DRAW_BLUE,num);CHKERRQ(ierr);
    sprintf(num,"%d",vertex_global[i]);
    ierr = DrawString(drawglobal,xm,ym,DRAW_BLUE,num);CHKERRQ(ierr);
  }
  for ( i=0; i<nverts; i++ ) {
    xm = vertex_value[2*verts[i]]; ym = vertex_value[2*verts[i] + 1];
    ierr = DrawPoint(drawglobal,xm,ym,DRAW_RED);CHKERRQ(ierr);
    ierr = DrawPoint(drawlocal,xm,ym,DRAW_RED);CHKERRQ(ierr);
  }
  ierr = DrawSynchronizedFlush(drawglobal);CHKERRQ(ierr);
  ierr = DrawSynchronizedFlush(drawlocal);CHKERRQ(ierr);

  ierr = ISRestoreIndices(grid->isvertex_boundary,&verts);CHKERRQ(ierr);
  ierr = ISRestoreIndices(grid->cell_global,&cell_global); CHKERRQ(ierr);
  ierr = ISRestoreIndices(grid->vertex_global,&vertex_global); CHKERRQ(ierr);

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
#undef __FUNC__
#define __FUNC__ "AppCxtViewMatlab"
PetscErrorCode AppCtxViewMatlab(AppCtx* appctx)
{
  PetscErrorCode    ierr;
  PetscInt *cell_vertex,rstart,rend;
#if defined(PETSC_USE_SOCKET_VIEWER)
  Viewer viewer = PETSC_VIEWER_SOCKET_WORLD;
#else
  Viewer viewer = PETSC_VIEWER_BINARY_WORLD;
#endif
  double *vertex_values;
  IS     isvertex;
PetscInt one = 1;
PetscInt i;
  PetscFunctionBegin;

  /* now send the cell_coords */
ierr = PetscDoubleView(8*appctx->grid.cell_n, appctx->grid.cell_coords, viewer);

/* send cell_df */
ierr = PetscIntView(9*appctx->grid.cell_n, appctx->grid.cell_df, viewer);CHKERRQ(ierr);


ierr = PetscIntView(1, &one, viewer);CHKERRQ(ierr);

/* next send the cell

  /* now send the number of steps coming */
  
 /******** Here need to send multiple solution vectors **********/
/*   for(i=0;i<NSTEPS;i++)*/
  /*   {ierr = VecView(appctx->algebra.solnv[i],viewer); CHKERRQ(ierr);} */


  /* Next, send vertices to Matlab */
/*   ierr = AODataKeyGetOwnershipRange(appctx->aodata,"vertex",&rstart,&rend); CHKERRQ(ierr); */
/*   ierr = ISCreateStride(PETSC_COMM_WORLD,rend-rstart,rstart,1,&isvertex); CHKERRQ(ierr); */
/*   ierr = AODataSegmentGetIS(appctx->aodata,"vertex","values",isvertex,(void **)&vertex_values); */
/*          CHKERRQ(ierr); */
/*   ierr = PetscDoubleView(2*(rend-rstart),vertex_values,viewer); CHKERRQ(ierr); */
/*   ierr = AODataSegmentRestoreIS(appctx->aodata,"vertex","values",PETSC_NULL,(void **)&vertex_values); */
/*          CHKERRQ(ierr); */
/*   ierr = ISDestroy(isvertex); CHKERRQ(ierr); */

  /* 
     Send list of vertices for each cell; these MUST be in the global (not local!) numbering); 
     this cannot use appctx->grid->cell_vertex 
  */
/*   ierr = AODataSegmentGetIS(appctx->aodata,"cell","vertex",appctx->grid.cell_global, */
/*         (void **)&cell_vertex); CHKERRQ(ierr); */
/*   ierr = PetscIntView(4*appctx->grid.cell_n,cell_vertex,viewer); CHKERRQ(ierr); */
/*   ierr = AODataSegmentRestoreIS(appctx->aodata,"cell","vertex",PETSC_NULL,(void **)&cell_vertex);  */
/*          CHKERRQ(ierr); */
  
  PetscFunctionReturn(0);
}


