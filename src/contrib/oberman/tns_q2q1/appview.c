
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
  int one = 1;
  int i,j;
  double *df_x, *df_y;
  int count;

  PetscFunctionBegin;
  /* now send the cell_coords */
  ierr = PetscDoubleView(appctx->element.dim*appctx->element.vel_basis_count*appctx->grid.cell_n, appctx->grid.cell_vcoords, viewer);
  /* send cell_df */
  ierr = PetscIntView((2*9+4)*appctx->grid.cell_n, appctx->grid.cell_df, viewer);CHKERRQ(ierr);

  /* send flag to show more values coming */
  ierr = PetscIntView(1, &one, viewer);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


