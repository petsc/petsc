/*$Id: main.c,v 1.4 2000/01/06 21:32:13 bsmith Exp bsmith $*/
static char help[] =
"Solves 2d-laplacian on quadrilateral grid.\n\
   Options:\n\
    -show_solution pipe solution to matlab (visualized with bscript.m).\n\
    -show_griddata print the local index sets and local to global mappings \n\
    -show_matrix visualize the sparsity structure of the stiffness matrix.\n\
    -show_grid visualize the global and local grids with numbering.\n";

/*
    The file appctx.h includes all the data structures used by this code
*/
#include "appctx.h"

int main(int argc,char **argv)
{
  int            ierr;
  AppCtx         *appctx;     /* contains all the data used by this PDE solver */

  /* ---------------------------------------------------------------------
     Initialize PETSc
     ----------------- ---------------------------------------------------*/
  PetscInitialize(&argc,&argv,(char *)0,help);

  /*  Load the grid database -- in appload.c              */
  ierr = AppCtxCreate(PETSC_COMM_WORLD,&appctx);CHKERRA(ierr);

  /*   Setup the graphics routines to view the grid -- in appview.c  */
  ierr = AppCtxGraphics(appctx);CHKERRA(ierr);
 
  /*   Setup the linear system and solve it -- in appsetalg.c */
  ierr = AppCtxSolve(appctx);CHKERRA(ierr);

  /*   Send solution to  matlab viewer -- in appview.c */
  if (appctx->view.show_solution) {
    ierr = AppCtxViewMatlab(appctx);CHKERRA(ierr);  
  }

  /*  Destroy all datastructures  -- in appload.c */
  ierr = AppCtxDestroy(appctx);CHKERRA(ierr);

  /* Close down PETSc and stop the program */
  PetscFinalize();
  return 0;
}










