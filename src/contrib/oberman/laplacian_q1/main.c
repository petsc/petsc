
static char help[] =
"Solves 2d-laplacian on quadrilateral grid.\n\
   Options:\n\
    -matlabgraphics pipe solution to matlab (visualize with bscript).\n\
    -show_griddata print the local index sets and local to global mappings \n\
    -show_solution plots the solution with a contour graph \n\
    -show_matrix visualize the sparsity structure of the stiffness matrix.\n\
    -show_grid visualize the global and local grids with numbering.\n";

#include "appctx.h"

int main( int argc, char **argv )
{
  int            ierr;
  AppCtx         *appctx;

  /* ---------------------------------------------------------------------
     Initialize PETSc
     --------------------- ---------------------------------------------------*/
  PetscInitialize(&argc,&argv,(char *)0,help);

  /*  Load the grid database 
      in appload.c              */
  ierr = AppCtxCreate(PETSC_COMM_WORLD,&appctx); CHKERRA(ierr);

  /*   Setup the graphics routines to view the grid
       in appview.c  */
  ierr = AppCtxGraphics(appctx); CHKERRA(ierr);
 
  /*   Setup the linear system and solve it 
       in appsetalg.c */
  ierr = AppCtxSolve(appctx);CHKERRA(ierr);

  /*   Send solution to  matlab viewer 
       in appview.c */
  if (appctx->view.matlabgraphics) {
    ierr = AppCtxViewMatlab(appctx); CHKERRA(ierr);  
  }

  /*  Destroy all datastructures  
      in appload.c */
  ierr = AppCtxDestroy(appctx); CHKERRA(ierr);

  /* Close down PETSc and stop the program */
  PetscFinalize();
  return 0;
}










