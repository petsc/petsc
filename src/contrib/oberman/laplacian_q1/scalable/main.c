
static char help[] = "Solves 2d-laplacian on quadrilateral grid.\n\
   Options:\n\
    -show_solution pipe solution to matlab (visualized with bscript.m).\n\
    -show_griddata print the local index sets and local to global mappings \n\
    -show_matrix visualize the sparsity structure of the stiffness matrix.\n\
    -show_grid visualize the global and local grids with numbering.\n";

/*
    The file appctx.h includes all the data structures used by this code
*/
#include "appctx.h"


#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  int            ierr,rank;
  double         norm;
  AppCtx         *appctx;     /* contains all the data used by this PDE solver */

  /* ---------------------------------------------------------------------
     Initialize PETSc
     --------------------------------------------------------------------- */

  PetscFunctionBegin;

  PetscInitialize(&argc,&argv,PETSC_NULL,help);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
                                                      
  /* Load the grid database -- in appload.c              */
  ierr = AppCtxCreate(PETSC_COMM_WORLD,&appctx);CHKERRQ(ierr);

  /* Setup the linear system and solve it -- in appalgebra.c */
  ierr = AppCtxSolve(appctx);CHKERRQ(ierr);

  /* Save the solution, if this is the case. */ 
  {
    PetscTruth flg;
    ierr = PetscOptionsHasName(PETSC_NULL,"-save_solution",&flg);CHKERRQ(ierr);
    if (flg) {
      PetscViewer viewer;
      ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,"solution.m",&viewer);CHKERRQ(ierr);
      ierr = PetscViewerSetFormat(viewer,PETSC_VIEWER_ASCII_MATLAB);
      ierr = VecView(appctx->algebra.x,viewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
    }
  }

  {
    Vec r;
    PetscScalar m1 = -1.0;
    PetscReal   petscnorm;
    ierr = VecDuplicate(appctx->algebra.b,&r);CHKERRQ(ierr);
    ierr = MatMult(appctx->algebra.A,appctx->algebra.x,r);CHKERRQ(ierr);
    ierr = VecAYPX(r,m1,appctx->algebra.b);CHKERRQ(ierr);
    ierr = VecNorm(r,NORM_2,&petscnorm);CHKERRQ(ierr);
    ierr = VecDestroy(r);CHKERRQ(ierr);
    norm = (double)petscnorm;
  }

  /* Destroy all datastructures  -- in appload.c */
  ierr = AppCtxDestroy(appctx);CHKERRQ(ierr);

  /* Close down PETSc and stop the program */
  ierr = PetscFinalize();CHKERRQ(ierr);

  if (!rank) { printf("\nFinal residual norm: %e\n",norm); }

  PetscFunctionReturn(0);
}











