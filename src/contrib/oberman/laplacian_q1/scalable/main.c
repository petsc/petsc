/*$Id: main.c,v 1.6 2000/01/16 03:29:05 bsmith Exp $*/
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

EXTERN_C_BEGIN
extern int PCCreate_NN (PC);
extern int PCCreate_FETI (PC);
extern int MatPartitioningCreate_Square (MatPartitioning);
EXTERN_C_END

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int            ierr,its,rank;
  double         norm;
  AppCtx         *appctx;     /* contains all the data used by this PDE solver */

  /* ---------------------------------------------------------------------
     Initialize PETSc
     --------------------------------------------------------------------- */

  PetscFunctionBegin;

  PetscInitialize(&argc,&argv,PETSC_NULL,help);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRA(ierr);
  PCRegisterDynamic("nn",PETSC_NULL,"PCCreate_NN",PCCreate_NN);
  PCRegisterDynamic("feti",PETSC_NULL,"PCCreate_FETI",PCCreate_FETI);
  MatPartitioningRegisterDynamic("square",PETSC_NULL,"MatPartitioningCreate_Square",MatPartitioningCreate_Square);
                                                      
  /* Load the grid database -- in appload.c              */
  ierr = AppCtxCreate(PETSC_COMM_WORLD,&appctx);CHKERRA(ierr);

  /* Setup the linear system and solve it -- in appalgebra.c */
  ierr = AppCtxSolve(appctx,&its);CHKERRA(ierr);

  /* Save the solution, if this is the case. */ 
  {
    PetscTruth flg;
    ierr = OptionsHasName(PETSC_NULL,"-save_solution",&flg);CHKERRA(ierr);
    if (flg) {
      Viewer viewer;
      ierr = ViewerASCIIOpen(PETSC_COMM_WORLD,"solution.m",&viewer);CHKERRA(ierr);
      ierr = ViewerSetFormat(viewer,VIEWER_FORMAT_ASCII_MATLAB,"X");
      ierr = VecView(appctx->algebra.x,viewer);CHKERRA(ierr);
      ierr = ViewerDestroy(viewer);CHKERRA(ierr);
    }
  }

  {
    Vec r;
    Scalar m1 = -1.0;
    PetscReal petscnorm;
    ierr = VecDuplicate(appctx->algebra.b,&r);CHKERRA(ierr);
    ierr = MatMult(appctx->algebra.A,appctx->algebra.x,r);CHKERRA(ierr);
    ierr = VecAYPX(&m1,appctx->algebra.b,r);CHKERRA(ierr);
    ierr = VecNorm(r,NORM_2,&petscnorm);CHKERRA(ierr);
    ierr = VecDestroy(r);CHKERRA(ierr);
    norm = (double)petscnorm;
  }

  /* Destroy all datastructures  -- in appload.c */
  ierr = AppCtxDestroy(appctx);CHKERRA(ierr);

  /* Close down PETSc and stop the program */
  PetscFinalize();

  if (!rank) { printf("\nFinal residual norm: %e\n",norm); }
  if (!rank) { printf("\nNumber of Iterations: %d\n\n",its); }

  PetscFunctionReturn(0);
}











