/* Program usage: mpiexec -n 1 rosenbrock1 [-help] [all TAO options] */

/*  Include "petsctao.h" so we can use TAO solvers.  */
#include <petsctao.h>
#include "rosenbrock1.h" // defines AppCtx, AppCtxFormFunctionGradient(), and AppCtxFormHessian()

static char help[] = "This example demonstrates use of the TAO package to \n\
solve an unconstrained minimization problem on a single processor.  We \n\
minimize the extended Rosenbrock function: \n\
   sum_{i=0}^{n/2-1} (alpha*(x_{2i+1}-x_{2i}^2)^2 + (1-x_{2i})^2) \n\
or the chained Rosenbrock function:\n\
   sum_{i=0}^{n-1} alpha*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2\n";

/* -------------- User-defined routines ---------- */
static PetscErrorCode FormFunctionGradient(Tao, Vec, PetscReal *, Vec, void *);
static PetscErrorCode FormHessian(Tao, Vec, Mat, Mat, void *);

int main(int argc, char **argv)
{
  Vec         x; /* solution vector */
  Mat         H, Hpre;
  Tao         tao;  /* Tao solver context */
  PetscMPIInt size; /* number of processes running */
  AppCtx      user; /* user-defined application context */
  MPI_Comm    comm;

  /* Initialize TAO and PETSc */
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCheck(size == 1, comm, PETSC_ERR_WRONG_MPI_SIZE, "Incorrect number of processors");

  /* Initialize problem parameters */
  PetscCall(AppCtxInitialize(comm, &user));

  /* Allocate vector for the solution */
  PetscCall(AppCtxCreateSolution(&user, &x));

  /* Allocate the Hessian matrix */
  PetscCall(AppCtxCreateHessianMatrices(&user, &H, &Hpre));

  /* The TAO code begins here */

  /* Create TAO solver with desired solution method */
  PetscCall(TaoCreate(comm, &tao));
  PetscCall(TaoSetType(tao, TAOLMVM));

  /* Set solution vec and an initial guess */
  PetscCall(VecZeroEntries(x));
  PetscCall(TaoSetSolution(tao, x));

  /* Set routines for function, gradient, hessian evaluation */
  PetscCall(TaoSetObjectiveAndGradient(tao, NULL, FormFunctionGradient, &user));
  PetscCall(TaoSetHessian(tao, H, Hpre, FormHessian, &user));

  /* Check for TAO command line options */
  PetscCall(TaoSetFromOptions(tao));

  /* SOLVE THE APPLICATION */
  PetscCall(TaoSolve(tao));

  /* Clean up */
  PetscCall(AppCtxFinalize(&user, tao));
  PetscCall(TaoDestroy(&tao));
  PetscCall(VecDestroy(&x));
  PetscCall(MatDestroy(&H));
  PetscCall(MatDestroy(&Hpre));

  PetscCall(PetscFinalize());
  return 0;
}

/*
  FormFunctionGradient - Evaluates the function, f(X), and gradient, G(X).

  Input Parameters:
+ tao  - the Tao context
. X    - input vector
- ptr  - optional user-defined context, as set by TaoSetFunctionGradient()

  Output Parameters:
+ f - function value
- G - vector containing the newly evaluated gradient

  Note:
  Some optimization methods ask for the function and the gradient evaluation
  at the same time.  Evaluating both at once may be more efficient that
  evaluating each separately.
*/
static PetscErrorCode FormFunctionGradient(Tao tao, Vec X, PetscReal *f, Vec G, void *ptr)
{
  AppCtx *user = (AppCtx *)ptr;

  PetscFunctionBeginUser;
  PetscCall(AppCtxFormFunctionGradient(user, X, f, G));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  FormHessian - Evaluates Hessian matrix.

  Input Parameters:
+ tao   - the Tao context
. x     - input vector
- ptr   - optional user-defined context, as set by TaoSetHessian()

  Output Parameters:
+ H     - Hessian matrix
- Hpre  - Preconditioning matrix

  Note:  Providing the Hessian may not be necessary.  Only some solvers
  require this matrix.
*/
static PetscErrorCode FormHessian(Tao tao, Vec X, Mat H, Mat Hpre, void *ptr)
{
  AppCtx *user = (AppCtx *)ptr;

  PetscFunctionBeginUser;
  PetscCall(AppCtxFormHessian(user, X, H));
  // Manual Jacobi preconditioner for testing
  if (user->jacobi_pc) {
    Vec v;

    PetscCall(VecDuplicate(X, &v));
    PetscCall(MatGetDiagonal(H, v));
    PetscCall(MatZeroEntries(Hpre));
    PetscCall(MatDiagonalSet(Hpre, v, INSERT_VALUES));
    PetscCall(VecDestroy(&v));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*TEST

   build:
     requires: !complex !single !quad !defined(PETSC_USE_64BIT_INDICES) !__float128

   test:
     args: -tao_monitor_short -tao_type nls -tao_gatol 1.e-4

   test:
     suffix: 2
     args: -tao_monitor_short -tao_type lmvm -tao_gatol 1.e-3

   test:
     suffix: 3
     args: -tao_monitor_short -tao_type ntr -tao_gatol 1.e-4

   test:
     suffix: 4
     args: -tao_monitor_short -tao_type ntr -tao_mf_hessian -tao_ntr_pc_type none -tao_gatol 1.e-4

   test:
     suffix: 5
     args: -tao_monitor_short -tao_type bntr -tao_gatol 1.e-4

   test:
     suffix: 6
     args: -tao_monitor_short -tao_type bntl -tao_gatol 1.e-4

   test:
     suffix: 7
     args: -tao_monitor_short -tao_type bnls -tao_gatol 1.e-4

   test:
     suffix: 8
     args: -tao_monitor_short -tao_type bntr -tao_bnk_max_cg_its 3 -tao_gatol 1.e-4 -tao_bnk_cg_tao_monitor_short

   test:
     suffix: 9
     args: -tao_monitor_short -tao_type bntl -tao_bnk_max_cg_its 3 -tao_gatol 1.e-4 -tao_bnk_cg_tao_monitor_short

   test:
     suffix: 10
     args: -tao_monitor_short -tao_type bnls -tao_bnk_max_cg_its 3 -tao_gatol 1.e-4 -tao_bnk_cg_tao_monitor_short

   test:
     suffix: 11
     args: -test_lmvm -tao_type bqnktr -tao_max_it 10 -tao_bqnk_mat_type lmvmbroyden

   test:
     suffix: 12
     args: -test_lmvm -tao_type bqnktr -tao_max_it 10 -tao_bqnk_mat_type lmvmbadbroyden

   test:
     suffix: 13
     args: -test_lmvm -tao_type bqnktr -tao_max_it 10 -tao_bqnk_mat_type lmvmsymbroyden -tao_bqnk_mat_lmvm_beta {{0.0 0.25 1.0}} -tao_bqnk_mat_lmvm_rho 0.75 -tao_bqnk_mat_lmvm_sigma_hist 2

   test:
     suffix: 14
     args: -test_lmvm -tao_type bqnktr -tao_max_it 10 -tao_bqnk_mat_type lmvmbfgs -tao_bqnk_mat_lmvm_scale_type {{scalar diagonal}} -tao_bqnk_mat_lmvm_alpha {{0.0 0.25 0.5}} -tao_bqnk_mat_lmvm_theta 1.0

   test:
     suffix: 15
     args: -test_lmvm -tao_type bqnktr -tao_max_it 10 -tao_bqnk_mat_type lmvmdfp

   test:
     suffix: 16
     args: -test_lmvm -tao_type bqnktr -tao_max_it 10 -tao_bqnk_mat_type lmvmsr1

   test:
     suffix: 17
     args: -tao_monitor_short -tao_gatol 1e-4 -tao_type bqnls

   test:
     suffix: 18
     args: -tao_monitor_short -tao_gatol 1e-4 -tao_type blmvm

   test:
     suffix: 19
     args: -tao_monitor_short -tao_gatol 1e-4 -tao_type bqnktr -tao_bqnk_mat_type lmvmsr1

   test:
     suffix: 20
     args: -tao_monitor -tao_gatol 1e-4 -tao_type blmvm -tao_ls_monitor

   test:
     suffix: 21
     args: -test_lmvm -tao_type bqnktr -tao_max_it 10 -tao_bqnk_mat_type lmvmsymbadbroyden

   test:
     suffix: 22
     args: -tao_max_it 1 -tao_converged_reason

   test:
     suffix: 23
     args: -tao_max_funcs 0 -tao_converged_reason

   test:
     suffix: 24
     args: -tao_gatol 10 -tao_converged_reason

   test:
     suffix: 25
     args: -tao_grtol 10 -tao_converged_reason

   test:
     suffix: 26
     args: -tao_gttol 10 -tao_converged_reason

   test:
     suffix: 27
     args: -tao_steptol 10 -tao_converged_reason

   test:
     suffix: 28
     args: -tao_fmin 10 -tao_converged_reason

   test:
     suffix: snes
     args: -snes_monitor ::ascii_info_detail -tao_type snes -snes_type newtontr -snes_atol 1.e-4 -pc_type none -tao_mf_hessian -ksp_type cg

   test:
     suffix: snes_ls_armijo
     args: -snes_monitor ::ascii_info_detail -tao_type snes -snes_type newtonls -snes_atol 1.e-4 -pc_type none -tao_mf_hessian -snes_linesearch_monitor -snes_linesearch_order 1

   test:
     suffix: snes_tr_cgnegcurve_kmdc
     args: -snes_monitor ::ascii_info_detail -tao_type snes -snes_type newtontr -snes_atol 1.e-4 -pc_type none -ksp_type cg -snes_tr_kmdc 0.9 -ksp_converged_neg_curve -ksp_converged_reason

   test:
     suffix: snes_ls_lmvm
     args: -snes_monitor ::ascii_info_detail -tao_type snes -snes_type newtonls -snes_atol 1.e-4 -pc_type lmvm -tao_mf_hessian

   test:
     suffix: add_terms_l2_no_pre
     args: -tao_type nls -tao_add_terms reg_ -reg_tao_term_type halfl2squared -tao_term_sum_reg_scale 0.3 -tao_monitor_short -tao_view ::ascii_info_detail

   test:
     suffix: add_terms_l1_no_pre
     args: -tao_type nls -tao_add_terms reg_ -reg_tao_term_type l1 -reg_tao_term_l1_epsilon 0.4 -tao_term_sum_reg_scale 0.3 -tao_monitor_short -tao_view ::ascii_info_detail

   test:
     suffix: hpre_is_not_h
     args: -tao_type nls -jacobi_pc 1 -tao_view ::ascii_info_detail -n 10

   test:
     suffix: param_none
     args: -tao_type nls -tao_add_terms reg_ -reg_tao_term_type halfl2squared -tao_term_sum_reg_scale 0.3 -tao_monitor_short
     args: -tao_view ::ascii_info_detail -reg_tao_term_parameters_mode none

TEST*/
