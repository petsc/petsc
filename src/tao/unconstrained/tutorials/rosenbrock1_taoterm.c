/*  Include "petsctao.h" so we can use TAO solvers.  */
#include <petsctao.h>
#include "rosenbrock1.h" // defines AppCtx, AppCtxFormFunctionGradient(), and AppCtxFormHessian()

static char help[] = "This example demonstrates use of the TaoTerm\n\
interface for defining problems in the Tao library.  This example\n\
should be compared to rosenbrock1.c, which uses the callback interface\n\
to define the Rosenbrock function.\n";

static PetscErrorCode FormFunctionGradient(TaoTerm, Vec, Vec, PetscReal *, Vec);
static PetscErrorCode FormHessian(TaoTerm, Vec, Vec, Mat, Mat);
static PetscErrorCode CreateSolutionVec(TaoTerm, Vec *);

static PetscErrorCode CtxDestroy(PetscCtxRt ctx)
{
  PetscFunctionBeginUser;
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  TaoTerm     objective;
  Tao         tao;  /* Tao solver context */
  PetscMPIInt size; /* number of processes running */
  AppCtx      user; /* user-defined application context */
  MPI_Comm    comm;
  PetscBool   test_gradient_fd_check = PETSC_FALSE; /* test that FD delta is preserved */
  PetscReal   fd_delta_set           = 1.e-6;

  /* Initialize TAO and PETSc */
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCheck(size == 1, comm, PETSC_ERR_WRONG_MPI_SIZE, "Incorrect number of processors");

  PetscOptionsBegin(comm, "", help, "none");
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_gradient_fd_check", &test_gradient_fd_check, NULL));
  PetscOptionsEnd();
  /* Initialize problem parameters */
  PetscCall(AppCtxInitialize(comm, &user));

  /* Define the objective function */
  PetscCall(TaoTermCreateShell(comm, &user, CtxDestroy, &objective));
  PetscCall(TaoTermSetParametersMode(objective, TAOTERM_PARAMETERS_NONE));
  PetscCall(TaoTermShellSetCreateSolutionVec(objective, CreateSolutionVec));
  PetscCall(TaoTermShellSetObjectiveAndGradient(objective, FormFunctionGradient));
  PetscCall(TaoTermShellSetCreateHessianMatrices(objective, TaoTermCreateHessianMatricesDefault));
  if (user.jacobi_pc) PetscCall(TaoTermSetCreateHessianMode(objective, PETSC_FALSE, MATBAIJ, MATBAIJ));
  else PetscCall(TaoTermSetCreateHessianMode(objective, PETSC_TRUE /* H == Hpre */, MATBAIJ, NULL));
  PetscCall(TaoTermShellSetHessian(objective, FormHessian));

  /* Create TAO solver with desired solution method */
  PetscCall(TaoCreate(PETSC_COMM_SELF, &tao));
  PetscCall(TaoSetType(tao, TAOLMVM));

  if (user.use_fd) {
    PetscCall(TaoTermSetFDDelta(objective, 7.e-9));
    PetscCall(TaoTermComputeGradientSetUseFD(objective, PETSC_TRUE));
    PetscCall(TaoTermComputeHessianSetUseFD(objective, PETSC_TRUE));
  }
  /* Set routines for function, gradient, hessian evaluation */
  PetscCall(TaoAddTerm(tao, NULL, 1.0, objective, NULL, NULL));

  /* Check for TAO command line options */
  PetscCall(TaoSetFromOptions(tao));

  /* Set FD delta for testing if requested (after options processing) */
  if (test_gradient_fd_check) PetscCall(TaoTermSetFDDelta(objective, fd_delta_set));

  /* SOLVE THE APPLICATION */
  PetscCall(TaoSolve(tao));

  /* Check that FD delta is preserved if testing */
  if (test_gradient_fd_check) {
    PetscReal fd_delta_get;

    PetscCall(TaoTermGetFDDelta(objective, &fd_delta_get));
    PetscCheck(PetscAbsReal(fd_delta_get - 1.e-6) < 1.e-15, comm, PETSC_ERR_PLIB, "FD delta changed: set %g, got %g", (double)fd_delta_set, (double)fd_delta_get);
  }

  /* Clean up */
  PetscCall(AppCtxFinalize(&user, tao));
  PetscCall(TaoDestroy(&tao));
  PetscCall(TaoTermDestroy(&objective));

  PetscCall(PetscFinalize());
  return 0;
}

/*
  FormFunctionGradient - Evaluates the function, f(X), and gradient, G(X).

  Input Parameters:
+ term              - the `TaoTerm` for the objective function
. X                 - input vector
- parameters_unused - optional vector of parameters that this rosenbrock function does not use

  Output Parameters:
+ f - function value
- G - vector containing the newly evaluated gradient

  Note:
  Some optimization methods ask for the function and the gradient evaluation
  at the same time.  Evaluating both at once may be more efficient that
  evaluating each separately.
*/
static PetscErrorCode FormFunctionGradient(TaoTerm term, Vec X, Vec parameters_unused, PetscReal *f, Vec G)
{
  AppCtx *user;

  PetscFunctionBeginUser;
  PetscCheck(parameters_unused == NULL, PetscObjectComm((PetscObject)term), PETSC_ERR_ARG_WRONG, "Rosenbrock function does not take a parameter vector");
  PetscCall(TaoTermShellGetContext(term, &user));
  PetscCall(AppCtxFormFunctionGradient(user, X, f, G));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  FormHessian - Evaluates Hessian matrix.

  Input Parameters:
+ tao    - the Tao context
. x      - input vector
. params - optional vector of parameters that this rosenbrock function does not use
- ptr    - optional user-defined context, as set by TaoSetHessian()

  Output Parameters:
+ H    - Hessian matrix
- Hpre - Preconditioning matrix

  Note:  Providing the Hessian may not be necessary.  Only some solvers
  require this matrix.
*/
static PetscErrorCode FormHessian(TaoTerm term, Vec X, Vec params, Mat H, Mat Hpre)
{
  AppCtx *user;

  PetscFunctionBeginUser;
  PetscCheck(params == NULL, PetscObjectComm((PetscObject)term), PETSC_ERR_ARG_WRONG, "Rosenbrock function does not take a parameter vector");
  PetscCall(TaoTermShellGetContext(term, &user));
  if (H) PetscCall(AppCtxFormHessian(user, X, H));
  if (Hpre && Hpre != H) {
    if (user->jacobi_pc) {
      Vec v;

      PetscCall(VecDuplicate(X, &v));
      PetscCall(MatGetDiagonal(H, v));
      PetscCall(MatZeroEntries(Hpre));
      PetscCall(MatDiagonalSet(Hpre, v, INSERT_VALUES));
      PetscCall(VecDestroy(&v));
    } else {
      if (H) PetscCall(MatCopy(H, Hpre, SAME_NONZERO_PATTERN));
      else PetscCall(AppCtxFormHessian(user, X, Hpre));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateSolutionVec(TaoTerm term, Vec *solution)
{
  AppCtx *user;

  PetscFunctionBeginUser;
  PetscCall(TaoTermShellGetContext(term, &user));
  PetscCall(AppCtxCreateSolution(user, solution));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*TEST

   build:
     requires: !complex !single !quad !defined(PETSC_USE_64BIT_INDICES) !__float128

   test:
     output_file: output/rosenbrock1_1.out
     args: -tao_monitor_short -tao_type nls -tao_gatol 1.e-4

   test:
     suffix: test_gradient
     args: -tao_monitor_short -tao_type nls -tao_gatol 1.e-4 -tao_test_gradient -tao_fd_delta 1.e-6 -n 4 -chained -tao_term_hessian_mat_type aij -alpha 49.0

   test:
     suffix: test_gradient_fd_check
     args: -tao_monitor_short -tao_type nls -tao_gatol 1.e-4 -tao_test_gradient
     args: -n 4 -chained -tao_term_hessian_mat_type aij -alpha 49.0 -test_gradient_fd_check 1

   test:
     suffix: fd_grad
     args: -tao_monitor_short -tao_type nls -tao_term_gradient_use_fd

   test:
     suffix: fd_hess
     args: -tao_monitor_short -tao_type nls -tao_term_hessian_use_fd

   test:
     suffix: fd_hess_diffpre
     args: -tao_monitor_short -tao_type nls -tao_term_hessian_use_fd -tao_term_hessian_pre_is_hessian 0

   test:
     suffix: use_fd
     args: -tao_monitor_short -tao_type nls -use_fd

   test:
     suffix: test_fd_hess
     args: -tao_type nls -tao_fd_hessian -tao_view -tao_monitor_short

   test:
     suffix: test_mf_hessian
     args: -tao_type nls -tao_term_hessian_mat_type mffd -tao_monitor_short -tao_view

   test:
     suffix: add_term
     args: -tao_type nls -tao_add_terms extra_ -extra_tao_term_type halfl2squared -tao_view ::ascii_info_detail

   test:
     suffix: separate_hessians
     requires: !single
     args: -tao_monitor_short -tao_type nls -tao_gatol 1.e-4 -tao_term_hessian_pre_is_hessian 0 -tao_term_hessian_mat_type aij -tao_term_hessian_pre_mat_type sbaij -tao_view

TEST*/
