#pragma once
// Common data structures for rosenbrock1.c and rosenbrock1_taoterm.c

#include <petsctao.h>

/* User-defined application context

   contains data needed by the application-provided call-back routines that
   evaluate the function, gradient, and hessian.
*/
typedef struct {
  MPI_Comm  comm;
  PetscInt  n;         /* dimension */
  PetscReal alpha;     /* condition parameter */
  PetscBool chained;   /* chained vs. unchained Rosenbrock function */
  PetscBool test;      /* run tests in AppCtxFinalize() */
  PetscBool jacobi_pc; /* Create Jacobi Hpre */
  PetscBool use_fd;    /* Use finite difference for grad and hess */
} AppCtx;

static PetscErrorCode AppCtxInitialize(MPI_Comm, AppCtx *); /* process options */
static PetscErrorCode AppCtxFinalize(AppCtx *, Tao);        /* clean up and optionally run tests */
static PetscErrorCode AppCtxCreateSolution(AppCtx *, Vec *);
static PetscErrorCode AppCtxCreateHessianMatrices(AppCtx *, Mat *, Mat *);

static PetscErrorCode AppCtxInitialize(MPI_Comm comm, AppCtx *usr)
{
  PetscBool flg;

  PetscFunctionBeginUser;
  usr->comm      = comm;
  usr->n         = 2;
  usr->alpha     = 99.0;
  usr->chained   = PETSC_FALSE;
  usr->test      = PETSC_FALSE;
  usr->jacobi_pc = PETSC_FALSE;
  usr->use_fd    = PETSC_FALSE;
  /* Check for command line arguments to override defaults */
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &usr->n, &flg));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-alpha", &usr->alpha, &flg));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-chained", &usr->chained, &flg));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_lmvm", &usr->test, &flg));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-jacobi_pc", &usr->jacobi_pc, &flg));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-use_fd", &usr->use_fd, &flg));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AppCtxFinalize(AppCtx *usr, Tao tao)
{
  PetscFunctionBeginUser;
  if (usr->test) {
    /* Test the LMVM matrix */
    KSP       ksp;
    PC        pc;
    Mat       M;
    Vec       in, out, out2;
    PetscReal mult_solve_dist;
    Vec       x;

    PetscCall(TaoGetSolution(tao, &x));
    PetscCall(TaoGetKSP(tao, &ksp));
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCLMVMGetMatLMVM(pc, &M));
    PetscCall(VecDuplicate(x, &in));
    PetscCall(VecDuplicate(x, &out));
    PetscCall(VecDuplicate(x, &out2));
    PetscCall(VecSet(in, 1.0));
    PetscCall(MatMult(M, in, out));
    PetscCall(MatSolve(M, out, out2));
    PetscCall(VecAXPY(out2, -1.0, in));
    PetscCall(VecNorm(out2, NORM_2, &mult_solve_dist));
    if (mult_solve_dist < 1.e-11) {
      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)tao), "Inverse error of LMVM MatMult and MatSolve: < 1.e-11\n"));
    } else if (mult_solve_dist < 1.e-6) {
      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)tao), "Inverse error of LMVM MatMult and MatSolve: < 1.e-6\n"));
    } else {
      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)tao), "Inverse error of LMVM MatMult and MatSolve is not small: %e\n", (double)mult_solve_dist));
    }
    PetscCall(VecDestroy(&in));
    PetscCall(VecDestroy(&out));
    PetscCall(VecDestroy(&out2));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode AppCtxCreateSolution(AppCtx *usr, Vec *solution)
{
  PetscFunctionBeginUser;
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, usr->n, solution));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PETSC_UNUSED PetscErrorCode AppCtxCreateHessianMatrices(AppCtx *usr, Mat *H, Mat *Hpre)
{
  Mat hessian;

  PetscFunctionBeginUser;
  PetscCall(MatCreateSeqBAIJ(PETSC_COMM_SELF, 2, usr->n, usr->n, 1, NULL, &hessian));
  if (H) {
    PetscCall(PetscObjectReference((PetscObject)hessian));
    *H = hessian;
  }
  if (Hpre) {
    if (usr->jacobi_pc) PetscCall(MatCreateSeqBAIJ(PETSC_COMM_SELF, 1, usr->n, usr->n, 1, NULL, Hpre));
    else {
      PetscCall(PetscObjectReference((PetscObject)hessian));
      *Hpre = hessian;
    }
  }
  PetscCall(MatDestroy(&hessian));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  AppCtxFormFunctionGradient - Evaluates the function, f(X), and gradient, G(X).

  Input Parameters:
+ user - AppCtx
- X    - input vector

  Output Parameters:
+ f - function value
- G - vector containing the newly evaluated gradient
*/
static PetscErrorCode AppCtxFormFunctionGradient(AppCtx *user, Vec X, PetscReal *f, Vec G)
{
  PetscInt           nn = user->n / 2;
  PetscReal          ff = 0, t1, t2, alpha = user->alpha;
  PetscScalar       *g;
  const PetscScalar *x;

  PetscFunctionBeginUser;
  /* Get pointers to vector data */
  PetscCall(VecGetArrayRead(X, &x));
  PetscCall(VecGetArray(G, &g));

  /* Compute G(X) */
  if (user->chained) {
    g[0] = 0;
    for (PetscInt i = 0; i < user->n - 1; i++) {
      t1 = x[i + 1] - x[i] * x[i];
      ff += PetscSqr(1 - x[i]) + alpha * t1 * t1;
      g[i] += -2 * (1 - x[i]) + 2 * alpha * t1 * (-2 * x[i]);
      g[i + 1] = 2 * alpha * t1;
    }
  } else {
    for (PetscInt i = 0; i < nn; i++) {
      t1 = x[2 * i + 1] - x[2 * i] * x[2 * i];
      t2 = 1 - x[2 * i];
      ff += alpha * t1 * t1 + t2 * t2;
      g[2 * i]     = -4 * alpha * t1 * x[2 * i] - 2.0 * t2;
      g[2 * i + 1] = 2 * alpha * t1;
    }
  }

  /* Restore vectors */
  PetscCall(VecRestoreArrayRead(X, &x));
  PetscCall(VecRestoreArray(G, &g));
  *f = ff;

  PetscCall(PetscLogFlops(15.0 * nn));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  AppCtxFormHessian - Evaluates Hessian matrix.

  Input Parameters:
+ user - the context
- X    - input vector

  Output Parameters:
. H - Hessian matrix
*/
static PetscErrorCode AppCtxFormHessian(AppCtx *user, Vec X, Mat H)
{
  PetscInt           ind[2];
  PetscReal          alpha = user->alpha;
  PetscReal          v[2][2];
  const PetscScalar *x;
  PetscBool          assembled;

  PetscFunctionBeginUser;
  /* Zero existing matrix entries */
  PetscCall(MatAssembled(H, &assembled));
  if (assembled) PetscCall(MatZeroEntries(H));

  /* Get a pointer to vector data */
  PetscCall(VecGetArrayRead(X, &x));

  /* Compute H(X) entries */
  if (user->chained) {
    PetscCall(MatZeroEntries(H));
    for (PetscInt i = 0; i < user->n - 1; i++) {
      PetscScalar t1 = x[i + 1] - x[i] * x[i];
      v[0][0]        = 2 + 2 * alpha * (t1 * (-2) + 4 * x[i] * x[i]);
      v[0][1]        = 2 * alpha * (-2 * x[i]);
      v[1][0]        = 2 * alpha * (-2 * x[i]);
      v[1][1]        = 2 * alpha;
      ind[0]         = i;
      ind[1]         = i + 1;
      PetscCall(MatSetValues(H, 2, ind, 2, ind, v[0], ADD_VALUES));
    }
  } else {
    for (PetscInt i = 0; i < user->n / 2; i++) {
      v[1][1] = 2 * alpha;
      v[0][0] = -4 * alpha * (x[2 * i + 1] - 3 * x[2 * i] * x[2 * i]) + 2;
      v[1][0] = v[0][1] = -4.0 * alpha * x[2 * i];
      ind[0]            = 2 * i;
      ind[1]            = 2 * i + 1;
      PetscCall(MatSetValues(H, 2, ind, 2, ind, v[0], INSERT_VALUES));
    }
  }
  PetscCall(VecRestoreArrayRead(X, &x));

  /* Assemble matrix */
  PetscCall(MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY));
  PetscCall(PetscLogFlops(9.0 * user->n / 2.0));
  PetscFunctionReturn(PETSC_SUCCESS);
}
