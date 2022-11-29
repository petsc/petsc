
static char help[] = "Solves u`` + u^{2} = f with Newton-like methods. Using\n\
 matrix-free techniques with user-provided explicit preconditioner matrix.\n\n";

#include <petscsnes.h>

extern PetscErrorCode FormJacobian(SNES, Vec, Mat, Mat, void *);
extern PetscErrorCode FormJacobianNoMatrix(SNES, Vec, Mat, Mat, void *);
extern PetscErrorCode FormFunction(SNES, Vec, Vec, void *);
extern PetscErrorCode FormFunctioni(void *, PetscInt, Vec, PetscScalar *);
extern PetscErrorCode OtherFunctionForDifferencing(void *, Vec, Vec);
extern PetscErrorCode FormInitialGuess(SNES, Vec);
extern PetscErrorCode Monitor(SNES, PetscInt, PetscReal, void *);

typedef struct {
  PetscViewer viewer;
} MonitorCtx;

typedef struct {
  PetscBool variant;
} AppCtx;

int main(int argc, char **argv)
{
  SNES        snes;                /* SNES context */
  SNESType    type = SNESNEWTONLS; /* default nonlinear solution method */
  Vec         x, r, F, U;          /* vectors */
  Mat         J, B;                /* Jacobian matrix-free, explicit preconditioner */
  AppCtx      user;                /* user-defined work context */
  PetscScalar h, xp  = 0.0, v;
  PetscInt    its, n = 5, i;
  PetscBool   puremf = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  PetscCall(PetscOptionsHasName(NULL, NULL, "-variant", &user.variant));
  h = 1.0 / (n - 1);

  /* Set up data structures */
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, n, &x));
  PetscCall(PetscObjectSetName((PetscObject)x, "Approximate Solution"));
  PetscCall(VecDuplicate(x, &r));
  PetscCall(VecDuplicate(x, &F));
  PetscCall(VecDuplicate(x, &U));
  PetscCall(PetscObjectSetName((PetscObject)U, "Exact Solution"));

  /* create explicit matrix preconditioner */
  PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, n, n, 3, NULL, &B));

  /* Store right-hand-side of PDE and exact solution */
  for (i = 0; i < n; i++) {
    v = 6.0 * xp + PetscPowScalar(xp + 1.e-12, 6.0); /* +1.e-12 is to prevent 0^6 */
    PetscCall(VecSetValues(F, 1, &i, &v, INSERT_VALUES));
    v = xp * xp * xp;
    PetscCall(VecSetValues(U, 1, &i, &v, INSERT_VALUES));
    xp += h;
  }

  /* Create nonlinear solver */
  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
  PetscCall(SNESSetType(snes, type));

  /* Set various routines and options */
  PetscCall(SNESSetFunction(snes, r, FormFunction, F));
  if (user.variant) {
    /* this approach is not normally needed, one should use the MatCreateSNESMF() below usually */
    PetscCall(MatCreateMFFD(PETSC_COMM_WORLD, n, n, n, n, &J));
    PetscCall(MatMFFDSetFunction(J, (PetscErrorCode(*)(void *, Vec, Vec))SNESComputeFunction, snes));
    PetscCall(MatMFFDSetFunctioni(J, FormFunctioni));
    /* Use the matrix free operator for both the Jacobian used to define the linear system and used to define the preconditioner */
    /* This tests MatGetDiagonal() for MATMFFD */
    PetscCall(PetscOptionsHasName(NULL, NULL, "-puremf", &puremf));
  } else {
    /* create matrix free matrix for Jacobian */
    PetscCall(MatCreateSNESMF(snes, &J));
    /* demonstrates differencing a different function than FormFunction() to apply a matrix operator */
    /* note we use the same context for this function as FormFunction, the F vector */
    PetscCall(MatMFFDSetFunction(J, OtherFunctionForDifferencing, F));
  }

  /* Set various routines and options */
  PetscCall(SNESSetJacobian(snes, J, puremf ? J : B, puremf ? FormJacobianNoMatrix : FormJacobian, &user));
  PetscCall(SNESSetFromOptions(snes));

  /* Solve nonlinear system */
  PetscCall(FormInitialGuess(snes, x));
  PetscCall(SNESSolve(snes, NULL, x));
  PetscCall(SNESGetIterationNumber(snes, &its));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "number of SNES iterations = %" PetscInt_FMT "\n\n", its));

  /* Free data structures */
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&r));
  PetscCall(VecDestroy(&U));
  PetscCall(VecDestroy(&F));
  PetscCall(MatDestroy(&J));
  PetscCall(MatDestroy(&B));
  PetscCall(SNESDestroy(&snes));
  PetscCall(PetscFinalize());
  return 0;
}
/* --------------------  Evaluate Function F(x) --------------------- */

PetscErrorCode FormFunction(SNES snes, Vec x, Vec f, void *dummy)
{
  const PetscScalar *xx, *FF;
  PetscScalar       *ff, d;
  PetscInt           i, n;

  PetscCall(VecGetArrayRead(x, &xx));
  PetscCall(VecGetArray(f, &ff));
  PetscCall(VecGetArrayRead((Vec)dummy, &FF));
  PetscCall(VecGetSize(x, &n));
  d     = (PetscReal)(n - 1);
  d     = d * d;
  ff[0] = xx[0];
  for (i = 1; i < n - 1; i++) ff[i] = d * (xx[i - 1] - 2.0 * xx[i] + xx[i + 1]) + xx[i] * xx[i] - FF[i];
  ff[n - 1] = xx[n - 1] - 1.0;
  PetscCall(VecRestoreArrayRead(x, &xx));
  PetscCall(VecRestoreArray(f, &ff));
  PetscCall(VecRestoreArrayRead((Vec)dummy, &FF));
  return 0;
}

PetscErrorCode FormFunctioni(void *dummy, PetscInt i, Vec x, PetscScalar *s)
{
  const PetscScalar *xx, *FF;
  PetscScalar        d;
  PetscInt           n;
  SNES               snes = (SNES)dummy;
  Vec                F;

  PetscCall(SNESGetFunction(snes, NULL, NULL, (void **)&F));
  PetscCall(VecGetArrayRead(x, &xx));
  PetscCall(VecGetArrayRead(F, &FF));
  PetscCall(VecGetSize(x, &n));
  d = (PetscReal)(n - 1);
  d = d * d;
  if (i == 0) {
    *s = xx[0];
  } else if (i == n - 1) {
    *s = xx[n - 1] - 1.0;
  } else {
    *s = d * (xx[i - 1] - 2.0 * xx[i] + xx[i + 1]) + xx[i] * xx[i] - FF[i];
  }
  PetscCall(VecRestoreArrayRead(x, &xx));
  PetscCall(VecRestoreArrayRead(F, &FF));
  return 0;
}

/*

   Example function that when differenced produces the same matrix free Jacobian as FormFunction()
   this is provided to show how a user can provide a different function
*/
PetscErrorCode OtherFunctionForDifferencing(void *dummy, Vec x, Vec f)
{
  PetscCall(FormFunction(NULL, x, f, dummy));
  PetscCall(VecShift(f, 1.0));
  return 0;
}

/* --------------------  Form initial approximation ----------------- */

PetscErrorCode FormInitialGuess(SNES snes, Vec x)
{
  PetscScalar pfive = .50;
  PetscCall(VecSet(x, pfive));
  return 0;
}
/* --------------------  Evaluate Jacobian F'(x) -------------------- */
/*  Evaluates a matrix that is used to precondition the matrix-free
    jacobian. In this case, the explicit preconditioner matrix is
    also EXACTLY the Jacobian. In general, it would be some lower
    order, simplified apprioximation */

PetscErrorCode FormJacobian(SNES snes, Vec x, Mat jac, Mat B, void *dummy)
{
  const PetscScalar *xx;
  PetscScalar        A[3], d;
  PetscInt           i, n, j[3];
  AppCtx            *user = (AppCtx *)dummy;

  PetscCall(VecGetArrayRead(x, &xx));
  PetscCall(VecGetSize(x, &n));
  d = (PetscReal)(n - 1);
  d = d * d;

  i    = 0;
  A[0] = 1.0;
  PetscCall(MatSetValues(B, 1, &i, 1, &i, &A[0], INSERT_VALUES));
  for (i = 1; i < n - 1; i++) {
    j[0] = i - 1;
    j[1] = i;
    j[2] = i + 1;
    A[0] = d;
    A[1] = -2.0 * d + 2.0 * xx[i];
    A[2] = d;
    PetscCall(MatSetValues(B, 1, &i, 3, j, A, INSERT_VALUES));
  }
  i    = n - 1;
  A[0] = 1.0;
  PetscCall(MatSetValues(B, 1, &i, 1, &i, &A[0], INSERT_VALUES));
  PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
  PetscCall(VecRestoreArrayRead(x, &xx));

  if (user->variant) PetscCall(MatMFFDSetBase(jac, x, NULL));
  PetscCall(MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY));
  return 0;
}

PetscErrorCode FormJacobianNoMatrix(SNES snes, Vec x, Mat jac, Mat B, void *dummy)
{
  AppCtx *user = (AppCtx *)dummy;

  if (user->variant) PetscCall(MatMFFDSetBase(jac, x, NULL));
  PetscCall(MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY));
  return 0;
}

/* --------------------  User-defined monitor ----------------------- */

PetscErrorCode Monitor(SNES snes, PetscInt its, PetscReal fnorm, void *dummy)
{
  MonitorCtx *monP = (MonitorCtx *)dummy;
  Vec         x;
  MPI_Comm    comm;

  PetscCall(PetscObjectGetComm((PetscObject)snes, &comm));
  PetscCall(PetscFPrintf(comm, stdout, "iter = %" PetscInt_FMT ", SNES Function norm %g \n", its, (double)fnorm));
  PetscCall(SNESGetSolution(snes, &x));
  PetscCall(VecView(x, monP->viewer));
  return 0;
}

/*TEST

   test:
      args: -ksp_gmres_cgs_refinement_type refine_always -snes_monitor_short

   test:
      suffix: 2
      args: -variant -ksp_gmres_cgs_refinement_type refine_always  -snes_monitor_short
      output_file: output/ex7_1.out

   # uses AIJ matrix to define diagonal matrix for Jacobian preconditioning
   test:
      suffix: 3
      args: -variant -pc_type jacobi -snes_view -ksp_monitor

   # uses MATMFFD matrix to define diagonal matrix for Jacobian preconditioning
   test:
      suffix: 4
      args: -variant -pc_type jacobi -puremf  -snes_view -ksp_monitor

TEST*/
