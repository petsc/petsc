
static const char help[] = "Tries to solve u`` + u^{2} = f for an easy case and an impossible case.\n\n";

/*
       This example was contributed by Peter Graf to show how SNES fails when given a nonlinear problem with no solution.

       Run with -n 14 to see fail to converge and -n 15 to see convergence

       The option -second_order uses a different discretization of the Neumann boundary condition and always converges

*/

#include <petscsnes.h>

PetscBool second_order = PETSC_FALSE;
#define X0DOT -2.0
#define X1    5.0
#define KPOW  2.0
const PetscScalar sperturb = 1.1;

/*
   User-defined routines
*/
PetscErrorCode FormJacobian(SNES, Vec, Mat, Mat, void *);
PetscErrorCode FormFunction(SNES, Vec, Vec, void *);

int main(int argc, char **argv)
{
  SNES              snes;    /* SNES context */
  Vec               x, r, F; /* vectors */
  Mat               J;       /* Jacobian */
  PetscInt          it, n = 11, i;
  PetscReal         h, xp = 0.0;
  PetscScalar       v;
  const PetscScalar a = X0DOT;
  const PetscScalar b = X1;
  const PetscScalar k = KPOW;
  PetscScalar       v2;
  PetscScalar      *xx;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-second_order", &second_order, NULL));
  h = 1.0 / (n - 1);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create nonlinear solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create vector data structures; set function evaluation routine
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(VecCreate(PETSC_COMM_SELF, &x));
  PetscCall(VecSetSizes(x, PETSC_DECIDE, n));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecDuplicate(x, &r));
  PetscCall(VecDuplicate(x, &F));

  PetscCall(SNESSetFunction(snes, r, FormFunction, (void *)F));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create matrix data structures; set Jacobian evaluation routine
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, n, n, 3, NULL, &J));

  /*
     Note that in this case we create separate matrices for the Jacobian
     and preconditioner matrix.  Both of these are computed in the
     routine FormJacobian()
  */
  /*  PetscCall(SNESSetJacobian(snes,NULL,JPrec,FormJacobian,0)); */
  PetscCall(SNESSetJacobian(snes, J, J, FormJacobian, 0));
  /*  PetscCall(SNESSetJacobian(snes,J,JPrec,FormJacobian,0)); */

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize nonlinear solver; set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(SNESSetFromOptions(snes));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize application:
     Store right-hand-side of PDE and exact solution
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* set right hand side and initial guess to be exact solution of continuim problem */
#define SQR(x) ((x) * (x))
  xp = 0.0;
  for (i = 0; i < n; i++) {
    v = k * (k - 1.) * (b - a) * PetscPowScalar(xp, k - 2.) + SQR(a * xp) + SQR(b - a) * PetscPowScalar(xp, 2. * k) + 2. * a * (b - a) * PetscPowScalar(xp, k + 1.);
    PetscCall(VecSetValues(F, 1, &i, &v, INSERT_VALUES));
    v2 = a * xp + (b - a) * PetscPowScalar(xp, k);
    PetscCall(VecSetValues(x, 1, &i, &v2, INSERT_VALUES));
    xp += h;
  }

  /* perturb initial guess */
  PetscCall(VecGetArray(x, &xx));
  for (i = 0; i < n; i++) {
    v2 = xx[i] * sperturb;
    PetscCall(VecSetValues(x, 1, &i, &v2, INSERT_VALUES));
  }
  PetscCall(VecRestoreArray(x, &xx));

  PetscCall(SNESSolve(snes, NULL, x));
  PetscCall(SNESGetIterationNumber(snes, &it));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "SNES iterations = %" PetscInt_FMT "\n\n", it));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&r));
  PetscCall(VecDestroy(&F));
  PetscCall(MatDestroy(&J));
  PetscCall(SNESDestroy(&snes));
  PetscCall(PetscFinalize());
  return 0;
}

PetscErrorCode FormFunction(SNES snes, Vec x, Vec f, void *dummy)
{
  const PetscScalar *xx;
  PetscScalar       *ff, *FF, d, d2;
  PetscInt           i, n;

  PetscCall(VecGetArrayRead(x, &xx));
  PetscCall(VecGetArray(f, &ff));
  PetscCall(VecGetArray((Vec)dummy, &FF));
  PetscCall(VecGetSize(x, &n));
  d  = (PetscReal)(n - 1);
  d2 = d * d;

  if (second_order) ff[0] = d * (0.5 * d * (-xx[2] + 4. * xx[1] - 3. * xx[0]) - X0DOT);
  else ff[0] = d * (d * (xx[1] - xx[0]) - X0DOT);

  for (i = 1; i < n - 1; i++) ff[i] = d2 * (xx[i - 1] - 2. * xx[i] + xx[i + 1]) + xx[i] * xx[i] - FF[i];

  ff[n - 1] = d * d * (xx[n - 1] - X1);
  PetscCall(VecRestoreArrayRead(x, &xx));
  PetscCall(VecRestoreArray(f, &ff));
  PetscCall(VecRestoreArray((Vec)dummy, &FF));
  return 0;
}

PetscErrorCode FormJacobian(SNES snes, Vec x, Mat jac, Mat prejac, void *dummy)
{
  const PetscScalar *xx;
  PetscScalar        A[3], d, d2;
  PetscInt           i, n, j[3];

  PetscCall(VecGetSize(x, &n));
  PetscCall(VecGetArrayRead(x, &xx));
  d  = (PetscReal)(n - 1);
  d2 = d * d;

  i = 0;
  if (second_order) {
    j[0] = 0;
    j[1] = 1;
    j[2] = 2;
    A[0] = -3. * d * d * 0.5;
    A[1] = 4. * d * d * 0.5;
    A[2] = -1. * d * d * 0.5;
    PetscCall(MatSetValues(prejac, 1, &i, 3, j, A, INSERT_VALUES));
  } else {
    j[0] = 0;
    j[1] = 1;
    A[0] = -d * d;
    A[1] = d * d;
    PetscCall(MatSetValues(prejac, 1, &i, 2, j, A, INSERT_VALUES));
  }
  for (i = 1; i < n - 1; i++) {
    j[0] = i - 1;
    j[1] = i;
    j[2] = i + 1;
    A[0] = d2;
    A[1] = -2. * d2 + 2. * xx[i];
    A[2] = d2;
    PetscCall(MatSetValues(prejac, 1, &i, 3, j, A, INSERT_VALUES));
  }

  i    = n - 1;
  A[0] = d * d;
  PetscCall(MatSetValues(prejac, 1, &i, 1, &i, &A[0], INSERT_VALUES));

  PetscCall(MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(prejac, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(prejac, MAT_FINAL_ASSEMBLY));

  PetscCall(VecRestoreArrayRead(x, &xx));
  return 0;
}

/*TEST

   test:
      args: -n 14 -snes_monitor_short -snes_converged_reason
      requires: !single

   test:
      suffix: 2
      args: -n 15 -snes_monitor_short -snes_converged_reason
      requires: !single

   test:
      suffix: 3
      args: -n 14 -second_order -snes_monitor_short -snes_converged_reason
      requires: !single

TEST*/
