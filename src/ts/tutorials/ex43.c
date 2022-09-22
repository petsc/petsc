static char help[] = "Single-DOF oscillator formulated as a second-order system.\n";

#include <petscts.h>

typedef struct {
  PetscReal Omega;  /* natural frequency */
  PetscReal Xi;     /* damping coefficient  */
  PetscReal u0, v0; /* initial conditions */
} UserParams;

static void Exact(PetscReal t, PetscReal omega, PetscReal xi, PetscReal u0, PetscReal v0, PetscReal *ut, PetscReal *vt)
{
  PetscReal u, v;
  if (xi < 1) {
    PetscReal a  = xi * omega;
    PetscReal w  = PetscSqrtReal(1 - xi * xi) * omega;
    PetscReal C1 = (v0 + a * u0) / w;
    PetscReal C2 = u0;
    u            = PetscExpReal(-a * t) * (C1 * PetscSinReal(w * t) + C2 * PetscCosReal(w * t));
    v            = (-a * PetscExpReal(-a * t) * (C1 * PetscSinReal(w * t) + C2 * PetscCosReal(w * t)) + w * PetscExpReal(-a * t) * (C1 * PetscCosReal(w * t) - C2 * PetscSinReal(w * t)));
  } else if (xi > 1) {
    PetscReal w  = PetscSqrtReal(xi * xi - 1) * omega;
    PetscReal C1 = (w * u0 + xi * u0 + v0) / (2 * w);
    PetscReal C2 = (w * u0 - xi * u0 - v0) / (2 * w);
    u            = C1 * PetscExpReal((-xi + w) * t) + C2 * PetscExpReal((-xi - w) * t);
    v            = C1 * (-xi + w) * PetscExpReal((-xi + w) * t) + C2 * (-xi - w) * PetscExpReal((-xi - w) * t);
  } else {
    PetscReal a  = xi * omega;
    PetscReal C1 = v0 + a * u0;
    PetscReal C2 = u0;
    u            = (C1 * t + C2) * PetscExpReal(-a * t);
    v            = (C1 - a * (C1 * t + C2)) * PetscExpReal(-a * t);
  }
  if (ut) *ut = u;
  if (vt) *vt = v;
}

PetscErrorCode Solution(TS ts, PetscReal t, Vec X, void *ctx)
{
  UserParams  *user = (UserParams *)ctx;
  PetscReal    u, v;
  PetscScalar *x;

  PetscFunctionBeginUser;
  Exact(t, user->Omega, user->Xi, user->u0, user->v0, &u, &v);
  PetscCall(VecGetArray(X, &x));
  x[0] = (PetscScalar)u;
  PetscCall(VecRestoreArray(X, &x));
  PetscFunctionReturn(0);
}

PetscErrorCode Residual1(TS ts, PetscReal t, Vec U, Vec A, Vec R, void *ctx)
{
  UserParams        *user  = (UserParams *)ctx;
  PetscReal          Omega = user->Omega;
  const PetscScalar *u, *a;
  PetscScalar       *r;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(U, &u));
  PetscCall(VecGetArrayRead(A, &a));
  PetscCall(VecGetArrayWrite(R, &r));

  r[0] = a[0] + (Omega * Omega) * u[0];

  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(VecRestoreArrayRead(A, &a));
  PetscCall(VecRestoreArrayWrite(R, &r));
  PetscCall(VecAssemblyBegin(R));
  PetscCall(VecAssemblyEnd(R));
  PetscFunctionReturn(0);
}

PetscErrorCode Tangent1(TS ts, PetscReal t, Vec U, Vec A, PetscReal shiftA, Mat J, Mat P, void *ctx)
{
  UserParams *user  = (UserParams *)ctx;
  PetscReal   Omega = user->Omega;
  PetscReal   T     = 0;

  PetscFunctionBeginUser;

  T = shiftA + (Omega * Omega);

  PetscCall(MatSetValue(P, 0, 0, T, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY));
  if (J != P) {
    PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode Residual2(TS ts, PetscReal t, Vec U, Vec V, Vec A, Vec R, void *ctx)
{
  UserParams        *user  = (UserParams *)ctx;
  PetscReal          Omega = user->Omega, Xi = user->Xi;
  const PetscScalar *u, *v, *a;
  PetscScalar       *r;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(U, &u));
  PetscCall(VecGetArrayRead(V, &v));
  PetscCall(VecGetArrayRead(A, &a));
  PetscCall(VecGetArrayWrite(R, &r));

  r[0] = a[0] + (2 * Xi * Omega) * v[0] + (Omega * Omega) * u[0];

  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(VecRestoreArrayRead(V, &v));
  PetscCall(VecRestoreArrayRead(A, &a));
  PetscCall(VecRestoreArrayWrite(R, &r));
  PetscCall(VecAssemblyBegin(R));
  PetscCall(VecAssemblyEnd(R));
  PetscFunctionReturn(0);
}

PetscErrorCode Tangent2(TS ts, PetscReal t, Vec U, Vec V, Vec A, PetscReal shiftV, PetscReal shiftA, Mat J, Mat P, void *ctx)
{
  UserParams *user  = (UserParams *)ctx;
  PetscReal   Omega = user->Omega, Xi = user->Xi;
  PetscReal   T = 0;

  PetscFunctionBeginUser;

  T = shiftA + shiftV * (2 * Xi * Omega) + (Omega * Omega);

  PetscCall(MatSetValue(P, 0, 0, T, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY));
  if (J != P) {
    PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(0);
}

int main(int argc, char *argv[])
{
  PetscMPIInt  size;
  TS           ts;
  Vec          R;
  Mat          J;
  Vec          U, V;
  PetscScalar *u, *v;
  UserParams   user = {/*Omega=*/1, /*Xi=*/0, /*u0=*/1, /*,v0=*/0};

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "Only for sequential runs");

  PetscOptionsBegin(PETSC_COMM_SELF, "", "ex43 options", "");
  PetscCall(PetscOptionsReal("-frequency", "Natual frequency", __FILE__, user.Omega, &user.Omega, NULL));
  PetscCall(PetscOptionsReal("-damping", "Damping coefficient", __FILE__, user.Xi, &user.Xi, NULL));
  PetscCall(PetscOptionsReal("-initial_u", "Initial displacement", __FILE__, user.u0, &user.u0, NULL));
  PetscCall(PetscOptionsReal("-initial_v", "Initial velocity", __FILE__, user.v0, &user.v0, NULL));
  PetscOptionsEnd();

  PetscCall(TSCreate(PETSC_COMM_SELF, &ts));
  PetscCall(TSSetType(ts, TSALPHA2));
  PetscCall(TSSetMaxTime(ts, 5 * (2 * PETSC_PI)));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER));
  PetscCall(TSSetTimeStep(ts, 0.01));

  PetscCall(VecCreateSeq(PETSC_COMM_SELF, 1, &R));
  PetscCall(VecSetUp(R));
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, 1, 1, NULL, &J));
  PetscCall(MatSetUp(J));
  if (user.Xi) {
    PetscCall(TSSetI2Function(ts, R, Residual2, &user));
    PetscCall(TSSetI2Jacobian(ts, J, J, Tangent2, &user));
  } else {
    PetscCall(TSSetIFunction(ts, R, Residual1, &user));
    PetscCall(TSSetIJacobian(ts, J, J, Tangent1, &user));
  }
  PetscCall(VecDestroy(&R));
  PetscCall(MatDestroy(&J));
  PetscCall(TSSetSolutionFunction(ts, Solution, &user));

  PetscCall(VecCreateSeq(PETSC_COMM_SELF, 1, &U));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, 1, &V));
  PetscCall(VecGetArrayWrite(U, &u));
  PetscCall(VecGetArrayWrite(V, &v));
  u[0] = user.u0;
  v[0] = user.v0;
  PetscCall(VecRestoreArrayWrite(U, &u));
  PetscCall(VecRestoreArrayWrite(V, &v));

  PetscCall(TS2SetSolution(ts, U, V));
  PetscCall(TSSetFromOptions(ts));
  PetscCall(TSSolve(ts, NULL));

  PetscCall(VecDestroy(&U));
  PetscCall(VecDestroy(&V));
  PetscCall(TSDestroy(&ts));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

    test:
      suffix: a
      args: -ts_max_steps 10 -ts_view
      requires: !single

    test:
      suffix: b
      args: -ts_max_steps 10 -ts_rtol 0 -ts_atol 1e-5 -ts_adapt_type basic -ts_adapt_monitor
      requires: !single

TEST*/
