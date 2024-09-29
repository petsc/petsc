static char help[] = "Constant acceleration check with 2nd-order generalized-alpha.\n";

#include <petscts.h>

typedef struct {
  PetscReal a0;     /* constant acceleration  */
  PetscReal u0, v0; /* initial conditions */
  PetscReal radius; /* spectral radius of integrator */
} UserParams;

static void Exact(PetscReal t, PetscReal a0, PetscReal u0, PetscReal v0, PetscScalar *ut, PetscScalar *vt)
{
  if (ut) *ut = u0 + v0 * t + 0.5 * a0 * t * t;
  if (vt) *vt = v0 + a0 * t;
}

PetscErrorCode Residual(TS ts, PetscReal t, Vec U, Vec V, Vec A, Vec R, void *ctx)
{
  UserParams        *user = (UserParams *)ctx;
  const PetscScalar *a;
  PetscScalar       *r;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(A, &a));
  PetscCall(VecGetArrayWrite(R, &r));

  r[0] = a[0] - user->a0;

  PetscCall(VecRestoreArrayRead(A, &a));
  PetscCall(VecRestoreArrayWrite(R, &r));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Tangent(TS ts, PetscReal t, Vec U, Vec V, Vec A, PetscReal shiftV, PetscReal shiftA, Mat J, Mat P, void *ctx)
{
  PetscReal T = 0;

  PetscFunctionBeginUser;
  T = shiftA;

  PetscCall(MatSetValue(P, 0, 0, T, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY));
  if (J != P) {
    PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char *argv[])
{
  PetscMPIInt  size;
  TS           ts;
  Vec          R;
  Mat          J;
  Vec          U, V;
  PetscScalar *u, *v, u_exact, v_exact;
  PetscReal    u_err, v_err;
  PetscReal    atol    = 1e-15;
  PetscReal    t_final = 3.0;
  PetscInt     n_step  = 8;
  UserParams   user    = {/*a0=*/1, /*u0=*/1, /*v0=*/0, /*radius=*/0.0};

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "Only for sequential runs");

  PetscOptionsBegin(PETSC_COMM_SELF, "", "ex80 options", "");
  PetscCall(PetscOptionsReal("-acceleration", "Constant acceleration", __FILE__, user.a0, &user.a0, NULL));
  PetscCall(PetscOptionsReal("-initial_u", "Initial displacement", __FILE__, user.u0, &user.u0, NULL));
  PetscCall(PetscOptionsReal("-initial_v", "Initial velocity", __FILE__, user.v0, &user.v0, NULL));
  PetscCall(PetscOptionsReal("-radius", "Spectral radius", __FILE__, user.radius, &user.radius, NULL));
  PetscOptionsEnd();

  PetscCall(TSCreate(PETSC_COMM_SELF, &ts));
  PetscCall(TSSetType(ts, TSALPHA2));
  PetscCall(TSSetMaxTime(ts, t_final));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetTimeStep(ts, t_final / n_step));
  PetscCall(TSAlpha2SetRadius(ts, user.radius));

  PetscCall(VecCreateSeq(PETSC_COMM_SELF, 1, &R));
  PetscCall(VecSetUp(R));
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, 1, 1, NULL, &J));
  PetscCall(MatSetUp(J));
  PetscCall(TSSetI2Function(ts, R, Residual, &user));
  PetscCall(TSSetI2Jacobian(ts, J, J, Tangent, &user));
  PetscCall(VecDestroy(&R));
  PetscCall(MatDestroy(&J));

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

  PetscCall(VecGetArray(U, &u));
  PetscCall(VecGetArray(V, &v));
  Exact(t_final, user.a0, user.u0, user.v0, &u_exact, &v_exact);
  u_err = PetscAbsScalar(u[0] - u_exact);
  v_err = PetscAbsScalar(v[0] - v_exact);
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "u(t=%g) = %g (error = %g)\n", (double)t_final, (double)PetscRealPart(u[0]), (double)u_err));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "v(t=%g) = %g (error = %g)\n", (double)t_final, (double)PetscRealPart(v[0]), (double)v_err));
  PetscCheck(u_err < atol, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Inexact displacement.");
  PetscCheck(v_err < atol, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Inexact velocity.");
  PetscCall(VecRestoreArray(U, &u));
  PetscCall(VecRestoreArray(V, &v));

  PetscCall(VecDestroy(&U));
  PetscCall(VecDestroy(&V));
  PetscCall(TSDestroy(&ts));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

    test:
      suffix: a
      args: -radius 0.0
      requires: !single

    test:
      suffix: b
      args: -radius 0.5
      requires: !single

    test:
      suffix: c
      args: -radius 1.0
      requires: !single

TEST*/
