static char help[] = "Constant velocity check with 1st-order generalized-alpha.\n";

#include <petscts.h>

typedef struct {
  PetscReal v0;     /* constant velocity */
  PetscReal u0;     /* initial condition */
  PetscReal radius; /* spectral radius of integrator */
} UserParams;

static void Exact(PetscReal t, PetscReal v0, PetscReal u0, PetscScalar *ut)
{
  if (ut) *ut = u0 + v0 * t;
}

PetscErrorCode Residual(TS ts, PetscReal t, Vec U, Vec V, Vec R, void *ctx)
{
  UserParams        *user = (UserParams *)ctx;
  const PetscScalar *v;
  PetscScalar       *r;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(V, &v));
  PetscCall(VecGetArrayWrite(R, &r));

  r[0] = v[0] - user->v0;

  PetscCall(VecRestoreArrayRead(V, &v));
  PetscCall(VecRestoreArrayWrite(R, &r));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Tangent(TS ts, PetscReal t, Vec U, Vec V, PetscReal shiftV, Mat J, Mat P, void *ctx)
{
  PetscReal T = 0;

  PetscFunctionBeginUser;
  T = shiftV;

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
  Vec          U;
  PetscScalar *u, u_exact;
  PetscReal    u_err;
  PetscReal    atol    = 1e-15;
  PetscReal    t_final = 3.0;
  PetscInt     n_step  = 8;
  UserParams   user    = {/*v0=*/1, /*u0=*/1, /*radius=*/0.0};

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "Only for sequential runs");

  PetscOptionsBegin(PETSC_COMM_SELF, "", "ex81 options", "");
  PetscCall(PetscOptionsReal("-velocity", "Constant velocity", __FILE__, user.v0, &user.v0, NULL));
  PetscCall(PetscOptionsReal("-initial_u", "Initial displacement", __FILE__, user.u0, &user.u0, NULL));
  PetscCall(PetscOptionsReal("-radius", "Spectral radius", __FILE__, user.radius, &user.radius, NULL));
  PetscOptionsEnd();

  PetscCall(TSCreate(PETSC_COMM_SELF, &ts));
  PetscCall(TSSetType(ts, TSALPHA));
  PetscCall(TSSetMaxTime(ts, t_final));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetTimeStep(ts, t_final / n_step));
  PetscCall(TSAlphaSetRadius(ts, user.radius));

  PetscCall(VecCreateSeq(PETSC_COMM_SELF, 1, &R));
  PetscCall(VecSetUp(R));
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, 1, 1, NULL, &J));
  PetscCall(MatSetUp(J));
  PetscCall(TSSetIFunction(ts, R, Residual, &user));
  PetscCall(TSSetIJacobian(ts, J, J, Tangent, &user));
  PetscCall(VecDestroy(&R));
  PetscCall(MatDestroy(&J));

  PetscCall(VecCreateSeq(PETSC_COMM_SELF, 1, &U));
  PetscCall(VecGetArrayWrite(U, &u));
  u[0] = user.u0;
  PetscCall(VecRestoreArrayWrite(U, &u));

  PetscCall(TSSetSolution(ts, U));
  PetscCall(TSSetFromOptions(ts));
  PetscCall(TSSolve(ts, NULL));

  PetscCall(VecGetArray(U, &u));
  Exact(t_final, user.v0, user.u0, &u_exact);
  u_err = PetscAbsScalar(u[0] - u_exact);
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "u(t=%g) = %g (error = %g)\n", (double)t_final, (double)PetscRealPart(u[0]), (double)u_err));
  PetscCheck(u_err < atol, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Inexact displacement.");
  PetscCall(VecRestoreArray(U, &u));

  PetscCall(VecDestroy(&U));
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
