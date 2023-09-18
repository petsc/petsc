static char help[] = "Solves a DAE with a non-trivial mass matrix. \n\n";
/*
   Solves:
        U * dU/dt = U*V
        U - V = 0

   that can be rewritten in implicit form F = 0, with F as
                 x[0] * xdot[0] - x[0] * x[1]
   F(t,x,xdot) =
                 x[0] - x[1]
   It is equivalent to solve dU/dt = U, U = U0 with solution U = U0 * exp(tfinal)
*/

#include <petscts.h>

PetscErrorCode IFunction(TS, PetscReal, Vec, Vec, Vec, void *);
PetscErrorCode IJacobian(TS, PetscReal, Vec, Vec, PetscReal, Mat, Mat, void *);

int main(int argc, char **argv)
{
  TS        ts;
  Vec       x;
  PetscBool dae = PETSC_TRUE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-dae", &dae, NULL));

  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(TSSetIFunction(ts, NULL, IFunction, &dae));
  PetscCall(TSSetIJacobian(ts, NULL, NULL, IJacobian, &dae));

  PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
  PetscCall(VecSetSizes(x, 2, PETSC_DECIDE));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecSetUp(x));
  PetscCall(VecSet(x, 0.5));
  PetscCall(TSSetSolution(ts, x));
  PetscCall(VecDestroy(&x));

  PetscCall(TSSetTimeStep(ts, 1.0 / 16.0));
  PetscCall(TSSetMaxTime(ts, 1.0));
  PetscCall(TSSetFromOptions(ts));
  PetscCall(TSSolve(ts, NULL));

  PetscCall(TSDestroy(&ts));
  PetscCall(PetscFinalize());
  return 0;
}

PetscErrorCode IFunction(TS ts, PetscReal t, Vec X, Vec Xdot, Vec F, void *ctx)
{
  const PetscScalar *xdot, *x;
  PetscScalar       *f;
  PetscBool          dae = *(PetscBool *)(ctx);

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(Xdot, &xdot));
  PetscCall(VecGetArrayRead(X, &x));
  PetscCall(VecGetArrayWrite(F, &f));
  if (dae) {
    f[0] = x[0] * xdot[0] - x[0] * x[1];
    f[1] = x[0] - x[1];
  } else {
    f[0] = xdot[0] - x[0];
    f[1] = xdot[1] - x[1];
  }
  PetscCall(VecRestoreArrayRead(Xdot, &xdot));
  PetscCall(VecRestoreArrayRead(X, &x));
  PetscCall(VecRestoreArrayWrite(F, &f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode IJacobian(TS ts, PetscReal t, Vec X, Vec Xdot, PetscReal shift, Mat A, Mat B, void *ctx)
{
  const PetscScalar *xdot, *x;
  PetscBool          dae = *(PetscBool *)(ctx);

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(Xdot, &xdot));
  PetscCall(VecGetArrayRead(X, &x));
  if (dae) {
    PetscCall(MatSetValue(B, 0, 0, shift * x[0] + xdot[0] - x[1], INSERT_VALUES));
    PetscCall(MatSetValue(B, 0, 1, -x[0], INSERT_VALUES));
    PetscCall(MatSetValue(B, 1, 0, 1.0, INSERT_VALUES));
    PetscCall(MatSetValue(B, 1, 1, -1.0, INSERT_VALUES));
  } else {
    PetscCall(MatZeroEntries(B));
    PetscCall(MatShift(B, shift));
  }
  PetscCall(VecRestoreArrayRead(X, &x));
  PetscCall(VecRestoreArrayRead(Xdot, &xdot));
  PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
  if (A != B) {
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*TEST

  testset:
    args: -ts_view_solution -ts_max_steps 10 -ts_dt 0.1 -ts_view_solution -ts_adapt_type {{none basic}} -ts_exact_final_time matchstep

    test:
      output_file: output/ex18_1.out
      suffix: bdf
      args: -ts_type bdf

    test:
      output_file: output/ex18_1.out
      suffix: dirk
      args: -dae {{0 1}} -ts_type dirk -ts_dirk_type {{s212 es122sal es213sal es324sal es325sal 657a es648sa 658a s659a 7510sal es7510sa 759a s7511sal 8614a 8616sal es8516sal}}

TEST*/
