/*
       Formatted test for TS routines.

          Solves U_t=F(t,u)
          Where:

                  [2*u1+u2   ]
          F(t,u)= [u1+2*u2+u3]
                  [   u2+2*u3]

       When run in parallel, each process solves the same set of equations separately.
*/

static char help[] = "Solves a linear ODE. \n\n";

#include <petscts.h>
#include <petscpc.h>

extern PetscErrorCode RHSFunction(TS, PetscReal, Vec, Vec, void *);
extern PetscErrorCode RHSJacobian(TS, PetscReal, Vec, Mat, Mat, void *);
extern PetscErrorCode Monitor(TS, PetscInt, PetscReal, Vec, void *);
extern PetscErrorCode Initial(Vec, void *);
extern PetscErrorCode MyMatMult(Mat, Vec, Vec);

extern PetscReal solx(PetscReal);
extern PetscReal soly(PetscReal);
extern PetscReal solz(PetscReal);

int main(int argc, char **argv)
{
  PetscInt  time_steps = 100, steps;
  Vec       global;
  PetscReal dt, ftime;
  TS        ts;
  Mat       A, S;
  PetscBool nest = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-time", &time_steps, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-nest", &nest, NULL));

  /* create vector to hold state */
  if (nest) {
    Vec g[3];

    PetscCall(VecCreate(PETSC_COMM_WORLD, &g[0]));
    PetscCall(VecSetSizes(g[0], 1, PETSC_DECIDE));
    PetscCall(VecSetFromOptions(g[0]));
    PetscCall(VecDuplicate(g[0], &g[1]));
    PetscCall(VecDuplicate(g[0], &g[2]));
    PetscCall(VecCreateNest(PETSC_COMM_WORLD, 3, NULL, g, &global));
    PetscCall(VecDestroy(&g[0]));
    PetscCall(VecDestroy(&g[1]));
    PetscCall(VecDestroy(&g[2]));
  } else {
    PetscCall(VecCreate(PETSC_COMM_WORLD, &global));
    PetscCall(VecSetSizes(global, 3, PETSC_DECIDE));
    PetscCall(VecSetFromOptions(global));
  }

  /* set initial conditions */
  PetscCall(Initial(global, NULL));

  /* make timestep context */
  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(TSSetProblemType(ts, TS_NONLINEAR));
  PetscCall(TSMonitorSet(ts, Monitor, NULL, NULL));
  dt = 0.001;

  /*
    The user provides the RHS and Jacobian
  */
  PetscCall(TSSetRHSFunction(ts, NULL, RHSFunction, NULL));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, 3, 3, PETSC_DECIDE, PETSC_DECIDE));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));
  PetscCall(RHSJacobian(ts, 0.0, global, A, A, NULL));
  PetscCall(TSSetRHSJacobian(ts, A, A, RHSJacobian, NULL));

  PetscCall(MatCreateShell(PETSC_COMM_WORLD, 3, 3, PETSC_DECIDE, PETSC_DECIDE, NULL, &S));
  PetscCall(MatShellSetOperation(S, MATOP_MULT, (PetscErrorCodeFn *)MyMatMult));
  PetscCall(TSSetRHSJacobian(ts, S, A, RHSJacobian, NULL));

  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetFromOptions(ts));

  PetscCall(TSSetTimeStep(ts, dt));
  PetscCall(TSSetMaxSteps(ts, time_steps));
  PetscCall(TSSetMaxTime(ts, 1));
  PetscCall(TSSetSolution(ts, global));

  PetscCall(TSSolve(ts, global));
  PetscCall(TSGetSolveTime(ts, &ftime));
  PetscCall(TSGetStepNumber(ts, &steps));

  /* free the memory */
  PetscCall(TSDestroy(&ts));
  PetscCall(VecDestroy(&global));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&S));

  PetscCall(PetscFinalize());
  return 0;
}

PetscErrorCode MyMatMult(Mat S, Vec x, Vec y)
{
  const PetscScalar *inptr;
  PetscScalar       *outptr;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(x, &inptr));
  PetscCall(VecGetArrayWrite(y, &outptr));

  outptr[0] = 2.0 * inptr[0] + inptr[1];
  outptr[1] = inptr[0] + 2.0 * inptr[1] + inptr[2];
  outptr[2] = inptr[1] + 2.0 * inptr[2];

  PetscCall(VecRestoreArrayRead(x, &inptr));
  PetscCall(VecRestoreArrayWrite(y, &outptr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Initial(Vec global, void *ctx)
{
  PetscScalar *localptr;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayWrite(global, &localptr));
  localptr[0] = solx(0.0);
  localptr[1] = soly(0.0);
  localptr[2] = solz(0.0);
  PetscCall(VecRestoreArrayWrite(global, &localptr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Monitor(TS ts, PetscInt step, PetscReal time, Vec global, void *ctx)
{
  const PetscScalar *tmp;
  PetscScalar        exact[] = {solx(time), soly(time), solz(time)};

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(global, &tmp));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "At t =%14.6e u = %14.6e  %14.6e  %14.6e \n", (double)time, (double)PetscRealPart(tmp[0]), (double)PetscRealPart(tmp[1]), (double)PetscRealPart(tmp[2])));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "At t =%14.6e errors = %14.6e  %14.6e  %14.6e \n", (double)time, (double)PetscRealPart(tmp[0] - exact[0]), (double)PetscRealPart(tmp[1] - exact[1]), (double)PetscRealPart(tmp[2] - exact[2])));
  PetscCall(VecRestoreArrayRead(global, &tmp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec globalin, Vec globalout, void *ctx)
{
  PetscScalar       *outptr;
  const PetscScalar *inptr;

  PetscFunctionBeginUser;
  /*Extract income array */
  PetscCall(VecGetArrayRead(globalin, &inptr));

  /* Extract outcome array*/
  PetscCall(VecGetArrayWrite(globalout, &outptr));

  outptr[0] = 2.0 * inptr[0] + inptr[1];
  outptr[1] = inptr[0] + 2.0 * inptr[1] + inptr[2];
  outptr[2] = inptr[1] + 2.0 * inptr[2];

  PetscCall(VecRestoreArrayRead(globalin, &inptr));
  PetscCall(VecRestoreArrayWrite(globalout, &outptr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RHSJacobian(TS ts, PetscReal t, Vec x, Mat A, Mat BB, void *ctx)
{
  PetscScalar v[3];
  PetscInt    idx[3], rst;

  PetscFunctionBeginUser;
  PetscCall(VecGetOwnershipRange(x, &rst, NULL));
  idx[0] = 0 + rst;
  idx[1] = 1 + rst;
  idx[2] = 2 + rst;

  v[0] = 2.0;
  v[1] = 1.0;
  v[2] = 0.0;
  PetscCall(MatSetValues(BB, 1, idx, 3, idx, v, INSERT_VALUES));

  v[0] = 1.0;
  v[1] = 2.0;
  v[2] = 1.0;
  PetscCall(MatSetValues(BB, 1, idx + 1, 3, idx, v, INSERT_VALUES));

  v[0] = 0.0;
  v[1] = 1.0;
  v[2] = 2.0;
  PetscCall(MatSetValues(BB, 1, idx + 2, 3, idx, v, INSERT_VALUES));

  PetscCall(MatAssemblyBegin(BB, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(BB, MAT_FINAL_ASSEMBLY));

  if (A != BB) {
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
      The exact solutions
*/
PetscReal solx(PetscReal t)
{
  return PetscExpReal((2.0 - PetscSqrtReal(2.0)) * t) / 2.0 - PetscExpReal((2.0 - PetscSqrtReal(2.0)) * t) / (2.0 * PetscSqrtReal(2.0)) + PetscExpReal((2.0 + PetscSqrtReal(2.0)) * t) / 2.0 + PetscExpReal((2.0 + PetscSqrtReal(2.0)) * t) / (2.0 * PetscSqrtReal(2.0));
}

PetscReal soly(PetscReal t)
{
  return PetscExpReal((2.0 - PetscSqrtReal(2.0)) * t) / 2.0 - PetscExpReal((2.0 - PetscSqrtReal(2.0)) * t) / PetscSqrtReal(2.0) + PetscExpReal((2.0 + PetscSqrtReal(2.0)) * t) / 2.0 + PetscExpReal((2.0 + PetscSqrtReal(2.0)) * t) / PetscSqrtReal(2.0);
}

PetscReal solz(PetscReal t)
{
  return PetscExpReal((2.0 - PetscSqrtReal(2.0)) * t) / 2.0 - PetscExpReal((2.0 - PetscSqrtReal(2.0)) * t) / (2.0 * PetscSqrtReal(2.0)) + PetscExpReal((2.0 + PetscSqrtReal(2.0)) * t) / 2.0 + PetscExpReal((2.0 + PetscSqrtReal(2.0)) * t) / (2.0 * PetscSqrtReal(2.0));
}

/*TEST

    test:
      suffix: euler
      args: -ts_type euler -nest {{0 1}}
      requires: !single

    test:
      suffix: beuler
      args: -ts_type beuler -nest {{0 1}}
      requires: !single

    test:
      suffix: rk
      args: -ts_type rk -nest {{0 1}} -ts_adapt_monitor
      requires: !single

    test:
      diff_args: -j
      requires: double !complex
      output_file: output/ex2_be_adapt.out
      suffix: bdf_1_adapt
      args: -ts_type bdf -ts_bdf_order 1 -ts_adapt_type basic -ts_adapt_clip 0,2

    test:
      diff_args: -j
      requires: double !complex
      suffix: be_adapt
      args: -ts_type beuler -ts_adapt_type basic -ts_adapt_clip 0,2

TEST*/
