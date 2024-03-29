/*
 * ex_vdp.c
 *
 *  Created on: Jun 1, 2012
 *      Author: Hong Zhang
 */
static char help[] = "Solves the van der Pol equation. \n Input parameters include:\n";

/*
 * This program solves the van der Pol equation
 * y' = z                               (1)
 * z' = (((1-y^2)*z-y)/eps              (2)
 * on the domain 0<=x<=0.5, with the initial conditions
 * y(0) = 2,
 * z(0) = -2/3 + 10/81*eps - 292/2187*eps^2-1814/19683*eps^3
 * IMEX schemes are applied to the split equation
 * [y'] = [z]  + [0                 ]
 * [z']   [0]    [(((1-y^2)*z-y)/eps]
 *
 * F(x)= [z]
 *       [0]
 *
 * G(x)= [y'] -   [0                 ]
 *       [z']     [(((1-y^2)*z-y)/eps]
 *
 * JG(x) =  G_x + a G_xdot
 */

#include <petscdmda.h>
#include <petscts.h>

typedef struct _User *User;
struct _User {
  PetscReal mu; /*stiffness control coefficient: epsilon*/
};

static PetscErrorCode RHSFunction(TS, PetscReal, Vec, Vec, void *);
static PetscErrorCode IFunction(TS, PetscReal, Vec, Vec, Vec, void *);
static PetscErrorCode IJacobian(TS, PetscReal, Vec, Vec, PetscReal, Mat, Mat, void *);

int main(int argc, char **argv)
{
  TS           ts;
  Vec          x; /* solution vector */
  Mat          A; /* Jacobian */
  PetscInt     steps, mx, eimex_rowcol[2], two;
  PetscScalar *x_ptr;
  PetscReal    ftime, dt, norm;
  Vec          ref;
  struct _User user; /* user-defined work context */
  PetscViewer  viewer;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  /* Initialize user application context */
  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "van der Pol options", "");
  user.mu = 1e0;
  PetscCall(PetscOptionsReal("-eps", "Stiffness controller", "", user.mu, &user.mu, NULL));
  PetscOptionsEnd();

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
   PetscCall(PetscOptionsGetBool(NULL,NULL,"-monitor",&monitor,NULL));
   */

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Create necessary matrix and vectors, solve same ODE on every process
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, 2, 2));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));
  PetscCall(MatCreateVecs(A, &x, NULL));

  PetscCall(MatCreateVecs(A, &ref, NULL));
  PetscCall(VecGetArray(ref, &x_ptr));
  /*
   * [0,1], mu=10^-3
   */
  /*
   x_ptr[0] = -1.8881254106283;
   x_ptr[1] =  0.7359074233370;*/

  /*
   * [0,0.5],mu=10^-3
   */
  /*
   x_ptr[0] = 1.596980778659137;
   x_ptr[1] = -1.029103015879544;
   */
  /*
   * [0,0.5],mu=1
   */
  x_ptr[0] = 1.619084329683235;
  x_ptr[1] = -0.803530465176385;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Create timestepping solver context
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(TSSetType(ts, TSEIMEX));
  PetscCall(TSSetRHSFunction(ts, NULL, RHSFunction, &user));
  PetscCall(TSSetIFunction(ts, NULL, IFunction, &user));
  PetscCall(TSSetIJacobian(ts, A, A, IJacobian, &user));

  dt    = 0.00001;
  ftime = 1.1;
  PetscCall(TSSetTimeStep(ts, dt));
  PetscCall(TSSetMaxTime(ts, ftime));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER));
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(VecGetArray(x, &x_ptr));
  x_ptr[0] = 2.;
  x_ptr[1] = -2. / 3. + 10. / 81. * (user.mu) - 292. / 2187. * (user.mu) * (user.mu) - 1814. / 19683. * (user.mu) * (user.mu) * (user.mu);
  PetscCall(TSSetSolution(ts, x));
  PetscCall(VecGetSize(x, &mx));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSetFromOptions(ts));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Solve nonlinear system
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSolve(ts, x));
  PetscCall(TSGetTime(ts, &ftime));
  PetscCall(TSGetStepNumber(ts, &steps));

  PetscCall(VecAXPY(x, -1.0, ref));
  PetscCall(VecNorm(x, NORM_2, &norm));
  PetscCall(TSGetTimeStep(ts, &dt));

  eimex_rowcol[0] = 0;
  eimex_rowcol[1] = 0;
  two             = 2;
  PetscCall(PetscOptionsGetIntArray(NULL, NULL, "-ts_eimex_row_col", eimex_rowcol, &two, NULL));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "order %11s %18s %37s\n", "dt", "norm", "final solution components 0 and 1"));
  PetscCall(VecGetArray(x, &x_ptr));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "(%" PetscInt_FMT ",%" PetscInt_FMT ") %10.8f %18.15f %18.15f %18.15f\n", eimex_rowcol[0], eimex_rowcol[1], (double)dt, (double)norm, (double)PetscRealPart(x_ptr[0]), (double)PetscRealPart(x_ptr[1])));
  PetscCall(VecRestoreArray(x, &x_ptr));

  /* Write line in convergence log */
  PetscCall(PetscViewerCreate(PETSC_COMM_WORLD, &viewer));
  PetscCall(PetscViewerSetType(viewer, PETSCVIEWERASCII));
  PetscCall(PetscViewerFileSetMode(viewer, FILE_MODE_APPEND));
  PetscCall(PetscViewerFileSetName(viewer, "eimex_nonstiff_vdp.txt"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "%" PetscInt_FMT " %" PetscInt_FMT " %10.8f %18.15f\n", eimex_rowcol[0], eimex_rowcol[1], (double)dt, (double)norm));
  PetscCall(PetscViewerDestroy(&viewer));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Free work space.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&ref));
  PetscCall(TSDestroy(&ts));
  PetscCall(PetscFinalize());
  return 0;
}

static PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec X, Vec F, void *ptr)
{
  PetscScalar       *f;
  const PetscScalar *x;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(X, &x));
  PetscCall(VecGetArray(F, &f));
  f[0] = x[1];
  f[1] = 0.0;
  PetscCall(VecRestoreArrayRead(X, &x));
  PetscCall(VecRestoreArray(F, &f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode IFunction(TS ts, PetscReal t, Vec X, Vec Xdot, Vec F, void *ptr)
{
  User               user = (User)ptr;
  PetscScalar       *f;
  const PetscScalar *x, *xdot;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(X, &x));
  PetscCall(VecGetArrayRead(Xdot, &xdot));
  PetscCall(VecGetArray(F, &f));
  f[0] = xdot[0];
  f[1] = xdot[1] - ((1. - x[0] * x[0]) * x[1] - x[0]) / user->mu;
  PetscCall(VecRestoreArrayRead(X, &x));
  PetscCall(VecRestoreArrayRead(Xdot, &xdot));
  PetscCall(VecRestoreArray(F, &f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode IJacobian(TS ts, PetscReal t, Vec X, Vec Xdot, PetscReal a, Mat A, Mat B, void *ptr)
{
  User               user     = (User)ptr;
  PetscReal          mu       = user->mu;
  PetscInt           rowcol[] = {0, 1};
  PetscScalar        J[2][2];
  const PetscScalar *x;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(X, &x));
  J[0][0] = a;
  J[0][1] = 0;
  J[1][0] = (2. * x[0] * x[1] + 1.) / mu;
  J[1][1] = a - (1. - x[0] * x[0]) / mu;
  PetscCall(MatSetValues(B, 2, rowcol, 2, rowcol, &J[0][0], INSERT_VALUES));
  PetscCall(VecRestoreArrayRead(X, &x));

  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  if (A != B) {
    PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*TEST

   test:
     args: -ts_type eimex -ts_adapt_type none -pc_type lu -ts_dt 0.01 -ts_max_time 10 -ts_eimex_row_col 3,3 -ts_monitor_lg_solution
     requires: x

   test:
     suffix: adapt
     args: -ts_type eimex -ts_adapt_type none -pc_type lu -ts_dt 0.01 -ts_max_time 10 -ts_eimex_order_adapt -ts_eimex_max_rows 7 -ts_monitor_lg_solution
     requires: x

   test:
     suffix: loop
     args: -ts_type eimex -ts_adapt_type none -pc_type lu -ts_dt {{0.005 0.001 0.0005}separate output} -ts_max_steps {{100 500 1000}separate output} -ts_eimex_row_col {{1,1 2,1 3,1 2,2 3,2 3,3}separate output}
     requires: x

 TEST*/
