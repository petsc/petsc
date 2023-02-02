
static char help[] = "Time-dependent PDE in 2d. Simplified from ex7.c for illustrating how to use TS on a structured domain. \n";
/*
   u_t = uxx + uyy
   0 < x < 1, 0 < y < 1;
   At t=0: u(x,y) = exp(c*r*r*r), if r=PetscSqrtReal((x-.5)*(x-.5) + (y-.5)*(y-.5)) < .125
           u(x,y) = 0.0           if r >= .125

    mpiexec -n 2 ./ex13 -da_grid_x 40 -da_grid_y 40 -ts_max_steps 2 -snes_monitor -ksp_monitor
    mpiexec -n 1 ./ex13 -snes_fd_color -ts_monitor_draw_solution
    mpiexec -n 2 ./ex13 -ts_type sundials -ts_monitor
*/

#include <petscdm.h>
#include <petscdmda.h>
#include <petscts.h>

/*
   User-defined data structures and routines
*/
typedef struct {
  PetscReal c;
} AppCtx;

extern PetscErrorCode RHSFunction(TS, PetscReal, Vec, Vec, void *);
extern PetscErrorCode RHSJacobian(TS, PetscReal, Vec, Mat, Mat, void *);
extern PetscErrorCode FormInitialSolution(DM, Vec, void *);

int main(int argc, char **argv)
{
  TS        ts;    /* nonlinear solver */
  Vec       u, r;  /* solution, residual vector */
  Mat       J;     /* Jacobian matrix */
  PetscInt  steps; /* iterations for convergence */
  DM        da;
  PetscReal ftime, dt;
  AppCtx    user; /* user-defined work context */

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, 8, 8, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));

  /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Extract global vectors from DMDA;
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(DMCreateGlobalVector(da, &u));
  PetscCall(VecDuplicate(u, &r));

  /* Initialize user application context */
  user.c = -30.0;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(TSSetDM(ts, da));
  PetscCall(TSSetType(ts, TSBEULER));
  PetscCall(TSSetRHSFunction(ts, r, RHSFunction, &user));

  /* Set Jacobian */
  PetscCall(DMSetMatType(da, MATAIJ));
  PetscCall(DMCreateMatrix(da, &J));
  PetscCall(TSSetRHSJacobian(ts, J, J, RHSJacobian, NULL));

  ftime = 1.0;
  PetscCall(TSSetMaxTime(ts, ftime));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(FormInitialSolution(da, u, &user));
  dt = .01;
  PetscCall(TSSetTimeStep(ts, dt));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSetFromOptions(ts));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSolve(ts, u));
  PetscCall(TSGetSolveTime(ts, &ftime));
  PetscCall(TSGetStepNumber(ts, &steps));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(MatDestroy(&J));
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&r));
  PetscCall(TSDestroy(&ts));
  PetscCall(DMDestroy(&da));

  PetscCall(PetscFinalize());
  return 0;
}
/* ------------------------------------------------------------------- */
/*
   RHSFunction - Evaluates nonlinear function, F(u).

   Input Parameters:
.  ts - the TS context
.  U - input vector
.  ptr - optional user-defined context, as set by TSSetFunction()

   Output Parameter:
.  F - function vector
 */
PetscErrorCode RHSFunction(TS ts, PetscReal ftime, Vec U, Vec F, void *ptr)
{
  /* PETSC_UNUSED AppCtx *user=(AppCtx*)ptr; */
  DM          da;
  PetscInt    i, j, Mx, My, xs, ys, xm, ym;
  PetscReal   two = 2.0, hx, hy, sx, sy;
  PetscScalar u, uxx, uyy, **uarray, **f;
  Vec         localU;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &da));
  PetscCall(DMGetLocalVector(da, &localU));
  PetscCall(DMDAGetInfo(da, PETSC_IGNORE, &Mx, &My, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE));

  hx = 1.0 / (PetscReal)(Mx - 1);
  sx = 1.0 / (hx * hx);
  hy = 1.0 / (PetscReal)(My - 1);
  sy = 1.0 / (hy * hy);

  /*
     Scatter ghost points to local vector,using the 2-step process
        DMGlobalToLocalBegin(),DMGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
  */
  PetscCall(DMGlobalToLocalBegin(da, U, INSERT_VALUES, localU));
  PetscCall(DMGlobalToLocalEnd(da, U, INSERT_VALUES, localU));

  /* Get pointers to vector data */
  PetscCall(DMDAVecGetArrayRead(da, localU, &uarray));
  PetscCall(DMDAVecGetArray(da, F, &f));

  /* Get local grid boundaries */
  PetscCall(DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL));

  /* Compute function over the locally owned part of the grid */
  for (j = ys; j < ys + ym; j++) {
    for (i = xs; i < xs + xm; i++) {
      if (i == 0 || j == 0 || i == Mx - 1 || j == My - 1) {
        f[j][i] = uarray[j][i];
        continue;
      }
      u       = uarray[j][i];
      uxx     = (-two * u + uarray[j][i - 1] + uarray[j][i + 1]) * sx;
      uyy     = (-two * u + uarray[j - 1][i] + uarray[j + 1][i]) * sy;
      f[j][i] = uxx + uyy;
    }
  }

  /* Restore vectors */
  PetscCall(DMDAVecRestoreArrayRead(da, localU, &uarray));
  PetscCall(DMDAVecRestoreArray(da, F, &f));
  PetscCall(DMRestoreLocalVector(da, &localU));
  PetscCall(PetscLogFlops(11.0 * ym * xm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* --------------------------------------------------------------------- */
/*
   RHSJacobian - User-provided routine to compute the Jacobian of
   the nonlinear right-hand-side function of the ODE.

   Input Parameters:
   ts - the TS context
   t - current time
   U - global input vector
   dummy - optional user-defined context, as set by TSetRHSJacobian()

   Output Parameters:
   J - Jacobian matrix
   Jpre - optionally different preconditioning matrix
   str - flag indicating matrix structure
*/
PetscErrorCode RHSJacobian(TS ts, PetscReal t, Vec U, Mat J, Mat Jpre, void *ctx)
{
  DM            da;
  DMDALocalInfo info;
  PetscInt      i, j;
  PetscReal     hx, hy, sx, sy;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &da));
  PetscCall(DMDAGetLocalInfo(da, &info));
  hx = 1.0 / (PetscReal)(info.mx - 1);
  sx = 1.0 / (hx * hx);
  hy = 1.0 / (PetscReal)(info.my - 1);
  sy = 1.0 / (hy * hy);
  for (j = info.ys; j < info.ys + info.ym; j++) {
    for (i = info.xs; i < info.xs + info.xm; i++) {
      PetscInt    nc = 0;
      MatStencil  row, col[5];
      PetscScalar val[5];
      row.i = i;
      row.j = j;
      if (i == 0 || j == 0 || i == info.mx - 1 || j == info.my - 1) {
        col[nc].i = i;
        col[nc].j = j;
        val[nc++] = 1.0;
      } else {
        col[nc].i = i - 1;
        col[nc].j = j;
        val[nc++] = sx;
        col[nc].i = i + 1;
        col[nc].j = j;
        val[nc++] = sx;
        col[nc].i = i;
        col[nc].j = j - 1;
        val[nc++] = sy;
        col[nc].i = i;
        col[nc].j = j + 1;
        val[nc++] = sy;
        col[nc].i = i;
        col[nc].j = j;
        val[nc++] = -2 * sx - 2 * sy;
      }
      PetscCall(MatSetValuesStencil(Jpre, 1, &row, nc, col, val, INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(Jpre, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Jpre, MAT_FINAL_ASSEMBLY));
  if (J != Jpre) {
    PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* ------------------------------------------------------------------- */
PetscErrorCode FormInitialSolution(DM da, Vec U, void *ptr)
{
  AppCtx       *user = (AppCtx *)ptr;
  PetscReal     c    = user->c;
  PetscInt      i, j, xs, ys, xm, ym, Mx, My;
  PetscScalar **u;
  PetscReal     hx, hy, x, y, r;

  PetscFunctionBeginUser;
  PetscCall(DMDAGetInfo(da, PETSC_IGNORE, &Mx, &My, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE));

  hx = 1.0 / (PetscReal)(Mx - 1);
  hy = 1.0 / (PetscReal)(My - 1);

  /* Get pointers to vector data */
  PetscCall(DMDAVecGetArray(da, U, &u));

  /* Get local grid boundaries */
  PetscCall(DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL));

  /* Compute function over the locally owned part of the grid */
  for (j = ys; j < ys + ym; j++) {
    y = j * hy;
    for (i = xs; i < xs + xm; i++) {
      x = i * hx;
      r = PetscSqrtReal((x - .5) * (x - .5) + (y - .5) * (y - .5));
      if (r < .125) u[j][i] = PetscExpReal(c * r * r * r);
      else u[j][i] = 0.0;
    }
  }

  /* Restore vectors */
  PetscCall(DMDAVecRestoreArray(da, U, &u));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*TEST

    test:
      args: -ts_max_steps 5 -ts_monitor

    test:
      suffix: 2
      args: -ts_max_steps 5 -ts_monitor

    test:
      suffix: 3
      args: -ts_max_steps 5 -snes_fd_color -ts_monitor

TEST*/
