
static char help[] = "Demonstrates Pattern Formation with Reaction-Diffusion Equations.\n";

/*F
     This example is taken from the book, Numerical Solution of Time-Dependent Advection-Diffusion-Reaction Equations by
      W. Hundsdorf and J.G. Verwer,  Page 21, Pattern Formation with Reaction-Diffusion Equations
\begin{eqnarray*}
        u_t = D_1 (u_{xx} + u_{yy})  - u*v^2 + \gamma(1 -u)           \\
        v_t = D_2 (v_{xx} + v_{yy})  + u*v^2 - (\gamma + \kappa)v
\end{eqnarray*}
    Unlike in the book this uses periodic boundary conditions instead of Neumann
    (since they are easier for finite differences).
F*/

/*
      Helpful runtime monitor options:
           -ts_monitor_draw_solution
           -draw_save -draw_save_movie

      Helpful runtime linear solver options:
           -pc_type mg -pc_mg_galerkin pmat -da_refine 1 -snes_monitor -ksp_monitor -ts_view  (note that these Jacobians are so well-conditioned multigrid may not be the best solver)

      Point your browser to localhost:8080 to monitor the simulation
           ./ex5  -ts_view_pre saws  -stack_view saws -draw_save -draw_save_single_file -x_virtual -ts_monitor_draw_solution -saws_root .

*/

/*

   Include "petscdmda.h" so that we can use distributed arrays (DMDAs).
   Include "petscts.h" so that we can use SNES numerical (ODE) integrators.  Note that this
   file automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h  - vectors
     petscmat.h - matrices                    petscis.h   - index sets
     petscksp.h - Krylov subspace methods     petscpc.h   - preconditioners
     petscviewer.h - viewers                  petscsnes.h - nonlinear solvers
*/
#include "reaction_diffusion.h"
#include <petscdm.h>
#include <petscdmda.h>

/* ------------------------------------------------------------------- */
PetscErrorCode InitialConditions(DM da, Vec U)
{
  PetscInt  i, j, xs, ys, xm, ym, Mx, My;
  Field   **u;
  PetscReal hx, hy, x, y;

  PetscFunctionBegin;
  PetscCall(DMDAGetInfo(da, PETSC_IGNORE, &Mx, &My, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE));

  hx = 2.5 / (PetscReal)(Mx);
  hy = 2.5 / (PetscReal)(My);

  /*
     Get pointers to actual vector data
  */
  PetscCall(DMDAVecGetArray(da, U, &u));

  /*
     Get local grid boundaries
  */
  PetscCall(DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL));

  /*
     Compute function over the locally owned part of the grid
  */
  for (j = ys; j < ys + ym; j++) {
    y = j * hy;
    for (i = xs; i < xs + xm; i++) {
      x = i * hx;
      if (PetscApproximateGTE(x, 1.0) && PetscApproximateLTE(x, 1.5) && PetscApproximateGTE(y, 1.0) && PetscApproximateLTE(y, 1.5))
        u[j][i].v = PetscPowReal(PetscSinReal(4.0 * PETSC_PI * x), 2.0) * PetscPowReal(PetscSinReal(4.0 * PETSC_PI * y), 2.0) / 4.0;
      else u[j][i].v = 0.0;

      u[j][i].u = 1.0 - 2.0 * u[j][i].v;
    }
  }

  /*
     Restore access to vector
  */
  PetscCall(DMDAVecRestoreArray(da, U, &u));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  TS     ts; /* ODE integrator */
  Vec    x;  /* solution */
  DM     da;
  AppCtx appctx;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscFunctionBeginUser;
  appctx.D1    = 8.0e-5;
  appctx.D2    = 4.0e-5;
  appctx.gamma = .024;
  appctx.kappa = .06;
  appctx.aijpc = PETSC_FALSE;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, DMDA_STENCIL_STAR, 65, 65, PETSC_DECIDE, PETSC_DECIDE, 2, 1, NULL, NULL, &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMDASetFieldName(da, 0, "u"));
  PetscCall(DMDASetFieldName(da, 1, "v"));

  /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create global vector from DMDA; this will be used to store the solution
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(DMCreateGlobalVector(da, &x));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(TSSetType(ts, TSARKIMEX));
  PetscCall(TSARKIMEXSetFullyImplicit(ts, PETSC_TRUE));
  PetscCall(TSSetDM(ts, da));
  PetscCall(TSSetProblemType(ts, TS_NONLINEAR));
  PetscCall(TSSetRHSFunction(ts, NULL, RHSFunction, &appctx));
  PetscCall(TSSetRHSJacobian(ts, NULL, NULL, RHSJacobian, &appctx));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(InitialConditions(da, x));
  PetscCall(TSSetSolution(ts, x));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solver options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSetMaxTime(ts, 2000.0));
  PetscCall(TSSetTimeStep(ts, .0001));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER));
  PetscCall(TSSetFromOptions(ts));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve ODE system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSolve(ts, x));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(VecDestroy(&x));
  PetscCall(TSDestroy(&ts));
  PetscCall(DMDestroy(&da));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
     depends: reaction_diffusion.c

   test:
      args: -ts_view  -ts_monitor -ts_max_time 500
      requires: double
      timeoutfactor: 3

   test:
      suffix: 2
      args: -ts_view  -ts_monitor -ts_max_time 500 -ts_monitor_draw_solution
      requires: x double
      output_file: output/ex5_1.out
      timeoutfactor: 3

TEST*/
