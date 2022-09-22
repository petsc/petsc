static char help[] = "Solves a time-dependent nonlinear PDE.\n";

/* ------------------------------------------------------------------------

   This program solves the two-dimensional time-dependent Bratu problem
       u_t = u_xx +  u_yy + \lambda*exp(u),
   on the domain 0 <= x,y <= 1,
   with the boundary conditions
       u(t,0,y) = 0, u_x(t,1,y) = 0,
       u(t,x,0) = 0, u_x(t,x,1) = 0,
   and the initial condition
       u(0,x,y) = 0.
   We discretize the right-hand side using finite differences with
   uniform grid spacings hx,hy:
       u_xx = (u_{i+1} - 2u_{i} + u_{i-1})/(hx^2)
       u_yy = (u_{j+1} - 2u_{j} + u_{j-1})/(hy^2)

  ------------------------------------------------------------------------- */

#include <petscdmda.h>
#include <petscts.h>

/*
   User-defined application context - contains data needed by the
   application-provided call-back routines.
*/
typedef struct {
  PetscReal lambda;
} AppCtx;

/*
   FormIFunctionLocal - Evaluates nonlinear implicit function on local process patch
 */
static PetscErrorCode FormIFunctionLocal(DMDALocalInfo *info, PetscReal t, PetscScalar **x, PetscScalar **xdot, PetscScalar **f, AppCtx *app)
{
  PetscInt    i, j;
  PetscReal   lambda, hx, hy;
  PetscScalar ut, u, ue, uw, un, us, uxx, uyy;

  PetscFunctionBeginUser;
  lambda = app->lambda;
  hx     = 1.0 / (PetscReal)(info->mx - 1);
  hy     = 1.0 / (PetscReal)(info->my - 1);

  /*
     Compute RHS function over the locally owned part of the grid
  */
  for (j = info->ys; j < info->ys + info->ym; j++) {
    for (i = info->xs; i < info->xs + info->xm; i++) {
      if (i == 0 || j == 0 || i == info->mx - 1 || j == info->my - 1) {
        /* boundary points */
        f[j][i] = x[j][i] - (PetscReal)0;
      } else {
        /* interior points */
        ut = xdot[j][i];
        u  = x[j][i];
        uw = x[j][i - 1];
        ue = x[j][i + 1];
        un = x[j + 1][i];
        us = x[j - 1][i];

        uxx     = (uw - 2.0 * u + ue) / (hx * hx);
        uyy     = (un - 2.0 * u + us) / (hy * hy);
        f[j][i] = ut - uxx - uyy - lambda * PetscExpScalar(u);
      }
    }
  }
  PetscFunctionReturn(0);
}

/*
   FormIJacobianLocal - Evaluates implicit Jacobian matrix on local process patch
*/
static PetscErrorCode FormIJacobianLocal(DMDALocalInfo *info, PetscReal t, PetscScalar **x, PetscScalar **xdot, PetscScalar shift, Mat jac, Mat jacpre, AppCtx *app)
{
  PetscInt    i, j, k;
  MatStencil  col[5], row;
  PetscScalar v[5], lambda, hx, hy;

  PetscFunctionBeginUser;
  lambda = app->lambda;
  hx     = 1.0 / (PetscReal)(info->mx - 1);
  hy     = 1.0 / (PetscReal)(info->my - 1);

  /*
     Compute Jacobian entries for the locally owned part of the grid
  */
  for (j = info->ys; j < info->ys + info->ym; j++) {
    for (i = info->xs; i < info->xs + info->xm; i++) {
      row.j = j;
      row.i = i;
      k     = 0;
      if (i == 0 || j == 0 || i == info->mx - 1 || j == info->my - 1) {
        /* boundary points */
        v[0] = 1.0;
        PetscCall(MatSetValuesStencil(jacpre, 1, &row, 1, &row, v, INSERT_VALUES));
      } else {
        /* interior points */
        v[k]     = -1.0 / (hy * hy);
        col[k].j = j - 1;
        col[k].i = i;
        k++;
        v[k]     = -1.0 / (hx * hx);
        col[k].j = j;
        col[k].i = i - 1;
        k++;

        v[k]     = shift + 2.0 / (hx * hx) + 2.0 / (hy * hy) - lambda * PetscExpScalar(x[j][i]);
        col[k].j = j;
        col[k].i = i;
        k++;

        v[k]     = -1.0 / (hx * hx);
        col[k].j = j;
        col[k].i = i + 1;
        k++;
        v[k]     = -1.0 / (hy * hy);
        col[k].j = j + 1;
        col[k].i = i;
        k++;

        PetscCall(MatSetValuesStencil(jacpre, 1, &row, k, col, v, INSERT_VALUES));
      }
    }
  }

  /*
     Assemble matrix
  */
  PetscCall(MatAssemblyBegin(jacpre, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(jacpre, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  TS              ts; /* ODE integrator */
  DM              da; /* DM context */
  Vec             U;  /* solution vector */
  DMBoundaryType  bt = DM_BOUNDARY_NONE;
  DMDAStencilType st = DMDA_STENCIL_STAR;
  PetscInt        sw = 1;
  PetscInt        N  = 17;
  PetscInt        n  = PETSC_DECIDE;
  AppCtx          app;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "ex21 options", "");
  {
    app.lambda = 6.8;
    app.lambda = 6.0;
    PetscCall(PetscOptionsReal("-lambda", "", "", app.lambda, &app.lambda, NULL));
  }
  PetscOptionsEnd();

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create DM context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, bt, bt, st, N, N, n, n, 1, sw, NULL, NULL, &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMDASetUniformCoordinates(da, 0.0, 1.0, 0.0, 1.0, 0, 1.0));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(TSSetProblemType(ts, TS_NONLINEAR));
  PetscCall(TSSetDM(ts, da));
  PetscCall(DMDestroy(&da));

  PetscCall(TSGetDM(ts, &da));
  PetscCall(DMDATSSetIFunctionLocal(da, INSERT_VALUES, (DMDATSIFunctionLocal)FormIFunctionLocal, &app));
  PetscCall(DMDATSSetIJacobianLocal(da, (DMDATSIJacobianLocal)FormIJacobianLocal, &app));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solver options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSetType(ts, TSBDF));
  PetscCall(TSSetTimeStep(ts, 1e-4));
  PetscCall(TSSetMaxTime(ts, 1.0));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER));
  PetscCall(TSSetFromOptions(ts));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSGetDM(ts, &da));
  PetscCall(DMCreateGlobalVector(da, &U));
  PetscCall(VecSet(U, 0.0));
  PetscCall(TSSetSolution(ts, U));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Run timestepping solver
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSolve(ts, U));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      All PETSc objects should be destroyed when they are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(VecDestroy(&U));
  PetscCall(TSDestroy(&ts));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

    testset:
      requires: !single !complex
      args: -da_grid_x 5 -da_grid_y 5 -da_refine 2 -dm_view -ts_type bdf -ts_adapt_type none -ts_dt 1e-3 -ts_monitor -ts_max_steps 5 -ts_view -snes_rtol 1e-6 -snes_type ngmres -npc_snes_type fas
      filter: grep -v "total number of"
      test:
        suffix: 1_bdf_ngmres_fas_ms
        args: -prefix_push npc_fas_levels_ -snes_type ms -snes_max_it 5 -ksp_type preonly -prefix_pop
      test:
        suffix: 2_bdf_ngmres_fas_ms
        args: -prefix_push npc_fas_levels_ -snes_type ms -snes_max_it 5 -ksp_type preonly -prefix_pop
        nsize: 2
      test:
        suffix: 1_bdf_ngmres_fas_ngs
        args: -prefix_push npc_fas_levels_ -snes_type ngs -snes_max_it 5 -prefix_pop
      test:
        suffix: 2_bdf_ngmres_fas_ngs
        args: -prefix_push npc_fas_levels_ -snes_type ngs -snes_max_it 5 -prefix_pop
        nsize: 2

TEST*/
