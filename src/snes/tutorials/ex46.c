static char help[] = "Surface processes in geophysics.\n\n";

#include <petscsnes.h>
#include <petscdm.h>
#include <petscdmda.h>

/*
   User-defined application context - contains data needed by the
   application-provided call-back routines, FormJacobianLocal() and
   FormFunctionLocal().
*/
typedef struct {
  PetscReal D; /* The diffusion coefficient */
  PetscReal K; /* The advection coefficient */
  PetscInt  m; /* Exponent for A */
} AppCtx;

/*
   User-defined routines
*/
extern PetscErrorCode FormFunctionLocal(DMDALocalInfo *, PetscScalar **, PetscScalar **, AppCtx *);
extern PetscErrorCode FormJacobianLocal(DMDALocalInfo *, PetscScalar **, Mat, AppCtx *);

int main(int argc, char **argv)
{
  SNES     snes; /* nonlinear solver */
  AppCtx   user; /* user-defined work context */
  PetscInt its;  /* iterations for convergence */
  DM       da;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize problem parameters
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscOptionsBegin(PETSC_COMM_WORLD, "", "Surface Process Problem Options", "SNES");
  user.D = 1.0;
  PetscCall(PetscOptionsReal("-D", "The diffusion coefficient D", __FILE__, user.D, &user.D, NULL));
  user.K = 1.0;
  PetscCall(PetscOptionsReal("-K", "The advection coefficient K", __FILE__, user.K, &user.K, NULL));
  user.m = 1;
  PetscCall(PetscOptionsInt("-m", "The exponent for A", __FILE__, user.m, &user.m, NULL));
  PetscOptionsEnd();

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, 4, 4, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMDASetUniformCoordinates(da, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0));
  PetscCall(DMSetApplicationContext(da, &user));
  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
  PetscCall(SNESSetDM(snes, da));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set local function evaluation routine
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(DMDASNESSetFunctionLocal(da, INSERT_VALUES, (PetscErrorCode(*)(DMDALocalInfo *, void *, void *, void *))FormFunctionLocal, &user));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize solver; set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(SNESSetFromOptions(snes));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(SNESSolve(snes, 0, 0));
  PetscCall(SNESGetIterationNumber(snes, &its));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Number of SNES iterations = %" PetscInt_FMT "\n", its));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(SNESDestroy(&snes));
  PetscCall(DMDestroy(&da));

  PetscCall(PetscFinalize());
  return 0;
}

PetscScalar funcU(DMDACoor2d *coords)
{
  return coords->x + coords->y;
}

PetscScalar funcA(PetscScalar z, AppCtx *user)
{
  PetscScalar v = 1.0;
  PetscInt    i;

  for (i = 0; i < user->m; ++i) v *= z;
  return v;
}

PetscScalar funcADer(PetscScalar z, AppCtx *user)
{
  PetscScalar v = 1.0;
  PetscInt    i;

  for (i = 0; i < user->m - 1; ++i) v *= z;
  return (PetscScalar)user->m * v;
}

/*
   FormFunctionLocal - Evaluates nonlinear function, F(x).
*/
PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, PetscScalar **x, PetscScalar **f, AppCtx *user)
{
  DM           coordDA;
  Vec          coordinates;
  DMDACoor2d **coords;
  PetscScalar  u, ux, uy, uxx, uyy;
  PetscReal    D, K, hx, hy, hxdhy, hydhx;
  PetscInt     i, j;

  PetscFunctionBeginUser;
  D     = user->D;
  K     = user->K;
  hx    = 1.0 / (PetscReal)(info->mx - 1);
  hy    = 1.0 / (PetscReal)(info->my - 1);
  hxdhy = hx / hy;
  hydhx = hy / hx;
  /*
     Compute function over the locally owned part of the grid
  */
  PetscCall(DMGetCoordinateDM(info->da, &coordDA));
  PetscCall(DMGetCoordinates(info->da, &coordinates));
  PetscCall(DMDAVecGetArray(coordDA, coordinates, &coords));
  for (j = info->ys; j < info->ys + info->ym; j++) {
    for (i = info->xs; i < info->xs + info->xm; i++) {
      if (i == 0 || j == 0 || i == info->mx - 1 || j == info->my - 1) f[j][i] = x[j][i];
      else {
        u       = x[j][i];
        ux      = (x[j][i + 1] - x[j][i]) / hx;
        uy      = (x[j + 1][i] - x[j][i]) / hy;
        uxx     = (2.0 * u - x[j][i - 1] - x[j][i + 1]) * hydhx;
        uyy     = (2.0 * u - x[j - 1][i] - x[j + 1][i]) * hxdhy;
        f[j][i] = D * (uxx + uyy) - (K * funcA(x[j][i], user) * PetscSqrtScalar(ux * ux + uy * uy) + funcU(&coords[j][i])) * hx * hy;
        PetscCheck(!PetscIsInfOrNanScalar(f[j][i]), PETSC_COMM_SELF, PETSC_ERR_FP, "Invalid residual: %g", (double)PetscRealPart(f[j][i]));
      }
    }
  }
  PetscCall(DMDAVecRestoreArray(coordDA, coordinates, &coords));
  PetscCall(PetscLogFlops(11.0 * info->ym * info->xm));
  PetscFunctionReturn(0);
}

/*
   FormJacobianLocal - Evaluates Jacobian matrix.
*/
PetscErrorCode FormJacobianLocal(DMDALocalInfo *info, PetscScalar **x, Mat jac, AppCtx *user)
{
  MatStencil  col[5], row;
  PetscScalar D, K, A, v[5], hx, hy, hxdhy, hydhx, ux, uy;
  PetscReal   normGradZ;
  PetscInt    i, j, k;

  PetscFunctionBeginUser;
  D     = user->D;
  K     = user->K;
  hx    = 1.0 / (PetscReal)(info->mx - 1);
  hy    = 1.0 / (PetscReal)(info->my - 1);
  hxdhy = hx / hy;
  hydhx = hy / hx;

  /*
     Compute entries for the locally owned part of the Jacobian.
      - Currently, all PETSc parallel matrix formats are partitioned by
        contiguous chunks of rows across the processors.
      - Each processor needs to insert only elements that it owns
        locally (but any non-local elements will be sent to the
        appropriate processor during matrix assembly).
      - Here, we set all entries for a particular row at once.
      - We can set matrix entries either using either
        MatSetValuesLocal() or MatSetValues(), as discussed above.
  */
  for (j = info->ys; j < info->ys + info->ym; j++) {
    for (i = info->xs; i < info->xs + info->xm; i++) {
      row.j = j;
      row.i = i;
      if (i == 0 || j == 0 || i == info->mx - 1 || j == info->my - 1) {
        /* boundary points */
        v[0] = 1.0;
        PetscCall(MatSetValuesStencil(jac, 1, &row, 1, &row, v, INSERT_VALUES));
      } else {
        /* interior grid points */
        ux        = (x[j][i + 1] - x[j][i]) / hx;
        uy        = (x[j + 1][i] - x[j][i]) / hy;
        normGradZ = PetscRealPart(PetscSqrtScalar(ux * ux + uy * uy));
        if (normGradZ < 1.0e-8) normGradZ = 1.0e-8;
        A = funcA(x[j][i], user);

        v[0]     = -D * hxdhy;
        col[0].j = j - 1;
        col[0].i = i;
        v[1]     = -D * hydhx;
        col[1].j = j;
        col[1].i = i - 1;
        v[2]     = D * 2.0 * (hydhx + hxdhy) + K * (funcADer(x[j][i], user) * normGradZ - A / normGradZ) * hx * hy;
        col[2].j = row.j;
        col[2].i = row.i;
        v[3]     = -D * hydhx + K * A * hx * hy / (2.0 * normGradZ);
        col[3].j = j;
        col[3].i = i + 1;
        v[4]     = -D * hxdhy + K * A * hx * hy / (2.0 * normGradZ);
        col[4].j = j + 1;
        col[4].i = i;
        for (k = 0; k < 5; ++k) PetscCheck(!PetscIsInfOrNanScalar(v[k]), PETSC_COMM_SELF, PETSC_ERR_FP, "Invalid residual: %g", (double)PetscRealPart(v[k]));
        PetscCall(MatSetValuesStencil(jac, 1, &row, 5, col, v, INSERT_VALUES));
      }
    }
  }

  /*
     Assemble matrix, using the 2-step process:
       MatAssemblyBegin(), MatAssemblyEnd().
  */
  PetscCall(MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY));
  /*
     Tell the matrix we will never add a new nonzero location to the
     matrix. If we do, it will generate an error.
  */
  PetscCall(MatSetOption(jac, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE));
  PetscFunctionReturn(0);
}

/*TEST

   test:
      args: -snes_view -snes_monitor_short -da_refine 1 -pc_type mg -ksp_type fgmres -pc_mg_type full -mg_levels_ksp_chebyshev_esteig 0.5,1.1

   test:
      suffix: ew_1
      args: -snes_monitor_short -ksp_converged_reason -da_grid_x 20 -da_grid_y 20 -snes_ksp_ew -snes_ksp_ew_version 1
      requires: !single

   test:
      suffix: ew_2
      args: -snes_monitor_short -ksp_converged_reason -da_grid_x 20 -da_grid_y 20 -snes_ksp_ew -snes_ksp_ew_version 2

   test:
      suffix: ew_3
      args: -snes_monitor_short -ksp_converged_reason -da_grid_x 20 -da_grid_y 20 -snes_ksp_ew -snes_ksp_ew_version 3
      requires: !single

   test:
      suffix: fm_rise_2
      args: -K 3 -m 1 -D 0.2 -snes_monitor_short -snes_type ngmres -snes_npc_side right -npc_snes_type newtonls -npc_snes_linesearch_type basic -snes_ngmres_restart_it 1 -snes_ngmres_restart_fm_rise
      requires: !single

   test:
      suffix: fm_rise_4
      args: -K 3 -m 1 -D 0.2 -snes_monitor_short -snes_type ngmres -snes_npc_side right -npc_snes_type newtonls -npc_snes_linesearch_type basic -snes_ngmres_restart_it 2 -snes_ngmres_restart_fm_rise -snes_rtol 1.e-2 -snes_max_it 5

TEST*/
