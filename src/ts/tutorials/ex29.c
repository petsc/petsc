static char help[] = "Time-Dependent Allan-Cahn example in 2D with Varying Coefficients";

/*
 This example is mainly here to show how to transfer coefficients between subdomains and levels in
 multigrid and domain decomposition.
 */

#include <petscdm.h>
#include <petscdmda.h>
#include <petscsnes.h>
#include <petscts.h>

typedef struct {
  PetscScalar epsilon;
  PetscScalar beta;
} Coeff;

typedef struct {
  PetscScalar u;
} Field;

extern PetscErrorCode FormInitialGuess(DM da, void *ctx, Vec X);
extern PetscErrorCode FormDiffusionCoefficient(DM da, void *ctx, Vec X);
extern PetscErrorCode FormIFunctionLocal(DMDALocalInfo *, PetscReal, Field **, Field **, Field **, void *);

/* hooks */

static PetscErrorCode CoefficientCoarsenHook(DM dm, DM dmc, void *ctx)
{
  Vec c, cc, ccl;
  Mat J;
  Vec vscale;
  DM  cdm, cdmc;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectQuery((PetscObject)dm, "coefficientdm", (PetscObject *)&cdm));

  PetscCheck(cdm, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "The coefficient DM needs to be set up!");

  PetscCall(DMDACreateCompatibleDMDA(dmc, 2, &cdmc));
  PetscCall(PetscObjectCompose((PetscObject)dmc, "coefficientdm", (PetscObject)cdmc));

  PetscCall(DMGetNamedGlobalVector(cdm, "coefficient", &c));
  PetscCall(DMGetNamedGlobalVector(cdmc, "coefficient", &cc));
  PetscCall(DMGetNamedLocalVector(cdmc, "coefficient", &ccl));

  PetscCall(DMCreateInterpolation(cdmc, cdm, &J, &vscale));
  PetscCall(MatRestrict(J, c, cc));
  PetscCall(VecPointwiseMult(cc, vscale, cc));

  PetscCall(MatDestroy(&J));
  PetscCall(VecDestroy(&vscale));

  PetscCall(DMGlobalToLocalBegin(cdmc, cc, INSERT_VALUES, ccl));
  PetscCall(DMGlobalToLocalEnd(cdmc, cc, INSERT_VALUES, ccl));

  PetscCall(DMRestoreNamedGlobalVector(cdm, "coefficient", &c));
  PetscCall(DMRestoreNamedGlobalVector(cdmc, "coefficient", &cc));
  PetscCall(DMRestoreNamedLocalVector(cdmc, "coefficient", &ccl));

  PetscCall(DMCoarsenHookAdd(dmc, CoefficientCoarsenHook, NULL, NULL));
  PetscCall(DMDestroy(&cdmc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* This could restrict auxiliary information to the coarse level.
 */
static PetscErrorCode CoefficientSubDomainRestrictHook(DM dm, DM subdm, void *ctx)
{
  Vec         c, cc;
  DM          cdm, csubdm;
  VecScatter *iscat, *oscat, *gscat;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectQuery((PetscObject)dm, "coefficientdm", (PetscObject *)&cdm));

  PetscCheck(cdm, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "The coefficient DM needs to be set up!");

  PetscCall(DMDACreateCompatibleDMDA(subdm, 2, &csubdm));
  PetscCall(PetscObjectCompose((PetscObject)subdm, "coefficientdm", (PetscObject)csubdm));

  PetscCall(DMGetNamedGlobalVector(cdm, "coefficient", &c));
  PetscCall(DMGetNamedLocalVector(csubdm, "coefficient", &cc));

  PetscCall(DMCreateDomainDecompositionScatters(cdm, 1, &csubdm, &iscat, &oscat, &gscat));

  PetscCall(VecScatterBegin(*gscat, c, cc, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(*gscat, c, cc, INSERT_VALUES, SCATTER_FORWARD));

  PetscCall(VecScatterDestroy(iscat));
  PetscCall(VecScatterDestroy(oscat));
  PetscCall(VecScatterDestroy(gscat));
  PetscCall(PetscFree(iscat));
  PetscCall(PetscFree(oscat));
  PetscCall(PetscFree(gscat));

  PetscCall(DMRestoreNamedGlobalVector(cdm, "coefficient", &c));
  PetscCall(DMRestoreNamedLocalVector(csubdm, "coefficient", &cc));

  PetscCall(DMDestroy(&csubdm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)

{
  TS  ts;
  Vec x, c, clocal;
  DM  da, cda;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(TSSetType(ts, TSARKIMEX));
  PetscCall(TSSetProblemType(ts, TS_NONLINEAR));
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, 4, 4, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMDASetUniformCoordinates(da, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0));

  PetscCall(DMDASetFieldName(da, 0, "u"));
  PetscCall(DMCreateGlobalVector(da, &x));

  PetscCall(TSSetDM(ts, da));

  PetscCall(FormInitialGuess(da, NULL, x));
  PetscCall(DMDATSSetIFunctionLocal(da, INSERT_VALUES, (PetscErrorCode(*)(DMDALocalInfo *, PetscReal, void *, void *, void *, void *))FormIFunctionLocal, NULL));

  /* set up the coefficient */

  PetscCall(DMDACreateCompatibleDMDA(da, 2, &cda));
  PetscCall(PetscObjectCompose((PetscObject)da, "coefficientdm", (PetscObject)cda));

  PetscCall(DMGetNamedGlobalVector(cda, "coefficient", &c));
  PetscCall(DMGetNamedLocalVector(cda, "coefficient", &clocal));

  PetscCall(FormDiffusionCoefficient(cda, NULL, c));

  PetscCall(DMGlobalToLocalBegin(cda, c, INSERT_VALUES, clocal));
  PetscCall(DMGlobalToLocalEnd(cda, c, INSERT_VALUES, clocal));

  PetscCall(DMRestoreNamedLocalVector(cda, "coefficient", &clocal));
  PetscCall(DMRestoreNamedGlobalVector(cda, "coefficient", &c));

  PetscCall(DMCoarsenHookAdd(da, CoefficientCoarsenHook, NULL, NULL));
  PetscCall(DMSubDomainHookAdd(da, CoefficientSubDomainRestrictHook, NULL, NULL));

  PetscCall(TSSetMaxSteps(ts, 10000));
  PetscCall(TSSetMaxTime(ts, 10000.0));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER));
  PetscCall(TSSetTimeStep(ts, 0.05));
  PetscCall(TSSetSolution(ts, x));
  PetscCall(TSSetFromOptions(ts));

  PetscCall(TSSolve(ts, x));

  PetscCall(VecDestroy(&x));
  PetscCall(TSDestroy(&ts));
  PetscCall(DMDestroy(&da));
  PetscCall(DMDestroy(&cda));

  PetscCall(PetscFinalize());
  return 0;
}

/* ------------------------------------------------------------------- */

PetscErrorCode FormInitialGuess(DM da, void *ctx, Vec X)
{
  PetscInt  i, j, Mx, My, xs, ys, xm, ym;
  Field   **x;
  PetscReal x0, x1;

  PetscFunctionBeginUser;
  PetscCall(DMDAGetInfo(da, PETSC_IGNORE, &Mx, &My, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE));

  PetscCall(DMDAVecGetArray(da, X, &x));
  PetscCall(DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL));

  for (j = ys; j < ys + ym; j++) {
    for (i = xs; i < xs + xm; i++) {
      x0        = 10.0 * (i - 0.5 * (Mx - 1)) / (Mx - 1);
      x1        = 10.0 * (j - 0.5 * (Mx - 1)) / (My - 1);
      x[j][i].u = PetscCosReal(2.0 * PetscSqrtReal(x1 * x1 + x0 * x0));
    }
  }

  PetscCall(DMDAVecRestoreArray(da, X, &x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FormDiffusionCoefficient(DM da, void *ctx, Vec X)
{
  PetscInt  i, j, Mx, My, xs, ys, xm, ym;
  Coeff   **x;
  PetscReal x1, x0;

  PetscFunctionBeginUser;
  PetscCall(DMDAGetInfo(da, PETSC_IGNORE, &Mx, &My, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE));

  /*
  ierr = VecSetRandom(X,NULL);
  PetscCall(VecMin(X,NULL,&min));
   */

  PetscCall(DMDAVecGetArray(da, X, &x));
  PetscCall(DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL));

  for (j = ys; j < ys + ym; j++) {
    for (i = xs; i < xs + xm; i++) {
      x0 = 10.0 * (i - 0.5 * (Mx - 1)) / (Mx - 1);
      x1 = 10.0 * (j - 0.5 * (My - 1)) / (My - 1);

      x[j][i].epsilon = 0.0;
      x[j][i].beta    = 0.05 + 0.05 * PetscSqrtReal(x0 * x0 + x1 * x1);
    }
  }

  PetscCall(DMDAVecRestoreArray(da, X, &x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FormIFunctionLocal(DMDALocalInfo *info, PetscReal ptime, Field **x, Field **xt, Field **f, void *ctx)
{
  PetscInt    i, j;
  PetscReal   hx, hy, dhx, dhy, hxdhy, hydhx, scale;
  PetscScalar u, uxx, uyy;
  PetscScalar ux, uy, bx, by;
  Vec         C;
  Coeff     **c;
  DM          cdm;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectQuery((PetscObject)info->da, "coefficientdm", (PetscObject *)&cdm));
  PetscCall(DMGetNamedLocalVector(cdm, "coefficient", &C));
  PetscCall(DMDAVecGetArray(cdm, C, &c));

  hx = 10.0 / ((PetscReal)(info->mx - 1));
  hy = 10.0 / ((PetscReal)(info->my - 1));

  dhx = 1. / hx;
  dhy = 1. / hy;

  hxdhy = hx / hy;
  hydhx = hy / hx;
  scale = hx * hy;

  for (j = info->ys; j < info->ys + info->ym; j++) {
    for (i = info->xs; i < info->xs + info->xm; i++) {
      f[j][i].u = xt[j][i].u * scale;

      u = x[j][i].u;

      f[j][i].u += scale * (u * u - 1.) * u;

      if (i == 0) f[j][i].u += (x[j][i].u - x[j][i + 1].u) * dhx;
      else if (i == info->mx - 1) f[j][i].u += (x[j][i].u - x[j][i - 1].u) * dhx;
      else if (j == 0) f[j][i].u += (x[j][i].u - x[j + 1][i].u) * dhy;
      else if (j == info->my - 1) f[j][i].u += (x[j][i].u - x[j - 1][i].u) * dhy;
      else {
        uyy = (2.0 * u - x[j - 1][i].u - x[j + 1][i].u) * hxdhy;
        uxx = (2.0 * u - x[j][i - 1].u - x[j][i + 1].u) * hydhx;

        bx = 0.5 * (c[j][i + 1].beta - c[j][i - 1].beta) * dhx;
        by = 0.5 * (c[j + 1][i].beta - c[j - 1][i].beta) * dhy;

        ux = 0.5 * (x[j][i + 1].u - x[j][i - 1].u) * dhx;
        uy = 0.5 * (x[j + 1][i].u - x[j - 1][i].u) * dhy;

        f[j][i].u += c[j][i].beta * (uxx + uyy) + scale * (bx * ux + by * uy);
      }
    }
  }
  PetscCall(PetscLogFlops(11. * info->ym * info->xm));

  PetscCall(DMDAVecRestoreArray(cdm, C, &c));
  PetscCall(DMRestoreNamedLocalVector(cdm, "coefficient", &C));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*TEST

    test:
      args: -da_refine 4 -ts_max_steps 10 -ts_rtol 1e-3 -ts_atol 1e-3 -ts_type arkimex -ts_monitor -snes_monitor -snes_type ngmres  -npc_snes_type nasm -npc_snes_nasm_type restrict -da_overlap 4
      nsize: 16
      requires: !single
      output_file: output/ex29.out

TEST*/
