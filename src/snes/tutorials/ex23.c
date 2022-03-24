static char help[] = "Poisson Problem with a split domain.\n\
We solve the Poisson problem in two halves of a domain.\n\
In one half, we include an additional field.\n\n\n";

#include <petscdmplex.h>
#include <petscsnes.h>
#include <petscds.h>
#include <petscconvest.h>

typedef struct {
  PetscInt dummy;
} AppCtx;

static PetscErrorCode quad_u(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = x[0]*x[0] + x[1]*x[1];
  return 0;
}

static PetscErrorCode quad_p(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = 2.0;
  return 0;
}

static void f0_quad_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                      const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                      const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                      PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f0[0] += 2.0;
}

static void f0_quad_p(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                      const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                      const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                      PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = u[uOff[1]] - 2.0;
}

static void f1_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f1[d] = u_x[d];
}

static void g3_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g3[d*dim+d] = 1.0;
}

static void g0_pp(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  g0[0] = 1.0;
}

static PetscErrorCode DivideDomain(DM dm, AppCtx *user)
{
  DMLabel        top, bottom;
  PetscReal      low[3], high[3], midy;
  PetscInt       cStart, cEnd, c;

  PetscFunctionBeginUser;
  CHKERRQ(DMCreateLabel(dm, "top"));
  CHKERRQ(DMCreateLabel(dm, "bottom"));
  CHKERRQ(DMGetLabel(dm, "top", &top));
  CHKERRQ(DMGetLabel(dm, "bottom", &bottom));
  CHKERRQ(DMGetBoundingBox(dm, low, high));
  midy = 0.5*(high[1] - low[1]);
  CHKERRQ(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  for (c = cStart; c < cEnd; ++c) {
    PetscReal centroid[3];

    CHKERRQ(DMPlexComputeCellGeometryFVM(dm, c, NULL, centroid, NULL));
    if (centroid[1] > midy) CHKERRQ(DMLabelSetValue(top, c, 1));
    else                    CHKERRQ(DMLabelSetValue(bottom, c, 1));
  }
  CHKERRQ(DMPlexLabelComplete(dm, top));
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscFunctionBeginUser;
  CHKERRQ(DMCreate(comm, dm));
  CHKERRQ(DMSetType(*dm, DMPLEX));
  CHKERRQ(DMSetFromOptions(*dm));
  CHKERRQ(DivideDomain(*dm, user));
  CHKERRQ(DMSetApplicationContext(*dm, user));
  CHKERRQ(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupPrimalProblem(DM dm, AppCtx *user)
{
  PetscDS        ds;
  PetscWeakForm  wf;
  DMLabel        label;
  const PetscInt id = 1;

  PetscFunctionBeginUser;
  CHKERRQ(DMGetRegionNumDS(dm, 0, &label, NULL, &ds));
  CHKERRQ(PetscDSGetWeakForm(ds, &wf));
  CHKERRQ(PetscWeakFormSetIndexResidual(wf, label, 1, 0, 0, 0, f0_quad_u, 0, f1_u));
  CHKERRQ(PetscWeakFormSetIndexJacobian(wf, label, 1, 0, 0, 0, 0, NULL, 0, NULL, 0, NULL, 0, g3_uu));
  CHKERRQ(PetscDSSetExactSolution(ds, 0, quad_u, user));
  CHKERRQ(DMGetRegionNumDS(dm, 1, &label, NULL, &ds));
  CHKERRQ(PetscDSGetWeakForm(ds, &wf));
  CHKERRQ(PetscWeakFormSetIndexResidual(wf, label, 1, 0, 0, 0, f0_quad_u, 0, f1_u));
  CHKERRQ(PetscWeakFormSetIndexJacobian(wf, label, 1, 0, 0, 0, 0, NULL, 0, NULL, 0, NULL, 0, g3_uu));
  CHKERRQ(PetscWeakFormSetIndexResidual(wf, label, 1, 1, 0, 0, f0_quad_p, 0, NULL));
  CHKERRQ(PetscWeakFormSetIndexJacobian(wf, label, 1, 1, 1, 0, 0, g0_pp, 0, NULL, 0, NULL, 0, NULL));
  CHKERRQ(PetscDSSetExactSolution(ds, 0, quad_u, user));
  CHKERRQ(PetscDSSetExactSolution(ds, 1, quad_p, user));
  CHKERRQ(DMGetLabel(dm, "marker", &label));
  CHKERRQ(DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0, NULL, (void (*)(void)) quad_u, NULL, user, NULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDiscretization(DM dm, const char name[], PetscErrorCode (*setup)(DM, AppCtx *), AppCtx *user)
{
  DM              cdm = dm;
  DMLabel         top;
  PetscFE         fe, feTop;
  PetscQuadrature q;
  PetscInt        dim;
  PetscBool       simplex;
  const char     *nameTop = "pressure";
  char            prefix[PETSC_MAX_PATH_LEN];

  PetscFunctionBeginUser;
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMPlexIsSimplex(dm, &simplex));
  CHKERRQ(PetscSNPrintf(prefix, PETSC_MAX_PATH_LEN, "%s_", name));
  CHKERRQ(PetscFECreateDefault(PetscObjectComm((PetscObject) dm), dim, 1, simplex, name ? prefix : NULL, -1, &fe));
  CHKERRQ(PetscObjectSetName((PetscObject) fe, name));
  CHKERRQ(DMSetField(dm, 0, NULL, (PetscObject) fe));
  CHKERRQ(PetscFEGetQuadrature(fe, &q));
  CHKERRQ(PetscFEDestroy(&fe));
  CHKERRQ(DMGetLabel(dm, "top", &top));
  CHKERRQ(PetscSNPrintf(prefix, PETSC_MAX_PATH_LEN, "%s_top_", nameTop));
  CHKERRQ(PetscFECreateDefault(PetscObjectComm((PetscObject) dm), dim, 1, simplex, name ? prefix : NULL, -1, &feTop));
  CHKERRQ(PetscObjectSetName((PetscObject) feTop, nameTop));
  CHKERRQ(PetscFESetQuadrature(feTop, q));
  CHKERRQ(DMSetField(dm, 1, top, (PetscObject) feTop));
  CHKERRQ(PetscFEDestroy(&feTop));
  CHKERRQ(DMCreateDS(dm));
  CHKERRQ((*setup)(dm, user));
  while (cdm) {
    CHKERRQ(DMCopyDisc(dm,cdm));
    CHKERRQ(DMGetCoarseDM(cdm, &cdm));
  }
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;   /* Problem specification */
  SNES           snes; /* Nonlinear solver */
  Vec            u;    /* Solutions */
  AppCtx         user; /* User-defined work context */

  CHKERRQ(PetscInitialize(&argc, &argv, NULL,help));
  /* Primal system */
  CHKERRQ(SNESCreate(PETSC_COMM_WORLD, &snes));
  CHKERRQ(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
  CHKERRQ(SNESSetDM(snes, dm));
  CHKERRQ(SetupDiscretization(dm, "potential", SetupPrimalProblem, &user));
  CHKERRQ(DMCreateGlobalVector(dm, &u));
  CHKERRQ(VecSet(u, 0.0));
  CHKERRQ(PetscObjectSetName((PetscObject) u, "solution"));
  CHKERRQ(DMPlexSetSNESLocalFEM(dm, &user, &user, &user));
  CHKERRQ(SNESSetFromOptions(snes));
  CHKERRQ(DMSNESCheckFromOptions(snes, u));
  CHKERRQ(SNESSolve(snes, NULL, u));
  CHKERRQ(SNESGetSolution(snes, &u));
  CHKERRQ(VecViewFromOptions(u, NULL, "-sol_view"));
  /* Cleanup */
  CHKERRQ(VecDestroy(&u));
  CHKERRQ(SNESDestroy(&snes));
  CHKERRQ(DMDestroy(&dm));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 2d_p1_0
    requires: triangle
    args: -potential_petscspace_degree 2 -pressure_top_petscspace_degree 0 -dm_refine 0 -dmsnes_check

  test:
    suffix: 2d_p1_1
    requires: triangle
    args: -potential_petscspace_degree 1 -pressure_top_petscspace_degree 0 -dm_refine 0 -convest_num_refine 3 -snes_convergence_estimate

TEST*/
