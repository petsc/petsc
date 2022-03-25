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
  PetscCall(DMCreateLabel(dm, "top"));
  PetscCall(DMCreateLabel(dm, "bottom"));
  PetscCall(DMGetLabel(dm, "top", &top));
  PetscCall(DMGetLabel(dm, "bottom", &bottom));
  PetscCall(DMGetBoundingBox(dm, low, high));
  midy = 0.5*(high[1] - low[1]);
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  for (c = cStart; c < cEnd; ++c) {
    PetscReal centroid[3];

    PetscCall(DMPlexComputeCellGeometryFVM(dm, c, NULL, centroid, NULL));
    if (centroid[1] > midy) PetscCall(DMLabelSetValue(top, c, 1));
    else                    PetscCall(DMLabelSetValue(bottom, c, 1));
  }
  PetscCall(DMPlexLabelComplete(dm, top));
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscFunctionBeginUser;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DivideDomain(*dm, user));
  PetscCall(DMSetApplicationContext(*dm, user));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupPrimalProblem(DM dm, AppCtx *user)
{
  PetscDS        ds;
  PetscWeakForm  wf;
  DMLabel        label;
  const PetscInt id = 1;

  PetscFunctionBeginUser;
  PetscCall(DMGetRegionNumDS(dm, 0, &label, NULL, &ds));
  PetscCall(PetscDSGetWeakForm(ds, &wf));
  PetscCall(PetscWeakFormSetIndexResidual(wf, label, 1, 0, 0, 0, f0_quad_u, 0, f1_u));
  PetscCall(PetscWeakFormSetIndexJacobian(wf, label, 1, 0, 0, 0, 0, NULL, 0, NULL, 0, NULL, 0, g3_uu));
  PetscCall(PetscDSSetExactSolution(ds, 0, quad_u, user));
  PetscCall(DMGetRegionNumDS(dm, 1, &label, NULL, &ds));
  PetscCall(PetscDSGetWeakForm(ds, &wf));
  PetscCall(PetscWeakFormSetIndexResidual(wf, label, 1, 0, 0, 0, f0_quad_u, 0, f1_u));
  PetscCall(PetscWeakFormSetIndexJacobian(wf, label, 1, 0, 0, 0, 0, NULL, 0, NULL, 0, NULL, 0, g3_uu));
  PetscCall(PetscWeakFormSetIndexResidual(wf, label, 1, 1, 0, 0, f0_quad_p, 0, NULL));
  PetscCall(PetscWeakFormSetIndexJacobian(wf, label, 1, 1, 1, 0, 0, g0_pp, 0, NULL, 0, NULL, 0, NULL));
  PetscCall(PetscDSSetExactSolution(ds, 0, quad_u, user));
  PetscCall(PetscDSSetExactSolution(ds, 1, quad_p, user));
  PetscCall(DMGetLabel(dm, "marker", &label));
  PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0, NULL, (void (*)(void)) quad_u, NULL, user, NULL));
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
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexIsSimplex(dm, &simplex));
  PetscCall(PetscSNPrintf(prefix, PETSC_MAX_PATH_LEN, "%s_", name));
  PetscCall(PetscFECreateDefault(PetscObjectComm((PetscObject) dm), dim, 1, simplex, name ? prefix : NULL, -1, &fe));
  PetscCall(PetscObjectSetName((PetscObject) fe, name));
  PetscCall(DMSetField(dm, 0, NULL, (PetscObject) fe));
  PetscCall(PetscFEGetQuadrature(fe, &q));
  PetscCall(PetscFEDestroy(&fe));
  PetscCall(DMGetLabel(dm, "top", &top));
  PetscCall(PetscSNPrintf(prefix, PETSC_MAX_PATH_LEN, "%s_top_", nameTop));
  PetscCall(PetscFECreateDefault(PetscObjectComm((PetscObject) dm), dim, 1, simplex, name ? prefix : NULL, -1, &feTop));
  PetscCall(PetscObjectSetName((PetscObject) feTop, nameTop));
  PetscCall(PetscFESetQuadrature(feTop, q));
  PetscCall(DMSetField(dm, 1, top, (PetscObject) feTop));
  PetscCall(PetscFEDestroy(&feTop));
  PetscCall(DMCreateDS(dm));
  PetscCall((*setup)(dm, user));
  while (cdm) {
    PetscCall(DMCopyDisc(dm,cdm));
    PetscCall(DMGetCoarseDM(cdm, &cdm));
  }
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;   /* Problem specification */
  SNES           snes; /* Nonlinear solver */
  Vec            u;    /* Solutions */
  AppCtx         user; /* User-defined work context */

  PetscCall(PetscInitialize(&argc, &argv, NULL,help));
  /* Primal system */
  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
  PetscCall(SNESSetDM(snes, dm));
  PetscCall(SetupDiscretization(dm, "potential", SetupPrimalProblem, &user));
  PetscCall(DMCreateGlobalVector(dm, &u));
  PetscCall(VecSet(u, 0.0));
  PetscCall(PetscObjectSetName((PetscObject) u, "solution"));
  PetscCall(DMPlexSetSNESLocalFEM(dm, &user, &user, &user));
  PetscCall(SNESSetFromOptions(snes));
  PetscCall(DMSNESCheckFromOptions(snes, u));
  PetscCall(SNESSolve(snes, NULL, u));
  PetscCall(SNESGetSolution(snes, &u));
  PetscCall(VecViewFromOptions(u, NULL, "-sol_view"));
  /* Cleanup */
  PetscCall(VecDestroy(&u));
  PetscCall(SNESDestroy(&snes));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
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
