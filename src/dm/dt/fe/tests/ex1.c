static const char help[] = "Performance Tests for FE Integration";

#include <petscdmplex.h>
#include <petscfe.h>
#include <petscds.h>

typedef struct {
  PetscInt  dim;     /* The topological dimension */
  PetscBool simplex; /* True for simplices, false for hexes */
  PetscInt  its;     /* Number of replications for timing */
  PetscInt  cbs;     /* Number of cells in an integration block */
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->dim     = 2;
  options->simplex = PETSC_TRUE;
  options->its     = 1;
  options->cbs     = 8;

  ierr = PetscOptionsBegin(comm, "", "FE Integration Performance Options", "PETSCFE");CHKERRQ(ierr);
  CHKERRQ(PetscOptionsInt("-dim", "The topological dimension", "ex1.c", options->dim, &options->dim, NULL));
  CHKERRQ(PetscOptionsBool("-simplex", "Simplex or hex cells", "ex1.c", options->simplex, &options->simplex, NULL));
  CHKERRQ(PetscOptionsInt("-its", "The number of replications for timing", "ex1.c", options->its, &options->its, NULL));
  CHKERRQ(PetscOptionsInt("-cbs", "The number of cells in an integration block", "ex1.c", options->cbs, &options->cbs, NULL));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode trig_u(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt d;
  *u = 0.0;
  for (d = 0; d < dim; ++d) *u += PetscSinReal(2.0*PETSC_PI*x[d]);
  return 0;
}

static void f0_trig_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                      const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                      const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                      PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f0[0] += -4.0*PetscSqr(PETSC_PI)*PetscSinReal(2.0*PETSC_PI*x[d]);
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

static PetscErrorCode SetupPrimalProblem(DM dm, AppCtx *user)
{
  PetscDS        prob;
  DMLabel        label;
  const PetscInt id = 1;

  PetscFunctionBeginUser;
  CHKERRQ(DMGetDS(dm, &prob));
  CHKERRQ(PetscDSSetResidual(prob, 0, f0_trig_u, f1_u));
  CHKERRQ(PetscDSSetJacobian(prob, 0, 0, NULL, NULL, NULL, g3_uu));
  CHKERRQ(PetscDSSetExactSolution(prob, 0, trig_u, user));
  CHKERRQ(DMGetLabel(dm, "marker", &label));
  CHKERRQ(DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0, NULL, (void (*)(void)) trig_u, NULL, user, NULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDiscretization(DM dm, const char name[], PetscErrorCode (*setup)(DM, AppCtx *), AppCtx *user)
{
  DM             cdm = dm;
  PetscFE        fe;
  char           prefix[PETSC_MAX_PATH_LEN];

  PetscFunctionBeginUser;
  /* Create finite element */
  CHKERRQ(PetscSNPrintf(prefix, PETSC_MAX_PATH_LEN, "%s_", name));
  CHKERRQ(PetscFECreateDefault(PetscObjectComm((PetscObject) dm), user->dim, 1, user->simplex, name ? prefix : NULL, -1, &fe));
  CHKERRQ(PetscObjectSetName((PetscObject) fe, name));
  /* Set discretization and boundary conditions for each mesh */
  CHKERRQ(DMSetField(dm, 0, NULL, (PetscObject) fe));
  CHKERRQ(DMCreateDS(dm));
  CHKERRQ((*setup)(dm, user));
  while (cdm) {
    CHKERRQ(DMCopyDisc(dm,cdm));
    /* TODO: Check whether the boundary of coarse meshes is marked */
    CHKERRQ(DMGetCoarseDM(cdm, &cdm));
  }
  CHKERRQ(PetscFEDestroy(&fe));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscContainerUserDestroy_PetscFEGeom(void *ctx)
{
  PetscFEGeom   *geom = (PetscFEGeom *) ctx;

  PetscFunctionBegin;
  CHKERRQ(PetscFEGeomDestroy(&geom));
  PetscFunctionReturn(0);
}

PetscErrorCode CellRangeGetFEGeom(IS cellIS, DMField coordField, PetscQuadrature quad, PetscBool faceData, PetscFEGeom **geom)
{
  char            composeStr[33] = {0};
  PetscObjectId   id;
  PetscContainer  container;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetId((PetscObject) quad, &id));
  CHKERRQ(PetscSNPrintf(composeStr, 32, "CellRangeGetFEGeom_%x\n", id));
  CHKERRQ(PetscObjectQuery((PetscObject) cellIS, composeStr, (PetscObject *) &container));
  if (container) {
    CHKERRQ(PetscContainerGetPointer(container, (void **) geom));
  } else {
    CHKERRQ(DMFieldCreateFEGeom(coordField, cellIS, quad, faceData, geom));
    CHKERRQ(PetscContainerCreate(PETSC_COMM_SELF, &container));
    CHKERRQ(PetscContainerSetPointer(container, (void *) *geom));
    CHKERRQ(PetscContainerSetUserDestroy(container, PetscContainerUserDestroy_PetscFEGeom));
    CHKERRQ(PetscObjectCompose((PetscObject) cellIS, composeStr, (PetscObject) container));
    CHKERRQ(PetscContainerDestroy(&container));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode CellRangeRestoreFEGeom(IS cellIS, DMField coordField, PetscQuadrature quad, PetscBool faceData, PetscFEGeom **geom)
{
  PetscFunctionBegin;
  *geom = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateFEGeometry(DM dm, PetscDS ds, IS cellIS, PetscQuadrature *affineQuad, PetscFEGeom **affineGeom, PetscQuadrature **quads, PetscFEGeom ***geoms)
{
  DMField        coordField;
  PetscInt       Nf, f, maxDegree;

  PetscFunctionBeginUser;
  *affineQuad = NULL;
  *affineGeom = NULL;
  *quads      = NULL;
  *geoms      = NULL;
  CHKERRQ(PetscDSGetNumFields(ds, &Nf));
  CHKERRQ(DMGetCoordinateField(dm, &coordField));
  CHKERRQ(DMFieldGetDegree(coordField, cellIS, NULL, &maxDegree));
  if (maxDegree <= 1) {
    CHKERRQ(DMFieldCreateDefaultQuadrature(coordField, cellIS, affineQuad));
    if (*affineQuad) CHKERRQ(CellRangeGetFEGeom(cellIS, coordField, *affineQuad, PETSC_FALSE, affineGeom));
  } else {
    CHKERRQ(PetscCalloc2(Nf, quads, Nf, geoms));
    for (f = 0; f < Nf; ++f) {
      PetscFE fe;

      CHKERRQ(PetscDSGetDiscretization(ds, f, (PetscObject *) &fe));
      CHKERRQ(PetscFEGetQuadrature(fe, &(*quads)[f]));
      CHKERRQ(PetscObjectReference((PetscObject) (*quads)[f]));
      CHKERRQ(CellRangeGetFEGeom(cellIS, coordField, (*quads)[f], PETSC_FALSE, &(*geoms)[f]));
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DestroyFEGeometry(DM dm, PetscDS ds, IS cellIS, PetscQuadrature *affineQuad, PetscFEGeom **affineGeom, PetscQuadrature **quads, PetscFEGeom ***geoms)
{
  DMField        coordField;
  PetscInt       Nf, f;

  PetscFunctionBeginUser;
  CHKERRQ(PetscDSGetNumFields(ds, &Nf));
  CHKERRQ(DMGetCoordinateField(dm, &coordField));
  if (*affineQuad) {
    CHKERRQ(CellRangeRestoreFEGeom(cellIS, coordField, *affineQuad, PETSC_FALSE, affineGeom));
    CHKERRQ(PetscQuadratureDestroy(affineQuad));
  } else {
    for (f = 0; f < Nf; ++f) {
      CHKERRQ(CellRangeRestoreFEGeom(cellIS, coordField, (*quads)[f], PETSC_FALSE, &(*geoms)[f]));
      CHKERRQ(PetscQuadratureDestroy(&(*quads)[f]));
    }
    CHKERRQ(PetscFree2(*quads, *geoms));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TestIntegration(DM dm, PetscInt cbs, PetscInt its)
{
  PetscDS         ds;
  PetscFEGeom    *chunkGeom = NULL;
  PetscQuadrature affineQuad,  *quads = NULL;
  PetscFEGeom    *affineGeom, **geoms = NULL;
  PetscScalar    *u, *elemVec;
  IS              cellIS;
  PetscInt        depth, cStart, cEnd, cell, chunkSize = cbs, Nch = 0, Nf, f, totDim, i, k;
#if defined(PETSC_USE_LOG)
  PetscLogStage   stage;
  PetscLogEvent   event;
#endif

  PetscFunctionBeginUser;
  CHKERRQ(PetscLogStageRegister("PetscFE Residual Integration Test", &stage));
  CHKERRQ(PetscLogEventRegister("FEIntegRes", PETSCFE_CLASSID, &event));
  CHKERRQ(PetscLogStagePush(stage));
  CHKERRQ(DMPlexGetDepth(dm, &depth));
  CHKERRQ(DMGetStratumIS(dm, "depth", depth, &cellIS));
  CHKERRQ(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  CHKERRQ(DMGetCellDS(dm, cStart, &ds));
  CHKERRQ(PetscDSGetNumFields(ds, &Nf));
  CHKERRQ(PetscDSGetTotalDimension(ds, &totDim));
  CHKERRQ(CreateFEGeometry(dm, ds, cellIS, &affineQuad, &affineGeom, &quads, &geoms));
  CHKERRQ(PetscMalloc2(chunkSize*totDim, &u, chunkSize*totDim, &elemVec));
  /* Assumptions:
    - Single field
    - No input data
    - No auxiliary data
    - No time-dependence
  */
  for (i = 0; i < its; ++i) {
    for (cell = cStart; cell < cEnd; cell += chunkSize, ++Nch) {
      const PetscInt cS = cell, cE = PetscMin(cS + chunkSize, cEnd), Ne = cE - cS;

      CHKERRQ(PetscArrayzero(elemVec, chunkSize*totDim));
      /* TODO Replace with DMPlexGetCellFields() */
      for (k = 0; k < chunkSize*totDim; ++k) u[k] = 1.0;
      for (f = 0; f < Nf; ++f) {
        PetscFormKey key;
        PetscFEGeom     *geom = affineGeom ? affineGeom : geoms[f];
        /* PetscQuadrature quad = affineQuad ? affineQuad : quads[f]; */

        key.label = NULL; key.value = 0; key.field = f;
        CHKERRQ(PetscFEGeomGetChunk(geom, cS, cE, &chunkGeom));
        CHKERRQ(PetscLogEventBegin(event,0,0,0,0));
        CHKERRQ(PetscFEIntegrateResidual(ds, key, Ne, chunkGeom, u, NULL, NULL, NULL, 0.0, elemVec));
        CHKERRQ(PetscLogEventEnd(event,0,0,0,0));
      }
    }
  }
  CHKERRQ(PetscFEGeomRestoreChunk(affineGeom, cStart, cEnd, &chunkGeom));
  CHKERRQ(DestroyFEGeometry(dm, ds, cellIS, &affineQuad, &affineGeom, &quads, &geoms));
  CHKERRQ(ISDestroy(&cellIS));
  CHKERRQ(PetscFree2(u, elemVec));
  CHKERRQ(PetscLogStagePop());
#if defined(PETSC_USE_LOG)
  {
    const char        *title = "Petsc FE Residual Integration";
    PetscEventPerfInfo eventInfo;
    PetscInt           N = (cEnd - cStart)*Nf*its;
    PetscReal          flopRate, cellRate;

    CHKERRQ(PetscLogEventGetPerfInfo(stage, event, &eventInfo));
    flopRate = eventInfo.time != 0.0 ? eventInfo.flops/eventInfo.time : 0.0;
    cellRate = eventInfo.time != 0.0 ? N/eventInfo.time : 0.0;
    CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject) dm), "%s: %D integrals %D chunks %D reps\n  Cell rate: %.2f/s flop rate: %.2f MF/s\n", title, N, Nch, its, (double)cellRate, (double)(flopRate/1.e6)));
  }
#endif
  PetscFunctionReturn(0);
}

static PetscErrorCode TestIntegration2(DM dm, PetscInt cbs, PetscInt its)
{
  Vec             X, F;
#if defined(PETSC_USE_LOG)
  PetscLogStage   stage;
#endif
  PetscInt        i;

  PetscFunctionBeginUser;
  CHKERRQ(PetscLogStageRegister("DMPlex Residual Integration Test", &stage));
  CHKERRQ(PetscLogStagePush(stage));
  CHKERRQ(DMGetLocalVector(dm, &X));
  CHKERRQ(DMGetLocalVector(dm, &F));
  for (i = 0; i < its; ++i) {
    CHKERRQ(DMPlexSNESComputeResidualFEM(dm, X, F, NULL));
  }
  CHKERRQ(DMRestoreLocalVector(dm, &X));
  CHKERRQ(DMRestoreLocalVector(dm, &F));
  CHKERRQ(PetscLogStagePop());
#if defined(PETSC_USE_LOG)
  {
    const char         *title = "DMPlex Residual Integration";
    PetscEventPerfInfo eventInfo;
    PetscReal          flopRate, cellRate;
    PetscInt           cStart, cEnd, Nf, N;
    PetscLogEvent      event;

    CHKERRQ(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
    CHKERRQ(DMGetNumFields(dm, &Nf));
    CHKERRQ(PetscLogEventGetId("DMPlexResidualFE", &event));
    CHKERRQ(PetscLogEventGetPerfInfo(stage, event, &eventInfo));
    N        = (cEnd - cStart)*Nf*eventInfo.count;
    flopRate = eventInfo.time != 0.0 ? eventInfo.flops/eventInfo.time : 0.0;
    cellRate = eventInfo.time != 0.0 ? N/eventInfo.time : 0.0;
    CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject) dm), "%s: %D integrals %D reps\n  Cell rate: %.2f/s flop rate: %.2f MF/s\n", title, N, eventInfo.count, (double)cellRate, (double)(flopRate/1.e6)));
  }
#endif
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  AppCtx         ctx;
  PetscMPIInt    size;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help); if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheckFalse(size > 1,PETSC_COMM_WORLD, PETSC_ERR_SUP, "This is a uniprocessor example only.");
  CHKERRQ(ProcessOptions(PETSC_COMM_WORLD, &ctx));
  CHKERRQ(PetscLogDefaultBegin());
  CHKERRQ(DMCreate(PETSC_COMM_WORLD, &dm));
  CHKERRQ(DMSetType(dm, DMPLEX));
  CHKERRQ(DMSetFromOptions(dm));
  CHKERRQ(PetscObjectSetName((PetscObject) dm, "Mesh"));
  CHKERRQ(PetscObjectViewFromOptions((PetscObject) dm, NULL, "-dm_view"));
  CHKERRQ(SetupDiscretization(dm, "potential", SetupPrimalProblem, &ctx));
  CHKERRQ(TestIntegration(dm, ctx.cbs, ctx.its));
  CHKERRQ(TestIntegration2(dm, ctx.cbs, ctx.its));
  CHKERRQ(DMDestroy(&dm));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST
  test:
    suffix: 0
    requires: triangle
    args: -dm_view

  test:
    suffix: 1
    requires: triangle
    args: -dm_view -potential_petscspace_degree 1

  test:
    suffix: 2
    requires: triangle
    args: -dm_view -potential_petscspace_degree 2

  test:
    suffix: 3
    requires: triangle
    args: -dm_view -potential_petscspace_degree 3
TEST*/
