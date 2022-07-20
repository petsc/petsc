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

  PetscFunctionBeginUser;
  options->dim     = 2;
  options->simplex = PETSC_TRUE;
  options->its     = 1;
  options->cbs     = 8;

  PetscOptionsBegin(comm, "", "FE Integration Performance Options", "PETSCFE");
  PetscCall(PetscOptionsInt("-dim", "The topological dimension", "ex1.c", options->dim, &options->dim, NULL));
  PetscCall(PetscOptionsBool("-simplex", "Simplex or hex cells", "ex1.c", options->simplex, &options->simplex, NULL));
  PetscCall(PetscOptionsInt("-its", "The number of replications for timing", "ex1.c", options->its, &options->its, NULL));
  PetscCall(PetscOptionsInt("-cbs", "The number of cells in an integration block", "ex1.c", options->cbs, &options->cbs, NULL));
  PetscOptionsEnd();
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
  PetscCall(DMGetDS(dm, &prob));
  PetscCall(PetscDSSetResidual(prob, 0, f0_trig_u, f1_u));
  PetscCall(PetscDSSetJacobian(prob, 0, 0, NULL, NULL, NULL, g3_uu));
  PetscCall(PetscDSSetExactSolution(prob, 0, trig_u, user));
  PetscCall(DMGetLabel(dm, "marker", &label));
  PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0, NULL, (void (*)(void)) trig_u, NULL, user, NULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDiscretization(DM dm, const char name[], PetscErrorCode (*setup)(DM, AppCtx *), AppCtx *user)
{
  DM             cdm = dm;
  PetscFE        fe;
  char           prefix[PETSC_MAX_PATH_LEN];

  PetscFunctionBeginUser;
  /* Create finite element */
  PetscCall(PetscSNPrintf(prefix, PETSC_MAX_PATH_LEN, "%s_", name));
  PetscCall(PetscFECreateDefault(PetscObjectComm((PetscObject) dm), user->dim, 1, user->simplex, name ? prefix : NULL, -1, &fe));
  PetscCall(PetscObjectSetName((PetscObject) fe, name));
  /* Set discretization and boundary conditions for each mesh */
  PetscCall(DMSetField(dm, 0, NULL, (PetscObject) fe));
  PetscCall(DMCreateDS(dm));
  PetscCall((*setup)(dm, user));
  while (cdm) {
    PetscCall(DMCopyDisc(dm,cdm));
    /* TODO: Check whether the boundary of coarse meshes is marked */
    PetscCall(DMGetCoarseDM(cdm, &cdm));
  }
  PetscCall(PetscFEDestroy(&fe));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscContainerUserDestroy_PetscFEGeom(void *ctx)
{
  PetscFEGeom   *geom = (PetscFEGeom *) ctx;

  PetscFunctionBegin;
  PetscCall(PetscFEGeomDestroy(&geom));
  PetscFunctionReturn(0);
}

PetscErrorCode CellRangeGetFEGeom(IS cellIS, DMField coordField, PetscQuadrature quad, PetscBool faceData, PetscFEGeom **geom)
{
  char            composeStr[33] = {0};
  PetscObjectId   id;
  PetscContainer  container;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetId((PetscObject) quad, &id));
  PetscCall(PetscSNPrintf(composeStr, 32, "CellRangeGetFEGeom_%" PetscInt64_FMT "\n", id));
  PetscCall(PetscObjectQuery((PetscObject) cellIS, composeStr, (PetscObject *) &container));
  if (container) {
    PetscCall(PetscContainerGetPointer(container, (void **) geom));
  } else {
    PetscCall(DMFieldCreateFEGeom(coordField, cellIS, quad, faceData, geom));
    PetscCall(PetscContainerCreate(PETSC_COMM_SELF, &container));
    PetscCall(PetscContainerSetPointer(container, (void *) *geom));
    PetscCall(PetscContainerSetUserDestroy(container, PetscContainerUserDestroy_PetscFEGeom));
    PetscCall(PetscObjectCompose((PetscObject) cellIS, composeStr, (PetscObject) container));
    PetscCall(PetscContainerDestroy(&container));
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
  PetscCall(PetscDSGetNumFields(ds, &Nf));
  PetscCall(DMGetCoordinateField(dm, &coordField));
  PetscCall(DMFieldGetDegree(coordField, cellIS, NULL, &maxDegree));
  if (maxDegree <= 1) {
    PetscCall(DMFieldCreateDefaultQuadrature(coordField, cellIS, affineQuad));
    if (*affineQuad) PetscCall(CellRangeGetFEGeom(cellIS, coordField, *affineQuad, PETSC_FALSE, affineGeom));
  } else {
    PetscCall(PetscCalloc2(Nf, quads, Nf, geoms));
    for (f = 0; f < Nf; ++f) {
      PetscFE fe;

      PetscCall(PetscDSGetDiscretization(ds, f, (PetscObject *) &fe));
      PetscCall(PetscFEGetQuadrature(fe, &(*quads)[f]));
      PetscCall(PetscObjectReference((PetscObject) (*quads)[f]));
      PetscCall(CellRangeGetFEGeom(cellIS, coordField, (*quads)[f], PETSC_FALSE, &(*geoms)[f]));
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DestroyFEGeometry(DM dm, PetscDS ds, IS cellIS, PetscQuadrature *affineQuad, PetscFEGeom **affineGeom, PetscQuadrature **quads, PetscFEGeom ***geoms)
{
  DMField        coordField;
  PetscInt       Nf, f;

  PetscFunctionBeginUser;
  PetscCall(PetscDSGetNumFields(ds, &Nf));
  PetscCall(DMGetCoordinateField(dm, &coordField));
  if (*affineQuad) {
    PetscCall(CellRangeRestoreFEGeom(cellIS, coordField, *affineQuad, PETSC_FALSE, affineGeom));
    PetscCall(PetscQuadratureDestroy(affineQuad));
  } else {
    for (f = 0; f < Nf; ++f) {
      PetscCall(CellRangeRestoreFEGeom(cellIS, coordField, (*quads)[f], PETSC_FALSE, &(*geoms)[f]));
      PetscCall(PetscQuadratureDestroy(&(*quads)[f]));
    }
    PetscCall(PetscFree2(*quads, *geoms));
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
  PetscCall(PetscLogStageRegister("PetscFE Residual Integration Test", &stage));
  PetscCall(PetscLogEventRegister("FEIntegRes", PETSCFE_CLASSID, &event));
  PetscCall(PetscLogStagePush(stage));
  PetscCall(DMPlexGetDepth(dm, &depth));
  PetscCall(DMGetStratumIS(dm, "depth", depth, &cellIS));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMGetCellDS(dm, cStart, &ds));
  PetscCall(PetscDSGetNumFields(ds, &Nf));
  PetscCall(PetscDSGetTotalDimension(ds, &totDim));
  PetscCall(CreateFEGeometry(dm, ds, cellIS, &affineQuad, &affineGeom, &quads, &geoms));
  PetscCall(PetscMalloc2(chunkSize*totDim, &u, chunkSize*totDim, &elemVec));
  /* Assumptions:
    - Single field
    - No input data
    - No auxiliary data
    - No time-dependence
  */
  for (i = 0; i < its; ++i) {
    for (cell = cStart; cell < cEnd; cell += chunkSize, ++Nch) {
      const PetscInt cS = cell, cE = PetscMin(cS + chunkSize, cEnd), Ne = cE - cS;

      PetscCall(PetscArrayzero(elemVec, chunkSize*totDim));
      /* TODO Replace with DMPlexGetCellFields() */
      for (k = 0; k < chunkSize*totDim; ++k) u[k] = 1.0;
      for (f = 0; f < Nf; ++f) {
        PetscFormKey key;
        PetscFEGeom     *geom = affineGeom ? affineGeom : geoms[f];
        /* PetscQuadrature quad = affineQuad ? affineQuad : quads[f]; */

        key.label = NULL; key.value = 0; key.field = f; key.part = 0;
        PetscCall(PetscFEGeomGetChunk(geom, cS, cE, &chunkGeom));
        PetscCall(PetscLogEventBegin(event,0,0,0,0));
        PetscCall(PetscFEIntegrateResidual(ds, key, Ne, chunkGeom, u, NULL, NULL, NULL, 0.0, elemVec));
        PetscCall(PetscLogEventEnd(event,0,0,0,0));
      }
    }
  }
  PetscCall(PetscFEGeomRestoreChunk(affineGeom, cStart, cEnd, &chunkGeom));
  PetscCall(DestroyFEGeometry(dm, ds, cellIS, &affineQuad, &affineGeom, &quads, &geoms));
  PetscCall(ISDestroy(&cellIS));
  PetscCall(PetscFree2(u, elemVec));
  PetscCall(PetscLogStagePop());
#if defined(PETSC_USE_LOG)
  {
    const char        *title = "Petsc FE Residual Integration";
    PetscEventPerfInfo eventInfo;
    PetscInt           N = (cEnd - cStart)*Nf*its;
    PetscReal          flopRate, cellRate;

    PetscCall(PetscLogEventGetPerfInfo(stage, event, &eventInfo));
    flopRate = eventInfo.time != 0.0 ? eventInfo.flops/eventInfo.time : 0.0;
    cellRate = eventInfo.time != 0.0 ? N/eventInfo.time : 0.0;
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject) dm), "%s: %" PetscInt_FMT " integrals %" PetscInt_FMT " chunks %" PetscInt_FMT " reps\n  Cell rate: %.2f/s flop rate: %.2f MF/s\n", title, N, Nch, its, (double)cellRate, (double)(flopRate/1.e6)));
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
  PetscCall(PetscLogStageRegister("DMPlex Residual Integration Test", &stage));
  PetscCall(PetscLogStagePush(stage));
  PetscCall(DMGetLocalVector(dm, &X));
  PetscCall(DMGetLocalVector(dm, &F));
  for (i = 0; i < its; ++i) {
    PetscCall(DMPlexSNESComputeResidualFEM(dm, X, F, NULL));
  }
  PetscCall(DMRestoreLocalVector(dm, &X));
  PetscCall(DMRestoreLocalVector(dm, &F));
  PetscCall(PetscLogStagePop());
#if defined(PETSC_USE_LOG)
  {
    const char         *title = "DMPlex Residual Integration";
    PetscEventPerfInfo eventInfo;
    PetscReal          flopRate, cellRate;
    PetscInt           cStart, cEnd, Nf, N;
    PetscLogEvent      event;

    PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
    PetscCall(DMGetNumFields(dm, &Nf));
    PetscCall(PetscLogEventGetId("DMPlexResidualFE", &event));
    PetscCall(PetscLogEventGetPerfInfo(stage, event, &eventInfo));
    N        = (cEnd - cStart)*Nf*eventInfo.count;
    flopRate = eventInfo.time != 0.0 ? eventInfo.flops/eventInfo.time : 0.0;
    cellRate = eventInfo.time != 0.0 ? N/eventInfo.time : 0.0;
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject) dm), "%s: %" PetscInt_FMT " integrals %d reps\n  Cell rate: %.2f/s flop rate: %.2f MF/s\n", title, N, eventInfo.count, (double)cellRate, (double)(flopRate/1.e6)));
  }
#endif
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  AppCtx         ctx;
  PetscMPIInt    size;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size <= 1,PETSC_COMM_WORLD, PETSC_ERR_SUP, "This is a uniprocessor example only.");
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &ctx));
  PetscCall(PetscLogDefaultBegin());
  PetscCall(DMCreate(PETSC_COMM_WORLD, &dm));
  PetscCall(DMSetType(dm, DMPLEX));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(PetscObjectSetName((PetscObject) dm, "Mesh"));
  PetscCall(PetscObjectViewFromOptions((PetscObject) dm, NULL, "-dm_view"));
  PetscCall(SetupDiscretization(dm, "potential", SetupPrimalProblem, &ctx));
  PetscCall(TestIntegration(dm, ctx.cbs, ctx.its));
  PetscCall(TestIntegration2(dm, ctx.cbs, ctx.its));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
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
