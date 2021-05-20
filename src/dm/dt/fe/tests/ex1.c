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
  ierr = PetscOptionsInt("-dim", "The topological dimension", "ex1.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-simplex", "Simplex or hex cells", "ex1.c", options->simplex, &options->simplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-its", "The number of replications for timing", "ex1.c", options->its, &options->its, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-cbs", "The number of cells in an integration block", "ex1.c", options->cbs, &options->cbs, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
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
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSSetResidual(prob, 0, f0_trig_u, f1_u);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob, 0, 0, NULL, NULL, NULL, g3_uu);CHKERRQ(ierr);
  ierr = PetscDSSetExactSolution(prob, 0, trig_u, user);CHKERRQ(ierr);
  ierr = DMGetLabel(dm, "marker", &label);CHKERRQ(ierr);
  ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0, NULL, (void (*)(void)) trig_u, NULL, user, NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDiscretization(DM dm, const char name[], PetscErrorCode (*setup)(DM, AppCtx *), AppCtx *user)
{
  DM             cdm = dm;
  PetscFE        fe;
  char           prefix[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  /* Create finite element */
  ierr = PetscSNPrintf(prefix, PETSC_MAX_PATH_LEN, "%s_", name);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(PetscObjectComm((PetscObject) dm), user->dim, 1, user->simplex, name ? prefix : NULL, -1, &fe);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe, name);CHKERRQ(ierr);
  /* Set discretization and boundary conditions for each mesh */
  ierr = DMSetField(dm, 0, NULL, (PetscObject) fe);CHKERRQ(ierr);
  ierr = DMCreateDS(dm);CHKERRQ(ierr);
  ierr = (*setup)(dm, user);CHKERRQ(ierr);
  while (cdm) {
    ierr = DMCopyDisc(dm,cdm);CHKERRQ(ierr);
    /* TODO: Check whether the boundary of coarse meshes is marked */
    ierr = DMGetCoarseDM(cdm, &cdm);CHKERRQ(ierr);
  }
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscContainerUserDestroy_PetscFEGeom(void *ctx)
{
  PetscFEGeom   *geom = (PetscFEGeom *) ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFEGeomDestroy(&geom);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode CellRangeGetFEGeom(IS cellIS, DMField coordField, PetscQuadrature quad, PetscBool faceData, PetscFEGeom **geom)
{
  char            composeStr[33] = {0};
  PetscObjectId   id;
  PetscContainer  container;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetId((PetscObject) quad, &id);CHKERRQ(ierr);
  ierr = PetscSNPrintf(composeStr, 32, "CellRangeGetFEGeom_%x\n", id);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) cellIS, composeStr, (PetscObject *) &container);CHKERRQ(ierr);
  if (container) {
    ierr = PetscContainerGetPointer(container, (void **) geom);CHKERRQ(ierr);
  } else {
    ierr = DMFieldCreateFEGeom(coordField, cellIS, quad, faceData, geom);CHKERRQ(ierr);
    ierr = PetscContainerCreate(PETSC_COMM_SELF, &container);CHKERRQ(ierr);
    ierr = PetscContainerSetPointer(container, (void *) *geom);CHKERRQ(ierr);
    ierr = PetscContainerSetUserDestroy(container, PetscContainerUserDestroy_PetscFEGeom);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject) cellIS, composeStr, (PetscObject) container);CHKERRQ(ierr);
    ierr = PetscContainerDestroy(&container);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  *affineQuad = NULL;
  *affineGeom = NULL;
  *quads      = NULL;
  *geoms      = NULL;
  ierr = PetscDSGetNumFields(ds, &Nf);CHKERRQ(ierr);
  ierr = DMGetCoordinateField(dm, &coordField);CHKERRQ(ierr);
  ierr = DMFieldGetDegree(coordField, cellIS, NULL, &maxDegree);CHKERRQ(ierr);
  if (maxDegree <= 1) {
    ierr = DMFieldCreateDefaultQuadrature(coordField, cellIS, affineQuad);CHKERRQ(ierr);
    if (*affineQuad) {ierr = CellRangeGetFEGeom(cellIS, coordField, *affineQuad, PETSC_FALSE, affineGeom);CHKERRQ(ierr);}
  } else {
    ierr = PetscCalloc2(Nf, quads, Nf, geoms);CHKERRQ(ierr);
    for (f = 0; f < Nf; ++f) {
      PetscFE fe;

      ierr = PetscDSGetDiscretization(ds, f, (PetscObject *) &fe);CHKERRQ(ierr);
      ierr = PetscFEGetQuadrature(fe, &(*quads)[f]);CHKERRQ(ierr);
      ierr = PetscObjectReference((PetscObject) (*quads)[f]);CHKERRQ(ierr);
      ierr = CellRangeGetFEGeom(cellIS, coordField, (*quads)[f], PETSC_FALSE, &(*geoms)[f]);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DestroyFEGeometry(DM dm, PetscDS ds, IS cellIS, PetscQuadrature *affineQuad, PetscFEGeom **affineGeom, PetscQuadrature **quads, PetscFEGeom ***geoms)
{
  DMField        coordField;
  PetscInt       Nf, f;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscDSGetNumFields(ds, &Nf);CHKERRQ(ierr);
  ierr = DMGetCoordinateField(dm, &coordField);CHKERRQ(ierr);
  if (*affineQuad) {
    ierr = CellRangeRestoreFEGeom(cellIS, coordField, *affineQuad, PETSC_FALSE, affineGeom);CHKERRQ(ierr);
    ierr = PetscQuadratureDestroy(affineQuad);CHKERRQ(ierr);
  } else {
    for (f = 0; f < Nf; ++f) {
      ierr = CellRangeRestoreFEGeom(cellIS, coordField, (*quads)[f], PETSC_FALSE, &(*geoms)[f]);CHKERRQ(ierr);
      ierr = PetscQuadratureDestroy(&(*quads)[f]);CHKERRQ(ierr);
    }
    ierr = PetscFree2(*quads, *geoms);CHKERRQ(ierr);
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
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  ierr = PetscLogStageRegister("PetscFE Residual Integration Test", &stage);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("FEIntegRes", PETSCFE_CLASSID, &event);CHKERRQ(ierr);
  ierr = PetscLogStagePush(stage);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = DMGetStratumIS(dm, "depth", depth, &cellIS);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMGetCellDS(dm, cStart, &ds);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(ds, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(ds, &totDim);CHKERRQ(ierr);
  ierr = CreateFEGeometry(dm, ds, cellIS, &affineQuad, &affineGeom, &quads, &geoms);CHKERRQ(ierr);
  ierr = PetscMalloc2(chunkSize*totDim, &u, chunkSize*totDim, &elemVec);CHKERRQ(ierr);
  /* Assumptions:
    - Single field
    - No input data
    - No auxiliary data
    - No time-dependence
  */
  for (i = 0; i < its; ++i) {
    for (cell = cStart; cell < cEnd; cell += chunkSize, ++Nch) {
      const PetscInt cS = cell, cE = PetscMin(cS + chunkSize, cEnd), Ne = cE - cS;

      ierr = PetscArrayzero(elemVec, chunkSize*totDim);CHKERRQ(ierr);
      /* TODO Replace with DMPlexGetCellFields() */
      for (k = 0; k < chunkSize*totDim; ++k) u[k] = 1.0;
      for (f = 0; f < Nf; ++f) {
        PetscFormKey key;
        PetscFEGeom     *geom = affineGeom ? affineGeom : geoms[f];
        /* PetscQuadrature quad = affineQuad ? affineQuad : quads[f]; */

        key.label = NULL; key.value = 0; key.field = f;
        ierr = PetscFEGeomGetChunk(geom, cS, cE, &chunkGeom);CHKERRQ(ierr);
        ierr = PetscLogEventBegin(event,0,0,0,0);CHKERRQ(ierr);
        ierr = PetscFEIntegrateResidual(ds, key, Ne, chunkGeom, u, NULL, NULL, NULL, 0.0, elemVec);CHKERRQ(ierr);
        ierr = PetscLogEventEnd(event,0,0,0,0);CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscFEGeomRestoreChunk(affineGeom, cStart, cEnd, &chunkGeom);CHKERRQ(ierr);
  ierr = DestroyFEGeometry(dm, ds, cellIS, &affineQuad, &affineGeom, &quads, &geoms);CHKERRQ(ierr);
  ierr = ISDestroy(&cellIS);CHKERRQ(ierr);
  ierr = PetscFree2(u, elemVec);CHKERRQ(ierr);
  ierr = PetscLogStagePop();CHKERRQ(ierr);
#if defined(PETSC_USE_LOG)
  {
    const char        *title = "Petsc FE Residual Integration";
    PetscEventPerfInfo eventInfo;
    PetscInt           N = (cEnd - cStart)*Nf*its;
    PetscReal          flopRate, cellRate;

    ierr = PetscLogEventGetPerfInfo(stage, event, &eventInfo);CHKERRQ(ierr);
    flopRate = eventInfo.time != 0.0 ? eventInfo.flops/eventInfo.time : 0.0;
    cellRate = eventInfo.time != 0.0 ? N/eventInfo.time : 0.0;
    ierr = PetscPrintf(PetscObjectComm((PetscObject) dm), "%s: %D integrals %D chunks %D reps\n  Cell rate: %.2f/s flop rate: %.2f MF/s\n", title, N, Nch, its, (double)cellRate, (double)(flopRate/1.e6));CHKERRQ(ierr);
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
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  ierr = PetscLogStageRegister("DMPlex Residual Integration Test", &stage);CHKERRQ(ierr);
  ierr = PetscLogStagePush(stage);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &X);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &F);CHKERRQ(ierr);
  for (i = 0; i < its; ++i) {
    ierr = DMPlexSNESComputeResidualFEM(dm, X, F, NULL);CHKERRQ(ierr);
  }
  ierr = DMRestoreLocalVector(dm, &X);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &F);CHKERRQ(ierr);
  ierr = PetscLogStagePop();CHKERRQ(ierr);
#if defined(PETSC_USE_LOG)
  {
    const char         *title = "DMPlex Residual Integration";
    PetscEventPerfInfo eventInfo;
    PetscReal          flopRate, cellRate;
    PetscInt           cStart, cEnd, Nf, N;
    PetscLogEvent      event;

    ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
    ierr = DMGetNumFields(dm, &Nf);CHKERRQ(ierr);
    ierr = PetscLogEventGetId("DMPlexResidualFE", &event);CHKERRQ(ierr);
    ierr = PetscLogEventGetPerfInfo(stage, event, &eventInfo);CHKERRQ(ierr);
    N        = (cEnd - cStart)*Nf*eventInfo.count;
    flopRate = eventInfo.time != 0.0 ? eventInfo.flops/eventInfo.time : 0.0;
    cellRate = eventInfo.time != 0.0 ? N/eventInfo.time : 0.0;
    ierr = PetscPrintf(PetscObjectComm((PetscObject) dm), "%s: %D integrals %D reps\n  Cell rate: %.2f/s flop rate: %.2f MF/s\n", title, N, eventInfo.count, (double)cellRate, (double)(flopRate/1.e6));CHKERRQ(ierr);
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
  ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);CHKERRMPI(ierr);
  if (size > 1) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "This is a uniprocessor example only.");
  ierr = ProcessOptions(PETSC_COMM_WORLD, &ctx);CHKERRQ(ierr);
  ierr = PetscLogDefaultBegin();CHKERRQ(ierr);
  ierr = DMCreate(PETSC_COMM_WORLD, &dm);CHKERRQ(ierr);
  ierr = DMSetType(dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) dm, "Mesh");CHKERRQ(ierr);
  ierr = PetscObjectViewFromOptions((PetscObject) dm, NULL, "-dm_view");CHKERRQ(ierr);
  ierr = SetupDiscretization(dm, "potential", SetupPrimalProblem, &ctx);CHKERRQ(ierr);
  ierr = TestIntegration(dm, ctx.cbs, ctx.its);CHKERRQ(ierr);
  ierr = TestIntegration2(dm, ctx.cbs, ctx.its);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
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
