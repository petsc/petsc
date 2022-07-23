static const char help[] = "Tests for injecting basis functions";

#include <petscdmplex.h>
#include <petscfe.h>
#include <petscds.h>

typedef struct {
  PetscInt its; /* Number of replications for timing */
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscFunctionBeginUser;
  options->its = 1;

  PetscOptionsBegin(comm, "", "FE Injection Options", "PETSCFE");
  PetscCall(PetscOptionsInt("-its", "The number of replications for timing", "ex1.c", options->its, &options->its, NULL));
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
  PetscDS        ds;
  DMLabel        label;
  const PetscInt id = 1;

  PetscFunctionBeginUser;
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSSetResidual(ds, 0, f0_trig_u, f1_u));
  PetscCall(PetscDSSetJacobian(ds, 0, 0, NULL, NULL, NULL, g3_uu));
  PetscCall(PetscDSSetExactSolution(ds, 0, trig_u, user));
  PetscCall(DMGetLabel(dm, "marker", &label));
  if (label) PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0, NULL, (void (*)(void)) trig_u, NULL, user, NULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDiscretization(DM dm, const char name[], PetscErrorCode (*setup)(DM, AppCtx *), AppCtx *user)
{
  DM             cdm = dm;
  PetscFE        fe;
  char           prefix[PETSC_MAX_PATH_LEN];
  PetscInt       dim;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(PetscSNPrintf(prefix, PETSC_MAX_PATH_LEN, "%s_", name));
  PetscCall(DMCreateFEDefault(dm, dim, name ? prefix : NULL, -1, &fe));
  PetscCall(PetscObjectSetName((PetscObject) fe, name));
  /* Set discretization and boundary conditions for each mesh */
  PetscCall(DMSetField(dm, 0, NULL, (PetscObject) fe));
  PetscCall(DMCreateDS(dm));
  PetscCall((*setup)(dm, user));
  while (cdm) {
    PetscCall(DMCopyDisc(dm,cdm));
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

static PetscErrorCode TestEvaluation(DM dm)
{
  PetscFE        fe;
  PetscSpace     sp;
  PetscReal     *points;
  PetscReal     *B, *D, *H;
  PetscInt       dim, Nb, b, Nc, c, Np, p;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetField(dm, 0, NULL, (PetscObject *) &fe));
  Np = 6;
  PetscCall(PetscMalloc1(Np*dim, &points));
  if (dim == 3) {
    points[0]  = -1.0; points[1]  = -1.0; points[2]  = -1.0;
    points[3]  =  1.0; points[4]  = -1.0; points[5]  = -1.0;
    points[6]  = -1.0; points[7]  =  1.0; points[8]  = -1.0;
    points[9]  = -1.0; points[10] = -1.0; points[11] =  1.0;
    points[12] =  1.0; points[13] = -1.0; points[14] =  1.0;
    points[15] = -1.0; points[16] =  1.0; points[17] =  1.0;
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Only for 3D right now");
  PetscCall(PetscFEGetBasisSpace(fe, &sp));
  PetscCall(PetscSpaceGetDimension(sp, &Nb));
  PetscCall(PetscSpaceGetNumComponents(sp, &Nc));
  PetscCall(DMGetWorkArray(dm, Np*Nb*Nc, MPIU_REAL, &B));
  PetscCall(DMGetWorkArray(dm, Np*Nb*Nc*dim, MPIU_REAL, &D));
  PetscCall(DMGetWorkArray(dm, Np*Nb*Nc*dim*dim, MPIU_REAL, &H));
  PetscCall(PetscSpaceEvaluate(sp, Np, points, B, NULL, NULL /*D, H*/));
  for (p = 0; p < Np; ++p) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "Point %" PetscInt_FMT "\n", p));
    for (b = 0; b < Nb; ++b) {
      PetscCall(PetscPrintf(PETSC_COMM_SELF, "B[%" PetscInt_FMT "]:", b));
      for (c = 0; c < Nc; ++c) PetscCall(PetscPrintf(PETSC_COMM_SELF, " %g", (double) B[(p*Nb+b)*Nc+c]));
      PetscCall(PetscPrintf(PETSC_COMM_SELF, "\n"));
#if 0
      for (c = 0; c < Nc; ++c) {
        PetscCall(PetscPrintf(PETSC_COMM_SELF, " D[%" PetscInt_FMT ",%" PetscInt_FMT "]:", b, c));
        for (d = 0; d < dim; ++d) PetscCall(PetscPrintf(PETSC_COMM_SELF, " %g", (double) B[((p*Nb+b)*Nc+c)*dim+d)]));
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "\n"));
      }
#endif
    }
  }
  PetscCall(DMRestoreWorkArray(dm, Np*Nb, MPIU_REAL, &B));
  PetscCall(DMRestoreWorkArray(dm, Np*Nb*dim, MPIU_REAL, &D));
  PetscCall(DMRestoreWorkArray(dm, Np*Nb*dim*dim, MPIU_REAL, &H));
  PetscCall(PetscFree(points));
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

  PetscFunctionBeginUser;
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
        PetscCall(PetscFEIntegrateResidual(ds, key, Ne, chunkGeom, u, NULL, NULL, NULL, 0.0, elemVec));
      }
    }
  }
  PetscCall(PetscFEGeomRestoreChunk(affineGeom, cStart, cEnd, &chunkGeom));
  PetscCall(DestroyFEGeometry(dm, ds, cellIS, &affineQuad, &affineGeom, &quads, &geoms));
  PetscCall(ISDestroy(&cellIS));
  PetscCall(PetscFree2(u, elemVec));
  PetscFunctionReturn(0);
}

static PetscErrorCode TestUnisolvence(DM dm)
{
  Mat M;
  Vec v;

  PetscFunctionBeginUser;
  PetscCall(DMGetLocalVector(dm, &v));
  PetscCall(DMRestoreLocalVector(dm, &v));
  PetscCall(DMCreateMassMatrix(dm, dm, &M));
  PetscCall(MatViewFromOptions(M, NULL, "-mass_view"));
  PetscCall(MatDestroy(&M));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  AppCtx         ctx;
  PetscMPIInt    size;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_SUP, "This is a uniprocessor example only.");
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &ctx));
  PetscCall(DMCreate(PETSC_COMM_WORLD, &dm));
  PetscCall(DMSetType(dm, DMPLEX));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(PetscObjectSetName((PetscObject) dm, "Mesh"));
  PetscCall(PetscObjectViewFromOptions((PetscObject) dm, NULL, "-dm_view"));
  PetscCall(SetupDiscretization(dm, "field", SetupPrimalProblem, &ctx));
  PetscCall(TestEvaluation(dm));
  PetscCall(TestIntegration(dm, 1, ctx.its));
  PetscCall(TestUnisolvence(dm));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
  test:
    suffix: 0
    args: -dm_plex_reference_cell_domain -dm_plex_cell triangular_prism -field_petscspace_degree 0

  test:
    suffix: 2
    args: -dm_plex_reference_cell_domain -dm_plex_cell triangular_prism \
          -field_petscspace_type sum \
          -field_petscspace_variables 3 \
          -field_petscspace_components 3 \
          -field_petscspace_sum_spaces 2 \
          -field_petscspace_sum_concatenate false \
          -field_sumcomp_0_petscspace_variables 3 \
          -field_sumcomp_0_petscspace_components 3 \
          -field_sumcomp_0_petscspace_degree 1 \
          -field_sumcomp_1_petscspace_variables 3 \
          -field_sumcomp_1_petscspace_components 3 \
          -field_sumcomp_1_petscspace_type wxy \
          -field_petscdualspace_form_degree 0 \
          -field_petscdualspace_order 1 \
          -field_petscdualspace_components 3

TEST*/
