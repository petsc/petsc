static const char help[] = "Tests for injecting basis functions";

#include <petscdmplex.h>
#include <petscfe.h>
#include <petscds.h>

typedef struct {
  PetscInt its; /* Number of replications for timing */
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->its = 1;

  ierr = PetscOptionsBegin(comm, "", "FE Injection Options", "PETSCFE");CHKERRQ(ierr);
  CHKERRQ(PetscOptionsInt("-its", "The number of replications for timing", "ex1.c", options->its, &options->its, NULL));
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
  PetscDS        ds;
  DMLabel        label;
  const PetscInt id = 1;

  PetscFunctionBeginUser;
  CHKERRQ(DMGetDS(dm, &ds));
  CHKERRQ(PetscDSSetResidual(ds, 0, f0_trig_u, f1_u));
  CHKERRQ(PetscDSSetJacobian(ds, 0, 0, NULL, NULL, NULL, g3_uu));
  CHKERRQ(PetscDSSetExactSolution(ds, 0, trig_u, user));
  CHKERRQ(DMGetLabel(dm, "marker", &label));
  if (label) CHKERRQ(DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0, NULL, (void (*)(void)) trig_u, NULL, user, NULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDiscretization(DM dm, const char name[], PetscErrorCode (*setup)(DM, AppCtx *), AppCtx *user)
{
  DM             cdm = dm;
  PetscFE        fe;
  char           prefix[PETSC_MAX_PATH_LEN];
  PetscInt       dim;

  PetscFunctionBeginUser;
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(PetscSNPrintf(prefix, PETSC_MAX_PATH_LEN, "%s_", name));
  CHKERRQ(DMCreateFEDefault(dm, dim, name ? prefix : NULL, -1, &fe));
  CHKERRQ(PetscObjectSetName((PetscObject) fe, name));
  /* Set discretization and boundary conditions for each mesh */
  CHKERRQ(DMSetField(dm, 0, NULL, (PetscObject) fe));
  CHKERRQ(DMCreateDS(dm));
  CHKERRQ((*setup)(dm, user));
  while (cdm) {
    CHKERRQ(DMCopyDisc(dm,cdm));
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

static PetscErrorCode TestEvaluation(DM dm)
{
  PetscFE        fe;
  PetscSpace     sp;
  PetscReal     *points;
  PetscReal     *B, *D, *H;
  PetscInt       dim, Nb, b, Nc, c, Np, p;

  PetscFunctionBeginUser;
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMGetField(dm, 0, NULL, (PetscObject *) &fe));
  Np = 6;
  CHKERRQ(PetscMalloc1(Np*dim, &points));
  if (dim == 3) {
    points[0]  = -1.0; points[1]  = -1.0; points[2]  = -1.0;
    points[3]  =  1.0; points[4]  = -1.0; points[5]  = -1.0;
    points[6]  = -1.0; points[7]  =  1.0; points[8]  = -1.0;
    points[9]  = -1.0; points[10] = -1.0; points[11] =  1.0;
    points[12] =  1.0; points[13] = -1.0; points[14] =  1.0;
    points[15] = -1.0; points[16] =  1.0; points[17] =  1.0;
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Only for 3D right now");
  CHKERRQ(PetscFEGetBasisSpace(fe, &sp));
  CHKERRQ(PetscSpaceGetDimension(sp, &Nb));
  CHKERRQ(PetscSpaceGetNumComponents(sp, &Nc));
  CHKERRQ(DMGetWorkArray(dm, Np*Nb*Nc, MPIU_REAL, &B));
  CHKERRQ(DMGetWorkArray(dm, Np*Nb*Nc*dim, MPIU_REAL, &D));
  CHKERRQ(DMGetWorkArray(dm, Np*Nb*Nc*dim*dim, MPIU_REAL, &H));
  CHKERRQ(PetscSpaceEvaluate(sp, Np, points, B, NULL, NULL /*D, H*/));
  for (p = 0; p < Np; ++p) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "Point %" PetscInt_FMT "\n", p));
    for (b = 0; b < Nb; ++b) {
      CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "B[%" PetscInt_FMT "]:", b));
      for (c = 0; c < Nc; ++c) CHKERRQ(PetscPrintf(PETSC_COMM_SELF, " %g", (double) B[(p*Nb+b)*Nc+c]));
      CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "\n"));
#if 0
      for (c = 0; c < Nc; ++c) {
        CHKERRQ(PetscPrintf(PETSC_COMM_SELF, " D[%" PetscInt_FMT ",%" PetscInt_FMT "]:", b, c));
        for (d = 0; d < dim; ++d) CHKERRQ(PetscPrintf(PETSC_COMM_SELF, " %g", (double) B[((p*Nb+b)*Nc+c)*dim+d)]));
        CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "\n"));
      }
#endif
    }
  }
  CHKERRQ(DMRestoreWorkArray(dm, Np*Nb, MPIU_REAL, &B));
  CHKERRQ(DMRestoreWorkArray(dm, Np*Nb*dim, MPIU_REAL, &D));
  CHKERRQ(DMRestoreWorkArray(dm, Np*Nb*dim*dim, MPIU_REAL, &H));
  CHKERRQ(PetscFree(points));
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
        CHKERRQ(PetscFEIntegrateResidual(ds, key, Ne, chunkGeom, u, NULL, NULL, NULL, 0.0, elemVec));
      }
    }
  }
  CHKERRQ(PetscFEGeomRestoreChunk(affineGeom, cStart, cEnd, &chunkGeom));
  CHKERRQ(DestroyFEGeometry(dm, ds, cellIS, &affineQuad, &affineGeom, &quads, &geoms));
  CHKERRQ(ISDestroy(&cellIS));
  CHKERRQ(PetscFree2(u, elemVec));
  PetscFunctionReturn(0);
}

static PetscErrorCode TestUnisolvence(DM dm)
{
  Mat M;
  Vec v;

  PetscFunctionBeginUser;
  CHKERRQ(DMGetLocalVector(dm, &v));
  CHKERRQ(DMRestoreLocalVector(dm, &v));
  CHKERRQ(DMCreateMassMatrix(dm, dm, &M));
  CHKERRQ(MatViewFromOptions(M, NULL, "-mass_view"));
  CHKERRQ(MatDestroy(&M));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  AppCtx         ctx;
  PetscMPIInt    size;

  CHKERRQ(PetscInitialize(&argc, &argv, NULL, help));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_SUP, "This is a uniprocessor example only.");
  CHKERRQ(ProcessOptions(PETSC_COMM_WORLD, &ctx));
  CHKERRQ(DMCreate(PETSC_COMM_WORLD, &dm));
  CHKERRQ(DMSetType(dm, DMPLEX));
  CHKERRQ(DMSetFromOptions(dm));
  CHKERRQ(PetscObjectSetName((PetscObject) dm, "Mesh"));
  CHKERRQ(PetscObjectViewFromOptions((PetscObject) dm, NULL, "-dm_view"));
  CHKERRQ(SetupDiscretization(dm, "field", SetupPrimalProblem, &ctx));
  CHKERRQ(TestEvaluation(dm));
  CHKERRQ(TestIntegration(dm, 1, ctx.its));
  CHKERRQ(TestUnisolvence(dm));
  CHKERRQ(DMDestroy(&dm));
  CHKERRQ(PetscFinalize());
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
