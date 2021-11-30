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
  ierr = PetscOptionsInt("-its", "The number of replications for timing", "ex1.c", options->its, &options->its, NULL);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDS(dm, &ds);CHKERRQ(ierr);
  ierr = PetscDSSetResidual(ds, 0, f0_trig_u, f1_u);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(ds, 0, 0, NULL, NULL, NULL, g3_uu);CHKERRQ(ierr);
  ierr = PetscDSSetExactSolution(ds, 0, trig_u, user);CHKERRQ(ierr);
  ierr = DMGetLabel(dm, "marker", &label);CHKERRQ(ierr);
  if (label) {ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0, NULL, (void (*)(void)) trig_u, NULL, user, NULL);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDiscretization(DM dm, const char name[], PetscErrorCode (*setup)(DM, AppCtx *), AppCtx *user)
{
  DM             cdm = dm;
  PetscFE        fe;
  char           prefix[PETSC_MAX_PATH_LEN];
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = PetscSNPrintf(prefix, PETSC_MAX_PATH_LEN, "%s_", name);CHKERRQ(ierr);
  ierr = DMCreateFEDefault(dm, dim, name ? prefix : NULL, -1, &fe);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe, name);CHKERRQ(ierr);
  /* Set discretization and boundary conditions for each mesh */
  ierr = DMSetField(dm, 0, NULL, (PetscObject) fe);CHKERRQ(ierr);
  ierr = DMCreateDS(dm);CHKERRQ(ierr);
  ierr = (*setup)(dm, user);CHKERRQ(ierr);
  while (cdm) {
    ierr = DMCopyDisc(dm,cdm);CHKERRQ(ierr);
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

static PetscErrorCode TestEvaluation(DM dm)
{
  PetscFE        fe;
  PetscSpace     sp;
  PetscReal     *points;
  PetscReal     *B, *D, *H;
  PetscInt       dim, Nb, b, Nc, c, Np, p;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetField(dm, 0, NULL, (PetscObject *) &fe);CHKERRQ(ierr);
  Np = 6;
  ierr = PetscMalloc1(Np*dim, &points);CHKERRQ(ierr);
  if (dim == 3) {
    points[0]  = -1.0; points[1]  = -1.0; points[2]  = -1.0;
    points[3]  =  1.0; points[4]  = -1.0; points[5]  = -1.0;
    points[6]  = -1.0; points[7]  =  1.0; points[8]  = -1.0;
    points[9]  = -1.0; points[10] = -1.0; points[11] =  1.0;
    points[12] =  1.0; points[13] = -1.0; points[14] =  1.0;
    points[15] = -1.0; points[16] =  1.0; points[17] =  1.0;
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Only for 3D right now");
  ierr = PetscFEGetBasisSpace(fe, &sp);CHKERRQ(ierr);
  ierr = PetscSpaceGetDimension(sp, &Nb);CHKERRQ(ierr);
  ierr = PetscSpaceGetNumComponents(sp, &Nc);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm, Np*Nb*Nc, MPIU_REAL, &B);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm, Np*Nb*Nc*dim, MPIU_REAL, &D);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm, Np*Nb*Nc*dim*dim, MPIU_REAL, &H);CHKERRQ(ierr);
  ierr = PetscSpaceEvaluate(sp, Np, points, B, NULL, NULL /*D, H*/);CHKERRQ(ierr);
  for (p = 0; p < Np; ++p) {
    ierr = PetscPrintf(PETSC_COMM_SELF, "Point %" PetscInt_FMT "\n", p);CHKERRQ(ierr);
    for (b = 0; b < Nb; ++b) {
      ierr = PetscPrintf(PETSC_COMM_SELF, "B[%" PetscInt_FMT "]:", b);CHKERRQ(ierr);
      for (c = 0; c < Nc; ++c) {ierr = PetscPrintf(PETSC_COMM_SELF, " %g", (double) B[(p*Nb+b)*Nc+c]);CHKERRQ(ierr);}
      ierr = PetscPrintf(PETSC_COMM_SELF, "\n");CHKERRQ(ierr);
#if 0
      for (c = 0; c < Nc; ++c) {
        ierr = PetscPrintf(PETSC_COMM_SELF, " D[%" PetscInt_FMT ",%" PetscInt_FMT "]:", b, c);CHKERRQ(ierr);
        for (d = 0; d < dim; ++d) {ierr = PetscPrintf(PETSC_COMM_SELF, " %g", (double) B[((p*Nb+b)*Nc+c)*dim+d)]);CHKERRQ(ierr);}
        ierr = PetscPrintf(PETSC_COMM_SELF, "\n");CHKERRQ(ierr);
      }
#endif
    }
  }
  ierr = DMRestoreWorkArray(dm, Np*Nb, MPIU_REAL, &B);CHKERRQ(ierr);
  ierr = DMRestoreWorkArray(dm, Np*Nb*dim, MPIU_REAL, &D);CHKERRQ(ierr);
  ierr = DMRestoreWorkArray(dm, Np*Nb*dim*dim, MPIU_REAL, &H);CHKERRQ(ierr);
  ierr = PetscFree(points);CHKERRQ(ierr);
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
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
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
        ierr = PetscFEIntegrateResidual(ds, key, Ne, chunkGeom, u, NULL, NULL, NULL, 0.0, elemVec);CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscFEGeomRestoreChunk(affineGeom, cStart, cEnd, &chunkGeom);CHKERRQ(ierr);
  ierr = DestroyFEGeometry(dm, ds, cellIS, &affineQuad, &affineGeom, &quads, &geoms);CHKERRQ(ierr);
  ierr = ISDestroy(&cellIS);CHKERRQ(ierr);
  ierr = PetscFree2(u, elemVec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TestUnisolvence(DM dm)
{
  Mat M;
  Vec v;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetLocalVector(dm, &v);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &v);CHKERRQ(ierr);
  ierr = DMCreateMassMatrix(dm, dm, &M);CHKERRQ(ierr);
  ierr = MatViewFromOptions(M, NULL, "-mass_view");CHKERRQ(ierr);
  ierr = MatDestroy(&M);CHKERRQ(ierr);
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
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_SUP, "This is a uniprocessor example only.");
  ierr = ProcessOptions(PETSC_COMM_WORLD, &ctx);CHKERRQ(ierr);
  ierr = DMCreate(PETSC_COMM_WORLD, &dm);CHKERRQ(ierr);
  ierr = DMSetType(dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) dm, "Mesh");CHKERRQ(ierr);
  ierr = PetscObjectViewFromOptions((PetscObject) dm, NULL, "-dm_view");CHKERRQ(ierr);
  ierr = SetupDiscretization(dm, "field", SetupPrimalProblem, &ctx);CHKERRQ(ierr);
  ierr = TestEvaluation(dm);CHKERRQ(ierr);
  ierr = TestIntegration(dm, 1, ctx.its);CHKERRQ(ierr);
  ierr = TestUnisolvence(dm);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
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
