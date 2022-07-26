static const char help[] = "Simple libCEED test to calculate surface area using 1^T M 1";

/*
  This is a recreation of libCeed Example 2: https://libceed.readthedocs.io/en/latest/examples/ceed/
*/

#include <petscdmceed.h>
#include <petscdmplexceed.h>
#include <petscfeceed.h>
#include <petscdmplex.h>
#include <petscds.h>

typedef struct {
  PetscReal         areaExact;
  CeedQFunctionUser setupgeo,       apply;
  const char       *setupgeofname, *applyfname;
} AppCtx;

typedef struct {
  CeedQFunction qf_apply;
  CeedOperator  op_apply;
  CeedVector    qdata, uceed, vceed;
} CeedData;

static PetscErrorCode CeedDataDestroy(CeedData *data)
{
  PetscFunctionBeginUser;
  PetscCall(CeedVectorDestroy(&data->qdata));
  PetscCall(CeedVectorDestroy(&data->uceed));
  PetscCall(CeedVectorDestroy(&data->vceed));
  PetscCall(CeedQFunctionDestroy(&data->qf_apply));
  PetscCall(CeedOperatorDestroy(&data->op_apply));
  PetscFunctionReturn(0);
}

CEED_QFUNCTION(Mass)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out)
{
  const CeedScalar *u = in[0], *qdata = in[1];
  CeedScalar       *v = out[0];

  CeedPragmaSIMD
  for (CeedInt i = 0; i < Q; ++i)
    v[i] = qdata[i] * u[i];

  return 0;
}

/*
// Reference (parent) 2D coordinates: X \in [-1, 1]^2
//
// Global physical coordinates given by the mesh (3D): xx \in [-l, l]^3
//
// Local physical coordinates on the manifold (2D): x \in [-l, l]^2
//
// Change of coordinates matrix computed by the library:
//   (physical 3D coords relative to reference 2D coords)
//   dxx_j/dX_i (indicial notation) [3 * 2]
//
// Change of coordinates x (physical 2D) relative to xx (phyisical 3D):
//   dx_i/dxx_j (indicial notation) [2 * 3]
//
// Change of coordinates x (physical 2D) relative to X (reference 2D):
//   (by chain rule)
//   dx_i/dX_j = dx_i/dxx_k * dxx_k/dX_j
//
// The quadrature data is stored in the array qdata.
//
// We require the determinant of the Jacobian to properly compute integrals of the form: int(u v)
//
// Qdata: w * det(dx_i/dX_j)
*/
CEED_QFUNCTION(SetupMassGeoCube)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out)
{
  const CeedScalar *J = in[1], *w = in[2];
  CeedScalar       *qdata = out[0];

  CeedPragmaSIMD
  for (CeedInt i = 0; i < Q; ++i) {
    // Read dxxdX Jacobian entries, stored as [[0 3], [1 4], [2 5]]
    const CeedScalar dxxdX[3][2] = {{J[i+Q*0], J[i+Q*3]},
                                    {J[i+Q*1], J[i+Q*4]},
                                    {J[i+Q*2], J[i+Q*5]}};
    // Modulus of dxxdX column vectors
    const CeedScalar modg1 = PetscSqrtReal(dxxdX[0][0]*dxxdX[0][0] + dxxdX[1][0]*dxxdX[1][0] + dxxdX[2][0]*dxxdX[2][0]);
    const CeedScalar modg2 = PetscSqrtReal(dxxdX[0][1]*dxxdX[0][1] + dxxdX[1][1]*dxxdX[1][1] + dxxdX[2][1]*dxxdX[2][1]);
    // Use normalized column vectors of dxxdX as rows of dxdxx
    const CeedScalar dxdxx[2][3] = {{dxxdX[0][0] / modg1, dxxdX[1][0] / modg1, dxxdX[2][0] / modg1},
                                    {dxxdX[0][1] / modg2, dxxdX[1][1] / modg2, dxxdX[2][1] / modg2}};

    CeedScalar dxdX[2][2];
    for (int j = 0; j < 2; ++j)
      for (int k = 0; k < 2; ++k) {
        dxdX[j][k] = 0;
        for (int l = 0; l < 3; ++l)
          dxdX[j][k] += dxdxx[j][l]*dxxdX[l][k];
      }
    qdata[i+Q*0] = (dxdX[0][0]*dxdX[1][1] - dxdX[1][0]*dxdX[0][1]) * w[i]; /* det J * weight */
  }
  return 0;
}

/*
// Reference (parent) 2D coordinates: X \in [-1, 1]^2
//
// Global 3D physical coordinates given by the mesh: xx \in [-R, R]^3
//   with R radius of the sphere
//
// Local 3D physical coordinates on the 2D manifold: x \in [-l, l]^3
//   with l half edge of the cube inscribed in the sphere
//
// Change of coordinates matrix computed by the library:
//   (physical 3D coords relative to reference 2D coords)
//   dxx_j/dX_i (indicial notation) [3 * 2]
//
// Change of coordinates x (on the 2D manifold) relative to xx (phyisical 3D):
//   dx_i/dxx_j (indicial notation) [3 * 3]
//
// Change of coordinates x (on the 2D manifold) relative to X (reference 2D):
//   (by chain rule)
//   dx_i/dX_j = dx_i/dxx_k * dxx_k/dX_j [3 * 2]
//
// modJ is given by the magnitude of the cross product of the columns of dx_i/dX_j
//
// The quadrature data is stored in the array qdata.
//
// We require the determinant of the Jacobian to properly compute integrals of
//   the form: int(u v)
//
// Qdata: modJ * w
*/
CEED_QFUNCTION(SetupMassGeoSphere)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar *X = in[0], *J = in[1], *w = in[2];
  CeedScalar       *qdata = out[0];

  CeedPragmaSIMD
  for (CeedInt i = 0; i < Q; ++i) {
    const CeedScalar xx[3][1] = {{X[i+0*Q]}, {X[i+1*Q]}, {X[i+2*Q]}};
    // Read dxxdX Jacobian entries, stored as [[0 3], [1 4], [2 5]]
    const CeedScalar dxxdX[3][2] = {{J[i+Q*0], J[i+Q*3]},
                                    {J[i+Q*1], J[i+Q*4]},
                                    {J[i+Q*2], J[i+Q*5]}};
    // Setup
    const CeedScalar modxxsq = xx[0][0]*xx[0][0]+xx[1][0]*xx[1][0]+xx[2][0]*xx[2][0];
    CeedScalar xxsq[3][3];
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k) {
        xxsq[j][k] = 0.;
        for (int l = 0; l < 1; ++l)
          xxsq[j][k] += xx[j][l]*xx[k][l] / (sqrt(modxxsq) * modxxsq);
      }

    const CeedScalar dxdxx[3][3] = {{1./sqrt(modxxsq) - xxsq[0][0], -xxsq[0][1], -xxsq[0][2]},
                                    {-xxsq[1][0], 1./sqrt(modxxsq) - xxsq[1][1], -xxsq[1][2]},
                                    {-xxsq[2][0], -xxsq[2][1], 1./sqrt(modxxsq) - xxsq[2][2]}};

    CeedScalar dxdX[3][2];
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 2; ++k) {
        dxdX[j][k] = 0.;
        for (int l = 0; l < 3; ++l)
          dxdX[j][k] += dxdxx[j][l]*dxxdX[l][k];
      }
    // J is given by the cross product of the columns of dxdX
    const CeedScalar J[3][1] = {{dxdX[1][0]*dxdX[2][1] - dxdX[2][0]*dxdX[1][1]},
                                {dxdX[2][0]*dxdX[0][1] - dxdX[0][0]*dxdX[2][1]},
                                {dxdX[0][0]*dxdX[1][1] - dxdX[1][0]*dxdX[0][1]}};
    // Use the magnitude of J as our detJ (volume scaling factor)
    const CeedScalar modJ = sqrt(J[0][0]*J[0][0]+J[1][0]*J[1][0]+J[2][0]*J[2][0]);
    qdata[i+Q*0] = modJ * w[i];
  }
  return 0;
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *ctx)
{
  DMPlexShape    shape = DM_SHAPE_UNKNOWN;

  PetscFunctionBeginUser;
  PetscOptionsBegin(comm, "", "libCEED Test Options", "DMPLEX");
  PetscOptionsEnd();
  PetscCall(PetscOptionsGetEnum(NULL, NULL, "-dm_plex_shape", DMPlexShapes, (PetscEnum *) &shape, NULL));
  ctx->setupgeo      = NULL;
  ctx->setupgeofname = NULL;
  ctx->apply         = Mass;
  ctx->applyfname    = Mass_loc;
  ctx->areaExact     = 0.0;
  switch (shape) {
    case DM_SHAPE_BOX_SURFACE:
      ctx->setupgeo      = SetupMassGeoCube;
      ctx->setupgeofname = SetupMassGeoCube_loc;
      ctx->areaExact     = 6.0;
      break;
    case DM_SHAPE_SPHERE:
      ctx->setupgeo      = SetupMassGeoSphere;
      ctx->setupgeofname = SetupMassGeoSphere_loc;
      ctx->areaExact     = 4.0*M_PI;
      break;
    default: break;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *ctx, DM *dm)
{
  PetscFunctionBegin;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
#ifdef PETSC_HAVE_LIBCEED
  {
    Ceed        ceed;
    const char *usedresource;

    PetscCall(DMGetCeed(*dm, &ceed));
    PetscCall(CeedGetResource(ceed, &usedresource));
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject) *dm), "libCEED Backend: %s\n", usedresource));
  }
#endif
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDiscretization(DM dm)
{
  DM        cdm;
  PetscFE   fe, cfe;
  PetscInt  dim, cnc;
  PetscBool simplex;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexIsSimplex(dm, &simplex));
  PetscCall(PetscFECreateDefault(PETSC_COMM_SELF, dim, 1, simplex, NULL, PETSC_DETERMINE, &fe));
  PetscCall(PetscFESetName(fe, "indicator"));
  PetscCall(DMAddField(dm, NULL, (PetscObject) fe));
  PetscCall(PetscFEDestroy(&fe));
  PetscCall(DMCreateDS(dm));
  PetscCall(DMPlexSetClosurePermutationTensor(dm, PETSC_DETERMINE, NULL));
  PetscCall(DMGetCoordinateDim(dm, &cnc));
  PetscCall(PetscFECreateDefault(PETSC_COMM_SELF, dim, cnc, simplex, NULL, PETSC_DETERMINE, &cfe));
  PetscCall(DMProjectCoordinates(dm, cfe));
  PetscCall(PetscFEDestroy(&cfe));
  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMPlexSetClosurePermutationTensor(cdm, PETSC_DETERMINE, NULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode LibCeedSetupByDegree(DM dm, AppCtx *ctx, CeedData *data)
{
  PetscDS             ds;
  PetscFE             fe, cfe;
  Ceed                ceed;
  CeedElemRestriction Erestrictx, Erestrictu, Erestrictq;
  CeedQFunction       qf_setupgeo;
  CeedOperator        op_setupgeo;
  CeedVector          xcoord;
  CeedBasis           basisu, basisx;
  CeedInt             Nqdata = 1;
  CeedInt             nqpts, nqptsx;
  DM                  cdm;
  Vec                 coords;
  const PetscScalar  *coordArray;
  PetscInt            dim, cdim, cStart, cEnd, Ncell;

  PetscFunctionBeginUser;
  PetscCall(DMGetCeed(dm, &ceed));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetCoordinateDim(dm, &cdim));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  Ncell = cEnd - cStart;
  // CEED bases
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSGetDiscretization(ds, 0, (PetscObject *) &fe));
  PetscCall(PetscFEGetCeedBasis(fe, &basisu));
  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMGetDS(cdm, &ds));
  PetscCall(PetscDSGetDiscretization(ds, 0, (PetscObject *) &cfe));
  PetscCall(PetscFEGetCeedBasis(cfe, &basisx));

  PetscCall(DMPlexGetCeedRestriction(cdm, NULL, 0, 0, 0, &Erestrictx));
  PetscCall(DMPlexGetCeedRestriction(dm,  NULL, 0, 0, 0, &Erestrictu));
  PetscCall(CeedBasisGetNumQuadraturePoints(basisu, &nqpts));
  PetscCall(CeedBasisGetNumQuadraturePoints(basisx, &nqptsx));
  PetscCheck(nqptsx == nqpts,PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Number of qpoints for u %" PetscInt_FMT " != %" PetscInt_FMT " Number of qpoints for x", nqpts, nqptsx);
  PetscCall(CeedElemRestrictionCreateStrided(ceed, Ncell, nqpts, Nqdata, Nqdata*Ncell*nqpts, CEED_STRIDES_BACKEND, &Erestrictq));

  PetscCall(DMGetCoordinatesLocal(dm, &coords));
  PetscCall(VecGetArrayRead(coords, &coordArray));
  PetscCall(CeedElemRestrictionCreateVector(Erestrictx, &xcoord, NULL));
  PetscCall(CeedVectorSetArray(xcoord, CEED_MEM_HOST, CEED_COPY_VALUES, (PetscScalar *) coordArray));
  PetscCall(VecRestoreArrayRead(coords, &coordArray));

  // Create the vectors that will be needed in setup and apply
  PetscCall(CeedElemRestrictionCreateVector(Erestrictu, &data->uceed, NULL));
  PetscCall(CeedElemRestrictionCreateVector(Erestrictu, &data->vceed, NULL));
  PetscCall(CeedElemRestrictionCreateVector(Erestrictq, &data->qdata, NULL));

  // Create the Q-function that builds the operator (i.e. computes its quadrature data) and set its context data
  PetscCall(CeedQFunctionCreateInterior(ceed, 1, ctx->setupgeo, ctx->setupgeofname, &qf_setupgeo));
  PetscCall(CeedQFunctionAddInput(qf_setupgeo,  "x",      cdim,     CEED_EVAL_INTERP));
  PetscCall(CeedQFunctionAddInput(qf_setupgeo,  "dx",     cdim*dim, CEED_EVAL_GRAD));
  PetscCall(CeedQFunctionAddInput(qf_setupgeo,  "weight", 1,        CEED_EVAL_WEIGHT));
  PetscCall(CeedQFunctionAddOutput(qf_setupgeo, "qdata",  Nqdata,   CEED_EVAL_NONE));

  // Set up the mass operator
  PetscCall(CeedQFunctionCreateInterior(ceed, 1, ctx->apply, ctx->applyfname, &data->qf_apply));
  PetscCall(CeedQFunctionAddInput(data->qf_apply,  "u",     1,      CEED_EVAL_INTERP));
  PetscCall(CeedQFunctionAddInput(data->qf_apply,  "qdata", Nqdata, CEED_EVAL_NONE));
  PetscCall(CeedQFunctionAddOutput(data->qf_apply, "v",     1,      CEED_EVAL_INTERP));

  // Create the operator that builds the quadrature data for the operator
  PetscCall(CeedOperatorCreate(ceed, qf_setupgeo, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_setupgeo));
  PetscCall(CeedOperatorSetField(op_setupgeo, "x",      Erestrictx, basisx, CEED_VECTOR_ACTIVE));
  PetscCall(CeedOperatorSetField(op_setupgeo, "dx",     Erestrictx, basisx, CEED_VECTOR_ACTIVE));
  PetscCall(CeedOperatorSetField(op_setupgeo, "weight", CEED_ELEMRESTRICTION_NONE, basisx, CEED_VECTOR_NONE));
  PetscCall(CeedOperatorSetField(op_setupgeo, "qdata",  Erestrictq, CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE));

  // Create the mass operator
  PetscCall(CeedOperatorCreate(ceed, data->qf_apply, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &data->op_apply));
  PetscCall(CeedOperatorSetField(data->op_apply, "u",     Erestrictu, basisu, CEED_VECTOR_ACTIVE));
  PetscCall(CeedOperatorSetField(data->op_apply, "qdata", Erestrictq, CEED_BASIS_COLLOCATED, data->qdata));
  PetscCall(CeedOperatorSetField(data->op_apply, "v",     Erestrictu, basisu, CEED_VECTOR_ACTIVE));

  // Setup qdata
  PetscCall(CeedOperatorApply(op_setupgeo, xcoord, data->qdata, CEED_REQUEST_IMMEDIATE));

  PetscCall(CeedElemRestrictionDestroy(&Erestrictq));
  PetscCall(CeedQFunctionDestroy(&qf_setupgeo));
  PetscCall(CeedOperatorDestroy(&op_setupgeo));
  PetscCall(CeedVectorDestroy(&xcoord));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  MPI_Comm       comm;
  DM             dm;
  AppCtx         ctx;
  Vec            U, Uloc, V, Vloc;
  PetscScalar   *v;
  PetscScalar    area;
  CeedData       ceeddata;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscCall(ProcessOptions(comm, &ctx));
  PetscCall(CreateMesh(comm, &ctx, &dm));
  PetscCall(SetupDiscretization(dm));

  PetscCall(LibCeedSetupByDegree(dm, &ctx, &ceeddata));

  PetscCall(DMCreateGlobalVector(dm, &U));
  PetscCall(DMCreateLocalVector(dm, &Uloc));
  PetscCall(VecDuplicate(U, &V));
  PetscCall(VecDuplicate(Uloc, &Vloc));

  /**/
  PetscCall(VecSet(Uloc, 1.));
  PetscCall(VecZeroEntries(V));
  PetscCall(VecZeroEntries(Vloc));
  PetscCall(VecGetArray(Vloc, &v));
  PetscCall(CeedVectorSetArray(ceeddata.vceed, CEED_MEM_HOST, CEED_USE_POINTER, v));
  PetscCall(CeedVectorSetValue(ceeddata.uceed, 1.0));
  PetscCall(CeedOperatorApply(ceeddata.op_apply, ceeddata.uceed, ceeddata.vceed, CEED_REQUEST_IMMEDIATE));
  PetscCall(CeedVectorTakeArray(ceeddata.vceed, CEED_MEM_HOST, NULL));
  PetscCall(VecRestoreArray(Vloc, &v));
  PetscCall(DMLocalToGlobalBegin(dm, Vloc, ADD_VALUES, V));
  PetscCall(DMLocalToGlobalEnd(dm, Vloc, ADD_VALUES, V));

  PetscCall(VecSum(V, &area));
  if (ctx.areaExact > 0.) {
    PetscReal error = PetscAbsReal(area - ctx.areaExact);
    PetscReal tol   = PETSC_SMALL;

    PetscCall(PetscPrintf(comm,   "Exact mesh surface area    : % .*f\n", PetscAbsReal(ctx.areaExact - round(ctx.areaExact)) > 1E-15 ? 14 : 1, (double) ctx.areaExact));
    PetscCall(PetscPrintf(comm,   "Computed mesh surface area : % .*f\n", PetscAbsScalar(area          - round(area))          > 1E-15 ? 14 : 1, (double)PetscRealPart(area)));
    if (error > tol) {
      PetscCall(PetscPrintf(comm, "Area error                 : % .14g\n", (double) error));
    } else {
      PetscCall(PetscPrintf(comm, "Area verifies!\n"));
    }
  }

  PetscCall(CeedDataDestroy(&ceeddata));
  PetscCall(VecDestroy(&U));
  PetscCall(VecDestroy(&Uloc));
  PetscCall(VecDestroy(&V));
  PetscCall(VecDestroy(&Vloc));
  PetscCall(DMDestroy(&dm));
  return PetscFinalize();
}

/*TEST

  build:
    requires: libceed

  testset:
    args: -dm_plex_simplex 0 -petscspace_degree 3 -dm_view -dm_petscds_view \
          -petscfe_default_quadrature_order 4 -coord_dm_default_quadrature_order 4

    test:
      suffix: cube_3
      args: -dm_plex_shape box_surface -dm_refine 2

    test:
      suffix: cube_3_p4
      nsize: 4
      args: -petscpartitioner_type simple -dm_refine_pre 1 -dm_plex_shape box_surface -dm_refine 1

    test:
      suffix: sphere_3
      args: -dm_plex_shape sphere -dm_refine 3

    test:
      suffix: sphere_3_p4
      nsize: 4
      args: -petscpartitioner_type simple -dm_refine_pre 1 -dm_plex_shape sphere -dm_refine 2

TEST*/
