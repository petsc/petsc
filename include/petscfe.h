/*
      Objects which encapsulate finite element spaces and operations
*/
#pragma once
#include <petscdm.h>
#include <petscdt.h>
#include <petscfetypes.h>
#include <petscdstypes.h>
#include <petscspace.h>
#include <petscdualspace.h>

/* MANSEC = DM */
/* SUBMANSEC = FE */

/*MC
    PetscFEGeom - Structure for geometric information for `PetscFE`

    Level: intermediate

    Note:
    This is a struct, not a `PetscObject`

.seealso: `PetscFE`, `PetscFEGeomCreate()`, `PetscFEGeomDestroy()`, `PetscFEGeomGetChunk()`, `PetscFEGeomRestoreChunk()`, `PetscFEGeomGetPoint()`, `PetscFEGeomGetCellPoint()`,
          `PetscFEGeomComplete()`, `PetscSpace`, `PetscDualSpace`
M*/
typedef struct {
  // We can represent several different types of geometry, which we call modes:
  //   basic:    dim == dE, only bulk data
  //     These are normal dim-cells
  //   embedded: dim < dE, only bulk data
  //     These are dim-cells embedded in a higher dimension, as an embedded manifold
  //   boundary: dim < dE, bulk and face data
  //     These are dim-cells on the boundary of a dE-mesh
  //   cohesive: dim < dE, bulk and face data
  //     These are dim-cells in the interior of a dE-mesh
  //   affine:
  //     For all modes, the transforms between real and reference are affine
  PetscFEGeomMode mode;     // The type of geometric data stored
  PetscBool       isAffine; // Flag for affine transforms
  // Sizes
  PetscInt dim;       // dim: topological dimension and reference coordinate dimension
  PetscInt dimEmbed;  // dE:  real coordinate dimension
  PetscInt numCells;  // Nc:  Number of mesh points represented in the arrays (points are assumed to be the same DMPolytopeType)
  PetscInt numPoints; // Np:  Number of evaluation points represented in the arrays
  // Bulk data
  const PetscReal *xi;   // xi[dim]                The first point in each cell in reference coordinates
  PetscReal       *v;    // v[Nc*Np*dE]:           The first point in each cell in real coordinates
  PetscReal       *J;    // J[Nc*Np*dE*dE]:        The Jacobian of the map from reference to real coordinates (if nonsquare it is completed with orthogonal columns)
  PetscReal       *invJ; // invJ[Nc*Np*dE*dE]:     The inverse of the Jacobian of the map from reference to real coordinates (if nonsquare it is completed with orthogonal columns)
  PetscReal       *detJ; // detJ[Nc*Np]:           The determinant of J, and if J is non-square it is the volume change
  // Face data
  PetscReal *n;           // n[Nc*Np*dE]:           For faces, the normal to the face in real coordinates, outward for the first supporting cell
  PetscInt (*face)[4];    // face[Nc][s*2]:         For faces, the local face number (cone index) and orientation for this face in each supporting cell
  PetscReal *suppJ[2];    // sJ[s][Nc*Np*dE*dE]:    For faces, the Jacobian for each supporting cell
  PetscReal *suppInvJ[2]; // sInvJ[s][Nc*Np*dE*dE]: For faces, the inverse Jacobian for each supporting cell
  PetscReal *suppDetJ[2]; // sdetJ[s][Nc*Np]:       For faces, the Jacobian determinant for each supporting cell
} PetscFEGeom;

PETSC_EXTERN PetscErrorCode PetscFEInitializePackage(void);

PETSC_EXTERN PetscErrorCode PetscFEGeomCreate(PetscQuadrature, PetscInt, PetscInt, PetscFEGeomMode, PetscFEGeom **);
PETSC_EXTERN PetscErrorCode PetscFEGeomGetChunk(PetscFEGeom *, PetscInt, PetscInt, PetscFEGeom **);
PETSC_EXTERN PetscErrorCode PetscFEGeomRestoreChunk(PetscFEGeom *, PetscInt, PetscInt, PetscFEGeom **);
PETSC_EXTERN PetscErrorCode PetscFEGeomGetPoint(PetscFEGeom *, PetscInt, PetscInt, const PetscReal[], PetscFEGeom *);
PETSC_EXTERN PetscErrorCode PetscFEGeomGetCellPoint(PetscFEGeom *, PetscInt, PetscInt, PetscFEGeom *);
PETSC_EXTERN PetscErrorCode PetscFEGeomComplete(PetscFEGeom *);
PETSC_EXTERN PetscErrorCode PetscFEGeomDestroy(PetscFEGeom **);

PETSC_EXTERN PetscErrorCode PetscDualSpaceApply(PetscDualSpace, PetscInt, PetscReal, PetscFEGeom *, PetscInt, PetscErrorCode (*)(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *), void *, PetscScalar *);
PETSC_EXTERN PetscErrorCode PetscDualSpaceApplyDefault(PetscDualSpace, PetscInt, PetscReal, PetscFEGeom *, PetscInt, PetscErrorCode (*)(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar *, void *), void *, PetscScalar *);

PETSC_EXTERN PetscErrorCode PetscDualSpaceTransform(PetscDualSpace, PetscDualSpaceTransformType, PetscBool, PetscFEGeom *, PetscInt, PetscInt, PetscScalar[]);
PETSC_EXTERN PetscErrorCode PetscDualSpaceTransformGradient(PetscDualSpace, PetscDualSpaceTransformType, PetscBool, PetscFEGeom *, PetscInt, PetscInt, PetscScalar[]);
PETSC_EXTERN PetscErrorCode PetscDualSpaceTransformHessian(PetscDualSpace, PetscDualSpaceTransformType, PetscBool, PetscFEGeom *, PetscInt, PetscInt, PetscScalar[]);
PETSC_EXTERN PetscErrorCode PetscDualSpacePullback(PetscDualSpace, PetscFEGeom *, PetscInt, PetscInt, PetscScalar[]);
PETSC_EXTERN PetscErrorCode PetscDualSpacePushforward(PetscDualSpace, PetscFEGeom *, PetscInt, PetscInt, PetscScalar[]);
PETSC_EXTERN PetscErrorCode PetscDualSpacePushforwardGradient(PetscDualSpace, PetscFEGeom *, PetscInt, PetscInt, PetscScalar[]);
PETSC_EXTERN PetscErrorCode PetscDualSpacePushforwardHessian(PetscDualSpace, PetscFEGeom *, PetscInt, PetscInt, PetscScalar[]);

PETSC_EXTERN PetscClassId PETSCFE_CLASSID;

/*J
  PetscFEType - String with the name of a PETSc finite element space

  Level: beginner

  Note:
  Currently, the classes are concerned with the implementation of element integration

.seealso: `PetscFESetType()`, `PetscFE`
J*/
typedef const char *PetscFEType;
#define PETSCFEBASIC     "basic"
#define PETSCFEOPENCL    "opencl"
#define PETSCFECOMPOSITE "composite"
#define PETSCFEVECTOR    "vector"

PETSC_EXTERN PetscFunctionList PetscFEList;
PETSC_EXTERN PetscErrorCode    PetscFECreate(MPI_Comm, PetscFE *);
PETSC_EXTERN PetscErrorCode    PetscFEDestroy(PetscFE *);
PETSC_EXTERN PetscErrorCode    PetscFESetType(PetscFE, PetscFEType);
PETSC_EXTERN PetscErrorCode    PetscFEGetType(PetscFE, PetscFEType *);
PETSC_EXTERN PetscErrorCode    PetscFESetUp(PetscFE);
PETSC_EXTERN PetscErrorCode    PetscFESetFromOptions(PetscFE);
PETSC_EXTERN PetscErrorCode    PetscFEViewFromOptions(PetscFE, PetscObject, const char[]);
PETSC_EXTERN PetscErrorCode    PetscFESetName(PetscFE, const char[]);
PETSC_EXTERN PetscErrorCode    PetscFECreateVector(PetscFE, PetscInt, PetscBool, PetscBool, PetscFE *);

PETSC_EXTERN PetscErrorCode PetscFEView(PetscFE, PetscViewer);
PETSC_EXTERN PetscErrorCode PetscFERegister(const char[], PetscErrorCode (*)(PetscFE));
PETSC_EXTERN PetscErrorCode PetscFERegisterDestroy(void);
PETSC_EXTERN PetscErrorCode PetscFECreateDefault(MPI_Comm, PetscInt, PetscInt, PetscBool, const char[], PetscInt, PetscFE *);
PETSC_EXTERN PetscErrorCode PetscFECreateByCell(MPI_Comm, PetscInt, PetscInt, DMPolytopeType, const char[], PetscInt, PetscFE *);
PETSC_EXTERN PetscErrorCode PetscFECreateLagrange(MPI_Comm, PetscInt, PetscInt, PetscBool, PetscInt, PetscInt, PetscFE *);
PETSC_EXTERN PetscErrorCode PetscFECreateLagrangeByCell(MPI_Comm, PetscInt, PetscInt, DMPolytopeType, PetscInt, PetscInt, PetscFE *);
PETSC_EXTERN PetscErrorCode PetscFECreateFromSpaces(PetscSpace, PetscDualSpace, PetscQuadrature, PetscQuadrature, PetscFE *);
PETSC_EXTERN PetscErrorCode PetscFELimitDegree(PetscFE, PetscInt, PetscInt, PetscFE *);
PETSC_EXTERN PetscErrorCode PetscFECreateBrokenElement(PetscFE, PetscFE *);

PETSC_EXTERN PetscErrorCode PetscFEGetDimension(PetscFE, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscFEGetSpatialDimension(PetscFE, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscFESetNumComponents(PetscFE, PetscInt);
PETSC_EXTERN PetscErrorCode PetscFEGetNumComponents(PetscFE, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscFEGetTileSizes(PetscFE, PetscInt *, PetscInt *, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscFESetTileSizes(PetscFE, PetscInt, PetscInt, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode PetscFESetBasisSpace(PetscFE, PetscSpace);
PETSC_EXTERN PetscErrorCode PetscFEGetBasisSpace(PetscFE, PetscSpace *);
PETSC_EXTERN PetscErrorCode PetscFESetDualSpace(PetscFE, PetscDualSpace);
PETSC_EXTERN PetscErrorCode PetscFEGetDualSpace(PetscFE, PetscDualSpace *);
PETSC_EXTERN PetscErrorCode PetscFESetQuadrature(PetscFE, PetscQuadrature);
PETSC_EXTERN PetscErrorCode PetscFEGetQuadrature(PetscFE, PetscQuadrature *);
PETSC_EXTERN PetscErrorCode PetscFESetFaceQuadrature(PetscFE, PetscQuadrature);
PETSC_EXTERN PetscErrorCode PetscFEGetFaceQuadrature(PetscFE, PetscQuadrature *);
PETSC_EXTERN PetscErrorCode PetscFEExpandFaceQuadrature(PetscFE, PetscQuadrature, PetscQuadrature *);
PETSC_EXTERN PetscErrorCode PetscFECopyQuadrature(PetscFE, PetscFE);
PETSC_EXTERN PetscErrorCode PetscFEGetNumDof(PetscFE, const PetscInt **);

/* TODO: Need a function to reuse the memory when retabulating the same FE at different points */
PETSC_EXTERN PetscErrorCode PetscFEGetCellTabulation(PetscFE, PetscInt, PetscTabulation *);
PETSC_EXTERN PetscErrorCode PetscFEGetFaceTabulation(PetscFE, PetscInt, PetscTabulation *);
PETSC_EXTERN PetscErrorCode PetscFEGetFaceCentroidTabulation(PetscFE, PetscTabulation *);
PETSC_EXTERN PetscErrorCode PetscFECreateTabulation(PetscFE, PetscInt, PetscInt, const PetscReal[], PetscInt, PetscTabulation *);
PETSC_EXTERN PetscErrorCode PetscFEComputeTabulation(PetscFE, PetscInt, const PetscReal[], PetscInt, PetscTabulation);
PETSC_EXTERN PetscErrorCode PetscTabulationDestroy(PetscTabulation *);

PETSC_EXTERN PetscErrorCode PetscFERefine(PetscFE, PetscFE *);
PETSC_EXTERN PetscErrorCode PetscFEGetHeightSubspace(PetscFE, PetscInt, PetscFE *);

PETSC_EXTERN PetscErrorCode PetscFECreateCellGeometry(PetscFE, PetscQuadrature, PetscFEGeom *);
PETSC_EXTERN PetscErrorCode PetscFEDestroyCellGeometry(PetscFE, PetscFEGeom *);
PETSC_EXTERN PetscErrorCode PetscFEPushforward(PetscFE, PetscFEGeom *, PetscInt, PetscScalar[]);
PETSC_EXTERN PetscErrorCode PetscFEPushforwardGradient(PetscFE, PetscFEGeom *, PetscInt, PetscScalar[]);
PETSC_EXTERN PetscErrorCode PetscFEPushforwardHessian(PetscFE, PetscFEGeom *, PetscInt, PetscScalar[]);

PETSC_EXTERN PetscErrorCode PetscFEIntegrate(PetscDS, PetscInt, PetscInt, PetscFEGeom *, const PetscScalar[], PetscDS, const PetscScalar[], PetscScalar[]);
PETSC_EXTERN PetscErrorCode PetscFEIntegrateBd(PetscDS, PetscInt, void (*)(PetscInt, PetscInt, PetscInt, const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[], PetscReal, const PetscReal[], const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]), PetscInt, PetscFEGeom *, const PetscScalar[], PetscDS, const PetscScalar[], PetscScalar[]);
PETSC_EXTERN PetscErrorCode PetscFEIntegrateResidual(PetscDS, PetscFormKey, PetscInt, PetscFEGeom *, const PetscScalar[], const PetscScalar[], PetscDS, const PetscScalar[], PetscReal, PetscScalar[]);
PETSC_EXTERN PetscErrorCode PetscFEIntegrateBdResidual(PetscDS, PetscWeakForm, PetscFormKey, PetscInt, PetscFEGeom *, const PetscScalar[], const PetscScalar[], PetscDS, const PetscScalar[], PetscReal, PetscScalar[]);
PETSC_EXTERN PetscErrorCode PetscFEIntegrateHybridResidual(PetscDS, PetscDS, PetscFormKey, PetscInt, PetscInt, PetscFEGeom *, PetscFEGeom *, const PetscScalar[], const PetscScalar[], PetscDS, const PetscScalar[], PetscReal, PetscScalar[]);
PETSC_EXTERN PetscErrorCode PetscFEIntegrateJacobian(PetscDS, PetscDS, PetscFEJacobianType, PetscFormKey, PetscInt, PetscFEGeom *, const PetscScalar[], const PetscScalar[], PetscDS, const PetscScalar[], PetscReal, PetscReal, PetscScalar[]);
PETSC_EXTERN PetscErrorCode PetscFEIntegrateBdJacobian(PetscDS, PetscWeakForm, PetscFEJacobianType, PetscFormKey, PetscInt, PetscFEGeom *, const PetscScalar[], const PetscScalar[], PetscDS, const PetscScalar[], PetscReal, PetscReal, PetscScalar[]);
PETSC_EXTERN PetscErrorCode PetscFEIntegrateHybridJacobian(PetscDS, PetscDS, PetscFEJacobianType, PetscFormKey, PetscInt, PetscInt, PetscFEGeom *, PetscFEGeom *, const PetscScalar[], const PetscScalar[], PetscDS, const PetscScalar[], PetscReal, PetscReal, PetscScalar[]);

PETSC_EXTERN PetscErrorCode PetscFECompositeGetMapping(PetscFE, PetscInt *, const PetscReal *[], const PetscReal *[], const PetscReal *[]);

PETSC_EXTERN PetscErrorCode PetscFECreateHeightTrace(PetscFE, PetscInt, PetscFE *);
PETSC_EXTERN PetscErrorCode PetscFECreatePointTrace(PetscFE, PetscInt, PetscFE *);

PETSC_EXTERN PetscErrorCode PetscFEOpenCLSetRealType(PetscFE, PetscDataType);
PETSC_EXTERN PetscErrorCode PetscFEOpenCLGetRealType(PetscFE, PetscDataType *);

#ifdef PETSC_HAVE_LIBCEED

  #ifndef PLEXFE_QFUNCTION
    #define PLEXFE_QFUNCTION(fname, f0_name, f1_name) \
      CEED_QFUNCTION(PlexQFunction##fname)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) \
      { \
        const CeedScalar *u = in[0], *du = in[1], *qdata = in[2]; \
        CeedScalar       *v = out[0], *dv = out[1]; \
        const PetscInt    Nc   = 1; \
        const PetscInt    cdim = 2; \
\
        CeedPragmaSIMD for (CeedInt i = 0; i < Q; ++i) \
        { \
          const PetscInt   uOff[2]    = {0, Nc}; \
          const PetscInt   uOff_x[2]  = {0, Nc * cdim}; \
          const CeedScalar x[2]       = {qdata[i + Q * 1], qdata[i + Q * 2]}; \
          const CeedScalar invJ[2][2] = { \
            {qdata[i + Q * 3], qdata[i + Q * 5]}, \
            {qdata[i + Q * 4], qdata[i + Q * 6]} \
          }; \
          const CeedScalar u_x[2] = {invJ[0][0] * du[i + Q * 0] + invJ[1][0] * du[i + Q * 1], invJ[0][1] * du[i + Q * 0] + invJ[1][1] * du[i + Q * 1]}; \
          PetscScalar      f0[Nc]; \
          PetscScalar      f1[Nc * cdim]; \
\
          for (PetscInt k = 0; k < Nc; ++k) f0[k] = 0; \
          for (PetscInt k = 0; k < Nc * cdim; ++k) f1[k] = 0; \
          f0_name(2, 1, 0, uOff, uOff_x, u, NULL, u_x, NULL, NULL, NULL, NULL, NULL, 0.0, x, 0, NULL, f0); \
          f1_name(2, 1, 0, uOff, uOff_x, u, NULL, u_x, NULL, NULL, NULL, NULL, NULL, 0.0, x, 0, NULL, f1); \
\
          dv[i + Q * 0] = qdata[i + Q * 0] * (invJ[0][0] * f1[0] + invJ[0][1] * f1[1]); \
          dv[i + Q * 1] = qdata[i + Q * 0] * (invJ[1][0] * f1[0] + invJ[1][1] * f1[1]); \
          v[i]          = qdata[i + Q * 0] * f0[0]; \
        } \
        return CEED_ERROR_SUCCESS; \
      }
  #endif

#else

  #ifndef PLEXFE_QFUNCTION
    #define PLEXFE_QFUNCTION(fname, f0_name, f1_name)
  #endif

#endif
