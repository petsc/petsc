/*
      Objects which encapsulate finite element spaces and operations
*/
#ifndef PETSCFE_H
#define PETSCFE_H
#include <petscdm.h>
#include <petscdt.h>
#include <petscfetypes.h>
#include <petscdstypes.h>
#include <petscspace.h>
#include <petscdualspace.h>

/* SUBMANSEC = FE */

/*MC
      PetscFEGeom - Structure for geometric information for `PetscFE`

    Level: intermediate

.seealso: `PetscFE`, `PetscFEGeomCreate()`, `PetscFEGeomDestroy()`, `PetscFEGeomGetChunk()`, `PetscFEGeomRestoreChunk()`, `PetscFEGeomGetPoint()`, `PetscFEGeomGetCellPoint()`,
          `PetscFEGeomComplete()`, `PetscSpace`, `PetscDualSpace`
M*/
typedef struct _n_PetscFEGeom {
  const PetscReal *xi;
  PetscReal       *v;     /* v[Nc*Np*dE]:           The first point in each each in real coordinates */
  PetscReal       *J;     /* J[Nc*Np*dE*dE]:        The Jacobian of the map from reference to real coordinates (if nonsquare it is completed with orthogonal columns) */
  PetscReal       *invJ;  /* invJ[Nc*Np*dE*dE]:     The inverse of the Jacobian of the map from reference to real coordinates (if nonsquare it is completed with orthogonal columns) */
  PetscReal       *detJ;  /* detJ[Nc*Np]:           The determinant of J, and if it is non-square its the volume change */
  PetscReal       *n;     /* n[Nc*Np*dE]:           For faces, the normal to the face in real coordinates, outward for the first supporting cell */
  PetscInt (*face)[4];    /* face[Nc][s*2]:         For faces, the local face number (cone index) and orientation for this face in each supporting cell s */
  PetscReal *suppJ[2];    /* sJ[s][Nc*Np*dE*dE]:    For faces, the Jacobian for each supporting cell s */
  PetscReal *suppInvJ[2]; /* sInvJ[s][Nc*Np*dE*dE]: For faces, the inverse Jacobian for each supporting cell s */
  PetscReal *suppDetJ[2]; /* sInvJ[s][Nc*Np*dE*dE]: For faces, the Jacobian determinant for each supporting cell s */
  PetscInt   dim;         /* dim: Topological dimension */
  PetscInt   dimEmbed;    /* dE:  coordinate dimension */
  PetscInt   numCells;    /* Nc:  Number of mesh points represented in the arrays */
  PetscInt   numPoints;   /* Np:  Number of evaluation points represented in the arrays */
  PetscBool  isAffine;    /* Flag for affine transforms */
  PetscBool  isCohesive;  /* Flag for a cohesive cell */
} PetscFEGeom;

PETSC_EXTERN PetscErrorCode PetscFEInitializePackage(void);

PETSC_EXTERN PetscErrorCode PetscFEGeomCreate(PetscQuadrature, PetscInt, PetscInt, PetscBool, PetscFEGeom **);
PETSC_EXTERN PetscErrorCode PetscFEGeomGetQuadrature(PetscFEGeom *, PetscQuadrature *);
PETSC_EXTERN PetscErrorCode PetscFEGeomSetQuadrature(PetscFEGeom *, PetscQuadrature);
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

PETSC_EXTERN PetscFunctionList PetscFEList;
PETSC_EXTERN PetscErrorCode    PetscFECreate(MPI_Comm, PetscFE *);
PETSC_EXTERN PetscErrorCode    PetscFEDestroy(PetscFE *);
PETSC_EXTERN PetscErrorCode    PetscFESetType(PetscFE, PetscFEType);
PETSC_EXTERN PetscErrorCode    PetscFEGetType(PetscFE, PetscFEType *);
PETSC_EXTERN PetscErrorCode    PetscFESetUp(PetscFE);
PETSC_EXTERN PetscErrorCode    PetscFESetFromOptions(PetscFE);
PETSC_EXTERN PetscErrorCode    PetscFEViewFromOptions(PetscFE, PetscObject, const char[]);
PETSC_EXTERN PetscErrorCode    PetscFESetName(PetscFE, const char[]);

PETSC_EXTERN PetscErrorCode PetscFEView(PetscFE, PetscViewer);
PETSC_EXTERN PetscErrorCode PetscFERegister(const char[], PetscErrorCode (*)(PetscFE));
PETSC_EXTERN PetscErrorCode PetscFERegisterDestroy(void);
PETSC_EXTERN PetscErrorCode PetscFECreateDefault(MPI_Comm, PetscInt, PetscInt, PetscBool, const char[], PetscInt, PetscFE *);
PETSC_EXTERN PetscErrorCode PetscFECreateByCell(MPI_Comm, PetscInt, PetscInt, DMPolytopeType, const char[], PetscInt, PetscFE *);
PETSC_EXTERN PetscErrorCode PetscFECreateLagrange(MPI_Comm, PetscInt, PetscInt, PetscBool, PetscInt, PetscInt, PetscFE *);
PETSC_EXTERN PetscErrorCode PetscFECreateLagrangeByCell(MPI_Comm, PetscInt, PetscInt, DMPolytopeType, PetscInt, PetscInt, PetscFE *);
PETSC_EXTERN PetscErrorCode PetscFECreateFromSpaces(PetscSpace, PetscDualSpace, PetscQuadrature, PetscQuadrature, PetscFE *);

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
PETSC_EXTERN PetscErrorCode PetscFEIntegrateHybridResidual(PetscDS, PetscDS, PetscFormKey, PetscInt, PetscInt, PetscFEGeom *, const PetscScalar[], const PetscScalar[], PetscDS, const PetscScalar[], PetscReal, PetscScalar[]);
PETSC_EXTERN PetscErrorCode PetscFEIntegrateJacobian(PetscDS, PetscFEJacobianType, PetscFormKey, PetscInt, PetscFEGeom *, const PetscScalar[], const PetscScalar[], PetscDS, const PetscScalar[], PetscReal, PetscReal, PetscScalar[]);
PETSC_EXTERN PetscErrorCode PetscFEIntegrateBdJacobian(PetscDS, PetscWeakForm, PetscFormKey, PetscInt, PetscFEGeom *, const PetscScalar[], const PetscScalar[], PetscDS, const PetscScalar[], PetscReal, PetscReal, PetscScalar[]);
PETSC_EXTERN PetscErrorCode PetscFEIntegrateHybridJacobian(PetscDS, PetscDS, PetscFEJacobianType, PetscFormKey, PetscInt, PetscInt, PetscFEGeom *, const PetscScalar[], const PetscScalar[], PetscDS, const PetscScalar[], PetscReal, PetscReal, PetscScalar[]);

PETSC_EXTERN PetscErrorCode PetscFECompositeGetMapping(PetscFE, PetscInt *, const PetscReal *[], const PetscReal *[], const PetscReal *[]);

PETSC_EXTERN PetscErrorCode PetscFECreateHeightTrace(PetscFE, PetscInt, PetscFE *);
PETSC_EXTERN PetscErrorCode PetscFECreatePointTrace(PetscFE, PetscInt, PetscFE *);

PETSC_EXTERN PetscErrorCode PetscFEOpenCLSetRealType(PetscFE, PetscDataType);
PETSC_EXTERN PetscErrorCode PetscFEOpenCLGetRealType(PetscFE, PetscDataType *);

#endif
