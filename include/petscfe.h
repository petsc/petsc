/*
      Objects which encapsulate finite element spaces and operations
*/
#if !defined(__PETSCFE_H)
#define __PETSCFE_H
#include <petscdm.h>
#include <petscdt.h>
#include <petscfetypes.h>
#include <petscdstypes.h>

/* Assuming dim <= 3 */
typedef struct {
  PetscReal v0[3];
  PetscReal J[9];
  PetscReal invJ[9];
  PetscReal detJ;
  PetscReal n[3];
  PetscInt  dim;
  PetscInt  dimEmbed;
} PetscFECellGeom;

PETSC_EXTERN PetscErrorCode PetscFEInitializePackage(void);

PETSC_EXTERN PetscClassId PETSCSPACE_CLASSID;

/*J
  PetscSpaceType - String with the name of a PETSc linear space

  Level: beginner

.seealso: PetscSpaceSetType(), PetscSpace
J*/
typedef const char *PetscSpaceType;
#define PETSCSPACEPOLYNOMIAL "poly"
#define PETSCSPACEDG         "dg"

PETSC_EXTERN PetscFunctionList PetscSpaceList;
PETSC_EXTERN PetscErrorCode PetscSpaceCreate(MPI_Comm, PetscSpace *);
PETSC_EXTERN PetscErrorCode PetscSpaceDestroy(PetscSpace *);
PETSC_EXTERN PetscErrorCode PetscSpaceSetType(PetscSpace, PetscSpaceType);
PETSC_EXTERN PetscErrorCode PetscSpaceGetType(PetscSpace, PetscSpaceType *);
PETSC_EXTERN PetscErrorCode PetscSpaceSetUp(PetscSpace);
PETSC_EXTERN PetscErrorCode PetscSpaceSetFromOptions(PetscSpace);
PETSC_STATIC_INLINE PetscErrorCode PetscSpaceViewFromOptions(PetscSpace A,PetscObject B,const char name[]) {return PetscObjectViewFromOptions((PetscObject)A,B,name);}

PETSC_EXTERN PetscErrorCode PetscSpaceView(PetscSpace,PetscViewer);
PETSC_EXTERN PetscErrorCode PetscSpaceRegister(const char [], PetscErrorCode (*)(PetscSpace));
PETSC_EXTERN PetscErrorCode PetscSpaceRegisterDestroy(void);

PETSC_EXTERN PetscErrorCode PetscSpaceGetDimension(PetscSpace, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscSpaceSetOrder(PetscSpace, PetscInt);
PETSC_EXTERN PetscErrorCode PetscSpaceGetOrder(PetscSpace, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscSpaceEvaluate(PetscSpace, PetscInt, const PetscReal[], PetscReal[], PetscReal[], PetscReal[]);

PETSC_EXTERN PetscErrorCode PetscSpacePolynomialSetNumVariables(PetscSpace, PetscInt);
PETSC_EXTERN PetscErrorCode PetscSpacePolynomialGetNumVariables(PetscSpace, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscSpacePolynomialSetSymmetric(PetscSpace, PetscBool);
PETSC_EXTERN PetscErrorCode PetscSpacePolynomialGetSymmetric(PetscSpace, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscSpacePolynomialSetTensor(PetscSpace, PetscBool);
PETSC_EXTERN PetscErrorCode PetscSpacePolynomialGetTensor(PetscSpace, PetscBool *);

PETSC_EXTERN PetscErrorCode PetscSpaceDGSetQuadrature(PetscSpace, PetscQuadrature);
PETSC_EXTERN PetscErrorCode PetscSpaceDGGetQuadrature(PetscSpace, PetscQuadrature *);

PETSC_EXTERN PetscClassId PETSCDUALSPACE_CLASSID;

/*J
  PetscDualSpaceType - String with the name of a PETSc dual space

  Level: beginner

.seealso: PetscDualSpaceSetType(), PetscDualSpace
J*/
typedef const char *PetscDualSpaceType;
#define PETSCDUALSPACELAGRANGE "lagrange"
#define PETSCDUALSPACESIMPLE   "simple"

PETSC_EXTERN PetscFunctionList PetscDualSpaceList;
PETSC_EXTERN PetscErrorCode PetscDualSpaceCreate(MPI_Comm, PetscDualSpace *);
PETSC_EXTERN PetscErrorCode PetscDualSpaceDestroy(PetscDualSpace *);
PETSC_EXTERN PetscErrorCode PetscDualSpaceDuplicate(PetscDualSpace, PetscDualSpace *);
PETSC_EXTERN PetscErrorCode PetscDualSpaceSetType(PetscDualSpace, PetscDualSpaceType);
PETSC_EXTERN PetscErrorCode PetscDualSpaceGetType(PetscDualSpace, PetscDualSpaceType *);
PETSC_EXTERN PetscErrorCode PetscDualSpaceGetNumDof(PetscDualSpace, const PetscInt **);
PETSC_EXTERN PetscErrorCode PetscDualSpaceSetUp(PetscDualSpace);
PETSC_EXTERN PetscErrorCode PetscDualSpaceSetFromOptions(PetscDualSpace);
PETSC_STATIC_INLINE PetscErrorCode PetscDualSpaceViewFromOptions(PetscDualSpace A,PetscObject B,const char name[]) {return PetscObjectViewFromOptions((PetscObject)A,B,name);}

PETSC_EXTERN PetscErrorCode PetscDualSpaceView(PetscDualSpace,PetscViewer);
PETSC_EXTERN PetscErrorCode PetscDualSpaceRegister(const char [], PetscErrorCode (*)(PetscDualSpace));
PETSC_EXTERN PetscErrorCode PetscDualSpaceRegisterDestroy(void);

PETSC_EXTERN PetscErrorCode PetscDualSpaceGetDimension(PetscDualSpace, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscDualSpaceSetOrder(PetscDualSpace, PetscInt);
PETSC_EXTERN PetscErrorCode PetscDualSpaceGetOrder(PetscDualSpace, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscDualSpaceSetDM(PetscDualSpace, DM);
PETSC_EXTERN PetscErrorCode PetscDualSpaceGetDM(PetscDualSpace, DM *);
PETSC_EXTERN PetscErrorCode PetscDualSpaceGetFunctional(PetscDualSpace, PetscInt, PetscQuadrature *);
PETSC_EXTERN PetscErrorCode PetscDualSpaceCreateReferenceCell(PetscDualSpace, PetscInt, PetscBool, DM *);

PETSC_EXTERN PetscErrorCode PetscDualSpaceApply(PetscDualSpace, PetscInt, PetscReal, PetscFECellGeom *, PetscInt, PetscErrorCode (*)(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *), void *, PetscScalar *);

PETSC_EXTERN PetscErrorCode PetscDualSpaceLagrangeGetContinuity(PetscDualSpace, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscDualSpaceLagrangeSetContinuity(PetscDualSpace, PetscBool);

PETSC_EXTERN PetscErrorCode PetscDualSpaceGetHeightSubspace(PetscDualSpace,PetscInt,PetscDualSpace *);
PETSC_EXTERN PetscErrorCode PetscDualSpaceSimpleSetDimension(PetscDualSpace, PetscInt);
PETSC_EXTERN PetscErrorCode PetscDualSpaceSimpleSetFunctional(PetscDualSpace, PetscInt, PetscQuadrature);

PETSC_EXTERN PetscClassId PETSCFE_CLASSID;

/*J
  PetscFEType - String with the name of a PETSc finite element space

  Level: beginner

  Note: Currently, the classes are concerned with the implementation of element integration

.seealso: PetscFESetType(), PetscFE
J*/
typedef const char *PetscFEType;
#define PETSCFEBASIC     "basic"
#define PETSCFENONAFFINE "nonaffine"
#define PETSCFEOPENCL    "opencl"
#define PETSCFECOMPOSITE "composite"

PETSC_EXTERN PetscFunctionList PetscFEList;
PETSC_EXTERN PetscErrorCode PetscFECreate(MPI_Comm, PetscFE *);
PETSC_EXTERN PetscErrorCode PetscFEDestroy(PetscFE *);
PETSC_EXTERN PetscErrorCode PetscFESetType(PetscFE, PetscFEType);
PETSC_EXTERN PetscErrorCode PetscFEGetType(PetscFE, PetscFEType *);
PETSC_EXTERN PetscErrorCode PetscFESetUp(PetscFE);
PETSC_EXTERN PetscErrorCode PetscFESetFromOptions(PetscFE);
PETSC_STATIC_INLINE PetscErrorCode PetscFEViewFromOptions(PetscFE A,PetscObject B,const char name[]) {return PetscObjectViewFromOptions((PetscObject)A,B,name);}

PETSC_EXTERN PetscErrorCode PetscFEView(PetscFE,PetscViewer);
PETSC_EXTERN PetscErrorCode PetscFERegister(const char [], PetscErrorCode (*)(PetscFE));
PETSC_EXTERN PetscErrorCode PetscFERegisterDestroy(void);
PETSC_EXTERN PetscErrorCode PetscFECreateDefault(DM, PetscInt, PetscInt, PetscBool, const char [], PetscInt, PetscFE *);

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
PETSC_EXTERN PetscErrorCode PetscFEGetNumDof(PetscFE, const PetscInt **);
PETSC_EXTERN PetscErrorCode PetscFEGetDefaultTabulation(PetscFE, PetscReal **, PetscReal **, PetscReal **);
PETSC_EXTERN PetscErrorCode PetscFEGetFaceTabulation(PetscFE, PetscReal **);
PETSC_EXTERN PetscErrorCode PetscFEGetTabulation(PetscFE, PetscInt, const PetscReal[], PetscReal **, PetscReal **, PetscReal **);
PETSC_EXTERN PetscErrorCode PetscFERestoreTabulation(PetscFE, PetscInt, const PetscReal[], PetscReal **, PetscReal **, PetscReal **);
PETSC_EXTERN PetscErrorCode PetscFERefine(PetscFE, PetscFE *);

PETSC_EXTERN PetscErrorCode PetscFEIntegrate(PetscFE, PetscDS, PetscInt, PetscInt, PetscFECellGeom *, const PetscScalar[], PetscDS, const PetscScalar[], PetscReal[]);
PETSC_EXTERN PetscErrorCode PetscFEIntegrateResidual(PetscFE, PetscDS, PetscInt, PetscInt, PetscFECellGeom *, const PetscScalar[], const PetscScalar[], PetscDS, const PetscScalar[], PetscScalar[]);
PETSC_EXTERN PetscErrorCode PetscFEIntegrateBdResidual(PetscFE, PetscDS, PetscInt, PetscInt, PetscFECellGeom *, const PetscScalar[], const PetscScalar[], PetscDS, const PetscScalar[], PetscScalar[]);
PETSC_EXTERN PetscErrorCode PetscFEIntegrateJacobian(PetscFE, PetscDS, PetscFEJacobianType, PetscInt, PetscInt, PetscInt, PetscFECellGeom *, const PetscScalar[], const PetscScalar[], PetscDS, const PetscScalar[], PetscScalar[]);
PETSC_EXTERN PetscErrorCode PetscFEIntegrateBdJacobian(PetscFE, PetscDS, PetscInt, PetscInt, PetscInt, PetscFECellGeom *, const PetscScalar[], const PetscScalar[], PetscDS, const PetscScalar[], PetscScalar[]);

PETSC_EXTERN PetscErrorCode PetscFECompositeGetMapping(PetscFE, PetscInt *, const PetscReal *[], const PetscReal *[], const PetscReal *[]);

PETSC_EXTERN PetscErrorCode PetscFEOpenCLSetRealType(PetscFE, PetscDataType);
PETSC_EXTERN PetscErrorCode PetscFEOpenCLGetRealType(PetscFE, PetscDataType *);

#endif
