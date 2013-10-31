/*
      Objects which encapsulate finite element spaces and operations
*/
#if !defined(__PETSCFE_H)
#define __PETSCFE_H
#include <petscdm.h>
#include <petscdt.h>
#include <petscfetypes.h>

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
PETSC_EXTERN PetscBool         PetscSpaceRegisterAllCalled;
PETSC_EXTERN PetscErrorCode PetscSpaceCreate(MPI_Comm, PetscSpace *);
PETSC_EXTERN PetscErrorCode PetscSpaceDestroy(PetscSpace *);
PETSC_EXTERN PetscErrorCode PetscSpaceSetType(PetscSpace, PetscSpaceType);
PETSC_EXTERN PetscErrorCode PetscSpaceGetType(PetscSpace, PetscSpaceType *);
PETSC_EXTERN PetscErrorCode PetscSpaceSetUp(PetscSpace);
PETSC_EXTERN PetscErrorCode PetscSpaceSetFromOptions(PetscSpace);
PETSC_EXTERN PetscErrorCode PetscSpaceRegister(const char [], PetscErrorCode (*)(PetscSpace));
PETSC_EXTERN PetscErrorCode PetscSpaceRegisterAll(void);
PETSC_EXTERN PetscErrorCode PetscSpaceRegisterDestroy(void);

PETSC_EXTERN PetscErrorCode PetscSpaceGetDimension(PetscSpace, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscSpaceSetOrder(PetscSpace, PetscInt);
PETSC_EXTERN PetscErrorCode PetscSpaceGetOrder(PetscSpace, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscSpaceEvaluate(PetscSpace, PetscInt, const PetscReal[], PetscReal[], PetscReal[], PetscReal[]);

PETSC_EXTERN PetscErrorCode PetscSpacePolynomialSetNumVariables(PetscSpace, PetscInt);
PETSC_EXTERN PetscErrorCode PetscSpacePolynomialGetNumVariables(PetscSpace, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscSpacePolynomialSetSymmetric(PetscSpace, PetscBool);
PETSC_EXTERN PetscErrorCode PetscSpacePolynomialGetSymmetric(PetscSpace, PetscBool *);

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

PETSC_EXTERN PetscFunctionList PetscDualSpaceList;
PETSC_EXTERN PetscBool         PetscDualSpaceRegisterAllCalled;
PETSC_EXTERN PetscErrorCode PetscDualSpaceCreate(MPI_Comm, PetscDualSpace *);
PETSC_EXTERN PetscErrorCode PetscDualSpaceDestroy(PetscDualSpace *);
PETSC_EXTERN PetscErrorCode PetscDualSpaceSetType(PetscDualSpace, PetscDualSpaceType);
PETSC_EXTERN PetscErrorCode PetscDualSpaceGetType(PetscDualSpace, PetscDualSpaceType *);
PETSC_EXTERN PetscErrorCode PetscDualSpaceSetUp(PetscDualSpace);
PETSC_EXTERN PetscErrorCode PetscDualSpaceSetFromOptions(PetscDualSpace);
PETSC_EXTERN PetscErrorCode PetscDualSpaceRegister(const char [], PetscErrorCode (*)(PetscDualSpace));
PETSC_EXTERN PetscErrorCode PetscDualSpaceRegisterAll(void);
PETSC_EXTERN PetscErrorCode PetscDualSpaceRegisterDestroy(void);

PETSC_EXTERN PetscErrorCode PetscDualSpaceGetDimension(PetscDualSpace, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscDualSpaceSetOrder(PetscDualSpace, PetscInt);
PETSC_EXTERN PetscErrorCode PetscDualSpaceGetOrder(PetscDualSpace, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscDualSpaceSetDM(PetscDualSpace, DM);
PETSC_EXTERN PetscErrorCode PetscDualSpaceGetDM(PetscDualSpace, DM *);
PETSC_EXTERN PetscErrorCode PetscDualSpaceGetFunctional(PetscDualSpace, PetscInt, PetscQuadrature *);
PETSC_EXTERN PetscErrorCode PetscDualSpaceCreateReferenceCell(PetscDualSpace, PetscInt, PetscBool, DM *);

PETSC_EXTERN PetscErrorCode PetscDualSpaceApply(PetscDualSpace, PetscInt, PetscCellGeometry, PetscInt, void (*)(const PetscReal [], PetscScalar *), PetscScalar *);

PETSC_EXTERN PetscClassId PETSCFE_CLASSID;

/*J
  PetscFEType - String with the name of a PETSc finite element space

  Level: beginner

  Note: Currently, the classes are concerned with the implementation of element integration

.seealso: PetscFESetType(), PetscFE
J*/
typedef const char *PetscFEType;
#define PETSCFEBASIC  "basic"
#define PETSCFEOPENCL "opencl"

PETSC_EXTERN PetscFunctionList PetscFEList;
PETSC_EXTERN PetscBool         PetscFERegisterAllCalled;
PETSC_EXTERN PetscErrorCode PetscFECreate(MPI_Comm, PetscFE *);
PETSC_EXTERN PetscErrorCode PetscFEDestroy(PetscFE *);
PETSC_EXTERN PetscErrorCode PetscFESetType(PetscFE, PetscFEType);
PETSC_EXTERN PetscErrorCode PetscFEGetType(PetscFE, PetscFEType *);
PETSC_EXTERN PetscErrorCode PetscFESetUp(PetscFE);
PETSC_EXTERN PetscErrorCode PetscFESetFromOptions(PetscFE);
PETSC_EXTERN PetscErrorCode PetscFERegister(const char [], PetscErrorCode (*)(PetscFE));
PETSC_EXTERN PetscErrorCode PetscFERegisterAll(void);
PETSC_EXTERN PetscErrorCode PetscFERegisterDestroy(void);

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
PETSC_EXTERN PetscErrorCode PetscFEGetTabulation(PetscFE, PetscInt, const PetscReal[], PetscReal **, PetscReal **, PetscReal **);
PETSC_EXTERN PetscErrorCode PetscFERestoreTabulation(PetscFE, PetscInt, const PetscReal[], PetscReal **, PetscReal **, PetscReal **);

PETSC_EXTERN PetscErrorCode PetscFEIntegrateResidual(PetscFE, PetscInt, PetscInt, PetscFE[], PetscInt, PetscCellGeometry, const PetscScalar[],
                                                     PetscInt, PetscFE[], const PetscScalar[],
                                                     void (*)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                                     void (*)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                                     PetscScalar[]);
PETSC_EXTERN PetscErrorCode PetscFEIntegrateBdResidual(PetscFE, PetscInt, PetscInt, PetscFE[], PetscInt, PetscCellGeometry, const PetscScalar[],
                                                       PetscInt, PetscFE[], const PetscScalar[],
                                                       void (*)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], const PetscReal[], PetscScalar[]),
                                                       void (*)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], const PetscReal[], PetscScalar[]),
                                                       PetscScalar[]);
PETSC_EXTERN PetscErrorCode PetscFEIntegrateJacobianAction(PetscFE, PetscInt, PetscInt, PetscFE[], PetscInt, PetscCellGeometry, const PetscScalar[], const PetscScalar[],
                                                           void (**)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                                           void (**)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                                           void (**)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                                           void (**)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                                           PetscScalar[]);
PETSC_EXTERN PetscErrorCode PetscFEIntegrateJacobian(PetscFE, PetscInt, PetscInt, PetscFE[], PetscInt, PetscInt, PetscCellGeometry, const PetscScalar[],
                                                     PetscInt, PetscFE[], const PetscScalar[],
                                                     void (*)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                                     void (*)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                                     void (*)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                                     void (*)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]),
                                                     PetscScalar[]);

PETSC_EXTERN PetscErrorCode PetscFEOpenCLSetRealType(PetscFE, PetscDataType);
PETSC_EXTERN PetscErrorCode PetscFEOpenCLGetRealType(PetscFE, PetscDataType *);

/* TODO: Should be moved inside DM */
typedef struct {
  PetscFE         *fe;
  PetscFE         *feAux;
  void (**f0Funcs)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]); /* The f_0 functions for each field */
  void (**f1Funcs)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]); /* The f_1 functions for each field */
  void (**g0Funcs)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]); /* The g_0 functions for each field pair */
  void (**g1Funcs)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]); /* The g_1 functions for each field pair */
  void (**g2Funcs)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]); /* The g_2 functions for each field pair */
  void (**g3Funcs)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], PetscScalar[]); /* The g_3 functions for each field pair */
  void (**bcFuncs)(const PetscReal[], PetscScalar *); /* The boundary condition function for each field component */
  PetscFE         *feBd;
  void (**f0BdFuncs)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], const PetscReal[], PetscScalar[]); /* The f_0 functions for each field */
  void (**f1BdFuncs)(const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscScalar[], const PetscReal[], const PetscReal[], PetscScalar[]); /* The f_1 functions for each field */
} PetscFEM;

#endif
