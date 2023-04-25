/*
      Objects which encapsulate finite element spaces
*/
#ifndef PETSCSPACE_H
#define PETSCSPACE_H
#include <petscdm.h>
#include <petscdt.h>

/* SUBMANSEC = SPACE */

/*S
  PetscSpace - PETSc object that manages a linear space, e.g. the space of d-dimensional polynomials of given degree

  Level: beginner

.seealso: `PetscSpaceCreate()`, `PetscDualSpace`, `PetscDualSpaceCreate()`, `PetscSpaceSetType()`, `PetscSpaceType`, PetscFE`
S*/
typedef struct _p_PetscSpace *PetscSpace;

PETSC_EXTERN PetscErrorCode PetscFEInitializePackage(void);

PETSC_EXTERN PetscClassId PETSCSPACE_CLASSID;

/*J
  PetscSpaceType - String with the name of a PETSc linear space

  Values:
+  `PETSCSPACEPOLYNOMIAL` - a polynomial space, e.g. P1 is the space of linear polynomials
.  `PETSCSPACEPTRIMMED` - a trimmed polynomial space
.  `PETSCSPACETENSOR` - a space consisting of the tensor product of two or more spaces
.  `PETSCSPACESUM` - a direct or a concatenation sum
.  `PETSCSPACEPOINT` - functions defined by values on a set of quadrature points
.  `PETSCSPACESUBSPACE` - some kind of subspace, no idea what
-  `PETSCSPACEWXY` - space that encapsulates the Wheeler-Xu-Yotov enrichments

  Level: beginner

.seealso: `PetscSpaceSetType()`, `PetscSpace`, `PetscSpaceType`
J*/
typedef const char *PetscSpaceType;
#define PETSCSPACEPOLYNOMIAL "poly"
#define PETSCSPACEPTRIMMED   "ptrimmed"
#define PETSCSPACETENSOR     "tensor"
#define PETSCSPACESUM        "sum"
#define PETSCSPACEPOINT      "point"
#define PETSCSPACESUBSPACE   "subspace"
#define PETSCSPACEWXY        "wxy"

PETSC_EXTERN PetscFunctionList PetscSpaceList;
PETSC_EXTERN PetscErrorCode    PetscSpaceCreate(MPI_Comm, PetscSpace *);
PETSC_EXTERN PetscErrorCode    PetscSpaceDestroy(PetscSpace *);
PETSC_EXTERN PetscErrorCode    PetscSpaceSetType(PetscSpace, PetscSpaceType);
PETSC_EXTERN PetscErrorCode    PetscSpaceGetType(PetscSpace, PetscSpaceType *);
PETSC_EXTERN PetscErrorCode    PetscSpaceSetUp(PetscSpace);
PETSC_EXTERN PetscErrorCode    PetscSpaceSetFromOptions(PetscSpace);
PETSC_EXTERN PetscErrorCode    PetscSpaceViewFromOptions(PetscSpace, PetscObject, const char[]);

PETSC_EXTERN PetscErrorCode PetscSpaceView(PetscSpace, PetscViewer);
PETSC_EXTERN PetscErrorCode PetscSpaceRegister(const char[], PetscErrorCode (*)(PetscSpace));
PETSC_EXTERN PetscErrorCode PetscSpaceRegisterDestroy(void);

PETSC_EXTERN PetscErrorCode PetscSpaceGetDimension(PetscSpace, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscSpaceSetNumComponents(PetscSpace, PetscInt);
PETSC_EXTERN PetscErrorCode PetscSpaceGetNumComponents(PetscSpace, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscSpaceSetNumVariables(PetscSpace, PetscInt);
PETSC_EXTERN PetscErrorCode PetscSpaceGetNumVariables(PetscSpace, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscSpaceSetDegree(PetscSpace, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode PetscSpaceGetDegree(PetscSpace, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscSpaceEvaluate(PetscSpace, PetscInt, const PetscReal[], PetscReal[], PetscReal[], PetscReal[]);
PETSC_EXTERN PetscErrorCode PetscSpaceGetHeightSubspace(PetscSpace, PetscInt, PetscSpace *);

static inline PETSC_DEPRECATED_FUNCTION("Property not used (since v3.17)") PetscErrorCode PetscSpacePolynomialSetSymmetric(PetscSpace sp, PetscBool s)
{
  PetscCheck(!s, PetscObjectComm((PetscObject)sp), PETSC_ERR_SUP, "PETSCSPACEPOLYNOMIAL does not support symmetric polynomials");
  return PETSC_SUCCESS;
}
static inline PETSC_DEPRECATED_FUNCTION("Property not used (since v3.17)") PetscErrorCode PetscSpacePolynomialGetSymmetric(PETSC_UNUSED PetscSpace sp, PetscBool *s)
{
  *s = PETSC_FALSE;
  return PETSC_SUCCESS;
}
PETSC_EXTERN PetscErrorCode PetscSpacePolynomialSetTensor(PetscSpace, PetscBool);
PETSC_EXTERN PetscErrorCode PetscSpacePolynomialGetTensor(PetscSpace, PetscBool *);

PETSC_EXTERN PetscErrorCode PetscSpacePTrimmedSetFormDegree(PetscSpace, PetscInt);
PETSC_EXTERN PetscErrorCode PetscSpacePTrimmedGetFormDegree(PetscSpace, PetscInt *);

PETSC_EXTERN PetscErrorCode PetscSpaceTensorSetNumSubspaces(PetscSpace, PetscInt);
PETSC_EXTERN PetscErrorCode PetscSpaceTensorGetNumSubspaces(PetscSpace, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscSpaceTensorSetSubspace(PetscSpace, PetscInt, PetscSpace);
PETSC_EXTERN PetscErrorCode PetscSpaceTensorGetSubspace(PetscSpace, PetscInt, PetscSpace *);

PETSC_EXTERN PetscErrorCode PetscSpaceSumSetNumSubspaces(PetscSpace, PetscInt);
PETSC_EXTERN PetscErrorCode PetscSpaceSumGetNumSubspaces(PetscSpace, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscSpaceSumSetSubspace(PetscSpace, PetscInt, PetscSpace);
PETSC_EXTERN PetscErrorCode PetscSpaceSumGetSubspace(PetscSpace, PetscInt, PetscSpace *);
PETSC_EXTERN PetscErrorCode PetscSpaceSumSetConcatenate(PetscSpace, PetscBool);
PETSC_EXTERN PetscErrorCode PetscSpaceSumGetConcatenate(PetscSpace, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscSpaceCreateSum(PetscInt numSubspaces, const PetscSpace subspaces[], PetscBool concatenate, PetscSpace *sumSpace);

PETSC_EXTERN PetscErrorCode PetscSpacePointGetPoints(PetscSpace, PetscQuadrature *);
PETSC_EXTERN PetscErrorCode PetscSpacePointSetPoints(PetscSpace, PetscQuadrature);

#endif
