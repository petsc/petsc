/*
      Objects which encapsulate finite element spaces
*/
#ifndef PETSCDUALSPACE_H
#define PETSCDUALSPACE_H
#include <petscdm.h>
#include <petscdt.h>
#include <petscfetypes.h>
#include <petscdstypes.h>
#include <petscspace.h>

/* SUBMANSEC = DUALSPACE */

/*S
  PetscDualSpace - PETSc object that manages the dual space to a linear space, e.g. the space of evaluation functionals at the vertices of a triangle

  Level: beginner

.seealso: `PetscDualSpaceCreate()`, `PetscSpace`, `PetscSpaceCreate()`, `PetscDualSpaceSetType()`, `PetscDualSpaceType`
S*/
typedef struct _p_PetscDualSpace *PetscDualSpace;

/*MC
  PetscDualSpaceReferenceCell - The type of reference cell

  Level: beginner

  Note:
  This is used only for automatic creation of reference cells. A `PetscDualSpace` can accept an arbitrary `DM` for a reference cell.

.seealso: `PetscSpace`, `PetscDualSpaceCreate()`, `PetscDualSpaceType`
M*/
typedef enum {
  PETSCDUALSPACE_REFCELL_SIMPLEX,
  PETSCDUALSPACE_REFCELL_TENSOR
} PetscDualSpaceReferenceCell;
PETSC_EXTERN const char *const PetscDualSpaceReferenceCells[];

/*MC
  PetscDualSpaceTransformType - The type of function transform

  Level: intermediate

  Values:
+  `IDENTITY_TRANSFORM` - make no changes in the function
.  `COVARIANT_PIOLA_TRANSFORM` - Covariant Piola: $\sigma^*(F) = J^{-T} F \circ \phi^{-1)$
-  `CONTRAVARIANT_PIOLA_TRANSFORM` - Contravariant Piola: $\sigma^*(F) = 1/|J| J F \circ \phi^{-1)$

  Notes:
  These transforms, and their inverses, are used to move functions and functionals between the reference element and real space.
  Suppose that we have a mapping $\phi$ which maps the reference cell to real space, and its Jacobian $J$. If we want to transform function $F$ on the reference element,
  so that it acts on real space, we use the pushforward transform $\sigma^*$. The pullback $\sigma_*$ is the inverse transform.

  References:
.    Rognes, Kirby, and Logg, Efficient Assembly of Hdiv and Hrot Conforming Finite Elements, SISC, 31(6), 4130-4151, arXiv 1205.3085, 2010

.seealso: `PetscDualSpaceGetDeRahm()`
M*/
typedef enum {
  IDENTITY_TRANSFORM,
  COVARIANT_PIOLA_TRANSFORM,
  CONTRAVARIANT_PIOLA_TRANSFORM
} PetscDualSpaceTransformType;

PETSC_EXTERN PetscClassId PETSCDUALSPACE_CLASSID;

/*J
  PetscDualSpaceType - String with the name of a PETSc dual space

  Values:
+ PETSCDUALSPACELAGRANGE  - a dual space of pointwise evaluation functionals
. PETSCDUALSPACESIMPLE -  a dual space defined by functionals provided with `PetscDualSpaceSimpleSetFunctional()`
. PETSCDUALSPACEREFINED - the joint dual space defined by a group of cells, usually refined from one larger cell
- PETSCDUALSPACEBDM - a dual space for Brezzi-Douglas-Marini elements

  Level: beginner

.seealso: `PetscDualSpaceSetType()`, `PetscDualSpace`, `PetscSpace`
J*/
typedef const char *PetscDualSpaceType;
#define PETSCDUALSPACELAGRANGE "lagrange"
#define PETSCDUALSPACESIMPLE   "simple"
#define PETSCDUALSPACEREFINED  "refined"
#define PETSCDUALSPACEBDM      "bdm"

/*MC
  PETSCDUALSPACEBDM = "bdm" - A `PetscDualSpace` object that encapsulates a dual space for Brezzi-Douglas-Marini elements

  Level: intermediate

  Note:
  This type is a constructor alias of `PETSCDUALSPACELAGRANGE`.  During
  `PetscDualSpaceSetUp()`, the correct value of `PetscDualSpaceSetFormDegree()` is
  set for H-div conforming spaces. The type of the dual space is then changed to
  to `PETSCDUALSPACELAGRANGE`.

.seealso: `PetscDualSpaceType`, `PetscDualSpaceCreate()`, `PetscDualSpaceSetType()`, `PETSCDUALSPACELAGRANGE`, `PetscDualSpaceSetFormDegree()`
M*/

PETSC_EXTERN PetscFunctionList PetscDualSpaceList;
PETSC_EXTERN PetscErrorCode    PetscDualSpaceCreate(MPI_Comm, PetscDualSpace *);
PETSC_EXTERN PetscErrorCode    PetscDualSpaceDestroy(PetscDualSpace *);
PETSC_EXTERN PetscErrorCode    PetscDualSpaceDuplicate(PetscDualSpace, PetscDualSpace *);
PETSC_EXTERN PetscErrorCode    PetscDualSpaceSetType(PetscDualSpace, PetscDualSpaceType);
PETSC_EXTERN PetscErrorCode    PetscDualSpaceGetType(PetscDualSpace, PetscDualSpaceType *);
PETSC_EXTERN PetscErrorCode    PetscDualSpaceGetUniform(PetscDualSpace, PetscBool *);
PETSC_EXTERN PetscErrorCode    PetscDualSpaceGetNumDof(PetscDualSpace, const PetscInt **);
PETSC_EXTERN PetscErrorCode    PetscDualSpaceGetSection(PetscDualSpace, PetscSection *);
PETSC_EXTERN PetscErrorCode    PetscDualSpaceSetUp(PetscDualSpace);
PETSC_EXTERN PetscErrorCode    PetscDualSpaceSetFromOptions(PetscDualSpace);
PETSC_EXTERN PetscErrorCode    PetscDualSpaceViewFromOptions(PetscDualSpace, PetscObject, const char[]);

PETSC_EXTERN PetscErrorCode PetscDualSpaceView(PetscDualSpace, PetscViewer);
PETSC_EXTERN PetscErrorCode PetscDualSpaceRegister(const char[], PetscErrorCode (*)(PetscDualSpace));
PETSC_EXTERN PetscErrorCode PetscDualSpaceRegisterDestroy(void);

PETSC_EXTERN PetscErrorCode PetscDualSpaceGetDimension(PetscDualSpace, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscDualSpaceGetInteriorDimension(PetscDualSpace, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscDualSpaceSetNumComponents(PetscDualSpace, PetscInt);
PETSC_EXTERN PetscErrorCode PetscDualSpaceGetNumComponents(PetscDualSpace, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscDualSpaceSetOrder(PetscDualSpace, PetscInt);
PETSC_EXTERN PetscErrorCode PetscDualSpaceGetOrder(PetscDualSpace, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscDualSpaceSetDM(PetscDualSpace, DM);
PETSC_EXTERN PetscErrorCode PetscDualSpaceGetDM(PetscDualSpace, DM *);
PETSC_EXTERN PetscErrorCode PetscDualSpaceGetFunctional(PetscDualSpace, PetscInt, PetscQuadrature *);
PETSC_EXTERN PetscErrorCode PetscDualSpaceGetSymmetries(PetscDualSpace, const PetscInt ****, const PetscScalar ****);

PETSC_EXTERN PetscErrorCode PetscDualSpaceGetAllData(PetscDualSpace, PetscQuadrature *, Mat *);
PETSC_EXTERN PetscErrorCode PetscDualSpaceCreateAllDataDefault(PetscDualSpace, PetscQuadrature *, Mat *);
PETSC_EXTERN PetscErrorCode PetscDualSpaceGetInteriorData(PetscDualSpace, PetscQuadrature *, Mat *);
PETSC_EXTERN PetscErrorCode PetscDualSpaceCreateInteriorDataDefault(PetscDualSpace, PetscQuadrature *, Mat *);
PETSC_EXTERN PetscErrorCode PetscDualSpaceEqual(PetscDualSpace, PetscDualSpace, PetscBool *);

PETSC_EXTERN PetscErrorCode PetscDualSpaceApplyAll(PetscDualSpace, const PetscScalar *, PetscScalar *);
PETSC_EXTERN PetscErrorCode PetscDualSpaceApplyAllDefault(PetscDualSpace, const PetscScalar *, PetscScalar *);
PETSC_EXTERN PetscErrorCode PetscDualSpaceApplyInterior(PetscDualSpace, const PetscScalar *, PetscScalar *);
PETSC_EXTERN PetscErrorCode PetscDualSpaceApplyInteriorDefault(PetscDualSpace, const PetscScalar *, PetscScalar *);

PETSC_EXTERN PetscErrorCode PetscDualSpaceGetFormDegree(PetscDualSpace, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscDualSpaceSetFormDegree(PetscDualSpace, PetscInt);
PETSC_EXTERN PetscErrorCode PetscDualSpaceGetDeRahm(PetscDualSpace, PetscInt *);

PETSC_EXTERN PetscErrorCode PetscDualSpaceLagrangeGetContinuity(PetscDualSpace, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscDualSpaceLagrangeSetContinuity(PetscDualSpace, PetscBool);
PETSC_EXTERN PetscErrorCode PetscDualSpaceLagrangeGetTensor(PetscDualSpace, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscDualSpaceLagrangeSetTensor(PetscDualSpace, PetscBool);
PETSC_EXTERN PetscErrorCode PetscDualSpaceLagrangeGetTrimmed(PetscDualSpace, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscDualSpaceLagrangeSetTrimmed(PetscDualSpace, PetscBool);
PETSC_EXTERN PetscErrorCode PetscDualSpaceLagrangeGetNodeType(PetscDualSpace, PetscDTNodeType *, PetscBool *, PetscReal *);
PETSC_EXTERN PetscErrorCode PetscDualSpaceLagrangeSetNodeType(PetscDualSpace, PetscDTNodeType, PetscBool, PetscReal);
PETSC_EXTERN PetscErrorCode PetscDualSpaceLagrangeGetUseMoments(PetscDualSpace, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscDualSpaceLagrangeSetUseMoments(PetscDualSpace, PetscBool);
PETSC_EXTERN PetscErrorCode PetscDualSpaceLagrangeGetMomentOrder(PetscDualSpace, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscDualSpaceLagrangeSetMomentOrder(PetscDualSpace, PetscInt);

PETSC_EXTERN PetscErrorCode PetscDualSpaceGetHeightSubspace(PetscDualSpace, PetscInt, PetscDualSpace *);
PETSC_EXTERN PetscErrorCode PetscDualSpaceGetPointSubspace(PetscDualSpace, PetscInt, PetscDualSpace *);
PETSC_EXTERN PetscErrorCode PetscDualSpaceSimpleSetDimension(PetscDualSpace, PetscInt);
PETSC_EXTERN PetscErrorCode PetscDualSpaceSimpleSetFunctional(PetscDualSpace, PetscInt, PetscQuadrature);

PETSC_EXTERN PetscErrorCode PetscDualSpaceRefinedSetCellSpaces(PetscDualSpace, const PetscDualSpace[]);
PETSC_EXTERN PetscErrorCode PetscSpaceCreateSubspace(PetscSpace, PetscDualSpace, PetscReal *, PetscReal *, PetscReal *, PetscReal *, PetscCopyMode, PetscSpace *);

#endif
