/*
      Objects which encapsulate finite element spaces and operations
*/
#if !defined(__PETSCFE_H)
#define __PETSCFE_H
#include <petscdm.h>
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

PETSC_EXTERN PetscFunctionList PetscSpaceList;
PETSC_EXTERN PetscBool         PetscSpaceRegisterAllCalled;
PETSC_EXTERN PetscErrorCode PetscSpaceCreate(MPI_Comm, PetscSpace *);
PETSC_EXTERN PetscErrorCode PetscSpaceDestroy(PetscSpace *);
PETSC_EXTERN PetscErrorCode PetscSpaceSetType(PetscSpace, PetscSpaceType);
PETSC_EXTERN PetscErrorCode PetscSpaceGetType(PetscSpace, PetscSpaceType *);
PETSC_EXTERN PetscErrorCode PetscSpaceSetFromOptions(PetscSpace);
PETSC_EXTERN PetscErrorCode PetscSpaceRegister(const char [], PetscErrorCode (*)(PetscSpace));
PETSC_EXTERN PetscErrorCode PetscSpaceRegisterAll(void);
PETSC_EXTERN PetscErrorCode PetscSpaceRegisterDestroy(void);

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
PETSC_EXTERN PetscErrorCode PetscDualSpaceSetFromOptions(PetscDualSpace);
PETSC_EXTERN PetscErrorCode PetscDualSpaceRegister(const char [], PetscErrorCode (*)(PetscDualSpace));
PETSC_EXTERN PetscErrorCode PetscDualSpaceRegisterAll(void);
PETSC_EXTERN PetscErrorCode PetscDualSpaceRegisterDestroy(void);

PETSC_EXTERN PetscErrorCode PetscDualSpaceGetDimension(PetscDualSpace, PetscInt *);

PETSC_EXTERN PetscClassId PETSCFE_CLASSID;

PETSC_EXTERN PetscErrorCode PetscFECreate(MPI_Comm, PetscFE *);
PETSC_EXTERN PetscErrorCode PetscFEDestroy(PetscFE *);
PETSC_EXTERN PetscErrorCode PetscFEGetTabulation(PetscFE, PetscInt, const PetscReal [], PetscReal **, PetscReal **, PetscReal **);
#endif
