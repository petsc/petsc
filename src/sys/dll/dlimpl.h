#if !defined(_DLIMPL_H)
#define _DLIMPL_H

#include <petscsys.h>

typedef void* PetscDLHandle;

#define PETSC_DL_DECIDE   0
#define PETSC_DL_NOW      1
#define PETSC_DL_LOCAL    2

extern PetscErrorCode  PetscDLOpen(const char[],int,PetscDLHandle *);
extern PetscErrorCode  PetscDLClose(PetscDLHandle *);
extern PetscErrorCode  PetscDLSym(PetscDLHandle,const char[],void **);

#endif
