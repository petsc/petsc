
/*
        EXPERIMENTAL CODE - Use at your own risk.

    Defines objects and routines to use the PETSc/ADIC libraries in 
  application codes.
*/

#if !defined(__PETSCADIC_H)
#define __PETSCADIC_H
#include "mat.h"  

PETSC_EXTERN PetscErrorCode ad_PetscInitialize(int *,char ***,char *,char*);
PETSC_EXTERN PetscErrorCode ad_PetscFinalize(void);
PETSC_EXTERN PetscErrorCode ad_AD_Init(void);
PETSC_EXTERN PetscErrorCode ad_AD_Final(void);

typedef struct _n_PetscADICFunction* PetscADICFunction;

PETSC_EXTERN PetscErrorCode PetscADICFunctionCreate(Vec,Vec,PetscErrorCode (*)(Vec,Vec),PetscErrorCode (*)(void **),PetscADICFunction*);
PETSC_EXTERN PetscErrorCode PetscADICFunctionInitialize(PetscADICFunction);
PETSC_EXTERN PetscErrorCode PetscADICFunctionEvaluateGradient(PetscADICFunction,Vec,Vec,Mat);
PETSC_EXTERN PetscErrorCode PetscADICFunctionApplyGradientInitialize(PetscADICFunction,Vec,Mat*);
PETSC_EXTERN PetscErrorCode PetscADICFunctionApplyGradientReset(Mat,Vec);
PETSC_EXTERN PetscErrorCode PetscADICFunctionDestroy(PetscADICFunction);

PETSC_EXTERN PetscErrorCode PetscADICFunctionSetFunction(PetscADICFunction,PetscErrorCode (*)(Vec,Vec),PetscErrorCode (*)(void **));
PETSC_EXTERN PetscErrorCode PetscADICFunctionEvaluateGradientFD(PetscADICFunction,Vec,Vec,Mat);

#endif
