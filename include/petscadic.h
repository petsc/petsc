/* $Id: petscadic.h,v 1.5 1998/03/24 20:38:58 balay Exp bsmith $ */

/*
        EXPERIMENTAL CODE - Use at your own risk.

    Defines objects and routines to use the PETSc/ADIC libraries in 
  application codes.
*/

#if !defined(__PETSCADIC_H)
#define __PETSCADIC_H
#include "mat.h"  
PETSC_EXTERN_CXX_BEGIN

EXTERN PetscErrorCode PETSC_DLLEXPORT ad_PetscInitialize(int *,char ***,char *,char*);
EXTERN PetscErrorCode PETSC_DLLEXPORT ad_PetscFinalize(void);
EXTERN PetscErrorCode PETSC_DLLEXPORT ad_AD_Init(void);
EXTERN PetscErrorCode PETSC_DLLEXPORT ad_AD_Final(void);

typedef struct _n_PetscADICFunction* PetscADICFunction;

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscADICFunctionCreate(Vec,Vec,PetscErrorCode (*)(Vec,Vec),PetscErrorCode (*)(void **),PetscADICFunction*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscADICFunctionInitialize(PetscADICFunction);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscADICFunctionEvaluateGradient(PetscADICFunction,Vec,Vec,Mat);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscADICFunctionApplyGradientInitialize(PetscADICFunction,Vec,Mat*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscADICFunctionApplyGradientReset(Mat,Vec);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscADICFunctionDestroy(PetscADICFunction);

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscADICFunctionSetFunction(PetscADICFunction,PetscErrorCode (*)(Vec,Vec),PetscErrorCode (*)(void **));
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscADICFunctionEvaluateGradientFD(PetscADICFunction,Vec,Vec,Mat);

PETSC_EXTERN_CXX_END
#endif
