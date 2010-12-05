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

extern PetscErrorCode PETSCSYS_DLLEXPORT ad_PetscInitialize(int *,char ***,char *,char*);
extern PetscErrorCode PETSCSYS_DLLEXPORT ad_PetscFinalize(void);
extern PetscErrorCode PETSCSYS_DLLEXPORT ad_AD_Init(void);
extern PetscErrorCode PETSCSYS_DLLEXPORT ad_AD_Final(void);

typedef struct _n_PetscADICFunction* PetscADICFunction;

extern PetscErrorCode PETSCSYS_DLLEXPORT PetscADICFunctionCreate(Vec,Vec,PetscErrorCode (*)(Vec,Vec),PetscErrorCode (*)(void **),PetscADICFunction*);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscADICFunctionInitialize(PetscADICFunction);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscADICFunctionEvaluateGradient(PetscADICFunction,Vec,Vec,Mat);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscADICFunctionApplyGradientInitialize(PetscADICFunction,Vec,Mat*);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscADICFunctionApplyGradientReset(Mat,Vec);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscADICFunctionDestroy(PetscADICFunction);

extern PetscErrorCode PETSCSYS_DLLEXPORT PetscADICFunctionSetFunction(PetscADICFunction,PetscErrorCode (*)(Vec,Vec),PetscErrorCode (*)(void **));
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscADICFunctionEvaluateGradientFD(PetscADICFunction,Vec,Vec,Mat);

PETSC_EXTERN_CXX_END
#endif
