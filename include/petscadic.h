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

EXTERN PetscErrorCode  ad_PetscInitialize(int *,char ***,char *,char*);
EXTERN PetscErrorCode  ad_PetscFinalize(void);
EXTERN PetscErrorCode  ad_AD_Init(void);
EXTERN PetscErrorCode  ad_AD_Final(void);

typedef struct _p_PetscADICFunction *PetscADICFunction;

EXTERN PetscErrorCode  PetscADICFunctionCreate(Vec,Vec,PetscErrorCode (*)(Vec,Vec),PetscErrorCode (*)(void **),PetscADICFunction*);
EXTERN PetscErrorCode  PetscADICFunctionInitialize(PetscADICFunction);
EXTERN PetscErrorCode  PetscADICFunctionEvaluateGradient(PetscADICFunction,Vec,Vec,Mat);
EXTERN PetscErrorCode  PetscADICFunctionApplyGradientInitialize(PetscADICFunction,Vec,Mat*);
EXTERN PetscErrorCode  PetscADICFunctionApplyGradientReset(Mat,Vec);
EXTERN PetscErrorCode  PetscADICFunctionDestroy(PetscADICFunction);

EXTERN PetscErrorCode  PetscADICFunctionSetFunction(PetscADICFunction,PetscErrorCode (*)(Vec,Vec),PetscErrorCode (*)(void **));
EXTERN PetscErrorCode  PetscADICFunctionEvaluateGradientFD(PetscADICFunction,Vec,Vec,Mat);

PETSC_EXTERN_CXX_END
#endif
