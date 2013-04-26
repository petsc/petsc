
/*
        EXPERIMENTAL CODE - Use at your own risk.

    Defines objects and routines to use the PETSc/ADIC libraries in 
  application codes.
*/

#if !defined(__PETSCADIC_H)
#define __PETSCADIC_H
#include "mat.h"  
PETSC_EXTERN_CXX_BEGIN

extern PetscErrorCode  ad_PetscInitialize(int *,char ***,char *,char*);
extern PetscErrorCode  ad_PetscFinalize(void);
extern PetscErrorCode  ad_AD_Init(void);
extern PetscErrorCode  ad_AD_Final(void);

typedef struct _n_PetscADICFunction* PetscADICFunction;

extern PetscErrorCode  PetscADICFunctionCreate(Vec,Vec,PetscErrorCode (*)(Vec,Vec),PetscErrorCode (*)(void **),PetscADICFunction*);
extern PetscErrorCode  PetscADICFunctionInitialize(PetscADICFunction);
extern PetscErrorCode  PetscADICFunctionEvaluateGradient(PetscADICFunction,Vec,Vec,Mat);
extern PetscErrorCode  PetscADICFunctionApplyGradientInitialize(PetscADICFunction,Vec,Mat*);
extern PetscErrorCode  PetscADICFunctionApplyGradientReset(Mat,Vec);
extern PetscErrorCode  PetscADICFunctionDestroy(PetscADICFunction);

extern PetscErrorCode  PetscADICFunctionSetFunction(PetscADICFunction,PetscErrorCode (*)(Vec,Vec),PetscErrorCode (*)(void **));
extern PetscErrorCode  PetscADICFunctionEvaluateGradientFD(PetscADICFunction,Vec,Vec,Mat);

PETSC_EXTERN_CXX_END
#endif
