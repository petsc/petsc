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

extern int ad_PetscInitialize(int *,char ***,char *,char*);
extern int ad_PetscFinalize(void);
extern int ad_AD_Init(void);
extern int ad_AD_Final(void);

typedef struct _p_PetscADICFunction *PetscADICFunction;

extern int PetscADICFunctionCreate(Vec,Vec,int (*)(Vec,Vec),int (*)(void **),PetscADICFunction*);
extern int PetscADICFunctionInitialize(PetscADICFunction);
extern int PetscADICFunctionEvaluateGradient(PetscADICFunction,Vec,Vec,Mat);
extern int PetscADICFunctionApplyGradientInitialize(PetscADICFunction,Vec,Mat*);
extern int PetscADICFunctionApplyGradientReset(Mat,Vec);
extern int PetscADICFunctionDestroy(PetscADICFunction);

extern int PetscADICFunctionSetFunction(PetscADICFunction,int (*)(Vec,Vec),int (*)(void **));
extern int PetscADICFunctionEvaluateGradientFD(PetscADICFunction,Vec,Vec,Mat);

PETSC_EXTERN_CXX_END
#endif
