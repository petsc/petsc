/* $Id: petscadic.h,v 1.1 1997/03/28 04:13:57 bsmith Exp bsmith $ */

/*
        EXPERIMENTAL CODE - Use at your own risk.

    Defines objects and routines to use the PETSc/ADIC libraries in 
  application codes.
*/

#if !defined(__PETSCADIC_PACKAGE)
#define __PETSCADIC_PACKAGE

#include "mat.h"  

extern int ad_PetscInitialize(int *,char ***,char *,char*);
extern int ad_PetscFinalize();
extern int ad_AD_Init();
extern int ad_AD_Final();

typedef struct _PetscADICFunction *PetscADICFunction;

extern int PetscADICFunctionCreate(Vec,Vec,int (*)(Vec,Vec),int (*)(void **),PetscADICFunction*);
extern int PetscADICFunctionInitialize(PetscADICFunction);
extern int PetscADICFunctionEvaluateGradient(PetscADICFunction,Vec,Vec,Mat);
extern int PetscADICFunctionApplyGradient(PetscADICFunction,Vec,Vec,Vec);
extern int PetscADICFunctionDestroy(PetscADICFunction);

#endif






