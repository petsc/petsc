/* $Id: petscsles.h,v 1.35 2000/09/02 02:50:55 bsmith Exp bsmith $ */
/*
   Defines PETSc interface to the linear solvers. The details of Krylov methods
  and preconditioners are handled in the petscksp.h and petscpc.h files
*/
#if !defined(__PETSCSLES_H)
#define __PETSCSLES_H
#include "petscpc.h"
#include "petscksp.h"

#define SLES_COOKIE PETSC_COOKIE+10

typedef struct _p_SLES* SLES;

EXTERN int SLESCreate(MPI_Comm,SLES*);
EXTERN int SLESDestroy(SLES);

EXTERN int SLESGetPC(SLES,PC*);
EXTERN int SLESGetKSP(SLES,KSP*);
EXTERN int SLESSetOperators(SLES,Mat,Mat,MatStructure);
EXTERN int SLESSolve(SLES,Vec,Vec,int*);
EXTERN int SLESSolveTranspose(SLES,Vec,Vec,int*);
EXTERN int SLESSetFromOptions(SLES);
EXTERN int SLESSetTypesFromOptions(SLES);
EXTERN int SLESView(SLES,PetscViewer);
EXTERN int SLESSetUp(SLES,Vec,Vec);
EXTERN int SLESSetUpOnBlocks(SLES);
EXTERN int SLESSetDiagonalScale(SLES,PetscTruth);
EXTERN int SLESGetDiagonalScale(SLES,PetscTruth*);
EXTERN int SLESSetDiagonalScaleFix(SLES);

EXTERN int SLESSetOptionsPrefix(SLES,char*);
EXTERN int SLESAppendOptionsPrefix(SLES,char*);
EXTERN int SLESGetOptionsPrefix(SLES,char**);

EXTERN int PCBJacobiGetSubSLES(PC,int*,int*,SLES**);
EXTERN int PCASMGetSubSLES(PC,int*,int*,SLES**);
EXTERN int PCSLESGetSLES(PC,SLES *);

#endif
