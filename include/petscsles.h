/* $Id: sles.h,v 1.29 1999/04/24 03:52:21 bsmith Exp bsmith $ */
/*
   Defines PETSc interface to the linear solvers. The details of Krylov methods
  and preconditioners are handled in the ksp.h and pc.h files
*/
#if !defined(__SLES_H)
#define __SLES_H
#include "pc.h"
#include "ksp.h"

#define SLES_COOKIE PETSC_COOKIE+10

typedef struct _p_SLES* SLES;

extern int SLESCreate(MPI_Comm,SLES*);
extern int SLESDestroy(SLES);

extern int SLESGetPC(SLES,PC*);
extern int SLESGetKSP(SLES,KSP*);
extern int SLESSetOperators(SLES,Mat,Mat,MatStructure);
extern int SLESSolve(SLES,Vec,Vec,int*);
extern int SLESSolveTrans(SLES,Vec,Vec,int*);
extern int SLESSetFromOptions(SLES);
extern int SLESSetTypesFromOptions(SLES);
extern int SLESPrintHelp(SLES);
extern int SLESView(SLES,Viewer);
extern int SLESSetUp(SLES,Vec,Vec);
extern int SLESSetUpOnBlocks(SLES);
extern int SLESSetDiagonalScale(SLES,PetscTruth);
extern int SLESGetDiagonalScale(SLES,PetscTruth*);
extern int SLESSetDiagonalScaleFix(SLES);

extern int SLESSetOptionsPrefix(SLES,char*);
extern int SLESAppendOptionsPrefix(SLES,char*);
extern int SLESGetOptionsPrefix(SLES,char**);

extern int PCBJacobiGetSubSLES(PC,int*,int*,SLES**);
extern int PCASMGetSubSLES(PC,int*,int*,SLES**);

#endif
