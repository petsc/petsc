/* $Id: sles.h,v 1.14 1995/07/09 19:11:38 bsmith Exp curfman $ */

#if !defined(__SLES_PACKAGE)
#define __SLES_PACKAGE
#include "pc.h"
#include "ksp.h"

#define SLES_COOKIE PETSC_COOKIE+10

typedef struct _SLES* SLES;

extern int SLESCreate(MPI_Comm,SLES*);
extern int SLESGetPC(SLES,PC*);
extern int SLESGetKSP(SLES,KSP*);
extern int SLESSetOperators(SLES,Mat,Mat,MatStructure);
extern int SLESSolve(SLES,Vec,Vec,int*);
extern int SLESSetFromOptions(SLES);
extern int SLESDestroy(SLES);
extern int SLESPrintHelp(SLES);
extern int SLESView(SLES,Viewer);
extern int SLESSetOptionsPrefix(SLES,char*);
extern int SLESSetUp(SLES,Vec,Vec);
extern int PCBJacobiGetSubSLES(PC,int*,int*,SLES**);

#endif
