/* $Id: sles.h,v 1.18 1996/03/01 00:01:56 balay Exp bsmith $ */

#if !defined(__SLES_PACKAGE)
#define __SLES_PACKAGE
#include "pc.h"
#include "ksp.h"

#define SLES_COOKIE PETSC_COOKIE+10

typedef struct _SLES* SLES;

extern int SLESCreate(MPI_Comm,SLES*);
extern int SLESDestroy(SLES);

extern int SLESGetPC(SLES,PC*);
extern int SLESGetKSP(SLES,KSP*);
extern int SLESSetOperators(SLES,Mat,Mat,MatStructure);
extern int SLESSolve(SLES,Vec,Vec,int*);
extern int SLESSetFromOptions(SLES);
extern int SLESPrintHelp(SLES);
extern int SLESView(SLES,Viewer);
extern int SLESSetUp(SLES,Vec,Vec);
extern int SLESSetUpOnBlocks(SLES);

extern int SLESSetOptionsPrefix(SLES,char*);
extern int SLESAppendOptionsPrefix(SLES,char*);
extern int SLESGetOptionsPrefix(SLES,char**);

extern int PCBJacobiGetSubSLES(PC,int*,int*,SLES**);
extern int PCBGSGetSubSLES(PC,int*,int*,SLES**);

#endif
