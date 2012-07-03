/* $Id: spai.h,v 1.1 1997/02/03 00:11:31 bsmith Exp bsmith $ */
/*
     Include file for the SPAI interface to PETSc. You should include
  this file if you wish to set SPAI options directly from your program.
*/
#ifndef __SPAI_PACKAGE
#define __SPAI_PACKAGE
#include <petscpc.h>

extern int  MatDumpSPAI(Mat,FILE *);
extern int  VecDumpSPAI(Vec,FILE *);

extern int  PCSPAISetEpsilon(PC,double);
extern int  PCSPAISetNBSteps(PC,int);
extern int  PCSPAISetMaxAPI(PC,int);
extern int  PCSPAISetMaxNew(PC,int);
extern int  PCSPAISetCacheSize(PC,int);

#endif



