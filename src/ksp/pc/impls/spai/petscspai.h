/* $Id: spai.h,v 1.1 1997/02/03 00:11:31 bsmith Exp bsmith $ */
/*
     Include file for the SPAI interface to PETSc. You should include
  this file if you wish to set SPAI options directly from your program.
*/
#ifndef __SPAI_PACKAGE
#define __SPAI_PACKAGE
#include "petscpc.h"

extern int PETSCKSP_DLLEXPORT MatDumpSPAI(Mat,FILE *);
extern int PETSCKSP_DLLEXPORT VecDumpSPAI(Vec,FILE *);

extern int PETSCKSP_DLLEXPORT PCSPAISetEpsilon(PC,double);
extern int PETSCKSP_DLLEXPORT PCSPAISetNBSteps(PC,int);
extern int PETSCKSP_DLLEXPORT PCSPAISetMaxAPI(PC,int);
extern int PETSCKSP_DLLEXPORT PCSPAISetMaxNew(PC,int);
extern int PETSCKSP_DLLEXPORT PCSPAISetCacheSize(PC,int);

#endif



