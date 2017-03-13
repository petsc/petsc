/* $Id: spai.h,v 1.1 1997/02/03 00:11:31 bsmith Exp bsmith $ */
/*
     Include file for the SPAI interface to PETSc. You should include
  this file if you wish to set SPAI options directly from your program.
*/
#if !defined(__SPAI_PACKAGE)
#define __SPAI_PACKAGE
#include <petscpc.h>

PETSC_EXTERN PetscErrorCode MatDumpSPAI(Mat,FILE*);
PETSC_EXTERN PetscErrorCode VecDumpSPAI(Vec,FILE*);

#endif



