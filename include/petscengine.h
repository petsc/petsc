/* $Id: petscengine.h,v 1.4 2001/08/06 21:19:20 bsmith Exp $ */

/*
    Defines an interface to the Matlab Engine from PETSc
*/

#if !defined(__PETSCENGINE_H)
#define __PETSCENGINE_H

#define MATLABENGINE_COOKIE PETSC_COOKIE+12

typedef struct _p_PetscMatlabEngine* PetscMatlabEngine;

EXTERN int PetscMatlabEngineCreate(MPI_Comm,char*,PetscMatlabEngine*);
EXTERN int PetscMatlabEngineDestroy(PetscMatlabEngine);
EXTERN int PetscMatlabEngineEvaluate(PetscMatlabEngine,char*,...);
EXTERN int PetscMatlabEngineGetOutput(PetscMatlabEngine,char **);
EXTERN int PetscMatlabEnginePrintOutput(PetscMatlabEngine,FILE*);
EXTERN int PetscMatlabEnginePut(PetscMatlabEngine,PetscObject);
EXTERN int PetscMatlabEngineGet(PetscMatlabEngine,PetscObject);
EXTERN int PetscMatlabEnginePutArray(PetscMatlabEngine,int,int,PetscScalar*,char*);
EXTERN int PetscMatlabEngineGetArray(PetscMatlabEngine,int,int,PetscScalar*,char*);

EXTERN PetscMatlabEngine PETSC_MATLAB_ENGINE_(MPI_Comm);
#define MATLAB_ENGINE_SELF  PETSC_MATLAB_ENGINE_(PETSC_COMM_SELF)
#define MATLAB_ENGINE_WORLD PETSC_MATLAB_ENGINE_(PETSC_COMM_WORLD)

#endif






