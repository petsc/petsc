/* $Id: petscengine.h,v 1.1 2000/04/18 23:08:28 bsmith Exp bsmith $ */

/*
    Defines an interface to the Matlab Engine from PETSc
*/

#if !defined(__PETSCENGINE_H)
#define __PETSCENGINE_H

#define MATLABENGINE_COOKIE PETSC_COOKIE+12

typedef struct _p_PetscMatlabEngine* PetscMatlabEngine;

extern int PetscMatlabEngineCreate(MPI_Comm,char*,PetscMatlabEngine*);
extern int PetscMatlabEngineDestroy(PetscMatlabEngine);
extern int PetscMatlabEngineEvaluate(PetscMatlabEngine,char*,...);
extern int PetscMatlabEngineGetOutput(PetscMatlabEngine,char **);
extern int PetscMatlabEnginePrintOutput(PetscMatlabEngine,FILE*);
extern int PetscMatlabEnginePut(PetscMatlabEngine,PetscObject);
extern int PetscMatlabEngineGet(PetscMatlabEngine,PetscObject);

extern PetscMatlabEngine MATLAB_ENGINE_(MPI_Comm);
#endif






