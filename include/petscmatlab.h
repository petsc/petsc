/* $Id: petscmatlab.h,v 1.4 2001/08/06 21:19:20 bsmith Exp $ */

/*
    Defines an interface to the Matlab Engine from PETSc
*/

#if !defined(__PETSCMATLAB_H)
#define __PETSCMATLAB_H
PETSC_EXTERN_CXX_BEGIN

extern int MATLABENGINE_COOKIE;

/*S
     PetscMatlabEngine - Object used to communicate with Matlab

   Level: intermediate

.seealso:  PetscMatlabEngineCreate(), PetscMatlabEngineDestroy(), PetscMatlabEngineEvaluate(),
           PetscMatlabEngineGetOutput(), PetscMatlabEnginePut(), PetscMatlabEngineGet(),
           PetscMatlabEnginePrintOutput(), PetscMatlabEnginePutArray(), PetscMatlabEngineGetArray(),
           PETSC_MATLAB_ENGINE_(), PETSC_MATLAB_ENGINE_SELF, PETSC_MATLAB_ENGINE_WORLD
S*/
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

/*MC
  PETSC_MATLAB_ENGINE_WORLD - same as PETSC_MATLAB_ENGINE_(PETSC_COMM_WORLD)

  Level: developer
M*/
#define PETSC_MATLAB_ENGINE_WORLD PETSC_MATLAB_ENGINE_(PETSC_COMM_WORLD)

/*MC
  PETSC_MATLAB_ENGINE_SELF - same as PETSC_MATLAB_ENGINE_(PETSC_COMM_SELF)

  Level: developer
M*/
#define PETSC_MATLAB_ENGINE_SELF  PETSC_MATLAB_ENGINE_(PETSC_COMM_SELF)

PETSC_EXTERN_CXX_END
#endif
