/*
    Defines an interface to the Matlab Engine from PETSc
*/

#if !defined(__PETSCMATLAB_H)
#define __PETSCMATLAB_H
PETSC_EXTERN_CXX_BEGIN

extern PetscCookie MATLABENGINE_COOKIE;

/*S
     PetscMatlabEngine - Object used to communicate with Matlab

   Level: intermediate

.seealso:  PetscMatlabEngineCreate(), PetscMatlabEngineDestroy(), PetscMatlabEngineEvaluate(),
           PetscMatlabEngineGetOutput(), PetscMatlabEnginePut(), PetscMatlabEngineGet(),
           PetscMatlabEnginePrintOutput(), PetscMatlabEnginePutArray(), PetscMatlabEngineGetArray(),
           PETSC_MATLAB_ENGINE_(), PETSC_MATLAB_ENGINE_SELF, PETSC_MATLAB_ENGINE_WORLD
S*/
typedef struct _p_PetscMatlabEngine* PetscMatlabEngine;

EXTERN PetscErrorCode PetscMatlabEngineCreate(MPI_Comm,const char[],PetscMatlabEngine*);
EXTERN PetscErrorCode PetscMatlabEngineDestroy(PetscMatlabEngine);
EXTERN PetscErrorCode PetscMatlabEngineEvaluate(PetscMatlabEngine,const char[],...);
EXTERN PetscErrorCode PetscMatlabEngineGetOutput(PetscMatlabEngine,char **);
EXTERN PetscErrorCode PetscMatlabEnginePrintOutput(PetscMatlabEngine,FILE*);
EXTERN PetscErrorCode PetscMatlabEnginePut(PetscMatlabEngine,PetscObject);
EXTERN PetscErrorCode PetscMatlabEngineGet(PetscMatlabEngine,PetscObject);
EXTERN PetscErrorCode PetscMatlabEnginePutArray(PetscMatlabEngine,int,int,PetscScalar*,const char[]);
EXTERN PetscErrorCode PetscMatlabEngineGetArray(PetscMatlabEngine,int,int,PetscScalar*,const char[]);

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
