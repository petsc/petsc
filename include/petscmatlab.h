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

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscMatlabEngineCreate(MPI_Comm,const char[],PetscMatlabEngine*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscMatlabEngineDestroy(PetscMatlabEngine);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscMatlabEngineEvaluate(PetscMatlabEngine,const char[],...);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscMatlabEngineGetOutput(PetscMatlabEngine,char **);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscMatlabEnginePrintOutput(PetscMatlabEngine,FILE*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscMatlabEnginePut(PetscMatlabEngine,PetscObject);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscMatlabEngineGet(PetscMatlabEngine,PetscObject);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscMatlabEnginePutArray(PetscMatlabEngine,int,int,PetscScalar*,const char[]);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscMatlabEngineGetArray(PetscMatlabEngine,int,int,PetscScalar*,const char[]);

EXTERN PetscMatlabEngine PETSC_DLLEXPORT PETSC_MATLAB_ENGINE_(MPI_Comm);

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
