/*
    Defines an interface to the Matlab Engine from PETSc
*/

#if !defined(__PETSCMATLAB_H)
#define __PETSCMATLAB_H
PETSC_EXTERN_CXX_BEGIN

extern PetscClassId MATLABENGINE_CLASSID;

/*S
     PetscMatlabEngine - Object used to communicate with Matlab

   Level: intermediate

.seealso:  PetscMatlabEngineCreate(), PetscMatlabEngineDestroy(), PetscMatlabEngineEvaluate(),
           PetscMatlabEngineGetOutput(), PetscMatlabEnginePut(), PetscMatlabEngineGet(),
           PetscMatlabEnginePrintOutput(), PetscMatlabEnginePutArray(), PetscMatlabEngineGetArray(),
           PETSC_MATLAB_ENGINE_(), PETSC_MATLAB_ENGINE_SELF, PETSC_MATLAB_ENGINE_WORLD
S*/
typedef struct _p_PetscMatlabEngine* PetscMatlabEngine;

extern PetscErrorCode  PetscMatlabEngineCreate(MPI_Comm,const char[],PetscMatlabEngine*);
extern PetscErrorCode  PetscMatlabEngineDestroy(PetscMatlabEngine);
extern PetscErrorCode  PetscMatlabEngineEvaluate(PetscMatlabEngine,const char[],...);
extern PetscErrorCode  PetscMatlabEngineGetOutput(PetscMatlabEngine,char **);
extern PetscErrorCode  PetscMatlabEnginePrintOutput(PetscMatlabEngine,FILE*);
extern PetscErrorCode  PetscMatlabEnginePut(PetscMatlabEngine,PetscObject);
extern PetscErrorCode  PetscMatlabEngineGet(PetscMatlabEngine,PetscObject);
extern PetscErrorCode  PetscMatlabEnginePutArray(PetscMatlabEngine,int,int,const PetscScalar*,const char[]);
extern PetscErrorCode  PetscMatlabEngineGetArray(PetscMatlabEngine,int,int,PetscScalar*,const char[]);

extern PetscMatlabEngine  PETSC_MATLAB_ENGINE_(MPI_Comm);

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
