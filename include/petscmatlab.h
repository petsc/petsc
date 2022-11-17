/*
    Defines an interface to the MATLAB Engine from PETSc
*/
#ifndef PETSCMATLAB_H
#define PETSCMATLAB_H

/* SUBMANSEC = Sys */

PETSC_EXTERN PetscClassId MATLABENGINE_CLASSID;

/*S
     PetscMatlabEngine - Object used to communicate with MATLAB

   Level: intermediate

   Notes:
   `Mat`s transferred between PETSc and MATLAB and vis versa are transposed in the other space
   (this is because MATLAB uses compressed column format and PETSc uses compressed row format)

   One must `./configure` PETSc with  `--with-matlab [-with-matlab-dir=matlab_root_directory]` to
   use this capability

.seealso: `PetscMatlabEngineCreate()`, `PetscMatlabEngineDestroy()`, `PetscMatlabEngineEvaluate()`,
          `PetscMatlabEngineGetOutput()`, `PetscMatlabEnginePut()`, `PetscMatlabEngineGet()`,
          `PetscMatlabEnginePrintOutput()`, `PetscMatlabEnginePutArray()`, `PetscMatlabEngineGetArray()`,
          `PETSC_MATLAB_ENGINE_()`, `PETSC_MATLAB_ENGINE_SELF`, `PETSC_MATLAB_ENGINE_WORLD`
S*/
typedef struct _p_PetscMatlabEngine *PetscMatlabEngine;

PETSC_EXTERN PetscErrorCode PetscMatlabEngineCreate(MPI_Comm, const char[], PetscMatlabEngine *);
PETSC_EXTERN PetscErrorCode PetscMatlabEngineDestroy(PetscMatlabEngine *);
PETSC_EXTERN PetscErrorCode PetscMatlabEngineEvaluate(PetscMatlabEngine, const char[], ...);
PETSC_EXTERN PetscErrorCode PetscMatlabEngineGetOutput(PetscMatlabEngine, char **);
PETSC_EXTERN PetscErrorCode PetscMatlabEnginePrintOutput(PetscMatlabEngine, FILE *);
PETSC_EXTERN PetscErrorCode PetscMatlabEnginePut(PetscMatlabEngine, PetscObject);
PETSC_EXTERN PetscErrorCode PetscMatlabEngineGet(PetscMatlabEngine, PetscObject);
PETSC_EXTERN PetscErrorCode PetscMatlabEnginePutArray(PetscMatlabEngine, int, int, const PetscScalar *, const char[]);
PETSC_EXTERN PetscErrorCode PetscMatlabEngineGetArray(PetscMatlabEngine, int, int, PetscScalar *, const char[]);

PETSC_EXTERN PetscMatlabEngine PETSC_MATLAB_ENGINE_(MPI_Comm);

/*MC
  PETSC_MATLAB_ENGINE_WORLD - same as PETSC_MATLAB_ENGINE_(PETSC_COMM_WORLD)

  Level: intermediate

.seealso: `PetscMatlabEngine`, `PETSC_MATLAB_ENGINE_()`, `PETSC_MATLAB_ENGINE_SELF`
M*/
#define PETSC_MATLAB_ENGINE_WORLD PETSC_MATLAB_ENGINE_(PETSC_COMM_WORLD)

/*MC
  PETSC_MATLAB_ENGINE_SELF - same as PETSC_MATLAB_ENGINE_(PETSC_COMM_SELF)

  Level: intermediate

.seealso: `PetscMatlabEngine`, `PETSC_MATLAB_ENGINE_()`, `PETSC_MATLAB_ENGINE_WORLD`
M*/
#define PETSC_MATLAB_ENGINE_SELF PETSC_MATLAB_ENGINE_(PETSC_COMM_SELF)

#endif
