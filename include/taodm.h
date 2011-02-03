#ifndef __TAODM_H
#define __TAODM_H
#include "petscmat.h"
#include "petscdm.h"
#include "tao.h"


PETSC_EXTERN_CXX_BEGIN

typedef struct _p_TaoDM* TaoDM;
extern PetscClassId TAODM_CLASSID;

extern PetscErrorCode TaoDMSetMatType(TaoDM *, const MatType);
extern PetscErrorCode TaoDMCreate(MPI_Comm, PetscInt, void*, TaoDM**);
extern PetscErrorCode TaoDMSetSolverType(TaoDM*, const TaoSolverType);
extern PetscErrorCode TaoDMSetOptionsPrefix(TaoDM *, const char []);
extern PetscErrorCode TaoDMDestroy(TaoDM*);
extern PetscErrorCode TaoDMDestroyLevel(TaoDM);
extern PetscErrorCode TaoDMSetFromOptions(TaoDM*);
extern PetscErrorCode TaoDMSetTolerances(TaoDM*,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal);
extern PetscErrorCode TaoDMSetUp(TaoDM*);
extern PetscErrorCode TaoDMSolve(TaoDM*);
extern PetscErrorCode TaoDMView(TaoDM*, PetscViewer);
extern PetscErrorCode TaoDMSetDM(TaoDM *, DM);
extern PetscErrorCode TaoDMGetDM(TaoDM , DM*);
extern PetscErrorCode TaoDMGetContext(TaoDM , void**);

extern PetscErrorCode TaoDMSetLevelMonitor(TaoDM*,PetscErrorCode(*)(TaoDM, PetscInt, void*),void*);
extern PetscErrorCode TaoDMSetInitialGuessRoutine(TaoDM*,PetscErrorCode(*)(TaoDM,Vec));
extern PetscErrorCode TaoDMSetVariableBoundsRoutine(TaoDM*,PetscErrorCode(*)(TaoDM,Vec, Vec));
extern PetscErrorCode TaoDMSetObjectiveAndGradientRoutine(TaoDM*,PetscErrorCode(*)(TaoSolver, Vec, PetscReal*, Vec, void*));
extern PetscErrorCode TaoDMSetHessianRoutine(TaoDM*,PetscErrorCode(*)(TaoSolver, Vec, Mat*, Mat*, MatStructure*,void*));

extern PetscErrorCode TaoDMSetLocalObjectiveRoutine(TaoDM*,PetscErrorCode(*)(DMDALocalInfo*,PetscScalar**,PetscScalar*,void*));
extern PetscErrorCode TaoDMSetLocalGradientRoutine(TaoDM*,PetscErrorCode(*)(DMDALocalInfo*,PetscScalar**,PetscScalar**,void*));
extern PetscErrorCode TaoDMSetLocalObjectiveAndGradientRoutine(TaoDM*,PetscErrorCode(*)(DMDALocalInfo*,PetscScalar**,PetscScalar*,PetscScalar**,void*));
extern PetscErrorCode TaoDMSetLocalHessianRoutine(TaoDM*,PetscErrorCode(*)(DMDALocalInfo*,PetscScalar**,Mat,void*));


extern PetscErrorCode TaoDMFormFunctionLocal(TaoSolver, Vec, PetscScalar *, void*);
extern PetscErrorCode TaoDMFormGradientLocal(TaoSolver, Vec, Vec, void*);
extern PetscErrorCode TaoDMFormFunctionGradientLocal(TaoSolver, Vec, PetscScalar *, Vec, void*);
extern PetscErrorCode TaoDMFormHessianLocal(TaoSolver, Vec, Mat*, Mat*, MatStructure*,void*);
extern PetscErrorCode TaoDMFormBounds(TaoSolver, Vec, Vec, void*); 

PETSC_EXTERN_CXX_END			 
#endif
