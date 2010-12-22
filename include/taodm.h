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
extern PetscErrorCode TaoDMSetOptionsPrefix(TaoDM *, const char []);
extern PetscErrorCode TaoDMDestroy(TaoDM*);
extern PetscErrorCode TaoDMSetFromOptions(TaoDM*);
extern PetscErrorCode TaoDMSetUp(TaoDM*);
extern PetscErrorCode TaoDMSolve(TaoDM*);
extern PetscErrorCode TaoDMView(TaoDM*, PetscViewer);
extern PetscErrorCode TaoDMSetDM(TaoDM *, DM);
extern PetscErrorCode TaoDMGetDM(TaoDM , DM*);
extern PetscErrorCode TaoDMGetContext(TaoDM , void**);

extern PetscErrorCode TaoDMSetInitialGuessRoutine(TaoDM*,PetscErrorCode(*)(TaoDM,Vec));
extern PetscErrorCode TaoDMSetVariableBoundsRoutine(TaoDM*,PetscErrorCode(*)(TaoDM,Vec, Vec));
extern PetscErrorCode TaoDMSetLocalObjectiveRoutine(TaoDM*,PetscErrorCode(*)(DMDALocalInfo*,PetscScalar**,PetscScalar*,void*));
extern PetscErrorCode TaoDMSetLocalGradientRoutine(TaoDM*,PetscErrorCode(*)(DMDALocalInfo*,PetscScalar**,PetscScalar**,void*));
extern PetscErrorCode TaoDMSetLocalObjectiveAndGradientRoutine(TaoDM*,PetscErrorCode(*)(DMDALocalInfo*,PetscScalar**,PetscScalar*,PetscScalar**,void*));
extern PetscErrorCode TaoDMSetLocalHessianRoutine(TaoDM*,PetscErrorCode(*)(DMDALocalInfo*,PetscScalar**,Mat,void*));


extern PetscErrorCode TaoDMFormFunction(TaoSolver, Vec, PetscScalar *, void*);
extern PetscErrorCode TaoDMFormGradient(TaoSolver, Vec, Vec, void*);
extern PetscErrorCode TaoDMFormFunctionGradient(TaoSolver, Vec, PetscScalar *, Vec, void*);
extern PetscErrorCode TaoDMFormBounds(TaoSolver, Vec, Vec, void*); 
extern PetscErrorCode TaoDMFormHessian(TaoSolver, Vec, Mat*, Mat*, MatStructure*,void*);

PETSC_EXTERN_CXX_END			 
#endif
