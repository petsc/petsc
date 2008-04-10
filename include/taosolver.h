#ifndef __TAOSOLVER_H
#define __TAOSOLVER_H

#include "petscvec.h"
#include "tao.h"
PETSC_EXTERN_CXX_BEGIN

typedef struct _p_TaoSolver*   TaoSolver;
#define TaoSolverType const char*
extern PetscCookie PETSC_DLLEXPORT TAOSOLVER_COOKIE;

/*  Convergence flags.
    Be sure to check that these match the flags in
    $TAO_DIR/include/finclude/tao_solver.h
*/
typedef enum {/* converged */
              TAO_CONVERGED_ATOL          =  2, /* F < F_minabs */
              TAO_CONVERGED_RTOL          =  3, /* F < F_mintol*F_initial */
              TAO_CONVERGED_TRTOL         =  4, /* step size small */
              TAO_CONVERGED_MINF          =  5, /* grad F < grad F_min */
              TAO_CONVERGED_USER          =  6, /* User defined */
              /* diverged */
              TAO_DIVERGED_MAXITS         = -2,
              TAO_DIVERGED_NAN            = -4,
              TAO_DIVERGED_MAXFCN         = -5,
              TAO_DIVERGED_LS_FAILURE     = -6,
              TAO_DIVERGED_TR_REDUCTION   = -7,
              TAO_DIVERGED_USER           = -8, /* User defined */
              /* keep going */
              TAO_CONTINUE_ITERATING      =  0} TaoSolverTerminateReason;


EXTERN PetscErrorCode PETSC_DLLEXPORT TaoSolverCreate(MPI_Comm,TaoSolver*);
EXTERN PetscErrorCode PETSC_DLLEXPORT TaoSolverSetFromOptions(TaoSolver);
EXTERN PetscErrorCode PETSC_DLLEXPORT TaoSolverSetUp(TaoSolver);
EXTERN PetscErrorCode PETSC_DLLEXPORT TaoSolverSetType(TaoSolver, TaoSolverType);
EXTERN PetscErrorCode PETSC_DLLEXPORT TaoSolverDestroy(TaoSolver);
EXTERN PetscErrorCode PETSC_DLLEXPORT TaoSolverSetOptionsPrefix(TaoSolver,const char []);
EXTERN PetscErrorCode PETSC_DLLEXPORT TaoSolverView(TaoSolver, PetscViewer);

EXTERN PetscErrorCode PETSC_DLLEXPORT TaoSolverSolve(TaoSolver);
EXTERN PetscErrorCode PETSC_DLLEXPORT TaoSolverRegister(const char [], const char[], const char[], 
							PetscErrorCode (*)(TaoSolver));
EXTERN PetscErrorCode PETSC_DLLEXPORT TaoSolverGetConvergedReason(TaoSolver,TaoSolverTerminateReason*);

EXTERN PetscErrorCode PETSC_DLLEXPORT TaoSolverSetInitialVector(TaoSolver, Vec);
EXTERN PetscErrorCode PETSC_DLLEXPORT TaoSolverSetObjective(TaoSolver, PetscErrorCode(*)(TaoSolver, Vec, PetscReal*,void*), void*);
EXTERN PetscErrorCode PETSC_DLLEXPORT TaoSolverSetGradient(TaoSolver, PetscErrorCode(*)(TaoSolver, Vec, Vec, void*), void*);
EXTERN PetscErrorCode PETSC_DLLEXPORT TaoSolverSetObjectiveGradient(TaoSolver, PetscErrorCode(*)(TaoSolver, Vec, PetscReal*, Vec, void*), void*);

EXTERN PetscErrorCode PETSC_DLLEXPORT TaoSolverComputeObjective(TaoSolver, Vec, PetscReal*);
EXTERN PetscErrorCode PETSC_DLLEXPORT TaoSolverComputeGradient(TaoSolver, Vec, Vec);
EXTERN PetscErrorCode PETSC_DLLEXPORT TaoSolverComputeObjectiveGradient(TaoSolver, Vec, PetscReal*, Vec);

PETSC_EXTERN_CXX_END
#endif
