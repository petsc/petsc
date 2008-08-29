#ifndef __TAOSOLVER_H
#define __TAOSOLVER_H

#include "petscvec.h"
#include "tao.h"
PETSC_EXTERN_CXX_BEGIN

typedef struct _p_TaoSolver*   TaoSolver;
#define TaoSolverType char*
extern PetscCookie TAOSOLVER_DLLEXPORT TAOSOLVER_COOKIE;

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
              TAO_CONTINUE_ITERATING      =  0} TaoSolverConvergedReason;

extern const char **TaoSolverConvergedReasons;

EXTERN PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverInitializePackage(const char []);

#if defined PETSC_USE_DYNAMIC_LIBRARIES
#define TaoSolverRegisterDynamic(a,b,c,d) TaoSolverRegister(a,b,c,0)
#else
#define TaoSolverRegisterDynamic(a,b,c,d) TaoSolverRegister(a,b,c,d)
#endif

EXTERN PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverCreate(MPI_Comm,TaoSolver*);
EXTERN PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverSetFromOptions(TaoSolver);
EXTERN PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverSetUp(TaoSolver);
EXTERN PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverSetType(TaoSolver, const TaoSolverType);
EXTERN PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverGetType(TaoSolver, const TaoSolverType*);
EXTERN PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverDestroy(TaoSolver);
EXTERN PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverSetOptionsPrefix(TaoSolver,const char []);
EXTERN PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverView(TaoSolver, PetscViewer);

EXTERN PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverSolve(TaoSolver);

EXTERN PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverRegister(const char [], const char[], const char[],  PetscErrorCode (*)(TaoSolver));
EXTERN PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverRegisterAll(const char[]);
EXTERN PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverRegisterDestroy(void);

EXTERN PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverGetConvergedReason(TaoSolver,TaoSolverConvergedReason*);

EXTERN PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverSetInitialVector(TaoSolver, Vec);
EXTERN PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverSetObjective(TaoSolver, PetscErrorCode(*)(TaoSolver, Vec, PetscScalar*,void*), void*);
EXTERN PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverSetGradient(TaoSolver, PetscErrorCode(*)(TaoSolver, Vec, Vec, void*), void*);
EXTERN PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverSetObjectiveAndGradient(TaoSolver, PetscErrorCode(*)(TaoSolver, Vec, PetscScalar*, Vec, void*), void*);

EXTERN PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverComputeObjective(TaoSolver, Vec, PetscScalar*);
EXTERN PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverComputeGradient(TaoSolver, Vec, Vec);
EXTERN PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverComputeObjectiveAndGradient(TaoSolver, Vec, PetscScalar*, Vec);

EXTERN PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverGetTolerances(TaoSolver, PetscScalar*, PetscScalar*, PetscScalar*, PetscScalar*, PetscScalar*);
EXTERN PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverSetTolerances(TaoSolver, PetscScalar, PetscScalar, PetscScalar, PetscScalar, PetscScalar);

EXTERN PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverResetStatistics(TaoSolver);
EXTERN PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverSetDefaultMonitors(TaoSolver);

EXTERN PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverDefaultConvergenceTest(TaoSolver,void*);

PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverMonitor(TaoSolver, PetscScalar, PetscScalar, PetscScalar); 


PETSC_EXTERN_CXX_END
#endif /* ifndef __TAOSOLVER_H */

