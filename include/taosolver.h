#ifndef __TAOSOLVER_H
#define __TAOSOLVER_H

#include "petscvec.h"
#include "petscmat.h"
#include "tao.h"
PETSC_EXTERN_CXX_BEGIN

typedef struct _p_TaoSolver*   TaoSolver;
#define TaoSolverType char*
extern PetscClassId TAOSOLVER_DLLEXPORT TAOSOLVER_CLASSID;

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
              TAO_CONTINUE_ITERATING      =  0} TaoSolverTerminationReason;

extern const char **TaoSolverTerminationReasons;
EXTERN PetscErrorCode TaoInitialize(int*,char***,const char[], const char[]);
EXTERN PetscErrorCode TaoFinalize();


EXTERN PetscErrorCode TaoInitialize_DynamicLibraries();
EXTERN PetscErrorCode TaoFinalize_DynamicLibraries();
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
EXTERN PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverGetType(TaoSolver, TaoSolverType *);
EXTERN PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverDestroy(TaoSolver);
EXTERN PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverSetOptionsPrefix(TaoSolver,const char []);
EXTERN PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverView(TaoSolver, PetscViewer);

EXTERN PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverSolve(TaoSolver);

EXTERN PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverRegister(const char [], const char[], const char[],  PetscErrorCode (*)(TaoSolver));
EXTERN PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverRegisterAll(const char[]);
EXTERN PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverRegisterDestroy(void);

EXTERN PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverGetConvergedReason(TaoSolver,TaoSolverTerminationReason*);

EXTERN PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverSetInitialVector(TaoSolver, Vec);
EXTERN PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverGetSolutionVector(TaoSolver, Vec*);
EXTERN PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverSetObjectiveRoutine(TaoSolver, PetscErrorCode(*)(TaoSolver, Vec, PetscReal*,void*), void*);
EXTERN PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverSetGradientRoutine(TaoSolver, PetscErrorCode(*)(TaoSolver, Vec, Vec, void*), void*);
EXTERN PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverSetObjectiveAndGradientRoutine(TaoSolver, PetscErrorCode(*)(TaoSolver, Vec, PetscReal*, Vec, void*), void*);
EXTERN PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverSetHessianMat(TaoSolver, Mat, Mat);
EXTERN PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverSetHessianRoutine(TaoSolver,Mat,Mat, PetscErrorCode(*)(TaoSolver,Vec, Mat*, Mat*, MatStructure*, void*), void*);
EXTERN PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverSetSeparableObjectiveRoutine(TaoSolver, Vec, PetscErrorCode(*)(TaoSolver, Vec, Vec, void*), void*);
EXTERN PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverSetJacobianRoutine(TaoSolver,Mat,Mat, PetscErrorCode(*)(TaoSolver,Vec, Mat*, Mat*, MatStructure*, void*), void*);

EXTERN PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverComputeObjective(TaoSolver, Vec, PetscReal*);
EXTERN PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverComputeSeparableObjective(TaoSolver, Vec, Vec);
EXTERN PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverComputeGradient(TaoSolver, Vec, Vec);
EXTERN PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverComputeObjectiveAndGradient(TaoSolver, Vec, PetscReal*, Vec);

EXTERN PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverComputeHessian(TaoSolver, Vec, Mat*, Mat*, MatStructure*);

EXTERN PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverDefaultComputeHessian(TaoSolver, Vec, Mat*, Mat*, MatStructure*, void*);

EXTERN PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverDefaultComputeHessianColor(TaoSolver, Vec, Mat*, Mat*, MatStructure*, void*);

EXTERN PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverComputeDualVariables(TaoSolver, Vec, Vec);
EXTERN PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverSetVariableBounds(TaoSolver, Vec, Vec);
EXTERN PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverGetVariableBounds(TaoSolver, Vec*, Vec*);

EXTERN PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverGetTolerances(TaoSolver, PetscReal*, PetscReal*, PetscReal*, PetscReal*, PetscReal*);
EXTERN PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverSetTolerances(TaoSolver, PetscReal, PetscReal, PetscReal, PetscReal, PetscReal);

EXTERN PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverResetStatistics(TaoSolver);
EXTERN PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverSetDefaultMonitors(TaoSolver);
EXTERN PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverGetKSP(TaoSolver, KSP*);
EXTERN PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverDefaultConvergenceTest(TaoSolver,void*);
EXTERN PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverSetMonitor(TaoSolver, PetscErrorCode (*)(TaoSolver,void*),void *);
EXTERN PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverDefaultMonitor(TaoSolver, void*);
EXTERN PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverDefaultSMonitor(TaoSolver, void*);
PetscErrorCode TAOSOLVER_DLLEXPORT TaoSolverMonitor(TaoSolver, PetscInt, PetscReal, PetscReal, PetscReal, PetscReal, TaoSolverTerminationReason*); 



PETSC_EXTERN_CXX_END
#endif /* ifndef __TAOSOLVER_H */

