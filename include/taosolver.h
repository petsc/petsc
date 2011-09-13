#ifndef __TAOSOLVER_H
#define __TAOSOLVER_H

#include "petsc.h"
#include "petscvec.h"
#include "petscmat.h"
#include "tao_version.h"
#include "tao_sys.h"
#include "tao_util.h"



PETSC_EXTERN_CXX_BEGIN

typedef struct _p_TaoSolver*   TaoSolver;
#define TaoSolverType char*
extern PetscClassId TAOSOLVER_CLASSID;

/*  Convergence flags.
    Be sure to check that these match the flags in
    $TAO_DIR/include/finclude/tao_solver.h
*/
typedef enum {/* converged */
  TAO_CONVERGED_FATOL          =  1, /* f(X)-f(X*) <= fatol */
  TAO_CONVERGED_FRTOL          =  2, /* |F(X) - f(X*)|/|f(X)| < frtol */
  TAO_CONVERGED_GATOL          =  3, /* ||g(X)|| < gatol */
  TAO_CONVERGED_GRTOL          =  4, /* ||g(X)|| / f(X)  < grtol */
  TAO_CONVERGED_GTTOL          =  5, /* ||g(X)|| / ||g(X0)|| < gttol */ 
  TAO_CONVERGED_STEPTOL        =  6, /* step size small */
  TAO_CONVERGED_MINF          =  7, /* F < F_min */
  TAO_CONVERGED_USER          =  8, /* User defined */
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

PETSC_EXTERN_CXX_END


#include "taolinesearch.h"

PETSC_EXTERN_CXX_BEGIN

extern PetscErrorCode TaoInitialize(int*,char***,const char[], const char[]);
extern PetscErrorCode TaoFinalize();


extern PetscErrorCode TaoInitialize_DynamicLibraries();
extern PetscErrorCode TaoFinalize_DynamicLibraries();
extern PetscErrorCode TaoSolverInitializePackage(const char []);

#if defined PETSC_USE_DYNAMIC_LIBRARIES
#define TaoSolverRegisterDynamic(a,b,c,d) TaoSolverRegister(a,b,c,0)
#else
#define TaoSolverRegisterDynamic(a,b,c,d) TaoSolverRegister(a,b,c,d)
#endif
extern PetscErrorCode TaoSolverCreate(MPI_Comm,TaoSolver*);
extern PetscErrorCode TaoSolverSetFromOptions(TaoSolver);
extern PetscErrorCode TaoSolverSetFiniteDifferencesOptions(TaoSolver);
extern PetscErrorCode TaoSolverSetUp(TaoSolver);
extern PetscErrorCode TaoSolverSetType(TaoSolver, const TaoSolverType);
extern PetscErrorCode TaoSolverGetType(TaoSolver, const TaoSolverType *);

extern PetscErrorCode TaoSolverDestroy(TaoSolver*);

extern PetscErrorCode TaoSolverSetOptionsPrefix(TaoSolver,const char []);
extern PetscErrorCode TaoSolverView(TaoSolver, PetscViewer);

extern PetscErrorCode TaoSolverSolve(TaoSolver);

extern PetscErrorCode TaoSolverRegister(const char [], const char[], const char[],  PetscErrorCode (*)(TaoSolver));
extern PetscErrorCode TaoSolverRegisterAll(const char[]);
extern PetscErrorCode TaoSolverRegisterDestroy(void);

extern PetscErrorCode TaoSolverGetTerminationReason(TaoSolver,TaoSolverTerminationReason*);
extern PetscErrorCode TaoSolverGetSolutionStatus(TaoSolver, PetscInt*, PetscReal*, PetscReal*, PetscReal*, PetscReal*, TaoSolverTerminationReason*);
extern PetscErrorCode TaoSolverSetTerminationReason(TaoSolver,TaoSolverTerminationReason);
extern PetscErrorCode TaoSolverSetInitialVector(TaoSolver, Vec);
extern PetscErrorCode TaoSolverGetSolutionVector(TaoSolver, Vec*);
extern PetscErrorCode TaoSolverGetGradientVector(TaoSolver, Vec*);
extern PetscErrorCode TaoSolverSetObjectiveRoutine(TaoSolver, PetscErrorCode(*)(TaoSolver, Vec, PetscReal*,void*), void*);
extern PetscErrorCode TaoSolverSetGradientRoutine(TaoSolver, PetscErrorCode(*)(TaoSolver, Vec, Vec, void*), void*);
extern PetscErrorCode TaoSolverSetObjectiveAndGradientRoutine(TaoSolver, PetscErrorCode(*)(TaoSolver, Vec, PetscReal*, Vec, void*), void*);
extern PetscErrorCode TaoSolverSetHessianMat(TaoSolver, Mat, Mat);
extern PetscErrorCode TaoSolverSetHessianRoutine(TaoSolver,Mat,Mat, PetscErrorCode(*)(TaoSolver,Vec, Mat*, Mat*, MatStructure*, void*), void*);
extern PetscErrorCode TaoSolverSetSeparableObjectiveRoutine(TaoSolver, Vec, PetscErrorCode(*)(TaoSolver, Vec, Vec, void*), void*);
extern PetscErrorCode TaoSolverSetConstraintsRoutine(TaoSolver, Vec, PetscErrorCode(*)(TaoSolver, Vec, Vec, void*), void*);
extern PetscErrorCode TaoSolverSetJacobianRoutine(TaoSolver,Mat,Mat, PetscErrorCode(*)(TaoSolver,Vec, Mat*, Mat*, MatStructure*, void*), void*);
extern PetscErrorCode TaoSolverSetJacobianStateRoutine(TaoSolver,Mat,Mat,Mat, PetscErrorCode(*)(TaoSolver,Vec, Mat*, Mat*, Mat*, MatStructure*, void*), void*);
extern PetscErrorCode TaoSolverSetJacobianDesignRoutine(TaoSolver,Mat,Mat, PetscErrorCode(*)(TaoSolver,Vec, Mat*, Mat*, MatStructure*, void*), void*);
extern PetscErrorCode TaoSolverSetStateDesignIS(TaoSolver, IS, IS);

extern PetscErrorCode TaoSolverComputeObjective(TaoSolver, Vec, PetscReal*);
extern PetscErrorCode TaoSolverComputeSeparableObjective(TaoSolver, Vec, Vec);
extern PetscErrorCode TaoSolverComputeGradient(TaoSolver, Vec, Vec);
extern PetscErrorCode TaoSolverComputeObjectiveAndGradient(TaoSolver, Vec, PetscReal*, Vec);
extern PetscErrorCode TaoSolverComputeConstraints(TaoSolver, Vec, Vec);
extern PetscErrorCode TaoSolverDefaultComputeGradient(TaoSolver, Vec, Vec, void*);
extern PetscErrorCode TaoSolverIsObjectiveDefined(TaoSolver,PetscBool*);
extern PetscErrorCode TaoSolverIsGradientDefined(TaoSolver,PetscBool*);
extern PetscErrorCode TaoSolverIsObjectiveAndGradientDefined(TaoSolver,PetscBool*);

extern PetscErrorCode TaoSolverComputeHessian(TaoSolver, Vec, Mat*, Mat*, MatStructure*);
extern PetscErrorCode TaoSolverComputeJacobian(TaoSolver, Vec, Mat*, Mat*, MatStructure*);
extern PetscErrorCode TaoSolverComputeJacobianState(TaoSolver, Vec, Mat*, Mat*, Mat*, MatStructure*);
extern PetscErrorCode TaoSolverComputeJacobianDesign(TaoSolver, Vec, Mat*, Mat*, MatStructure*);

extern PetscErrorCode TaoSolverDefaultComputeHessian(TaoSolver, Vec, Mat*, Mat*, MatStructure*, void*);

extern PetscErrorCode TaoSolverDefaultComputeHessianColor(TaoSolver, Vec, Mat*, Mat*, MatStructure*, void*);
extern PetscErrorCode TaoSolverComputeDualVariables(TaoSolver, Vec, Vec);
extern PetscErrorCode TaoSolverSetVariableBounds(TaoSolver, Vec, Vec);
extern PetscErrorCode TaoSolverGetVariableBounds(TaoSolver, Vec*, Vec*);
extern PetscErrorCode TaoSolverSetVariableBoundsRoutine(TaoSolver, PetscErrorCode(*)(TaoSolver, Vec, Vec, void*), void*);
extern PetscErrorCode TaoSolverComputeVariableBounds(TaoSolver);

extern PetscErrorCode TaoSolverGetTolerances(TaoSolver, PetscReal*, PetscReal*, PetscReal*, PetscReal*, PetscReal*);
extern PetscErrorCode TaoSolverSetTolerances(TaoSolver, PetscReal, PetscReal, PetscReal, PetscReal, PetscReal);
extern PetscErrorCode TaoSolverSetFunctionLowerBound(TaoSolver, PetscReal);
extern PetscErrorCode TaoSolverSetInitialTrustRegionRadius(TaoSolver, PetscReal);
extern PetscErrorCode TaoSolverSetMaximumIterations(TaoSolver, PetscInt);
extern PetscErrorCode TaoSolverSetMaximumFunctionEvaluations(TaoSolver, PetscInt);
extern PetscErrorCode TaoSolverGetFunctionLowerBound(TaoSolver, PetscReal*);
extern PetscErrorCode TaoSolverGetInitialTrustRegionRadius(TaoSolver, PetscReal*);
extern PetscErrorCode TaoSolverGetCurrentTrustRegionRadius(TaoSolver, PetscReal*);
extern PetscErrorCode TaoSolverGetMaximumIterations(TaoSolver, PetscInt*);
extern PetscErrorCode TaoSolverGetMaximumFunctionEvaluations(TaoSolver, PetscInt*);
extern PetscErrorCode TaoSolverSetDefaultKSPType(TaoSolver, KSPType);
extern PetscErrorCode TaoSolverSetDefaultPCType(TaoSolver, PCType);
extern PetscErrorCode TaoSolverSetDefaultLineSearchType(TaoSolver, TaoLineSearchType);
extern PetscErrorCode TaoSolverSetOptionsPrefix(TaoSolver, const char p[]);
extern PetscErrorCode TaoSolverAppendOptionsPrefix(TaoSolver, const char p[]);
extern PetscErrorCode TaoSolverGetOptionsPrefix(TaoSolver, const char *p[]);
extern PetscErrorCode TaoSolverResetStatistics(TaoSolver);

extern PetscErrorCode TaoSolverGetKSP(TaoSolver, KSP*);
extern PetscErrorCode TaoSolverGetLineSearch(TaoSolver, TaoLineSearch*);

extern PetscErrorCode TaoSolverSetHistory(TaoSolver,PetscReal*,PetscReal*,PetscReal*,PetscInt,PetscBool);
extern PetscErrorCode TaoSolverGetHistory(TaoSolver,PetscReal**,PetscReal**,PetscReal**,PetscInt*);
extern PetscErrorCode TaoSolverSetMonitor(TaoSolver, PetscErrorCode (*)(TaoSolver,void*),void *,PetscErrorCode (*)(void**));
extern PetscErrorCode TaoSolverCancelMonitors(TaoSolver);
extern PetscErrorCode TaoSolverDefaultMonitor(TaoSolver, void*);
extern PetscErrorCode TaoSolverDefaultSMonitor(TaoSolver, void*);
extern PetscErrorCode TaoSolverDefaultCMonitor(TaoSolver, void*);
extern PetscErrorCode TaoSolverSolutionMonitor(TaoSolver, void*);
extern PetscErrorCode TaoSolverSeparableObjectiveMonitor(TaoSolver, void*);
extern PetscErrorCode TaoSolverGradientMonitor(TaoSolver, void*);
extern PetscErrorCode TaoSolverDrawSolutionMonitor(TaoSolver, void*);
extern PetscErrorCode TaoSolverDrawStepMonitor(TaoSolver, void*);
extern PetscErrorCode TaoSolverDrawGradientMonitor(TaoSolver, void*);


extern PetscErrorCode TaoSolverDefaultConvergenceTest(TaoSolver,void*);
extern PetscErrorCode TaoSolverSetConvergenceTest(TaoSolver, PetscErrorCode (*)(TaoSolver, void*),void *);

extern PetscErrorCode TaoSolverSQPCONSetStateDesignIS(TaoSolver, IS, IS);
extern PetscErrorCode TaoSolverLCLSetStateDesignIS(TaoSolver, IS, IS);
PetscErrorCode TaoSolverMonitor(TaoSolver, PetscInt, PetscReal, PetscReal, PetscReal, PetscReal, TaoSolverTerminationReason*); 




PETSC_EXTERN_CXX_END
#endif /* ifndef __TAOSOLVER_H */

