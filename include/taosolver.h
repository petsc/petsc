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
extern PetscErrorCode TaoInitializePackage(const char []);

#if defined PETSC_USE_DYNAMIC_LIBRARIES
#define TaoSolverRegisterDynamic(a,b,c,d) TaoSolverRegister(a,b,c,0)
#else
#define TaoSolverRegisterDynamic(a,b,c,d) TaoSolverRegister(a,b,c,d)
#endif
extern PetscErrorCode TaoCreate(MPI_Comm,TaoSolver*);
extern PetscErrorCode TaoSetFromOptions(TaoSolver);
extern PetscErrorCode TaoSetFiniteDifferencesOptions(TaoSolver);
extern PetscErrorCode TaoSetUp(TaoSolver);
extern PetscErrorCode TaoSetType(TaoSolver, const TaoSolverType);
extern PetscErrorCode TaoGetType(TaoSolver, const TaoSolverType *);

extern PetscErrorCode TaoDestroy(TaoSolver*);

extern PetscErrorCode TaoSetOptionsPrefix(TaoSolver,const char []);
extern PetscErrorCode TaoView(TaoSolver, PetscViewer);

extern PetscErrorCode TaoSolve(TaoSolver);

extern PetscErrorCode TaoSolverRegister(const char [], const char[], const char[],  PetscErrorCode (*)(TaoSolver));
extern PetscErrorCode TaoSolverRegisterAll(const char[]);
extern PetscErrorCode TaoSolverRegisterDestroy(void);

extern PetscErrorCode TaoGetTerminationReason(TaoSolver,TaoSolverTerminationReason*);
extern PetscErrorCode TaoGetSolutionStatus(TaoSolver, PetscInt*, PetscReal*, PetscReal*, PetscReal*, PetscReal*, TaoSolverTerminationReason*);
extern PetscErrorCode TaoSetTerminationReason(TaoSolver,TaoSolverTerminationReason);
extern PetscErrorCode TaoSetInitialVector(TaoSolver, Vec);
extern PetscErrorCode TaoGetSolutionVector(TaoSolver, Vec*);
extern PetscErrorCode TaoGetGradientVector(TaoSolver, Vec*);
extern PetscErrorCode TaoSetObjectiveRoutine(TaoSolver, PetscErrorCode(*)(TaoSolver, Vec, PetscReal*,void*), void*);
extern PetscErrorCode TaoSetGradientRoutine(TaoSolver, PetscErrorCode(*)(TaoSolver, Vec, Vec, void*), void*);
extern PetscErrorCode TaoSetObjectiveAndGradientRoutine(TaoSolver, PetscErrorCode(*)(TaoSolver, Vec, PetscReal*, Vec, void*), void*);
extern PetscErrorCode TaoSetHessianMat(TaoSolver, Mat, Mat);
extern PetscErrorCode TaoSetHessianRoutine(TaoSolver,Mat,Mat, PetscErrorCode(*)(TaoSolver,Vec, Mat*, Mat*, MatStructure*, void*), void*);
extern PetscErrorCode TaoSetSeparableObjectiveRoutine(TaoSolver, Vec, PetscErrorCode(*)(TaoSolver, Vec, Vec, void*), void*);
extern PetscErrorCode TaoSetConstraintsRoutine(TaoSolver, Vec, PetscErrorCode(*)(TaoSolver, Vec, Vec, void*), void*);
extern PetscErrorCode TaoSetJacobianRoutine(TaoSolver,Mat,Mat, PetscErrorCode(*)(TaoSolver,Vec, Mat*, Mat*, MatStructure*, void*), void*);
extern PetscErrorCode TaoSetJacobianStateRoutine(TaoSolver,Mat,Mat,Mat, PetscErrorCode(*)(TaoSolver,Vec, Mat*, Mat*, Mat*, MatStructure*, void*), void*);
extern PetscErrorCode TaoSetJacobianDesignRoutine(TaoSolver,Mat,PetscErrorCode(*)(TaoSolver,Vec, Mat*, void*), void*);
extern PetscErrorCode TaoSetStateDesignIS(TaoSolver, IS, IS);

extern PetscErrorCode TaoComputeObjective(TaoSolver, Vec, PetscReal*);
extern PetscErrorCode TaoComputeSeparableObjective(TaoSolver, Vec, Vec);
extern PetscErrorCode TaoComputeGradient(TaoSolver, Vec, Vec);
extern PetscErrorCode TaoComputeObjectiveAndGradient(TaoSolver, Vec, PetscReal*, Vec);
extern PetscErrorCode TaoComputeConstraints(TaoSolver, Vec, Vec);
extern PetscErrorCode TaoDefaultComputeGradient(TaoSolver, Vec, Vec, void*);
extern PetscErrorCode TaoIsObjectiveDefined(TaoSolver,PetscBool*);
extern PetscErrorCode TaoIsGradientDefined(TaoSolver,PetscBool*);
extern PetscErrorCode TaoIsObjectiveAndGradientDefined(TaoSolver,PetscBool*);

extern PetscErrorCode TaoComputeHessian(TaoSolver, Vec, Mat*, Mat*, MatStructure*);
extern PetscErrorCode TaoComputeJacobian(TaoSolver, Vec, Mat*, Mat*, MatStructure*);
extern PetscErrorCode TaoComputeJacobianState(TaoSolver, Vec, Mat*, Mat*, Mat*, MatStructure*);
extern PetscErrorCode TaoComputeJacobianDesign(TaoSolver, Vec, Mat*);

extern PetscErrorCode TaoDefaultComputeHessian(TaoSolver, Vec, Mat*, Mat*, MatStructure*, void*);

extern PetscErrorCode TaoDefaultComputeHessianColor(TaoSolver, Vec, Mat*, Mat*, MatStructure*, void*);
extern PetscErrorCode TaoComputeDualVariables(TaoSolver, Vec, Vec);
extern PetscErrorCode TaoSetVariableBounds(TaoSolver, Vec, Vec);
extern PetscErrorCode TaoGetVariableBounds(TaoSolver, Vec*, Vec*);
extern PetscErrorCode TaoSetVariableBoundsRoutine(TaoSolver, PetscErrorCode(*)(TaoSolver, Vec, Vec, void*), void*);
extern PetscErrorCode TaoComputeVariableBounds(TaoSolver);

extern PetscErrorCode TaoGetTolerances(TaoSolver, PetscReal*, PetscReal*, PetscReal*, PetscReal*, PetscReal*);
extern PetscErrorCode TaoSetTolerances(TaoSolver, PetscReal, PetscReal, PetscReal, PetscReal, PetscReal);
extern PetscErrorCode TaoSetFunctionLowerBound(TaoSolver, PetscReal);
extern PetscErrorCode TaoSetInitialTrustRegionRadius(TaoSolver, PetscReal);
extern PetscErrorCode TaoSetMaximumIterations(TaoSolver, PetscInt);
extern PetscErrorCode TaoSetMaximumFunctionEvaluations(TaoSolver, PetscInt);
extern PetscErrorCode TaoGetFunctionLowerBound(TaoSolver, PetscReal*);
extern PetscErrorCode TaoGetInitialTrustRegionRadius(TaoSolver, PetscReal*);
extern PetscErrorCode TaoGetCurrentTrustRegionRadius(TaoSolver, PetscReal*);
extern PetscErrorCode TaoGetMaximumIterations(TaoSolver, PetscInt*);
extern PetscErrorCode TaoGetMaximumFunctionEvaluations(TaoSolver, PetscInt*);
extern PetscErrorCode TaoSetDefaultKSPType(TaoSolver, KSPType);
extern PetscErrorCode TaoSetDefaultPCType(TaoSolver, PCType);
extern PetscErrorCode TaoSetDefaultLineSearchType(TaoSolver, TaoLineSearchType);
extern PetscErrorCode TaoSetOptionsPrefix(TaoSolver, const char p[]);
extern PetscErrorCode TaoAppendOptionsPrefix(TaoSolver, const char p[]);
extern PetscErrorCode TaoGetOptionsPrefix(TaoSolver, const char *p[]);
extern PetscErrorCode TaoResetStatistics(TaoSolver);

extern PetscErrorCode TaoGetKSP(TaoSolver, KSP*);
extern PetscErrorCode TaoGetLineSearch(TaoSolver, TaoLineSearch*);

extern PetscErrorCode TaoSetHistory(TaoSolver,PetscReal*,PetscReal*,PetscReal*,PetscInt,PetscBool);
extern PetscErrorCode TaoGetHistory(TaoSolver,PetscReal**,PetscReal**,PetscReal**,PetscInt*);
extern PetscErrorCode TaoSetMonitor(TaoSolver, PetscErrorCode (*)(TaoSolver,void*),void *,PetscErrorCode (*)(void**));
extern PetscErrorCode TaoCancelMonitors(TaoSolver);
extern PetscErrorCode TaoDefaultMonitor(TaoSolver, void*);
extern PetscErrorCode TaoDefaultSMonitor(TaoSolver, void*);
extern PetscErrorCode TaoDefaultCMonitor(TaoSolver, void*);
extern PetscErrorCode TaoSolutionMonitor(TaoSolver, void*);
extern PetscErrorCode TaoSeparableObjectiveMonitor(TaoSolver, void*);
extern PetscErrorCode TaoGradientMonitor(TaoSolver, void*);
extern PetscErrorCode TaoDrawSolutionMonitor(TaoSolver, void*);
extern PetscErrorCode TaoDrawStepMonitor(TaoSolver, void*);
extern PetscErrorCode TaoDrawGradientMonitor(TaoSolver, void*);
extern PetscErrorCode TaoAddLineSearchCounts(TaoSolver);

extern PetscErrorCode TaoDefaultConvergenceTest(TaoSolver,void*);
extern PetscErrorCode TaoSetConvergenceTest(TaoSolver, PetscErrorCode (*)(TaoSolver, void*),void *);

extern PetscErrorCode TaoSQPCONSetStateDesignIS(TaoSolver, IS, IS);
extern PetscErrorCode TaoLCLSetStateDesignIS(TaoSolver, IS, IS);
PetscErrorCode TaoMonitor(TaoSolver, PetscInt, PetscReal, PetscReal, PetscReal, PetscReal, TaoSolverTerminationReason*); 




PETSC_EXTERN_CXX_END
#endif /* ifndef __TAOSOLVER_H */

