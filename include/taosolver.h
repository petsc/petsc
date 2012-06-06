#ifndef __TAOSOLVER_H
#define __TAOSOLVER_H

#include "petsc.h"
#include "petscvec.h"
#include "petscmat.h"
#include "tao_version.h"
#include "tao_sys.h"
#include "tao_util.h"




typedef struct _p_TaoSolver*   TaoSolver;
#define TaoSolverType char*
PETSC_EXTERN PetscClassId TAOSOLVER_CLASSID;

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

PETSC_EXTERN const char **TaoSolverTerminationReasons;



#include "taolinesearch.h"


PETSC_EXTERN PetscErrorCode TaoInitialize(int*,char***,const char[], const char[]);
PETSC_EXTERN PetscErrorCode TaoFinalize();


PETSC_EXTERN PetscErrorCode TaoInitializePackage(const char []);

#if defined PETSC_USE_DYNAMIC_LIBRARIES
#define TaoSolverRegisterDynamic(a,b,c,d) TaoSolverRegister(a,b,c,0)
#else
#define TaoSolverRegisterDynamic(a,b,c,d) TaoSolverRegister(a,b,c,d)
#endif
PETSC_EXTERN PetscErrorCode TaoCreate(MPI_Comm,TaoSolver*);
PETSC_EXTERN PetscErrorCode TaoSetFromOptions(TaoSolver);
PETSC_EXTERN PetscErrorCode TaoSetFiniteDifferencesOptions(TaoSolver);
PETSC_EXTERN PetscErrorCode TaoSetUp(TaoSolver);
PETSC_EXTERN PetscErrorCode TaoSetType(TaoSolver, const TaoSolverType);
PETSC_EXTERN PetscErrorCode TaoGetType(TaoSolver, const TaoSolverType *);
PETSC_EXTERN PetscErrorCode TaoSetApplicationContext(TaoSolver, void*);
PETSC_EXTERN PetscErrorCode TaoGetApplicationContext(TaoSolver, void*);
PETSC_EXTERN PetscErrorCode TaoDestroy(TaoSolver*);

PETSC_EXTERN PetscErrorCode TaoSetOptionsPrefix(TaoSolver,const char []);
PETSC_EXTERN PetscErrorCode TaoView(TaoSolver, PetscViewer);

PETSC_EXTERN PetscErrorCode TaoSolve(TaoSolver);

PETSC_EXTERN PetscErrorCode TaoSolverRegister(const char [], const char[], const char[],  PetscErrorCode (*)(TaoSolver));
PETSC_EXTERN PetscErrorCode TaoSolverRegisterAll(const char[]);
PETSC_EXTERN PetscErrorCode TaoSolverRegisterDestroy(void);

PETSC_EXTERN PetscErrorCode TaoGetTerminationReason(TaoSolver,TaoSolverTerminationReason*);
PETSC_EXTERN PetscErrorCode TaoGetSolutionStatus(TaoSolver, PetscInt*, PetscReal*, PetscReal*, PetscReal*, PetscReal*, TaoSolverTerminationReason*);
PETSC_EXTERN PetscErrorCode TaoSetTerminationReason(TaoSolver,TaoSolverTerminationReason);
PETSC_EXTERN PetscErrorCode TaoSetInitialVector(TaoSolver, Vec);
PETSC_EXTERN PetscErrorCode TaoGetSolutionVector(TaoSolver, Vec*);
PETSC_EXTERN PetscErrorCode TaoGetGradientVector(TaoSolver, Vec*);
PETSC_EXTERN PetscErrorCode TaoSetObjectiveRoutine(TaoSolver, PetscErrorCode(*)(TaoSolver, Vec, PetscReal*,void*), void*);
PETSC_EXTERN PetscErrorCode TaoSetGradientRoutine(TaoSolver, PetscErrorCode(*)(TaoSolver, Vec, Vec, void*), void*);
PETSC_EXTERN PetscErrorCode TaoSetObjectiveAndGradientRoutine(TaoSolver, PetscErrorCode(*)(TaoSolver, Vec, PetscReal*, Vec, void*), void*);
PETSC_EXTERN PetscErrorCode TaoSetHessianMat(TaoSolver, Mat, Mat);
PETSC_EXTERN PetscErrorCode TaoSetHessianRoutine(TaoSolver,Mat,Mat, PetscErrorCode(*)(TaoSolver,Vec, Mat*, Mat*, MatStructure*, void*), void*);
PETSC_EXTERN PetscErrorCode TaoSetSeparableObjectiveRoutine(TaoSolver, Vec, PetscErrorCode(*)(TaoSolver, Vec, Vec, void*), void*);
PETSC_EXTERN PetscErrorCode TaoSetConstraintsRoutine(TaoSolver, Vec, PetscErrorCode(*)(TaoSolver, Vec, Vec, void*), void*);
PETSC_EXTERN PetscErrorCode TaoSetJacobianRoutine(TaoSolver,Mat,Mat, PetscErrorCode(*)(TaoSolver,Vec, Mat*, Mat*, MatStructure*, void*), void*);
PETSC_EXTERN PetscErrorCode TaoSetJacobianStateRoutine(TaoSolver,Mat,Mat,Mat, PetscErrorCode(*)(TaoSolver,Vec, Mat*, Mat*, Mat*, MatStructure*, void*), void*);
PETSC_EXTERN PetscErrorCode TaoSetJacobianDesignRoutine(TaoSolver,Mat,PetscErrorCode(*)(TaoSolver,Vec, Mat*, void*), void*);
PETSC_EXTERN PetscErrorCode TaoSetStateDesignIS(TaoSolver, IS, IS);

PETSC_EXTERN PetscErrorCode TaoComputeObjective(TaoSolver, Vec, PetscReal*);
PETSC_EXTERN PetscErrorCode TaoComputeSeparableObjective(TaoSolver, Vec, Vec);
PETSC_EXTERN PetscErrorCode TaoComputeGradient(TaoSolver, Vec, Vec);
PETSC_EXTERN PetscErrorCode TaoComputeObjectiveAndGradient(TaoSolver, Vec, PetscReal*, Vec);
PETSC_EXTERN PetscErrorCode TaoComputeConstraints(TaoSolver, Vec, Vec);
PETSC_EXTERN PetscErrorCode TaoDefaultComputeGradient(TaoSolver, Vec, Vec, void*);
PETSC_EXTERN PetscErrorCode TaoIsObjectiveDefined(TaoSolver,PetscBool*);
PETSC_EXTERN PetscErrorCode TaoIsGradientDefined(TaoSolver,PetscBool*);
PETSC_EXTERN PetscErrorCode TaoIsObjectiveAndGradientDefined(TaoSolver,PetscBool*);

PETSC_EXTERN PetscErrorCode TaoComputeHessian(TaoSolver, Vec, Mat*, Mat*, MatStructure*);
PETSC_EXTERN PetscErrorCode TaoComputeJacobian(TaoSolver, Vec, Mat*, Mat*, MatStructure*);
PETSC_EXTERN PetscErrorCode TaoComputeJacobianState(TaoSolver, Vec, Mat*, Mat*, Mat*, MatStructure*);
PETSC_EXTERN PetscErrorCode TaoComputeJacobianDesign(TaoSolver, Vec, Mat*);

PETSC_EXTERN PetscErrorCode TaoDefaultComputeHessian(TaoSolver, Vec, Mat*, Mat*, MatStructure*, void*);

PETSC_EXTERN PetscErrorCode TaoDefaultComputeHessianColor(TaoSolver, Vec, Mat*, Mat*, MatStructure*, void*);
PETSC_EXTERN PetscErrorCode TaoComputeDualVariables(TaoSolver, Vec, Vec);
PETSC_EXTERN PetscErrorCode TaoSetVariableBounds(TaoSolver, Vec, Vec);
PETSC_EXTERN PetscErrorCode TaoGetVariableBounds(TaoSolver, Vec*, Vec*);
PETSC_EXTERN PetscErrorCode TaoSetVariableBoundsRoutine(TaoSolver, PetscErrorCode(*)(TaoSolver, Vec, Vec, void*), void*);
PETSC_EXTERN PetscErrorCode TaoComputeVariableBounds(TaoSolver);

PETSC_EXTERN PetscErrorCode TaoGetTolerances(TaoSolver, PetscReal*, PetscReal*, PetscReal*, PetscReal*, PetscReal*);
PETSC_EXTERN PetscErrorCode TaoSetTolerances(TaoSolver, PetscReal, PetscReal, PetscReal, PetscReal, PetscReal);
PETSC_EXTERN PetscErrorCode TaoGetConstraintTolerances(TaoSolver, PetscReal*, PetscReal*);
PETSC_EXTERN PetscErrorCode TaoSetConstraintTolerances(TaoSolver, PetscReal, PetscReal);
PETSC_EXTERN PetscErrorCode TaoSetFunctionLowerBound(TaoSolver, PetscReal);
PETSC_EXTERN PetscErrorCode TaoSetInitialTrustRegionRadius(TaoSolver, PetscReal);
PETSC_EXTERN PetscErrorCode TaoSetMaximumIterations(TaoSolver, PetscInt);
PETSC_EXTERN PetscErrorCode TaoSetMaximumFunctionEvaluations(TaoSolver, PetscInt);
PETSC_EXTERN PetscErrorCode TaoGetFunctionLowerBound(TaoSolver, PetscReal*);
PETSC_EXTERN PetscErrorCode TaoGetInitialTrustRegionRadius(TaoSolver, PetscReal*);
PETSC_EXTERN PetscErrorCode TaoGetCurrentTrustRegionRadius(TaoSolver, PetscReal*);
PETSC_EXTERN PetscErrorCode TaoGetMaximumIterations(TaoSolver, PetscInt*);
PETSC_EXTERN PetscErrorCode TaoGetMaximumFunctionEvaluations(TaoSolver, PetscInt*);
PETSC_EXTERN PetscErrorCode TaoSetDefaultKSPType(TaoSolver, KSPType);
PETSC_EXTERN PetscErrorCode TaoSetDefaultPCType(TaoSolver, PCType);
PETSC_EXTERN PetscErrorCode TaoSetDefaultLineSearchType(TaoSolver, TaoLineSearchType);
PETSC_EXTERN PetscErrorCode TaoSetOptionsPrefix(TaoSolver, const char p[]);
PETSC_EXTERN PetscErrorCode TaoAppendOptionsPrefix(TaoSolver, const char p[]);
PETSC_EXTERN PetscErrorCode TaoGetOptionsPrefix(TaoSolver, const char *p[]);
PETSC_EXTERN PetscErrorCode TaoResetStatistics(TaoSolver);

PETSC_EXTERN PetscErrorCode TaoGetKSP(TaoSolver, KSP*);
PETSC_EXTERN PetscErrorCode TaoGetLineSearch(TaoSolver, TaoLineSearch*);

PETSC_EXTERN PetscErrorCode TaoSetHistory(TaoSolver,PetscReal*,PetscReal*,PetscReal*,PetscInt,PetscBool);
PETSC_EXTERN PetscErrorCode TaoGetHistory(TaoSolver,PetscReal**,PetscReal**,PetscReal**,PetscInt*);
PETSC_EXTERN PetscErrorCode TaoSetMonitor(TaoSolver, PetscErrorCode (*)(TaoSolver,void*),void *,PetscErrorCode (*)(void**));
PETSC_EXTERN PetscErrorCode TaoCancelMonitors(TaoSolver);
PETSC_EXTERN PetscErrorCode TaoDefaultMonitor(TaoSolver, void*);
PETSC_EXTERN PetscErrorCode TaoDefaultSMonitor(TaoSolver, void*);
PETSC_EXTERN PetscErrorCode TaoDefaultCMonitor(TaoSolver, void*);
PETSC_EXTERN PetscErrorCode TaoSolutionMonitor(TaoSolver, void*);
PETSC_EXTERN PetscErrorCode TaoSeparableObjectiveMonitor(TaoSolver, void*);
PETSC_EXTERN PetscErrorCode TaoGradientMonitor(TaoSolver, void*);
PETSC_EXTERN PetscErrorCode TaoStepDirectionMonitor(TaoSolver, void*);
PETSC_EXTERN PetscErrorCode TaoDrawSolutionMonitor(TaoSolver, void*);
PETSC_EXTERN PetscErrorCode TaoDrawStepMonitor(TaoSolver, void*);
PETSC_EXTERN PetscErrorCode TaoDrawGradientMonitor(TaoSolver, void*);
PETSC_EXTERN PetscErrorCode TaoAddLineSearchCounts(TaoSolver);

PETSC_EXTERN PetscErrorCode TaoDefaultConvergenceTest(TaoSolver,void*);
PETSC_EXTERN PetscErrorCode TaoSetConvergenceTest(TaoSolver, PetscErrorCode (*)(TaoSolver, void*),void *);

PETSC_EXTERN PetscErrorCode TaoSQPCONSetStateDesignIS(TaoSolver, IS, IS);
PETSC_EXTERN PetscErrorCode TaoLCLSetStateDesignIS(TaoSolver, IS, IS);
PetscErrorCode TaoMonitor(TaoSolver, PetscInt, PetscReal, PetscReal, PetscReal, PetscReal, TaoSolverTerminationReason*); 


#endif /* ifndef __TAOSOLVER_H */

