/* $Id: snes.h,v 1.35 1995/12/13 15:59:10 curfman Exp bsmith $ */
/*
    User interface for the nonlinear solvers package.
*/
#if !defined(__SNES_PACKAGE)
#define __SNES_PACKAGE
#include "sles.h"

typedef struct _SNES* SNES;
#define SNES_COOKIE PETSC_COOKIE+13

typedef enum { SNES_UNKNOWN_METHOD=-1,
               SNES_EQ_NLS,
               SNES_EQ_NTR,
               SNES_EQ_NTR_DOG_LEG,
               SNES_EQ_NTR2_LIN,
               SNES_EQ_NTEST,
               SNES_UM_NLS,
               SNES_UM_NTR 
} SNESMethod;

typedef enum { POSITIVE_FUNCTION_VALUE, NEGATIVE_FUNCTION_VALUE} SNESFunctionSign;

typedef enum { SNES_NONLINEAR_EQUATIONS, SNES_UNCONSTRAINED_MINIMIZATION} SNESType;

extern int SNESCreate(MPI_Comm,SNESType,SNES*);
extern int SNESSetMethod(SNES,SNESMethod);
extern int SNESSetMonitor(SNES,int(*)(SNES,int,double,void*),void *);
extern int SNESSetSolution(SNES,Vec,int(*)(SNES,Vec,void*),void *);
extern int SNESSetFunction(SNES,Vec,int(*)(SNES,Vec,Vec,void*),void *,SNESFunctionSign);
extern int SNESSetJacobian(SNES,Mat,Mat,int(*)(SNES,Vec,Mat*,Mat*,MatStructure*,void*),void *);
extern int SNESDestroy(SNES);
extern int SNESSetUp(SNES);
extern int SNESSolve(SNES,int*);
extern int SNESRegister(int,char*,int(*)(SNES));
extern int SNESRegisterDestroy();
extern int SNESRegisterAll();
extern int SNESGetSLES(SNES,SLES*);
extern int SNESNoLineSearch(SNES,Vec,Vec,Vec,Vec,Vec,double,double*,double*,int*);
extern int SNESCubicLineSearch(SNES,Vec,Vec,Vec,Vec,Vec,double,double*,double*,int*);
extern int SNESQuadraticLineSearch(SNES,Vec,Vec,Vec,Vec,Vec,double,double*,double*,int*);
extern int SNESSetLineSearchRoutine(SNES,int(*)(SNES,Vec,Vec,Vec,Vec,Vec,double,double*,double*,int*));

extern int SNESGetSolution(SNES,Vec*);
extern int SNESGetSolutionUpdate(SNES,Vec*);
extern int SNESGetFunction(SNES,Vec*);

extern int SNESPrintHelp(SNES);
extern int SNESView(SNES,Viewer);
extern int SNESSetFromOptions(SNES);
extern int SNESGetMethodName(SNESMethod,char **);
extern int SNESGetMethodFromContext(SNES,SNESMethod*);
extern int SNESDefaultMonitor(SNES,int,double,void *);
extern int SNESDefaultSMonitor(SNES,int,double,void *);
extern int SNESDefaultConverged(SNES,double,double,double,void*);
extern int SNESTrustRegionDefaultConverged(SNES,double,double,double,void*);
extern int SNESSetSolutionTolerance(SNES,double);
extern int SNESSetAbsoluteTolerance(SNES,double);
extern int SNESSetRelativeTolerance(SNES,double);
extern int SNESSetTruncationTolerance(SNES,double);
extern int SNESSetTrustRegionTolerance(SNES,double);
extern int SNESSetMaxIterations(SNES,int);
extern int SNESSetMaxFunctionEvaluations(SNES,int);
extern int SNESGetIterationNumber(SNES,int*);
extern int SNESGetFunctionNorm(SNES,Scalar*);
extern int SNESGetNumberUnsuccessfulSteps(SNES,int*);
extern int SNES_KSP_SetParametersEW(SNES,int,double,double,double,double,double,double);
extern int SNES_KSP_SetConvergenceTestEW(SNES);

#if defined(__DRAW_PACKAGE)
#define SNESLGMonitorCreate  KSPLGMonitorCreate
#define SNESLGMonitorDestroy KSPLGMonitorDestroy
#define SNESLGMonitor        ((int (*)(SNES,int,double,void*))KSPLGMonitor)
#endif

extern int SNESComputeInitialGuess(SNES,Vec);

extern int SNESDefaultComputeJacobian(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
extern int SNESDefaultMatrixFreeMatCreate(SNES,Vec x,Mat*);

extern int SNESComputeFunction(SNES,Vec,Vec);
extern int SNESDestroy(SNES);

extern int SNESSetApplicationContext(SNES,void *);
extern int SNESGetApplicationContext(SNES,void **);
extern int SNESSetConvergenceTest(SNES,int (*)(SNES,double,double,double,void*),void*);

/* Unconstrained minimization routines */

extern int SNESSetHessian(SNES,Mat,Mat,int(*)(SNES,Vec,Mat*,Mat*,MatStructure*,void*),void *);
extern int SNESSetGradient(SNES,Vec,int(*)(SNES,Vec,Vec,void*),void*);
extern int SNESGetGradient(SNES,Vec*);
extern int SNESGetGradientNorm(SNES,Scalar*);
extern int SNESComputeGradient(SNES,Vec,Vec);
extern int SNESSetMinimizationFunction(SNES,int(*)(SNES,Vec,double*,void*),void*);
extern int SNESComputeMinimizationFunction(SNES,Vec,double*);
extern int SNESGetMinimizationFunction(SNES,double*);
extern int SNESSetMinFunctionTolerance(SNES,double);
extern int SNESGetLineSearchDampingParameter(SNES,double*);
extern int SNESConverged_UMLS(SNES,double,double,double,void*);
extern int SNESConverged_UMTR(SNES,double,double,double,void*);

/* Should these 2 routines be private? */
extern int SNESComputeHessian(SNES,Vec,Mat*,Mat*,MatStructure*);
extern int SNESComputeJacobian(SNES,Vec,Mat*,Mat*,MatStructure*);

extern int SNESDefaultMatrixFreeMatAddNullSpace(Mat,int,int,Vec *);

#endif

