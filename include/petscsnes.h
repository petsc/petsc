/* $Id: snes.h,v 1.27 1995/07/20 15:34:44 curfman Exp curfman $ */

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
               SNES_UM_NTR }
  SNESMethod;

#define SNES_NLS         SNES_EQ_NLS
#define SNES_NTR         SNES_EQ_NTR
#define SNES_NTR2_LIN    SNES_EQ_NTR2_LIN
#define SNES_NTR_DOG_LEG SNES_EQ_NTR_DOG_LEG
#define SNES_NTEST       SNES_EQ_NTEST

typedef enum { SNES_NONLINEAR_EQUATIONS, SNES_UNCONSTRAINED_MINIMIZATION } 
        SNESType;

extern int SNESCreate(MPI_Comm,SNESType,SNES*);
extern int SNESSetMethod(SNES,SNESMethod);
extern int SNESSetMonitor(SNES,int(*)(SNES,int,double,void*),void *);
extern int SNESSetSolution(SNES,Vec,int(*)(SNES,Vec,void*),void *);
extern int SNESSetFunction(SNES,Vec,int(*)(SNES,Vec,Vec,void*),void *,int);
extern int SNESSetJacobian(SNES,Mat,Mat,int(*)(SNES,Vec,Mat*,Mat*,MatStructure*,void*),void *);
extern int SNESDestroy(SNES);
extern int SNESSetUp(SNES);
extern int SNESSolve(SNES,int*);
extern int SNESRegister(int,char*,int(*)(SNES));
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
extern int SNESDefaultTrustRegionConverged(SNES,double,double,double,void*);

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

#if defined(__DRAW_PACKAGE)
#define SNESLGMonitorCreate  KSPLGMonitorCreate
#define SNESLGMonitorDestroy KSPLGMonitorDestroy
#define SNESLGMonitor        ((int (*)(SNES,int,double,void*))KSPLGMonitor)
#endif

extern int SNESComputeInitialGuess(SNES,Vec);

extern int SNESDefaultComputeJacobian(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
extern int SNESDefaultMatrixFreeComputeJacobian(SNES,Vec,Mat*,Mat*,
                                                MatStructure*,void*);
extern int SNESDefaultMatrixFreeMatCreate(SNES,Vec x,Mat*);

extern int SNESComputeFunction(SNES,Vec,Vec);
extern int SNESDestroy(SNES);


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

/* Should these 2 routines be private? */
extern int SNESComputeHessian(SNES,Vec,Mat*,Mat*,MatStructure*);
extern int SNESComputeJacobian(SNES,Vec,Mat*,Mat*,MatStructure*);

#endif

