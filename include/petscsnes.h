/* $Id: snes.h,v 1.52 1996/08/09 23:32:39 curfman Exp curfman $ */
/*
    User interface for the nonlinear solvers package.
*/
#if !defined(__SNES_PACKAGE)
#define __SNES_PACKAGE
#include "sles.h"

typedef struct _SNES* SNES;
#define SNES_COOKIE PETSC_COOKIE+13

typedef enum { SNES_UNKNOWN_METHOD=-1,
               SNES_EQ_LS,
               SNES_EQ_TR,
               SNES_EQ_TR_DOG_LEG,
               SNES_EQ_TR2_LIN,
               SNES_EQ_TEST,
               SNES_UM_LS,
               SNES_UM_TR 
} SNESType;

typedef enum {SNES_NONLINEAR_EQUATIONS, SNES_UNCONSTRAINED_MINIMIZATION} SNESProblemType;
extern int SNESCreate(MPI_Comm,SNESProblemType,SNES*);
extern int SNESDestroy(SNES);
extern int SNESSetType(SNES,SNESType);
extern int SNESSetMonitor(SNES,int(*)(SNES,int,double,void*),void *);
extern int SNESSetUp(SNES,Vec);
extern int SNESSolve(SNES,Vec,int*);
extern int SNESRegister(int,char*,int(*)(SNES));
extern int SNESRegisterDestroy();
extern int SNESRegisterAll();
extern int SNESGetSLES(SNES,SLES*);
extern int SNESGetSolution(SNES,Vec*);
extern int SNESGetSolutionUpdate(SNES,Vec*);
extern int SNESGetFunction(SNES,Vec*);
extern int SNESPrintHelp(SNES);
extern int SNESView(SNES,Viewer);

extern int SNESSetOptionsPrefix(SNES,char*);
extern int SNESAppendOptionsPrefix(SNES,char*);
extern int SNESGetOptionsPrefix(SNES,char**);
extern int SNESSetFromOptions(SNES);

extern int SNESDefaultMatrixFreeMatCreate(SNES,Vec x,Mat*);
extern int SNESSetMatrixFreeParameters(SNES,double,double);
extern int SNESGetType(SNES,SNESType*,char**);
extern int SNESDefaultMonitor(SNES,int,double,void *);
extern int SNESDefaultSMonitor(SNES,int,double,void *);
extern int SNESSetTolerances(SNES,double,double,double,int,int);
extern int SNESGetTolerances(SNES,double*,double*,double*,int*,int*);
extern int SNESSetTrustRegionTolerance(SNES,double);
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

extern int SNESSetApplicationContext(SNES,void *);
extern int SNESGetApplicationContext(SNES,void **);
extern int SNESSetConvergenceTest(SNES,int (*)(SNES,double,double,double,void*),void*);

/* Routines for solving systems of nonlinear equations */
extern int SNESSetFunction(SNES,Vec,int(*)(SNES,Vec,Vec,void*),void *);
extern int SNESComputeFunction(SNES,Vec,Vec);
extern int SNESSetJacobian(SNES,Mat,Mat,int(*)(SNES,Vec,Mat*,Mat*,MatStructure*,void*),void *);
extern int SNESGetJacobian(SNES,Mat*,Mat*,void **);
extern int SNESDefaultComputeJacobian(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
extern int SNESConverged_EQ_LS(SNES,double,double,double,void*);
extern int SNESConverged_EQ_TR(SNES,double,double,double,void*);
extern int SNESNoLineSearch(SNES,Vec,Vec,Vec,Vec,Vec,double,double*,double*,int*);
extern int SNESCubicLineSearch(SNES,Vec,Vec,Vec,Vec,Vec,double,double*,double*,int*);
extern int SNESQuadraticLineSearch(SNES,Vec,Vec,Vec,Vec,Vec,double,double*,double*,int*);
extern int SNESSetLineSearch(SNES,int(*)(SNES,Vec,Vec,Vec,Vec,Vec,double,double*,double*,int*));

/* Unconstrained minimization routines */
extern int SNESSetHessian(SNES,Mat,Mat,int(*)(SNES,Vec,Mat*,Mat*,MatStructure*,void*),void *);
extern int SNESGetHessian(SNES,Mat*,Mat*,void **);
extern int SNESDefaultComputeHessian(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
extern int SNESSetGradient(SNES,Vec,int(*)(SNES,Vec,Vec,void*),void*);
extern int SNESGetGradient(SNES,Vec*);
extern int SNESGetGradientNorm(SNES,Scalar*);
extern int SNESComputeGradient(SNES,Vec,Vec);
extern int SNESSetMinimizationFunction(SNES,int(*)(SNES,Vec,double*,void*),void*);
extern int SNESComputeMinimizationFunction(SNES,Vec,double*);
extern int SNESGetMinimizationFunction(SNES,double*);
extern int SNESSetMinimizationFunctionTolerance(SNES,double);
extern int SNESGetLineSearchDampingParameter(SNES,Scalar*);
extern int SNESConverged_UM_LS(SNES,double,double,double,void*);
extern int SNESConverged_UM_TR(SNES,double,double,double,void*);

extern int SNESDefaultMatrixFreeMatAddNullSpace(Mat,int,int,Vec *);

/* Should these 2 routines be private? */
extern int SNESComputeHessian(SNES,Vec,Mat*,Mat*,MatStructure*);
extern int SNESComputeJacobian(SNES,Vec,Mat*,Mat*,MatStructure*);

#endif

