/* $Id: snes.h,v 1.83 1999/03/17 23:25:44 bsmith Exp bsmith $ */
/*
    User interface for the nonlinear solvers and unconstrained minimization package.
*/
#if !defined(__SNES_H)
#define __SNES_H
#include "sles.h"

typedef struct _p_SNES* SNES;
#define SNES_COOKIE PETSC_COOKIE+13

#define SNES_EQ_LS          "ls"
#define SNES_EQ_TR          "tr"
#define SNES_EQ_TR_DOG_LEG  
#define SNES_EQ_TR2_LIN
#define SNES_EQ_TEST        "test"
#define SNES_UM_LS          "umls"
#define SNES_UM_TR          "umtr"

typedef char *SNESType;

typedef enum {SNES_NONLINEAR_EQUATIONS, SNES_UNCONSTRAINED_MINIMIZATION, SNES_LEAST_SQUARES} SNESProblemType;

extern int SNESCreate(MPI_Comm,SNESProblemType,SNES*);
extern int SNESDestroy(SNES);
extern int SNESSetType(SNES,SNESType);
extern int SNESSetMonitor(SNES,int(*)(SNES,int,double,void*),void *);
extern int SNESClearMonitor(SNES);
extern int SNESSetConvergenceHistory(SNES,double*,int *,int,PetscTruth);
extern int SNESGetConvergenceHistory(SNES,double**,int **,int *);
extern int SNESSetUp(SNES,Vec);
extern int SNESSolve(SNES,Vec,int*);

extern FList SNESList;
extern int SNESRegisterDestroy(void);
extern int SNESRegisterAll(char *);

extern int SNESRegister_Private(char*,char*,char*,int(*)(SNES));
#if defined(USE_DYNAMIC_LIBRARIES)
#define SNESRegister(a,b,c,d) SNESRegister_Private(a,b,c,0)
#else
#define SNESRegister(a,b,c,d) SNESRegister_Private(a,b,c,d)
#endif

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
extern int SNESSetTypeFromOptions(SNES);
extern int SNESAddOptionsChecker(int (*)(SNES));

extern int MatCreateSNESMF(SNES,Vec,Mat*);
extern int MatSNESMFAddNullSpace(Mat,int,int,Vec *);
extern int MatSNESMFSetHHistory(Mat,Scalar *,int);
extern int MatSNESMFResetHHistory(Mat);
extern int MatSNESMFSetFunctionError(Mat,double);
extern int MatSNESMFGetH(Mat,Scalar *);
extern int MatSNESMFKSPMonitor(KSP,int,double,void *);
extern int MatSNESMFSetFromOptions(Mat);
typedef struct _p_MatSNESMFCtx   *MatSNESMFCtx;
extern int MatSNESMFSetType(Mat,char*);
extern int MatSNESMFRegister_Private(char *,char *,char *,int (*)(MatSNESMFCtx));
#if defined(USE_DYNAMIC_LIBRARIES)
#define MatSNESMFRegister(a,b,c,d) MatSNESMFRegister_Private(a,b,c,0)
#else
#define MatSNESMFRegister(a,b,c,d) MatSNESMFRegister_Private(a,b,c,d)
#endif
extern int MatSNESMFRegisterAll(char *);
extern int MatSNESMFRegisterDestroy(void);
extern int MatSNESMFDefaultSetUmin(Mat,double);
extern int MatSNESMFWPSetComputeNormA(Mat,PetscTruth);
extern int MatSNESMFWPSetComputeNormU(Mat,PetscTruth);

extern int SNESGetType(SNES,SNESType*);
extern int SNESDefaultMonitor(SNES,int,double,void *);
extern int SNESVecViewMonitor(SNES,int,double,void *);
extern int SNESDefaultSMonitor(SNES,int,double,void *);
extern int SNESSetTolerances(SNES,double,double,double,int,int);
extern int SNESGetTolerances(SNES,double*,double*,double*,int*,int*);
extern int SNESSetTrustRegionTolerance(SNES,double);
extern int SNESGetIterationNumber(SNES,int*);
extern int SNESGetFunctionNorm(SNES,Scalar*);
extern int SNESGetNumberUnsuccessfulSteps(SNES,int*);
extern int SNESGetNumberLinearIterations(SNES,int*);
extern int SNES_KSP_SetParametersEW(SNES,int,double,double,double,double,double,double);
extern int SNES_KSP_SetConvergenceTestEW(SNES);

/*
     Reuse the default KSP monitor routines for SNES
*/
#define SNESLGMonitorCreate  KSPLGMonitorCreate
#define SNESLGMonitorDestroy KSPLGMonitorDestroy
extern int SNESLGMonitor(SNES,int,double,void*);

extern int SNESSetApplicationContext(SNES,void *);
extern int SNESGetApplicationContext(SNES,void **);
extern int SNESSetConvergenceTest(SNES,int (*)(SNES,double,double,double,void*),void*);

/* --------- Solving systems of nonlinear equations --------------- */
extern int SNESSetFunction(SNES,Vec,int(*)(SNES,Vec,Vec,void*),void *);
extern int SNESComputeFunction(SNES,Vec,Vec);
extern int SNESSetJacobian(SNES,Mat,Mat,int(*)(SNES,Vec,Mat*,Mat*,MatStructure*,void*),void *);
extern int SNESGetJacobian(SNES,Mat*,Mat*,void **);
extern int SNESDefaultComputeJacobian(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
extern int SNESDefaultComputeJacobianColor(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
extern int SNESConverged_EQ_LS(SNES,double,double,double,void*);
extern int SNESConverged_EQ_TR(SNES,double,double,double,void*);
extern int SNESSetLineSearch(SNES,int(*)(SNES,void*,Vec,Vec,Vec,Vec,Vec,double,double*,double*,int*),void*);
extern int SNESNoLineSearch(SNES,void*,Vec,Vec,Vec,Vec,Vec,double,double*,double*,int*);
extern int SNESNoLineSearchNoNorms(SNES,void*,Vec,Vec,Vec,Vec,Vec,double,double*,double*,int*);
extern int SNESCubicLineSearch(SNES,void*,Vec,Vec,Vec,Vec,Vec,double,double*,double*,int*);
extern int SNESQuadraticLineSearch(SNES,void*,Vec,Vec,Vec,Vec,Vec,double,double*,double*,int*);
extern int SNESSetLineSearchCheck(SNES,int(*)(SNES,void*,Vec,PetscTruth*),void*);

/* --------- Unconstrained minimization routines --------------------------------*/
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
extern int SNESLineSearchSetDampingParameter(SNES,Scalar*);
extern int SNESConverged_UM_LS(SNES,double,double,double,void*);
extern int SNESConverged_UM_TR(SNES,double,double,double,void*);


/* Should these 2 routines be private? */
extern int SNESComputeHessian(SNES,Vec,Mat*,Mat*,MatStructure*);
extern int SNESComputeJacobian(SNES,Vec,Mat*,Mat*,MatStructure*);

#endif

