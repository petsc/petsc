/* $Id: snes.h,v 1.95 1999/11/05 14:48:27 bsmith Exp bsmith $ */
/*
    User interface for the nonlinear solvers and unconstrained minimization package.
*/
#if !defined(__SNES_H)
#define __SNES_H
#include "sles.h"

typedef struct _p_SNES* SNES;
#define SNES_COOKIE         PETSC_COOKIE+13
#define MATSNESMFCTX_COOKIE PETSC_COOKIE+29

#define SNESEQLS          "ls"
#define SNESEQTR          "tr"
#define SNESEQTEST        "test"
#define SNESUMLS          "umls"
#define SNESUMTR          "umtr"

typedef char *SNESType;

typedef enum {SNES_NONLINEAR_EQUATIONS, SNES_UNCONSTRAINED_MINIMIZATION, SNES_LEAST_SQUARES} SNESProblemType;

extern int SNESCreate(MPI_Comm,SNESProblemType,SNES*);
extern int SNESDestroy(SNES);
extern int SNESSetType(SNES,SNESType);
extern int SNESSetMonitor(SNES,int(*)(SNES,int,double,void*),void *,int (*)(void *));
extern int SNESClearMonitor(SNES);
extern int SNESSetConvergenceHistory(SNES,double*,int *,int,PetscTruth);
extern int SNESGetConvergenceHistory(SNES,double**,int **,int *);
extern int SNESSetUp(SNES,Vec);
extern int SNESSolve(SNES,Vec,int*);

extern FList SNESList;
extern int SNESRegisterDestroy(void);
extern int SNESRegisterAll(char *);

extern int SNESRegister(char*,char*,char*,int(*)(SNES));
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define SNESRegisterDynamic(a,b,c,d) SNESRegister(a,b,c,0)
#else
#define SNESRegisterDynamic(a,b,c,d) SNESRegister(a,b,c,d)
#endif

extern int SNESGetSLES(SNES,SLES*);
extern int SNESGetSolution(SNES,Vec*);
extern int SNESGetSolutionUpdate(SNES,Vec*);
extern int SNESGetFunction(SNES,Vec*,void**);
extern int SNESPrintHelp(SNES);
extern int SNESView(SNES,Viewer);

extern int SNESSetOptionsPrefix(SNES,char*);
extern int SNESAppendOptionsPrefix(SNES,char*);
extern int SNESGetOptionsPrefix(SNES,char**);
extern int SNESSetFromOptions(SNES);
extern int SNESSetTypeFromOptions(SNES);
extern int SNESAddOptionsChecker(int (*)(SNES));

extern int MatCreateSNESMF(SNES,Vec,Mat*);
extern int MatSNESMFSetFunction(Mat,Vec,int(*)(SNES,Vec,Vec,void*),void *);
extern int MatSNESMFAddNullSpace(Mat,PCNullSpace);
extern int MatSNESMFSetHHistory(Mat,Scalar *,int);
extern int MatSNESMFResetHHistory(Mat);
extern int MatSNESMFSetFunctionError(Mat,double);
extern int MatSNESMFGetH(Mat,Scalar *);
extern int MatSNESMFKSPMonitor(KSP,int,double,void *);
extern int MatSNESMFSetFromOptions(Mat);
typedef struct _p_MatSNESMFCtx   *MatSNESMFCtx;
extern int MatSNESMFSetType(Mat,char*);
extern int MatSNESMFRegister(char *,char *,char *,int (*)(MatSNESMFCtx));
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define MatSNESMFRegisterDynamic(a,b,c,d) MatSNESMFRegister(a,b,c,0)
#else
#define MatSNESMFRegisterDynamic(a,b,c,d) MatSNESMFRegister(a,b,c,d)
#endif
extern int MatSNESMFRegisterAll(char *);
extern int MatSNESMFRegisterDestroy(void);
extern int MatSNESMFDefaultSetUmin(Mat,double);
extern int MatSNESMFWPSetComputeNormA(Mat,PetscTruth);
extern int MatSNESMFWPSetComputeNormU(Mat,PetscTruth);

extern int SNESGetType(SNES,SNESType*);
extern int SNESDefaultMonitor(SNES,int,double,void *);
extern int SNESVecViewMonitor(SNES,int,double,void *);
extern int SNESVecViewUpdateMonitor(SNES,int,double,void *);
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
extern int SNESLGMonitorCreate(char*,char*,int,int,int,int,DrawLG*);
extern int SNESLGMonitor(SNES,int,double,void*);
extern int SNESLGMonitorDestroy(DrawLG);

extern int SNESSetApplicationContext(SNES,void *);
extern int SNESGetApplicationContext(SNES,void **);

typedef enum {/* converged */
              SNES_CONVERGED_FNORM_ABS         =  2, /* F < F_minabs */
              SNES_CONVERGED_FNORM_RELATIVE    =  3, /* F < F_mintol*F_initial */
              SNES_CONVERGED_PNORM_RELATIVE    =  4, /* step size small */
              SNES_CONVERGED_GNORM_ABS         =  5, /* grad F < grad F_min */
              SNES_CONVERGED_TR_REDUCTION      =  6,
              SNES_CONVERGED_TR_DELTA          =  7,
              /* diverged */
              SNES_DIVERGED_FUNCTION_COUNT     = -2,  
              SNES_DIVERGED_FNORM_NAN          = -4, 
              SNES_DIVERGED_MAX_IT             = -5,
              SNES_DIVERGED_LS_FAILURE         = -6,
              SNES_DIVERGED_TR_REDUCTION       = -7,
              SNES_CONVERGED_ITERATING         =  0} SNESConvergedReason;

extern int SNESSetConvergenceTest(SNES,int (*)(SNES,double,double,double,SNESConvergedReason*,void*),void*);
extern int SNESConverged_UM_LS(SNES,double,double,double,SNESConvergedReason*,void*);
extern int SNESConverged_UM_TR(SNES,double,double,double,SNESConvergedReason*,void*);
extern int SNESConverged_EQ_LS(SNES,double,double,double,SNESConvergedReason*,void*);
extern int SNESConverged_EQ_TR(SNES,double,double,double,SNESConvergedReason*,void*);
extern int SNESGetConvergedReason(SNES,SNESConvergedReason*);

/* --------- Solving systems of nonlinear equations --------------- */
extern int SNESSetFunction(SNES,Vec,int(*)(SNES,Vec,Vec,void*),void *);
extern int SNESComputeFunction(SNES,Vec,Vec);
extern int SNESSetJacobian(SNES,Mat,Mat,int(*)(SNES,Vec,Mat*,Mat*,MatStructure*,void*),void *);
extern int SNESGetJacobian(SNES,Mat*,Mat*,void **);
extern int SNESDefaultComputeJacobian(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
extern int SNESDefaultComputeJacobianColor(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
extern int SNESSetLineSearch(SNES,int(*)(SNES,void*,Vec,Vec,Vec,Vec,Vec,double,double*,double*,int*),void*);
extern int SNESNoLineSearch(SNES,void*,Vec,Vec,Vec,Vec,Vec,double,double*,double*,int*);
extern int SNESNoLineSearchNoNorms(SNES,void*,Vec,Vec,Vec,Vec,Vec,double,double*,double*,int*);
extern int SNESCubicLineSearch(SNES,void*,Vec,Vec,Vec,Vec,Vec,double,double*,double*,int*);
extern int SNESQuadraticLineSearch(SNES,void*,Vec,Vec,Vec,Vec,Vec,double,double*,double*,int*);
extern int SNESSetLineSearchCheck(SNES,int(*)(SNES,void*,Vec,PetscTruth*),void*);
extern int SNESSetLineSearchParams(SNES, double, double, double);
extern int SNESGetLineSearchParams(SNES, double*, double*, double*);

/* --------- Unconstrained minimization routines --------------------------------*/
extern int SNESSetHessian(SNES,Mat,Mat,int(*)(SNES,Vec,Mat*,Mat*,MatStructure*,void*),void *);
extern int SNESGetHessian(SNES,Mat*,Mat*,void **);
extern int SNESDefaultComputeHessian(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
extern int SNESDefaultComputeHessianColor(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
extern int SNESSetGradient(SNES,Vec,int(*)(SNES,Vec,Vec,void*),void*);
extern int SNESGetGradient(SNES,Vec*,void**);
extern int SNESGetGradientNorm(SNES,Scalar*);
extern int SNESComputeGradient(SNES,Vec,Vec);
extern int SNESSetMinimizationFunction(SNES,int(*)(SNES,Vec,double*,void*),void*);
extern int SNESComputeMinimizationFunction(SNES,Vec,double*);
extern int SNESGetMinimizationFunction(SNES,double*,void**);
extern int SNESSetMinimizationFunctionTolerance(SNES,double);
extern int SNESLineSearchSetDampingParameter(SNES,Scalar*);


/* Should these 2 routines be private? */
extern int SNESComputeHessian(SNES,Vec,Mat*,Mat*,MatStructure*);
extern int SNESComputeJacobian(SNES,Vec,Mat*,Mat*,MatStructure*);

#endif

