/* $Id: petscsnes.h,v 1.101 2000/07/21 19:40:49 bsmith Exp bsmith $ */
/*
    User interface for the nonlinear solvers and unconstrained minimization package.
*/
#if !defined(__PETSCSNES_H)
#define __PETSCSNES_H
#include "petscsles.h"

typedef struct _p_SNES* SNES;
#define SNES_COOKIE         PETSC_COOKIE+13
#define MATSNESMFCTX_COOKIE PETSC_COOKIE+29

#define SNESEQLS          "ls"
#define SNESEQTR          "tr"
#define SNESEQTEST        "test"
#define SNESUMLS          "umls"
#define SNESUMTR          "umtr"

typedef char *SNESType;

typedef enum {SNES_NONLINEAR_EQUATIONS,SNES_UNCONSTRAINED_MINIMIZATION,SNES_LEAST_SQUARES} SNESProblemType;

EXTERN int SNESCreate(MPI_Comm,SNESProblemType,SNES*);
EXTERN int SNESDestroy(SNES);
EXTERN int SNESSetType(SNES,SNESType);
EXTERN int SNESSetMonitor(SNES,int(*)(SNES,int,double,void*),void *,int (*)(void *));
EXTERN int SNESClearMonitor(SNES);
EXTERN int SNESSetConvergenceHistory(SNES,double*,int *,int,PetscTruth);
EXTERN int SNESGetConvergenceHistory(SNES,double**,int **,int *);
EXTERN int SNESSetUp(SNES,Vec);
EXTERN int SNESSolve(SNES,Vec,int*);

extern FList SNESList;
EXTERN int SNESRegisterDestroy(void);
EXTERN int SNESRegisterAll(char *);

EXTERN int SNESRegister(char*,char*,char*,int(*)(SNES));
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define SNESRegisterDynamic(a,b,c,d) SNESRegister(a,b,c,0)
#else
#define SNESRegisterDynamic(a,b,c,d) SNESRegister(a,b,c,d)
#endif

EXTERN int SNESGetSLES(SNES,SLES*);
EXTERN int SNESGetSolution(SNES,Vec*);
EXTERN int SNESGetSolutionUpdate(SNES,Vec*);
EXTERN int SNESGetFunction(SNES,Vec*,void**,int(**)(SNES,Vec,Vec,void*));
EXTERN int SNESPrintHelp(SNES);
EXTERN int SNESView(SNES,Viewer);

EXTERN int SNESSetOptionsPrefix(SNES,char*);
EXTERN int SNESAppendOptionsPrefix(SNES,char*);
EXTERN int SNESGetOptionsPrefix(SNES,char**);
EXTERN int SNESSetFromOptions(SNES);
EXTERN int SNESSetTypeFromOptions(SNES);
EXTERN int SNESAddOptionsChecker(int (*)(SNES));

#define MATSNESMF_DEFAULT "default"
#define MATSNESMF_WP      "wp"
EXTERN int MatCreateSNESMF(SNES,Vec,Mat*);
EXTERN int MatSNESMFSetFunction(Mat,Vec,int(*)(SNES,Vec,Vec,void*),void *);
EXTERN int MatSNESMFAddNullSpace(Mat,PCNullSpace);
EXTERN int MatSNESMFSetHHistory(Mat,Scalar *,int);
EXTERN int MatSNESMFResetHHistory(Mat);
EXTERN int MatSNESMFSetFunctionError(Mat,double);
EXTERN int MatSNESMFSetPeriod(Mat,int);
EXTERN int MatSNESMFGetH(Mat,Scalar *);
EXTERN int MatSNESMFKSPMonitor(KSP,int,double,void *);
EXTERN int MatSNESMFSetFromOptions(Mat);
typedef struct _p_MatSNESMFCtx   *MatSNESMFCtx;
typedef char* MatSNESMFType;
EXTERN int MatSNESMFSetType(Mat,MatSNESMFType);
EXTERN int MatSNESMFRegister(char *,char *,char *,int (*)(MatSNESMFCtx));
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define MatSNESMFRegisterDynamic(a,b,c,d) MatSNESMFRegister(a,b,c,0)
#else
#define MatSNESMFRegisterDynamic(a,b,c,d) MatSNESMFRegister(a,b,c,d)
#endif
EXTERN int MatSNESMFRegisterAll(char *);
EXTERN int MatSNESMFRegisterDestroy(void);
EXTERN int MatSNESMFDefaultSetUmin(Mat,double);
EXTERN int MatSNESMFWPSetComputeNormA(Mat,PetscTruth);
EXTERN int MatSNESMFWPSetComputeNormU(Mat,PetscTruth);

EXTERN int SNESGetType(SNES,SNESType*);
EXTERN int SNESDefaultMonitor(SNES,int,double,void *);
EXTERN int SNESVecViewMonitor(SNES,int,double,void *);
EXTERN int SNESVecViewUpdateMonitor(SNES,int,double,void *);
EXTERN int SNESDefaultSMonitor(SNES,int,double,void *);
EXTERN int SNESSetTolerances(SNES,double,double,double,int,int);
EXTERN int SNESGetTolerances(SNES,double*,double*,double*,int*,int*);
EXTERN int SNESSetTrustRegionTolerance(SNES,double);
EXTERN int SNESGetIterationNumber(SNES,int*);
EXTERN int SNESGetFunctionNorm(SNES,Scalar*);
EXTERN int SNESGetNumberUnsuccessfulSteps(SNES,int*);
EXTERN int SNESGetNumberLinearIterations(SNES,int*);
EXTERN int SNES_KSP_SetParametersEW(SNES,int,double,double,double,double,double,double);
EXTERN int SNES_KSP_SetConvergenceTestEW(SNES);

/*
     Reuse the default KSP monitor routines for SNES
*/
EXTERN int SNESLGMonitorCreate(char*,char*,int,int,int,int,DrawLG*);
EXTERN int SNESLGMonitor(SNES,int,double,void*);
EXTERN int SNESLGMonitorDestroy(DrawLG);

EXTERN int SNESSetApplicationContext(SNES,void *);
EXTERN int SNESGetApplicationContext(SNES,void **);

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
              SNES_DIVERGED_LOCAL_MIN          = -8,  /* || J^T b || is small, implies converged to local minimum of F() */
              SNES_CONVERGED_ITERATING         =  0} SNESConvergedReason;

EXTERN int SNESSetConvergenceTest(SNES,int (*)(SNES,double,double,double,SNESConvergedReason*,void*),void*);
EXTERN int SNESConverged_UM_LS(SNES,double,double,double,SNESConvergedReason*,void*);
EXTERN int SNESConverged_UM_TR(SNES,double,double,double,SNESConvergedReason*,void*);
EXTERN int SNESConverged_EQ_LS(SNES,double,double,double,SNESConvergedReason*,void*);
EXTERN int SNESConverged_EQ_TR(SNES,double,double,double,SNESConvergedReason*,void*);
EXTERN int SNESGetConvergedReason(SNES,SNESConvergedReason*);

/* --------- Solving systems of nonlinear equations --------------- */
EXTERN int SNESSetFunction(SNES,Vec,int(*)(SNES,Vec,Vec,void*),void *);
EXTERN int SNESComputeFunction(SNES,Vec,Vec);
EXTERN int SNESSetJacobian(SNES,Mat,Mat,int(*)(SNES,Vec,Mat*,Mat*,MatStructure*,void*),void *);
EXTERN int SNESGetJacobian(SNES,Mat*,Mat*,void **,int(**)(SNES,Vec,Mat*,Mat*,MatStructure*,void*));
EXTERN int SNESDefaultComputeJacobian(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
EXTERN int SNESDefaultComputeJacobianColor(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
EXTERN int SNESSetLineSearch(SNES,int(*)(SNES,void*,Vec,Vec,Vec,Vec,Vec,double,double*,double*,int*),void*);
EXTERN int SNESNoLineSearch(SNES,void*,Vec,Vec,Vec,Vec,Vec,double,double*,double*,int*);
EXTERN int SNESNoLineSearchNoNorms(SNES,void*,Vec,Vec,Vec,Vec,Vec,double,double*,double*,int*);
EXTERN int SNESCubicLineSearch(SNES,void*,Vec,Vec,Vec,Vec,Vec,double,double*,double*,int*);
EXTERN int SNESQuadraticLineSearch(SNES,void*,Vec,Vec,Vec,Vec,Vec,double,double*,double*,int*);
EXTERN int SNESSetLineSearchCheck(SNES,int(*)(SNES,void*,Vec,PetscTruth*),void*);
EXTERN int SNESSetLineSearchParams(SNES,double,double,double);
EXTERN int SNESGetLineSearchParams(SNES,double*,double*,double*);

/* --------- Unconstrained minimization routines --------------------------------*/
EXTERN int SNESSetHessian(SNES,Mat,Mat,int(*)(SNES,Vec,Mat*,Mat*,MatStructure*,void*),void *);
EXTERN int SNESGetHessian(SNES,Mat*,Mat*,void **);
EXTERN int SNESDefaultComputeHessian(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
EXTERN int SNESDefaultComputeHessianColor(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
EXTERN int SNESSetGradient(SNES,Vec,int(*)(SNES,Vec,Vec,void*),void*);
EXTERN int SNESGetGradient(SNES,Vec*,void**);
EXTERN int SNESGetGradientNorm(SNES,Scalar*);
EXTERN int SNESComputeGradient(SNES,Vec,Vec);
EXTERN int SNESSetMinimizationFunction(SNES,int(*)(SNES,Vec,double*,void*),void*);
EXTERN int SNESComputeMinimizationFunction(SNES,Vec,double*);
EXTERN int SNESGetMinimizationFunction(SNES,double*,void**);
EXTERN int SNESSetMinimizationFunctionTolerance(SNES,double);
EXTERN int SNESLineSearchSetDampingParameter(SNES,Scalar*);


/* Should these 2 routines be private? */
EXTERN int SNESComputeHessian(SNES,Vec,Mat*,Mat*,MatStructure*);
EXTERN int SNESComputeJacobian(SNES,Vec,Mat*,Mat*,MatStructure*);

#endif

