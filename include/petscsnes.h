/* $Id: petscsnes.h,v 1.111 2001/08/06 21:17:05 bsmith Exp $ */
/*
    User interface for the nonlinear solvers package.
*/
#if !defined(__PETSCSNES_H)
#define __PETSCSNES_H
#include "petscsles.h"

/*S
     SNES - Abstract PETSc object that manages all nonlinear solves

   Level: beginner

  Concepts: nonlinear solvers

.seealso:  SNESCreate(), SNESSetType(), SNESType, TS, SLES, KSP, PC
S*/
typedef struct _p_SNES* SNES;

/*E
    SNESType - String with the name of a PETSc SNES method or the creation function
       with an optional dynamic library name, for example
       http://www.mcs.anl.gov/petsc/lib.a:mysnescreate()

   Level: beginner

.seealso: SNESSetType(), SNES
E*/
#define SNESLS          "ls"
#define SNESTR          "tr"
#define SNESTEST        "test"
typedef char *SNESType;

/* Logging support */
extern int SNES_COOKIE;
extern int SNES_Solve, SNES_LineSearch, SNES_FunctionEval, SNES_JacobianEval;


EXTERN int SNESInitializePackage(char *);

EXTERN int SNESCreate(MPI_Comm,SNES*);
EXTERN int SNESDestroy(SNES);
EXTERN int SNESSetType(SNES,SNESType);
EXTERN int SNESSetMonitor(SNES,int(*)(SNES,int,PetscReal,void*),void *,int (*)(void *));
EXTERN int SNESClearMonitor(SNES);
EXTERN int SNESSetConvergenceHistory(SNES,PetscReal*,int *,int,PetscTruth);
EXTERN int SNESGetConvergenceHistory(SNES,PetscReal**,int **,int *);
EXTERN int SNESSetUp(SNES,Vec);
EXTERN int SNESSolve(SNES,Vec,int*);

EXTERN int SNESAddOptionsChecker(int (*)(SNES));

EXTERN int SNESSetRhsBC(SNES, int (*)(SNES, Vec, void *));
EXTERN int SNESDefaultRhsBC(SNES, Vec, void *);
EXTERN int SNESSetSolutionBC(SNES, int (*)(SNES, Vec, void *));
EXTERN int SNESDefaultSolutionBC(SNES, Vec, void *);
EXTERN int SNESSetUpdate(SNES, int (*)(SNES, int));
EXTERN int SNESDefaultUpdate(SNES, int);

extern PetscFList SNESList;
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
EXTERN int SNESView(SNES,PetscViewer);

EXTERN int SNESSetOptionsPrefix(SNES,char*);
EXTERN int SNESAppendOptionsPrefix(SNES,char*);
EXTERN int SNESGetOptionsPrefix(SNES,char**);
EXTERN int SNESSetFromOptions(SNES);

EXTERN int MatCreateSNESMF(SNES,Vec,Mat*);
EXTERN int MatCreateMF(Vec,Mat*);
EXTERN int MatSNESMFSetBase(Mat,Vec);
EXTERN int MatSNESMFComputeJacobian(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
EXTERN int MatSNESMFSetFunction(Mat,Vec,int(*)(SNES,Vec,Vec,void*),void*);
EXTERN int MatSNESMFSetFunctioni(Mat,int (*)(int,Vec,PetscScalar*,void*));
EXTERN int MatSNESMFSetFunctioniBase(Mat,int (*)(Vec,void*));
EXTERN int MatSNESMFAddNullSpace(Mat,MatNullSpace);
EXTERN int MatSNESMFSetHHistory(Mat,PetscScalar *,int);
EXTERN int MatSNESMFResetHHistory(Mat);
EXTERN int MatSNESMFSetFunctionError(Mat,PetscReal);
EXTERN int MatSNESMFSetPeriod(Mat,int);
EXTERN int MatSNESMFGetH(Mat,PetscScalar *);
EXTERN int MatSNESMFKSPMonitor(KSP,int,PetscReal,void *);
EXTERN int MatSNESMFSetFromOptions(Mat);
EXTERN int MatSNESMFCheckPositivity(Vec,Vec,PetscScalar*,void*);
EXTERN int MatSNESMFSetCheckh(Mat,int (*)(Vec,Vec,PetscScalar*,void*),void*);

typedef struct _p_MatSNESMFCtx   *MatSNESMFCtx;

#define MATSNESMF_DEFAULT "default"
#define MATSNESMF_WP      "wp"
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
EXTERN int MatSNESMFDefaultSetUmin(Mat,PetscReal);
EXTERN int MatSNESMFWPSetComputeNormA(Mat,PetscTruth);
EXTERN int MatSNESMFWPSetComputeNormU(Mat,PetscTruth);

EXTERN int MatDAADSetSNES(Mat,SNES);

EXTERN int SNESGetType(SNES,SNESType*);
EXTERN int SNESDefaultMonitor(SNES,int,PetscReal,void *);
EXTERN int SNESRatioMonitor(SNES,int,PetscReal,void *);
EXTERN int SNESSetRatioMonitor(SNES);
EXTERN int SNESVecViewMonitor(SNES,int,PetscReal,void *);
EXTERN int SNESVecViewResidualMonitor(SNES,int,PetscReal,void *);
EXTERN int SNESVecViewUpdateMonitor(SNES,int,PetscReal,void *);
EXTERN int SNESDefaultSMonitor(SNES,int,PetscReal,void *);
EXTERN int SNESSetTolerances(SNES,PetscReal,PetscReal,PetscReal,int,int);
EXTERN int SNESGetTolerances(SNES,PetscReal*,PetscReal*,PetscReal*,int*,int*);
EXTERN int SNESSetTrustRegionTolerance(SNES,PetscReal);
EXTERN int SNESGetIterationNumber(SNES,int*);
EXTERN int SNESGetFunctionNorm(SNES,PetscScalar*);
EXTERN int SNESGetNumberUnsuccessfulSteps(SNES,int*);
EXTERN int SNESSetMaximumUnsuccessfulSteps(SNES,int);
EXTERN int SNESGetMaximumUnsuccessfulSteps(SNES,int*);
EXTERN int SNESGetNumberLinearIterations(SNES,int*);
EXTERN int SNES_KSP_SetParametersEW(SNES,int,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal);
EXTERN int SNES_KSP_SetConvergenceTestEW(SNES);

/*
     Reuse the default KSP monitor routines for SNES
*/
EXTERN int SNESLGMonitorCreate(char*,char*,int,int,int,int,PetscDrawLG*);
EXTERN int SNESLGMonitor(SNES,int,PetscReal,void*);
EXTERN int SNESLGMonitorDestroy(PetscDrawLG);

EXTERN int SNESSetApplicationContext(SNES,void *);
EXTERN int SNESGetApplicationContext(SNES,void **);

/*E
    SNESConvergedReason - reason a SNES method was said to 
         have converged or diverged

   Level: beginner

   Notes: this must match finclude/petscsnes.h 

.seealso: SNESSolve(), SNESGetConvergedReason(), KSPConvergedReason, SNESSetConvergenceTest()
E*/
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

EXTERN int SNESSetConvergenceTest(SNES,int (*)(SNES,PetscReal,PetscReal,PetscReal,SNESConvergedReason*,void*),void*);
EXTERN int SNESConverged_LS(SNES,PetscReal,PetscReal,PetscReal,SNESConvergedReason*,void*);
EXTERN int SNESConverged_TR(SNES,PetscReal,PetscReal,PetscReal,SNESConvergedReason*,void*);
EXTERN int SNESGetConvergedReason(SNES,SNESConvergedReason*);

EXTERN int SNESDAFormFunction(SNES,Vec,Vec,void*);
EXTERN int SNESDAComputeJacobianWithAdic(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
EXTERN int SNESDAComputeJacobianWithAdifor(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
EXTERN int SNESDAComputeJacobian(SNES,Vec,Mat*,Mat*,MatStructure*,void*);

/* --------- Solving systems of nonlinear equations --------------- */
EXTERN int SNESSetFunction(SNES,Vec,int(*)(SNES,Vec,Vec,void*),void *);
EXTERN int SNESComputeFunction(SNES,Vec,Vec);
EXTERN int SNESSetJacobian(SNES,Mat,Mat,int(*)(SNES,Vec,Mat*,Mat*,MatStructure*,void*),void *);
EXTERN int SNESGetJacobian(SNES,Mat*,Mat*,void **,int(**)(SNES,Vec,Mat*,Mat*,MatStructure*,void*));
EXTERN int SNESDefaultComputeJacobian(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
EXTERN int SNESDefaultComputeJacobianColor(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
EXTERN int SNESSetLineSearch(SNES,int(*)(SNES,void*,Vec,Vec,Vec,Vec,Vec,PetscReal,PetscReal*,PetscReal*,int*),void*);
EXTERN int SNESNoLineSearch(SNES,void*,Vec,Vec,Vec,Vec,Vec,PetscReal,PetscReal*,PetscReal*,int*);
EXTERN int SNESNoLineSearchNoNorms(SNES,void*,Vec,Vec,Vec,Vec,Vec,PetscReal,PetscReal*,PetscReal*,int*);
EXTERN int SNESCubicLineSearch(SNES,void*,Vec,Vec,Vec,Vec,Vec,PetscReal,PetscReal*,PetscReal*,int*);
EXTERN int SNESQuadraticLineSearch(SNES,void*,Vec,Vec,Vec,Vec,Vec,PetscReal,PetscReal*,PetscReal*,int*);
EXTERN int SNESSetLineSearchCheck(SNES,int(*)(SNES,void*,Vec,PetscTruth*),void*);
EXTERN int SNESSetLineSearchParams(SNES,PetscReal,PetscReal,PetscReal);
EXTERN int SNESGetLineSearchParams(SNES,PetscReal*,PetscReal*,PetscReal*);

EXTERN int SNESTestLocalMin(SNES snes);

/* Should this routine be private? */
EXTERN int SNESComputeJacobian(SNES,Vec,Mat*,Mat*,MatStructure*);

#endif

