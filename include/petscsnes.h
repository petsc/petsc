/*
    User interface for the nonlinear solvers package.
*/
#if !defined(__PETSCSNES_H)
#define __PETSCSNES_H
#include "petscksp.h"
PETSC_EXTERN_CXX_BEGIN

/*S
     SNES - Abstract PETSc object that manages all nonlinear solves

   Level: beginner

  Concepts: nonlinear solvers

.seealso:  SNESCreate(), SNESSetType(), SNESType, TS, KSP, KSP, PC
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
#define SNESType const char*

/* Logging support */
extern PetscCookie PETSCSNES_DLLEXPORT SNES_COOKIE;
extern PetscEvent    SNES_Solve, SNES_LineSearch, SNES_FunctionEval, SNES_JacobianEval;
extern PetscCookie PETSCSNES_DLLEXPORT MATSNESMFCTX_COOKIE;
extern PetscEvent    MATSNESMF_Mult;

EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESInitializePackage(const char[]);

EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESCreate(MPI_Comm,SNES*);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESDestroy(SNES);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESSetType(SNES,SNESType);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESSetMonitor(SNES,PetscErrorCode(*)(SNES,PetscInt,PetscReal,void*),void *,PetscErrorCode (*)(void*));
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESClearMonitor(SNES);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESSetConvergenceHistory(SNES,PetscReal[],PetscInt[],PetscInt,PetscTruth);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESGetConvergenceHistory(SNES,PetscReal*[],PetscInt *[],PetscInt *);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESSetUp(SNES);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESSolve(SNES,Vec,Vec);

EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESAddOptionsChecker(PetscErrorCode (*)(SNES));

EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESSetUpdate(SNES, PetscErrorCode (*)(SNES, PetscInt));
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESDefaultUpdate(SNES, PetscInt);

extern PetscFList SNESList;
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESRegisterDestroy(void);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESRegisterAll(const char[]);

EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESRegister(const char[],const char[],const char[],PetscErrorCode (*)(SNES));

/*MC
   SNESRegisterDynamic - Adds a method to the nonlinear solver package.

   Synopsis:
   PetscErrorCode SNESRegisterDynamic(char *name_solver,char *path,char *name_create,PetscErrorCode (*routine_create)(SNES))

   Not collective

   Input Parameters:
+  name_solver - name of a new user-defined solver
.  path - path (either absolute or relative) the library containing this solver
.  name_create - name of routine to create method context
-  routine_create - routine to create method context

   Notes:
   SNESRegisterDynamic() may be called multiple times to add several user-defined solvers.

   If dynamic libraries are used, then the fourth input argument (routine_create)
   is ignored.

   Environmental variables such as ${PETSC_ARCH}, ${PETSC_DIR}, ${PETSC_LIB_DIR},
   and others of the form ${any_environmental_variable} occuring in pathname will be 
   replaced with appropriate values.

   Sample usage:
.vb
   SNESRegisterDynamic("my_solver",/home/username/my_lib/lib/libg/solaris/mylib.a,
                "MySolverCreate",MySolverCreate);
.ve

   Then, your solver can be chosen with the procedural interface via
$     SNESSetType(snes,"my_solver")
   or at runtime via the option
$     -snes_type my_solver

   Level: advanced

    Note: If your function is not being put into a shared library then use SNESRegister() instead

.keywords: SNES, nonlinear, register

.seealso: SNESRegisterAll(), SNESRegisterDestroy()
M*/
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define SNESRegisterDynamic(a,b,c,d) SNESRegister(a,b,c,0)
#else
#define SNESRegisterDynamic(a,b,c,d) SNESRegister(a,b,c,d)
#endif

EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESGetKSP(SNES,KSP*);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESGetSolution(SNES,Vec*);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESGetSolutionUpdate(SNES,Vec*);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESGetFunction(SNES,Vec*,PetscErrorCode(**)(SNES,Vec,Vec,void*),void**);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESView(SNES,PetscViewer);

EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESSetOptionsPrefix(SNES,const char[]);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESAppendOptionsPrefix(SNES,const char[]);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESGetOptionsPrefix(SNES,const char*[]);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESSetFromOptions(SNES);

EXTERN PetscErrorCode PETSCSNES_DLLEXPORT MatCreateSNESMF(SNES,Vec,Mat*);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT MatCreateMF(Vec,Mat*);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT MatSNESMFSetBase(Mat,Vec);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT MatSNESMFComputeJacobian(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT MatSNESMFSetFunction(Mat,Vec,PetscErrorCode(*)(SNES,Vec,Vec,void*),void*);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT MatSNESMFSetFunctioni(Mat,PetscErrorCode (*)(PetscInt,Vec,PetscScalar*,void*));
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT MatSNESMFSetFunctioniBase(Mat,PetscErrorCode (*)(Vec,void*));
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT MatSNESMFAddNullSpace(Mat,MatNullSpace);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT MatSNESMFSetHHistory(Mat,PetscScalar[],PetscInt);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT MatSNESMFResetHHistory(Mat);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT MatSNESMFSetFunctionError(Mat,PetscReal);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT MatSNESMFSetPeriod(Mat,PetscInt);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT MatSNESMFGetH(Mat,PetscScalar *);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT MatSNESMFKSPMonitor(KSP,PetscInt,PetscReal,void *);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT MatSNESMFSetFromOptions(Mat);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT MatSNESMFCheckPositivity(Vec,Vec,PetscScalar*,void*);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT MatSNESMFSetCheckh(Mat,PetscErrorCode (*)(Vec,Vec,PetscScalar*,void*),void*);

typedef struct _p_MatSNESMFCtx* MatSNESMFCtx;

#define MATSNESMF_DS  "ds"
#define MATSNESMF_WP  "wp"
#define MatSNESMFType char*
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT MatSNESMFSetType(Mat,const MatSNESMFType);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT MatSNESMFRegister(const char[],const char[],const char[],PetscErrorCode (*)(MatSNESMFCtx));

/*MC
   MatSNESMFRegisterDynamic - Adds a method to the MatSNESMF registry.

   Synopsis:
   PetscErrorCode MatSNESMFRegisterDynamic(char *name_solver,char *path,char *name_create,PetscErrorCode (*routine_create)(MatSNESMF))

   Not Collective

   Input Parameters:
+  name_solver - name of a new user-defined compute-h module
.  path - path (either absolute or relative) the library containing this solver
.  name_create - name of routine to create method context
-  routine_create - routine to create method context

   Level: developer

   Notes:
   MatSNESMFRegisterDynamic) may be called multiple times to add several user-defined solvers.

   If dynamic libraries are used, then the fourth input argument (routine_create)
   is ignored.

   Sample usage:
.vb
   MatSNESMFRegisterDynamic"my_h",/home/username/my_lib/lib/libO/solaris/mylib.a,
               "MyHCreate",MyHCreate);
.ve

   Then, your solver can be chosen with the procedural interface via
$     MatSNESMFSetType(mfctx,"my_h")
   or at runtime via the option
$     -snes_mf_type my_h

.keywords: MatSNESMF, register

.seealso: MatSNESMFRegisterAll(), MatSNESMFRegisterDestroy()
M*/
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define MatSNESMFRegisterDynamic(a,b,c,d) MatSNESMFRegister(a,b,c,0)
#else
#define MatSNESMFRegisterDynamic(a,b,c,d) MatSNESMFRegister(a,b,c,d)
#endif

EXTERN PetscErrorCode PETSCSNES_DLLEXPORT MatSNESMFRegisterAll(const char[]);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT MatSNESMFRegisterDestroy(void);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT MatSNESMFDefaultSetUmin(Mat,PetscReal);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT MatSNESMFWPSetComputeNormU(Mat,PetscTruth);

EXTERN PetscErrorCode PETSCSNES_DLLEXPORT MatDAADSetSNES(Mat,SNES);

EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESGetType(SNES,SNESType*);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESDefaultMonitor(SNES,PetscInt,PetscReal,void *);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESRatioMonitor(SNES,PetscInt,PetscReal,void *);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESSetRatioMonitor(SNES);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESVecViewMonitor(SNES,PetscInt,PetscReal,void *);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESVecViewResidualMonitor(SNES,PetscInt,PetscReal,void *);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESVecViewUpdateMonitor(SNES,PetscInt,PetscReal,void *);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESDefaultSMonitor(SNES,PetscInt,PetscReal,void *);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESSetTolerances(SNES,PetscReal,PetscReal,PetscReal,PetscInt,PetscInt);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESGetTolerances(SNES,PetscReal*,PetscReal*,PetscReal*,PetscInt*,PetscInt*);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESSetTrustRegionTolerance(SNES,PetscReal);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESGetIterationNumber(SNES,PetscInt*);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESGetFunctionNorm(SNES,PetscScalar*);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESGetNumberUnsuccessfulSteps(SNES,PetscInt*);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESSetMaximumUnsuccessfulSteps(SNES,PetscInt);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESGetMaximumUnsuccessfulSteps(SNES,PetscInt*);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESGetNumberLinearIterations(SNES,PetscInt*);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNES_KSP_SetParametersEW(SNES,PetscInt,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNES_KSP_SetConvergenceTestEW(SNES);

/*
     Reuse the default KSP monitor routines for SNES
*/
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESLGMonitorCreate(const char[],const char[],int,int,int,int,PetscDrawLG*);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESLGMonitor(SNES,PetscInt,PetscReal,void*);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESLGMonitorDestroy(PetscDrawLG);

EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESSetApplicationContext(SNES,void *);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESGetApplicationContext(SNES,void **);

/*E
    SNESConvergedReason - reason a SNES method was said to 
         have converged or diverged

   Level: beginner

   Notes: this must match finclude/petscsnes.h 

   Developer note: The string versions of these are in 
     src/snes/interface/snes.c called convergedreasons.
     If these enums are changed you much change those.

.seealso: SNESSolve(), SNESGetConvergedReason(), KSPConvergedReason, SNESSetConvergenceTest()
E*/
typedef enum {/* converged */
              SNES_CONVERGED_FNORM_ABS         =  2, /* F < F_minabs */
              SNES_CONVERGED_FNORM_RELATIVE    =  3, /* F < F_mintol*F_initial */
              SNES_CONVERGED_PNORM_RELATIVE    =  4, /* step size small */
              SNES_CONVERGED_TR_DELTA          =  7,
              /* diverged */
              SNES_DIVERGED_FUNCTION_DOMAIN    = -1,  
              SNES_DIVERGED_FUNCTION_COUNT     = -2,  
              SNES_DIVERGED_FNORM_NAN          = -4, 
              SNES_DIVERGED_MAX_IT             = -5,
              SNES_DIVERGED_LS_FAILURE         = -6,
              SNES_DIVERGED_LOCAL_MIN          = -8,  /* || J^T b || is small, implies converged to local minimum of F() */
              SNES_CONVERGED_ITERATING         =  0} SNESConvergedReason;
extern const char **SNESConvergedReasons;

/*MC
     SNES_CONVERGED_FNORM_ABS - 2-norm(F) <= abstol

   Level: beginner

.seealso:  SNESSolve(), SNESGetConvergedReason(), SNESConvergedReason, SNESSetTolerances()

M*/

/*MC
     SNES_CONVERGED_FNORM_RELATIVE - 2-norm(F) <= rtol*2-norm(F(x_0)) where x_0 is the initial guess

   Level: beginner

.seealso:  SNESSolve(), SNESGetConvergedReason(), SNESConvergedReason, SNESSetTolerances()

M*/

/*MC
     SNES_CONVERGED_PNORM_RELATIVE - The 2-norm of the last step <= xtol * 2-norm(x) where x is the current
          solution and xtol is the 4th argument to SNESSetTolerances()

   Level: beginner

.seealso:  SNESSolve(), SNESGetConvergedReason(), SNESConvergedReason, SNESSetTolerances()

M*/

/*MC
     SNES_DIVERGED_FUNCTION_COUNT - The user provided function has been called more times then the final
         argument to SNESSetTolerances()

   Level: beginner

.seealso:  SNESSolve(), SNESGetConvergedReason(), SNESConvergedReason, SNESSetTolerances()

M*/

/*MC
     SNES_DIVERGED_FNORM_NAN - the 2-norm of the current function evaluation is not-a-number (NaN), this
      is usually caused by a division of 0 by 0.

   Level: beginner

.seealso:  SNESSolve(), SNESGetConvergedReason(), SNESConvergedReason, SNESSetTolerances()

M*/

/*MC
     SNES_DIVERGED_MAX_IT - SNESSolve() has reached the maximum number of iterations requested

   Level: beginner

.seealso:  SNESSolve(), SNESGetConvergedReason(), SNESConvergedReason, SNESSetTolerances()

M*/

/*MC
     SNES_DIVERGED_LS_FAILURE - The line search has failed. This only occurs for a SNESType of SNESLS

   Level: beginner

.seealso:  SNESSolve(), SNESGetConvergedReason(), SNESConvergedReason, SNESSetTolerances()

M*/

/*MC
     SNES_DIVERGED_LOCAL_MIN - the algorithm seems to have stagnated at a local minimum that is not zero

   Level: beginner

.seealso:  SNESSolve(), SNESGetConvergedReason(), SNESConvergedReason, SNESSetTolerances()

M*/

/*MC
     SNES_CONERGED_ITERATING - this only occurs if SNESGetConvergedReason() is called during the SNESSolve()

   Level: beginner

.seealso:  SNESSolve(), SNESGetConvergedReason(), SNESConvergedReason, SNESSetTolerances()

M*/

EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESSetConvergenceTest(SNES,PetscErrorCode (*)(SNES,PetscReal,PetscReal,PetscReal,SNESConvergedReason*,void*),void*);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESConverged_LS(SNES,PetscReal,PetscReal,PetscReal,SNESConvergedReason*,void*);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESConverged_TR(SNES,PetscReal,PetscReal,PetscReal,SNESConvergedReason*,void*);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESGetConvergedReason(SNES,SNESConvergedReason*);

EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESDAFormFunction(SNES,Vec,Vec,void*);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESDAComputeJacobianWithAdic(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESDAComputeJacobianWithAdifor(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESDAComputeJacobian(SNES,Vec,Mat*,Mat*,MatStructure*,void*);

/* --------- Solving systems of nonlinear equations --------------- */
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESSetFunction(SNES,Vec,PetscErrorCode(*)(SNES,Vec,Vec,void*),void *);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESComputeFunction(SNES,Vec,Vec);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESSetJacobian(SNES,Mat,Mat,PetscErrorCode(*)(SNES,Vec,Mat*,Mat*,MatStructure*,void*),void *);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESGetJacobian(SNES,Mat*,Mat*,PetscErrorCode(**)(SNES,Vec,Mat*,Mat*,MatStructure*,void*),void **);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESDefaultComputeJacobian(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESDefaultComputeJacobianColor(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESSetRhs(SNES,Vec);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESGetRhs(SNES,Vec*);

/* --------- Routines specifically for line search methods --------------- */
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESLineSearchSet(SNES,PetscErrorCode(*)(SNES,void*,Vec,Vec,Vec,Vec,Vec,PetscReal,PetscReal*,PetscReal*,PetscTruth*),void*);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESLineSearchNo(SNES,void*,Vec,Vec,Vec,Vec,Vec,PetscReal,PetscReal*,PetscReal*,PetscTruth*);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESLineSearchNoNorms(SNES,void*,Vec,Vec,Vec,Vec,Vec,PetscReal,PetscReal*,PetscReal*,PetscTruth*);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESLineSearchCubic(SNES,void*,Vec,Vec,Vec,Vec,Vec,PetscReal,PetscReal*,PetscReal*,PetscTruth*);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESLineSearchQuadratic(SNES,void*,Vec,Vec,Vec,Vec,Vec,PetscReal,PetscReal*,PetscReal*,PetscTruth*);

EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESLineSearchSetPostCheck(SNES,PetscErrorCode(*)(SNES,Vec,Vec,Vec,void*,PetscTruth*,PetscTruth*),void*);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESLineSearchSetPreCheck(SNES,PetscErrorCode(*)(SNES,Vec,Vec,void*,PetscTruth*),void*);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESLineSearchSetParams(SNES,PetscReal,PetscReal,PetscReal);
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESLineSearchGetParams(SNES,PetscReal*,PetscReal*,PetscReal*);

EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESTestLocalMin(SNES snes);

/* Should this routine be private? */
EXTERN PetscErrorCode PETSCSNES_DLLEXPORT SNESComputeJacobian(SNES,Vec,Mat*,Mat*,MatStructure*);

PETSC_EXTERN_CXX_END
#endif
