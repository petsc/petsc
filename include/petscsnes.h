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
#define SNESType char*

/* Logging support */
extern PetscCookie SNES_COOKIE;
extern PetscEvent    SNES_Solve, SNES_LineSearch, SNES_FunctionEval, SNES_JacobianEval;


EXTERN PetscErrorCode SNESInitializePackage(const char[]);

EXTERN PetscErrorCode SNESCreate(MPI_Comm,SNES*);
EXTERN PetscErrorCode SNESDestroy(SNES);
EXTERN PetscErrorCode SNESSetType(SNES,const SNESType);
EXTERN PetscErrorCode SNESSetMonitor(SNES,PetscErrorCode(*)(SNES,PetscInt,PetscReal,void*),void *,PetscErrorCode (*)(void*));
EXTERN PetscErrorCode SNESClearMonitor(SNES);
EXTERN PetscErrorCode SNESSetConvergenceHistory(SNES,PetscReal[],PetscInt[],PetscInt,PetscTruth);
EXTERN PetscErrorCode SNESGetConvergenceHistory(SNES,PetscReal*[],PetscInt *[],PetscInt *);
EXTERN PetscErrorCode SNESSetUp(SNES,Vec);
EXTERN PetscErrorCode SNESSolve(SNES,Vec);

EXTERN PetscErrorCode SNESAddOptionsChecker(PetscErrorCode (*)(SNES));

EXTERN PetscErrorCode SNESSetRhsBC(SNES, PetscErrorCode (*)(SNES, Vec, void *));
EXTERN PetscErrorCode SNESDefaultRhsBC(SNES, Vec, void *);
EXTERN PetscErrorCode SNESSetSolutionBC(SNES, PetscErrorCode (*)(SNES, Vec, void *));
EXTERN PetscErrorCode SNESDefaultSolutionBC(SNES, Vec, void *);
EXTERN PetscErrorCode SNESSetUpdate(SNES, PetscErrorCode (*)(SNES, PetscInt));
EXTERN PetscErrorCode SNESDefaultUpdate(SNES, PetscInt);

extern PetscFList SNESList;
EXTERN PetscErrorCode SNESRegisterDestroy(void);
EXTERN PetscErrorCode SNESRegisterAll(const char[]);

EXTERN PetscErrorCode SNESRegister(const char[],const char[],const char[],PetscErrorCode (*)(SNES));

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

EXTERN PetscErrorCode SNESGetKSP(SNES,KSP*);
EXTERN PetscErrorCode SNESGetSolution(SNES,Vec*);
EXTERN PetscErrorCode SNESGetSolutionUpdate(SNES,Vec*);
EXTERN PetscErrorCode SNESGetFunction(SNES,Vec*,void**,PetscErrorCode(**)(SNES,Vec,Vec,void*));
EXTERN PetscErrorCode SNESView(SNES,PetscViewer);

EXTERN PetscErrorCode SNESSetOptionsPrefix(SNES,const char[]);
EXTERN PetscErrorCode SNESAppendOptionsPrefix(SNES,const char[]);
EXTERN PetscErrorCode SNESGetOptionsPrefix(SNES,char*[]);
EXTERN PetscErrorCode SNESSetFromOptions(SNES);

EXTERN PetscErrorCode MatCreateSNESMF(SNES,Vec,Mat*);
EXTERN PetscErrorCode MatCreateMF(Vec,Mat*);
EXTERN PetscErrorCode MatSNESMFSetBase(Mat,Vec);
EXTERN PetscErrorCode MatSNESMFComputeJacobian(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
EXTERN PetscErrorCode MatSNESMFSetFunction(Mat,Vec,PetscErrorCode(*)(SNES,Vec,Vec,void*),void*);
EXTERN PetscErrorCode MatSNESMFSetFunctioni(Mat,PetscErrorCode (*)(PetscInt,Vec,PetscScalar*,void*));
EXTERN PetscErrorCode MatSNESMFSetFunctioniBase(Mat,PetscErrorCode (*)(Vec,void*));
EXTERN PetscErrorCode MatSNESMFAddNullSpace(Mat,MatNullSpace);
EXTERN PetscErrorCode MatSNESMFSetHHistory(Mat,PetscScalar[],PetscInt);
EXTERN PetscErrorCode MatSNESMFResetHHistory(Mat);
EXTERN PetscErrorCode MatSNESMFSetFunctionError(Mat,PetscReal);
EXTERN PetscErrorCode MatSNESMFSetPeriod(Mat,PetscInt);
EXTERN PetscErrorCode MatSNESMFGetH(Mat,PetscScalar *);
EXTERN PetscErrorCode MatSNESMFKSPMonitor(KSP,PetscInt,PetscReal,void *);
EXTERN PetscErrorCode MatSNESMFSetFromOptions(Mat);
EXTERN PetscErrorCode MatSNESMFCheckPositivity(Vec,Vec,PetscScalar*,void*);
EXTERN PetscErrorCode MatSNESMFSetCheckh(Mat,PetscErrorCode (*)(Vec,Vec,PetscScalar*,void*),void*);

typedef struct _p_MatSNESMFCtx   *MatSNESMFCtx;

#define MATSNESMF_DEFAULT "default"
#define MATSNESMF_WP      "wp"
#define MatSNESMFType char*
EXTERN PetscErrorCode MatSNESMFSetType(Mat,const MatSNESMFType);
EXTERN PetscErrorCode MatSNESMFRegister(const char[],const char[],const char[],PetscErrorCode (*)(MatSNESMFCtx));

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

EXTERN PetscErrorCode MatSNESMFRegisterAll(const char[]);
EXTERN PetscErrorCode MatSNESMFRegisterDestroy(void);
EXTERN PetscErrorCode MatSNESMFDefaultSetUmin(Mat,PetscReal);
EXTERN PetscErrorCode MatSNESMFWPSetComputeNormA(Mat,PetscTruth);
EXTERN PetscErrorCode MatSNESMFWPSetComputeNormU(Mat,PetscTruth);

EXTERN PetscErrorCode MatDAADSetSNES(Mat,SNES);

EXTERN PetscErrorCode SNESGetType(SNES,SNESType*);
EXTERN PetscErrorCode SNESDefaultMonitor(SNES,PetscInt,PetscReal,void *);
EXTERN PetscErrorCode SNESRatioMonitor(SNES,PetscInt,PetscReal,void *);
EXTERN PetscErrorCode SNESSetRatioMonitor(SNES);
EXTERN PetscErrorCode SNESVecViewMonitor(SNES,PetscInt,PetscReal,void *);
EXTERN PetscErrorCode SNESVecViewResidualMonitor(SNES,PetscInt,PetscReal,void *);
EXTERN PetscErrorCode SNESVecViewUpdateMonitor(SNES,PetscInt,PetscReal,void *);
EXTERN PetscErrorCode SNESDefaultSMonitor(SNES,PetscInt,PetscReal,void *);
EXTERN PetscErrorCode SNESSetTolerances(SNES,PetscReal,PetscReal,PetscReal,PetscInt,PetscInt);
EXTERN PetscErrorCode SNESGetTolerances(SNES,PetscReal*,PetscReal*,PetscReal*,PetscInt*,PetscInt*);
EXTERN PetscErrorCode SNESSetTrustRegionTolerance(SNES,PetscReal);
EXTERN PetscErrorCode SNESGetIterationNumber(SNES,PetscInt*);
EXTERN PetscErrorCode SNESGetFunctionNorm(SNES,PetscScalar*);
EXTERN PetscErrorCode SNESGetNumberUnsuccessfulSteps(SNES,PetscInt*);
EXTERN PetscErrorCode SNESSetMaximumUnsuccessfulSteps(SNES,PetscInt);
EXTERN PetscErrorCode SNESGetMaximumUnsuccessfulSteps(SNES,PetscInt*);
EXTERN PetscErrorCode SNESGetNumberLinearIterations(SNES,PetscInt*);
EXTERN PetscErrorCode SNES_KSP_SetParametersEW(SNES,PetscInt,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal);
EXTERN PetscErrorCode SNES_KSP_SetConvergenceTestEW(SNES);

/*
     Reuse the default KSP monitor routines for SNES
*/
EXTERN PetscErrorCode SNESLGMonitorCreate(const char[],const char[],int,int,int,int,PetscDrawLG*);
EXTERN PetscErrorCode SNESLGMonitor(SNES,PetscInt,PetscReal,void*);
EXTERN PetscErrorCode SNESLGMonitorDestroy(PetscDrawLG);

EXTERN PetscErrorCode SNESSetApplicationContext(SNES,void *);
EXTERN PetscErrorCode SNESGetApplicationContext(SNES,void **);

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
              SNES_DIVERGED_FUNCTION_COUNT     = -2,  
              SNES_DIVERGED_FNORM_NAN          = -4, 
              SNES_DIVERGED_MAX_IT             = -5,
              SNES_DIVERGED_LS_FAILURE         = -6,
              SNES_DIVERGED_LOCAL_MIN          = -8,  /* || J^T b || is small, implies converged to local minimum of F() */
              SNES_CONVERGED_ITERATING         =  0} SNESConvergedReason;

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

EXTERN PetscErrorCode SNESSetConvergenceTest(SNES,PetscErrorCode (*)(SNES,PetscReal,PetscReal,PetscReal,SNESConvergedReason*,void*),void*);
EXTERN PetscErrorCode SNESConverged_LS(SNES,PetscReal,PetscReal,PetscReal,SNESConvergedReason*,void*);
EXTERN PetscErrorCode SNESConverged_TR(SNES,PetscReal,PetscReal,PetscReal,SNESConvergedReason*,void*);
EXTERN PetscErrorCode SNESGetConvergedReason(SNES,SNESConvergedReason*);

EXTERN PetscErrorCode SNESDAFormFunction(SNES,Vec,Vec,void*);
EXTERN PetscErrorCode SNESDAComputeJacobianWithAdic(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
EXTERN PetscErrorCode SNESDAComputeJacobianWithAdifor(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
EXTERN PetscErrorCode SNESDAComputeJacobian(SNES,Vec,Mat*,Mat*,MatStructure*,void*);

/* --------- Solving systems of nonlinear equations --------------- */
EXTERN PetscErrorCode SNESSetFunction(SNES,Vec,PetscErrorCode(*)(SNES,Vec,Vec,void*),void *);
EXTERN PetscErrorCode SNESComputeFunction(SNES,Vec,Vec);
EXTERN PetscErrorCode SNESSetJacobian(SNES,Mat,Mat,PetscErrorCode(*)(SNES,Vec,Mat*,Mat*,MatStructure*,void*),void *);
EXTERN PetscErrorCode SNESGetJacobian(SNES,Mat*,Mat*,void **,PetscErrorCode(**)(SNES,Vec,Mat*,Mat*,MatStructure*,void*));
EXTERN PetscErrorCode SNESDefaultComputeJacobian(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
EXTERN PetscErrorCode SNESDefaultComputeJacobianColor(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
EXTERN PetscErrorCode SNESSetRhs(SNES,Vec);
EXTERN PetscErrorCode SNESSetLineSearch(SNES,PetscErrorCode(*)(SNES,void*,Vec,Vec,Vec,Vec,Vec,PetscReal,PetscReal*,PetscReal*,PetscTruth*),void*);
EXTERN PetscErrorCode SNESNoLineSearch(SNES,void*,Vec,Vec,Vec,Vec,Vec,PetscReal,PetscReal*,PetscReal*,PetscTruth*);
EXTERN PetscErrorCode SNESNoLineSearchNoNorms(SNES,void*,Vec,Vec,Vec,Vec,Vec,PetscReal,PetscReal*,PetscReal*,PetscTruth*);
EXTERN PetscErrorCode SNESCubicLineSearch(SNES,void*,Vec,Vec,Vec,Vec,Vec,PetscReal,PetscReal*,PetscReal*,PetscTruth*);
EXTERN PetscErrorCode SNESQuadraticLineSearch(SNES,void*,Vec,Vec,Vec,Vec,Vec,PetscReal,PetscReal*,PetscReal*,PetscTruth*);

EXTERN PetscErrorCode SNESSetLineSearchCheck(SNES,PetscErrorCode(*)(SNES,void*,Vec,PetscTruth*),void*);
EXTERN PetscErrorCode SNESSetLineSearchParams(SNES,PetscReal,PetscReal,PetscReal);
EXTERN PetscErrorCode SNESGetLineSearchParams(SNES,PetscReal*,PetscReal*,PetscReal*);

EXTERN PetscErrorCode SNESTestLocalMin(SNES snes);

/* Should this routine be private? */
EXTERN PetscErrorCode SNESComputeJacobian(SNES,Vec,Mat*,Mat*,MatStructure*);

PETSC_EXTERN_CXX_END
#endif
