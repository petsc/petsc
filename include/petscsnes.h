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

/*J
    SNESType - String with the name of a PETSc SNES method or the creation function
       with an optional dynamic library name, for example
       http://www.mcs.anl.gov/petsc/lib.a:mysnescreate()

   Level: beginner

.seealso: SNESSetType(), SNES
J*/
#define SNESType char*
#define SNESLS           "ls"
#define SNESTR           "tr"
#define SNESPYTHON       "python"
#define SNESTEST         "test"
#define SNESNRICHARDSON  "nrichardson"
#define SNESKSPONLY      "ksponly"
#define SNESVIRS         "virs"
#define SNESVISS         "viss"
#define SNESNGMRES       "ngmres"
#define SNESQN           "qn"
#define SNESSHELL        "shell"
#define SNESGS           "gs"
#define SNESNCG          "ncg"
#define SNESSORQN        "sorqn"
#define SNESFAS          "fas"

/* Logging support */
extern PetscClassId  SNES_CLASSID;

extern PetscErrorCode  SNESInitializePackage(const char[]);

extern PetscErrorCode  SNESCreate(MPI_Comm,SNES*);
extern PetscErrorCode  SNESReset(SNES);
extern PetscErrorCode  SNESDestroy(SNES*);
extern PetscErrorCode  SNESSetType(SNES,const SNESType);
extern PetscErrorCode  SNESMonitor(SNES,PetscInt,PetscReal);
extern PetscErrorCode  SNESMonitorSet(SNES,PetscErrorCode(*)(SNES,PetscInt,PetscReal,void*),void *,PetscErrorCode (*)(void**));
extern PetscErrorCode  SNESMonitorCancel(SNES);
extern PetscErrorCode  SNESSetConvergenceHistory(SNES,PetscReal[],PetscInt[],PetscInt,PetscBool );
extern PetscErrorCode  SNESGetConvergenceHistory(SNES,PetscReal*[],PetscInt *[],PetscInt *);
extern PetscErrorCode  SNESSetUp(SNES);
extern PetscErrorCode  SNESSolve(SNES,Vec,Vec);
extern PetscErrorCode  SNESSetErrorIfNotConverged(SNES,PetscBool );
extern PetscErrorCode  SNESGetErrorIfNotConverged(SNES,PetscBool  *);


extern PetscErrorCode  SNESAddOptionsChecker(PetscErrorCode (*)(SNES));

extern PetscErrorCode  SNESSetUpdate(SNES, PetscErrorCode (*)(SNES, PetscInt));
extern PetscErrorCode  SNESDefaultUpdate(SNES, PetscInt);

extern PetscFList SNESList;
extern PetscErrorCode  SNESRegisterDestroy(void);
extern PetscErrorCode  SNESRegisterAll(const char[]);

extern PetscErrorCode  SNESRegister(const char[],const char[],const char[],PetscErrorCode (*)(SNES));

/*MC
   SNESRegisterDynamic - Adds a method to the nonlinear solver package.

   Synopsis:
   PetscErrorCode SNESRegisterDynamic(const char *name_solver,const char *path,const char *name_create,PetscErrorCode (*routine_create)(SNES))

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

extern PetscErrorCode  SNESGetKSP(SNES,KSP*);
extern PetscErrorCode  SNESSetKSP(SNES,KSP);
extern PetscErrorCode  SNESGetSolution(SNES,Vec*);
extern PetscErrorCode  SNESGetSolutionUpdate(SNES,Vec*);
extern PetscErrorCode  SNESGetRhs(SNES,Vec*);
extern PetscErrorCode  SNESView(SNES,PetscViewer);

extern PetscErrorCode  SNESSetOptionsPrefix(SNES,const char[]);
extern PetscErrorCode  SNESAppendOptionsPrefix(SNES,const char[]);
extern PetscErrorCode  SNESGetOptionsPrefix(SNES,const char*[]);
extern PetscErrorCode  SNESSetFromOptions(SNES);
extern PetscErrorCode  SNESDefaultGetWork(SNES,PetscInt);

extern PetscErrorCode  MatCreateSNESMF(SNES,Mat*);
extern PetscErrorCode  MatMFFDComputeJacobian(SNES,Vec,Mat*,Mat*,MatStructure*,void*);

extern PetscErrorCode  MatDAADSetSNES(Mat,SNES);

extern PetscErrorCode  SNESGetType(SNES,const SNESType*);
extern PetscErrorCode  SNESMonitorDefault(SNES,PetscInt,PetscReal,void *);
extern PetscErrorCode  SNESMonitorRange(SNES,PetscInt,PetscReal,void *);
extern PetscErrorCode  SNESMonitorRatio(SNES,PetscInt,PetscReal,void *);
extern PetscErrorCode  SNESMonitorSetRatio(SNES,PetscViewer);
extern PetscErrorCode  SNESMonitorSolution(SNES,PetscInt,PetscReal,void *);
extern PetscErrorCode  SNESMonitorResidual(SNES,PetscInt,PetscReal,void *);
extern PetscErrorCode  SNESMonitorSolutionUpdate(SNES,PetscInt,PetscReal,void *);
extern PetscErrorCode  SNESMonitorDefaultShort(SNES,PetscInt,PetscReal,void *);
extern PetscErrorCode  SNESSetTolerances(SNES,PetscReal,PetscReal,PetscReal,PetscInt,PetscInt);
extern PetscErrorCode  SNESGetTolerances(SNES,PetscReal*,PetscReal*,PetscReal*,PetscInt*,PetscInt*);
extern PetscErrorCode  SNESSetTrustRegionTolerance(SNES,PetscReal);
extern PetscErrorCode  SNESGetFunctionNorm(SNES,PetscReal*);
extern PetscErrorCode  SNESGetIterationNumber(SNES,PetscInt*);

extern PetscErrorCode  SNESGetNonlinearStepFailures(SNES,PetscInt*);
extern PetscErrorCode  SNESSetMaxNonlinearStepFailures(SNES,PetscInt);
extern PetscErrorCode  SNESGetMaxNonlinearStepFailures(SNES,PetscInt*);
extern PetscErrorCode  SNESGetNumberFunctionEvals(SNES,PetscInt*);

extern PetscErrorCode  SNESSetLagPreconditioner(SNES,PetscInt);
extern PetscErrorCode  SNESGetLagPreconditioner(SNES,PetscInt*);
extern PetscErrorCode  SNESSetLagJacobian(SNES,PetscInt);
extern PetscErrorCode  SNESGetLagJacobian(SNES,PetscInt*);
extern PetscErrorCode  SNESSetGridSequence(SNES,PetscInt);

extern PetscErrorCode  SNESGetLinearSolveIterations(SNES,PetscInt*);
extern PetscErrorCode  SNESGetLinearSolveFailures(SNES,PetscInt*);
extern PetscErrorCode  SNESSetMaxLinearSolveFailures(SNES,PetscInt);
extern PetscErrorCode  SNESGetMaxLinearSolveFailures(SNES,PetscInt*);

extern PetscErrorCode  SNESKSPSetUseEW(SNES,PetscBool );
extern PetscErrorCode  SNESKSPGetUseEW(SNES,PetscBool *);
extern PetscErrorCode  SNESKSPSetParametersEW(SNES,PetscInt,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal);
extern PetscErrorCode  SNESKSPGetParametersEW(SNES,PetscInt*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*,PetscReal*);

extern PetscErrorCode  SNESMonitorLGCreate(const char[],const char[],int,int,int,int,PetscDrawLG*);
extern PetscErrorCode  SNESMonitorLG(SNES,PetscInt,PetscReal,void*);
extern PetscErrorCode  SNESMonitorLGDestroy(PetscDrawLG*);
extern PetscErrorCode  SNESMonitorLGRangeCreate(const char[],const char[],int,int,int,int,PetscDrawLG*);
extern PetscErrorCode  SNESMonitorLGRange(SNES,PetscInt,PetscReal,void*);
extern PetscErrorCode  SNESMonitorLGRangeDestroy(PetscDrawLG*);

extern PetscErrorCode  SNESSetApplicationContext(SNES,void *);
extern PetscErrorCode  SNESGetApplicationContext(SNES,void *);
extern PetscErrorCode  SNESSetComputeApplicationContext(SNES,PetscErrorCode (*)(SNES,void**),PetscErrorCode (*)(void**));

extern PetscErrorCode  SNESPythonSetType(SNES,const char[]);

extern PetscErrorCode  SNESSetFunctionDomainError(SNES);
/*E
    SNESConvergedReason - reason a SNES method was said to 
         have converged or diverged

   Level: beginner

   The two most common reasons for divergence are 
$   1) an incorrectly coded or computed Jacobian or 
$   2) failure or lack of convergence in the linear system (in this case we recommend
$      testing with -pc_type lu to eliminate the linear solver as the cause of the problem).

   Diverged Reasons:
.    SNES_DIVERGED_LOCAL_MIN - this can only occur when using the line-search variant of SNES.
       The line search wants to minimize Q(alpha) = 1/2 || F(x + alpha s) ||^2_2  this occurs
       at Q'(alpha) = s^T F'(x+alpha s)^T F(x+alpha s) = 0. If s is the Newton direction - F'(x)^(-1)F(x) then
       you get Q'(alpha) = -F(x)^T F'(x)^(-1)^T F'(x+alpha s)F(x+alpha s); when alpha = 0
       Q'(0) = - ||F(x)||^2_2 which is always NEGATIVE if F'(x) is invertible. This means the Newton
       direction is a descent direction and the line search should succeed if alpha is small enough.

       If F'(x) is NOT invertible AND F'(x)^T F(x) = 0 then Q'(0) = 0 and the Newton direction 
       is NOT a descent direction so the line search will fail. All one can do at this point
       is change the initial guess and try again.

       An alternative explanation: Newton's method can be regarded as replacing the function with
       its linear approximation and minimizing the 2-norm of that. That is F(x+s) approx F(x) + F'(x)s
       so we minimize || F(x) + F'(x) s ||^2_2; do this using Least Squares. If F'(x) is invertible then
       s = - F'(x)^(-1)F(x) otherwise F'(x)^T F'(x) s = -F'(x)^T F(x). If F'(x)^T F(x) is NOT zero then there
       exists a nontrival (that is F'(x)s != 0) solution to the equation and this direction is 
       s = - [F'(x)^T F'(x)]^(-1) F'(x)^T F(x) so Q'(0) = - F(x)^T F'(x) [F'(x)^T F'(x)]^(-T) F'(x)^T F(x)
       = - (F'(x)^T F(x)) [F'(x)^T F'(x)]^(-T) (F'(x)^T F(x)). Since we are assuming (F'(x)^T F(x)) != 0
       and F'(x)^T F'(x) has no negative eigenvalues Q'(0) < 0 so s is a descent direction and the line
       search should succeed for small enough alpha.

       Note that this RARELY happens in practice. Far more likely the linear system is not being solved
       (well enough?) or the Jacobian is wrong.
     
   SNES_DIVERGED_MAX_IT means that the solver reached the maximum number of iterations without satisfying any
   convergence criteria. SNES_CONVERGED_ITS means that SNESSkipConverged() was chosen as the convergence test;
   thus the usual convergence criteria have not been checked and may or may not be satisfied.

   Developer Notes: this must match finclude/petscsnes.h 

       The string versions of these are in SNESConvergedReason, if you change any value here you must
     also adjust that array.

   Each reason has its own manual page.

.seealso: SNESSolve(), SNESGetConvergedReason(), KSPConvergedReason, SNESSetConvergenceTest()
E*/
typedef enum {/* converged */
              SNES_CONVERGED_FNORM_ABS         =  2, /* ||F|| < atol */
              SNES_CONVERGED_FNORM_RELATIVE    =  3, /* ||F|| < rtol*||F_initial|| */
              SNES_CONVERGED_PNORM_RELATIVE    =  4, /* Newton computed step size small; || delta x || < stol */
              SNES_CONVERGED_ITS               =  5, /* maximum iterations reached */
              SNES_CONVERGED_TR_DELTA          =  7,
              /* diverged */
              SNES_DIVERGED_FUNCTION_DOMAIN     = -1, /* the new x location passed the function is not in the domain of F */
              SNES_DIVERGED_FUNCTION_COUNT      = -2,  
              SNES_DIVERGED_LINEAR_SOLVE        = -3, /* the linear solve failed */
              SNES_DIVERGED_FNORM_NAN           = -4, 
              SNES_DIVERGED_MAX_IT              = -5,
              SNES_DIVERGED_LINE_SEARCH         = -6, /* the line search failed */ 
              SNES_DIVERGED_INNER               = -7, /* inner solve failed */
              SNES_DIVERGED_LOCAL_MIN           = -8, /* || J^T b || is small, implies converged to local minimum of F() */
              SNES_CONVERGED_ITERATING          =  0} SNESConvergedReason;
extern const char *const*SNESConvergedReasons;

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
     SNES_CONVERGED_PNORM_RELATIVE - The 2-norm of the last step <= stol * 2-norm(x) where x is the current
          solution and stol is the 4th argument to SNESSetTolerances()

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
     SNES_DIVERGED_LINE_SEARCH - The line search has failed. This only occurs for a SNESType of SNESLS

   Level: beginner

.seealso:  SNESSolve(), SNESGetConvergedReason(), SNESConvergedReason, SNESSetTolerances()

M*/

/*MC
     SNES_DIVERGED_LOCAL_MIN - the algorithm seems to have stagnated at a local minimum that is not zero. 
        See the manual page for SNESConvergedReason for more details

   Level: beginner

.seealso:  SNESSolve(), SNESGetConvergedReason(), SNESConvergedReason, SNESSetTolerances()

M*/

/*MC
     SNES_CONERGED_ITERATING - this only occurs if SNESGetConvergedReason() is called during the SNESSolve()

   Level: beginner

.seealso:  SNESSolve(), SNESGetConvergedReason(), SNESConvergedReason, SNESSetTolerances()

M*/

extern PetscErrorCode  SNESSetConvergenceTest(SNES,PetscErrorCode (*)(SNES,PetscInt,PetscReal,PetscReal,PetscReal,SNESConvergedReason*,void*),void*,PetscErrorCode (*)(void*));
extern PetscErrorCode  SNESDefaultConverged(SNES,PetscInt,PetscReal,PetscReal,PetscReal,SNESConvergedReason*,void*);
extern PetscErrorCode  SNESSkipConverged(SNES,PetscInt,PetscReal,PetscReal,PetscReal,SNESConvergedReason*,void*);
extern PetscErrorCode  SNESGetConvergedReason(SNES,SNESConvergedReason*);

extern PetscErrorCode  SNESDMDAComputeFunction(SNES,Vec,Vec,void*);
extern PetscErrorCode  SNESDMDAComputeJacobianWithAdic(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
extern PetscErrorCode  SNESDMDAComputeJacobianWithAdifor(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
extern PetscErrorCode  SNESDMDAComputeJacobian(SNES,Vec,Mat*,Mat*,MatStructure*,void*);

extern PetscErrorCode  SNESDMMeshComputeFunction(SNES,Vec,Vec,void*);
extern PetscErrorCode SNESDMMeshComputeJacobian(SNES,Vec,Mat*,Mat*,MatStructure*,void*);

/* --------- Solving systems of nonlinear equations --------------- */
typedef PetscErrorCode (*SNESFunction)(SNES,Vec,Vec,void*);
typedef PetscErrorCode (*SNESJacobian)(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
typedef PetscErrorCode (*SNESGSFunction)(SNES,Vec,Vec,void*);
extern PetscErrorCode  SNESSetFunction(SNES,Vec,SNESFunction,void*);
extern PetscErrorCode  SNESGetFunction(SNES,Vec*,SNESFunction*,void**);
extern PetscErrorCode  SNESComputeFunction(SNES,Vec,Vec);
extern PetscErrorCode  SNESSetJacobian(SNES,Mat,Mat,SNESJacobian,void*);
extern PetscErrorCode  SNESGetJacobian(SNES,Mat*,Mat*,SNESJacobian*,void**);
extern PetscErrorCode  SNESDefaultComputeJacobian(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
extern PetscErrorCode  SNESDefaultComputeJacobianColor(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
extern PetscErrorCode  SNESSetComputeInitialGuess(SNES,PetscErrorCode (*)(SNES,Vec,void*),void*);
extern PetscErrorCode  SNESSetPicard(SNES,Vec,SNESFunction,Mat,Mat,SNESJacobian,void*);
extern PetscErrorCode  SNESGetPicard(SNES,Vec*,SNESFunction*,Mat*,SNESJacobian*,void**);

extern PetscErrorCode  SNESSetGS(SNES,SNESGSFunction,void*);
extern PetscErrorCode  SNESGetGS(SNES,SNESGSFunction*,void**);
extern PetscErrorCode  SNESSetUseGS(SNES,PetscBool);
extern PetscErrorCode  SNESGetUseGS(SNES,PetscBool *);
extern PetscErrorCode  SNESSetGSSweeps(SNES,PetscInt);
extern PetscErrorCode  SNESGetGSSweeps(SNES,PetscInt *);
extern PetscErrorCode  SNESComputeGS(SNES,Vec,Vec);

/* --------- Routines specifically for line search methods --------------- */
/*E
    SNESLineSearchType - type of line search used in Newton's method as well as VI solvers and Richardson solvers

    Level: beginner

.seealso: SNESSetFromOptions(), SNESLineSearchSet()
E*/
typedef enum {SNES_LS_BASIC, SNES_LS_BASIC_NONORMS, SNES_LS_QUADRATIC, SNES_LS_CUBIC, SNES_LS_EXACT, SNES_LS_TEST, SNES_LS_SECANT,SNES_LS_USER_DEFINED} SNESLineSearchType;
extern const char *const SNESLineSearchTypes[];
extern const char *SNESLineSearchTypeName(SNESLineSearchType); /* Does bounds checking, use this for viewing */

extern PetscErrorCode  SNESLineSearchSet(SNES,PetscErrorCode(*)(SNES,void*,Vec,Vec,Vec,PetscReal,PetscReal,Vec,Vec,PetscReal*,PetscReal*,PetscBool *),void*);
extern PetscErrorCode  SNESLineSearchSetType(SNES,SNESLineSearchType);
extern PetscErrorCode  SNESLineSearchNo(SNES,void*,Vec,Vec,Vec,PetscReal,PetscReal,Vec,Vec,PetscReal*,PetscReal*,PetscBool *);
extern PetscErrorCode  SNESLineSearchNoNorms(SNES,void*,Vec,Vec,Vec,PetscReal,PetscReal,Vec,Vec,PetscReal*,PetscReal*,PetscBool *);
extern PetscErrorCode  SNESLineSearchQuadratic(SNES,void*,Vec,Vec,Vec,PetscReal,PetscReal,Vec,Vec,PetscReal*,PetscReal*,PetscBool *);
extern PetscErrorCode  SNESLineSearchCubic(SNES,void*,Vec,Vec,Vec,PetscReal,PetscReal,Vec,Vec,PetscReal*,PetscReal*,PetscBool *);
extern PetscErrorCode  SNESLineSearchSecant(SNES,void*,Vec,Vec,Vec,PetscReal,PetscReal,Vec,Vec,PetscReal*,PetscReal*,PetscBool *);
extern PetscErrorCode  SNESLineSearchQuadraticSecant(SNES,void*,Vec,Vec,Vec,PetscReal,PetscReal,Vec,Vec,PetscReal*,PetscReal*,PetscBool *);

extern PetscErrorCode  SNESLineSearchSetPostCheck(SNES,PetscErrorCode(*)(SNES,Vec,Vec,Vec,void*,PetscBool *,PetscBool *),void*);
extern PetscErrorCode  SNESLineSearchSetPreCheck(SNES,PetscErrorCode(*)(SNES,Vec,Vec,void*,PetscBool *),void*);
extern PetscErrorCode  SNESLineSearchPreCheckPicard(SNES,Vec,Vec,void*,PetscBool*);
extern PetscErrorCode  SNESLineSearchSetParams(SNES,PetscReal,PetscReal,PetscReal);
extern PetscErrorCode  SNESLineSearchGetParams(SNES,PetscReal*,PetscReal*,PetscReal*);
extern PetscErrorCode  SNESLineSearchSetMonitor(SNES,PetscBool );

extern PetscErrorCode  SNESShellGetContext(SNES,void**);
extern PetscErrorCode  SNESShellSetContext(SNES,void*);
extern PetscErrorCode  SNESShellSetSolve(SNES,PetscErrorCode (*)(SNES,Vec));

/* Routines for VI solver */
extern PetscErrorCode  SNESVISetVariableBounds(SNES,Vec,Vec);
extern PetscErrorCode  SNESVISetComputeVariableBounds(SNES, PetscErrorCode (*)(SNES,Vec,Vec));
extern PetscErrorCode  SNESVIGetInactiveSet(SNES,IS*);
extern PetscErrorCode  SNESVIGetActiveSetIS(SNES,Vec,Vec,IS*);
extern PetscErrorCode  SNESVIComputeInactiveSetFnorm(SNES,Vec,Vec,PetscReal*);
extern PetscErrorCode  SNESVISetRedundancyCheck(SNES,PetscErrorCode(*)(SNES,IS,IS*,void*),void*);
#define SNES_VI_INF   1.0e20
#define SNES_VI_NINF -1.0e20

extern PetscErrorCode  SNESTestLocalMin(SNES);

/* Should this routine be private? */
extern PetscErrorCode  SNESComputeJacobian(SNES,Vec,Mat*,Mat*,MatStructure*);

extern PetscErrorCode SNESSetDM(SNES,DM);
extern PetscErrorCode SNESGetDM(SNES,DM*);
extern PetscErrorCode SNESSetPC(SNES,SNES);
extern PetscErrorCode SNESGetPC(SNES,SNES*);

/* Routines for Multiblock solver */
extern PetscErrorCode SNESMultiblockSetFields(SNES, const char [], PetscInt, const PetscInt *);
extern PetscErrorCode SNESMultiblockSetIS(SNES, const char [], IS);
extern PetscErrorCode SNESMultiblockSetBlockSize(SNES, PetscInt);
extern PetscErrorCode SNESMultiblockSetType(SNES, PCCompositeType);

PETSC_EXTERN_CXX_END
#endif
