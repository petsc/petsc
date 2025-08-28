/*
    User interface for the nonlinear solvers package.
*/
#pragma once

#include <petscksp.h>
#include <petscdmtypes.h>
#include <petscfvtypes.h>
#include <petscdmdatypes.h>
#include <petscsnestypes.h>

/* SUBMANSEC = SNES */

/*J
   SNESType - String with the name of a PETSc `SNES` method. These are all the nonlinear solvers that PETSc provides.

   Level: beginner

   Note:
   Use `SNESSetType()` or the options database key `-snes_type` to set the specific nonlinear solver algorithm to use with a given `SNES` object

.seealso: [](doc_nonlinsolve), [](ch_snes), `SNESSetType()`, `SNES`, `SNESCreate()`, `SNESDestroy()`, `SNESSetFromOptions()`
J*/
typedef const char *SNESType;
#define SNESNEWTONLS         "newtonls"
#define SNESNEWTONTR         "newtontr"
#define SNESNEWTONTRDC       "newtontrdc"
#define SNESPYTHON           "python"
#define SNESNRICHARDSON      "nrichardson"
#define SNESKSPONLY          "ksponly"
#define SNESKSPTRANSPOSEONLY "ksptransposeonly"
#define SNESVINEWTONRSLS     "vinewtonrsls"
#define SNESVINEWTONSSLS     "vinewtonssls"
#define SNESNGMRES           "ngmres"
#define SNESQN               "qn"
#define SNESSHELL            "shell"
#define SNESNGS              "ngs"
#define SNESNCG              "ncg"
#define SNESFAS              "fas"
#define SNESMS               "ms"
#define SNESNASM             "nasm"
#define SNESANDERSON         "anderson"
#define SNESASPIN            "aspin"
#define SNESCOMPOSITE        "composite"
#define SNESPATCH            "patch"
#define SNESNEWTONAL         "newtonal"

/* Logging support */
PETSC_EXTERN PetscClassId SNES_CLASSID;
PETSC_EXTERN PetscClassId DMSNES_CLASSID;

PETSC_EXTERN PetscErrorCode SNESInitializePackage(void);
PETSC_EXTERN PetscErrorCode SNESFinalizePackage(void);

PETSC_EXTERN PetscErrorCode SNESCreate(MPI_Comm, SNES *);
PETSC_EXTERN PetscErrorCode SNESParametersInitialize(SNES);
PETSC_EXTERN PetscErrorCode SNESReset(SNES);
PETSC_EXTERN PetscErrorCode SNESDestroy(SNES *);
PETSC_EXTERN PetscErrorCode SNESSetType(SNES, SNESType);
PETSC_EXTERN PetscErrorCode SNESMonitor(SNES, PetscInt, PetscReal);
PETSC_EXTERN PetscErrorCode SNESMonitorSet(SNES, PetscErrorCode (*)(SNES, PetscInt, PetscReal, void *), void *, PetscCtxDestroyFn *);
PETSC_EXTERN PetscErrorCode SNESMonitorSetFromOptions(SNES, const char[], const char[], const char[], PetscErrorCode (*)(SNES, PetscInt, PetscReal, PetscViewerAndFormat *), PetscErrorCode (*)(SNES, PetscViewerAndFormat *));
PETSC_EXTERN PetscErrorCode SNESMonitorCancel(SNES);
PETSC_EXTERN PetscErrorCode SNESMonitorSAWs(SNES, PetscInt, PetscReal, void *);
PETSC_EXTERN PetscErrorCode SNESMonitorSAWsCreate(SNES, void **);
PETSC_EXTERN PetscErrorCode SNESMonitorSAWsDestroy(void **);
PETSC_EXTERN PetscErrorCode SNESSetConvergenceHistory(SNES, PetscReal[], PetscInt[], PetscInt, PetscBool);
PETSC_EXTERN PetscErrorCode SNESGetConvergenceHistory(SNES, PetscReal *[], PetscInt *[], PetscInt *);
PETSC_EXTERN PetscErrorCode SNESSetUp(SNES);
PETSC_EXTERN PetscErrorCode SNESSolve(SNES, Vec, Vec);
PETSC_EXTERN PetscErrorCode SNESSetErrorIfNotConverged(SNES, PetscBool);
PETSC_EXTERN PetscErrorCode SNESGetErrorIfNotConverged(SNES, PetscBool *);
PETSC_EXTERN PetscErrorCode SNESConverged(SNES, PetscInt, PetscReal, PetscReal, PetscReal);

PETSC_EXTERN PetscErrorCode SNESSetWorkVecs(SNES, PetscInt);

PETSC_EXTERN PetscErrorCode SNESAddOptionsChecker(PetscErrorCode (*)(SNES));

PETSC_EXTERN PetscErrorCode SNESRegister(const char[], PetscErrorCode (*)(SNES));

PETSC_EXTERN PetscErrorCode SNESGetKSP(SNES, KSP *);
PETSC_EXTERN PetscErrorCode SNESSetKSP(SNES, KSP);
PETSC_EXTERN PetscErrorCode SNESSetSolution(SNES, Vec);
PETSC_EXTERN PetscErrorCode SNESGetSolution(SNES, Vec *);
PETSC_EXTERN PetscErrorCode SNESGetSolutionUpdate(SNES, Vec *);
PETSC_EXTERN PetscErrorCode SNESGetRhs(SNES, Vec *);
PETSC_EXTERN PetscErrorCode SNESView(SNES, PetscViewer);
PETSC_EXTERN PetscErrorCode SNESLoad(SNES, PetscViewer);
PETSC_EXTERN PetscErrorCode SNESConvergedReasonViewSet(SNES, PetscErrorCode (*)(SNES, void *), void *, PetscCtxDestroyFn *);
PETSC_EXTERN PetscErrorCode SNESViewFromOptions(SNES, PetscObject, const char[]);
PETSC_EXTERN PetscErrorCode SNESConvergedReasonView(SNES, PetscViewer);
PETSC_EXTERN PetscErrorCode SNESConvergedReasonViewFromOptions(SNES);
PETSC_EXTERN PetscErrorCode SNESConvergedReasonViewCancel(SNES);

PETSC_DEPRECATED_FUNCTION(3, 14, 0, "SNESConvergedReasonView()", ) static inline PetscErrorCode SNESReasonView(SNES snes, PetscViewer v)
{
  return SNESConvergedReasonView(snes, v);
}
PETSC_DEPRECATED_FUNCTION(3, 14, 0, "SNESConvergedReasonViewFromOptions()", ) static inline PetscErrorCode SNESReasonViewFromOptions(SNES snes)
{
  return SNESConvergedReasonViewFromOptions(snes);
}

#define SNES_FILE_CLASSID 1211224

PETSC_EXTERN PetscErrorCode SNESSetOptionsPrefix(SNES, const char[]);
PETSC_EXTERN PetscErrorCode SNESAppendOptionsPrefix(SNES, const char[]);
PETSC_EXTERN PetscErrorCode SNESGetOptionsPrefix(SNES, const char *[]);
PETSC_EXTERN PetscErrorCode SNESSetFromOptions(SNES);
PETSC_EXTERN PetscErrorCode SNESResetFromOptions(SNES);

PETSC_EXTERN PetscErrorCode SNESSetUseMatrixFree(SNES, PetscBool, PetscBool);
PETSC_EXTERN PetscErrorCode SNESGetUseMatrixFree(SNES, PetscBool *, PetscBool *);
PETSC_EXTERN PetscErrorCode MatCreateSNESMF(SNES, Mat *);
PETSC_EXTERN PetscErrorCode MatSNESMFGetSNES(Mat, SNES *);
PETSC_EXTERN PetscErrorCode MatSNESMFSetReuseBase(Mat, PetscBool);
PETSC_EXTERN PetscErrorCode MatSNESMFGetReuseBase(Mat, PetscBool *);
PETSC_EXTERN PetscErrorCode MatMFFDComputeJacobian(SNES, Vec, Mat, Mat, void *);
PETSC_EXTERN PetscErrorCode MatCreateSNESMFMore(SNES, Vec, Mat *);
PETSC_EXTERN PetscErrorCode MatSNESMFMoreSetParameters(Mat, PetscReal, PetscReal, PetscReal);

PETSC_EXTERN PetscErrorCode SNESGetType(SNES, SNESType *);
PETSC_EXTERN PetscErrorCode SNESMonitorDefaultSetUp(SNES, PetscViewerAndFormat *);
PETSC_EXTERN PetscErrorCode SNESMonitorDefault(SNES, PetscInt, PetscReal, PetscViewerAndFormat *);
PETSC_EXTERN PetscErrorCode SNESMonitorScaling(SNES, PetscInt, PetscReal, PetscViewerAndFormat *);
PETSC_EXTERN PetscErrorCode SNESMonitorRange(SNES, PetscInt, PetscReal, PetscViewerAndFormat *);
PETSC_EXTERN PetscErrorCode SNESMonitorRatio(SNES, PetscInt, PetscReal, PetscViewerAndFormat *);
PETSC_EXTERN PetscErrorCode SNESMonitorRatioSetUp(SNES, PetscViewerAndFormat *);
PETSC_EXTERN PetscErrorCode SNESMonitorSolution(SNES, PetscInt, PetscReal, PetscViewerAndFormat *);
PETSC_EXTERN PetscErrorCode SNESMonitorResidual(SNES, PetscInt, PetscReal, PetscViewerAndFormat *);
PETSC_EXTERN PetscErrorCode SNESMonitorSolutionUpdate(SNES, PetscInt, PetscReal, PetscViewerAndFormat *);
PETSC_EXTERN PetscErrorCode SNESMonitorDefaultShort(SNES, PetscInt, PetscReal, PetscViewerAndFormat *);
PETSC_EXTERN PetscErrorCode SNESMonitorDefaultField(SNES, PetscInt, PetscReal, PetscViewerAndFormat *);
PETSC_EXTERN PetscErrorCode SNESMonitorJacUpdateSpectrum(SNES, PetscInt, PetscReal, PetscViewerAndFormat *);
PETSC_EXTERN PetscErrorCode SNESMonitorFields(SNES, PetscInt, PetscReal, PetscViewerAndFormat *);
PETSC_EXTERN PetscErrorCode KSPMonitorSNESResidual(KSP, PetscInt, PetscReal, PetscViewerAndFormat *);
PETSC_EXTERN PetscErrorCode KSPMonitorSNESResidualDrawLG(KSP, PetscInt, PetscReal, PetscViewerAndFormat *);
PETSC_EXTERN PetscErrorCode KSPMonitorSNESResidualDrawLGCreate(PetscViewer, PetscViewerFormat, void *, PetscViewerAndFormat **);

PETSC_EXTERN PetscErrorCode SNESSetTolerances(SNES, PetscReal, PetscReal, PetscReal, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode SNESSetDivergenceTolerance(SNES, PetscReal);
PETSC_EXTERN PetscErrorCode SNESGetTolerances(SNES, PetscReal *, PetscReal *, PetscReal *, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode SNESGetDivergenceTolerance(SNES, PetscReal *);
PETSC_EXTERN PetscErrorCode SNESGetForceIteration(SNES, PetscBool *);
PETSC_EXTERN PetscErrorCode SNESSetForceIteration(SNES, PetscBool);
PETSC_EXTERN PetscErrorCode SNESGetIterationNumber(SNES, PetscInt *);
PETSC_EXTERN PetscErrorCode SNESSetIterationNumber(SNES, PetscInt);

/*E
   SNESNewtonTRFallbackType - type of fallback in case the solution of the trust-region subproblem is outside of the radius

   Values:
+  `SNES_TR_FALLBACK_NEWTON` - use scaled Newton step
.  `SNES_TR_FALLBACK_CAUCHY` - use Cauchy direction
-  `SNES_TR_FALLBACK_DOGLEG` - use dogleg method

   Level: intermediate

.seealso: [](ch_snes), `SNES`, `SNESNEWTONTR`, `SNESNEWTONTRDC`
E*/
typedef enum {
  SNES_TR_FALLBACK_NEWTON,
  SNES_TR_FALLBACK_CAUCHY,
  SNES_TR_FALLBACK_DOGLEG,
} SNESNewtonTRFallbackType;

PETSC_EXTERN const char *const SNESNewtonTRFallbackTypes[];

PETSC_EXTERN PetscErrorCode SNESNewtonTRSetPreCheck(SNES, PetscErrorCode (*)(SNES, Vec, Vec, PetscBool *, void *), void *ctx);
PETSC_EXTERN PetscErrorCode SNESNewtonTRGetPreCheck(SNES, PetscErrorCode (**)(SNES, Vec, Vec, PetscBool *, void *), void **ctx);
PETSC_EXTERN PetscErrorCode SNESNewtonTRSetPostCheck(SNES, PetscErrorCode (*)(SNES, Vec, Vec, Vec, PetscBool *, PetscBool *, void *), void *ctx);
PETSC_EXTERN PetscErrorCode SNESNewtonTRGetPostCheck(SNES, PetscErrorCode (**)(SNES, Vec, Vec, Vec, PetscBool *, PetscBool *, void *), void **ctx);
PETSC_EXTERN PetscErrorCode SNESNewtonTRSetFallbackType(SNES, SNESNewtonTRFallbackType);
PETSC_EXTERN PetscErrorCode SNESNewtonTRPreCheck(SNES, Vec, Vec, PetscBool *);
PETSC_EXTERN PetscErrorCode SNESNewtonTRPostCheck(SNES, Vec, Vec, Vec, PetscBool *, PetscBool *);
PETSC_EXTERN PetscErrorCode SNESNewtonTRSetNormType(SNES, NormType);

/*E
    SNESNewtonTRQNType - type of quasi-Newton model to use

   Values:
+  `SNES_TR_QN_NONE` - do not use a quasi-Newton model
.  `SNES_TR_QN_SAME` - use the same quasi-Newton model for matrix and preconditioner
-  `SNES_TR_QN_DIFFERENT` - use different quasi-Newton models for matrix and preconditioner

   Level: intermediate

.seealso: [](ch_snes), `SNES`, `SNESNEWTONTR`
E*/
typedef enum {
  SNES_TR_QN_NONE,
  SNES_TR_QN_SAME,
  SNES_TR_QN_DIFFERENT,
} SNESNewtonTRQNType;

PETSC_EXTERN const char *const SNESNewtonTRQNTypes[];

PETSC_EXTERN PetscErrorCode SNESNewtonTRSetQNType(SNES, SNESNewtonTRQNType);

PETSC_EXTERN PETSC_DEPRECATED_FUNCTION(3, 22, 0, "SNESNewtonTRSetTolerances()", ) PetscErrorCode SNESSetTrustRegionTolerance(SNES, PetscReal);
PETSC_EXTERN PetscErrorCode SNESNewtonTRSetTolerances(SNES, PetscReal, PetscReal, PetscReal);
PETSC_EXTERN PetscErrorCode SNESNewtonTRGetTolerances(SNES, PetscReal *, PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode SNESNewtonTRSetUpdateParameters(SNES, PetscReal, PetscReal, PetscReal, PetscReal, PetscReal);
PETSC_EXTERN PetscErrorCode SNESNewtonTRGetUpdateParameters(SNES, PetscReal *, PetscReal *, PetscReal *, PetscReal *, PetscReal *);

PETSC_EXTERN PetscErrorCode SNESNewtonTRDCGetRhoFlag(SNES, PetscBool *);
PETSC_EXTERN PetscErrorCode SNESNewtonTRDCSetPreCheck(SNES, PetscErrorCode (*)(SNES, Vec, Vec, PetscBool *, void *), void *ctx);
PETSC_EXTERN PetscErrorCode SNESNewtonTRDCGetPreCheck(SNES, PetscErrorCode (**)(SNES, Vec, Vec, PetscBool *, void *), void **ctx);
PETSC_EXTERN PetscErrorCode SNESNewtonTRDCSetPostCheck(SNES, PetscErrorCode (*)(SNES, Vec, Vec, Vec, PetscBool *, PetscBool *, void *), void *ctx);
PETSC_EXTERN PetscErrorCode SNESNewtonTRDCGetPostCheck(SNES, PetscErrorCode (**)(SNES, Vec, Vec, Vec, PetscBool *, PetscBool *, void *), void **ctx);

PETSC_EXTERN PetscErrorCode SNESGetNonlinearStepFailures(SNES, PetscInt *);
PETSC_EXTERN PetscErrorCode SNESSetMaxNonlinearStepFailures(SNES, PetscInt);
PETSC_EXTERN PetscErrorCode SNESGetMaxNonlinearStepFailures(SNES, PetscInt *);
PETSC_EXTERN PetscErrorCode SNESGetNumberFunctionEvals(SNES, PetscInt *);

PETSC_EXTERN PetscErrorCode SNESSetLagPreconditioner(SNES, PetscInt);
PETSC_EXTERN PetscErrorCode SNESGetLagPreconditioner(SNES, PetscInt *);
PETSC_EXTERN PetscErrorCode SNESSetLagJacobian(SNES, PetscInt);
PETSC_EXTERN PetscErrorCode SNESGetLagJacobian(SNES, PetscInt *);
PETSC_EXTERN PetscErrorCode SNESSetLagPreconditionerPersists(SNES, PetscBool);
PETSC_EXTERN PetscErrorCode SNESSetLagJacobianPersists(SNES, PetscBool);
PETSC_EXTERN PetscErrorCode SNESSetGridSequence(SNES, PetscInt);
PETSC_EXTERN PetscErrorCode SNESGetGridSequence(SNES, PetscInt *);

PETSC_EXTERN PetscErrorCode SNESGetLinearSolveIterations(SNES, PetscInt *);
PETSC_EXTERN PetscErrorCode SNESGetLinearSolveFailures(SNES, PetscInt *);
PETSC_EXTERN PetscErrorCode SNESSetMaxLinearSolveFailures(SNES, PetscInt);
PETSC_EXTERN PetscErrorCode SNESGetMaxLinearSolveFailures(SNES, PetscInt *);
PETSC_EXTERN PetscErrorCode SNESSetCountersReset(SNES, PetscBool);
PETSC_EXTERN PetscErrorCode SNESResetCounters(SNES);

PETSC_EXTERN PetscErrorCode SNESKSPSetUseEW(SNES, PetscBool);
PETSC_EXTERN PetscErrorCode SNESKSPGetUseEW(SNES, PetscBool *);
PETSC_EXTERN PetscErrorCode SNESKSPSetParametersEW(SNES, PetscInt, PetscReal, PetscReal, PetscReal, PetscReal, PetscReal, PetscReal);
PETSC_EXTERN PetscErrorCode SNESKSPGetParametersEW(SNES, PetscInt *, PetscReal *, PetscReal *, PetscReal *, PetscReal *, PetscReal *, PetscReal *);

PETSC_EXTERN PetscErrorCode SNESMonitorLGRange(SNES, PetscInt, PetscReal, void *);

PETSC_EXTERN PetscErrorCode SNESSetApplicationContext(SNES, void *);
PETSC_EXTERN PetscErrorCode SNESGetApplicationContext(SNES, void *);
PETSC_EXTERN PetscErrorCode SNESSetComputeApplicationContext(SNES, PetscErrorCode (*)(SNES, void **), PetscCtxDestroyFn *);

PETSC_EXTERN PetscErrorCode SNESPythonSetType(SNES, const char[]);
PETSC_EXTERN PetscErrorCode SNESPythonGetType(SNES, const char *[]);

PETSC_EXTERN PetscErrorCode SNESSetFunctionDomainError(SNES);
PETSC_EXTERN PetscErrorCode SNESGetFunctionDomainError(SNES, PetscBool *);
PETSC_EXTERN PetscErrorCode SNESGetJacobianDomainError(SNES, PetscBool *);
PETSC_EXTERN PetscErrorCode SNESSetJacobianDomainError(SNES);
PETSC_EXTERN PetscErrorCode SNESSetCheckJacobianDomainError(SNES, PetscBool);
PETSC_EXTERN PetscErrorCode SNESGetCheckJacobianDomainError(SNES, PetscBool *);

#define SNES_CONVERGED_TR_DELTA_DEPRECATED SNES_CONVERGED_TR_DELTA PETSC_DEPRECATED_ENUM(3, 12, 0, "SNES_DIVERGED_TR_DELTA", )
/*E
    SNESConvergedReason - reason a `SNESSolve()` was determined to have converged or diverged

   Values:
+  `SNES_CONVERGED_FNORM_ABS`      - 2-norm(F) <= abstol
.  `SNES_CONVERGED_FNORM_RELATIVE` - 2-norm(F) <= rtol*2-norm(F(x_0)) where x_0 is the initial guess
.  `SNES_CONVERGED_SNORM_RELATIVE` - The 2-norm of the last step <= stol * 2-norm(x) where x is the current
.  `SNES_CONVERGED_USER`           - The user has indicated convergence for an arbitrary reason
.  `SNES_DIVERGED_FUNCTION_COUNT`  - The user provided function has been called more times than the maximum set in `SNESSetTolerances()`
.  `SNES_DIVERGED_DTOL`            - The norm of the function has increased by a factor of divtol set with `SNESSetDivergenceTolerance()`
.  `SNES_DIVERGED_FNORM_NAN`       - the 2-norm of the current function evaluation is not-a-number (NaN), this
                                     is usually caused by a division of 0 by 0.
.  `SNES_DIVERGED_MAX_IT`          - `SNESSolve()` has reached the maximum number of iterations requested
.  `SNES_DIVERGED_LINE_SEARCH`     - The line search has failed. This only occurs for `SNES` solvers that use a line search
.  `SNES_DIVERGED_LOCAL_MIN`       - the algorithm seems to have stagnated at a local minimum that is not zero.
.  `SNES_DIVERGED_USER`            - The user has indicated divergence for an arbitrary reason
-  `SNES_CONVERGED_ITERATING       - this only occurs if `SNESGetConvergedReason()` is called during the `SNESSolve()`

   Level: beginner

    Notes:
   The two most common reasons for divergence are an incorrectly coded or computed Jacobian or failure or lack of convergence in the linear system
   (in this case we recommend
   testing with `-pc_type lu` to eliminate the linear solver as the cause of the problem).

   `SNES_DIVERGED_LOCAL_MIN` can only occur when using a `SNES` solver that uses a line search (`SNESLineSearch`).
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
   exists a nontrivial (that is F'(x)s != 0) solution to the equation and this direction is
   s = - [F'(x)^T F'(x)]^(-1) F'(x)^T F(x) so Q'(0) = - F(x)^T F'(x) [F'(x)^T F'(x)]^(-T) F'(x)^T F(x)
   = - (F'(x)^T F(x)) [F'(x)^T F'(x)]^(-T) (F'(x)^T F(x)). Since we are assuming (F'(x)^T F(x)) != 0
   and F'(x)^T F'(x) has no negative eigenvalues Q'(0) < 0 so s is a descent direction and the line
   search should succeed for small enough alpha.

   Note that this RARELY happens in practice. Far more likely the linear system is not being solved
   (well enough?) or the Jacobian is wrong.

   `SNES_DIVERGED_MAX_IT` means that the solver reached the maximum number of iterations without satisfying any
   convergence criteria. `SNES_CONVERGED_ITS` means that `SNESConvergedSkip()` was chosen as the convergence test;
   thus the usual convergence criteria have not been checked and may or may not be satisfied.

.seealso: [](ch_snes), `SNES`, `SNESSolve()`, `SNESGetConvergedReason()`, `KSPConvergedReason`, `SNESSetConvergenceTest()`, `SNESSetTolerances()`
E*/
typedef enum {                       /* converged */
  SNES_CONVERGED_FNORM_ABS      = 2, /* ||F|| < atol */
  SNES_CONVERGED_FNORM_RELATIVE = 3, /* ||F|| < rtol*||F_initial|| */
  SNES_CONVERGED_SNORM_RELATIVE = 4, /* Newton computed step size small; || delta x || < stol || x || */
  SNES_CONVERGED_ITS            = 5, /* maximum iterations reached */
  SNES_BREAKOUT_INNER_ITER      = 6, /* Flag to break out of inner loop after checking custom convergence, used in multi-phase flow when state changes */
  SNES_CONVERGED_USER           = 7, /* The user has indicated convergence for an arbitrary reason */
  /* diverged */
  SNES_DIVERGED_FUNCTION_DOMAIN      = -1, /* the new x location passed the function is not in the domain of F */
  SNES_DIVERGED_FUNCTION_COUNT       = -2,
  SNES_DIVERGED_LINEAR_SOLVE         = -3, /* the linear solve failed */
  SNES_DIVERGED_FNORM_NAN            = -4,
  SNES_DIVERGED_MAX_IT               = -5,
  SNES_DIVERGED_LINE_SEARCH          = -6,  /* the line search failed */
  SNES_DIVERGED_INNER                = -7,  /* inner solve failed */
  SNES_DIVERGED_LOCAL_MIN            = -8,  /* || J^T b || is small, implies converged to local minimum of F() */
  SNES_DIVERGED_DTOL                 = -9,  /* || F || > divtol*||F_initial|| */
  SNES_DIVERGED_JACOBIAN_DOMAIN      = -10, /* Jacobian calculation does not make sense */
  SNES_DIVERGED_TR_DELTA             = -11,
  SNES_CONVERGED_TR_DELTA_DEPRECATED = -11,
  SNES_DIVERGED_USER                 = -12, /* The user has indicated divergence for an arbitrary reason */

  SNES_CONVERGED_ITERATING = 0
} SNESConvergedReason;
PETSC_EXTERN const char *const *SNESConvergedReasons;

/*MC
   SNES_CONVERGED_FNORM_ABS - 2-norm(F) <= abstol

   Level: beginner

.seealso: [](ch_snes), `SNES`, `SNESSolve()`, `SNESGetConvergedReason()`, `SNESConvergedReason`, `SNESSetTolerances()`
M*/

/*MC
   SNES_CONVERGED_FNORM_RELATIVE - 2-norm(F) <= rtol*2-norm(F(x_0)) where x_0 is the initial guess

   Level: beginner

.seealso: [](ch_snes), `SNES`, `SNESSolve()`, `SNESGetConvergedReason()`, `SNESConvergedReason`, `SNESSetTolerances()`
M*/

/*MC
  SNES_CONVERGED_SNORM_RELATIVE - The 2-norm of the last step <= stol * 2-norm(x) where x is the current
  solution and stol is the 4th argument to `SNESSetTolerances()`

  Options Database Key:
  -snes_stol <stol> - the step tolerance

   Level: beginner

.seealso: [](ch_snes), `SNES`, `SNESSolve()`, `SNESGetConvergedReason()`, `SNESConvergedReason`, `SNESSetTolerances()`
M*/

/*MC
   SNES_DIVERGED_FUNCTION_COUNT - The user provided function has been called more times then the final
   argument to `SNESSetTolerances()`

   Level: beginner

.seealso: [](ch_snes), `SNES`, `SNESSolve()`, `SNESGetConvergedReason()`, `SNESConvergedReason`, `SNESSetTolerances()`
M*/

/*MC
   SNES_DIVERGED_DTOL - The norm of the function has increased by a factor of divtol set with `SNESSetDivergenceTolerance()`

   Level: beginner

.seealso: [](ch_snes), `SNES`, `SNESSolve()`, `SNESGetConvergedReason()`, `SNESConvergedReason`, `SNESSetTolerances()`, `SNESSetDivergenceTolerance()`
M*/

/*MC
   SNES_DIVERGED_FNORM_NAN - the 2-norm of the current function evaluation is not-a-number (NaN), this
   is usually caused by a division of 0 by 0.

   Level: beginner

.seealso: [](ch_snes), `SNES`, `SNESSolve()`, `SNESGetConvergedReason()`, `SNESConvergedReason`, `SNESSetTolerances()`
M*/

/*MC
   SNES_DIVERGED_MAX_IT - SNESSolve() has reached the maximum number of iterations requested

   Level: beginner

.seealso: [](ch_snes), `SNES`, `SNESSolve()`, `SNESGetConvergedReason()`, `SNESConvergedReason`, `SNESSetTolerances()`
M*/

/*MC
   SNES_DIVERGED_LINE_SEARCH - The line search has failed. This only occurs for a `SNES` solvers that use a line search

   Level: beginner

.seealso: [](ch_snes), `SNES`, `SNESSolve()`, `SNESGetConvergedReason()`, `SNESConvergedReason`, `SNESSetTolerances()`, `SNESLineSearch`
M*/

/*MC
   SNES_DIVERGED_LOCAL_MIN - the algorithm seems to have stagnated at a local minimum that is not zero.
   See the manual page for `SNESConvergedReason` for more details

   Level: beginner

.seealso: [](ch_snes), `SNES`, `SNESSolve()`, `SNESGetConvergedReason()`, `SNESConvergedReason`, `SNESSetTolerances()`
M*/

/*MC
   SNES_CONERGED_ITERATING - this only occurs if `SNESGetConvergedReason()` is called during the `SNESSolve()`

   Level: beginner

.seealso: [](ch_snes), `SNES`, `SNESSolve()`, `SNESGetConvergedReason()`, `SNESConvergedReason`, `SNESSetTolerances()`
M*/

PETSC_EXTERN PetscErrorCode SNESSetConvergenceTest(SNES, PetscErrorCode (*)(SNES, PetscInt, PetscReal, PetscReal, PetscReal, SNESConvergedReason *, void *), void *, PetscErrorCode (*)(void *));
PETSC_EXTERN PetscErrorCode SNESConvergedDefault(SNES, PetscInt, PetscReal, PetscReal, PetscReal, SNESConvergedReason *, void *);
PETSC_EXTERN PetscErrorCode SNESConvergedSkip(SNES, PetscInt, PetscReal, PetscReal, PetscReal, SNESConvergedReason *, void *);
PETSC_EXTERN PetscErrorCode SNESConvergedCorrectPressure(SNES, PetscInt, PetscReal, PetscReal, PetscReal, SNESConvergedReason *, void *);
PETSC_EXTERN PetscErrorCode SNESGetConvergedReason(SNES, SNESConvergedReason *);
PETSC_EXTERN PetscErrorCode SNESGetConvergedReasonString(SNES, const char **);
PETSC_EXTERN PetscErrorCode SNESSetConvergedReason(SNES, SNESConvergedReason);

PETSC_DEPRECATED_FUNCTION(3, 5, 0, "SNESConvergedSkip()", ) static inline void SNESSkipConverged(void)
{ /* never called */
}
#define SNESSkipConverged (SNESSkipConverged, SNESConvergedSkip)

/*S
  SNESInitialGuessFn - A prototype of a `SNES` compute initial guess function that would be passed to `SNESSetComputeInitialGuess()`

  Calling Sequence:
+ snes  - `SNES` context
. u   - output vector to contain initial guess
- ctx - [optional] user-defined function context

  Level: beginner

.seealso: [](ch_snes), `SNES`, `SNESSetComputeInitialGuess()`, `SNESSetFunction()`, `SNESGetFunction()`, `SNESJacobianFn`, `SNESFunctionFn`
S*/
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode SNESInitialGuessFn(SNES snes, Vec u, void *ctx);

/*S
  SNESFunctionFn - A prototype of a `SNES` evaluation function that would be passed to `SNESSetFunction()`

  Calling Sequence:
+ snes  - `SNES` context
. u   - input vector
. F   - function vector
- ctx - [optional] user-defined function context

  Level: beginner

.seealso: [](ch_snes), `SNES`, `SNESSetFunction()`, `SNESGetFunction()`, `SNESJacobianFn`, `SNESNGSFn`
S*/
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode SNESFunctionFn(SNES snes, Vec u, Vec F, void *ctx);

/*S
  SNESObjectiveFn - A prototype of a `SNES` objective evaluation function that would be passed to `SNESSetObjective()`

  Calling Sequence:
+ snes  - `SNES` context
. u   - input vector
. o   - output value
- ctx - [optional] user-defined function context

  Level: beginner

.seealso: [](ch_snes), `SNES`, `SNESSetFunction()`, `SNESGetFunction()`, `SNESJacobianFn`, `SNESNGSFn`
S*/
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode SNESObjectiveFn(SNES snes, Vec u, PetscReal *o, void *ctx);

/*S
  SNESJacobianFn - A prototype of a `SNES` Jacobian evaluation function that would be passed to `SNESSetJacobian()`

  Calling Sequence:
+ snes   - the `SNES` context obtained from `SNESCreate()`
. u    - input vector
. Amat - (approximate) Jacobian matrix
. Pmat - matrix used to construct the preconditioner, often the same as `Amat`
- ctx  - [optional] user-defined context for matrix evaluation routine

  Level: beginner

.seealso: [](ch_snes), `SNES`, `SNESSetJacobian()`, `SNESGetJacobian()`, `SNESFunctionFn`, `SNESNGSFn`
S*/
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode SNESJacobianFn(SNES snes, Vec u, Mat Amat, Mat Pmat, void *ctx);

/*S
  SNESNGSFn - A prototype of a `SNES` nonlinear Gauss-Seidel function that would be passed to `SNESSetNGS()`

  Calling Sequence:
+ snes   - the `SNES` context obtained from `SNESCreate()`
. u    - the current solution, updated in place
. b    - the right-hand side vector (which may be `NULL`)
- ctx  - [optional] user-defined context for matrix evaluation routine

  Level: beginner

.seealso: [](ch_snes), `SNES`, `SNESSetJacobian()`, `SNESGetJacobian()`, `SNESFunctionFn`, `SNESSetFunction()`, `SNESGetFunction()`, `SNESJacobianFn`
S*/
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode SNESNGSFn(SNES snes, Vec u, Vec b, void *ctx);

/*S
  SNESUpdateFn - A prototype of a `SNES` update function that would be passed to `SNESSetUpdate()`

  Calling Sequence:
+ snes - `SNES` context
- step - the current iteration index

  Level: advanced

.seealso: [](ch_snes), `SNES`, `SNESSetUpdate()`
S*/
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode SNESUpdateFn(SNES snes, PetscInt step);

/* --------- Solving systems of nonlinear equations --------------- */
PETSC_EXTERN PetscErrorCode SNESSetFunction(SNES, Vec, SNESFunctionFn *, void *);
PETSC_EXTERN PetscErrorCode SNESGetFunction(SNES, Vec *, SNESFunctionFn **, void **);
PETSC_EXTERN PetscErrorCode SNESComputeFunction(SNES, Vec, Vec);
PETSC_EXTERN PetscErrorCode SNESComputeMFFunction(SNES, Vec, Vec);
PETSC_EXTERN PetscErrorCode SNESSetInitialFunction(SNES, Vec);

PETSC_EXTERN PetscErrorCode SNESSetJacobian(SNES, Mat, Mat, SNESJacobianFn *, void *);
PETSC_EXTERN PetscErrorCode SNESGetJacobian(SNES, Mat *, Mat *, SNESJacobianFn **, void **);
PETSC_EXTERN SNESFunctionFn SNESObjectiveComputeFunctionDefaultFD;
PETSC_EXTERN SNESJacobianFn SNESComputeJacobianDefault;
PETSC_EXTERN SNESJacobianFn SNESComputeJacobianDefaultColor;
PETSC_EXTERN PetscErrorCode SNESPruneJacobianColor(SNES, Mat, Mat);
PETSC_EXTERN PetscErrorCode SNESSetComputeInitialGuess(SNES, SNESInitialGuessFn *, void *);
PETSC_EXTERN PetscErrorCode SNESSetPicard(SNES, Vec, SNESFunctionFn *, Mat, Mat, SNESJacobianFn *, void *);
PETSC_EXTERN PetscErrorCode SNESGetPicard(SNES, Vec *, SNESFunctionFn **, Mat *, Mat *, SNESJacobianFn **, void **);
PETSC_EXTERN SNESFunctionFn SNESPicardComputeFunction;
PETSC_EXTERN SNESFunctionFn SNESPicardComputeMFFunction;
PETSC_EXTERN SNESJacobianFn SNESPicardComputeJacobian;

PETSC_EXTERN PetscErrorCode SNESSetObjective(SNES, SNESObjectiveFn *, void *);
PETSC_EXTERN PetscErrorCode SNESGetObjective(SNES, SNESObjectiveFn **, void **);
PETSC_EXTERN PetscErrorCode SNESComputeObjective(SNES, Vec, PetscReal *);

PETSC_EXTERN PetscErrorCode SNESSetUpdate(SNES, SNESUpdateFn *);

/*E
   SNESNormSchedule - Frequency with which the norm is computed during a nonliner solve

   Values:
+   `SNES_NORM_DEFAULT`            - use the default behavior for the current `SNESType`
.   `SNES_NORM_NONE`               - avoid all norm computations
.   `SNES_NORM_ALWAYS`             - compute the norms whenever possible
.   `SNES_NORM_INITIAL_ONLY`       - compute the norm only when the algorithm starts
.   `SNES_NORM_FINAL_ONLY`         - compute the norm only when the algorithm finishes
-   `SNES_NORM_INITIAL_FINAL_ONLY` - compute the norm at the start and end of the algorithm

   Level: advanced

   Notes:
   Support for these is highly dependent on the solver.

   Some options limit the convergence tests that can be used.

   The `SNES_NORM_NONE` option is most commonly used when the nonlinear solver is being used as a smoother, for example for `SNESFAS`

   This is primarily used to turn off extra norm and function computation
   when the solvers are composed.

.seealso: [](ch_snes), `SNES`, `SNESSolve()`, `SNESGetConvergedReason()`, `KSPSetNormType()`,
          `KSPSetConvergenceTest()`, `KSPSetPCSide()`
E*/
typedef enum {
  SNES_NORM_DEFAULT            = -1,
  SNES_NORM_NONE               = 0,
  SNES_NORM_ALWAYS             = 1,
  SNES_NORM_INITIAL_ONLY       = 2,
  SNES_NORM_FINAL_ONLY         = 3,
  SNES_NORM_INITIAL_FINAL_ONLY = 4
} SNESNormSchedule;
PETSC_EXTERN const char *const *const SNESNormSchedules;

/*MC
   SNES_NORM_NONE - Don't compute function and its L2 norm when possible

   Level: advanced

   Note:
   This is most useful for stationary solvers with a fixed number of iterations used as smoothers.

.seealso: [](ch_snes), `SNESNormSchedule`, `SNES`, `SNESSetNormSchedule()`, `SNES_NORM_DEFAULT`
M*/

/*MC
   SNES_NORM_ALWAYS - Compute the function and its L2 norm at each iteration.

   Level: advanced

   Note:
   Most solvers will use this no matter what norm type is passed to them.

.seealso: [](ch_snes), `SNESNormSchedule`, `SNES`, `SNESSetNormSchedule()`, `SNES_NORM_NONE`
M*/

/*MC
   SNES_NORM_INITIAL_ONLY - Compute the function and its L2 at iteration 0, but do not update it.

   Level: advanced

   Notes:
   This method is useful in composed methods, when a true solution might actually be found before `SNESSolve()` is called.
   This option enables the solve to abort on the zeroth iteration if this is the case.

   For solvers that require the computation of the L2 norm of the function as part of the method, this merely cancels
   the norm computation at the last iteration (if possible).

.seealso: [](ch_snes), `SNESNormSchedule`, `SNES`, `SNESSetNormSchedule()`, `SNES_NORM_FINAL_ONLY`, `SNES_NORM_INITIAL_FINAL_ONLY`
M*/

/*MC
   SNES_NORM_FINAL_ONLY - Compute the function and its L2 norm on only the final iteration.

   Level: advanced

   Note:
   For solvers that require the computation of the L2 norm of the function as part of the method, behaves
   exactly as `SNES_NORM_DEFAULT`.  This method is useful when the function is gotten after `SNESSolve()` and
   used in subsequent computation for methods that do not need the norm computed during the rest of the
   solution procedure.

.seealso: [](ch_snes), `SNESNormSchedule`, `SNES`, `SNESSetNormSchedule()`, `SNES_NORM_INITIAL_ONLY`, `SNES_NORM_INITIAL_FINAL_ONLY`
M*/

/*MC
   SNES_NORM_INITIAL_FINAL_ONLY - Compute the function and its L2 norm on only the initial and final iterations.

   Level: advanced

   Note:
   This method combines the benefits of `SNES_NORM_INITIAL_ONLY` and `SNES_NORM_FINAL_ONLY`.

.seealso: [](ch_snes), `SNESNormSchedule`, `SNES`, `SNESSetNormSchedule()`, `SNES_NORM_SNES_NORM_INITIAL_ONLY`, `SNES_NORM_FINAL_ONLY`
M*/

PETSC_EXTERN PetscErrorCode SNESSetNormSchedule(SNES, SNESNormSchedule);
PETSC_EXTERN PetscErrorCode SNESGetNormSchedule(SNES, SNESNormSchedule *);
PETSC_EXTERN PetscErrorCode SNESSetFunctionNorm(SNES, PetscReal);
PETSC_EXTERN PetscErrorCode SNESGetFunctionNorm(SNES, PetscReal *);
PETSC_EXTERN PetscErrorCode SNESGetUpdateNorm(SNES, PetscReal *);
PETSC_EXTERN PetscErrorCode SNESGetSolutionNorm(SNES, PetscReal *);

/*E
   SNESFunctionType - Type of function computed

   Values:
+  `SNES_FUNCTION_DEFAULT`          - the default behavior for the current `SNESType`
.  `SNES_FUNCTION_UNPRECONDITIONED` - the original function provided
-  `SNES_FUNCTION_PRECONDITIONED`   - the modification of the function by the preconditioner

   Level: advanced

   Note:
   Support for these is dependent on the solver.

.seealso: [](ch_snes), `SNES`, `SNESSolve()`, `SNESGetConvergedReason()`, `KSPSetNormType()`,
          `KSPSetConvergenceTest()`, `KSPSetPCSide()`
E*/
typedef enum {
  SNES_FUNCTION_DEFAULT          = -1,
  SNES_FUNCTION_UNPRECONDITIONED = 0,
  SNES_FUNCTION_PRECONDITIONED   = 1
} SNESFunctionType;
PETSC_EXTERN const char *const *const SNESFunctionTypes;

PETSC_EXTERN PetscErrorCode SNESSetFunctionType(SNES, SNESFunctionType);
PETSC_EXTERN PetscErrorCode SNESGetFunctionType(SNES, SNESFunctionType *);

PETSC_EXTERN PetscErrorCode SNESSetNGS(SNES, SNESNGSFn *, void *);
PETSC_EXTERN PetscErrorCode SNESGetNGS(SNES, SNESNGSFn **, void **);
PETSC_EXTERN PetscErrorCode SNESComputeNGS(SNES, Vec, Vec);

PETSC_EXTERN PetscErrorCode SNESNGSSetSweeps(SNES, PetscInt);
PETSC_EXTERN PetscErrorCode SNESNGSGetSweeps(SNES, PetscInt *);
PETSC_EXTERN PetscErrorCode SNESNGSSetTolerances(SNES, PetscReal, PetscReal, PetscReal, PetscInt);
PETSC_EXTERN PetscErrorCode SNESNGSGetTolerances(SNES, PetscReal *, PetscReal *, PetscReal *, PetscInt *);

PETSC_EXTERN PetscErrorCode SNESSetAlwaysComputesFinalResidual(SNES, PetscBool);
PETSC_EXTERN PetscErrorCode SNESGetAlwaysComputesFinalResidual(SNES, PetscBool *);

PETSC_EXTERN PetscErrorCode SNESShellGetContext(SNES, void *);
PETSC_EXTERN PetscErrorCode SNESShellSetContext(SNES, void *);
PETSC_EXTERN PetscErrorCode SNESShellSetSolve(SNES, PetscErrorCode (*)(SNES, Vec));

/* --------- Routines specifically for line search methods --------------- */

/*S
   SNESLineSearch - Abstract PETSc object that manages line-search operations for nonlinear solvers

   Level: beginner

.seealso: [](ch_snes), `SNESLineSearchType`, `SNESLineSearchCreate()`, `SNESLineSearchSetType()`, `SNES`
S*/
typedef struct _p_LineSearch *SNESLineSearch;

/*J
   SNESLineSearchType - String with the name of a PETSc line search method `SNESLineSearch`. Provides all the linesearches for the nonlinear solvers, `SNES`,
                        in PETSc.

   Values:
+  `SNESLINESEARCHBASIC`     - (or equivalently `SNESLINESEARCHNONE`) Simple damping line search, defaults to using the full Newton step
.  `SNESLINESEARCHBT`        - Backtracking line search over the L2 norm of the function or an objective function
.  `SNESLINESEARCHSECANT`    - Secant line search over the L2 norm of the function or an objective function
.  `SNESLINESEARCHCP`        - Critical point secant line search assuming $F(x) = \nabla G(x)$ for some unknown $G(x)$
.  `SNESLINESEARCHNLEQERR`   - Affine-covariant error-oriented linesearch
-  `SNESLINESEARCHBISECTION` - bisection line search for a root in the directional derivative
-  `SNESLINESEARCHSHELL`     - User provided `SNESLineSearch` implementation

   Level: beginner

   Note:
   Use `SNESLineSearchSetType()` or the options database key `-snes_linesearch_type` to set
   the specific line search algorithm to use with a given `SNES` object. Not all `SNESType` can utilize a line search.

.seealso: [](ch_snes), `SNESLineSearch`, `SNESLineSearchSetType()`, `SNES`
J*/
typedef const char *SNESLineSearchType;
#define SNESLINESEARCHBT        "bt"
#define SNESLINESEARCHNLEQERR   "nleqerr"
#define SNESLINESEARCHBASIC     "basic"
#define SNESLINESEARCHNONE      "none"
#define SNESLINESEARCHSECANT    "secant"
#define SNESLINESEARCHL2        PETSC_DEPRECATED_MACRO(3, 24, 0, "SNESLINESEARCHSECANT", ) "secant"
#define SNESLINESEARCHCP        "cp"
#define SNESLINESEARCHSHELL     "shell"
#define SNESLINESEARCHNCGLINEAR "ncglinear"
#define SNESLINESEARCHBISECTION "bisection"

PETSC_EXTERN PetscFunctionList SNESList;
PETSC_EXTERN PetscClassId      SNESLINESEARCH_CLASSID;
PETSC_EXTERN PetscFunctionList SNESLineSearchList;

#define SNES_LINESEARCH_ORDER_LINEAR    1
#define SNES_LINESEARCH_ORDER_QUADRATIC 2
#define SNES_LINESEARCH_ORDER_CUBIC     3

/*S
  SNESLineSearchVIProjectFn - A prototype of a `SNES` function that projects a vector onto the VI bounds, passed to `SNESLineSearchSetVIFunctions()`

  Calling Sequence:
+ snes  - `SNES` context
- u     - the vector to project to the bounds

  Level: advanced

  Note:
  The deprecated `SNESLineSearchVIProjectFunc` still works as a replacement for `SNESLineSearchVIProjectFn` *.

.seealso: [](ch_snes), `SNES`, `SNESLineSearch`
S*/
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode             SNESLineSearchVIProjectFn(SNES snes, Vec u);
PETSC_EXTERN_TYPEDEF typedef SNESLineSearchVIProjectFn *SNESLineSearchVIProjectFunc PETSC_DEPRECATED_TYPEDEF(3, 21, 0, "SNESLineSearchVIProjectFn*", );

/*S
  SNESLineSearchVIProjectFn - A prototype of a `SNES` function that computes the norm of the active set variables in a vector in a VI solve,
  passed to `SNESLineSearchSetVIFunctions()`

  Calling Sequence:
+ snes  - `SNES` context
. f     - the vector to compute the norm of
. u     - the current solution, entries that are on the VI bounds are ignored
- fnorm - the resulting norm

  Level: advanced

  Note:
  The deprecated `SNESLineSearchVINormFunc` still works as a replacement for `SNESLineSearchVINormFn` *.

.seealso: [](ch_snes), `SNES`, `SNESLineSearch`
S*/
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode          SNESLineSearchVINormFn(SNES snes, Vec f, Vec u, PetscReal *fnorm);
PETSC_EXTERN_TYPEDEF typedef SNESLineSearchVINormFn *SNESLineSearchVINormFunc PETSC_DEPRECATED_TYPEDEF(3, 21, 0, "SNESLineSearchVINormFnn*", );

/*S
  SNESLineSearchVIDirDerivFn - A prototype of a `SNES` function that computes the directional derivative considering the VI bounds, passed to `SNESLineSearchSetVIFunctions()`

  Calling Sequence:
+ snes  - `SNES` context
. f     - the function vector to compute the directional derivative with
. u     - the current solution, entries that are on the VI bounds are ignored
. y     - the direction to compute the directional derivative
- fty   - the resulting directional derivative

  Level: advanced

.seealso: [](ch_snes), `SNES`, `SNESLineSearch`, `SNESLineSearchVIProjectFn`, `SNESLineSearchVIProjectFn`, `SNESLineSearchSetVIFunctions()`, `SNESLineSearchGetVIFunctions()`
S*/
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode SNESLineSearchVIDirDerivFn(SNES snes, Vec f, Vec u, Vec y, PetscScalar *fty);

PETSC_EXTERN_TYPEDEF typedef PetscErrorCode              SNESLineSearchApplyFn(SNESLineSearch);
PETSC_EXTERN_TYPEDEF typedef SNESLineSearchApplyFn      *SNESLineSearchApplyFunc PETSC_DEPRECATED_TYPEDEF(3, 21, 0, "SNESLineSearchApplyFn*", );
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode              SNESLineSearchShellApplyFn(SNESLineSearch, void *);
PETSC_EXTERN_TYPEDEF typedef SNESLineSearchShellApplyFn *SNESLineSearchUserFunc PETSC_DEPRECATED_TYPEDEF(3, 21, 0, "SNESLineSearchApplyFn*", );

PETSC_EXTERN PetscErrorCode SNESLineSearchCreate(MPI_Comm, SNESLineSearch *);
PETSC_EXTERN PetscErrorCode SNESLineSearchReset(SNESLineSearch);
PETSC_EXTERN PetscErrorCode SNESLineSearchView(SNESLineSearch, PetscViewer);
PETSC_EXTERN PetscErrorCode SNESLineSearchDestroy(SNESLineSearch *);
PETSC_EXTERN PetscErrorCode SNESLineSearchGetType(SNESLineSearch, SNESLineSearchType *);
PETSC_EXTERN PetscErrorCode SNESLineSearchSetType(SNESLineSearch, SNESLineSearchType);
PETSC_EXTERN PetscErrorCode SNESLineSearchSetFromOptions(SNESLineSearch);
PETSC_EXTERN PetscErrorCode SNESLineSearchSetFunction(SNESLineSearch, PetscErrorCode (*)(SNES, Vec, Vec));
PETSC_EXTERN PetscErrorCode SNESLineSearchSetUp(SNESLineSearch);
PETSC_EXTERN PetscErrorCode SNESLineSearchApply(SNESLineSearch, Vec, Vec, PetscReal *, Vec);
PETSC_EXTERN PetscErrorCode SNESLineSearchPreCheck(SNESLineSearch, Vec, Vec, PetscBool *);
PETSC_EXTERN PetscErrorCode SNESLineSearchPostCheck(SNESLineSearch, Vec, Vec, Vec, PetscBool *, PetscBool *);
PETSC_EXTERN PetscErrorCode SNESLineSearchSetWorkVecs(SNESLineSearch, PetscInt);

/* set the functions for precheck and postcheck */

PETSC_EXTERN PetscErrorCode SNESLineSearchSetPreCheck(SNESLineSearch, PetscErrorCode (*)(SNESLineSearch, Vec, Vec, PetscBool *, void *), void *ctx);
PETSC_EXTERN PetscErrorCode SNESLineSearchSetPostCheck(SNESLineSearch, PetscErrorCode (*)(SNESLineSearch, Vec, Vec, Vec, PetscBool *, PetscBool *, void *), void *ctx);

PETSC_EXTERN PetscErrorCode SNESLineSearchGetPreCheck(SNESLineSearch, PetscErrorCode (**)(SNESLineSearch, Vec, Vec, PetscBool *, void *), void **ctx);
PETSC_EXTERN PetscErrorCode SNESLineSearchGetPostCheck(SNESLineSearch, PetscErrorCode (**)(SNESLineSearch, Vec, Vec, Vec, PetscBool *, PetscBool *, void *), void **ctx);

/* set the functions for VI-specific line search operations */

PETSC_EXTERN PetscErrorCode SNESLineSearchSetVIFunctions(SNESLineSearch, SNESLineSearchVIProjectFn *, SNESLineSearchVINormFn *, SNESLineSearchVIDirDerivFn *);
PETSC_EXTERN PetscErrorCode SNESLineSearchGetVIFunctions(SNESLineSearch, SNESLineSearchVIProjectFn **, SNESLineSearchVINormFn **, SNESLineSearchVIDirDerivFn **);

/* pointers to the associated SNES in order to be able to get the function evaluation out */
PETSC_EXTERN PetscErrorCode SNESLineSearchSetSNES(SNESLineSearch, SNES);
PETSC_EXTERN PetscErrorCode SNESLineSearchGetSNES(SNESLineSearch, SNES *);

/* set and get the parameters and vectors */
PETSC_EXTERN PetscErrorCode SNESLineSearchGetTolerances(SNESLineSearch, PetscReal *, PetscReal *, PetscReal *, PetscReal *, PetscReal *, PetscInt *);
PETSC_EXTERN PetscErrorCode SNESLineSearchSetTolerances(SNESLineSearch, PetscReal, PetscReal, PetscReal, PetscReal, PetscReal, PetscInt);

PETSC_EXTERN PetscErrorCode SNESLineSearchPreCheckPicard(SNESLineSearch, Vec, Vec, PetscBool *, void *);

PETSC_EXTERN PetscErrorCode SNESLineSearchGetLambda(SNESLineSearch, PetscReal *);
PETSC_EXTERN PetscErrorCode SNESLineSearchSetLambda(SNESLineSearch, PetscReal);

PETSC_EXTERN PetscErrorCode SNESLineSearchGetDamping(SNESLineSearch, PetscReal *);
PETSC_EXTERN PetscErrorCode SNESLineSearchSetDamping(SNESLineSearch, PetscReal);

PETSC_EXTERN PetscErrorCode SNESLineSearchGetOrder(SNESLineSearch, PetscInt *);
PETSC_EXTERN PetscErrorCode SNESLineSearchSetOrder(SNESLineSearch, PetscInt);

/*E
    SNESLineSearchReason - indication if the line search has succeeded or failed and why

  Values:
+  `SNES_LINESEARCH_SUCCEEDED`       - the line search succeeded
.  `SNES_LINESEARCH_FAILED_NANORINF` - a not a number of infinity appeared in the computions
.  `SNES_LINESEARCH_FAILED_DOMAIN`   - the function was evaluated outside of its domain, see `SNESSetFunctionDomainError()` and `SNESSetJacobianDomainError()`
.  `SNES_LINESEARCH_FAILED_REDUCT`   - the linear search failed to get the requested decrease in its norm or objective
.  `SNES_LINESEARCH_FAILED_USER`     - used by `SNESLINESEARCHNLEQERR` to indicate the user changed the search direction inappropriately
-  `SNES_LINESEARCH_FAILED_FUNCTION` - indicates the maximum number of function evaluations allowed has been surpassed, `SNESConvergedReason` is also
                                       set to `SNES_DIVERGED_FUNCTION_COUNT`

   Level: intermediate

   Developer Note:
   Some of these reasons overlap with values of `SNESConvergedReason`

.seealso: [](ch_snes), `SNES`, `SNESSolve()`, `SNESGetConvergedReason()`, `KSPConvergedReason`, `SNESSetConvergenceTest()`,
          `SNESSetFunctionDomainError()` and `SNESSetJacobianDomainError()`
E*/
typedef enum {
  SNES_LINESEARCH_SUCCEEDED,
  SNES_LINESEARCH_FAILED_NANORINF,
  SNES_LINESEARCH_FAILED_DOMAIN,
  SNES_LINESEARCH_FAILED_REDUCT, /* INSUFFICIENT REDUCTION */
  SNES_LINESEARCH_FAILED_USER,
  SNES_LINESEARCH_FAILED_FUNCTION
} SNESLineSearchReason;

PETSC_EXTERN PetscErrorCode SNESLineSearchGetReason(SNESLineSearch, SNESLineSearchReason *);
PETSC_EXTERN PetscErrorCode SNESLineSearchSetReason(SNESLineSearch, SNESLineSearchReason);

PETSC_EXTERN PetscErrorCode SNESLineSearchGetVecs(SNESLineSearch, Vec *, Vec *, Vec *, Vec *, Vec *);
PETSC_EXTERN PetscErrorCode SNESLineSearchSetVecs(SNESLineSearch, Vec, Vec, Vec, Vec, Vec);

PETSC_EXTERN PetscErrorCode SNESLineSearchGetNorms(SNESLineSearch, PetscReal *, PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode SNESLineSearchSetNorms(SNESLineSearch, PetscReal, PetscReal, PetscReal);
PETSC_EXTERN PetscErrorCode SNESLineSearchComputeNorms(SNESLineSearch);
PETSC_EXTERN PetscErrorCode SNESLineSearchSetComputeNorms(SNESLineSearch, PetscBool);

PETSC_EXTERN PetscErrorCode SNESLineSearchMonitor(SNESLineSearch);
PETSC_EXTERN PetscErrorCode SNESLineSearchMonitorSet(SNESLineSearch, PetscErrorCode (*)(SNESLineSearch, void *), void *, PetscCtxDestroyFn *);
PETSC_EXTERN PetscErrorCode SNESLineSearchMonitorSetFromOptions(SNESLineSearch, const char[], const char[], const char[], PetscErrorCode (*)(SNESLineSearch, PetscViewerAndFormat *), PetscErrorCode (*)(SNESLineSearch, PetscViewerAndFormat *));
PETSC_EXTERN PetscErrorCode SNESLineSearchMonitorCancel(SNESLineSearch);
PETSC_EXTERN PetscErrorCode SNESLineSearchSetDefaultMonitor(SNESLineSearch, PetscViewer);
PETSC_EXTERN PetscErrorCode SNESLineSearchGetDefaultMonitor(SNESLineSearch, PetscViewer *);
PETSC_EXTERN PetscErrorCode SNESLineSearchMonitorSolutionUpdate(SNESLineSearch, PetscViewerAndFormat *);

PETSC_EXTERN PetscErrorCode SNESLineSearchAppendOptionsPrefix(SNESLineSearch, const char[]);
PETSC_EXTERN PetscErrorCode SNESLineSearchGetOptionsPrefix(SNESLineSearch, const char *[]);

/* Shell interface functions */
PETSC_EXTERN PetscErrorCode SNESLineSearchShellSetApply(SNESLineSearch, SNESLineSearchShellApplyFn *, void *);
PETSC_EXTERN PetscErrorCode SNESLineSearchShellGetApply(SNESLineSearch, SNESLineSearchShellApplyFn **, void **);

PETSC_DEPRECATED_FUNCTION(3, 21, 0, "SNESLinesearchShellSetApply()", ) static inline PetscErrorCode SNESLineSearchShellSetUserFunc(SNESLineSearch ls, SNESLineSearchShellApplyFn *f, void *ctx)
{
  return SNESLineSearchShellSetApply(ls, f, ctx);
}

PETSC_DEPRECATED_FUNCTION(3, 21, 0, "SNESLinesearchShellGetApply()", ) static inline PetscErrorCode SNESLineSearchShellGetUserFunc(SNESLineSearch ls, SNESLineSearchShellApplyFn **f, void **ctx)
{
  return SNESLineSearchShellGetApply(ls, f, ctx);
}

/* BT interface functions */
PETSC_EXTERN PetscErrorCode SNESLineSearchBTSetAlpha(SNESLineSearch, PetscReal);
PETSC_EXTERN PetscErrorCode SNESLineSearchBTGetAlpha(SNESLineSearch, PetscReal *);

/*register line search types */
PETSC_EXTERN PetscErrorCode SNESLineSearchRegister(const char[], PetscErrorCode (*)(SNESLineSearch));

/* Routines for VI solver */
PETSC_EXTERN PetscErrorCode SNESVISetVariableBounds(SNES, Vec, Vec);
PETSC_EXTERN PetscErrorCode SNESVIGetVariableBounds(SNES, Vec *, Vec *);
PETSC_EXTERN PetscErrorCode SNESVISetComputeVariableBounds(SNES, PetscErrorCode (*)(SNES, Vec, Vec));
PETSC_EXTERN PetscErrorCode SNESVIGetInactiveSet(SNES, IS *);
PETSC_EXTERN PetscErrorCode SNESVIGetActiveSetIS(SNES, Vec, Vec, IS *);
PETSC_EXTERN PetscErrorCode SNESVIComputeInactiveSetFnorm(SNES, Vec, Vec, PetscReal *);
PETSC_EXTERN PetscErrorCode SNESVIComputeInactiveSetFtY(SNES, Vec, Vec, Vec, PetscScalar *);
PETSC_EXTERN PetscErrorCode SNESVISetRedundancyCheck(SNES, PetscErrorCode (*)(SNES, IS, IS *, void *), void *);
PETSC_EXTERN PetscErrorCode SNESVIComputeMeritFunction(Vec, PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode SNESVIComputeFunction(SNES, Vec, Vec, void *);
PETSC_EXTERN PetscErrorCode DMSetVI(DM, IS);
PETSC_EXTERN PetscErrorCode DMDestroyVI(DM);

PETSC_EXTERN PetscErrorCode SNESTestLocalMin(SNES);

/* Should this routine be private? */
PETSC_EXTERN PetscErrorCode SNESComputeJacobian(SNES, Vec, Mat, Mat);
PETSC_EXTERN PetscErrorCode SNESTestJacobian(SNES, PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode SNESTestFunction(SNES);

PETSC_EXTERN PetscErrorCode SNESSetDM(SNES, DM);
PETSC_EXTERN PetscErrorCode SNESGetDM(SNES, DM *);
PETSC_EXTERN PetscErrorCode SNESSetNPC(SNES, SNES);
PETSC_EXTERN PetscErrorCode SNESGetNPC(SNES, SNES *);
PETSC_EXTERN PetscErrorCode SNESHasNPC(SNES, PetscBool *);
PETSC_EXTERN PetscErrorCode SNESApplyNPC(SNES, Vec, Vec, Vec);
PETSC_EXTERN PetscErrorCode SNESGetNPCFunction(SNES, Vec, PetscReal *);
PETSC_EXTERN PetscErrorCode SNESComputeFunctionDefaultNPC(SNES, Vec, Vec);
PETSC_EXTERN PetscErrorCode SNESSetNPCSide(SNES, PCSide);
PETSC_EXTERN PetscErrorCode SNESGetNPCSide(SNES, PCSide *);
PETSC_EXTERN PetscErrorCode SNESSetLineSearch(SNES, SNESLineSearch);
PETSC_EXTERN PetscErrorCode SNESGetLineSearch(SNES, SNESLineSearch *);

PETSC_DEPRECATED_FUNCTION(3, 4, 0, "SNESGetLineSearch()", ) static inline PetscErrorCode SNESGetSNESLineSearch(SNES snes, SNESLineSearch *ls)
{
  return SNESGetLineSearch(snes, ls);
}
PETSC_DEPRECATED_FUNCTION(3, 4, 0, "SNESSetLineSearch()", ) static inline PetscErrorCode SNESSetSNESLineSearch(SNES snes, SNESLineSearch ls)
{
  return SNESSetLineSearch(snes, ls);
}

PETSC_EXTERN PetscErrorCode SNESSetUpMatrices(SNES);
PETSC_EXTERN PetscErrorCode DMSNESSetFunction(DM, SNESFunctionFn *, void *);
PETSC_EXTERN PetscErrorCode DMSNESGetFunction(DM, SNESFunctionFn **, void **);
PETSC_EXTERN PetscErrorCode DMSNESSetFunctionContextDestroy(DM, PetscCtxDestroyFn *);
PETSC_EXTERN PetscErrorCode DMSNESSetMFFunction(DM, SNESFunctionFn *, void *);
PETSC_EXTERN PetscErrorCode DMSNESSetNGS(DM, SNESNGSFn *, void *);
PETSC_EXTERN PetscErrorCode DMSNESGetNGS(DM, SNESNGSFn **, void **);
PETSC_EXTERN PetscErrorCode DMSNESSetJacobian(DM, SNESJacobianFn *, void *);
PETSC_EXTERN PetscErrorCode DMSNESGetJacobian(DM, SNESJacobianFn **, void **);
PETSC_EXTERN PetscErrorCode DMSNESSetJacobianContextDestroy(DM, PetscCtxDestroyFn *);
PETSC_EXTERN PetscErrorCode DMSNESSetPicard(DM, SNESFunctionFn *, SNESJacobianFn *, void *);
PETSC_EXTERN PetscErrorCode DMSNESGetPicard(DM, SNESFunctionFn **, SNESJacobianFn **, void **);
PETSC_EXTERN PetscErrorCode DMSNESSetObjective(DM, SNESObjectiveFn *, void *);
PETSC_EXTERN PetscErrorCode DMSNESGetObjective(DM, SNESObjectiveFn **, void **);
PETSC_EXTERN PetscErrorCode DMCopyDMSNES(DM, DM);

PETSC_EXTERN_TYPEDEF typedef PetscErrorCode DMDASNESFunctionFn(DMDALocalInfo *, void *, void *, void *);
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode DMDASNESJacobianFn(DMDALocalInfo *, void *, Mat, Mat, void *);
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode DMDASNESObjectiveFn(DMDALocalInfo *, void *, PetscReal *, void *);

PETSC_EXTERN_TYPEDEF typedef PetscErrorCode DMDASNESFunctionVecFn(DMDALocalInfo *, Vec, Vec, void *);
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode DMDASNESJacobianVecFn(DMDALocalInfo *, Vec, Mat, Mat, void *);
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode DMDASNESObjectiveVecFn(DMDALocalInfo *, Vec, PetscReal *, void *);

PETSC_EXTERN PetscErrorCode DMDASNESSetFunctionLocal(DM, InsertMode, DMDASNESFunctionFn *, void *);
PETSC_EXTERN PetscErrorCode DMDASNESSetJacobianLocal(DM, DMDASNESJacobianFn *, void *);
PETSC_EXTERN PetscErrorCode DMDASNESSetObjectiveLocal(DM, DMDASNESObjectiveFn *, void *);
PETSC_EXTERN PetscErrorCode DMDASNESSetPicardLocal(DM, InsertMode, DMDASNESFunctionFn *, DMDASNESJacobianFn, void *);

PETSC_EXTERN PetscErrorCode DMDASNESSetFunctionLocalVec(DM, InsertMode, DMDASNESFunctionVecFn *, void *);
PETSC_EXTERN PetscErrorCode DMDASNESSetJacobianLocalVec(DM, DMDASNESJacobianVecFn *, void *);
PETSC_EXTERN PetscErrorCode DMDASNESSetObjectiveLocalVec(DM, DMDASNESObjectiveVecFn *, void *);

PETSC_EXTERN PetscErrorCode DMSNESSetBoundaryLocal(DM, PetscErrorCode (*)(DM, Vec, void *), void *);
PETSC_EXTERN PetscErrorCode DMSNESSetObjectiveLocal(DM, PetscErrorCode (*)(DM, Vec, PetscReal *, void *), void *);
PETSC_EXTERN PetscErrorCode DMSNESSetFunctionLocal(DM, PetscErrorCode (*)(DM, Vec, Vec, void *), void *);
PETSC_EXTERN PetscErrorCode DMSNESSetJacobianLocal(DM, PetscErrorCode (*)(DM, Vec, Mat, Mat, void *), void *);
PETSC_EXTERN PetscErrorCode DMSNESGetBoundaryLocal(DM, PetscErrorCode (**)(DM, Vec, void *), void **);
PETSC_EXTERN PetscErrorCode DMSNESGetObjectiveLocal(DM, PetscErrorCode (**)(DM, Vec, PetscReal *, void *), void **);
PETSC_EXTERN PetscErrorCode DMSNESGetFunctionLocal(DM, PetscErrorCode (**)(DM, Vec, Vec, void *), void **);
PETSC_EXTERN PetscErrorCode DMSNESGetJacobianLocal(DM, PetscErrorCode (**)(DM, Vec, Mat, Mat, void *), void **);

/* Routines for Multiblock solver */
PETSC_EXTERN PetscErrorCode SNESMultiblockSetFields(SNES, const char[], PetscInt, const PetscInt *);
PETSC_EXTERN PetscErrorCode SNESMultiblockSetIS(SNES, const char[], IS);
PETSC_EXTERN PetscErrorCode SNESMultiblockSetBlockSize(SNES, PetscInt);
PETSC_EXTERN PetscErrorCode SNESMultiblockSetType(SNES, PCCompositeType);
PETSC_EXTERN PetscErrorCode SNESMultiblockGetSubSNES(SNES, PetscInt *, SNES *[]);

/*J
   SNESMSType - String with the name of a PETSc `SNESMS` method.

   Level: intermediate

.seealso: [](ch_snes), `SNESMS`, `SNESMSGetType()`, `SNESMSSetType()`, `SNES`
J*/
typedef const char *SNESMSType;
#define SNESMSM62       "m62"
#define SNESMSEULER     "euler"
#define SNESMSJAMESON83 "jameson83"
#define SNESMSVLTP11    "vltp11"
#define SNESMSVLTP21    "vltp21"
#define SNESMSVLTP31    "vltp31"
#define SNESMSVLTP41    "vltp41"
#define SNESMSVLTP51    "vltp51"
#define SNESMSVLTP61    "vltp61"

PETSC_EXTERN PetscErrorCode SNESMSRegister(SNESMSType, PetscInt, PetscInt, PetscReal, const PetscReal[], const PetscReal[], const PetscReal[]);
PETSC_EXTERN PetscErrorCode SNESMSRegisterAll(void);
PETSC_EXTERN PetscErrorCode SNESMSGetType(SNES, SNESMSType *);
PETSC_EXTERN PetscErrorCode SNESMSSetType(SNES, SNESMSType);
PETSC_EXTERN PetscErrorCode SNESMSGetDamping(SNES, PetscReal *);
PETSC_EXTERN PetscErrorCode SNESMSSetDamping(SNES, PetscReal);
PETSC_EXTERN PetscErrorCode SNESMSFinalizePackage(void);
PETSC_EXTERN PetscErrorCode SNESMSInitializePackage(void);
PETSC_EXTERN PetscErrorCode SNESMSRegisterDestroy(void);

/*MC
   SNESNGMRESRestartType - the restart approach used by `SNESNGMRES`

  Values:
+   `SNES_NGMRES_RESTART_NONE`       - never restart
.   `SNES_NGMRES_RESTART_DIFFERENCE` - restart based upon difference criteria
-   `SNES_NGMRES_RESTART_PERIODIC`   - restart after a fixed number of iterations

  Options Database Keys:
+ -snes_ngmres_restart_type <difference,periodic,none> - set the restart type
- -snes_ngmres_restart <30>                            - sets the number of iterations before restart for periodic

   Level: intermediate

.seealso: `SNES, `SNESNGMRES`, `SNESNGMRESSetSelectType()`, `SNESNGMRESGetSelectType()`, `SNESNGMRESSetRestartType()`,
          `SNESNGMRESGetRestartType()`, `SNESNGMRESSelectType`
M*/
typedef enum {
  SNES_NGMRES_RESTART_NONE       = 0,
  SNES_NGMRES_RESTART_PERIODIC   = 1,
  SNES_NGMRES_RESTART_DIFFERENCE = 2
} SNESNGMRESRestartType;
PETSC_EXTERN const char *const SNESNGMRESRestartTypes[];

/*MC
   SNESNGMRESSelectType - the approach used by `SNESNGMRES` to determine how the candidate solution and
  combined solution are used to create the next iterate.

   Values:
+   `SNES_NGMRES_SELECT_NONE`       - choose the combined solution all the time
.   `SNES_NGMRES_SELECT_DIFFERENCE` - choose based upon the selection criteria
-   `SNES_NGMRES_SELECT_LINESEARCH` - choose based upon line search combination

  Options Database Key:
. -snes_ngmres_select_type<difference,none,linesearch> - select type

   Level: intermediate

.seealso: `SNES, `SNESNGMRES`, `SNESNGMRESSetSelectType()`, `SNESNGMRESGetSelectType()`, `SNESNGMRESSetRestartType()`,
          `SNESNGMRESGetRestartType()`, `SNESNGMRESRestartType`
M*/
typedef enum {
  SNES_NGMRES_SELECT_NONE       = 0,
  SNES_NGMRES_SELECT_DIFFERENCE = 1,
  SNES_NGMRES_SELECT_LINESEARCH = 2
} SNESNGMRESSelectType;
PETSC_EXTERN const char *const SNESNGMRESSelectTypes[];

PETSC_EXTERN PetscErrorCode SNESNGMRESSetRestartType(SNES, SNESNGMRESRestartType);
PETSC_EXTERN PetscErrorCode SNESNGMRESSetSelectType(SNES, SNESNGMRESSelectType);
PETSC_EXTERN PetscErrorCode SNESNGMRESSetRestartFmRise(SNES, PetscBool);
PETSC_EXTERN PetscErrorCode SNESNGMRESGetRestartFmRise(SNES, PetscBool *);

/*MC
   SNESNCGType - the conjugate update approach for `SNESNCG`

   Values:
+   `SNES_NCG_FR`  - Fletcher-Reeves update
.   `SNES_NCG_PRP` - Polak-Ribiere-Polyak update, the default and the only one that tolerates generalized search directions
.   `SNES_NCG_HS`  - Hestenes-Steifel update
.   `SNES_NCG_DY`  - Dai-Yuan update
-   `SNES_NCG_CD`  - Conjugate Descent update

  Options Database Key:
. -snes_ncg_type<fr,prp,hs,dy,cd> - select type

   Level: intermediate

.seealso: `SNES, `SNESNCG`, `SNESNCGSetType()`
M*/
typedef enum {
  SNES_NCG_FR  = 0,
  SNES_NCG_PRP = 1,
  SNES_NCG_HS  = 2,
  SNES_NCG_DY  = 3,
  SNES_NCG_CD  = 4
} SNESNCGType;
PETSC_EXTERN const char *const SNESNCGTypes[];

PETSC_EXTERN PetscErrorCode SNESNCGSetType(SNES, SNESNCGType);

/*MC
   SNESQNScaleType - the scaling type used by `SNESQN`

   Values:
+   `SNES_QN_SCALE_NONE`     - don't scale the problem
.   `SNES_QN_SCALE_SCALAR`   - use Shanno scaling
.   `SNES_QN_SCALE_DIAGONAL` - scale with a diagonalized BFGS formula (see Gilbert and Lemarechal 1989), available
-   `SNES_QN_SCALE_JACOBIAN` - scale by solving a linear system coming from the Jacobian you provided with `SNESSetJacobian()`
                               computed at the first iteration of `SNESQN` and at ever restart.

    Options Database Key:
. -snes_qn_scale_type <diagonal,none,scalar,jacobian> - Scaling type

   Level: intermediate

.seealso: `SNES, `SNESQN`, `SNESQNSetScaleType()`, `SNESQNType`, `SNESQNSetType()`, `SNESQNSetRestartType()`, `SNESQNRestartType`
M*/
typedef enum {
  SNES_QN_SCALE_DEFAULT  = 0,
  SNES_QN_SCALE_NONE     = 1,
  SNES_QN_SCALE_SCALAR   = 2,
  SNES_QN_SCALE_DIAGONAL = 3,
  SNES_QN_SCALE_JACOBIAN = 4
} SNESQNScaleType;
PETSC_EXTERN const char *const SNESQNScaleTypes[];

/*MC
   SNESQNRestartType - the restart approached used by `SNESQN`

   Values:
+   `SNES_QN_RESTART_NONE`     - never restart
.   `SNES_QN_RESTART_POWELL`   - restart based upon descent criteria
-   `SNES_QN_RESTART_PERIODIC` - restart after a fixed number of iterations

  Options Database Keys:
+ -snes_qn_restart_type <powell,periodic,none> - set the restart type
- -snes_qn_m <m>                               - sets the number of stored updates and the restart period for periodic

   Level: intermediate

.seealso: `SNES, `SNESQN`, `SNESQNSetScaleType()`, `SNESQNType`, `SNESQNSetType()`, `SNESQNSetRestartType()`, `SNESQNScaleType`
M*/
typedef enum {
  SNES_QN_RESTART_DEFAULT  = 0,
  SNES_QN_RESTART_NONE     = 1,
  SNES_QN_RESTART_POWELL   = 2,
  SNES_QN_RESTART_PERIODIC = 3
} SNESQNRestartType;
PETSC_EXTERN const char *const SNESQNRestartTypes[];

/*MC
   SNESQNType - the type used by `SNESQN`

  Values:
+   `SNES_QN_LBFGS`      - LBFGS variant
.   `SNES_QN_BROYDEN`    - Broyden variant
-   `SNES_QN_BADBROYDEN` - Bad Broyden variant

  Options Database Key:
. -snes_qn_type <lbfgs,broyden,badbroyden> - quasi-Newton type

   Level: intermediate

.seealso: `SNES, `SNESQN`, `SNESQNSetScaleType()`, `SNESQNSetType()`, `SNESQNScaleType`, `SNESQNRestartType`, `SNESQNSetRestartType()`
M*/
typedef enum {
  SNES_QN_LBFGS      = 0,
  SNES_QN_BROYDEN    = 1,
  SNES_QN_BADBROYDEN = 2
} SNESQNType;
PETSC_EXTERN const char *const SNESQNTypes[];

PETSC_EXTERN PetscErrorCode SNESQNSetType(SNES, SNESQNType);
PETSC_EXTERN PetscErrorCode SNESQNSetScaleType(SNES, SNESQNScaleType);
PETSC_EXTERN PetscErrorCode SNESQNSetRestartType(SNES, SNESQNRestartType);

PETSC_EXTERN PetscErrorCode SNESNASMGetType(SNES, PCASMType *);
PETSC_EXTERN PetscErrorCode SNESNASMSetType(SNES, PCASMType);
PETSC_EXTERN PetscErrorCode SNESNASMGetSubdomains(SNES, PetscInt *, SNES *[], VecScatter *[], VecScatter *[], VecScatter *[]);
PETSC_EXTERN PetscErrorCode SNESNASMSetSubdomains(SNES, PetscInt, SNES[], VecScatter[], VecScatter[], VecScatter[]);
PETSC_EXTERN PetscErrorCode SNESNASMSetDamping(SNES, PetscReal);
PETSC_EXTERN PetscErrorCode SNESNASMGetDamping(SNES, PetscReal *);
PETSC_EXTERN PetscErrorCode SNESNASMGetSubdomainVecs(SNES, PetscInt *, Vec *[], Vec *[], Vec *[], Vec *[]);
PETSC_EXTERN PetscErrorCode SNESNASMSetComputeFinalJacobian(SNES, PetscBool);
PETSC_EXTERN PetscErrorCode SNESNASMGetSNES(SNES, PetscInt, SNES *);
PETSC_EXTERN PetscErrorCode SNESNASMGetNumber(SNES, PetscInt *);
PETSC_EXTERN PetscErrorCode SNESNASMSetWeight(SNES, Vec);

/*E
  SNESCompositeType - Determines how two or more preconditioners are composed with the `SNESType` of `SNESCOMPOSITE`

  Values:
+ `SNES_COMPOSITE_ADDITIVE`        - results from application of all preconditioners are added together
. `SNES_COMPOSITE_MULTIPLICATIVE`  - preconditioners are applied sequentially to the residual freshly
                                     computed after the previous preconditioner application
- `SNES_COMPOSITE_ADDITIVEOPTIMAL` - uses a linear combination of the solutions obtained with each preconditioner that approximately minimize the function
                                     value at the new iteration.

   Level: beginner

.seealso: [](sec_pc), `PCCOMPOSITE`, `PCFIELDSPLIT`, `PC`, `PCCompositeSetType()`, `PCCompositeType`
E*/
typedef enum {
  SNES_COMPOSITE_ADDITIVE,
  SNES_COMPOSITE_MULTIPLICATIVE,
  SNES_COMPOSITE_ADDITIVEOPTIMAL
} SNESCompositeType;
PETSC_EXTERN const char *const SNESCompositeTypes[];

PETSC_EXTERN PetscErrorCode SNESCompositeSetType(SNES, SNESCompositeType);
PETSC_EXTERN PetscErrorCode SNESCompositeAddSNES(SNES, SNESType);
PETSC_EXTERN PetscErrorCode SNESCompositeGetSNES(SNES, PetscInt, SNES *);
PETSC_EXTERN PetscErrorCode SNESCompositeGetNumber(SNES, PetscInt *);
PETSC_EXTERN PetscErrorCode SNESCompositeSetDamping(SNES, PetscInt, PetscReal);

PETSC_EXTERN PetscErrorCode SNESPatchSetDiscretisationInfo(SNES, PetscInt, DM *, PetscInt *, PetscInt *, const PetscInt **, const PetscInt *, PetscInt, const PetscInt *, PetscInt, const PetscInt *);
PETSC_EXTERN PetscErrorCode SNESPatchSetComputeOperator(SNES, PetscErrorCode (*func)(PC, PetscInt, Vec, Mat, IS, PetscInt, const PetscInt *, const PetscInt *, void *), void *);
PETSC_EXTERN PetscErrorCode SNESPatchSetComputeFunction(SNES, PetscErrorCode (*func)(PC, PetscInt, Vec, Vec, IS, PetscInt, const PetscInt *, const PetscInt *, void *), void *);
PETSC_EXTERN PetscErrorCode SNESPatchSetConstructType(SNES, PCPatchConstructType, PetscErrorCode (*func)(PC, PetscInt *, IS **, IS *, void *), void *);
PETSC_EXTERN PetscErrorCode SNESPatchSetCellNumbering(SNES, PetscSection);

/*E
    SNESFASType - Determines the type of nonlinear multigrid method that is run.

   Values:
+  `SNES_FAS_MULTIPLICATIVE` (default) - traditional V or W cycle as determined by `SNESFASSetCycles()`
.  `SNES_FAS_ADDITIVE`                 - additive FAS cycle
.  `SNES_FAS_FULL`                     - full FAS cycle
-  `SNES_FAS_KASKADE`                  - Kaskade FAS cycle

   Level: beginner

.seealso: [](ch_snes), `SNESFAS`, `PCMGSetType()`, `PCMGType`
E*/
typedef enum {
  SNES_FAS_MULTIPLICATIVE,
  SNES_FAS_ADDITIVE,
  SNES_FAS_FULL,
  SNES_FAS_KASKADE
} SNESFASType;
PETSC_EXTERN const char *const SNESFASTypes[];

/* called on the finest level FAS instance*/
PETSC_EXTERN PetscErrorCode SNESFASSetType(SNES, SNESFASType);
PETSC_EXTERN PetscErrorCode SNESFASGetType(SNES, SNESFASType *);
PETSC_EXTERN PetscErrorCode SNESFASSetLevels(SNES, PetscInt, MPI_Comm *);
PETSC_EXTERN PetscErrorCode SNESFASGetLevels(SNES, PetscInt *);
PETSC_EXTERN PetscErrorCode SNESFASGetCycleSNES(SNES, PetscInt, SNES *);
PETSC_EXTERN PetscErrorCode SNESFASSetNumberSmoothUp(SNES, PetscInt);
PETSC_EXTERN PetscErrorCode SNESFASSetNumberSmoothDown(SNES, PetscInt);
PETSC_EXTERN PetscErrorCode SNESFASSetCycles(SNES, PetscInt);
PETSC_EXTERN PetscErrorCode SNESFASSetMonitor(SNES, PetscViewerAndFormat *, PetscBool);
PETSC_EXTERN PetscErrorCode SNESFASSetLog(SNES, PetscBool);

PETSC_EXTERN PetscErrorCode SNESFASSetGalerkin(SNES, PetscBool);
PETSC_EXTERN PetscErrorCode SNESFASGetGalerkin(SNES, PetscBool *);
PETSC_EXTERN PetscErrorCode SNESFASGalerkinFunctionDefault(SNES, Vec, Vec, void *);

/* called on any level -- "Cycle" FAS instance */
PETSC_EXTERN PetscErrorCode SNESFASCycleGetSmoother(SNES, SNES *);
PETSC_EXTERN PetscErrorCode SNESFASCycleGetSmootherUp(SNES, SNES *);
PETSC_EXTERN PetscErrorCode SNESFASCycleGetSmootherDown(SNES, SNES *);
PETSC_EXTERN PetscErrorCode SNESFASCycleGetCorrection(SNES, SNES *);
PETSC_EXTERN PetscErrorCode SNESFASCycleGetInterpolation(SNES, Mat *);
PETSC_EXTERN PetscErrorCode SNESFASCycleGetRestriction(SNES, Mat *);
PETSC_EXTERN PetscErrorCode SNESFASCycleGetInjection(SNES, Mat *);
PETSC_EXTERN PetscErrorCode SNESFASCycleGetRScale(SNES, Vec *);
PETSC_EXTERN PetscErrorCode SNESFASCycleSetCycles(SNES, PetscInt);
PETSC_EXTERN PetscErrorCode SNESFASCycleIsFine(SNES, PetscBool *);

/* called on the (outer) finest level FAS to set/get parameters on any level instance */
PETSC_EXTERN PetscErrorCode SNESFASSetInterpolation(SNES, PetscInt, Mat);
PETSC_EXTERN PetscErrorCode SNESFASGetInterpolation(SNES, PetscInt, Mat *);
PETSC_EXTERN PetscErrorCode SNESFASSetRestriction(SNES, PetscInt, Mat);
PETSC_EXTERN PetscErrorCode SNESFASGetRestriction(SNES, PetscInt, Mat *);
PETSC_EXTERN PetscErrorCode SNESFASSetInjection(SNES, PetscInt, Mat);
PETSC_EXTERN PetscErrorCode SNESFASGetInjection(SNES, PetscInt, Mat *);
PETSC_EXTERN PetscErrorCode SNESFASSetRScale(SNES, PetscInt, Vec);
PETSC_EXTERN PetscErrorCode SNESFASGetRScale(SNES, PetscInt, Vec *);
PETSC_EXTERN PetscErrorCode SNESFASSetContinuation(SNES, PetscBool);

PETSC_EXTERN PetscErrorCode SNESFASGetSmoother(SNES, PetscInt, SNES *);
PETSC_EXTERN PetscErrorCode SNESFASGetSmootherUp(SNES, PetscInt, SNES *);
PETSC_EXTERN PetscErrorCode SNESFASGetSmootherDown(SNES, PetscInt, SNES *);
PETSC_EXTERN PetscErrorCode SNESFASGetCoarseSolve(SNES, SNES *);

/* parameters for full FAS */
PETSC_EXTERN PetscErrorCode SNESFASFullSetDownSweep(SNES, PetscBool);
PETSC_EXTERN PetscErrorCode SNESFASCreateCoarseVec(SNES, Vec *);
PETSC_EXTERN PetscErrorCode SNESFASRestrict(SNES, Vec, Vec);
PETSC_EXTERN PetscErrorCode SNESFASFullSetTotal(SNES, PetscBool);
PETSC_EXTERN PetscErrorCode SNESFASFullGetTotal(SNES, PetscBool *);

PETSC_EXTERN PetscErrorCode DMPlexSetSNESVariableBounds(DM, SNES);
PETSC_EXTERN PetscErrorCode DMSNESCheckDiscretization(SNES, DM, PetscReal, Vec, PetscReal, PetscReal[]);
PETSC_EXTERN PetscErrorCode DMSNESCheckResidual(SNES, DM, Vec, PetscReal, PetscReal *);
PETSC_EXTERN PetscErrorCode DMSNESCheckJacobian(SNES, DM, Vec, PetscReal, PetscBool *, PetscReal *);
PETSC_EXTERN PetscErrorCode DMSNESCheckFromOptions(SNES, Vec);
PETSC_EXTERN PetscErrorCode DMSNESComputeJacobianAction(DM, Vec, Vec, Vec, void *);
PETSC_EXTERN PetscErrorCode DMSNESCreateJacobianMF(DM, Vec, void *, Mat *);

PETSC_EXTERN PetscErrorCode SNESNewtonALSetFunction(SNES, SNESFunctionFn *, void *ctx);
PETSC_EXTERN PetscErrorCode SNESNewtonALGetFunction(SNES, SNESFunctionFn **, void **ctx);
PETSC_EXTERN PetscErrorCode SNESNewtonALComputeFunction(SNES, Vec, Vec);
PETSC_EXTERN PetscErrorCode SNESNewtonALGetLoadParameter(SNES, PetscReal *);

/*MC
   SNESNewtonALCorrectionType - the approach used by `SNESNEWTONAL` to determine
   the correction to the current increment. While the exact correction satisfies
   the constraint surface at every iteration, it also requires solving a quadratic
   equation which may not have real roots. Conversely, the normal correction is more
   efficient and always yields a real correction and is the default.

   Values:
+   `SNES_NEWTONAL_CORRECTION_EXACT` - choose the correction which exactly satisfies the constraint
-   `SNES_NEWTONAL_CORRECTION_NORMAL` - choose the correction in the updated normal hyper-surface to the constraint surface

   Options Database Key:
. -snes_newtonal_correction_type <exact> - select type from <exact,normal>

   Level: intermediate

.seealso: `SNES`, `SNESNEWTONAL`, `SNESNewtonALSetCorrectionType()`
M*/
typedef enum {
  SNES_NEWTONAL_CORRECTION_EXACT  = 0,
  SNES_NEWTONAL_CORRECTION_NORMAL = 1,
} SNESNewtonALCorrectionType;
PETSC_EXTERN const char *const SNESNewtonALCorrectionTypes[];

PETSC_EXTERN PetscErrorCode SNESNewtonALSetCorrectionType(SNES, SNESNewtonALCorrectionType);
