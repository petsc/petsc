/*
   User interface for the timestepping package. This package
   is for use in solving time-dependent PDEs.
*/
#ifndef PETSCTS_H
#define PETSCTS_H

#include <petscsnes.h>
#include <petscconvest.h>

/* SUBMANSEC = TS */

/*S
     TS - Abstract PETSc object that manages all time-steppers (ODE integrators)

   Level: beginner

.seealso: [](integrator_table), [](chapter_ts), `TSCreate()`, `TSSetType()`, `TSType`, `SNES`, `KSP`, `PC`, `TSDestroy()`
S*/
typedef struct _p_TS *TS;

/*J
    TSType - String with the name of a PETSc `TS` method.

   Level: beginner

.seealso: [](integrator_table), [](chapter_ts), `TSSetType()`, `TS`, `TSRegister()`
J*/
typedef const char *TSType;
#define TSEULER           "euler"
#define TSBEULER          "beuler"
#define TSBASICSYMPLECTIC "basicsymplectic"
#define TSPSEUDO          "pseudo"
#define TSCN              "cn"
#define TSSUNDIALS        "sundials"
#define TSRK              "rk"
#define TSPYTHON          "python"
#define TSTHETA           "theta"
#define TSALPHA           "alpha"
#define TSALPHA2          "alpha2"
#define TSGLLE            "glle"
#define TSGLEE            "glee"
#define TSSSP             "ssp"
#define TSARKIMEX         "arkimex"
#define TSROSW            "rosw"
#define TSEIMEX           "eimex"
#define TSMIMEX           "mimex"
#define TSBDF             "bdf"
#define TSRADAU5          "radau5"
#define TSMPRK            "mprk"
#define TSDISCGRAD        "discgrad"
#define TSIRK             "irk"

/*E
    TSProblemType - Determines the type of problem this `TS` object is to be used to solve

   Values:
 + `TS_LINEAR` - a linear ODE or DAE
 - `TS_NONLINEAR` - a nonlinear ODE or DAE

   Level: beginner

.seealso: [](chapter_ts), `TS`, `TSCreate()`
E*/
typedef enum {
  TS_LINEAR,
  TS_NONLINEAR
} TSProblemType;

/*E
   TSEquationType - type of `TS` problem that is solved

   Level: beginner

   Values:
+  `TS_EQ_UNSPECIFIED` - (default)
.  `TS_EQ_EXPLICIT` - {ODE and DAE index 1, 2, 3, HI} F(t,U,U_t) := M(t) U_t - G(U,t) = 0
-  `TS_EQ_IMPLICIT` - {ODE and DAE index 1, 2, 3, HI} F(t,U,U_t) = 0

.seealso: [](chapter_ts), `TS`, `TSGetEquationType()`, `TSSetEquationType()`
E*/
typedef enum {
  TS_EQ_UNSPECIFIED               = -1,
  TS_EQ_EXPLICIT                  = 0,
  TS_EQ_ODE_EXPLICIT              = 1,
  TS_EQ_DAE_SEMI_EXPLICIT_INDEX1  = 100,
  TS_EQ_DAE_SEMI_EXPLICIT_INDEX2  = 200,
  TS_EQ_DAE_SEMI_EXPLICIT_INDEX3  = 300,
  TS_EQ_DAE_SEMI_EXPLICIT_INDEXHI = 500,
  TS_EQ_IMPLICIT                  = 1000,
  TS_EQ_ODE_IMPLICIT              = 1001,
  TS_EQ_DAE_IMPLICIT_INDEX1       = 1100,
  TS_EQ_DAE_IMPLICIT_INDEX2       = 1200,
  TS_EQ_DAE_IMPLICIT_INDEX3       = 1300,
  TS_EQ_DAE_IMPLICIT_INDEXHI      = 1500
} TSEquationType;
PETSC_EXTERN const char *const *TSEquationTypes;

/*E
   TSConvergedReason - reason a `TS` method has converged (integrated to the requested time) or not

   Values:
+  `TS_CONVERGED_ITERATING`          - this only occurs if `TSGetConvergedReason()` is called during the `TSSolve()`
.  `TS_CONVERGED_TIME`               - the final time was reached
.  `TS_CONVERGED_ITS`                - the maximum number of iterations (time-steps) was reached prior to the final time
.  `TS_CONVERGED_USER`               - user requested termination
.  `TS_CONVERGED_EVENT`              - user requested termination on event detection
.  `TS_CONVERGED_PSEUDO_FATOL`       - stops when function norm decreased by a set amount, used only for `TSPSEUDO`
.  `TS_CONVERGED_PSEUDO_FRTOL`       - stops when function norm decreases below a set amount, used only for `TSPSEUDO`
.  `TS_DIVERGED_NONLINEAR_SOLVE`     - too many nonlinear solve failures have occurred
.  `TS_DIVERGED_STEP_REJECTED`       - too many steps were rejected
.  `TSFORWARD_DIVERGED_LINEAR_SOLVE` - tangent linear solve failed
-  `TSADJOINT_DIVERGED_LINEAR_SOLVE` - transposed linear solve failed

   Level: beginner

.seealso: [](chapter_ts), `TS`, `TSGetConvergedReason()`
E*/
typedef enum {
  TS_CONVERGED_ITERATING          = 0,
  TS_CONVERGED_TIME               = 1,
  TS_CONVERGED_ITS                = 2,
  TS_CONVERGED_USER               = 3,
  TS_CONVERGED_EVENT              = 4,
  TS_CONVERGED_PSEUDO_FATOL       = 5,
  TS_CONVERGED_PSEUDO_FRTOL       = 6,
  TS_DIVERGED_NONLINEAR_SOLVE     = -1,
  TS_DIVERGED_STEP_REJECTED       = -2,
  TSFORWARD_DIVERGED_LINEAR_SOLVE = -3,
  TSADJOINT_DIVERGED_LINEAR_SOLVE = -4
} TSConvergedReason;
PETSC_EXTERN const char *const *TSConvergedReasons;

/*MC
   TS_CONVERGED_ITERATING - this only occurs if `TSGetConvergedReason()` is called during the `TSSolve()`

   Level: beginner

.seealso: [](chapter_ts), `TS`, `TSSolve()`, `TSGetConvergedReason()`, `TSGetAdapt()`
M*/

/*MC
   TS_CONVERGED_TIME - the final time was reached

   Level: beginner

.seealso: [](chapter_ts), `TS`, `TSSolve()`, `TSGetConvergedReason()`, `TSGetAdapt()`, `TSSetMaxTime()`, `TSGetMaxTime()`, `TSGetSolveTime()`
M*/

/*MC
   TS_CONVERGED_ITS - the maximum number of iterations (time-steps) was reached prior to the final time

   Level: beginner

.seealso: [](chapter_ts), `TS`, `TSSolve()`, `TSGetConvergedReason()`, `TSGetAdapt()`, `TSSetMaxSteps()`, `TSGetMaxSteps()`
M*/

/*MC
   TS_CONVERGED_USER - user requested termination

   Level: beginner

.seealso: [](chapter_ts), `TS`, `TSSolve()`, `TSGetConvergedReason()`, `TSSetConvergedReason()`
M*/

/*MC
   TS_CONVERGED_EVENT - user requested termination on event detection

   Level: beginner

.seealso: [](chapter_ts), `TS`, `TSSolve()`, `TSGetConvergedReason()`, `TSSetConvergedReason()`
M*/

/*MC
   TS_CONVERGED_PSEUDO_FRTOL - stops when function norm decreased by a set amount, used only for `TSPSEUDO`

   Options Database Key:
.   -ts_pseudo_frtol <rtol> - use specified rtol

   Level: beginner

.seealso: [](chapter_ts), `TS`, `TSSolve()`, `TSGetConvergedReason()`, `TSSetConvergedReason()`, `TS_CONVERGED_PSEUDO_FATOL`
M*/

/*MC
   TS_CONVERGED_PSEUDO_FATOL - stops when function norm decreases below a set amount, used only for `TSPSEUDO`

   Options Database Key:
.   -ts_pseudo_fatol <atol> - use specified atol

   Level: beginner

.seealso: [](chapter_ts), `TS`, `TSSolve()`, `TSGetConvergedReason()`, `TSSetConvergedReason()`, `TS_CONVERGED_PSEUDO_FRTOL`
M*/

/*MC
   TS_DIVERGED_NONLINEAR_SOLVE - too many nonlinear solves failed

   Level: beginner

   Note:
    See `TSSetMaxSNESFailures()` for how to allow more nonlinear solver failures.

.seealso: [](chapter_ts), `TS`, `TSSolve()`, `TSGetConvergedReason()`, `TSGetAdapt()`, `TSGetSNES()`, `SNESGetConvergedReason()`, `TSSetMaxSNESFailures()`
M*/

/*MC
   TS_DIVERGED_STEP_REJECTED - too many steps were rejected

   Level: beginner

   Notes:
    See `TSSetMaxStepRejections()` for how to allow more step rejections.

.seealso: [](chapter_ts), `TS`, `TSSolve()`, `TSGetConvergedReason()`, `TSGetAdapt()`, `TSSetMaxStepRejections()`
M*/

/*E
   TSExactFinalTimeOption - option for handling of final time step

   Values:
+  `TS_EXACTFINALTIME_STEPOVER`    - Don't do anything if requested final time is exceeded
.  `TS_EXACTFINALTIME_INTERPOLATE` - Interpolate back to final time
-  `TS_EXACTFINALTIME_MATCHSTEP` - Adapt final time step to match the final time requested

   Level: beginner

.seealso: [](chapter_ts), `TS`, `TSGetConvergedReason()`, `TSSetExactFinalTime()`, `TSGetExactFinalTime()`
E*/
typedef enum {
  TS_EXACTFINALTIME_UNSPECIFIED = 0,
  TS_EXACTFINALTIME_STEPOVER    = 1,
  TS_EXACTFINALTIME_INTERPOLATE = 2,
  TS_EXACTFINALTIME_MATCHSTEP   = 3
} TSExactFinalTimeOption;
PETSC_EXTERN const char *const TSExactFinalTimeOptions[];

/* Logging support */
PETSC_EXTERN PetscClassId TS_CLASSID;
PETSC_EXTERN PetscClassId DMTS_CLASSID;
PETSC_EXTERN PetscClassId TSADAPT_CLASSID;

PETSC_EXTERN PetscErrorCode TSInitializePackage(void);
PETSC_EXTERN PetscErrorCode TSFinalizePackage(void);

PETSC_EXTERN PetscErrorCode TSCreate(MPI_Comm, TS *);
PETSC_EXTERN PetscErrorCode TSClone(TS, TS *);
PETSC_EXTERN PetscErrorCode TSDestroy(TS *);

PETSC_EXTERN PetscErrorCode TSSetProblemType(TS, TSProblemType);
PETSC_EXTERN PetscErrorCode TSGetProblemType(TS, TSProblemType *);
PETSC_EXTERN PetscErrorCode TSMonitor(TS, PetscInt, PetscReal, Vec);
PETSC_EXTERN PetscErrorCode TSMonitorSet(TS, PetscErrorCode (*)(TS, PetscInt, PetscReal, Vec, void *), void *, PetscErrorCode (*)(void **));
PETSC_EXTERN PetscErrorCode TSMonitorSetFromOptions(TS, const char[], const char[], const char[], PetscErrorCode (*)(TS, PetscInt, PetscReal, Vec, PetscViewerAndFormat *), PetscErrorCode (*)(TS, PetscViewerAndFormat *));
PETSC_EXTERN PetscErrorCode TSMonitorCancel(TS);

PETSC_EXTERN PetscErrorCode TSSetOptionsPrefix(TS, const char[]);
PETSC_EXTERN PetscErrorCode TSAppendOptionsPrefix(TS, const char[]);
PETSC_EXTERN PetscErrorCode TSGetOptionsPrefix(TS, const char *[]);
PETSC_EXTERN PetscErrorCode TSSetFromOptions(TS);
PETSC_EXTERN PetscErrorCode TSSetUp(TS);
PETSC_EXTERN PetscErrorCode TSReset(TS);

PETSC_EXTERN PetscErrorCode TSSetSolution(TS, Vec);
PETSC_EXTERN PetscErrorCode TSGetSolution(TS, Vec *);

PETSC_EXTERN PetscErrorCode TS2SetSolution(TS, Vec, Vec);
PETSC_EXTERN PetscErrorCode TS2GetSolution(TS, Vec *, Vec *);

PETSC_EXTERN PetscErrorCode TSGetSolutionComponents(TS, PetscInt *, Vec *);
PETSC_EXTERN PetscErrorCode TSGetAuxSolution(TS, Vec *);
PETSC_EXTERN PetscErrorCode TSGetTimeError(TS, PetscInt, Vec *);
PETSC_EXTERN PetscErrorCode TSSetTimeError(TS, Vec);

PETSC_EXTERN PetscErrorCode TSSetRHSJacobianP(TS, Mat, PetscErrorCode (*)(TS, PetscReal, Vec, Mat, void *), void *);
PETSC_EXTERN PetscErrorCode TSGetRHSJacobianP(TS, Mat *, PetscErrorCode (**)(TS, PetscReal, Vec, Mat, void *), void **);
PETSC_EXTERN PetscErrorCode TSComputeRHSJacobianP(TS, PetscReal, Vec, Mat);
PETSC_EXTERN PetscErrorCode TSSetIJacobianP(TS, Mat, PetscErrorCode (*)(TS, PetscReal, Vec, Vec, PetscReal, Mat, void *), void *);
PETSC_EXTERN PetscErrorCode TSComputeIJacobianP(TS, PetscReal, Vec, Vec, PetscReal, Mat, PetscBool);
PETSC_EXTERN                PETSC_DEPRECATED_FUNCTION("Use TSGetQuadratureTS then TSComputeRHSJacobianP") PetscErrorCode TSComputeDRDPFunction(TS, PetscReal, Vec, Vec *);
PETSC_EXTERN                PETSC_DEPRECATED_FUNCTION("Use TSGetQuadratureTS then TSComputeRHSJacobian") PetscErrorCode TSComputeDRDUFunction(TS, PetscReal, Vec, Vec *);
PETSC_EXTERN PetscErrorCode TSSetIHessianProduct(TS, Vec *, PetscErrorCode (*)(TS, PetscReal, Vec, Vec *, Vec, Vec *, void *), Vec *, PetscErrorCode (*)(TS, PetscReal, Vec, Vec *, Vec, Vec *, void *), Vec *, PetscErrorCode (*)(TS, PetscReal, Vec, Vec *, Vec, Vec *, void *), Vec *, PetscErrorCode (*)(TS, PetscReal, Vec, Vec *, Vec, Vec *, void *), void *);
PETSC_EXTERN PetscErrorCode TSComputeIHessianProductFunctionUU(TS, PetscReal, Vec, Vec *, Vec, Vec *);
PETSC_EXTERN PetscErrorCode TSComputeIHessianProductFunctionUP(TS, PetscReal, Vec, Vec *, Vec, Vec *);
PETSC_EXTERN PetscErrorCode TSComputeIHessianProductFunctionPU(TS, PetscReal, Vec, Vec *, Vec, Vec *);
PETSC_EXTERN PetscErrorCode TSComputeIHessianProductFunctionPP(TS, PetscReal, Vec, Vec *, Vec, Vec *);
PETSC_EXTERN PetscErrorCode TSSetRHSHessianProduct(TS, Vec *, PetscErrorCode (*)(TS, PetscReal, Vec, Vec *, Vec, Vec *, void *), Vec *, PetscErrorCode (*)(TS, PetscReal, Vec, Vec *, Vec, Vec *, void *), Vec *, PetscErrorCode (*)(TS, PetscReal, Vec, Vec *, Vec, Vec *, void *), Vec *, PetscErrorCode (*)(TS, PetscReal, Vec, Vec *, Vec, Vec *, void *), void *);
PETSC_EXTERN PetscErrorCode TSComputeRHSHessianProductFunctionUU(TS, PetscReal, Vec, Vec *, Vec, Vec *);
PETSC_EXTERN PetscErrorCode TSComputeRHSHessianProductFunctionUP(TS, PetscReal, Vec, Vec *, Vec, Vec *);
PETSC_EXTERN PetscErrorCode TSComputeRHSHessianProductFunctionPU(TS, PetscReal, Vec, Vec *, Vec, Vec *);
PETSC_EXTERN PetscErrorCode TSComputeRHSHessianProductFunctionPP(TS, PetscReal, Vec, Vec *, Vec, Vec *);
PETSC_EXTERN PetscErrorCode TSSetCostHessianProducts(TS, PetscInt, Vec *, Vec *, Vec);
PETSC_EXTERN PetscErrorCode TSGetCostHessianProducts(TS, PetscInt *, Vec **, Vec **, Vec *);
PETSC_EXTERN PetscErrorCode TSComputeSNESJacobian(TS, Vec, Mat, Mat);

/*S
     TSTrajectory - Abstract PETSc object that stores the trajectory (solution of ODE/DAE at each time step)

   Level: advanced

.seealso: [](chapter_ts), `TS`, `TSSetSaveTrajectory()`, `TSTrajectoryCreate()`, `TSTrajectorySetType()`, `TSTrajectoryDestroy()`, `TSTrajectoryReset()`
S*/
typedef struct _p_TSTrajectory *TSTrajectory;

/*J
    TSTrajectoryType - String with the name of a PETSc `TS` trajectory storage method

   Level: intermediate

.seealso: [](chapter_ts), `TS`, `TSSetSaveTrajectory()`, `TSTrajectoryCreate()`, `TSTrajectoryDestroy()`
J*/
typedef const char *TSTrajectoryType;
#define TSTRAJECTORYBASIC         "basic"
#define TSTRAJECTORYSINGLEFILE    "singlefile"
#define TSTRAJECTORYMEMORY        "memory"
#define TSTRAJECTORYVISUALIZATION "visualization"

PETSC_EXTERN PetscFunctionList TSTrajectoryList;
PETSC_EXTERN PetscClassId      TSTRAJECTORY_CLASSID;
PETSC_EXTERN PetscBool         TSTrajectoryRegisterAllCalled;

PETSC_EXTERN PetscErrorCode TSSetSaveTrajectory(TS);
PETSC_EXTERN PetscErrorCode TSResetTrajectory(TS);
PETSC_EXTERN PetscErrorCode TSRemoveTrajectory(TS);

PETSC_EXTERN PetscErrorCode TSTrajectoryCreate(MPI_Comm, TSTrajectory *);
PETSC_EXTERN PetscErrorCode TSTrajectoryReset(TSTrajectory);
PETSC_EXTERN PetscErrorCode TSTrajectoryDestroy(TSTrajectory *);
PETSC_EXTERN PetscErrorCode TSTrajectoryView(TSTrajectory, PetscViewer);
PETSC_EXTERN PetscErrorCode TSTrajectorySetType(TSTrajectory, TS, TSTrajectoryType);
PETSC_EXTERN PetscErrorCode TSTrajectoryGetType(TSTrajectory, TS, TSTrajectoryType *);
PETSC_EXTERN PetscErrorCode TSTrajectorySet(TSTrajectory, TS, PetscInt, PetscReal, Vec);
PETSC_EXTERN PetscErrorCode TSTrajectoryGet(TSTrajectory, TS, PetscInt, PetscReal *);
PETSC_EXTERN PetscErrorCode TSTrajectoryGetVecs(TSTrajectory, TS, PetscInt, PetscReal *, Vec, Vec);
PETSC_EXTERN PetscErrorCode TSTrajectoryGetUpdatedHistoryVecs(TSTrajectory, TS, PetscReal, Vec *, Vec *);
PETSC_EXTERN PetscErrorCode TSTrajectoryGetNumSteps(TSTrajectory, PetscInt *);
PETSC_EXTERN PetscErrorCode TSTrajectoryRestoreUpdatedHistoryVecs(TSTrajectory, Vec *, Vec *);
PETSC_EXTERN PetscErrorCode TSTrajectorySetFromOptions(TSTrajectory, TS);
PETSC_EXTERN PetscErrorCode TSTrajectoryRegister(const char[], PetscErrorCode (*)(TSTrajectory, TS));
PETSC_EXTERN PetscErrorCode TSTrajectoryRegisterAll(void);
PETSC_EXTERN PetscErrorCode TSTrajectorySetUp(TSTrajectory, TS);
PETSC_EXTERN PetscErrorCode TSTrajectorySetUseHistory(TSTrajectory, PetscBool);
PETSC_EXTERN PetscErrorCode TSTrajectorySetMonitor(TSTrajectory, PetscBool);
PETSC_EXTERN PetscErrorCode TSTrajectorySetVariableNames(TSTrajectory, const char *const *);
PETSC_EXTERN PetscErrorCode TSTrajectorySetTransform(TSTrajectory, PetscErrorCode (*)(void *, Vec, Vec *), PetscErrorCode (*)(void *), void *);
PETSC_EXTERN PetscErrorCode TSTrajectorySetSolutionOnly(TSTrajectory, PetscBool);
PETSC_EXTERN PetscErrorCode TSTrajectoryGetSolutionOnly(TSTrajectory, PetscBool *);
PETSC_EXTERN PetscErrorCode TSTrajectorySetKeepFiles(TSTrajectory, PetscBool);
PETSC_EXTERN PetscErrorCode TSTrajectorySetDirname(TSTrajectory, const char[]);
PETSC_EXTERN PetscErrorCode TSTrajectorySetFiletemplate(TSTrajectory, const char[]);
PETSC_EXTERN PetscErrorCode TSGetTrajectory(TS, TSTrajectory *);

PETSC_EXTERN PetscErrorCode TSSetCostGradients(TS, PetscInt, Vec *, Vec *);
PETSC_EXTERN PetscErrorCode TSGetCostGradients(TS, PetscInt *, Vec **, Vec **);
PETSC_EXTERN PETSC_DEPRECATED_FUNCTION("Use TSCreateQuadratureTS() then set up the sub-TS (since version 3.12)") PetscErrorCode TSSetCostIntegrand(TS, PetscInt, Vec, PetscErrorCode (*)(TS, PetscReal, Vec, Vec, void *), PetscErrorCode (*)(TS, PetscReal, Vec, Vec *, void *), PetscErrorCode (*)(TS, PetscReal, Vec, Vec *, void *), PetscBool, void *);
PETSC_EXTERN PetscErrorCode TSGetCostIntegral(TS, Vec *);
PETSC_EXTERN PetscErrorCode TSComputeCostIntegrand(TS, PetscReal, Vec, Vec);
PETSC_EXTERN PetscErrorCode TSCreateQuadratureTS(TS, PetscBool, TS *);
PETSC_EXTERN PetscErrorCode TSGetQuadratureTS(TS, PetscBool *, TS *);

PETSC_EXTERN PetscErrorCode TSAdjointSetFromOptions(TS, PetscOptionItems *);
PETSC_EXTERN PetscErrorCode TSAdjointMonitor(TS, PetscInt, PetscReal, Vec, PetscInt, Vec *, Vec *);
PETSC_EXTERN PetscErrorCode TSAdjointMonitorSet(TS, PetscErrorCode (*)(TS, PetscInt, PetscReal, Vec, PetscInt, Vec *, Vec *, void *), void *, PetscErrorCode (*)(void **));
PETSC_EXTERN PetscErrorCode TSAdjointMonitorCancel(TS);
PETSC_EXTERN PetscErrorCode TSAdjointMonitorSetFromOptions(TS, const char[], const char[], const char[], PetscErrorCode (*)(TS, PetscInt, PetscReal, Vec, PetscInt, Vec *, Vec *, PetscViewerAndFormat *), PetscErrorCode (*)(TS, PetscViewerAndFormat *));

PETSC_EXTERN                PETSC_DEPRECATED_FUNCTION("Use TSSetRHSJacobianP()") PetscErrorCode TSAdjointSetRHSJacobian(TS, Mat, PetscErrorCode (*)(TS, PetscReal, Vec, Mat, void *), void *);
PETSC_EXTERN                PETSC_DEPRECATED_FUNCTION("Use TSComputeRHSJacobianP()") PetscErrorCode TSAdjointComputeRHSJacobian(TS, PetscReal, Vec, Mat);
PETSC_EXTERN                PETSC_DEPRECATED_FUNCTION("Use TSGetQuadratureTS then TSComputeRHSJacobianP") PetscErrorCode TSAdjointComputeDRDPFunction(TS, PetscReal, Vec, Vec *);
PETSC_EXTERN                PETSC_DEPRECATED_FUNCTION("Use TSGetQuadratureTS then TSComputeRHSJacobian") PetscErrorCode TSAdjointComputeDRDYFunction(TS, PetscReal, Vec, Vec *);
PETSC_EXTERN PetscErrorCode TSAdjointSolve(TS);
PETSC_EXTERN PetscErrorCode TSAdjointSetSteps(TS, PetscInt);

PETSC_EXTERN PetscErrorCode TSAdjointStep(TS);
PETSC_EXTERN PetscErrorCode TSAdjointSetUp(TS);
PETSC_EXTERN PetscErrorCode TSAdjointReset(TS);
PETSC_EXTERN PetscErrorCode TSAdjointCostIntegral(TS);
PETSC_EXTERN PetscErrorCode TSAdjointSetForward(TS, Mat);
PETSC_EXTERN PetscErrorCode TSAdjointResetForward(TS);

PETSC_EXTERN PetscErrorCode TSForwardSetSensitivities(TS, PetscInt, Mat);
PETSC_EXTERN PetscErrorCode TSForwardGetSensitivities(TS, PetscInt *, Mat *);
PETSC_EXTERN                PETSC_DEPRECATED_FUNCTION("Use TSCreateQuadratureTS() and TSForwardSetSensitivities() (since version 3.12)") PetscErrorCode TSForwardSetIntegralGradients(TS, PetscInt, Vec *);
PETSC_EXTERN                PETSC_DEPRECATED_FUNCTION("Use TSForwardGetSensitivities()") PetscErrorCode TSForwardGetIntegralGradients(TS, PetscInt *, Vec **);
PETSC_EXTERN PetscErrorCode TSForwardSetUp(TS);
PETSC_EXTERN PetscErrorCode TSForwardReset(TS);
PETSC_EXTERN PetscErrorCode TSForwardCostIntegral(TS);
PETSC_EXTERN PetscErrorCode TSForwardStep(TS);
PETSC_EXTERN PetscErrorCode TSForwardSetInitialSensitivities(TS, Mat);
PETSC_EXTERN PetscErrorCode TSForwardGetStages(TS, PetscInt *, Mat *[]);

PETSC_EXTERN PetscErrorCode TSSetMaxSteps(TS, PetscInt);
PETSC_EXTERN PetscErrorCode TSGetMaxSteps(TS, PetscInt *);
PETSC_EXTERN PetscErrorCode TSSetMaxTime(TS, PetscReal);
PETSC_EXTERN PetscErrorCode TSGetMaxTime(TS, PetscReal *);
PETSC_EXTERN PetscErrorCode TSSetExactFinalTime(TS, TSExactFinalTimeOption);
PETSC_EXTERN PetscErrorCode TSGetExactFinalTime(TS, TSExactFinalTimeOption *);
PETSC_EXTERN PetscErrorCode TSSetTimeSpan(TS, PetscInt, PetscReal *);
PETSC_EXTERN PetscErrorCode TSGetTimeSpan(TS, PetscInt *, const PetscReal **);
PETSC_EXTERN PetscErrorCode TSGetTimeSpanSolutions(TS, PetscInt *, Vec **);

PETSC_EXTERN PETSC_DEPRECATED_FUNCTION("Use TSSetTime[Step]() (since version 3.8)") PetscErrorCode TSSetInitialTimeStep(TS, PetscReal, PetscReal);
PETSC_EXTERN PETSC_DEPRECATED_FUNCTION("Use TSSetMax{Steps|Time}() (since version 3.8)") PetscErrorCode TSSetDuration(TS, PetscInt, PetscReal);
PETSC_EXTERN PETSC_DEPRECATED_FUNCTION("Use TSGetMax{Steps|Time}() (since version 3.8)") PetscErrorCode TSGetDuration(TS, PetscInt *, PetscReal *);
PETSC_EXTERN PETSC_DEPRECATED_FUNCTION("Use TSGetStepNumber() (since version 3.8)") PetscErrorCode TSGetTimeStepNumber(TS, PetscInt *);
PETSC_EXTERN PETSC_DEPRECATED_FUNCTION("Use TSGetStepNumber() (since version 3.8)") PetscErrorCode TSGetTotalSteps(TS, PetscInt *);

PETSC_EXTERN PetscErrorCode TSMonitorDefault(TS, PetscInt, PetscReal, Vec, PetscViewerAndFormat *);
PETSC_EXTERN PetscErrorCode TSMonitorExtreme(TS, PetscInt, PetscReal, Vec, PetscViewerAndFormat *);

typedef struct _n_TSMonitorDrawCtx *TSMonitorDrawCtx;
PETSC_EXTERN PetscErrorCode         TSMonitorDrawCtxCreate(MPI_Comm, const char[], const char[], int, int, int, int, PetscInt, TSMonitorDrawCtx *);
PETSC_EXTERN PetscErrorCode         TSMonitorDrawCtxDestroy(TSMonitorDrawCtx *);
PETSC_EXTERN PetscErrorCode         TSMonitorDrawSolution(TS, PetscInt, PetscReal, Vec, void *);
PETSC_EXTERN PetscErrorCode         TSMonitorDrawSolutionPhase(TS, PetscInt, PetscReal, Vec, void *);
PETSC_EXTERN PetscErrorCode         TSMonitorDrawError(TS, PetscInt, PetscReal, Vec, void *);
PETSC_EXTERN PetscErrorCode         TSMonitorDrawSolutionFunction(TS, PetscInt, PetscReal, Vec, void *);

PETSC_EXTERN PetscErrorCode TSAdjointMonitorDefault(TS, PetscInt, PetscReal, Vec, PetscInt, Vec *, Vec *, PetscViewerAndFormat *);
PETSC_EXTERN PetscErrorCode TSAdjointMonitorDrawSensi(TS, PetscInt, PetscReal, Vec, PetscInt, Vec *, Vec *, void *);

PETSC_EXTERN PetscErrorCode TSMonitorSolution(TS, PetscInt, PetscReal, Vec, PetscViewerAndFormat *);
PETSC_EXTERN PetscErrorCode TSMonitorSolutionVTK(TS, PetscInt, PetscReal, Vec, void *);
PETSC_EXTERN PetscErrorCode TSMonitorSolutionVTKDestroy(void *);

PETSC_EXTERN PetscErrorCode TSStep(TS);
PETSC_EXTERN PetscErrorCode TSEvaluateWLTE(TS, NormType, PetscInt *, PetscReal *);
PETSC_EXTERN PetscErrorCode TSEvaluateStep(TS, PetscInt, Vec, PetscBool *);
PETSC_EXTERN PetscErrorCode TSSolve(TS, Vec);
PETSC_EXTERN PetscErrorCode TSGetEquationType(TS, TSEquationType *);
PETSC_EXTERN PetscErrorCode TSSetEquationType(TS, TSEquationType);
PETSC_EXTERN PetscErrorCode TSGetConvergedReason(TS, TSConvergedReason *);
PETSC_EXTERN PetscErrorCode TSSetConvergedReason(TS, TSConvergedReason);
PETSC_EXTERN PetscErrorCode TSGetSolveTime(TS, PetscReal *);
PETSC_EXTERN PetscErrorCode TSGetSNESIterations(TS, PetscInt *);
PETSC_EXTERN PetscErrorCode TSGetKSPIterations(TS, PetscInt *);
PETSC_EXTERN PetscErrorCode TSGetStepRejections(TS, PetscInt *);
PETSC_EXTERN PetscErrorCode TSSetMaxStepRejections(TS, PetscInt);
PETSC_EXTERN PetscErrorCode TSGetSNESFailures(TS, PetscInt *);
PETSC_EXTERN PetscErrorCode TSSetMaxSNESFailures(TS, PetscInt);
PETSC_EXTERN PetscErrorCode TSSetErrorIfStepFails(TS, PetscBool);
PETSC_EXTERN PetscErrorCode TSRestartStep(TS);
PETSC_EXTERN PetscErrorCode TSRollBack(TS);

PETSC_EXTERN PetscErrorCode TSGetStages(TS, PetscInt *, Vec *[]);

PETSC_EXTERN PetscErrorCode TSGetTime(TS, PetscReal *);
PETSC_EXTERN PetscErrorCode TSSetTime(TS, PetscReal);
PETSC_EXTERN PetscErrorCode TSGetPrevTime(TS, PetscReal *);
PETSC_EXTERN PetscErrorCode TSGetTimeStep(TS, PetscReal *);
PETSC_EXTERN PetscErrorCode TSSetTimeStep(TS, PetscReal);
PETSC_EXTERN PetscErrorCode TSGetStepNumber(TS, PetscInt *);
PETSC_EXTERN PetscErrorCode TSSetStepNumber(TS, PetscInt);

PETSC_EXTERN_TYPEDEF typedef PetscErrorCode (*TSRHSFunction)(TS, PetscReal, Vec, Vec, void *);
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode (*TSRHSJacobian)(TS, PetscReal, Vec, Mat, Mat, void *);
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode (*TSRHSJacobianP)(TS, PetscReal, Vec, Mat, void *);
PETSC_EXTERN PetscErrorCode TSSetRHSFunction(TS, Vec, TSRHSFunction, void *);
PETSC_EXTERN PetscErrorCode TSGetRHSFunction(TS, Vec *, TSRHSFunction *, void **);
PETSC_EXTERN PetscErrorCode TSSetRHSJacobian(TS, Mat, Mat, TSRHSJacobian, void *);
PETSC_EXTERN PetscErrorCode TSGetRHSJacobian(TS, Mat *, Mat *, TSRHSJacobian *, void **);
PETSC_EXTERN PetscErrorCode TSRHSJacobianSetReuse(TS, PetscBool);

PETSC_EXTERN_TYPEDEF typedef PetscErrorCode (*TSSolutionFunction)(TS, PetscReal, Vec, void *);
PETSC_EXTERN PetscErrorCode TSSetSolutionFunction(TS, TSSolutionFunction, void *);
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode (*TSForcingFunction)(TS, PetscReal, Vec, void *);
PETSC_EXTERN PetscErrorCode TSSetForcingFunction(TS, TSForcingFunction, void *);

PETSC_EXTERN_TYPEDEF typedef PetscErrorCode (*TSIFunction)(TS, PetscReal, Vec, Vec, Vec, void *);
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode (*TSIJacobian)(TS, PetscReal, Vec, Vec, PetscReal, Mat, Mat, void *);
PETSC_EXTERN PetscErrorCode TSSetIFunction(TS, Vec, TSIFunction, void *);
PETSC_EXTERN PetscErrorCode TSGetIFunction(TS, Vec *, TSIFunction *, void **);
PETSC_EXTERN PetscErrorCode TSSetIJacobian(TS, Mat, Mat, TSIJacobian, void *);
PETSC_EXTERN PetscErrorCode TSGetIJacobian(TS, Mat *, Mat *, TSIJacobian *, void **);

PETSC_EXTERN_TYPEDEF typedef PetscErrorCode (*TSI2Function)(TS, PetscReal, Vec, Vec, Vec, Vec, void *);
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode (*TSI2Jacobian)(TS, PetscReal, Vec, Vec, Vec, PetscReal, PetscReal, Mat, Mat, void *);
PETSC_EXTERN PetscErrorCode TSSetI2Function(TS, Vec, TSI2Function, void *);
PETSC_EXTERN PetscErrorCode TSGetI2Function(TS, Vec *, TSI2Function *, void **);
PETSC_EXTERN PetscErrorCode TSSetI2Jacobian(TS, Mat, Mat, TSI2Jacobian, void *);
PETSC_EXTERN PetscErrorCode TSGetI2Jacobian(TS, Mat *, Mat *, TSI2Jacobian *, void **);

PETSC_EXTERN PetscErrorCode TSRHSSplitSetIS(TS, const char[], IS);
PETSC_EXTERN PetscErrorCode TSRHSSplitGetIS(TS, const char[], IS *);
PETSC_EXTERN PetscErrorCode TSRHSSplitSetRHSFunction(TS, const char[], Vec, TSRHSFunction, void *);
PETSC_EXTERN PetscErrorCode TSRHSSplitGetSubTS(TS, const char[], TS *);
PETSC_EXTERN PetscErrorCode TSRHSSplitGetSubTSs(TS, PetscInt *, TS *[]);
PETSC_EXTERN PetscErrorCode TSSetUseSplitRHSFunction(TS, PetscBool);
PETSC_EXTERN PetscErrorCode TSGetUseSplitRHSFunction(TS, PetscBool *);

PETSC_EXTERN PetscErrorCode TSComputeRHSFunctionLinear(TS, PetscReal, Vec, Vec, void *);
PETSC_EXTERN PetscErrorCode TSComputeRHSJacobianConstant(TS, PetscReal, Vec, Mat, Mat, void *);
PETSC_EXTERN PetscErrorCode TSComputeIFunctionLinear(TS, PetscReal, Vec, Vec, Vec, void *);
PETSC_EXTERN PetscErrorCode TSComputeIJacobianConstant(TS, PetscReal, Vec, Vec, PetscReal, Mat, Mat, void *);
PETSC_EXTERN PetscErrorCode TSComputeSolutionFunction(TS, PetscReal, Vec);
PETSC_EXTERN PetscErrorCode TSComputeForcingFunction(TS, PetscReal, Vec);
PETSC_EXTERN PetscErrorCode TSComputeIJacobianDefaultColor(TS, PetscReal, Vec, Vec, PetscReal, Mat, Mat, void *);
PETSC_EXTERN PetscErrorCode TSPruneIJacobianColor(TS, Mat, Mat);

PETSC_EXTERN PetscErrorCode TSSetPreStep(TS, PetscErrorCode (*)(TS));
PETSC_EXTERN PetscErrorCode TSSetPreStage(TS, PetscErrorCode (*)(TS, PetscReal));
PETSC_EXTERN PetscErrorCode TSSetPostStage(TS, PetscErrorCode (*)(TS, PetscReal, PetscInt, Vec *));
PETSC_EXTERN PetscErrorCode TSSetPostEvaluate(TS, PetscErrorCode (*)(TS));
PETSC_EXTERN PetscErrorCode TSSetPostStep(TS, PetscErrorCode (*)(TS));
PETSC_EXTERN PetscErrorCode TSPreStep(TS);
PETSC_EXTERN PetscErrorCode TSPreStage(TS, PetscReal);
PETSC_EXTERN PetscErrorCode TSPostStage(TS, PetscReal, PetscInt, Vec *);
PETSC_EXTERN PetscErrorCode TSPostEvaluate(TS);
PETSC_EXTERN PetscErrorCode TSPostStep(TS);
PETSC_EXTERN PetscErrorCode TSInterpolate(TS, PetscReal, Vec);
PETSC_EXTERN PetscErrorCode TSSetTolerances(TS, PetscReal, Vec, PetscReal, Vec);
PETSC_EXTERN PetscErrorCode TSGetTolerances(TS, PetscReal *, Vec *, PetscReal *, Vec *);
PETSC_EXTERN PetscErrorCode TSErrorWeightedNormInfinity(TS, Vec, Vec, PetscReal *, PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode TSErrorWeightedNorm2(TS, Vec, Vec, PetscReal *, PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode TSErrorWeightedNorm(TS, Vec, Vec, NormType, PetscReal *, PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode TSErrorWeightedENormInfinity(TS, Vec, Vec, Vec, PetscReal *, PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode TSErrorWeightedENorm2(TS, Vec, Vec, Vec, PetscReal *, PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode TSErrorWeightedENorm(TS, Vec, Vec, Vec, NormType, PetscReal *, PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode TSSetCFLTimeLocal(TS, PetscReal);
PETSC_EXTERN PetscErrorCode TSGetCFLTime(TS, PetscReal *);
PETSC_EXTERN PetscErrorCode TSSetFunctionDomainError(TS, PetscErrorCode (*)(TS, PetscReal, Vec, PetscBool *));
PETSC_EXTERN PetscErrorCode TSFunctionDomainError(TS, PetscReal, Vec, PetscBool *);

PETSC_EXTERN PetscErrorCode TSPseudoSetTimeStep(TS, PetscErrorCode (*)(TS, PetscReal *, void *), void *);
PETSC_EXTERN PetscErrorCode TSPseudoTimeStepDefault(TS, PetscReal *, void *);
PETSC_EXTERN PetscErrorCode TSPseudoComputeTimeStep(TS, PetscReal *);
PETSC_EXTERN PetscErrorCode TSPseudoSetMaxTimeStep(TS, PetscReal);
PETSC_EXTERN PetscErrorCode TSPseudoSetVerifyTimeStep(TS, PetscErrorCode (*)(TS, Vec, void *, PetscReal *, PetscBool *), void *);
PETSC_EXTERN PetscErrorCode TSPseudoVerifyTimeStepDefault(TS, Vec, void *, PetscReal *, PetscBool *);
PETSC_EXTERN PetscErrorCode TSPseudoVerifyTimeStep(TS, Vec, PetscReal *, PetscBool *);
PETSC_EXTERN PetscErrorCode TSPseudoSetTimeStepIncrement(TS, PetscReal);
PETSC_EXTERN PetscErrorCode TSPseudoIncrementDtFromInitialDt(TS);

PETSC_EXTERN PetscErrorCode TSPythonSetType(TS, const char[]);
PETSC_EXTERN PetscErrorCode TSPythonGetType(TS, const char *[]);

PETSC_EXTERN PetscErrorCode TSComputeRHSFunction(TS, PetscReal, Vec, Vec);
PETSC_EXTERN PetscErrorCode TSComputeRHSJacobian(TS, PetscReal, Vec, Mat, Mat);
PETSC_EXTERN PetscErrorCode TSComputeIFunction(TS, PetscReal, Vec, Vec, Vec, PetscBool);
PETSC_EXTERN PetscErrorCode TSComputeIJacobian(TS, PetscReal, Vec, Vec, PetscReal, Mat, Mat, PetscBool);
PETSC_EXTERN PetscErrorCode TSComputeI2Function(TS, PetscReal, Vec, Vec, Vec, Vec);
PETSC_EXTERN PetscErrorCode TSComputeI2Jacobian(TS, PetscReal, Vec, Vec, Vec, PetscReal, PetscReal, Mat, Mat);
PETSC_EXTERN PetscErrorCode TSComputeLinearStability(TS, PetscReal, PetscReal, PetscReal *, PetscReal *);

PETSC_EXTERN PetscErrorCode TSVISetVariableBounds(TS, Vec, Vec);

PETSC_EXTERN PetscErrorCode DMTSSetBoundaryLocal(DM, PetscErrorCode (*)(DM, PetscReal, Vec, Vec, void *), void *);
PETSC_EXTERN PetscErrorCode DMTSSetRHSFunction(DM, TSRHSFunction, void *);
PETSC_EXTERN PetscErrorCode DMTSGetRHSFunction(DM, TSRHSFunction *, void **);
PETSC_EXTERN PetscErrorCode DMTSSetRHSFunctionContextDestroy(DM, PetscErrorCode (*)(void *));
PETSC_EXTERN PetscErrorCode DMTSSetRHSJacobian(DM, TSRHSJacobian, void *);
PETSC_EXTERN PetscErrorCode DMTSGetRHSJacobian(DM, TSRHSJacobian *, void **);
PETSC_EXTERN PetscErrorCode DMTSSetRHSJacobianContextDestroy(DM, PetscErrorCode (*)(void *));
PETSC_EXTERN PetscErrorCode DMTSSetIFunction(DM, TSIFunction, void *);
PETSC_EXTERN PetscErrorCode DMTSGetIFunction(DM, TSIFunction *, void **);
PETSC_EXTERN PetscErrorCode DMTSSetIFunctionContextDestroy(DM, PetscErrorCode (*)(void *));
PETSC_EXTERN PetscErrorCode DMTSSetIJacobian(DM, TSIJacobian, void *);
PETSC_EXTERN PetscErrorCode DMTSGetIJacobian(DM, TSIJacobian *, void **);
PETSC_EXTERN PetscErrorCode DMTSSetIJacobianContextDestroy(DM, PetscErrorCode (*)(void *));
PETSC_EXTERN PetscErrorCode DMTSSetI2Function(DM, TSI2Function, void *);
PETSC_EXTERN PetscErrorCode DMTSGetI2Function(DM, TSI2Function *, void **);
PETSC_EXTERN PetscErrorCode DMTSSetI2FunctionContextDestroy(DM, PetscErrorCode (*)(void *));
PETSC_EXTERN PetscErrorCode DMTSSetI2Jacobian(DM, TSI2Jacobian, void *);
PETSC_EXTERN PetscErrorCode DMTSGetI2Jacobian(DM, TSI2Jacobian *, void **);
PETSC_EXTERN PetscErrorCode DMTSSetI2JacobianContextDestroy(DM, PetscErrorCode (*)(void *));

PETSC_EXTERN_TYPEDEF typedef PetscErrorCode (*TSTransientVariable)(TS, Vec, Vec, void *);
PETSC_EXTERN PetscErrorCode TSSetTransientVariable(TS, TSTransientVariable, void *);
PETSC_EXTERN PetscErrorCode DMTSSetTransientVariable(DM, TSTransientVariable, void *);
PETSC_EXTERN PetscErrorCode DMTSGetTransientVariable(DM, TSTransientVariable *, void *);
PETSC_EXTERN PetscErrorCode TSComputeTransientVariable(TS, Vec, Vec);
PETSC_EXTERN PetscErrorCode TSHasTransientVariable(TS, PetscBool *);

PETSC_EXTERN PetscErrorCode DMTSSetSolutionFunction(DM, TSSolutionFunction, void *);
PETSC_EXTERN PetscErrorCode DMTSGetSolutionFunction(DM, TSSolutionFunction *, void **);
PETSC_EXTERN PetscErrorCode DMTSSetForcingFunction(DM, TSForcingFunction, void *);
PETSC_EXTERN PetscErrorCode DMTSGetForcingFunction(DM, TSForcingFunction *, void **);
PETSC_EXTERN PetscErrorCode DMTSCheckResidual(TS, DM, PetscReal, Vec, Vec, PetscReal, PetscReal *);
PETSC_EXTERN PetscErrorCode DMTSCheckJacobian(TS, DM, PetscReal, Vec, Vec, PetscReal, PetscBool *, PetscReal *);
PETSC_EXTERN PetscErrorCode DMTSCheckFromOptions(TS, Vec);

PETSC_EXTERN PetscErrorCode DMTSGetIFunctionLocal(DM, PetscErrorCode (**)(DM, PetscReal, Vec, Vec, Vec, void *), void **);
PETSC_EXTERN PetscErrorCode DMTSSetIFunctionLocal(DM, PetscErrorCode (*)(DM, PetscReal, Vec, Vec, Vec, void *), void *);
PETSC_EXTERN PetscErrorCode DMTSGetIJacobianLocal(DM, PetscErrorCode (**)(DM, PetscReal, Vec, Vec, PetscReal, Mat, Mat, void *), void **);
PETSC_EXTERN PetscErrorCode DMTSSetIJacobianLocal(DM, PetscErrorCode (*)(DM, PetscReal, Vec, Vec, PetscReal, Mat, Mat, void *), void *);
PETSC_EXTERN PetscErrorCode DMTSGetRHSFunctionLocal(DM, PetscErrorCode (**)(DM, PetscReal, Vec, Vec, void *), void **);
PETSC_EXTERN PetscErrorCode DMTSSetRHSFunctionLocal(DM, PetscErrorCode (*)(DM, PetscReal, Vec, Vec, void *), void *);
PETSC_EXTERN PetscErrorCode DMTSCreateRHSMassMatrix(DM);
PETSC_EXTERN PetscErrorCode DMTSCreateRHSMassMatrixLumped(DM);
PETSC_EXTERN PetscErrorCode DMTSDestroyRHSMassMatrix(DM);

PETSC_EXTERN PetscErrorCode DMTSSetIFunctionSerialize(DM, PetscErrorCode (*)(void *, PetscViewer), PetscErrorCode (*)(void **, PetscViewer));
PETSC_EXTERN PetscErrorCode DMTSSetIJacobianSerialize(DM, PetscErrorCode (*)(void *, PetscViewer), PetscErrorCode (*)(void **, PetscViewer));

PETSC_EXTERN_TYPEDEF typedef PetscErrorCode (*DMDATSRHSFunctionLocal)(DMDALocalInfo *, PetscReal, void *, void *, void *);
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode (*DMDATSRHSJacobianLocal)(DMDALocalInfo *, PetscReal, void *, Mat, Mat, void *);
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode (*DMDATSIFunctionLocal)(DMDALocalInfo *, PetscReal, void *, void *, void *, void *);
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode (*DMDATSIJacobianLocal)(DMDALocalInfo *, PetscReal, void *, void *, PetscReal, Mat, Mat, void *);

PETSC_EXTERN PetscErrorCode DMDATSSetRHSFunctionLocal(DM, InsertMode, PetscErrorCode (*)(DMDALocalInfo *, PetscReal, void *, void *, void *), void *);
PETSC_EXTERN PetscErrorCode DMDATSSetRHSJacobianLocal(DM, PetscErrorCode (*)(DMDALocalInfo *, PetscReal, void *, Mat, Mat, void *), void *);
PETSC_EXTERN PetscErrorCode DMDATSSetIFunctionLocal(DM, InsertMode, PetscErrorCode (*)(DMDALocalInfo *, PetscReal, void *, void *, void *, void *), void *);
PETSC_EXTERN PetscErrorCode DMDATSSetIJacobianLocal(DM, PetscErrorCode (*)(DMDALocalInfo *, PetscReal, void *, void *, PetscReal, Mat, Mat, void *), void *);

PETSC_EXTERN PetscErrorCode DMPlexTSGetGeometryFVM(DM, Vec *, Vec *, PetscReal *);

typedef struct _n_TSMonitorLGCtx *TSMonitorLGCtx;
typedef struct {
  Vec            ray;
  VecScatter     scatter;
  PetscViewer    viewer;
  TSMonitorLGCtx lgctx;
} TSMonitorDMDARayCtx;
PETSC_EXTERN PetscErrorCode TSMonitorDMDARayDestroy(void **);
PETSC_EXTERN PetscErrorCode TSMonitorDMDARay(TS, PetscInt, PetscReal, Vec, void *);
PETSC_EXTERN PetscErrorCode TSMonitorLGDMDARay(TS, PetscInt, PetscReal, Vec, void *);

/* Dynamic creation and loading functions */
PETSC_EXTERN PetscFunctionList TSList;
PETSC_EXTERN PetscErrorCode    TSGetType(TS, TSType *);
PETSC_EXTERN PetscErrorCode    TSSetType(TS, TSType);
PETSC_EXTERN PetscErrorCode    TSRegister(const char[], PetscErrorCode (*)(TS));

PETSC_EXTERN PetscErrorCode TSGetSNES(TS, SNES *);
PETSC_EXTERN PetscErrorCode TSSetSNES(TS, SNES);
PETSC_EXTERN PetscErrorCode TSGetKSP(TS, KSP *);

PETSC_EXTERN PetscErrorCode TSView(TS, PetscViewer);
PETSC_EXTERN PetscErrorCode TSLoad(TS, PetscViewer);
PETSC_EXTERN PetscErrorCode TSViewFromOptions(TS, PetscObject, const char[]);
PETSC_EXTERN PetscErrorCode TSTrajectoryViewFromOptions(TSTrajectory, PetscObject, const char[]);

#define TS_FILE_CLASSID 1211225

PETSC_EXTERN PetscErrorCode TSSetApplicationContext(TS, void *);
PETSC_EXTERN PetscErrorCode TSGetApplicationContext(TS, void *);

PETSC_EXTERN PetscErrorCode TSMonitorLGCtxCreate(MPI_Comm, const char[], const char[], int, int, int, int, PetscInt, TSMonitorLGCtx *);
PETSC_EXTERN PetscErrorCode TSMonitorLGCtxDestroy(TSMonitorLGCtx *);
PETSC_EXTERN PetscErrorCode TSMonitorLGTimeStep(TS, PetscInt, PetscReal, Vec, void *);
PETSC_EXTERN PetscErrorCode TSMonitorLGSolution(TS, PetscInt, PetscReal, Vec, void *);
PETSC_EXTERN PetscErrorCode TSMonitorLGSetVariableNames(TS, const char *const *);
PETSC_EXTERN PetscErrorCode TSMonitorLGGetVariableNames(TS, const char *const **);
PETSC_EXTERN PetscErrorCode TSMonitorLGCtxSetVariableNames(TSMonitorLGCtx, const char *const *);
PETSC_EXTERN PetscErrorCode TSMonitorLGSetDisplayVariables(TS, const char *const *);
PETSC_EXTERN PetscErrorCode TSMonitorLGCtxSetDisplayVariables(TSMonitorLGCtx, const char *const *);
PETSC_EXTERN PetscErrorCode TSMonitorLGSetTransform(TS, PetscErrorCode (*)(void *, Vec, Vec *), PetscErrorCode (*)(void *), void *);
PETSC_EXTERN PetscErrorCode TSMonitorLGCtxSetTransform(TSMonitorLGCtx, PetscErrorCode (*)(void *, Vec, Vec *), PetscErrorCode (*)(void *), void *);
PETSC_EXTERN PetscErrorCode TSMonitorLGError(TS, PetscInt, PetscReal, Vec, void *);
PETSC_EXTERN PetscErrorCode TSMonitorLGSNESIterations(TS, PetscInt, PetscReal, Vec, void *);
PETSC_EXTERN PetscErrorCode TSMonitorLGKSPIterations(TS, PetscInt, PetscReal, Vec, void *);
PETSC_EXTERN PetscErrorCode TSMonitorError(TS, PetscInt, PetscReal, Vec, PetscViewerAndFormat *);
PETSC_EXTERN PetscErrorCode TSDMSwarmMonitorMoments(TS, PetscInt, PetscReal, Vec, PetscViewerAndFormat *);

struct _n_TSMonitorLGCtxNetwork {
  PetscInt     nlg;
  PetscDrawLG *lg;
  PetscBool    semilogy;
  PetscInt     howoften; /* when > 0 uses step % howoften, when negative only final solution plotted */
};
typedef struct _n_TSMonitorLGCtxNetwork *TSMonitorLGCtxNetwork;
PETSC_EXTERN PetscErrorCode              TSMonitorLGCtxNetworkDestroy(TSMonitorLGCtxNetwork *);
PETSC_EXTERN PetscErrorCode              TSMonitorLGCtxNetworkCreate(TS, const char[], const char[], int, int, int, int, PetscInt, TSMonitorLGCtxNetwork *);
PETSC_EXTERN PetscErrorCode              TSMonitorLGCtxNetworkSolution(TS, PetscInt, PetscReal, Vec, void *);

typedef struct _n_TSMonitorEnvelopeCtx *TSMonitorEnvelopeCtx;
PETSC_EXTERN PetscErrorCode             TSMonitorEnvelopeCtxCreate(TS, TSMonitorEnvelopeCtx *);
PETSC_EXTERN PetscErrorCode             TSMonitorEnvelope(TS, PetscInt, PetscReal, Vec, void *);
PETSC_EXTERN PetscErrorCode             TSMonitorEnvelopeGetBounds(TS, Vec *, Vec *);
PETSC_EXTERN PetscErrorCode             TSMonitorEnvelopeCtxDestroy(TSMonitorEnvelopeCtx *);

typedef struct _n_TSMonitorSPEigCtx *TSMonitorSPEigCtx;
PETSC_EXTERN PetscErrorCode          TSMonitorSPEigCtxCreate(MPI_Comm, const char[], const char[], int, int, int, int, PetscInt, TSMonitorSPEigCtx *);
PETSC_EXTERN PetscErrorCode          TSMonitorSPEigCtxDestroy(TSMonitorSPEigCtx *);
PETSC_EXTERN PetscErrorCode          TSMonitorSPEig(TS, PetscInt, PetscReal, Vec, void *);

typedef struct _n_TSMonitorSPCtx *TSMonitorSPCtx;
PETSC_EXTERN PetscErrorCode       TSMonitorSPCtxCreate(MPI_Comm, const char[], const char[], int, int, int, int, PetscInt, PetscInt, PetscBool, PetscBool, TSMonitorSPCtx *);
PETSC_EXTERN PetscErrorCode       TSMonitorSPCtxDestroy(TSMonitorSPCtx *);
PETSC_EXTERN PetscErrorCode       TSMonitorSPSwarmSolution(TS, PetscInt, PetscReal, Vec, void *);

typedef struct _n_TSMonitorHGCtx *TSMonitorHGCtx;
PETSC_EXTERN PetscErrorCode       TSMonitorHGCtxCreate(MPI_Comm, const char[], const char[], int, int, int, int, PetscInt, PetscInt, PetscInt, PetscBool, TSMonitorHGCtx *);
PETSC_EXTERN PetscErrorCode       TSMonitorHGSwarmSolution(TS, PetscInt, PetscReal, Vec, void *);
PETSC_EXTERN PetscErrorCode       TSMonitorHGCtxDestroy(TSMonitorHGCtx *);
PETSC_EXTERN PetscErrorCode       TSMonitorHGSwarmSolution(TS, PetscInt, PetscReal, Vec, void *);

PETSC_EXTERN PetscErrorCode TSSetEventHandler(TS, PetscInt, PetscInt[], PetscBool[], PetscErrorCode (*)(TS, PetscReal, Vec, PetscScalar[], void *), PetscErrorCode (*)(TS, PetscInt, PetscInt[], PetscReal, Vec, PetscBool, void *), void *);
PETSC_EXTERN PetscErrorCode TSSetPostEventIntervalStep(TS, PetscReal);
PETSC_EXTERN PetscErrorCode TSSetEventTolerances(TS, PetscReal, PetscReal[]);
PETSC_EXTERN PetscErrorCode TSGetNumEvents(TS, PetscInt *);

/*J
   TSSSPType - string with the name of a `TSSSP` scheme.

   Level: beginner

.seealso: [](chapter_ts), `TSSSPSetType()`, `TS`, `TSSSP`
J*/
typedef const char *TSSSPType;
#define TSSSPRKS2  "rks2"
#define TSSSPRKS3  "rks3"
#define TSSSPRK104 "rk104"

PETSC_EXTERN PetscErrorCode    TSSSPSetType(TS, TSSSPType);
PETSC_EXTERN PetscErrorCode    TSSSPGetType(TS, TSSSPType *);
PETSC_EXTERN PetscErrorCode    TSSSPSetNumStages(TS, PetscInt);
PETSC_EXTERN PetscErrorCode    TSSSPGetNumStages(TS, PetscInt *);
PETSC_EXTERN PetscErrorCode    TSSSPInitializePackage(void);
PETSC_EXTERN PetscErrorCode    TSSSPFinalizePackage(void);
PETSC_EXTERN PetscFunctionList TSSSPList;

/*S
   TSAdapt - Abstract object that manages time-step adaptivity

   Level: beginner

.seealso: [](chapter_ts), `TS`, `TSAdaptCreate()`, `TSAdaptType`
S*/
typedef struct _p_TSAdapt *TSAdapt;

/*J
    TSAdaptType - String with the name of `TSAdapt` scheme.

   Level: beginner

.seealso: [](chapter_ts), `TSAdaptSetType()`, `TS`, `TSAdapt`
J*/
typedef const char *TSAdaptType;
#define TSADAPTNONE    "none"
#define TSADAPTBASIC   "basic"
#define TSADAPTDSP     "dsp"
#define TSADAPTCFL     "cfl"
#define TSADAPTGLEE    "glee"
#define TSADAPTHISTORY "history"

PETSC_EXTERN PetscErrorCode TSGetAdapt(TS, TSAdapt *);
PETSC_EXTERN PetscErrorCode TSAdaptRegister(const char[], PetscErrorCode (*)(TSAdapt));
PETSC_EXTERN PetscErrorCode TSAdaptInitializePackage(void);
PETSC_EXTERN PetscErrorCode TSAdaptFinalizePackage(void);
PETSC_EXTERN PetscErrorCode TSAdaptCreate(MPI_Comm, TSAdapt *);
PETSC_EXTERN PetscErrorCode TSAdaptSetType(TSAdapt, TSAdaptType);
PETSC_EXTERN PetscErrorCode TSAdaptGetType(TSAdapt, TSAdaptType *);
PETSC_EXTERN PetscErrorCode TSAdaptSetOptionsPrefix(TSAdapt, const char[]);
PETSC_EXTERN PetscErrorCode TSAdaptCandidatesClear(TSAdapt);
PETSC_EXTERN PetscErrorCode TSAdaptCandidateAdd(TSAdapt, const char[], PetscInt, PetscInt, PetscReal, PetscReal, PetscBool);
PETSC_EXTERN PetscErrorCode TSAdaptCandidatesGet(TSAdapt, PetscInt *, const PetscInt **, const PetscInt **, const PetscReal **, const PetscReal **);
PETSC_EXTERN PetscErrorCode TSAdaptChoose(TSAdapt, TS, PetscReal, PetscInt *, PetscReal *, PetscBool *);
PETSC_EXTERN PetscErrorCode TSAdaptCheckStage(TSAdapt, TS, PetscReal, Vec, PetscBool *);
PETSC_EXTERN PetscErrorCode TSAdaptView(TSAdapt, PetscViewer);
PETSC_EXTERN PetscErrorCode TSAdaptLoad(TSAdapt, PetscViewer);
PETSC_EXTERN PetscErrorCode TSAdaptSetFromOptions(TSAdapt, PetscOptionItems *);
PETSC_EXTERN PetscErrorCode TSAdaptReset(TSAdapt);
PETSC_EXTERN PetscErrorCode TSAdaptDestroy(TSAdapt *);
PETSC_EXTERN PetscErrorCode TSAdaptSetMonitor(TSAdapt, PetscBool);
PETSC_EXTERN PetscErrorCode TSAdaptSetAlwaysAccept(TSAdapt, PetscBool);
PETSC_EXTERN PetscErrorCode TSAdaptSetSafety(TSAdapt, PetscReal, PetscReal);
PETSC_EXTERN PetscErrorCode TSAdaptGetSafety(TSAdapt, PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode TSAdaptSetMaxIgnore(TSAdapt, PetscReal);
PETSC_EXTERN PetscErrorCode TSAdaptGetMaxIgnore(TSAdapt, PetscReal *);
PETSC_EXTERN PetscErrorCode TSAdaptSetClip(TSAdapt, PetscReal, PetscReal);
PETSC_EXTERN PetscErrorCode TSAdaptGetClip(TSAdapt, PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode TSAdaptSetScaleSolveFailed(TSAdapt, PetscReal);
PETSC_EXTERN PetscErrorCode TSAdaptGetScaleSolveFailed(TSAdapt, PetscReal *);
PETSC_EXTERN PetscErrorCode TSAdaptSetStepLimits(TSAdapt, PetscReal, PetscReal);
PETSC_EXTERN PetscErrorCode TSAdaptGetStepLimits(TSAdapt, PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode TSAdaptSetCheckStage(TSAdapt, PetscErrorCode (*)(TSAdapt, TS, PetscReal, Vec, PetscBool *));
PETSC_EXTERN PetscErrorCode TSAdaptHistorySetHistory(TSAdapt, PetscInt n, PetscReal hist[], PetscBool);
PETSC_EXTERN PetscErrorCode TSAdaptHistorySetTrajectory(TSAdapt, TSTrajectory, PetscBool);
PETSC_EXTERN PetscErrorCode TSAdaptHistoryGetStep(TSAdapt, PetscInt, PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode TSAdaptSetTimeStepIncreaseDelay(TSAdapt, PetscInt);
PETSC_EXTERN PetscErrorCode TSAdaptDSPSetFilter(TSAdapt, const char *);
PETSC_EXTERN PetscErrorCode TSAdaptDSPSetPID(TSAdapt, PetscReal, PetscReal, PetscReal);

/*S
   TSGLLEAdapt - Abstract object that manages time-step adaptivity for `TSGLLE`

   Level: beginner

   Developer Note:
   This functionality should be replaced by the `TSAdapt`.

.seealso: [](chapter_ts), `TS`, `TSGLLE`, `TSGLLEAdaptCreate()`, `TSGLLEAdaptType`
S*/
typedef struct _p_TSGLLEAdapt *TSGLLEAdapt;

/*J
    TSGLLEAdaptType - String with the name of `TSGLLEAdapt` scheme

   Level: beginner

   Developer Note:
   This functionality should be replaced by the `TSAdaptType`.

.seealso: [](chapter_ts), `TSGLLEAdaptSetType()`, `TS`
J*/
typedef const char *TSGLLEAdaptType;
#define TSGLLEADAPT_NONE "none"
#define TSGLLEADAPT_SIZE "size"
#define TSGLLEADAPT_BOTH "both"

PETSC_EXTERN PetscErrorCode TSGLLEAdaptRegister(const char[], PetscErrorCode (*)(TSGLLEAdapt));
PETSC_EXTERN PetscErrorCode TSGLLEAdaptInitializePackage(void);
PETSC_EXTERN PetscErrorCode TSGLLEAdaptFinalizePackage(void);
PETSC_EXTERN PetscErrorCode TSGLLEAdaptCreate(MPI_Comm, TSGLLEAdapt *);
PETSC_EXTERN PetscErrorCode TSGLLEAdaptSetType(TSGLLEAdapt, TSGLLEAdaptType);
PETSC_EXTERN PetscErrorCode TSGLLEAdaptSetOptionsPrefix(TSGLLEAdapt, const char[]);
PETSC_EXTERN PetscErrorCode TSGLLEAdaptChoose(TSGLLEAdapt, PetscInt, const PetscInt[], const PetscReal[], const PetscReal[], PetscInt, PetscReal, PetscReal, PetscInt *, PetscReal *, PetscBool *);
PETSC_EXTERN PetscErrorCode TSGLLEAdaptView(TSGLLEAdapt, PetscViewer);
PETSC_EXTERN PetscErrorCode TSGLLEAdaptSetFromOptions(TSGLLEAdapt, PetscOptionItems *);
PETSC_EXTERN PetscErrorCode TSGLLEAdaptDestroy(TSGLLEAdapt *);

/*J
    TSGLLEAcceptType - String with the name of `TSGLLEAccept` scheme

   Level: beginner

.seealso: [](chapter_ts), `TSGLLESetAcceptType()`, `TS`, `TSGLLEAccept`
J*/
typedef const char *TSGLLEAcceptType;
#define TSGLLEACCEPT_ALWAYS "always"

PETSC_EXTERN_TYPEDEF typedef PetscErrorCode (*TSGLLEAcceptFunction)(TS, PetscReal, PetscReal, const PetscReal[], PetscBool *);
PETSC_EXTERN PetscErrorCode TSGLLEAcceptRegister(const char[], TSGLLEAcceptFunction);

/*J
  TSGLLEType - string with the name of a General Linear `TSGLLE` type

  Level: beginner

.seealso: [](chapter_ts), `TS`, `TSGLLE`, `TSGLLESetType()`, `TSGLLERegister()`, `TSGLLEAccept`
J*/
typedef const char *TSGLLEType;
#define TSGLLE_IRKS "irks"

PETSC_EXTERN PetscErrorCode TSGLLERegister(const char[], PetscErrorCode (*)(TS));
PETSC_EXTERN PetscErrorCode TSGLLEInitializePackage(void);
PETSC_EXTERN PetscErrorCode TSGLLEFinalizePackage(void);
PETSC_EXTERN PetscErrorCode TSGLLESetType(TS, TSGLLEType);
PETSC_EXTERN PetscErrorCode TSGLLEGetAdapt(TS, TSGLLEAdapt *);
PETSC_EXTERN PetscErrorCode TSGLLESetAcceptType(TS, TSGLLEAcceptType);

/*J
    TSEIMEXType - String with the name of an Extrapolated IMEX `TSEIMEX` type

   Level: beginner

.seealso: [](chapter_ts), `TSEIMEXSetType()`, `TS`, `TSEIMEX`, `TSEIMEXRegister()`
J*/
#define TSEIMEXType char *

PETSC_EXTERN PetscErrorCode TSEIMEXSetMaxRows(TS ts, PetscInt);
PETSC_EXTERN PetscErrorCode TSEIMEXSetRowCol(TS ts, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode TSEIMEXSetOrdAdapt(TS, PetscBool);

/*J
    TSRKType - String with the name of a Runge-Kutta `TSRK` type

   Level: beginner

.seealso: [](chapter_ts), `TS`, `TSRKSetType()`, `TS`, `TSRK`, `TSRKRegister()`
J*/
typedef const char *TSRKType;
#define TSRK1FE "1fe"
#define TSRK2A  "2a"
#define TSRK2B  "2b"
#define TSRK3   "3"
#define TSRK3BS "3bs"
#define TSRK4   "4"
#define TSRK5F  "5f"
#define TSRK5DP "5dp"
#define TSRK5BS "5bs"
#define TSRK6VR "6vr"
#define TSRK7VR "7vr"
#define TSRK8VR "8vr"

PETSC_EXTERN PetscErrorCode TSRKGetOrder(TS, PetscInt *);
PETSC_EXTERN PetscErrorCode TSRKGetType(TS, TSRKType *);
PETSC_EXTERN PetscErrorCode TSRKSetType(TS, TSRKType);
PETSC_EXTERN PetscErrorCode TSRKGetTableau(TS, PetscInt *, const PetscReal **, const PetscReal **, const PetscReal **, const PetscReal **, PetscInt *, const PetscReal **, PetscBool *);
PETSC_EXTERN PetscErrorCode TSRKSetMultirate(TS, PetscBool);
PETSC_EXTERN PetscErrorCode TSRKGetMultirate(TS, PetscBool *);
PETSC_EXTERN PetscErrorCode TSRKRegister(TSRKType, PetscInt, PetscInt, const PetscReal[], const PetscReal[], const PetscReal[], const PetscReal[], PetscInt, const PetscReal[]);
PETSC_EXTERN PetscErrorCode TSRKInitializePackage(void);
PETSC_EXTERN PetscErrorCode TSRKFinalizePackage(void);
PETSC_EXTERN PetscErrorCode TSRKRegisterDestroy(void);

/*J
   TSMPRKType - String with the name of a Partitioned Runge-Kutta `TSMPRK` type

   Level: beginner

.seealso: [](chapter_ts), `TSMPRKSetType()`, `TS`, `TSMPRK`, `TSMPRKRegister()`
J*/
typedef const char *TSMPRKType;
#define TSMPRK2A22 "2a22"
#define TSMPRK2A23 "2a23"
#define TSMPRK2A32 "2a32"
#define TSMPRK2A33 "2a33"
#define TSMPRKP2   "p2"
#define TSMPRKP3   "p3"

PETSC_EXTERN PetscErrorCode TSMPRKGetType(TS ts, TSMPRKType *);
PETSC_EXTERN PetscErrorCode TSMPRKSetType(TS ts, TSMPRKType);
PETSC_EXTERN PetscErrorCode TSMPRKRegister(TSMPRKType, PetscInt, PetscInt, PetscInt, PetscInt, const PetscReal[], const PetscReal[], const PetscReal[], const PetscInt[], const PetscReal[], const PetscReal[], const PetscReal[], const PetscInt[], const PetscReal[], const PetscReal[], const PetscReal[]);
PETSC_EXTERN PetscErrorCode TSMPRKInitializePackage(void);
PETSC_EXTERN PetscErrorCode TSMPRKFinalizePackage(void);
PETSC_EXTERN PetscErrorCode TSMPRKRegisterDestroy(void);

/*J
    TSIRKType - String with the name of an implicit Runge-Kutta `TSIRK` type

   Level: beginner

.seealso: [](chapter_ts), `TSIRKSetType()`, `TS`, `TSIRK`, `TSIRKRegister()`
J*/
typedef const char *TSIRKType;
#define TSIRKGAUSS "gauss"

PETSC_EXTERN PetscErrorCode TSIRKGetOrder(TS, PetscInt *);
PETSC_EXTERN PetscErrorCode TSIRKGetType(TS, TSIRKType *);
PETSC_EXTERN PetscErrorCode TSIRKSetType(TS, TSIRKType);
PETSC_EXTERN PetscErrorCode TSIRKGetNumStages(TS, PetscInt *);
PETSC_EXTERN PetscErrorCode TSIRKSetNumStages(TS, PetscInt);
PETSC_EXTERN PetscErrorCode TSIRKRegister(const char[], PetscErrorCode (*function)(TS));
PETSC_EXTERN PetscErrorCode TSIRKTableauCreate(TS, PetscInt, const PetscReal *, const PetscReal *, const PetscReal *, const PetscReal *, const PetscScalar *, const PetscScalar *, const PetscScalar *);
PETSC_EXTERN PetscErrorCode TSIRKInitializePackage(void);
PETSC_EXTERN PetscErrorCode TSIRKFinalizePackage(void);
PETSC_EXTERN PetscErrorCode TSIRKRegisterDestroy(void);

/*J
    TSGLEEType - String with the name of a General Linear with Error Estimation `TSGLEE` type

   Level: beginner

.seealso: [](chapter_ts), `TSGLEESetType()`, `TS`, `TSGLEE`, `TSGLEERegister()`
J*/
typedef const char *TSGLEEType;
#define TSGLEEi1      "BE1"
#define TSGLEE23      "23"
#define TSGLEE24      "24"
#define TSGLEE25I     "25i"
#define TSGLEE35      "35"
#define TSGLEEEXRK2A  "exrk2a"
#define TSGLEERK32G1  "rk32g1"
#define TSGLEERK285EX "rk285ex"

/*J
    TSGLEEMode - String with the mode of error estimation for a General Linear with Error Estimation `TSGLEE` type

   Level: beginner

.seealso: [](chapter_ts), `TSGLEESetMode()`, `TS`, `TSGLEE`, `TSGLEERegister()`
J*/
PETSC_EXTERN PetscErrorCode TSGLEEGetType(TS ts, TSGLEEType *);
PETSC_EXTERN PetscErrorCode TSGLEESetType(TS ts, TSGLEEType);
PETSC_EXTERN PetscErrorCode TSGLEERegister(TSGLEEType, PetscInt, PetscInt, PetscInt, PetscReal, const PetscReal[], const PetscReal[], const PetscReal[], const PetscReal[], const PetscReal[], const PetscReal[], const PetscReal[], const PetscReal[], const PetscReal[], const PetscReal[], PetscInt, const PetscReal[]);
PETSC_EXTERN PetscErrorCode TSGLEEFinalizePackage(void);
PETSC_EXTERN PetscErrorCode TSGLEEInitializePackage(void);
PETSC_EXTERN PetscErrorCode TSGLEERegisterDestroy(void);

/*J
    TSARKIMEXType - String with the name of an Additive Runge-Kutta IMEX `TSARKIMEX` type

   Level: beginner

.seealso: [](chapter_ts), `TSARKIMEXSetType()`, `TS`, `TSARKIMEX`, `TSARKIMEXRegister()`
J*/
typedef const char *TSARKIMEXType;
#define TSARKIMEX1BEE   "1bee"
#define TSARKIMEXA2     "a2"
#define TSARKIMEXL2     "l2"
#define TSARKIMEXARS122 "ars122"
#define TSARKIMEX2C     "2c"
#define TSARKIMEX2D     "2d"
#define TSARKIMEX2E     "2e"
#define TSARKIMEXPRSSP2 "prssp2"
#define TSARKIMEX3      "3"
#define TSARKIMEXBPR3   "bpr3"
#define TSARKIMEXARS443 "ars443"
#define TSARKIMEX4      "4"
#define TSARKIMEX5      "5"
PETSC_EXTERN PetscErrorCode TSARKIMEXGetType(TS ts, TSARKIMEXType *);
PETSC_EXTERN PetscErrorCode TSARKIMEXSetType(TS ts, TSARKIMEXType);
PETSC_EXTERN PetscErrorCode TSARKIMEXSetFullyImplicit(TS, PetscBool);
PETSC_EXTERN PetscErrorCode TSARKIMEXGetFullyImplicit(TS, PetscBool *);
PETSC_EXTERN PetscErrorCode TSARKIMEXRegister(TSARKIMEXType, PetscInt, PetscInt, const PetscReal[], const PetscReal[], const PetscReal[], const PetscReal[], const PetscReal[], const PetscReal[], const PetscReal[], const PetscReal[], PetscInt, const PetscReal[], const PetscReal[]);
PETSC_EXTERN PetscErrorCode TSARKIMEXInitializePackage(void);
PETSC_EXTERN PetscErrorCode TSARKIMEXFinalizePackage(void);
PETSC_EXTERN PetscErrorCode TSARKIMEXRegisterDestroy(void);

/*J
    TSRosWType - String with the name of a Rosenbrock-W `TSROSW` type

   Level: beginner

.seealso: [](chapter_ts), `TSRosWSetType()`, `TS`, `TSROSW`, `TSRosWRegister()`
J*/
typedef const char *TSRosWType;
#define TSROSW2M          "2m"
#define TSROSW2P          "2p"
#define TSROSWRA3PW       "ra3pw"
#define TSROSWRA34PW2     "ra34pw2"
#define TSROSWRODAS3      "rodas3"
#define TSROSWSANDU3      "sandu3"
#define TSROSWASSP3P3S1C  "assp3p3s1c"
#define TSROSWLASSP3P4S2C "lassp3p4s2c"
#define TSROSWLLSSP3P4S2C "llssp3p4s2c"
#define TSROSWARK3        "ark3"
#define TSROSWTHETA1      "theta1"
#define TSROSWTHETA2      "theta2"
#define TSROSWGRK4T       "grk4t"
#define TSROSWSHAMP4      "shamp4"
#define TSROSWVELDD4      "veldd4"
#define TSROSW4L          "4l"

PETSC_EXTERN PetscErrorCode TSRosWGetType(TS, TSRosWType *);
PETSC_EXTERN PetscErrorCode TSRosWSetType(TS, TSRosWType);
PETSC_EXTERN PetscErrorCode TSRosWSetRecomputeJacobian(TS, PetscBool);
PETSC_EXTERN PetscErrorCode TSRosWRegister(TSRosWType, PetscInt, PetscInt, const PetscReal[], const PetscReal[], const PetscReal[], const PetscReal[], PetscInt, const PetscReal[]);
PETSC_EXTERN PetscErrorCode TSRosWRegisterRos4(TSRosWType, PetscReal, PetscReal, PetscReal, PetscReal, PetscReal);
PETSC_EXTERN PetscErrorCode TSRosWInitializePackage(void);
PETSC_EXTERN PetscErrorCode TSRosWFinalizePackage(void);
PETSC_EXTERN PetscErrorCode TSRosWRegisterDestroy(void);

PETSC_EXTERN PetscErrorCode TSBDFSetOrder(TS, PetscInt);
PETSC_EXTERN PetscErrorCode TSBDFGetOrder(TS, PetscInt *);

/*J
  TSBasicSymplecticType - String with the name of a basic symplectic integration `TSBASICSYMPLECTIC` type

  Level: beginner

.seealso: [](chapter_ts), `TSBasicSymplecticSetType()`, `TS`, `TSBASICSYMPLECTIC`, `TSBasicSymplecticRegister()`
J*/
typedef const char *TSBasicSymplecticType;
#define TSBASICSYMPLECTICSIEULER   "1"
#define TSBASICSYMPLECTICVELVERLET "2"
#define TSBASICSYMPLECTIC3         "3"
#define TSBASICSYMPLECTIC4         "4"
PETSC_EXTERN PetscErrorCode TSBasicSymplecticSetType(TS, TSBasicSymplecticType);
PETSC_EXTERN PetscErrorCode TSBasicSymplecticGetType(TS, TSBasicSymplecticType *);
PETSC_EXTERN PetscErrorCode TSBasicSymplecticRegister(TSBasicSymplecticType, PetscInt, PetscInt, PetscReal[], PetscReal[]);
PETSC_EXTERN PetscErrorCode TSBasicSymplecticInitializePackage(void);
PETSC_EXTERN PetscErrorCode TSBasicSymplecticFinalizePackage(void);
PETSC_EXTERN PetscErrorCode TSBasicSymplecticRegisterDestroy(void);

/*J
  TSDISCGRAD - The Discrete Gradient integrator is a timestepper for Hamiltonian systems designed to conserve the first integral (energy),
  but also has the property for some systems of monotonicity in a functional.

  Level: beginner

.seealso: [](chapter_ts), `TS`, TSDiscGradSetFormulation()`, `TSDiscGradGetFormulation()`
J*/
PETSC_EXTERN PetscErrorCode TSDiscGradSetFormulation(TS, PetscErrorCode (*)(TS, PetscReal, Vec, Mat, void *), PetscErrorCode (*)(TS, PetscReal, Vec, PetscScalar *, void *), PetscErrorCode (*)(TS, PetscReal, Vec, Vec, void *), void *);
PETSC_EXTERN PetscErrorCode TSDiscGradIsGonzalez(TS, PetscBool *);
PETSC_EXTERN PetscErrorCode TSDiscGradUseGonzalez(TS, PetscBool);

/*
       PETSc interface to Sundials
*/
#ifdef PETSC_HAVE_SUNDIALS2
typedef enum {
  SUNDIALS_ADAMS = 1,
  SUNDIALS_BDF   = 2
} TSSundialsLmmType;
PETSC_EXTERN const char *const TSSundialsLmmTypes[];
typedef enum {
  SUNDIALS_MODIFIED_GS  = 1,
  SUNDIALS_CLASSICAL_GS = 2
} TSSundialsGramSchmidtType;
PETSC_EXTERN const char *const TSSundialsGramSchmidtTypes[];
PETSC_EXTERN PetscErrorCode    TSSundialsSetType(TS, TSSundialsLmmType);
PETSC_EXTERN PetscErrorCode    TSSundialsGetPC(TS, PC *);
PETSC_EXTERN PetscErrorCode    TSSundialsSetTolerance(TS, PetscReal, PetscReal);
PETSC_EXTERN PetscErrorCode    TSSundialsSetMinTimeStep(TS, PetscReal);
PETSC_EXTERN PetscErrorCode    TSSundialsSetMaxTimeStep(TS, PetscReal);
PETSC_EXTERN PetscErrorCode    TSSundialsGetIterations(TS, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode    TSSundialsSetGramSchmidtType(TS, TSSundialsGramSchmidtType);
PETSC_EXTERN PetscErrorCode    TSSundialsSetGMRESRestart(TS, PetscInt);
PETSC_EXTERN PetscErrorCode    TSSundialsSetLinearTolerance(TS, PetscReal);
PETSC_EXTERN PetscErrorCode    TSSundialsMonitorInternalSteps(TS, PetscBool);
PETSC_EXTERN PetscErrorCode    TSSundialsGetParameters(TS, PetscInt *, long *[], double *[]);
PETSC_EXTERN PetscErrorCode    TSSundialsSetMaxl(TS, PetscInt);
PETSC_EXTERN PetscErrorCode    TSSundialsSetMaxord(TS, PetscInt);
PETSC_EXTERN PetscErrorCode    TSSundialsSetUseDense(TS, PetscBool);
#endif

PETSC_EXTERN PetscErrorCode TSThetaSetTheta(TS, PetscReal);
PETSC_EXTERN PetscErrorCode TSThetaGetTheta(TS, PetscReal *);
PETSC_EXTERN PetscErrorCode TSThetaGetEndpoint(TS, PetscBool *);
PETSC_EXTERN PetscErrorCode TSThetaSetEndpoint(TS, PetscBool);

PETSC_EXTERN PetscErrorCode TSAlphaSetRadius(TS, PetscReal);
PETSC_EXTERN PetscErrorCode TSAlphaSetParams(TS, PetscReal, PetscReal, PetscReal);
PETSC_EXTERN PetscErrorCode TSAlphaGetParams(TS, PetscReal *, PetscReal *, PetscReal *);

PETSC_EXTERN PetscErrorCode TSAlpha2SetRadius(TS, PetscReal);
PETSC_EXTERN PetscErrorCode TSAlpha2SetParams(TS, PetscReal, PetscReal, PetscReal, PetscReal);
PETSC_EXTERN PetscErrorCode TSAlpha2GetParams(TS, PetscReal *, PetscReal *, PetscReal *, PetscReal *);

PETSC_EXTERN PetscErrorCode TSSetDM(TS, DM);
PETSC_EXTERN PetscErrorCode TSGetDM(TS, DM *);

PETSC_EXTERN PetscErrorCode SNESTSFormFunction(SNES, Vec, Vec, void *);
PETSC_EXTERN PetscErrorCode SNESTSFormJacobian(SNES, Vec, Mat, Mat, void *);

PETSC_EXTERN PetscErrorCode TSRHSJacobianTest(TS, PetscBool *);
PETSC_EXTERN PetscErrorCode TSRHSJacobianTestTranspose(TS, PetscBool *);

PETSC_EXTERN PetscErrorCode TSGetComputeInitialCondition(TS, PetscErrorCode (**)(TS, Vec));
PETSC_EXTERN PetscErrorCode TSSetComputeInitialCondition(TS, PetscErrorCode (*)(TS, Vec));
PETSC_EXTERN PetscErrorCode TSComputeInitialCondition(TS, Vec);
PETSC_EXTERN PetscErrorCode TSGetComputeExactError(TS, PetscErrorCode (**)(TS, Vec, Vec));
PETSC_EXTERN PetscErrorCode TSSetComputeExactError(TS, PetscErrorCode (*)(TS, Vec, Vec));
PETSC_EXTERN PetscErrorCode TSComputeExactError(TS, Vec, Vec);
PETSC_EXTERN PetscErrorCode PetscConvEstUseTS(PetscConvEst, PetscBool);

PETSC_EXTERN PetscErrorCode TSSetMatStructure(TS, MatStructure);
#endif
