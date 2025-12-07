/*
   User interface for the timestepping package. This package
   is for use in solving time-dependent PDEs.
*/
#pragma once

#include <petscsnes.h>
#include <petscconvest.h>

/*I <petscts.h> I*/

/* SUBMANSEC = TS */

/*S
   TS - Abstract PETSc object that manages integrating an ODE.

   Level: beginner

.seealso: [](integrator_table), [](ch_ts), `TSCreate()`, `TSSetType()`, `TSType`, `SNES`, `KSP`, `PC`, `TSDestroy()`
S*/
typedef struct _p_TS *TS;

/*J
   TSType - String with the name of a PETSc `TS` method. These are all the time/ODE integrators that PETSc provides.

   Level: beginner

   Note:
   Use `TSSetType()` or the options database key `-ts_type` to set the ODE integrator method to use with a given `TS` object

.seealso: [](integrator_table), [](ch_ts), `TSSetType()`, `TS`, `TSRegister()`
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
#define TSDIRK            "dirk"

/*E
   TSProblemType - Determines the type of problem this `TS` object is to be used to solve

   Values:
 + `TS_LINEAR`    - a linear ODE or DAE
 - `TS_NONLINEAR` - a nonlinear ODE or DAE

   Level: beginner

.seealso: [](ch_ts), `TS`, `TSCreate()`
E*/
typedef enum {
  TS_LINEAR,
  TS_NONLINEAR
} TSProblemType;

/*E
   TSEquationType - type of `TS` problem that is solved

   Values:
+  `TS_EQ_UNSPECIFIED` - (default)
.  `TS_EQ_EXPLICIT`    - {ODE and DAE index 1, 2, 3, HI} F(t,U,U_t) := M(t) U_t - G(U,t) = 0
-  `TS_EQ_IMPLICIT`    - {ODE and DAE index 1, 2, 3, HI} F(t,U,U_t) = 0

   Level: beginner

.seealso: [](ch_ts), `TS`, `TSGetEquationType()`, `TSSetEquationType()`
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

.seealso: [](ch_ts), `TS`, `TSGetConvergedReason()`
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

.seealso: [](ch_ts), `TS`, `TSSolve()`, `TSGetConvergedReason()`, `TSGetAdapt()`
M*/

/*MC
   TS_CONVERGED_TIME - the final time was reached

   Level: beginner

.seealso: [](ch_ts), `TS`, `TSSolve()`, `TSGetConvergedReason()`, `TSGetAdapt()`, `TSSetMaxTime()`, `TSGetMaxTime()`, `TSGetSolveTime()`
M*/

/*MC
   TS_CONVERGED_ITS - the maximum number of iterations (time-steps) was reached prior to the final time

   Level: beginner

.seealso: [](ch_ts), `TS`, `TSSolve()`, `TSGetConvergedReason()`, `TSGetAdapt()`, `TSSetMaxSteps()`, `TSGetMaxSteps()`
M*/

/*MC
   TS_CONVERGED_USER - user requested termination

   Level: beginner

.seealso: [](ch_ts), `TS`, `TSSolve()`, `TSGetConvergedReason()`, `TSSetConvergedReason()`
M*/

/*MC
   TS_CONVERGED_EVENT - user requested termination on event detection

   Level: beginner

.seealso: [](ch_ts), `TS`, `TSSolve()`, `TSGetConvergedReason()`, `TSSetConvergedReason()`
M*/

/*MC
   TS_CONVERGED_PSEUDO_FRTOL - stops when function norm decreased by a set amount, used only for `TSPSEUDO`

   Options Database Key:
.   -ts_pseudo_frtol <rtol> - use specified rtol

   Level: beginner

.seealso: [](ch_ts), `TS`, `TSSolve()`, `TSGetConvergedReason()`, `TSSetConvergedReason()`, `TS_CONVERGED_PSEUDO_FATOL`
M*/

/*MC
   TS_CONVERGED_PSEUDO_FATOL - stops when function norm decreases below a set amount, used only for `TSPSEUDO`

   Options Database Key:
.   -ts_pseudo_fatol <atol> - use specified atol

   Level: beginner

.seealso: [](ch_ts), `TS`, `TSSolve()`, `TSGetConvergedReason()`, `TSSetConvergedReason()`, `TS_CONVERGED_PSEUDO_FRTOL`
M*/

/*MC
   TS_DIVERGED_NONLINEAR_SOLVE - too many nonlinear solves failed

   Level: beginner

   Note:
   See `TSSetMaxSNESFailures()` for how to allow more nonlinear solver failures.

.seealso: [](ch_ts), `TS`, `TSSolve()`, `TSGetConvergedReason()`, `TSGetAdapt()`, `TSGetSNES()`, `SNESGetConvergedReason()`, `TSSetMaxSNESFailures()`
M*/

/*MC
   TS_DIVERGED_STEP_REJECTED - too many steps were rejected

   Level: beginner

   Notes:
   See `TSSetMaxStepRejections()` for how to allow more step rejections.

.seealso: [](ch_ts), `TS`, `TSSolve()`, `TSGetConvergedReason()`, `TSGetAdapt()`, `TSSetMaxStepRejections()`
M*/

/*E
   TSExactFinalTimeOption - option for handling of final time step

   Values:
+  `TS_EXACTFINALTIME_STEPOVER`    - Don't do anything if requested final time is exceeded
.  `TS_EXACTFINALTIME_INTERPOLATE` - Interpolate back to final time
-  `TS_EXACTFINALTIME_MATCHSTEP`   - Adapt final time step to match the final time requested

   Level: beginner

.seealso: [](ch_ts), `TS`, `TSGetConvergedReason()`, `TSSetExactFinalTime()`, `TSGetExactFinalTime()`
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
PETSC_EXTERN PetscErrorCode TSMonitorSet(TS, PetscErrorCode (*)(TS, PetscInt, PetscReal, Vec, void *), void *, PetscCtxDestroyFn *);
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
PETSC_EXTERN PetscErrorCode TSGetIJacobianP(TS, Mat *, PetscErrorCode (**)(TS, PetscReal, Vec, Vec, PetscReal, Mat, void *), void **);
PETSC_EXTERN PetscErrorCode TSComputeIJacobianP(TS, PetscReal, Vec, Vec, PetscReal, Mat, PetscBool);
PETSC_EXTERN PETSC_DEPRECATED_FUNCTION(3, 5, 0, "TSGetQuadratureTS() then TSComputeRHSJacobianP()", ) PetscErrorCode TSComputeDRDPFunction(TS, PetscReal, Vec, Vec *);
PETSC_EXTERN PETSC_DEPRECATED_FUNCTION(3, 5, 0, "TSGetQuadratureTS() then TSComputeRHSJacobian()", ) PetscErrorCode TSComputeDRDUFunction(TS, PetscReal, Vec, Vec *);
PETSC_EXTERN PetscErrorCode TSSetIHessianProduct(TS, Vec *, PetscErrorCode (*)(TS, PetscReal, Vec, Vec *, Vec, Vec *, void *), Vec *, PetscErrorCode (*)(TS, PetscReal, Vec, Vec *, Vec, Vec *, void *), Vec *, PetscErrorCode (*)(TS, PetscReal, Vec, Vec *, Vec, Vec *, void *), Vec *, PetscErrorCode (*)(TS, PetscReal, Vec, Vec *, Vec, Vec *, void *), void *);
PETSC_EXTERN PetscErrorCode TSComputeIHessianProductFunctionUU(TS, PetscReal, Vec, Vec[], Vec, Vec[]);
PETSC_EXTERN PetscErrorCode TSComputeIHessianProductFunctionUP(TS, PetscReal, Vec, Vec[], Vec, Vec[]);
PETSC_EXTERN PetscErrorCode TSComputeIHessianProductFunctionPU(TS, PetscReal, Vec, Vec[], Vec, Vec[]);
PETSC_EXTERN PetscErrorCode TSComputeIHessianProductFunctionPP(TS, PetscReal, Vec, Vec[], Vec, Vec[]);
PETSC_EXTERN PetscErrorCode TSSetRHSHessianProduct(TS, Vec[], PetscErrorCode (*)(TS, PetscReal, Vec, Vec *, Vec, Vec *, void *), Vec[], PetscErrorCode (*)(TS, PetscReal, Vec, Vec *, Vec, Vec *, void *), Vec[], PetscErrorCode (*)(TS, PetscReal, Vec, Vec *, Vec, Vec *, void *), Vec[], PetscErrorCode (*)(TS, PetscReal, Vec, Vec *, Vec, Vec *, void *), void *);
PETSC_EXTERN PetscErrorCode TSComputeRHSHessianProductFunctionUU(TS, PetscReal, Vec, Vec[], Vec, Vec[]);
PETSC_EXTERN PetscErrorCode TSComputeRHSHessianProductFunctionUP(TS, PetscReal, Vec, Vec[], Vec, Vec[]);
PETSC_EXTERN PetscErrorCode TSComputeRHSHessianProductFunctionPU(TS, PetscReal, Vec, Vec[], Vec, Vec[]);
PETSC_EXTERN PetscErrorCode TSComputeRHSHessianProductFunctionPP(TS, PetscReal, Vec, Vec[], Vec, Vec[]);
PETSC_EXTERN PetscErrorCode TSSetCostHessianProducts(TS, PetscInt, Vec[], Vec[], Vec);
PETSC_EXTERN PetscErrorCode TSGetCostHessianProducts(TS, PetscInt *, Vec *[], Vec *[], Vec *);
PETSC_EXTERN PetscErrorCode TSComputeSNESJacobian(TS, Vec, Mat, Mat);

/*S
   TSTrajectory - Abstract PETSc object that stores the trajectory (solution of ODE/DAE at each time step)

   Level: advanced

.seealso: [](ch_ts), `TS`, `TSSetSaveTrajectory()`, `TSTrajectoryCreate()`, `TSTrajectorySetType()`, `TSTrajectoryDestroy()`, `TSTrajectoryReset()`
S*/
typedef struct _p_TSTrajectory *TSTrajectory;

/*J
   TSTrajectoryType - String with the name of a PETSc `TS` trajectory storage method

   Level: intermediate

.seealso: [](ch_ts), `TS`, `TSSetSaveTrajectory()`, `TSTrajectoryCreate()`, `TSTrajectoryDestroy()`
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

typedef enum {
  TJ_REVOLVE,
  TJ_CAMS,
  TJ_PETSC
} TSTrajectoryMemoryType;
PETSC_EXTERN const char *const TSTrajectoryMemoryTypes[];

PETSC_EXTERN PetscErrorCode TSTrajectoryMemorySetType(TSTrajectory, TSTrajectoryMemoryType);
PETSC_EXTERN PetscErrorCode TSTrajectorySetMaxCpsRAM(TSTrajectory, PetscInt);
PETSC_EXTERN PetscErrorCode TSTrajectorySetMaxCpsDisk(TSTrajectory, PetscInt);
PETSC_EXTERN PetscErrorCode TSTrajectorySetMaxUnitsRAM(TSTrajectory, PetscInt);
PETSC_EXTERN PetscErrorCode TSTrajectorySetMaxUnitsDisk(TSTrajectory, PetscInt);

PETSC_EXTERN PetscErrorCode TSSetCostGradients(TS, PetscInt, Vec[], Vec[]);
PETSC_EXTERN PetscErrorCode TSGetCostGradients(TS, PetscInt *, Vec *[], Vec *[]);
PETSC_EXTERN PETSC_DEPRECATED_FUNCTION(3, 12, 0, "TSCreateQuadratureTS() and TSForwardSetSensitivities()", ) PetscErrorCode TSSetCostIntegrand(TS, PetscInt, Vec, PetscErrorCode (*)(TS, PetscReal, Vec, Vec, void *), PetscErrorCode (*)(TS, PetscReal, Vec, Vec *, void *), PetscErrorCode (*)(TS, PetscReal, Vec, Vec *, void *), PetscBool, void *);
PETSC_EXTERN PetscErrorCode TSGetCostIntegral(TS, Vec *);
PETSC_EXTERN PetscErrorCode TSComputeCostIntegrand(TS, PetscReal, Vec, Vec);
PETSC_EXTERN PetscErrorCode TSCreateQuadratureTS(TS, PetscBool, TS *);
PETSC_EXTERN PetscErrorCode TSGetQuadratureTS(TS, PetscBool *, TS *);

PETSC_EXTERN PetscErrorCode TSAdjointSetFromOptions(TS, PetscOptionItems);
PETSC_EXTERN PetscErrorCode TSAdjointMonitor(TS, PetscInt, PetscReal, Vec, PetscInt, Vec[], Vec[]);
PETSC_EXTERN PetscErrorCode TSAdjointMonitorSet(TS, PetscErrorCode (*)(TS, PetscInt, PetscReal, Vec, PetscInt, Vec *, Vec *, void *), void *, PetscCtxDestroyFn *);
PETSC_EXTERN PetscErrorCode TSAdjointMonitorCancel(TS);
PETSC_EXTERN PetscErrorCode TSAdjointMonitorSetFromOptions(TS, const char[], const char[], const char[], PetscErrorCode (*)(TS, PetscInt, PetscReal, Vec, PetscInt, Vec *, Vec *, PetscViewerAndFormat *), PetscErrorCode (*)(TS, PetscViewerAndFormat *));

PETSC_EXTERN PETSC_DEPRECATED_FUNCTION(3, 5, 0, "TSSetRHSJacobianP()", ) PetscErrorCode TSAdjointSetRHSJacobian(TS, Mat, PetscErrorCode (*)(TS, PetscReal, Vec, Mat, void *), void *);
PETSC_EXTERN PETSC_DEPRECATED_FUNCTION(3, 5, 0, "TSComputeRHSJacobianP()", ) PetscErrorCode TSAdjointComputeRHSJacobian(TS, PetscReal, Vec, Mat);
PETSC_EXTERN PETSC_DEPRECATED_FUNCTION(3, 5, 0, "TSGetQuadratureTS()", ) PetscErrorCode TSAdjointComputeDRDPFunction(TS, PetscReal, Vec, Vec *);
PETSC_EXTERN PETSC_DEPRECATED_FUNCTION(3, 5, 0, "TSGetQuadratureTS()", ) PetscErrorCode TSAdjointComputeDRDYFunction(TS, PetscReal, Vec, Vec *);
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
PETSC_EXTERN PETSC_DEPRECATED_FUNCTION(3, 12, 0, "TSCreateQuadratureTS()", ) PetscErrorCode TSForwardSetIntegralGradients(TS, PetscInt, Vec[]);
PETSC_EXTERN PETSC_DEPRECATED_FUNCTION(3, 12, 0, "TSForwardGetSensitivities()", ) PetscErrorCode TSForwardGetIntegralGradients(TS, PetscInt *, Vec *[]);
PETSC_EXTERN PetscErrorCode TSForwardSetUp(TS);
PETSC_EXTERN PetscErrorCode TSForwardReset(TS);
PETSC_EXTERN PetscErrorCode TSForwardCostIntegral(TS);
PETSC_EXTERN PetscErrorCode TSForwardStep(TS);
PETSC_EXTERN PetscErrorCode TSForwardSetInitialSensitivities(TS, Mat);
PETSC_EXTERN PetscErrorCode TSForwardGetStages(TS, PetscInt *, Mat *[]);

PETSC_EXTERN PetscErrorCode TSSetMaxSteps(TS, PetscInt);
PETSC_EXTERN PetscErrorCode TSGetMaxSteps(TS, PetscInt *);
PETSC_EXTERN PetscErrorCode TSSetRunSteps(TS, PetscInt);
PETSC_EXTERN PetscErrorCode TSGetRunSteps(TS, PetscInt *);
PETSC_EXTERN PetscErrorCode TSSetMaxTime(TS, PetscReal);
PETSC_EXTERN PetscErrorCode TSGetMaxTime(TS, PetscReal *);
PETSC_EXTERN PetscErrorCode TSSetExactFinalTime(TS, TSExactFinalTimeOption);
PETSC_EXTERN PetscErrorCode TSGetExactFinalTime(TS, TSExactFinalTimeOption *);
PETSC_EXTERN PetscErrorCode TSSetEvaluationTimes(TS, PetscInt, PetscReal[]);
PETSC_EXTERN PetscErrorCode TSGetEvaluationTimes(TS, PetscInt *, const PetscReal *[]);
PETSC_EXTERN PetscErrorCode TSGetEvaluationSolutions(TS, PetscInt *, const PetscReal *[], Vec *[]);
PETSC_EXTERN PetscErrorCode TSSetTimeSpan(TS, PetscInt, PetscReal[]);

/*@C
  TSGetTimeSpan - gets the time span set with `TSSetTimeSpan()`

  Not Collective

  Input Parameter:
. ts - the time-stepper

  Output Parameters:
+ n          - number of the time points (>=2)
- span_times - array of the time points. The first element and the last element are the initial time and the final time respectively.

  Level: deprecated

  Note:
  Deprecated, use `TSGetEvaluationTimes()`.

  The values obtained are valid until the `TS` object is destroyed.

  Both `n` and `span_times` can be `NULL`.

.seealso: [](ch_ts), `TS`, `TSGetEvaluationTimes()`, `TSSetTimeSpan()`, `TSSetEvaluationTimes()`, `TSGetEvaluationSolutions()`
 @*/
PETSC_DEPRECATED_FUNCTION(3, 23, 0, "TSGetEvaluationTimes()", ) static inline PetscErrorCode TSGetTimeSpan(TS ts, PetscInt *n, const PetscReal *span_times[])
{
  return TSGetEvaluationTimes(ts, n, span_times);
}

/*@C
  TSGetTimeSpanSolutions - Get the number of solutions and the solutions at the time points specified by the time span.

  Input Parameter:
. ts - the `TS` context obtained from `TSCreate()`

  Output Parameters:
+ nsol - the number of solutions
- Sols - the solution vectors

  Level: deprecated

  Notes:
  Deprecated, use `TSGetEvaluationSolutions()`.

  Both `nsol` and `Sols` can be `NULL`.

  Some time points in the time span may be skipped by `TS` so that `nsol` is less than the number of points specified by `TSSetTimeSpan()`.
  For example, manipulating the step size, especially with a reduced precision, may cause `TS` to step over certain points in the span.
  This issue is alleviated in `TSGetEvaluationSolutions()` by returning the solution times that `Sols` were recorded at.

.seealso: [](ch_ts), `TS`, `TSGetEvaluationSolutions()`, `TSSetTimeSpan()`, `TSGetEvaluationTimes()`, `TSSetEvaluationTimes()`
 @*/
PETSC_DEPRECATED_FUNCTION(3, 23, 0, "TSGetEvaluationSolutions()", ) static inline PetscErrorCode TSGetTimeSpanSolutions(TS ts, PetscInt *nsol, Vec **Sols)
{
  return TSGetEvaluationSolutions(ts, nsol, NULL, Sols);
}

PETSC_EXTERN PETSC_DEPRECATED_FUNCTION(3, 8, 0, "TSSetTime()", ) PetscErrorCode TSSetInitialTimeStep(TS, PetscReal, PetscReal);
PETSC_EXTERN PETSC_DEPRECATED_FUNCTION(3, 8, 0, "TSSetMax()", ) PetscErrorCode TSSetDuration(TS, PetscInt, PetscReal);
PETSC_EXTERN PETSC_DEPRECATED_FUNCTION(3, 8, 0, "TSGetMax()", ) PetscErrorCode TSGetDuration(TS, PetscInt *, PetscReal *);
PETSC_EXTERN PETSC_DEPRECATED_FUNCTION(3, 8, 0, "TSGetStepNumber()", ) PetscErrorCode TSGetTimeStepNumber(TS, PetscInt *);
PETSC_EXTERN PETSC_DEPRECATED_FUNCTION(3, 8, 0, "TSGetStepNumber()", ) PetscErrorCode TSGetTotalSteps(TS, PetscInt *);

PETSC_EXTERN PetscErrorCode TSMonitorDefault(TS, PetscInt, PetscReal, Vec, PetscViewerAndFormat *);
PETSC_EXTERN PetscErrorCode TSMonitorWallClockTime(TS, PetscInt, PetscReal, Vec, PetscViewerAndFormat *);
PETSC_EXTERN PetscErrorCode TSMonitorWallClockTimeSetUp(TS, PetscViewerAndFormat *);
PETSC_EXTERN PetscErrorCode TSMonitorExtreme(TS, PetscInt, PetscReal, Vec, PetscViewerAndFormat *);

typedef struct _n_TSMonitorDrawCtx *TSMonitorDrawCtx;
PETSC_EXTERN PetscErrorCode         TSMonitorDrawCtxCreate(MPI_Comm, const char[], const char[], int, int, int, int, PetscInt, TSMonitorDrawCtx *);
PETSC_EXTERN PetscErrorCode         TSMonitorDrawCtxDestroy(TSMonitorDrawCtx *);
PETSC_EXTERN PetscErrorCode         TSMonitorDrawSolution(TS, PetscInt, PetscReal, Vec, void *);
PETSC_EXTERN PetscErrorCode         TSMonitorDrawSolutionPhase(TS, PetscInt, PetscReal, Vec, void *);
PETSC_EXTERN PetscErrorCode         TSMonitorDrawError(TS, PetscInt, PetscReal, Vec, void *);
PETSC_EXTERN PetscErrorCode         TSMonitorDrawSolutionFunction(TS, PetscInt, PetscReal, Vec, void *);

PETSC_EXTERN PetscErrorCode TSAdjointMonitorDefault(TS, PetscInt, PetscReal, Vec, PetscInt, Vec[], Vec[], PetscViewerAndFormat *);
PETSC_EXTERN PetscErrorCode TSAdjointMonitorDrawSensi(TS, PetscInt, PetscReal, Vec, PetscInt, Vec[], Vec[], void *);

typedef struct _n_TSMonitorSolutionCtx *TSMonitorSolutionCtx;
PETSC_EXTERN PetscErrorCode             TSMonitorSolution(TS, PetscInt, PetscReal, Vec, PetscViewerAndFormat *);
PETSC_EXTERN PetscErrorCode             TSMonitorSolutionSetup(TS, PetscViewerAndFormat *);

typedef struct _n_TSMonitorVTKCtx *TSMonitorVTKCtx;
PETSC_EXTERN PetscErrorCode        TSMonitorSolutionVTK(TS, PetscInt, PetscReal, Vec, TSMonitorVTKCtx);
PETSC_EXTERN PetscErrorCode        TSMonitorSolutionVTKDestroy(TSMonitorVTKCtx *);
PETSC_EXTERN PetscErrorCode        TSMonitorSolutionVTKCtxCreate(const char *, TSMonitorVTKCtx *);

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
PETSC_EXTERN PetscErrorCode TSGetStepRollBack(TS, PetscBool *);

PETSC_EXTERN PetscErrorCode TSGetStages(TS, PetscInt *, Vec *[]);

PETSC_EXTERN PetscErrorCode TSGetTime(TS, PetscReal *);
PETSC_EXTERN PetscErrorCode TSSetTime(TS, PetscReal);
PETSC_EXTERN PetscErrorCode TSGetPrevTime(TS, PetscReal *);
PETSC_EXTERN PetscErrorCode TSGetTimeStep(TS, PetscReal *);
PETSC_EXTERN PetscErrorCode TSSetTimeStep(TS, PetscReal);
PETSC_EXTERN PetscErrorCode TSGetStepNumber(TS, PetscInt *);
PETSC_EXTERN PetscErrorCode TSSetStepNumber(TS, PetscInt);

/*S
  TSRHSFunctionFn - A prototype of a `TS` right-hand-side evaluation function that would be passed to `TSSetRHSFunction()`

  Calling Sequence:
+ ts  - timestep context
. t   - current time
. u   - input vector
. F   - function vector
- ctx - [optional] user-defined function context

  Level: beginner

  Note:
  The deprecated `TSRHSFunction` still works as a replacement for `TSRHSFunctionFn` *.

.seealso: [](ch_ts), `TS`, `TSSetRHSFunction()`, `DMTSSetRHSFunction()`, `TSIFunctionFn`,
`TSIJacobianFn`, `TSRHSJacobianFn`
S*/
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode TSRHSFunctionFn(TS ts, PetscReal t, Vec u, Vec F, void *ctx);

PETSC_EXTERN_TYPEDEF typedef TSRHSFunctionFn *TSRHSFunction;

/*S
  TSRHSJacobianFn - A prototype of a `TS` right-hand-side Jacobian evaluation function that would be passed to `TSSetRHSJacobian()`

  Calling Sequence:
+ ts   - the `TS` context obtained from `TSCreate()`
. t    - current time
. u    - input vector
. Amat - (approximate) Jacobian matrix
. Pmat - matrix from which preconditioner is to be constructed (usually the same as `Amat`)
- ctx  - [optional] user-defined context for matrix evaluation routine

  Level: beginner

  Note:
  The deprecated `TSRHSJacobian` still works as a replacement for `TSRHSJacobianFn` *.

.seealso: [](ch_ts), `TS`, `TSSetRHSJacobian()`, `DMTSSetRHSJacobian()`, `TSRHSFunctionFn`,
`TSIFunctionFn`, `TSIJacobianFn`
S*/
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode TSRHSJacobianFn(TS ts, PetscReal t, Vec u, Mat Amat, Mat Pmat, void *ctx);

PETSC_EXTERN_TYPEDEF typedef TSRHSJacobianFn *TSRHSJacobian;

/*S
  TSRHSJacobianPFn - A prototype of a function that computes the Jacobian of G w.r.t. the parameters P where
  U_t = G(U,P,t), as well as the location to store the matrix that would be passed to `TSSetRHSJacobianP()`

  Calling Sequence:
+ ts  - the `TS` context
. t   - current timestep
. U   - input vector (current ODE solution)
. A   - output matrix
- ctx - [optional] user-defined function context

  Level: beginner

  Note:
  The deprecated `TSRHSJacobianP` still works as a replacement for `TSRHSJacobianPFn` *.

.seealso: [](ch_ts), `TS`, `TSSetRHSJacobianP()`, `TSGetRHSJacobianP()`
S*/
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode TSRHSJacobianPFn(TS ts, PetscReal t, Vec U, Mat A, void *ctx);

PETSC_EXTERN_TYPEDEF typedef TSRHSJacobianPFn *TSRHSJacobianP;

PETSC_EXTERN PetscErrorCode TSSetRHSFunction(TS, Vec, TSRHSFunctionFn *, void *);
PETSC_EXTERN PetscErrorCode TSGetRHSFunction(TS, Vec *, TSRHSFunctionFn **, void **);
PETSC_EXTERN PetscErrorCode TSSetRHSJacobian(TS, Mat, Mat, TSRHSJacobianFn *, void *);
PETSC_EXTERN PetscErrorCode TSGetRHSJacobian(TS, Mat *, Mat *, TSRHSJacobianFn **, void **);
PETSC_EXTERN PetscErrorCode TSRHSJacobianSetReuse(TS, PetscBool);

/*S
  TSSolutionFn - A prototype of a `TS` solution evaluation function that would be passed to `TSSetSolutionFunction()`

  Calling Sequence:
+ ts  - timestep context
. t   - current time
. u   - output vector
- ctx - [optional] user-defined function context

  Level: advanced

  Note:
  The deprecated `TSSolutionFunction` still works as a replacement for `TSSolutionFn` *.

.seealso: [](ch_ts), `TS`, `TSSetSolutionFunction()`, `DMTSSetSolutionFunction()`
S*/
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode TSSolutionFn(TS ts, PetscReal t, Vec u, void *ctx);

PETSC_EXTERN_TYPEDEF typedef TSSolutionFn *TSSolutionFunction;

PETSC_EXTERN PetscErrorCode TSSetSolutionFunction(TS, TSSolutionFn *, void *);

/*S
  TSForcingFn - A prototype of a `TS` forcing function evaluation function that would be passed to `TSSetForcingFunction()`

  Calling Sequence:
+ ts  - timestep context
. t   - current time
. f   - output vector
- ctx - [optional] user-defined function context

  Level: advanced

  Note:
  The deprecated `TSForcingFunction` still works as a replacement for `TSForcingFn` *.

.seealso: [](ch_ts), `TS`, `TSSetForcingFunction()`, `DMTSSetForcingFunction()`
S*/
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode TSForcingFn(TS ts, PetscReal t, Vec f, void *ctx);

PETSC_EXTERN_TYPEDEF typedef TSForcingFn *TSForcingFunction;

PETSC_EXTERN PetscErrorCode TSSetForcingFunction(TS, TSForcingFn *, void *);

/*S
  TSIFunctionFn - A prototype of a `TS` implicit function evaluation function that would be passed to `TSSetIFunction()

  Calling Sequence:
+ ts  - the `TS` context obtained from `TSCreate()`
. t   - time at step/stage being solved
. U   - state vector
. U_t - time derivative of state vector
. F   - function vector
- ctx - [optional] user-defined context for function

  Level: beginner

  Note:
  The deprecated `TSIFunction` still works as a replacement for `TSIFunctionFn` *.

.seealso: [](ch_ts), `TS`, `TSSetIFunction()`, `DMTSSetIFunction()`, `TSIJacobianFn`, `TSRHSFunctionFn`, `TSRHSJacobianFn`
S*/
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode TSIFunctionFn(TS ts, PetscReal t, Vec U, Vec U_t, Vec F, void *ctx);

PETSC_EXTERN_TYPEDEF typedef TSIFunctionFn *TSIFunction;

/*S
  TSIJacobianFn - A prototype of a `TS` Jacobian evaluation function that would be passed to `TSSetIJacobian()`

  Calling Sequence:
+ ts   - the `TS` context obtained from `TSCreate()`
. t    - time at step/stage being solved
. U    - state vector
. U_t  - time derivative of state vector
. a    - shift
. Amat - (approximate) Jacobian of F(t,U,W+a*U), equivalent to dF/dU + a*dF/dU_t
. Pmat - matrix used for constructing preconditioner, usually the same as `Amat`
- ctx  - [optional] user-defined context for Jacobian evaluation routine

  Level: beginner

  Note:
  The deprecated `TSIJacobian` still works as a replacement for `TSIJacobianFn` *.

.seealso: [](ch_ts), `TSSetIJacobian()`, `DMTSSetIJacobian()`, `TSIFunctionFn`, `TSRHSFunctionFn`, `TSRHSJacobianFn`
S*/
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode TSIJacobianFn(TS ts, PetscReal t, Vec U, Vec U_t, PetscReal a, Mat Amat, Mat Pmat, void *ctx);

PETSC_EXTERN_TYPEDEF typedef TSIJacobianFn *TSIJacobian;

PETSC_EXTERN PetscErrorCode TSSetIFunction(TS, Vec, TSIFunctionFn *, void *);
PETSC_EXTERN PetscErrorCode TSGetIFunction(TS, Vec *, TSIFunctionFn **, void **);
PETSC_EXTERN PetscErrorCode TSSetIJacobian(TS, Mat, Mat, TSIJacobianFn *, void *);
PETSC_EXTERN PetscErrorCode TSGetIJacobian(TS, Mat *, Mat *, TSIJacobianFn **, void **);

/*S
  TSI2FunctionFn - A prototype of a `TS` implicit function evaluation function for 2nd order systems that would be passed to `TSSetI2Function()`

  Calling Sequence:
+ ts   - the `TS` context obtained from `TSCreate()`
. t    - time at step/stage being solved
. U    - state vector
. U_t  - time derivative of state vector
. U_tt - second time derivative of state vector
. F    - function vector
- ctx  - [optional] user-defined context for matrix evaluation routine (may be `NULL`)

  Level: advanced

  Note:
  The deprecated `TSI2Function` still works as a replacement for `TSI2FunctionFn` *.

.seealso: [](ch_ts), `TS`, `TSSetI2Function()`, `DMTSSetI2Function()`, `TSIFunctionFn`
S*/
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode TSI2FunctionFn(TS ts, PetscReal t, Vec U, Vec U_t, Vec U_tt, Vec F, void *ctx);

PETSC_EXTERN_TYPEDEF typedef TSI2FunctionFn *TSI2Function;

/*S
  TSI2JacobianFn - A prototype of a `TS` implicit Jacobian evaluation function for 2nd order systems that would be passed to `TSSetI2Jacobian()`

  Calling Sequence:
+ ts   - the `TS` context obtained from `TSCreate()`
. t    - time at step/stage being solved
. U    - state vector
. U_t  - time derivative of state vector
. U_tt - second time derivative of state vector
. v    - shift for U_t
. a    - shift for U_tt
. J    - Jacobian of G(U) = F(t,U,W+v*U,W'+a*U), equivalent to dF/dU + v*dF/dU_t  + a*dF/dU_tt
. jac  - matrix from which to construct the preconditioner, may be same as `J`
- ctx  - [optional] user-defined context for matrix evaluation routine

  Level: advanced

  Note:
  The deprecated `TSI2Jacobian` still works as a replacement for `TSI2JacobianFn` *.

.seealso: [](ch_ts), `TS`, `TSSetI2Jacobian()`, `DMTSSetI2Jacobian()`, `TSIFunctionFn`, `TSIJacobianFn`, `TSRHSFunctionFn`, `TSRHSJacobianFn`
S*/
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode TSI2JacobianFn(TS ts, PetscReal t, Vec U, Vec U_t, Vec U_tt, PetscReal v, PetscReal a, Mat J, Mat Jac, void *ctx);

PETSC_EXTERN_TYPEDEF typedef TSI2JacobianFn *TSI2Jacobian;

PETSC_EXTERN PetscErrorCode TSSetI2Function(TS, Vec, TSI2FunctionFn *, void *);
PETSC_EXTERN PetscErrorCode TSGetI2Function(TS, Vec *, TSI2FunctionFn **, void **);
PETSC_EXTERN PetscErrorCode TSSetI2Jacobian(TS, Mat, Mat, TSI2JacobianFn *, void *);
PETSC_EXTERN PetscErrorCode TSGetI2Jacobian(TS, Mat *, Mat *, TSI2JacobianFn **, void **);

PETSC_EXTERN PetscErrorCode TSRHSSplitSetIS(TS, const char[], IS);
PETSC_EXTERN PetscErrorCode TSRHSSplitGetIS(TS, const char[], IS *);
PETSC_EXTERN PetscErrorCode TSRHSSplitSetRHSFunction(TS, const char[], Vec, TSRHSFunctionFn *, void *);
PETSC_EXTERN PetscErrorCode TSRHSSplitSetIFunction(TS, const char[], Vec, TSIFunctionFn *, void *);
PETSC_EXTERN PetscErrorCode TSRHSSplitSetIJacobian(TS, const char[], Mat, Mat, TSIJacobianFn *, void *);
PETSC_EXTERN PetscErrorCode TSRHSSplitGetSubTS(TS, const char[], TS *);
PETSC_EXTERN PetscErrorCode TSRHSSplitGetSubTSs(TS, PetscInt *, TS *[]);
PETSC_EXTERN PetscErrorCode TSSetUseSplitRHSFunction(TS, PetscBool);
PETSC_EXTERN PetscErrorCode TSGetUseSplitRHSFunction(TS, PetscBool *);
PETSC_EXTERN PetscErrorCode TSRHSSplitGetSNES(TS, SNES *);
PETSC_EXTERN PetscErrorCode TSRHSSplitSetSNES(TS, SNES);

PETSC_EXTERN TSRHSFunctionFn TSComputeRHSFunctionLinear;
PETSC_EXTERN TSRHSJacobianFn TSComputeRHSJacobianConstant;
PETSC_EXTERN PetscErrorCode  TSComputeIFunctionLinear(TS, PetscReal, Vec, Vec, Vec, void *);
PETSC_EXTERN PetscErrorCode  TSComputeIJacobianConstant(TS, PetscReal, Vec, Vec, PetscReal, Mat, Mat, void *);
PETSC_EXTERN PetscErrorCode  TSComputeSolutionFunction(TS, PetscReal, Vec);
PETSC_EXTERN PetscErrorCode  TSComputeForcingFunction(TS, PetscReal, Vec);
PETSC_EXTERN PetscErrorCode  TSComputeIJacobianDefaultColor(TS, PetscReal, Vec, Vec, PetscReal, Mat, Mat, void *);
PETSC_EXTERN PetscErrorCode  TSPruneIJacobianColor(TS, Mat, Mat);

PETSC_EXTERN PetscErrorCode TSSetPreStep(TS, PetscErrorCode (*)(TS));
PETSC_EXTERN PetscErrorCode TSSetPreStage(TS, PetscErrorCode (*)(TS, PetscReal));
PETSC_EXTERN PetscErrorCode TSSetPostStage(TS, PetscErrorCode (*)(TS, PetscReal, PetscInt, Vec *));
PETSC_EXTERN PetscErrorCode TSSetPostEvaluate(TS, PetscErrorCode (*)(TS));
PETSC_EXTERN PetscErrorCode TSSetPostStep(TS, PetscErrorCode (*)(TS));
PETSC_EXTERN PetscErrorCode TSSetResize(TS, PetscBool, PetscErrorCode (*)(TS, PetscInt, PetscReal, Vec, PetscBool *, void *), PetscErrorCode (*)(TS, PetscInt, Vec[], Vec[], void *), void *);
PETSC_EXTERN PetscErrorCode TSPreStep(TS);
PETSC_EXTERN PetscErrorCode TSPreStage(TS, PetscReal);
PETSC_EXTERN PetscErrorCode TSPostStage(TS, PetscReal, PetscInt, Vec[]);
PETSC_EXTERN PetscErrorCode TSPostEvaluate(TS);
PETSC_EXTERN PetscErrorCode TSPostStep(TS);
PETSC_EXTERN PetscErrorCode TSResize(TS);
PETSC_EXTERN PetscErrorCode TSResizeRetrieveVec(TS, const char *, Vec *);
PETSC_EXTERN PetscErrorCode TSResizeRegisterVec(TS, const char *, Vec);
PETSC_EXTERN PetscErrorCode TSGetStepResize(TS, PetscBool *);

PETSC_EXTERN PetscErrorCode TSInterpolate(TS, PetscReal, Vec);
PETSC_EXTERN PetscErrorCode TSSetTolerances(TS, PetscReal, Vec, PetscReal, Vec);
PETSC_EXTERN PetscErrorCode TSGetTolerances(TS, PetscReal *, Vec *, PetscReal *, Vec *);
PETSC_EXTERN PetscErrorCode TSErrorWeightedNorm(TS, Vec, Vec, NormType, PetscReal *, PetscReal *, PetscReal *);
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
PETSC_EXTERN PetscErrorCode DMTSSetRHSFunction(DM, TSRHSFunctionFn *, void *);
PETSC_EXTERN PetscErrorCode DMTSGetRHSFunction(DM, TSRHSFunctionFn **, void **);
PETSC_EXTERN PetscErrorCode DMTSSetRHSFunctionContextDestroy(DM, PetscCtxDestroyFn *);
PETSC_EXTERN PetscErrorCode DMTSSetRHSJacobian(DM, TSRHSJacobianFn *, void *);
PETSC_EXTERN PetscErrorCode DMTSGetRHSJacobian(DM, TSRHSJacobianFn **, void **);
PETSC_EXTERN PetscErrorCode DMTSSetRHSJacobianContextDestroy(DM, PetscCtxDestroyFn *);
PETSC_EXTERN PetscErrorCode DMTSSetIFunction(DM, TSIFunctionFn *, void *);
PETSC_EXTERN PetscErrorCode DMTSGetIFunction(DM, TSIFunctionFn **, void **);
PETSC_EXTERN PetscErrorCode DMTSSetIFunctionContextDestroy(DM, PetscCtxDestroyFn *);
PETSC_EXTERN PetscErrorCode DMTSSetIJacobian(DM, TSIJacobianFn *, void *);
PETSC_EXTERN PetscErrorCode DMTSGetIJacobian(DM, TSIJacobianFn **, void **);
PETSC_EXTERN PetscErrorCode DMTSSetIJacobianContextDestroy(DM, PetscCtxDestroyFn *);
PETSC_EXTERN PetscErrorCode DMTSSetI2Function(DM, TSI2FunctionFn *, void *);
PETSC_EXTERN PetscErrorCode DMTSGetI2Function(DM, TSI2FunctionFn **, void **);
PETSC_EXTERN PetscErrorCode DMTSSetI2FunctionContextDestroy(DM, PetscCtxDestroyFn *);
PETSC_EXTERN PetscErrorCode DMTSSetI2Jacobian(DM, TSI2JacobianFn *, void *);
PETSC_EXTERN PetscErrorCode DMTSGetI2Jacobian(DM, TSI2JacobianFn **, void **);
PETSC_EXTERN PetscErrorCode DMTSSetI2JacobianContextDestroy(DM, PetscCtxDestroyFn *);

/*S
  TSTransientVariableFn - A prototype of a function to transform from state to transient variables that would be passed to `TSSetTransientVariable()`

  Calling Sequence:
+ ts  - timestep context
. p   - input vector (primitive form)
. c   - output vector, transient variables (conservative form)
- ctx - [optional] user-defined function context

  Level: advanced

  Note:
  The deprecated `TSTransientVariable` still works as a replacement for `TSTransientVariableFn` *.

.seealso: [](ch_ts), `TS`, `TSSetTransientVariable()`, `DMTSSetTransientVariable()`
S*/
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode TSTransientVariableFn(TS ts, Vec p, Vec c, void *ctx);

PETSC_EXTERN_TYPEDEF typedef TSTransientVariableFn *TSTransientVariable;

PETSC_EXTERN PetscErrorCode TSSetTransientVariable(TS, TSTransientVariableFn *, void *);
PETSC_EXTERN PetscErrorCode DMTSSetTransientVariable(DM, TSTransientVariableFn *, void *);
PETSC_EXTERN PetscErrorCode DMTSGetTransientVariable(DM, TSTransientVariableFn **, void *);
PETSC_EXTERN PetscErrorCode TSComputeTransientVariable(TS, Vec, Vec);
PETSC_EXTERN PetscErrorCode TSHasTransientVariable(TS, PetscBool *);

PETSC_EXTERN PetscErrorCode DMTSSetSolutionFunction(DM, TSSolutionFn *, void *);
PETSC_EXTERN PetscErrorCode DMTSGetSolutionFunction(DM, TSSolutionFn **, void **);
PETSC_EXTERN PetscErrorCode DMTSSetForcingFunction(DM, TSForcingFn *, void *);
PETSC_EXTERN PetscErrorCode DMTSGetForcingFunction(DM, TSForcingFn **, void **);
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

/*S
  DMDATSRHSFunctionLocalFn - A prototype of a local `TS` right-hand side residual evaluation function for use with `DMDA` that would be passed to `DMDATSSetRHSFunctionLocal()`

  Calling Sequence:
+ info - defines the subdomain to evaluate the residual on
. t    - time at which to evaluate residual
. x    - array of local state information
. f    - output array of local residual information
- ctx  - optional user context

  Level: beginner

  Note:
  The deprecated `DMDATSRHSFunctionLocal` still works as a replacement for `DMDATSRHSFunctionLocalFn` *.

.seealso: `DMDA`, `DMDATSSetRHSFunctionLocal()`, `TSRHSFunctionFn`, `DMDATSRHSJacobianLocalFn`, `DMDATSIJacobianLocalFn`, `DMDATSIFunctionLocalFn`
S*/
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode DMDATSRHSFunctionLocalFn(DMDALocalInfo *info, PetscReal t, void *x, void *f, void *ctx);

PETSC_EXTERN_TYPEDEF typedef DMDATSRHSFunctionLocalFn *DMDATSRHSFunctionLocal;

/*S
  DMDATSRHSJacobianLocalFn - A prototype of a local residual evaluation function for use with `DMDA` that would be passed to `DMDATSSetRHSJacobianLocal()`

  Calling Sequence:
+ info - defines the subdomain to evaluate the residual on
. t    - time at which to evaluate residual
. x    - array of local state information
. J    - Jacobian matrix
. B    - matrix from which to construct the preconditioner; often same as `J`
- ctx  - optional context

  Level: beginner

  Note:
  The deprecated `DMDATSRHSJacobianLocal` still works as a replacement for `DMDATSRHSJacobianLocalFn` *.

.seealso: `DMDA`, `DMDATSSetRHSJacobianLocal()`, `TSRHSJacobianFn`, `DMDATSRHSFunctionLocalFn`, `DMDATSIJacobianLocalFn`, `DMDATSIFunctionLocalFn`
S*/
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode DMDATSRHSJacobianLocalFn(DMDALocalInfo *info, PetscReal t, void *x, Mat J, Mat B, void *ctx);

PETSC_EXTERN_TYPEDEF typedef DMDATSRHSJacobianLocalFn *DMDATSRHSJacobianLocal;

/*S
  DMDATSIFunctionLocalFn - A prototype of a local residual evaluation function for use with `DMDA` that would be passed to `DMDATSSetIFunctionLocal()`

  Calling Sequence:
+ info  - defines the subdomain to evaluate the residual on
. t     - time at which to evaluate residual
. x     - array of local state information
. xdot  - array of local time derivative information
. imode - output array of local function evaluation information
- ctx   - optional context

  Level: beginner

  Note:
  The deprecated `DMDATSIFunctionLocal` still works as a replacement for `DMDATSIFunctionLocalFn` *.

.seealso: `DMDA`, `DMDATSSetIFunctionLocal()`, `DMDATSIJacobianLocalFn`, `TSIFunctionFn`
S*/
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode DMDATSIFunctionLocalFn(DMDALocalInfo *info, PetscReal t, void *x, void *xdot, void *imode, void *ctx);

PETSC_EXTERN_TYPEDEF typedef DMDATSIFunctionLocalFn *DMDATSIFunctionLocal;

/*S
  DMDATSIJacobianLocalFn - A prototype of a local residual evaluation function for use with `DMDA` that would be passed to `DMDATSSetIJacobianLocal()`

  Calling Sequence:
+ info  - defines the subdomain to evaluate the residual on
. t     - time at which to evaluate the jacobian
. x     - array of local state information
. xdot  - time derivative at this state
. shift - see `TSSetIJacobian()` for the meaning of this parameter
. J     - Jacobian matrix
. B     - matrix from which to construct the preconditioner; often same as `J`
- ctx   - optional context

  Level: beginner

  Note:
  The deprecated `DMDATSIJacobianLocal` still works as a replacement for `DMDATSIJacobianLocalFn` *.

.seealso: `DMDA` `DMDATSSetIJacobianLocal()`, `TSIJacobianFn`, `DMDATSIFunctionLocalFn`, `DMDATSRHSFunctionLocalFn`,  `DMDATSRHSJacobianlocal()`
S*/
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode DMDATSIJacobianLocalFn(DMDALocalInfo *info, PetscReal t, void *x, void *xdot, PetscReal shift, Mat J, Mat B, void *ctx);

PETSC_EXTERN_TYPEDEF typedef DMDATSIJacobianLocalFn *DMDATSIJacobianLocal;

PETSC_EXTERN PetscErrorCode DMDATSSetRHSFunctionLocal(DM, InsertMode, DMDATSRHSFunctionLocalFn *, void *);
PETSC_EXTERN PetscErrorCode DMDATSSetRHSJacobianLocal(DM, DMDATSRHSJacobianLocalFn *, void *);
PETSC_EXTERN PetscErrorCode DMDATSSetIFunctionLocal(DM, InsertMode, DMDATSIFunctionLocalFn *, void *);
PETSC_EXTERN PetscErrorCode DMDATSSetIJacobianLocal(DM, DMDATSIJacobianLocalFn *, void *);

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
PETSC_EXTERN PetscErrorCode TSMonitorLGSetTransform(TS, PetscErrorCode (*)(void *, Vec, Vec *), PetscCtxDestroyFn *, void *);
PETSC_EXTERN PetscErrorCode TSMonitorLGCtxSetTransform(TSMonitorLGCtx, PetscErrorCode (*)(void *, Vec, Vec *), PetscCtxDestroyFn *, void *);
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

PETSC_EXTERN PetscErrorCode TSSetEventHandler(TS, PetscInt, PetscInt[], PetscBool[], PetscErrorCode (*)(TS, PetscReal, Vec, PetscReal[], void *), PetscErrorCode (*)(TS, PetscInt, PetscInt[], PetscReal, Vec, PetscBool, void *), void *);
PETSC_EXTERN PetscErrorCode TSSetPostEventStep(TS, PetscReal);
PETSC_EXTERN PetscErrorCode TSSetPostEventSecondStep(TS, PetscReal);
PETSC_DEPRECATED_FUNCTION(3, 21, 0, "TSSetPostEventSecondStep()", ) static inline PetscErrorCode TSSetPostEventIntervalStep(TS ts, PetscReal dt)
{
  return TSSetPostEventSecondStep(ts, dt);
}
PETSC_EXTERN PetscErrorCode TSSetEventTolerances(TS, PetscReal, PetscReal[]);
PETSC_EXTERN PetscErrorCode TSGetNumEvents(TS, PetscInt *);

/*J
   TSSSPType - string with the name of a `TSSSP` scheme.

   Level: beginner

.seealso: [](ch_ts), `TSSSPSetType()`, `TS`, `TSSSP`
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

.seealso: [](ch_ts), [](sec_ts_error_control), `TS`, `TSGetAdapt()`, `TSAdaptCreate()`, `TSAdaptType`
S*/
typedef struct _p_TSAdapt *TSAdapt;

/*J
   TSAdaptType - String with the name of `TSAdapt` scheme.

   Level: beginner

.seealso: [](ch_ts), [](sec_ts_error_control), `TSGetAdapt()`, `TSAdaptSetType()`, `TS`, `TSAdapt`
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
PETSC_EXTERN PetscErrorCode TSAdaptSetFromOptions(TSAdapt, PetscOptionItems);
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
PETSC_EXTERN PetscErrorCode TSAdaptHistorySetHistory(TSAdapt, PetscInt, PetscReal[], PetscBool);
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

.seealso: [](ch_ts), `TS`, `TSGLLE`, `TSGLLEAdaptCreate()`, `TSGLLEAdaptType`
S*/
typedef struct _p_TSGLLEAdapt *TSGLLEAdapt;

/*J
   TSGLLEAdaptType - String with the name of `TSGLLEAdapt` scheme

   Level: beginner

   Developer Note:
   This functionality should be replaced by the `TSAdaptType`.

.seealso: [](ch_ts), `TSGLLEAdaptSetType()`, `TS`
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
PETSC_EXTERN PetscErrorCode TSGLLEAdaptSetFromOptions(TSGLLEAdapt, PetscOptionItems);
PETSC_EXTERN PetscErrorCode TSGLLEAdaptDestroy(TSGLLEAdapt *);

/*J
   TSGLLEAcceptType - String with the name of `TSGLLEAccept` scheme

   Level: beginner

.seealso: [](ch_ts), `TSGLLESetAcceptType()`, `TS`, `TSGLLEAccept`
J*/
typedef const char *TSGLLEAcceptType;
#define TSGLLEACCEPT_ALWAYS "always"

/*S
  TSGLLEAcceptFn - A prototype of a `TS` accept function that would be passed to `TSGLLEAcceptRegister()`

  Calling Sequence:
+ ts  - timestep context
. nt - time to end of solution time
. h - the proposed step-size
. enorm - unknown
- accept - output, if the proposal is accepted

  Level: beginner

  Note:
  The deprecated `TSGLLEAcceptFunction` still works as a replacement for `TSGLLEAcceptFn` *

.seealso: [](ch_ts), `TS`, `TSSetRHSFunction()`, `DMTSSetRHSFunction()`, `TSIFunctionFn`,
`TSIJacobianFn`, `TSRHSJacobianFn`, `TSGLLEAcceptRegister()`
S*/
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode(TSGLLEAcceptFn)(TS ts, PetscReal nt, PetscReal h, const PetscReal enorm[], PetscBool *accept);

PETSC_EXTERN_TYPEDEF typedef TSGLLEAcceptFn *TSGLLEAcceptFunction;

PETSC_EXTERN PetscErrorCode TSGLLEAcceptRegister(const char[], TSGLLEAcceptFn *);

/*J
  TSGLLEType - string with the name of a General Linear `TSGLLE` type

  Level: beginner

.seealso: [](ch_ts), `TS`, `TSGLLE`, `TSGLLESetType()`, `TSGLLERegister()`, `TSGLLEAccept`
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

.seealso: [](ch_ts), `TSEIMEXSetType()`, `TS`, `TSEIMEX`, `TSEIMEXRegister()`
J*/
#define TSEIMEXType char *

PETSC_EXTERN PetscErrorCode TSEIMEXSetMaxRows(TS, PetscInt);
PETSC_EXTERN PetscErrorCode TSEIMEXSetRowCol(TS, PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode TSEIMEXSetOrdAdapt(TS, PetscBool);

/*J
   TSRKType - String with the name of a Runge-Kutta `TSRK` type

   Level: beginner

.seealso: [](ch_ts), `TS`, `TSRKSetType()`, `TS`, `TSRK`, `TSRKRegister()`
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
   TSMPRKType - String with the name of a partitioned Runge-Kutta `TSMPRK` type

   Level: beginner

.seealso: [](ch_ts), `TSMPRKSetType()`, `TS`, `TSMPRK`, `TSMPRKRegister()`
J*/
typedef const char *TSMPRKType;
#define TSMPRK2A22 "2a22"
#define TSMPRK2A23 "2a23"
#define TSMPRK2A32 "2a32"
#define TSMPRK2A33 "2a33"
#define TSMPRKP2   "p2"
#define TSMPRKP3   "p3"

PETSC_EXTERN PetscErrorCode TSMPRKGetType(TS, TSMPRKType *);
PETSC_EXTERN PetscErrorCode TSMPRKSetType(TS, TSMPRKType);
PETSC_EXTERN PetscErrorCode TSMPRKRegister(TSMPRKType, PetscInt, PetscInt, PetscInt, PetscInt, const PetscReal[], const PetscReal[], const PetscReal[], const PetscInt[], const PetscReal[], const PetscReal[], const PetscReal[], const PetscInt[], const PetscReal[], const PetscReal[], const PetscReal[]);
PETSC_EXTERN PetscErrorCode TSMPRKInitializePackage(void);
PETSC_EXTERN PetscErrorCode TSMPRKFinalizePackage(void);
PETSC_EXTERN PetscErrorCode TSMPRKRegisterDestroy(void);

/*J
   TSIRKType - String with the name of an implicit Runge-Kutta `TSIRK` type

   Level: beginner

.seealso: [](ch_ts), `TSIRKSetType()`, `TS`, `TSIRK`, `TSIRKRegister()`
J*/
typedef const char *TSIRKType;
#define TSIRKGAUSS "gauss"

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

.seealso: [](ch_ts), `TSGLEESetType()`, `TS`, `TSGLEE`, `TSGLEERegister()`
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

.seealso: [](ch_ts), `TSGLEESetMode()`, `TS`, `TSGLEE`, `TSGLEERegister()`
J*/
PETSC_EXTERN PetscErrorCode TSGLEEGetType(TS, TSGLEEType *);
PETSC_EXTERN PetscErrorCode TSGLEESetType(TS, TSGLEEType);
PETSC_EXTERN PetscErrorCode TSGLEERegister(TSGLEEType, PetscInt, PetscInt, PetscInt, PetscReal, const PetscReal[], const PetscReal[], const PetscReal[], const PetscReal[], const PetscReal[], const PetscReal[], const PetscReal[], const PetscReal[], const PetscReal[], const PetscReal[], PetscInt, const PetscReal[]);
PETSC_EXTERN PetscErrorCode TSGLEERegisterAll(void);
PETSC_EXTERN PetscErrorCode TSGLEEFinalizePackage(void);
PETSC_EXTERN PetscErrorCode TSGLEEInitializePackage(void);
PETSC_EXTERN PetscErrorCode TSGLEERegisterDestroy(void);

/*J
   TSARKIMEXType - String with the name of an Additive Runge-Kutta IMEX `TSARKIMEX` type

   Level: beginner

.seealso: [](ch_ts), `TSARKIMEXSetType()`, `TS`, `TSARKIMEX`, `TSARKIMEXRegister()`
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

PETSC_EXTERN PetscErrorCode TSARKIMEXGetType(TS, TSARKIMEXType *);
PETSC_EXTERN PetscErrorCode TSARKIMEXSetType(TS, TSARKIMEXType);
PETSC_EXTERN PetscErrorCode TSARKIMEXSetFullyImplicit(TS, PetscBool);
PETSC_EXTERN PetscErrorCode TSARKIMEXGetFullyImplicit(TS, PetscBool *);
PETSC_EXTERN PetscErrorCode TSARKIMEXSetFastSlowSplit(TS, PetscBool);
PETSC_EXTERN PetscErrorCode TSARKIMEXGetFastSlowSplit(TS, PetscBool *);
PETSC_EXTERN PetscErrorCode TSARKIMEXRegister(TSARKIMEXType, PetscInt, PetscInt, const PetscReal[], const PetscReal[], const PetscReal[], const PetscReal[], const PetscReal[], const PetscReal[], const PetscReal[], const PetscReal[], PetscInt, const PetscReal[], const PetscReal[]);
PETSC_EXTERN PetscErrorCode TSARKIMEXInitializePackage(void);
PETSC_EXTERN PetscErrorCode TSARKIMEXFinalizePackage(void);
PETSC_EXTERN PetscErrorCode TSARKIMEXRegisterDestroy(void);

/*J
   TSDIRKType - String with the name of a Diagonally Implicit Runge-Kutta `TSDIRK` type

   Level: beginner

.seealso: [](ch_ts), `TSDIRKSetType()`, `TS`, `TSDIRK`, `TSDIRKRegister()`
J*/
typedef const char *TSDIRKType;
#define TSDIRKS212      "s212"
#define TSDIRKES122SAL  "es122sal"
#define TSDIRKES213SAL  "es213sal"
#define TSDIRKES324SAL  "es324sal"
#define TSDIRKES325SAL  "es325sal"
#define TSDIRK657A      "657a"
#define TSDIRKES648SA   "es648sa"
#define TSDIRK658A      "658a"
#define TSDIRKS659A     "s659a"
#define TSDIRK7510SAL   "7510sal"
#define TSDIRKES7510SA  "es7510sa"
#define TSDIRK759A      "759a"
#define TSDIRKS7511SAL  "s7511sal"
#define TSDIRK8614A     "8614a"
#define TSDIRK8616SAL   "8616sal"
#define TSDIRKES8516SAL "es8516sal"

PETSC_EXTERN PetscErrorCode TSDIRKGetType(TS, TSDIRKType *);
PETSC_EXTERN PetscErrorCode TSDIRKSetType(TS, TSDIRKType);
PETSC_EXTERN PetscErrorCode TSDIRKRegister(TSDIRKType, PetscInt, PetscInt, const PetscReal[], const PetscReal[], const PetscReal[], const PetscReal[], PetscInt, const PetscReal[]);

/*J
   TSRosWType - String with the name of a Rosenbrock-W `TSROSW` type

   Level: beginner

.seealso: [](ch_ts), `TSRosWSetType()`, `TS`, `TSROSW`, `TSRosWRegister()`
J*/
typedef const char *TSRosWType;
#define TSROSW2M          "2m"
#define TSROSW2P          "2p"
#define TSROSWRA3PW       "ra3pw"
#define TSROSWRA34PW2     "ra34pw2"
#define TSROSWR34PRW      "r34prw"
#define TSROSWR3PRL2      "r3prl2"
#define TSROSWRODAS3      "rodas3"
#define TSROSWRODASPR     "rodaspr"
#define TSROSWRODASPR2    "rodaspr2"
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

.seealso: [](ch_ts), `TSBasicSymplecticSetType()`, `TS`, `TSBASICSYMPLECTIC`, `TSBasicSymplecticRegister()`
J*/
typedef const char *TSBasicSymplecticType;
#define TSBASICSYMPLECTICSIEULER   "1"
#define TSBASICSYMPLECTICVELVERLET "2"
#define TSBASICSYMPLECTIC3         "3"
#define TSBASICSYMPLECTIC4         "4"

PETSC_EXTERN PetscErrorCode TSBasicSymplecticSetType(TS, TSBasicSymplecticType);
PETSC_EXTERN PetscErrorCode TSBasicSymplecticGetType(TS, TSBasicSymplecticType *);
PETSC_EXTERN PetscErrorCode TSBasicSymplecticRegister(TSBasicSymplecticType, PetscInt, PetscInt, PetscReal[], PetscReal[]);
PETSC_EXTERN PetscErrorCode TSBasicSymplecticRegisterAll(void);
PETSC_EXTERN PetscErrorCode TSBasicSymplecticInitializePackage(void);
PETSC_EXTERN PetscErrorCode TSBasicSymplecticFinalizePackage(void);
PETSC_EXTERN PetscErrorCode TSBasicSymplecticRegisterDestroy(void);

/*J
  TSDISCGRAD - The Discrete Gradient integrator is a timestepper for Hamiltonian systems designed to conserve the first integral (energy),
  but also has the property for some systems of monotonicity in a functional.

  Level: beginner

.seealso: [](ch_ts), `TS`, TSDiscGradSetFormulation()`, `TSDiscGradGetFormulation()`, `TSDiscGradSetType()`, `TSDiscGradGetType()`
J*/
typedef enum {
  TS_DG_GONZALEZ,
  TS_DG_AVERAGE,
  TS_DG_NONE
} TSDGType;
PETSC_EXTERN PetscErrorCode TSDiscGradSetFormulation(TS, PetscErrorCode (*)(TS, PetscReal, Vec, Mat, void *), PetscErrorCode (*)(TS, PetscReal, Vec, PetscScalar *, void *), PetscErrorCode (*)(TS, PetscReal, Vec, Vec, void *), void *);
PETSC_EXTERN PetscErrorCode TSDiscGradGetFormulation(TS, PetscErrorCode (**)(TS, PetscReal, Vec, Mat, void *), PetscErrorCode (**)(TS, PetscReal, Vec, PetscScalar *, void *), PetscErrorCode (**)(TS, PetscReal, Vec, Vec, void *), void *);
PETSC_EXTERN PetscErrorCode TSDiscGradSetType(TS, TSDGType);
PETSC_EXTERN PetscErrorCode TSDiscGradGetType(TS, TSDGType *);

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

PETSC_EXTERN PetscErrorCode TSSundialsSetType(TS, TSSundialsLmmType);
PETSC_EXTERN PetscErrorCode TSSundialsGetPC(TS, PC *);
PETSC_EXTERN PetscErrorCode TSSundialsSetTolerance(TS, PetscReal, PetscReal);
PETSC_EXTERN PetscErrorCode TSSundialsSetMinTimeStep(TS, PetscReal);
PETSC_EXTERN PetscErrorCode TSSundialsSetMaxTimeStep(TS, PetscReal);
PETSC_EXTERN PetscErrorCode TSSundialsGetIterations(TS, PetscInt *, PetscInt *);
PETSC_EXTERN PetscErrorCode TSSundialsSetGramSchmidtType(TS, TSSundialsGramSchmidtType);
PETSC_EXTERN PetscErrorCode TSSundialsSetGMRESRestart(TS, PetscInt);
PETSC_EXTERN PetscErrorCode TSSundialsSetLinearTolerance(TS, PetscReal);
PETSC_EXTERN PetscErrorCode TSSundialsMonitorInternalSteps(TS, PetscBool);
PETSC_EXTERN PetscErrorCode TSSundialsSetMaxl(TS, PetscInt);
PETSC_EXTERN PetscErrorCode TSSundialsSetMaxord(TS, PetscInt);
PETSC_EXTERN PetscErrorCode TSSundialsSetUseDense(TS, PetscBool);
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

/*S
  TSAlpha2PredictorFn - A callback to set the predictor (i.e., the initial guess for the nonlinear solver) in
  a second-order generalized-alpha time integrator.

  Calling Sequence:
+ ts   - the `TS` context obtained from `TSCreate()`
. X0   - the previous time step's state vector
. V0   - the previous time step's first derivative of the state vector
. A0   - the previous time step's second derivative of the state vector
. X1   - the vector into which the initial guess for the current time step will be written
- ctx  - [optional] user-defined context for the predictor evaluation routine (may be `NULL`)

  Level: intermediate

  Note:
  The deprecated `TSAlpha2Predictor` still works as a replacement for `TSAlpha2PredictorFn` *.

.seealso: [](ch_ts), `TS`, `TSAlpha2SetPredictor()`
S*/
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode TSAlpha2PredictorFn(TS ts, Vec X0, Vec V0, Vec A0, Vec X1, void *ctx);

PETSC_EXTERN_TYPEDEF typedef TSAlpha2PredictorFn *TSAlpha2Predictor;

PETSC_EXTERN PetscErrorCode TSAlpha2SetPredictor(TS, TSAlpha2PredictorFn *, void *);

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
