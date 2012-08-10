/*
   User interface for the timestepping package. This package
   is for use in solving time-dependent PDEs.
*/
#if !defined(__PETSCTS_H)
#define __PETSCTS_H
#include "petscsnes.h"

/*S
     TS - Abstract PETSc object that manages all time-steppers (ODE integrators)

   Level: beginner

  Concepts: ODE solvers

.seealso:  TSCreate(), TSSetType(), TSType, SNES, KSP, PC
S*/
typedef struct _p_TS* TS;

/*J
    TSType - String with the name of a PETSc TS method or the creation function
       with an optional dynamic library name, for example
       http://www.mcs.anl.gov/petsc/lib.a:mytscreate()

   Level: beginner

.seealso: TSSetType(), TS
J*/
#define TSType char*
#define TSEULER           "euler"
#define TSBEULER          "beuler"
#define TSPSEUDO          "pseudo"
#define TSCN              "cn"
#define TSSUNDIALS        "sundials"
#define TSRK              "rk"
#define TSPYTHON          "python"
#define TSTHETA           "theta"
#define TSALPHA           "alpha"
#define TSGL              "gl"
#define TSSSP             "ssp"
#define TSARKIMEX         "arkimex"
#define TSROSW            "rosw"

/*E
    TSProblemType - Determines the type of problem this TS object is to be used to solve

   Level: beginner

.seealso: TSCreate()
E*/
typedef enum {TS_LINEAR,TS_NONLINEAR} TSProblemType;

/*E
   TSConvergedReason - reason a TS method has converged or not

   Level: beginner

   Developer Notes: this must match finclude/petscts.h

   Each reason has its own manual page.

.seealso: TSGetConvergedReason()
E*/
typedef enum {
  TS_CONVERGED_ITERATING      = 0,
  TS_CONVERGED_TIME           = 1,
  TS_CONVERGED_ITS            = 2,
  TS_DIVERGED_NONLINEAR_SOLVE = -1,
  TS_DIVERGED_STEP_REJECTED   = -2
} TSConvergedReason;
PETSC_EXTERN const char *const*TSConvergedReasons;

/*MC
   TS_CONVERGED_ITERATING - this only occurs if TSGetConvergedReason() is called during the TSSolve()

   Level: beginner

.seealso: TSSolve(), TSConvergedReason(), TSGetAdapt()
M*/

/*MC
   TS_CONVERGED_TIME - the final time was reached

   Level: beginner

.seealso: TSSolve(), TSConvergedReason(), TSGetAdapt(), TSSetDuration()
M*/

/*MC
   TS_CONVERGED_ITS - the maximum number of iterations was reached prior to the final time

   Level: beginner

.seealso: TSSolve(), TSConvergedReason(), TSGetAdapt(), TSSetDuration()
M*/

/*MC
   TS_DIVERGED_NONLINEAR_SOLVE - too many nonlinear solves failed

   Level: beginner

.seealso: TSSolve(), TSConvergedReason(), TSGetAdapt(), TSGetSNES(), SNESGetConvergedReason()
M*/

/*MC
   TS_DIVERGED_STEP_REJECTED - too many steps were rejected

   Level: beginner

.seealso: TSSolve(), TSConvergedReason(), TSGetAdapt()
M*/

/* Logging support */
PETSC_EXTERN PetscClassId TS_CLASSID;

PETSC_EXTERN PetscErrorCode TSInitializePackage(const char[]);

PETSC_EXTERN PetscErrorCode TSCreate(MPI_Comm,TS*);
PETSC_EXTERN PetscErrorCode TSDestroy(TS*);

PETSC_EXTERN PetscErrorCode TSSetProblemType(TS,TSProblemType);
PETSC_EXTERN PetscErrorCode TSGetProblemType(TS,TSProblemType*);
PETSC_EXTERN PetscErrorCode TSMonitor(TS,PetscInt,PetscReal,Vec);
PETSC_EXTERN PetscErrorCode TSMonitorSet(TS,PetscErrorCode(*)(TS,PetscInt,PetscReal,Vec,void*),void *,PetscErrorCode (*)(void**));
PETSC_EXTERN PetscErrorCode TSMonitorCancel(TS);

PETSC_EXTERN PetscErrorCode TSSetOptionsPrefix(TS,const char[]);
PETSC_EXTERN PetscErrorCode TSAppendOptionsPrefix(TS,const char[]);
PETSC_EXTERN PetscErrorCode TSGetOptionsPrefix(TS,const char *[]);
PETSC_EXTERN PetscErrorCode TSSetFromOptions(TS);
PETSC_EXTERN PetscErrorCode TSSetUp(TS);
PETSC_EXTERN PetscErrorCode TSReset(TS);

PETSC_EXTERN PetscErrorCode TSSetSolution(TS,Vec);
PETSC_EXTERN PetscErrorCode TSGetSolution(TS,Vec*);

PETSC_EXTERN PetscErrorCode TSSetDuration(TS,PetscInt,PetscReal);
PETSC_EXTERN PetscErrorCode TSGetDuration(TS,PetscInt*,PetscReal*);
PETSC_EXTERN PetscErrorCode TSSetExactFinalTime(TS,PetscBool);

PETSC_EXTERN PetscErrorCode TSMonitorDefault(TS,PetscInt,PetscReal,Vec,void*);
PETSC_EXTERN PetscErrorCode TSMonitorSolution(TS,PetscInt,PetscReal,Vec,void*);
PETSC_EXTERN PetscErrorCode TSMonitorSolutionCreate(TS,PetscViewer,void**);
PETSC_EXTERN PetscErrorCode TSMonitorSolutionDestroy(void**);
PETSC_EXTERN PetscErrorCode TSMonitorSolutionBinary(TS,PetscInt,PetscReal,Vec,void*);
PETSC_EXTERN PetscErrorCode TSMonitorSolutionVTK(TS,PetscInt,PetscReal,Vec,void*);
PETSC_EXTERN PetscErrorCode TSMonitorSolutionVTKDestroy(void*);

PETSC_EXTERN PetscErrorCode TSStep(TS);
PETSC_EXTERN PetscErrorCode TSEvaluateStep(TS,PetscInt,Vec,PetscBool*);
PETSC_EXTERN PetscErrorCode TSSolve(TS,Vec,PetscReal*);
PETSC_EXTERN PetscErrorCode TSGetConvergedReason(TS,TSConvergedReason*);
PETSC_EXTERN PetscErrorCode TSGetSNESIterations(TS,PetscInt*);
PETSC_EXTERN PetscErrorCode TSGetKSPIterations(TS,PetscInt*);
PETSC_EXTERN PetscErrorCode TSGetStepRejections(TS,PetscInt*);
PETSC_EXTERN PetscErrorCode TSSetMaxStepRejections(TS,PetscInt);
PETSC_EXTERN PetscErrorCode TSGetSNESFailures(TS,PetscInt*);
PETSC_EXTERN PetscErrorCode TSSetMaxSNESFailures(TS,PetscInt);
PETSC_EXTERN PetscErrorCode TSSetErrorIfStepFails(TS,PetscBool);

PETSC_EXTERN PetscErrorCode TSSetInitialTimeStep(TS,PetscReal,PetscReal);
PETSC_EXTERN PetscErrorCode TSGetTimeStep(TS,PetscReal*);
PETSC_EXTERN PetscErrorCode TSGetTime(TS,PetscReal*);
PETSC_EXTERN PetscErrorCode TSSetTime(TS,PetscReal);
PETSC_EXTERN PetscErrorCode TSGetTimeStepNumber(TS,PetscInt*);
PETSC_EXTERN PetscErrorCode TSSetTimeStep(TS,PetscReal);

PETSC_EXTERN_TYPEDEF typedef PetscErrorCode (*TSRHSFunction)(TS,PetscReal,Vec,Vec,void*);
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode (*TSRHSJacobian)(TS,PetscReal,Vec,Mat*,Mat*,MatStructure*,void*);
PETSC_EXTERN PetscErrorCode TSSetRHSFunction(TS,Vec,TSRHSFunction,void*);
PETSC_EXTERN PetscErrorCode TSGetRHSFunction(TS,Vec*,TSRHSFunction*,void**);
PETSC_EXTERN PetscErrorCode TSSetRHSJacobian(TS,Mat,Mat,TSRHSJacobian,void*);
PETSC_EXTERN PetscErrorCode TSGetRHSJacobian(TS,Mat*,Mat*,TSRHSJacobian*,void**);

PETSC_EXTERN_TYPEDEF typedef PetscErrorCode (*TSIFunction)(TS,PetscReal,Vec,Vec,Vec,void*);
PETSC_EXTERN_TYPEDEF typedef PetscErrorCode (*TSIJacobian)(TS,PetscReal,Vec,Vec,PetscReal,Mat*,Mat*,MatStructure*,void*);
PETSC_EXTERN PetscErrorCode TSSetIFunction(TS,Vec,TSIFunction,void*);
PETSC_EXTERN PetscErrorCode TSGetIFunction(TS,Vec*,TSIFunction*,void**);
PETSC_EXTERN PetscErrorCode TSSetIJacobian(TS,Mat,Mat,TSIJacobian,void*);
PETSC_EXTERN PetscErrorCode TSGetIJacobian(TS,Mat*,Mat*,TSIJacobian*,void**);

PETSC_EXTERN PetscErrorCode TSComputeRHSFunctionLinear(TS,PetscReal,Vec,Vec,void*);
PETSC_EXTERN PetscErrorCode TSComputeRHSJacobianConstant(TS,PetscReal,Vec,Mat*,Mat*,MatStructure*,void*);
PETSC_EXTERN PetscErrorCode TSComputeIFunctionLinear(TS,PetscReal,Vec,Vec,Vec,void*);
PETSC_EXTERN PetscErrorCode TSComputeIJacobianConstant(TS,PetscReal,Vec,Vec,PetscReal,Mat*,Mat*,MatStructure*,void*);

PETSC_EXTERN PetscErrorCode TSSetPreStep(TS, PetscErrorCode (*)(TS));
PETSC_EXTERN PetscErrorCode TSSetPreStage(TS, PetscErrorCode (*)(TS,PetscReal));
PETSC_EXTERN PetscErrorCode TSSetPostStep(TS, PetscErrorCode (*)(TS));
PETSC_EXTERN PetscErrorCode TSPreStep(TS);
PETSC_EXTERN PetscErrorCode TSPreStage(TS,PetscReal);
PETSC_EXTERN PetscErrorCode TSPostStep(TS);
PETSC_EXTERN PetscErrorCode TSSetRetainStages(TS,PetscBool);
PETSC_EXTERN PetscErrorCode TSInterpolate(TS,PetscReal,Vec);
PETSC_EXTERN PetscErrorCode TSSetTolerances(TS,PetscReal,Vec,PetscReal,Vec);
PETSC_EXTERN PetscErrorCode TSGetTolerances(TS,PetscReal*,Vec*,PetscReal*,Vec*);
PETSC_EXTERN PetscErrorCode TSErrorNormWRMS(TS,Vec,PetscReal*);
PETSC_EXTERN PetscErrorCode TSSetCFLTimeLocal(TS,PetscReal);
PETSC_EXTERN PetscErrorCode TSGetCFLTime(TS,PetscReal*);

PETSC_EXTERN PetscErrorCode TSPseudoSetTimeStep(TS,PetscErrorCode(*)(TS,PetscReal*,void*),void*);
PETSC_EXTERN PetscErrorCode TSPseudoDefaultTimeStep(TS,PetscReal*,void*);
PETSC_EXTERN PetscErrorCode TSPseudoComputeTimeStep(TS,PetscReal *);
PETSC_EXTERN PetscErrorCode TSPseudoSetMaxTimeStep(TS,PetscReal);

PETSC_EXTERN PetscErrorCode TSPseudoSetVerifyTimeStep(TS,PetscErrorCode(*)(TS,Vec,void*,PetscReal*,PetscBool *),void*);
PETSC_EXTERN PetscErrorCode TSPseudoDefaultVerifyTimeStep(TS,Vec,void*,PetscReal*,PetscBool *);
PETSC_EXTERN PetscErrorCode TSPseudoVerifyTimeStep(TS,Vec,PetscReal*,PetscBool *);
PETSC_EXTERN PetscErrorCode TSPseudoSetTimeStepIncrement(TS,PetscReal);
PETSC_EXTERN PetscErrorCode TSPseudoIncrementDtFromInitialDt(TS);

PETSC_EXTERN PetscErrorCode TSPythonSetType(TS,const char[]);

PETSC_EXTERN PetscErrorCode TSComputeRHSFunction(TS,PetscReal,Vec,Vec);
PETSC_EXTERN PetscErrorCode TSComputeRHSJacobian(TS,PetscReal,Vec,Mat*,Mat*,MatStructure*);
PETSC_EXTERN PetscErrorCode TSComputeIFunction(TS,PetscReal,Vec,Vec,Vec,PetscBool);
PETSC_EXTERN PetscErrorCode TSComputeIJacobian(TS,PetscReal,Vec,Vec,PetscReal,Mat*,Mat*,MatStructure*,PetscBool);

PETSC_EXTERN PetscErrorCode TSVISetVariableBounds(TS,Vec,Vec);

/* Dynamic creation and loading functions */
PETSC_EXTERN PetscFList TSList;
PETSC_EXTERN PetscBool TSRegisterAllCalled;
PETSC_EXTERN PetscErrorCode TSGetType(TS,const TSType*);
PETSC_EXTERN PetscErrorCode TSSetType(TS,const TSType);
PETSC_EXTERN PetscErrorCode TSRegister(const char[], const char[], const char[], PetscErrorCode (*)(TS));
PETSC_EXTERN PetscErrorCode TSRegisterAll(const char[]);
PETSC_EXTERN PetscErrorCode TSRegisterDestroy(void);

/*MC
  TSRegisterDynamic - Adds a creation method to the TS package.

  Synopsis:
  PetscErrorCode TSRegisterDynamic(const char *name, const char *path, const char *func_name, PetscErrorCode (*create_func)(TS))

  Not Collective

  Input Parameters:
+ name        - The name of a new user-defined creation routine
. path        - The path (either absolute or relative) of the library containing this routine
. func_name   - The name of the creation routine
- create_func - The creation routine itself

  Notes:
  TSRegisterDynamic() may be called multiple times to add several user-defined tses.

  If dynamic libraries are used, then the fourth input argument (create_func) is ignored.

  Sample usage:
.vb
  TSRegisterDynamic("my_ts", "/home/username/my_lib/lib/libO/solaris/libmy.a", "MyTSCreate", MyTSCreate);
.ve

  Then, your ts type can be chosen with the procedural interface via
.vb
    TS ts;
    TSCreate(MPI_Comm, &ts);
    TSSetType(ts, "my_ts")
.ve
  or at runtime via the option
.vb
    -ts_type my_ts
.ve

  Notes: $PETSC_ARCH occuring in pathname will be replaced with appropriate values.
        If your function is not being put into a shared library then use TSRegister() instead

  Level: advanced

.keywords: TS, register
.seealso: TSRegisterAll(), TSRegisterDestroy()
M*/
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define TSRegisterDynamic(a,b,c,d) TSRegister(a,b,c,0)
#else
#define TSRegisterDynamic(a,b,c,d) TSRegister(a,b,c,d)
#endif

PETSC_EXTERN PetscErrorCode TSGetSNES(TS,SNES*);
PETSC_EXTERN PetscErrorCode TSGetKSP(TS,KSP*);

PETSC_EXTERN PetscErrorCode TSView(TS,PetscViewer);

PETSC_EXTERN PetscErrorCode TSSetApplicationContext(TS,void *);
PETSC_EXTERN PetscErrorCode TSGetApplicationContext(TS,void *);

PETSC_EXTERN PetscErrorCode TSMonitorLGCreate(const char[],const char[],int,int,int,int,PetscDrawLG *);
PETSC_EXTERN PetscErrorCode TSMonitorLG(TS,PetscInt,PetscReal,Vec,void *);
PETSC_EXTERN PetscErrorCode TSMonitorLGDestroy(PetscDrawLG*);

/*J
   TSSSPType - string with the name of TSSSP scheme.

   Level: beginner

.seealso: TSSSPSetType(), TS
J*/
#define TSSSPType char*
#define TSSSPRKS2  "rks2"
#define TSSSPRKS3  "rks3"
#define TSSSPRK104 "rk104"

PETSC_EXTERN PetscErrorCode TSSSPSetType(TS,const TSSSPType);
PETSC_EXTERN PetscErrorCode TSSSPGetType(TS,const TSSSPType*);
PETSC_EXTERN PetscErrorCode TSSSPSetNumStages(TS,PetscInt);
PETSC_EXTERN PetscErrorCode TSSSPGetNumStages(TS,PetscInt*);

/*S
   TSAdapt - Abstract object that manages time-step adaptivity

   Level: beginner

.seealso: TS, TSAdaptCreate(), TSAdaptType
S*/
typedef struct _p_TSAdapt *TSAdapt;

/*E
    TSAdaptType - String with the name of TSAdapt scheme or the creation function
       with an optional dynamic library name, for example
       http://www.mcs.anl.gov/petsc/lib.a:mytsgladaptcreate()

   Level: beginner

.seealso: TSAdaptSetType(), TS
E*/
#define TSAdaptType  char*
#define TSADAPTBASIC "basic"
#define TSADAPTNONE  "none"
#define TSADAPTCFL   "cfl"

/*MC
   TSAdaptRegisterDynamic - adds a TSAdapt implementation

   Synopsis:
   PetscErrorCode TSAdaptRegisterDynamic(const char *name_scheme,const char *path,const char *name_create,PetscErrorCode (*routine_create)(TS))

   Not Collective

   Input Parameters:
+  name_scheme - name of user-defined adaptivity scheme
.  path - path (either absolute or relative) the library containing this scheme
.  name_create - name of routine to create method context
-  routine_create - routine to create method context

   Notes:
   TSAdaptRegisterDynamic() may be called multiple times to add several user-defined families.

   If dynamic libraries are used, then the fourth input argument (routine_create)
   is ignored.

   Sample usage:
.vb
   TSAdaptRegisterDynamic("my_scheme",/home/username/my_lib/lib/libO/solaris/mylib.a,
                            "MySchemeCreate",MySchemeCreate);
.ve

   Then, your scheme can be chosen with the procedural interface via
$     TSAdaptSetType(ts,"my_scheme")
   or at runtime via the option
$     -ts_adapt_type my_scheme

   Level: advanced

   Notes: Environmental variables such as ${PETSC_ARCH}, ${PETSC_DIR}, ${PETSC_LIB_DIR},
          and others of the form ${any_environmental_variable} occuring in pathname will be 
          replaced with appropriate values.

.keywords: TSAdapt, register

.seealso: TSAdaptRegisterAll()
M*/
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#  define TSAdaptRegisterDynamic(a,b,c,d)  TSAdaptRegister(a,b,c,0)
#else
#  define TSAdaptRegisterDynamic(a,b,c,d)  TSAdaptRegister(a,b,c,d)
#endif

PETSC_EXTERN PetscErrorCode TSGetAdapt(TS,TSAdapt*);
PETSC_EXTERN PetscErrorCode TSAdaptRegister(const char[],const char[],const char[],PetscErrorCode (*)(TSAdapt));
PETSC_EXTERN PetscErrorCode TSAdaptRegisterAll(const char[]);
PETSC_EXTERN PetscErrorCode TSAdaptRegisterDestroy(void);
PETSC_EXTERN PetscErrorCode TSAdaptInitializePackage(const char[]);
PETSC_EXTERN PetscErrorCode TSAdaptFinalizePackage(void);
PETSC_EXTERN PetscErrorCode TSAdaptCreate(MPI_Comm,TSAdapt*);
PETSC_EXTERN PetscErrorCode TSAdaptSetType(TSAdapt,const TSAdaptType);
PETSC_EXTERN PetscErrorCode TSAdaptSetOptionsPrefix(TSAdapt,const char[]);
PETSC_EXTERN PetscErrorCode TSAdaptCandidatesClear(TSAdapt);
PETSC_EXTERN PetscErrorCode TSAdaptCandidateAdd(TSAdapt,const char[],PetscInt,PetscInt,PetscReal,PetscReal,PetscBool);
PETSC_EXTERN PetscErrorCode TSAdaptCandidatesGet(TSAdapt,PetscInt*,const PetscInt**,const PetscInt**,const PetscReal**,const PetscReal**);
PETSC_EXTERN PetscErrorCode TSAdaptChoose(TSAdapt,TS,PetscReal,PetscInt*,PetscReal*,PetscBool*);
PETSC_EXTERN PetscErrorCode TSAdaptCheckStage(TSAdapt,TS,PetscBool*);
PETSC_EXTERN PetscErrorCode TSAdaptView(TSAdapt,PetscViewer);
PETSC_EXTERN PetscErrorCode TSAdaptSetFromOptions(TSAdapt);
PETSC_EXTERN PetscErrorCode TSAdaptDestroy(TSAdapt*);
PETSC_EXTERN PetscErrorCode TSAdaptSetMonitor(TSAdapt,PetscBool);
PETSC_EXTERN PetscErrorCode TSAdaptSetStepLimits(TSAdapt,PetscReal,PetscReal);
PETSC_EXTERN PetscErrorCode TSAdaptSetCheckStage(TSAdapt,PetscErrorCode(*)(TSAdapt,TS,PetscBool*));

/*S
   TSGLAdapt - Abstract object that manages time-step adaptivity

   Level: beginner

   Developer Notes:
   This functionality should be replaced by the TSAdapt.

.seealso: TSGL, TSGLAdaptCreate(), TSGLAdaptType
S*/
typedef struct _p_TSGLAdapt *TSGLAdapt;

/*J
    TSGLAdaptType - String with the name of TSGLAdapt scheme or the creation function
       with an optional dynamic library name, for example
       http://www.mcs.anl.gov/petsc/lib.a:mytsgladaptcreate()

   Level: beginner

.seealso: TSGLAdaptSetType(), TS
J*/
#define TSGLAdaptType  char*
#define TSGLADAPT_NONE "none"
#define TSGLADAPT_SIZE "size"
#define TSGLADAPT_BOTH "both"

/*MC
   TSGLAdaptRegisterDynamic - adds a TSGLAdapt implementation

   Synopsis:
   PetscErrorCode TSGLAdaptRegisterDynamic(const char *name_scheme,const char *path,const char *name_create,PetscErrorCode (*routine_create)(TS))

   Not Collective

   Input Parameters:
+  name_scheme - name of user-defined adaptivity scheme
.  path - path (either absolute or relative) the library containing this scheme
.  name_create - name of routine to create method context
-  routine_create - routine to create method context

   Notes:
   TSGLAdaptRegisterDynamic() may be called multiple times to add several user-defined families.

   If dynamic libraries are used, then the fourth input argument (routine_create)
   is ignored.

   Sample usage:
.vb
   TSGLAdaptRegisterDynamic("my_scheme",/home/username/my_lib/lib/libO/solaris/mylib.a,
                            "MySchemeCreate",MySchemeCreate);
.ve

   Then, your scheme can be chosen with the procedural interface via
$     TSGLAdaptSetType(ts,"my_scheme")
   or at runtime via the option
$     -ts_adapt_type my_scheme

   Level: advanced

   Notes: Environmental variables such as ${PETSC_ARCH}, ${PETSC_DIR}, ${PETSC_LIB_DIR},
          and others of the form ${any_environmental_variable} occuring in pathname will be 
          replaced with appropriate values.

.keywords: TSGLAdapt, register

.seealso: TSGLAdaptRegisterAll()
M*/
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#  define TSGLAdaptRegisterDynamic(a,b,c,d)  TSGLAdaptRegister(a,b,c,0)
#else
#  define TSGLAdaptRegisterDynamic(a,b,c,d)  TSGLAdaptRegister(a,b,c,d)
#endif

PETSC_EXTERN PetscErrorCode TSGLAdaptRegister(const char[],const char[],const char[],PetscErrorCode (*)(TSGLAdapt));
PETSC_EXTERN PetscErrorCode TSGLAdaptRegisterAll(const char[]);
PETSC_EXTERN PetscErrorCode TSGLAdaptRegisterDestroy(void);
PETSC_EXTERN PetscErrorCode TSGLAdaptInitializePackage(const char[]);
PETSC_EXTERN PetscErrorCode TSGLAdaptFinalizePackage(void);
PETSC_EXTERN PetscErrorCode TSGLAdaptCreate(MPI_Comm,TSGLAdapt*);
PETSC_EXTERN PetscErrorCode TSGLAdaptSetType(TSGLAdapt,const TSGLAdaptType);
PETSC_EXTERN PetscErrorCode TSGLAdaptSetOptionsPrefix(TSGLAdapt,const char[]);
PETSC_EXTERN PetscErrorCode TSGLAdaptChoose(TSGLAdapt,PetscInt,const PetscInt[],const PetscReal[],const PetscReal[],PetscInt,PetscReal,PetscReal,PetscInt*,PetscReal*,PetscBool *);
PETSC_EXTERN PetscErrorCode TSGLAdaptView(TSGLAdapt,PetscViewer);
PETSC_EXTERN PetscErrorCode TSGLAdaptSetFromOptions(TSGLAdapt);
PETSC_EXTERN PetscErrorCode TSGLAdaptDestroy(TSGLAdapt*);

/*J
    TSGLAcceptType - String with the name of TSGLAccept scheme or the function
       with an optional dynamic library name, for example
       http://www.mcs.anl.gov/petsc/lib.a:mytsglaccept()

   Level: beginner

.seealso: TSGLSetAcceptType(), TS
J*/
#define TSGLAcceptType  char*
#define TSGLACCEPT_ALWAYS "always"

PETSC_EXTERN_TYPEDEF typedef PetscErrorCode (*TSGLAcceptFunction)(TS,PetscReal,PetscReal,const PetscReal[],PetscBool *);
PETSC_EXTERN PetscErrorCode TSGLAcceptRegister(const char[],const char[],const char[],TSGLAcceptFunction);

/*MC
   TSGLAcceptRegisterDynamic - adds a TSGL acceptance scheme

   Synopsis:
   PetscErrorCode TSGLAcceptRegisterDynamic(const char *name_scheme,const char *path,const char *name_create,PetscErrorCode (*routine_create)(TS))

   Not Collective

   Input Parameters:
+  name_scheme - name of user-defined acceptance scheme
.  path - path (either absolute or relative) the library containing this scheme
.  name_create - name of routine to create method context
-  routine_create - routine to create method context

   Notes:
   TSGLAcceptRegisterDynamic() may be called multiple times to add several user-defined families.

   If dynamic libraries are used, then the fourth input argument (routine_create)
   is ignored.

   Sample usage:
.vb
   TSGLAcceptRegisterDynamic("my_scheme",/home/username/my_lib/lib/libO/solaris/mylib.a,
                             "MySchemeCreate",MySchemeCreate);
.ve

   Then, your scheme can be chosen with the procedural interface via
$     TSGLSetAcceptType(ts,"my_scheme")
   or at runtime via the option
$     -ts_gl_accept_type my_scheme

   Level: advanced

   Notes: Environmental variables such as ${PETSC_ARCH}, ${PETSC_DIR}, ${PETSC_LIB_DIR},
          and others of the form ${any_environmental_variable} occuring in pathname will be 
          replaced with appropriate values.

.keywords: TSGL, TSGLAcceptType, register

.seealso: TSGLRegisterAll()
M*/
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#  define TSGLAcceptRegisterDynamic(a,b,c,d) TSGLAcceptRegister(a,b,c,0)
#else
#  define TSGLAcceptRegisterDynamic(a,b,c,d) TSGLAcceptRegister(a,b,c,d)
#endif

/*J
  TSGLType - family of time integration method within the General Linear class

  Level: beginner

.seealso: TSGLSetType(), TSGLRegister()
J*/
#define TSGLType char*
#define TSGL_IRKS   "irks"

/*MC
   TSGLRegisterDynamic - adds a TSGL implementation

   Synopsis:
   PetscErrorCode TSGLRegisterDynamic(const char *name_scheme,const char *path,const char *name_create,PetscErrorCode (*routine_create)(TS))

   Not Collective

   Input Parameters:
+  name_scheme - name of user-defined general linear scheme
.  path - path (either absolute or relative) the library containing this scheme
.  name_create - name of routine to create method context
-  routine_create - routine to create method context

   Notes:
   TSGLRegisterDynamic() may be called multiple times to add several user-defined families.

   If dynamic libraries are used, then the fourth input argument (routine_create)
   is ignored.

   Sample usage:
.vb
   TSGLRegisterDynamic("my_scheme",/home/username/my_lib/lib/libO/solaris/mylib.a,
                       "MySchemeCreate",MySchemeCreate);
.ve

   Then, your scheme can be chosen with the procedural interface via
$     TSGLSetType(ts,"my_scheme")
   or at runtime via the option
$     -ts_gl_type my_scheme

   Level: advanced

   Notes: Environmental variables such as ${PETSC_ARCH}, ${PETSC_DIR}, ${PETSC_LIB_DIR},
          and others of the form ${any_environmental_variable} occuring in pathname will be 
          replaced with appropriate values.

.keywords: TSGL, register

.seealso: TSGLRegisterAll()
M*/
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#  define TSGLRegisterDynamic(a,b,c,d)       TSGLRegister(a,b,c,0)
#else
#  define TSGLRegisterDynamic(a,b,c,d)       TSGLRegister(a,b,c,d)
#endif

PETSC_EXTERN PetscErrorCode TSGLRegister(const char[],const char[],const char[],PetscErrorCode(*)(TS));
PETSC_EXTERN PetscErrorCode TSGLRegisterAll(const char[]);
PETSC_EXTERN PetscErrorCode TSGLRegisterDestroy(void);
PETSC_EXTERN PetscErrorCode TSGLInitializePackage(const char[]);
PETSC_EXTERN PetscErrorCode TSGLFinalizePackage(void);
PETSC_EXTERN PetscErrorCode TSGLSetType(TS,const TSGLType);
PETSC_EXTERN PetscErrorCode TSGLGetAdapt(TS,TSGLAdapt*);
PETSC_EXTERN PetscErrorCode TSGLSetAcceptType(TS,const TSGLAcceptType);

/*J
    TSARKIMEXType - String with the name of an Additive Runge-Kutta IMEX method.

   Level: beginner

.seealso: TSARKIMEXSetType(), TS, TSARKIMEX, TSARKIMEXRegister()
J*/
#define TSARKIMEXType char*
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
PETSC_EXTERN PetscErrorCode TSARKIMEXGetType(TS ts,const TSARKIMEXType*);
PETSC_EXTERN PetscErrorCode TSARKIMEXSetType(TS ts,const TSARKIMEXType);
PETSC_EXTERN PetscErrorCode TSARKIMEXSetFullyImplicit(TS,PetscBool);
PETSC_EXTERN PetscErrorCode TSARKIMEXRegister(const TSARKIMEXType,PetscInt,PetscInt,const PetscReal[],const PetscReal[],const PetscReal[],const PetscReal[],const PetscReal[],const PetscReal[],const PetscReal[],const PetscReal[],PetscInt,const PetscReal[],const PetscReal[]);
PETSC_EXTERN PetscErrorCode TSARKIMEXFinalizePackage(void);
PETSC_EXTERN PetscErrorCode TSARKIMEXInitializePackage(const char path[]);
PETSC_EXTERN PetscErrorCode TSARKIMEXRegisterDestroy(void);
PETSC_EXTERN PetscErrorCode TSARKIMEXRegisterAll(void);

/*J
    TSRosWType - String with the name of a Rosenbrock-W method.

   Level: beginner

.seealso: TSRosWSetType(), TS, TSROSW, TSRosWRegister()
J*/
#define TSRosWType char*
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


PETSC_EXTERN PetscErrorCode TSRosWGetType(TS ts,const TSRosWType*);
PETSC_EXTERN PetscErrorCode TSRosWSetType(TS ts,const TSRosWType);
PETSC_EXTERN PetscErrorCode TSRosWSetRecomputeJacobian(TS,PetscBool);
PETSC_EXTERN PetscErrorCode TSRosWRegister(const TSRosWType,PetscInt,PetscInt,const PetscReal[],const PetscReal[],const PetscReal[],const PetscReal[],PetscInt,const PetscReal[]);
PETSC_EXTERN PetscErrorCode TSRosWFinalizePackage(void);
PETSC_EXTERN PetscErrorCode TSRosWInitializePackage(const char path[]);
PETSC_EXTERN PetscErrorCode TSRosWRegisterDestroy(void);
PETSC_EXTERN PetscErrorCode TSRosWRegisterAll(void);

/*
       PETSc interface to Sundials
*/
#ifdef PETSC_HAVE_SUNDIALS
typedef enum { SUNDIALS_ADAMS=1,SUNDIALS_BDF=2} TSSundialsLmmType;
PETSC_EXTERN const char *TSSundialsLmmTypes[];
typedef enum { SUNDIALS_MODIFIED_GS = 1,SUNDIALS_CLASSICAL_GS = 2 } TSSundialsGramSchmidtType;
PETSC_EXTERN const char *TSSundialsGramSchmidtTypes[];
PETSC_EXTERN PetscErrorCode TSSundialsSetType(TS,TSSundialsLmmType);
PETSC_EXTERN PetscErrorCode TSSundialsGetPC(TS,PC*);
PETSC_EXTERN PetscErrorCode TSSundialsSetTolerance(TS,PetscReal,PetscReal);
PETSC_EXTERN PetscErrorCode TSSundialsSetMinTimeStep(TS,PetscReal);
PETSC_EXTERN PetscErrorCode TSSundialsSetMaxTimeStep(TS,PetscReal);
PETSC_EXTERN PetscErrorCode TSSundialsGetIterations(TS,PetscInt *,PetscInt *);
PETSC_EXTERN PetscErrorCode TSSundialsSetGramSchmidtType(TS,TSSundialsGramSchmidtType);
PETSC_EXTERN PetscErrorCode TSSundialsSetGMRESRestart(TS,PetscInt);
PETSC_EXTERN PetscErrorCode TSSundialsSetLinearTolerance(TS,PetscReal);
PETSC_EXTERN PetscErrorCode TSSundialsMonitorInternalSteps(TS,PetscBool );
PETSC_EXTERN PetscErrorCode TSSundialsGetParameters(TS,PetscInt *,long*[],double*[]);
PETSC_EXTERN PetscErrorCode TSSundialsSetMaxl(TS,PetscInt);
#endif

PETSC_EXTERN PetscErrorCode TSRKSetTolerance(TS,PetscReal);

PETSC_EXTERN PetscErrorCode TSThetaSetTheta(TS,PetscReal);
PETSC_EXTERN PetscErrorCode TSThetaGetTheta(TS,PetscReal*);
PETSC_EXTERN PetscErrorCode TSThetaGetEndpoint(TS,PetscBool*);
PETSC_EXTERN PetscErrorCode TSThetaSetEndpoint(TS,PetscBool);

PETSC_EXTERN PetscErrorCode TSAlphaSetAdapt(TS,PetscErrorCode(*)(TS,PetscReal,Vec,Vec,PetscReal*,PetscBool*,void*),void*);
PETSC_EXTERN PetscErrorCode TSAlphaAdaptDefault(TS,PetscReal,Vec,Vec,PetscReal*,PetscBool*,void*);
PETSC_EXTERN PetscErrorCode TSAlphaSetRadius(TS,PetscReal);
PETSC_EXTERN PetscErrorCode TSAlphaSetParams(TS,PetscReal,PetscReal,PetscReal);
PETSC_EXTERN PetscErrorCode TSAlphaGetParams(TS,PetscReal*,PetscReal*,PetscReal*);

PETSC_EXTERN PetscErrorCode TSSetDM(TS,DM);
PETSC_EXTERN PetscErrorCode TSGetDM(TS,DM*);

PETSC_EXTERN PetscErrorCode SNESTSFormFunction(SNES,Vec,Vec,void*);
PETSC_EXTERN PetscErrorCode SNESTSFormJacobian(SNES,Vec,Mat*,Mat*,MatStructure*,void*);

#endif
