/*
   User interface for the timestepping package. This package
   is for use in solving time-dependent PDEs.
*/
#if !defined(__PETSCTS_H)
#define __PETSCTS_H
#include "petscsnes.h"
PETSC_EXTERN_CXX_BEGIN

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
extern const char *const*TSConvergedReasons;

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
extern PetscClassId  TS_CLASSID;

extern PetscErrorCode   TSInitializePackage(const char[]);

extern PetscErrorCode   TSCreate(MPI_Comm,TS*);
extern PetscErrorCode   TSDestroy(TS*);

extern PetscErrorCode   TSSetProblemType(TS,TSProblemType);
extern PetscErrorCode   TSGetProblemType(TS,TSProblemType*);
extern PetscErrorCode   TSMonitor(TS,PetscInt,PetscReal,Vec);
extern PetscErrorCode   TSMonitorSet(TS,PetscErrorCode(*)(TS,PetscInt,PetscReal,Vec,void*),void *,PetscErrorCode (*)(void**));
extern PetscErrorCode   TSMonitorCancel(TS);

extern PetscErrorCode   TSSetOptionsPrefix(TS,const char[]);
extern PetscErrorCode   TSAppendOptionsPrefix(TS,const char[]);
extern PetscErrorCode   TSGetOptionsPrefix(TS,const char *[]);
extern PetscErrorCode   TSSetFromOptions(TS);
extern PetscErrorCode   TSSetUp(TS);
extern PetscErrorCode   TSReset(TS);

extern PetscErrorCode   TSSetSolution(TS,Vec);
extern PetscErrorCode   TSGetSolution(TS,Vec*);

extern PetscErrorCode   TSSetDuration(TS,PetscInt,PetscReal);
extern PetscErrorCode   TSGetDuration(TS,PetscInt*,PetscReal*);
extern PetscErrorCode   TSSetExactFinalTime(TS,PetscBool);

extern PetscErrorCode   TSMonitorDefault(TS,PetscInt,PetscReal,Vec,void*);
extern PetscErrorCode   TSMonitorSolution(TS,PetscInt,PetscReal,Vec,void*);
extern PetscErrorCode   TSMonitorSolutionCreate(TS,PetscViewer,void**);
extern PetscErrorCode   TSMonitorSolutionDestroy(void**);
extern PetscErrorCode   TSMonitorSolutionBinary(TS,PetscInt,PetscReal,Vec,void*);
extern PetscErrorCode   TSMonitorSolutionVTK(TS,PetscInt,PetscReal,Vec,void*);
extern PetscErrorCode   TSMonitorSolutionVTKDestroy(void*);

extern PetscErrorCode   TSStep(TS);
extern PetscErrorCode   TSEvaluateStep(TS,PetscInt,Vec,PetscBool*);
extern PetscErrorCode   TSSolve(TS,Vec,PetscReal*);
extern PetscErrorCode   TSGetConvergedReason(TS,TSConvergedReason*);
extern PetscErrorCode   TSGetNonlinearSolveIterations(TS,PetscInt*);
extern PetscErrorCode   TSGetLinearSolveIterations(TS,PetscInt*);

extern PetscErrorCode   TSSetInitialTimeStep(TS,PetscReal,PetscReal);
extern PetscErrorCode   TSGetTimeStep(TS,PetscReal*);
extern PetscErrorCode   TSGetTime(TS,PetscReal*);
extern PetscErrorCode   TSSetTime(TS,PetscReal);
extern PetscErrorCode   TSGetTimeStepNumber(TS,PetscInt*);
extern PetscErrorCode   TSSetTimeStep(TS,PetscReal);

typedef PetscErrorCode (*TSRHSFunction)(TS,PetscReal,Vec,Vec,void*);
typedef PetscErrorCode (*TSRHSJacobian)(TS,PetscReal,Vec,Mat*,Mat*,MatStructure*,void*);
extern PetscErrorCode   TSSetRHSFunction(TS,Vec,TSRHSFunction,void*);
extern PetscErrorCode   TSGetRHSFunction(TS,Vec*,TSRHSFunction*,void**);
extern PetscErrorCode   TSSetRHSJacobian(TS,Mat,Mat,TSRHSJacobian,void*);
extern PetscErrorCode   TSGetRHSJacobian(TS,Mat*,Mat*,TSRHSJacobian*,void**);

typedef PetscErrorCode (*TSIFunction)(TS,PetscReal,Vec,Vec,Vec,void*);
typedef PetscErrorCode (*TSIJacobian)(TS,PetscReal,Vec,Vec,PetscReal,Mat*,Mat*,MatStructure*,void*);
extern PetscErrorCode   TSSetIFunction(TS,Vec,TSIFunction,void*);
extern PetscErrorCode   TSGetIFunction(TS,Vec*,TSIFunction*,void**);
extern PetscErrorCode   TSSetIJacobian(TS,Mat,Mat,TSIJacobian,void*);
extern PetscErrorCode   TSGetIJacobian(TS,Mat*,Mat*,TSIJacobian*,void**);

extern PetscErrorCode   TSComputeRHSFunctionLinear(TS,PetscReal,Vec,Vec,void*);
extern PetscErrorCode   TSComputeRHSJacobianConstant(TS,PetscReal,Vec,Mat*,Mat*,MatStructure*,void*);
extern PetscErrorCode   TSComputeIFunctionLinear(TS,PetscReal,Vec,Vec,Vec,void*);
extern PetscErrorCode   TSComputeIJacobianConstant(TS,PetscReal,Vec,Vec,PetscReal,Mat*,Mat*,MatStructure*,void*);

extern PetscErrorCode   TSSetPreStep(TS, PetscErrorCode (*)(TS));
extern PetscErrorCode   TSSetPostStep(TS, PetscErrorCode (*)(TS));
extern PetscErrorCode   TSPreStep(TS);
extern PetscErrorCode   TSPostStep(TS);
extern PetscErrorCode   TSSetRetainStages(TS,PetscBool);
extern PetscErrorCode   TSInterpolate(TS,PetscReal,Vec);
extern PetscErrorCode   TSSetTolerances(TS,PetscReal,Vec,PetscReal,Vec);
extern PetscErrorCode   TSErrorNormWRMS(TS,Vec,PetscReal*);
extern PetscErrorCode   TSSetCFLTimeLocal(TS,PetscReal);
extern PetscErrorCode   TSGetCFLTime(TS,PetscReal*);

extern PetscErrorCode   TSPseudoSetTimeStep(TS,PetscErrorCode(*)(TS,PetscReal*,void*),void*);
extern PetscErrorCode   TSPseudoDefaultTimeStep(TS,PetscReal*,void*);
extern PetscErrorCode   TSPseudoComputeTimeStep(TS,PetscReal *);
extern PetscErrorCode   TSPseudoSetMaxTimeStep(TS,PetscReal);

extern PetscErrorCode   TSPseudoSetVerifyTimeStep(TS,PetscErrorCode(*)(TS,Vec,void*,PetscReal*,PetscBool *),void*);
extern PetscErrorCode   TSPseudoDefaultVerifyTimeStep(TS,Vec,void*,PetscReal*,PetscBool *);
extern PetscErrorCode   TSPseudoVerifyTimeStep(TS,Vec,PetscReal*,PetscBool *);
extern PetscErrorCode   TSPseudoSetTimeStepIncrement(TS,PetscReal);
extern PetscErrorCode   TSPseudoIncrementDtFromInitialDt(TS);

extern PetscErrorCode   TSPythonSetType(TS,const char[]);

extern PetscErrorCode   TSComputeRHSFunction(TS,PetscReal,Vec,Vec);
extern PetscErrorCode   TSComputeRHSJacobian(TS,PetscReal,Vec,Mat*,Mat*,MatStructure*);
extern PetscErrorCode   TSComputeIFunction(TS,PetscReal,Vec,Vec,Vec,PetscBool);
extern PetscErrorCode   TSComputeIJacobian(TS,PetscReal,Vec,Vec,PetscReal,Mat*,Mat*,MatStructure*,PetscBool);

extern PetscErrorCode   TSVISetVariableBounds(TS,Vec,Vec);

/* Dynamic creation and loading functions */
extern PetscFList TSList;
extern PetscBool  TSRegisterAllCalled;
extern PetscErrorCode   TSGetType(TS,const TSType*);
extern PetscErrorCode   TSSetType(TS,const TSType);
extern PetscErrorCode   TSRegister(const char[], const char[], const char[], PetscErrorCode (*)(TS));
extern PetscErrorCode   TSRegisterAll(const char[]);
extern PetscErrorCode   TSRegisterDestroy(void);

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

extern PetscErrorCode   TSGetSNES(TS,SNES*);
extern PetscErrorCode   TSGetKSP(TS,KSP*);

extern PetscErrorCode   TSView(TS,PetscViewer);

extern PetscErrorCode   TSSetApplicationContext(TS,void *);
extern PetscErrorCode   TSGetApplicationContext(TS,void *);

extern PetscErrorCode   TSMonitorLGCreate(const char[],const char[],int,int,int,int,PetscDrawLG *);
extern PetscErrorCode   TSMonitorLG(TS,PetscInt,PetscReal,Vec,void *);
extern PetscErrorCode   TSMonitorLGDestroy(PetscDrawLG*);

/*J
   TSSSPType - string with the name of TSSSP scheme.

   Level: beginner

.seealso: TSSSPSetType(), TS
J*/
#define TSSSPType char*
#define TSSSPRKS2  "rks2"
#define TSSSPRKS3  "rks3"
#define TSSSPRK104 "rk104"

extern PetscErrorCode TSSSPSetType(TS,const TSSSPType);
extern PetscErrorCode TSSSPGetType(TS,const TSSSPType*);
extern PetscErrorCode TSSSPSetNumStages(TS,PetscInt);
extern PetscErrorCode TSSSPGetNumStages(TS,PetscInt*);

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

extern PetscErrorCode TSGetAdapt(TS,TSAdapt*);
extern PetscErrorCode TSAdaptRegister(const char[],const char[],const char[],PetscErrorCode (*)(TSAdapt));
extern PetscErrorCode TSAdaptRegisterAll(const char[]);
extern PetscErrorCode TSAdaptRegisterDestroy(void);
extern PetscErrorCode TSAdaptInitializePackage(const char[]);
extern PetscErrorCode TSAdaptFinalizePackage(void);
extern PetscErrorCode TSAdaptCreate(MPI_Comm,TSAdapt*);
extern PetscErrorCode TSAdaptSetType(TSAdapt,const TSAdaptType);
extern PetscErrorCode TSAdaptSetOptionsPrefix(TSAdapt,const char[]);
extern PetscErrorCode TSAdaptCandidatesClear(TSAdapt);
extern PetscErrorCode TSAdaptCandidateAdd(TSAdapt,const char[],PetscInt,PetscInt,PetscReal,PetscReal,PetscBool);
extern PetscErrorCode TSAdaptCandidatesGet(TSAdapt,PetscInt*,const PetscInt**,const PetscInt**,const PetscReal**,const PetscReal**);
extern PetscErrorCode TSAdaptChoose(TSAdapt,TS,PetscReal,PetscInt*,PetscReal*,PetscBool*);
extern PetscErrorCode TSAdaptCheckStage(TSAdapt,TS,PetscBool*);
extern PetscErrorCode TSAdaptView(TSAdapt,PetscViewer);
extern PetscErrorCode TSAdaptSetFromOptions(TSAdapt);
extern PetscErrorCode TSAdaptDestroy(TSAdapt*);
extern PetscErrorCode TSAdaptSetMonitor(TSAdapt,PetscBool);
extern PetscErrorCode TSAdaptSetStepLimits(TSAdapt,PetscReal,PetscReal);

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

extern PetscErrorCode  TSGLAdaptRegister(const char[],const char[],const char[],PetscErrorCode (*)(TSGLAdapt));
extern PetscErrorCode  TSGLAdaptRegisterAll(const char[]);
extern PetscErrorCode  TSGLAdaptRegisterDestroy(void);
extern PetscErrorCode  TSGLAdaptInitializePackage(const char[]);
extern PetscErrorCode  TSGLAdaptFinalizePackage(void);
extern PetscErrorCode  TSGLAdaptCreate(MPI_Comm,TSGLAdapt*);
extern PetscErrorCode  TSGLAdaptSetType(TSGLAdapt,const TSGLAdaptType);
extern PetscErrorCode  TSGLAdaptSetOptionsPrefix(TSGLAdapt,const char[]);
extern PetscErrorCode  TSGLAdaptChoose(TSGLAdapt,PetscInt,const PetscInt[],const PetscReal[],const PetscReal[],PetscInt,PetscReal,PetscReal,PetscInt*,PetscReal*,PetscBool *);
extern PetscErrorCode  TSGLAdaptView(TSGLAdapt,PetscViewer);
extern PetscErrorCode  TSGLAdaptSetFromOptions(TSGLAdapt);
extern PetscErrorCode  TSGLAdaptDestroy(TSGLAdapt*);

/*J
    TSGLAcceptType - String with the name of TSGLAccept scheme or the function
       with an optional dynamic library name, for example
       http://www.mcs.anl.gov/petsc/lib.a:mytsglaccept()

   Level: beginner

.seealso: TSGLSetAcceptType(), TS
J*/
#define TSGLAcceptType  char*
#define TSGLACCEPT_ALWAYS "always"

typedef PetscErrorCode (*TSGLAcceptFunction)(TS,PetscReal,PetscReal,const PetscReal[],PetscBool *);
extern PetscErrorCode  TSGLAcceptRegister(const char[],const char[],const char[],TSGLAcceptFunction);

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

extern PetscErrorCode  TSGLRegister(const char[],const char[],const char[],PetscErrorCode(*)(TS));
extern PetscErrorCode  TSGLRegisterAll(const char[]);
extern PetscErrorCode  TSGLRegisterDestroy(void);
extern PetscErrorCode  TSGLInitializePackage(const char[]);
extern PetscErrorCode  TSGLFinalizePackage(void);
extern PetscErrorCode  TSGLSetType(TS,const TSGLType);
extern PetscErrorCode  TSGLGetAdapt(TS,TSGLAdapt*);
extern PetscErrorCode  TSGLSetAcceptType(TS,const TSGLAcceptType);

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
extern PetscErrorCode TSARKIMEXGetType(TS ts,const TSARKIMEXType*);
extern PetscErrorCode TSARKIMEXSetType(TS ts,const TSARKIMEXType);
extern PetscErrorCode TSARKIMEXSetFullyImplicit(TS,PetscBool);
extern PetscErrorCode TSARKIMEXRegister(const TSARKIMEXType,PetscInt,PetscInt,const PetscReal[],const PetscReal[],const PetscReal[],const PetscReal[],const PetscReal[],const PetscReal[],const PetscReal[],const PetscReal[],PetscInt,const PetscReal[],const PetscReal[]);
extern PetscErrorCode TSARKIMEXFinalizePackage(void);
extern PetscErrorCode TSARKIMEXInitializePackage(const char path[]);
extern PetscErrorCode TSARKIMEXRegisterDestroy(void);
extern PetscErrorCode TSARKIMEXRegisterAll(void);

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


extern PetscErrorCode TSRosWGetType(TS ts,const TSRosWType*);
extern PetscErrorCode TSRosWSetType(TS ts,const TSRosWType);
extern PetscErrorCode TSRosWSetRecomputeJacobian(TS,PetscBool);
extern PetscErrorCode TSRosWRegister(const TSRosWType,PetscInt,PetscInt,const PetscReal[],const PetscReal[],const PetscReal[],const PetscReal[],PetscInt,const PetscReal[]);
extern PetscErrorCode TSRosWFinalizePackage(void);
extern PetscErrorCode TSRosWInitializePackage(const char path[]);
extern PetscErrorCode TSRosWRegisterDestroy(void);
extern PetscErrorCode TSRosWRegisterAll(void);

/*
       PETSc interface to Sundials
*/
#ifdef PETSC_HAVE_SUNDIALS
typedef enum { SUNDIALS_ADAMS=1,SUNDIALS_BDF=2} TSSundialsLmmType;
extern const char *TSSundialsLmmTypes[];
typedef enum { SUNDIALS_MODIFIED_GS = 1,SUNDIALS_CLASSICAL_GS = 2 } TSSundialsGramSchmidtType;
extern const char *TSSundialsGramSchmidtTypes[];
extern PetscErrorCode   TSSundialsSetType(TS,TSSundialsLmmType);
extern PetscErrorCode   TSSundialsGetPC(TS,PC*);
extern PetscErrorCode   TSSundialsSetTolerance(TS,PetscReal,PetscReal);
extern PetscErrorCode   TSSundialsSetMinTimeStep(TS,PetscReal);
extern PetscErrorCode   TSSundialsSetMaxTimeStep(TS,PetscReal);
extern PetscErrorCode   TSSundialsGetIterations(TS,PetscInt *,PetscInt *);
extern PetscErrorCode   TSSundialsSetGramSchmidtType(TS,TSSundialsGramSchmidtType);
extern PetscErrorCode   TSSundialsSetGMRESRestart(TS,PetscInt);
extern PetscErrorCode   TSSundialsSetLinearTolerance(TS,PetscReal);
extern PetscErrorCode   TSSundialsMonitorInternalSteps(TS,PetscBool );
extern PetscErrorCode   TSSundialsGetParameters(TS,PetscInt *,long*[],double*[]);
extern PetscErrorCode   TSSundialsSetMaxl(TS,PetscInt);
#endif

extern PetscErrorCode   TSRKSetTolerance(TS,PetscReal);

extern PetscErrorCode  TSThetaSetTheta(TS,PetscReal);
extern PetscErrorCode  TSThetaGetTheta(TS,PetscReal*);
extern PetscErrorCode  TSThetaGetEndpoint(TS,PetscBool*);
extern PetscErrorCode  TSThetaSetEndpoint(TS,PetscBool);

extern PetscErrorCode  TSAlphaSetAdapt(TS,PetscErrorCode(*)(TS,PetscReal,Vec,Vec,PetscReal*,PetscBool*,void*),void*);
extern PetscErrorCode  TSAlphaAdaptDefault(TS,PetscReal,Vec,Vec,PetscReal*,PetscBool*,void*);
extern PetscErrorCode  TSAlphaSetRadius(TS,PetscReal);
extern PetscErrorCode  TSAlphaSetParams(TS,PetscReal,PetscReal,PetscReal);
extern PetscErrorCode  TSAlphaGetParams(TS,PetscReal*,PetscReal*,PetscReal*);

extern PetscErrorCode  TSSetDM(TS,DM);
extern PetscErrorCode  TSGetDM(TS,DM*);

extern PetscErrorCode  SNESTSFormFunction(SNES,Vec,Vec,void*);
extern PetscErrorCode  SNESTSFormJacobian(SNES,Vec,Mat*,Mat*,MatStructure*,void*);

PETSC_EXTERN_CXX_END
#endif
