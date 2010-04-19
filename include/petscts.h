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

/*E
    TSType - String with the name of a PETSc TS method or the creation function
       with an optional dynamic library name, for example
       http://www.mcs.anl.gov/petsc/lib.a:mytscreate()

   Level: beginner

.seealso: TSSetType(), TS
E*/
#define TSType char*
#define TSEULER           "euler"
#define TSBEULER          "beuler"
#define TSPSEUDO          "pseudo"
#define TSCRANK_NICHOLSON "crank-nicholson"
#define TSSUNDIALS        "sundials"
#define TSRUNGE_KUTTA     "runge-kutta"
#define TSPYTHON          "python"
#define TSTHETA           "theta"
#define TSGL              "gl"
#define TSSSP             "ssp"

/*E
    TSProblemType - Determines the type of problem this TS object is to be used to solve

   Level: beginner

.seealso: TSCreate()
E*/
typedef enum {TS_LINEAR,TS_NONLINEAR} TSProblemType;

/* Logging support */
extern PetscCookie PETSCTS_DLLEXPORT TS_COOKIE;

EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSInitializePackage(const char[]);

EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSCreate(MPI_Comm,TS*);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSDestroy(TS);

EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSSetProblemType(TS,TSProblemType);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSGetProblemType(TS,TSProblemType*);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSMonitorSet(TS,PetscErrorCode(*)(TS,PetscInt,PetscReal,Vec,void*),void *,PetscErrorCode (*)(void*));
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSMonitorCancel(TS);

EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSSetOptionsPrefix(TS,const char[]);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSAppendOptionsPrefix(TS,const char[]);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSGetOptionsPrefix(TS,const char *[]);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSSetFromOptions(TS);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSSetUp(TS);

EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSSetSolution(TS,Vec);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSGetSolution(TS,Vec*);

EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSSetDuration(TS,PetscInt,PetscReal);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSGetDuration(TS,PetscInt*,PetscReal*);

EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSMonitorDefault(TS,PetscInt,PetscReal,Vec,void*);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSMonitorSolution(TS,PetscInt,PetscReal,Vec,void*);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSStep(TS,PetscInt *,PetscReal*);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSSolve(TS,Vec);


EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSSetInitialTimeStep(TS,PetscReal,PetscReal);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSGetTimeStep(TS,PetscReal*);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSGetTime(TS,PetscReal*);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSSetTime(TS,PetscReal);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSGetTimeStepNumber(TS,PetscInt*);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSSetTimeStep(TS,PetscReal);

EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSSetRHSFunction(TS,PetscErrorCode (*)(TS,PetscReal,Vec,Vec,void*),void*);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSSetMatrices(TS,Mat,PetscErrorCode (*)(TS,PetscReal,Mat*,Mat*,MatStructure*,void*),Mat,PetscErrorCode (*)(TS,PetscReal,Mat*,Mat*,MatStructure*,void*),MatStructure,void*);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSGetMatrices(TS,Mat*,Mat*,void**);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSSetRHSJacobian(TS,Mat,Mat,PetscErrorCode (*)(TS,PetscReal,Vec,Mat*,Mat*,MatStructure*,void*),void*);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSGetRHSJacobian(TS,Mat*,Mat*,void**);

typedef PetscErrorCode (*TSIFunction)(TS,PetscReal,Vec,Vec,Vec,void*);
typedef PetscErrorCode (*TSIJacobian)(TS,PetscReal,Vec,Vec,PetscReal,Mat*,Mat*,MatStructure*,void*);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSSetIFunction(TS,TSIFunction,void*);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSSetIJacobian(TS,Mat,Mat,TSIJacobian,void*);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSGetIJacobian(TS,Mat*,Mat*,TSIJacobian*,void**);

EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSDefaultComputeJacobianColor(TS,PetscReal,Vec,Mat*,Mat*,MatStructure*,void*);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSDefaultComputeJacobian(TS,PetscReal,Vec,Mat*,Mat*,MatStructure*,void*);

EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSSetPreStep(TS, PetscErrorCode (*)(TS));
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSSetPostStep(TS, PetscErrorCode (*)(TS));
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSDefaultPreStep(TS);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSDefaultPostStep(TS);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSPreStep(TS);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSPostStep(TS);

EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSPseudoSetTimeStep(TS,PetscErrorCode(*)(TS,PetscReal*,void*),void*);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSPseudoDefaultTimeStep(TS,PetscReal*,void*);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSPseudoComputeTimeStep(TS,PetscReal *);

EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSPseudoSetVerifyTimeStep(TS,PetscErrorCode(*)(TS,Vec,void*,PetscReal*,PetscTruth*),void*);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSPseudoDefaultVerifyTimeStep(TS,Vec,void*,PetscReal*,PetscTruth*);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSPseudoVerifyTimeStep(TS,Vec,PetscReal*,PetscTruth*);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSPseudoSetTimeStepIncrement(TS,PetscReal);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSPseudoIncrementDtFromInitialDt(TS);

EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSPythonSetType(TS,const char[]);

EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSComputeRHSFunction(TS,PetscReal,Vec,Vec);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSComputeRHSJacobian(TS,PetscReal,Vec,Mat*,Mat*,MatStructure*);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSComputeIFunction(TS,PetscReal,Vec,Vec,Vec);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSComputeIJacobian(TS,PetscReal,Vec,Vec,PetscReal,Mat*,Mat*,MatStructure*);

/* Dynamic creation and loading functions */
extern PetscFList TSList;
extern PetscTruth TSRegisterAllCalled;
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSGetType(TS,const TSType*);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSSetType(TS,const TSType);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSRegister(const char[], const char[], const char[], PetscErrorCode (*)(TS));
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSRegisterAll(const char[]);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSRegisterDestroy(void);

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

EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSGetSNES(TS,SNES*);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSGetKSP(TS,KSP*);

EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSView(TS,PetscViewer);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSViewFromOptions(TS,const char[]);

EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSSetApplicationContext(TS,void *);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSGetApplicationContext(TS,void **);

EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSMonitorLGCreate(const char[],const char[],int,int,int,int,PetscDrawLG *);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSMonitorLG(TS,PetscInt,PetscReal,Vec,void *);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSMonitorLGDestroy(PetscDrawLG);

/*S
   TSGLAdapt - Abstract object that manages time-step adaptivity

   Level: beginner

.seealso: TSGL, TSGLAdaptCreate(), TSGLAdaptType
S*/
typedef struct _p_TSGLAdapt *TSGLAdapt;

/*E
    TSGLAdaptType - String with the name of TSGLAdapt scheme or the creation function
       with an optional dynamic library name, for example
       http://www.mcs.anl.gov/petsc/lib.a:mytsgladaptcreate()

   Level: beginner

.seealso: TSGLAdaptSetType(), TS
E*/
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

EXTERN PetscErrorCode PETSCTS_DLLEXPORT TSGLAdaptRegister(const char[],const char[],const char[],PetscErrorCode (*)(TSGLAdapt));
EXTERN PetscErrorCode PETSCTS_DLLEXPORT TSGLAdaptRegisterAll(const char[]);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT TSGLAdaptRegisterDestroy(void);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT TSGLAdaptInitializePackage(const char[]);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT TSGLAdaptFinalizePackage(void);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT TSGLAdaptCreate(MPI_Comm,TSGLAdapt*);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT TSGLAdaptSetType(TSGLAdapt,const TSGLAdaptType);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT TSGLAdaptSetOptionsPrefix(TSGLAdapt,const char[]);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT TSGLAdaptChoose(TSGLAdapt,PetscInt,const PetscInt[],const PetscReal[],const PetscReal[],PetscInt,PetscReal,PetscReal,PetscInt*,PetscReal*,PetscTruth*);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT TSGLAdaptView(TSGLAdapt,PetscViewer);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT TSGLAdaptSetFromOptions(TSGLAdapt);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT TSGLAdaptDestroy(TSGLAdapt);

/*E
    TSGLAcceptType - String with the name of TSGLAccept scheme or the function
       with an optional dynamic library name, for example
       http://www.mcs.anl.gov/petsc/lib.a:mytsglaccept()

   Level: beginner

.seealso: TSGLSetAcceptType(), TS
E*/
#define TSGLAcceptType  char*
#define TSGLACCEPT_ALWAYS "always"

typedef PetscErrorCode (*TSGLAcceptFunction)(TS,PetscReal,PetscReal,const PetscReal[],PetscTruth*);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT TSGLAcceptRegister(const char[],const char[],const char[],TSGLAcceptFunction);

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

/*E
  TSGLType - family of time integration method within the General Linear class

  Level: beginner

.seealso: TSGLSetType(), TSGLRegister()
E*/
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

EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSGLRegister(const char[],const char[],const char[],PetscErrorCode(*)(TS));
EXTERN PetscErrorCode PETSCTS_DLLEXPORT TSGLRegisterAll(const char[]);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT TSGLRegisterDestroy(void);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT TSGLInitializePackage(const char[]);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT TSGLFinalizePackage(void);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT TSGLSetType(TS,const TSGLType);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT TSGLGetAdapt(TS,TSGLAdapt*);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT TSGLSetAcceptType(TS,const TSGLAcceptType);

/*
       PETSc interface to Sundials
*/
#ifdef PETSC_HAVE_SUNDIALS
typedef enum { SUNDIALS_ADAMS=1,SUNDIALS_BDF=2} TSSundialsLmmType;
extern const char *TSSundialsLmmTypes[];
typedef enum { SUNDIALS_MODIFIED_GS = 1,SUNDIALS_CLASSICAL_GS = 2 } TSSundialsGramSchmidtType;
extern const char *TSSundialsGramSchmidtTypes[];
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSSundialsSetType(TS,TSSundialsLmmType);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSSundialsGetPC(TS,PC*);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSSundialsSetTolerance(TS,PetscReal,PetscReal);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSSundialsGetIterations(TS,PetscInt *,PetscInt *);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSSundialsSetGramSchmidtType(TS,TSSundialsGramSchmidtType);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSSundialsSetGMRESRestart(TS,PetscInt);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSSundialsSetLinearTolerance(TS,PetscReal);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSSundialsSetExactFinalTime(TS,PetscTruth);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSSundialsMonitorInternalSteps(TS,PetscTruth);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSSundialsGetParameters(TS,PetscInt *,long*[],double*[]);
#endif

EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSRKSetTolerance(TS,PetscReal);

EXTERN PetscErrorCode PETSCTS_DLLEXPORT TSThetaSetTheta(TS,PetscReal);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT TSThetaGetTheta(TS,PetscReal*);

PETSC_EXTERN_CXX_END
#endif
