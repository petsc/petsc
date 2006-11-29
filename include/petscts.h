/*
   User interface for the timestepping package. This is package
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
#define TS_EULER           "euler"
#define TS_BEULER          "beuler"
#define TS_PSEUDO          "pseudo"
#define TS_CRANK_NICHOLSON "crank-nicholson"
#define TS_SUNDIALS        "sundials"
#define TS_RUNGE_KUTTA     "runge-kutta"
#define TSType char*

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
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSSetRHSMatrix(TS,Mat,Mat,PetscErrorCode (*)(TS,PetscReal,Mat*,Mat*,MatStructure*,void*),void*);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSSetLHSMatrix(TS,Mat,Mat,PetscErrorCode (*)(TS,PetscReal,Mat*,Mat*,MatStructure*,void*),void*);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSSetRHSJacobian(TS,Mat,Mat,PetscErrorCode (*)(TS,PetscReal,Vec,Mat*,Mat*,MatStructure*,void*),void*);

EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSDefaultComputeJacobianColor(TS,PetscReal,Vec,Mat*,Mat*,MatStructure*,void*);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSDefaultComputeJacobian(TS,PetscReal,Vec,Mat*,Mat*,MatStructure*,void*);

EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSGetRHSMatrix(TS,Mat*,Mat*,void**);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSGetRHSJacobian(TS,Mat*,Mat*,void**);

EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSSetPreStep(TS, PetscErrorCode (*)(TS));
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSSetUpdate(TS, PetscErrorCode (*)(TS, PetscReal, PetscReal *));
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSSetPostStep(TS, PetscErrorCode (*)(TS));
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSDefaultPreStep(TS);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSDefaultUpdate(TS, PetscReal, PetscReal *);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSDefaultPostStep(TS);

EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSPseudoSetTimeStep(TS,PetscErrorCode(*)(TS,PetscReal*,void*),void*);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSPseudoDefaultTimeStep(TS,PetscReal*,void*);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSPseudoComputeTimeStep(TS,PetscReal *);

EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSPseudoSetVerifyTimeStep(TS,PetscErrorCode(*)(TS,Vec,void*,PetscReal*,PetscTruth*),void*);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSPseudoDefaultVerifyTimeStep(TS,Vec,void*,PetscReal*,PetscTruth*);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSPseudoVerifyTimeStep(TS,Vec,PetscReal*,PetscTruth*);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSPseudoSetTimeStepIncrement(TS,PetscReal);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSPseudoIncrementDtFromInitialDt(TS);

EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSComputeRHSFunction(TS,PetscReal,Vec,Vec);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSComputeRHSJacobian(TS,PetscReal,Vec,Mat*,Mat*,MatStructure*);

/* Dynamic creation and loading functions */
extern PetscFList TSList;
extern PetscTruth TSRegisterAllCalled;
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSGetType(TS,TSType*);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSSetType(TS,const TSType);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSRegister(const char[], const char[], const char[], PetscErrorCode (*)(TS));
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSRegisterAll(const char[]);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSRegisterDestroy(void);

/*MC
  TSRegisterDynamic - Adds a creation method to the TS package.

  Synopsis:
  PetscErrorCode TSRegisterDynamic(char *name, char *path, char *func_name, PetscErrorCode (*create_func)(TS))

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
    TSCreate(MPI_Comm, TS *);
    TSSetType(vec, "my_ts")
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

/*
       PETSc interface to Sundials
*/
#ifdef PETSC_HAVE_SUNDIALS
#define SUNDIALS_UNMODIFIED_GS SUNDIALS_CLASSICAL_GS
typedef enum { SUNDIALS_ADAMS,SUNDIALS_BDF } TSSundialsType;
typedef enum { SUNDIALS_MODIFIED_GS = 0,SUNDIALS_CLASSICAL_GS = 1 } TSSundialsGramSchmidtType;
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSSundialsSetType(TS,TSSundialsType);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSSundialsGetPC(TS,PC*);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSSundialsSetTolerance(TS,PetscReal,PetscReal);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSSundialsGetIterations(TS,PetscInt *,PetscInt *);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSSundialsSetGramSchmidtType(TS,TSSundialsGramSchmidtType);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSSundialsSetGMRESRestart(TS,PetscInt);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSSundialsSetLinearTolerance(TS,PetscReal);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSSundialsSetExactFinalTime(TS,PetscTruth);
EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSSundialsGetParameters(TS,PetscInt *,long int*[],double*[]);
#endif

EXTERN PetscErrorCode PETSCTS_DLLEXPORT  TSRKSetTolerance(TS,PetscReal);

PETSC_EXTERN_CXX_END
#endif
