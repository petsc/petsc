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
#define TS_PVODE           "pvode"
#define TS_RUNGE_KUTTA     "runge-kutta"
#define TSType char*

/*E
    TSProblemType - Determines the type of problem this TS object is to be used to solve

   Level: beginner

.seealso: TSCreate()
E*/
typedef enum {TS_LINEAR,TS_NONLINEAR} TSProblemType;

/* Logging support */
extern int TS_COOKIE;
extern int TS_Step, TS_PseudoComputeTimeStep, TS_FunctionEval, TS_JacobianEval;

EXTERN PetscErrorCode TSInitializePackage(const char[]);

EXTERN PetscErrorCode TSCreate(MPI_Comm,TS*);
EXTERN PetscErrorCode TSDestroy(TS);

EXTERN PetscErrorCode TSSetProblemType(TS,TSProblemType);
EXTERN PetscErrorCode TSGetProblemType(TS,TSProblemType*);
EXTERN PetscErrorCode TSSetMonitor(TS,int(*)(TS,int,PetscReal,Vec,void*),void *,int (*)(void*));
EXTERN PetscErrorCode TSClearMonitor(TS);

EXTERN PetscErrorCode TSSetOptionsPrefix(TS,const char[]);
EXTERN PetscErrorCode TSAppendOptionsPrefix(TS,const char[]);
EXTERN PetscErrorCode TSGetOptionsPrefix(TS,char *[]);
EXTERN PetscErrorCode TSSetFromOptions(TS);
EXTERN PetscErrorCode TSSetUp(TS);

EXTERN PetscErrorCode TSSetSolution(TS,Vec);
EXTERN PetscErrorCode TSGetSolution(TS,Vec*);

EXTERN PetscErrorCode TSSetDuration(TS,int,PetscReal);
EXTERN PetscErrorCode TSGetDuration(TS,int*,PetscReal*);

EXTERN PetscErrorCode TSDefaultMonitor(TS,int,PetscReal,Vec,void*);
EXTERN PetscErrorCode TSVecViewMonitor(TS,int,PetscReal,Vec,void*);
EXTERN PetscErrorCode TSStep(TS,int *,PetscReal*);

EXTERN PetscErrorCode TSSetInitialTimeStep(TS,PetscReal,PetscReal);
EXTERN PetscErrorCode TSGetTimeStep(TS,PetscReal*);
EXTERN PetscErrorCode TSGetTime(TS,PetscReal*);
EXTERN PetscErrorCode TSGetTimeStepNumber(TS,int*);
EXTERN PetscErrorCode TSSetTimeStep(TS,PetscReal);

EXTERN PetscErrorCode TSSetRHSFunction(TS,int (*)(TS,PetscReal,Vec,Vec,void*),void*);
EXTERN PetscErrorCode TSSetRHSMatrix(TS,Mat,Mat,int (*)(TS,PetscReal,Mat*,Mat*,MatStructure*,void*),void*);
EXTERN PetscErrorCode TSSetRHSJacobian(TS,Mat,Mat,int(*)(TS,PetscReal,Vec,Mat*,Mat*,MatStructure*,void*),void*);
EXTERN PetscErrorCode TSSetRHSBoundaryConditions(TS,int (*)(TS,PetscReal,Vec,void*),void*);

EXTERN PetscErrorCode TSDefaultComputeJacobianColor(TS,PetscReal,Vec,Mat*,Mat*,MatStructure*,void*);
EXTERN PetscErrorCode TSDefaultComputeJacobian(TS,PetscReal,Vec,Mat*,Mat*,MatStructure*,void*);

EXTERN PetscErrorCode TSGetRHSMatrix(TS,Mat*,Mat*,void**);
EXTERN PetscErrorCode TSGetRHSJacobian(TS,Mat*,Mat*,void**);

EXTERN PetscErrorCode TSSetRhsBC(TS, int (*)(TS, Vec, void *));
EXTERN PetscErrorCode TSSetSystemMatrixBC(TS, int (*)(TS, Mat, Mat, void *));
EXTERN PetscErrorCode TSSetSolutionBC(TS, int (*)(TS, Vec, void *));
EXTERN PetscErrorCode TSSetPreStep(TS, int (*)(TS));
EXTERN PetscErrorCode TSSetUpdate(TS, int (*)(TS, PetscReal, PetscReal *));
EXTERN PetscErrorCode TSSetPostStep(TS, int (*)(TS));
EXTERN PetscErrorCode TSDefaultRhsBC(TS, Vec, void *);
EXTERN PetscErrorCode TSDefaultSystemMatrixBC(TS, Mat, Mat, void *);
EXTERN PetscErrorCode TSDefaultSolutionBC(TS, Vec, void *);
EXTERN PetscErrorCode TSDefaultPreStep(TS);
EXTERN PetscErrorCode TSDefaultUpdate(TS, PetscReal, PetscReal *);
EXTERN PetscErrorCode TSDefaultPostStep(TS);
EXTERN PetscErrorCode TSSetIdentity(TS, int (*)(TS, double, Mat *, void *));

EXTERN PetscErrorCode TSPseudoSetTimeStep(TS,int(*)(TS,PetscReal*,void*),void*);
EXTERN PetscErrorCode TSPseudoDefaultTimeStep(TS,PetscReal*,void*);
EXTERN PetscErrorCode TSPseudoComputeTimeStep(TS,PetscReal *);

EXTERN PetscErrorCode TSPseudoSetVerifyTimeStep(TS,int(*)(TS,Vec,void*,PetscReal*,int*),void*);
EXTERN PetscErrorCode TSPseudoDefaultVerifyTimeStep(TS,Vec,void*,PetscReal*,int*);
EXTERN PetscErrorCode TSPseudoVerifyTimeStep(TS,Vec,PetscReal*,int*);
EXTERN PetscErrorCode TSPseudoSetTimeStepIncrement(TS,PetscReal);
EXTERN PetscErrorCode TSPseudoIncrementDtFromInitialDt(TS);

EXTERN PetscErrorCode TSComputeRHSFunction(TS,PetscReal,Vec,Vec);
EXTERN PetscErrorCode TSComputeRHSBoundaryConditions(TS,PetscReal,Vec);
EXTERN PetscErrorCode TSComputeRHSJacobian(TS,PetscReal,Vec,Mat*,Mat*,MatStructure*);

/* Dynamic creation and loading functions */
extern PetscFList TSList;
extern PetscTruth TSRegisterAllCalled;
EXTERN PetscErrorCode TSGetType(TS,TSType*);
EXTERN PetscErrorCode TSSetType(TS,const TSType);
EXTERN PetscErrorCode TSRegister(const char[], const char[], const char[], int (*)(TS));
EXTERN PetscErrorCode TSRegisterAll(const char[]);
EXTERN PetscErrorCode TSRegisterDestroy(void);

/*MC
  TSRegisterDynamic - Adds a creation method to the TS package.

  Synopsis:

  TSRegisterDynamic(char *name, char *path, char *func_name, int (*create_func)(TS))

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

  Notes: $PETSC_ARCH and $BOPT occuring in pathname will be replaced with appropriate values.
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

EXTERN PetscErrorCode TSGetSNES(TS,SNES*);
EXTERN PetscErrorCode TSGetKSP(TS,KSP*);

EXTERN PetscErrorCode TSView(TS,PetscViewer);
EXTERN PetscErrorCode TSViewFromOptions(TS,const char[]);

EXTERN PetscErrorCode TSSetApplicationContext(TS,void *);
EXTERN PetscErrorCode TSGetApplicationContext(TS,void **);

EXTERN PetscErrorCode TSLGMonitorCreate(const char[],const char[],int,int,int,int,PetscDrawLG *);
EXTERN PetscErrorCode TSLGMonitor(TS,int,PetscReal,Vec,void *);
EXTERN PetscErrorCode TSLGMonitorDestroy(PetscDrawLG);

/*
       PETSc interface to PVode
*/
#define PVODE_UNMODIFIED_GS PVODE_CLASSICAL_GS
typedef enum { PVODE_ADAMS,PVODE_BDF } TSPVodeType;
typedef enum { PVODE_MODIFIED_GS = 0,PVODE_CLASSICAL_GS = 1 } TSPVodeGramSchmidtType;
EXTERN PetscErrorCode TSPVodeSetType(TS,TSPVodeType);
EXTERN PetscErrorCode TSPVodeGetPC(TS,PC*);
EXTERN PetscErrorCode TSPVodeSetTolerance(TS,PetscReal,PetscReal);
EXTERN PetscErrorCode TSPVodeGetIterations(TS,int *,int *);
EXTERN PetscErrorCode TSPVodeSetGramSchmidtType(TS,TSPVodeGramSchmidtType);
EXTERN PetscErrorCode TSPVodeSetGMRESRestart(TS,int);
EXTERN PetscErrorCode TSPVodeSetLinearTolerance(TS,PetscReal);
EXTERN PetscErrorCode TSPVodeSetExactFinalTime(TS,PetscTruth);
EXTERN PetscErrorCode TSPVodeGetParameters(TS,int *,long int*[],double*[]);

EXTERN PetscErrorCode TSRKSetTolerance(TS,PetscReal);

PETSC_EXTERN_CXX_END
#endif
