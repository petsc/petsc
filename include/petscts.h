/* $Id: petscts.h,v 1.47 2001/08/06 15:42:42 bsmith Exp $ */
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

.seealso:  TSCreate(), TSSetType(), TSType, SNES, SLES, KSP, PC
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
typedef char *TSType;

typedef char *TSSerializeType;

/*E
    TSProblemType - Determines the type of problem this TS object is to be used to solve

   Level: beginner

.seealso: TSCreate()
E*/
typedef enum {TS_LINEAR,TS_NONLINEAR} TSProblemType;

/* Logging support */
extern int TS_COOKIE;
extern int TS_Step, TS_PseudoComputeTimeStep, TS_FunctionEval, TS_JacobianEval;

EXTERN int TSInitializePackage(const char[]);

EXTERN int TSCreate(MPI_Comm,TS*);
EXTERN int TSSerialize(MPI_Comm, TS *, PetscViewer, PetscTruth);
EXTERN int TSDestroy(TS);

EXTERN int TSSetProblemType(TS,TSProblemType);
EXTERN int TSGetProblemType(TS,TSProblemType*);
EXTERN int TSSetMonitor(TS,int(*)(TS,int,PetscReal,Vec,void*),void *,int (*)(void*));
EXTERN int TSClearMonitor(TS);

EXTERN int TSSetOptionsPrefix(TS,const char[]);
EXTERN int TSAppendOptionsPrefix(TS,const char[]);
EXTERN int TSGetOptionsPrefix(TS,char *[]);
EXTERN int TSSetFromOptions(TS);
EXTERN int TSSetUp(TS);

EXTERN int TSSetSolution(TS,Vec);
EXTERN int TSGetSolution(TS,Vec*);

EXTERN int TSSetDuration(TS,int,PetscReal);
EXTERN int TSGetDuration(TS,int*,PetscReal*);

EXTERN int TSDefaultMonitor(TS,int,PetscReal,Vec,void*);
EXTERN int TSVecViewMonitor(TS,int,PetscReal,Vec,void*);
EXTERN int TSStep(TS,int *,PetscReal*);

EXTERN int TSSetInitialTimeStep(TS,PetscReal,PetscReal);
EXTERN int TSGetTimeStep(TS,PetscReal*);
EXTERN int TSGetTime(TS,PetscReal*);
EXTERN int TSGetTimeStepNumber(TS,int*);
EXTERN int TSSetTimeStep(TS,PetscReal);

EXTERN int TSSetRHSFunction(TS,int (*)(TS,PetscReal,Vec,Vec,void*),void*);
EXTERN int TSSetRHSMatrix(TS,Mat,Mat,int (*)(TS,PetscReal,Mat*,Mat*,MatStructure*,void*),void*);
EXTERN int TSSetRHSJacobian(TS,Mat,Mat,int(*)(TS,PetscReal,Vec,Mat*,Mat*,MatStructure*,void*),void*);
EXTERN int TSSetRHSBoundaryConditions(TS,int (*)(TS,PetscReal,Vec,void*),void*);

EXTERN int TSDefaultComputeJacobianColor(TS,PetscReal,Vec,Mat*,Mat*,MatStructure*,void*);
EXTERN int TSDefaultComputeJacobian(TS,PetscReal,Vec,Mat*,Mat*,MatStructure*,void*);

EXTERN int TSGetRHSMatrix(TS,Mat*,Mat*,void**);
EXTERN int TSGetRHSJacobian(TS,Mat*,Mat*,void**);

extern int TSSetRhsBC(TS, int (*)(TS, Vec, void *));
extern int TSSetSystemMatrixBC(TS, int (*)(TS, Mat, Mat, void *));
extern int TSSetSolutionBC(TS, int (*)(TS, Vec, void *));
extern int TSSetPreStep(TS, int (*)(TS));
extern int TSSetUpdate(TS, int (*)(TS, PetscReal, PetscReal *));
extern int TSSetPostStep(TS, int (*)(TS));
extern int TSDefaultRhsBC(TS, Vec, void *);
extern int TSDefaultSystemMatrixBC(TS, Mat, Mat, void *);
extern int TSDefaultSolutionBC(TS, Vec, void *);
extern int TSDefaultPreStep(TS);
extern int TSDefaultUpdate(TS, PetscReal, PetscReal *);
extern int TSDefaultPostStep(TS);
extern int TSSetIdentity(TS, int (*)(TS, double, Mat *, void *));

EXTERN int TSPseudoSetTimeStep(TS,int(*)(TS,PetscReal*,void*),void*);
EXTERN int TSPseudoDefaultTimeStep(TS,PetscReal*,void*);
EXTERN int TSPseudoComputeTimeStep(TS,PetscReal *);

EXTERN int TSPseudoSetVerifyTimeStep(TS,int(*)(TS,Vec,void*,PetscReal*,int*),void*);
EXTERN int TSPseudoDefaultVerifyTimeStep(TS,Vec,void*,PetscReal*,int*);
EXTERN int TSPseudoVerifyTimeStep(TS,Vec,PetscReal*,int*);
EXTERN int TSPseudoSetTimeStepIncrement(TS,PetscReal);
EXTERN int TSPseudoIncrementDtFromInitialDt(TS);

EXTERN int TSComputeRHSFunction(TS,PetscReal,Vec,Vec);
EXTERN int TSComputeRHSBoundaryConditions(TS,PetscReal,Vec);
EXTERN int TSComputeRHSJacobian(TS,PetscReal,Vec,Mat*,Mat*,MatStructure*);

/* Dynamic creation and loading functions */
extern PetscFList TSList;
extern PetscTruth TSRegisterAllCalled;
EXTERN int TSGetType(TS,TSType*);
EXTERN int TSSetType(TS,TSType);
EXTERN int TSRegister(const char[], const char[], const char[], int (*)(TS));
EXTERN int TSRegisterAll(const char[]);
EXTERN int TSRegisterDestroy(void);

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

extern PetscFList TSSerializeList;
extern PetscTruth TSSerializeRegisterAllCalled;
EXTERN int TSSetSerializeType(TS, TSSerializeType);
EXTERN int TSGetSerializeType(TS, TSSerializeType *);
EXTERN int TSSerializeRegister(const char [], const char [], const char [], int (*)(MPI_Comm, TS *, PetscViewer, PetscTruth));
EXTERN int TSSerializeRegisterAll(const char []);
EXTERN int TSSerializeRegisterDestroy(void);
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define TSSerializeRegisterDynamic(a,b,c,d) TSSerializeRegister(a,b,c,0)
#else
#define TSSerializeRegisterDynamic(a,b,c,d) TSSerializeRegister(a,b,c,d)
#endif

EXTERN int TSGetSNES(TS,SNES*);
EXTERN int TSGetSLES(TS,SLES*);

EXTERN int TSView(TS,PetscViewer);
EXTERN int TSViewFromOptions(TS,const char[]);

EXTERN int TSSetApplicationContext(TS,void *);
EXTERN int TSGetApplicationContext(TS,void **);

EXTERN int TSLGMonitorCreate(const char[],const char[],int,int,int,int,PetscDrawLG *);
EXTERN int TSLGMonitor(TS,int,PetscReal,Vec,void *);
EXTERN int TSLGMonitorDestroy(PetscDrawLG);

/*
       PETSc interface to PVode
*/
#define PVODE_UNMODIFIED_GS PVODE_CLASSICAL_GS
typedef enum { PVODE_ADAMS,PVODE_BDF } TSPVodeType;
typedef enum { PVODE_MODIFIED_GS = 0,PVODE_CLASSICAL_GS = 1 } TSPVodeGramSchmidtType;
EXTERN int TSPVodeSetType(TS,TSPVodeType);
EXTERN int TSPVodeGetPC(TS,PC*);
EXTERN int TSPVodeSetTolerance(TS,PetscReal,PetscReal);
EXTERN int TSPVodeGetIterations(TS,int *,int *);
EXTERN int TSPVodeSetGramSchmidtType(TS,TSPVodeGramSchmidtType);
EXTERN int TSPVodeSetGMRESRestart(TS,int);
EXTERN int TSPVodeSetLinearTolerance(TS,PetscReal);
EXTERN int TSPVodeSetExactFinalTime(TS,PetscTruth);
EXTERN int TSPVodeGetParameters(TS,int *,long int*[],double*[]);

EXTERN int TSRKSetTolerance(TS,PetscReal);

PETSC_EXTERN_CXX_END
#endif
