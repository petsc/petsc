/* $Id: petscts.h,v 1.47 2001/08/06 15:42:42 bsmith Exp $ */
/*
   User interface for the timestepping package. This is package
   is for use in solving time-dependent PDEs.
*/
#if !defined(__PETSCTS_H)
#define __PETSCTS_H
#include "petscsnes.h"

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

EXTERN int TSInitializePackage(char *);

EXTERN int TSCreate(MPI_Comm,TS*);
EXTERN int TSSerialize(MPI_Comm, TS *, PetscViewer, PetscTruth);
EXTERN int TSDestroy(TS);

EXTERN int TSSetProblemType(TS,TSProblemType);
EXTERN int TSGetProblemType(TS,TSProblemType*);
EXTERN int TSSetMonitor(TS,int(*)(TS,int,PetscReal,Vec,void*),void *,int (*)(void*));
EXTERN int TSClearMonitor(TS);

EXTERN int TSSetOptionsPrefix(TS,char *);
EXTERN int TSAppendOptionsPrefix(TS,char *);
EXTERN int TSGetOptionsPrefix(TS,char **);
EXTERN int TSSetFromOptions(TS);
EXTERN int TSSetUp(TS);

EXTERN int TSSetSolution(TS,Vec);
EXTERN int TSGetSolution(TS,Vec*);

EXTERN int TSSetDuration(TS,int,PetscReal);

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
extern int TSSetUpdate(TS, int (*)(TS, double, double *));
extern int TSSetPostStep(TS, int (*)(TS));
extern int TSDefaultRhsBC(TS, Vec, void *);
extern int TSDefaultSystemMatrixBC(TS, Mat, Mat, void *);
extern int TSDefaultSolutionBC(TS, Vec, void *);
extern int TSDefaultPreStep(TS);
extern int TSDefaultUpdate(TS, double, double *);
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
EXTERN int TSViewFromOptions(TS,char *);

EXTERN int TSSetApplicationContext(TS,void *);
EXTERN int TSGetApplicationContext(TS,void **);

EXTERN int TSLGMonitorCreate(char *,char *,int,int,int,int,PetscDrawLG *);
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

#endif





