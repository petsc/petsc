/* $Id: petscts.h,v 1.42 2000/05/10 16:44:25 bsmith Exp bsmith $ */
/*
   User interface for the timestepping package. This is package
   is for use in solving time-dependent PDEs.
*/
#if !defined(__PETSCTS_H)
#define __PETSCTS_H
#include "petscsnes.h"

typedef struct _p_TS* TS;
#define TS_COOKIE PETSC_COOKIE+18

#define TS_EULER           "euler"
#define TS_BEULER          "beuler"
#define TS_PSEUDO          "pseudo"
#define TS_CRANK_NICHOLSON "crank-nicholson"
#define TS_PVODE           "pvode"

typedef char *TSType;

typedef enum {TS_LINEAR,TS_NONLINEAR} TSProblemType;

EXTERN int TSCreate(MPI_Comm,TSProblemType,TS*);
EXTERN int TSSetType(TS,TSType);
EXTERN int TSGetProblemType(TS,TSProblemType*);
EXTERN int TSDestroy(TS);

EXTERN int TSSetMonitor(TS,int(*)(TS,int,double,Vec,void*),void *,int (*)(void*));
EXTERN int TSClearMonitor(TS);
EXTERN int TSGetType(TS,TSType*);

EXTERN int TSSetOptionsPrefix(TS,char *);
EXTERN int TSAppendOptionsPrefix(TS,char *);
EXTERN int TSGetOptionsPrefix(TS,char **);
EXTERN int TSSetFromOptions(TS);
EXTERN int TSSetUp(TS);

EXTERN int TSSetSolution(TS,Vec);
EXTERN int TSGetSolution(TS,Vec*);

EXTERN int TSSetDuration(TS,int,double);

EXTERN int TSDefaultMonitor(TS,int,double,Vec,void*);
EXTERN int TSStep(TS,int *,double*);

EXTERN int TSSetInitialTimeStep(TS,double,double);
EXTERN int TSGetTimeStep(TS,double*);
EXTERN int TSGetTime(TS,double*);
EXTERN int TSGetTimeStepNumber(TS,int*);
EXTERN int TSSetTimeStep(TS,double);

EXTERN int TSSetRHSFunction(TS,int (*)(TS,double,Vec,Vec,void*),void*);
EXTERN int TSSetRHSMatrix(TS,Mat,Mat,int (*)(TS,double,Mat*,Mat*,MatStructure*,void*),void*);
EXTERN int TSSetRHSJacobian(TS,Mat,Mat,int(*)(TS,double,Vec,Mat*,Mat*,MatStructure*,void*),void*);
EXTERN int TSSetRHSBoundaryConditions(TS,int (*)(TS,double,Vec,void*),void*);

EXTERN int TSDefaultComputeJacobianColor(TS,double,Vec,Mat*,Mat*,MatStructure*,void*);
EXTERN int TSDefaultComputeJacobian(TS,double,Vec,Mat*,Mat*,MatStructure*,void*);

EXTERN int TSGetRHSMatrix(TS,Mat*,Mat*,void**);
EXTERN int TSGetRHSJacobian(TS,Mat*,Mat*,void**);

EXTERN int TSPseudoSetTimeStep(TS,int(*)(TS,double*,void*),void*);
EXTERN int TSPseudoDefaultTimeStep(TS,double*,void*);
EXTERN int TSPseudoComputeTimeStep(TS,double *);

EXTERN int TSPseudoSetVerifyTimeStep(TS,int(*)(TS,Vec,void*,double*,int*),void*);
EXTERN int TSPseudoDefaultVerifyTimeStep(TS,Vec,void*,double*,int*);
EXTERN int TSPseudoVerifyTimeStep(TS,Vec,double*,int*);
EXTERN int TSPseudoSetTimeStepIncrement(TS,double);
EXTERN int TSPseudoIncrementDtFromInitialDt(TS);

EXTERN int TSComputeRHSFunction(TS,double,Vec,Vec);
EXTERN int TSComputeRHSBoundaryConditions(TS,double,Vec);
EXTERN int TSComputeRHSJacobian(TS,double,Vec,Mat*,Mat*,MatStructure*);

extern FList      TSList;
EXTERN int        TSRegisterAll(char*);
EXTERN int        TSRegisterDestroy(void);
extern PetscTruth TSRegisterAllCalled;

EXTERN int TSRegister(char*,char*,char*,int(*)(TS));
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define TSRegisterDynamic(a,b,c,d) TSRegister(a,b,c,0)
#else
#define TSRegisterDynamic(a,b,c,d) TSRegister(a,b,c,d)
#endif

EXTERN int TSGetSNES(TS,SNES*);
EXTERN int TSGetSLES(TS,SLES*);

EXTERN int TSView(TS,Viewer);

EXTERN int TSSetApplicationContext(TS,void *);
EXTERN int TSGetApplicationContext(TS,void **);

EXTERN int TSLGMonitorCreate(char *,char *,int,int,int,int,DrawLG *);
EXTERN int TSLGMonitor(TS,int,double,Vec,void *);
EXTERN int TSLGMonitorDestroy(DrawLG);

/*
       PETSc interface to PVode
*/
#define PVODE_UNMODIFIED_GS PVODE_CLASSICAL_GS
typedef enum { PVODE_ADAMS,PVODE_BDF } TSPVodeType;
typedef enum { PVODE_MODIFIED_GS = 0,PVODE_CLASSICAL_GS = 1 } TSPVodeGramSchmidtType;
EXTERN int TSPVodeSetType(TS,TSPVodeType);
EXTERN int TSPVodeGetPC(TS,PC*);
EXTERN int TSPVodeSetTolerance(TS,double,double);
EXTERN int TSPVodeGetIterations(TS,int *,int *);
EXTERN int TSPVodeSetGramSchmidtType(TS,TSPVodeGramSchmidtType);
EXTERN int TSPVodeSetGMRESRestart(TS,int);
EXTERN int TSPVodeSetLinearTolerance(TS,double);
EXTERN int TSPVodeSetExactFinalTime(TS,PetscTruth);

#endif





