/* $Id: ts.h,v 1.38 2000/01/11 21:04:04 bsmith Exp bsmith $ */
/*
   User interface for the timestepping package. This is package
   is for use in solving time-dependent PDEs.
*/
#if !defined(__TS_H)
#define __TS_H
#include "snes.h"

typedef struct _p_TS* TS;
#define TS_COOKIE PETSC_COOKIE+18

#define TS_EULER           "euler"
#define TS_BEULER          "beuler"
#define TS_PSEUDO          "pseudo"
#define TS_CRANK_NICHOLSON "crank-nicholson"
#define TS_PVODE           "pvode"

typedef char *TSType;

typedef enum {TS_LINEAR,TS_NONLINEAR} TSProblemType;

extern int TSCreate(MPI_Comm,TSProblemType,TS*);
extern int TSSetType(TS,TSType);
extern int TSGetProblemType(TS,TSProblemType*);
extern int TSDestroy(TS);

extern int TSSetMonitor(TS,int(*)(TS,int,double,Vec,void*),void *,int (*)(void*));
extern int TSClearMonitor(TS);
extern int TSGetType(TS,TSType*);

extern int TSSetOptionsPrefix(TS,char *);
extern int TSAppendOptionsPrefix(TS,char *);
extern int TSGetOptionsPrefix(TS,char **);
extern int TSSetFromOptions(TS);
extern int TSSetTypeFromOptions(TS);
extern int TSSetUp(TS);

extern int TSSetSolution(TS,Vec);
extern int TSGetSolution(TS,Vec*);

extern int TSSetDuration(TS,int,double);
extern int TSPrintHelp(TS);

extern int TSDefaultMonitor(TS,int,double,Vec,void*);
extern int TSStep(TS,int *,double*);

extern int TSSetInitialTimeStep(TS,double,double);
extern int TSGetTimeStep(TS,double*);
extern int TSGetTime(TS,double*);
extern int TSGetTimeStepNumber(TS,int*);
extern int TSSetTimeStep(TS,double);

extern int TSSetRHSFunction(TS,int (*)(TS,double,Vec,Vec,void*),void*);
extern int TSSetRHSMatrix(TS,Mat,Mat,int (*)(TS,double,Mat*,Mat*,MatStructure*,void*),void*);
extern int TSSetRHSJacobian(TS,Mat,Mat,int(*)(TS,double,Vec,Mat*,Mat*,MatStructure*,void*),void*);
extern int TSSetRHSBoundaryConditions(TS,int (*)(TS,double,Vec,void*),void*);

extern int TSDefaultComputeJacobianColor(TS,double,Vec,Mat*,Mat*,MatStructure*,void*);
extern int TSDefaultComputeJacobian(TS,double,Vec,Mat*,Mat*,MatStructure*,void*);

extern int TSGetRHSMatrix(TS,Mat*,Mat*,void**);
extern int TSGetRHSJacobian(TS,Mat*,Mat*,void**);

extern int TSPseudoSetTimeStep(TS,int(*)(TS,double*,void*),void*);
extern int TSPseudoDefaultTimeStep(TS,double*,void*);
extern int TSPseudoComputeTimeStep(TS,double *);

extern int TSPseudoSetVerifyTimeStep(TS,int(*)(TS,Vec,void*,double*,int*),void*);
extern int TSPseudoDefaultVerifyTimeStep(TS,Vec,void*,double*,int*);
extern int TSPseudoVerifyTimeStep(TS,Vec,double*,int*);
extern int TSPseudoSetTimeStepIncrement(TS,double);
extern int TSPseudoIncrementDtFromInitialDt(TS);

extern int TSComputeRHSFunction(TS,double,Vec,Vec);
extern int TSComputeRHSBoundaryConditions(TS,double,Vec);

extern FList      TSList;
extern int        TSRegisterAll(char*);
extern int        TSRegisterDestroy(void);
extern PetscTruth TSRegisterAllCalled;

extern int TSRegister(char*,char*,char*,int(*)(TS));
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define TSRegisterDynamic(a,b,c,d) TSRegister(a,b,c,0)
#else
#define TSRegisterDynamic(a,b,c,d) TSRegister(a,b,c,d)
#endif

extern int TSGetSNES(TS,SNES*);
extern int TSGetSLES(TS,SLES*);

extern int TSView(TS,Viewer);

extern int TSSetApplicationContext(TS,void *);
extern int TSGetApplicationContext(TS,void **);

extern int TSLGMonitorCreate(char *,char *,int,int,int,int,DrawLG *);
extern int TSLGMonitor(TS,int,double,Vec,void *);
extern int TSLGMonitorDestroy(DrawLG);

/*
       PETSc interface to PVode
*/
#define PVODE_UNMODIFIED_GS PVODE_CLASSICAL_GS
typedef enum { PVODE_ADAMS,PVODE_BDF } TSPVodeType;
typedef enum { PVODE_MODIFIED_GS = 0,PVODE_CLASSICAL_GS = 1 } TSPVodeGramSchmidtType;
extern int TSPVodeSetType(TS,TSPVodeType);
extern int TSPVodeGetPC(TS,PC*);
extern int TSPVodeSetTolerance(TS,double,double);
extern int TSPVodeGetIterations(TS,int *,int *);
extern int TSPVodeSetGramSchmidtType(TS,TSPVodeGramSchmidtType);
extern int TSPVodeSetGMRESRestart(TS,int);
extern int TSPVodeSetLinearTolerance(TS,double);
extern int TSPVodeSetExactFinalTime(TS,PetscTruth);

#endif





