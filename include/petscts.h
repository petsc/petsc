/* $Id: ts.h,v 1.4 1996/03/10 17:30:09 bsmith Exp bsmith $ */
/*
    User interface for the time-stepping package. This is package
  is for use in solving time dependent PDES.
*/
#if !defined(__TS_PACKAGE)
#define __TS_PACKAGE
#include "snes.h"

typedef struct _TS* TS;
#define TS_COOKIE PETSC_COOKIE+18

typedef enum { TS_EULER, TS_BEULER, TS_PSEUDO} TSType;
typedef enum { TS_LINEAR, TS_NONLINEAR} TSProblemType;

extern int TSCreate(MPI_Comm,TSProblemType,TS*);
extern int TSSetType(TS,TSType);
extern int TSDestroy(TS);

extern int TSSetMonitor(TS,int(*)(TS,int,Scalar,Vec,void*),void *);
extern int TSGetType(TS,TSType*,char**);

extern int TSSetFromOptions(TS);
extern int TSSetUp(TS);

extern int TSSetSolution(TS,Vec);
extern int TSGetSolution(TS,Vec*);

extern int TSSetDuration(TS,int,Scalar);
extern int TSPrintHelp(TS);

extern int TSDefaultMonitor(TS,int,Scalar,Vec,void*);
extern int TSStep(TS,int *,Scalar*);

extern int TSSetInitialTimeStep(TS,double,double);
extern int TSGetTimeStep(TS,double*);
extern int TSSetTimeStep(TS,double);

extern int TSSetRHSFunction(TS,int (*)(TS,Scalar,Vec,Vec,void*),void*);
extern int TSSetRHSMatrix(TS,Mat,Mat,int (*)(TS,double,Mat*,Mat*,MatStructure*,void*),void*);
extern int TSSetRHSJacobian(TS,Mat,Mat,int(*)(TS,double,Vec,Mat*,Mat*,MatStructure*,void*),void*);

extern int TSComputeRHSFunction(TS,double,Vec,Vec);

extern int TSRegisterAll();
extern int TSRegister(int,char*,int (*)(TS));

extern int TSGetSNES(TS,SNES*);
extern int TSGetSLES(TS,SLES*);

extern int TSView(TS,Viewer);
#endif

