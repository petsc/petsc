/* $Id: snes.h,v 1.20 1995/06/14 02:10:31 bsmith Exp curfman $ */

#if !defined(__SNES_PACKAGE)
#define __SNES_PACKAGE
#include "sles.h"

typedef struct _SNES* SNES;
#define SNES_COOKIE PETSC_COOKIE+13

typedef enum { SNES_NLS,
               SNES_NTR,
               SNES_NTR_DOG_LEG,
               SNES_NTR2_LIN,
               SUMS_NLS,
               SUMS_NTR,
               SNES_NTEST }
  SNESMethod;

typedef enum { SNES_T, SUMS_T } SNESType;

extern int SNESCreate(MPI_Comm,SNES*);
extern int SNESSetMethod(SNES,SNESMethod);
extern int SNESSetMonitor(SNES, int (*)(SNES,int,double,void*),void *);
extern int SNESSetSolution(SNES,Vec,int (*)(SNES,Vec,void*),void *);
extern int SNESSetFunction(SNES, Vec, int (*)(SNES,Vec,Vec,void*),void *,int);
extern int SNESSetJacobian(SNES,Mat,Mat,int(*)(SNES,Vec,Mat*,Mat*,MatStructure*,void*),void *);
extern int SNESDestroy(SNES);
extern int SNESSetUp(SNES);
extern int SNESSolve(SNES,int*);
extern int SNESRegister(int, char*, int (*)(SNES));
extern int SNESRegisterAll();
extern int SNESGetSLES(SNES,SLES*);
extern int SNESNoLineSearch(SNES,Vec,Vec,Vec,Vec,Vec,double,double*,double*,int*);
extern int SNESCubicLineSearch(SNES,Vec,Vec,Vec,Vec,Vec,double,double*,double*,int*);
extern int SNESQuadraticLineSearch(SNES,Vec,Vec,Vec,Vec,Vec,double,double*,double*,int*);
extern int SNESSetLineSearchRoutine(SNES,int(*)(SNES,Vec,Vec,Vec,Vec,Vec,double,double*,double*,int*));

extern int SNESGetSolution(SNES,Vec*);
extern int SNESGetSolutionUpdate(SNES,Vec*);
extern int SNESGetFunction(SNES,Vec*);

extern int SNESPrintHelp(SNES);
extern int SNESSetFromOptions(SNES);
extern int SNESGetMethodName(SNESMethod,char **);
extern int SNESDefaultMonitor(SNES,int,double,void *);
extern int SNESDefaultSMonitor(SNES,int,double,void *);
extern int SNESDefaultConverged(SNES,double,double,double,void*);

extern int SNESSetSolutionTolerance(SNES,double);
extern int SNESSetAbsoluteTolerance(SNES,double);
extern int SNESSetRelativeTolerance(SNES,double);
extern int SNESSetTruncationTolerance(SNES,double);
extern int SNESSetMaxIterations(SNES,int);
extern int SNESSetMaxFunctionEvaluations(SNES,int);
extern int SNESGetIterationNumber(SNES,int*);

#if defined(__DRAW_PACKAGE)
#define SNESLGMonitorCreate  KSPLGMonitorCreate
#define SNESLGMonitorDestroy KSPLGMonitorDestroy
#define SNESLGMonitor        ((int (*)(SNES,int,double,void*))KSPLGMonitor)
#endif

extern int SNESComputeInitialGuess(SNES,Vec);

extern int SNESDefaultComputeJacobian(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
extern int SNESDefaultMatrixFreeComputeJacobian(SNES,Vec,Mat*,Mat*,
                                                MatStructure*,void*);
extern int SNESDefaultMatrixFreeMatCreate(SNES,Vec x,Mat*);

extern int SNESComputeFunction(SNES,Vec,Vec);
extern int SNESComputeJacobian(SNES,Vec,Mat*,Mat*,MatStructure*,void *);
#endif

