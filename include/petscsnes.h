/* $Id: snes.h,v 1.7 1995/04/19 03:01:32 bsmith Exp bsmith $ */

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
               SUMS_NTR }
  SNESMETHOD;

typedef enum { SNES_T, SUMS_T } SNESTYPE;

extern int SNESCreate(MPI_Comm,SNES*);
extern int SNESSetMethod(SNES,SNESMETHOD);
extern int SNESSetMonitor(SNES, int (*)(SNES,int,double,void*),void *);
extern int SNESSetSolution(SNES,Vec,int (*)(Vec,void*),void *);
extern int SNESSetFunction(SNES, Vec, int (*)(Vec,Vec,void*),void *,int);
extern int SNESSetJacobian(SNES,Mat,Mat,int(*)(Vec,Mat*,Mat*,int*,void*),void *);
extern int SNESDestroy(SNES);
extern int SNESSetUp(SNES);
extern int SNESSolve(SNES,int*);
extern int SNESRegister(int, char*, int (*)(SNES));
extern int SNESRegisterAll();
extern int SNESGetSLES(SNES,SLES*);
extern int SNESNoLineSearch(SNES,Vec,Vec,Vec,Vec,Vec,double,double*,double*);
extern int SNESCubicLineSearch(SNES,Vec,Vec,Vec,Vec,Vec,double,double*,double*);
extern int SNESQuadraticLineSearch(SNES,Vec,Vec,Vec,Vec,Vec,double,double*,double*);

extern int SNESGetSolution(SNES,Vec*);
extern int SNESGetFunction(SNES,Vec*);

extern int SNESPrintHelp(SNES);
extern int SNESSetFromOptions(SNES);
extern int SNESGetMethodName(SNESMETHOD,char **);
extern int SNESDefaultMonitor(SNES,int,double,void *);
extern int SNESDefaultConverged(SNES,double,double,double,void*);

extern int SNESSetSolutionTolerance(SNES,double);
extern int SNESSetAbsoluteTolerance(SNES,double);
extern int SNESSetRelativeTolerance(SNES,double);
extern int SNESSetTruncationTolerance(SNES,double);
extern int SNESSetMaxIterations(SNES,int);
extern int SNESSetMaxResidualEvaluations(SNES,int);

#if defined(__DRAW_PACKAGE)
#define SNESLGMonitorCreate  KSPLGMonitorCreate
#define SNESLGMonitorDestroy KSPLGMonitorDestroy
#define SNESLGMonitor        ((int (*)(SNES,int,double,void*))KSPLGMonitor)
#endif

#endif

