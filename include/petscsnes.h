/* $Id: snes.h,v 1.5 1995/04/16 22:13:48 curfman Exp curfman $ */

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

typedef enum { SNESINITIAL_GUESS,
               SNESRESIDUAL,
               SNESFUNCTION,
               SNESGRADIENT,
               SNESMATRIX,
               SNESCORE,
               SNESSTEP_SETUP,
               SNESSTEP_COMPUTE,
               SNESSTEP_DESTROY,
               SNESTOTAL }
   SNESPHASE;

typedef enum { SNES_T, SUMS_T } SNESTYPE;

extern int SNESCreate(MPI_Comm,SNES*);
extern int SNESSetMethod(SNES,SNESMETHOD);
extern int SNESSetMonitor(SNES, int (*)(SNES,int,Vec,Vec,double,void*),void *);
extern int SNESSetSolution(SNES,Vec,int (*)(Vec,void*),void *);
extern int SNESSetResidual(SNES, Vec, int (*)(Vec,Vec,void*),void *,int);
extern int SNESSetJacobian(SNES,Mat,int (*)(Vec,Mat*,void*),void *);
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
extern int SNESGetResidual(SNES,Vec*);

extern int SNESPrintHelp(SNES);
extern int SNESSetFromOptions(SNES);
extern int SNESGetMethodName(SNESMETHOD,char **);
extern int SNESDefaultMonitor(SNES,int, Vec,Vec,double,void *);
extern int SNESDefaultConverged(SNES,double,double,double,void*);

extern int SNESSetSolutionTolerance(SNES,double);
extern int SNESSetAbsoluteTolerance(SNES,double);
extern int SNESSetRelativeTolerance(SNES,double);
extern int SNESSetTruncationTolerance(SNES,double);
extern int SNESSetMaxIterations(SNES,int);
extern int SNESSetMaxResidualEvaluations(SNES,int);
#endif

