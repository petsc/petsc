
#ifndef __KSP_PACKAGE
#define __KSP_PACKAGE
#include "petsc.h"
#include "vec.h"

typedef struct _KSP*     KSP;

/*  Possible Krylov Space Methods */
typedef enum { KSPRICHARDSON, KSPCHEBYCHEV, KSPCG, KSPGMRES, 
               KSPTCQMR, KSPBCGS, KSPCGS, KSPTFQMR, KSPCR, KSPLSQR,
               KSPPREONLY } KSPMETHOD;


extern int KSPCreate          ANSI_ARGS((KSP *));
extern int KSPSetMethod       ANSI_ARGS((KSP,KSPMETHOD));
extern int KSPSetUp           ANSI_ARGS((KSP));
extern int KSPSolve           ANSI_ARGS((KSP,int *));
extern int KSPDestroy         ANSI_ARGS((KSP));

extern int KSPRegisterAll     ANSI_ARGS(());
extern int KSPRegister        ANSI_ARGS((KSPMETHOD,char *,void (*)()));

extern int KSPGetMethodName   ANSI_ARGS((KSPMETHOD,char **));
extern int KSPGetWorkCounts   ANSI_ARGS((KSP,int*,int*,int*,int*,int*));
extern int KSPClearWorkCounts ANSI_ARGS((KSP));

extern int KSPSetIterations   ANSI_ARGS((KSP,int));
extern int KSPSetRightPreconditioner ANSI_ARGS((KSP));
extern int KSPGetPreconditionerSide ANSI_ARGS((KSP,int *));
extern int KSPGetMethodFromContext  ANSI_ARGS((KSP,KSPMETHOD *));
extern int KSPGetMethodFromCommandLine
                    ANSI_ARGS((int *,char **,int,char *,KSPMETHOD *));
extern int KSPSetRelativeTolerance  ANSI_ARGS((KSP,double));
extern int KSPSetAbsoluteTolerance ANSI_ARGS((KSP,double));
extern int KSPSetDivergenceTolerance  ANSI_ARGS((KSP,double));
extern int KSPSetCalculateResidual ANSI_ARGS((KSP));
extern int KSPSetDoNotCalculateResidual ANSI_ARGS((KSP));
extern int KSPSetUsePreconditionedResidual ANSI_ARGS((KSP));
extern int KSPSetInitialGuessZero ANSI_ARGS((KSP));
extern int KSPSetCalculateEigenvalues ANSI_ARGS((KSP));
extern int KSPSetRhs ANSI_ARGS((KSP,VeVector));
extern int KSPGetRhs ANSI_ARGS((KSP,Vec *));
extern int KSPSetSolution ANSI_ARGS((KSP,Vec));
extern int KSPGetSolution ANSI_ARGS((KSP,Vec *));
extern int KSPSetAmult  ANSI_ARGS((KSP,void (*)(void *,Vec,Vec),void *));

extern int KSPGetAmultContext ANSI_ARGS((KSP,void **));
extern int KSPSetAmultTranspose ANSI_ARGS((KSP,void (*)(void *,Vec,Vec)));
extern int KSPSetBinv     ANSI_ARGS((KSP,void(*)(void *,Vec,Vec),void *));
extern int KSPGetBinvContext ANSI_ARGS((KSP,void **));
extern int KSPSetBinvTranspose  
                    ANSI_ARGS((KSP,void (*)(void *,Vec,Vec)));
extern int KSPSetMatop 
                    ANSI_ARGS((KSP,void (*)(void *,Vec,Vec)));
extern int KSPSetMatopTranspose  ANSI_ARGS((KSP,void (*)(void *,Vec,Vec)));
extern int KSPSetMonitor 
               ANSI_ARGS((KSP,void (*)(KSP,int,double, void*), void *));
extern int KSPGetMonitorContext  ANSI_ARGS((KSP,void **));
extern int KSPSetResidualHistory  ANSI_ARGS((KSP, double *,int));
extern int KSPSetConvergenceTest  
               ANSI_ARGS((KSP,int (*)(KSP,int,double, void*), void *));
extern int KSPGetConvergenceContext  ANSI_ARGS((KSP,void **));

extern int KSPBuildSolution  ANSI_ARGS((KSP, Vec,Vec *));
extern int KSPBuildResidual  ANSI_ARGS((KSP, Vec, Vec,Vec *));

extern int KSPRichardsonSetScale  ANSI_ARGS((KSP , double));
extern int KSPChebychevSetEigenvalues  ANSI_ARGS((KSP , double, double));
extern int KSPGMRESSetRestart  ANSI_ARGS((KSP, int));

extern int KSPDefaultCGMonitor  ANSI_ARGS((KSP,int,double, void * ));
extern int KSPDefaultCGConverged  ANSI_ARGS((KSP,int,double, void *));
extern int KSPDefaultMonitor    ANSI_ARGS((KSP,int,double, void *));
extern int KSPLineGraphMonitor  ANSI_ARGS((KSP,int,double, void *));
extern int KSPDefaultConverged  ANSI_ARGS((KSP,int,double, void *));

#endif


