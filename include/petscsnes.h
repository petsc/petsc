/* $Id: snes.h,v 1.1 1995/03/20 00:12:47 bsmith Exp bsmith $ */

#if !defined(__SNES_PACKAGE)
#define __SNES_PACKAGE

typedef enum { NLE_NLS1,
               NLE_NTR1,
               NLE_NTR2_DOG,
               NLE_NTR2_LIN,
               NLE_NBASIC,
               NLM_NLS1,
               NLM_NTR1 }
  NLMETHOD;

typedef enum { NLINITIAL_GUESS,
               NLRESIDUAL,
               NLFUNCTION,
               NLGRADIENT,
               NLMATRIX,
               NLCORE,
               NLSTEP_SETUP,
               NLSTEP_COMPUTE,
               NLSTEP_DESTROY,
               NLTOTAL }
   NLPHASE;

typedef enum { NLE, NLM } NLTYPE;

extern void NLSetResidual               ANSI_ARGS((NLCtx*, void *));
extern void NLSetResidualRoutine        ANSI_ARGS((NLCtx*, 
                                        void (*)(NLCtx*, void*, void*), int ));
extern void NLSetMaxResidualEvaluations ANSI_ARGS((NLCtx*, int ));
extern void *NLGetResidual              ANSI_ARGS((NLCtx *));
extern void NLiResid                    ANSI_ARGS((NLCtx*, void *, void *));

extern void NLENewtonStatisticsMonitor  ANSI_ARGS((NLCtx*, void*, void*, 
                                        double *));
extern void NLENewtonDefaultMonitor     ANSI_ARGS((NLCtx*, void*, void*, 
                                        double *));
extern int  NLENewtonDefaultConverged   ANSI_ARGS((NLCtx*, double*, double*, 
                                        double *));
extern char *NLENewtonDefaultConvergedType ANSI_ARGS((NLCtx *));

extern int SNESGetSLES(SNES,SLES*);

extern void     NLSlesSetDampingFlag	ANSI_ARGS((NLCtx*, int));
extern void     NLSlesSetSaveSolverCtx	ANSI_ARGS((NLCtx *));

/* ---- Routines to extract information from the NLSles context ---- */
extern int      NLSlesGetDampingFlag	ANSI_ARGS((NLCtx *));
extern int      NLSlesSaveSolverCtx	ANSI_ARGS((NLCtx *));

/* --- Miscellaneous routines --- */
extern void     NLSlesSetUp 		ANSI_ARGS((NLCtx*, void *));
extern void     NLSlesSetGeneralStep	ANSI_ARGS((NLCtx*,
					void (*)(NLCtx *, void *) ));
extern void     NLSlesSetRoutines	ANSI_ARGS((NLCtx*,
					void (*)(NLCtx *, void *) ));
extern void     NLSlesApplyForwardSolve ANSI_ARGS((NLCtx*, void*, void *));
extern void     NLSlesApplyBackwardSolve ANSI_ARGS((NLCtx*, void*, void *));
       void     NLSlesCheck		ANSI_ARGS((NLCtx*, char*));
extern int      NLSlesSolveScale	ANSI_ARGS((NLCtx*, void*, void*, 
					void*, double*, double*, double*, 
					double*, double*, void *));

