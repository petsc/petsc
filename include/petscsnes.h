/* $Id: snes.h,v 1.2 1995/03/20 00:23:58 bsmith Exp bsmith $ */

#if !defined(__SNES_PACKAGE)
#define __SNES_PACKAGE

typedef struct _SNES* SNES;

typedef enum { SNES_NLS1,
               SNES_NTR1,
               SNES_NTR2_DOG,
               SNES_NTR2_LIN,
               SNES_NBASIC,
               SUMS_NLS1,
               SUMS_NTR1 }
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

typedef enum { SNES, SUMS } SNESTYPE;

extern int SNESSetResidual(SNES,Vec);
extern int SNESGetResidual(SNES,Vec*);
extern int SNESSetResidualRoutine(SNES,int (*)(void*,Vec,Vec),int,void*);
extern int SNESSetMaxResidualEvaluations(SNES,int);

extern int SNESDefaultMonitor     ANSI_ARGS((NLCtx*, void*, void*, 
                                        double *));
extern int SNESDefaultConverged   ANSI_ARGS((NLCtx*, double*, double*, 
                                        double *));
extern int SNESGetSLES(SNES,SLES*);

extern int     SNESSetDampingFlag	ANSI_ARGS((NLCtx*, int));
extern int      NLSlesGetDampingFlag	ANSI_ARGS((NLCtx *));


/* --- Miscellaneous routines --- */
extern int     SNESSetUp(SNES);
extern int     NLSlesSetGeneralStep	ANSI_ARGS((NLCtx*,
					void (*)(NLCtx *, void *) ));
extern int     NLSlesSetRoutines	ANSI_ARGS((NLCtx*,
					void (*)(NLCtx *, int *) ));
extern int     NLSlesApplyForwardSolve ANSI_ARGS((NLCtx*, void*, void *));
extern int     NLSlesApplyBackwardSolve ANSI_ARGS((NLCtx*, void*, void *));
       int     NLSlesCheck		ANSI_ARGS((NLCtx*, char*));
extern int      NLSlesSolveScale	ANSI_ARGS((NLCtx*, void*, void*, 
					void*, double*, double*, double*, 
					double*, double*, void *));

