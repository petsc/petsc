/* $Id: nlsles.h,v 1.1 1995/03/20 00:11:33 bsmith Exp $ */

#if !defined(__NL_SLES)
#define __NL_SLES

/* Include file for use of SLES solvers within SNES routines */

#include "nonlin/nlspmat.h"

/* ---- Routines to set information within the NLSles context ---- */
extern void     NLSlesSetMethod		ANSI_ARGS((NLCtx*, SVMETHOD ));
extern void     NLSlesSetKSPMethod	ANSI_ARGS((NLCtx*, ITMETHOD ));
extern void     NLSlesSetDefaults	ANSI_ARGS((NLCtx*, SVMETHOD, 
					ITMETHOD ));
extern void     NLSlesSetDampingFlag	ANSI_ARGS((NLCtx*, int));
extern void     NLSlesSetSaveSolverCtx	ANSI_ARGS((NLCtx *));

/* ---- Routines to extract information from the NLSles context ---- */
extern SVMETHOD NLSlesGetMethod		ANSI_ARGS((NLCtx *));
extern ITMETHOD NLSlesGetKSPMethod	ANSI_ARGS((NLCtx *));
extern ITCntx   *NLSlesGetKSPCtx        ANSI_ARGS((NLCtx *));
extern int      NLSlesGetDampingFlag	ANSI_ARGS((NLCtx *));
extern int      NLSlesSaveSolverCtx	ANSI_ARGS((NLCtx *));

/* --- Analogues of SLES routines --- */
extern SVctx    *NLSlesSVCreate 	ANSI_ARGS((NLCtx*, SpMat*, SVMETHOD ));
extern void     NLSlesSVSetUp 		ANSI_ARGS((NLCtx *));
extern void     NLSlesSVSetOperators 	ANSI_ARGS((NLCtx*, SpMat*, SpMat*, 
					int, int ));
extern int      NLSlesSVSolve 		ANSI_ARGS((NLCtx*, void*, void *));
extern void     NLSlesSVDestroy 	ANSI_ARGS((NLCtx *));
extern void     NLSlesSVSetMonitor	ANSI_ARGS((NLCtx*, 
					void (*)(ITCntx*, int, double), void*));
extern void     NLSlesSVSetConvergenceTest ANSI_ARGS((NLCtx*, 
					int (*)(ITCntx*, int, double), void*));

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

/*  
   NLSlesCheck - Checks that a SLES linear solver is set within the 
   nonlinear solver context.  The usage is:
             NLSlesCheck( nlP, name ) {code if context exists;}
 */

#define NLSlesCheck( nlP, name ) \
 if (!(nlP)->sc.svctx || !(nlP)->sc.is_sles) {char buf[200];\
   sprintf(buf,"Must set SLES linear solver context before calling %s\n",name);\
   sprintf(buf,"Use NLSlesSVCreate or NLSetLinearSolverCtx.");\
   SETERRC(1,buf);}\
 else

#endif
