/* $Id: newls1.h,v 1.4 1995/02/22 00:52:14 curfman Exp $ */

/* 
   Context for a Newton line search method (NLE_NLS1) for solving
   systems of nonlinear equations. 
 */

#ifndef __NLE_NLS1
#define __NLE_NLS1

typedef struct {
/* ------------------------- Line search routine ------------------------- */
  int    (*line_search)();      

/* --------------- Parameters used by line search method ----------------- */
  double alpha;			/* used to determine sufficient reduction */
  double maxstep;               /* maximum step size */
  double steptol;               /* step convergence tolerance */
} NLENewtonLS1Ctx;

void   NLSetLineSearchRoutine		ANSI_ARGS((NLCtx*, int (*)(NLCtx*, 
					void*, void*, void*, void*, void*, 
					double, double*, double *) ));
int    NLStepSimpleLineSearch		ANSI_ARGS((NLCtx*, void*, void*, 
					void*, void*, void*, double, 
					double*, double *));
int    NLStepDefaultLineSearch		ANSI_ARGS((NLCtx*, void*, void*,
					void*, void*, void*, double, 
					double*, double *));

/* Routines for the NLE_NLS1 method */
void   NLENewtonLS1Create		ANSI_ARGS((NLCtx *));
void   NLENewtonLS1SetUp		ANSI_ARGS((NLCtx *));
int    NLENewtonLS1Solve		ANSI_ARGS((NLCtx *));
void   NLENewtonLS1Destroy		ANSI_ARGS((NLCtx *));
void   NLENewtonLS1SetParameter		ANSI_ARGS((NLCtx*, char*, double *));
double NLENewtonLS1GetParameter		ANSI_ARGS((NLCtx*, char *));

#endif
