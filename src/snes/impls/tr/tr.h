/* $Id: newtr1.h,v 1.3 1995/02/22 00:52:50 curfman Exp $ */

/*
   Context for a Newton trust region method (NLE_NTR1) 
 */

#ifndef __NLE_NTR1
#define __NLE_NTR1

typedef struct {
  /* ---- Parameters used by the trust region method NLE_NTR1 ---- */
  double mu;		/* used to compute trust region parameter */
  double eta;		/* used to compute trust region parameter */
  double delta;		/* trust region parameter */
  double delta0;	/* used to initialize trust region parameter */
  double delta1;	/* used to compute trust region parameter */
  double delta2;	/* used to compute trust region parameter */
  double delta3;	/* used to compute trust region parameter */
  double sigma;		/* used to detemine termination */
  int    itflag;	/* flag for convergence testing */
} NLENewtonTR1Ctx;

/* Routines for the method NLE_NTR1 */
void   NLENewtonTR1Create		ANSI_ARGS((NLCtx *));
void   NLENewtonTR1SetUp		ANSI_ARGS((NLCtx *));
int    NLENewtonTR1Solve		ANSI_ARGS((NLCtx *));
void   NLENewtonTR1Destroy		ANSI_ARGS((NLCtx *));
void   NLENewtonTR1SetParameter		ANSI_ARGS((NLCtx*, char*, double *));
double NLENewtonTR1GetParameter		ANSI_ARGS((NLCtx*, char *));
int    NLENewtonTR1DefaultConverged	ANSI_ARGS((NLCtx*, double*, double*,
                                        double *));
char   *NLENewtonTR1DefaultConvergedType ANSI_ARGS((NLCtx *));

#endif
