/* $Id: tr.h,v 1.5 1995/07/22 19:47:21 curfman Exp curfman $ */

/*
   Context for a Newton trust region method for solving a system 
   of nonlinear equations
 */

#ifndef __SNES_EQTR_H
#define __SNES_EQTR_H
#include "snesimpl.h"

typedef struct {
  /* ---- Parameters used by the trust region method  ---- */
  double mu;		/* used to compute trust region parameter */
  double eta;		/* used to compute trust region parameter */
  double delta;		/* trust region parameter */
  double delta0;	/* used to initialize trust region parameter */
  double delta1;	/* used to compute trust region parameter */
  double delta2;	/* used to compute trust region parameter */
  double delta3;	/* used to compute trust region parameter */
  double sigma;		/* used to detemine termination */
  int    itflag;	/* flag for convergence testing */
  double rnorm0,ttol;   /* used for KSP convergence test */
} SNES_TR;

#endif
