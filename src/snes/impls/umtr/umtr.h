/* $Id: umtr.h,v 1.1 1995/07/20 04:02:33 curfman Exp $ */

/*
    Context for a Newton trust region method (unconstrained minimization)
 */

#ifndef __TRM_H
#define __TRM_H
#include "snesimpl.h"

typedef struct {
  double delta0;	/* used to initialize trust region parameter */
  double delta;		/* trust region parameter */
  double eta1;		/* step is unsuccessful if actred < eta1 * prered,
			   where prered is the predicted reduction and 
			   actred is the actual reduction */
  double eta2;		/* used to compute trust region parameter */
  double eta3;		/* used to compute trust region parameter */
  double eta4;		/* used to compute trust region parameter */
  double factor1;	/* used to initialize trust region parameter */
  double actred;	/* actual reduction */
  double prered;	/* predicted reduction */
  int    success;	/* indicator for successful step */
  int    sflag;		/* flag for convergence testing */
} SNES_UMTR;

#endif
