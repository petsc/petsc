/* $Id: umtr.h,v 1.3 1996/08/08 14:46:56 bsmith Exp bsmith $ */

/*
    Context for a Newton trust region method (unconstrained minimization)
 */

#ifndef __SNES_UMTR_H
#define __SNES_UMTR_H
#include "src/snes/snesimpl.h"

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
} SNES_UM_TR;

#endif
