/* $Id: umtr.h,v 1.4 1999/10/23 00:01:00 bsmith Exp bsmith $ */

/*
    Context for a Newton trust region method (unconstrained minimization)
 */

#ifndef __SNES_UMTR_H
#define __SNES_UMTR_H
#include "src/snes/snesimpl.h"

typedef struct {
  PetscReal delta0;	/* used to initialize trust region parameter */
  PetscReal delta;		/* trust region parameter */
  PetscReal eta1;		/* step is unsuccessful if actred < eta1 * prered,
			   where prered is the predicted reduction and 
			   actred is the actual reduction */
  PetscReal eta2;		/* used to compute trust region parameter */
  PetscReal eta3;		/* used to compute trust region parameter */
  PetscReal eta4;		/* used to compute trust region parameter */
  PetscReal factor1;	/* used to initialize trust region parameter */
  PetscReal actred;	/* actual reduction */
  PetscReal prered;	/* predicted reduction */
  int       success;	/* indicator for successful step */
  int       sflag;		/* flag for convergence testing */
} SNES_UM_TR;

#endif
