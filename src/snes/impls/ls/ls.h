/* $Id: newls1.h,v 1.1 1995/03/20 22:59:55 bsmith Exp bsmith $ */

/* 
   Context for a Newton line search method (NLE_NLS1) for solving
   systems of nonlinear equations. 
 */

#ifndef __SNES_NLS1
#define __SNES_NLS1

#include "snesimpl.h"

typedef struct {
  int (*LineSearch)(SNES, Vec, Vec, Vec, Vec, Vec, double, double*, double*);
/* --------------- Parameters used by line search method ----------------- */
  double alpha;			/* used to determine sufficient reduction */
  double maxstep;               /* maximum step size */
  double steptol;               /* step convergence tolerance */
} SNES_LS;


#endif
