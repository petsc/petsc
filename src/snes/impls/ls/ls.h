/* $Id: ls.h,v 1.2 1995/05/05 03:51:33 bsmith Exp curfman $ */

/* 
   Context for a Newton line search method (NLE_NLS1) for solving
   systems of nonlinear equations. 
 */

#ifndef __SNES_NLS1
#define __SNES_NLS1

#include "snesimpl.h"

typedef struct {
  int (*LineSearch)(SNES,Vec,Vec,Vec,Vec,Vec,double,double*,double*,int*);
/* --------------- Parameters used by line search method ----------------- */
  double alpha;			/* used to determine sufficient reduction */
  double maxstep;               /* maximum step size */
  double steptol;               /* step convergence tolerance */
} SNES_LS;


#endif
