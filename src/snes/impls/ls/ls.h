/* $Id: ls.h,v 1.3 1995/06/14 19:55:54 curfman Exp curfman $ */

/* 
   Context for a Newton line search method for solving
   systems of nonlinear equations
 */

#ifndef __SNES_EQLS_H
#define __SNES_EQLS_H
#include "snesimpl.h"

typedef struct {
  int (*LineSearch)(SNES,Vec,Vec,Vec,Vec,Vec,double,double*,double*,int*);
/* --------------- Parameters used by line search method ----------------- */
  double alpha;			/* used to determine sufficient reduction */
  double maxstep;               /* maximum step size */
  double steptol;               /* step convergence tolerance */
} SNES_LS;

#endif
