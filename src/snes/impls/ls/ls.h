/* $Id: ls.h,v 1.6 1998/04/21 23:48:28 curfman Exp curfman $ */

/* 
   Private context for a Newton line search method for solving
   systems of nonlinear equations
 */

#ifndef __SNES_EQLS_H
#define __SNES_EQLS_H
#include "src/snes/snesimpl.h"

typedef struct {
  int    (*LineSearch)(SNES,Vec,Vec,Vec,Vec,Vec,double,double*,double*,int*);
  /* --------------- Parameters used by line search method ----------------- */
  double alpha;			/* used to determine sufficient reduction */
  double maxstep;               /* maximum step size */
  double steptol;               /* step convergence tolerance */
  int    (*CheckStep)(SNES,Vec,int*,void*); /* step-checking routine (optional) */
  void   *checkP;                           /* user-defined step-checking context */
} SNES_LS;

#endif

