/* $Id: ls.h,v 1.9 1999/03/14 17:54:36 curfman Exp curfman $ */

/* 
   Private context for a Newton line search method for solving
   systems of nonlinear equations
 */

#ifndef __SNES_EQLS_H
#define __SNES_EQLS_H
#include "src/snes/snesimpl.h"

typedef struct {
  int    (*LineSearch)(SNES,void*,Vec,Vec,Vec,Vec,Vec,double,double*,double*,int*);
  void   *lsP;                              /* user-defined line-search context (optional) */
  /* --------------- Parameters used by line search method ----------------- */
  double alpha;		                    /* used to determine sufficient reduction */
  double maxstep;                           /* maximum step size */
  double steptol;                           /* step convergence tolerance */
  int    (*CheckStep)(SNES,void*,Vec,PetscTruth*); /* step-checking routine (optional) */
  void   *checkP;                           /* user-defined step-checking context (optional) */
} SNES_LS;

#endif

