/*$Id: rich.c,v 1.85 1999/11/05 14:46:46 bsmith Exp bsmith $*/
/*
    Context for a Newton line search method (unconstrained minimization)
 */

#ifndef __SNES_UMLS_H
#define __SNES_UMLS_H
#include "src/snes/snesimpl.h"

typedef struct {

/* --------------- Parameters used by line search method ----------------- */
  Scalar gamma;		     /* damping parameter */
  double maxstep;	     /* maximum step size */
  double gamma_factor;	     /* damping parameter */
  double rtol;		     /* relative tol for acceptable step (rtol>0) */
  double ftol;		     /* tol for sufficient decrease condition (ftol>0) */
  double gtol;		     /* tol for curvature condition (gtol>0)*/
  double stepmin;	     /* lower bound for step */
  double stepmax;	     /* upper bound for step */
  double step;		     /* step size */
  int    max_kspiter_factor; /* computes max KSP iterations */
  int    maxfev;	     /* maximum funct evals per line search call */
  int    nfev;		     /* number of funct evals per line search call */
  int    bracket;
  int    infoc;

/* ------------------------- Line search routine ------------------------- */
  int    (*LineSearch)(SNES,Vec,Vec,Vec,Vec,double*,double*,double*,int*);
  int    line;		     /* line search termination code (set line=1 on success) */
} SNES_UM_LS;

#endif
