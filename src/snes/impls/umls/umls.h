/*$Id: umls.h,v 1.9 2001/08/07 03:04:12 balay Exp bsmith $*/
/*
    Context for a Newton line search method (unconstrained minimization)
 */

#ifndef __SNES_UMLS_H
#define __SNES_UMLS_H
#include "src/snes/snesimpl.h"

typedef struct {

/* --------------- Parameters used by line search method ----------------- */
  PetscScalar gamma;		     /* damping parameter */
  PetscReal maxstep;	     /* maximum step size */
  PetscReal gamma_factor;	     /* damping parameter */
  PetscReal rtol;		     /* relative tol for acceptable step (rtol>0) */
  PetscReal ftol;		     /* tol for sufficient decrease condition (ftol>0) */
  PetscReal gtol;		     /* tol for curvature condition (gtol>0)*/
  PetscReal stepmin;	     /* lower bound for step */
  PetscReal stepmax;	     /* upper bound for step */
  PetscReal step;		     /* step size */
  int       max_kspiter_factor; /* computes max KSP iterations */
  int       maxfev;	     /* maximum funct evals per line search call */
  int       nfev;		     /* number of funct evals per line search call */
  int       bracket;
  int       infoc;

/* ------------------------- Line search routine ------------------------- */
  int       (*LineSearch)(SNES,Vec,Vec,Vec,Vec,PetscReal*,PetscReal*,PetscReal*,int*);
  int       line;		     /* line search termination code (set line=1 on success) */
} SNES_UM_LS;

#endif
