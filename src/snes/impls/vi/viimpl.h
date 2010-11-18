#include "private/snesimpl.h"

#define PETSC_VI_INF   1.0e20
#define PETSC_VI_NINF -1.0e20

#define PetscScalarNorm(a,b) (PetscSqrtScalar((a)*(a)+(b)*(b)))
/* 
   Private context for semismooth newton method with line search for solving
   system of mixed complementarity equations
 */

#ifndef __SNES_VI_H
#define __SNES_VI_H

typedef struct {
  PetscErrorCode           (*LineSearch)(SNES,void*,Vec,Vec,Vec,Vec,Vec,PetscReal,PetscReal,PetscReal*,PetscReal*,PetscBool *);
  void                     *lsP;                              /* user-defined line-search context (optional) */
  /* --------------- Parameters used by line search method ----------------- */
  PetscReal                alpha;		                                                   /* used to determine sufficient reduction */
  PetscReal                maxstep;                                                          /* maximum step size */
  PetscReal                minlambda;                                                        /* determines smallest line search lambda used */
  PetscErrorCode           (*precheckstep)(SNES,Vec,Vec,void*,PetscBool *);                  /* step-checking routine (optional) */
  void                     *precheck;                                                        /* user-defined step-checking context (optional) */
  PetscErrorCode           (*postcheckstep)(SNES,Vec,Vec,Vec,void*,PetscBool *,PetscBool *); /* step-checking routine (optional) */
  void                     *postcheck;                                                       /* user-defined step-checking context (optional) */
  PetscViewerASCIIMonitor  lsmonitor;

  /* ------------------ Semismooth algorithm stuff ------------------------------ */
  Vec                      phi;                      /* pointer to semismooth function */
  PetscReal                phinorm;                 /* 2-norm of the semismooth function */
  PetscErrorCode           (*computessfunction)(PetscScalar,PetscScalar,PetscScalar*); /* Semismooth function evaluation routine */
  PetscReal                merit;           /* Merit function */
  Vec                      Da;            /* B sub-differential work vector (diag perturbation) */
  Vec                      Db;            /* B sub-differential work vector (row scaling) */
  Vec                      z;    /* B subdifferential work vector */
  Vec                      t;    /* B subdifferential work vector */
  Vec                      xl;            /* lower bound on variables */
  Vec                      xu;            /* upper bound on variables */
  PetscBool                usersetxbounds; /* flag to indicate whether the user has set bounds on variables */

  PetscScalar             norm_d;         /* two norm of the descent direction */

  /* Tolerance to check whether the constraint is satisfied */
  PetscReal             const_tol;
  /* Copy of user supplied function evaluation and jacobian evaluation function pointers */
  PetscErrorCode (*computeuserfunction)(SNES,Vec,Vec,void*);
} SNES_VI;

#endif

