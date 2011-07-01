#include <private/snesimpl.h>



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
  PetscViewer              lsmonitor;

  /* ------------------ Semismooth algorithm stuff ------------------------------ */
  Vec                      phi;                      /* pointer to semismooth function */
  PetscReal                phinorm;                 /* 2-norm of the semismooth function */
  PetscReal                merit;           /* Merit function */
  Vec                      dpsi;           /* Merit function gradient */
  Vec                      Da;            /* B sub-differential work vector (diag perturbation) */
  Vec                      Db;            /* B sub-differential work vector (row scaling) */
  Vec                      z;    /* B subdifferential work vector */
  Vec                      t;    /* B subdifferential work vector */
  Vec                      xl;            /* lower bound on variables */
  Vec                      xu;            /* upper bound on variables */

  PetscScalar              norm_d;         /* two norm of the descent direction */
  IS                       IS_inact_prev; /* Inctive set IS for the previous iteration or previous snes solve */

  /* Tolerance to check whether the constraint is satisfied */
  PetscReal                const_tol;
  /* Copy of user supplied function evaluation routine  */
  PetscErrorCode (*computeuserfunction)(SNES,Vec,Vec,void*);
  /* user supplied function for checking redundant equations for SNESSolveVI_RS2 */
  PetscErrorCode (*checkredundancy)(SNES,IS,IS*,void*);
  void                     *ctxP; /* user defined check redundancy context */

  PetscErrorCode           (*computevariablebounds)(SNES,Vec,Vec);
} SNES_VI;

#endif

