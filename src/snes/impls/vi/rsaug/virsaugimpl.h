#include <petsc-private/snesimpl.h>



#define PetscScalarNorm(a,b) (PetscSqrtScalar((a)*(a)+(b)*(b)))
/* 
   Private context for semismooth newton method with line search for solving
   system of mixed complementarity equations
 */

#ifndef __SNES_VI_H
#define __SNES_VI_H

typedef struct {

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
  PetscInt                 ntruebounds;   /* number of variables that have at least one non-infinite bound given */

  PetscScalar              norm_d;         /* two norm of the descent direction */
  IS                       IS_inact_prev; /* Inctive set IS for the previous iteration or previous snes solve */

  /* Tolerance to check whether the constraint is satisfied */
  PetscReal                const_tol;
  /* Copy of user supplied function evaluation routine  */
  PetscErrorCode (*computeuserfunction)(SNES,Vec,Vec,void*);
  /* user supplied function for checking redundant equations for SNESSolveVI_RS2 */
  PetscErrorCode (*checkredundancy)(SNES,IS,IS*,void*);
  PetscErrorCode (*computevariablebounds)(SNES,Vec,Vec);        /* user provided routine to set box constrained variable bounds */
  void                     *ctxP; /* user defined check redundancy context */


  PetscBool                ignorefunctionsign;    /* when computing active set ignore the sign of the function values */
} SNES_VIRSAUG;

#endif

