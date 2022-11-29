/*
   Private context for semismooth newton method with line search for solving
   system of mixed complementarity equations
 */

#ifndef __SNES_VISS_H
#define __SNES_VISS_H

#include <petsc/private/snesimpl.h>

#define PetscScalarNorm(a, b) (PetscSqrtScalar((a) * (a) + (b) * (b)))

typedef struct {
  Vec         phi;     /* pointer to semismooth function */
  PetscReal   phinorm; /* 2-norm of the semismooth function */
  PetscReal   merit;   /* Merit function */
  Vec         dpsi;    /* Merit function gradient */
  Vec         Da;      /* B sub-differential work vector (diag perturbation) */
  Vec         Db;      /* B sub-differential work vector (row scaling) */
  Vec         z;       /* B subdifferential work vector */
  Vec         t;       /* B subdifferential work vector */
  PetscScalar norm_d;  /* two norm of the descent direction */

  /* Copy of user supplied function evaluation routine  */
  PetscErrorCode (*computeuserfunction)(SNES, Vec, Vec, void *);
  /* user supplied function for checking redundant equations for SNESSolveVI_RS2 */
  PetscErrorCode (*checkredundancy)(SNES, IS, IS *, void *);
} SNES_VINEWTONSSLS;

#endif
