
/*
   Context for a Newton trust region method for solving a system
   of nonlinear equations
 */

#ifndef __SNES_TR_H
#define __SNES_TR_H
#include <petsc/private/snesimpl.h>

typedef struct {
  /* ---- Parameters used by the trust region method  ---- */
  PetscReal mu;           /* used to compute trust region parameter */
  PetscReal eta;          /* used to compute trust region parameter */
  PetscReal delta;        /* trust region parameter */
  PetscReal delta0;       /* used to initialize trust region parameter */
  PetscReal delta1;       /* used to compute trust region parameter */
  PetscReal delta2;       /* used to compute trust region parameter */
  PetscReal delta3;       /* used to compute trust region parameter */
  PetscReal sigma;        /* used to detemine termination */
  PetscBool itflag;       /* flag for convergence testing */
  PetscReal rnorm0, ttol; /* used for KSP convergence test */
  PetscErrorCode (*precheck)(SNES, Vec, Vec, PetscBool *, void *);
  void *precheckctx;
  PetscErrorCode (*postcheck)(SNES, Vec, Vec, Vec, PetscBool *, PetscBool *, void *);
  void *postcheckctx;
} SNES_NEWTONTR;

#endif
