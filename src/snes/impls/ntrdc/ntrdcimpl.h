
/*
   Context for a Newton trust region method for solving a system
   of nonlinear equations
 */

#ifndef __SNES_TR_H
#define __SNES_TR_H
#include <petsc/private/snesimpl.h>

typedef struct {
  /* ---- Parameters used by the trust region method  ---- */
  PetscReal mu;     /* used to compute trust region parameter */
  PetscReal eta;    /* used to compute trust region parameter */
  PetscReal delta;  /* trust region parameter */
  PetscReal delta0; /* used to initialize trust region parameter */
  PetscReal delta1; /* used to compute trust region parameter */
  PetscReal delta2; /* used to compute trust region parameter */
  PetscReal delta3; /* used to compute trust region parameter */

  PetscReal eta1;   /* Heeho's new TR-dogleg */
  PetscReal eta2;   /* Heeho's new TR-dogleg */
  PetscReal eta3;   /* Heeho's new TR-dogleg */
  PetscReal t1;     /* Heeho's new TR-dogleg */
  PetscReal t2;     /* Heeho's new TR-dogleg */
  PetscReal deltaM; /* Heeho's new TR-dogleg */
  /* currently using fixed array for the block size because of memory leak */
  /* PetscReal      *inorms;         Heeho's new TR-dogleg, stores largest inf norm */
  /* PetscInt       bs;              Heeho's new TR-dogleg, solution vector block size */

  PetscReal sigma;                 /* used to detemine termination */
  PetscBool itflag;                /* flag for convergence testing */
  PetscBool use_cauchy;            /* flag to use/not use Cauchy step and direction (S&D) */
  PetscBool auto_scale_multiphase; /* flag to use/not use autoscaling for Cauchy S&D for multiphase*/
  PetscReal auto_scale_max;        /* max cap value for auto-scaling muste be > 1 */
  PetscBool rho_satisfied;         /* flag for whether inner iteration satisfied rho */
  PetscReal rnorm0, ttol;          /* used for KSP convergence test */
  PetscErrorCode (*precheck)(SNES, Vec, Vec, PetscBool *, void *);
  void *precheckctx;
  PetscErrorCode (*postcheck)(SNES, Vec, Vec, Vec, PetscBool *, PetscBool *, void *);
  void *postcheckctx;
} SNES_NEWTONTRDC;

#endif
