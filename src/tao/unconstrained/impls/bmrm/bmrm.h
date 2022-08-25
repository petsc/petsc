#ifndef __TAO_BMRM_H
#define __TAO_BMRM_H

#include <petsc/private/taoimpl.h>
#include <petscmath.h>

#define BMRM_INFTY 1e30 /* single precision: ~\pm 10^{38.53}; PetscReal precision: ~\pm 10^{308.25} */
#define ALPHA_MIN  1e-10
#define ALPHA_MAX  1e10
#define EPS_SV     1e-15
#define EPS        1e-20
#define TOL_LAM    1e-15
#define TOL_R      1e-10
#define INCRE_DIM  1000

/* Context for BMRM solver */
typedef struct {
  VecScatter scatter; /* Scatter context  */
  Vec        local_w;
  PetscReal  lambda;
} TAO_BMRM;

typedef struct Vec_Chain {
  Vec               V;
  struct Vec_Chain *next;
} Vec_Chain;

/* Context for Dai-Fletcher solver */
typedef struct {
  PetscInt   maxProjIter;
  PetscInt   maxPGMIter;
  PetscInt  *ipt, *ipt2, *uv;
  PetscReal *g, *y, *tempv, *d, *Qd, *t, *xplus, *tplus, *sk, *yk;

  PetscInt dim;

  PetscInt cur_num_cp;

  /* Variables (i.e. Lagrangian multipliers) */
  PetscReal *x;

  /* Linear part of the objective function  */
  PetscReal *f;

  /* Hessian of the QP */
  PetscReal **Q;

  /* Constraint matrix  */
  PetscReal *a;

  /* RHS of the equality constraint */
  PetscReal b;

  /* Lower bound vector for the variables */
  PetscReal *l;

  /* Upper bound vector for the variables */
  PetscReal *u;

  /* Tolerance for optimization error */
  PetscReal tol;
} TAO_DF;

#endif
