#ifndef __TAO_LCL_H
#define __TAO_LCL_H

#include <petsc/private/taoimpl.h>
#include <petscis.h>
#define LCL_FORWARD1        0
#define LCL_ADJOINT1        1
#define LCL_FORWARD2        2
#define LCL_ADJOINT2        3

typedef struct {
  Mat M;    /* Quasi-newton hessian matrix */
  Vec dbar;   /* Reduced gradient */
  Vec GL;
  Vec GAugL;
  Vec GL_U;   /* Gradient of lagrangian */
  Vec GL_V;   /* Gradient of lagrangian */
  Vec GAugL_U; /* Augmented lagrangian gradient */
  Vec GAugL_V; /* Augmented lagrangian gradient */
  Vec GL_U0;   /* Gradient of lagrangian */
  Vec GL_V0;   /* Gradient of lagrangian */
  Vec GAugL_U0; /* Augmented lagrangian gradient */
  Vec GAugL_V0; /* Augmented lagrangian gradient */

  IS UIS;   /* Index set to state */
  IS UID;   /* Index set to design */
  IS UIM;   /* Full index set to all constraints */
  VecScatter state_scatter;
  VecScatter design_scatter;

  Vec U;    /* State variable */
  Vec V;    /* Design variable */
  Vec U0;    /* State variable */
  Vec V0;    /* Design variable */
  Vec V1;    /* Design variable */

  Vec DU;   /* State step */
  Vec DV;   /* Design step */
  Vec DL;   /* Multipliers step */

  Vec GU;   /* Gradient wrt U */
  Vec GV;   /* Gradient wrt V */
  Vec GU0;   /* Gradient wrt U */
  Vec GV0;   /* Gradient wrt V */

  Vec W;    /* work vector */
  Vec X0;
  Vec G0;
  Vec WU;   /* state work vector */
  Vec WV;   /* design work vector */
  Vec r;
  Vec s;
  Vec g1,g2;
  Vec con1;

  PetscInt m; /* number of constraints */
  PetscInt n; /* number of variables */

  Mat jacobian_state0;   /* Jacobian wrt U */
  Mat jacobian_state0_pre; /* preconditioning matrix wrt U */
  Mat jacobian_design0;   /* Jacobian wrt V */
  Mat jacobian_state_inv0; /* Inverse of Jacobian wrt U */
  Mat R;

  Vec lamda;   /* Lagrange Multiplier */
  Vec lamda0;   /* Lagrange Multiplier */
  Vec lamda1;   /* Lagrange Multiplier */

  Vec WL;   /* Work vector */
  PetscReal rho; /* Penalty parameter */
  PetscReal rho0;
  PetscReal rhomax;
  PetscReal eps1,eps2;
  PetscReal aug,aug0,lgn,lgn0;
  PetscInt    subset_type;
  PetscInt    solve_type;
  PetscBool recompute_jacobian_flag;
  PetscInt phase2_niter;
  PetscBool verbose;
  PetscReal tau[4];

} TAO_LCL;

#endif
