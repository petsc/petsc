#ifndef __TAO_SQPCON_H
#define __TAO_SQPCON_H

#include "private/taosolver_impl.h"
#include "petscis.h"

typedef struct {
  Mat M;    /* Quasi-newton hessian matrix */
  Vec dbar;   /* Reduced gradient */
  Vec GL;   /* Gradient of lagrangian */

  IS UIS;   /* Index set to state */
  IS UID;   /* Index set to design */
  IS UIM;   /* Full index set to all constraints */
  VecScatter state_scatter;
  VecScatter design_scatter;

  Vec U;    /* State variable */
  Vec V;    /* Design variable */
  
  Vec DU;   /* State step */
  Vec DV;   /* Design step */
  Vec DL;   /* Multipliers step */

  Vec GU;   /* Gradient wrt U */
  Vec GV;   /* Gradient wrt V */

  Vec W;    /* work vector */
  Vec Xold;
  Vec Gold;
  Vec WU;   /* state work vector */
  Vec WV;   /* design work vector */
  Vec Tbar; 
  Vec aqwac;

  PetscInt m; /* number of constraints */
  PetscInt n; /* number of variables */

  Mat JU;   /* Jacobian wrt U */
  Mat Jpre_U; /* preconditioning matrix wrt U */
  Mat JV;   /* Jacobian wrt V */
  Mat R,Q;
  Vec LM;   /* Lagrange Multiplier */
  Vec WL;   /* Work vector */
  PetscReal rho; /* Penalty parameter */
  PetscInt    subset_type;
  MatStructure statematflag,designmatflag;

  

} TAO_SQPCON;


#endif
