#ifndef __TAO_NELDERMEAD_H
#define __TAO_NELDERMEAD_H
#include <petsc/private/taoimpl.h>

typedef struct {
  PetscReal mu_ic;
  PetscReal mu_oc;
  PetscReal mu_r;
  PetscReal mu_e;

  PetscReal lambda; /*  starting point delta for finding starting simplex */

  PetscInt  N;
  PetscReal oneOverN;
  Vec       Xbar, Xmuc, Xmur, Xmue;
  Vec       G;
  Vec      *simplex;

  PetscReal *f_values;
  PetscInt  *indices;

  PetscInt nshrink;
  PetscInt nexpand;
  PetscInt nreflect;
  PetscInt nincontract;
  PetscInt noutcontract;

} TAO_NelderMead;

#endif /* ifndef __TAO_NELDERMEAD_H */
