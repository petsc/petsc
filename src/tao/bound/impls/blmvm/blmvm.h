#ifndef __TAO_BLMVM_H
#define __TAO_BLMVM_H
#include <petsc/private/taoimpl.h>

/*
 Context for limited memory variable metric method for bound constrained
 optimization.
*/
typedef struct {

  Mat M;

  Vec unprojected_gradient;
  Vec Xold;
  Vec Gold;

  PetscInt n_free;
  PetscInt n_bind;

  PetscInt grad;
  PetscInt reset;
  Mat      H0;
} TAO_BLMVM;

#endif  /* ifndef __TAO_BLMVM_H */
