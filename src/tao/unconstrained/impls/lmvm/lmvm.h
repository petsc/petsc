/*
 Context for limited memory variable metric method for unconstrained
 optimization.
*/

#ifndef __TAO_LMVM_H
#define __TAO_LMVM_H
#include <petsc/private/taoimpl.h>

typedef struct {
  Mat M;

  Vec X;
  Vec G;
  Vec D;
  Vec W;

  Vec Xold;
  Vec Gold;

  PetscInt bfgs;
  PetscInt grad;
  Mat      H0;

  PetscBool recycle;
} TAO_LMVM;

#endif /* ifndef __TAO_LMVM_H */
