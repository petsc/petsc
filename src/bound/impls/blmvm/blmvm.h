#ifndef __TAO_BLMVM_H
#define __TAO_BLMVM_H
#include "include/private/taosolver_impl.h"

/*
 Context for limited memory variable metric method for bound constrained 
 optimization.
*/
typedef struct {

  Mat M;

  Vec unprojected_gradient;
  
  PetscInt n_free;
  PetscInt n_bind;

  PetscInt grad;
  PetscInt reset;
} TAO_BLMVM;

#endif  /* ifndef __TAO_BLMVM_H */
