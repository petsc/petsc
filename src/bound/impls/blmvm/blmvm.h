/*$Id$*/

/*
 Context for limited memory variable metric method for bound constrained 
 optimization.
*/

#ifndef __TAO_BLMVM_H
#define __TAO_BLMVM_H
#include "include/private/taosolver_impl.h"

typedef struct {

  Mat M;

  Vec GP; /*  gradient projection */
  
  PetscInt n_free;
  PetscInt n_bind;

  PetscInt grad;
  PetscInt reset;
} TAO_BLMVM;

#endif  /* ifndef __TAO_BLMVM_H */
