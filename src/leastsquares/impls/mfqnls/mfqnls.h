#ifndef __TAO_MFQNLS_H
#define __TAO_MFQNLS_H
#include "include/private/taosolver_impl.h"

typedef struct {
  PetscInt npmax;  /* Max number of interpolation points (>n+1) (def: 2n+1) */
  PetscReal delta; /* Trust region radius (>0) (def: 0.1) */
  PetscInt m; /* number of components */  

} TAO_MFQNLS;

#endif /* ifndef __TAO_MFQNLS */
