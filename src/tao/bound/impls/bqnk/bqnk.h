/*
Context for bounded quasi-Newton-Krylov type optimization algorithms
*/

#if !defined(__TAO_BQNK_H)
#define __TAO_BQNK_H

#include <../src/tao/bound/impls/bnk/bnk.h>

typedef struct {
  Mat B;
} TAO_BQNK;

PETSC_INTERN PetscErrorCode TaoCreate_BQNK(Tao);

#endif /* if !defined(__TAO_BQNK_H) */