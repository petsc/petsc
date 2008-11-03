
#if !defined(__ICC_H)
#define __ICC_H

#include "../src/ksp/pc/impls/factor/factor.h"

/* Incomplete Cholesky factorization context */

typedef struct {
  PC_Factor       hdr;
  PetscReal       actualfill;
  void            *implctx;
} PC_ICC;

#endif
