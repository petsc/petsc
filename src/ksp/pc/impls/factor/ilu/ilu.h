/*
   Private data structure for ILU preconditioner.
*/
#pragma once

#include <../src/ksp/pc/impls/factor/factor.h>

typedef struct {
  PC_Factor hdr;
  IS        row, col; /* row and column permutations for reordering */
  void     *implctx;  /* private implementation context */
  PetscBool nonzerosalongdiagonal;
  PetscReal nonzerosalongdiagonaltol;
} PC_ILU;
