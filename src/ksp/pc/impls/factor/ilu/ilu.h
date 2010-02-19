/* 
   Private data structure for ILU preconditioner.
*/
#if !defined(__ILU_H)
#define __ILU_H

#include "../src/ksp/pc/impls/factor/factor.h"

typedef struct {
  PC_Factor         hdr;
  IS                row,col;         /* row and column permutations for reordering */
  void              *implctx;         /* private implementation context */
  PetscTruth        inplace;          /* in-place ILU factorization */
  PetscTruth        reuseordering;    /* reuses previous reordering computed */

  PetscTruth        reusefill;        /* reuse fill from previous ILUDT */
  PetscReal         actualfill;       /* expected fill in factorization */
  PetscTruth        nonzerosalongdiagonal;
  PetscReal         nonzerosalongdiagonaltol;
} PC_ILU;

#endif
