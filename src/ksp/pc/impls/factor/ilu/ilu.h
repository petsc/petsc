/* 
   Private data structure for ILU preconditioner.
*/
#if !defined(__ILU_H)
#define __ILU_H

#include <../src/ksp/pc/impls/factor/factor.h>

typedef struct {
  PC_Factor         hdr;
  IS                row,col;         /* row and column permutations for reordering */
  void              *implctx;         /* private implementation context */
  PetscBool         inplace;          /* in-place ILU factorization */
  PetscBool         reuseordering;    /* reuses previous reordering computed */

  PetscBool         reusefill;        /* reuse fill from previous ILUDT */
  PetscReal         actualfill;       /* expected fill in factorization */
  PetscBool         nonzerosalongdiagonal;
  PetscReal         nonzerosalongdiagonaltol;
} PC_ILU;

#endif
