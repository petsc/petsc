/*
   Private data structure for LU preconditioner.
*/
#if !defined(__LU_H)
#define __LU_H

#include <../src/ksp/pc/impls/factor/factor.h>

typedef struct {
  PC_Factor hdr;
  IS        row,col;            /* index sets used for reordering */
  PetscBool nonzerosalongdiagonal;
  PetscReal nonzerosalongdiagonaltol;
} PC_LU;

#endif
