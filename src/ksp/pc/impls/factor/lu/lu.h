/* 
   Private data structure for LU preconditioner.
*/
#if !defined(__LU_H)
#define __LU_H

#include "petscmat.h"

typedef struct {
  Mat             fact;             /* factored matrix */
  PetscReal       actualfill;       /* actual fill in factor */
  PetscTruth      inplace;          /* flag indicating in-place factorization */
  IS              row,col;          /* index sets used for reordering */
  MatOrderingType ordering;         /* matrix ordering */
  PetscTruth      reuseordering;    /* reuses previous reordering computed */
  PetscTruth      reusefill;        /* reuse fill from previous LU */
  MatFactorInfo   info;
  PetscTruth      nonzerosalongdiagonal;
  PetscReal       nonzerosalongdiagonaltol;
} PC_LU;

#endif
