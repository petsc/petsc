
#if !defined(__ADJ_H)
#define __ADJ_H
#include "src/mat/matimpl.h"


/*  
  MATMPIAdj format - Compressed row storage for storing adjacency lists, and possibly weights
                     This is for grid reorderings (to reduce bandwidth)
                     grid partitionings, etc. This is NOT currently a dynamic data-structure.
                     
*/

typedef struct {
  PetscInt         nz;
  PetscInt         *diag;            /* pointers to diagonal elements, if they exist */
  PetscInt         *i;               /* pointer to beginning of each row */
  PetscInt         *j;               /* column values: j + i[k] - 1 is start of row k */
  PetscInt         *values;          /* numerical values */
  PetscTruth       symmetric;        /* user indicates the nonzero structure is symmetric */
  PetscTruth       freeaij;          /* call PetscFree() on a, i,j at destroy */
} Mat_MPIAdj;

#endif
