/* $Id: mpiadj.h,v 1.3 1998/12/03 04:01:42 bsmith Exp bsmith $ */

#include "src/mat/matimpl.h"

#if !defined(__CSR_H)
#define __CSR_H

/*  
  MATMPICSR format - Compressed row storage for storing adjacency lists, and possibly weights
                     This is for grid reorderings (to reduce bandwidth)
                     grid partitionings, etc. This is NOT currently a dynamic data-structure.
                     
*/

typedef struct {
  int              *rowners;         /* ranges owned by each processor */
  int              rstart,rend;     /* start and end of local rows */
  int              m;                /* local rows */
  int              nz;
  int              *diag;            /* pointers to diagonal elements, if they exist */
  int              *i;               /* pointer to beginning of each row */
  int              *j;               /* column values: j + i[k] - 1 is start of row k */
  Scalar           *values;          /* numerical values */
  PetscTruth       symmetric;        /* user indicates the nonzero structure is symmetric */
} Mat_MPICSR;

#endif
