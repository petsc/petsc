/* $Id: adj.h,v 1.2 1997/07/02 22:25:57 bsmith Exp $ */

#include "src/mat/matimpl.h"
#include <math.h>

#if !defined(__ADJ_H)
#define __ADJ_H

/*  
  MATSEQADJ format - Compressed row storage for storing adjacency lists, but no 
                     matrix values. This is for grid reorderings (to reduce bandwidth)
                     grid partitionings, etc. This is NOT currently a dynamic data-structure.
                     
*/

typedef struct {
  int              m, n;             /* rows, columns */
  int              nz;
  int              *diag;            /* pointers to diagonal elements */
  int              *i;               /* pointer to beginning of each row */
  int              *j;               /* column values: j + i[k] - 1 is start of row k */
  PetscTruth       symmetric;        /* user indicates the nonzero structure is symmetric */
} Mat_SeqAdj;

#endif
