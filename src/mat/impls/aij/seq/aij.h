/* $Id: aij.h,v 1.11 1995/08/17 01:31:13 curfman Exp bsmith $ */

#include "matimpl.h"
#include <math.h>

#if !defined(__AIJ_H)
#define __AIJ_H

/*  
  MATSEQAIJ format - Compressed row storage (also called Yale sparse matrix
  format), compatible with Fortran.  The i[] and j[] arrays start at 1,
  not zero, to support Fortran 77.  For example, in Fortran 
  j[i[k]+p-1] is the pth column in row k.
*/

typedef struct {
  int    sorted;           /* if true, rows are sorted by increasing columns */
  int    roworiented;      /* if true, row-oriented storage */
  int    nonew;            /* if true, don't allow new elements to be added */
  int    singlemalloc;     /* if true a, i, and j have been obtained with
                               one big malloc */
  int    assembled;        /* if true, matrix is fully assembled */
  int    m, n;             /* rows, columns */
  int    nz, maxnz;        /* nonzeros, allocated nonzeros */
  int    *diag;            /* pointers to diagonal elements */
  int    *i;               /* pointer to beginning of each row */
  int    *imax;            /* maximum space allocated for each row */
  int    *ilen;            /* actual length of each row */
  int    *j;               /* column values: j + i[k] - 1 is start of row k */
  Scalar *a;               /* nonzero elements */
  IS     row, col;         /* index sets, used for reorderings */
  Scalar *solve_work;      /* work space used in MatSolve_AIJ */
} Mat_SeqAIJ;

#endif
