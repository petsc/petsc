/* $Id: bdiag.h,v 1.16 1995/10/27 01:06:31 curfman Exp bsmith $ */

#include "matimpl.h"
#include <math.h>

#if !defined(__BDIAG_H)
#define __BDIAG_H

/*
   Mat_SeqBDiag (MATSEQBDIAG) - block-diagonal format, where each diagonal
   element consists of a square block of size nb x nb.  Dense storage
   within each block is in column-major order.  The diagonals are the
   full length of the matrix.  As a special case, blocks of size nb=1
   (scalars) are supported as well.
*/

typedef struct {
  int    m, n;             /* rows, columns */
  int    mblock, nblock;   /* block rows and columns */
  int    assembled;        /* if true, matrix is assembled */
  int    nonew;            /* if true, no new nonzeros allowed in matrix */
  int    nonew_diag;       /* if true, no new diagonals allowed in matrix */
  int    nz,maxnz;         /* nonzeros, allocated nonzeros */
  int    nd;               /* number of block diagonals */
  int    mainbd;           /* the number of the main block diagonal */
  int    nb;               /* Each diagonal element is an nb x nb matrix */
  int    *diag;            /* value of (row-col)/nb for each diagonal */
  int    *bdlen;           /* block-length of each diagonal */
  int    ndim;             /* diagonals come from an ndim pde (if 0, ignore) */
  int    ndims[3];         /* sizes of the mesh if ndim > 0 */
  int    user_alloc;       /* true if the user provided the diagonals */
  int    *colloc;          /* holds column locations if using MatGetRow */
  Scalar **diagv;          /* The actual diagonals */
  Scalar *dvalue;          /* Used to hold a row if MatGetRow is used */
  int    *pivot;           /* pivots for LU factorization (temporary loc) */
  int    roworiented;      /* inputs to MatSetValue() are row oriented (default = 1)*/
} Mat_SeqBDiag;

extern int MatConvert_SeqBDiag(Mat,MatType,Mat *);

#endif
