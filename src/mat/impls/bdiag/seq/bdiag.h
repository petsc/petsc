
#include "matimpl.h"
#include "math.h"

#if !defined(__BDIAG_H)
#define __BDIAG_H

/*
   Mat_Bdiag (MATBDIAG) - block-diagonal format, where each diagonal
   element consists of a square block of size nb x nb.  Dense storage
   within each block is in column-major order.  The diagonals are the
   full length of the matrix.  As a special case, blocks of size nb=1
   (scalars) are supported as well.
*/

typedef struct {
  int    m, n;         /* rows, columns */
  int    mblock, nblock;       /* block rows and columns */
  int    assembled, nonew;
  int    nz,maxnz,mem; /* nonzeros, allocated nonzeros, memory */
  int    nd;         /* Number of block diagonals */
  int    mainbd;     /* the number of the main block diagonal */
  int    nb;         /* Each diagonal element is an nb x nb matrix */
  int    *diag;      /* value of (row-col)/nb for each diagonal */
  int    *bdlen;     /* block-length of each diagonal */
  int    ndim;       /* Diagonals come from an ndim pde (if 0, ignore) */
  int    ndims[3];   /* Sizes of the mesh if ndim > 0 */
  int    user_alloc; /* True if the user provided the diagonals */
  int    *colloc;    /* Used to hold the column locations if
			      ScatterFromRow is used */
  Scalar **diagv;    /* The actual diagonals */
  Scalar *dvalue;    /* Used to hold a row if ScatterFromRow is used */
} Mat_BDiag;

/* need something like this.
   SpBDGetDiagNumbers - Returns a pointer to the array of diagonal numbers.

#define SpBDGetDiagNumbers( mat )	((SpMatBDiag *)(mat)->data)->diag
 */

#endif
