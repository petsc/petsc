/* $Id: bdiag.h,v 1.31 2001/08/07 03:02:53 balay Exp $ */

#include "src/mat/matimpl.h"

#if !defined(__BDIAG_H)
#define __BDIAG_H

/*
   Mat_SeqBDiag (MATSEQBDIAG) - block-diagonal format, where each diagonal
   element consists of a square block of size bs  x bs.  Dense storage
   within each block is in column-major order.  The diagonals are the
   full length of the matrix.  As a special case, blocks of size bs=1
   (scalars) are supported as well.
*/

typedef struct {
  int          mblock,nblock;    /* block rows and columns */
  int          nonew;            /* if true, no new nonzeros allowed in matrix */
  int          nonew_diag;       /* if true, no new diagonals allowed in matrix */
  int          nz,maxnz;         /* nonzeros, allocated nonzeros */
  int          nd;               /* number of block diagonals */
  int          mainbd;           /* the number of the main block diagonal */
  int          bs;               /* Each diagonal element is an bs x bs matrix */
  int          *diag;            /* value of (row-col)/bs for each diagonal */
  int          *bdlen;           /* block-length of each diagonal */
  int          ndim;             /* diagonals come from an ndim pde (if 0, ignore) */
  int          ndims[3];         /* sizes of the mesh if ndim > 0 */
  PetscTruth   user_alloc;       /* true if the user provided the diagonals */
  int          *colloc;          /* holds column locations if using MatGetRow */
  PetscScalar  **diagv;          /* The actual diagonals */
  PetscScalar  *dvalue;          /* Used to hold a row if MatGetRow is used */
  int          *pivot;           /* pivots for LU factorization (temporary loc) */
  PetscTruth   roworiented;      /* inputs to MatSetValue() are row oriented (default = 1) */
  int          reallocs;         /* number of allocations during MatSetValues */
} Mat_SeqBDiag;

EXTERN int MatNorm_SeqBDiag_Columns(Mat,PetscReal*,int);
EXTERN int MatMult_SeqBDiag_N(Mat A,Vec,Vec);
EXTERN int MatMultAdd_SeqBDiag_N(Mat A,Vec,Vec,Vec);
EXTERN int MatMultTranspose_SeqBDiag_N(Mat A,Vec,Vec);
EXTERN int MatMultTransposeAdd_SeqBDiag_N(Mat A,Vec,Vec,Vec);

#endif
