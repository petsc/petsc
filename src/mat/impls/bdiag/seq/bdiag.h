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
EXTERN int MatMult_SeqBDiag_1(Mat,Vec,Vec);
EXTERN int MatMult_SeqBDiag_2(Mat,Vec,Vec);
EXTERN int MatMult_SeqBDiag_3(Mat,Vec,Vec);
EXTERN int MatMult_SeqBDiag_4(Mat,Vec,Vec);
EXTERN int MatMult_SeqBDiag_5(Mat,Vec,Vec);
EXTERN int MatMult_SeqBDiag_N(Mat A,Vec,Vec);
EXTERN int MatMultAdd_SeqBDiag_1(Mat,Vec,Vec,Vec);
EXTERN int MatMultAdd_SeqBDiag_2(Mat,Vec,Vec,Vec);
EXTERN int MatMultAdd_SeqBDiag_3(Mat,Vec,Vec,Vec);
EXTERN int MatMultAdd_SeqBDiag_4(Mat,Vec,Vec,Vec);
EXTERN int MatMultAdd_SeqBDiag_5(Mat,Vec,Vec,Vec);
EXTERN int MatMultAdd_SeqBDiag_N(Mat A,Vec,Vec,Vec);
EXTERN int MatMultTranspose_SeqBDiag_1(Mat,Vec,Vec);
EXTERN int MatMultTranspose_SeqBDiag_N(Mat A,Vec,Vec);
EXTERN int MatMultTransposeAdd_SeqBDiag_1(Mat,Vec,Vec,Vec);
EXTERN int MatMultTransposeAdd_SeqBDiag_N(Mat A,Vec,Vec,Vec);
EXTERN int MatSetValues_SeqBDiag_1(Mat,int,const int [],int,const int [],const PetscScalar [],InsertMode);
EXTERN int MatSetValues_SeqBDiag_N(Mat,int,const int [],int,const int [],const PetscScalar [],InsertMode);
EXTERN int MatGetValues_SeqBDiag_1(Mat,int,const int [],int,const int [],PetscScalar []);
EXTERN int MatGetValues_SeqBDiag_N(Mat,int,const int [],int,const int [],PetscScalar []);
EXTERN int MatRelax_SeqBDiag_1(Mat,Vec,PetscReal,MatSORType,PetscReal,int,int,Vec);
EXTERN int MatRelax_SeqBDiag_N(Mat,Vec,PetscReal,MatSORType,PetscReal,int,int,Vec);
EXTERN int MatView_SeqBDiag(Mat,PetscViewer);
EXTERN int MatGetInfo_SeqBDiag(Mat,MatInfoType,MatInfo*);
EXTERN int MatGetRow_SeqBDiag(Mat,int,int *,int **,PetscScalar **);
EXTERN int MatRestoreRow_SeqBDiag(Mat,int,int *,int **,PetscScalar **);
EXTERN int MatTranspose_SeqBDiag(Mat,Mat *);
EXTERN int MatNorm_SeqBDiag(Mat,NormType,PetscReal *);
EXTERN int MatLUFactorSymbolic_SeqBDiag(Mat,IS,IS,MatFactorInfo*,Mat*);
EXTERN int MatILUFactorSymbolic_SeqBDiag(Mat,IS,IS,MatFactorInfo*,Mat*);
EXTERN int MatILUFactor_SeqBDiag(Mat,IS,IS,MatFactorInfo*);
EXTERN int MatLUFactorNumeric_SeqBDiag_N(Mat,Mat*);
EXTERN int MatLUFactorNumeric_SeqBDiag_1(Mat,Mat*);
EXTERN int MatSolve_SeqBDiag_1(Mat,Vec,Vec);
EXTERN int MatSolve_SeqBDiag_2(Mat,Vec,Vec);
EXTERN int MatSolve_SeqBDiag_3(Mat,Vec,Vec);
EXTERN int MatSolve_SeqBDiag_4(Mat,Vec,Vec);
EXTERN int MatSolve_SeqBDiag_5(Mat,Vec,Vec);
EXTERN int MatSolve_SeqBDiag_N(Mat,Vec,Vec);
EXTERN int MatLoad_SeqBDiag(PetscViewer,MatType,Mat*);

#endif
