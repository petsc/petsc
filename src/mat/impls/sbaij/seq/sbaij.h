/* $Id: sbaij.h,v 1.9 2000/09/12 14:03:46 hzhang Exp hzhang $ */

#include "src/mat/matimpl.h"

#if !defined(__SBAIJ_H)
#define __SBAIJ_H

/*  
  MATSEQSBAIJ format - Block compressed row storage. The i[] and j[] 
  arrays start at 0.
*/

typedef struct {
  PetscTruth       sorted;       /* if true, rows are sorted by increasing columns */
  PetscTruth       roworiented;  /* if true, row-oriented input, default */
  int              nonew;        /* 1 don't add new nonzeros, -1 generate error on new */
  PetscTruth       singlemalloc; /* if true a, i, and j have been obtained with
                                        one big malloc */
  int              m,n;          /* rows or columns */
  int              bs,bs2;       /* block size, square of block size */
  int              mbs,nbs;      /* rows/bs or columns/bs */
  int              s_nz,s_maxnz; /* total diagonal and superdiagonal nonzero blocks, 
                                    total allocated diagonal and superdiagonal nonzero blocks*/                            
  int              *diag;        /* pointers to diagonal elements */
  int              *i;     /* pointer to beginning of each row */
  int              *imax;        /* maximum space allocated for each row */
  int              *ilen;        /* actual length of each row */
  int              *j;     /* column values: j + i[k] is start of row k */
  MatScalar        *a;     /* nonzero diagonal and subdiagonal elements */
  IS               row,icol;     /* index sets, used for reorderings */
  Scalar           *solve_work;  /* work space used in MatSolve */
  void             *spptr;       /* pointer for special library like SuperLU */
  int              reallocs;     /* number of mallocs done during MatSetValues() 
                                    as more values are set then were preallocated */
  Scalar           *mult_work;   /* work array for matrix vector product*/
  Scalar           *saved_values; 

  PetscTruth       keepzeroedrows; /* if true, MatZeroRows() will not change nonzero structure */
  /* IS               a2anew; */        /* map used for symm permutation */
} Mat_SeqSBAIJ;

extern int MatIncompleteCholeskyFactorSymbolic_SeqSBAIJ(Mat,IS,PetscReal,int,Mat *);
extern int MatConvert_SeqSBAIJ(Mat,MatType,Mat *);
extern int MatDuplicate_SeqSBAIJ(Mat,MatDuplicateOption,Mat*);
extern int MatMarkDiagonal_SeqSBAIJ(Mat);

extern int MatSolveTranspose_SeqSBAIJ_1_NaturalOrdering(Mat,Vec,Vec);
extern int MatCholeskyFactorNumeric_SeqSBAIJ_2_NaturalOrdering(Mat,Mat*);
extern int MatSolve_SeqSBAIJ_2_NaturalOrdering(Mat,Vec,Vec);
extern int MatSolveTranspose_SeqSBAIJ_2_NaturalOrdering(Mat,Vec,Vec);
extern int MatCholeskyFactorNumeric_SeqSBAIJ_3_NaturalOrdering(Mat,Mat*);
extern int MatSolve_SeqSBAIJ_3_NaturalOrdering(Mat,Vec,Vec);
extern int MatSolveTranspose_SeqSBAIJ_3_NaturalOrdering(Mat,Vec,Vec);
extern int MatCholeskyFactorNumeric_SeqSBAIJ_4_NaturalOrdering(Mat,Mat*);
extern int MatSolve_SeqSBAIJ_4_NaturalOrdering(Mat,Vec,Vec);
extern int MatSolveTranspose_SeqSBAIJ_4_NaturalOrdering(Mat,Vec,Vec);
extern int MatCholeskyFactorNumeric_SeqSBAIJ_5_NaturalOrdering(Mat,Mat*);
extern int MatSolve_SeqSBAIJ_5_NaturalOrdering(Mat,Vec,Vec);
extern int MatSolveTranspose_SeqSBAIJ_5_NaturalOrdering(Mat,Vec,Vec);
extern int MatCholeskyFactorNumeric_SeqSBAIJ_6_NaturalOrdering(Mat,Mat*);
extern int MatSolve_SeqSBAIJ_6_NaturalOrdering(Mat,Vec,Vec);
extern int MatSolveTranspose_SeqSBAIJ_6_NaturalOrdering(Mat,Vec,Vec);
extern int MatCholeskyFactorNumeric_SeqSBAIJ_7_NaturalOrdering(Mat,Mat*);
extern int MatSolve_SeqSBAIJ_7_NaturalOrdering(Mat,Vec,Vec);
extern int MatSolveTranspose_SeqSBAIJ_7_NaturalOrdering(Mat,Vec,Vec);

extern int MatReorderingSeqSBAIJ(Mat,IS);

#endif
