/* $Id: baij.h,v 1.35 2001/08/07 03:02:55 balay Exp $ */

#include "src/mat/matimpl.h"

#if !defined(__BAIJ_H)
#define __BAIJ_H

/*  
  MATSEQBAIJ format - Block compressed row storage. The i[] and j[] 
  arrays start at 0.
*/

typedef struct {
  PetscTruth       sorted;       /* if true, rows are sorted by increasing columns */
  PetscTruth       roworiented;  /* if true, row-oriented input, default */
  int              nonew;        /* 1 don't add new nonzeros, -1 generate error on new */
  PetscTruth       singlemalloc; /* if true a, i, and j have been obtained with
                                        one big malloc */
  int              bs,bs2;       /* block size, square of block size */
  int              mbs,nbs;      /* rows/bs, columns/bs */
  int              nz,maxnz;     /* nonzeros, allocated nonzeros */
  int              *diag;        /* pointers to diagonal elements */
  int              *i;           /* pointer to beginning of each row */
  int              *imax;        /* maximum space allocated for each row */
  int              *ilen;        /* actual length of each row */
  int              *j;           /* column values: j + i[k] - 1 is start of row k */
  MatScalar        *a;           /* nonzero elements */
  IS               row,col,icol; /* index sets, used for reorderings */
  PetscScalar      *solve_work;  /* work space used in MatSolve */
  int              reallocs;     /* number of mallocs done during MatSetValues() 
                                    as more values are set then were preallocated */
  PetscScalar      *mult_work;   /* work array for matrix vector product*/
  PetscScalar      *saved_values; 

  PetscTruth       keepzeroedrows; /* if true, MatZeroRows() will not change nonzero structure */

#if defined(PETSC_USE_MAT_SINGLE)
  int              setvalueslen;
  MatScalar        *setvaluescopy; /* area double precision values in MatSetValuesXXX() are copied
                                      before calling MatSetValuesXXX_SeqBAIJ_MatScalar() */
#endif
  PetscTruth       pivotinblocks;  /* pivot inside factorization of each diagonal block */

  int              *xtoy,*xtoyB;     /* map nonzero pattern of X into Y's, used by MatAXPY() */
  Mat              XtoY;             /* used by MatAXPY() */
} Mat_SeqBAIJ;

EXTERN int MatILUFactorSymbolic_SeqBAIJ(Mat,IS,IS,MatFactorInfo*,Mat *);
EXTERN int MatDuplicate_SeqBAIJ(Mat,MatDuplicateOption,Mat*);
EXTERN int MatMarkDiagonal_SeqBAIJ(Mat);

EXTERN int MatLUFactorSymbolic_SeqBAIJ(Mat,IS,IS,MatFactorInfo*,Mat*);
EXTERN int MatLUFactor_SeqBAIJ(Mat,IS,IS,MatFactorInfo*);
EXTERN int MatIncreaseOverlap_SeqBAIJ(Mat,int,IS*,int);
EXTERN int MatGetSubMatrix_SeqBAIJ(Mat,IS,IS,int,MatReuse,Mat*);
EXTERN int MatGetSubMatrices_SeqBAIJ(Mat,int,IS*,IS*,MatReuse,Mat**);
EXTERN int MatMultTranspose_SeqBAIJ(Mat,Vec,Vec);
EXTERN int MatMultTransposeAdd_SeqBAIJ(Mat,Vec,Vec,Vec);
EXTERN int MatScale_SeqBAIJ(PetscScalar*,Mat);
EXTERN int MatNorm_SeqBAIJ(Mat,NormType,PetscReal *);
EXTERN int MatEqual_SeqBAIJ(Mat,Mat,PetscTruth*);
EXTERN int MatGetDiagonal_SeqBAIJ(Mat,Vec);
EXTERN int MatDiagonalScale_SeqBAIJ(Mat,Vec,Vec);
EXTERN int MatGetInfo_SeqBAIJ(Mat,MatInfoType,MatInfo *);
EXTERN int MatZeroEntries_SeqBAIJ(Mat);

EXTERN int MatSeqBAIJ_UpdateFactorNumeric_NaturalOrdering(Mat);
EXTERN int MatSeqBAIJ_UpdateSolvers(Mat);

EXTERN int MatSolve_SeqBAIJ_Update(Mat,Vec,Vec);
EXTERN int MatSolve_SeqBAIJ_1(Mat,Vec,Vec);
EXTERN int MatSolve_SeqBAIJ_1_NaturalOrdering(Mat,Vec,Vec);
EXTERN int MatSolve_SeqBAIJ_2(Mat,Vec,Vec);
EXTERN int MatSolve_SeqBAIJ_2_NaturalOrdering(Mat,Vec,Vec);
EXTERN int MatSolve_SeqBAIJ_3(Mat,Vec,Vec);
EXTERN int MatSolve_SeqBAIJ_3_NaturalOrdering(Mat,Vec,Vec);
EXTERN int MatSolve_SeqBAIJ_4(Mat,Vec,Vec);
EXTERN int MatSolve_SeqBAIJ_4_NaturalOrdering(Mat,Vec,Vec);
#if defined(PETSC_HAVE_SSE)
EXTERN int MatSolve_SeqBAIJ_4_SSE_Demotion(Mat,Vec,Vec);
EXTERN int MatSolve_SeqBAIJ_4_NaturalOrdering_SSE_Demotion(Mat,Vec,Vec);
EXTERN int MatSolve_SeqBAIJ_4_NaturalOrdering_SSE_Demotion_usj(Mat,Vec,Vec);
#endif
EXTERN int MatSolve_SeqBAIJ_5(Mat,Vec,Vec);
EXTERN int MatSolve_SeqBAIJ_5_NaturalOrdering(Mat,Vec,Vec);
EXTERN int MatSolve_SeqBAIJ_6(Mat,Vec,Vec);
EXTERN int MatSolve_SeqBAIJ_6_NaturalOrdering(Mat,Vec,Vec);
EXTERN int MatSolve_SeqBAIJ_7(Mat,Vec,Vec);
EXTERN int MatSolve_SeqBAIJ_7_NaturalOrdering(Mat,Vec,Vec);
EXTERN int MatSolve_SeqBAIJ_N(Mat,Vec,Vec);

EXTERN int MatSolveTranspose_SeqBAIJ_Update(Mat,Vec,Vec);
EXTERN int MatSolveTranspose_SeqBAIJ_1(Mat,Vec,Vec);
EXTERN int MatSolveTranspose_SeqBAIJ_1_NaturalOrdering(Mat,Vec,Vec);
EXTERN int MatSolveTranspose_SeqBAIJ_2(Mat,Vec,Vec);
EXTERN int MatSolveTranspose_SeqBAIJ_2_NaturalOrdering(Mat,Vec,Vec);
EXTERN int MatSolveTranspose_SeqBAIJ_3(Mat,Vec,Vec);
EXTERN int MatSolveTranspose_SeqBAIJ_3_NaturalOrdering(Mat,Vec,Vec);
EXTERN int MatSolveTranspose_SeqBAIJ_4(Mat,Vec,Vec);
EXTERN int MatSolveTranspose_SeqBAIJ_4_NaturalOrdering(Mat,Vec,Vec);
EXTERN int MatSolveTranspose_SeqBAIJ_5(Mat,Vec,Vec);
EXTERN int MatSolveTranspose_SeqBAIJ_5_NaturalOrdering(Mat,Vec,Vec);
EXTERN int MatSolveTranspose_SeqBAIJ_6(Mat,Vec,Vec);
EXTERN int MatSolveTranspose_SeqBAIJ_6_NaturalOrdering(Mat,Vec,Vec);
EXTERN int MatSolveTranspose_SeqBAIJ_7(Mat,Vec,Vec);
EXTERN int MatSolveTranspose_SeqBAIJ_7_NaturalOrdering(Mat,Vec,Vec);

EXTERN int MatLUFactorNumeric_SeqBAIJ_1(Mat,Mat*);
EXTERN int MatLUFactorNumeric_SeqBAIJ_2(Mat,Mat*);
EXTERN int MatLUFactorNumeric_SeqBAIJ_2_NaturalOrdering(Mat,Mat*);
EXTERN int MatLUFactorNumeric_SeqBAIJ_3(Mat,Mat*);
EXTERN int MatLUFactorNumeric_SeqBAIJ_3_NaturalOrdering(Mat,Mat*);
EXTERN int MatLUFactorNumeric_SeqBAIJ_4(Mat,Mat*);
EXTERN int MatLUFactorNumeric_SeqBAIJ_4_NaturalOrdering(Mat,Mat*);
#if defined(PETSC_HAVE_SSE)
EXTERN int MatLUFactorNumeric_SeqBAIJ_4_NaturalOrdering_SSE(Mat,Mat*);
EXTERN int MatLUFactorNumeric_SeqBAIJ_4_NaturalOrdering_SSE_usj(Mat,Mat*);
#else
#endif
EXTERN int MatLUFactorNumeric_SeqBAIJ_5(Mat,Mat*);
EXTERN int MatLUFactorNumeric_SeqBAIJ_5_NaturalOrdering(Mat,Mat*);
EXTERN int MatLUFactorNumeric_SeqBAIJ_6(Mat,Mat*);
EXTERN int MatLUFactorNumeric_SeqBAIJ_6_NaturalOrdering(Mat,Mat*);
EXTERN int MatLUFactorNumeric_SeqBAIJ_7(Mat,Mat*);
EXTERN int MatLUFactorNumeric_SeqBAIJ_7_NaturalOrdering(Mat,Mat*);
EXTERN int MatLUFactorNumeric_SeqBAIJ_N(Mat,Mat*);

EXTERN int MatMult_SeqBAIJ_1(Mat,Vec,Vec);
EXTERN int MatMult_SeqBAIJ_2(Mat,Vec,Vec);
EXTERN int MatMult_SeqBAIJ_3(Mat,Vec,Vec);
EXTERN int MatMult_SeqBAIJ_4(Mat,Vec,Vec);
EXTERN int MatMult_SeqBAIJ_5(Mat,Vec,Vec);
EXTERN int MatMult_SeqBAIJ_6(Mat,Vec,Vec);
EXTERN int MatMult_SeqBAIJ_7(Mat,Vec,Vec);
EXTERN int MatMult_SeqBAIJ_N(Mat,Vec,Vec);

EXTERN int MatMultAdd_SeqBAIJ_1(Mat,Vec,Vec,Vec);
EXTERN int MatMultAdd_SeqBAIJ_2(Mat,Vec,Vec,Vec);
EXTERN int MatMultAdd_SeqBAIJ_3(Mat,Vec,Vec,Vec);
EXTERN int MatMultAdd_SeqBAIJ_4(Mat,Vec,Vec,Vec);
EXTERN int MatMultAdd_SeqBAIJ_5(Mat,Vec,Vec,Vec);
EXTERN int MatMultAdd_SeqBAIJ_6(Mat,Vec,Vec,Vec);
EXTERN int MatMultAdd_SeqBAIJ_7(Mat,Vec,Vec,Vec);
EXTERN int MatMultAdd_SeqBAIJ_N(Mat,Vec,Vec,Vec);

#endif
