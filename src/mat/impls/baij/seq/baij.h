
#if !defined(__BAIJ_H)
#define __BAIJ_H
#include "src/mat/matimpl.h"


/*  
  MATSEQBAIJ format - Block compressed row storage. The i[] and j[] 
  arrays start at 0.
*/

/* This header is shared by the SeqSBAIJ matrix */
#define SEQBAIJHEADER \
  PetscTruth       sorted;       /* if true, rows are sorted by increasing columns */                \
  PetscTruth       roworiented;  /* if true, row-oriented input, default */                          \
  PetscInt         nonew;        /* 1 don't add new nonzeros, -1 generate error on new */            \
  PetscTruth       singlemalloc; /* if true a, i, and j have been obtained with                      \
                                        one big malloc */                                            \
  PetscInt         bs2;          /*  square of block size */                                         \
  PetscInt         mbs,nbs;      /* rows/bs, columns/bs */                                           \
  PetscInt         nz,maxnz;     /* nonzeros, allocated nonzeros */                                  \
  PetscInt         *diag;        /* pointers to diagonal elements */                                 \
  PetscInt         *i;           /* pointer to beginning of each row */                              \
  PetscInt         *imax;        /* maximum space allocated for each row */                          \
  PetscInt         *ilen;        /* actual length of each row */                                     \
  PetscInt         *j;           /* column values: j + i[k] - 1 is start of row k */                 \
  MatScalar        *a;           /* nonzero elements */                                              \
  IS               row,col,icol; /* index sets, used for reorderings */                              \
  PetscScalar      *solve_work;  /* work space used in MatSolve */                                   \
  PetscInt         reallocs;     /* number of mallocs done during MatSetValues()                     \
                                    as more values are set then were preallocated */                 \
  PetscScalar      *mult_work;   /* work array for matrix vector product*/                           \
  PetscScalar      *saved_values;                                                                    \
                                                                                                     \
  PetscTruth       keepzeroedrows; /* if true, MatZeroRows() will not change nonzero structure */    \
  Mat              sbaijMat;         /* mat in sbaij format */                                       \
                                                                                                     \
  PetscInt         setvalueslen;   /* only used for single precision */                              \
  MatScalar        *setvaluescopy; /* area double precision values in MatSetValuesXXX() are copied   \
                                      before calling MatSetValuesXXX_SeqBAIJ_MatScalar() */          \
                                                                                                     \
  PetscTruth       pivotinblocks;  /* pivot inside factorization of each diagonal block */           \
                                                                                                     \
  PetscInt         *xtoy,*xtoyB;     /* map nonzero pattern of X into Y's, used by MatAXPY() */      \
  Mat              XtoY;             /* used by MatAXPY() */                                         \
  PetscScalar      *idiag;           /* inverse of block diagonal  */                                \
  PetscTruth       idiagvalid;       /* if above has correct/current values */                       \
  Mat_CompressedRow compressedrow;   /* use compressed row format */

typedef struct {
  SEQBAIJHEADER
} Mat_SeqBAIJ;

EXTERN PetscErrorCode MatILUFactorSymbolic_SeqBAIJ(Mat,IS,IS,MatFactorInfo*,Mat *);
EXTERN PetscErrorCode MatICCFactorSymbolic_SeqBAIJ(Mat,IS,MatFactorInfo*,Mat *);
EXTERN PetscErrorCode MatCholeskyFactorSymbolic_SeqBAIJ(Mat,IS,MatFactorInfo*,Mat*);
EXTERN PetscErrorCode MatCholeskyFactorNumeric_SeqBAIJ_N(Mat,Mat *);
EXTERN PetscErrorCode MatCholeskyFactorNumeric_SeqBAIJ_N_NaturalOrdering(Mat,Mat *);
EXTERN PetscErrorCode MatDuplicate_SeqBAIJ(Mat,MatDuplicateOption,Mat*);
EXTERN PetscErrorCode MatMissingDiagonal_SeqBAIJ(Mat);
EXTERN PetscErrorCode MatMarkDiagonal_SeqBAIJ(Mat);

EXTERN PetscErrorCode MatLUFactorSymbolic_SeqBAIJ(Mat,IS,IS,MatFactorInfo*,Mat*);
EXTERN PetscErrorCode MatLUFactor_SeqBAIJ(Mat,IS,IS,MatFactorInfo*);
EXTERN PetscErrorCode MatIncreaseOverlap_SeqBAIJ(Mat,PetscInt,IS*,PetscInt);
EXTERN PetscErrorCode MatGetSubMatrix_SeqBAIJ(Mat,IS,IS,PetscInt,MatReuse,Mat*);
EXTERN PetscErrorCode MatGetSubMatrices_SeqBAIJ(Mat,PetscInt,const IS[],const IS[],MatReuse,Mat*[]);
EXTERN PetscErrorCode MatMultTranspose_SeqBAIJ(Mat,Vec,Vec);
EXTERN PetscErrorCode MatMultTransposeAdd_SeqBAIJ(Mat,Vec,Vec,Vec);
EXTERN PetscErrorCode MatScale_SeqBAIJ(const PetscScalar*,Mat);
EXTERN PetscErrorCode MatNorm_SeqBAIJ(Mat,NormType,PetscReal *);
EXTERN PetscErrorCode MatEqual_SeqBAIJ(Mat,Mat,PetscTruth*);
EXTERN PetscErrorCode MatGetDiagonal_SeqBAIJ(Mat,Vec);
EXTERN PetscErrorCode MatDiagonalScale_SeqBAIJ(Mat,Vec,Vec);
EXTERN PetscErrorCode MatGetInfo_SeqBAIJ(Mat,MatInfoType,MatInfo *);
EXTERN PetscErrorCode MatZeroEntries_SeqBAIJ(Mat);

EXTERN PetscErrorCode MatSeqBAIJ_UpdateFactorNumeric_NaturalOrdering(Mat);
EXTERN PetscErrorCode MatSeqBAIJ_UpdateSolvers(Mat);

EXTERN PetscErrorCode MatSolve_SeqBAIJ_Update(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolve_SeqBAIJ_1(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolve_SeqBAIJ_1_NaturalOrdering(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolve_SeqBAIJ_2(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolve_SeqBAIJ_2_NaturalOrdering(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolve_SeqBAIJ_3(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolve_SeqBAIJ_3_NaturalOrdering(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolve_SeqBAIJ_4(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolve_SeqBAIJ_4_NaturalOrdering(Mat,Vec,Vec);
#if defined(PETSC_HAVE_SSE)
EXTERN PetscErrorCode MatSolve_SeqBAIJ_4_SSE_Demotion(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolve_SeqBAIJ_4_NaturalOrdering_SSE_Demotion(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolve_SeqBAIJ_4_NaturalOrdering_SSE_Demotion_usj(Mat,Vec,Vec);
#endif
EXTERN PetscErrorCode MatSolve_SeqBAIJ_5(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolve_SeqBAIJ_5_NaturalOrdering(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolve_SeqBAIJ_6(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolve_SeqBAIJ_6_NaturalOrdering(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolve_SeqBAIJ_7(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolve_SeqBAIJ_7_NaturalOrdering(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolve_SeqBAIJ_N(Mat,Vec,Vec);

EXTERN PetscErrorCode MatSolveTranspose_SeqBAIJ_Update(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolveTranspose_SeqBAIJ_1(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolveTranspose_SeqBAIJ_1_NaturalOrdering(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolveTranspose_SeqBAIJ_2(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolveTranspose_SeqBAIJ_2_NaturalOrdering(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolveTranspose_SeqBAIJ_3(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolveTranspose_SeqBAIJ_3_NaturalOrdering(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolveTranspose_SeqBAIJ_4(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolveTranspose_SeqBAIJ_4_NaturalOrdering(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolveTranspose_SeqBAIJ_5(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolveTranspose_SeqBAIJ_5_NaturalOrdering(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolveTranspose_SeqBAIJ_6(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolveTranspose_SeqBAIJ_6_NaturalOrdering(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolveTranspose_SeqBAIJ_7(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolveTranspose_SeqBAIJ_7_NaturalOrdering(Mat,Vec,Vec);

EXTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_1(Mat,Mat*);
EXTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_2(Mat,Mat*);
EXTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_2_NaturalOrdering(Mat,Mat*);
EXTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_3(Mat,Mat*);
EXTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_3_NaturalOrdering(Mat,Mat*);
EXTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_4(Mat,Mat*);
EXTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_4_NaturalOrdering(Mat,Mat*);
#if defined(PETSC_HAVE_SSE)
EXTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_4_NaturalOrdering_SSE(Mat,Mat*);
EXTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_4_NaturalOrdering_SSE_usj(Mat,Mat*);
#else
#endif
EXTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_5(Mat,Mat*);
EXTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_5_NaturalOrdering(Mat,Mat*);
EXTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_6(Mat,Mat*);
EXTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_6_NaturalOrdering(Mat,Mat*);
EXTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_7(Mat,Mat*);
EXTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_7_NaturalOrdering(Mat,Mat*);
EXTERN PetscErrorCode MatLUFactorNumeric_SeqBAIJ_N(Mat,Mat*);

EXTERN PetscErrorCode MatMult_SeqBAIJ_1(Mat,Vec,Vec);
EXTERN PetscErrorCode MatMult_SeqBAIJ_2(Mat,Vec,Vec);
EXTERN PetscErrorCode MatMult_SeqBAIJ_3(Mat,Vec,Vec);
EXTERN PetscErrorCode MatMult_SeqBAIJ_4(Mat,Vec,Vec);
EXTERN PetscErrorCode MatMult_SeqBAIJ_5(Mat,Vec,Vec);
EXTERN PetscErrorCode MatMult_SeqBAIJ_6(Mat,Vec,Vec);
EXTERN PetscErrorCode MatMult_SeqBAIJ_7(Mat,Vec,Vec);
EXTERN PetscErrorCode MatMult_SeqBAIJ_N(Mat,Vec,Vec);

EXTERN PetscErrorCode MatMultAdd_SeqBAIJ_1(Mat,Vec,Vec,Vec);
EXTERN PetscErrorCode MatMultAdd_SeqBAIJ_2(Mat,Vec,Vec,Vec);
EXTERN PetscErrorCode MatMultAdd_SeqBAIJ_3(Mat,Vec,Vec,Vec);
EXTERN PetscErrorCode MatMultAdd_SeqBAIJ_4(Mat,Vec,Vec,Vec);
EXTERN PetscErrorCode MatMultAdd_SeqBAIJ_5(Mat,Vec,Vec,Vec);
EXTERN PetscErrorCode MatMultAdd_SeqBAIJ_6(Mat,Vec,Vec,Vec);
EXTERN PetscErrorCode MatMultAdd_SeqBAIJ_7(Mat,Vec,Vec,Vec);
EXTERN PetscErrorCode MatMultAdd_SeqBAIJ_N(Mat,Vec,Vec,Vec);
EXTERN PetscErrorCode MatLoad_SeqBAIJ(PetscViewer,const MatType,Mat*);

#endif
