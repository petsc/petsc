
#if !defined(__AIJ_H)
#define __AIJ_H
#include "src/mat/impls/csr/inode/inode.h"
/*  
  MATSEQAIJ format - Compressed row storage (also called Yale sparse matrix
  format).  The i[] and j[] arrays start at 0. For example,
  j[i[k]+p] is the pth column in row k.  Note that the diagonal
  matrix elements are stored with the rest of the nonzeros (not separately).
*/

typedef struct {
  MAT_INODE_HEADER;
  PetscTruth       sorted;           /* if true, rows are sorted by increasing columns */
  PetscTruth       roworiented;      /* if true, row-oriented input, default */
  PetscInt         nonew;            /* 1 don't add new nonzeros, -1 generate error on new */
  PetscTruth       singlemalloc;     /* if true a, i, and j have been obtained with
                                          one big malloc */
  PetscTruth       freedata;         /* free the i,j,a data when the matrix is destroyed; true by default */
  PetscInt         maxnz;            /* allocated nonzeros */
  PetscInt         *imax;            /* maximum space allocated for each row */
  PetscInt         *ilen;            /* actual length of each row */
  PetscInt         reallocs;         /* number of mallocs done during MatSetValues() 
                                        as more values are set than were prealloced */
  PetscInt         rmax;             /* max nonzeros in any row */
  PetscTruth       ilu_preserve_row_sums;
  PetscReal        lu_dtcol;
  PetscReal        lu_shift;         /* Manteuffel shift switch, fraction */
  PetscReal        lu_shift_fraction;
  PetscScalar      *saved_values;    /* location for stashing nonzero values of matrix */
  PetscScalar      *idiag,*ssor;     /* inverse of diagonal entries; space for eisen */

  PetscTruth       keepzeroedrows;   /* keeps matrix structure same in calls to MatZeroRows()*/
  PetscTruth       ignorezeroentries;
  ISColoring       coloring;         /* set with MatADSetColoring() used by MatADSetValues() */

  PetscInt         *xtoy,*xtoyB;     /* map nonzero pattern of X into Y's, used by MatAXPY() */
  Mat              XtoY;             /* used by MatAXPY() */
  Mat_CompressedRow compressedrow;   /* use compressed row format */
} Mat_SeqAIJ;

EXTERN PetscErrorCode MatILUFactorSymbolic_SeqAIJ(Mat,IS,IS,MatFactorInfo*,Mat *);
EXTERN PetscErrorCode MatICCFactorSymbolic_SeqAIJ(Mat,IS,MatFactorInfo*,Mat *);
EXTERN PetscErrorCode MatCholeskyFactorSymbolic_SeqAIJ(Mat,IS,MatFactorInfo*,Mat*);
EXTERN PetscErrorCode MatCholeskyFactorNumeric_SeqAIJ(Mat,MatFactorInfo*,Mat*);
EXTERN PetscErrorCode MatDuplicate_SeqAIJ(Mat,MatDuplicateOption,Mat*);
EXTERN PetscErrorCode MatMissingDiagonal_SeqAIJ(Mat);
EXTERN PetscErrorCode MatMarkDiagonal_SeqAIJ(Mat);

EXTERN PetscErrorCode MatMult_SeqAIJ(Mat A,Vec,Vec);
EXTERN PetscErrorCode MatMultAdd_SeqAIJ(Mat A,Vec,Vec,Vec);
EXTERN PetscErrorCode MatMultTranspose_SeqAIJ(Mat A,Vec,Vec);
EXTERN PetscErrorCode MatMultTransposeAdd_SeqAIJ(Mat A,Vec,Vec,Vec);
EXTERN PetscErrorCode MatRelax_SeqAIJ(Mat,Vec,PetscReal,MatSORType,PetscReal,PetscInt,PetscInt,Vec);

EXTERN PetscErrorCode MatSetColoring_SeqAIJ(Mat,ISColoring);
EXTERN PetscErrorCode MatSetValuesAdic_SeqAIJ(Mat,void*);
EXTERN PetscErrorCode MatSetValuesAdifor_SeqAIJ(Mat,PetscInt,void*);

EXTERN PetscErrorCode MatGetSymbolicTranspose_SeqAIJ(Mat,PetscInt *[],PetscInt *[]);
EXTERN PetscErrorCode MatGetSymbolicTransposeReduced_SeqAIJ(Mat,PetscInt,PetscInt,PetscInt *[],PetscInt *[]);
EXTERN PetscErrorCode MatRestoreSymbolicTranspose_SeqAIJ(Mat,PetscInt *[],PetscInt *[]);
EXTERN PetscErrorCode MatToSymmetricIJ_SeqAIJ(PetscInt,PetscInt*,PetscInt*,PetscInt,PetscInt,PetscInt**,PetscInt**);
EXTERN PetscErrorCode MatLUFactorSymbolic_SeqAIJ(Mat,IS,IS,MatFactorInfo*,Mat*);
EXTERN PetscErrorCode MatLUFactorNumeric_SeqAIJ(Mat,MatFactorInfo*,Mat*);
EXTERN PetscErrorCode MatLUFactor_SeqAIJ(Mat,IS,IS,MatFactorInfo*);
EXTERN PetscErrorCode MatSolve_SeqAIJ(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolveAdd_SeqAIJ(Mat,Vec,Vec,Vec);
EXTERN PetscErrorCode MatSolveTranspose_SeqAIJ(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolveTransposeAdd_SeqAIJ(Mat,Vec,Vec,Vec);
EXTERN PetscErrorCode MatEqual_SeqAIJ(Mat A,Mat B,PetscTruth* flg);
EXTERN PetscErrorCode MatFDColoringCreate_SeqAIJ(Mat,ISColoring,MatFDColoring);
EXTERN PetscErrorCode MatILUDTFactor_SeqAIJ(Mat,MatFactorInfo*,IS,IS,Mat*);
EXTERN PetscErrorCode MatLoad_SeqAIJ(PetscViewer,const MatType,Mat*);
EXTERN PetscErrorCode RegisterApplyPtAPRoutines_Private(Mat);
EXTERN PetscErrorCode MatMatMult_SeqAIJ_SeqAIJ(Mat,Mat,MatReuse,PetscReal,Mat*);
EXTERN PetscErrorCode MatMatMultSymbolic_SeqAIJ_SeqAIJ(Mat,Mat,PetscReal,Mat*);
EXTERN PetscErrorCode MatMatMultNumeric_SeqAIJ_SeqAIJ(Mat,Mat,Mat);
EXTERN PetscErrorCode MatPtAPSymbolic_SeqAIJ(Mat,Mat,PetscReal,Mat*);
EXTERN PetscErrorCode MatPtAPNumeric_SeqAIJ(Mat,Mat,Mat);
EXTERN PetscErrorCode MatPtAPSymbolic_SeqAIJ_SeqAIJ(Mat,Mat,PetscReal,Mat*);
EXTERN PetscErrorCode MatPtAPNumeric_SeqAIJ_SeqAIJ(Mat,Mat,Mat);
EXTERN PetscErrorCode MatMatMultTranspose_SeqAIJ_SeqAIJ(Mat,Mat,MatReuse,PetscReal,Mat*);
EXTERN PetscErrorCode MatMatMultTransposeSymbolic_SeqAIJ_SeqAIJ(Mat,Mat,PetscReal,Mat*);
EXTERN PetscErrorCode MatMatMultTransposeNumeric_SeqAIJ_SeqAIJ(Mat,Mat,Mat);
EXTERN PetscErrorCode MatSetValues_SeqAIJ(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],const PetscScalar[],InsertMode);
EXTERN PetscErrorCode MatGetRow_SeqAIJ(Mat,PetscInt,PetscInt*,PetscInt**,PetscScalar**);
EXTERN PetscErrorCode MatRestoreRow_SeqAIJ(Mat,PetscInt,PetscInt*,PetscInt**,PetscScalar**);
EXTERN PetscErrorCode MatPrintHelp_SeqAIJ(Mat);
EXTERN PetscErrorCode MatAXPY_SeqAIJ(const PetscScalar[],Mat,Mat,MatStructure);

EXTERN_C_BEGIN
EXTERN PetscErrorCode MatConvert_SeqAIJ_SeqSBAIJ(Mat,const MatType,MatReuse,Mat*);
EXTERN PetscErrorCode MatConvert_SeqAIJ_SeqBAIJ(Mat,const MatType,MatReuse,Mat*);
EXTERN PetscErrorCode MatReorderForNonzeroDiagonal_SeqAIJ(Mat,PetscReal,IS,IS);
EXTERN_C_END

#endif
