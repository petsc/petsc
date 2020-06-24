
#if !defined(__SBAIJ_H)
#define __SBAIJ_H
#include <petsc/private/matimpl.h>
#include <../src/mat/impls/baij/seq/baij.h>

/*
  MATSEQSBAIJ format - Block compressed row storage. The i[] and j[]
  arrays start at 0.
*/

typedef struct {
  SEQAIJHEADER(MatScalar);
  SEQBAIJHEADER;
  PetscInt         *inew;        /* pointer to beginning of each row of reordered matrix */
  PetscInt         *jnew;        /* column values: jnew + i[k] is start of row k */
  MatScalar        *anew;        /* nonzero diagonal and superdiagonal elements of reordered matrix */
  PetscScalar      *solves_work; /* work space used in MatSolves */
  PetscInt         solves_work_n; /* size of solves_work */
  PetscInt         *a2anew;        /* map used for symm permutation */
  PetscBool        permute;        /* if true, a non-trivial permutation is used for factorization */
  PetscBool        ignore_ltriangular; /* if true, ignore the lower triangular values inserted by users */
  PetscBool        getrow_utriangular; /* if true, MatGetRow_SeqSBAIJ() is enabled to get the upper part of the row */
  Mat_SeqAIJ_Inode inode;
  unsigned short   *jshort;
  PetscBool        free_jshort;
} Mat_SeqSBAIJ;

PETSC_INTERN PetscErrorCode MatCholeskyFactorSymbolic_SeqSBAIJ(Mat,Mat,IS,const MatFactorInfo*);
PETSC_INTERN PetscErrorCode MatCholeskyFactorSymbolic_SeqSBAIJ_inplace(Mat,Mat,IS,const MatFactorInfo*);
PETSC_INTERN PetscErrorCode MatCholeskyFactor_SeqSBAIJ(Mat,IS,const MatFactorInfo*);
PETSC_INTERN PetscErrorCode MatICCFactorSymbolic_SeqSBAIJ(Mat,Mat,IS,const MatFactorInfo*);
PETSC_INTERN PetscErrorCode MatICCFactorSymbolic_SeqSBAIJ_inplace(Mat,Mat,IS,const MatFactorInfo*);
PETSC_INTERN PetscErrorCode MatDuplicate_SeqSBAIJ(Mat,MatDuplicateOption,Mat*);
PETSC_INTERN PetscErrorCode MatMarkDiagonal_SeqSBAIJ(Mat);
PETSC_INTERN PetscErrorCode MatIncreaseOverlap_SeqSBAIJ(Mat,PetscInt,IS[],PetscInt);
PETSC_INTERN PetscErrorCode MatSeqSBAIJZeroOps_Private(Mat);
PETSC_INTERN PetscErrorCode MatCreateSubMatrix_SeqSBAIJ(Mat,IS,IS,MatReuse,Mat*);
PETSC_INTERN PetscErrorCode MatCreateSubMatrices_SeqSBAIJ(Mat,PetscInt,const IS[],const IS[],MatReuse,Mat*[]);
PETSC_INTERN PetscErrorCode MatScale_SeqSBAIJ(Mat,PetscScalar);
PETSC_INTERN PetscErrorCode MatNorm_SeqSBAIJ(Mat,NormType,PetscReal*);
PETSC_INTERN PetscErrorCode MatEqual_SeqSBAIJ(Mat,Mat,PetscBool*);
PETSC_INTERN PetscErrorCode MatGetDiagonal_SeqSBAIJ(Mat,Vec);
PETSC_INTERN PetscErrorCode MatDiagonalScale_SeqSBAIJ(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatGetInfo_SeqSBAIJ(Mat,MatInfoType,MatInfo*);
PETSC_INTERN PetscErrorCode MatZeroEntries_SeqSBAIJ(Mat);
PETSC_INTERN PetscErrorCode MatGetRowMaxAbs_SeqSBAIJ(Mat,Vec,PetscInt[]);
PETSC_INTERN PetscErrorCode MatGetInertia_SeqSBAIJ(Mat,PetscInt*,PetscInt*,PetscInt*);
PETSC_INTERN PetscErrorCode MatDestroy_SeqSBAIJ(Mat);
PETSC_INTERN PetscErrorCode MatView_SeqSBAIJ(Mat,PetscViewer);

PETSC_INTERN PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_1_NaturalOrdering(Mat,Mat,const MatFactorInfo*);
PETSC_INTERN PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_1_NaturalOrdering_inplace(Mat,Mat,const MatFactorInfo*);
PETSC_INTERN PetscErrorCode MatSolve_SeqSBAIJ_1_NaturalOrdering_inplace(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatSolve_SeqSBAIJ_1_NaturalOrdering(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatMatSolve_SeqSBAIJ_1_NaturalOrdering(Mat,Mat,Mat);

PETSC_INTERN PetscErrorCode MatForwardSolve_SeqSBAIJ_1_NaturalOrdering_inplace(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatBackwardSolve_SeqSBAIJ_1_NaturalOrdering_inplace(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatForwardSolve_SeqSBAIJ_1_inplace(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatBackwardSolve_SeqSBAIJ_1_inplace(Mat,Vec,Vec);

PETSC_INTERN PetscErrorCode MatForwardSolve_SeqSBAIJ_1_NaturalOrdering(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatBackwardSolve_SeqSBAIJ_1_NaturalOrdering(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatForwardSolve_SeqSBAIJ_1(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatBackwardSolve_SeqSBAIJ_1(Mat,Vec,Vec);

PETSC_INTERN PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_2_NaturalOrdering(Mat,Mat,const MatFactorInfo*);
PETSC_INTERN PetscErrorCode MatSolve_SeqSBAIJ_2_NaturalOrdering_inplace(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatForwardSolve_SeqSBAIJ_2_NaturalOrdering_inplace(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatBackwardSolve_SeqSBAIJ_2_NaturalOrdering_inplace(Mat,Vec,Vec);

PETSC_INTERN PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_3_NaturalOrdering(Mat,Mat,const MatFactorInfo*);
PETSC_INTERN PetscErrorCode MatSolve_SeqSBAIJ_3_NaturalOrdering_inplace(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatForwardSolve_SeqSBAIJ_3_NaturalOrdering_inplace(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatBackwardSolve_SeqSBAIJ_3_NaturalOrdering_inplace(Mat,Vec,Vec);

PETSC_INTERN PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_4_NaturalOrdering(Mat,Mat,const MatFactorInfo*);
PETSC_INTERN PetscErrorCode MatSolve_SeqSBAIJ_4_NaturalOrdering_inplace(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatForwardSolve_SeqSBAIJ_4_NaturalOrdering_inplace(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatBackwardSolve_SeqSBAIJ_4_NaturalOrdering_inplace(Mat,Vec,Vec);

PETSC_INTERN PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_5_NaturalOrdering(Mat,Mat,const MatFactorInfo*);
PETSC_INTERN PetscErrorCode MatSolve_SeqSBAIJ_5_NaturalOrdering_inplace(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatForwardSolve_SeqSBAIJ_5_NaturalOrdering_inplace(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatBackwardSolve_SeqSBAIJ_5_NaturalOrdering_inplace(Mat,Vec,Vec);

PETSC_INTERN PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_6_NaturalOrdering(Mat,Mat,const MatFactorInfo*);
PETSC_INTERN PetscErrorCode MatSolve_SeqSBAIJ_6_NaturalOrdering_inplace(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatForwardSolve_SeqSBAIJ_6_NaturalOrdering_inplace(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatBackwardSolve_SeqSBAIJ_6_NaturalOrdering_inplace(Mat,Vec,Vec);

PETSC_INTERN PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_7_NaturalOrdering(Mat,Mat,const MatFactorInfo*);
PETSC_INTERN PetscErrorCode MatSolve_SeqSBAIJ_7_NaturalOrdering_inplace(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatForwardSolve_SeqSBAIJ_7_NaturalOrdering_inplace(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatBackwardSolve_SeqSBAIJ_7_NaturalOrdering_inplace(Mat,Vec,Vec);

PETSC_INTERN PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_N_NaturalOrdering(Mat,Mat,const MatFactorInfo*);
PETSC_INTERN PetscErrorCode MatSolve_SeqSBAIJ_N_NaturalOrdering_inplace(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatForwardSolve_SeqSBAIJ_N_NaturalOrdering_inplace(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatBackwardSolve_SeqSBAIJ_N_NaturalOrdering_inplace(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatForwardSolve_SeqSBAIJ_N_inplace(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatBackwardSolve_SeqSBAIJ_N_inplace(Mat,Vec,Vec);

PETSC_INTERN PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_N(Mat,Mat,const MatFactorInfo*);
PETSC_INTERN PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_1_inplace(Mat,Mat,const MatFactorInfo*);
PETSC_INTERN PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_2(Mat,Mat,const MatFactorInfo*);
PETSC_INTERN PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_3(Mat,Mat,const MatFactorInfo*);
PETSC_INTERN PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_4(Mat,Mat,const MatFactorInfo*);
PETSC_INTERN PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_5(Mat,Mat,const MatFactorInfo*);
PETSC_INTERN PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_6(Mat,Mat,const MatFactorInfo*);
PETSC_INTERN PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_7(Mat,Mat,const MatFactorInfo*);

PETSC_INTERN PetscErrorCode MatSolve_SeqSBAIJ_N_inplace(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatSolve_SeqSBAIJ_1_inplace(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatSolve_SeqSBAIJ_1(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatSolve_SeqSBAIJ_2_inplace(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatSolve_SeqSBAIJ_3_inplace(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatSolve_SeqSBAIJ_4_inplace(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatSolve_SeqSBAIJ_5_inplace(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatSolve_SeqSBAIJ_6_inplace(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatSolve_SeqSBAIJ_7_inplace(Mat,Vec,Vec);

PETSC_INTERN PetscErrorCode MatSolves_SeqSBAIJ_1_inplace(Mat,Vecs,Vecs);
PETSC_INTERN PetscErrorCode MatSolves_SeqSBAIJ_1(Mat,Vecs,Vecs);

PETSC_INTERN PetscErrorCode MatMult_SeqSBAIJ_1(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatMult_SeqSBAIJ_2(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatMult_SeqSBAIJ_3(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatMult_SeqSBAIJ_4(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatMult_SeqSBAIJ_5(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatMult_SeqSBAIJ_6(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatMult_SeqSBAIJ_7(Mat,Vec,Vec);
PETSC_INTERN PetscErrorCode MatMult_SeqSBAIJ_N(Mat,Vec,Vec);

PETSC_INTERN PetscErrorCode MatMultAdd_SeqSBAIJ_1(Mat,Vec,Vec,Vec);
PETSC_INTERN PetscErrorCode MatMultAdd_SeqSBAIJ_2(Mat,Vec,Vec,Vec);
PETSC_INTERN PetscErrorCode MatMultAdd_SeqSBAIJ_3(Mat,Vec,Vec,Vec);
PETSC_INTERN PetscErrorCode MatMultAdd_SeqSBAIJ_4(Mat,Vec,Vec,Vec);
PETSC_INTERN PetscErrorCode MatMultAdd_SeqSBAIJ_5(Mat,Vec,Vec,Vec);
PETSC_INTERN PetscErrorCode MatMultAdd_SeqSBAIJ_6(Mat,Vec,Vec,Vec);
PETSC_INTERN PetscErrorCode MatMultAdd_SeqSBAIJ_7(Mat,Vec,Vec,Vec);
PETSC_INTERN PetscErrorCode MatMultAdd_SeqSBAIJ_N(Mat,Vec,Vec,Vec);

PETSC_INTERN PetscErrorCode MatSOR_SeqSBAIJ(Mat,Vec,PetscReal,MatSORType,PetscReal,PetscInt,PetscInt,Vec);
PETSC_INTERN PetscErrorCode MatLoad_SeqSBAIJ(Mat,PetscViewer);
PETSC_INTERN PetscErrorCode MatSeqSBAIJSetNumericFactorization_inplace(Mat,PetscBool);

PETSC_INTERN PetscErrorCode MatAXPYGetPreallocation_SeqSBAIJ(Mat,Mat,PetscInt*);

PETSC_INTERN PetscErrorCode MatCreateMPIMatConcatenateSeqMat_SeqSBAIJ(MPI_Comm,Mat,PetscInt,MatReuse,Mat*);
PETSC_INTERN PetscErrorCode MatCreateMPIMatConcatenateSeqMat_MPISBAIJ(MPI_Comm,Mat,PetscInt,MatReuse,Mat*);
/* required by mpisbaij.c */
PETSC_INTERN PetscErrorCode MatGetValues_SeqSBAIJ(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],PetscScalar []);
PETSC_INTERN PetscErrorCode MatSetValues_SeqSBAIJ(Mat,PetscInt,const PetscInt [],PetscInt,const PetscInt [],const PetscScalar [],InsertMode);
PETSC_INTERN PetscErrorCode MatSetValuesBlocked_SeqSBAIJ(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],const PetscScalar[],InsertMode);
PETSC_INTERN PetscErrorCode MatGetRow_SeqSBAIJ(Mat,PetscInt,PetscInt*,PetscInt**,PetscScalar**);
PETSC_INTERN PetscErrorCode MatRestoreRow_SeqSBAIJ(Mat,PetscInt,PetscInt*,PetscInt**,PetscScalar**);
PETSC_INTERN PetscErrorCode MatZeroRows_SeqSBAIJ(Mat,IS,PetscScalar*,Vec,Vec);

#endif
