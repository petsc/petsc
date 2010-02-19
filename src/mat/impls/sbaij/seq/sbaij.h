
#if !defined(__SBAIJ_H)
#define __SBAIJ_H
#include "private/matimpl.h"
#include "../src/mat/impls/baij/seq/baij.h"

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
  PetscInt         solves_work_n;/* size of solves_work */  
  PetscScalar      *sor_work;  
  PetscInt         *a2anew;        /* map used for symm permutation */
  PetscTruth       permute;        /* if true, a non-trivial permutation is used for factorization */
  PetscTruth       ignore_ltriangular; /* if true, ignore the lower triangular values inserted by users */
  PetscTruth       getrow_utriangular; /* if true, MatGetRow_SeqSBAIJ() is enabled to get the upper part of the row */
  Mat_SeqAIJ_Inode inode;
  unsigned short   *jshort;
  PetscTruth       free_jshort;
} Mat_SeqSBAIJ;

EXTERN_C_BEGIN
EXTERN PetscErrorCode MatSeqSBAIJSetPreallocation_SeqSBAIJ(Mat,PetscInt,PetscInt,PetscInt*);
EXTERN_C_END
EXTERN PetscErrorCode MatCholeskyFactorSymbolic_SeqSBAIJ(Mat,Mat,IS,const MatFactorInfo*);
EXTERN PetscErrorCode MatCholeskyFactorSymbolic_SeqSBAIJ_inplace(Mat,Mat,IS,const MatFactorInfo*);
EXTERN PetscErrorCode MatCholeskyFactor_SeqSBAIJ(Mat,IS,const MatFactorInfo*);
EXTERN PetscErrorCode MatICCFactorSymbolic_SeqSBAIJ(Mat,Mat,IS,const MatFactorInfo*);
EXTERN PetscErrorCode MatICCFactorSymbolic_SeqSBAIJ_inplace(Mat,Mat,IS,const MatFactorInfo*);
EXTERN PetscErrorCode MatDuplicate_SeqSBAIJ(Mat,MatDuplicateOption,Mat*);
EXTERN PetscErrorCode MatMarkDiagonal_SeqSBAIJ(Mat);
EXTERN PetscErrorCode MatIncreaseOverlap_SeqSBAIJ(Mat,PetscInt,IS[],PetscInt);
EXTERN PetscErrorCode MatGetSubMatrix_SeqSBAIJ(Mat,IS,IS,MatReuse,Mat*);
EXTERN PetscErrorCode MatGetSubMatrices_SeqSBAIJ(Mat,PetscInt,const IS[],const IS[],MatReuse,Mat*[]);
EXTERN PetscErrorCode MatScale_SeqSBAIJ(Mat,PetscScalar);
EXTERN PetscErrorCode MatNorm_SeqSBAIJ(Mat,NormType,PetscReal *);
EXTERN PetscErrorCode MatEqual_SeqSBAIJ(Mat,Mat,PetscTruth*);
EXTERN PetscErrorCode MatGetDiagonal_SeqSBAIJ(Mat,Vec);
EXTERN PetscErrorCode MatDiagonalScale_SeqSBAIJ(Mat,Vec,Vec);
EXTERN PetscErrorCode MatGetInfo_SeqSBAIJ(Mat,MatInfoType,MatInfo *);
EXTERN PetscErrorCode MatZeroEntries_SeqSBAIJ(Mat);
EXTERN PetscErrorCode MatGetRowMaxAbs_SeqSBAIJ(Mat,Vec,PetscInt[]);
EXTERN PetscErrorCode MatGetInertia_SeqSBAIJ(Mat,PetscInt*,PetscInt*,PetscInt*);

EXTERN PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_1_NaturalOrdering(Mat,Mat,const MatFactorInfo*);
EXTERN PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_1_NaturalOrdering_inplace(Mat,Mat,const MatFactorInfo*);
EXTERN PetscErrorCode MatSolve_SeqSBAIJ_1_NaturalOrdering_inplace(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolve_SeqSBAIJ_1_NaturalOrdering(Mat,Vec,Vec);

EXTERN PetscErrorCode MatForwardSolve_SeqSBAIJ_1_NaturalOrdering_inplace(Mat,Vec,Vec);
EXTERN PetscErrorCode MatBackwardSolve_SeqSBAIJ_1_NaturalOrdering_inplace(Mat,Vec,Vec);
EXTERN PetscErrorCode MatForwardSolve_SeqSBAIJ_1_inplace(Mat,Vec,Vec);
EXTERN PetscErrorCode MatBackwardSolve_SeqSBAIJ_1_inplace(Mat,Vec,Vec);

EXTERN PetscErrorCode MatForwardSolve_SeqSBAIJ_1_NaturalOrdering(Mat,Vec,Vec);
EXTERN PetscErrorCode MatBackwardSolve_SeqSBAIJ_1_NaturalOrdering(Mat,Vec,Vec);
EXTERN PetscErrorCode MatForwardSolve_SeqSBAIJ_1(Mat,Vec,Vec);
EXTERN PetscErrorCode MatBackwardSolve_SeqSBAIJ_1(Mat,Vec,Vec);

EXTERN PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_2_NaturalOrdering(Mat,Mat,const MatFactorInfo*);
EXTERN PetscErrorCode MatSolve_SeqSBAIJ_2_NaturalOrdering_inplace(Mat,Vec,Vec);
EXTERN PetscErrorCode MatForwardSolve_SeqSBAIJ_2_NaturalOrdering_inplace(Mat,Vec,Vec);
EXTERN PetscErrorCode MatBackwardSolve_SeqSBAIJ_2_NaturalOrdering_inplace(Mat,Vec,Vec);

EXTERN PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_3_NaturalOrdering(Mat,Mat,const MatFactorInfo*);
EXTERN PetscErrorCode MatSolve_SeqSBAIJ_3_NaturalOrdering_inplace(Mat,Vec,Vec);
EXTERN PetscErrorCode MatForwardSolve_SeqSBAIJ_3_NaturalOrdering_inplace(Mat,Vec,Vec);
EXTERN PetscErrorCode MatBackwardSolve_SeqSBAIJ_3_NaturalOrdering_inplace(Mat,Vec,Vec);

EXTERN PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_4_NaturalOrdering(Mat,Mat,const MatFactorInfo*);
EXTERN PetscErrorCode MatSolve_SeqSBAIJ_4_NaturalOrdering_inplace(Mat,Vec,Vec);
EXTERN PetscErrorCode MatForwardSolve_SeqSBAIJ_4_NaturalOrdering_inplace(Mat,Vec,Vec);
EXTERN PetscErrorCode MatBackwardSolve_SeqSBAIJ_4_NaturalOrdering_inplace(Mat,Vec,Vec);

EXTERN PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_5_NaturalOrdering(Mat,Mat,const MatFactorInfo*);
EXTERN PetscErrorCode MatSolve_SeqSBAIJ_5_NaturalOrdering_inplace(Mat,Vec,Vec);
EXTERN PetscErrorCode MatForwardSolve_SeqSBAIJ_5_NaturalOrdering_inplace(Mat,Vec,Vec);
EXTERN PetscErrorCode MatBackwardSolve_SeqSBAIJ_5_NaturalOrdering_inplace(Mat,Vec,Vec);

EXTERN PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_6_NaturalOrdering(Mat,Mat,const MatFactorInfo*);
EXTERN PetscErrorCode MatSolve_SeqSBAIJ_6_NaturalOrdering_inplace(Mat,Vec,Vec);
EXTERN PetscErrorCode MatForwardSolve_SeqSBAIJ_6_NaturalOrdering_inplace(Mat,Vec,Vec);
EXTERN PetscErrorCode MatBackwardSolve_SeqSBAIJ_6_NaturalOrdering_inplace(Mat,Vec,Vec);

EXTERN PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_7_NaturalOrdering(Mat,Mat,const MatFactorInfo*);
EXTERN PetscErrorCode MatSolve_SeqSBAIJ_7_NaturalOrdering_inplace(Mat,Vec,Vec);
EXTERN PetscErrorCode MatForwardSolve_SeqSBAIJ_7_NaturalOrdering_inplace(Mat,Vec,Vec);
EXTERN PetscErrorCode MatBackwardSolve_SeqSBAIJ_7_NaturalOrdering_inplace(Mat,Vec,Vec);

EXTERN PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_N_NaturalOrdering(Mat,Mat,const MatFactorInfo*);
EXTERN PetscErrorCode MatSolve_SeqSBAIJ_N_NaturalOrdering_inplace(Mat,Vec,Vec);
EXTERN PetscErrorCode MatForwardSolve_SeqSBAIJ_N_NaturalOrdering_inplace(Mat,Vec,Vec);
EXTERN PetscErrorCode MatBackwardSolve_SeqSBAIJ_N_NaturalOrdering_inplace(Mat,Vec,Vec);
EXTERN PetscErrorCode MatForwardSolve_SeqSBAIJ_N_inplace(Mat,Vec,Vec);
EXTERN PetscErrorCode MatBackwardSolve_SeqSBAIJ_N_inplace(Mat,Vec,Vec);

EXTERN PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_N(Mat,Mat,const MatFactorInfo*);
EXTERN PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_1_inplace(Mat,Mat,const MatFactorInfo*);
EXTERN PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_2(Mat,Mat,const MatFactorInfo*);
EXTERN PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_3(Mat,Mat,const MatFactorInfo*);
EXTERN PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_4(Mat,Mat,const MatFactorInfo*);
EXTERN PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_5(Mat,Mat,const MatFactorInfo*);
EXTERN PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_6(Mat,Mat,const MatFactorInfo*);
EXTERN PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_7(Mat,Mat,const MatFactorInfo*);

EXTERN PetscErrorCode MatSolve_SeqSBAIJ_N_inplace(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolve_SeqSBAIJ_1_inplace(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolve_SeqSBAIJ_1(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolve_SeqSBAIJ_2_inplace(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolve_SeqSBAIJ_3_inplace(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolve_SeqSBAIJ_4_inplace(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolve_SeqSBAIJ_5_inplace(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolve_SeqSBAIJ_6_inplace(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolve_SeqSBAIJ_7_inplace(Mat,Vec,Vec);

EXTERN PetscErrorCode MatSolves_SeqSBAIJ_1_inplace(Mat,Vecs,Vecs);
EXTERN PetscErrorCode MatSolves_SeqSBAIJ_1(Mat,Vecs,Vecs);

EXTERN PetscErrorCode MatSolve_SeqSBAIJ_1_NaturalOrdering_inplace(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolve_SeqSBAIJ_1_NaturalOrdering(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolve_SeqSBAIJ_2_NaturalOrdering_inplace(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolve_SeqSBAIJ_3_NaturalOrdering_inplace(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolve_SeqSBAIJ_4_NaturalOrdering_inplace(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolve_SeqSBAIJ_5_NaturalOrdering_inplace(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolve_SeqSBAIJ_6_NaturalOrdering_inplace(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolve_SeqSBAIJ_7_NaturalOrdering_inplace(Mat,Vec,Vec);
EXTERN PetscErrorCode MatSolve_SeqSBAIJ_N_NaturalOrdering_inplace(Mat,Vec,Vec);

EXTERN PetscErrorCode MatMult_SeqSBAIJ_1(Mat,Vec,Vec);
EXTERN PetscErrorCode MatMult_SeqSBAIJ_2(Mat,Vec,Vec);
EXTERN PetscErrorCode MatMult_SeqSBAIJ_3(Mat,Vec,Vec);
EXTERN PetscErrorCode MatMult_SeqSBAIJ_4(Mat,Vec,Vec);
EXTERN PetscErrorCode MatMult_SeqSBAIJ_5(Mat,Vec,Vec);
EXTERN PetscErrorCode MatMult_SeqSBAIJ_6(Mat,Vec,Vec);
EXTERN PetscErrorCode MatMult_SeqSBAIJ_7(Mat,Vec,Vec);
EXTERN PetscErrorCode MatMult_SeqSBAIJ_N(Mat,Vec,Vec);

EXTERN PetscErrorCode MatMult_SeqSBAIJ_1_Hermitian(Mat,Vec,Vec);

EXTERN PetscErrorCode MatMultAdd_SeqSBAIJ_1(Mat,Vec,Vec,Vec);
EXTERN PetscErrorCode MatMultAdd_SeqSBAIJ_2(Mat,Vec,Vec,Vec);
EXTERN PetscErrorCode MatMultAdd_SeqSBAIJ_3(Mat,Vec,Vec,Vec);
EXTERN PetscErrorCode MatMultAdd_SeqSBAIJ_4(Mat,Vec,Vec,Vec);
EXTERN PetscErrorCode MatMultAdd_SeqSBAIJ_5(Mat,Vec,Vec,Vec);
EXTERN PetscErrorCode MatMultAdd_SeqSBAIJ_6(Mat,Vec,Vec,Vec);
EXTERN PetscErrorCode MatMultAdd_SeqSBAIJ_7(Mat,Vec,Vec,Vec);
EXTERN PetscErrorCode MatMultAdd_SeqSBAIJ_N(Mat,Vec,Vec,Vec);

EXTERN PetscErrorCode MatSOR_SeqSBAIJ(Mat,Vec,PetscReal,MatSORType,PetscReal,PetscInt,PetscInt,Vec);
EXTERN PetscErrorCode MatLoad_SeqSBAIJ(PetscViewer, const MatType,Mat*);

EXTERN PetscErrorCode MatSeqSBAIJSetNumericFactorization_inplace(Mat,PetscTruth);

#endif
