
#if !defined(__SBAIJ_H)
#define __SBAIJ_H
#include "src/mat/matimpl.h"
#include "src/mat/impls/baij/seq/baij.h"

/*  
  MATSEQSBAIJ format - Block compressed row storage. The i[] and j[] 
  arrays start at 0.
*/

typedef struct {
  SEQAIJHEADER(PetscScalar);
  SEQBAIJHEADER;
  PetscInt         *inew;        /* pointer to beginning of each row of reordered matrix */
  PetscInt         *jnew;        /* column values: jnew + i[k] is start of row k */
  MatScalar        *anew;        /* nonzero diagonal and superdiagonal elements of reordered matrix */
  PetscScalar      *solves_work; /* work space used in MatSolves */
  PetscInt         solves_work_n;/* size of solves_work */  
  PetscScalar      *relax_work;  
  PetscInt         *a2anew;        /* map used for symm permutation */
  PetscTruth       permute;        /* if true, a non-trivial permutation is used for factorization */
  PetscTruth       ignore_ltriangular; /* if true, ignore the lower triangular values inserted by users */
  PetscTruth       getrow_utriangular; /* if true, MatGetRow_SeqSBAIJ() is enabled to get the upper part of the row */
} Mat_SeqSBAIJ;

EXTERN_C_BEGIN
EXTERN PetscErrorCode MatSeqSBAIJSetPreallocation_SeqSBAIJ(Mat,PetscInt,PetscInt,PetscInt*);
EXTERN_C_END
EXTERN PetscErrorCode MatICCFactorSymbolic_SeqSBAIJ(Mat,IS,MatFactorInfo*,Mat *);
EXTERN PetscErrorCode MatDuplicate_SeqSBAIJ(Mat,MatDuplicateOption,Mat*);
EXTERN PetscErrorCode MatMarkDiagonal_SeqSBAIJ(Mat);

EXTERN PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_1_NaturalOrdering(Mat,MatFactorInfo*,Mat*);
EXTERN PetscErrorCode MatSolve_SeqSBAIJ_1_NaturalOrdering(Mat,Vec,Vec);
EXTERN PetscErrorCode MatForwardSolve_SeqSBAIJ_1_NaturalOrdering(Mat,Vec,Vec);
EXTERN PetscErrorCode MatBackwardSolve_SeqSBAIJ_1_NaturalOrdering(Mat,Vec,Vec);

EXTERN PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_2_NaturalOrdering(Mat,MatFactorInfo*,Mat*);
EXTERN PetscErrorCode MatSolve_SeqSBAIJ_2_NaturalOrdering(Mat,Vec,Vec);

EXTERN PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_3_NaturalOrdering(Mat,MatFactorInfo*,Mat*);
EXTERN PetscErrorCode MatSolve_SeqSBAIJ_3_NaturalOrdering(Mat,Vec,Vec);

EXTERN PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_4_NaturalOrdering(Mat,MatFactorInfo*,Mat*);
EXTERN PetscErrorCode MatSolve_SeqSBAIJ_4_NaturalOrdering(Mat,Vec,Vec);

EXTERN PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_5_NaturalOrdering(Mat,MatFactorInfo*,Mat*);
EXTERN PetscErrorCode MatSolve_SeqSBAIJ_5_NaturalOrdering(Mat,Vec,Vec);

EXTERN PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_6_NaturalOrdering(Mat,MatFactorInfo*,Mat*);
EXTERN PetscErrorCode MatSolve_SeqSBAIJ_6_NaturalOrdering(Mat,Vec,Vec);

EXTERN PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_7_NaturalOrdering(Mat,MatFactorInfo*,Mat*);
EXTERN PetscErrorCode MatSolve_SeqSBAIJ_7_NaturalOrdering(Mat,Vec,Vec);

EXTERN PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_N_NaturalOrdering(Mat,MatFactorInfo*,Mat*);
EXTERN PetscErrorCode MatSolve_SeqSBAIJ_N_NaturalOrdering(Mat,Vec,Vec);

EXTERN PetscErrorCode MatRelax_SeqSBAIJ(Mat,Vec,PetscReal,MatSORType,PetscReal,PetscInt,PetscInt,Vec);
EXTERN PetscErrorCode MatLoad_SeqSBAIJ(PetscViewer, MatType,Mat*);

#endif
