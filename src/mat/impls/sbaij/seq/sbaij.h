
#if !defined(__SBAIJ_H)
#define __SBAIJ_H
#include "src/mat/matimpl.h"
#include "src/mat/impls/baij/seq/baij.h"

/*  
  MATSEQSBAIJ format - Block compressed row storage. The i[] and j[] 
  arrays start at 0.
*/

typedef struct {
  SEQBAIJHEADER

  int              *inew;        /* pointer to beginning of each row of reordered matrix */
  int              *jnew;        /* column values: jnew + i[k] is start of row k */
  MatScalar        *anew;        /* nonzero diagonal and superdiagonal elements of reordered matrix */
  PetscScalar      *solves_work; /* work space used in MatSolves */
  int              solves_work_n;/* size of solves_work */  
  int              *a2anew;        /* map used for symm permutation */
  PetscTruth       permute;        /* if true, a non-trivial permutation is used for factorization */

  /* carry MatFactorInfo from symbolic factor to numeric factor */
  int              factor_levels;
  PetscReal        factor_damping;     
  PetscReal        factor_shift;
  PetscReal        factor_zeropivot;
} Mat_SeqSBAIJ;

EXTERN PetscErrorCode MatICCFactorSymbolic_SeqSBAIJ(Mat,IS,MatFactorInfo*,Mat *);
EXTERN PetscErrorCode MatDuplicate_SeqSBAIJ(Mat,MatDuplicateOption,Mat*);
EXTERN PetscErrorCode MatMarkDiagonal_SeqSBAIJ(Mat);

EXTERN PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_1_NaturalOrdering(Mat,Mat*);
EXTERN PetscErrorCode MatSolve_SeqSBAIJ_1_NaturalOrdering(Mat,Vec,Vec);

EXTERN PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_2_NaturalOrdering(Mat,Mat*);
EXTERN PetscErrorCode MatSolve_SeqSBAIJ_2_NaturalOrdering(Mat,Vec,Vec);

EXTERN PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_3_NaturalOrdering(Mat,Mat*);
EXTERN PetscErrorCode MatSolve_SeqSBAIJ_3_NaturalOrdering(Mat,Vec,Vec);

EXTERN PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_4_NaturalOrdering(Mat,Mat*);
EXTERN PetscErrorCode MatSolve_SeqSBAIJ_4_NaturalOrdering(Mat,Vec,Vec);

EXTERN PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_5_NaturalOrdering(Mat,Mat*);
EXTERN PetscErrorCode MatSolve_SeqSBAIJ_5_NaturalOrdering(Mat,Vec,Vec);

EXTERN PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_6_NaturalOrdering(Mat,Mat*);
EXTERN PetscErrorCode MatSolve_SeqSBAIJ_6_NaturalOrdering(Mat,Vec,Vec);

EXTERN PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_7_NaturalOrdering(Mat,Mat*);
EXTERN PetscErrorCode MatSolve_SeqSBAIJ_7_NaturalOrdering(Mat,Vec,Vec);

EXTERN PetscErrorCode MatCholeskyFactorNumeric_SeqSBAIJ_N_NaturalOrdering(Mat,Mat*);
EXTERN PetscErrorCode MatSolve_SeqSBAIJ_N_NaturalOrdering(Mat,Vec,Vec);

EXTERN PetscErrorCode MatRelax_SeqSBAIJ(Mat,Vec,PetscReal,MatSORType,PetscReal,int,int,Vec);
EXTERN PetscErrorCode MatLoad_SeqSBAIJ(PetscViewer,const MatType,Mat*);

#endif
