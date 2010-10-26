#include "private/matimpl.h"
#include "private/vecimpl.h"

typedef struct{

  Mat A;
  Vec D1;
  Vec D2;
  Vec W;
  Vec W2;
  Vec ADADiag;
  PetscInt GotDiag;

} _p_TaoMatADACtx;

typedef  _p_TaoMatADACtx* TaoMatADACtx;

PetscErrorCode MatCreateADA(Mat,Vec,Vec,Mat*);
PetscErrorCode MatMult_ADA(Mat,Vec,Vec);
PetscErrorCode MatMultTranspose_ADA(Mat,Vec,Vec);
PetscErrorCode MatDiagonalSet_ADA(Vec,Mat);
PetscErrorCode MatDestroy_ADA(Mat);
PetscErrorCode MatView_ADA(Mat,PetscViewer);
PetscErrorCode MatShift_ADA(Mat,PetscReal);
PetscErrorCode MatDuplicate_ADA(Mat,MatDuplicateOption,Mat*);
PetscErrorCode MatEqual_ADA(Mat,Mat,PetscBool*);
PetscErrorCode MatScale_ADA(Mat,PetscReal);
PetscErrorCode MatGetSubMatrix_ADA(Mat,IS,IS,MatReuse,Mat *);
PetscErrorCode MatGetSubMatrices_ADA(Mat,PetscInt,IS*,IS*,MatReuse,Mat**);
PetscErrorCode MatTranspose_ADA(Mat,Mat*);
PetscErrorCode MatGetDiagonal_ADA(Mat,Vec);
PetscErrorCode MatGetColumnVector_ADA(Mat,Vec, PetscInt);
PetscErrorCode MatNorm_ADA(Mat,NormType,PetscReal *);
PetscErrorCode MatADAComputeDiagonal(Mat);

