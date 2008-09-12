#include "private/matimpl.h"
#include "private/vecimpl.h"

typedef struct{

  Mat A;
  Vec D1;
  Vec D2;
  Vec W;
  Vec W2;
  Vec ADADiag;
  int GotDiag;

} _p_TaoMatADACtx;

typedef  _p_TaoMatADACtx* TaoMatADACtx;

int MatCreateADA(Mat,Vec,Vec,Mat*);
int MatMult_ADA(Mat,Vec,Vec);
int MatMultTranspose_ADA(Mat,Vec,Vec);
int MatDiagonalShift_ADA(Vec,Mat);
int MatDestroy_ADA(Mat);
int MatView_ADA(Mat,PetscViewer);
int MatShift_ADA(Mat,PetscReal);
int MatDuplicate_ADA(Mat,MatDuplicateOption,Mat*);
int MatEqual_ADA(Mat,Mat,PetscTruth*);
int MatScale_ADA(Mat,PetscReal);
int MatGetSubMatrix_ADA(Mat,IS,IS,int,MatReuse,Mat *);
int MatGetSubMatrices_ADA(Mat,int,IS*,IS*,MatReuse,Mat**);
int MatTranspose_ADA(Mat,Mat*);
int MatGetDiagonal_ADA(Mat,Vec);
int MatGetColumnVector_ADA(Mat,Vec, int);
int MatNorm_ADA(Mat,NormType,PetscReal *);
int MatADAComputeDiagonal(Mat);

