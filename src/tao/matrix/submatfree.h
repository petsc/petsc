
typedef struct {
  Mat A;
  Vec VC; /* Work vector in the column space */
  Vec VR; /* Work vector in the row space */
  IS  Rows, Cols;

} _p_MatSubMatFreeCtx;

typedef _p_MatSubMatFreeCtx *MatSubMatFreeCtx;

PetscErrorCode MatCreateSubMatrixFree(Mat, IS, IS, Mat *);
PetscErrorCode MatSMFResetRowColumn(Mat, IS, IS);
PetscErrorCode MatMult_SMF(Mat, Vec, Vec);
PetscErrorCode MatMultTranspose_SMF(Mat, Vec, Vec);
PetscErrorCode MatDiagonalSet_SMF(Mat, Vec, InsertMode);
PetscErrorCode MatDestroy_SMF(Mat);
PetscErrorCode MatView_SMF(Mat, PetscViewer);
PetscErrorCode MatShift_SMF(Mat, PetscReal);
PetscErrorCode MatDuplicate_SMF(Mat, MatDuplicateOption, Mat *);
PetscErrorCode MatEqual_SMF(Mat, Mat, PetscBool *);
PetscErrorCode MatScale_SMF(Mat, PetscReal);
PetscErrorCode MatCreateSubMatrix_SMF(Mat, IS, IS, MatReuse, Mat *);
PetscErrorCode MatCreateSubMatrices_SMF(Mat, PetscInt, IS *, IS *, MatReuse, Mat **);
PetscErrorCode MatTranspose_SMF(Mat, Mat *);
PetscErrorCode MatGetDiagonal_SMF(Mat, Vec);
PetscErrorCode MatGetColumnVector_SMF(Mat, Vec, PetscInt);
PetscErrorCode MatNorm_SMF(Mat, NormType, PetscReal *);
PetscErrorCode MatGetRowMax_SMF(Mat, Vec);
PetscErrorCode MatGetRow_SMF(Mat, PetscInt, PetscInt *, const PetscInt **, const PetscScalar **);
PetscErrorCode MatRestoreRow_SMF(Mat, PetscInt, PetscInt *, const PetscInt **, const PetscScalar **);
