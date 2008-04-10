#include "private/matimpl.h"


typedef struct{
    PetscInt lm;
    PetscReale eps;
    PetscInt limitType;
    PetscInt scaleType;
    PetscInt rScaleType;
    
    PetscReal s_alpha;	// Factor for scalar scaling
    PetscReal r_alpha;	// Factor on scalar for rescaling diagonal matrix
    PetscReal r_beta;	// Factor on diagonal for rescaling diagonal matrix
    PetscReal mu;		// Factor for using historical information
    PetscReal nu;		// Factor for using historical information
    PetscReal phi;		// Factor for Broyden scaling

  PetscInt scalar_history;	// Amount of history to keep for scalar scaling
  PetscReal *yy_history;	// Past information for scalar scaling
  PetscReal *ys_history;	// Past information for scalar scaling
  PetscReal *ss_history;	// Past information for scalar scaling

  PetscInt rescale_history;  // Amount of history to keep for rescaling diagonal
  PetscReal *yy_rhistory;	// Past information for scalar rescaling
  PetscReal *ys_rhistory;	// Past information for scalar rescaling
  PetscReal *ss_rhistory;	// Past information for scalar rescaling

  PetscReal delta_max;	// Maximum value for delta
  PetscReal delta_min;	// Minimum value for delta

  PetscInt lmnow;
  PetscInt iter;
  PetscInt updates;
  PetscInt rejects

  Vec *S;
  Vec *Y;
  Vec Gprev;
  Vec Xprev;

  Vec D;
  Vec U;
  Vec V;
  Vec W;
  Vec P;
  Vec Q;

  PetscReal delta;
  PetscReal sigma;
  PetscReal *rho;
  PetscReal *beta;

  PetscTruth H0default;
  Mat H0;

  Vec scale;
    

} _p_LmvmMatCtx;

typedef  _p_MatLMVMCtx* MatLMVMCtx;

int MatCreateLMVM(Mat,Vec,Vec,Mat*);

/* PETSc Mat overrides */
int MatMult_LMVM(Mat,Vec,Vec);
int MatView_LMVM(Mat,PetscViewer);

/*
int MatMultTranspose_LMVM(Mat,Vec,Vec);
int MatDiagonalShift_LMVM(Vec,Mat);
int MatDestroy_LMVM(Mat);
int MatShift_LMVM(Mat,PetscScalar);
int MatDuplicate_LMVM(Mat,MatDuplicateOption,Mat*);
int MatEqual_LMVM(Mat,Mat,PetscTruth*);
int MatScale_LMVM(Mat,PetscScalar);
int MatGetSubMatrix_LMVM(Mat,IS,IS,int,MatReuse,Mat *);
int MatGetSubMatrices_LMVM(Mat,int,IS*,IS*,MatReuse,Mat**);
int MatTranspose_LMVM(Mat,Mat*);
int MatGetDiagonal_LMVM(Mat,Vec);
int MatGetColumnVector_LMVM(Mat,Vec, int);
int MatNorm_LMVM(Mat,NormType,PetscReal *);
*/

/* Functions used by TAO */
PetscErrorCode MatLMVM_Reset();
PetscErrorCode MatLMVM_Solve(Vec, Vec, PetscTruth*);
PetscErrorCode MatLMVM_Update(Vec, Vec);
PetscErrorCode MatLMVM_SetDelta(PetscReal);
PetscErrorCode MatLMVM_SetScale(Vec);
PetscErrorCode MatLMVM_GetRejects();
PetscErrorCode MatLMVM_SetH0(Mat);
PetscErrorCode MatLMVM_GetX0(Vec);
PetscErrorCode MatLMVM_Refine(Mat, Mat, Vec, Vec);
