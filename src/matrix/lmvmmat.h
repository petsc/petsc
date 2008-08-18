#ifndef __LMVMMAT_H
#define __LMVMMAT_H


#include "private/matimpl.h"
#include "tao_sys.h"

#define MatLMVM_Scale_None		0
#define MatLMVM_Scale_Scalar		1
#define MatLMVM_Scale_Broyden		2
#define MatLMVM_Scale_Types             3

#define MatLMVM_Rescale_None		0
#define MatLMVM_Rescale_Scalar		1
#define MatLMVM_Rescale_GL		2
#define MatLMVM_Rescale_Types          	3

#define MatLMVM_Limit_None		0
#define MatLMVM_Limit_Average		1
#define MatLMVM_Limit_Relative		2
#define MatLMVM_Limit_Absolute		3
#define MatLMVM_Limit_Types		4

#define TAO_ZERO_SAFEGUARD	1e-8
#define TAO_INF_SAFEGUARD	1e+8

typedef struct{
    PetscTruth allocated;
    PetscInt lm;
    PetscScalar eps;
    PetscInt limitType;
    PetscInt scaleType;
    PetscInt rScaleType;
    
    PetscScalar s_alpha;	// Factor for scalar scaling
    PetscScalar r_alpha;	// Factor on scalar for rescaling diagonal matrix
    PetscScalar r_beta;	// Factor on diagonal for rescaling diagonal matrix
    PetscScalar mu;		// Factor for using historical information
    PetscScalar nu;		// Factor for using historical information
    PetscScalar phi;		// Factor for Broyden scaling

  PetscInt scalar_history;	// Amount of history to keep for scalar scaling
  PetscScalar *yy_history;	// Past information for scalar scaling
  PetscScalar *ys_history;	// Past information for scalar scaling
  PetscScalar *ss_history;	// Past information for scalar scaling

  PetscInt rescale_history;  // Amount of history to keep for rescaling diagonal
  PetscScalar *yy_rhistory;	// Past information for scalar rescaling
  PetscScalar *ys_rhistory;	// Past information for scalar rescaling
  PetscScalar *ss_rhistory;	// Past information for scalar rescaling

  PetscScalar delta_max;	// Maximum value for delta
  PetscScalar delta_min;	// Minimum value for delta

  PetscInt lmnow;
  PetscInt iter;
  PetscInt updates;
  PetscInt rejects;

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

  PetscScalar delta;
  PetscScalar sigma;
  PetscScalar *rho;
  PetscScalar *beta;

  PetscTruth useDefaultH0;
  Mat H0;

  PetscTruth useScale;
  Vec scale;
     

} MatLMVMCtx;

//typedef  struct _p_MatLMVMCtx* MatLMVMCtx;

EXTERN PetscErrorCode MatCreateLMVM(MPI_Comm,PetscInt,PetscInt,Mat*);


/* PETSc Mat overrides */
EXTERN PetscErrorCode MatView_LMVM(Mat,PetscViewer);
EXTERN PetscErrorCode MatDestroy_LMVM(Mat);

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
int MatNorm_LMVM(Mat,NormType,PetscScalar *);
*/

/* Functions used by TAO */
PetscErrorCode MatLMVMReset(Mat);
PetscErrorCode MatLMVMUpdate(Mat,Vec, Vec);
PetscErrorCode MatLMVMSetDelta(Mat,PetscScalar);
PetscErrorCode MatLMVMSetScale(Mat,Vec);
PetscErrorCode MatLMVMGetRejects(Mat);
PetscErrorCode MatLMVMSetH0(Mat,Mat);
PetscErrorCode MatLMVMGetX0(Mat,Vec);
PetscErrorCode MatLMVMRefine(Mat, Mat, Vec, Vec);
PetscErrorCode MatLMVMAllocateVectors(Mat m, Vec v);
PetscErrorCode MatLMVMSolve(Mat, Vec, Vec);


#endif
