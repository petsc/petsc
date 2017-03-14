#ifndef __LMVMMAT_H
#define __LMVMMAT_H

#include <petscksp.h>

#define MatLMVM_Scale_None              0
#define MatLMVM_Scale_Scalar            1
#define MatLMVM_Scale_Broyden           2
#define MatLMVM_Scale_H0                3
#define MatLMVM_Scale_Types             4

#define MatLMVM_Rescale_None            0
#define MatLMVM_Rescale_Scalar          1
#define MatLMVM_Rescale_GL              2
#define MatLMVM_Rescale_Types           3

#define MatLMVM_Limit_None              0
#define MatLMVM_Limit_Average           1
#define MatLMVM_Limit_Relative          2
#define MatLMVM_Limit_Absolute          3
#define MatLMVM_Limit_Types             4

#define TAO_ZERO_SAFEGUARD      1e-8
#define TAO_INF_SAFEGUARD       1e+8

typedef struct{
    PetscBool allocated;
    PetscInt lm;
    PetscReal eps;
    PetscInt limitType;
    PetscInt scaleType;
    PetscInt rScaleType;

    PetscReal s_alpha;  /*  Factor for scalar scaling */
    PetscReal r_alpha;  /*  Factor on scalar for rescaling diagonal matrix */
    PetscReal r_beta;   /*  Factor on diagonal for rescaling diagonal matrix */
    PetscReal mu;               /*  Factor for using historical information */
    PetscReal nu;               /*  Factor for using historical information */
    PetscReal phi;              /*  Factor for Broyden scaling */

  PetscInt scalar_history;      /*  Amount of history to keep for scalar scaling */
  PetscReal *yy_history;        /*  Past information for scalar scaling */
  PetscReal *ys_history;        /*  Past information for scalar scaling */
  PetscReal *ss_history;        /*  Past information for scalar scaling */

  PetscInt rescale_history;  /*  Amount of history to keep for rescaling diagonal */
  PetscReal *yy_rhistory;       /*  Past information for scalar rescaling */
  PetscReal *ys_rhistory;       /*  Past information for scalar rescaling */
  PetscReal *ss_rhistory;       /*  Past information for scalar rescaling */

  PetscReal delta_max;  /*  Maximum value for delta */
  PetscReal delta_min;  /*  Minimum value for delta */

  PetscInt lmnow;
  PetscInt iter;
  PetscInt nupdates;
  PetscInt nrejects;

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
  PetscReal theta;
  PetscReal *rho;
  PetscReal *beta;

  PetscBool useDefaultH0;
  Mat H0_mat;
  KSP H0_ksp;
  Vec H0_norm;

  PetscBool useScale;
  Vec scale;


} MatLMVMCtx;


extern PetscErrorCode MatCreateLMVM(MPI_Comm,PetscInt,PetscInt,Mat*);


/* PETSc Mat overrides */
extern PetscErrorCode MatView_LMVM(Mat,PetscViewer);
extern PetscErrorCode MatDestroy_LMVM(Mat);

/*
int MatMultTranspose_LMVM(Mat,Vec,Vec);
int MatDiagonalShift_LMVM(Vec,Mat);
int MatDestroy_LMVM(Mat);
int MatShift_LMVM(Mat,PetscReal);
int MatDuplicate_LMVM(Mat,MatDuplicateOption,Mat*);
int MatEqual_LMVM(Mat,Mat,PetscBool*);
int MatScale_LMVM(Mat,PetscReal);
int MatGetCreateMatrix_LMVM(Mat,IS,IS,int,MatReuse,Mat *);
int MatCreateSubMatrices_LMVM(Mat,int,IS*,IS*,MatReuse,Mat**);
int MatTranspose_LMVM(Mat,Mat*);
int MatGetDiagonal_LMVM(Mat,Vec);
int MatGetColumnVector_LMVM(Mat,Vec, int);
int MatNorm_LMVM(Mat,NormType,PetscReal *);
*/

/* Functions used by TAO */
PetscErrorCode MatLMVMReset(Mat);
PetscErrorCode MatLMVMUpdate(Mat,Vec, Vec);
PetscErrorCode MatLMVMSetDelta(Mat,PetscReal);
PetscErrorCode MatLMVMSetScale(Mat,Vec);
PetscErrorCode MatLMVMGetRejects(Mat,PetscInt*);
PetscErrorCode MatLMVMSetH0(Mat,Mat);
PetscErrorCode MatLMVMGetH0(Mat,Mat*);
PetscErrorCode MatLMVMGetH0KSP(Mat,KSP*);
PetscErrorCode MatLMVMSetPrev(Mat,Vec,Vec);
PetscErrorCode MatLMVMGetX0(Mat,Vec);
PetscErrorCode MatLMVMRefine(Mat, Mat, Vec, Vec);
PetscErrorCode MatLMVMAllocateVectors(Mat m, Vec v);
PetscErrorCode MatLMVMSolve(Mat, Vec, Vec);


#endif
