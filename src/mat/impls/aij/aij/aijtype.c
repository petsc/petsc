#include "src/mat/matimpl.h"

typedef struct {
  PetscErrorCode (*MatConvert)(Mat,const MatType,Mat*);
  PetscErrorCode (*MatDestroy)(Mat);
} Mat_AIJ;

EXTERN_C_BEGIN
EXTERN PetscErrorCode MatConvert_SeqAIJ_AIJ(Mat,const MatType,Mat *);
EXTERN PetscErrorCode MatConvert_MPIAIJ_AIJ(Mat,const MatType,Mat *);
EXTERN PetscErrorCode MatConvert_AIJ_SeqAIJ(Mat,const MatType,Mat *);
EXTERN PetscErrorCode MatConvert_AIJ_MPIAIJ(Mat,const MatType,Mat *);
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatConvert_AIJ_SeqAIJ"
PetscErrorCode MatConvert_AIJ_SeqAIJ(Mat A, const MatType type, Mat *newmat)
{
  PetscErrorCode ierr;
  Mat            B=*newmat;
  Mat_AIJ        *aij=(Mat_AIJ *)A->spptr;

  PetscFunctionBegin;
  if (B != A) {
    ierr = MatDuplicate(A,MAT_COPY_VALUES, &B);CHKERRQ(ierr);
  }

  B->ops->convert = aij->MatConvert;
  B->ops->destroy = aij->MatDestroy;

  ierr = PetscFree(aij);CHKERRQ(ierr);

  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_aij_seqaij_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_seqaij_aij_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATSEQAIJ);CHKERRQ(ierr);
  *newmat = B;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatConvert_AIJ_MPIAIJ"
PetscErrorCode MatConvert_AIJ_MPIAIJ(Mat A, const MatType type, Mat *newmat)
{
  PetscErrorCode ierr;
  Mat            B=*newmat;
  Mat_AIJ        *aij=(Mat_AIJ *)A->spptr;

  PetscFunctionBegin;
  if (B != A) {
    ierr = MatDuplicate(A,MAT_COPY_VALUES, &B);CHKERRQ(ierr);
  }

  B->ops->convert = aij->MatConvert;
  B->ops->destroy = aij->MatDestroy;

  ierr = PetscFree(aij);CHKERRQ(ierr);

  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_aij_mpiaij_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_mpiaij_aij_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATMPIAIJ);CHKERRQ(ierr);
  *newmat = B;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_AIJ"
PetscErrorCode MatDestroy_AIJ(Mat A)
{
  PetscMPIInt    size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(A->comm,&size);CHKERRQ(ierr);
  if (size == 1) {
    ierr = MatConvert_AIJ_SeqAIJ(A,MATSEQAIJ,&A);CHKERRQ(ierr);
  } else {
    ierr = MatConvert_AIJ_MPIAIJ(A,MATMPIAIJ,&A);CHKERRQ(ierr);
  }
  ierr = (*A->ops->destroy)(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatConvertFrom_AIJviaSeqAIJ"
PetscErrorCode MatConvertFrom_AIJviaSeqAIJ(Mat A, const MatType type, Mat *newmat)
{
  PetscErrorCode ierr;
  PetscTruth     inplace=PETSC_FALSE;

  PetscFunctionBegin;
  if (*newmat == A) inplace = PETSC_TRUE;
  ierr = MatConvert_AIJ_SeqAIJ(A,MATSEQAIJ,&A);CHKERRQ(ierr);
  ierr = MatConvert(A,type,newmat);CHKERRQ(ierr);
  if (!inplace) {
    ierr = MatConvert_SeqAIJ_AIJ(A,MATAIJ,&A);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatConvertFrom_AIJviaMPIAIJ"
PetscErrorCode MatConvertFrom_AIJviaMPIAIJ(Mat A, const MatType type, Mat *newmat)
{
  PetscErrorCode ierr;
  PetscTruth     inplace=PETSC_FALSE;

  PetscFunctionBegin;
  if (A == *newmat) inplace = PETSC_TRUE;
  ierr = MatConvert_AIJ_MPIAIJ(A,PETSC_NULL,&A);CHKERRQ(ierr);
  ierr = MatConvert(A,type,newmat);CHKERRQ(ierr);
  if (!inplace) { 
    ierr = MatConvert_MPIAIJ_AIJ(A,MATAIJ,&A);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatConvert_SeqAIJ_AIJ"
PetscErrorCode MatConvert_SeqAIJ_AIJ(Mat A, const MatType type, Mat *newmat)
{
  PetscErrorCode ierr;
  Mat            B=*newmat;
  Mat_AIJ        *aij;

  PetscFunctionBegin;
  if (B != A) {
    ierr = MatDuplicate(A,MAT_COPY_VALUES, &B);CHKERRQ(ierr);
  }
  
  ierr = PetscNew(Mat_AIJ,&aij);CHKERRQ(ierr);
  aij->MatConvert = A->ops->convert;
  aij->MatDestroy = A->ops->destroy;

  B->spptr        = (void *)aij;
  B->ops->convert = MatConvertFrom_AIJviaSeqAIJ;
  B->ops->destroy = MatDestroy_AIJ;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatConvert_aij_seqaij_C",
                                    "MatConvert_AIJ_SeqAIJ",MatConvert_AIJ_SeqAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatConvert_seqaij_aij_C",
                                    "MatConvert_SeqAIJ_AIJ",MatConvert_SeqAIJ_AIJ);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATAIJ);CHKERRQ(ierr);
  *newmat = B;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatConvert_MPIAIJ_AIJ"
PetscErrorCode MatConvert_MPIAIJ_AIJ(Mat A, const MatType type, Mat *newmat)
{
  PetscErrorCode ierr;
  Mat            B=*newmat;
  Mat_AIJ        *aij;

  PetscFunctionBegin;
  if (B != A) {
    ierr = MatDuplicate(A,MAT_COPY_VALUES, &B);CHKERRQ(ierr);
  }

  ierr = PetscNew(Mat_AIJ,&aij);CHKERRQ(ierr);
  aij->MatConvert = A->ops->convert;
  aij->MatDestroy = A->ops->destroy;

  B->spptr        = (void *)aij;
  B->ops->convert = MatConvertFrom_AIJviaMPIAIJ;
  B->ops->destroy = MatDestroy_AIJ;

  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_aij_mpiaij_C",
                                    "MatConvert_AIJ_MPIAIJ",
                                    (void (*)(void))MatConvert_AIJ_MPIAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_mpiaij_aij_C",
                                    "MatConvert_MPIAIJ_AIJ",
                                    (void (*)(void))MatConvert_MPIAIJ_AIJ);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATAIJ);CHKERRQ(ierr);
  *newmat = B;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatConvertTo_AIJ"
PetscErrorCode MatConvertTo_AIJ(Mat A, const MatType type, Mat *newmat)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;
  Mat            B=*newmat;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(A->comm,&size);
  if (size == 1) {
    ierr = MatConvert(A,MATSEQAIJ,&B);CHKERRQ(ierr);
    ierr = MatConvert_SeqAIJ_AIJ(B,MATAIJ,&B);CHKERRQ(ierr);
  } else {
    ierr = MatConvert(A,MATMPIAIJ,&B);CHKERRQ(ierr);
    ierr = MatConvert_MPIAIJ_AIJ(B,MATAIJ,&B);CHKERRQ(ierr);
  }
  *newmat = B;
  PetscFunctionReturn(0);
}
EXTERN_C_END

/*MC
   MATAIJ - MATAIJ = "aij" - A matrix type to be used for sparse matrices.

   This matrix type is identical to MATSEQAIJ when constructed with a single process communicator,
   and MATMPIAIJ otherwise.  As a result, for single process communicators, 
  MatSeqAIJSetPreallocation is supported, and similarly MatMPIAIJSetPreallocation is supported 
  for communicators controlling multiple processes.  It is recommended that you call both of
  the above preallocation routines for simplicity.

   Options Database Keys:
. -mat_type aij - sets the matrix type to "aij" during a call to MatSetFromOptions()

  Level: beginner

.seealso: MatCreateMPIAIJ,MATSEQAIJ,MATMPIAIJ
M*/

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatCreate_AIJ"
PetscErrorCode MatCreate_AIJ(Mat A) 
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = PetscObjectChangeTypeName((PetscObject)A,MATAIJ);CHKERRQ(ierr);
  ierr = MPI_Comm_size(A->comm,&size);CHKERRQ(ierr);
  if (size == 1) {
    ierr = MatSetType(A,MATSEQAIJ);CHKERRQ(ierr);
    ierr = MatConvert_SeqAIJ_AIJ(A,MATAIJ,&A);CHKERRQ(ierr);
  } else {
    ierr = MatSetType(A,MATMPIAIJ);CHKERRQ(ierr);
    ierr = MatConvert_MPIAIJ_AIJ(A,MATAIJ,&A);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END
