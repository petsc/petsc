#define PETSCMAT_DLL

#include "src/mat/matimpl.h"

typedef struct {
  PetscErrorCode (*MatConvert)(Mat, MatType,MatReuse,Mat*);
  PetscErrorCode (*MatDestroy)(Mat);
} Mat_AIJ;

EXTERN_C_BEGIN
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatConvert_SeqAIJ_AIJ(Mat, MatType,MatReuse,Mat *);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatConvert_MPIAIJ_AIJ(Mat, MatType,MatReuse,Mat *);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatConvert_AIJ_SeqAIJ(Mat, MatType,MatReuse,Mat *);
EXTERN PetscErrorCode PETSCMAT_DLLEXPORT MatConvert_AIJ_MPIAIJ(Mat, MatType,MatReuse,Mat *);
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatConvert_AIJ_SeqAIJ"
PetscErrorCode PETSCMAT_DLLEXPORT MatConvert_AIJ_SeqAIJ(Mat A, MatType type, MatReuse reuse,Mat *newmat)
{
  PetscErrorCode ierr;
  Mat            B=*newmat;
  Mat_AIJ        *aij=(Mat_AIJ *)A->spptr;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
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
PetscErrorCode PETSCMAT_DLLEXPORT MatConvert_AIJ_MPIAIJ(Mat A, MatType type, MatReuse reuse, Mat *newmat)
{
  PetscErrorCode ierr;
  Mat            B=*newmat;
  Mat_AIJ        *aij=(Mat_AIJ *)A->spptr;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
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
    ierr = MatConvert_AIJ_SeqAIJ(A,MATSEQAIJ,MAT_REUSE_MATRIX,&A);CHKERRQ(ierr);
  } else {
    ierr = MatConvert_AIJ_MPIAIJ(A,MATMPIAIJ,MAT_REUSE_MATRIX,&A);CHKERRQ(ierr);
  }
  ierr = (*A->ops->destroy)(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatConvertFrom_AIJviaSeqAIJ"
PetscErrorCode MatConvertFrom_AIJviaSeqAIJ(Mat A, MatType type,MatReuse reuse,Mat *newmat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatConvert_AIJ_SeqAIJ(A,MATSEQAIJ,MAT_REUSE_MATRIX,&A);CHKERRQ(ierr);
  ierr = MatConvert(A,type,reuse,newmat);CHKERRQ(ierr);
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatConvert_SeqAIJ_AIJ(A,MATAIJ,MAT_REUSE_MATRIX,&A);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatConvertFrom_AIJviaMPIAIJ"
PetscErrorCode MatConvertFrom_AIJviaMPIAIJ(Mat A, MatType type,MatReuse reuse,Mat *newmat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatConvert_AIJ_MPIAIJ(A,PETSC_NULL,MAT_REUSE_MATRIX,&A);CHKERRQ(ierr);
  ierr = MatConvert(A,type,reuse,newmat);CHKERRQ(ierr);
  if (reuse == MAT_INITIAL_MATRIX) { 
    ierr = MatConvert_MPIAIJ_AIJ(A,MATAIJ,MAT_REUSE_MATRIX,&A);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatConvert_SeqAIJ_AIJ"
PetscErrorCode PETSCMAT_DLLEXPORT MatConvert_SeqAIJ_AIJ(Mat A, MatType type,MatReuse reuse,Mat *newmat)
{
  PetscErrorCode ierr;
  Mat            B=*newmat;
  Mat_AIJ        *aij;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
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
PetscErrorCode PETSCMAT_DLLEXPORT MatConvert_MPIAIJ_AIJ(Mat A, MatType type,MatReuse reuse,Mat *newmat)
{
  PetscErrorCode ierr;
  Mat            B=*newmat;
  Mat_AIJ        *aij;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
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
PetscErrorCode PETSCMAT_DLLEXPORT MatConvertTo_AIJ(Mat A, MatType type,MatReuse reuse,Mat *newmat)
{
  /*
    This method is to be registered using MatConvertRegisterDynamic.  Perhaps a better mechanism would be to
    add a second MatConvert function pointer (one for "from" -- which we already have, and a second for "to").
    We should maintain the current registry list as well in order to provide a measure of backwards compatibility
    if indeed we make this change.
  */
  PetscErrorCode ierr;
  PetscMPIInt    size;
  Mat            B=*newmat;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(A->comm,&size);
  if (size == 1) {
    ierr = MatConvert(A,MATSEQAIJ,reuse,&B);CHKERRQ(ierr);
    ierr = MatConvert_SeqAIJ_AIJ(B,MATAIJ,MAT_REUSE_MATRIX,&B);CHKERRQ(ierr);
  } else {
    ierr = MatConvert(A,MATMPIAIJ,reuse,&B);CHKERRQ(ierr);
    ierr = MatConvert_MPIAIJ_AIJ(B,MATAIJ,MAT_REUSE_MATRIX,&B);CHKERRQ(ierr);
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
PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_AIJ(Mat A) 
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = PetscObjectChangeTypeName((PetscObject)A,MATAIJ);CHKERRQ(ierr);
  ierr = MPI_Comm_size(A->comm,&size);CHKERRQ(ierr);
  if (size == 1) {
    ierr = MatSetType(A,MATSEQAIJ);CHKERRQ(ierr);
    ierr = MatConvert_SeqAIJ_AIJ(A,MATAIJ,MAT_REUSE_MATRIX,&A);CHKERRQ(ierr);
  } else {
    ierr = MatSetType(A,MATMPIAIJ);CHKERRQ(ierr);
    ierr = MatConvert_MPIAIJ_AIJ(A,MATAIJ,MAT_REUSE_MATRIX,&A);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

/*MC
   MATCRL - MATCRL = "crl" - A matrix type to be used for sparse matrices.

   This matrix type is identical to MATSEQCRL when constructed with a single process communicator,
   and MATMPICRL otherwise.  As a result, for single process communicators, 
  MatSeqAIJSetPreallocation is supported, and similarly MatMPIAIJSetPreallocation is supported 
  for communicators controlling multiple processes.  It is recommended that you call both of
  the above preallocation routines for simplicity.

   Options Database Keys:
. -mat_type aij - sets the matrix type to "aij" during a call to MatSetFromOptions()

  Level: beginner

.seealso: MatCreateMPICRL,MATSEQCRL,MATMPICRL, MATSEQCRL, MATMPICRL
M*/

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatCreate_AIJ"
PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_CRL(Mat A) 
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = PetscObjectChangeTypeName((PetscObject)A,MATCRL);CHKERRQ(ierr);
  ierr = MPI_Comm_size(A->comm,&size);CHKERRQ(ierr);
  /* this is not really correct; should do all the complicated stuff like for MatCreate_AIJ() */
  if (size == 1) {
    ierr = MatSetType(A,MATSEQCRL);CHKERRQ(ierr);
  } else {
    ierr = MatSetType(A,MATMPICRL);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END
