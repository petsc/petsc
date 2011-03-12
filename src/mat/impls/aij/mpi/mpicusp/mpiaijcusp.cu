
#include "../src/mat/impls/aij/mpi/mpiaij.h"   /*I "petscmat.h" I*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatMPIAIJSetPreallocation_MPIAIJCUSP"
PetscErrorCode  MatMPIAIJSetPreallocation_MPIAIJCUSP(Mat B,PetscInt d_nz,const PetscInt d_nnz[],PetscInt o_nz,const PetscInt o_nnz[])
{
  Mat_MPIAIJ     *b;
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  if (d_nz == PETSC_DEFAULT || d_nz == PETSC_DECIDE) d_nz = 5;
  if (o_nz == PETSC_DEFAULT || o_nz == PETSC_DECIDE) o_nz = 2;
  if (d_nz < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"d_nz cannot be less than 0: value %D",d_nz);
  if (o_nz < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"o_nz cannot be less than 0: value %D",o_nz);

  ierr = PetscLayoutSetBlockSize(B->rmap,1);CHKERRQ(ierr);
  ierr = PetscLayoutSetBlockSize(B->cmap,1);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(B->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(B->cmap);CHKERRQ(ierr);
  if (d_nnz) {
    for (i=0; i<B->rmap->n; i++) {
      if (d_nnz[i] < 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"d_nnz cannot be less than 0: local row %D value %D",i,d_nnz[i]);
    }
  }
  if (o_nnz) {
    for (i=0; i<B->rmap->n; i++) {
      if (o_nnz[i] < 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"o_nnz cannot be less than 0: local row %D value %D",i,o_nnz[i]);
    }
  }
  b = (Mat_MPIAIJ*)B->data;

  if (!B->preallocated) {
    /* Explicitly create 2 MATSEQAIJ matrices. */
    ierr = MatCreate(PETSC_COMM_SELF,&b->A);CHKERRQ(ierr);
    ierr = MatSetSizes(b->A,B->rmap->n,B->cmap->n,B->rmap->n,B->cmap->n);CHKERRQ(ierr);
    ierr = MatSetType(b->A,MATSEQAIJCUSP);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(B,b->A);CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_SELF,&b->B);CHKERRQ(ierr);
    ierr = MatSetSizes(b->B,B->rmap->n,B->cmap->N,B->rmap->n,B->cmap->N);CHKERRQ(ierr);
    ierr = MatSetType(b->B,MATSEQAIJCUSP);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(B,b->B);CHKERRQ(ierr);
  }

  ierr = MatSeqAIJSetPreallocation(b->A,d_nz,d_nnz);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(b->B,o_nz,o_nnz);CHKERRQ(ierr);
  B->preallocated = PETSC_TRUE;
  PetscFunctionReturn(0);
}
EXTERN_C_END
#undef __FUNCT__  
#define __FUNCT__ "MatGetVecs_MPIAIJCUSP"
PetscErrorCode  MatGetVecs_MPIAIJCUSP(Mat mat,Vec *right,Vec *left)
{
  PetscErrorCode ierr;
  PetscMPIInt size;

  PetscFunctionBegin;

  ierr = MPI_Comm_size(((PetscObject)mat)->comm, &size);CHKERRQ(ierr);
  if (right) {
    ierr = VecCreate(((PetscObject)mat)->comm,right);CHKERRQ(ierr);
    ierr = VecSetSizes(*right,mat->cmap->n,PETSC_DETERMINE);CHKERRQ(ierr);
    ierr = VecSetBlockSize(*right,mat->rmap->bs);CHKERRQ(ierr);
    if (size > 1) {
      /* New vectors uses Mat cmap and does not create a new one */
      ierr = PetscLayoutDestroy((*right)->map);CHKERRQ(ierr);
      (*right)->map = mat->cmap;
      mat->cmap->refcnt++;
      ierr = VecSetType(*right,VECMPICUSP);CHKERRQ(ierr);
      } else {ierr = VecSetType(*right,VECSEQCUSP);CHKERRQ(ierr);}
  }
  if (left) {
    ierr = VecCreate(((PetscObject)mat)->comm,left);CHKERRQ(ierr);
    ierr = VecSetSizes(*left,mat->rmap->n,PETSC_DETERMINE);CHKERRQ(ierr);
    ierr = VecSetBlockSize(*left,mat->rmap->bs);CHKERRQ(ierr);
    if (size > 1) {
      /* New vectors uses Mat rmap and does not create a new one */
      ierr = PetscLayoutDestroy((*left)->map);CHKERRQ(ierr);
      (*left)->map = mat->rmap;
      mat->rmap->refcnt++;
      ierr = VecSetType(*left,VECMPICUSP);CHKERRQ(ierr);
    } else {ierr = VecSetType(*left,VECSEQCUSP);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
PetscErrorCode  MatCreate_MPIAIJ(Mat);
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatCreate_MPIAIJCUSP"
PetscErrorCode  MatCreate_MPIAIJCUSP(Mat B)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate_MPIAIJ(B);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatMPIAIJSetPreallocation_C",
                                     "MatMPIAIJSetPreallocation_MPIAIJCUSP",
                                      MatMPIAIJSetPreallocation_MPIAIJCUSP);CHKERRQ(ierr);
  B->ops->getvecs = MatGetVecs_MPIAIJCUSP;
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATMPIAIJCUSP);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END


#undef __FUNCT__  
#define __FUNCT__ "MatCreateMPIAIJCUSP"
PetscErrorCode  MatCreateMPIAIJCUSP(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt M,PetscInt N,PetscInt d_nz,const PetscInt d_nnz[],PetscInt o_nz,const PetscInt o_nnz[],Mat *A)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,M,N);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size > 1) {
    ierr = MatSetType(*A,MATMPIAIJCUSP);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(*A,d_nz,d_nnz,o_nz,o_nnz);CHKERRQ(ierr);
  } else {
    ierr = MatSetType(*A,MATSEQAIJCUSP);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(*A,d_nz,d_nnz);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*MC
   MATAIJCUSP - MATAIJCUSP = "aijcusp" - A matrix type to be used for sparse matrices.

   This matrix type is identical to MATSEQAIJCUSP when constructed with a single process communicator,
   and MATMPIAIJCUSP otherwise.  As a result, for single process communicators, 
  MatSeqAIJSetPreallocation is supported, and similarly MatMPIAIJSetPreallocation is supported 
  for communicators controlling multiple processes.  It is recommended that you call both of
  the above preallocation routines for simplicity.

   Options Database Keys:
. -mat_type aij - sets the matrix type to "aij" during a call to MatSetFromOptions()

  Level: beginner

.seealso: MatCreateMPIAIJ,MATSEQAIJ,MATMPIAIJ, MATMPIAIJCUSP, MATSEQAIJCUSP
M*/

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatCreate_AIJCUSP"
PetscErrorCode  MatCreate_AIJCUSP(Mat A) 
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(((PetscObject)A)->comm,&size);CHKERRQ(ierr);
  if (size == 1) {
    ierr = MatSetType(A,MATSEQAIJCUSP);CHKERRQ(ierr);
  } else {
    ierr = MatSetType(A,MATMPIAIJCUSP);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END
