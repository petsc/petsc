#define PETSCMAT_DLL

#include "../src/mat/impls/im/matim.h"          /*I "petscmat.h" I*/
#include "private/isimpl.h"

#undef  __FUNCT__
#define __FUNCT__ "MatIMSetIS"
PetscErrorCode PETSCMAT_DLLEXPORT MatIMSetIS(Mat A, IS in, IS out) {
  PetscErrorCode        ierr;
  Mat_IM*               im = (Mat_IM*)A->data;
  PetscMPIInt           flag;
  PetscInt              in_size, out_size;

  
  PetscFunctionBegin;

  /* FIX: check that sizes have been set and are valid; check that IS min,max are within Mat size limits */

  /* check IS validity */
  ierr = MPI_Comm_compare(((PetscObject)A)->comm, ((PetscObject)in)->comm, &flag); CHKERRQ(ierr);
  if(flag == MPI_IDENT) {
    ierr = MPI_Comm_compare(((PetscObject)A)->comm, ((PetscObject)out)->comm, &flag); CHKERRQ(ierr);
    if(flag != MPI_IDENT) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER, "Communicators differ between MatIM and the output IS");
  }
  else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER, "Communicators differ between MatIM and the input IS");
  if(!in && !out) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER, "Cannot have both input and output IS be NULL");
  /* Check that IS's min and max are within the A's size limits */
  if(in && (in->min < 0 || in->max >= A->rmap->N)) {
    SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_USER, "Input IS min or max values out of range: min = %d, max %d, must be between 0 and %d", in->min, in->max, A->rmap->N);
  }
  if(out && (out->min < 0 || out->max >= A->cmap->N)) {
    SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_USER, "Output IS min or max values out of range: min = %d, max %d, must be between 0 and %d", out->min, out->max, A->cmap->N);
  }
  if(in && out) {
    ierr = ISGetLocalSize(in,  &in_size); CHKERRQ(ierr);
    ierr = ISGetLocalSize(out, &out_size); CHKERRQ(ierr);
    if(in_size != out_size) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER, "IS local size mismatch: input %d and output %d", in_size, out_size);
  }
  im->in = in; 
  im->out = out; 
  if(im->in) {
    ierr = PetscObjectReference((PetscObject)(im->in)); CHKERRQ(ierr);
  }
  else {
    /* 
       If the input IS is NULL, it is assumed to be an identity on the initial segment of the local rows of A.
       The length of the segment is the local size of the output IS. 
    */
    ierr = ISGetLocalSize(im->out, &out_size); CHKERRQ(ierr);
    if(out_size > A->rmap->n) {
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER, "Output IS local size %d exceeds the local row count %d and input IS is NULL", out_size, A->rmap->n);
    }
    ierr = ISCreateStride(((PetscObject)A)->comm, out_size, A->rmap->rstart, 1, &(im->in)); CHKERRQ(ierr);
  }
  if(im->out) {
    ierr = PetscObjectReference((PetscObject)(im->out)); CHKERRQ(ierr);
  }
  else {
    /* 
       If the output IS is NULL, it is assumed to be an identity on the initial segment of the local columns of A.
       The length of the segment is the local size of the input IS. 
    */
    ierr = ISGetLocalSize(im->in, &in_size); CHKERRQ(ierr);
    if(in_size > A->cmap->n) {
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER, "Input IS local size %d exceeds the local column count %d and output IS is NULL", in_size, A->cmap->n);
    }
    ierr = ISCreateStride(((PetscObject)A)->comm, in_size, A->cmap->rstart, 1, &(im->out)); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}/* MatIMSetIS() */


#undef  __FUNCT__
#define __FUNCT__ "MatIMGetIS"
PetscErrorCode PETSCMAT_DLLEXPORT MatIMGetIS(Mat A, IS *_in, IS *_out) {
  PetscErrorCode ierr;
  Mat_IM*               im = (Mat_IM*)A->data;
  PetscFunctionBegin;
  if(_in) {
    *_in = im->in;
    ierr = PetscObjectReference((PetscObject)(im->in)); CHKERRQ(ierr);
  }
  if(_out) {
    *_out = im->out;
    ierr = PetscObjectReference((PetscObject)(im->out)); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
  
}/* MatIMGetIS() */



#undef __FUNCT__  
#define __FUNCT__ "MatIMSetUpPreallocation"
PetscErrorCode PETSCMAT_DLLEXPORT MatIMSetUpPreallocation(Mat A) 
{
  PetscErrorCode ierr;
  Mat_IM   *im = (Mat_IM*)A->data;
  Vec invec, outvec;
  PetscMPIInt commsize;
  PetscFunctionBegin;
  
  
  /* Not that 'invec' corresponds to columns and 'outvec' corresponds to rows, which is the opposite
     of the 'in' and 'out' IS designation: data movement is dual to index movement.
  */
  ierr = MPI_Comm_size(((PetscObject)A)->comm, &commsize); CHKERRQ(ierr);
  if(commsize == 1) {
    ierr = VecCreateSeqWithArray(((PetscObject)A)->comm, A->cmap->n, PETSC_NULL, &invec); CHKERRQ(ierr);
    ierr = VecCreateSeqWithArray(((PetscObject)A)->comm, A->rmap->n, PETSC_NULL, &outvec); CHKERRQ(ierr);
  }
  else {
    ierr = VecCreateMPIWithArray(((PetscObject)A)->comm, A->cmap->n, A->cmap->N, PETSC_NULL, &invec); CHKERRQ(ierr);
    ierr = VecCreateMPIWithArray(((PetscObject)A)->comm, A->rmap->n, A->rmap->N, PETSC_NULL, &outvec); CHKERRQ(ierr);  
  }
  ierr = VecScatterCreate(invec, im->out, outvec, im->in, &(im->scatter)); CHKERRQ(ierr);
  ierr = VecDestroy(invec); CHKERRQ(ierr);
  ierr = VecDestroy(outvec); CHKERRQ(ierr);

  A->assembled = PETSC_TRUE;
  PetscFunctionReturn(0);
}/* MatIMSetUpPreallocation() */



#undef  __FUNCT__
#define __FUNCT__ "MatMult_IM"
PetscErrorCode PETSCMAT_DLLEXPORT MatMult_IM(Mat A, Vec x, Vec y) {
  Mat_IM  *im = (Mat_IM*)A->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = VecZeroEntries(y); CHKERRQ(ierr);
  ierr = VecScatterBegin(im->scatter,x,y,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(im->scatter,x,y,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}// MatMult_IM()

#undef  __FUNCT__
#define __FUNCT__ "MatMultAdd_IM"
PetscErrorCode PETSCMAT_DLLEXPORT MatMultAdd_IM(Mat A, Vec x, Vec y, Vec z) {
  Mat_IM  *im = (Mat_IM*)A->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (z != y) {ierr = VecCopy(y,z);CHKERRQ(ierr);}
  ierr = VecScatterBegin(im->scatter,x,z,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(im->scatter,x,z,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}// MatMultAdd_IM()


#undef  __FUNCT__
#define __FUNCT__ "MatMultTranspose_IM"
PetscErrorCode PETSCMAT_DLLEXPORT MatMultTranspose_IM(Mat A, Vec x, Vec y) {
  Mat_IM  *im = (Mat_IM*)A->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = VecZeroEntries(y); CHKERRQ(ierr);
  ierr = VecScatterBegin(im->scatter,x,y,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(im->scatter,x,y,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}// MatMultTranspose_IM()

#undef  __FUNCT__
#define __FUNCT__ "MatMultTransposeAdd_IM"
PetscErrorCode PETSCMAT_DLLEXPORT MatMultTransposeAdd_IM(Mat A, Vec x, Vec y, Vec z) {
  Mat_IM  *im = (Mat_IM*)A->data;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (z != y) {ierr = VecCopy(y,z);CHKERRQ(ierr);}
  ierr = VecScatterBegin(im->scatter,x,z,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(im->scatter,x,z,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}// MatMultTransposeAdd_IM()


#undef  __FUNCT__
#define __FUNCT__ "MatDestroy_IM"
PetscErrorCode PETSCMAT_DLLEXPORT MatDestroy_IM(Mat M) {
  Mat_IM     *im = (Mat_IM *)M->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if(im->scatter) {
    ierr = VecScatterDestroy(im->scatter); CHKERRQ(ierr);
  }
  if(im->in) {
    ierr = ISDestroy(im->in); CHKERRQ(ierr); 
  }
  if(im->out) {
    ierr = ISDestroy(im->out); CHKERRQ(ierr); 
  }
  ierr = PetscFree(im);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)M,0);CHKERRQ(ierr);
  M->data = 0;
  PetscFunctionReturn(0);
}/* MatDestroy_IM() */


EXTERN_C_BEGIN
#undef  __FUNCT__
#define __FUNCT__ "MatCreate_IM"
PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_IM(Mat A) {
  /* Assume that this is called after MatSetSizes() */
  Mat_IM  *im;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = PetscLayoutSetBlockSize(A->rmap,1);CHKERRQ(ierr);
  ierr = PetscLayoutSetBlockSize(A->cmap,1);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(A->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(A->cmap);CHKERRQ(ierr);

  A->ops->setuppreallocation = MatIMSetUpPreallocation;
  A->ops->mult               = MatMult_IM;
  A->ops->multtranspose      = MatMultTranspose_IM;
  A->ops->multadd            = MatMultAdd_IM;
  A->ops->multtransposeadd   = MatMultTransposeAdd_IM;
  A->ops->destroy            = MatDestroy_IM;

  A->assembled    = PETSC_FALSE;
  A->same_nonzero = PETSC_FALSE;

  ierr = PetscNewLog(A,Mat_IM,&im);CHKERRQ(ierr);
  A->data = (void*)im;
  
  im->in = PETSC_NULL;
  im->out = PETSC_NULL;
  im->scatter = PETSC_NULL;


  ierr = PetscObjectChangeTypeName((PetscObject)A,MATIM);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* MatCreate_IM() */
EXTERN_C_END
