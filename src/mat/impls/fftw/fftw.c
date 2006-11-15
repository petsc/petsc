#define PETSCMAT_DLL

/*
    Provides an interface to the FFTW package
*/

#include "src/mat/matimpl.h"          /*I "petscvec.h" I*/
EXTERN_C_BEGIN 
#if defined(PETSC_USE_COMPLEX)
#include "fftw3.h"
#endif
EXTERN_C_END 

typedef struct {
  PetscInt       ndim;
  PetscInt       *dim;
  fftw_plan      p_forward,p_backward;
} Mat_FFTW;

#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqFFTW"
PetscErrorCode MatMult_SeqFFTW(Mat A,Vec x,Vec y)
{
  PetscErrorCode ierr;
  Mat_FFTW       *fftw = (Mat_FFTW*)A->data;
  PetscScalar    *x_array,*y_array;
  PetscInt       ndim=fftw->ndim,*dim=fftw->dim;

  PetscFunctionBegin;
  ierr = VecGetArray(x,&x_array);CHKERRQ(ierr);
  ierr = VecGetArray(y,&y_array);CHKERRQ(ierr);
  switch (ndim){
  case 1:
    fftw->p_forward = fftw_plan_dft_1d(dim[0],(fftw_complex*)x_array,(fftw_complex*)y_array,FFTW_FORWARD,FFTW_ESTIMATE);
    break;
  case 2:
    fftw->p_forward = fftw_plan_dft_2d(dim[0],dim[1],(fftw_complex*)x_array,(fftw_complex*)y_array,FFTW_FORWARD,FFTW_ESTIMATE);
    break;
  case 3:
    fftw->p_forward = fftw_plan_dft_3d(dim[0],dim[1],dim[2],(fftw_complex*)x_array,(fftw_complex*)y_array,FFTW_FORWARD,FFTW_ESTIMATE);
    break;
  default:
    fftw->p_forward = fftw_plan_dft(ndim,dim,(fftw_complex*)x_array,(fftw_complex*)y_array,FFTW_FORWARD,FFTW_ESTIMATE);
    break;
  }
  fftw_execute(fftw->p_forward);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&y_array);CHKERRQ(ierr);
  ierr = VecRestoreArray(x,&x_array);CHKERRQ(ierr);
  fftw_destroy_plan(fftw->p_forward);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultTranspose_SeqFFTW"
PetscErrorCode MatMultTranspose_SeqFFTW(Mat A,Vec x,Vec y)
{
  PetscErrorCode ierr;
  Mat_FFTW       *fftw = (Mat_FFTW*)A->data;
  PetscScalar    *x_array,*y_array;
  PetscInt       ndim=fftw->ndim,*dim=fftw->dim;

  PetscFunctionBegin;
  ierr = VecGetArray(x,&x_array);CHKERRQ(ierr);
  ierr = VecGetArray(y,&y_array);CHKERRQ(ierr);
  switch (ndim){
  case 1:
    fftw->p_backward = fftw_plan_dft_1d(dim[0],(fftw_complex*)x_array,(fftw_complex*)y_array,FFTW_BACKWARD,FFTW_ESTIMATE);
    break;
  case 2:
    fftw->p_backward = fftw_plan_dft_2d(dim[0],dim[1],(fftw_complex*)x_array,(fftw_complex*)y_array,FFTW_BACKWARD,FFTW_ESTIMATE);
    break;
  case 3:
    fftw->p_backward = fftw_plan_dft_3d(dim[0],dim[1],dim[2],(fftw_complex*)x_array,(fftw_complex*)y_array,FFTW_BACKWARD,FFTW_ESTIMATE);
    break;
  default:
    fftw->p_backward = fftw_plan_dft(ndim,dim,(fftw_complex*)x_array,(fftw_complex*)y_array,FFTW_BACKWARD,FFTW_ESTIMATE);
    break;
  }
  fftw_execute(fftw->p_backward);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&y_array);CHKERRQ(ierr);
  ierr = VecRestoreArray(x,&x_array);CHKERRQ(ierr);
  fftw_destroy_plan(fftw->p_backward);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_SeqFFTW"
PetscErrorCode MatDestroy_SeqFFTW(Mat A)
{
  Mat_FFTW       *fftw = (Mat_FFTW*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;  
  ierr = PetscFree(fftw->dim);CHKERRQ(ierr);
  ierr = PetscFree(fftw);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)A,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
  MATFFTW - MATFFTW = "fftw" - A matrix type providing sequential FFT
  via the external package FFTW.

  If FFTW is installed (see the manual for
  instructions on how to declare the existence of external packages),
  a matrix type can be constructed which invokes FFFTW solver.
  After calling MatCreate(...,A), simply call MatSetType(A,MATSEQFFTW).

  Options Database Keys:
+ -mat_type seqfftw - sets the matrix type to "seqfftw" during a call to MatSetFromOptions()
. -mat_fftw_: 
- -mat_fftw_:

   Level: beginner
M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatCreate_SeqFFTW"
PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_SeqFFTW(Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatCreateSeqFFTW"
PetscErrorCode PETSCMAT_DLLEXPORT MatCreateSeqFFTW(MPI_Comm comm,PetscInt ndim,const PetscInt dim[],Mat* A)
{
  PetscErrorCode ierr;
  Mat_FFTW       *fftw;
  PetscInt       m,i,*dim_tmp;

  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  m = dim[0];
  for (i=1; i<ndim; i++) m *= dim[i];
  ierr = MatSetSizes(*A,m,m,m,m);CHKERRQ(ierr);  
  ierr = PetscObjectChangeTypeName((PetscObject)*A,MATSEQFFTW);CHKERRQ(ierr);

  ierr = PetscNew(Mat_FFTW,&fftw);CHKERRQ(ierr);
  (*A)->data = (void*)fftw;
  ierr = PetscMalloc((ndim+1)*sizeof(PetscInt),&fftw->dim);CHKERRQ(ierr);
  ierr = PetscMemcpy(fftw->dim,dim,ndim*sizeof(PetscInt));CHKERRQ(ierr);
  fftw->ndim       = ndim;
  fftw->p_forward  = 0;
  fftw->p_backward = 0;

  (*A)->ops->mult          = MatMult_SeqFFTW;
  (*A)->ops->multtranspose = MatMultTranspose_SeqFFTW;
  (*A)->assembled          = PETSC_TRUE;
  (*A)->ops->destroy       = MatDestroy_SeqFFTW;
  PetscFunctionReturn(0);
}
