
/*
    Provides an interface to the CUFFT package.
    Testing examples can be found in ~src/mat/examples/tests
*/

#include <petsc-private/matimpl.h>          /*I "petscmat.h" I*/
EXTERN_C_BEGIN 
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
EXTERN_C_END 

typedef struct {
  PetscInt      ndim;
  PetscInt     *dim;
  cufftHandle   p_forward, p_backward;
  cufftComplex *devArray;
} Mat_CUFFT;

#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqCUFFT"
PetscErrorCode MatMult_SeqCUFFT(Mat A, Vec x, Vec y)
{
  Mat_CUFFT     *cufft    = (Mat_CUFFT *) A->data;
  cufftComplex  *devArray = cufft->devArray;
  PetscInt       ndim     = cufft->ndim, *dim = cufft->dim;
  PetscScalar   *x_array, *y_array;
  cufftResult    result;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetArray(x, &x_array);CHKERRQ(ierr);
  ierr = VecGetArray(y, &y_array);CHKERRQ(ierr);
  if (!cufft->p_forward) {
    cufftResult result;
    /* create a plan, then execute it */
    switch(ndim) {
    case 1:
      result = cufftPlan1d(&cufft->p_forward, dim[0], CUFFT_C2C, 1);CHKERRQ(result != CUFFT_SUCCESS);
      break;
    case 2:
      result = cufftPlan2d(&cufft->p_forward, dim[0], dim[1], CUFFT_C2C);CHKERRQ(result != CUFFT_SUCCESS);
      break;
    case 3:
      result = cufftPlan3d(&cufft->p_forward, dim[0], dim[1], dim[2], CUFFT_C2C);CHKERRQ(result != CUFFT_SUCCESS);
      break;
    default:
      SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_USER, "Cannot create plan for %d-dimensional transform", ndim);
    }
  }
  /* transfer to GPU memory */
  cudaMemcpy(devArray, x_array, sizeof(cufftComplex)*dim[ndim], cudaMemcpyHostToDevice);
  /* execute transform */
  result = cufftExecC2C(cufft->p_forward, devArray, devArray, CUFFT_FORWARD);CHKERRQ(result != CUFFT_SUCCESS);
  /* transfer from GPU memory */
  cudaMemcpy(y_array, devArray, sizeof(cufftComplex)*dim[ndim], cudaMemcpyDeviceToHost);
  ierr = VecRestoreArray(y, &y_array);CHKERRQ(ierr);
  ierr = VecRestoreArray(x, &x_array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultTranspose_SeqCUFFT"
PetscErrorCode MatMultTranspose_SeqCUFFT(Mat A, Vec x, Vec y)
{
  Mat_CUFFT     *cufft    = (Mat_CUFFT *) A->data;
  cufftComplex  *devArray = cufft->devArray;
  PetscInt       ndim     = cufft->ndim, *dim = cufft->dim;
  PetscScalar   *x_array, *y_array;
  cufftResult    result;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetArray(x, &x_array);CHKERRQ(ierr);
  ierr = VecGetArray(y, &y_array);CHKERRQ(ierr);
  if (!cufft->p_backward) {
    /* create a plan, then execute it */
    switch(ndim) {
    case 1:
      result = cufftPlan1d(&cufft->p_backward, dim[0], CUFFT_C2C, 1);CHKERRQ(result != CUFFT_SUCCESS);
      break;
    case 2:
      result = cufftPlan2d(&cufft->p_backward, dim[0], dim[1], CUFFT_C2C);CHKERRQ(result != CUFFT_SUCCESS);
      break;
    case 3:
      result = cufftPlan3d(&cufft->p_backward, dim[0], dim[1], dim[2], CUFFT_C2C);CHKERRQ(result != CUFFT_SUCCESS);
      break;
    default:
      SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_USER, "Cannot create plan for %d-dimensional transform", ndim);
    }
  }
  /* transfer to GPU memory */
  cudaMemcpy(devArray, x_array, sizeof(cufftComplex)*dim[ndim], cudaMemcpyHostToDevice);
  /* execute transform */
  result = cufftExecC2C(cufft->p_forward, devArray, devArray, CUFFT_INVERSE);CHKERRQ(result != CUFFT_SUCCESS);
  /* transfer from GPU memory */
  cudaMemcpy(y_array, devArray, sizeof(cufftComplex)*dim[ndim], cudaMemcpyDeviceToHost);
  ierr = VecRestoreArray(y, &y_array);CHKERRQ(ierr);
  ierr = VecRestoreArray(x, &x_array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_SeqCUFFT"
PetscErrorCode MatDestroy_SeqCUFFT(Mat A)
{
  Mat_CUFFT     *cufft = (Mat_CUFFT *) A->data;
  cufftResult    result;
  PetscErrorCode ierr;

  PetscFunctionBegin;  
  ierr = PetscFree(cufft->dim);CHKERRQ(ierr);
  if (cufft->p_forward)  {result = cufftDestroy(cufft->p_forward);CHKERRQ(result != CUFFT_SUCCESS);}
  if (cufft->p_backward) {result = cufftDestroy(cufft->p_backward);CHKERRQ(result != CUFFT_SUCCESS);}
  cudaFree(cufft->devArray);
  ierr = PetscFree(A->data);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)A,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatCreateSeqCUFFT"
/*@
  MatCreateSeqCUFFT - Creates a matrix object that provides sequential FFT via the external package CUFFT

  Collective on MPI_Comm

  Input Parameters:
+ comm - MPI communicator, set to PETSC_COMM_SELF
. ndim - the ndim-dimensional transform
- dim  - array of size ndim, dim[i] contains the vector length in the i-dimension

  Output Parameter:
. A - the matrix

  Options Database Keys:
. -mat_cufft_plannerflags - set CUFFT planner flags

  Level: intermediate
@*/
PetscErrorCode  MatCreateSeqCUFFT(MPI_Comm comm, PetscInt ndim, const PetscInt dim[], Mat* A)
{
  Mat_CUFFT     *cufft;
  PetscInt       m, d;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ndim < 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_USER, "ndim %d must be > 0", ndim);
  ierr = MatCreate(comm, A);CHKERRQ(ierr);
  m = 1;
  for (d = 0; d < ndim; ++d){
    if (dim[d] < 0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_USER, "dim[%d]=%d must be > 0", d, dim[d]);
    m *= dim[d];
  }
  ierr = MatSetSizes(*A, m, m, m, m);CHKERRQ(ierr);  
  ierr = PetscObjectChangeTypeName((PetscObject)*A, MATSEQCUFFT);CHKERRQ(ierr);

  ierr = PetscNewLog(*A, Mat_CUFFT, &cufft);CHKERRQ(ierr);
  (*A)->data = (void*) cufft;
  ierr = PetscMalloc((ndim+1)*sizeof(PetscInt), &cufft->dim);CHKERRQ(ierr);
  ierr = PetscMemcpy(cufft->dim, dim, ndim*sizeof(PetscInt));CHKERRQ(ierr);
  cufft->ndim       = ndim;
  cufft->p_forward  = 0;
  cufft->p_backward = 0;
  cufft->dim[ndim]  = m;

  /* GPU memory allocation */
  cudaMalloc((void **) &cufft->devArray, sizeof(cufftComplex)*m);

  (*A)->ops->mult          = MatMult_SeqCUFFT;
  (*A)->ops->multtranspose = MatMultTranspose_SeqCUFFT;
  (*A)->assembled          = PETSC_TRUE;
  (*A)->ops->destroy       = MatDestroy_SeqCUFFT;

  /* get runtime options */
  ierr = PetscOptionsBegin(comm, ((PetscObject)(*A))->prefix, "CUFFT Options", "Mat");CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
