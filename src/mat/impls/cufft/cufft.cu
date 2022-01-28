
/*
    Provides an interface to the CUFFT package.
    Testing examples can be found in ~src/mat/tests
*/

#include <petscdevice.h>
#include <petsc/private/matimpl.h>          /*I "petscmat.h" I*/

typedef struct {
  PetscInt     ndim;
  PetscInt     *dim;
  cufftHandle  p_forward, p_backward;
  cufftComplex *devArray;
} Mat_CUFFT;

PetscErrorCode MatMult_SeqCUFFT(Mat A, Vec x, Vec y)
{
  Mat_CUFFT      *cufft    = (Mat_CUFFT*) A->data;
  cufftComplex   *devArray = cufft->devArray;
  PetscInt       ndim      = cufft->ndim, *dim = cufft->dim;
  PetscScalar    *x_array, *y_array;
  cufftResult    result;
  cudaError_t    cerr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetArray(x, &x_array);CHKERRQ(ierr);
  ierr = VecGetArray(y, &y_array);CHKERRQ(ierr);
  if (!cufft->p_forward) {
    cufftResult result;
    /* create a plan, then execute it */
    switch (ndim) {
    case 1:
      result = cufftPlan1d(&cufft->p_forward, dim[0], CUFFT_C2C, 1);CHKERRCUFFT(result);
      break;
    case 2:
      result = cufftPlan2d(&cufft->p_forward, dim[0], dim[1], CUFFT_C2C);CHKERRCUFFT(result);
      break;
    case 3:
      result = cufftPlan3d(&cufft->p_forward, dim[0], dim[1], dim[2], CUFFT_C2C);CHKERRCUFFT(result);
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_USER, "Cannot create plan for %d-dimensional transform", ndim);
    }
  }
  /* transfer to GPU memory */
  cerr = cudaMemcpy(devArray, x_array, sizeof(cufftComplex)*dim[ndim], cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
  /* execute transform */
  result = cufftExecC2C(cufft->p_forward, devArray, devArray, CUFFT_FORWARD);CHKERRCUFFT(result);
  /* transfer from GPU memory */
  cerr = cudaMemcpy(y_array, devArray, sizeof(cufftComplex)*dim[ndim], cudaMemcpyDeviceToHost);CHKERRCUDA(cerr);
  ierr = VecRestoreArray(y, &y_array);CHKERRQ(ierr);
  ierr = VecRestoreArray(x, &x_array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultTranspose_SeqCUFFT(Mat A, Vec x, Vec y)
{
  Mat_CUFFT      *cufft    = (Mat_CUFFT*) A->data;
  cufftComplex   *devArray = cufft->devArray;
  PetscInt       ndim      = cufft->ndim, *dim = cufft->dim;
  PetscScalar    *x_array, *y_array;
  cufftResult    result;
  cudaError_t    cerr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetArray(x, &x_array);CHKERRQ(ierr);
  ierr = VecGetArray(y, &y_array);CHKERRQ(ierr);
  if (!cufft->p_backward) {
    /* create a plan, then execute it */
    switch (ndim) {
    case 1:
      result = cufftPlan1d(&cufft->p_backward, dim[0], CUFFT_C2C, 1);CHKERRCUFFT(result);
      break;
    case 2:
      result = cufftPlan2d(&cufft->p_backward, dim[0], dim[1], CUFFT_C2C);CHKERRCUFFT(result);
      break;
    case 3:
      result = cufftPlan3d(&cufft->p_backward, dim[0], dim[1], dim[2], CUFFT_C2C);CHKERRCUFFT(result);
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_USER, "Cannot create plan for %d-dimensional transform", ndim);
    }
  }
  /* transfer to GPU memory */
  cerr = cudaMemcpy(devArray, x_array, sizeof(cufftComplex)*dim[ndim], cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
  /* execute transform */
  result = cufftExecC2C(cufft->p_forward, devArray, devArray, CUFFT_INVERSE);CHKERRCUFFT(result);
  /* transfer from GPU memory */
  cerr = cudaMemcpy(y_array, devArray, sizeof(cufftComplex)*dim[ndim], cudaMemcpyDeviceToHost);CHKERRCUDA(cerr);
  ierr = VecRestoreArray(y, &y_array);CHKERRQ(ierr);
  ierr = VecRestoreArray(x, &x_array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_SeqCUFFT(Mat A)
{
  Mat_CUFFT      *cufft = (Mat_CUFFT*) A->data;
  cufftResult    result;
  cudaError_t    cerr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(cufft->dim);CHKERRQ(ierr);
  if (cufft->p_forward)  {result = cufftDestroy(cufft->p_forward);CHKERRCUFFT(result);}
  if (cufft->p_backward) {result = cufftDestroy(cufft->p_backward);CHKERRCUFFT(result);}
  cerr = cudaFree(cufft->devArray);CHKERRCUDA(cerr);
  ierr = PetscFree(A->data);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)A,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  MatCreateSeqCUFFT - Creates a matrix object that provides sequential FFT via the external package CUFFT

  Collective

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
PetscErrorCode  MatCreateSeqCUFFT(MPI_Comm comm, PetscInt ndim, const PetscInt dim[], Mat *A)
{
  Mat_CUFFT      *cufft;
  PetscInt       m, d;
  PetscErrorCode ierr;
  cudaError_t    cerr;

  PetscFunctionBegin;
  PetscAssertFalse(ndim < 0,PETSC_COMM_SELF, PETSC_ERR_USER, "ndim %d must be > 0", ndim);
  ierr = MatCreate(comm, A);CHKERRQ(ierr);
  m    = 1;
  for (d = 0; d < ndim; ++d) {
    PetscAssertFalse(dim[d] < 0,PETSC_COMM_SELF, PETSC_ERR_USER, "dim[%d]=%d must be > 0", d, dim[d]);
    m *= dim[d];
  }
  ierr = MatSetSizes(*A, m, m, m, m);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)*A, MATSEQCUFFT);CHKERRQ(ierr);

  ierr       = PetscNewLog(*A,&cufft);CHKERRQ(ierr);
  (*A)->data = (void*) cufft;
  ierr       = PetscMalloc1(ndim+1, &cufft->dim);CHKERRQ(ierr);
  ierr       = PetscArraycpy(cufft->dim, dim, ndim);CHKERRQ(ierr);

  cufft->ndim       = ndim;
  cufft->p_forward  = 0;
  cufft->p_backward = 0;
  cufft->dim[ndim]  = m;

  /* GPU memory allocation */
  cerr = cudaMalloc((void**) &cufft->devArray, sizeof(cufftComplex)*m);CHKERRCUDA(cerr);

  (*A)->ops->mult          = MatMult_SeqCUFFT;
  (*A)->ops->multtranspose = MatMultTranspose_SeqCUFFT;
  (*A)->assembled          = PETSC_TRUE;
  (*A)->ops->destroy       = MatDestroy_SeqCUFFT;

  /* get runtime options */
  ierr = PetscOptionsBegin(comm, ((PetscObject)(*A))->prefix, "CUFFT Options", "Mat");CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
