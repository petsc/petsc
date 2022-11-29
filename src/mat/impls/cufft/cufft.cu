
/*
    Provides an interface to the CUFFT package.
    Testing examples can be found in ~src/mat/tests
*/

#include <petscdevice_cuda.h>
#include <petsc/private/matimpl.h> /*I "petscmat.h" I*/

typedef struct {
  PetscInt      ndim;
  PetscInt     *dim;
  cufftHandle   p_forward, p_backward;
  cufftComplex *devArray;
} Mat_CUFFT;

PetscErrorCode MatMult_SeqCUFFT(Mat A, Vec x, Vec y)
{
  Mat_CUFFT    *cufft    = (Mat_CUFFT *)A->data;
  cufftComplex *devArray = cufft->devArray;
  PetscInt      ndim = cufft->ndim, *dim = cufft->dim;
  PetscScalar  *x_array, *y_array;

  PetscFunctionBegin;
  PetscCall(VecGetArray(x, &x_array));
  PetscCall(VecGetArray(y, &y_array));
  if (!cufft->p_forward) {
    /* create a plan, then execute it */
    switch (ndim) {
    case 1:
      PetscCallCUFFT(cufftPlan1d(&cufft->p_forward, dim[0], CUFFT_C2C, 1));
      break;
    case 2:
      PetscCallCUFFT(cufftPlan2d(&cufft->p_forward, dim[0], dim[1], CUFFT_C2C));
      break;
    case 3:
      PetscCallCUFFT(cufftPlan3d(&cufft->p_forward, dim[0], dim[1], dim[2], CUFFT_C2C));
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_USER, "Cannot create plan for %" PetscInt_FMT "-dimensional transform", ndim);
    }
  }
  /* transfer to GPU memory */
  PetscCallCUDA(cudaMemcpy(devArray, x_array, sizeof(cufftComplex) * dim[ndim], cudaMemcpyHostToDevice));
  /* execute transform */
  PetscCallCUFFT(cufftExecC2C(cufft->p_forward, devArray, devArray, CUFFT_FORWARD));
  /* transfer from GPU memory */
  PetscCallCUDA(cudaMemcpy(y_array, devArray, sizeof(cufftComplex) * dim[ndim], cudaMemcpyDeviceToHost));
  PetscCall(VecRestoreArray(y, &y_array));
  PetscCall(VecRestoreArray(x, &x_array));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultTranspose_SeqCUFFT(Mat A, Vec x, Vec y)
{
  Mat_CUFFT    *cufft    = (Mat_CUFFT *)A->data;
  cufftComplex *devArray = cufft->devArray;
  PetscInt      ndim = cufft->ndim, *dim = cufft->dim;
  PetscScalar  *x_array, *y_array;

  PetscFunctionBegin;
  PetscCall(VecGetArray(x, &x_array));
  PetscCall(VecGetArray(y, &y_array));
  if (!cufft->p_backward) {
    /* create a plan, then execute it */
    switch (ndim) {
    case 1:
      PetscCallCUFFT(cufftPlan1d(&cufft->p_backward, dim[0], CUFFT_C2C, 1));
      break;
    case 2:
      PetscCallCUFFT(cufftPlan2d(&cufft->p_backward, dim[0], dim[1], CUFFT_C2C));
      break;
    case 3:
      PetscCallCUFFT(cufftPlan3d(&cufft->p_backward, dim[0], dim[1], dim[2], CUFFT_C2C));
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_USER, "Cannot create plan for %" PetscInt_FMT "-dimensional transform", ndim);
    }
  }
  /* transfer to GPU memory */
  PetscCallCUDA(cudaMemcpy(devArray, x_array, sizeof(cufftComplex) * dim[ndim], cudaMemcpyHostToDevice));
  /* execute transform */
  PetscCallCUFFT(cufftExecC2C(cufft->p_forward, devArray, devArray, CUFFT_INVERSE));
  /* transfer from GPU memory */
  PetscCallCUDA(cudaMemcpy(y_array, devArray, sizeof(cufftComplex) * dim[ndim], cudaMemcpyDeviceToHost));
  PetscCall(VecRestoreArray(y, &y_array));
  PetscCall(VecRestoreArray(x, &x_array));
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_SeqCUFFT(Mat A)
{
  Mat_CUFFT *cufft = (Mat_CUFFT *)A->data;

  PetscFunctionBegin;
  PetscCall(PetscFree(cufft->dim));
  if (cufft->p_forward) PetscCallCUFFT(cufftDestroy(cufft->p_forward));
  if (cufft->p_backward) PetscCallCUFFT(cufftDestroy(cufft->p_backward));
  PetscCallCUDA(cudaFree(cufft->devArray));
  PetscCall(PetscFree(A->data));
  PetscCall(PetscObjectChangeTypeName((PetscObject)A, 0));
  PetscFunctionReturn(0);
}

/*@
  MatCreateSeqCUFFT - Creates a matrix object that provides `MATSEQCUFFT` via the NVIDIA package CuFFT

  Collective

  Input Parameters:
+ comm - MPI communicator, set to `PETSC_COMM_SELF`
. ndim - the ndim-dimensional transform
- dim  - array of size ndim, dim[i] contains the vector length in the i-dimension

  Output Parameter:
. A - the matrix

  Options Database Keys:
. -mat_cufft_plannerflags - set CUFFT planner flags

  Level: intermediate

.seealso: `MATSEQCUFFT`
@*/
PetscErrorCode MatCreateSeqCUFFT(MPI_Comm comm, PetscInt ndim, const PetscInt dim[], Mat *A)
{
  Mat_CUFFT *cufft;
  PetscInt   m = 1;

  PetscFunctionBegin;
  PetscCheck(ndim >= 0, PETSC_COMM_SELF, PETSC_ERR_USER, "ndim %" PetscInt_FMT " must be > 0", ndim);
  if (ndim) PetscValidIntPointer(dim, 3);
  PetscValidPointer(A, 4);
  PetscCall(MatCreate(comm, A));
  for (PetscInt d = 0; d < ndim; ++d) {
    PetscCheck(dim[d] >= 0, PETSC_COMM_SELF, PETSC_ERR_USER, "dim[%" PetscInt_FMT "]=%" PetscInt_FMT " must be > 0", d, dim[d]);
    m *= dim[d];
  }
  PetscCall(MatSetSizes(*A, m, m, m, m));
  PetscCall(PetscObjectChangeTypeName((PetscObject)*A, MATSEQCUFFT));

  PetscCall(PetscNew(&cufft));
  (*A)->data = (void *)cufft;
  PetscCall(PetscMalloc1(ndim + 1, &cufft->dim));
  PetscCall(PetscArraycpy(cufft->dim, dim, ndim));

  cufft->ndim       = ndim;
  cufft->p_forward  = 0;
  cufft->p_backward = 0;
  cufft->dim[ndim]  = m;

  /* GPU memory allocation */
  PetscCallCUDA(cudaMalloc((void **)&cufft->devArray, sizeof(cufftComplex) * m));

  (*A)->ops->mult          = MatMult_SeqCUFFT;
  (*A)->ops->multtranspose = MatMultTranspose_SeqCUFFT;
  (*A)->assembled          = PETSC_TRUE;
  (*A)->ops->destroy       = MatDestroy_SeqCUFFT;

  /* get runtime options ...what options????? */
  PetscOptionsBegin(comm, ((PetscObject)(*A))->prefix, "CUFFT Options", "Mat");
  PetscOptionsEnd();
  PetscFunctionReturn(0);
}
