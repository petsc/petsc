/*
    Provides an interface to the FFT packages.
*/

#include <../src/mat/impls/fft/fft.h>   /*I "petscmat.h" I*/

PetscErrorCode MatDestroy_FFT(Mat A)
{
  Mat_FFT *fft = (Mat_FFT*)A->data;

  PetscFunctionBegin;
  if (fft->matdestroy) CHKERRQ((fft->matdestroy)(A));
  CHKERRQ(PetscFree(fft->dim));
  CHKERRQ(PetscFree(A->data));
  CHKERRQ(PetscObjectChangeTypeName((PetscObject)A,NULL));
  PetscFunctionReturn(0);
}

/*@C
      MatCreateFFT - Creates a matrix object that provides FFT via an external package

   Collective

   Input Parameters:
+   comm - MPI communicator
.   ndim - the ndim-dimensional transform
.   dim - array of size ndim, dim[i] contains the vector length in the i-dimension
-   type - package type, e.g., FFTW or MATSEQCUFFT

   Output Parameter:
.   A  - the matrix

   Options Database Keys:
.   -mat_fft_type - set FFT type fft or seqcufft

   Note: this serves as a base class for all FFT marix classes, currently MATFFTW or MATSEQCUFFT

   Level: intermediate

.seealso: MatCreateVecsFFTW()
@*/
PetscErrorCode MatCreateFFT(MPI_Comm comm,PetscInt ndim,const PetscInt dim[],MatType mattype,Mat *A)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;
  Mat            FFT;
  PetscInt       N,i;
  Mat_FFT        *fft;

  PetscFunctionBegin;
  PetscValidIntPointer(dim,3);
  PetscValidPointer(A,5);
  PetscCheck(ndim >= 1,comm,PETSC_ERR_USER,"ndim %" PetscInt_FMT " must be > 0",ndim);
  CHKERRMPI(MPI_Comm_size(comm, &size));

  CHKERRQ(MatCreate(comm,&FFT));
  CHKERRQ(PetscNewLog(FFT,&fft));
  FFT->data = (void*)fft;
  N         = 1;
  for (i=0; i<ndim; i++) {
    PetscCheck(dim[i] >= 1,PETSC_COMM_SELF,PETSC_ERR_USER,"dim[%" PetscInt_FMT "]=%" PetscInt_FMT " must be > 0",i,dim[i]);
    N *= dim[i];
  }

  CHKERRQ(PetscMalloc1(ndim,&fft->dim));
  CHKERRQ(PetscArraycpy(fft->dim,dim,ndim));

  fft->ndim = ndim;
  fft->n    = PETSC_DECIDE;
  fft->N    = N;
  fft->data = NULL;

  CHKERRQ(MatSetType(FFT,mattype));

  FFT->ops->destroy = MatDestroy_FFT;

  /* get runtime options... what options? */
  ierr = PetscObjectOptionsBegin((PetscObject)FFT);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  *A = FFT;
  PetscFunctionReturn(0);
}
