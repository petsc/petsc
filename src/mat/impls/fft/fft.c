
/*
    Provides an interface to the FFT packages.
*/

#include <../src/mat/impls/fft/fft.h>   /*I "petscmat.h" I*/

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_FFT"
PetscErrorCode MatDestroy_FFT(Mat A)
{
  PetscErrorCode ierr;
  Mat_FFT        *fft = (Mat_FFT*)A->data;

  PetscFunctionBegin;
  if (fft->matdestroy){
    ierr = (fft->matdestroy)(A);CHKERRQ(ierr);
  }
  ierr = PetscFree(fft->dim);CHKERRQ(ierr);
  ierr = PetscFree(A->data);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)A,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatCreateFFT"
/*@
      MatCreateFFT - Creates a matrix object that provides FFT via an external package

   Collective on MPI_Comm

   Input Parameter:
+   comm - MPI communicator
.   ndim - the ndim-dimensional transform
.   dim - array of size ndim, dim[i] contains the vector length in the i-dimension
-   type - package type, e.g., FFTW or FFTCU

   Output Parameter:
.   A  - the matrix

  Options Database Keys:
+ -mat_fft_type - set FFT type

   Level: intermediate
   
@*/
PetscErrorCode MatCreateFFT(MPI_Comm comm,PetscInt ndim,const PetscInt dim[],const MatType mattype,Mat* A)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;
  Mat            FFT;
  PetscInt       N,i;
  Mat_FFT        *fft;

  PetscFunctionBegin;
  if (ndim < 1) SETERRQ1(comm,PETSC_ERR_USER,"ndim %d must be > 0",ndim);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);

  ierr = MatCreate(comm,&FFT);CHKERRQ(ierr);
  ierr = PetscNewLog(FFT,Mat_FFT,&fft);CHKERRQ(ierr);
  FFT->data = (void*)fft;
  N = 1;
  for (i=0; i<ndim; i++){
    if (dim[i] < 1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"dim[%d]=%d must be > 0",i,dim[i]);
    N *= dim[i];
  }

  ierr = PetscMalloc(ndim*sizeof(PetscInt),&fft->dim);CHKERRQ(ierr);
  ierr = PetscMemcpy(fft->dim,dim,ndim*sizeof(PetscInt));CHKERRQ(ierr);

  fft->ndim = ndim;
  fft->n    = PETSC_DECIDE;
  fft->N    = N;
  fft->data = PETSC_NULL;
  
  ierr = MatSetType(FFT,mattype);CHKERRQ(ierr); 
  FFT->ops->destroy = MatDestroy_FFT;
 
  /* get runtime options */
  ierr = PetscOptionsBegin(((PetscObject)FFT)->comm,((PetscObject)FFT)->prefix,"FFT Options","Mat");CHKERRQ(ierr);
  PetscOptionsEnd();

  *A = FFT;
  PetscFunctionReturn(0);
}
