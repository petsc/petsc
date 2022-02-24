
/*
    Provides an interface to the FFTW package.
    Testing examples can be found in ~src/mat/tests
*/

#include <../src/mat/impls/fft/fft.h>   /*I "petscmat.h" I*/
EXTERN_C_BEGIN
#if !PetscDefined(HAVE_MPIUNI)
#include <fftw3-mpi.h>
#else
#include <fftw3.h>
#endif
EXTERN_C_END

typedef struct {
  ptrdiff_t    ndim_fftw,*dim_fftw;
#if defined(PETSC_USE_64BIT_INDICES)
  fftw_iodim64 *iodims;
#else
  fftw_iodim   *iodims;
#endif
  PetscInt     partial_dim;
  fftw_plan    p_forward,p_backward;
  unsigned     p_flag; /* planner flags, FFTW_ESTIMATE,FFTW_MEASURE, FFTW_PATIENT, FFTW_EXHAUSTIVE */
  PetscScalar  *finarray,*foutarray,*binarray,*boutarray; /* keep track of arrays becaue fftw plan should be
                                                            executed for the arrays with which the plan was created */
} Mat_FFTW;

extern PetscErrorCode MatMult_SeqFFTW(Mat,Vec,Vec);
extern PetscErrorCode MatMultTranspose_SeqFFTW(Mat,Vec,Vec);
extern PetscErrorCode MatDestroy_FFTW(Mat);
extern PetscErrorCode MatCreateVecsFFTW_FFTW(Mat,Vec*,Vec*,Vec*);
#if !PetscDefined(HAVE_MPIUNI)
extern PetscErrorCode MatMult_MPIFFTW(Mat,Vec,Vec);
extern PetscErrorCode MatMultTranspose_MPIFFTW(Mat,Vec,Vec);
extern PetscErrorCode VecDestroy_MPIFFTW(Vec);
#endif

/*
   MatMult_SeqFFTW performs forward DFT
   Input parameter:
     A - the matrix
     x - the vector on which FDFT will be performed

   Output parameter:
     y - vector that stores result of FDFT
*/
PetscErrorCode MatMult_SeqFFTW(Mat A,Vec x,Vec y)
{
  Mat_FFT        *fft  = (Mat_FFT*)A->data;
  Mat_FFTW       *fftw = (Mat_FFTW*)fft->data;
  const PetscScalar *x_array;
  PetscScalar    *y_array;
#if defined(PETSC_USE_COMPLEX)
#if defined(PETSC_USE_64BIT_INDICES)
  fftw_iodim64   *iodims;
#else
  fftw_iodim     *iodims;
#endif
  PetscInt       i;
#endif
  PetscInt       ndim = fft->ndim,*dim = fft->dim;

  PetscFunctionBegin;
  CHKERRQ(VecGetArrayRead(x,&x_array));
  CHKERRQ(VecGetArray(y,&y_array));
  if (!fftw->p_forward) { /* create a plan, then excute it */
    switch (ndim) {
    case 1:
#if defined(PETSC_USE_COMPLEX)
      fftw->p_forward = fftw_plan_dft_1d(dim[0],(fftw_complex*)x_array,(fftw_complex*)y_array,FFTW_FORWARD,fftw->p_flag);
#else
      fftw->p_forward = fftw_plan_dft_r2c_1d(dim[0],(double*)x_array,(fftw_complex*)y_array,fftw->p_flag);
#endif
      break;
    case 2:
#if defined(PETSC_USE_COMPLEX)
      fftw->p_forward = fftw_plan_dft_2d(dim[0],dim[1],(fftw_complex*)x_array,(fftw_complex*)y_array,FFTW_FORWARD,fftw->p_flag);
#else
      fftw->p_forward = fftw_plan_dft_r2c_2d(dim[0],dim[1],(double*)x_array,(fftw_complex*)y_array,fftw->p_flag);
#endif
      break;
    case 3:
#if defined(PETSC_USE_COMPLEX)
      fftw->p_forward = fftw_plan_dft_3d(dim[0],dim[1],dim[2],(fftw_complex*)x_array,(fftw_complex*)y_array,FFTW_FORWARD,fftw->p_flag);
#else
      fftw->p_forward = fftw_plan_dft_r2c_3d(dim[0],dim[1],dim[2],(double*)x_array,(fftw_complex*)y_array,fftw->p_flag);
#endif
      break;
    default:
#if defined(PETSC_USE_COMPLEX)
      iodims = fftw->iodims;
#if defined(PETSC_USE_64BIT_INDICES)
      if (ndim) {
        iodims[ndim-1].n = (ptrdiff_t)dim[ndim-1];
        iodims[ndim-1].is = iodims[ndim-1].os = 1;
        for (i=ndim-2; i>=0; --i) {
          iodims[i].n = (ptrdiff_t)dim[i];
          iodims[i].is = iodims[i].os = iodims[i+1].is * iodims[i+1].n;
        }
      }
      fftw->p_forward = fftw_plan_guru64_dft((int)ndim,(fftw_iodim64*)iodims,0,NULL,(fftw_complex*)x_array,(fftw_complex*)y_array,FFTW_FORWARD,fftw->p_flag);
#else
      if (ndim) {
        iodims[ndim-1].n = (int)dim[ndim-1];
        iodims[ndim-1].is = iodims[ndim-1].os = 1;
        for (i=ndim-2; i>=0; --i) {
          iodims[i].n = (int)dim[i];
          iodims[i].is = iodims[i].os = iodims[i+1].is * iodims[i+1].n;
        }
      }
      fftw->p_forward = fftw_plan_guru_dft((int)ndim,(fftw_iodim*)iodims,0,NULL,(fftw_complex*)x_array,(fftw_complex*)y_array,FFTW_FORWARD,fftw->p_flag);
#endif

#else
      fftw->p_forward = fftw_plan_dft_r2c(ndim,(int*)dim,(double*)x_array,(fftw_complex*)y_array,fftw->p_flag);
#endif
      break;
    }
    fftw->finarray  = (PetscScalar *) x_array;
    fftw->foutarray = y_array;
    /* Warning: if (fftw->p_flag!==FFTW_ESTIMATE) The data in the in/out arrays is overwritten!
                planning should be done before x is initialized! See FFTW manual sec2.1 or sec4 */
    fftw_execute(fftw->p_forward);
  } else { /* use existing plan */
    if (fftw->finarray != x_array || fftw->foutarray != y_array) { /* use existing plan on new arrays */
#if defined(PETSC_USE_COMPLEX)
      fftw_execute_dft(fftw->p_forward,(fftw_complex*)x_array,(fftw_complex*)y_array);
#else
      fftw_execute_dft_r2c(fftw->p_forward,(double*)x_array,(fftw_complex*)y_array);
#endif
    } else {
      fftw_execute(fftw->p_forward);
    }
  }
  CHKERRQ(VecRestoreArray(y,&y_array));
  CHKERRQ(VecRestoreArrayRead(x,&x_array));
  PetscFunctionReturn(0);
}

/* MatMultTranspose_SeqFFTW performs serial backward DFT
   Input parameter:
     A - the matrix
     x - the vector on which BDFT will be performed

   Output parameter:
     y - vector that stores result of BDFT
*/

PetscErrorCode MatMultTranspose_SeqFFTW(Mat A,Vec x,Vec y)
{
  Mat_FFT        *fft  = (Mat_FFT*)A->data;
  Mat_FFTW       *fftw = (Mat_FFTW*)fft->data;
  const PetscScalar *x_array;
  PetscScalar    *y_array;
  PetscInt       ndim = fft->ndim,*dim = fft->dim;
#if defined(PETSC_USE_COMPLEX)
#if defined(PETSC_USE_64BIT_INDICES)
  fftw_iodim64   *iodims = fftw->iodims;
#else
  fftw_iodim     *iodims = fftw->iodims;
#endif
#endif

  PetscFunctionBegin;
  CHKERRQ(VecGetArrayRead(x,&x_array));
  CHKERRQ(VecGetArray(y,&y_array));
  if (!fftw->p_backward) { /* create a plan, then excute it */
    switch (ndim) {
    case 1:
#if defined(PETSC_USE_COMPLEX)
      fftw->p_backward = fftw_plan_dft_1d(dim[0],(fftw_complex*)x_array,(fftw_complex*)y_array,FFTW_BACKWARD,fftw->p_flag);
#else
      fftw->p_backward= fftw_plan_dft_c2r_1d(dim[0],(fftw_complex*)x_array,(double*)y_array,fftw->p_flag);
#endif
      break;
    case 2:
#if defined(PETSC_USE_COMPLEX)
      fftw->p_backward = fftw_plan_dft_2d(dim[0],dim[1],(fftw_complex*)x_array,(fftw_complex*)y_array,FFTW_BACKWARD,fftw->p_flag);
#else
      fftw->p_backward= fftw_plan_dft_c2r_2d(dim[0],dim[1],(fftw_complex*)x_array,(double*)y_array,fftw->p_flag);
#endif
      break;
    case 3:
#if defined(PETSC_USE_COMPLEX)
      fftw->p_backward = fftw_plan_dft_3d(dim[0],dim[1],dim[2],(fftw_complex*)x_array,(fftw_complex*)y_array,FFTW_BACKWARD,fftw->p_flag);
#else
      fftw->p_backward= fftw_plan_dft_c2r_3d(dim[0],dim[1],dim[2],(fftw_complex*)x_array,(double*)y_array,fftw->p_flag);
#endif
      break;
    default:
#if defined(PETSC_USE_COMPLEX)
#if defined(PETSC_USE_64BIT_INDICES)
      fftw->p_backward = fftw_plan_guru64_dft((int)ndim,(fftw_iodim64*)iodims,0,NULL,(fftw_complex*)x_array,(fftw_complex*)y_array,FFTW_BACKWARD,fftw->p_flag);
#else
      fftw->p_backward = fftw_plan_guru_dft((int)ndim,iodims,0,NULL,(fftw_complex*)x_array,(fftw_complex*)y_array,FFTW_BACKWARD,fftw->p_flag);
#endif
#else
      fftw->p_backward= fftw_plan_dft_c2r((int)ndim,(int*)dim,(fftw_complex*)x_array,(double*)y_array,fftw->p_flag);
#endif
      break;
    }
    fftw->binarray  = (PetscScalar *) x_array;
    fftw->boutarray = y_array;
    fftw_execute(fftw->p_backward);
  } else { /* use existing plan */
    if (fftw->binarray != x_array || fftw->boutarray != y_array) { /* use existing plan on new arrays */
#if defined(PETSC_USE_COMPLEX)
      fftw_execute_dft(fftw->p_backward,(fftw_complex*)x_array,(fftw_complex*)y_array);
#else
      fftw_execute_dft_c2r(fftw->p_backward,(fftw_complex*)x_array,(double*)y_array);
#endif
    } else {
      fftw_execute(fftw->p_backward);
    }
  }
  CHKERRQ(VecRestoreArray(y,&y_array));
  CHKERRQ(VecRestoreArrayRead(x,&x_array));
  PetscFunctionReturn(0);
}

#if !PetscDefined(HAVE_MPIUNI)
/* MatMult_MPIFFTW performs forward DFT in parallel
   Input parameter:
     A - the matrix
     x - the vector on which FDFT will be performed

   Output parameter:
   y   - vector that stores result of FDFT
*/
PetscErrorCode MatMult_MPIFFTW(Mat A,Vec x,Vec y)
{
  Mat_FFT        *fft  = (Mat_FFT*)A->data;
  Mat_FFTW       *fftw = (Mat_FFTW*)fft->data;
  const PetscScalar *x_array;
  PetscScalar    *y_array;
  PetscInt       ndim = fft->ndim,*dim = fft->dim;
  MPI_Comm       comm;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)A,&comm));
  CHKERRQ(VecGetArrayRead(x,&x_array));
  CHKERRQ(VecGetArray(y,&y_array));
  if (!fftw->p_forward) { /* create a plan, then excute it */
    switch (ndim) {
    case 1:
#if defined(PETSC_USE_COMPLEX)
      fftw->p_forward = fftw_mpi_plan_dft_1d(dim[0],(fftw_complex*)x_array,(fftw_complex*)y_array,comm,FFTW_FORWARD,fftw->p_flag);
#else
      SETERRQ(comm,PETSC_ERR_SUP,"not support for real numbers yet");
#endif
      break;
    case 2:
#if defined(PETSC_USE_COMPLEX) /* For complex transforms call fftw_mpi_plan_dft, for real transforms call fftw_mpi_plan_dft_r2c */
      fftw->p_forward = fftw_mpi_plan_dft_2d(dim[0],dim[1],(fftw_complex*)x_array,(fftw_complex*)y_array,comm,FFTW_FORWARD,fftw->p_flag);
#else
      fftw->p_forward = fftw_mpi_plan_dft_r2c_2d(dim[0],dim[1],(double*)x_array,(fftw_complex*)y_array,comm,FFTW_ESTIMATE);
#endif
      break;
    case 3:
#if defined(PETSC_USE_COMPLEX)
      fftw->p_forward = fftw_mpi_plan_dft_3d(dim[0],dim[1],dim[2],(fftw_complex*)x_array,(fftw_complex*)y_array,comm,FFTW_FORWARD,fftw->p_flag);
#else
      fftw->p_forward = fftw_mpi_plan_dft_r2c_3d(dim[0],dim[1],dim[2],(double*)x_array,(fftw_complex*)y_array,comm,FFTW_ESTIMATE);
#endif
      break;
    default:
#if defined(PETSC_USE_COMPLEX)
      fftw->p_forward = fftw_mpi_plan_dft(fftw->ndim_fftw,fftw->dim_fftw,(fftw_complex*)x_array,(fftw_complex*)y_array,comm,FFTW_FORWARD,fftw->p_flag);
#else
      fftw->p_forward = fftw_mpi_plan_dft_r2c(fftw->ndim_fftw,fftw->dim_fftw,(double*)x_array,(fftw_complex*)y_array,comm,FFTW_ESTIMATE);
#endif
      break;
    }
    fftw->finarray  = (PetscScalar *) x_array;
    fftw->foutarray = y_array;
    /* Warning: if (fftw->p_flag!==FFTW_ESTIMATE) The data in the in/out arrays is overwritten!
                planning should be done before x is initialized! See FFTW manual sec2.1 or sec4 */
    fftw_execute(fftw->p_forward);
  } else { /* use existing plan */
    if (fftw->finarray != x_array || fftw->foutarray != y_array) { /* use existing plan on new arrays */
      fftw_execute_dft(fftw->p_forward,(fftw_complex*)x_array,(fftw_complex*)y_array);
    } else {
      fftw_execute(fftw->p_forward);
    }
  }
  CHKERRQ(VecRestoreArray(y,&y_array));
  CHKERRQ(VecRestoreArrayRead(x,&x_array));
  PetscFunctionReturn(0);
}

/*
   MatMultTranspose_MPIFFTW performs parallel backward DFT
   Input parameter:
     A - the matrix
     x - the vector on which BDFT will be performed

   Output parameter:
     y - vector that stores result of BDFT
*/
PetscErrorCode MatMultTranspose_MPIFFTW(Mat A,Vec x,Vec y)
{
  Mat_FFT        *fft  = (Mat_FFT*)A->data;
  Mat_FFTW       *fftw = (Mat_FFTW*)fft->data;
  const PetscScalar *x_array;
  PetscScalar    *y_array;
  PetscInt       ndim = fft->ndim,*dim = fft->dim;
  MPI_Comm       comm;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)A,&comm));
  CHKERRQ(VecGetArrayRead(x,&x_array));
  CHKERRQ(VecGetArray(y,&y_array));
  if (!fftw->p_backward) { /* create a plan, then excute it */
    switch (ndim) {
    case 1:
#if defined(PETSC_USE_COMPLEX)
      fftw->p_backward = fftw_mpi_plan_dft_1d(dim[0],(fftw_complex*)x_array,(fftw_complex*)y_array,comm,FFTW_BACKWARD,fftw->p_flag);
#else
      SETERRQ(comm,PETSC_ERR_SUP,"not support for real numbers yet");
#endif
      break;
    case 2:
#if defined(PETSC_USE_COMPLEX) /* For complex transforms call fftw_mpi_plan_dft with flag FFTW_BACKWARD, for real transforms call fftw_mpi_plan_dft_c2r */
      fftw->p_backward = fftw_mpi_plan_dft_2d(dim[0],dim[1],(fftw_complex*)x_array,(fftw_complex*)y_array,comm,FFTW_BACKWARD,fftw->p_flag);
#else
      fftw->p_backward = fftw_mpi_plan_dft_c2r_2d(dim[0],dim[1],(fftw_complex*)x_array,(double*)y_array,comm,FFTW_ESTIMATE);
#endif
      break;
    case 3:
#if defined(PETSC_USE_COMPLEX)
      fftw->p_backward = fftw_mpi_plan_dft_3d(dim[0],dim[1],dim[2],(fftw_complex*)x_array,(fftw_complex*)y_array,comm,FFTW_BACKWARD,fftw->p_flag);
#else
      fftw->p_backward = fftw_mpi_plan_dft_c2r_3d(dim[0],dim[1],dim[2],(fftw_complex*)x_array,(double*)y_array,comm,FFTW_ESTIMATE);
#endif
      break;
    default:
#if defined(PETSC_USE_COMPLEX)
      fftw->p_backward = fftw_mpi_plan_dft(fftw->ndim_fftw,fftw->dim_fftw,(fftw_complex*)x_array,(fftw_complex*)y_array,comm,FFTW_BACKWARD,fftw->p_flag);
#else
      fftw->p_backward = fftw_mpi_plan_dft_c2r(fftw->ndim_fftw,fftw->dim_fftw,(fftw_complex*)x_array,(double*)y_array,comm,FFTW_ESTIMATE);
#endif
      break;
    }
    fftw->binarray  = (PetscScalar *) x_array;
    fftw->boutarray = y_array;
    fftw_execute(fftw->p_backward);
  } else { /* use existing plan */
    if (fftw->binarray != x_array || fftw->boutarray != y_array) { /* use existing plan on new arrays */
      fftw_execute_dft(fftw->p_backward,(fftw_complex*)x_array,(fftw_complex*)y_array);
    } else {
      fftw_execute(fftw->p_backward);
    }
  }
  CHKERRQ(VecRestoreArray(y,&y_array));
  CHKERRQ(VecRestoreArrayRead(x,&x_array));
  PetscFunctionReturn(0);
}
#endif

PetscErrorCode MatDestroy_FFTW(Mat A)
{
  Mat_FFT        *fft  = (Mat_FFT*)A->data;
  Mat_FFTW       *fftw = (Mat_FFTW*)fft->data;

  PetscFunctionBegin;
  fftw_destroy_plan(fftw->p_forward);
  fftw_destroy_plan(fftw->p_backward);
  if (fftw->iodims) {
    free(fftw->iodims);
  }
  CHKERRQ(PetscFree(fftw->dim_fftw));
  CHKERRQ(PetscFree(fft->data));
#if !PetscDefined(HAVE_MPIUNI)
  fftw_mpi_cleanup();
#endif
  PetscFunctionReturn(0);
}

#if !PetscDefined(HAVE_MPIUNI)
#include <../src/vec/vec/impls/mpi/pvecimpl.h>   /*I  "petscvec.h"   I*/
PetscErrorCode VecDestroy_MPIFFTW(Vec v)
{
  PetscScalar    *array;

  PetscFunctionBegin;
  CHKERRQ(VecGetArray(v,&array));
  fftw_free((fftw_complex*)array);
  CHKERRQ(VecRestoreArray(v,&array));
  CHKERRQ(VecDestroy_MPI(v));
  PetscFunctionReturn(0);
}
#endif

#if !PetscDefined(HAVE_MPIUNI)
static PetscErrorCode VecDuplicate_FFTW_fin(Vec fin,Vec *fin_new)
{
  Mat            A;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectQuery((PetscObject)fin,"FFTmatrix",(PetscObject*)&A));
  CHKERRQ(MatCreateVecsFFTW_FFTW(A,fin_new,NULL,NULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode VecDuplicate_FFTW_fout(Vec fout,Vec *fout_new)
{
  Mat            A;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectQuery((PetscObject)fout,"FFTmatrix",(PetscObject*)&A));
  CHKERRQ(MatCreateVecsFFTW_FFTW(A,NULL,fout_new,NULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode VecDuplicate_FFTW_bout(Vec bout, Vec *bout_new)
{
  Mat            A;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectQuery((PetscObject)bout,"FFTmatrix",(PetscObject*)&A));
  CHKERRQ(MatCreateVecsFFTW_FFTW(A,NULL,NULL,bout_new));
  PetscFunctionReturn(0);
}
#endif

/*@
   MatCreateVecsFFTW - Get vector(s) compatible with the matrix, i.e. with the
     parallel layout determined by FFTW

   Collective on Mat

   Input Parameter:
.   A - the matrix

   Output Parameters:
+   x - (optional) input vector of forward FFTW
.   y - (optional) output vector of forward FFTW
-   z - (optional) output vector of backward FFTW

  Level: advanced

  Note: The parallel layout of output of forward FFTW is always same as the input
        of the backward FFTW. But parallel layout of the input vector of forward
        FFTW might not be same as the output of backward FFTW.
        Also note that we need to provide enough space while doing parallel real transform.
        We need to pad extra zeros at the end of last dimension. For this reason the one needs to
        invoke the routine fftw_mpi_local_size_transposed routines. Remember one has to change the
        last dimension from n to n/2+1 while invoking this routine. The number of zeros to be padded
        depends on if the last dimension is even or odd. If the last dimension is even need to pad two
        zeros if it is odd only one zero is needed.
        Lastly one needs some scratch space at the end of data set in each process. alloc_local
        figures out how much space is needed, i.e. it figures out the data+scratch space for
        each processor and returns that.

.seealso: MatCreateFFT()
@*/
PetscErrorCode MatCreateVecsFFTW(Mat A,Vec *x,Vec *y,Vec *z)
{
  PetscFunctionBegin;
  CHKERRQ(PetscUseMethod(A,"MatCreateVecsFFTW_C",(Mat,Vec*,Vec*,Vec*),(A,x,y,z)));
  PetscFunctionReturn(0);
}

PetscErrorCode  MatCreateVecsFFTW_FFTW(Mat A,Vec *fin,Vec *fout,Vec *bout)
{
  PetscMPIInt    size,rank;
  MPI_Comm       comm;
  Mat_FFT        *fft  = (Mat_FFT*)A->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  CHKERRQ(PetscObjectGetComm((PetscObject)A,&comm));

  CHKERRMPI(MPI_Comm_size(comm, &size));
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  if (size == 1) { /* sequential case */
#if defined(PETSC_USE_COMPLEX)
    if (fin)  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,fft->N,fin));
    if (fout) CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,fft->N,fout));
    if (bout) CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,fft->N,bout));
#else
    if (fin) CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,fft->n,fin));
    if (fout) CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,fft->n,fout));
    if (bout) CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,fft->n,bout));
#endif
#if !PetscDefined(HAVE_MPIUNI)
  } else { /* parallel cases */
    Mat_FFTW     *fftw = (Mat_FFTW*)fft->data;
    PetscInt     ndim  = fft->ndim,*dim = fft->dim;
    ptrdiff_t    alloc_local,local_n0,local_0_start;
    ptrdiff_t    local_n1;
    fftw_complex *data_fout;
    ptrdiff_t    local_1_start;
#if defined(PETSC_USE_COMPLEX)
    fftw_complex *data_fin,*data_bout;
#else
    double       *data_finr,*data_boutr;
    PetscInt     n1,N1;
    ptrdiff_t    temp;
#endif

    switch (ndim) {
    case 1:
#if !defined(PETSC_USE_COMPLEX)
      SETERRQ(comm,PETSC_ERR_SUP,"FFTW does not allow parallel real 1D transform");
#else
      alloc_local = fftw_mpi_local_size_1d(dim[0],comm,FFTW_FORWARD,FFTW_ESTIMATE,&local_n0,&local_0_start,&local_n1,&local_1_start);
      if (fin) {
        data_fin  = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*alloc_local);
        CHKERRQ(VecCreateMPIWithArray(comm,1,local_n0,fft->N,(const PetscScalar*)data_fin,fin));
        CHKERRQ(PetscObjectCompose((PetscObject)*fin,"FFTmatrix",(PetscObject)A));
        (*fin)->ops->duplicate = VecDuplicate_FFTW_fin;
        (*fin)->ops->destroy   = VecDestroy_MPIFFTW;
      }
      if (fout) {
        data_fout = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*alloc_local);
        CHKERRQ(VecCreateMPIWithArray(comm,1,local_n1,fft->N,(const PetscScalar*)data_fout,fout));
        CHKERRQ(PetscObjectCompose((PetscObject)*fout,"FFTmatrix",(PetscObject)A));
        (*fout)->ops->duplicate = VecDuplicate_FFTW_fout;
        (*fout)->ops->destroy   = VecDestroy_MPIFFTW;
      }
      alloc_local = fftw_mpi_local_size_1d(dim[0],comm,FFTW_BACKWARD,FFTW_ESTIMATE,&local_n0,&local_0_start,&local_n1,&local_1_start);
      if (bout) {
        data_bout = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*alloc_local);
        CHKERRQ(VecCreateMPIWithArray(comm,1,local_n1,fft->N,(const PetscScalar*)data_bout,bout));
        CHKERRQ(PetscObjectCompose((PetscObject)*bout,"FFTmatrix",(PetscObject)A));
        (*bout)->ops->duplicate = VecDuplicate_FFTW_fout;
        (*bout)->ops->destroy   = VecDestroy_MPIFFTW;
      }
      break;
#endif
    case 2:
#if !defined(PETSC_USE_COMPLEX) /* Note that N1 is no more the product of individual dimensions */
      alloc_local =  fftw_mpi_local_size_2d_transposed(dim[0],dim[1]/2+1,comm,&local_n0,&local_0_start,&local_n1,&local_1_start);
      N1          = 2*dim[0]*(dim[1]/2+1); n1 = 2*local_n0*(dim[1]/2+1);
      if (fin) {
        data_finr = (double*)fftw_malloc(sizeof(double)*alloc_local*2);
        CHKERRQ(VecCreateMPIWithArray(comm,1,(PetscInt)n1,N1,(PetscScalar*)data_finr,fin));
        CHKERRQ(PetscObjectCompose((PetscObject)*fin,"FFTmatrix",(PetscObject)A));
        (*fin)->ops->duplicate = VecDuplicate_FFTW_fin;
        (*fin)->ops->destroy   = VecDestroy_MPIFFTW;
      }
      if (fout) {
        data_fout = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*alloc_local);
        CHKERRQ(VecCreateMPIWithArray(comm,1,(PetscInt)n1,N1,(PetscScalar*)data_fout,fout));
        CHKERRQ(PetscObjectCompose((PetscObject)*fout,"FFTmatrix",(PetscObject)A));
        (*fout)->ops->duplicate = VecDuplicate_FFTW_fout;
        (*fout)->ops->destroy   = VecDestroy_MPIFFTW;
      }
      if (bout) {
        data_boutr = (double*)fftw_malloc(sizeof(double)*alloc_local*2);
        CHKERRQ(VecCreateMPIWithArray(comm,1,(PetscInt)n1,N1,(PetscScalar*)data_boutr,bout));
        CHKERRQ(PetscObjectCompose((PetscObject)*bout,"FFTmatrix",(PetscObject)A));
        (*bout)->ops->duplicate = VecDuplicate_FFTW_bout;
        (*bout)->ops->destroy   = VecDestroy_MPIFFTW;
      }
#else
      /* Get local size */
      alloc_local = fftw_mpi_local_size_2d(dim[0],dim[1],comm,&local_n0,&local_0_start);
      if (fin) {
        data_fin = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*alloc_local);
        CHKERRQ(VecCreateMPIWithArray(comm,1,fft->n,fft->N,(const PetscScalar*)data_fin,fin));
        CHKERRQ(PetscObjectCompose((PetscObject)*fin,"FFTmatrix",(PetscObject)A));
        (*fin)->ops->duplicate = VecDuplicate_FFTW_fin;
        (*fin)->ops->destroy   = VecDestroy_MPIFFTW;
      }
      if (fout) {
        data_fout = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*alloc_local);
        CHKERRQ(VecCreateMPIWithArray(comm,1,fft->n,fft->N,(const PetscScalar*)data_fout,fout));
        CHKERRQ(PetscObjectCompose((PetscObject)*fout,"FFTmatrix",(PetscObject)A));
        (*fout)->ops->duplicate = VecDuplicate_FFTW_fout;
        (*fout)->ops->destroy   = VecDestroy_MPIFFTW;
      }
      if (bout) {
        data_bout = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*alloc_local);
        CHKERRQ(VecCreateMPIWithArray(comm,1,fft->n,fft->N,(const PetscScalar*)data_bout,bout));
        CHKERRQ(PetscObjectCompose((PetscObject)*bout,"FFTmatrix",(PetscObject)A));
        (*bout)->ops->duplicate = VecDuplicate_FFTW_bout;
        (*bout)->ops->destroy   = VecDestroy_MPIFFTW;
      }
#endif
      break;
    case 3:
#if !defined(PETSC_USE_COMPLEX)
      alloc_local =  fftw_mpi_local_size_3d_transposed(dim[0],dim[1],dim[2]/2+1,comm,&local_n0,&local_0_start,&local_n1,&local_1_start);
      N1 = 2*dim[0]*dim[1]*(dim[2]/2+1); n1 = 2*local_n0*dim[1]*(dim[2]/2+1);
      if (fin) {
        data_finr = (double*)fftw_malloc(sizeof(double)*alloc_local*2);
        CHKERRQ(VecCreateMPIWithArray(comm,1,(PetscInt)n1,N1,(PetscScalar*)data_finr,fin));
        CHKERRQ(PetscObjectCompose((PetscObject)*fin,"FFTmatrix",(PetscObject)A));
        (*fin)->ops->duplicate = VecDuplicate_FFTW_fin;
        (*fin)->ops->destroy   = VecDestroy_MPIFFTW;
      }
      if (fout) {
        data_fout=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*alloc_local);
        CHKERRQ(VecCreateMPIWithArray(comm,1,n1,N1,(PetscScalar*)data_fout,fout));
        CHKERRQ(PetscObjectCompose((PetscObject)*fout,"FFTmatrix",(PetscObject)A));
        (*fout)->ops->duplicate = VecDuplicate_FFTW_fout;
        (*fout)->ops->destroy   = VecDestroy_MPIFFTW;
      }
      if (bout) {
        data_boutr=(double*)fftw_malloc(sizeof(double)*alloc_local*2);
        CHKERRQ(VecCreateMPIWithArray(comm,1,(PetscInt)n1,N1,(PetscScalar*)data_boutr,bout));
        CHKERRQ(PetscObjectCompose((PetscObject)*bout,"FFTmatrix",(PetscObject)A));
        (*bout)->ops->duplicate = VecDuplicate_FFTW_bout;
        (*bout)->ops->destroy   = VecDestroy_MPIFFTW;
      }
#else
      alloc_local = fftw_mpi_local_size_3d(dim[0],dim[1],dim[2],comm,&local_n0,&local_0_start);
      if (fin) {
        data_fin  = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*alloc_local);
        CHKERRQ(VecCreateMPIWithArray(comm,1,fft->n,fft->N,(const PetscScalar*)data_fin,fin));
        CHKERRQ(PetscObjectCompose((PetscObject)*fin,"FFTmatrix",(PetscObject)A));
        (*fin)->ops->duplicate = VecDuplicate_FFTW_fin;
        (*fin)->ops->destroy   = VecDestroy_MPIFFTW;
      }
      if (fout) {
        data_fout = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*alloc_local);
        CHKERRQ(VecCreateMPIWithArray(comm,1,fft->n,fft->N,(const PetscScalar*)data_fout,fout));
        CHKERRQ(PetscObjectCompose((PetscObject)*fout,"FFTmatrix",(PetscObject)A));
        (*fout)->ops->duplicate = VecDuplicate_FFTW_fout;
        (*fout)->ops->destroy   = VecDestroy_MPIFFTW;
      }
      if (bout) {
        data_bout  = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*alloc_local);
        CHKERRQ(VecCreateMPIWithArray(comm,1,fft->n,fft->N,(const PetscScalar*)data_bout,bout));
        CHKERRQ(PetscObjectCompose((PetscObject)*bout,"FFTmatrix",(PetscObject)A));
        (*bout)->ops->duplicate = VecDuplicate_FFTW_bout;
        (*bout)->ops->destroy   = VecDestroy_MPIFFTW;
      }
#endif
      break;
    default:
#if !defined(PETSC_USE_COMPLEX)
      temp = (fftw->dim_fftw)[fftw->ndim_fftw-1];
      (fftw->dim_fftw)[fftw->ndim_fftw-1] = temp/2 + 1;
      alloc_local = fftw_mpi_local_size_transposed(fftw->ndim_fftw,fftw->dim_fftw,comm,&local_n0,&local_0_start,&local_n1,&local_1_start);
      N1          = 2*fft->N*(PetscInt)((fftw->dim_fftw)[fftw->ndim_fftw-1])/((PetscInt) temp);
      (fftw->dim_fftw)[fftw->ndim_fftw-1] = temp;

      if (fin) {
        data_finr=(double*)fftw_malloc(sizeof(double)*alloc_local*2);
        CHKERRQ(VecCreateMPIWithArray(comm,1,fft->n,N1,(PetscScalar*)data_finr,fin));
        CHKERRQ(PetscObjectCompose((PetscObject)*fin,"FFTmatrix",(PetscObject)A));
        (*fin)->ops->duplicate = VecDuplicate_FFTW_fin;
        (*fin)->ops->destroy   = VecDestroy_MPIFFTW;
      }
      if (fout) {
        data_fout=(fftw_complex*)fftw_malloc(sizeof(fftw_complex)*alloc_local);
        CHKERRQ(VecCreateMPIWithArray(comm,1,fft->n,N1,(PetscScalar*)data_fout,fout));
        CHKERRQ(PetscObjectCompose((PetscObject)*fout,"FFTmatrix",(PetscObject)A));
        (*fout)->ops->duplicate = VecDuplicate_FFTW_fout;
        (*fout)->ops->destroy   = VecDestroy_MPIFFTW;
      }
      if (bout) {
        data_boutr=(double*)fftw_malloc(sizeof(double)*alloc_local*2);
        CHKERRQ(VecCreateMPIWithArray(comm,1,fft->n,N1,(PetscScalar*)data_boutr,bout));
        CHKERRQ(PetscObjectCompose((PetscObject)*bout,"FFTmatrix",(PetscObject)A));
        (*bout)->ops->duplicate = VecDuplicate_FFTW_bout;
        (*bout)->ops->destroy   = VecDestroy_MPIFFTW;
      }
#else
      alloc_local = fftw_mpi_local_size(fftw->ndim_fftw,fftw->dim_fftw,comm,&local_n0,&local_0_start);
      if (fin) {
        data_fin  = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*alloc_local);
        CHKERRQ(VecCreateMPIWithArray(comm,1,fft->n,fft->N,(const PetscScalar*)data_fin,fin));
        CHKERRQ(PetscObjectCompose((PetscObject)*fin,"FFTmatrix",(PetscObject)A));
        (*fin)->ops->duplicate = VecDuplicate_FFTW_fin;
        (*fin)->ops->destroy   = VecDestroy_MPIFFTW;
      }
      if (fout) {
        data_fout = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*alloc_local);
        CHKERRQ(VecCreateMPIWithArray(comm,1,fft->n,fft->N,(const PetscScalar*)data_fout,fout));
        CHKERRQ(PetscObjectCompose((PetscObject)*fout,"FFTmatrix",(PetscObject)A));
        (*fout)->ops->duplicate = VecDuplicate_FFTW_fout;
        (*fout)->ops->destroy   = VecDestroy_MPIFFTW;
      }
      if (bout) {
        data_bout  = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*alloc_local);
        CHKERRQ(VecCreateMPIWithArray(comm,1,fft->n,fft->N,(const PetscScalar*)data_bout,bout));
        CHKERRQ(PetscObjectCompose((PetscObject)*bout,"FFTmatrix",(PetscObject)A));
        (*bout)->ops->duplicate = VecDuplicate_FFTW_bout;
        (*bout)->ops->destroy   = VecDestroy_MPIFFTW;
      }
#endif
      break;
    }
    /* fftw vectors have their data array allocated by fftw_malloc, such that v->array=xxx but
       v->array_allocated=NULL. A regular replacearray call won't free the memory and only causes
       memory leaks. We void these pointers here to avoid acident uses.
    */
    if (fin)  (*fin)->ops->replacearray = NULL;
    if (fout) (*fout)->ops->replacearray = NULL;
    if (bout) (*bout)->ops->replacearray = NULL;
#endif
  }
  PetscFunctionReturn(0);
}

/*@
   VecScatterPetscToFFTW - Copies the PETSc vector to the vector that goes into FFTW block.

   Collective on Mat

   Input Parameters:
+  A - FFTW matrix
-  x - the PETSc vector

   Output Parameters:
.  y - the FFTW vector

  Options Database Keys:
. -mat_fftw_plannerflags - set FFTW planner flags

   Level: intermediate

   Note: For real parallel FFT, FFTW requires insertion of extra space at the end of last dimension. This required even when
         one is not doing in-place transform. The last dimension size must be changed to 2*(dim[last]/2+1) to accommodate these extra
         zeros. This routine does that job by scattering operation.

.seealso: VecScatterFFTWToPetsc()
@*/
PetscErrorCode VecScatterPetscToFFTW(Mat A,Vec x,Vec y)
{
  PetscFunctionBegin;
  CHKERRQ(PetscUseMethod(A,"VecScatterPetscToFFTW_C",(Mat,Vec,Vec),(A,x,y)));
  PetscFunctionReturn(0);
}

PetscErrorCode VecScatterPetscToFFTW_FFTW(Mat A,Vec x,Vec y)
{
  MPI_Comm       comm;
  Mat_FFT        *fft  = (Mat_FFT*)A->data;
  PetscInt       low;
  PetscMPIInt    rank,size;
  PetscInt       vsize,vsize1;
  VecScatter     vecscat;
  IS             list1;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)A,&comm));
  CHKERRMPI(MPI_Comm_size(comm, &size));
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  CHKERRQ(VecGetOwnershipRange(y,&low,NULL));

  if (size==1) {
    CHKERRQ(VecGetSize(x,&vsize));
    CHKERRQ(VecGetSize(y,&vsize1));
    CHKERRQ(ISCreateStride(PETSC_COMM_SELF,fft->N,0,1,&list1));
    CHKERRQ(VecScatterCreate(x,list1,y,list1,&vecscat));
    CHKERRQ(VecScatterBegin(vecscat,x,y,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(vecscat,x,y,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterDestroy(&vecscat));
    CHKERRQ(ISDestroy(&list1));
#if !PetscDefined(HAVE_MPIUNI)
  } else {
    Mat_FFTW   *fftw = (Mat_FFTW*)fft->data;
    PetscInt   ndim  = fft->ndim,*dim = fft->dim;
    ptrdiff_t  local_n0,local_0_start;
    ptrdiff_t  local_n1,local_1_start;
    IS         list2;
#if !defined(PETSC_USE_COMPLEX)
    PetscInt   i,j,k,partial_dim;
    PetscInt   *indx1, *indx2, tempindx, tempindx1;
    PetscInt   NM;
    ptrdiff_t  temp;
#endif

    switch (ndim) {
    case 1:
#if defined(PETSC_USE_COMPLEX)
      fftw_mpi_local_size_1d(dim[0],comm,FFTW_FORWARD,FFTW_ESTIMATE,&local_n0,&local_0_start,&local_n1,&local_1_start);

      CHKERRQ(ISCreateStride(comm,local_n0,local_0_start,1,&list1));
      CHKERRQ(ISCreateStride(comm,local_n0,low,1,&list2));
      CHKERRQ(VecScatterCreate(x,list1,y,list2,&vecscat));
      CHKERRQ(VecScatterBegin(vecscat,x,y,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecScatterEnd(vecscat,x,y,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecScatterDestroy(&vecscat));
      CHKERRQ(ISDestroy(&list1));
      CHKERRQ(ISDestroy(&list2));
#else
      SETERRQ(comm,PETSC_ERR_SUP,"FFTW does not support parallel 1D real transform");
#endif
      break;
    case 2:
#if defined(PETSC_USE_COMPLEX)
      fftw_mpi_local_size_2d(dim[0],dim[1],comm,&local_n0,&local_0_start);

      CHKERRQ(ISCreateStride(comm,local_n0*dim[1],local_0_start*dim[1],1,&list1));
      CHKERRQ(ISCreateStride(comm,local_n0*dim[1],low,1,&list2));
      CHKERRQ(VecScatterCreate(x,list1,y,list2,&vecscat));
      CHKERRQ(VecScatterBegin(vecscat,x,y,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecScatterEnd(vecscat,x,y,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecScatterDestroy(&vecscat));
      CHKERRQ(ISDestroy(&list1));
      CHKERRQ(ISDestroy(&list2));
#else
      fftw_mpi_local_size_2d_transposed(dim[0],dim[1]/2+1,comm,&local_n0,&local_0_start,&local_n1,&local_1_start);

      CHKERRQ(PetscMalloc1(((PetscInt)local_n0)*dim[1],&indx1));
      CHKERRQ(PetscMalloc1(((PetscInt)local_n0)*dim[1],&indx2));

      if (dim[1]%2==0) {
        NM = dim[1]+2;
      } else {
        NM = dim[1]+1;
      }
      for (i=0; i<local_n0; i++) {
        for (j=0; j<dim[1]; j++) {
          tempindx  = i*dim[1] + j;
          tempindx1 = i*NM + j;

          indx1[tempindx]=local_0_start*dim[1]+tempindx;
          indx2[tempindx]=low+tempindx1;
        }
      }

      CHKERRQ(ISCreateGeneral(comm,local_n0*dim[1],indx1,PETSC_COPY_VALUES,&list1));
      CHKERRQ(ISCreateGeneral(comm,local_n0*dim[1],indx2,PETSC_COPY_VALUES,&list2));

      CHKERRQ(VecScatterCreate(x,list1,y,list2,&vecscat));
      CHKERRQ(VecScatterBegin(vecscat,x,y,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecScatterEnd(vecscat,x,y,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecScatterDestroy(&vecscat));
      CHKERRQ(ISDestroy(&list1));
      CHKERRQ(ISDestroy(&list2));
      CHKERRQ(PetscFree(indx1));
      CHKERRQ(PetscFree(indx2));
#endif
      break;

    case 3:
#if defined(PETSC_USE_COMPLEX)
      fftw_mpi_local_size_3d(dim[0],dim[1],dim[2],comm,&local_n0,&local_0_start);

      CHKERRQ(ISCreateStride(comm,local_n0*dim[1]*dim[2],local_0_start*dim[1]*dim[2],1,&list1));
      CHKERRQ(ISCreateStride(comm,local_n0*dim[1]*dim[2],low,1,&list2));
      CHKERRQ(VecScatterCreate(x,list1,y,list2,&vecscat));
      CHKERRQ(VecScatterBegin(vecscat,x,y,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecScatterEnd(vecscat,x,y,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecScatterDestroy(&vecscat));
      CHKERRQ(ISDestroy(&list1));
      CHKERRQ(ISDestroy(&list2));
#else
      /* buggy, needs to be fixed. See src/mat/tests/ex158.c */
      SETERRQ(comm,PETSC_ERR_SUP,"FFTW does not support parallel 3D real transform");
      fftw_mpi_local_size_3d_transposed(dim[0],dim[1],dim[2]/2+1,comm,&local_n0,&local_0_start,&local_n1,&local_1_start);

      CHKERRQ(PetscMalloc1(((PetscInt)local_n0)*dim[1]*dim[2],&indx1));
      CHKERRQ(PetscMalloc1(((PetscInt)local_n0)*dim[1]*dim[2],&indx2));

      if (dim[2]%2==0) NM = dim[2]+2;
      else             NM = dim[2]+1;

      for (i=0; i<local_n0; i++) {
        for (j=0; j<dim[1]; j++) {
          for (k=0; k<dim[2]; k++) {
            tempindx  = i*dim[1]*dim[2] + j*dim[2] + k;
            tempindx1 = i*dim[1]*NM + j*NM + k;

            indx1[tempindx]=local_0_start*dim[1]*dim[2]+tempindx;
            indx2[tempindx]=low+tempindx1;
          }
        }
      }

      CHKERRQ(ISCreateGeneral(comm,local_n0*dim[1]*dim[2],indx1,PETSC_COPY_VALUES,&list1));
      CHKERRQ(ISCreateGeneral(comm,local_n0*dim[1]*dim[2],indx2,PETSC_COPY_VALUES,&list2));
      CHKERRQ(VecScatterCreate(x,list1,y,list2,&vecscat));
      CHKERRQ(VecScatterBegin(vecscat,x,y,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecScatterEnd(vecscat,x,y,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecScatterDestroy(&vecscat));
      CHKERRQ(ISDestroy(&list1));
      CHKERRQ(ISDestroy(&list2));
      CHKERRQ(PetscFree(indx1));
      CHKERRQ(PetscFree(indx2));
#endif
      break;

    default:
#if defined(PETSC_USE_COMPLEX)
      fftw_mpi_local_size(fftw->ndim_fftw,fftw->dim_fftw,comm,&local_n0,&local_0_start);

      CHKERRQ(ISCreateStride(comm,local_n0*(fftw->partial_dim),local_0_start*(fftw->partial_dim),1,&list1));
      CHKERRQ(ISCreateStride(comm,local_n0*(fftw->partial_dim),low,1,&list2));
      CHKERRQ(VecScatterCreate(x,list1,y,list2,&vecscat));
      CHKERRQ(VecScatterBegin(vecscat,x,y,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecScatterEnd(vecscat,x,y,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecScatterDestroy(&vecscat));
      CHKERRQ(ISDestroy(&list1));
      CHKERRQ(ISDestroy(&list2));
#else
      /* buggy, needs to be fixed. See src/mat/tests/ex158.c */
      SETERRQ(comm,PETSC_ERR_SUP,"FFTW does not support parallel DIM>3 real transform");
      temp = (fftw->dim_fftw)[fftw->ndim_fftw-1];

      (fftw->dim_fftw)[fftw->ndim_fftw-1] = temp/2 + 1;

      fftw_mpi_local_size_transposed(fftw->ndim_fftw,fftw->dim_fftw,comm,&local_n0,&local_0_start,&local_n1,&local_1_start);

      (fftw->dim_fftw)[fftw->ndim_fftw-1] = temp;

      partial_dim = fftw->partial_dim;

      CHKERRQ(PetscMalloc1(((PetscInt)local_n0)*partial_dim,&indx1));
      CHKERRQ(PetscMalloc1(((PetscInt)local_n0)*partial_dim,&indx2));

      if (dim[ndim-1]%2==0) NM = 2;
      else                  NM = 1;

      j = low;
      for (i=0,k=1; i<((PetscInt)local_n0)*partial_dim;i++,k++) {
        indx1[i] = local_0_start*partial_dim + i;
        indx2[i] = j;
        if (k%dim[ndim-1]==0) j+=NM;
        j++;
      }
      CHKERRQ(ISCreateGeneral(comm,local_n0*partial_dim,indx1,PETSC_COPY_VALUES,&list1));
      CHKERRQ(ISCreateGeneral(comm,local_n0*partial_dim,indx2,PETSC_COPY_VALUES,&list2));
      CHKERRQ(VecScatterCreate(x,list1,y,list2,&vecscat));
      CHKERRQ(VecScatterBegin(vecscat,x,y,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecScatterEnd(vecscat,x,y,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecScatterDestroy(&vecscat));
      CHKERRQ(ISDestroy(&list1));
      CHKERRQ(ISDestroy(&list2));
      CHKERRQ(PetscFree(indx1));
      CHKERRQ(PetscFree(indx2));
#endif
      break;
    }
#endif
  }
  PetscFunctionReturn(0);
}

/*@
   VecScatterFFTWToPetsc - Converts FFTW output to the PETSc vector.

   Collective on Mat

    Input Parameters:
+   A - FFTW matrix
-   x - FFTW vector

   Output Parameters:
.  y - PETSc vector

   Level: intermediate

   Note: While doing real transform the FFTW output of backward DFT contains extra zeros at the end of last dimension.
         VecScatterFFTWToPetsc removes those extra zeros.

.seealso: VecScatterPetscToFFTW()
@*/
PetscErrorCode VecScatterFFTWToPetsc(Mat A,Vec x,Vec y)
{
  PetscFunctionBegin;
  CHKERRQ(PetscUseMethod(A,"VecScatterFFTWToPetsc_C",(Mat,Vec,Vec),(A,x,y)));
  PetscFunctionReturn(0);
}

PetscErrorCode VecScatterFFTWToPetsc_FFTW(Mat A,Vec x,Vec y)
{
  MPI_Comm       comm;
  Mat_FFT        *fft  = (Mat_FFT*)A->data;
  PetscInt       low;
  PetscMPIInt    rank,size;
  VecScatter     vecscat;
  IS             list1;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)A,&comm));
  CHKERRMPI(MPI_Comm_size(comm, &size));
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  CHKERRQ(VecGetOwnershipRange(x,&low,NULL));

  if (size==1) {
    CHKERRQ(ISCreateStride(comm,fft->N,0,1,&list1));
    CHKERRQ(VecScatterCreate(x,list1,y,list1,&vecscat));
    CHKERRQ(VecScatterBegin(vecscat,x,y,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(vecscat,x,y,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterDestroy(&vecscat));
    CHKERRQ(ISDestroy(&list1));

#if !PetscDefined(HAVE_MPIUNI)
  } else {
    Mat_FFTW       *fftw = (Mat_FFTW*)fft->data;
    PetscInt       ndim  = fft->ndim,*dim = fft->dim;
    ptrdiff_t      local_n0,local_0_start;
    ptrdiff_t      local_n1,local_1_start;
    IS             list2;
#if !defined(PETSC_USE_COMPLEX)
    PetscInt       i,j,k,partial_dim;
    PetscInt       *indx1, *indx2, tempindx, tempindx1;
    PetscInt       NM;
    ptrdiff_t      temp;
#endif
    switch (ndim) {
    case 1:
#if defined(PETSC_USE_COMPLEX)
      fftw_mpi_local_size_1d(dim[0],comm,FFTW_BACKWARD,FFTW_ESTIMATE,&local_n0,&local_0_start,&local_n1,&local_1_start);

      CHKERRQ(ISCreateStride(comm,local_n1,local_1_start,1,&list1));
      CHKERRQ(ISCreateStride(comm,local_n1,low,1,&list2));
      CHKERRQ(VecScatterCreate(x,list1,y,list2,&vecscat));
      CHKERRQ(VecScatterBegin(vecscat,x,y,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecScatterEnd(vecscat,x,y,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecScatterDestroy(&vecscat));
      CHKERRQ(ISDestroy(&list1));
      CHKERRQ(ISDestroy(&list2));
#else
      SETERRQ(comm,PETSC_ERR_SUP,"No support for real parallel 1D FFT");
#endif
      break;
    case 2:
#if defined(PETSC_USE_COMPLEX)
      fftw_mpi_local_size_2d(dim[0],dim[1],comm,&local_n0,&local_0_start);

      CHKERRQ(ISCreateStride(comm,local_n0*dim[1],local_0_start*dim[1],1,&list1));
      CHKERRQ(ISCreateStride(comm,local_n0*dim[1],low,1,&list2));
      CHKERRQ(VecScatterCreate(x,list2,y,list1,&vecscat));
      CHKERRQ(VecScatterBegin(vecscat,x,y,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecScatterEnd(vecscat,x,y,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecScatterDestroy(&vecscat));
      CHKERRQ(ISDestroy(&list1));
      CHKERRQ(ISDestroy(&list2));
#else
      fftw_mpi_local_size_2d_transposed(dim[0],dim[1]/2+1,comm,&local_n0,&local_0_start,&local_n1,&local_1_start);

      CHKERRQ(PetscMalloc1(((PetscInt)local_n0)*dim[1],&indx1));
      CHKERRQ(PetscMalloc1(((PetscInt)local_n0)*dim[1],&indx2));

      if (dim[1]%2==0) NM = dim[1]+2;
      else             NM = dim[1]+1;

      for (i=0; i<local_n0; i++) {
        for (j=0; j<dim[1]; j++) {
          tempindx = i*dim[1] + j;
          tempindx1 = i*NM + j;

          indx1[tempindx]=local_0_start*dim[1]+tempindx;
          indx2[tempindx]=low+tempindx1;
        }
      }

      CHKERRQ(ISCreateGeneral(comm,local_n0*dim[1],indx1,PETSC_COPY_VALUES,&list1));
      CHKERRQ(ISCreateGeneral(comm,local_n0*dim[1],indx2,PETSC_COPY_VALUES,&list2));

      CHKERRQ(VecScatterCreate(x,list2,y,list1,&vecscat));
      CHKERRQ(VecScatterBegin(vecscat,x,y,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecScatterEnd(vecscat,x,y,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecScatterDestroy(&vecscat));
      CHKERRQ(ISDestroy(&list1));
      CHKERRQ(ISDestroy(&list2));
      CHKERRQ(PetscFree(indx1));
      CHKERRQ(PetscFree(indx2));
#endif
      break;
    case 3:
#if defined(PETSC_USE_COMPLEX)
      fftw_mpi_local_size_3d(dim[0],dim[1],dim[2],comm,&local_n0,&local_0_start);

      CHKERRQ(ISCreateStride(comm,local_n0*dim[1]*dim[2],local_0_start*dim[1]*dim[2],1,&list1));
      CHKERRQ(ISCreateStride(comm,local_n0*dim[1]*dim[2],low,1,&list2));
      CHKERRQ(VecScatterCreate(x,list1,y,list2,&vecscat));
      CHKERRQ(VecScatterBegin(vecscat,x,y,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecScatterEnd(vecscat,x,y,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecScatterDestroy(&vecscat));
      CHKERRQ(ISDestroy(&list1));
      CHKERRQ(ISDestroy(&list2));
#else
      fftw_mpi_local_size_3d_transposed(dim[0],dim[1],dim[2]/2+1,comm,&local_n0,&local_0_start,&local_n1,&local_1_start);

      CHKERRQ(PetscMalloc1(((PetscInt)local_n0)*dim[1]*dim[2],&indx1));
      CHKERRQ(PetscMalloc1(((PetscInt)local_n0)*dim[1]*dim[2],&indx2));

      if (dim[2]%2==0) NM = dim[2]+2;
      else             NM = dim[2]+1;

      for (i=0; i<local_n0; i++) {
        for (j=0; j<dim[1]; j++) {
          for (k=0; k<dim[2]; k++) {
            tempindx  = i*dim[1]*dim[2] + j*dim[2] + k;
            tempindx1 = i*dim[1]*NM + j*NM + k;

            indx1[tempindx]=local_0_start*dim[1]*dim[2]+tempindx;
            indx2[tempindx]=low+tempindx1;
          }
        }
      }

      CHKERRQ(ISCreateGeneral(comm,local_n0*dim[1]*dim[2],indx1,PETSC_COPY_VALUES,&list1));
      CHKERRQ(ISCreateGeneral(comm,local_n0*dim[1]*dim[2],indx2,PETSC_COPY_VALUES,&list2));

      CHKERRQ(VecScatterCreate(x,list2,y,list1,&vecscat));
      CHKERRQ(VecScatterBegin(vecscat,x,y,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecScatterEnd(vecscat,x,y,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecScatterDestroy(&vecscat));
      CHKERRQ(ISDestroy(&list1));
      CHKERRQ(ISDestroy(&list2));
      CHKERRQ(PetscFree(indx1));
      CHKERRQ(PetscFree(indx2));
#endif
      break;
    default:
#if defined(PETSC_USE_COMPLEX)
      fftw_mpi_local_size(fftw->ndim_fftw,fftw->dim_fftw,comm,&local_n0,&local_0_start);

      CHKERRQ(ISCreateStride(comm,local_n0*(fftw->partial_dim),local_0_start*(fftw->partial_dim),1,&list1));
      CHKERRQ(ISCreateStride(comm,local_n0*(fftw->partial_dim),low,1,&list2));
      CHKERRQ(VecScatterCreate(x,list1,y,list2,&vecscat));
      CHKERRQ(VecScatterBegin(vecscat,x,y,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecScatterEnd(vecscat,x,y,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecScatterDestroy(&vecscat));
      CHKERRQ(ISDestroy(&list1));
      CHKERRQ(ISDestroy(&list2));
#else
      temp = (fftw->dim_fftw)[fftw->ndim_fftw-1];

      (fftw->dim_fftw)[fftw->ndim_fftw-1] = temp/2 + 1;

      fftw_mpi_local_size_transposed(fftw->ndim_fftw,fftw->dim_fftw,comm,&local_n0,&local_0_start,&local_n1,&local_1_start);

      (fftw->dim_fftw)[fftw->ndim_fftw-1] = temp;

      partial_dim = fftw->partial_dim;

      CHKERRQ(PetscMalloc1(((PetscInt)local_n0)*partial_dim,&indx1));
      CHKERRQ(PetscMalloc1(((PetscInt)local_n0)*partial_dim,&indx2));

      if (dim[ndim-1]%2==0) NM = 2;
      else                  NM = 1;

      j = low;
      for (i=0,k=1; i<((PetscInt)local_n0)*partial_dim; i++,k++) {
        indx1[i] = local_0_start*partial_dim + i;
        indx2[i] = j;
        if (k%dim[ndim-1]==0) j+=NM;
        j++;
      }
      CHKERRQ(ISCreateGeneral(comm,local_n0*partial_dim,indx1,PETSC_COPY_VALUES,&list1));
      CHKERRQ(ISCreateGeneral(comm,local_n0*partial_dim,indx2,PETSC_COPY_VALUES,&list2));

      CHKERRQ(VecScatterCreate(x,list2,y,list1,&vecscat));
      CHKERRQ(VecScatterBegin(vecscat,x,y,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecScatterEnd(vecscat,x,y,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecScatterDestroy(&vecscat));
      CHKERRQ(ISDestroy(&list1));
      CHKERRQ(ISDestroy(&list2));
      CHKERRQ(PetscFree(indx1));
      CHKERRQ(PetscFree(indx2));
#endif
      break;
    }
#endif
  }
  PetscFunctionReturn(0);
}

/*
    MatCreate_FFTW - Creates a matrix object that provides FFT via the external package FFTW

  Options Database Keys:
+ -mat_fftw_plannerflags - set FFTW planner flags

   Level: intermediate

*/
PETSC_EXTERN PetscErrorCode MatCreate_FFTW(Mat A)
{
  MPI_Comm       comm;
  Mat_FFT        *fft = (Mat_FFT*)A->data;
  Mat_FFTW       *fftw;
  PetscInt       ndim = fft->ndim,*dim = fft->dim;
  const char     *plans[]={"FFTW_ESTIMATE","FFTW_MEASURE","FFTW_PATIENT","FFTW_EXHAUSTIVE"};
  unsigned       iplans[]={FFTW_ESTIMATE,FFTW_MEASURE,FFTW_PATIENT,FFTW_EXHAUSTIVE};
  PetscBool      flg;
  PetscInt       p_flag,partial_dim=1,ctr;
  PetscMPIInt    size,rank;
  ptrdiff_t      *pdim;
  PetscErrorCode ierr;
#if !defined(PETSC_USE_COMPLEX)
  PetscInt       tot_dim;
#endif

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)A,&comm));
  CHKERRMPI(MPI_Comm_size(comm, &size));
  CHKERRMPI(MPI_Comm_rank(comm, &rank));

#if !PetscDefined(HAVE_MPIUNI)
  fftw_mpi_init();
#endif
  pdim    = (ptrdiff_t*)calloc(ndim,sizeof(ptrdiff_t));
  pdim[0] = dim[0];
#if !defined(PETSC_USE_COMPLEX)
  tot_dim = 2*dim[0];
#endif
  for (ctr=1; ctr<ndim; ctr++) {
    partial_dim *= dim[ctr];
    pdim[ctr]    = dim[ctr];
#if !defined(PETSC_USE_COMPLEX)
    if (ctr==ndim-1) tot_dim *= (dim[ctr]/2+1);
    else             tot_dim *= dim[ctr];
#endif
  }

  if (size == 1) {
#if defined(PETSC_USE_COMPLEX)
    CHKERRQ(MatSetSizes(A,fft->N,fft->N,fft->N,fft->N));
    fft->n = fft->N;
#else
    CHKERRQ(MatSetSizes(A,tot_dim,tot_dim,tot_dim,tot_dim));
    fft->n = tot_dim;
#endif
#if !PetscDefined(HAVE_MPIUNI)
  } else {
    ptrdiff_t local_n0,local_0_start,local_n1,local_1_start;
#if !defined(PETSC_USE_COMPLEX)
    ptrdiff_t temp;
    PetscInt  N1;
#endif

    switch (ndim) {
    case 1:
#if !defined(PETSC_USE_COMPLEX)
      SETERRQ(comm,PETSC_ERR_SUP,"FFTW does not support parallel 1D real transform");
#else
      fftw_mpi_local_size_1d(dim[0],comm,FFTW_FORWARD,FFTW_ESTIMATE,&local_n0,&local_0_start,&local_n1,&local_1_start);
      fft->n = (PetscInt)local_n0;
      CHKERRQ(MatSetSizes(A,local_n1,fft->n,fft->N,fft->N));
#endif
      break;
    case 2:
#if defined(PETSC_USE_COMPLEX)
      fftw_mpi_local_size_2d(dim[0],dim[1],comm,&local_n0,&local_0_start);
      fft->n    = (PetscInt)local_n0*dim[1];
      CHKERRQ(MatSetSizes(A,fft->n,fft->n,fft->N,fft->N));
#else
      fftw_mpi_local_size_2d_transposed(dim[0],dim[1]/2+1,comm,&local_n0,&local_0_start,&local_n1,&local_1_start);

      fft->n = 2*(PetscInt)local_n0*(dim[1]/2+1);
      CHKERRQ(MatSetSizes(A,fft->n,fft->n,2*dim[0]*(dim[1]/2+1),2*dim[0]*(dim[1]/2+1)));
#endif
      break;
    case 3:
#if defined(PETSC_USE_COMPLEX)
      fftw_mpi_local_size_3d(dim[0],dim[1],dim[2],comm,&local_n0,&local_0_start);

      fft->n = (PetscInt)local_n0*dim[1]*dim[2];
      CHKERRQ(MatSetSizes(A,fft->n,fft->n,fft->N,fft->N));
#else
      fftw_mpi_local_size_3d_transposed(dim[0],dim[1],dim[2]/2+1,comm,&local_n0,&local_0_start,&local_n1,&local_1_start);

      fft->n = 2*(PetscInt)local_n0*dim[1]*(dim[2]/2+1);
      CHKERRQ(MatSetSizes(A,fft->n,fft->n,2*dim[0]*dim[1]*(dim[2]/2+1),2*dim[0]*dim[1]*(dim[2]/2+1)));
#endif
      break;
    default:
#if defined(PETSC_USE_COMPLEX)
      fftw_mpi_local_size(ndim,pdim,comm,&local_n0,&local_0_start);

      fft->n = (PetscInt)local_n0*partial_dim;
      CHKERRQ(MatSetSizes(A,fft->n,fft->n,fft->N,fft->N));
#else
      temp = pdim[ndim-1];

      pdim[ndim-1] = temp/2 + 1;

      fftw_mpi_local_size_transposed(ndim,pdim,comm,&local_n0,&local_0_start,&local_n1,&local_1_start);

      fft->n  = 2*(PetscInt)local_n0*partial_dim*pdim[ndim-1]/temp;
      N1 = 2*fft->N*(PetscInt)pdim[ndim-1]/((PetscInt) temp);

      pdim[ndim-1] = temp;

      CHKERRQ(MatSetSizes(A,fft->n,fft->n,N1,N1));
#endif
      break;
    }
#endif
  }
  free(pdim);
  CHKERRQ(PetscObjectChangeTypeName((PetscObject)A,MATFFTW));
  CHKERRQ(PetscNewLog(A,&fftw));
  fft->data = (void*)fftw;

  fftw->ndim_fftw   = (ptrdiff_t)ndim; /* This is dimension of fft */
  fftw->partial_dim = partial_dim;

  CHKERRQ(PetscMalloc1(ndim, &(fftw->dim_fftw)));
  if (size == 1) {
#if defined(PETSC_USE_64BIT_INDICES)
    fftw->iodims = (fftw_iodim64 *) malloc(sizeof(fftw_iodim64) * ndim);
#else
    fftw->iodims = (fftw_iodim *) malloc(sizeof(fftw_iodim) * ndim);
#endif
  }

  for (ctr=0;ctr<ndim;ctr++) (fftw->dim_fftw)[ctr]=dim[ctr];

  fftw->p_forward  = NULL;
  fftw->p_backward = NULL;
  fftw->p_flag     = FFTW_ESTIMATE;

  if (size == 1) {
    A->ops->mult          = MatMult_SeqFFTW;
    A->ops->multtranspose = MatMultTranspose_SeqFFTW;
#if !PetscDefined(HAVE_MPIUNI)
  } else {
    A->ops->mult          = MatMult_MPIFFTW;
    A->ops->multtranspose = MatMultTranspose_MPIFFTW;
#endif
  }
  fft->matdestroy = MatDestroy_FFTW;
  A->assembled    = PETSC_TRUE;
  A->preallocated = PETSC_TRUE;

  CHKERRQ(PetscObjectComposeFunction((PetscObject)A,"MatCreateVecsFFTW_C",MatCreateVecsFFTW_FFTW));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)A,"VecScatterPetscToFFTW_C",VecScatterPetscToFFTW_FFTW));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)A,"VecScatterFFTWToPetsc_C",VecScatterFFTWToPetsc_FFTW));

  /* get runtime options */
  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)A),((PetscObject)A)->prefix,"FFTW Options","Mat");CHKERRQ(ierr);
  CHKERRQ(PetscOptionsEList("-mat_fftw_plannerflags","Planner Flags","None",plans,4,plans[0],&p_flag,&flg));
  if (flg) fftw->p_flag = iplans[p_flag];
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
