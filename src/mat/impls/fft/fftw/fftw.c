
/*
    Provides an interface to the FFTW package.
    Testing examples can be found in ~src/mat/examples/tests
*/

#include "../src/mat/impls/fft/fft.h"   /*I "petscmat.h" I*/
EXTERN_C_BEGIN 
#include "fftw3-mpi.h"
EXTERN_C_END 

typedef struct {
  fftw_plan   p_forward,p_backward;
  unsigned    p_flag; /* planner flags, FFTW_ESTIMATE,FFTW_MEASURE, FFTW_PATIENT, FFTW_EXHAUSTIVE */
  PetscScalar *finarray,*foutarray,*binarray,*boutarray; /* keep track of arrays becaue fftw plan should be 
                                                            executed for the arrays with which the plan was created */
} Mat_FFTW;

extern PetscErrorCode MatMult_SeqFFTW(Mat,Vec,Vec);
extern PetscErrorCode MatMultTranspose_SeqFFTW(Mat,Vec,Vec);
extern PetscErrorCode MatMult_MPIFFTW(Mat,Vec,Vec);
extern PetscErrorCode MatMultTranspose_MPIFFTW(Mat,Vec,Vec);
extern PetscErrorCode MatDestroy_FFTW(Mat);
extern PetscErrorCode VecDestroy_MPIFFTW(Vec);
extern PetscErrorCode MatGetVecs_FFTW(Mat,Vec*,Vec*);

#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqFFTW"
PetscErrorCode MatMult_SeqFFTW(Mat A,Vec x,Vec y)
{
  PetscErrorCode ierr;
  Mat_FFT        *fft  = (Mat_FFT*)A->data;
  Mat_FFTW       *fftw = (Mat_FFTW*)fft->data;
  PetscScalar    *x_array,*y_array;
  PetscInt       ndim=fft->ndim,*dim=fft->dim;

  PetscFunctionBegin;
#if !defined(PETSC_USE_COMPLEX)
  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"not support for real numbers");
#endif
  ierr = VecGetArray(x,&x_array);CHKERRQ(ierr);
  ierr = VecGetArray(y,&y_array);CHKERRQ(ierr);
  if (!fftw->p_forward){ /* create a plan, then excute it */
    switch (ndim){
    case 1:
      fftw->p_forward = fftw_plan_dft_1d(dim[0],(fftw_complex*)x_array,(fftw_complex*)y_array,FFTW_FORWARD,fftw->p_flag);   
      break;
    case 2:
      fftw->p_forward = fftw_plan_dft_2d(dim[0],dim[1],(fftw_complex*)x_array,(fftw_complex*)y_array,FFTW_FORWARD,fftw->p_flag);
      break;
    case 3:
      fftw->p_forward = fftw_plan_dft_3d(dim[0],dim[1],dim[2],(fftw_complex*)x_array,(fftw_complex*)y_array,FFTW_FORWARD,fftw->p_flag);
      break;
    default:
      fftw->p_forward = fftw_plan_dft(ndim,dim,(fftw_complex*)x_array,(fftw_complex*)y_array,FFTW_FORWARD,fftw->p_flag);
      break;
    }
    fftw->finarray  = x_array;
    fftw->foutarray = y_array;
    /* Warning: if (fftw->p_flag!==FFTW_ESTIMATE) The data in the in/out arrays is overwritten! 
                planning should be done before x is initialized! See FFTW manual sec2.1 or sec4 */
    fftw_execute(fftw->p_forward);
  } else { /* use existing plan */
    if (fftw->finarray != x_array || fftw->foutarray != y_array){ /* use existing plan on new arrays */
      fftw_execute_dft(fftw->p_forward,(fftw_complex*)x_array,(fftw_complex*)y_array);
    } else {
      fftw_execute(fftw->p_forward);
    }
  }
  ierr = VecRestoreArray(y,&y_array);CHKERRQ(ierr);
  ierr = VecRestoreArray(x,&x_array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultTranspose_SeqFFTW"
PetscErrorCode MatMultTranspose_SeqFFTW(Mat A,Vec x,Vec y)
{
  PetscErrorCode ierr;
  Mat_FFT        *fft = (Mat_FFT*)A->data;
  Mat_FFTW       *fftw = (Mat_FFTW*)fft->data;
  PetscScalar    *x_array,*y_array;
  PetscInt       ndim=fft->ndim,*dim=fft->dim;

  PetscFunctionBegin;
#if !defined(PETSC_USE_COMPLEX)
  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"not support for real numbers");
#endif
  ierr = VecGetArray(x,&x_array);CHKERRQ(ierr);
  ierr = VecGetArray(y,&y_array);CHKERRQ(ierr);
  if (!fftw->p_backward){ /* create a plan, then excute it */
    switch (ndim){
    case 1:
      fftw->p_backward = fftw_plan_dft_1d(dim[0],(fftw_complex*)x_array,(fftw_complex*)y_array,FFTW_BACKWARD,fftw->p_flag);
      break;
    case 2:
      fftw->p_backward = fftw_plan_dft_2d(dim[0],dim[1],(fftw_complex*)x_array,(fftw_complex*)y_array,FFTW_BACKWARD,fftw->p_flag);
      break;
    case 3:
      fftw->p_backward = fftw_plan_dft_3d(dim[0],dim[1],dim[2],(fftw_complex*)x_array,(fftw_complex*)y_array,FFTW_BACKWARD,fftw->p_flag);
      break;
    default:
      fftw->p_backward = fftw_plan_dft(ndim,dim,(fftw_complex*)x_array,(fftw_complex*)y_array,FFTW_BACKWARD,fftw->p_flag);
      break;
    }
    fftw->binarray  = x_array;
    fftw->boutarray = y_array;
    fftw_execute(fftw->p_backward);CHKERRQ(ierr);
  } else { /* use existing plan */
    if (fftw->binarray != x_array || fftw->boutarray != y_array){ /* use existing plan on new arrays */
      fftw_execute_dft(fftw->p_backward,(fftw_complex*)x_array,(fftw_complex*)y_array);
    } else {
      fftw_execute(fftw->p_backward);CHKERRQ(ierr);
    }
  }
  ierr = VecRestoreArray(y,&y_array);CHKERRQ(ierr);
  ierr = VecRestoreArray(x,&x_array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMult_MPIFFTW"
PetscErrorCode MatMult_MPIFFTW(Mat A,Vec x,Vec y)
{
  PetscErrorCode ierr;
  Mat_FFT        *fft  = (Mat_FFT*)A->data;
  Mat_FFTW       *fftw = (Mat_FFTW*)fft->data;
  PetscScalar    *x_array,*y_array;
  PetscInt       ndim=fft->ndim,*dim=fft->dim;
  MPI_Comm       comm=((PetscObject)A)->comm;

  PetscFunctionBegin;
#if !defined(PETSC_USE_COMPLEX)
  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"not support for real numbers");
#endif
  ierr = VecGetArray(x,&x_array);CHKERRQ(ierr);
  ierr = VecGetArray(y,&y_array);CHKERRQ(ierr);
  if (!fftw->p_forward){ /* create a plan, then excute it */
    switch (ndim){
    case 1:
      fftw->p_forward = fftw_mpi_plan_dft_1d(dim[0],(fftw_complex*)x_array,(fftw_complex*)y_array,comm,FFTW_FORWARD,fftw->p_flag);   
      break;
    case 2:
      fftw->p_forward = fftw_mpi_plan_dft_2d(dim[0],dim[1],(fftw_complex*)x_array,(fftw_complex*)y_array,comm,FFTW_FORWARD,fftw->p_flag);
      break;
    case 3:
      fftw->p_forward = fftw_mpi_plan_dft_3d(dim[0],dim[1],dim[2],(fftw_complex*)x_array,(fftw_complex*)y_array,comm,FFTW_FORWARD,fftw->p_flag);
      break;
    default:
      /*
       fftw->p_forward = fftw_mpi_plan_dft(ndim,dim,(fftw_complex*)x_array,(fftw_complex*)y_array,comm,FFTW_FORWARD,fftw->p_flag);
       */
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Not supported yet");
      break;
    }
    fftw->finarray  = x_array;
    fftw->foutarray = y_array;
    /* Warning: if (fftw->p_flag!==FFTW_ESTIMATE) The data in the in/out arrays is overwritten! 
                planning should be done before x is initialized! See FFTW manual sec2.1 or sec4 */
    fftw_execute(fftw->p_forward);
  } else { /* use existing plan */
    if (fftw->finarray != x_array || fftw->foutarray != y_array){ /* use existing plan on new arrays */
      fftw_execute_dft(fftw->p_forward,(fftw_complex*)x_array,(fftw_complex*)y_array);
    } else {
      fftw_execute(fftw->p_forward);
    }
  }
  ierr = VecRestoreArray(y,&y_array);CHKERRQ(ierr);
  ierr = VecRestoreArray(x,&x_array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultTranspose_MPIFFTW"
PetscErrorCode MatMultTranspose_MPIFFTW(Mat A,Vec x,Vec y)
{
  PetscErrorCode ierr;
  Mat_FFT        *fft  = (Mat_FFT*)A->data;
  Mat_FFTW       *fftw = (Mat_FFTW*)fft->data;
  PetscScalar    *x_array,*y_array;
  PetscInt       ndim=fft->ndim,*dim=fft->dim;
  MPI_Comm       comm=((PetscObject)A)->comm;

  PetscFunctionBegin;
#if !defined(PETSC_USE_COMPLEX)
  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"not support for real numbers");
#endif
  ierr = VecGetArray(x,&x_array);CHKERRQ(ierr);
  ierr = VecGetArray(y,&y_array);CHKERRQ(ierr);
  if (!fftw->p_backward){ /* create a plan, then excute it */
    switch (ndim){
    case 1:
      fftw->p_backward = fftw_mpi_plan_dft_1d(dim[0],(fftw_complex*)x_array,(fftw_complex*)y_array,comm,FFTW_BACKWARD,fftw->p_flag);
      break;
    case 2:
      fftw->p_backward = fftw_mpi_plan_dft_2d(dim[0],dim[1],(fftw_complex*)x_array,(fftw_complex*)y_array,comm,FFTW_BACKWARD,fftw->p_flag);
      break;
    case 3:
      fftw->p_backward = fftw_mpi_plan_dft_3d(dim[0],dim[1],dim[2],(fftw_complex*)x_array,(fftw_complex*)y_array,comm,FFTW_BACKWARD,fftw->p_flag);
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Not supported yet");
      /* fftw->p_backward = fftw_mpi_plan_dft(ndim,dim,(fftw_complex*)x_array,(fftw_complex*)y_array,FFTW_BACKWARD,fftw->p_flag); */
      break;
    }
    fftw->binarray  = x_array;
    fftw->boutarray = y_array;
    fftw_execute(fftw->p_backward);CHKERRQ(ierr);
  } else { /* use existing plan */
    if (fftw->binarray != x_array || fftw->boutarray != y_array){ /* use existing plan on new arrays */
      fftw_execute_dft(fftw->p_backward,(fftw_complex*)x_array,(fftw_complex*)y_array);
    } else {
      fftw_execute(fftw->p_backward);CHKERRQ(ierr);
    }
  }
  ierr = VecRestoreArray(y,&y_array);CHKERRQ(ierr);
  ierr = VecRestoreArray(x,&x_array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_FFTW"
PetscErrorCode MatDestroy_FFTW(Mat A)
{
  Mat_FFT        *fft = (Mat_FFT*)A->data;
  Mat_FFTW       *fftw = (Mat_FFTW*)fft->data;
  PetscErrorCode ierr;

  PetscFunctionBegin; 
#if !defined(PETSC_USE_COMPLEX)
  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"not support for real numbers");
#endif
  fftw_destroy_plan(fftw->p_forward); 
  fftw_destroy_plan(fftw->p_backward);
  ierr = PetscFree(fftw);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include "../src/vec/vec/impls/mpi/pvecimpl.h"   /*I  "petscvec.h"   I*/
#undef __FUNCT__
#define __FUNCT__ "VecDestroy_MPIFFTW"
PetscErrorCode VecDestroy_MPIFFTW(Vec v)
{
  PetscErrorCode  ierr;
  PetscScalar     *array;

  PetscFunctionBegin;
#if !defined(PETSC_USE_COMPLEX)
  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"not support for real numbers");
#endif
  ierr = VecGetArray(v,&array);CHKERRQ(ierr);
  fftw_free((fftw_complex*)array);CHKERRQ(ierr);
  ierr = VecRestoreArray(v,&array);CHKERRQ(ierr);
  ierr = VecDestroy_MPI(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetVecs_FFTW"
/*
   MatGetVecs_FFTW - Get vector(s) compatible with the matrix, i.e. with the
     parallel layout determined by FFTW

   Collective on Mat

   Input Parameter:
.  mat - the matrix

   Output Parameter:
+   fin - (optional) input vector of forward FFTW
-   fout - (optional) output vector of forward FFTW

  Level: advanced

.seealso: MatCreateFFTW()
*/
PetscErrorCode  MatGetVecs_FFTW(Mat A,Vec *fin,Vec *fout)
{
  PetscErrorCode ierr;
  PetscMPIInt    size,rank;
  MPI_Comm       comm=((PetscObject)A)->comm;
  Mat_FFT        *fft = (Mat_FFT*)A->data;
  PetscInt       N=fft->N;  

  PetscFunctionBegin;
#if !defined(PETSC_USE_COMPLEX)
  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"not support for real numbers");
#endif
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);

  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  if (size == 1){ /* sequential case */
    if (fin) {ierr = VecCreateSeq(PETSC_COMM_SELF,N,fin);CHKERRQ(ierr);}
    if (fout){ierr = VecCreateSeq(PETSC_COMM_SELF,N,fout);CHKERRQ(ierr);}
  } else {        /* mpi case */
    ptrdiff_t      alloc_local,local_n0,local_0_start;
    PetscInt       ndim=fft->ndim,*dim=fft->dim,n=fft->n;
    fftw_complex   *data_fin,*data_fout;

    switch (ndim){
    case 1:
      SETERRQ(((PetscObject)A)->comm,PETSC_ERR_SUP,"Not supported yet");
      break;
    case 2:
      /* Get local size */
      alloc_local = fftw_mpi_local_size_2d(dim[0],dim[1],comm,&local_n0,&local_0_start);
      if (fin) {
        data_fin  = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*alloc_local);
        ierr = VecCreateMPIWithArray(comm,n,N,(const PetscScalar*)data_fin,fin);CHKERRQ(ierr);
        (*fin)->ops->destroy   = VecDestroy_MPIFFTW;
      }
      if (fout) {
        data_fout = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*alloc_local);
        ierr = VecCreateMPIWithArray(comm,n,N,(const PetscScalar*)data_fout,fout);CHKERRQ(ierr);
        (*fout)->ops->destroy   = VecDestroy_MPIFFTW;
      }
      break;
    case 3:
      SETERRQ(((PetscObject)A)->comm,PETSC_ERR_SUP,"Not supported yet");
      break;
    default:
      SETERRQ(((PetscObject)A)->comm,PETSC_ERR_SUP,"Not supported yet");
      break;
    }
  } 
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatCreate_FFTW"
/*
      MatCreate_FFTW - Creates a matrix object that provides FFT
  via the external package FFTW

  Options Database Keys:
+ -mat_fftw_plannerflags - set FFTW planner flags

   Level: intermediate
   
*/
PetscErrorCode MatCreate_FFTW(Mat A)
{
  PetscErrorCode ierr;
  MPI_Comm       comm=((PetscObject)A)->comm;
  Mat_FFT        *fft=(Mat_FFT*)A->data;
  Mat_FFTW       *fftw;
  PetscInt       n=fft->n,N=fft->N,ndim=fft->ndim,*dim = fft->dim;
  const char     *p_flags[]={"FFTW_ESTIMATE","FFTW_MEASURE","FFTW_PATIENT","FFTW_EXHAUSTIVE"};
  PetscBool      flg;
  PetscInt       p_flag;
  PetscMPIInt    size;

  PetscFunctionBegin;
#if !defined(PETSC_USE_COMPLEX)
  SETERRQ(comm,PETSC_ERR_SUP,"not support for real numbers");
#endif
 
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  //ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);

  
  if (size == 1) {
    ierr = MatSetSizes(A,N,N,N,N);CHKERRQ(ierr);  
    n = N;
  } else {
    ptrdiff_t alloc_local,local_n0,local_0_start;
    switch (ndim){
    case 1:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"not implemented yet");
      break;
    case 2:
      alloc_local = fftw_mpi_local_size_2d(dim[0],dim[1],comm,&local_n0,&local_0_start);
      /*
       PetscMPIInt    rank;
       PetscSynchronizedPrintf(comm,"[%d] MatCreateSeqFFTW: local_n0, local_0_start %d %d, N %d,dim %d, %d\n",rank,(PetscInt)local_n0*dim[1],(PetscInt)local_0_start,m,dim[0],dim[1]);
       PetscSynchronizedFlush(comm);
       */
      n = (PetscInt)local_n0*dim[1];
      ierr = MatSetSizes(A,n,n,N,N);CHKERRQ(ierr);  
      break;
    case 3:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"not implemented yet");
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"not implemented yet");
      break;
    }
  }
  ierr = PetscObjectChangeTypeName((PetscObject)A,MATFFTW);CHKERRQ(ierr);

  ierr = PetscNewLog(A,Mat_FFTW,&fftw);CHKERRQ(ierr);
  fft->data = (void*)fftw;
  
  fft->n           = n;
  fftw->p_forward  = 0;
  fftw->p_backward = 0;
  fftw->p_flag     = FFTW_ESTIMATE;

  if (size == 1){
    A->ops->mult          = MatMult_SeqFFTW;
    A->ops->multtranspose = MatMultTranspose_SeqFFTW;
  } else {
    A->ops->mult          = MatMult_MPIFFTW;
    A->ops->multtranspose = MatMultTranspose_MPIFFTW;
  }
  fft->matdestroy          = MatDestroy_FFTW;
  A->ops->getvecs       = MatGetVecs_FFTW;
  A->assembled          = PETSC_TRUE;

  /* get runtime options */
  ierr = PetscOptionsBegin(((PetscObject)A)->comm,((PetscObject)A)->prefix,"FFTW Options","Mat");CHKERRQ(ierr);
    ierr = PetscOptionsEList("-mat_fftw_plannerflags","Planner Flags","None",p_flags,4,p_flags[0],&p_flag,&flg);CHKERRQ(ierr);
    if (flg) {fftw->p_flag = (unsigned)p_flag;}
  PetscOptionsEnd();
  PetscFunctionReturn(0);
}
EXTERN_C_END
