
/*
    Provides an interface to the FFTW package.
    Testing examples can be found in ~src/mat/examples/tests
*/

#include <../src/mat/impls/fft/fft.h>   /*I "petscmat.h" I*/
EXTERN_C_BEGIN 
#include <fftw3-mpi.h>
EXTERN_C_END 

typedef struct {
  ptrdiff_t   ndim_fftw,*dim_fftw;
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
extern PetscErrorCode InputTransformFFT_FFTW(Mat,Vec,Vec);
extern PetscErrorCode InputTransformFFT(Mat,Vec,Vec);

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
// PetscInt ctr;
//  ptrdiff_t      ndim1=(ptrdiff_t) ndim,*pdim;
//  ndim1=(ptrdiff_t) ndim;
//  pdim = (ptrdiff_t *)calloc(ndim,sizeof(ptrdiff_t));

//  for(ctr=0;ctr<ndim;ctr++)
//     {
//      pdim[ctr] = dim[ctr];
//     } 

  PetscFunctionBegin;
//#if !defined(PETSC_USE_COMPLEX)
//  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"not support for real numbers");
//#endif
//  pdim = (ptrdiff_t *)calloc(ndim,sizeof(ptrdiff_t));
//  for (ctr=0; ctr<ndim; ctr++) pdim[ctr] = dim[ctr];
    
  ierr = VecGetArray(x,&x_array);CHKERRQ(ierr);
  ierr = VecGetArray(y,&y_array);CHKERRQ(ierr);
  if (!fftw->p_forward){ /* create a plan, then excute it */
    switch (ndim){
    case 1:
#if defined(PETSC_USE_COMPLEX)
      fftw->p_forward = fftw_mpi_plan_dft_1d(dim[0],(fftw_complex*)x_array,(fftw_complex*)y_array,comm,FFTW_FORWARD,fftw->p_flag);   
#endif 
      break;
    case 2:
#if defined(PETSC_USE_COMPLEX)
      fftw->p_forward = fftw_mpi_plan_dft_2d(dim[0],dim[1],(fftw_complex*)x_array,(fftw_complex*)y_array,comm,FFTW_FORWARD,fftw->p_flag);
#else
      printf("The code comes here \n");
      fftw->p_forward = fftw_mpi_plan_dft_r2c_2d(dim[0],dim[1],(double *)x_array,(fftw_complex*)y_array,comm,FFTW_ESTIMATE);
#endif 
      break;
    case 3:
#if defined(PETSC_USE_COMPLEX)
      fftw->p_forward = fftw_mpi_plan_dft_3d(dim[0],dim[1],dim[2],(fftw_complex*)x_array,(fftw_complex*)y_array,comm,FFTW_FORWARD,fftw->p_flag);
#else
      fftw->p_forward = fftw_mpi_plan_dft_r2c_3d(dim[0],dim[1],dim[3],(double *)x_array,(fftw_complex*)y_array,comm,FFTW_ESTIMATE);
#endif 
      break;
    default:
#if defined(PETSC_USE_COMPLEX)
      fftw->p_forward = fftw_mpi_plan_dft(fftw->ndim_fftw,fftw->dim_fftw,(fftw_complex*)x_array,(fftw_complex*)y_array,comm,FFTW_FORWARD,fftw->p_flag);
#else
      fftw->p_forward = fftw_mpi_plan_dft_r2c(fftw->ndim_fftw,fftw->dim_fftw,(double *)x_array,(fftw_complex*)y_array,comm,FFTW_ESTIMATE);
#endif 
 //     fftw->p_forward = fftw_mpi_plan_dft(ndim,dim,(fftw_complex*)x_array,(fftw_complex*)y_array,comm,FFTW_FORWARD,fftw->p_flag);
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
//  PetscInt       ctr;
//  ptrdiff_t      ndim1=(ptrdiff_t)ndim,*pdim;
//  ndim1=(ptrdiff_t) ndim;
//  pdim = (ptrdiff_t *)calloc(ndim,sizeof(ptrdiff_t));

//  for(ctr=0;ctr<ndim;ctr++)
//     {
//      pdim[ctr] = dim[ctr];
//     } 
 
  PetscFunctionBegin;
//#if !defined(PETSC_USE_COMPLEX)
//  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"not support for real numbers");
//#endif
//  ierr = PetscMalloc(ndim*sizeof(ptrdiff_t), (ptrdiff_t *)&pdim);CHKERRQ(ierr); 
// should pdim be a member of Mat_FFTW?
//  for (ctr=0; ctr<ndim; ctr++) pdim[ctr] = dim[ctr];
    
  ierr = VecGetArray(x,&x_array);CHKERRQ(ierr);
  ierr = VecGetArray(y,&y_array);CHKERRQ(ierr);
  if (!fftw->p_backward){ /* create a plan, then excute it */
    switch (ndim){
    case 1:
#if defined(PETSC_USE_COMPLEX)
      fftw->p_backward = fftw_mpi_plan_dft_1d(dim[0],(fftw_complex*)x_array,(fftw_complex*)y_array,comm,FFTW_BACKWARD,fftw->p_flag);
#endif 
      break;
    case 2:
#if defined(PETSC_USE_COMPLEX)
      fftw->p_backward = fftw_mpi_plan_dft_2d(dim[0],dim[1],(fftw_complex*)x_array,(fftw_complex*)y_array,comm,FFTW_BACKWARD,fftw->p_flag);
#else
      fftw->p_backward = fftw_mpi_plan_dft_c2r_2d(dim[0],dim[1],(fftw_complex*)x_array,(double *)y_array,comm,FFTW_ESTIMATE);
#endif 
      break;
    case 3:
#if defined(PETSC_USE_COMPLEX)
      fftw->p_backward = fftw_mpi_plan_dft_3d(dim[0],dim[1],dim[2],(fftw_complex*)x_array,(fftw_complex*)y_array,comm,FFTW_BACKWARD,fftw->p_flag);
#else
      fftw->p_backward = fftw_mpi_plan_dft_c2r_3d(dim[0],dim[1],dim[2],(fftw_complex*)x_array,(double *)y_array,comm,FFTW_ESTIMATE);
#endif 
      break;
    default:
#if defined(PETSC_USE_COMPLEX)
      fftw->p_backward = fftw_mpi_plan_dft(fftw->ndim_fftw,fftw->dim_fftw,(fftw_complex*)x_array,(fftw_complex*)y_array,comm,FFTW_BACKWARD,fftw->p_flag);  
#else
      fftw->p_backward = fftw_mpi_plan_dft_c2r(fftw->ndim_fftw,fftw->dim_fftw,(fftw_complex*)x_array,(double *)y_array,comm,FFTW_ESTIMATE);
#endif 
//      fftw->p_backward = fftw_mpi_plan_dft(ndim,dim,(fftw_complex*)x_array,(fftw_complex*)y_array,comm,FFTW_BACKWARD,fftw->p_flag); 
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
  ierr = PetscFree(fftw->dim_fftw);CHKERRQ(ierr);
  ierr = PetscFree(fft->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include <../src/vec/vec/impls/mpi/pvecimpl.h>   /*I  "petscvec.h"   I*/
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
#define __FUNCT__ "MatGetVecs1DC_FFTW"
/* 
   MatGetVecs_FFTW1D - Get Vectors(s) compatible with matrix, i.e. with the 
     parallel layout determined by FFTW-1D 

*/
PetscErrorCode  MatGetVecs_FFTW1D(Mat A,Vec *fin,Vec *fout,Vec *bin,Vec *bout)
{
  PetscErrorCode ierr;
  PetscMPIInt    size,rank;
  MPI_Comm       comm=((PetscObject)A)->comm;
  Mat_FFT        *fft = (Mat_FFT*)A->data;
//  Mat_FFTW       *fftw = (Mat_FFTW*)fft->data;
  PetscInt       N=fft->N;  
  PetscInt       ndim=fft->ndim,*dim=fft->dim;
  ptrdiff_t      f_alloc_local,f_local_n0,f_local_0_start;
  ptrdiff_t      f_local_n1,f_local_1_end;
  ptrdiff_t      b_alloc_local,b_local_n0,b_local_0_start;
  ptrdiff_t      b_local_n1,b_local_1_end;
  fftw_complex   *data_fin,*data_fout,*data_bin,*data_bout;

  PetscFunctionBegin;
#if !defined(PETSC_USE_COMPLEX)
  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"not support for real numbers");
#endif
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  if (size == 1){
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Works only for parallel 1D");
  }
  else {
      if (ndim>1){
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Works only for parallel 1D");}
      else {
          f_alloc_local = fftw_mpi_local_size_1d(dim[0],comm,FFTW_FORWARD,FFTW_ESTIMATE,&f_local_n0,&f_local_0_start,&f_local_n1,&f_local_1_end);
          b_alloc_local = fftw_mpi_local_size_1d(dim[0],comm,FFTW_BACKWARD,FFTW_ESTIMATE,&b_local_n0,&b_local_0_start,&b_local_n1,&b_local_1_end);
          if (fin) {
            data_fin  = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*f_alloc_local);
            ierr = VecCreateMPIWithArray(comm,f_local_n0,N,(const PetscScalar*)data_fin,fin);CHKERRQ(ierr);
            (*fin)->ops->destroy   = VecDestroy_MPIFFTW;
          } 
          if (fout) {
            data_fout = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*f_alloc_local);
            ierr = VecCreateMPIWithArray(comm,f_local_n1,N,(const PetscScalar*)data_fout,fout);CHKERRQ(ierr);
            (*fout)->ops->destroy   = VecDestroy_MPIFFTW;
          }
          if (bin) {
            data_bin  = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*b_alloc_local);
            ierr = VecCreateMPIWithArray(comm,b_local_n0,N,(const PetscScalar*)data_bin,bin);CHKERRQ(ierr);
            (*bin)->ops->destroy   = VecDestroy_MPIFFTW;
          } 
          if (bout) {
            data_bout = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*b_alloc_local);
            ierr = VecCreateMPIWithArray(comm,b_local_n1,N,(const PetscScalar*)data_bout,bout);CHKERRQ(ierr);
            (*bout)->ops->destroy   = VecDestroy_MPIFFTW;
          }
  }
  if (fin){
    ierr = PetscLayoutReference(A->cmap,&(*fin)->map);CHKERRQ(ierr);
  }
  if (fout){
    ierr = PetscLayoutReference(A->rmap,&(*fout)->map);CHKERRQ(ierr);
  }
  if (bin){
    ierr = PetscLayoutReference(A->rmap,&(*bin)->map);CHKERRQ(ierr);
  }
  if (bout){
    ierr = PetscLayoutReference(A->rmap,&(*bout)->map);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
           

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
  Mat_FFTW       *fftw = (Mat_FFTW*)fft->data;
  PetscInt       N=fft->N, N1, n1,vsize;  

  PetscFunctionBegin;
//#if !defined(PETSC_USE_COMPLEX)
//  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"not support for real numbers");
//#endif
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);

  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  if (size == 1){ /* sequential case */
    if (fin) {ierr = VecCreateSeq(PETSC_COMM_SELF,N,fin);CHKERRQ(ierr);}
    if (fout){ierr = VecCreateSeq(PETSC_COMM_SELF,N,fout);CHKERRQ(ierr);}
    printf("The code successfully comes at the end of the routine with one processor\n");
  } else {        /* mpi case */
    ptrdiff_t      alloc_local,local_n0,local_0_start;
    ptrdiff_t      local_n1,local_1_end;
    PetscInt       ndim=fft->ndim,*dim=fft->dim,n=fft->n;
    fftw_complex   *data_fin,*data_fout;
    double         *data_finr, *data_foutr;
    ptrdiff_t local_1_start;
//    PetscInt ctr;
//    ptrdiff_t      ndim1,*pdim;
//    ndim1=(ptrdiff_t) ndim;
//    pdim = (ptrdiff_t *)calloc(ndim,sizeof(ptrdiff_t));

//    for(ctr=0;ctr<ndim;ctr++)
//        {k
//           pdim[ctr] = dim[ctr];
//       }

 

    switch (ndim){
    case 1:
      /* Get local size */
      /* We need to write an error message here saying that one cannot call this routine when doing paralllel 1D complex FFTW */
//      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Works only for parallel Multi-dimensional FFTW, Dimension>1. Check Documentation for MatGetVecs_FFTW1D routine");
      alloc_local = fftw_mpi_local_size_1d(dim[0],comm,FFTW_FORWARD,FFTW_ESTIMATE,&local_n0,&local_0_start,&local_n1,&local_1_end);
      if (fin) {
        data_fin  = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*alloc_local);
        ierr = VecCreateMPIWithArray(comm,local_n0,N,(const PetscScalar*)data_fin,fin);CHKERRQ(ierr);
        (*fin)->ops->destroy   = VecDestroy_MPIFFTW;
      } 
      if (fout) {
        data_fout = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*alloc_local);
        ierr = VecCreateMPIWithArray(comm,local_n1,N,(const PetscScalar*)data_fout,fout);CHKERRQ(ierr);
        (*fout)->ops->destroy   = VecDestroy_MPIFFTW;
      }
      break;
    case 2:
#if !defined(PETSC_USE_COMPLEX)
      alloc_local =  fftw_mpi_local_size_2d_transposed(dim[0],dim[1]/2+1,comm,&local_n0,&local_0_start,&local_n1,&local_1_start);
      N1 = 2*dim[0]*(dim[1]/2+1); n1 = 2*local_n0*(dim[1]/2+1);
      if (fin) {
        data_finr=(double *)fftw_malloc(sizeof(double)*alloc_local*2);
        ierr = VecCreateMPIWithArray(PETSC_COMM_WORLD,(PetscInt)n1,N1,(PetscScalar*)data_finr,fin);CHKERRQ(ierr);
        ierr = VecGetSize(*fin,&vsize);CHKERRQ(ierr);
        //printf("The code comes here with vector size %d\n",vsize);
        (*fin)->ops->destroy   = VecDestroy_MPIFFTW;
      }
      if (fout) {
        data_fout=(fftw_complex *)fftw_malloc(sizeof(fftw_complex)*alloc_local);
        ierr = VecCreateMPIWithArray(PETSC_COMM_WORLD,n1,N1,(PetscScalar*)data_fout,fout);CHKERRQ(ierr);
        (*fout)->ops->destroy   = VecDestroy_MPIFFTW;
      }
      printf("Vector size from fftw.c is  given by %d, %d\n",n1,N1);
     
#else
      /* Get local size */
     printf("Hope this does not come here");
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
     printf("Hope this does not come here");
#endif
      break;
    case 3:
      /* Get local size */
#if !defined(PETSC_USE_COMPLEX)
  SETERRQ(comm,PETSC_ERR_SUP,"Not done yet");
#else
      alloc_local = fftw_mpi_local_size_3d(dim[0],dim[1],dim[2],comm,&local_n0,&local_0_start);
//      printf("The quantity n is %d",n);
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
#endif
      break;
    default:
      /* Get local size */
#if !defined(PETSC_USE_COMPLEX)
  SETERRQ(comm,PETSC_ERR_SUP,"Not done yet");
#else
      alloc_local = fftw_mpi_local_size(fftw->ndim_fftw,fftw->dim_fftw,comm,&local_n0,&local_0_start);
//      printf("The value of alloc local is %d from process %d\n",alloc_local,rank);
//      printf("The value of alloc local is %d",alloc_local);
//      pdim=(ptrdiff_t *)calloc(ndim,sizeof(ptrdiff_t));
//      for(i=0;i<ndim;i++)
//         {
//          pdim[i]=dim[i];printf("%d",pdim[i]);
//         }
//      alloc_local = fftw_mpi_local_size(ndim,pdim,comm,&local_n0,&local_0_start);
//      printf("The quantity n is %d",n);
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
#endif
      break;
    }
  } 
//  if (fin){
//    ierr = PetscLayoutReference(A->cmap,&(*fin)->map);CHKERRQ(ierr);
//  }
//  if (fout){
//    ierr = PetscLayoutReference(A->rmap,&(*fout)->map);CHKERRQ(ierr);
//  }
  PetscFunctionReturn(0);
}

//EXTERN_C_BEGIN - Do we need this?
#undef __FUNCT__  
#define __FUNCT__ "InputTransformFFT"
PetscErrorCode InputTransformFFT(Mat A,Vec x,Vec y)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscTryMethod(A,"InputTransformFFT_C",(Mat,Vec,Vec),(A,x,y));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
//EXTERN_C_END - Do we need this?
/*
      InputTransformFFT_FFTW - Copies the user data to the vector that goes into FFTW block
  Input A, x, y
  A - FFTW matrix
  x - user data
  Options Database Keys:
+ -mat_fftw_plannerflags - set FFTW planner flags

   Level: intermediate
   
*/

EXTERN_C_BEGIN 
#undef __FUNCT__  
#define __FUNCT__ "InputTransformFFT_FTTW"
PetscErrorCode InputTransformFFT_FFTW(Mat A,Vec x,Vec y)
{
  PetscErrorCode ierr;
  MPI_Comm       comm=((PetscObject)A)->comm;
  Mat_FFT        *fft = (Mat_FFT*)A->data;
  Mat_FFTW       *fftw = (Mat_FFTW*)fft->data;
  PetscInt       N=fft->N, N1, n1 ,NM;
  PetscInt       ndim=fft->ndim,*dim=fft->dim,n=fft->n;
  PetscInt       low, *indx1, *indx2, tempindx, tempindx1, *indx3, *indx4; 
  PetscInt       i,j,rank,size; 
  ptrdiff_t      alloc_local,local_n0,local_0_start;
  ptrdiff_t      local_n1,local_1_start;
  VecScatter     vecscat;
  IS             list1,list2;

  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(y,&low,PETSC_NULL);
  printf("Local ownership starts at %d\n",low);

 switch (ndim){
 case 1:
  SETERRQ(comm,PETSC_ERR_SUP,"Not Supported by FFTW");
  break;
 case 2:
      alloc_local =  fftw_mpi_local_size_2d_transposed(dim[0],dim[1]/2+1,comm,&local_n0,&local_0_start,&local_n1,&local_1_start);
      N1 = 2*dim[0]*(dim[1]/2+1); n1 = 2*local_n0*(dim[1]/2+1);
     
     ierr = PetscMalloc(sizeof(PetscInt)*((PetscInt)local_n0)*N1,&indx1);CHKERRQ(ierr);
     ierr = PetscMalloc(sizeof(PetscInt)*((PetscInt)local_n0)*N1,&indx2);CHKERRQ(ierr);
     printf("Val local_0_start = %ld\n",local_0_start);
      
     if (dim[1]%2==0)
      NM = dim[1]+2;
    else
      NM = dim[1]+1; 
 
     for (i=0;i<local_n0;i++){
        for (j=0;j<dim[1];j++){
            tempindx = i*dim[1] + j;
            tempindx1 = i*NM + j;
            indx1[tempindx]=local_0_start*dim[1]+tempindx;
            indx2[tempindx]=low+tempindx1;
           // printf("Val tempindx1 = %d\n",tempindx1);
  //          printf("index1 %d from proc %d is \n",indx1[tempindx],rank);
  //          printf("index2 %d from proc %d is \n",indx2[tempindx],rank);
  //          printf("-------------------------\n",indx2[tempindx],rank);
        }
     }
   
     ierr = ISCreateGeneral(comm,local_n0*dim[1],indx1,PETSC_COPY_VALUES,&list1);CHKERRQ(ierr);
     ierr = ISCreateGeneral(comm,local_n0*dim[1],indx2,PETSC_COPY_VALUES,&list2);CHKERRQ(ierr);

     ierr = VecScatterCreate(x,list1,y,list2,&vecscat);CHKERRQ(ierr); 
     ierr = VecScatterBegin(vecscat,x,y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
     ierr = VecScatterEnd(vecscat,x,y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
     ierr = VecScatterDestroy(&vecscat);CHKERRQ(ierr);
     break;

 case 3:
  SETERRQ(comm,PETSC_ERR_SUP,"Not Done Yet");
  break;

 default:
  SETERRQ(comm,PETSC_ERR_SUP,"Not Done Yet");
  break;
 }

 return 0;
}
EXTERN_C_END


//EXTERN_C_BEGIN - Do we need this?
#undef __FUNCT__  
#define __FUNCT__ "OutputTransformFFT"
PetscErrorCode OutputTransformFFT(Mat A,Vec x,Vec y)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscTryMethod(A,"OutputTransformFFT_C",(Mat,Vec,Vec),(A,x,y));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
//EXTERN_C_END - Do we need this?
/*
      OutputTransformFFT_FFTW - Copies the FFTW output to the PETSc vector that user can use 
  Input A, x, y
  A - FFTW matrix
  x - FFTW vector
  y - PETSc vector that user can read
  Options Database Keys:
+ -mat_fftw_plannerflags - set FFTW planner flags

   Level: intermediate
   
*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "OutputTransformFFT_FTTW"
PetscErrorCode OutputTransformFFT_FFTW(Mat A,Vec x,Vec y)
{
  PetscErrorCode ierr;
  MPI_Comm       comm=((PetscObject)A)->comm;
  Mat_FFT        *fft = (Mat_FFT*)A->data;
  Mat_FFTW       *fftw = (Mat_FFTW*)fft->data;
  PetscInt       N=fft->N, N1, n1 ,NM;
  PetscInt       ndim=fft->ndim,*dim=fft->dim,n=fft->n;
  PetscInt       low, *indx1, *indx2, tempindx, tempindx1, *indx3, *indx4;
  PetscInt       i,j,rank,size;
  ptrdiff_t      alloc_local,local_n0,local_0_start;
  ptrdiff_t      local_n1,local_1_start;
  VecScatter     vecscat;
  IS             list1,list2;

  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(y,&low,PETSC_NULL);
  printf("Local ownership starts at %d\n",low);

 switch (ndim){
 case 1:
  SETERRQ(comm,PETSC_ERR_SUP,"Not Supported by FFTW");
  break;
 case 2:
      alloc_local =  fftw_mpi_local_size_2d_transposed(dim[0],dim[1]/2+1,comm,&local_n0,&local_0_start,&local_n1,&local_1_start);
      N1 = 2*dim[0]*(dim[1]/2+1); n1 = 2*local_n0*(dim[1]/2+1);

     ierr = PetscMalloc(sizeof(PetscInt)*((PetscInt)local_n0)*N1,&indx1);CHKERRQ(ierr);
     ierr = PetscMalloc(sizeof(PetscInt)*((PetscInt)local_n0)*N1,&indx2);CHKERRQ(ierr);
     printf("Val local_0_start = %ld\n",local_0_start);

     if (dim[1]%2==0)
      NM = dim[1]+2;
    else
      NM = dim[1]+1;


    
     for (i=0;i<local_n0;i++){
        for (j=0;j<dim[1];j++){
            tempindx = i*dim[1] + j;
            tempindx1 = i*NM + j;
            indx1[tempindx]=local_0_start*dim[1]+tempindx;
            indx2[tempindx]=low+tempindx1;
            printf("Val tempindx1 = %d\n",tempindx1);
            printf("index1 %d from proc %d is \n",indx1[tempindx],rank);
            printf("index2 %d from proc %d is \n",indx2[tempindx],rank);
            printf("-------------------------\n",indx2[tempindx],rank);
        }
     }

     ierr = ISCreateGeneral(comm,local_n0*dim[1],indx1,PETSC_COPY_VALUES,&list1);CHKERRQ(ierr);
     ierr = ISCreateGeneral(comm,local_n0*dim[1],indx2,PETSC_COPY_VALUES,&list2);CHKERRQ(ierr);

     ierr = VecScatterCreate(x,list2,y,list1,&vecscat);CHKERRQ(ierr);
     ierr = VecScatterBegin(vecscat,x,y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
     ierr = VecScatterEnd(vecscat,x,y,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
     ierr = VecScatterDestroy(&vecscat);CHKERRQ(ierr);
     break;

 case 3:
  SETERRQ(comm,PETSC_ERR_SUP,"Not Done Yet");
  break;

 default:
  SETERRQ(comm,PETSC_ERR_SUP,"Not Done Yet");
  break;
 }

 return 0;
}
EXTERN_C_END




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
  PetscInt       p_flag,partial_dim=1,ctr;
  PetscMPIInt    size,rank;
  ptrdiff_t      *pdim;
  ptrdiff_t      local_n1,local_1_start;

  PetscFunctionBegin;
//#if !defined(PETSC_USE_COMPLEX)
//  SETERRQ(comm,PETSC_ERR_SUP,"not support for real numbers");
//#endif
 
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Barrier(PETSC_COMM_WORLD);

  pdim = (ptrdiff_t *)calloc(ndim,sizeof(ptrdiff_t));
  pdim[0] = dim[0];
  for(ctr=1;ctr<ndim;ctr++)
      {
          partial_dim *= dim[ctr]; 
          pdim[ctr] = dim[ctr];
      } 
//#if !defined(PETSC_USE_COMPLEX)
//  SETERRQ(comm,PETSC_ERR_SUP,"not support for real numbers");
//#endif

//  printf("partial dimension is %d",partial_dim);              
  if (size == 1) {
    ierr = MatSetSizes(A,N,N,N,N);CHKERRQ(ierr);  
    n = N;
  } else {
    ptrdiff_t alloc_local,local_n0,local_0_start,local_n1,local_1_end;
    switch (ndim){
    case 1:
#if !defined(PETSC_USE_COMPLEX)
  SETERRQ(comm,PETSC_ERR_SUP,"FFTW does not support parallel 1D real transform");
#endif
      alloc_local = fftw_mpi_local_size_1d(dim[0],comm,FFTW_FORWARD,FFTW_ESTIMATE,&local_n0,&local_0_start,&local_n1,&local_1_end);
      n = (PetscInt)local_n0;
      ierr = MatSetSizes(A,n,n,N,N);CHKERRQ(ierr);  
//      PetscObjectComposeFunctionDynamic((PetscObject)A,"MatGetVecs1DC_C","MatGetVecs1DC_FFTW",MatGetVecs1DC_FFTW);   
      break;
    case 2:
#if defined(PETSC_USE_COMPLEX)
      alloc_local = fftw_mpi_local_size_2d(dim[0],dim[1],comm,&local_n0,&local_0_start);
      /*
       PetscMPIInt    rank;
       PetscSynchronizedPrintf(comm,"[%d] MatCreateSeqFFTW: local_n0, local_0_start %d %d, N %d,dim %d, %d\n",rank,(PetscInt)local_n0*dim[1],(PetscInt)local_0_start,m,dim[0],dim[1]);
       PetscSynchronizedFlush(comm);
       */
      n = (PetscInt)local_n0*dim[1];
      ierr = MatSetSizes(A,n,n,N,N);CHKERRQ(ierr);  
#else
      alloc_local = fftw_mpi_local_size_2d_transposed(dim[0],dim[1]/2+1,PETSC_COMM_WORLD,&local_n0,&local_0_start,&local_n1,&local_1_start);
      n = 2*(PetscInt)local_n0*(dim[1]/2+1);
      ierr = MatSetSizes(A,n,n,2*dim[0]*(dim[1]/2+1),2*dim[0]*(dim[1]/2+1));
#endif 
      break;
    case 3:
//      printf("The value of alloc local is %d",alloc_local);
      n = (PetscInt)local_n0*dim[1]*dim[2];
      ierr = MatSetSizes(A,n,n,N,N);CHKERRQ(ierr);  
      break;
    default:
      alloc_local = fftw_mpi_local_size(ndim,pdim,comm,&local_n0,&local_0_start);
//      printf("The value of alloc local is %ld from process %d\n",alloc_local,rank);
//      alloc_local = fftw_mpi_local_size(ndim,dim,comm,&local_n0,&local_0_start);
      n = (PetscInt)local_n0*partial_dim;
//      printf("New partial dimension is %d %d %d",n,N,ndim);              
      ierr = MatSetSizes(A,n,n,N,N);CHKERRQ(ierr);  
      break;
    }
  }
  ierr = PetscObjectChangeTypeName((PetscObject)A,MATFFTW);CHKERRQ(ierr);

  ierr = PetscNewLog(A,Mat_FFTW,&fftw);CHKERRQ(ierr);
  fft->data = (void*)fftw;
  
  fft->n           = n;
  fftw->ndim_fftw  = (ptrdiff_t)ndim; // This is dimension of fft
  ierr = PetscMalloc(ndim*sizeof(ptrdiff_t), (ptrdiff_t *)&(fftw->dim_fftw));CHKERRQ(ierr); 
  for(ctr=0;ctr<ndim;ctr++) (fftw->dim_fftw)[ctr]=dim[ctr];
  
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
#if !defined(PETSC_USE_COMPLEX)
  PetscObjectComposeFunctionDynamic((PetscObject)A,"InputTransformFFT_C","InputTransformFFT_FFTW",InputTransformFFT_FFTW);   
  PetscObjectComposeFunctionDynamic((PetscObject)A,"OutputTransformFFT_C","OutputTransformFFT_FFTW",OutputTransformFFT_FFTW);  
#endif 
  
  /* get runtime options */
  ierr = PetscOptionsBegin(((PetscObject)A)->comm,((PetscObject)A)->prefix,"FFTW Options","Mat");CHKERRQ(ierr);
    ierr = PetscOptionsEList("-mat_fftw_plannerflags","Planner Flags","None",p_flags,4,p_flags[0],&p_flag,&flg);CHKERRQ(ierr);
    if (flg) {fftw->p_flag = (unsigned)p_flag;}
  PetscOptionsEnd();
  PetscFunctionReturn(0);
}
EXTERN_C_END




