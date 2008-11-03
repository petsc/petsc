#define PETSCMAT_DLL

/*
    Provides an interface to the FFTW package.
    Testing examples can be found in ~src/mat/examples/tests
*/

#include "private/matimpl.h"          /*I "petscmat.h" I*/
EXTERN_C_BEGIN 
#if defined(PETSC_USE_COMPLEX)
#include "fftw3.h"
#endif
EXTERN_C_END 

typedef struct {
  PetscInt       ndim;
  PetscInt       *dim;
  fftw_plan      p_forward,p_backward;
  unsigned       p_flag; /* planner flags, FFTW_ESTIMATE,FFTW_MEASURE, FFTW_PATIENT, FFTW_EXHAUSTIVE */
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
    fftw_execute(fftw->p_forward);
  } else {
    /* use existing plan */
    fftw_execute_dft(fftw->p_forward,(fftw_complex*)x_array,(fftw_complex*)y_array);
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
  Mat_FFTW       *fftw = (Mat_FFTW*)A->data;
  PetscScalar    *x_array,*y_array;
  PetscInt       ndim=fftw->ndim,*dim=fftw->dim;

  PetscFunctionBegin;
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
    fftw_execute(fftw->p_backward);CHKERRQ(ierr);
  } else { /* use existing plan */
    fftw_execute_dft(fftw->p_backward,(fftw_complex*)x_array,(fftw_complex*)y_array);
  }
  ierr = VecRestoreArray(y,&y_array);CHKERRQ(ierr);
  ierr = VecRestoreArray(x,&x_array);CHKERRQ(ierr);
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
  fftw_destroy_plan(fftw->p_forward);
  fftw_destroy_plan(fftw->p_backward);
  ierr = PetscFree(fftw);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)A,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatCreateSeqFFTW"
/*@
      MatCreateSeqFFTW - Creates a matrix object that provides sequential FFT
  via the external package FFTW

   Collective on MPI_Comm

   Input Parameter:
+   comm - MPI communicator, set to PETSC_COMM_SELF
.   ndim - the ndim-dimensional transform
-   dim - array of size ndim, dim[i] contains the vector length in the i-dimension

   Output Parameter:
.   A  - the matrix

  Options Database Keys:
+ -mat_fftw_plannerflags - set FFTW planner flags

   Level: intermediate
   
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatCreateSeqFFTW(MPI_Comm comm,PetscInt ndim,const PetscInt dim[],Mat* A)
{
  PetscErrorCode ierr;
  Mat_FFTW       *fftw;
  PetscInt       m,i;
  const char     *p_flags[]={"FFTW_ESTIMATE","FFTW_MEASURE","FFTW_PATIENT","FFTW_EXHAUSTIVE"};
  PetscTruth     flg;
  PetscInt       p_flag;

  PetscFunctionBegin;
  if (ndim < 0) SETERRQ1(PETSC_ERR_USER,"ndim %d must be > 0",ndim);
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  m = 1;
  for (i=0; i<ndim; i++){
    if (dim[i] < 0) SETERRQ2(PETSC_ERR_USER,"dim[%d]=%d must be > 0",i,dim[i]);
    m *= dim[i];
  }
  ierr = MatSetSizes(*A,m,m,m,m);CHKERRQ(ierr);  
  ierr = PetscObjectChangeTypeName((PetscObject)*A,MATSEQFFTW);CHKERRQ(ierr);

  ierr = PetscNewLog(*A,Mat_FFTW,&fftw);CHKERRQ(ierr);
  (*A)->data = (void*)fftw;
  ierr = PetscMalloc((ndim+1)*sizeof(PetscInt),&fftw->dim);CHKERRQ(ierr);
  ierr = PetscMemcpy(fftw->dim,dim,ndim*sizeof(PetscInt));CHKERRQ(ierr);
  fftw->ndim       = ndim;
  fftw->p_forward  = 0;
  fftw->p_backward = 0;
  fftw->p_flag     = FFTW_ESTIMATE;

  (*A)->ops->mult          = MatMult_SeqFFTW;
  (*A)->ops->multtranspose = MatMultTranspose_SeqFFTW;
  (*A)->assembled          = PETSC_TRUE;
  (*A)->ops->destroy       = MatDestroy_SeqFFTW;

  /* get runtime options */
  ierr = PetscOptionsBegin(((PetscObject)(*A))->comm,((PetscObject)(*A))->prefix,"FFTW Options","Mat");CHKERRQ(ierr);
  ierr = PetscOptionsEList("-mat_fftw_plannerflags","Planner Flags","None",p_flags,4,p_flags[0],&p_flag,&flg);CHKERRQ(ierr);
  if (flg) {fftw->p_flag = (unsigned)p_flag;}
  PetscOptionsEnd();
  PetscFunctionReturn(0);
}
