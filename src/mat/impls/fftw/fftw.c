#define PETSCMAT_DLL

/*
    Provides an interface to the FFTW package.
    Testing examples can be found in ~src/mat/examples/tests
*/

#include "private/matimpl.h"          /*I "petscmat.h" I*/
EXTERN_C_BEGIN 
#include "fftw3.h"
EXTERN_C_END 

typedef struct {
  PetscInt       ndim;
  PetscInt       *dim;
  fftw_plan      p_forward,p_backward;
  unsigned       p_flag; /* planner flags, FFTW_ESTIMATE,FFTW_MEASURE, FFTW_PATIENT, FFTW_EXHAUSTIVE */
  PetscScalar    *finarray,*foutarray,*binarray,*boutarray; /* keep track of arrays becaue fftw plan should be 
                                                               executed for the arrays with which the plan was created */
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
  Mat_FFTW       *fftw = (Mat_FFTW*)A->data;
  PetscScalar    *x_array,*y_array;
  PetscInt       ndim=fftw->ndim,*dim=fftw->dim;

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
#define __FUNCT__ "MatDestroy_SeqFFTW"
PetscErrorCode MatDestroy_SeqFFTW(Mat A)
{
  Mat_FFTW       *fftw = (Mat_FFTW*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin; 
#if !defined(PETSC_USE_COMPLEX)
  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"not support for real numbers");
#endif
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
PetscErrorCode  MatCreateSeqFFTW(MPI_Comm comm,PetscInt ndim,const PetscInt dim[],Mat* A)
{
  PetscErrorCode ierr;
  Mat_FFTW       *fftw;
  PetscInt       m,i;
  const char     *p_flags[]={"FFTW_ESTIMATE","FFTW_MEASURE","FFTW_PATIENT","FFTW_EXHAUSTIVE"};
  PetscBool      flg;
  PetscInt       p_flag;

  PetscFunctionBegin;
#if !defined(PETSC_USE_COMPLEX)
  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"not support for real numbers");
#endif
  if (ndim < 1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"ndim %d must be > 0",ndim);
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  m = 1;
  for (i=0; i<ndim; i++){
    if (dim[i] < 1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_USER,"dim[%d]=%d must be > 0",i,dim[i]);
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
