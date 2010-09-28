#define PETSCKSP_DLL

/*  -------------------------------------------------------------------- */

/* 
   Include files needed for the CUDA Smoothed Aggregation preconditioner:
     pcimpl.h - private include file intended for use by all preconditioners 
*/

#include "private/pcimpl.h"   /*I "petscpc.h" I*/
#include "../src/mat/impls/aij/seq/aij.h"
#include <cusp/precond/smoothed_aggregation.h>
#include "../src/vec/vec/impls/dvecimpl.h"
#include "../src/mat/impls/aij/seq/seqcuda/cudamatimpl.h"

#define cudasaprecond cusp::precond::smoothed_aggregation<PetscInt,PetscScalar,cusp::device_memory>

/* 
   Private context (data structure) for the SACUDA preconditioner.  
*/
typedef struct {
 cudasaprecond* SACUDA;
} PC_SACUDA;


/* -------------------------------------------------------------------------- */
/*
   PCSetUp_SACUDA - Prepares for the use of the SACUDA preconditioner
                    by setting data structures and options.   

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCSetUp()

   Notes:
   The interface routine PCSetUp() is not usually called directly by
   the user, but instead is called by PCApply() if necessary.
*/
#undef __FUNCT__  
#define __FUNCT__ "PCSetUp_SACUDA"
static PetscErrorCode PCSetUp_SACUDA(PC pc)
{
  PC_SACUDA      *sa = (PC_SACUDA*)pc->data;
  PetscBool      flg = PETSC_FALSE;
  PetscErrorCode ierr;
  Mat_SeqAIJCUDA *gpustruct;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)pc->pmat,MATSEQAIJCUDA,&flg);CHKERRQ(ierr);;
  if (!flg) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_SUP,"Currently only handles CUDA matrices");
  if (pc->setupcalled != 0){
    try {
      delete sa->SACUDA;
    } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUDA error: %s", ex);
    } 
  }
  try {
    gpustruct  = (Mat_SeqAIJCUDA *)(pc->pmat->spptr);
    sa->SACUDA = new cudasaprecond(*(CUSPMATRIX*)gpustruct->mat);
  } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUDA error: %s", ex);
  } 
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCApply_SACUDA - Applies the SACUDA preconditioner to a vector.

   Input Parameters:
.  pc - the preconditioner context
.  x - input vector

   Output Parameter:
.  y - output vector

   Application Interface Routine: PCApply()
 */
#undef __FUNCT__  
#define __FUNCT__ "PCApply_SACUDA"
static PetscErrorCode PCApply_SACUDA(PC pc,Vec x,Vec y)
{
  PC_SACUDA      *sac = (PC_SACUDA*)pc->data;
  PetscErrorCode ierr;
  PetscBool      flg1,flg2;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)x,VECSEQCUDA,&flg1);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)y,VECSEQCUDA,&flg2);CHKERRQ(ierr);
  if (!(flg1 && flg2)) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_SUP, "Currently only handles CUDA vectors");
  if (!sac->SACUDA) {
    ierr = PCSetUp_SACUDA(pc);CHKERRQ(ierr);
  }
  ierr = VecCUDACopyToGPU(x);CHKERRQ(ierr);
  ierr = VecCUDAAllocateCheck(y);CHKERRQ(ierr);
  try {
    cusp::multiply(*sac->SACUDA,*((Vec_CUDA *)x->spptr)->GPUarray,*((Vec_CUDA *)y->spptr)->GPUarray);
    y->valid_GPU_array = PETSC_CUDA_GPU;
  } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUDA error: %s", ex);
  } 
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
/*
   PCDestroy_SACUDA - Destroys the private context for the SACUDA preconditioner
   that was created with PCCreate_SACUDA().

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCDestroy()
*/
#undef __FUNCT__  
#define __FUNCT__ "PCDestroy_SACUDA"
static PetscErrorCode PCDestroy_SACUDA(PC pc)
{
  PC_SACUDA      *sac  = (PC_SACUDA*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (sac->SACUDA) {
    try {
      delete sac->SACUDA;
    } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUDA error: %s", ex);
    } 
}

  /*
      Free the private data structure that was hanging off the PC
  */
  ierr = PetscFree(sac);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSetFromOptions_SACUDA"
static PetscErrorCode PCSetFromOptions_SACUDA(PC pc)
{
    PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("SACUDA options");CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */



EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCreate_SACUDA"
PetscErrorCode PETSCKSP_DLLEXPORT PCCreate_SACUDA(PC pc)
{
  PC_SACUDA      *sac;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*
     Creates the private data structure for this preconditioner and
     attach it to the PC object.
  */
  ierr      = PetscNewLog(pc,PC_SACUDA,&sac);CHKERRQ(ierr);
  pc->data  = (void*)sac;

  /*
     Initialize the pointer to zero
  */
  sac->SACUDA          = 0;


  /*
      Set the pointers for the functions that are provided above.
      Now when the user-level routines (such as PCApply(), PCDestroy(), etc.)
      are called, they will automatically call these functions.  Note we
      choose not to provide a couple of these functions since they are
      not needed.
  */
  pc->ops->apply               = PCApply_SACUDA;
  pc->ops->applytranspose      = 0;
  pc->ops->setup               = PCSetUp_SACUDA;
  pc->ops->destroy             = PCDestroy_SACUDA;
  pc->ops->setfromoptions      = PCSetFromOptions_SACUDA;
  pc->ops->view                = 0;
  pc->ops->applyrichardson     = 0;
  pc->ops->applysymmetricleft  = 0;
  pc->ops->applysymmetricright = 0;
  PetscFunctionReturn(0);
}
EXTERN_C_END
