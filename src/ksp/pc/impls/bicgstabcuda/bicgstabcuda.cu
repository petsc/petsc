#define PETSCKSP_DLL

/*  -------------------------------------------------------------------- */

/* 
   Include files needed for the CUDA BiCGSTAB preconditioner:
     pcimpl.h - private include file intended for use by all preconditioners 
*/

#include "private/pcimpl.h"   /*I "petscpc.h" I*/
#include "../src/mat/impls/aij/seq/aij.h"
#include <cusp/monitor.h>
#undef VecType
#include <cusp/krylov/bicgstab.h>
#define VecType char*
#include "../src/vec/vec/impls/dvecimpl.h"
#include "../src/mat/impls/aij/seq/seqcuda/cudamatimpl.h"


/*
   Private context (data structure) for the CUDA BiCGStab preconditioner.
 */
typedef struct {
  PetscInt maxits;
  PetscReal rtol;
  CUSPMATRIX* mat;
} PC_BiCGStabCUDA;


/* -------------------------------------------------------------------------- */
/*
   PCSetUp_BiCGStabCUDA - Prepares for the use of the CUDA BiCGStab preconditioner
                    by setting data structures and options.   

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCSetUp()

   Notes:
   The interface routine PCSetUp() is not usually called directly by
   the user, but instead is called by PCApply() if necessary.
*/
#undef __FUNCT__  
#define __FUNCT__ "PCSetUp_BiCGStabCUDA"
static PetscErrorCode PCSetUp_BiCGStabCUDA(PC pc)
{
  PC_BiCGStabCUDA *bicg = (PC_BiCGStabCUDA*)pc->data;
  PetscBool       flg = PETSC_FALSE;
  Mat_SeqAIJCUDA  *gpustruct;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)pc->pmat,MATSEQAIJCUDA,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_SUP,"Currently only handles CUDA matrices");
  /* defaults */
  bicg->rtol = 1.e-1;
  bicg->maxits = 100;
  try{
    ierr = MatCUDACopyToGPU(pc->pmat);CHKERRCUDA(ierr);
    gpustruct = (Mat_SeqAIJCUDA *)(pc->pmat->spptr);
    bicg->mat = (CUSPMATRIX*)gpustruct->mat;
  } catch(char* ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUDA error: %s",ex);
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCApply_BiCGStabCUDA - Applies the BiCGStabCUDA preconditioner to a vector.

   Input Parameters:
.  pc - the preconditioner context
.  x - input vector

   Output Parameter:
.  y - output vector

   Application Interface Routine: PCApply()
 */
#undef __FUNCT__  
#define __FUNCT__ "PCApply_BiCGStabCUDA"
static PetscErrorCode PCApply_BiCGStabCUDA(PC pc,Vec x,Vec y)
{
  PC_BiCGStabCUDA *bicg = (PC_BiCGStabCUDA*)pc->data;
  PetscErrorCode  ierr;
  PetscBool       flg1,flg2;
  CUSPARRAY       *xarray,*yarray;

  PetscFunctionBegin;
  /*how to apply a certain fixed number of iterations?*/
  ierr = PetscTypeCompare((PetscObject)x,VECSEQCUDA,&flg1);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)y,VECSEQCUDA,&flg2);CHKERRQ(ierr);
  if (!(flg1 && flg2)) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_SUP, "Currently only handles CUDA vectors");
  if (!bicg->mat) {
    ierr = PCSetUp_BiCGStabCUDA(pc);CHKERRQ(ierr);
  }
  ierr = VecSet(y,0.0);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayRead(x,&xarray);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayWrite(y,&yarray);CHKERRQ(ierr);
  try {
    cusp::default_monitor<PetscScalar> monitor(*xarray,bicg->maxits,bicg->rtol);
    cusp::krylov::bicgstab(*bicg->mat,*yarray,*xarray,monitor);
  } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUDA error: %s", ex);
  }
  ierr = VecCUDARestoreArrayRead(x,&xarray);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayWrite(y,&yarray);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
/*
   PCDestroy_BiCGStabCUDA - Destroys the private context for the BiCGStabCUDA preconditioner
   that was created with PCCreate_BiCGStabCUDA().

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCDestroy()
*/
#undef __FUNCT__  
#define __FUNCT__ "PCDestroy_BiCGStabCUDA"
static PetscErrorCode PCDestroy_BiCGStabCUDA(PC pc)
{
  PC_BiCGStabCUDA *bicg = (PC_BiCGStabCUDA*)pc->data;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  /*
      Free the private data structure that was hanging off the PC
  */
  ierr = PetscFree(bicg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */



EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCreate_BiCGStabCUDA"
PetscErrorCode  PCCreate_BiCGStabCUDA(PC pc)
{
  PC_BiCGStabCUDA *bicg;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  /*
     Creates the private data structure for this preconditioner and
     attach it to the PC object.
   */
  ierr      = PetscNewLog(pc,PC_BiCGStabCUDA,&bicg);CHKERRQ(ierr);
  pc->data  = (void*)bicg;
  /*
      Set the pointers for the functions that are provided above.
      Now when the user-level routines (such as PCApply(), PCDestroy(), etc.)
      are called, they will automatically call these functions.  Note we
      choose not to provide a couple of these functions since they are
      not needed.
  */
  pc->ops->apply               = PCApply_BiCGStabCUDA;
  pc->ops->applytranspose      = 0;
  pc->ops->setup               = PCSetUp_BiCGStabCUDA;
  pc->ops->destroy             = PCDestroy_BiCGStabCUDA;
  pc->ops->setfromoptions      = 0;
  pc->ops->view                = 0;
  pc->ops->applyrichardson     = 0;
  pc->ops->applysymmetricleft  = 0;
  pc->ops->applysymmetricright = 0;
  PetscFunctionReturn(0);
}
EXTERN_C_END

