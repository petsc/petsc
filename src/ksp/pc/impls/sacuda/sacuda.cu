#define PETSCKSP_DLL

#define cudasaprecond cusp::precond::smoothed_aggregation<PetscInt,PetscScalar,cusp::device_memory>
/*  -------------------------------------------------------------------- */

/* 
   Include files needed for the CUDA Smoothed Aggregation preconditioner:
     pcimpl.h - private include file intended for use by all preconditioners 
*/

#include "private/pcimpl.h"   /*I "petscpc.h" I*/
#include "../src/vec/vec/impls/seq/seqcuda/cudavecimpl.h"

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
  PetscTruth     flg1 = PETSC_FALSE, flg2 = PETSC_FALSE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)pc->pmat,MATSEQAIJCUDA,&flg1);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)pc->pmat,MATMPIAIJCUDA,&flg2);CHKERRQ(ierr);
  if (!(flg1 || flg2)) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_SUP,"Currently only handles CUDA matrices");
  if (pc->setupcalled == 0){/* allocate space for preconditioner */
    sa->SACUDA = new cudasaprecond;
    PetscLogObjectParent(pc,sa->SACUDA);
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
  PC_Jacobi      *sac = (PC_SACUDA*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!sac->SACUDA) {
    ierr = PCSetUp_SACUDA(pc);CHKERRQ(ierr);
  }
  /* what goes here? */
  ierr = VecPointwiseMult(y,x,jac->diag);CHKERRQ(ierr);
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
  if (sac->SACUDA)     {delete sac->SACUDA;}

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
  PC_SACUDA      *sac = (PC_SACUDA*)pc->data;
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
  sac->SACCUDA          = 0;


  /*
      Set the pointers for the functions that are provided above.
      Now when the user-level routines (such as PCApply(), PCDestroy(), etc.)
      are called, they will automatically call these functions.  Note we
      choose not to provide a couple of these functions since they are
      not needed.
  */
  pc->ops->apply               = PCApply_SACUDA;
  pc->ops->applytranspose      = PCApply_SACUDA;
  pc->ops->setup               = PCSetUp_SACUDA;
  pc->ops->destroy             = PCDestroy_SACUDA;
  pc->ops->setfromoptions      = PCSetFromOptions_SACUDA;
  pc->ops->view                = 0;
  pc->ops->applyrichardson     = 0;
  pc->ops->applysymmetricleft  = 0
  pc->ops->applysymmetricright = 0
  PetscFunctionReturn(0);
}
EXTERN_C_END
