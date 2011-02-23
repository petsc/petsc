#define PETSCKSP_DLL

/*  -------------------------------------------------------------------- */

/*
   Include files needed for the CUSP AINV preconditioner:
     pcimpl.h - private include file intended for use by all preconditioners
*/

#include "private/pcimpl.h"   /*I "petscpc.h" I*/
#include "../src/mat/impls/aij/seq/aij.h"
#include <cusp/monitor.h>
#undef VecType
#include <cusp/precond/ainv.h>
#define VecType char*
#include "../src/vec/vec/impls/dvecimpl.h"
#include "../src/mat/impls/aij/seq/seqcusp/cuspmatimpl.h"

#define cuspainvprecond cusp::precond::scaled_bridson_ainv<PetscScalar,cusp::device_memory>

/*
   Private context (data structure) for the CUSP AINV preconditioner.  Note that this only works on CUSP SPD matrices.
 */
typedef struct {
  cuspainvprecond* AINVCUSP;
} PC_AINVCUSP;

/* -------------------------------------------------------------------------- */
/*
   PCSetUp_AINVCUSP - Prepares for the use of the CUSP AINV preconditioner
                    by setting data structures and options.

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCSetUp()

   Notes:
   The interface routine PCSetUp() is not usually called directly by
   the user, but instead is called by PCApply() if necessary.
*/
#undef __FUNCT__
#define __FUNCT__ "PCSetUp_AINVCUSP"
static PetscErrorCode PCSetUp_AINVCUSP(PC pc)
{
  PC_AINVCUSP     *ainv = (PC_AINVCUSP*)pc->data;
  PetscBool       flg = PETSC_FALSE;
  Mat_SeqAIJCUSP  *gpustruct;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)pc->pmat,MATSEQAIJCUSP,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_SUP,"Currently only handles CUSP matrices");
  if (pc->setupcalled != 0){
    try {
      delete ainv->AINVCUSP;
    } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
    }
  }
  try {
    ierr = MatCUSPCopyToGPU(pc->pmat);CHKERRCUSP(ierr);
    gpustruct = (Mat_SeqAIJCUSP *)(pc->pmat->spptr);
    /* This currently uses what http://code.google.com/p/cusp-library/source/browse/examples/Preconditioners/ainv.cu calls 'standard drop tolerance strategy'--must investigate.*/
    ainv->AINVCUSP =  new cuspainvprecond(*(CUSPMATRIX*)gpustruct->mat, 0.1);
  } catch(char* ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s",ex);
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCApply_AINVCUSP - Applies the CUSP AINV preconditioner to a vector.

   Input Parameters:
.  pc - the preconditioner context
.  x - input vector

   Output Parameter:
.  y - output vector

   Application Interface Routine: PCApply()
 */
#undef __FUNCT__
#define __FUNCT__ "PCApply_AINVCUSP"
static PetscErrorCode PCApply_AINVCUSP(PC pc,Vec x,Vec y)
{
  PC_AINVCUSP     *ainv = (PC_AINVCUSP*)pc->data;
  PetscErrorCode  ierr;
  PetscBool       flg1,flg2;
  CUSPARRAY       *xarray,*yarray;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)x,VECSEQCUSP,&flg1);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)y,VECSEQCUSP,&flg2);CHKERRQ(ierr);
  if (!(flg1 && flg2)) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_SUP, "Currently only handles CUSP vectors");
  if (!ainv->AINVCUSP) {
    ierr = PCSetUp_AINVCUSP(pc);CHKERRQ(ierr);
  }
  ierr = VecSet(y,0.0);CHKERRQ(ierr);
  ierr = VecCUSPGetArrayRead(x,&xarray);CHKERRQ(ierr);
  ierr = VecCUSPGetArrayWrite(y,&yarray);CHKERRQ(ierr);
  try {
    cusp::multiply(*ainv->AINVCUSP,*xarray,*yarray);
  } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
  }
  ierr = VecCUSPRestoreArrayRead(x,&xarray);CHKERRQ(ierr);
  ierr = VecCUSPRestoreArrayWrite(y,&yarray);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
/*
   PCDestroy_AINVCUSP - Destroys the private context for the AINVCUSP preconditioner
   that was created with PCCreate_AINVCUSP().

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCDestroy()
*/
#undef __FUNCT__
#define __FUNCT__ "PCDestroy_AINVCUSP"
static PetscErrorCode PCDestroy_AINVCUSP(PC pc)
{
  PC_AINVCUSP    *ainv  = (PC_AINVCUSP*)pc->data;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (ainv->AINVCUSP) {
    try {
      delete ainv->AINVCUSP;
    } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
    }
  }

  /*
      Free the private data structure that was hanging off the PC
  */
  ierr = PetscFree(ainv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCSetFromOptions_AINVCUSP"
static PetscErrorCode PCSetFromOptions_AINVCUSP(PC pc)
{
  /*PC_AINVCUSP     *ainv = (PC_AINVCUSP*)pc->data;*/
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("AINVCUSP options");CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */


EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PCCreate_AINVCUSP"
PetscErrorCode  PCCreate_AINVCUSP(PC pc)
{
  PC_AINVCUSP     *ainv;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  /*
     Creates the private data structure for this preconditioner and
     attach it to the PC object.
   */
  ierr                 = PetscNewLog(pc,PC_AINVCUSP,&ainv);CHKERRQ(ierr);
  pc->data             = (void*)ainv;
  ainv->AINVCUSP       = 0;
  /*
      Set the pointers for the functions that are provided above.
      Now when the user-level routines (such as PCApply(), PCDestroy(), etc.)
      are called, they will automatically call these functions.  Note we
      choose not to provide a couple of these functions since they are
      not needed.
  */
  pc->ops->apply               = PCApply_AINVCUSP;
  pc->ops->applytranspose      = 0;
  pc->ops->setup               = PCSetUp_AINVCUSP;
  pc->ops->destroy             = PCDestroy_AINVCUSP;
  pc->ops->setfromoptions      = PCSetFromOptions_AINVCUSP;
  pc->ops->view                = 0;
  pc->ops->applyrichardson     = 0;
  pc->ops->applysymmetricleft  = 0;
  pc->ops->applysymmetricright = 0;

  PetscFunctionReturn(0);
}
EXTERN_C_END

