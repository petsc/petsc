
/*  -------------------------------------------------------------------- */

/* 
   Include files needed for the CUSP Smoothed Aggregation preconditioner with Chebyshev polynomial smoothing:
     pcimpl.h - private include file intended for use by all preconditioners 
*/

#include <petsc-private/pcimpl.h>   /*I "petscpc.h" I*/
#include <../src/mat/impls/aij/seq/aij.h>
#include <cusp/monitor.h>
#undef VecType
#define USE_POLY_SMOOTHER 1
#include <cusp/precond/smoothed_aggregation.h>
#undef USE_POLY_SMOOTHER
#define VecType char*
#include <../src/vec/vec/impls/dvecimpl.h>
#include <../src/mat/impls/aij/seq/seqcusp/cuspmatimpl.h>

#define cuspsaprecond cusp::precond::smoothed_aggregation<PetscInt,PetscScalar,cusp::device_memory>

/* 
   Private context (data structure) for the SACUSPPoly preconditioner.  
*/
typedef struct {
 cuspsaprecond* SACUSPPoly;
  /*int cycles; */
} PC_SACUSPPoly;


/* -------------------------------------------------------------------------- */
/*
   PCSetUp_SACUSPPoly - Prepares for the use of the SACUSPPoly preconditioner
                    by setting data structures and options.   

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCSetUp()

   Notes:
   The interface routine PCSetUp() is not usually called directly by
   the user, but instead is called by PCApply() if necessary.
*/
#undef __FUNCT__  
#define __FUNCT__ "PCSetUp_SACUSPPoly"
static PetscErrorCode PCSetUp_SACUSPPoly(PC pc)
{
  PC_SACUSPPoly      *sa = (PC_SACUSPPoly*)pc->data;
  PetscBool      flg = PETSC_FALSE;
  PetscErrorCode ierr;
  Mat_SeqAIJCUSP *gpustruct;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)pc->pmat,MATSEQAIJCUSP,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_SUP,"Currently only handles CUSP matrices");
  if (pc->setupcalled != 0){
    try {
      delete sa->SACUSPPoly;
    } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
    } 
  }
  try {
    ierr = MatCUSPCopyToGPU(pc->pmat);CHKERRQ(ierr);
    gpustruct  = (Mat_SeqAIJCUSP *)(pc->pmat->spptr);
    sa->SACUSPPoly = new cuspsaprecond(*(CUSPMATRIX*)gpustruct->mat);
  } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
  } 
  /*ierr = PetscOptionsInt("-pc_sacusp_cycles","Number of v-cycles to perform","PCSACUSPSetCycles",sa->cycles,
    &sa->cycles,PETSC_NULL);CHKERRQ(ierr);*/
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCApplyRichardson_SACUSPPoly"
static PetscErrorCode PCApplyRichardson_SACUSPPoly(PC pc, Vec b, Vec y, Vec w,PetscReal rtol, PetscReal abstol, PetscReal dtol, PetscInt its, PetscBool guesszero,PetscInt *outits,PCRichardsonConvergedReason *reason)
{
  PC_SACUSPPoly      *sac = (PC_SACUSPPoly*)pc->data;
  PetscErrorCode ierr;
  CUSPARRAY      *barray,*yarray;
  
  PetscFunctionBegin;
  /* how to incorporate dtol, guesszero, w?*/
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = VecCUSPGetArrayRead(b,&barray);CHKERRQ(ierr);
  ierr = VecCUSPGetArrayReadWrite(y,&yarray);CHKERRQ(ierr);
  cusp::default_monitor<PetscScalar> monitor(*barray,its,rtol,abstol);
  sac->SACUSPPoly->solve(*barray,*yarray,monitor);
  *outits = monitor.iteration_count();
  if (monitor.converged()){
    /* how to discern between converging from RTOL or ATOL?*/
    *reason = PCRICHARDSON_CONVERGED_RTOL;
  } else{
    *reason = PCRICHARDSON_CONVERGED_ITS;
  }
  ierr = PetscObjectStateIncrease((PetscObject)y);CHKERRQ(ierr);
  ierr = VecCUSPRestoreArrayRead(b,&barray);CHKERRQ(ierr);
  ierr = VecCUSPRestoreArrayReadWrite(y,&yarray);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCApply_SACUSPPoly - Applies the SACUSPPoly preconditioner to a vector.

   Input Parameters:
.  pc - the preconditioner context
.  x - input vector

   Output Parameter:
.  y - output vector

   Application Interface Routine: PCApply()
 */
#undef __FUNCT__  
#define __FUNCT__ "PCApply_SACUSPPoly"
static PetscErrorCode PCApply_SACUSPPoly(PC pc,Vec x,Vec y)
{
  PC_SACUSPPoly      *sac = (PC_SACUSPPoly*)pc->data;
  PetscErrorCode ierr;
  PetscBool      flg1,flg2;
  CUSPARRAY      *xarray,*yarray;

  PetscFunctionBegin;
  /*how to apply a certain fixed number of iterations?*/
  ierr = PetscObjectTypeCompare((PetscObject)x,VECSEQCUSP,&flg1);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)y,VECSEQCUSP,&flg2);CHKERRQ(ierr);
  if (!(flg1 && flg2)) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_SUP, "Currently only handles CUSP vectors");
  if (!sac->SACUSPPoly) {
    ierr = PCSetUp_SACUSPPoly(pc);CHKERRQ(ierr);
  }
  ierr = VecSet(y,0.0);CHKERRQ(ierr);
  ierr = VecCUSPGetArrayRead(x,&xarray);CHKERRQ(ierr);
  ierr = VecCUSPGetArrayWrite(y,&yarray);CHKERRQ(ierr);
  try {
    cusp::multiply(*sac->SACUSPPoly,*xarray,*yarray);
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
   PCDestroy_SACUSPPoly - Destroys the private context for the SACUSPPoly preconditioner
   that was created with PCCreate_SACUSPPoly().

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCDestroy()
*/
#undef __FUNCT__  
#define __FUNCT__ "PCDestroy_SACUSPPoly"
static PetscErrorCode PCDestroy_SACUSPPoly(PC pc)
{
  PC_SACUSPPoly      *sac  = (PC_SACUSPPoly*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (sac->SACUSPPoly) {
    try {
      delete sac->SACUSPPoly;
    } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
    } 
}

  /*
      Free the private data structure that was hanging off the PC
  */
  ierr = PetscFree(sac);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSetFromOptions_SACUSPPoly"
static PetscErrorCode PCSetFromOptions_SACUSPPoly(PC pc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("SACUSPPoly options");CHKERRQ(ierr);			 
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */



EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCreate_SACUSPPoly"
PetscErrorCode  PCCreate_SACUSPPoly(PC pc)
{
  PC_SACUSPPoly      *sac;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*
     Creates the private data structure for this preconditioner and
     attach it to the PC object.
  */
  ierr      = PetscNewLog(pc,PC_SACUSPPoly,&sac);CHKERRQ(ierr);
  pc->data  = (void*)sac;

  /*
     Initialize the pointer to zero
     Initialize number of v-cycles to default (1)
  */
  sac->SACUSPPoly          = 0;
  /*sac->cycles=1;*/


  /*
      Set the pointers for the functions that are provided above.
      Now when the user-level routines (such as PCApply(), PCDestroy(), etc.)
      are called, they will automatically call these functions.  Note we
      choose not to provide a couple of these functions since they are
      not needed.
  */
  pc->ops->apply               = PCApply_SACUSPPoly;
  pc->ops->applytranspose      = 0;
  pc->ops->setup               = PCSetUp_SACUSPPoly;
  pc->ops->destroy             = PCDestroy_SACUSPPoly;
  pc->ops->setfromoptions      = PCSetFromOptions_SACUSPPoly;
  pc->ops->view                = 0;
  pc->ops->applyrichardson     = PCApplyRichardson_SACUSPPoly;
  pc->ops->applysymmetricleft  = 0;
  pc->ops->applysymmetricright = 0;
  PetscFunctionReturn(0);
}
EXTERN_C_END
