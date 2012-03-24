
/*  -------------------------------------------------------------------- */

/*
   Include files needed for the CUSP BiCGSTAB preconditioner:
     pcimpl.h - private include file intended for use by all preconditioners
*/

#include <petsc-private/pcimpl.h>   /*I "petscpc.h" I*/
#include <../src/mat/impls/aij/seq/aij.h>
#include <cusp/monitor.h>
#undef VecType
#include <cusp/krylov/bicgstab.h>
#define VecType char*
#include <../src/vec/vec/impls/dvecimpl.h>
#include <../src/mat/impls/aij/seq/seqcusp/cuspmatimpl.h>


/*
   Private context (data structure) for the CUSP BiCGStab preconditioner.
 */
typedef struct {
  PetscInt maxits;
  PetscReal rtol;
  PetscBool monitorverbose;
  CUSPMATRIX* mat;
} PC_BiCGStabCUSP;

#undef __FUNCT__
#define __FUNCT__ "PCBiCGStabCUSPSetTolerance_BiCGStabCUSP"
static PetscErrorCode PCBiCGStabCUSPSetTolerance_BiCGStabCUSP(PC pc,PetscReal rtol)
{
  PC_BiCGStabCUSP *bicg = (PC_BiCGStabCUSP*)pc->data;

  PetscFunctionBegin;
  bicg->rtol = rtol;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBiCGStabCUSPSetUseVerboseMonitor_BiCGStabCUSP"
static PetscErrorCode PCBiCGStabCUSPSetUseVerboseMonitor_BiCGStabCUSP(PC pc, PetscBool useverbose)
{
  PC_BiCGStabCUSP *bicg = (PC_BiCGStabCUSP*)pc->data;

  PetscFunctionBegin;
  bicg->monitorverbose = useverbose;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBiCGStabCUSPSetUseVerboseMonitor"
PetscErrorCode PCBiCGStabCUSPSetUseVerboseMonitor(PC pc, PetscBool useverbose)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID,1);
  ierr = PetscTryMethod(pc, "PCBiCGStabCUSPSetUseVerboseMonitors_C",(PC,PetscBool),(pc,useverbose));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBiCGStabCUSPSetIterations_BiCGStabCUSP"
static PetscErrorCode PCBiCGStabCUSPSetIterations_BiCGStabCUSP(PC pc, PetscInt its)
{
  PC_BiCGStabCUSP *bicg = (PC_BiCGStabCUSP*)pc->data;

  PetscFunctionBegin;
  bicg->maxits = its;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBiCGStabCUSPSetIterations"
PetscErrorCode PCBiCGStabCUSPSetITerations(PC pc, PetscInt its)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc, "PCBiCGStabCUSPSetIterations_C",(PC,PetscInt),(pc,its));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBiCGStabCUSPSetTolerance"
PetscErrorCode PCBiCGStabCUSPSetTolerance(PC pc, PetscReal rtol)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc, "PCBiCGStabCUSPSetTolerance_C",(PC,PetscReal),(pc,rtol));CHKERRQ(ierr);
  PetscFunctionReturn(0);
  }

/* -------------------------------------------------------------------------- */
/*
   PCSetUp_BiCGStabCUSP - Prepares for the use of the CUSP BiCGStab preconditioner
                    by setting data structures and options.

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCSetUp()

   Notes:
   The interface routine PCSetUp() is not usually called directly by
   the user, but instead is called by PCApply() if necessary.
*/
#undef __FUNCT__
#define __FUNCT__ "PCSetUp_BiCGStabCUSP"
static PetscErrorCode PCSetUp_BiCGStabCUSP(PC pc)
{
  PC_BiCGStabCUSP *bicg = (PC_BiCGStabCUSP*)pc->data;
  PetscBool       flg = PETSC_FALSE;
  Mat_SeqAIJCUSP  *gpustruct;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)pc->pmat,MATSEQAIJCUSP,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_SUP,"Currently only handles CUSP matrices");
  try{
    ierr = MatCUSPCopyToGPU(pc->pmat);CHKERRQ(ierr);
    gpustruct = (Mat_SeqAIJCUSP *)(pc->pmat->spptr);
    bicg->mat = (CUSPMATRIX*)gpustruct->mat;
  } catch(char* ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s",ex);
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCApply_BiCGStabCUSP - Applies the BiCGStabCUSP preconditioner to a vector.

   Input Parameters:
.  pc - the preconditioner context
.  x - input vector

   Output Parameter:
.  y - output vector

   Application Interface Routine: PCApply()
 */
#undef __FUNCT__
#define __FUNCT__ "PCApply_BiCGStabCUSP"
static PetscErrorCode PCApply_BiCGStabCUSP(PC pc,Vec x,Vec y)
{
  PC_BiCGStabCUSP *bicg = (PC_BiCGStabCUSP*)pc->data;
  PetscErrorCode  ierr;
  PetscBool       flg1,flg2;
  CUSPARRAY       *xarray,*yarray;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)x,VECSEQCUSP,&flg1);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)y,VECSEQCUSP,&flg2);CHKERRQ(ierr);
  if (!(flg1 && flg2)) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_SUP, "Currently only handles CUSP vectors");
  if (!bicg->mat) {
    ierr = PCSetUp_BiCGStabCUSP(pc);CHKERRQ(ierr);
  }
  ierr = VecSet(y,0.0);CHKERRQ(ierr);
  ierr = VecCUSPGetArrayRead(x,&xarray);CHKERRQ(ierr);
  ierr = VecCUSPGetArrayWrite(y,&yarray);CHKERRQ(ierr);
  try {
    cusp::default_monitor<PetscScalar> monitor(*xarray,bicg->maxits,bicg->rtol);
    if (bicg->monitorverbose){
      cusp::verbose_monitor<PetscScalar> verbosemonitor(*xarray,bicg->maxits,bicg->rtol);
      cusp::krylov::bicgstab(*bicg->mat,*yarray,*xarray,verbosemonitor);
    } else {
      cusp::krylov::bicgstab(*bicg->mat,*yarray,*xarray,monitor);
    }
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
   PCDestroy_BiCGStabCUSP - Destroys the private context for the BiCGStabCUSP preconditioner
   that was created with PCCreate_BiCGStabCUSP().

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCDestroy()
*/
#undef __FUNCT__
#define __FUNCT__ "PCDestroy_BiCGStabCUSP"
static PetscErrorCode PCDestroy_BiCGStabCUSP(PC pc)
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  /*
      Free the private data structure that was hanging off the PC
  */
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCSetFromOptions_BiCGStabCUSP"
static PetscErrorCode PCSetFromOptions_BiCGStabCUSP(PC pc)
{
  PC_BiCGStabCUSP *bicg = (PC_BiCGStabCUSP*)pc->data;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("BiCGStabCUSP options");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-pc_bicgstabcusp_rtol","relative tolerance for BiCGStabCUSP preconditioner","PCBiCGStabCUSPSetTolerance",bicg->rtol,&bicg->rtol,0);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_bicgstabcusp_max_it","maximum iterations for BiCGStabCUSP preconditioner","PCBiCGStabCUSPSetIterations",bicg->maxits,&bicg->maxits,0);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_bicgstabcusp_monitor_verbose","Print information about GPU BiCGStabCUSP iterations","PCBiCGStabCUSPSetUseVerboseMonitor",bicg->monitorverbose,&bicg->monitorverbose,0);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */


EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PCCreate_BiCGStabCUSP"
PetscErrorCode  PCCreate_BiCGStabCUSP(PC pc)
{
  PC_BiCGStabCUSP *bicg;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  /*
     Creates the private data structure for this preconditioner and
     attach it to the PC object.
   */
  ierr         = PetscNewLog(pc,PC_BiCGStabCUSP,&bicg);CHKERRQ(ierr);
  /*
     Set default values.  We don't actually want to set max iterations as far as I know, but the Cusp monitor requires them so we use a large number.
   */
  bicg->maxits         = 1000;
  bicg->rtol           = 1.e-1;
  bicg->monitorverbose = PETSC_FALSE;
  pc->data             = (void*)bicg;
  /*
      Set the pointers for the functions that are provided above.
      Now when the user-level routines (such as PCApply(), PCDestroy(), etc.)
      are called, they will automatically call these functions.  Note we
      choose not to provide a couple of these functions since they are
      not needed.
  */
  pc->ops->apply               = PCApply_BiCGStabCUSP;
  pc->ops->applytranspose      = 0;
  pc->ops->setup               = PCSetUp_BiCGStabCUSP;
  pc->ops->destroy             = PCDestroy_BiCGStabCUSP;
  pc->ops->setfromoptions      = PCSetFromOptions_BiCGStabCUSP;
  pc->ops->view                = 0;
  pc->ops->applyrichardson     = 0;
  pc->ops->applysymmetricleft  = 0;
  pc->ops->applysymmetricright = 0;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCBiCGStabCUSPSetTolerance_C","PCBiCGStabCUSPSetTolerance_BiCGStabCUSP",PCBiCGStabCUSPSetTolerance_BiCGStabCUSP);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc, "PCBiCGStabCUSPSetIterations_C","PCBiCGStabCUSPSetIterations_BiCGStabCUSP", PCBiCGStabCUSPSetIterations_BiCGStabCUSP);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc, "PCBiCGStabCUSPSetUseVerboseMonitor_C", "PCBiCGStabCUSPSetUseVerboseMonitor_BiCGStabCUSP", PCBiCGStabCUSPSetUseVerboseMonitor_BiCGStabCUSP);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

