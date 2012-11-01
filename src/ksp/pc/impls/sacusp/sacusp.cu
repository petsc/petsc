
/*  -------------------------------------------------------------------- */

/*
   Include files needed for the CUSP Smoothed Aggregation preconditioner:
     pcimpl.h - private include file intended for use by all preconditioners
*/

#include <petsc-private/pcimpl.h>   /*I "petscpc.h" I*/
#include <../src/mat/impls/aij/seq/aij.h>
#include <cusp/monitor.h>
#include <cusp/precond/smoothed_aggregation.h>
#include <../src/vec/vec/impls/dvecimpl.h>
#include <../src/mat/impls/aij/seq/seqcusp/cuspmatimpl.h>

#define cuspsaprecond cusp::precond::smoothed_aggregation<PetscInt,PetscScalar,cusp::device_memory>

/*
   Private context (data structure) for the SACUSP preconditioner.
*/
typedef struct {
 cuspsaprecond* SACUSP;
  /*int cycles; */
} PC_SACUSP;

/*#undef __FUNCT__
#define __FUNCT__ "PCSACUSPSetCycles"
static PetscErrorCode PCSACUSPSetCycles(PC pc, int n)
{
  PC_SACUSP      *sac = (PC_SACUSP*)pc->data;

  PetscFunctionBegin;
  sac->cycles = n;
  PetscFunctionReturn(0);

  }*/

/* -------------------------------------------------------------------------- */
/*
   PCSetUp_SACUSP - Prepares for the use of the SACUSP preconditioner
                    by setting data structures and options.

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCSetUp()

   Notes:
   The interface routine PCSetUp() is not usually called directly by
   the user, but instead is called by PCApply() if necessary.
*/
#undef __FUNCT__
#define __FUNCT__ "PCSetUp_SACUSP"
static PetscErrorCode PCSetUp_SACUSP(PC pc)
{
  PC_SACUSP      *sa = (PC_SACUSP*)pc->data;
  PetscBool      flg = PETSC_FALSE;
  PetscErrorCode ierr;
#if !defined(PETSC_USE_COMPLEX)
  // protect these in order to avoid compiler warnings. This preconditioner does 
  // not work for complex types.
  Mat_SeqAIJCUSP *gpustruct;
  CUSPMATRIX* mat;	
#endif

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)pc->pmat,MATSEQAIJCUSP,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_SUP,"Currently only handles CUSP matrices");
  if (pc->setupcalled != 0){
    try {
      delete sa->SACUSP;
    } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
    }
  }
  try {
#if defined(PETSC_USE_COMPLEX)
    sa->SACUSP = 0; CHKERRQ(1); /* TODO */
#else
    ierr = MatCUSPCopyToGPU(pc->pmat);CHKERRQ(ierr);
    gpustruct  = (Mat_SeqAIJCUSP *)(pc->pmat->spptr);
#ifdef PETSC_HAVE_TXPETSCGPU
    ierr = gpustruct->mat->getCsrMatrix(&mat);CHKERRCUSP(ierr);
#else
    mat = (CUSPMATRIX*)gpustruct->mat;
#endif
    sa->SACUSP = new cuspsaprecond(*mat);
#endif

  } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
  }
  /*ierr = PetscOptionsInt("-pc_sacusp_cycles","Number of v-cycles to perform","PCSACUSPSetCycles",sa->cycles,
    &sa->cycles,PETSC_NULL);CHKERRQ(ierr);*/
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCApplyRichardson_SACUSP"
static PetscErrorCode PCApplyRichardson_SACUSP(PC pc, Vec b, Vec y, Vec w,PetscReal rtol, PetscReal abstol, PetscReal dtol, PetscInt its, PetscBool guesszero,PetscInt *outits,PCRichardsonConvergedReason *reason)
{
#if !defined(PETSC_USE_COMPLEX)
  // protect these in order to avoid compiler warnings. This preconditioner does 
  // not work for complex types.
  PC_SACUSP      *sac = (PC_SACUSP*)pc->data;
#endif
  PetscErrorCode ierr;
  CUSPARRAY      *barray,*yarray;

  PetscFunctionBegin;
  /* how to incorporate dtol, guesszero, w?*/
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = VecCUSPGetArrayRead(b,&barray);CHKERRQ(ierr);
  ierr = VecCUSPGetArrayReadWrite(y,&yarray);CHKERRQ(ierr);
  cusp::default_monitor<PetscReal> monitor(*barray,its,rtol,abstol);
#if defined(PETSC_USE_COMPLEX)
  CHKERRQ(1);
  /* TODO */
#else
  sac->SACUSP->solve(*barray,*yarray,monitor);
  *outits = monitor.iteration_count();
  if (monitor.converged()){
    /* how to discern between converging from RTOL or ATOL?*/
    *reason = PCRICHARDSON_CONVERGED_RTOL;
  } else{
    *reason = PCRICHARDSON_CONVERGED_ITS;
  }
#endif
  ierr = PetscObjectStateIncrease((PetscObject)y);CHKERRQ(ierr);
  ierr = VecCUSPRestoreArrayRead(b,&barray);CHKERRQ(ierr);
  ierr = VecCUSPRestoreArrayReadWrite(y,&yarray);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*
   PCApply_SACUSP - Applies the SACUSP preconditioner to a vector.

   Input Parameters:
.  pc - the preconditioner context
.  x - input vector

   Output Parameter:
.  y - output vector

   Application Interface Routine: PCApply()
 */
#undef __FUNCT__
#define __FUNCT__ "PCApply_SACUSP"
static PetscErrorCode PCApply_SACUSP(PC pc,Vec x,Vec y)
{
  PC_SACUSP      *sac = (PC_SACUSP*)pc->data;
  PetscErrorCode ierr;
  PetscBool      flg1,flg2;
  CUSPARRAY      *xarray,*yarray;

  PetscFunctionBegin;
  /*how to apply a certain fixed number of iterations?*/
  ierr = PetscObjectTypeCompare((PetscObject)x,VECSEQCUSP,&flg1);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)y,VECSEQCUSP,&flg2);CHKERRQ(ierr);
  if (!(flg1 && flg2)) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_SUP, "Currently only handles CUSP vectors");
  if (!sac->SACUSP) {
    ierr = PCSetUp_SACUSP(pc);CHKERRQ(ierr);
  }
  ierr = VecSet(y,0.0);CHKERRQ(ierr);
  ierr = VecCUSPGetArrayRead(x,&xarray);CHKERRQ(ierr);
  ierr = VecCUSPGetArrayWrite(y,&yarray);CHKERRQ(ierr);
  try {
#if defined(PETSC_USE_COMPLEX)

#else
    cusp::multiply(*sac->SACUSP,*xarray,*yarray);
#endif
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
   PCDestroy_SACUSP - Destroys the private context for the SACUSP preconditioner
   that was created with PCCreate_SACUSP().

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCDestroy()
*/
#undef __FUNCT__
#define __FUNCT__ "PCDestroy_SACUSP"
static PetscErrorCode PCDestroy_SACUSP(PC pc)
{
  PC_SACUSP      *sac  = (PC_SACUSP*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (sac->SACUSP) {
    try {
      delete sac->SACUSP;
    } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
    }
}

  /*
      Free the private data structure that was hanging off the PC
  */
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCSetFromOptions_SACUSP"
static PetscErrorCode PCSetFromOptions_SACUSP(PC pc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("SACUSP options");CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */


/*MC
     PCSACUSP  - A smoothed agglomeration algorithm that runs on the Nvidia GPU.


    http://research.nvidia.com/sites/default/files/publications/nvr-2011-002.pdf

   Level: advanced

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC

M*/

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PCCreate_SACUSP"
PetscErrorCode  PCCreate_SACUSP(PC pc)
{
  PC_SACUSP      *sac;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*
     Creates the private data structure for this preconditioner and
     attach it to the PC object.
  */
  ierr      = PetscNewLog(pc,PC_SACUSP,&sac);CHKERRQ(ierr);
  pc->data  = (void*)sac;

  /*
     Initialize the pointer to zero
     Initialize number of v-cycles to default (1)
  */
  sac->SACUSP          = 0;
  /*sac->cycles=1;*/


  /*
      Set the pointers for the functions that are provided above.
      Now when the user-level routines (such as PCApply(), PCDestroy(), etc.)
      are called, they will automatically call these functions.  Note we
      choose not to provide a couple of these functions since they are
      not needed.
  */
  pc->ops->apply               = PCApply_SACUSP;
  pc->ops->applytranspose      = 0;
  pc->ops->setup               = PCSetUp_SACUSP;
  pc->ops->destroy             = PCDestroy_SACUSP;
  pc->ops->setfromoptions      = PCSetFromOptions_SACUSP;
  pc->ops->view                = 0;
  pc->ops->applyrichardson     = PCApplyRichardson_SACUSP;
  pc->ops->applysymmetricleft  = 0;
  pc->ops->applysymmetricright = 0;
  PetscFunctionReturn(0);
}
EXTERN_C_END
