
/*  -------------------------------------------------------------------- */

/*
   Include files needed for the CUSP AINV preconditioner:
     pcimpl.h - private include file intended for use by all preconditioners
*/
#define PETSC_SKIP_SPINLOCK
#include <petsc/private/pcimpl.h>   /*I "petscpc.h" I*/
#include <../src/mat/impls/aij/seq/aij.h>
#include <cusp/monitor.h>
#undef VecType
#include <cusp/precond/ainv.h>
#define VecType char*
#include <../src/vec/vec/impls/dvecimpl.h>
#include <../src/mat/impls/aij/seq/seqcusp/cuspmatimpl.h>
#include <../src/vec/vec/impls/seq/seqcusp/cuspvecimpl.h>

#define cuspainvprecondscaled cusp::precond::scaled_bridson_ainv<PetscScalar,cusp::device_memory>
#define cuspainvprecond cusp::precond::bridson_ainv<PetscScalar,cusp::device_memory>

/*
   Private context (data structure) for the CUSP AINV preconditioner.  Note that this only works on CUSP SPD matrices.
 */
typedef struct {
  void      *AINVCUSP;
  PetscBool scaled; /* Whether to use the scaled version of the Bridson AINV or not */

  PetscInt  nonzeros; /* can only use one of nonzeros, droptolerance, linparam at once */
  PetscReal droptolerance;
  PetscInt  linparam;
  PetscBool uselin;
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
static PetscErrorCode PCSetUp_AINVCUSP(PC pc)
{
  PC_AINVCUSP *ainv = (PC_AINVCUSP*)pc->data;
  PetscBool   flg   = PETSC_FALSE;
#if !defined(PETSC_USE_COMPLEX)
  // protect these in order to avoid compiler warnings. This preconditioner does
  // not work for complex types.
  Mat_SeqAIJCUSP *gpustruct;
#endif
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)pc->pmat,MATSEQAIJCUSP,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Currently only handles CUSP matrices");
  if (pc->setupcalled != 0) {
    try {
      if (ainv->scaled) delete (cuspainvprecondscaled*)ainv->AINVCUSP;
      else delete (cuspainvprecond*)ainv->AINVCUSP;
    } catch(char *ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
    }
  }
  try {
    ierr = MatCUSPCopyToGPU(pc->pmat);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
    ainv->AINVCUSP =  0;CHKERRQ(1); /* TODO */
#else
    gpustruct = (Mat_SeqAIJCUSP*)(pc->pmat->spptr);

   if (gpustruct->format==MAT_CUSP_ELL) {
      CUSPMATRIXELL *mat = (CUSPMATRIXELL*)gpustruct->mat;
      if (ainv->scaled) ainv->AINVCUSP = new cuspainvprecondscaled(*mat, ainv->droptolerance,ainv->nonzeros,ainv->uselin,ainv->linparam);
      else ainv->AINVCUSP = new cuspainvprecond(*mat, ainv->droptolerance,ainv->nonzeros,ainv->uselin,ainv->linparam);
    } else if (gpustruct->format==MAT_CUSP_DIA) {
      CUSPMATRIXDIA *mat = (CUSPMATRIXDIA*)gpustruct->mat;
      if (ainv->scaled) ainv->AINVCUSP = new cuspainvprecondscaled(*mat, ainv->droptolerance,ainv->nonzeros,ainv->uselin,ainv->linparam);
      else ainv->AINVCUSP = new cuspainvprecond(*mat, ainv->droptolerance,ainv->nonzeros,ainv->uselin,ainv->linparam);
    } else {
      CUSPMATRIX *mat = (CUSPMATRIX*)gpustruct->mat;
      if (ainv->scaled) ainv->AINVCUSP = new cuspainvprecondscaled(*mat, ainv->droptolerance,ainv->nonzeros,ainv->uselin,ainv->linparam);
      else ainv->AINVCUSP = new cuspainvprecond(*mat, ainv->droptolerance,ainv->nonzeros,ainv->uselin,ainv->linparam);
    }
#endif
  } catch(char *ex) {
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
static PetscErrorCode PCApply_AINVCUSP(PC pc,Vec x,Vec y)
{
  PC_AINVCUSP    *ainv = (PC_AINVCUSP*)pc->data;
  PetscErrorCode ierr;
  PetscBool      flg1,flg2;
  CUSPARRAY      *xarray=NULL,*yarray=NULL;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)x,VECSEQCUSP,&flg1);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)y,VECSEQCUSP,&flg2);CHKERRQ(ierr);
  if (!(flg1 && flg2)) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP, "Currently only handles CUSP vectors");
  if (!ainv->AINVCUSP) {
    ierr = PCSetUp_AINVCUSP(pc);CHKERRQ(ierr);
  }
  ierr = VecSet(y,0.0);CHKERRQ(ierr);
  ierr = VecCUSPGetArrayRead(x,&xarray);CHKERRQ(ierr);
  ierr = VecCUSPGetArrayWrite(y,&yarray);CHKERRQ(ierr);
  try {
    if (ainv->scaled) cusp::multiply(*(cuspainvprecondscaled*)ainv->AINVCUSP,*xarray,*yarray);
    else cusp::multiply(*(cuspainvprecond*)ainv->AINVCUSP,*xarray,*yarray);
  } catch(char* ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
  }
  ierr = VecCUSPRestoreArrayRead(x,&xarray);CHKERRQ(ierr);
  ierr = VecCUSPRestoreArrayWrite(y,&yarray);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */

static PetscErrorCode PCReset_AINVCUSP(PC pc)
{
  PC_AINVCUSP *ainv = (PC_AINVCUSP*)pc->data;

  PetscFunctionBegin;
  if (ainv->AINVCUSP) {
    try {
      if (ainv->scaled) delete (cuspainvprecondscaled*)ainv->AINVCUSP;
      else delete (cuspainvprecond*)ainv->AINVCUSP;
    } catch(char* ex) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"CUSP error: %s", ex);
    }
    ainv->AINVCUSP = NULL;
  }
  PetscFunctionReturn(0);
}

/*
   PCDestroy_AINVCUSP - Destroys the private context for the AINVCUSP preconditioner
   that was created with PCCreate_AINVCUSP().

   Input Parameter:
.  pc - the preconditioner context

   Application Interface Routine: PCDestroy()
*/
static PetscErrorCode PCDestroy_AINVCUSP(PC pc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCReset_AINVCUSP(pc);CHKERRQ(ierr);

  /*
      Free the private data structure that was hanging off the PC
  */
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCAINVCUSPSetDropTolerance_AINVCUSP(PC pc, PetscReal droptolerance)
{
  PC_AINVCUSP *ainv = (PC_AINVCUSP*)pc->data;

  PetscFunctionBegin;
  ainv->droptolerance = droptolerance;
  ainv->uselin        = PETSC_FALSE;
  ainv->linparam      = 1;
  ainv->nonzeros      = -1;
  PetscFunctionReturn(0);
}

PetscErrorCode PCAINVCUSPSetDropTolerance(PC pc, PetscReal droptolerance)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID,1);
  ierr = PetscTryMethod(pc, "PCAINVCUSPSetDropTolerance_C",(PC,PetscReal),(pc,droptolerance));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCAINVCUSPSetNonzeros_AINVCUSP(PC pc, PetscInt nonzeros)
{
  PC_AINVCUSP *ainv = (PC_AINVCUSP*)pc->data;

  PetscFunctionBegin;
  ainv->droptolerance = 0;
  ainv->uselin        = PETSC_FALSE;
  ainv->linparam      = 1;
  ainv->nonzeros      = nonzeros;
  PetscFunctionReturn(0);
}

PetscErrorCode PCAINVCUSPSetNonzeros(PC pc, PetscInt nonzeros)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID,1);
  ierr = PetscTryMethod(pc, "PCAINVCUSPSetNonzeros_C",(PC,PetscInt),(pc,nonzeros));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCAINVCUSPSetLinParameter_AINVCUSP(PC pc, PetscInt param)
{
  PC_AINVCUSP *ainv = (PC_AINVCUSP*)pc->data;

  PetscFunctionBegin;
  ainv->droptolerance = 0;
  ainv->uselin        = PETSC_TRUE;
  ainv->linparam      = param;
  ainv->nonzeros      = -1;
  PetscFunctionReturn(0);
}

PetscErrorCode PCAINVCUSPSetLinParameter(PC pc, PetscInt param)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID,1);
  ierr = PetscTryMethod(pc, "PCAINVCUSPSetLinParameter_C",(PC,PetscInt),(pc,param));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCAINVCUSPUseScaling_AINVCUSP(PC pc, PetscBool scaled)
{
  PC_AINVCUSP *ainv = (PC_AINVCUSP*)pc->data;

  PetscFunctionBegin;
  ainv->scaled = scaled;
  PetscFunctionReturn(0);
}

PetscErrorCode PCAINVCUSPUseScaling(PC pc, PetscBool scaled)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID,1);
  ierr = PetscTryMethod(pc, "PCAINVCUSPUseScaling_C",(PC,PetscBool),(pc,scaled));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetFromOptions_AINVCUSP(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PC_AINVCUSP    *ainv = (PC_AINVCUSP*)pc->data;
  PetscBool      flag  = PETSC_FALSE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"AINVCUSP options");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-pc_ainvcusp_droptol","drop tolerance for AINVCUSP preconditioner","PCAINVCUSPSetDropTolerance",ainv->droptolerance,&ainv->droptolerance,&flag);
  if (flag) {
    ainv->nonzeros = -1;
    ainv->uselin   = PETSC_FALSE;
    ainv->linparam = 1;
    flag           = PETSC_FALSE;
  }
  ierr = PetscOptionsInt("-pc_ainvcusp_nonzeros","nonzeros/row for AINVCUSP preconditioner","PCAINVCUSPSetNonzeros",ainv->nonzeros,&ainv->nonzeros,&flag);
  if (flag) {
    ainv->droptolerance = 0;
    ainv->uselin        = PETSC_FALSE;
    ainv->linparam      = 1;
    flag                = PETSC_FALSE;
  }
  ierr = PetscOptionsInt("-pc_ainvcusp_linparameter","Lin parameter for AINVCUSP preconditioner","PCAINVCUSPSetLinParameter",ainv->linparam,&ainv->linparam,&flag);
  if (flag) {
    ainv->droptolerance = 0;
    ainv->uselin        = PETSC_TRUE;
    ainv->droptolerance = 0;
    ainv->nonzeros      = -1;
  }
  ierr = PetscOptionsBool("-pc_ainvcusp_scale","Whether to use scaled AINVCUSP preconditioner or not","PCAINVCUSPUseScaling",ainv->scaled,&ainv->scaled,0);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */

/*MC
     PCAINVCUSP  - A sparse approximate inverse precondition that runs on the Nvidia GPU.


   http://docs.cusp-library.googlecode.com/hg/classcusp_1_1precond_1_1bridson__ainv.html

   Level: advanced

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC

M*/

PETSC_EXTERN PetscErrorCode PCCreate_AINVCUSP(PC pc)
{
  PC_AINVCUSP    *ainv;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*
     Creates the private data structure for this preconditioner and
     attach it to the PC object.
   */
  ierr                = PetscNewLog(pc,&ainv);CHKERRQ(ierr);
  pc->data            = (void*)ainv;
  ainv->AINVCUSP      = 0;
  ainv->droptolerance = 0.1;
  ainv->nonzeros      = -1;
  ainv->scaled        = PETSC_TRUE;
  ainv->linparam      = 1;
  ainv->uselin        = PETSC_FALSE;
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
  pc->ops->reset               = PCReset_AINVCUSP;
  pc->ops->destroy             = PCDestroy_AINVCUSP;
  pc->ops->setfromoptions      = PCSetFromOptions_AINVCUSP;
  pc->ops->view                = 0;
  pc->ops->applyrichardson     = 0;
  pc->ops->applysymmetricleft  = 0;
  pc->ops->applysymmetricright = 0;

  ierr = PetscObjectComposeFunction((PetscObject)pc, "PCAINVCUSPSetDropTolerance_C", PCAINVCUSPSetDropTolerance_AINVCUSP);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc, "PCAINVCUSPUseScaling_C",  PCAINVCUSPUseScaling_AINVCUSP);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc, "PCAINVCUSPSetLinParameter_C", PCAINVCUSPSetLinParameter_AINVCUSP);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc, "PCAINVCUSPSetNonzeros_C",  PCAINVCUSPSetNonzeros_AINVCUSP);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

