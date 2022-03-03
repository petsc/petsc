/*
   This provides a thin wrapper around LMVM matrices in order to use their MatLMVMSolve
   methods as preconditioner applications in KSP solves.
*/

#include <petsc/private/pcimpl.h>        /*I "petscpc.h" I*/
#include <petsc/private/matimpl.h>

typedef struct {
  Vec  xwork, ywork;
  IS   inactive;
  Mat  B;
  PetscBool allocated;
} PC_LMVM;

/*@
   PCLMVMSetMatLMVM - Replaces the LMVM matrix inside the preconditioner with
   the one provided by the user.

   Input Parameters:
+  pc - An LMVM preconditioner
-  B  - An LMVM-type matrix (MATLDFP, MATLBFGS, MATLSR1, MATLBRDN, MATLMBRDN, MATLSBRDN)

   Level: intermediate
@*/
PetscErrorCode PCLMVMSetMatLMVM(PC pc, Mat B)
{
  PC_LMVM          *ctx = (PC_LMVM*)pc->data;
  PetscBool        same;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscValidHeaderSpecific(B, MAT_CLASSID, 2);
  CHKERRQ(PetscObjectTypeCompare((PetscObject)pc, PCLMVM, &same));
  PetscCheck(same,PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONG, "PC must be a PCLMVM type.");
  CHKERRQ(PetscObjectBaseTypeCompare((PetscObject)B, MATLMVM, &same));
  PetscCheck(same,PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONG, "Matrix must be an LMVM-type.");
  CHKERRQ(MatDestroy(&ctx->B));
  CHKERRQ(PetscObjectReference((PetscObject)B));
  ctx->B = B;
  PetscFunctionReturn(0);
}

/*@
   PCLMVMGetMatLMVM - Returns a pointer to the underlying LMVM matrix.

   Input Parameters:
.  pc - An LMVM preconditioner

   Output Parameters:
.  B - LMVM matrix inside the preconditioner

   Level: intermediate
@*/
PetscErrorCode PCLMVMGetMatLMVM(PC pc, Mat *B)
{
  PC_LMVM          *ctx = (PC_LMVM*)pc->data;
  PetscBool        same;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  CHKERRQ(PetscObjectTypeCompare((PetscObject)pc, PCLMVM, &same));
  PetscCheck(same,PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONG, "PC must be a PCLMVM type.");
  *B = ctx->B;
  PetscFunctionReturn(0);
}

/*@
   PCLMVMSetIS - Sets the index sets that reduce the PC application.

   Input Parameters:
+  pc - An LMVM preconditioner
-  inactive - Index set defining the variables removed from the problem

   Level: intermediate

.seealso:  MatLMVMUpdate()
@*/
PetscErrorCode PCLMVMSetIS(PC pc, IS inactive)
{
  PC_LMVM          *ctx = (PC_LMVM*)pc->data;
  PetscBool        same;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscValidHeaderSpecific(inactive, IS_CLASSID, 2);
  CHKERRQ(PetscObjectTypeCompare((PetscObject)pc, PCLMVM, &same));
  PetscCheck(same,PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONG, "PC must be a PCLMVM type.");
  CHKERRQ(PCLMVMClearIS(pc));
  CHKERRQ(PetscObjectReference((PetscObject)inactive));
  ctx->inactive = inactive;
  PetscFunctionReturn(0);
}

/*@
   PCLMVMClearIS - Removes the inactive variable index set.

   Input Parameters:
.  pc - An LMVM preconditioner

   Level: intermediate

.seealso:  MatLMVMUpdate()
@*/
PetscErrorCode PCLMVMClearIS(PC pc)
{
  PC_LMVM          *ctx = (PC_LMVM*)pc->data;
  PetscBool        same;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  CHKERRQ(PetscObjectTypeCompare((PetscObject)pc, PCLMVM, &same));
  PetscCheck(same,PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONG, "PC must be a PCLMVM type.");
  if (ctx->inactive) {
    CHKERRQ(ISDestroy(&ctx->inactive));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApply_LMVM(PC pc,Vec x,Vec y)
{
  PC_LMVM          *ctx = (PC_LMVM*)pc->data;
  Vec              xsub, ysub;

  PetscFunctionBegin;
  if (ctx->inactive) {
    CHKERRQ(VecZeroEntries(ctx->xwork));
    CHKERRQ(VecGetSubVector(ctx->xwork, ctx->inactive, &xsub));
    CHKERRQ(VecCopy(x, xsub));
    CHKERRQ(VecRestoreSubVector(ctx->xwork, ctx->inactive, &xsub));
  } else {
    CHKERRQ(VecCopy(x, ctx->xwork));
  }
  CHKERRQ(MatSolve(ctx->B, ctx->xwork, ctx->ywork));
  if (ctx->inactive) {
    CHKERRQ(VecGetSubVector(ctx->ywork, ctx->inactive, &ysub));
    CHKERRQ(VecCopy(ysub, y));
    CHKERRQ(VecRestoreSubVector(ctx->ywork, ctx->inactive, &ysub));
  } else {
    CHKERRQ(VecCopy(ctx->ywork, y));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCReset_LMVM(PC pc)
{
  PC_LMVM        *ctx = (PC_LMVM*)pc->data;

  PetscFunctionBegin;
  if (ctx->xwork) {
    CHKERRQ(VecDestroy(&ctx->xwork));
  }
  if (ctx->ywork) {
    CHKERRQ(VecDestroy(&ctx->ywork));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetUp_LMVM(PC pc)
{
  PC_LMVM        *ctx = (PC_LMVM*)pc->data;
  PetscInt       n, N;
  PetscBool      allocated;

  PetscFunctionBegin;
  CHKERRQ(MatLMVMIsAllocated(ctx->B, &allocated));
  if (!allocated) {
    CHKERRQ(MatCreateVecs(pc->mat, &ctx->xwork, &ctx->ywork));
    CHKERRQ(VecGetLocalSize(ctx->xwork, &n));
    CHKERRQ(VecGetSize(ctx->xwork, &N));
    CHKERRQ(MatSetSizes(ctx->B, n, n, N, N));
    CHKERRQ(MatLMVMAllocate(ctx->B, ctx->xwork, ctx->ywork));
  } else {
    CHKERRQ(MatCreateVecs(ctx->B, &ctx->xwork, &ctx->ywork));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCView_LMVM(PC pc,PetscViewer viewer)
{
  PC_LMVM        *ctx = (PC_LMVM*)pc->data;
  PetscBool      iascii;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii && ctx->B->assembled) {
    CHKERRQ(PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_INFO));
    CHKERRQ(MatView(ctx->B, viewer));
    CHKERRQ(PetscViewerPopFormat(viewer));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetFromOptions_LMVM(PetscOptionItems* PetscOptionsObject, PC pc)
{
  PC_LMVM        *ctx = (PC_LMVM*)pc->data;

  PetscFunctionBegin;
  CHKERRQ(MatSetFromOptions(ctx->B));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_LMVM(PC pc)
{
  PC_LMVM        *ctx = (PC_LMVM*)pc->data;

  PetscFunctionBegin;
  if (ctx->inactive) {
    CHKERRQ(ISDestroy(&ctx->inactive));
  }
  if (pc->setupcalled) {
    CHKERRQ(VecDestroy(&ctx->xwork));
    CHKERRQ(VecDestroy(&ctx->ywork));
  }
  CHKERRQ(MatDestroy(&ctx->B));
  CHKERRQ(PetscFree(pc->data));
  PetscFunctionReturn(0);
}

/*MC
   PCLMVM - Creates a preconditioner around an LMVM matrix. Options for the
            underlying LMVM matrix can be access with the "-pc_lmvm_" prefix.

   Level: intermediate

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types),
           PC, MATLMVM, PCLMVMUpdate(), PCLMVMSetMatLMVM(), PCLMVMGetMatLMVM()
M*/
PETSC_EXTERN PetscErrorCode PCCreate_LMVM(PC pc)
{
  PC_LMVM        *ctx;

  PetscFunctionBegin;
  CHKERRQ(PetscNewLog(pc,&ctx));
  pc->data = (void*)ctx;

  pc->ops->reset           = PCReset_LMVM;
  pc->ops->setup           = PCSetUp_LMVM;
  pc->ops->destroy         = PCDestroy_LMVM;
  pc->ops->view            = PCView_LMVM;
  pc->ops->apply           = PCApply_LMVM;
  pc->ops->setfromoptions  = PCSetFromOptions_LMVM;
  pc->ops->applysymmetricleft  = NULL;
  pc->ops->applysymmetricright = NULL;
  pc->ops->applytranspose  = NULL;
  pc->ops->applyrichardson = NULL;
  pc->ops->presolve        = NULL;
  pc->ops->postsolve       = NULL;

  CHKERRQ(PCSetReusePreconditioner(pc, PETSC_TRUE));

  CHKERRQ(MatCreate(PetscObjectComm((PetscObject)pc), &ctx->B));
  CHKERRQ(MatSetType(ctx->B, MATLMVMBFGS));
  CHKERRQ(PetscObjectIncrementTabLevel((PetscObject)ctx->B, (PetscObject)pc, 1));
  CHKERRQ(MatSetOptionsPrefix(ctx->B, "pc_lmvm_"));
  PetscFunctionReturn(0);
}
