/*
   This provides a thin wrapper around LMVM matrices in order to use their MatLMVMSolve
   methods as preconditioner applications in KSP solves.
*/

#include <petsc/private/pcimpl.h> /*I "petscpc.h" I*/
#include <petsc/private/matimpl.h>

typedef struct {
  Vec       xwork, ywork;
  IS        inactive;
  Mat       B;
  PetscBool allocated;
} PC_LMVM;

/*@
   PCLMVMSetMatLMVM - Replaces the `MATLMVM` matrix inside the preconditioner with
   the one provided by the user.

   Input Parameters:
+  pc - An `PCLMVM` preconditioner
-  B  - An  LMVM-type matrix (`MATLMVM`, `MATLDFP`, `MATLBFGS`, `MATLSR1`, `MATLBRDN`, `MATLMBRDN`, `MATLSBRDN`)

   Level: intermediate

.seealso: `PCLMVM`, `MATLDFP`, `MATLBFGS`, `MATLSR1`, `MATLBRDN`, `MATLMBRDN`, `MATLSBRDN`, `PCLMVMGetMatLMVM()`
@*/
PetscErrorCode PCLMVMSetMatLMVM(PC pc, Mat B)
{
  PC_LMVM  *ctx = (PC_LMVM *)pc->data;
  PetscBool same;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscValidHeaderSpecific(B, MAT_CLASSID, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)pc, PCLMVM, &same));
  PetscCheck(same, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONG, "PC must be a PCLMVM type.");
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)B, MATLMVM, &same));
  PetscCheck(same, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONG, "Matrix must be an LMVM-type.");
  PetscCall(MatDestroy(&ctx->B));
  PetscCall(PetscObjectReference((PetscObject)B));
  ctx->B = B;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   PCLMVMGetMatLMVM - Returns a pointer to the underlying `MATLMVM` matrix.

   Input Parameter:
.  pc - An `PCLMVM` preconditioner

   Output Parameter:
.  B - matrix inside the preconditioner, one of type `MATLMVM`, `MATLDFP`, `MATLBFGS`, `MATLSR1`, `MATLBRDN`, `MATLMBRDN`, `MATLSBRDN`

   Level: intermediate

.seealso: `PCLMVM`, `MATLMVM`, `MATLDFP`, `MATLBFGS`, `MATLSR1`, `MATLBRDN`, `MATLMBRDN`, `MATLSBRDN`, `PCLMVMSetMatLMVM()`
@*/
PetscErrorCode PCLMVMGetMatLMVM(PC pc, Mat *B)
{
  PC_LMVM  *ctx = (PC_LMVM *)pc->data;
  PetscBool same;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscCall(PetscObjectTypeCompare((PetscObject)pc, PCLMVM, &same));
  PetscCheck(same, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONG, "PC must be a PCLMVM type.");
  *B = ctx->B;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   PCLMVMSetIS - Sets the index sets that reduce the `PC` application.

   Input Parameters:
+  pc - An `PCLMVM` preconditioner
-  inactive - Index set defining the variables removed from the problem

   Level: intermediate

   Developer Note:
   Need to explain the purpose of this `IS`

.seealso: `PCLMVM`, `MatLMVMUpdate()`
@*/
PetscErrorCode PCLMVMSetIS(PC pc, IS inactive)
{
  PC_LMVM  *ctx = (PC_LMVM *)pc->data;
  PetscBool same;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscValidHeaderSpecific(inactive, IS_CLASSID, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)pc, PCLMVM, &same));
  PetscCheck(same, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONG, "PC must be a PCLMVM type.");
  PetscCall(PCLMVMClearIS(pc));
  PetscCall(PetscObjectReference((PetscObject)inactive));
  ctx->inactive = inactive;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   PCLMVMClearIS - Removes the inactive variable index set from a `PCLMVM`

   Input Parameter:
.  pc - An `PCLMVM` preconditioner

   Level: intermediate

.seealso: `PCLMVM`, `MatLMVMUpdate()`
@*/
PetscErrorCode PCLMVMClearIS(PC pc)
{
  PC_LMVM  *ctx = (PC_LMVM *)pc->data;
  PetscBool same;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscCall(PetscObjectTypeCompare((PetscObject)pc, PCLMVM, &same));
  PetscCheck(same, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONG, "PC must be a PCLMVM type.");
  if (ctx->inactive) PetscCall(ISDestroy(&ctx->inactive));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCApply_LMVM(PC pc, Vec x, Vec y)
{
  PC_LMVM *ctx = (PC_LMVM *)pc->data;
  Vec      xsub, ysub;

  PetscFunctionBegin;
  if (ctx->inactive) {
    PetscCall(VecZeroEntries(ctx->xwork));
    PetscCall(VecGetSubVector(ctx->xwork, ctx->inactive, &xsub));
    PetscCall(VecCopy(x, xsub));
    PetscCall(VecRestoreSubVector(ctx->xwork, ctx->inactive, &xsub));
  } else {
    PetscCall(VecCopy(x, ctx->xwork));
  }
  PetscCall(MatSolve(ctx->B, ctx->xwork, ctx->ywork));
  if (ctx->inactive) {
    PetscCall(VecGetSubVector(ctx->ywork, ctx->inactive, &ysub));
    PetscCall(VecCopy(ysub, y));
    PetscCall(VecRestoreSubVector(ctx->ywork, ctx->inactive, &ysub));
  } else {
    PetscCall(VecCopy(ctx->ywork, y));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCReset_LMVM(PC pc)
{
  PC_LMVM *ctx = (PC_LMVM *)pc->data;

  PetscFunctionBegin;
  if (ctx->xwork) PetscCall(VecDestroy(&ctx->xwork));
  if (ctx->ywork) PetscCall(VecDestroy(&ctx->ywork));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetUp_LMVM(PC pc)
{
  PC_LMVM    *ctx = (PC_LMVM *)pc->data;
  PetscInt    n, N;
  PetscBool   allocated;
  const char *prefix;

  PetscFunctionBegin;
  if (pc->setupcalled) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(MatLMVMIsAllocated(ctx->B, &allocated));
  if (!allocated) {
    PetscCall(MatCreateVecs(pc->mat, &ctx->xwork, &ctx->ywork));
    PetscCall(VecGetLocalSize(ctx->xwork, &n));
    PetscCall(VecGetSize(ctx->xwork, &N));
    PetscCall(MatSetSizes(ctx->B, n, n, N, N));
    PetscCall(MatLMVMAllocate(ctx->B, ctx->xwork, ctx->ywork));
  } else {
    PetscCall(MatCreateVecs(ctx->B, &ctx->xwork, &ctx->ywork));
  }
  PetscCall(PCGetOptionsPrefix(pc, &prefix));
  PetscCall(MatSetOptionsPrefix(ctx->B, prefix));
  PetscCall(MatAppendOptionsPrefix(ctx->B, "pc_lmvm_"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCView_LMVM(PC pc, PetscViewer viewer)
{
  PC_LMVM  *ctx = (PC_LMVM *)pc->data;
  PetscBool iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii && ctx->B->assembled) {
    PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_INFO));
    PetscCall(MatView(ctx->B, viewer));
    PetscCall(PetscViewerPopFormat(viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetFromOptions_LMVM(PC pc, PetscOptionItems *PetscOptionsObject)
{
  PC_LMVM    *ctx = (PC_LMVM *)pc->data;
  const char *prefix;

  PetscFunctionBegin;
  PetscCall(PCGetOptionsPrefix(pc, &prefix));
  PetscCall(MatSetOptionsPrefix(ctx->B, prefix));
  PetscCall(MatAppendOptionsPrefix(ctx->B, "pc_lmvm_"));
  PetscCall(MatSetFromOptions(ctx->B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCDestroy_LMVM(PC pc)
{
  PC_LMVM *ctx = (PC_LMVM *)pc->data;

  PetscFunctionBegin;
  if (ctx->inactive) PetscCall(ISDestroy(&ctx->inactive));
  if (pc->setupcalled) {
    PetscCall(VecDestroy(&ctx->xwork));
    PetscCall(VecDestroy(&ctx->ywork));
  }
  PetscCall(MatDestroy(&ctx->B));
  PetscCall(PetscFree(pc->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   PCLMVM - A a preconditioner constructed from a `MATLMVM` matrix. Options for the
            underlying `MATLMVM` matrix can be access with the -pc_lmvm_ prefix.

   Level: intermediate

.seealso: `PCCreate()`, `PCSetType()`, `PCType`, `PCLMVM`, `MATLDFP`, `MATLBFGS`, `MATLSR1`, `MATLBRDN`, `MATLMBRDN`, `MATLSBRDN`,
          `PC`, `MATLMVM`, `PCLMVMUpdate()`, `PCLMVMSetMatLMVM()`, `PCLMVMGetMatLMVM()`
M*/
PETSC_EXTERN PetscErrorCode PCCreate_LMVM(PC pc)
{
  PC_LMVM *ctx;

  PetscFunctionBegin;
  PetscCall(PetscNew(&ctx));
  pc->data = (void *)ctx;

  pc->ops->reset               = PCReset_LMVM;
  pc->ops->setup               = PCSetUp_LMVM;
  pc->ops->destroy             = PCDestroy_LMVM;
  pc->ops->view                = PCView_LMVM;
  pc->ops->apply               = PCApply_LMVM;
  pc->ops->setfromoptions      = PCSetFromOptions_LMVM;
  pc->ops->applysymmetricleft  = NULL;
  pc->ops->applysymmetricright = NULL;
  pc->ops->applytranspose      = NULL;
  pc->ops->applyrichardson     = NULL;
  pc->ops->presolve            = NULL;
  pc->ops->postsolve           = NULL;

  PetscCall(MatCreate(PetscObjectComm((PetscObject)pc), &ctx->B));
  PetscCall(MatSetType(ctx->B, MATLMVMBFGS));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)ctx->B, (PetscObject)pc, 1));
  PetscFunctionReturn(PETSC_SUCCESS);
}
