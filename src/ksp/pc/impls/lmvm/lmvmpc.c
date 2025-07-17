/*
   This provides a thin wrapper around LMVM matrices in order to use their MatLMVMSolve
   methods as preconditioner applications in KSP solves.
*/

#include <petsc/private/pcimpl.h> /*I "petscpc.h" I*/
#include <petsc/private/matimpl.h>

typedef struct {
  Vec              xwork, ywork;
  IS               inactive;
  Mat              B;
  Vec              X;
  PetscObjectState Xstate;
  PetscBool        setfromoptionscalled;
} PC_LMVM;

/*@
  PCLMVMSetUpdateVec - Set the vector to be used as solution update for the internal LMVM matrix.

  Input Parameters:
+ pc - The preconditioner
- X  - Solution vector

  Level: intermediate

  Notes:
  This is only needed if you want the preconditioner to automatically update the internal matrix.
  It is called in some `SNES` implementations to update the preconditioner.
  The right-hand side of the linear system is used as function vector.

.seealso: `MatLMVMUpdate()`, `PCLMVMSetMatLMVM()`
@*/
PetscErrorCode PCLMVMSetUpdateVec(PC pc, Vec X)
{
  PC_LMVM  *ctx;
  PetscBool same;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  if (X) PetscValidHeaderSpecific(X, VEC_CLASSID, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)pc, PCLMVM, &same));
  if (!same) PetscFunctionReturn(PETSC_SUCCESS);
  ctx = (PC_LMVM *)pc->data;
  PetscCall(PetscObjectReference((PetscObject)X));
  PetscCall(VecDestroy(&ctx->X));
  ctx->X      = X;
  ctx->Xstate = -1;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCLMVMSetMatLMVM - Replaces the `MATLMVM` matrix inside the preconditioner with the one provided by the user.

  Input Parameters:
+ pc - An `PCLMVM` preconditioner
- B  - An `MATLMVM` type matrix

  Level: intermediate

.seealso: [](ch_ksp), `PCLMVMGetMatLMVM()`
@*/
PetscErrorCode PCLMVMSetMatLMVM(PC pc, Mat B)
{
  PC_LMVM  *ctx;
  PetscBool same;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscValidHeaderSpecific(B, MAT_CLASSID, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)pc, PCLMVM, &same));
  if (!same) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)B, MATLMVM, &same));
  PetscCheck(same, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONG, "Matrix must be an MATLMVM.");
  ctx = (PC_LMVM *)pc->data;
  PetscCall(PetscObjectReference((PetscObject)B));
  PetscCall(MatDestroy(&ctx->B));
  ctx->B = B;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCLMVMGetMatLMVM - Returns a pointer to the underlying `MATLMVM` matrix.

  Input Parameter:
. pc - An `PCLMVM` preconditioner

  Output Parameter:
. B - `MATLMVM` matrix used by the preconditioner

  Level: intermediate

.seealso: [](ch_ksp), `PCLMVMSetMatLMVM()`
@*/
PetscErrorCode PCLMVMGetMatLMVM(PC pc, Mat *B)
{
  PC_LMVM  *ctx;
  PetscBool same;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscCall(PetscObjectTypeCompare((PetscObject)pc, PCLMVM, &same));
  PetscCheck(same, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONG, "PC must be a PCLMVM type.");
  ctx = (PC_LMVM *)pc->data;
  if (!ctx->B) {
    Mat J;

    if (pc->useAmat) J = pc->mat;
    else J = pc->pmat;
    PetscCall(PetscObjectBaseTypeCompare((PetscObject)J, MATLMVM, &same));
    if (same) *B = J;
    else {
      const char *prefix;

      PetscCall(PCGetOptionsPrefix(pc, &prefix));
      PetscCall(MatCreate(PetscObjectComm((PetscObject)pc), &ctx->B));
      PetscCall(MatSetOptionsPrefix(ctx->B, prefix));
      PetscCall(MatAppendOptionsPrefix(ctx->B, "pc_lmvm_"));
      PetscCall(MatSetType(ctx->B, MATLMVMBFGS));
      PetscCall(PetscObjectIncrementTabLevel((PetscObject)ctx->B, (PetscObject)pc, 1));
      *B = ctx->B;
    }
  } else *B = ctx->B;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCLMVMSetIS - Sets the index sets that reduce the `PC` application.

  Input Parameters:
+ pc       - An `PCLMVM` preconditioner
- inactive - Index set defining the variables removed from the problem

  Level: intermediate

  Developer Notes:
  Need to explain the purpose of this `IS`

.seealso: [](ch_ksp), `PCLMVMClearIS()`
@*/
PetscErrorCode PCLMVMSetIS(PC pc, IS inactive)
{
  PC_LMVM  *ctx;
  PetscBool same;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscValidHeaderSpecific(inactive, IS_CLASSID, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)pc, PCLMVM, &same));
  if (!same) PetscFunctionReturn(PETSC_SUCCESS);
  ctx = (PC_LMVM *)pc->data;
  PetscCall(PCLMVMClearIS(pc));
  PetscCall(PetscObjectReference((PetscObject)inactive));
  ctx->inactive = inactive;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PCLMVMClearIS - Removes the inactive variable index set from a `PCLMVM`

  Input Parameter:
. pc - An `PCLMVM` preconditioner

  Level: intermediate

.seealso: [](ch_ksp), `PCLMVMSetIS()`
@*/
PetscErrorCode PCLMVMClearIS(PC pc)
{
  PC_LMVM  *ctx;
  PetscBool same;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscCall(PetscObjectTypeCompare((PetscObject)pc, PCLMVM, &same));
  if (!same) PetscFunctionReturn(PETSC_SUCCESS);
  ctx = (PC_LMVM *)pc->data;
  PetscCall(ISDestroy(&ctx->inactive));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCApply_LMVM(PC pc, Vec x, Vec y)
{
  PC_LMVM *ctx = (PC_LMVM *)pc->data;
  Vec      xsub, ysub, Bx = x, By = y;
  Mat      B = ctx->B ? ctx->B : (pc->useAmat ? pc->mat : pc->pmat);

  PetscFunctionBegin;
  if (ctx->inactive) {
    if (!ctx->xwork) PetscCall(MatCreateVecs(B, &ctx->xwork, &ctx->ywork));
    PetscCall(VecZeroEntries(ctx->xwork));
    PetscCall(VecGetSubVector(ctx->xwork, ctx->inactive, &xsub));
    PetscCall(VecCopy(x, xsub));
    PetscCall(VecRestoreSubVector(ctx->xwork, ctx->inactive, &xsub));
    Bx = ctx->xwork;
    By = ctx->ywork;
  }
  PetscCall(MatSolve(B, Bx, By));
  if (ctx->inactive) {
    PetscCall(VecGetSubVector(ctx->ywork, ctx->inactive, &ysub));
    PetscCall(VecCopy(ysub, y));
    PetscCall(VecRestoreSubVector(ctx->ywork, ctx->inactive, &ysub));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCReset_LMVM(PC pc)
{
  PC_LMVM *ctx = (PC_LMVM *)pc->data;

  PetscFunctionBegin;
  PetscCall(VecDestroy(&ctx->xwork));
  PetscCall(VecDestroy(&ctx->ywork));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetUp_LMVM(PC pc)
{
  PetscInt  n, N;
  PetscBool allocated;
  Mat       B;
  PC_LMVM  *ctx = (PC_LMVM *)pc->data;

  PetscFunctionBegin;
  PetscCall(PCLMVMGetMatLMVM(pc, &B));
  PetscCall(MatLMVMIsAllocated(B, &allocated));
  if (!allocated) {
    Vec t1, t2;

    PetscCall(MatCreateVecs(pc->mat, &t1, &t2));
    PetscCall(VecGetLocalSize(t1, &n));
    PetscCall(VecGetSize(t1, &N));
    PetscCall(MatSetSizes(B, n, n, N, N));
    PetscCall(MatLMVMAllocate(B, t1, t2));
    PetscCall(VecDestroy(&t1));
    PetscCall(VecDestroy(&t2));
  }
  /* Only call SetFromOptions if we internally handle the LMVM matrix */
  if (B == ctx->B && ctx->setfromoptionscalled) PetscCall(MatSetFromOptions(ctx->B));
  ctx->setfromoptionscalled = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCView_LMVM(PC pc, PetscViewer viewer)
{
  PC_LMVM  *ctx = (PC_LMVM *)pc->data;
  PetscBool isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii && ctx->B && ctx->B->assembled) {
    PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_INFO));
    PetscCall(MatView(ctx->B, viewer));
    PetscCall(PetscViewerPopFormat(viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetFromOptions_LMVM(PC pc, PetscOptionItems PetscOptionsObject)
{
  PC_LMVM *ctx = (PC_LMVM *)pc->data;

  PetscFunctionBegin;
  /* defer SetFromOptions calls to PCSetUp_LMVM */
  ctx->setfromoptionscalled = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCPreSolve_LMVM(PC pc, KSP ksp, Vec F, Vec X)
{
  PC_LMVM *ctx = (PC_LMVM *)pc->data;

  PetscFunctionBegin;
  if (ctx->X && ctx->B) { /* Perform update only if requested. Otherwise we assume the user, e.g. TAO, has already taken care of it */
    PetscObjectState Xstate;

    PetscCall(PetscObjectStateGet((PetscObject)ctx->X, &Xstate));
    if (ctx->Xstate != Xstate) PetscCall(MatLMVMUpdate(ctx->B, ctx->X, F));
    ctx->Xstate = Xstate;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCDestroy_LMVM(PC pc)
{
  PC_LMVM *ctx = (PC_LMVM *)pc->data;

  PetscFunctionBegin;
  PetscCall(ISDestroy(&ctx->inactive));
  PetscCall(VecDestroy(&ctx->xwork));
  PetscCall(VecDestroy(&ctx->ywork));
  PetscCall(VecDestroy(&ctx->X));
  PetscCall(MatDestroy(&ctx->B));
  PetscCall(PetscFree(pc->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   PCLMVM - A preconditioner constructed from a `MATLMVM` matrix.
            If the matrix used to construct the preconditioner is not of type `MATLMVM`, an internal matrix is used.
            Options for the internal `MATLMVM` matrix can be accessed with the `-pc_lmvm_` prefix.
            Alternatively, the user can pass a suitable matrix with `PCLMVMSetMatLMVM()`.

   Level: intermediate

.seealso: [](ch_ksp), `PCCreate()`, `PCSetType()`, `PCLMVMSetUpdateVec()`, `PCLMVMSetMatLMVM()`, `PCLMVMGetMatLMVM()`
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
  pc->ops->presolve            = PCPreSolve_LMVM;
  pc->ops->postsolve           = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}
