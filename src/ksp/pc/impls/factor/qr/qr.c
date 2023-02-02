
/*
   Defines a direct QR factorization preconditioner for any Mat implementation
   Note: this need not be considered a preconditioner since it supplies
         a direct solver.
*/
#include <../src/ksp/pc/impls/factor/qr/qr.h> /*I "petscpc.h" I*/

static PetscErrorCode PCSetUp_QR(PC pc)
{
  PC_QR         *dir = (PC_QR *)pc->data;
  MatSolverType  stype;
  MatFactorError err;
  const char    *prefix;

  PetscFunctionBegin;
  PetscCall(PCGetOptionsPrefix(pc, &prefix));
  PetscCall(MatSetOptionsPrefix(pc->pmat, prefix));
  pc->failedreason = PC_NOERROR;
  if (dir->hdr.reusefill && pc->setupcalled) ((PC_Factor *)dir)->info.fill = dir->hdr.actualfill;

  PetscCall(MatSetErrorIfFailure(pc->pmat, pc->erroriffailure));
  if (dir->hdr.inplace) {
    MatFactorType ftype;

    PetscCall(MatGetFactorType(pc->pmat, &ftype));
    if (ftype == MAT_FACTOR_NONE) {
      PetscCall(MatQRFactor(pc->pmat, dir->col, &((PC_Factor *)dir)->info));
      PetscCall(MatFactorGetError(pc->pmat, &err));
      if (err) { /* Factor() fails */
        pc->failedreason = (PCFailedReason)err;
        PetscFunctionReturn(PETSC_SUCCESS);
      }
    }
    ((PC_Factor *)dir)->fact = pc->pmat;
  } else {
    MatInfo info;

    if (!pc->setupcalled) {
      if (!((PC_Factor *)dir)->fact) { PetscCall(MatGetFactor(pc->pmat, ((PC_Factor *)dir)->solvertype, MAT_FACTOR_QR, &((PC_Factor *)dir)->fact)); }
      PetscCall(MatQRFactorSymbolic(((PC_Factor *)dir)->fact, pc->pmat, dir->col, &((PC_Factor *)dir)->info));
      PetscCall(MatGetInfo(((PC_Factor *)dir)->fact, MAT_LOCAL, &info));
      dir->hdr.actualfill = info.fill_ratio_needed;
    } else if (pc->flag != SAME_NONZERO_PATTERN) {
      PetscCall(MatQRFactorSymbolic(((PC_Factor *)dir)->fact, pc->pmat, dir->col, &((PC_Factor *)dir)->info));
      PetscCall(MatGetInfo(((PC_Factor *)dir)->fact, MAT_LOCAL, &info));
      dir->hdr.actualfill = info.fill_ratio_needed;
    } else {
      PetscCall(MatFactorGetError(((PC_Factor *)dir)->fact, &err));
    }
    PetscCall(MatFactorGetError(((PC_Factor *)dir)->fact, &err));
    if (err) { /* FactorSymbolic() fails */
      pc->failedreason = (PCFailedReason)err;
      PetscFunctionReturn(PETSC_SUCCESS);
    }

    PetscCall(MatQRFactorNumeric(((PC_Factor *)dir)->fact, pc->pmat, &((PC_Factor *)dir)->info));
    PetscCall(MatFactorGetError(((PC_Factor *)dir)->fact, &err));
    if (err) { /* FactorNumeric() fails */
      pc->failedreason = (PCFailedReason)err;
    }
  }

  PetscCall(PCFactorGetMatSolverType(pc, &stype));
  if (!stype) {
    MatSolverType solverpackage;
    PetscCall(MatFactorGetSolverType(((PC_Factor *)dir)->fact, &solverpackage));
    PetscCall(PCFactorSetMatSolverType(pc, solverpackage));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCReset_QR(PC pc)
{
  PC_QR *dir = (PC_QR *)pc->data;

  PetscFunctionBegin;
  if (!dir->hdr.inplace && ((PC_Factor *)dir)->fact) PetscCall(MatDestroy(&((PC_Factor *)dir)->fact));
  PetscCall(ISDestroy(&dir->col));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCDestroy_QR(PC pc)
{
  PC_QR *dir = (PC_QR *)pc->data;

  PetscFunctionBegin;
  PetscCall(PCReset_QR(pc));
  PetscCall(PetscFree(((PC_Factor *)dir)->ordering));
  PetscCall(PetscFree(((PC_Factor *)dir)->solvertype));
  PetscCall(PCFactorClearComposedFunctions(pc));
  PetscCall(PetscFree(pc->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCApply_QR(PC pc, Vec x, Vec y)
{
  PC_QR *dir = (PC_QR *)pc->data;
  Mat    fact;

  PetscFunctionBegin;
  fact = dir->hdr.inplace ? pc->pmat : ((PC_Factor *)dir)->fact;
  PetscCall(MatSolve(fact, x, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCMatApply_QR(PC pc, Mat X, Mat Y)
{
  PC_QR *dir = (PC_QR *)pc->data;
  Mat    fact;

  PetscFunctionBegin;
  fact = dir->hdr.inplace ? pc->pmat : ((PC_Factor *)dir)->fact;
  PetscCall(MatMatSolve(fact, X, Y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCApplyTranspose_QR(PC pc, Vec x, Vec y)
{
  PC_QR *dir = (PC_QR *)pc->data;
  Mat    fact;

  PetscFunctionBegin;
  fact = dir->hdr.inplace ? pc->pmat : ((PC_Factor *)dir)->fact;
  PetscCall(MatSolveTranspose(fact, x, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   PCQR - Uses a direct solver, based on QR factorization, as a preconditioner

   Level: beginner

   Note:
   Usually this will compute an "exact" solution in one iteration and does
   not need a Krylov method (i.e. you can use -ksp_type preonly, or
   `KSPSetType`(ksp,`KSPPREONLY`) for the Krylov method

.seealso: `PCCreate()`, `PCSetType()`, `PCType`, `PC`, `PCSVD`,
          `PCILU`, `PCLU`, `PCCHOLESKY`, `PCICC`, `PCFactorSetReuseOrdering()`, `PCFactorSetReuseFill()`, `PCFactorGetMatrix()`,
          `PCFactorSetFill()`, `PCFactorSetUseInPlace()`, `PCFactorSetMatOrderingType()`, `PCFactorSetColumnPivot()`,
          `PCFactorSetPivotingInBlocks()`, `PCFactorSetShiftType()`, `PCFactorSetShiftAmount()`
          `PCFactorReorderForNonzeroDiagonal()`
M*/

PETSC_EXTERN PetscErrorCode PCCreate_QR(PC pc)
{
  PC_QR *dir;

  PetscFunctionBegin;
  PetscCall(PetscNew(&dir));
  pc->data = (void *)dir;
  PetscCall(PCFactorInitialize(pc, MAT_FACTOR_QR));

  dir->col                 = NULL;
  pc->ops->reset           = PCReset_QR;
  pc->ops->destroy         = PCDestroy_QR;
  pc->ops->apply           = PCApply_QR;
  pc->ops->matapply        = PCMatApply_QR;
  pc->ops->applytranspose  = PCApplyTranspose_QR;
  pc->ops->setup           = PCSetUp_QR;
  pc->ops->setfromoptions  = PCSetFromOptions_Factor;
  pc->ops->view            = PCView_Factor;
  pc->ops->applyrichardson = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}
