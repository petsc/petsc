
/*
   Defines a direct factorization preconditioner for any Mat implementation
   Note: this need not be consided a preconditioner since it supplies
         a direct solver.
*/
#include <../src/ksp/pc/impls/factor/factor.h> /*I "petscpc.h" I*/

typedef struct {
  PC_Factor hdr;
  IS        row, col; /* index sets used for reordering */
} PC_Cholesky;

static PetscErrorCode PCSetFromOptions_Cholesky(PC pc, PetscOptionItems *PetscOptionsObject)
{
  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "Cholesky options");
  PetscCall(PCSetFromOptions_Factor(pc, PetscOptionsObject));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetUp_Cholesky(PC pc)
{
  PetscBool      flg;
  PC_Cholesky   *dir = (PC_Cholesky *)pc->data;
  MatSolverType  stype;
  MatFactorError err;
  const char    *prefix;

  PetscFunctionBegin;
  pc->failedreason = PC_NOERROR;
  if (dir->hdr.reusefill && pc->setupcalled) ((PC_Factor *)dir)->info.fill = dir->hdr.actualfill;

  PetscCall(PCGetOptionsPrefix(pc, &prefix));
  PetscCall(MatSetOptionsPrefixFactor(pc->pmat, prefix));

  PetscCall(MatSetErrorIfFailure(pc->pmat, pc->erroriffailure));
  if (dir->hdr.inplace) {
    if (dir->row && dir->col && (dir->row != dir->col)) PetscCall(ISDestroy(&dir->row));
    PetscCall(ISDestroy(&dir->col));
    /* should only get reordering if the factor matrix uses it but cannot determine because MatGetFactor() not called */
    PetscCall(PCFactorSetDefaultOrdering_Factor(pc));
    PetscCall(MatGetOrdering(pc->pmat, ((PC_Factor *)dir)->ordering, &dir->row, &dir->col));
    if (dir->col && (dir->row != dir->col)) { /* only use row ordering for SBAIJ */
      PetscCall(ISDestroy(&dir->col));
    }
    PetscCall(MatCholeskyFactor(pc->pmat, dir->row, &((PC_Factor *)dir)->info));
    PetscCall(MatFactorGetError(pc->pmat, &err));
    if (err) { /* Factor() fails */
      pc->failedreason = (PCFailedReason)err;
      PetscFunctionReturn(0);
    }

    ((PC_Factor *)dir)->fact = pc->pmat;
  } else {
    MatInfo info;

    if (!pc->setupcalled) {
      PetscBool canuseordering;
      if (!((PC_Factor *)dir)->fact) { PetscCall(MatGetFactor(pc->pmat, ((PC_Factor *)dir)->solvertype, MAT_FACTOR_CHOLESKY, &((PC_Factor *)dir)->fact)); }
      PetscCall(MatFactorGetCanUseOrdering(((PC_Factor *)dir)->fact, &canuseordering));
      if (canuseordering) {
        PetscCall(PCFactorSetDefaultOrdering_Factor(pc));
        PetscCall(MatGetOrdering(pc->pmat, ((PC_Factor *)dir)->ordering, &dir->row, &dir->col));
        /* check if dir->row == dir->col */
        if (dir->row) {
          PetscCall(ISEqual(dir->row, dir->col, &flg));
          PetscCheck(flg, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "row and column permutations must be equal");
        }
        PetscCall(ISDestroy(&dir->col)); /* only pass one ordering into CholeskyFactor */

        flg = PETSC_FALSE;
        PetscCall(PetscOptionsGetBool(((PetscObject)pc)->options, ((PetscObject)pc)->prefix, "-pc_factor_nonzeros_along_diagonal", &flg, NULL));
        if (flg) {
          PetscReal tol = 1.e-10;
          PetscCall(PetscOptionsGetReal(((PetscObject)pc)->options, ((PetscObject)pc)->prefix, "-pc_factor_nonzeros_along_diagonal", &tol, NULL));
          PetscCall(MatReorderForNonzeroDiagonal(pc->pmat, tol, dir->row, dir->row));
        }
      }
      PetscCall(MatCholeskyFactorSymbolic(((PC_Factor *)dir)->fact, pc->pmat, dir->row, &((PC_Factor *)dir)->info));
      PetscCall(MatGetInfo(((PC_Factor *)dir)->fact, MAT_LOCAL, &info));
      dir->hdr.actualfill = info.fill_ratio_needed;
    } else if (pc->flag != SAME_NONZERO_PATTERN) {
      if (!dir->hdr.reuseordering) {
        PetscBool canuseordering;
        PetscCall(MatDestroy(&((PC_Factor *)dir)->fact));
        PetscCall(MatGetFactor(pc->pmat, ((PC_Factor *)dir)->solvertype, MAT_FACTOR_CHOLESKY, &((PC_Factor *)dir)->fact));
        PetscCall(MatFactorGetCanUseOrdering(((PC_Factor *)dir)->fact, &canuseordering));
        if (canuseordering) {
          PetscCall(ISDestroy(&dir->row));
          PetscCall(PCFactorSetDefaultOrdering_Factor(pc));
          PetscCall(MatGetOrdering(pc->pmat, ((PC_Factor *)dir)->ordering, &dir->row, &dir->col));
          PetscCall(ISDestroy(&dir->col)); /* only use dir->row ordering in CholeskyFactor */

          flg = PETSC_FALSE;
          PetscCall(PetscOptionsGetBool(((PetscObject)pc)->options, ((PetscObject)pc)->prefix, "-pc_factor_nonzeros_along_diagonal", &flg, NULL));
          if (flg) {
            PetscReal tol = 1.e-10;
            PetscCall(PetscOptionsGetReal(((PetscObject)pc)->options, ((PetscObject)pc)->prefix, "-pc_factor_nonzeros_along_diagonal", &tol, NULL));
            PetscCall(MatReorderForNonzeroDiagonal(pc->pmat, tol, dir->row, dir->row));
          }
        }
      }
      PetscCall(MatCholeskyFactorSymbolic(((PC_Factor *)dir)->fact, pc->pmat, dir->row, &((PC_Factor *)dir)->info));
      PetscCall(MatGetInfo(((PC_Factor *)dir)->fact, MAT_LOCAL, &info));
      dir->hdr.actualfill = info.fill_ratio_needed;
    } else {
      PetscCall(MatFactorGetError(((PC_Factor *)dir)->fact, &err));
      if (err == MAT_FACTOR_NUMERIC_ZEROPIVOT) {
        PetscCall(MatFactorClearError(((PC_Factor *)dir)->fact));
        pc->failedreason = PC_NOERROR;
      }
    }
    PetscCall(MatFactorGetError(((PC_Factor *)dir)->fact, &err));
    if (err) { /* FactorSymbolic() fails */
      pc->failedreason = (PCFailedReason)err;
      PetscFunctionReturn(0);
    }

    PetscCall(MatCholeskyFactorNumeric(((PC_Factor *)dir)->fact, pc->pmat, &((PC_Factor *)dir)->info));
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
  PetscFunctionReturn(0);
}

static PetscErrorCode PCReset_Cholesky(PC pc)
{
  PC_Cholesky *dir = (PC_Cholesky *)pc->data;

  PetscFunctionBegin;
  if (!dir->hdr.inplace && ((PC_Factor *)dir)->fact) PetscCall(MatDestroy(&((PC_Factor *)dir)->fact));
  PetscCall(ISDestroy(&dir->row));
  PetscCall(ISDestroy(&dir->col));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_Cholesky(PC pc)
{
  PC_Cholesky *dir = (PC_Cholesky *)pc->data;

  PetscFunctionBegin;
  PetscCall(PCReset_Cholesky(pc));
  PetscCall(PetscFree(((PC_Factor *)dir)->ordering));
  PetscCall(PetscFree(((PC_Factor *)dir)->solvertype));
  PetscCall(PCFactorClearComposedFunctions(pc));
  PetscCall(PetscFree(pc->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApply_Cholesky(PC pc, Vec x, Vec y)
{
  PC_Cholesky *dir = (PC_Cholesky *)pc->data;

  PetscFunctionBegin;
  if (dir->hdr.inplace) {
    PetscCall(MatSolve(pc->pmat, x, y));
  } else {
    PetscCall(MatSolve(((PC_Factor *)dir)->fact, x, y));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCMatApply_Cholesky(PC pc, Mat X, Mat Y)
{
  PC_Cholesky *dir = (PC_Cholesky *)pc->data;

  PetscFunctionBegin;
  if (dir->hdr.inplace) {
    PetscCall(MatMatSolve(pc->pmat, X, Y));
  } else {
    PetscCall(MatMatSolve(((PC_Factor *)dir)->fact, X, Y));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplySymmetricLeft_Cholesky(PC pc, Vec x, Vec y)
{
  PC_Cholesky *dir = (PC_Cholesky *)pc->data;

  PetscFunctionBegin;
  if (dir->hdr.inplace) {
    PetscCall(MatForwardSolve(pc->pmat, x, y));
  } else {
    PetscCall(MatForwardSolve(((PC_Factor *)dir)->fact, x, y));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplySymmetricRight_Cholesky(PC pc, Vec x, Vec y)
{
  PC_Cholesky *dir = (PC_Cholesky *)pc->data;

  PetscFunctionBegin;
  if (dir->hdr.inplace) {
    PetscCall(MatBackwardSolve(pc->pmat, x, y));
  } else {
    PetscCall(MatBackwardSolve(((PC_Factor *)dir)->fact, x, y));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplyTranspose_Cholesky(PC pc, Vec x, Vec y)
{
  PC_Cholesky *dir = (PC_Cholesky *)pc->data;

  PetscFunctionBegin;
  if (dir->hdr.inplace) {
    PetscCall(MatSolveTranspose(pc->pmat, x, y));
  } else {
    PetscCall(MatSolveTranspose(((PC_Factor *)dir)->fact, x, y));
  }
  PetscFunctionReturn(0);
}

/*@
   PCFactorSetReuseOrdering - When similar matrices are factored, this
   causes the ordering computed in the first factor to be used for all
   following factors.

   Logically Collective on pc

   Input Parameters:
+  pc - the preconditioner context
-  flag - `PETSC_TRUE` to reuse else `PETSC_FALSE`

   Options Database Key:
.  -pc_factor_reuse_ordering - Activate `PCFactorSetReuseOrdering()`

   Level: intermediate

.seealso: `PCLU`, `PCCHOLESKY`, `PCFactorSetReuseFill()`
@*/
PetscErrorCode PCFactorSetReuseOrdering(PC pc, PetscBool flag)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscValidLogicalCollectiveBool(pc, flag, 2);
  PetscTryMethod(pc, "PCFactorSetReuseOrdering_C", (PC, PetscBool), (pc, flag));
  PetscFunctionReturn(0);
}

/*MC
   PCCHOLESKY - Uses a direct solver, based on Cholesky factorization, as a preconditioner

   Options Database Keys:
+  -pc_factor_reuse_ordering - Activate `PCFactorSetReuseOrdering()`
.  -pc_factor_mat_solver_type - Actives `PCFactorSetMatSolverType()` to choose the direct solver, like superlu
.  -pc_factor_reuse_fill - Activates `PCFactorSetReuseFill()`
.  -pc_factor_fill <fill> - Sets fill amount
.  -pc_factor_in_place - Activates in-place factorization
-  -pc_factor_mat_ordering_type <nd,rcm,...> - Sets ordering routine

   Level: beginner

   Notes:
   Not all options work for all matrix formats

   Usually this will compute an "exact" solution in one iteration and does
   not need a Krylov method (i.e. you can use -ksp_type preonly, or
   `KSPSetType`(ksp,`KSPPREONLY`) for the Krylov method

.seealso: `PCCreate()`, `PCSetType()`, `PCType`, `PC`,
          `PCILU`, `PCLU`, `PCICC`, `PCFactorSetReuseOrdering()`, `PCFactorSetReuseFill()`, `PCFactorGetMatrix()`,
          `PCFactorSetFill()`, `PCFactorSetShiftNonzero()`, `PCFactorSetShiftType()`, `PCFactorSetShiftAmount()`
          `PCFactorSetUseInPlace()`, `PCFactorGetUseInPlace()`, `PCFactorSetMatOrderingType()`, `PCFactorSetReuseOrdering()`
M*/

PETSC_EXTERN PetscErrorCode PCCreate_Cholesky(PC pc)
{
  PC_Cholesky *dir;

  PetscFunctionBegin;
  PetscCall(PetscNew(&dir));
  pc->data = (void *)dir;
  PetscCall(PCFactorInitialize(pc, MAT_FACTOR_CHOLESKY));

  ((PC_Factor *)dir)->info.fill = 5.0;

  pc->ops->destroy             = PCDestroy_Cholesky;
  pc->ops->reset               = PCReset_Cholesky;
  pc->ops->apply               = PCApply_Cholesky;
  pc->ops->matapply            = PCMatApply_Cholesky;
  pc->ops->applysymmetricleft  = PCApplySymmetricLeft_Cholesky;
  pc->ops->applysymmetricright = PCApplySymmetricRight_Cholesky;
  pc->ops->applytranspose      = PCApplyTranspose_Cholesky;
  pc->ops->setup               = PCSetUp_Cholesky;
  pc->ops->setfromoptions      = PCSetFromOptions_Cholesky;
  pc->ops->view                = PCView_Factor;
  pc->ops->applyrichardson     = NULL;
  PetscFunctionReturn(0);
}
