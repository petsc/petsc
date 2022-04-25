
/*
   Defines a direct factorization preconditioner for any Mat implementation
   Note: this need not be consided a preconditioner since it supplies
         a direct solver.
*/

#include <../src/ksp/pc/impls/factor/lu/lu.h>  /*I "petscpc.h" I*/

PetscErrorCode PCFactorReorderForNonzeroDiagonal_LU(PC pc,PetscReal z)
{
  PC_LU *lu = (PC_LU*)pc->data;

  PetscFunctionBegin;
  lu->nonzerosalongdiagonal = PETSC_TRUE;
  if (z == PETSC_DECIDE) lu->nonzerosalongdiagonaltol = 1.e-10;
  else lu->nonzerosalongdiagonaltol = z;
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetFromOptions_LU(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PC_LU          *lu = (PC_LU*)pc->data;
  PetscBool      flg = PETSC_FALSE;
  PetscReal      tol;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject,"LU options");
  PetscCall(PCSetFromOptions_Factor(PetscOptionsObject,pc));

  PetscCall(PetscOptionsName("-pc_factor_nonzeros_along_diagonal","Reorder to remove zeros from diagonal","PCFactorReorderForNonzeroDiagonal",&flg));
  if (flg) {
    tol  = PETSC_DECIDE;
    PetscCall(PetscOptionsReal("-pc_factor_nonzeros_along_diagonal","Reorder to remove zeros from diagonal","PCFactorReorderForNonzeroDiagonal",lu->nonzerosalongdiagonaltol,&tol,NULL));
    PetscCall(PCFactorReorderForNonzeroDiagonal(pc,tol));
  }
  PetscOptionsHeadEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetUp_LU(PC pc)
{
  PC_LU                  *dir = (PC_LU*)pc->data;
  MatSolverType          stype;
  MatFactorError         err;

  PetscFunctionBegin;
  pc->failedreason = PC_NOERROR;
  if (dir->hdr.reusefill && pc->setupcalled) ((PC_Factor*)dir)->info.fill = dir->hdr.actualfill;

  PetscCall(MatSetErrorIfFailure(pc->pmat,pc->erroriffailure));
  if (dir->hdr.inplace) {
    MatFactorType ftype;

    PetscCall(MatGetFactorType(pc->pmat, &ftype));
    if (ftype == MAT_FACTOR_NONE) {
      if (dir->row && dir->col && dir->row != dir->col) PetscCall(ISDestroy(&dir->row));
      PetscCall(ISDestroy(&dir->col));
      /* This should only get the ordering if needed, but since MatGetFactor() is not called we can't know if it is needed */
      PetscCall(PCFactorSetDefaultOrdering_Factor(pc));
      PetscCall(MatGetOrdering(pc->pmat,((PC_Factor*)dir)->ordering,&dir->row,&dir->col));
      if (dir->row) {
        PetscCall(PetscLogObjectParent((PetscObject)pc,(PetscObject)dir->row));
        PetscCall(PetscLogObjectParent((PetscObject)pc,(PetscObject)dir->col));
      }
      PetscCall(MatLUFactor(pc->pmat,dir->row,dir->col,&((PC_Factor*)dir)->info));
      PetscCall(MatFactorGetError(pc->pmat,&err));
      if (err) { /* Factor() fails */
        pc->failedreason = (PCFailedReason)err;
        PetscFunctionReturn(0);
      }
    }
    ((PC_Factor*)dir)->fact = pc->pmat;
  } else {
    MatInfo info;

    if (!pc->setupcalled) {
      PetscBool canuseordering;
      if (!((PC_Factor*)dir)->fact) {
        PetscCall(MatGetFactor(pc->pmat,((PC_Factor*)dir)->solvertype,MAT_FACTOR_LU,&((PC_Factor*)dir)->fact));
        PetscCall(PetscLogObjectParent((PetscObject)pc,(PetscObject)((PC_Factor*)dir)->fact));
      }
      PetscCall(MatFactorGetCanUseOrdering(((PC_Factor*)dir)->fact,&canuseordering));
      if (canuseordering) {
        PetscCall(PCFactorSetDefaultOrdering_Factor(pc));
        PetscCall(MatGetOrdering(pc->pmat,((PC_Factor*)dir)->ordering,&dir->row,&dir->col));
        if (dir->nonzerosalongdiagonal) {
          PetscCall(MatReorderForNonzeroDiagonal(pc->pmat,dir->nonzerosalongdiagonaltol,dir->row,dir->col));
        }
        PetscCall(PetscLogObjectParent((PetscObject)pc,(PetscObject)dir->row));
        PetscCall(PetscLogObjectParent((PetscObject)pc,(PetscObject)dir->col));
      }
      PetscCall(MatLUFactorSymbolic(((PC_Factor*)dir)->fact,pc->pmat,dir->row,dir->col,&((PC_Factor*)dir)->info));
      PetscCall(MatGetInfo(((PC_Factor*)dir)->fact,MAT_LOCAL,&info));
      dir->hdr.actualfill = info.fill_ratio_needed;
    } else if (pc->flag != SAME_NONZERO_PATTERN) {
      PetscBool canuseordering;
      if (!dir->hdr.reuseordering) {
        PetscCall(MatDestroy(&((PC_Factor*)dir)->fact));
        PetscCall(MatGetFactor(pc->pmat,((PC_Factor*)dir)->solvertype,MAT_FACTOR_LU,&((PC_Factor*)dir)->fact));
        PetscCall(PetscLogObjectParent((PetscObject)pc,(PetscObject)((PC_Factor*)dir)->fact));
        PetscCall(MatFactorGetCanUseOrdering(((PC_Factor*)dir)->fact,&canuseordering));
        if (canuseordering) {
          if (dir->row && dir->col && dir->row != dir->col) PetscCall(ISDestroy(&dir->row));
          PetscCall(ISDestroy(&dir->col));
          PetscCall(PCFactorSetDefaultOrdering_Factor(pc));
          PetscCall(MatGetOrdering(pc->pmat,((PC_Factor*)dir)->ordering,&dir->row,&dir->col));
          if (dir->nonzerosalongdiagonal) {
            PetscCall(MatReorderForNonzeroDiagonal(pc->pmat,dir->nonzerosalongdiagonaltol,dir->row,dir->col));
          }
          PetscCall(PetscLogObjectParent((PetscObject)pc,(PetscObject)dir->row));
          PetscCall(PetscLogObjectParent((PetscObject)pc,(PetscObject)dir->col));
        }
      }
      PetscCall(MatLUFactorSymbolic(((PC_Factor*)dir)->fact,pc->pmat,dir->row,dir->col,&((PC_Factor*)dir)->info));
      PetscCall(MatGetInfo(((PC_Factor*)dir)->fact,MAT_LOCAL,&info));
      dir->hdr.actualfill = info.fill_ratio_needed;
    } else {
      PetscCall(MatFactorGetError(((PC_Factor*)dir)->fact,&err));
      if (err == MAT_FACTOR_NUMERIC_ZEROPIVOT) {
        PetscCall(MatFactorClearError(((PC_Factor*)dir)->fact));
        pc->failedreason = PC_NOERROR;
      }
    }
    PetscCall(MatFactorGetError(((PC_Factor*)dir)->fact,&err));
    if (err) { /* FactorSymbolic() fails */
      pc->failedreason = (PCFailedReason)err;
      PetscFunctionReturn(0);
    }

    PetscCall(MatLUFactorNumeric(((PC_Factor*)dir)->fact,pc->pmat,&((PC_Factor*)dir)->info));
    PetscCall(MatFactorGetError(((PC_Factor*)dir)->fact,&err));
    if (err) { /* FactorNumeric() fails */
      pc->failedreason = (PCFailedReason)err;
    }

  }

  PetscCall(PCFactorGetMatSolverType(pc,&stype));
  if (!stype) {
    MatSolverType solverpackage;
    PetscCall(MatFactorGetSolverType(((PC_Factor*)dir)->fact,&solverpackage));
    PetscCall(PCFactorSetMatSolverType(pc,solverpackage));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCReset_LU(PC pc)
{
  PC_LU          *dir = (PC_LU*)pc->data;

  PetscFunctionBegin;
  if (!dir->hdr.inplace && ((PC_Factor*)dir)->fact) PetscCall(MatDestroy(&((PC_Factor*)dir)->fact));
  if (dir->row && dir->col && dir->row != dir->col) PetscCall(ISDestroy(&dir->row));
  PetscCall(ISDestroy(&dir->col));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_LU(PC pc)
{
  PC_LU          *dir = (PC_LU*)pc->data;

  PetscFunctionBegin;
  PetscCall(PCReset_LU(pc));
  PetscCall(PetscFree(((PC_Factor*)dir)->ordering));
  PetscCall(PetscFree(((PC_Factor*)dir)->solvertype));
  PetscCall(PetscFree(pc->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApply_LU(PC pc,Vec x,Vec y)
{
  PC_LU          *dir = (PC_LU*)pc->data;

  PetscFunctionBegin;
  if (dir->hdr.inplace) {
    PetscCall(MatSolve(pc->pmat,x,y));
  } else {
    PetscCall(MatSolve(((PC_Factor*)dir)->fact,x,y));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCMatApply_LU(PC pc,Mat X,Mat Y)
{
  PC_LU          *dir = (PC_LU*)pc->data;

  PetscFunctionBegin;
  if (dir->hdr.inplace) {
    PetscCall(MatMatSolve(pc->pmat,X,Y));
  } else {
    PetscCall(MatMatSolve(((PC_Factor*)dir)->fact,X,Y));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplyTranspose_LU(PC pc,Vec x,Vec y)
{
  PC_LU          *dir = (PC_LU*)pc->data;

  PetscFunctionBegin;
  if (dir->hdr.inplace) {
    PetscCall(MatSolveTranspose(pc->pmat,x,y));
  } else {
    PetscCall(MatSolveTranspose(((PC_Factor*)dir)->fact,x,y));
  }
  PetscFunctionReturn(0);
}

/* -----------------------------------------------------------------------------------*/

/*MC
   PCLU - Uses a direct solver, based on LU factorization, as a preconditioner

   Options Database Keys:
+  -pc_factor_reuse_ordering - Activate PCFactorSetReuseOrdering()
.  -pc_factor_mat_solver_type - Actives PCFactorSetMatSolverType() to choose the direct solver, like superlu
.  -pc_factor_reuse_fill - Activates PCFactorSetReuseFill()
.  -pc_factor_fill <fill> - Sets fill amount
.  -pc_factor_in_place - Activates in-place factorization
.  -pc_factor_mat_ordering_type <nd,rcm,...> - Sets ordering routine
.  -pc_factor_pivot_in_blocks <true,false> - allow pivoting within the small blocks during factorization (may increase
                                         stability of factorization.
.  -pc_factor_shift_type <shifttype> - Sets shift type or PETSC_DECIDE for the default; use '-help' for a list of available types
.  -pc_factor_shift_amount <shiftamount> - Sets shift amount or PETSC_DECIDE for the default
-   -pc_factor_nonzeros_along_diagonal - permutes the rows and columns to try to put nonzero value along the
        diagonal.

   Notes:
    Not all options work for all matrix formats
          Run with -help to see additional options for particular matrix formats or factorization
          algorithms

   Level: beginner

   Notes:
    Usually this will compute an "exact" solution in one iteration and does
          not need a Krylov method (i.e. you can use -ksp_type preonly, or
          KSPSetType(ksp,KSPPREONLY) for the Krylov method

.seealso: `PCCreate()`, `PCSetType()`, `PCType`, `PC`,
          `PCILU`, `PCCHOLESKY`, `PCICC`, `PCFactorSetReuseOrdering()`, `PCFactorSetReuseFill()`, `PCFactorGetMatrix()`,
          `PCFactorSetFill()`, `PCFactorSetUseInPlace()`, `PCFactorSetMatOrderingType()`, `PCFactorSetColumnPivot()`,
          `PCFactorSetPivotInBlocks(),PCFactorSetShiftType(),PCFactorSetShiftAmount()`
          `PCFactorReorderForNonzeroDiagonal()`
M*/

PETSC_EXTERN PetscErrorCode PCCreate_LU(PC pc)
{
  PC_LU          *dir;

  PetscFunctionBegin;
  PetscCall(PetscNewLog(pc,&dir));
  pc->data = (void*)dir;
  PetscCall(PCFactorInitialize(pc,MAT_FACTOR_LU));
  dir->nonzerosalongdiagonal    = PETSC_FALSE;

  ((PC_Factor*)dir)->info.fill          = 5.0;
  ((PC_Factor*)dir)->info.dtcol         = 1.e-6;  /* default to pivoting; this is only thing PETSc LU supports */
  ((PC_Factor*)dir)->info.shifttype     = (PetscReal)MAT_SHIFT_NONE;
  dir->col                              = NULL;
  dir->row                              = NULL;

  pc->ops->reset             = PCReset_LU;
  pc->ops->destroy           = PCDestroy_LU;
  pc->ops->apply             = PCApply_LU;
  pc->ops->matapply          = PCMatApply_LU;
  pc->ops->applytranspose    = PCApplyTranspose_LU;
  pc->ops->setup             = PCSetUp_LU;
  pc->ops->setfromoptions    = PCSetFromOptions_LU;
  pc->ops->view              = PCView_Factor;
  pc->ops->applyrichardson   = NULL;
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCFactorReorderForNonzeroDiagonal_C",PCFactorReorderForNonzeroDiagonal_LU));
  PetscFunctionReturn(0);
}
