
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
  CHKERRQ(PetscOptionsHead(PetscOptionsObject,"LU options"));
  CHKERRQ(PCSetFromOptions_Factor(PetscOptionsObject,pc));

  CHKERRQ(PetscOptionsName("-pc_factor_nonzeros_along_diagonal","Reorder to remove zeros from diagonal","PCFactorReorderForNonzeroDiagonal",&flg));
  if (flg) {
    tol  = PETSC_DECIDE;
    CHKERRQ(PetscOptionsReal("-pc_factor_nonzeros_along_diagonal","Reorder to remove zeros from diagonal","PCFactorReorderForNonzeroDiagonal",lu->nonzerosalongdiagonaltol,&tol,NULL));
    CHKERRQ(PCFactorReorderForNonzeroDiagonal(pc,tol));
  }
  CHKERRQ(PetscOptionsTail());
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

  CHKERRQ(MatSetErrorIfFailure(pc->pmat,pc->erroriffailure));
  if (dir->hdr.inplace) {
    MatFactorType ftype;

    CHKERRQ(MatGetFactorType(pc->pmat, &ftype));
    if (ftype == MAT_FACTOR_NONE) {
      if (dir->row && dir->col && dir->row != dir->col) CHKERRQ(ISDestroy(&dir->row));
      CHKERRQ(ISDestroy(&dir->col));
      /* This should only get the ordering if needed, but since MatGetFactor() is not called we can't know if it is needed */
      CHKERRQ(PCFactorSetDefaultOrdering_Factor(pc));
      CHKERRQ(MatGetOrdering(pc->pmat,((PC_Factor*)dir)->ordering,&dir->row,&dir->col));
      if (dir->row) {
        CHKERRQ(PetscLogObjectParent((PetscObject)pc,(PetscObject)dir->row));
        CHKERRQ(PetscLogObjectParent((PetscObject)pc,(PetscObject)dir->col));
      }
      CHKERRQ(MatLUFactor(pc->pmat,dir->row,dir->col,&((PC_Factor*)dir)->info));
      CHKERRQ(MatFactorGetError(pc->pmat,&err));
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
        CHKERRQ(MatGetFactor(pc->pmat,((PC_Factor*)dir)->solvertype,MAT_FACTOR_LU,&((PC_Factor*)dir)->fact));
        CHKERRQ(PetscLogObjectParent((PetscObject)pc,(PetscObject)((PC_Factor*)dir)->fact));
      }
      CHKERRQ(MatFactorGetCanUseOrdering(((PC_Factor*)dir)->fact,&canuseordering));
      if (canuseordering) {
        CHKERRQ(PCFactorSetDefaultOrdering_Factor(pc));
        CHKERRQ(MatGetOrdering(pc->pmat,((PC_Factor*)dir)->ordering,&dir->row,&dir->col));
        if (dir->nonzerosalongdiagonal) {
          CHKERRQ(MatReorderForNonzeroDiagonal(pc->pmat,dir->nonzerosalongdiagonaltol,dir->row,dir->col));
        }
        CHKERRQ(PetscLogObjectParent((PetscObject)pc,(PetscObject)dir->row));
        CHKERRQ(PetscLogObjectParent((PetscObject)pc,(PetscObject)dir->col));
      }
      CHKERRQ(MatLUFactorSymbolic(((PC_Factor*)dir)->fact,pc->pmat,dir->row,dir->col,&((PC_Factor*)dir)->info));
      CHKERRQ(MatGetInfo(((PC_Factor*)dir)->fact,MAT_LOCAL,&info));
      dir->hdr.actualfill = info.fill_ratio_needed;
    } else if (pc->flag != SAME_NONZERO_PATTERN) {
      PetscBool canuseordering;
      if (!dir->hdr.reuseordering) {
        CHKERRQ(MatDestroy(&((PC_Factor*)dir)->fact));
        CHKERRQ(MatGetFactor(pc->pmat,((PC_Factor*)dir)->solvertype,MAT_FACTOR_LU,&((PC_Factor*)dir)->fact));
        CHKERRQ(PetscLogObjectParent((PetscObject)pc,(PetscObject)((PC_Factor*)dir)->fact));
        CHKERRQ(MatFactorGetCanUseOrdering(((PC_Factor*)dir)->fact,&canuseordering));
        if (canuseordering) {
          if (dir->row && dir->col && dir->row != dir->col) CHKERRQ(ISDestroy(&dir->row));
          CHKERRQ(ISDestroy(&dir->col));
          CHKERRQ(PCFactorSetDefaultOrdering_Factor(pc));
          CHKERRQ(MatGetOrdering(pc->pmat,((PC_Factor*)dir)->ordering,&dir->row,&dir->col));
          if (dir->nonzerosalongdiagonal) {
            CHKERRQ(MatReorderForNonzeroDiagonal(pc->pmat,dir->nonzerosalongdiagonaltol,dir->row,dir->col));
          }
          CHKERRQ(PetscLogObjectParent((PetscObject)pc,(PetscObject)dir->row));
          CHKERRQ(PetscLogObjectParent((PetscObject)pc,(PetscObject)dir->col));
        }
      }
      CHKERRQ(MatLUFactorSymbolic(((PC_Factor*)dir)->fact,pc->pmat,dir->row,dir->col,&((PC_Factor*)dir)->info));
      CHKERRQ(MatGetInfo(((PC_Factor*)dir)->fact,MAT_LOCAL,&info));
      dir->hdr.actualfill = info.fill_ratio_needed;
    } else {
      CHKERRQ(MatFactorGetError(((PC_Factor*)dir)->fact,&err));
      if (err == MAT_FACTOR_NUMERIC_ZEROPIVOT) {
        CHKERRQ(MatFactorClearError(((PC_Factor*)dir)->fact));
        pc->failedreason = PC_NOERROR;
      }
    }
    CHKERRQ(MatFactorGetError(((PC_Factor*)dir)->fact,&err));
    if (err) { /* FactorSymbolic() fails */
      pc->failedreason = (PCFailedReason)err;
      PetscFunctionReturn(0);
    }

    CHKERRQ(MatLUFactorNumeric(((PC_Factor*)dir)->fact,pc->pmat,&((PC_Factor*)dir)->info));
    CHKERRQ(MatFactorGetError(((PC_Factor*)dir)->fact,&err));
    if (err) { /* FactorNumeric() fails */
      pc->failedreason = (PCFailedReason)err;
    }

  }

  CHKERRQ(PCFactorGetMatSolverType(pc,&stype));
  if (!stype) {
    MatSolverType solverpackage;
    CHKERRQ(MatFactorGetSolverType(((PC_Factor*)dir)->fact,&solverpackage));
    CHKERRQ(PCFactorSetMatSolverType(pc,solverpackage));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCReset_LU(PC pc)
{
  PC_LU          *dir = (PC_LU*)pc->data;

  PetscFunctionBegin;
  if (!dir->hdr.inplace && ((PC_Factor*)dir)->fact) CHKERRQ(MatDestroy(&((PC_Factor*)dir)->fact));
  if (dir->row && dir->col && dir->row != dir->col) CHKERRQ(ISDestroy(&dir->row));
  CHKERRQ(ISDestroy(&dir->col));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_LU(PC pc)
{
  PC_LU          *dir = (PC_LU*)pc->data;

  PetscFunctionBegin;
  CHKERRQ(PCReset_LU(pc));
  CHKERRQ(PetscFree(((PC_Factor*)dir)->ordering));
  CHKERRQ(PetscFree(((PC_Factor*)dir)->solvertype));
  CHKERRQ(PetscFree(pc->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApply_LU(PC pc,Vec x,Vec y)
{
  PC_LU          *dir = (PC_LU*)pc->data;

  PetscFunctionBegin;
  if (dir->hdr.inplace) {
    CHKERRQ(MatSolve(pc->pmat,x,y));
  } else {
    CHKERRQ(MatSolve(((PC_Factor*)dir)->fact,x,y));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCMatApply_LU(PC pc,Mat X,Mat Y)
{
  PC_LU          *dir = (PC_LU*)pc->data;

  PetscFunctionBegin;
  if (dir->hdr.inplace) {
    CHKERRQ(MatMatSolve(pc->pmat,X,Y));
  } else {
    CHKERRQ(MatMatSolve(((PC_Factor*)dir)->fact,X,Y));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplyTranspose_LU(PC pc,Vec x,Vec y)
{
  PC_LU          *dir = (PC_LU*)pc->data;

  PetscFunctionBegin;
  if (dir->hdr.inplace) {
    CHKERRQ(MatSolveTranspose(pc->pmat,x,y));
  } else {
    CHKERRQ(MatSolveTranspose(((PC_Factor*)dir)->fact,x,y));
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

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC,
           PCILU, PCCHOLESKY, PCICC, PCFactorSetReuseOrdering(), PCFactorSetReuseFill(), PCFactorGetMatrix(),
           PCFactorSetFill(), PCFactorSetUseInPlace(), PCFactorSetMatOrderingType(), PCFactorSetColumnPivot(),
           PCFactorSetPivotInBlocks(),PCFactorSetShiftType(),PCFactorSetShiftAmount()
           PCFactorReorderForNonzeroDiagonal()
M*/

PETSC_EXTERN PetscErrorCode PCCreate_LU(PC pc)
{
  PC_LU          *dir;

  PetscFunctionBegin;
  CHKERRQ(PetscNewLog(pc,&dir));
  pc->data = (void*)dir;
  CHKERRQ(PCFactorInitialize(pc,MAT_FACTOR_LU));
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
  CHKERRQ(PetscObjectComposeFunction((PetscObject)pc,"PCFactorReorderForNonzeroDiagonal_C",PCFactorReorderForNonzeroDiagonal_LU));
  PetscFunctionReturn(0);
}
