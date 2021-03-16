
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
  PetscErrorCode ierr;
  PetscBool      flg = PETSC_FALSE;
  PetscReal      tol;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"LU options");CHKERRQ(ierr);
  ierr = PCSetFromOptions_Factor(PetscOptionsObject,pc);CHKERRQ(ierr);

  ierr = PetscOptionsName("-pc_factor_nonzeros_along_diagonal","Reorder to remove zeros from diagonal","PCFactorReorderForNonzeroDiagonal",&flg);CHKERRQ(ierr);
  if (flg) {
    tol  = PETSC_DECIDE;
    ierr = PetscOptionsReal("-pc_factor_nonzeros_along_diagonal","Reorder to remove zeros from diagonal","PCFactorReorderForNonzeroDiagonal",lu->nonzerosalongdiagonaltol,&tol,NULL);CHKERRQ(ierr);
    ierr = PCFactorReorderForNonzeroDiagonal(pc,tol);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetUp_LU(PC pc)
{
  PetscErrorCode         ierr;
  PC_LU                  *dir = (PC_LU*)pc->data;
  MatSolverType          stype;
  MatFactorError         err;

  PetscFunctionBegin;
  pc->failedreason = PC_NOERROR;
  if (dir->hdr.reusefill && pc->setupcalled) ((PC_Factor*)dir)->info.fill = dir->hdr.actualfill;

  ierr = MatSetErrorIfFailure(pc->pmat,pc->erroriffailure);CHKERRQ(ierr);
  if (dir->hdr.inplace) {
    MatFactorType ftype;

    ierr = MatGetFactorType(pc->pmat, &ftype);CHKERRQ(ierr);
    if (ftype == MAT_FACTOR_NONE) {
      if (dir->row && dir->col && dir->row != dir->col) {ierr = ISDestroy(&dir->row);CHKERRQ(ierr);}
      ierr = ISDestroy(&dir->col);CHKERRQ(ierr);
      /* This should only get the ordering if needed, but since MatGetFactor() is not called we can't know if it is needed */
      ierr = MatGetOrdering(pc->pmat,((PC_Factor*)dir)->ordering,&dir->row,&dir->col);CHKERRQ(ierr);
      if (dir->row) {
        ierr = PetscLogObjectParent((PetscObject)pc,(PetscObject)dir->row);CHKERRQ(ierr);
        ierr = PetscLogObjectParent((PetscObject)pc,(PetscObject)dir->col);CHKERRQ(ierr);
      }
      ierr = MatLUFactor(pc->pmat,dir->row,dir->col,&((PC_Factor*)dir)->info);CHKERRQ(ierr);
      ierr = MatFactorGetError(pc->pmat,&err);CHKERRQ(ierr);
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
        ierr = MatGetFactor(pc->pmat,((PC_Factor*)dir)->solvertype,MAT_FACTOR_LU,&((PC_Factor*)dir)->fact);CHKERRQ(ierr);
        ierr = PetscLogObjectParent((PetscObject)pc,(PetscObject)((PC_Factor*)dir)->fact);CHKERRQ(ierr);
      }
      ierr = MatFactorGetCanUseOrdering(((PC_Factor*)dir)->fact,&canuseordering);CHKERRQ(ierr);
      if (canuseordering) {
        ierr = MatGetOrdering(pc->pmat,((PC_Factor*)dir)->ordering,&dir->row,&dir->col);CHKERRQ(ierr);
        if (dir->nonzerosalongdiagonal) {
          ierr = MatReorderForNonzeroDiagonal(pc->pmat,dir->nonzerosalongdiagonaltol,dir->row,dir->col);CHKERRQ(ierr);
        }
        ierr = PetscLogObjectParent((PetscObject)pc,(PetscObject)dir->row);CHKERRQ(ierr);
        ierr = PetscLogObjectParent((PetscObject)pc,(PetscObject)dir->col);CHKERRQ(ierr);
      }
      ierr                = MatLUFactorSymbolic(((PC_Factor*)dir)->fact,pc->pmat,dir->row,dir->col,&((PC_Factor*)dir)->info);CHKERRQ(ierr);
      ierr                = MatGetInfo(((PC_Factor*)dir)->fact,MAT_LOCAL,&info);CHKERRQ(ierr);
      dir->hdr.actualfill = info.fill_ratio_needed;
    } else if (pc->flag != SAME_NONZERO_PATTERN) {
      PetscBool canuseordering;
      if (!dir->hdr.reuseordering) {
        ierr = MatDestroy(&((PC_Factor*)dir)->fact);CHKERRQ(ierr);
        ierr = MatGetFactor(pc->pmat,((PC_Factor*)dir)->solvertype,MAT_FACTOR_LU,&((PC_Factor*)dir)->fact);CHKERRQ(ierr);
        ierr = PetscLogObjectParent((PetscObject)pc,(PetscObject)((PC_Factor*)dir)->fact);CHKERRQ(ierr);
        ierr = MatFactorGetCanUseOrdering(((PC_Factor*)dir)->fact,&canuseordering);CHKERRQ(ierr);
        if (canuseordering) {
          if (dir->row && dir->col && dir->row != dir->col) {ierr = ISDestroy(&dir->row);CHKERRQ(ierr);}
          ierr = ISDestroy(&dir->col);CHKERRQ(ierr);
          ierr = MatGetOrdering(pc->pmat,((PC_Factor*)dir)->ordering,&dir->row,&dir->col);CHKERRQ(ierr);
          if (dir->nonzerosalongdiagonal) {
            ierr = MatReorderForNonzeroDiagonal(pc->pmat,dir->nonzerosalongdiagonaltol,dir->row,dir->col);CHKERRQ(ierr);
          }
          ierr = PetscLogObjectParent((PetscObject)pc,(PetscObject)dir->row);CHKERRQ(ierr);
          ierr = PetscLogObjectParent((PetscObject)pc,(PetscObject)dir->col);CHKERRQ(ierr);
        }
      }
      ierr                = MatLUFactorSymbolic(((PC_Factor*)dir)->fact,pc->pmat,dir->row,dir->col,&((PC_Factor*)dir)->info);CHKERRQ(ierr);
      ierr                = MatGetInfo(((PC_Factor*)dir)->fact,MAT_LOCAL,&info);CHKERRQ(ierr);
      dir->hdr.actualfill = info.fill_ratio_needed;
    } else {
      ierr = MatFactorGetError(((PC_Factor*)dir)->fact,&err);CHKERRQ(ierr);
      if (err == MAT_FACTOR_NUMERIC_ZEROPIVOT) {
        ierr = MatFactorClearError(((PC_Factor*)dir)->fact);CHKERRQ(ierr);
        pc->failedreason = PC_NOERROR;
      }
    }
    ierr = MatFactorGetError(((PC_Factor*)dir)->fact,&err);CHKERRQ(ierr);
    if (err) { /* FactorSymbolic() fails */
      pc->failedreason = (PCFailedReason)err;
      PetscFunctionReturn(0);
    }

    ierr = MatLUFactorNumeric(((PC_Factor*)dir)->fact,pc->pmat,&((PC_Factor*)dir)->info);CHKERRQ(ierr);
    ierr = MatFactorGetError(((PC_Factor*)dir)->fact,&err);CHKERRQ(ierr);
    if (err) { /* FactorNumeric() fails */
      pc->failedreason = (PCFailedReason)err;
    }

  }

  ierr = PCFactorGetMatSolverType(pc,&stype);CHKERRQ(ierr);
  if (!stype) {
    MatSolverType solverpackage;
    ierr = MatFactorGetSolverType(((PC_Factor*)dir)->fact,&solverpackage);CHKERRQ(ierr);
    ierr = PCFactorSetMatSolverType(pc,solverpackage);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCReset_LU(PC pc)
{
  PC_LU          *dir = (PC_LU*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!dir->hdr.inplace && ((PC_Factor*)dir)->fact) {ierr = MatDestroy(&((PC_Factor*)dir)->fact);CHKERRQ(ierr);}
  if (dir->row && dir->col && dir->row != dir->col) {ierr = ISDestroy(&dir->row);CHKERRQ(ierr);}
  ierr = ISDestroy(&dir->col);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_LU(PC pc)
{
  PC_LU          *dir = (PC_LU*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCReset_LU(pc);CHKERRQ(ierr);
  ierr = PetscFree(((PC_Factor*)dir)->ordering);CHKERRQ(ierr);
  ierr = PetscFree(((PC_Factor*)dir)->solvertype);CHKERRQ(ierr);
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApply_LU(PC pc,Vec x,Vec y)
{
  PC_LU          *dir = (PC_LU*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (dir->hdr.inplace) {
    ierr = MatSolve(pc->pmat,x,y);CHKERRQ(ierr);
  } else {
    ierr = MatSolve(((PC_Factor*)dir)->fact,x,y);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCMatApply_LU(PC pc,Mat X,Mat Y)
{
  PC_LU          *dir = (PC_LU*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (dir->hdr.inplace) {
    ierr = MatMatSolve(pc->pmat,X,Y);CHKERRQ(ierr);
  } else {
    ierr = MatMatSolve(((PC_Factor*)dir)->fact,X,Y);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplyTranspose_LU(PC pc,Vec x,Vec y)
{
  PC_LU          *dir = (PC_LU*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (dir->hdr.inplace) {
    ierr = MatSolveTranspose(pc->pmat,x,y);CHKERRQ(ierr);
  } else {
    ierr = MatSolveTranspose(((PC_Factor*)dir)->fact,x,y);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PC_LU          *dir;

  PetscFunctionBegin;
  ierr     = PetscNewLog(pc,&dir);CHKERRQ(ierr);
  pc->data = (void*)dir;
  ierr     = PCFactorInitialize(pc);CHKERRQ(ierr);
  ((PC_Factor*)dir)->factortype = MAT_FACTOR_LU;
  dir->nonzerosalongdiagonal    = PETSC_FALSE;

  ((PC_Factor*)dir)->info.fill          = 5.0;
  ((PC_Factor*)dir)->info.dtcol         = 1.e-6;  /* default to pivoting; this is only thing PETSc LU supports */
  ((PC_Factor*)dir)->info.shifttype     = (PetscReal)MAT_SHIFT_NONE;
  dir->col                              = NULL;
  dir->row                              = NULL;

  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)pc),&size);CHKERRMPI(ierr);
  if (size == 1) {
    ierr = PetscStrallocpy(MATORDERINGND,(char**)&((PC_Factor*)dir)->ordering);CHKERRQ(ierr);
  } else {
    ierr = PetscStrallocpy(MATORDERINGNATURAL,(char**)&((PC_Factor*)dir)->ordering);CHKERRQ(ierr);
  }

  pc->ops->reset             = PCReset_LU;
  pc->ops->destroy           = PCDestroy_LU;
  pc->ops->apply             = PCApply_LU;
  pc->ops->matapply          = PCMatApply_LU;
  pc->ops->applytranspose    = PCApplyTranspose_LU;
  pc->ops->setup             = PCSetUp_LU;
  pc->ops->setfromoptions    = PCSetFromOptions_LU;
  pc->ops->view              = PCView_Factor;
  pc->ops->applyrichardson   = NULL;
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFactorReorderForNonzeroDiagonal_C",PCFactorReorderForNonzeroDiagonal_LU);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
