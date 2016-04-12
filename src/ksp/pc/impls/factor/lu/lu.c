
/*
   Defines a direct factorization preconditioner for any Mat implementation
   Note: this need not be consided a preconditioner since it supplies
         a direct solver.
*/

#include <../src/ksp/pc/impls/factor/lu/lu.h>  /*I "petscpc.h" I*/


#undef __FUNCT__
#define __FUNCT__ "PCFactorReorderForNonzeroDiagonal_LU"
PetscErrorCode PCFactorReorderForNonzeroDiagonal_LU(PC pc,PetscReal z)
{
  PC_LU *lu = (PC_LU*)pc->data;

  PetscFunctionBegin;
  lu->nonzerosalongdiagonal = PETSC_TRUE;
  if (z == PETSC_DECIDE) lu->nonzerosalongdiagonaltol = 1.e-10;
  else lu->nonzerosalongdiagonaltol = z;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCFactorSetReuseOrdering_LU"
PetscErrorCode PCFactorSetReuseOrdering_LU(PC pc,PetscBool flag)
{
  PC_LU *lu = (PC_LU*)pc->data;

  PetscFunctionBegin;
  lu->reuseordering = flag;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCFactorSetReuseFill_LU"
PetscErrorCode PCFactorSetReuseFill_LU(PC pc,PetscBool flag)
{
  PC_LU *lu = (PC_LU*)pc->data;

  PetscFunctionBegin;
  lu->reusefill = flag;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCSetFromOptions_LU"
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
    ierr = PetscOptionsReal("-pc_factor_nonzeros_along_diagonal","Reorder to remove zeros from diagonal","PCFactorReorderForNonzeroDiagonal",lu->nonzerosalongdiagonaltol,&tol,0);CHKERRQ(ierr);
    ierr = PCFactorReorderForNonzeroDiagonal(pc,tol);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCView_LU"
static PetscErrorCode PCView_LU(PC pc,PetscViewer viewer)
{
  PC_LU          *lu = (PC_LU*)pc->data;
  PetscErrorCode ierr;
  PetscBool      iascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    if (lu->inplace) {
      ierr = PetscViewerASCIIPrintf(viewer,"  LU: in-place factorization\n");CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  LU: out-of-place factorization\n");CHKERRQ(ierr);
    }

    if (lu->reusefill)    {ierr = PetscViewerASCIIPrintf(viewer,"       Reusing fill from past factorization\n");CHKERRQ(ierr);}
    if (lu->reuseordering) {ierr = PetscViewerASCIIPrintf(viewer,"       Reusing reordering from past factorization\n");CHKERRQ(ierr);}
  }
  ierr = PCView_Factor(pc,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCSetUp_LU"
static PetscErrorCode PCSetUp_LU(PC pc)
{
  PetscErrorCode ierr;
  PC_LU          *dir = (PC_LU*)pc->data;
  const MatSolverPackage stype;

  PetscFunctionBegin;
  if (dir->reusefill && pc->setupcalled) ((PC_Factor*)dir)->info.fill = dir->actualfill;

  ierr = MatSetErrorIfFailure(pc->pmat,pc->erroriffailure);CHKERRQ(ierr);
  if (dir->inplace) {
    if (dir->row && dir->col && dir->row != dir->col) {ierr = ISDestroy(&dir->row);CHKERRQ(ierr);}
    ierr = ISDestroy(&dir->col);CHKERRQ(ierr);
    ierr = MatGetOrdering(pc->pmat,((PC_Factor*)dir)->ordering,&dir->row,&dir->col);CHKERRQ(ierr);
    if (dir->row) {
      ierr = PetscLogObjectParent((PetscObject)pc,(PetscObject)dir->row);CHKERRQ(ierr);
      ierr = PetscLogObjectParent((PetscObject)pc,(PetscObject)dir->col);CHKERRQ(ierr);
    }
    ierr = MatLUFactor(pc->pmat,dir->row,dir->col,&((PC_Factor*)dir)->info);CHKERRQ(ierr);
    if (pc->pmat->errortype) { /* Factor() fails */
      pc->failedreason = (PCFailedReason)pc->pmat->errortype;
      PetscFunctionReturn(0);
    }

    ((PC_Factor*)dir)->fact = pc->pmat;
  } else {
    MatInfo info;
    Mat     F;
    if (!pc->setupcalled) {
      ierr = MatGetOrdering(pc->pmat,((PC_Factor*)dir)->ordering,&dir->row,&dir->col);CHKERRQ(ierr);
      if (dir->nonzerosalongdiagonal) {
        ierr = MatReorderForNonzeroDiagonal(pc->pmat,dir->nonzerosalongdiagonaltol,dir->row,dir->col);CHKERRQ(ierr);
      }
      if (dir->row) {
        ierr = PetscLogObjectParent((PetscObject)pc,(PetscObject)dir->row);CHKERRQ(ierr);
        ierr = PetscLogObjectParent((PetscObject)pc,(PetscObject)dir->col);CHKERRQ(ierr);
      }
      if (!((PC_Factor*)dir)->fact) {
        ierr = MatGetFactor(pc->pmat,((PC_Factor*)dir)->solvertype,MAT_FACTOR_LU,&((PC_Factor*)dir)->fact);CHKERRQ(ierr);
      }
      ierr            = MatLUFactorSymbolic(((PC_Factor*)dir)->fact,pc->pmat,dir->row,dir->col,&((PC_Factor*)dir)->info);CHKERRQ(ierr);
      ierr            = MatGetInfo(((PC_Factor*)dir)->fact,MAT_LOCAL,&info);CHKERRQ(ierr);
      dir->actualfill = info.fill_ratio_needed;
      ierr            = PetscLogObjectParent((PetscObject)pc,(PetscObject)((PC_Factor*)dir)->fact);CHKERRQ(ierr);
    } else if (pc->flag != SAME_NONZERO_PATTERN) {
      if (!dir->reuseordering) {
        if (dir->row && dir->col && dir->row != dir->col) {ierr = ISDestroy(&dir->row);CHKERRQ(ierr);}
        ierr = ISDestroy(&dir->col);CHKERRQ(ierr);
        ierr = MatGetOrdering(pc->pmat,((PC_Factor*)dir)->ordering,&dir->row,&dir->col);CHKERRQ(ierr);
        if (dir->nonzerosalongdiagonal) {
          ierr = MatReorderForNonzeroDiagonal(pc->pmat,dir->nonzerosalongdiagonaltol,dir->row,dir->col);CHKERRQ(ierr);
        }
        if (dir->row) {
          ierr = PetscLogObjectParent((PetscObject)pc,(PetscObject)dir->row);CHKERRQ(ierr);
          ierr = PetscLogObjectParent((PetscObject)pc,(PetscObject)dir->col);CHKERRQ(ierr);
        }
      }
      ierr            = MatDestroy(&((PC_Factor*)dir)->fact);CHKERRQ(ierr);
      ierr            = MatGetFactor(pc->pmat,((PC_Factor*)dir)->solvertype,MAT_FACTOR_LU,&((PC_Factor*)dir)->fact);CHKERRQ(ierr);
      ierr            = MatLUFactorSymbolic(((PC_Factor*)dir)->fact,pc->pmat,dir->row,dir->col,&((PC_Factor*)dir)->info);CHKERRQ(ierr);
      ierr            = MatGetInfo(((PC_Factor*)dir)->fact,MAT_LOCAL,&info);CHKERRQ(ierr);
      dir->actualfill = info.fill_ratio_needed;
      ierr            = PetscLogObjectParent((PetscObject)pc,(PetscObject)((PC_Factor*)dir)->fact);CHKERRQ(ierr);
    }
    F = ((PC_Factor*)dir)->fact;
    if (F->errortype) { /* FactorSymbolic() fails */
      pc->failedreason = (PCFailedReason)F->errortype;
      PetscFunctionReturn(0);
    }

    ierr = MatLUFactorNumeric(((PC_Factor*)dir)->fact,pc->pmat,&((PC_Factor*)dir)->info);CHKERRQ(ierr);
    if (F->errortype) { /* FactorNumeric() fails */
      pc->failedreason = (PCFailedReason)F->errortype;
    }

  }

  ierr = PCFactorGetMatSolverPackage(pc,&stype);CHKERRQ(ierr);
  if (!stype) {
    ierr = PCFactorSetMatSolverPackage(pc,((PC_Factor*)dir)->fact->solvertype);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCReset_LU"
static PetscErrorCode PCReset_LU(PC pc)
{
  PC_LU          *dir = (PC_LU*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!dir->inplace && ((PC_Factor*)dir)->fact) {ierr = MatDestroy(&((PC_Factor*)dir)->fact);CHKERRQ(ierr);}
  if (dir->row && dir->col && dir->row != dir->col) {ierr = ISDestroy(&dir->row);CHKERRQ(ierr);}
  ierr = ISDestroy(&dir->col);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCDestroy_LU"
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

#undef __FUNCT__
#define __FUNCT__ "PCApply_LU"
static PetscErrorCode PCApply_LU(PC pc,Vec x,Vec y)
{
  PC_LU          *dir = (PC_LU*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (dir->inplace) {
    ierr = MatSolve(pc->pmat,x,y);CHKERRQ(ierr);
  } else {
    ierr = MatSolve(((PC_Factor*)dir)->fact,x,y);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCApplyTranspose_LU"
static PetscErrorCode PCApplyTranspose_LU(PC pc,Vec x,Vec y)
{
  PC_LU          *dir = (PC_LU*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (dir->inplace) {
    ierr = MatSolveTranspose(pc->pmat,x,y);CHKERRQ(ierr);
  } else {
    ierr = MatSolveTranspose(((PC_Factor*)dir)->fact,x,y);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* -----------------------------------------------------------------------------------*/

#undef __FUNCT__
#define __FUNCT__ "PCFactorSetUseInPlace_LU"
PetscErrorCode  PCFactorSetUseInPlace_LU(PC pc,PetscBool flg)
{
  PC_LU *dir = (PC_LU*)pc->data;

  PetscFunctionBegin;
  dir->inplace = flg;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCFactorGetUseInPlace_LU"
PetscErrorCode  PCFactorGetUseInPlace_LU(PC pc,PetscBool *flg)
{
  PC_LU *dir = (PC_LU*)pc->data;

  PetscFunctionBegin;
  *flg = dir->inplace;
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------ */

/*MC
   PCLU - Uses a direct solver, based on LU factorization, as a preconditioner

   Options Database Keys:
+  -pc_factor_reuse_ordering - Activate PCFactorSetReuseOrdering()
.  -pc_factor_mat_solver_package - Actives PCFactorSetMatSolverPackage() to choose the direct solver, like superlu
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

   Notes: Not all options work for all matrix formats
          Run with -help to see additional options for particular matrix formats or factorization
          algorithms

   Level: beginner

   Concepts: LU factorization, direct solver

   Notes: Usually this will compute an "exact" solution in one iteration and does
          not need a Krylov method (i.e. you can use -ksp_type preonly, or
          KSPSetType(ksp,KSPPREONLY) for the Krylov method

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC,
           PCILU, PCCHOLESKY, PCICC, PCFactorSetReuseOrdering(), PCFactorSetReuseFill(), PCFactorGetMatrix(),
           PCFactorSetFill(), PCFactorSetUseInPlace(), PCFactorSetMatOrderingType(), PCFactorSetColumnPivot(),
           PCFactorSetPivotingInBlocks(),PCFactorSetShiftType(),PCFactorSetShiftAmount()
           PCFactorReorderForNonzeroDiagonal()
M*/

#undef __FUNCT__
#define __FUNCT__ "PCCreate_LU"
PETSC_EXTERN PetscErrorCode PCCreate_LU(PC pc)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PC_LU          *dir;

  PetscFunctionBegin;
  ierr = PetscNewLog(pc,&dir);CHKERRQ(ierr);

  ierr = MatFactorInfoInitialize(&((PC_Factor*)dir)->info);CHKERRQ(ierr);

  ((PC_Factor*)dir)->fact       = NULL;
  ((PC_Factor*)dir)->factortype = MAT_FACTOR_LU;
  dir->inplace                  = PETSC_FALSE;
  dir->nonzerosalongdiagonal    = PETSC_FALSE;

  ((PC_Factor*)dir)->info.fill          = 5.0;
  ((PC_Factor*)dir)->info.dtcol         = 1.e-6;  /* default to pivoting; this is only thing PETSc LU supports */
  ((PC_Factor*)dir)->info.shifttype     = (PetscReal)MAT_SHIFT_NONE;
  ((PC_Factor*)dir)->info.shiftamount   = 0.0;
  ((PC_Factor*)dir)->info.zeropivot     = 100.0*PETSC_MACHINE_EPSILON;
  ((PC_Factor*)dir)->info.pivotinblocks = 1.0;
  dir->col                              = 0;
  dir->row                              = 0;

  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)pc),&size);CHKERRQ(ierr);
  if (size == 1) {
    ierr = PetscStrallocpy(MATORDERINGND,(char**)&((PC_Factor*)dir)->ordering);CHKERRQ(ierr);
  } else {
    ierr = PetscStrallocpy(MATORDERINGNATURAL,(char**)&((PC_Factor*)dir)->ordering);CHKERRQ(ierr);
  }
  dir->reusefill     = PETSC_FALSE;
  dir->reuseordering = PETSC_FALSE;
  pc->data           = (void*)dir;

  pc->ops->reset             = PCReset_LU;
  pc->ops->destroy           = PCDestroy_LU;
  pc->ops->apply             = PCApply_LU;
  pc->ops->applytranspose    = PCApplyTranspose_LU;
  pc->ops->setup             = PCSetUp_LU;
  pc->ops->setfromoptions    = PCSetFromOptions_LU;
  pc->ops->view              = PCView_LU;
  pc->ops->applyrichardson   = 0;
  pc->ops->getfactoredmatrix = PCFactorGetMatrix_Factor;

  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFactorSetUpMatSolverPackage_C",PCFactorSetUpMatSolverPackage_Factor);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFactorGetMatSolverPackage_C",PCFactorGetMatSolverPackage_Factor);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFactorSetMatSolverPackage_C",PCFactorSetMatSolverPackage_Factor);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFactorSetZeroPivot_C",PCFactorSetZeroPivot_Factor);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFactorSetShiftType_C",PCFactorSetShiftType_Factor);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFactorSetShiftAmount_C",PCFactorSetShiftAmount_Factor);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFactorSetFill_C",PCFactorSetFill_Factor);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFactorSetUseInPlace_C",PCFactorSetUseInPlace_LU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFactorGetUseInPlace_C",PCFactorGetUseInPlace_LU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFactorSetMatOrderingType_C",PCFactorSetMatOrderingType_Factor);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFactorSetReuseOrdering_C",PCFactorSetReuseOrdering_LU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFactorSetReuseFill_C",PCFactorSetReuseFill_LU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFactorSetColumnPivot_C",PCFactorSetColumnPivot_Factor);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFactorSetPivotInBlocks_C",PCFactorSetPivotInBlocks_Factor);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFactorReorderForNonzeroDiagonal_C",PCFactorReorderForNonzeroDiagonal_LU);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
