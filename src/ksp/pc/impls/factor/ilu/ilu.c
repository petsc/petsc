
/*
   Defines a ILU factorization preconditioner for any Mat implementation
*/
#include <../src/ksp/pc/impls/factor/ilu/ilu.h>     /*I "petscpc.h"  I*/

/* ------------------------------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "PCFactorSetReuseFill_ILU"
PetscErrorCode  PCFactorSetReuseFill_ILU(PC pc,PetscBool flag)
{
  PC_ILU *lu = (PC_ILU*)pc->data;

  PetscFunctionBegin;
  lu->reusefill = flag;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCFactorReorderForNonzeroDiagonal_ILU"
PetscErrorCode  PCFactorReorderForNonzeroDiagonal_ILU(PC pc,PetscReal z)
{
  PC_ILU *ilu = (PC_ILU*)pc->data;

  PetscFunctionBegin;
  ilu->nonzerosalongdiagonal = PETSC_TRUE;
  if (z == PETSC_DECIDE) ilu->nonzerosalongdiagonaltol = 1.e-10;
  else ilu->nonzerosalongdiagonaltol = z;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCReset_ILU"
PetscErrorCode PCReset_ILU(PC pc)
{
  PC_ILU         *ilu = (PC_ILU*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!ilu->inplace) {ierr = MatDestroy(&((PC_Factor*)ilu)->fact);CHKERRQ(ierr);}
  if (ilu->row && ilu->col && ilu->row != ilu->col) {ierr = ISDestroy(&ilu->row);CHKERRQ(ierr);}
  ierr = ISDestroy(&ilu->col);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCFactorSetDropTolerance_ILU"
PetscErrorCode  PCFactorSetDropTolerance_ILU(PC pc,PetscReal dt,PetscReal dtcol,PetscInt dtcount)
{
  PC_ILU *ilu = (PC_ILU*)pc->data;

  PetscFunctionBegin;
  if (pc->setupcalled && (((PC_Factor*)ilu)->info.dt != dt || ((PC_Factor*)ilu)->info.dtcol != dtcol || ((PC_Factor*)ilu)->info.dtcount != dtcount)) {
    SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Cannot change drop tolerance after using PC");
  }
  ((PC_Factor*)ilu)->info.dt      = dt;
  ((PC_Factor*)ilu)->info.dtcol   = dtcol;
  ((PC_Factor*)ilu)->info.dtcount = dtcount;
  ((PC_Factor*)ilu)->info.usedt   = 1.0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCFactorSetReuseOrdering_ILU"
PetscErrorCode  PCFactorSetReuseOrdering_ILU(PC pc,PetscBool flag)
{
  PC_ILU *ilu = (PC_ILU*)pc->data;

  PetscFunctionBegin;
  ilu->reuseordering = flag;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCFactorSetUseInPlace_ILU"
PetscErrorCode  PCFactorSetUseInPlace_ILU(PC pc,PetscBool flg)
{
  PC_ILU *dir = (PC_ILU*)pc->data;

  PetscFunctionBegin;
  dir->inplace = flg;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCFactorGetUseInPlace_ILU"
PetscErrorCode  PCFactorGetUseInPlace_ILU(PC pc,PetscBool *flg)
{
  PC_ILU *dir = (PC_ILU*)pc->data;

  PetscFunctionBegin;
  *flg = dir->inplace;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCSetFromOptions_ILU"
static PetscErrorCode PCSetFromOptions_ILU(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PetscErrorCode ierr;
  PetscInt       itmp;
  PetscBool      flg,set;
  PC_ILU         *ilu = (PC_ILU*)pc->data;
  PetscReal      tol;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"ILU Options");CHKERRQ(ierr);
  ierr = PCSetFromOptions_Factor(PetscOptionsObject,pc);CHKERRQ(ierr);

  ierr = PetscOptionsInt("-pc_factor_levels","levels of fill","PCFactorSetLevels",(PetscInt)((PC_Factor*)ilu)->info.levels,&itmp,&flg);CHKERRQ(ierr);
  if (flg) ((PC_Factor*)ilu)->info.levels = itmp;

  ierr = PetscOptionsBool("-pc_factor_diagonal_fill","Allow fill into empty diagonal entry","PCFactorSetAllowDiagonalFill",((PC_Factor*)ilu)->info.diagonal_fill ? PETSC_TRUE : PETSC_FALSE,&flg,&set);CHKERRQ(ierr);
  if (set) ((PC_Factor*)ilu)->info.diagonal_fill = (PetscReal) flg;
  ierr = PetscOptionsName("-pc_factor_nonzeros_along_diagonal","Reorder to remove zeros from diagonal","PCFactorReorderForNonzeroDiagonal",&flg);CHKERRQ(ierr);
  if (flg) {
    tol  = PETSC_DECIDE;
    ierr = PetscOptionsReal("-pc_factor_nonzeros_along_diagonal","Reorder to remove zeros from diagonal","PCFactorReorderForNonzeroDiagonal",ilu->nonzerosalongdiagonaltol,&tol,0);CHKERRQ(ierr);
    ierr = PCFactorReorderForNonzeroDiagonal(pc,tol);CHKERRQ(ierr);
  }

  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCView_ILU"
static PetscErrorCode PCView_ILU(PC pc,PetscViewer viewer)
{
  PC_ILU         *ilu = (PC_ILU*)pc->data;
  PetscErrorCode ierr;
  PetscBool      iascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    if (ilu->inplace) {
      ierr = PetscViewerASCIIPrintf(viewer,"  ILU: in-place factorization\n");CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  ILU: out-of-place factorization\n");CHKERRQ(ierr);
    }

    if (ilu->reusefill)     {ierr = PetscViewerASCIIPrintf(viewer,"  ILU: Reusing fill from past factorization\n");CHKERRQ(ierr);}
    if (ilu->reuseordering) {ierr = PetscViewerASCIIPrintf(viewer,"  ILU: Reusing reordering from past factorization\n");CHKERRQ(ierr);}
  }
  ierr = PCView_Factor(pc,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCSetUp_ILU"
static PetscErrorCode PCSetUp_ILU(PC pc)
{
  PetscErrorCode ierr;
  PC_ILU         *ilu = (PC_ILU*)pc->data;
  MatInfo        info;
  PetscBool      flg;
  const MatSolverPackage stype;

  PetscFunctionBegin;
  /* ugly hack to change default, since it is not support by some matrix types */
  if (((PC_Factor*)ilu)->info.shifttype == (PetscReal)MAT_SHIFT_NONZERO) {
    ierr = PetscObjectTypeCompare((PetscObject)pc->pmat,MATSEQAIJ,&flg);CHKERRQ(ierr);
    if (!flg) {
      ierr = PetscObjectTypeCompare((PetscObject)pc->pmat,MATMPIAIJ,&flg);CHKERRQ(ierr);
      if (!flg) {
        ((PC_Factor*)ilu)->info.shifttype = (PetscReal)MAT_SHIFT_INBLOCKS;
        PetscInfo(pc,"Changing shift type from NONZERO to INBLOCKS because block matrices do not support NONZERO\n");CHKERRQ(ierr);
      }
    }
  }

  ierr = MatSetErrorIfFailure(pc->pmat,pc->erroriffailure);CHKERRQ(ierr);
  if (ilu->inplace) {
    if (!pc->setupcalled) {

      /* In-place factorization only makes sense with the natural ordering,
         so we only need to get the ordering once, even if nonzero structure changes */
      ierr = MatGetOrdering(pc->pmat,((PC_Factor*)ilu)->ordering,&ilu->row,&ilu->col);CHKERRQ(ierr);
      if (ilu->row) {ierr = PetscLogObjectParent((PetscObject)pc,(PetscObject)ilu->row);CHKERRQ(ierr);}
      if (ilu->col) {ierr = PetscLogObjectParent((PetscObject)pc,(PetscObject)ilu->col);CHKERRQ(ierr);}
    }

    /* In place ILU only makes sense with fill factor of 1.0 because
       cannot have levels of fill */
    ((PC_Factor*)ilu)->info.fill          = 1.0;
    ((PC_Factor*)ilu)->info.diagonal_fill = 0.0;

    ierr = MatILUFactor(pc->pmat,ilu->row,ilu->col,&((PC_Factor*)ilu)->info);CHKERRQ(ierr);CHKERRQ(ierr);
    if (pc->pmat->errortype) { /* Factor() fails */
      pc->failedreason = (PCFailedReason)pc->pmat->errortype;
      PetscFunctionReturn(0);
    }

    ((PC_Factor*)ilu)->fact = pc->pmat;
    /* must update the pc record of the matrix state or the PC will attempt to run PCSetUp() yet again */
    ierr = PetscObjectStateGet((PetscObject)pc->pmat,&pc->matstate);CHKERRQ(ierr);
  } else {
    Mat F;
    if (!pc->setupcalled) {
      /* first time in so compute reordering and symbolic factorization */
      ierr = MatGetOrdering(pc->pmat,((PC_Factor*)ilu)->ordering,&ilu->row,&ilu->col);CHKERRQ(ierr);
      if (ilu->row) {ierr = PetscLogObjectParent((PetscObject)pc,(PetscObject)ilu->row);CHKERRQ(ierr);}
      if (ilu->col) {ierr = PetscLogObjectParent((PetscObject)pc,(PetscObject)ilu->col);CHKERRQ(ierr);}
      /*  Remove zeros along diagonal?     */
      if (ilu->nonzerosalongdiagonal) {
        ierr = MatReorderForNonzeroDiagonal(pc->pmat,ilu->nonzerosalongdiagonaltol,ilu->row,ilu->col);CHKERRQ(ierr);
      }
      if (!((PC_Factor*)ilu)->fact) {
        ierr = MatGetFactor(pc->pmat,((PC_Factor*)ilu)->solvertype,MAT_FACTOR_ILU,&((PC_Factor*)ilu)->fact);CHKERRQ(ierr);
      }
      ierr = MatILUFactorSymbolic(((PC_Factor*)ilu)->fact,pc->pmat,ilu->row,ilu->col,&((PC_Factor*)ilu)->info);CHKERRQ(ierr);
      ierr = MatGetInfo(((PC_Factor*)ilu)->fact,MAT_LOCAL,&info);CHKERRQ(ierr);

      ilu->actualfill = info.fill_ratio_needed;

      ierr = PetscLogObjectParent((PetscObject)pc,(PetscObject)((PC_Factor*)ilu)->fact);CHKERRQ(ierr);
    } else if (pc->flag != SAME_NONZERO_PATTERN) {
      if (!ilu->reuseordering) {
        /* compute a new ordering for the ILU */
        ierr = ISDestroy(&ilu->row);CHKERRQ(ierr);
        ierr = ISDestroy(&ilu->col);CHKERRQ(ierr);
        ierr = MatGetOrdering(pc->pmat,((PC_Factor*)ilu)->ordering,&ilu->row,&ilu->col);CHKERRQ(ierr);
        if (ilu->row) {ierr = PetscLogObjectParent((PetscObject)pc,(PetscObject)ilu->row);CHKERRQ(ierr);}
        if (ilu->col) {ierr = PetscLogObjectParent((PetscObject)pc,(PetscObject)ilu->col);CHKERRQ(ierr);}
        /*  Remove zeros along diagonal?     */
        if (ilu->nonzerosalongdiagonal) {
          ierr = MatReorderForNonzeroDiagonal(pc->pmat,ilu->nonzerosalongdiagonaltol,ilu->row,ilu->col);CHKERRQ(ierr);
        }
      }
      ierr = MatDestroy(&((PC_Factor*)ilu)->fact);CHKERRQ(ierr);
      ierr = MatGetFactor(pc->pmat,((PC_Factor*)ilu)->solvertype,MAT_FACTOR_ILU,&((PC_Factor*)ilu)->fact);CHKERRQ(ierr);
      ierr = MatILUFactorSymbolic(((PC_Factor*)ilu)->fact,pc->pmat,ilu->row,ilu->col,&((PC_Factor*)ilu)->info);CHKERRQ(ierr);
      ierr = MatGetInfo(((PC_Factor*)ilu)->fact,MAT_LOCAL,&info);CHKERRQ(ierr);

      ilu->actualfill = info.fill_ratio_needed;

      ierr = PetscLogObjectParent((PetscObject)pc,(PetscObject)((PC_Factor*)ilu)->fact);CHKERRQ(ierr);
    }
    F = ((PC_Factor*)ilu)->fact;
    if (F->errortype) { /* FactorSymbolic() fails */
      pc->failedreason = (PCFailedReason)F->errortype;
      PetscFunctionReturn(0);
    }

    ierr = MatLUFactorNumeric(((PC_Factor*)ilu)->fact,pc->pmat,&((PC_Factor*)ilu)->info);CHKERRQ(ierr);
    if (F->errortype) { /* FactorNumeric() fails */
      pc->failedreason = (PCFailedReason)F->errortype;
    }
  }

  ierr = PCFactorGetMatSolverPackage(pc,&stype);CHKERRQ(ierr);
  if (!stype) {
    ierr = PCFactorSetMatSolverPackage(pc,((PC_Factor*)ilu)->fact->solvertype);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCDestroy_ILU"
static PetscErrorCode PCDestroy_ILU(PC pc)
{
  PC_ILU         *ilu = (PC_ILU*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCReset_ILU(pc);CHKERRQ(ierr);
  ierr = PetscFree(((PC_Factor*)ilu)->solvertype);CHKERRQ(ierr);
  ierr = PetscFree(((PC_Factor*)ilu)->ordering);CHKERRQ(ierr);
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCApply_ILU"
static PetscErrorCode PCApply_ILU(PC pc,Vec x,Vec y)
{
  PC_ILU         *ilu = (PC_ILU*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSolve(((PC_Factor*)ilu)->fact,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCApplyTranspose_ILU"
static PetscErrorCode PCApplyTranspose_ILU(PC pc,Vec x,Vec y)
{
  PC_ILU         *ilu = (PC_ILU*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSolveTranspose(((PC_Factor*)ilu)->fact,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCApplySymmetricLeft_ILU"
static PetscErrorCode PCApplySymmetricLeft_ILU(PC pc,Vec x,Vec y)
{
  PetscErrorCode ierr;
  PC_ILU         *icc = (PC_ILU*)pc->data;

  PetscFunctionBegin;
  ierr = MatForwardSolve(((PC_Factor*)icc)->fact,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCApplySymmetricRight_ILU"
static PetscErrorCode PCApplySymmetricRight_ILU(PC pc,Vec x,Vec y)
{
  PetscErrorCode ierr;
  PC_ILU         *icc = (PC_ILU*)pc->data;

  PetscFunctionBegin;
  ierr = MatBackwardSolve(((PC_Factor*)icc)->fact,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
     PCILU - Incomplete factorization preconditioners.

   Options Database Keys:
+  -pc_factor_levels <k> - number of levels of fill for ILU(k)
.  -pc_factor_in_place - only for ILU(0) with natural ordering, reuses the space of the matrix for
                      its factorization (overwrites original matrix)
.  -pc_factor_diagonal_fill - fill in a zero diagonal even if levels of fill indicate it wouldn't be fill
.  -pc_factor_reuse_ordering - reuse ordering of factorized matrix from previous factorization
.  -pc_factor_fill <nfill> - expected amount of fill in factored matrix compared to original matrix, nfill > 1
.  -pc_factor_nonzeros_along_diagonal - reorder the matrix before factorization to remove zeros from the diagonal,
                                   this decreases the chance of getting a zero pivot
.  -pc_factor_mat_ordering_type <natural,nd,1wd,rcm,qmd> - set the row/column ordering of the factored matrix
-  -pc_factor_pivot_in_blocks - for block ILU(k) factorization, i.e. with BAIJ matrices with block size larger
                             than 1 the diagonal blocks are factored with partial pivoting (this increases the
                             stability of the ILU factorization

   Level: beginner

  Concepts: incomplete factorization

   Notes: Only implemented for some matrix formats. (for parallel see PCHYPRE for hypre's ILU)

          For BAIJ matrices this implements a point block ILU

          The "symmetric" application of this preconditioner is not actually symmetric since L is not transpose(U)
          even when the matrix is not symmetric since the U stores the diagonals of the factorization.

          If you are using MATSEQAIJCUSPARSE matrices (or MATMPIAIJCUSPARESE matrices with block Jacobi), factorization 
          is never done on the GPU).

   References:
+  1. - T. Dupont, R. Kendall, and H. Rachford. An approximate factorization procedure for solving
   self adjoint elliptic difference equations. SIAM J. Numer. Anal., 5, 1968.
.  2. -  T.A. Oliphant. An implicit numerical method for solving two dimensional timedependent diffusion problems. Quart. Appl. Math., 19, 1961.
-  3. -  TONY F. CHAN AND HENK A. VAN DER VORST, APPROXIMATE AND INCOMPLETE FACTORIZATIONS, 
      Chapter in Parallel Numerical
      Algorithms, edited by D. Keyes, A. Semah, V. Venkatakrishnan, ICASE/LaRC Interdisciplinary Series in
      Science and Engineering, Kluwer.


.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC, PCSOR, MatOrderingType,
           PCFactorSetZeroPivot(), PCFactorSetShiftSetType(), PCFactorSetAmount(),
           PCFactorSetDropTolerance(),PCFactorSetFill(), PCFactorSetMatOrderingType(), PCFactorSetReuseOrdering(),
           PCFactorSetLevels(), PCFactorSetUseInPlace(), PCFactorSetAllowDiagonalFill(), PCFactorSetPivotInBlocks(),
           PCFactorGetAllowDiagonalFill(), PCFactorGetUseInPlace()

M*/

#undef __FUNCT__
#define __FUNCT__ "PCCreate_ILU"
PETSC_EXTERN PetscErrorCode PCCreate_ILU(PC pc)
{
  PetscErrorCode ierr;
  PC_ILU         *ilu;

  PetscFunctionBegin;
  ierr = PetscNewLog(pc,&ilu);CHKERRQ(ierr);

  ((PC_Factor*)ilu)->fact               = 0;
  ierr                                  = MatFactorInfoInitialize(&((PC_Factor*)ilu)->info);CHKERRQ(ierr);
  ((PC_Factor*)ilu)->factortype         = MAT_FACTOR_ILU;
  ((PC_Factor*)ilu)->info.levels        = 0.;
  ((PC_Factor*)ilu)->info.fill          = 1.0;
  ilu->col                              = 0;
  ilu->row                              = 0;
  ilu->inplace                          = PETSC_FALSE;
  ierr                                  = PetscStrallocpy(MATORDERINGNATURAL,(char**)&((PC_Factor*)ilu)->ordering);CHKERRQ(ierr);
  ilu->reuseordering                    = PETSC_FALSE;
  ((PC_Factor*)ilu)->info.dt            = PETSC_DEFAULT;
  ((PC_Factor*)ilu)->info.dtcount       = PETSC_DEFAULT;
  ((PC_Factor*)ilu)->info.dtcol         = PETSC_DEFAULT;
  ((PC_Factor*)ilu)->info.shifttype     = (PetscReal)MAT_SHIFT_NONE;
  ((PC_Factor*)ilu)->info.shiftamount   = 100.0*PETSC_MACHINE_EPSILON;
  ((PC_Factor*)ilu)->info.zeropivot     = 100.0*PETSC_MACHINE_EPSILON;
  ((PC_Factor*)ilu)->info.pivotinblocks = 1.0;
  ilu->reusefill                        = PETSC_FALSE;
  ((PC_Factor*)ilu)->info.diagonal_fill = 0.0;
  pc->data                              = (void*)ilu;

  pc->ops->reset               = PCReset_ILU;
  pc->ops->destroy             = PCDestroy_ILU;
  pc->ops->apply               = PCApply_ILU;
  pc->ops->applytranspose      = PCApplyTranspose_ILU;
  pc->ops->setup               = PCSetUp_ILU;
  pc->ops->setfromoptions      = PCSetFromOptions_ILU;
  pc->ops->getfactoredmatrix   = PCFactorGetMatrix_Factor;
  pc->ops->view                = PCView_ILU;
  pc->ops->applysymmetricleft  = PCApplySymmetricLeft_ILU;
  pc->ops->applysymmetricright = PCApplySymmetricRight_ILU;
  pc->ops->applyrichardson     = 0;

  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFactorSetZeroPivot_C",PCFactorSetZeroPivot_Factor);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFactorSetShiftType_C",PCFactorSetShiftType_Factor);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFactorSetShiftAmount_C",PCFactorSetShiftAmount_Factor);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFactorGetMatSolverPackage_C",PCFactorGetMatSolverPackage_Factor);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFactorSetMatSolverPackage_C",PCFactorSetMatSolverPackage_Factor);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFactorSetUpMatSolverPackage_C",PCFactorSetUpMatSolverPackage_Factor);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFactorSetDropTolerance_C",PCFactorSetDropTolerance_ILU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFactorSetFill_C",PCFactorSetFill_Factor);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFactorSetMatOrderingType_C",PCFactorSetMatOrderingType_Factor);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFactorSetReuseOrdering_C",PCFactorSetReuseOrdering_ILU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFactorSetReuseFill_C",PCFactorSetReuseFill_ILU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFactorSetLevels_C",PCFactorSetLevels_Factor);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFactorGetLevels_C",PCFactorGetLevels_Factor);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFactorSetUseInPlace_C",PCFactorSetUseInPlace_ILU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFactorGetUseInPlace_C",PCFactorGetUseInPlace_ILU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFactorSetAllowDiagonalFill_C",PCFactorSetAllowDiagonalFill_Factor);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFactorGetAllowDiagonalFill_C",PCFactorGetAllowDiagonalFill_Factor);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFactorSetPivotInBlocks_C",PCFactorSetPivotInBlocks_Factor);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFactorReorderForNonzeroDiagonal_C",PCFactorReorderForNonzeroDiagonal_ILU);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
