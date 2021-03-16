
/*
   Defines a ILU factorization preconditioner for any Mat implementation
*/
#include <../src/ksp/pc/impls/factor/ilu/ilu.h>     /*I "petscpc.h"  I*/

PetscErrorCode  PCFactorReorderForNonzeroDiagonal_ILU(PC pc,PetscReal z)
{
  PC_ILU *ilu = (PC_ILU*)pc->data;

  PetscFunctionBegin;
  ilu->nonzerosalongdiagonal = PETSC_TRUE;
  if (z == PETSC_DECIDE) ilu->nonzerosalongdiagonaltol = 1.e-10;
  else ilu->nonzerosalongdiagonaltol = z;
  PetscFunctionReturn(0);
}

PetscErrorCode PCReset_ILU(PC pc)
{
  PC_ILU         *ilu = (PC_ILU*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!ilu->hdr.inplace) {ierr = MatDestroy(&((PC_Factor*)ilu)->fact);CHKERRQ(ierr);}
  if (ilu->row && ilu->col && ilu->row != ilu->col) {ierr = ISDestroy(&ilu->row);CHKERRQ(ierr);}
  ierr = ISDestroy(&ilu->col);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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
    ierr = PetscOptionsReal("-pc_factor_nonzeros_along_diagonal","Reorder to remove zeros from diagonal","PCFactorReorderForNonzeroDiagonal",ilu->nonzerosalongdiagonaltol,&tol,NULL);CHKERRQ(ierr);
    ierr = PCFactorReorderForNonzeroDiagonal(pc,tol);CHKERRQ(ierr);
  }

  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetUp_ILU(PC pc)
{
  PetscErrorCode         ierr;
  PC_ILU                 *ilu = (PC_ILU*)pc->data;
  MatInfo                info;
  PetscBool              flg;
  MatSolverType          stype;
  MatFactorError         err;

  PetscFunctionBegin;
  pc->failedreason = PC_NOERROR;
  /* ugly hack to change default, since it is not support by some matrix types */
  if (((PC_Factor*)ilu)->info.shifttype == (PetscReal)MAT_SHIFT_NONZERO) {
    ierr = PetscObjectTypeCompare((PetscObject)pc->pmat,MATSEQAIJ,&flg);CHKERRQ(ierr);
    if (!flg) {
      ierr = PetscObjectTypeCompare((PetscObject)pc->pmat,MATMPIAIJ,&flg);CHKERRQ(ierr);
      if (!flg) {
        ((PC_Factor*)ilu)->info.shifttype = (PetscReal)MAT_SHIFT_INBLOCKS;
        ierr = PetscInfo(pc,"Changing shift type from NONZERO to INBLOCKS because block matrices do not support NONZERO\n");CHKERRQ(ierr);
      }
    }
  }

  ierr = MatSetErrorIfFailure(pc->pmat,pc->erroriffailure);CHKERRQ(ierr);
  if (ilu->hdr.inplace) {
    if (!pc->setupcalled) {

      /* In-place factorization only makes sense with the natural ordering,
         so we only need to get the ordering once, even if nonzero structure changes */
      /* Should not get the ordering if the factorization routine does not use it, but do not yet have access to the factor matrix */
      ierr = MatGetOrdering(pc->pmat,((PC_Factor*)ilu)->ordering,&ilu->row,&ilu->col);CHKERRQ(ierr);
      if (ilu->row) {ierr = PetscLogObjectParent((PetscObject)pc,(PetscObject)ilu->row);CHKERRQ(ierr);}
      if (ilu->col) {ierr = PetscLogObjectParent((PetscObject)pc,(PetscObject)ilu->col);CHKERRQ(ierr);}
    }

    /* In place ILU only makes sense with fill factor of 1.0 because
       cannot have levels of fill */
    ((PC_Factor*)ilu)->info.fill          = 1.0;
    ((PC_Factor*)ilu)->info.diagonal_fill = 0.0;

    ierr = MatILUFactor(pc->pmat,ilu->row,ilu->col,&((PC_Factor*)ilu)->info);CHKERRQ(ierr);
    ierr = MatFactorGetError(pc->pmat,&err);CHKERRQ(ierr);
    if (err) { /* Factor() fails */
      pc->failedreason = (PCFailedReason)err;
      PetscFunctionReturn(0);
    }

    ((PC_Factor*)ilu)->fact = pc->pmat;
    /* must update the pc record of the matrix state or the PC will attempt to run PCSetUp() yet again */
    ierr = PetscObjectStateGet((PetscObject)pc->pmat,&pc->matstate);CHKERRQ(ierr);
  } else {
    if (!pc->setupcalled) {
      /* first time in so compute reordering and symbolic factorization */
      PetscBool canuseordering;
      if (!((PC_Factor*)ilu)->fact) {
        ierr = MatGetFactor(pc->pmat,((PC_Factor*)ilu)->solvertype,MAT_FACTOR_ILU,&((PC_Factor*)ilu)->fact);CHKERRQ(ierr);
        ierr = PetscLogObjectParent((PetscObject)pc,(PetscObject)((PC_Factor*)ilu)->fact);CHKERRQ(ierr);
      }
      ierr = MatFactorGetCanUseOrdering(((PC_Factor*)ilu)->fact,&canuseordering);CHKERRQ(ierr);
      if (canuseordering) {
        ierr = MatGetOrdering(pc->pmat,((PC_Factor*)ilu)->ordering,&ilu->row,&ilu->col);CHKERRQ(ierr);
        ierr = PetscLogObjectParent((PetscObject)pc,(PetscObject)ilu->row);CHKERRQ(ierr);
        ierr = PetscLogObjectParent((PetscObject)pc,(PetscObject)ilu->col);CHKERRQ(ierr);
        /*  Remove zeros along diagonal?     */
        if (ilu->nonzerosalongdiagonal) {
          ierr = MatReorderForNonzeroDiagonal(pc->pmat,ilu->nonzerosalongdiagonaltol,ilu->row,ilu->col);CHKERRQ(ierr);
        }
      }
      ierr = MatILUFactorSymbolic(((PC_Factor*)ilu)->fact,pc->pmat,ilu->row,ilu->col,&((PC_Factor*)ilu)->info);CHKERRQ(ierr);
      ierr = MatGetInfo(((PC_Factor*)ilu)->fact,MAT_LOCAL,&info);CHKERRQ(ierr);
      ilu->hdr.actualfill = info.fill_ratio_needed;
    } else if (pc->flag != SAME_NONZERO_PATTERN) {
      if (!ilu->hdr.reuseordering) {
        PetscBool canuseordering;
        ierr = MatDestroy(&((PC_Factor*)ilu)->fact);CHKERRQ(ierr);
        ierr = MatGetFactor(pc->pmat,((PC_Factor*)ilu)->solvertype,MAT_FACTOR_ILU,&((PC_Factor*)ilu)->fact);CHKERRQ(ierr);
        ierr = PetscLogObjectParent((PetscObject)pc,(PetscObject)((PC_Factor*)ilu)->fact);CHKERRQ(ierr);
        ierr = MatFactorGetCanUseOrdering(((PC_Factor*)ilu)->fact,&canuseordering);CHKERRQ(ierr);
        if (canuseordering) {
          /* compute a new ordering for the ILU */
          ierr = ISDestroy(&ilu->row);CHKERRQ(ierr);
          ierr = ISDestroy(&ilu->col);CHKERRQ(ierr);
          ierr = MatGetOrdering(pc->pmat,((PC_Factor*)ilu)->ordering,&ilu->row,&ilu->col);CHKERRQ(ierr);
          ierr = PetscLogObjectParent((PetscObject)pc,(PetscObject)ilu->row);CHKERRQ(ierr);
          ierr = PetscLogObjectParent((PetscObject)pc,(PetscObject)ilu->col);CHKERRQ(ierr);
          /*  Remove zeros along diagonal?     */
          if (ilu->nonzerosalongdiagonal) {
            ierr = MatReorderForNonzeroDiagonal(pc->pmat,ilu->nonzerosalongdiagonaltol,ilu->row,ilu->col);CHKERRQ(ierr);
          }
        }
      }
      ierr = MatILUFactorSymbolic(((PC_Factor*)ilu)->fact,pc->pmat,ilu->row,ilu->col,&((PC_Factor*)ilu)->info);CHKERRQ(ierr);
      ierr = MatGetInfo(((PC_Factor*)ilu)->fact,MAT_LOCAL,&info);CHKERRQ(ierr);
      ilu->hdr.actualfill = info.fill_ratio_needed;
    }
    ierr = MatFactorGetError(((PC_Factor*)ilu)->fact,&err);CHKERRQ(ierr);
    if (err) { /* FactorSymbolic() fails */
      pc->failedreason = (PCFailedReason)err;
      PetscFunctionReturn(0);
    }

    ierr = MatLUFactorNumeric(((PC_Factor*)ilu)->fact,pc->pmat,&((PC_Factor*)ilu)->info);CHKERRQ(ierr);
    ierr = MatFactorGetError(((PC_Factor*)ilu)->fact,&err);CHKERRQ(ierr);
    if (err) { /* FactorNumeric() fails */
      pc->failedreason = (PCFailedReason)err;
    }
  }

  ierr = PCFactorGetMatSolverType(pc,&stype);CHKERRQ(ierr);
  if (!stype) {
    MatSolverType solverpackage;
    ierr = MatFactorGetSolverType(((PC_Factor*)ilu)->fact,&solverpackage);CHKERRQ(ierr);
    ierr = PCFactorSetMatSolverType(pc,solverpackage);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

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

static PetscErrorCode PCApply_ILU(PC pc,Vec x,Vec y)
{
  PC_ILU         *ilu = (PC_ILU*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSolve(((PC_Factor*)ilu)->fact,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCMatApply_ILU(PC pc,Mat X,Mat Y)
{
  PC_ILU         *ilu = (PC_ILU*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMatSolve(((PC_Factor*)ilu)->fact,X,Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplyTranspose_ILU(PC pc,Vec x,Vec y)
{
  PC_ILU         *ilu = (PC_ILU*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSolveTranspose(((PC_Factor*)ilu)->fact,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplySymmetricLeft_ILU(PC pc,Vec x,Vec y)
{
  PetscErrorCode ierr;
  PC_ILU         *icc = (PC_ILU*)pc->data;

  PetscFunctionBegin;
  ierr = MatForwardSolve(((PC_Factor*)icc)->fact,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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

   Notes:
    Only implemented for some matrix formats. (for parallel see PCHYPRE for hypre's ILU)

          For BAIJ matrices this implements a point block ILU

          The "symmetric" application of this preconditioner is not actually symmetric since L is not transpose(U)
          even when the matrix is not symmetric since the U stores the diagonals of the factorization.

          If you are using MATSEQAIJCUSPARSE matrices (or MATMPIAIJCUSPARSE matrices with block Jacobi), factorization
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

PETSC_EXTERN PetscErrorCode PCCreate_ILU(PC pc)
{
  PetscErrorCode ierr;
  PC_ILU         *ilu;

  PetscFunctionBegin;
  ierr     = PetscNewLog(pc,&ilu);CHKERRQ(ierr);
  pc->data = (void*)ilu;
  ierr     = PCFactorInitialize(pc);CHKERRQ(ierr);

  ((PC_Factor*)ilu)->factortype         = MAT_FACTOR_ILU;
  ((PC_Factor*)ilu)->info.levels        = 0.;
  ((PC_Factor*)ilu)->info.fill          = 1.0;
  ilu->col                              = NULL;
  ilu->row                              = NULL;
  ierr                                  = PetscStrallocpy(MATORDERINGNATURAL,(char**)&((PC_Factor*)ilu)->ordering);CHKERRQ(ierr);
  ((PC_Factor*)ilu)->info.dt            = PETSC_DEFAULT;
  ((PC_Factor*)ilu)->info.dtcount       = PETSC_DEFAULT;
  ((PC_Factor*)ilu)->info.dtcol         = PETSC_DEFAULT;

  pc->ops->reset               = PCReset_ILU;
  pc->ops->destroy             = PCDestroy_ILU;
  pc->ops->apply               = PCApply_ILU;
  pc->ops->matapply            = PCMatApply_ILU;
  pc->ops->applytranspose      = PCApplyTranspose_ILU;
  pc->ops->setup               = PCSetUp_ILU;
  pc->ops->setfromoptions      = PCSetFromOptions_ILU;
  pc->ops->view                = PCView_Factor;
  pc->ops->applysymmetricleft  = PCApplySymmetricLeft_ILU;
  pc->ops->applysymmetricright = PCApplySymmetricRight_ILU;
  pc->ops->applyrichardson     = NULL;
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFactorSetDropTolerance_C",PCFactorSetDropTolerance_ILU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCFactorReorderForNonzeroDiagonal_C",PCFactorReorderForNonzeroDiagonal_ILU);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
