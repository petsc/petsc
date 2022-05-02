
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

  PetscFunctionBegin;
  if (!ilu->hdr.inplace) PetscCall(MatDestroy(&((PC_Factor*)ilu)->fact));
  if (ilu->row && ilu->col && ilu->row != ilu->col) PetscCall(ISDestroy(&ilu->row));
  PetscCall(ISDestroy(&ilu->col));
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
  PetscInt       itmp;
  PetscBool      flg,set;
  PC_ILU         *ilu = (PC_ILU*)pc->data;
  PetscReal      tol;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject,"ILU Options");
  PetscCall(PCSetFromOptions_Factor(PetscOptionsObject,pc));

  PetscCall(PetscOptionsInt("-pc_factor_levels","levels of fill","PCFactorSetLevels",(PetscInt)((PC_Factor*)ilu)->info.levels,&itmp,&flg));
  if (flg) ((PC_Factor*)ilu)->info.levels = itmp;

  PetscCall(PetscOptionsBool("-pc_factor_diagonal_fill","Allow fill into empty diagonal entry","PCFactorSetAllowDiagonalFill",((PC_Factor*)ilu)->info.diagonal_fill ? PETSC_TRUE : PETSC_FALSE,&flg,&set));
  if (set) ((PC_Factor*)ilu)->info.diagonal_fill = (PetscReal) flg;
  PetscCall(PetscOptionsName("-pc_factor_nonzeros_along_diagonal","Reorder to remove zeros from diagonal","PCFactorReorderForNonzeroDiagonal",&flg));
  if (flg) {
    tol  = PETSC_DECIDE;
    PetscCall(PetscOptionsReal("-pc_factor_nonzeros_along_diagonal","Reorder to remove zeros from diagonal","PCFactorReorderForNonzeroDiagonal",ilu->nonzerosalongdiagonaltol,&tol,NULL));
    PetscCall(PCFactorReorderForNonzeroDiagonal(pc,tol));
  }

  PetscOptionsHeadEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetUp_ILU(PC pc)
{
  PC_ILU                 *ilu = (PC_ILU*)pc->data;
  MatInfo                info;
  PetscBool              flg;
  MatSolverType          stype;
  MatFactorError         err;
  const char             *prefix;

  PetscFunctionBegin;
  PetscCall(PCGetOptionsPrefix(pc,&prefix));
  PetscCall(MatSetOptionsPrefix(pc->pmat,prefix));
  pc->failedreason = PC_NOERROR;
  /* ugly hack to change default, since it is not support by some matrix types */
  if (((PC_Factor*)ilu)->info.shifttype == (PetscReal)MAT_SHIFT_NONZERO) {
    PetscCall(PetscObjectTypeCompare((PetscObject)pc->pmat,MATSEQAIJ,&flg));
    if (!flg) {
      PetscCall(PetscObjectTypeCompare((PetscObject)pc->pmat,MATMPIAIJ,&flg));
      if (!flg) {
        ((PC_Factor*)ilu)->info.shifttype = (PetscReal)MAT_SHIFT_INBLOCKS;
        PetscCall(PetscInfo(pc,"Changing shift type from NONZERO to INBLOCKS because block matrices do not support NONZERO\n"));
      }
    }
  }

  PetscCall(MatSetErrorIfFailure(pc->pmat,pc->erroriffailure));
  if (ilu->hdr.inplace) {
    if (!pc->setupcalled) {

      /* In-place factorization only makes sense with the natural ordering,
         so we only need to get the ordering once, even if nonzero structure changes */
      /* Should not get the ordering if the factorization routine does not use it, but do not yet have access to the factor matrix */
      PetscCall(PCFactorSetDefaultOrdering_Factor(pc));
      PetscCall(MatDestroy(&((PC_Factor*)ilu)->fact));
      PetscCall(MatGetOrdering(pc->pmat,((PC_Factor*)ilu)->ordering,&ilu->row,&ilu->col));
      if (ilu->row) PetscCall(PetscLogObjectParent((PetscObject)pc,(PetscObject)ilu->row));
      if (ilu->col) PetscCall(PetscLogObjectParent((PetscObject)pc,(PetscObject)ilu->col));
    }

    /* In place ILU only makes sense with fill factor of 1.0 because
       cannot have levels of fill */
    ((PC_Factor*)ilu)->info.fill          = 1.0;
    ((PC_Factor*)ilu)->info.diagonal_fill = 0.0;

    PetscCall(MatILUFactor(pc->pmat,ilu->row,ilu->col,&((PC_Factor*)ilu)->info));
    PetscCall(MatFactorGetError(pc->pmat,&err));
    if (err) { /* Factor() fails */
      pc->failedreason = (PCFailedReason)err;
      PetscFunctionReturn(0);
    }

    ((PC_Factor*)ilu)->fact = pc->pmat;
    /* must update the pc record of the matrix state or the PC will attempt to run PCSetUp() yet again */
    PetscCall(PetscObjectStateGet((PetscObject)pc->pmat,&pc->matstate));
  } else {
    if (!pc->setupcalled) {
      /* first time in so compute reordering and symbolic factorization */
      PetscBool canuseordering;
      if (!((PC_Factor*)ilu)->fact) {
        PetscCall(MatGetFactor(pc->pmat,((PC_Factor*)ilu)->solvertype,MAT_FACTOR_ILU,&((PC_Factor*)ilu)->fact));
        PetscCall(PetscLogObjectParent((PetscObject)pc,(PetscObject)((PC_Factor*)ilu)->fact));
      }
      PetscCall(MatFactorGetCanUseOrdering(((PC_Factor*)ilu)->fact,&canuseordering));
      if (canuseordering) {
        PetscCall(PCFactorSetDefaultOrdering_Factor(pc));
        PetscCall(MatGetOrdering(pc->pmat,((PC_Factor*)ilu)->ordering,&ilu->row,&ilu->col));
        PetscCall(PetscLogObjectParent((PetscObject)pc,(PetscObject)ilu->row));
        PetscCall(PetscLogObjectParent((PetscObject)pc,(PetscObject)ilu->col));
        /*  Remove zeros along diagonal?     */
        if (ilu->nonzerosalongdiagonal) {
          PetscCall(MatReorderForNonzeroDiagonal(pc->pmat,ilu->nonzerosalongdiagonaltol,ilu->row,ilu->col));
        }
      }
      PetscCall(MatILUFactorSymbolic(((PC_Factor*)ilu)->fact,pc->pmat,ilu->row,ilu->col,&((PC_Factor*)ilu)->info));
      PetscCall(MatGetInfo(((PC_Factor*)ilu)->fact,MAT_LOCAL,&info));
      ilu->hdr.actualfill = info.fill_ratio_needed;
    } else if (pc->flag != SAME_NONZERO_PATTERN) {
      if (!ilu->hdr.reuseordering) {
        PetscBool canuseordering;
        PetscCall(MatDestroy(&((PC_Factor*)ilu)->fact));
        PetscCall(MatGetFactor(pc->pmat,((PC_Factor*)ilu)->solvertype,MAT_FACTOR_ILU,&((PC_Factor*)ilu)->fact));
        PetscCall(PetscLogObjectParent((PetscObject)pc,(PetscObject)((PC_Factor*)ilu)->fact));
        PetscCall(MatFactorGetCanUseOrdering(((PC_Factor*)ilu)->fact,&canuseordering));
        if (canuseordering) {
          /* compute a new ordering for the ILU */
          PetscCall(ISDestroy(&ilu->row));
          PetscCall(ISDestroy(&ilu->col));
          PetscCall(PCFactorSetDefaultOrdering_Factor(pc));
          PetscCall(MatGetOrdering(pc->pmat,((PC_Factor*)ilu)->ordering,&ilu->row,&ilu->col));
          PetscCall(PetscLogObjectParent((PetscObject)pc,(PetscObject)ilu->row));
          PetscCall(PetscLogObjectParent((PetscObject)pc,(PetscObject)ilu->col));
          /*  Remove zeros along diagonal?     */
          if (ilu->nonzerosalongdiagonal) {
            PetscCall(MatReorderForNonzeroDiagonal(pc->pmat,ilu->nonzerosalongdiagonaltol,ilu->row,ilu->col));
          }
        }
      }
      PetscCall(MatILUFactorSymbolic(((PC_Factor*)ilu)->fact,pc->pmat,ilu->row,ilu->col,&((PC_Factor*)ilu)->info));
      PetscCall(MatGetInfo(((PC_Factor*)ilu)->fact,MAT_LOCAL,&info));
      ilu->hdr.actualfill = info.fill_ratio_needed;
    }
    PetscCall(MatFactorGetError(((PC_Factor*)ilu)->fact,&err));
    if (err) { /* FactorSymbolic() fails */
      pc->failedreason = (PCFailedReason)err;
      PetscFunctionReturn(0);
    }

    PetscCall(MatLUFactorNumeric(((PC_Factor*)ilu)->fact,pc->pmat,&((PC_Factor*)ilu)->info));
    PetscCall(MatFactorGetError(((PC_Factor*)ilu)->fact,&err));
    if (err) { /* FactorNumeric() fails */
      pc->failedreason = (PCFailedReason)err;
    }
  }

  PetscCall(PCFactorGetMatSolverType(pc,&stype));
  if (!stype) {
    MatSolverType solverpackage;
    PetscCall(MatFactorGetSolverType(((PC_Factor*)ilu)->fact,&solverpackage));
    PetscCall(PCFactorSetMatSolverType(pc,solverpackage));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_ILU(PC pc)
{
  PC_ILU         *ilu = (PC_ILU*)pc->data;

  PetscFunctionBegin;
  PetscCall(PCReset_ILU(pc));
  PetscCall(PetscFree(((PC_Factor*)ilu)->solvertype));
  PetscCall(PetscFree(((PC_Factor*)ilu)->ordering));
  PetscCall(PetscFree(pc->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApply_ILU(PC pc,Vec x,Vec y)
{
  PC_ILU         *ilu = (PC_ILU*)pc->data;

  PetscFunctionBegin;
  PetscCall(MatSolve(((PC_Factor*)ilu)->fact,x,y));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCMatApply_ILU(PC pc,Mat X,Mat Y)
{
  PC_ILU         *ilu = (PC_ILU*)pc->data;

  PetscFunctionBegin;
  PetscCall(MatMatSolve(((PC_Factor*)ilu)->fact,X,Y));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplyTranspose_ILU(PC pc,Vec x,Vec y)
{
  PC_ILU         *ilu = (PC_ILU*)pc->data;

  PetscFunctionBegin;
  PetscCall(MatSolveTranspose(((PC_Factor*)ilu)->fact,x,y));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplySymmetricLeft_ILU(PC pc,Vec x,Vec y)
{
  PC_ILU         *icc = (PC_ILU*)pc->data;

  PetscFunctionBegin;
  PetscCall(MatForwardSolve(((PC_Factor*)icc)->fact,x,y));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplySymmetricRight_ILU(PC pc,Vec x,Vec y)
{
  PC_ILU         *icc = (PC_ILU*)pc->data;

  PetscFunctionBegin;
  PetscCall(MatBackwardSolve(((PC_Factor*)icc)->fact,x,y));
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
+  * - T. Dupont, R. Kendall, and H. Rachford. An approximate factorization procedure for solving
   self adjoint elliptic difference equations. SIAM J. Numer. Anal., 5, 1968.
.  * -  T.A. Oliphant. An implicit numerical method for solving two dimensional timedependent diffusion problems. Quart. Appl. Math., 19, 1961.
-  * -  TONY F. CHAN AND HENK A. VAN DER VORST, APPROXIMATE AND INCOMPLETE FACTORIZATIONS,
      Chapter in Parallel Numerical
      Algorithms, edited by D. Keyes, A. Semah, V. Venkatakrishnan, ICASE/LaRC Interdisciplinary Series in
      Science and Engineering, Kluwer.

.seealso: `PCCreate()`, `PCSetType()`, `PCType`, `PC`, `PCSOR`, `MatOrderingType`,
          `PCFactorSetZeroPivot()`, `PCFactorSetShiftSetType()`, `PCFactorSetAmount()`,
          `PCFactorSetDropTolerance(),PCFactorSetFill()`, `PCFactorSetMatOrderingType()`, `PCFactorSetReuseOrdering()`,
          `PCFactorSetLevels()`, `PCFactorSetUseInPlace()`, `PCFactorSetAllowDiagonalFill()`, `PCFactorSetPivotInBlocks()`,
          `PCFactorGetAllowDiagonalFill()`, `PCFactorGetUseInPlace()`

M*/

PETSC_EXTERN PetscErrorCode PCCreate_ILU(PC pc)
{
  PC_ILU         *ilu;

  PetscFunctionBegin;
  PetscCall(PetscNewLog(pc,&ilu));
  pc->data = (void*)ilu;
  PetscCall(PCFactorInitialize(pc,MAT_FACTOR_ILU));

  ((PC_Factor*)ilu)->info.levels        = 0.;
  ((PC_Factor*)ilu)->info.fill          = 1.0;
  ilu->col                              = NULL;
  ilu->row                              = NULL;
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
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCFactorSetDropTolerance_C",PCFactorSetDropTolerance_ILU));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCFactorReorderForNonzeroDiagonal_C",PCFactorReorderForNonzeroDiagonal_ILU));
  PetscFunctionReturn(0);
}
