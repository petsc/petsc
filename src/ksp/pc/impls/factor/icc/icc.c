
#include <../src/ksp/pc/impls/factor/icc/icc.h>   /*I "petscpc.h" I*/

static PetscErrorCode PCSetUp_ICC(PC pc)
{
  PC_ICC                 *icc = (PC_ICC*)pc->data;
  IS                     perm = NULL,cperm = NULL;
  PetscErrorCode         ierr;
  MatInfo                info;
  MatSolverType          stype;
  MatFactorError         err;

  PetscFunctionBegin;
  pc->failedreason = PC_NOERROR;

  ierr = MatSetErrorIfFailure(pc->pmat,pc->erroriffailure);CHKERRQ(ierr);
  if (!pc->setupcalled) {
    if (!((PC_Factor*)icc)->fact) {
      PetscBool useordering;
      ierr = MatGetFactor(pc->pmat,((PC_Factor*)icc)->solvertype,MAT_FACTOR_ICC,&((PC_Factor*)icc)->fact);CHKERRQ(ierr);
      ierr = MatFactorGetUseOrdering(((PC_Factor*)icc)->fact,&useordering);CHKERRQ(ierr);
      if (useordering) {
        ierr = MatGetOrdering(pc->pmat, ((PC_Factor*)icc)->ordering,&perm,&cperm);CHKERRQ(ierr);
      }
    }
    ierr = MatICCFactorSymbolic(((PC_Factor*)icc)->fact,pc->pmat,perm,&((PC_Factor*)icc)->info);CHKERRQ(ierr);
  } else if (pc->flag != SAME_NONZERO_PATTERN) {
    PetscBool useordering;
    ierr = MatDestroy(&((PC_Factor*)icc)->fact);CHKERRQ(ierr);
    ierr = MatGetFactor(pc->pmat,((PC_Factor*)icc)->solvertype,MAT_FACTOR_ICC,&((PC_Factor*)icc)->fact);CHKERRQ(ierr);
    ierr = MatFactorGetUseOrdering(((PC_Factor*)icc)->fact,&useordering);CHKERRQ(ierr);
    if (useordering) {
      ierr = MatGetOrdering(pc->pmat, ((PC_Factor*)icc)->ordering,&perm,&cperm);CHKERRQ(ierr);
    }
    ierr = MatICCFactorSymbolic(((PC_Factor*)icc)->fact,pc->pmat,perm,&((PC_Factor*)icc)->info);CHKERRQ(ierr);
  }
  ierr                = MatGetInfo(((PC_Factor*)icc)->fact,MAT_LOCAL,&info);CHKERRQ(ierr);
  icc->hdr.actualfill = info.fill_ratio_needed;

  ierr = ISDestroy(&cperm);CHKERRQ(ierr);
  ierr = ISDestroy(&perm);CHKERRQ(ierr);

  ierr = MatFactorGetError(((PC_Factor*)icc)->fact,&err);CHKERRQ(ierr);
  if (err) { /* FactorSymbolic() fails */
    pc->failedreason = (PCFailedReason)err;
    PetscFunctionReturn(0);
  }

  ierr = MatCholeskyFactorNumeric(((PC_Factor*)icc)->fact,pc->pmat,&((PC_Factor*)icc)->info);CHKERRQ(ierr);
  ierr = MatFactorGetError(((PC_Factor*)icc)->fact,&err);CHKERRQ(ierr);
  if (err) { /* FactorNumeric() fails */
    pc->failedreason = (PCFailedReason)err;
  }

  ierr = PCFactorGetMatSolverType(pc,&stype);CHKERRQ(ierr);
  if (!stype) {
    MatSolverType solverpackage;
    ierr = MatFactorGetSolverType(((PC_Factor*)icc)->fact,&solverpackage);CHKERRQ(ierr);
    ierr = PCFactorSetMatSolverType(pc,solverpackage);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCReset_ICC(PC pc)
{
  PC_ICC         *icc = (PC_ICC*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDestroy(&((PC_Factor*)icc)->fact);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_ICC(PC pc)
{
  PC_ICC         *icc = (PC_ICC*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCReset_ICC(pc);CHKERRQ(ierr);
  ierr = PetscFree(((PC_Factor*)icc)->ordering);CHKERRQ(ierr);
  ierr = PetscFree(((PC_Factor*)icc)->solvertype);CHKERRQ(ierr);
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApply_ICC(PC pc,Vec x,Vec y)
{
  PC_ICC         *icc = (PC_ICC*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSolve(((PC_Factor*)icc)->fact,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCMatApply_ICC(PC pc,Mat X,Mat Y)
{
  PC_ICC         *icc = (PC_ICC*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMatSolve(((PC_Factor*)icc)->fact,X,Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplySymmetricLeft_ICC(PC pc,Vec x,Vec y)
{
  PetscErrorCode ierr;
  PC_ICC         *icc = (PC_ICC*)pc->data;

  PetscFunctionBegin;
  ierr = MatForwardSolve(((PC_Factor*)icc)->fact,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplySymmetricRight_ICC(PC pc,Vec x,Vec y)
{
  PetscErrorCode ierr;
  PC_ICC         *icc = (PC_ICC*)pc->data;

  PetscFunctionBegin;
  ierr = MatBackwardSolve(((PC_Factor*)icc)->fact,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetFromOptions_ICC(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PC_ICC         *icc = (PC_ICC*)pc->data;
  PetscBool      flg;
  PetscErrorCode ierr;
  /* PetscReal      dt[3];*/

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"ICC Options");CHKERRQ(ierr);
  ierr = PCSetFromOptions_Factor(PetscOptionsObject,pc);CHKERRQ(ierr);

  ierr = PetscOptionsReal("-pc_factor_levels","levels of fill","PCFactorSetLevels",((PC_Factor*)icc)->info.levels,&((PC_Factor*)icc)->info.levels,&flg);CHKERRQ(ierr);
  /*dt[0] = ((PC_Factor*)icc)->info.dt;
  dt[1] = ((PC_Factor*)icc)->info.dtcol;
  dt[2] = ((PC_Factor*)icc)->info.dtcount;
  PetscInt       dtmax = 3;
  ierr = PetscOptionsRealArray("-pc_factor_drop_tolerance","<dt,dtcol,maxrowcount>","PCFactorSetDropTolerance",dt,&dtmax,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PCFactorSetDropTolerance(pc,dt[0],dt[1],(PetscInt)dt[2]);CHKERRQ(ierr);
  }
  */
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

extern PetscErrorCode  PCFactorSetDropTolerance_ILU(PC,PetscReal,PetscReal,PetscInt);

/*MC
     PCICC - Incomplete Cholesky factorization preconditioners.

   Options Database Keys:
+  -pc_factor_levels <k> - number of levels of fill for ICC(k)
.  -pc_factor_in_place - only for ICC(0) with natural ordering, reuses the space of the matrix for
                      its factorization (overwrites original matrix)
.  -pc_factor_fill <nfill> - expected amount of fill in factored matrix compared to original matrix, nfill > 1
-  -pc_factor_mat_ordering_type <natural,nd,1wd,rcm,qmd> - set the row/column ordering of the factored matrix

   Level: beginner

   Notes:
    Only implemented for some matrix formats. Not implemented in parallel.

          For BAIJ matrices this implements a point block ICC.

          The Manteuffel shift is only implemented for matrices with block size 1

          By default, the Manteuffel is applied (for matrices with block size 1). Call PCFactorSetShiftType(pc,MAT_SHIFT_POSITIVE_DEFINITE);
          to turn off the shift.

   References:
.  1. - TONY F. CHAN AND HENK A. VAN DER VORST, Review article: APPROXIMATE AND INCOMPLETE FACTORIZATIONS,
      Chapter in Parallel Numerical Algorithms, edited by D. Keyes, A. Semah, V. Venkatakrishnan, ICASE/LaRC Interdisciplinary Series in
      Science and Engineering, Kluwer.


.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC, PCSOR, MatOrderingType,
           PCFactorSetZeroPivot(), PCFactorSetShiftType(), PCFactorSetShiftAmount(),
           PCFactorSetFill(), PCFactorSetMatOrderingType(), PCFactorSetReuseOrdering(),
           PCFactorSetLevels()

M*/

PETSC_EXTERN PetscErrorCode PCCreate_ICC(PC pc)
{
  PetscErrorCode ierr;
  PC_ICC         *icc;

  PetscFunctionBegin;
  ierr     = PetscNewLog(pc,&icc);CHKERRQ(ierr);
  pc->data = (void*)icc;
  ierr     = PCFactorInitialize(pc);CHKERRQ(ierr);
  ierr     = PetscStrallocpy(MATORDERINGNATURAL,(char**)&((PC_Factor*)icc)->ordering);CHKERRQ(ierr);

  ((PC_Factor*)icc)->factortype     = MAT_FACTOR_ICC;
  ((PC_Factor*)icc)->info.fill      = 1.0;
  ((PC_Factor*)icc)->info.dtcol     = PETSC_DEFAULT;
  ((PC_Factor*)icc)->info.shifttype = (PetscReal) MAT_SHIFT_POSITIVE_DEFINITE;

  pc->ops->apply               = PCApply_ICC;
  pc->ops->matapply            = PCMatApply_ICC;
  pc->ops->applytranspose      = PCApply_ICC;
  pc->ops->setup               = PCSetUp_ICC;
  pc->ops->reset               = PCReset_ICC;
  pc->ops->destroy             = PCDestroy_ICC;
  pc->ops->setfromoptions      = PCSetFromOptions_ICC;
  pc->ops->view                = PCView_Factor;
  pc->ops->applysymmetricleft  = PCApplySymmetricLeft_ICC;
  pc->ops->applysymmetricright = PCApplySymmetricRight_ICC;
  PetscFunctionReturn(0);
}
