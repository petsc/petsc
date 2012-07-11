
#include <../src/ksp/pc/impls/factor/icc/icc.h>   /*I "petscpc.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "PCSetup_ICC"
static PetscErrorCode PCSetup_ICC(PC pc)
{
  PC_ICC         *icc = (PC_ICC*)pc->data;
  IS             perm,cperm;
  PetscErrorCode ierr;
  MatInfo        info;

  PetscFunctionBegin;
  ierr = MatGetOrdering(pc->pmat, ((PC_Factor*)icc)->ordering,&perm,&cperm);CHKERRQ(ierr);

  if (!pc->setupcalled) {
    if (!((PC_Factor*)icc)->fact){
      ierr = MatGetFactor(pc->pmat,((PC_Factor*)icc)->solvertype,MAT_FACTOR_ICC,& ((PC_Factor*)icc)->fact);CHKERRQ(ierr);
    }
    ierr = MatICCFactorSymbolic(((PC_Factor*)icc)->fact,pc->pmat,perm,&((PC_Factor*)icc)->info);CHKERRQ(ierr);
  } else if (pc->flag != SAME_NONZERO_PATTERN) {
    ierr = MatDestroy(&((PC_Factor*)icc)->fact);CHKERRQ(ierr);
    ierr = MatGetFactor(pc->pmat,((PC_Factor*)icc)->solvertype,MAT_FACTOR_ICC,&((PC_Factor*)icc)->fact);CHKERRQ(ierr);
    ierr = MatICCFactorSymbolic(((PC_Factor*)icc)->fact,pc->pmat,perm,&((PC_Factor*)icc)->info);CHKERRQ(ierr);
  }
  ierr = MatGetInfo(((PC_Factor*)icc)->fact,MAT_LOCAL,&info);CHKERRQ(ierr);
  icc->actualfill = info.fill_ratio_needed;

  ierr = ISDestroy(&cperm);CHKERRQ(ierr);
  ierr = ISDestroy(&perm);CHKERRQ(ierr);
  ierr = MatCholeskyFactorNumeric(((PC_Factor*)icc)->fact,pc->pmat,&((PC_Factor*)icc)->info);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCReset_ICC"
static PetscErrorCode PCReset_ICC(PC pc)
{
  PC_ICC         *icc = (PC_ICC*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDestroy(&((PC_Factor*)icc)->fact);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCDestroy_ICC"
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

#undef __FUNCT__  
#define __FUNCT__ "PCApply_ICC"
static PetscErrorCode PCApply_ICC(PC pc,Vec x,Vec y)
{
  PC_ICC         *icc = (PC_ICC*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSolve(((PC_Factor*)icc)->fact,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);  
}

#undef __FUNCT__  
#define __FUNCT__ "PCApplySymmetricLeft_ICC"
static PetscErrorCode PCApplySymmetricLeft_ICC(PC pc,Vec x,Vec y)
{
  PetscErrorCode ierr;
  PC_ICC         *icc = (PC_ICC*)pc->data;

  PetscFunctionBegin;
  ierr = MatForwardSolve(((PC_Factor*)icc)->fact,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCApplySymmetricRight_ICC"
static PetscErrorCode PCApplySymmetricRight_ICC(PC pc,Vec x,Vec y)
{
  PetscErrorCode ierr;
  PC_ICC         *icc = (PC_ICC*)pc->data;

  PetscFunctionBegin;
  ierr = MatBackwardSolve(((PC_Factor*)icc)->fact,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSetFromOptions_ICC"
static PetscErrorCode PCSetFromOptions_ICC(PC pc)
{
  PC_ICC         *icc = (PC_ICC*)pc->data;
  PetscBool      flg;
  /* PetscReal      dt[3];*/
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("ICC Options");CHKERRQ(ierr);
    ierr = PCSetFromOptions_Factor(pc);CHKERRQ(ierr);

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

#undef __FUNCT__  
#define __FUNCT__ "PCView_ICC"
static PetscErrorCode PCView_ICC(PC pc,PetscViewer viewer)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = PCView_Factor(pc,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
extern PetscErrorCode  PCFactorSetDropTolerance_ILU(PC,PetscReal,PetscReal,PetscInt);
EXTERN_C_END

/*MC
     PCICC - Incomplete Cholesky factorization preconditioners.

   Options Database Keys:
+  -pc_factor_levels <k> - number of levels of fill for ICC(k)
.  -pc_factor_in_place - only for ICC(0) with natural ordering, reuses the space of the matrix for
                      its factorization (overwrites original matrix)
.  -pc_factor_fill <nfill> - expected amount of fill in factored matrix compared to original matrix, nfill > 1
-  -pc_factor_mat_ordering_type <natural,nd,1wd,rcm,qmd> - set the row/column ordering of the factored matrix

   Level: beginner

  Concepts: incomplete Cholesky factorization

   Notes: Only implemented for some matrix formats. Not implemented in parallel.

          For BAIJ matrices this implements a point block ICC.

          The Manteuffel shift is only implemented for matrices with block size 1

          By default, the Manteuffel is applied (for matrices with block size 1). Call PCFactorSetShiftType(pc,MAT_SHIFT_POSITIVE_DEFINITE);
          to turn off the shift.

   References:
   Review article: APPROXIMATE AND INCOMPLETE FACTORIZATIONS, TONY F. CHAN AND HENK A. VAN DER VORST
      http://igitur-archive.library.uu.nl/math/2001-0621-115821/proc.pdf chapter in Parallel Numerical
      Algorithms, edited by D. Keyes, A. Semah, V. Venkatakrishnan, ICASE/LaRC Interdisciplinary Series in
      Science and Engineering, Kluwer, pp. 167--202.


.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC, PCSOR, MatOrderingType,
           PCFactorSetZeroPivot(), PCFactorSetShiftType(), PCFactorSetShiftAmount(), 
           PCFactorSetFill(), PCFactorSetMatOrderingType(), PCFactorSetReuseOrdering(), 
           PCFactorSetLevels()

M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCreate_ICC"
PetscErrorCode  PCCreate_ICC(PC pc)
{
  PetscErrorCode ierr;
  PC_ICC         *icc;

  PetscFunctionBegin;
  ierr = PetscNewLog(pc,PC_ICC,&icc);CHKERRQ(ierr);

  ((PC_Factor*)icc)->fact	          = 0;
  ierr = PetscStrallocpy(MATORDERINGNATURAL,&((PC_Factor*)icc)->ordering);CHKERRQ(ierr);
  ierr = PetscStrallocpy(MATSOLVERPETSC,&((PC_Factor*)icc)->solvertype);CHKERRQ(ierr);
  ierr = MatFactorInfoInitialize(&((PC_Factor*)icc)->info);CHKERRQ(ierr);
  ((PC_Factor*)icc)->factortype         = MAT_FACTOR_ICC;
  ((PC_Factor*)icc)->info.levels	= 0.;
  ((PC_Factor*)icc)->info.fill          = 1.0;
  icc->implctx            = 0;

  ((PC_Factor*)icc)->info.dtcol       = PETSC_DEFAULT;
  ((PC_Factor*)icc)->info.shifttype   = (PetscReal) MAT_SHIFT_POSITIVE_DEFINITE;
  ((PC_Factor*)icc)->info.shiftamount = 100.0*PETSC_MACHINE_EPSILON;
  ((PC_Factor*)icc)->info.zeropivot   = 100.0*PETSC_MACHINE_EPSILON;

  pc->data	               = (void*)icc;
  pc->ops->apply	       = PCApply_ICC;
  pc->ops->applytranspose      = PCApply_ICC;
  pc->ops->setup               = PCSetup_ICC;
  pc->ops->reset  	       = PCReset_ICC;
  pc->ops->destroy	       = PCDestroy_ICC;
  pc->ops->setfromoptions      = PCSetFromOptions_ICC;
  pc->ops->view                = PCView_ICC;
  pc->ops->getfactoredmatrix   = PCFactorGetMatrix_Factor;
  pc->ops->applysymmetricleft  = PCApplySymmetricLeft_ICC;
  pc->ops->applysymmetricright = PCApplySymmetricRight_ICC;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetUpMatSolverPackage_C","PCFactorSetUpMatSolverPackage_Factor",
                    PCFactorSetUpMatSolverPackage_Factor);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorGetMatSolverPackage_C","PCFactorGetMatSolverPackage_Factor",
                    PCFactorGetMatSolverPackage_Factor);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetZeroPivot_C","PCFactorSetZeroPivot_Factor",
                    PCFactorSetZeroPivot_Factor);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetShiftType_C","PCFactorSetShiftType_Factor",
                    PCFactorSetShiftType_Factor);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetShiftAmount_C","PCFactorSetShiftAmount_Factor",
                    PCFactorSetShiftAmount_Factor);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetLevels_C","PCFactorSetLevels_Factor",
                    PCFactorSetLevels_Factor);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetFill_C","PCFactorSetFill_Factor",
                    PCFactorSetFill_Factor);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetMatOrderingType_C","PCFactorSetMatOrderingType_Factor",
                    PCFactorSetMatOrderingType_Factor);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetMatSolverPackage_C","PCFactorSetMatSolverPackage_Factor",
                    PCFactorSetMatSolverPackage_Factor);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetDropTolerance_C","PCFactorSetDropTolerance_ILU",
                    PCFactorSetDropTolerance_ILU);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END


