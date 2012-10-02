
/*
   Defines a direct factorization preconditioner for any Mat implementation
   Note: this need not be consided a preconditioner since it supplies
         a direct solver.
*/
#include <../src/ksp/pc/impls/factor/factor.h>         /*I "petscpc.h" I*/

typedef struct {
  PC_Factor        hdr;
  PetscReal        actualfill;       /* actual fill in factor */
  PetscBool        inplace;          /* flag indicating in-place factorization */
  IS               row,col;          /* index sets used for reordering */
  PetscBool        reuseordering;    /* reuses previous reordering computed */
  PetscBool        reusefill;        /* reuse fill from previous Cholesky */
} PC_Cholesky;

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetReuseOrdering_Cholesky"
PetscErrorCode  PCFactorSetReuseOrdering_Cholesky(PC pc,PetscBool  flag)
{
  PC_Cholesky *lu;
  
  PetscFunctionBegin;
  lu               = (PC_Cholesky*)pc->data;
  lu->reuseordering = flag;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetReuseFill_Cholesky"
PetscErrorCode  PCFactorSetReuseFill_Cholesky(PC pc,PetscBool  flag)
{
  PC_Cholesky *lu;
  
  PetscFunctionBegin;
  lu = (PC_Cholesky*)pc->data;
  lu->reusefill = flag;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "PCSetFromOptions_Cholesky"
static PetscErrorCode PCSetFromOptions_Cholesky(PC pc)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = PetscOptionsHead("Cholesky options");CHKERRQ(ierr);
    ierr = PCSetFromOptions_Factor(pc);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCView_Cholesky"
static PetscErrorCode PCView_Cholesky(PC pc,PetscViewer viewer)
{
  PC_Cholesky    *chol = (PC_Cholesky*)pc->data;
  PetscErrorCode ierr;
  PetscBool      iascii;
  
  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {  
    if (chol->inplace) {
      ierr = PetscViewerASCIIPrintf(viewer,"  Cholesky: in-place factorization\n");CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  Cholesky: out-of-place factorization\n");CHKERRQ(ierr);
    }
    
    if (chol->reusefill)    {ierr = PetscViewerASCIIPrintf(viewer,"  Reusing fill from past factorization\n");CHKERRQ(ierr);}
    if (chol->reuseordering) {ierr = PetscViewerASCIIPrintf(viewer,"  Reusing reordering from past factorization\n");CHKERRQ(ierr);}
  } 
  ierr = PCView_Factor(pc,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "PCSetUp_Cholesky"
static PetscErrorCode PCSetUp_Cholesky(PC pc)
{
  PetscErrorCode ierr;
  PetscBool      flg;
  PC_Cholesky    *dir = (PC_Cholesky*)pc->data;

  PetscFunctionBegin;
  if (dir->reusefill && pc->setupcalled) ((PC_Factor*)dir)->info.fill = dir->actualfill;
  
  if (dir->inplace) {
    if (dir->row && dir->col && (dir->row != dir->col)) {
      ierr = ISDestroy(&dir->row);CHKERRQ(ierr);
    }
    ierr = ISDestroy(&dir->col);CHKERRQ(ierr);
    ierr = MatGetOrdering(pc->pmat,((PC_Factor*)dir)->ordering,&dir->row,&dir->col);CHKERRQ(ierr);
    if (dir->col && (dir->row != dir->col)) {  /* only use row ordering for SBAIJ */
      ierr = ISDestroy(&dir->col);CHKERRQ(ierr);
    }
    if (dir->row) {ierr = PetscLogObjectParent(pc,dir->row);CHKERRQ(ierr);}
    ierr = MatCholeskyFactor(pc->pmat,dir->row,&((PC_Factor*)dir)->info);CHKERRQ(ierr);
    ((PC_Factor*)dir)->fact = pc->pmat;
  } else {
    MatInfo info;
    if (!pc->setupcalled) {
      ierr = MatGetOrdering(pc->pmat,((PC_Factor*)dir)->ordering,&dir->row,&dir->col);CHKERRQ(ierr);
      /* check if dir->row == dir->col */
      ierr = ISEqual(dir->row,dir->col,&flg);CHKERRQ(ierr);
      if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"row and column permutations must equal");
      ierr = ISDestroy(&dir->col);CHKERRQ(ierr); /* only pass one ordering into CholeskyFactor */

      flg  = PETSC_FALSE;
      ierr = PetscOptionsGetBool(((PetscObject)pc)->prefix,"-pc_factor_nonzeros_along_diagonal",&flg,PETSC_NULL);CHKERRQ(ierr);
      if (flg) {
        PetscReal tol = 1.e-10;
        ierr = PetscOptionsGetReal(((PetscObject)pc)->prefix,"-pc_factor_nonzeros_along_diagonal",&tol,PETSC_NULL);CHKERRQ(ierr);
        ierr = MatReorderForNonzeroDiagonal(pc->pmat,tol,dir->row,dir->row);CHKERRQ(ierr);
      }
      if (dir->row) {ierr = PetscLogObjectParent(pc,dir->row);CHKERRQ(ierr);}
      if (!((PC_Factor*)dir)->fact){
        ierr = MatGetFactor(pc->pmat,((PC_Factor*)dir)->solvertype,MAT_FACTOR_CHOLESKY,&((PC_Factor*)dir)->fact);CHKERRQ(ierr);
      }
      ierr = MatCholeskyFactorSymbolic(((PC_Factor*)dir)->fact,pc->pmat,dir->row,&((PC_Factor*)dir)->info);CHKERRQ(ierr);
      ierr = MatGetInfo(((PC_Factor*)dir)->fact,MAT_LOCAL,&info);CHKERRQ(ierr);
      dir->actualfill = info.fill_ratio_needed;
      ierr = PetscLogObjectParent(pc,((PC_Factor*)dir)->fact);CHKERRQ(ierr);
    } else if (pc->flag != SAME_NONZERO_PATTERN) {
      if (!dir->reuseordering) {
        ierr = ISDestroy(&dir->row);CHKERRQ(ierr);
        ierr = MatGetOrdering(pc->pmat,((PC_Factor*)dir)->ordering,&dir->row,&dir->col);CHKERRQ(ierr);
        ierr = ISDestroy(&dir->col);CHKERRQ(ierr); /* only use dir->row ordering in CholeskyFactor */

        flg  = PETSC_FALSE;
        ierr = PetscOptionsGetBool(((PetscObject)pc)->prefix,"-pc_factor_nonzeros_along_diagonal",&flg,PETSC_NULL);CHKERRQ(ierr);
        if (flg) {
          PetscReal tol = 1.e-10;
          ierr = PetscOptionsGetReal(((PetscObject)pc)->prefix,"-pc_factor_nonzeros_along_diagonal",&tol,PETSC_NULL);CHKERRQ(ierr);
          ierr = MatReorderForNonzeroDiagonal(pc->pmat,tol,dir->row,dir->row);CHKERRQ(ierr);
        }
        if (dir->row) {ierr = PetscLogObjectParent(pc,dir->row);CHKERRQ(ierr);}
      }
      ierr = MatDestroy(&((PC_Factor*)dir)->fact);CHKERRQ(ierr);
      ierr = MatGetFactor(pc->pmat,((PC_Factor*)dir)->solvertype,MAT_FACTOR_CHOLESKY,&((PC_Factor*)dir)->fact);CHKERRQ(ierr);
      ierr = MatCholeskyFactorSymbolic(((PC_Factor*)dir)->fact,pc->pmat,dir->row,&((PC_Factor*)dir)->info);CHKERRQ(ierr);
      ierr = MatGetInfo(((PC_Factor*)dir)->fact,MAT_LOCAL,&info);CHKERRQ(ierr);
      dir->actualfill = info.fill_ratio_needed;
      ierr = PetscLogObjectParent(pc,((PC_Factor*)dir)->fact);CHKERRQ(ierr);
    }
    ierr = MatCholeskyFactorNumeric(((PC_Factor*)dir)->fact,pc->pmat,&((PC_Factor*)dir)->info);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCReset_Cholesky"
static PetscErrorCode PCReset_Cholesky(PC pc)
{
  PC_Cholesky    *dir = (PC_Cholesky*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!dir->inplace && ((PC_Factor*)dir)->fact) {ierr = MatDestroy(&((PC_Factor*)dir)->fact);CHKERRQ(ierr);}
  ierr = ISDestroy(&dir->row);CHKERRQ(ierr);
  ierr = ISDestroy(&dir->col);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCDestroy_Cholesky"
static PetscErrorCode PCDestroy_Cholesky(PC pc)
{
  PC_Cholesky    *dir = (PC_Cholesky*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCReset_Cholesky(pc);CHKERRQ(ierr);
  ierr = PetscFree(((PC_Factor*)dir)->ordering);CHKERRQ(ierr);
  ierr = PetscFree(((PC_Factor*)dir)->solvertype);CHKERRQ(ierr);
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCApply_Cholesky"
static PetscErrorCode PCApply_Cholesky(PC pc,Vec x,Vec y)
{
  PC_Cholesky    *dir = (PC_Cholesky*)pc->data;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  if (dir->inplace) {ierr = MatSolve(pc->pmat,x,y);CHKERRQ(ierr);}
  else              {ierr = MatSolve(((PC_Factor*)dir)->fact,x,y);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCApplyTranspose_Cholesky"
static PetscErrorCode PCApplyTranspose_Cholesky(PC pc,Vec x,Vec y)
{
  PC_Cholesky    *dir = (PC_Cholesky*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (dir->inplace) {ierr = MatSolveTranspose(pc->pmat,x,y);CHKERRQ(ierr);}
  else              {ierr = MatSolveTranspose(((PC_Factor*)dir)->fact,x,y);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/* -----------------------------------------------------------------------------------*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetUseInPlace_Cholesky"
PetscErrorCode  PCFactorSetUseInPlace_Cholesky(PC pc)
{
  PC_Cholesky *dir;

  PetscFunctionBegin;
  dir = (PC_Cholesky*)pc->data;
  dir->inplace = PETSC_TRUE;
  PetscFunctionReturn(0);
}
EXTERN_C_END

/* -----------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetReuseOrdering"
/*@
   PCFactorSetReuseOrdering - When similar matrices are factored, this
   causes the ordering computed in the first factor to be used for all
   following factors.

   Logically Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  flag - PETSC_TRUE to reuse else PETSC_FALSE

   Options Database Key:
.  -pc_factor_reuse_ordering - Activate PCFactorSetReuseOrdering()

   Level: intermediate

.keywords: PC, levels, reordering, factorization, incomplete, LU

.seealso: PCFactorSetReuseFill()
@*/
PetscErrorCode  PCFactorSetReuseOrdering(PC pc,PetscBool  flag)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveBool(pc,flag,2);
  ierr = PetscTryMethod(pc,"PCFactorSetReuseOrdering_C",(PC,PetscBool),(pc,flag));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*MC
   PCCHOLESKY - Uses a direct solver, based on Cholesky factorization, as a preconditioner

   Options Database Keys:
+  -pc_factor_reuse_ordering - Activate PCFactorSetReuseOrdering()
.  -pc_factor_mat_solver_package - Actives PCFactorSetMatSolverPackage() to choose the direct solver, like spooles
.  -pc_factor_reuse_fill - Activates PCFactorSetReuseFill()
.  -pc_factor_fill <fill> - Sets fill amount
.  -pc_factor_in_place - Activates in-place factorization
-  -pc_factor_mat_ordering_type <nd,rcm,...> - Sets ordering routine

   Notes: Not all options work for all matrix formats

   Level: beginner

   Concepts: Cholesky factorization, direct solver

   Notes: Usually this will compute an "exact" solution in one iteration and does 
          not need a Krylov method (i.e. you can use -ksp_type preonly, or 
          KSPSetType(ksp,KSPPREONLY) for the Krylov method

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC,
           PCILU, PCLU, PCICC, PCFactorSetReuseOrdering(), PCFactorSetReuseFill(), PCFactorGetMatrix(),
           PCFactorSetFill(), PCFactorSetShiftNonzero(), PCFactorSetShiftType(), PCFactorSetShiftAmount()
	   PCFactorSetUseInPlace(), PCFactorSetMatOrderingType()

M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCreate_Cholesky"
PetscErrorCode  PCCreate_Cholesky(PC pc)
{
  PetscErrorCode ierr;
  PC_Cholesky    *dir;

  PetscFunctionBegin;
  ierr = PetscNewLog(pc,PC_Cholesky,&dir);CHKERRQ(ierr);

  ((PC_Factor*)dir)->fact                   = 0;
  dir->inplace                = PETSC_FALSE;
  ierr = MatFactorInfoInitialize(&((PC_Factor*)dir)->info);CHKERRQ(ierr);
  ((PC_Factor*)dir)->factortype         = MAT_FACTOR_CHOLESKY;
  ((PC_Factor*)dir)->info.fill          = 5.0;
  ((PC_Factor*)dir)->info.shifttype     = (PetscReal) MAT_SHIFT_NONE;
  ((PC_Factor*)dir)->info.shiftamount   = 0.0;
  ((PC_Factor*)dir)->info.zeropivot     = 100.0*PETSC_MACHINE_EPSILON;
  ((PC_Factor*)dir)->info.pivotinblocks = 1.0;
  dir->col                    = 0;
  dir->row                    = 0;
  ierr = PetscStrallocpy(MATORDERINGNATURAL,&((PC_Factor*)dir)->ordering);CHKERRQ(ierr);
  ierr = PetscStrallocpy(MATSOLVERPETSC,&((PC_Factor*)dir)->solvertype);CHKERRQ(ierr);
  dir->reusefill        = PETSC_FALSE;
  dir->reuseordering    = PETSC_FALSE;
  pc->data              = (void*)dir;

  pc->ops->destroy           = PCDestroy_Cholesky;
  pc->ops->reset             = PCReset_Cholesky;
  pc->ops->apply             = PCApply_Cholesky;
  pc->ops->applytranspose    = PCApplyTranspose_Cholesky;
  pc->ops->setup             = PCSetUp_Cholesky;
  pc->ops->setfromoptions    = PCSetFromOptions_Cholesky;
  pc->ops->view              = PCView_Cholesky;
  pc->ops->applyrichardson   = 0;
  pc->ops->getfactoredmatrix = PCFactorGetMatrix_Factor;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetUpMatSolverPackage_C","PCFactorSetUpMatSolverPackage_Factor",
                    PCFactorSetUpMatSolverPackage_Factor);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetMatSolverPackage_C","PCFactorSetMatSolverPackage_Factor",
                    PCFactorSetMatSolverPackage_Factor);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorGetMatSolverPackage_C","PCFactorGetMatSolverPackage_Factor",
                    PCFactorGetMatSolverPackage_Factor);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetZeroPivot_C","PCFactorSetZeroPivot_Factor",
                    PCFactorSetZeroPivot_Factor);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetShiftType_C","PCFactorSetShiftType_Factor",
                    PCFactorSetShiftType_Factor);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetShiftAmount_C","PCFactorSetShiftAmount_Factor",
                    PCFactorSetShiftAmount_Factor);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetFill_C","PCFactorSetFill_Factor",
                    PCFactorSetFill_Factor);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetUseInPlace_C","PCFactorSetUseInPlace_Cholesky",
                    PCFactorSetUseInPlace_Cholesky);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetMatOrderingType_C","PCFactorSetMatOrderingType_Factor",
                    PCFactorSetMatOrderingType_Factor);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetReuseOrdering_C","PCFactorSetReuseOrdering_Cholesky",
                    PCFactorSetReuseOrdering_Cholesky);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetReuseFill_C","PCFactorSetReuseFill_Cholesky",
                    PCFactorSetReuseFill_Cholesky);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
