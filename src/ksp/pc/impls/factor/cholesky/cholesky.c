
/*
   Defines a direct factorization preconditioner for any Mat implementation
   Note: this need not be consided a preconditioner since it supplies
         a direct solver.
*/
#include <../src/ksp/pc/impls/factor/factor.h>         /*I "petscpc.h" I*/

typedef struct {
  PC_Factor hdr;
  IS        row,col;                 /* index sets used for reordering */
} PC_Cholesky;

static PetscErrorCode PCSetFromOptions_Cholesky(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"Cholesky options");CHKERRQ(ierr);
  ierr = PCSetFromOptions_Factor(PetscOptionsObject,pc);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetUp_Cholesky(PC pc)
{
  PetscErrorCode         ierr;
  PetscBool              flg;
  PC_Cholesky            *dir = (PC_Cholesky*)pc->data;
  MatSolverType          stype;
  MatFactorError         err;

  PetscFunctionBegin;
  pc->failedreason = PC_NOERROR;
  if (dir->hdr.reusefill && pc->setupcalled) ((PC_Factor*)dir)->info.fill = dir->hdr.actualfill;

  ierr = MatSetErrorIfFailure(pc->pmat,pc->erroriffailure);CHKERRQ(ierr);
  if (dir->hdr.inplace) {
    if (dir->row && dir->col && (dir->row != dir->col)) {
      ierr = ISDestroy(&dir->row);CHKERRQ(ierr);
    }
    ierr = ISDestroy(&dir->col);CHKERRQ(ierr);
    /* should only get reordering if the factor matrix uses it but cannot determine because MatGetFactor() not called */
    ierr = MatGetOrdering(pc->pmat,((PC_Factor*)dir)->ordering,&dir->row,&dir->col);CHKERRQ(ierr);
    if (dir->col && (dir->row != dir->col)) {  /* only use row ordering for SBAIJ */
      ierr = ISDestroy(&dir->col);CHKERRQ(ierr);
    }
    if (dir->row) {ierr = PetscLogObjectParent((PetscObject)pc,(PetscObject)dir->row);CHKERRQ(ierr);}
    ierr = MatCholeskyFactor(pc->pmat,dir->row,&((PC_Factor*)dir)->info);CHKERRQ(ierr);
    ierr = MatFactorGetError(pc->pmat,&err);CHKERRQ(ierr);
    if (err) { /* Factor() fails */
      pc->failedreason = (PCFailedReason)err;
      PetscFunctionReturn(0);
    }

    ((PC_Factor*)dir)->fact = pc->pmat;
  } else {
    MatInfo info;

    if (!pc->setupcalled) {
      PetscBool useordering;
      if (!((PC_Factor*)dir)->fact) {
        ierr = MatGetFactor(pc->pmat,((PC_Factor*)dir)->solvertype,MAT_FACTOR_CHOLESKY,&((PC_Factor*)dir)->fact);CHKERRQ(ierr);
        ierr = PetscLogObjectParent((PetscObject)pc,(PetscObject)((PC_Factor*)dir)->fact);CHKERRQ(ierr);
      }
      ierr = MatFactorGetUseOrdering(((PC_Factor*)dir)->fact,&useordering);CHKERRQ(ierr);
      if (useordering) {
        ierr = MatGetOrdering(pc->pmat,((PC_Factor*)dir)->ordering,&dir->row,&dir->col);CHKERRQ(ierr);
        /* check if dir->row == dir->col */
        ierr = ISEqual(dir->row,dir->col,&flg);CHKERRQ(ierr);
        if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"row and column permutations must equal");
        ierr = ISDestroy(&dir->col);CHKERRQ(ierr); /* only pass one ordering into CholeskyFactor */

        flg  = PETSC_FALSE;
        ierr = PetscOptionsGetBool(((PetscObject)pc)->options,((PetscObject)pc)->prefix,"-pc_factor_nonzeros_along_diagonal",&flg,NULL);CHKERRQ(ierr);
        if (flg) {
          PetscReal tol = 1.e-10;
          ierr = PetscOptionsGetReal(((PetscObject)pc)->options,((PetscObject)pc)->prefix,"-pc_factor_nonzeros_along_diagonal",&tol,NULL);CHKERRQ(ierr);
          ierr = MatReorderForNonzeroDiagonal(pc->pmat,tol,dir->row,dir->row);CHKERRQ(ierr);
        }
        ierr = PetscLogObjectParent((PetscObject)pc,(PetscObject)dir->row);CHKERRQ(ierr);
      }
      ierr                = MatCholeskyFactorSymbolic(((PC_Factor*)dir)->fact,pc->pmat,dir->row,&((PC_Factor*)dir)->info);CHKERRQ(ierr);
      ierr                = MatGetInfo(((PC_Factor*)dir)->fact,MAT_LOCAL,&info);CHKERRQ(ierr);
      dir->hdr.actualfill = info.fill_ratio_needed;
    } else if (pc->flag != SAME_NONZERO_PATTERN) {
      if (!dir->hdr.reuseordering) {
        PetscBool useordering;
        ierr = MatDestroy(&((PC_Factor*)dir)->fact);CHKERRQ(ierr);
        ierr = MatGetFactor(pc->pmat,((PC_Factor*)dir)->solvertype,MAT_FACTOR_CHOLESKY,&((PC_Factor*)dir)->fact);CHKERRQ(ierr);
        ierr = PetscLogObjectParent((PetscObject)pc,(PetscObject)((PC_Factor*)dir)->fact);CHKERRQ(ierr);
        ierr = MatFactorGetUseOrdering(((PC_Factor*)dir)->fact,&useordering);CHKERRQ(ierr);
        if (useordering) {
          ierr = ISDestroy(&dir->row);CHKERRQ(ierr);
          ierr = MatGetOrdering(pc->pmat,((PC_Factor*)dir)->ordering,&dir->row,&dir->col);CHKERRQ(ierr);
          ierr = ISDestroy(&dir->col);CHKERRQ(ierr); /* only use dir->row ordering in CholeskyFactor */

          flg  = PETSC_FALSE;
          ierr = PetscOptionsGetBool(((PetscObject)pc)->options,((PetscObject)pc)->prefix,"-pc_factor_nonzeros_along_diagonal",&flg,NULL);CHKERRQ(ierr);
          if (flg) {
            PetscReal tol = 1.e-10;
            ierr = PetscOptionsGetReal(((PetscObject)pc)->options,((PetscObject)pc)->prefix,"-pc_factor_nonzeros_along_diagonal",&tol,NULL);CHKERRQ(ierr);
            ierr = MatReorderForNonzeroDiagonal(pc->pmat,tol,dir->row,dir->row);CHKERRQ(ierr);
          }
          ierr = PetscLogObjectParent((PetscObject)pc,(PetscObject)dir->row);CHKERRQ(ierr);
        }
      }
      ierr                = MatCholeskyFactorSymbolic(((PC_Factor*)dir)->fact,pc->pmat,dir->row,&((PC_Factor*)dir)->info);CHKERRQ(ierr);
      ierr                = MatGetInfo(((PC_Factor*)dir)->fact,MAT_LOCAL,&info);CHKERRQ(ierr);
      dir->hdr.actualfill = info.fill_ratio_needed;
      ierr                = PetscLogObjectParent((PetscObject)pc,(PetscObject)((PC_Factor*)dir)->fact);CHKERRQ(ierr);
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

    ierr = MatCholeskyFactorNumeric(((PC_Factor*)dir)->fact,pc->pmat,&((PC_Factor*)dir)->info);CHKERRQ(ierr);
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

static PetscErrorCode PCReset_Cholesky(PC pc)
{
  PC_Cholesky    *dir = (PC_Cholesky*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!dir->hdr.inplace && ((PC_Factor*)dir)->fact) {ierr = MatDestroy(&((PC_Factor*)dir)->fact);CHKERRQ(ierr);}
  ierr = ISDestroy(&dir->row);CHKERRQ(ierr);
  ierr = ISDestroy(&dir->col);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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

static PetscErrorCode PCApply_Cholesky(PC pc,Vec x,Vec y)
{
  PC_Cholesky    *dir = (PC_Cholesky*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (dir->hdr.inplace) {
    ierr = MatSolve(pc->pmat,x,y);CHKERRQ(ierr);
  } else {
    ierr = MatSolve(((PC_Factor*)dir)->fact,x,y);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCMatApply_Cholesky(PC pc,Mat X,Mat Y)
{
  PC_Cholesky    *dir = (PC_Cholesky*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (dir->hdr.inplace) {
    ierr = MatMatSolve(pc->pmat,X,Y);CHKERRQ(ierr);
  } else {
    ierr = MatMatSolve(((PC_Factor*)dir)->fact,X,Y);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplySymmetricLeft_Cholesky(PC pc,Vec x,Vec y)
{
  PC_Cholesky    *dir = (PC_Cholesky*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (dir->hdr.inplace) {
    ierr = MatForwardSolve(pc->pmat,x,y);CHKERRQ(ierr);
  } else {
    ierr = MatForwardSolve(((PC_Factor*)dir)->fact,x,y);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplySymmetricRight_Cholesky(PC pc,Vec x,Vec y)
{
  PC_Cholesky    *dir = (PC_Cholesky*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (dir->hdr.inplace) {
    ierr = MatBackwardSolve(pc->pmat,x,y);CHKERRQ(ierr);
  } else {
    ierr = MatBackwardSolve(((PC_Factor*)dir)->fact,x,y);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplyTranspose_Cholesky(PC pc,Vec x,Vec y)
{
  PC_Cholesky    *dir = (PC_Cholesky*)pc->data;
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

/* -----------------------------------------------------------------------------------*/

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

.seealso: PCFactorSetReuseFill()
@*/
PetscErrorCode  PCFactorSetReuseOrdering(PC pc,PetscBool flag)
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
.  -pc_factor_mat_solver_type - Actives PCFactorSetMatSolverType() to choose the direct solver, like superlu
.  -pc_factor_reuse_fill - Activates PCFactorSetReuseFill()
.  -pc_factor_fill <fill> - Sets fill amount
.  -pc_factor_in_place - Activates in-place factorization
-  -pc_factor_mat_ordering_type <nd,rcm,...> - Sets ordering routine

   Notes:
    Not all options work for all matrix formats

   Level: beginner

   Notes:
    Usually this will compute an "exact" solution in one iteration and does
          not need a Krylov method (i.e. you can use -ksp_type preonly, or
          KSPSetType(ksp,KSPPREONLY) for the Krylov method

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC,
           PCILU, PCLU, PCICC, PCFactorSetReuseOrdering(), PCFactorSetReuseFill(), PCFactorGetMatrix(),
           PCFactorSetFill(), PCFactorSetShiftNonzero(), PCFactorSetShiftType(), PCFactorSetShiftAmount()
           PCFactorSetUseInPlace(), PCFactorGetUseInPlace(), PCFactorSetMatOrderingType()

M*/

PETSC_EXTERN PetscErrorCode PCCreate_Cholesky(PC pc)
{
  PetscErrorCode ierr;
  PC_Cholesky    *dir;

  PetscFunctionBegin;
  ierr     = PetscNewLog(pc,&dir);CHKERRQ(ierr);
  pc->data = (void*)dir;
  ierr     = PCFactorInitialize(pc);CHKERRQ(ierr);

  ((PC_Factor*)dir)->factortype         = MAT_FACTOR_CHOLESKY;
  ((PC_Factor*)dir)->info.fill          = 5.0;

  dir->col = NULL;
  dir->row = NULL;

  /* MATORDERINGNATURAL_OR_ND allows selecting type based on matrix type sbaij or aij */
  ierr = PetscStrallocpy(MATORDERINGNATURAL_OR_ND,(char**)&((PC_Factor*)dir)->ordering);CHKERRQ(ierr);

  pc->ops->destroy             = PCDestroy_Cholesky;
  pc->ops->reset               = PCReset_Cholesky;
  pc->ops->apply               = PCApply_Cholesky;
  pc->ops->matapply            = PCMatApply_Cholesky;
  pc->ops->applysymmetricleft  = PCApplySymmetricLeft_Cholesky;
  pc->ops->applysymmetricright = PCApplySymmetricRight_Cholesky;
  pc->ops->applytranspose      = PCApplyTranspose_Cholesky;
  pc->ops->setup               = PCSetUp_Cholesky;
  pc->ops->setfromoptions      = PCSetFromOptions_Cholesky;
  pc->ops->view                = PCView_Factor;
  pc->ops->applyrichardson     = NULL;
  PetscFunctionReturn(0);
}
