
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
  PetscFunctionBegin;
  CHKERRQ(PetscOptionsHead(PetscOptionsObject,"Cholesky options"));
  CHKERRQ(PCSetFromOptions_Factor(PetscOptionsObject,pc));
  CHKERRQ(PetscOptionsTail());
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetUp_Cholesky(PC pc)
{
  PetscBool              flg;
  PC_Cholesky            *dir = (PC_Cholesky*)pc->data;
  MatSolverType          stype;
  MatFactorError         err;

  PetscFunctionBegin;
  pc->failedreason = PC_NOERROR;
  if (dir->hdr.reusefill && pc->setupcalled) ((PC_Factor*)dir)->info.fill = dir->hdr.actualfill;

  CHKERRQ(MatSetErrorIfFailure(pc->pmat,pc->erroriffailure));
  if (dir->hdr.inplace) {
    if (dir->row && dir->col && (dir->row != dir->col)) {
      CHKERRQ(ISDestroy(&dir->row));
    }
    CHKERRQ(ISDestroy(&dir->col));
    /* should only get reordering if the factor matrix uses it but cannot determine because MatGetFactor() not called */
    CHKERRQ(PCFactorSetDefaultOrdering_Factor(pc));
    CHKERRQ(MatGetOrdering(pc->pmat,((PC_Factor*)dir)->ordering,&dir->row,&dir->col));
    if (dir->col && (dir->row != dir->col)) {  /* only use row ordering for SBAIJ */
      CHKERRQ(ISDestroy(&dir->col));
    }
    if (dir->row) CHKERRQ(PetscLogObjectParent((PetscObject)pc,(PetscObject)dir->row));
    CHKERRQ(MatCholeskyFactor(pc->pmat,dir->row,&((PC_Factor*)dir)->info));
    CHKERRQ(MatFactorGetError(pc->pmat,&err));
    if (err) { /* Factor() fails */
      pc->failedreason = (PCFailedReason)err;
      PetscFunctionReturn(0);
    }

    ((PC_Factor*)dir)->fact = pc->pmat;
  } else {
    MatInfo info;

    if (!pc->setupcalled) {
      PetscBool canuseordering;
      if (!((PC_Factor*)dir)->fact) {
        CHKERRQ(MatGetFactor(pc->pmat,((PC_Factor*)dir)->solvertype,MAT_FACTOR_CHOLESKY,&((PC_Factor*)dir)->fact));
        CHKERRQ(PetscLogObjectParent((PetscObject)pc,(PetscObject)((PC_Factor*)dir)->fact));
      }
      CHKERRQ(MatFactorGetCanUseOrdering(((PC_Factor*)dir)->fact,&canuseordering));
      if (canuseordering) {
        CHKERRQ(PCFactorSetDefaultOrdering_Factor(pc));
        CHKERRQ(MatGetOrdering(pc->pmat,((PC_Factor*)dir)->ordering,&dir->row,&dir->col));
        /* check if dir->row == dir->col */
        if (dir->row) {
          CHKERRQ(ISEqual(dir->row,dir->col,&flg));
          PetscCheckFalse(!flg,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"row and column permutations must be equal");
        }
        CHKERRQ(ISDestroy(&dir->col)); /* only pass one ordering into CholeskyFactor */

        flg  = PETSC_FALSE;
        CHKERRQ(PetscOptionsGetBool(((PetscObject)pc)->options,((PetscObject)pc)->prefix,"-pc_factor_nonzeros_along_diagonal",&flg,NULL));
        if (flg) {
          PetscReal tol = 1.e-10;
          CHKERRQ(PetscOptionsGetReal(((PetscObject)pc)->options,((PetscObject)pc)->prefix,"-pc_factor_nonzeros_along_diagonal",&tol,NULL));
          CHKERRQ(MatReorderForNonzeroDiagonal(pc->pmat,tol,dir->row,dir->row));
        }
        CHKERRQ(PetscLogObjectParent((PetscObject)pc,(PetscObject)dir->row));
      }
      CHKERRQ(MatCholeskyFactorSymbolic(((PC_Factor*)dir)->fact,pc->pmat,dir->row,&((PC_Factor*)dir)->info));
      CHKERRQ(MatGetInfo(((PC_Factor*)dir)->fact,MAT_LOCAL,&info));
      dir->hdr.actualfill = info.fill_ratio_needed;
    } else if (pc->flag != SAME_NONZERO_PATTERN) {
      if (!dir->hdr.reuseordering) {
        PetscBool canuseordering;
        CHKERRQ(MatDestroy(&((PC_Factor*)dir)->fact));
        CHKERRQ(MatGetFactor(pc->pmat,((PC_Factor*)dir)->solvertype,MAT_FACTOR_CHOLESKY,&((PC_Factor*)dir)->fact));
        CHKERRQ(PetscLogObjectParent((PetscObject)pc,(PetscObject)((PC_Factor*)dir)->fact));
        CHKERRQ(MatFactorGetCanUseOrdering(((PC_Factor*)dir)->fact,&canuseordering));
        if (canuseordering) {
          CHKERRQ(ISDestroy(&dir->row));
          CHKERRQ(PCFactorSetDefaultOrdering_Factor(pc));
          CHKERRQ(MatGetOrdering(pc->pmat,((PC_Factor*)dir)->ordering,&dir->row,&dir->col));
          CHKERRQ(ISDestroy(&dir->col)); /* only use dir->row ordering in CholeskyFactor */

          flg  = PETSC_FALSE;
          CHKERRQ(PetscOptionsGetBool(((PetscObject)pc)->options,((PetscObject)pc)->prefix,"-pc_factor_nonzeros_along_diagonal",&flg,NULL));
          if (flg) {
            PetscReal tol = 1.e-10;
            CHKERRQ(PetscOptionsGetReal(((PetscObject)pc)->options,((PetscObject)pc)->prefix,"-pc_factor_nonzeros_along_diagonal",&tol,NULL));
            CHKERRQ(MatReorderForNonzeroDiagonal(pc->pmat,tol,dir->row,dir->row));
          }
          CHKERRQ(PetscLogObjectParent((PetscObject)pc,(PetscObject)dir->row));
        }
      }
      CHKERRQ(MatCholeskyFactorSymbolic(((PC_Factor*)dir)->fact,pc->pmat,dir->row,&((PC_Factor*)dir)->info));
      CHKERRQ(MatGetInfo(((PC_Factor*)dir)->fact,MAT_LOCAL,&info));
      dir->hdr.actualfill = info.fill_ratio_needed;
      CHKERRQ(PetscLogObjectParent((PetscObject)pc,(PetscObject)((PC_Factor*)dir)->fact));
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

    CHKERRQ(MatCholeskyFactorNumeric(((PC_Factor*)dir)->fact,pc->pmat,&((PC_Factor*)dir)->info));
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

static PetscErrorCode PCReset_Cholesky(PC pc)
{
  PC_Cholesky    *dir = (PC_Cholesky*)pc->data;

  PetscFunctionBegin;
  if (!dir->hdr.inplace && ((PC_Factor*)dir)->fact) CHKERRQ(MatDestroy(&((PC_Factor*)dir)->fact));
  CHKERRQ(ISDestroy(&dir->row));
  CHKERRQ(ISDestroy(&dir->col));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_Cholesky(PC pc)
{
  PC_Cholesky    *dir = (PC_Cholesky*)pc->data;

  PetscFunctionBegin;
  CHKERRQ(PCReset_Cholesky(pc));
  CHKERRQ(PetscFree(((PC_Factor*)dir)->ordering));
  CHKERRQ(PetscFree(((PC_Factor*)dir)->solvertype));
  CHKERRQ(PetscFree(pc->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApply_Cholesky(PC pc,Vec x,Vec y)
{
  PC_Cholesky    *dir = (PC_Cholesky*)pc->data;

  PetscFunctionBegin;
  if (dir->hdr.inplace) {
    CHKERRQ(MatSolve(pc->pmat,x,y));
  } else {
    CHKERRQ(MatSolve(((PC_Factor*)dir)->fact,x,y));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCMatApply_Cholesky(PC pc,Mat X,Mat Y)
{
  PC_Cholesky    *dir = (PC_Cholesky*)pc->data;

  PetscFunctionBegin;
  if (dir->hdr.inplace) {
    CHKERRQ(MatMatSolve(pc->pmat,X,Y));
  } else {
    CHKERRQ(MatMatSolve(((PC_Factor*)dir)->fact,X,Y));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplySymmetricLeft_Cholesky(PC pc,Vec x,Vec y)
{
  PC_Cholesky    *dir = (PC_Cholesky*)pc->data;

  PetscFunctionBegin;
  if (dir->hdr.inplace) {
    CHKERRQ(MatForwardSolve(pc->pmat,x,y));
  } else {
    CHKERRQ(MatForwardSolve(((PC_Factor*)dir)->fact,x,y));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplySymmetricRight_Cholesky(PC pc,Vec x,Vec y)
{
  PC_Cholesky    *dir = (PC_Cholesky*)pc->data;

  PetscFunctionBegin;
  if (dir->hdr.inplace) {
    CHKERRQ(MatBackwardSolve(pc->pmat,x,y));
  } else {
    CHKERRQ(MatBackwardSolve(((PC_Factor*)dir)->fact,x,y));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplyTranspose_Cholesky(PC pc,Vec x,Vec y)
{
  PC_Cholesky    *dir = (PC_Cholesky*)pc->data;

  PetscFunctionBegin;
  if (dir->hdr.inplace) {
    CHKERRQ(MatSolveTranspose(pc->pmat,x,y));
  } else {
    CHKERRQ(MatSolveTranspose(((PC_Factor*)dir)->fact,x,y));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveBool(pc,flag,2);
  CHKERRQ(PetscTryMethod(pc,"PCFactorSetReuseOrdering_C",(PC,PetscBool),(pc,flag)));
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
  PC_Cholesky    *dir;

  PetscFunctionBegin;
  CHKERRQ(PetscNewLog(pc,&dir));
  pc->data = (void*)dir;
  CHKERRQ(PCFactorInitialize(pc,MAT_FACTOR_CHOLESKY));

  ((PC_Factor*)dir)->info.fill  = 5.0;

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
