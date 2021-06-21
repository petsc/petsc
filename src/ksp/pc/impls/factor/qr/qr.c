
/*
   Defines a direct QR factorization preconditioner for any Mat implementation
   Note: this need not be considered a preconditioner since it supplies
         a direct solver.
*/
#include <../src/ksp/pc/impls/factor/qr/qr.h>  /*I "petscpc.h" I*/

static PetscErrorCode PCSetUp_QR(PC pc)
{
  PetscErrorCode         ierr;
  PC_QR                  *dir = (PC_QR*)pc->data;
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
      ierr = MatQRFactor(pc->pmat,dir->col,&((PC_Factor*)dir)->info);CHKERRQ(ierr);
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
      if (!((PC_Factor*)dir)->fact) {
        ierr = MatGetFactor(pc->pmat,((PC_Factor*)dir)->solvertype,MAT_FACTOR_QR,&((PC_Factor*)dir)->fact);CHKERRQ(ierr);
        ierr = PetscLogObjectParent((PetscObject)pc,(PetscObject)((PC_Factor*)dir)->fact);CHKERRQ(ierr);
      }
      ierr = MatQRFactorSymbolic(((PC_Factor*)dir)->fact,pc->pmat,dir->col,&((PC_Factor*)dir)->info);CHKERRQ(ierr);
      ierr = MatGetInfo(((PC_Factor*)dir)->fact,MAT_LOCAL,&info);CHKERRQ(ierr);
      dir->hdr.actualfill = info.fill_ratio_needed;
    } else if (pc->flag != SAME_NONZERO_PATTERN) {
      ierr = MatQRFactorSymbolic(((PC_Factor*)dir)->fact,pc->pmat,dir->col,&((PC_Factor*)dir)->info);CHKERRQ(ierr);
      ierr = MatGetInfo(((PC_Factor*)dir)->fact,MAT_LOCAL,&info);CHKERRQ(ierr);
      dir->hdr.actualfill = info.fill_ratio_needed;
    } else {
      ierr = MatFactorGetError(((PC_Factor*)dir)->fact,&err);CHKERRQ(ierr);
    }
    ierr = MatFactorGetError(((PC_Factor*)dir)->fact,&err);CHKERRQ(ierr);
    if (err) { /* FactorSymbolic() fails */
      pc->failedreason = (PCFailedReason)err;
      PetscFunctionReturn(0);
    }

    ierr = MatQRFactorNumeric(((PC_Factor*)dir)->fact,pc->pmat,&((PC_Factor*)dir)->info);CHKERRQ(ierr);
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

static PetscErrorCode PCReset_QR(PC pc)
{
  PC_QR          *dir = (PC_QR*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!dir->hdr.inplace && ((PC_Factor*)dir)->fact) {ierr = MatDestroy(&((PC_Factor*)dir)->fact);CHKERRQ(ierr);}
  ierr = ISDestroy(&dir->col);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_QR(PC pc)
{
  PC_QR          *dir = (PC_QR*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCReset_QR(pc);CHKERRQ(ierr);
  ierr = PetscFree(((PC_Factor*)dir)->ordering);CHKERRQ(ierr);
  ierr = PetscFree(((PC_Factor*)dir)->solvertype);CHKERRQ(ierr);
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApply_QR(PC pc,Vec x,Vec y)
{
  PC_QR          *dir = (PC_QR*)pc->data;
  Mat            fact;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  fact = dir->hdr.inplace ? pc->pmat : ((PC_Factor*)dir)->fact;
  ierr = MatSolve(fact,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCMatApply_QR(PC pc,Mat X,Mat Y)
{
  PC_QR          *dir = (PC_QR*)pc->data;
  Mat            fact;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  fact = dir->hdr.inplace ? pc->pmat : ((PC_Factor*)dir)->fact;
  ierr = MatMatSolve(fact,X,Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplyTranspose_QR(PC pc,Vec x,Vec y)
{
  PC_QR          *dir = (PC_QR*)pc->data;
  Mat            fact;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  fact = dir->hdr.inplace ? pc->pmat : ((PC_Factor*)dir)->fact;
  ierr = MatSolveTranspose(fact,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -----------------------------------------------------------------------------------*/

/*MC
   PCQR - Uses a direct solver, based on QR factorization, as a preconditioner

   Level: beginner

   Notes:
    Usually this will compute an "exact" solution in one iteration and does
          not need a Krylov method (i.e. you can use -ksp_type preonly, or
          KSPSetType(ksp,KSPPREONLY) for the Krylov method

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC,
           PCILU, PCLU, PCCHOLESKY, PCICC, PCFactorSetReuseOrdering(), PCFactorSetReuseFill(), PCFactorGetMatrix(),
           PCFactorSetFill(), PCFactorSetUseInPlace(), PCFactorSetMatOrderingType(), PCFactorSetColumnPivot(),
           PCFactorSetPivotingInBlocks(),PCFactorSetShiftType(),PCFactorSetShiftAmount()
           PCFactorReorderForNonzeroDiagonal()
M*/

PETSC_EXTERN PetscErrorCode PCCreate_QR(PC pc)
{
  PetscErrorCode ierr;
  PC_QR          *dir;

  PetscFunctionBegin;
  ierr     = PetscNewLog(pc,&dir);CHKERRQ(ierr);
  pc->data = (void*)dir;
  ierr     = PCFactorInitialize(pc, MAT_FACTOR_QR);CHKERRQ(ierr);

  dir->col                   = NULL;
  pc->ops->reset             = PCReset_QR;
  pc->ops->destroy           = PCDestroy_QR;
  pc->ops->apply             = PCApply_QR;
  pc->ops->matapply          = PCMatApply_QR;
  pc->ops->applytranspose    = PCApplyTranspose_QR;
  pc->ops->setup             = PCSetUp_QR;
  pc->ops->setfromoptions    = PCSetFromOptions_Factor;
  pc->ops->view              = PCView_Factor;
  pc->ops->applyrichardson   = NULL;
  PetscFunctionReturn(0);
}
