#define PETSCKSP_DLL

/*
   Defines a Cholesky factorization preconditioner for any Mat implementation.
  Presently only provided for MPIRowbs format (i.e. BlockSolve).
*/

#include "src/ksp/pc/impls/factor/icc/icc.h"   /*I "petscpc.h" I*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetZeroPivot_ICC"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetZeroPivot_ICC(PC pc,PetscReal z)
{
  PC_ICC *icc;

  PetscFunctionBegin;
  icc                 = (PC_ICC*)pc->data;
  icc->info.zeropivot = z;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetShiftNonzero_ICC"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetShiftNonzero_ICC(PC pc,PetscReal shift)
{
  PC_ICC *dir;

  PetscFunctionBegin;
  dir = (PC_ICC*)pc->data;
  if (shift == (PetscReal) PETSC_DECIDE) {
    dir->info.shiftnz = 1.e-12;
  } else {
    dir->info.shiftnz = shift;
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetShiftPd_ICC"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetShiftPd_ICC(PC pc,PetscTruth shift)
{
  PC_ICC *dir;
 
  PetscFunctionBegin;
  dir = (PC_ICC*)pc->data;
  if (shift) {
    dir->info.shift_fraction = 0.0;
    dir->info.shiftpd = 1.0;
  } else {
    dir->info.shiftpd = 0.0;
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetMatOrdering_ICC"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetMatOrdering_ICC(PC pc,MatOrderingType ordering)
{
  PC_ICC         *dir = (PC_ICC*)pc->data;
  PetscErrorCode ierr;
 
  PetscFunctionBegin;
  ierr = PetscStrfree(dir->ordering);CHKERRQ(ierr);
  ierr = PetscStrallocpy(ordering,&dir->ordering);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetFill_ICC"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetFill_ICC(PC pc,PetscReal fill)
{
  PC_ICC *dir;

  PetscFunctionBegin;
  dir            = (PC_ICC*)pc->data;
  dir->info.fill = fill;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetLevels_ICC"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetLevels_ICC(PC pc,PetscInt levels)
{
  PC_ICC *icc;

  PetscFunctionBegin;
  icc = (PC_ICC*)pc->data;
  icc->info.levels = levels;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "PCSetup_ICC"
static PetscErrorCode PCSetup_ICC(PC pc)
{
  PC_ICC         *icc = (PC_ICC*)pc->data;
  IS             perm,cperm;
  PetscErrorCode ierr;
  MatInfo        info;

  PetscFunctionBegin;
  ierr = MatGetOrdering(pc->pmat,icc->ordering,&perm,&cperm);CHKERRQ(ierr);

  if (!pc->setupcalled) {
    ierr = MatICCFactorSymbolic(pc->pmat,perm,&icc->info,&icc->fact);CHKERRQ(ierr);
  } else if (pc->flag != SAME_NONZERO_PATTERN) {
    ierr = MatDestroy(icc->fact);CHKERRQ(ierr);
    ierr = MatICCFactorSymbolic(pc->pmat,perm,&icc->info,&icc->fact);CHKERRQ(ierr);
  }
  ierr = MatGetInfo(icc->fact,MAT_LOCAL,&info);CHKERRQ(ierr);
  icc->actualfill = info.fill_ratio_needed;

  ierr = ISDestroy(cperm);CHKERRQ(ierr);
  ierr = ISDestroy(perm);CHKERRQ(ierr);
  ierr = MatCholeskyFactorNumeric(pc->pmat,&icc->info,&icc->fact);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCDestroy_ICC"
static PetscErrorCode PCDestroy_ICC(PC pc)
{
  PC_ICC         *icc = (PC_ICC*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (icc->fact) {ierr = MatDestroy(icc->fact);CHKERRQ(ierr);}
  ierr = PetscStrfree(icc->ordering);CHKERRQ(ierr);
  ierr = PetscFree(icc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCApply_ICC"
static PetscErrorCode PCApply_ICC(PC pc,Vec x,Vec y)
{
  PC_ICC         *icc = (PC_ICC*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSolve(icc->fact,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);  
}

#undef __FUNCT__  
#define __FUNCT__ "PCApplySymmetricLeft_ICC"
static PetscErrorCode PCApplySymmetricLeft_ICC(PC pc,Vec x,Vec y)
{
  PetscErrorCode ierr;
  PC_ICC         *icc = (PC_ICC*)pc->data;

  PetscFunctionBegin;
  ierr = MatForwardSolve(icc->fact,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCApplySymmetricRight_ICC"
static PetscErrorCode PCApplySymmetricRight_ICC(PC pc,Vec x,Vec y)
{
  PetscErrorCode ierr;
  PC_ICC         *icc = (PC_ICC*)pc->data;

  PetscFunctionBegin;
  ierr = MatBackwardSolve(icc->fact,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCGetFactoredMatrix_ICC"
static PetscErrorCode PCGetFactoredMatrix_ICC(PC pc,Mat *mat)
{
  PC_ICC *icc = (PC_ICC*)pc->data;

  PetscFunctionBegin;
  *mat = icc->fact;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSetFromOptions_ICC"
static PetscErrorCode PCSetFromOptions_ICC(PC pc)
{
  PC_ICC         *icc = (PC_ICC*)pc->data;
  char           tname[256];
  PetscTruth     flg;
  PetscErrorCode ierr;
  PetscFList     ordlist;

  PetscFunctionBegin;
  ierr = MatOrderingRegisterAll(PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsHead("ICC Options");CHKERRQ(ierr);
    ierr = PetscOptionsReal("-pc_factor_levels","levels of fill","PCFactorSetLevels",icc->info.levels,&icc->info.levels,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-pc_factor_fill","Expected fill in factorization","PCFactorSetFill",icc->info.fill,&icc->info.fill,&flg);CHKERRQ(ierr);
    ierr = MatGetOrderingList(&ordlist);CHKERRQ(ierr);
    ierr = PetscOptionsList("-pc_factor_mat_ordering_type","Reorder to reduce nonzeros in ICC","PCFactorSetMatOrdering",ordlist,icc->ordering,tname,256,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PCFactorSetMatOrdering(pc,tname);CHKERRQ(ierr);
    }
    ierr = PetscOptionsName("-pc_factor_shift_nonzero","Shift added to diagonal","PCFactorSetShiftNonzero",&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PCFactorSetShiftNonzero(pc,(PetscReal)PETSC_DECIDE);CHKERRQ(ierr);
    }
    ierr = PetscOptionsReal("-pc_factor_shift_nonzero","Shift added to diagonal","PCFactorSetShiftNonzero",icc->info.shiftnz,&icc->info.shiftnz,0);CHKERRQ(ierr);
    ierr = PetscOptionsName("-pc_factor_shift_positive_definite","Manteuffel shift applied to diagonal","PCFactorSetShift",&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PCFactorSetShiftPd(pc,PETSC_TRUE);CHKERRQ(ierr);
    } else {
      ierr = PCFactorSetShiftPd(pc,PETSC_FALSE);CHKERRQ(ierr);
    }
    ierr = PetscOptionsReal("-pc_factor_zeropivot","Pivot is considered zero if less than","PCFactorSetZeroPivot",icc->info.zeropivot,&icc->info.zeropivot,0);CHKERRQ(ierr);
 
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCView_ICC"
static PetscErrorCode PCView_ICC(PC pc,PetscViewer viewer)
{
  PC_ICC         *icc = (PC_ICC*)pc->data;
  PetscErrorCode ierr;
  PetscTruth     isstring,iascii;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_STRING,&isstring);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    if (icc->info.levels == 1) {
        ierr = PetscViewerASCIIPrintf(viewer,"  ICC: %D level of fill\n",(PetscInt)icc->info.levels);CHKERRQ(ierr);
    } else {
        ierr = PetscViewerASCIIPrintf(viewer,"  ICC: %D levels of fill\n",(PetscInt)icc->info.levels);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  ICC: factor fill ratio allocated %G\n",icc->info.fill);CHKERRQ(ierr);
    if (icc->info.shiftpd) {ierr = PetscViewerASCIIPrintf(viewer,"  ICC: using Manteuffel shift\n");CHKERRQ(ierr);}
    if (icc->fact) {
      ierr = PetscViewerASCIIPrintf(viewer,"  ICC: factor fill ratio needed %G\n",icc->actualfill);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"       Factored matrix follows\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr);
      ierr = MatView(icc->fact,viewer);CHKERRQ(ierr);
      ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
  } else if (isstring) {
    ierr = PetscViewerStringSPrintf(viewer," lvls=%D",(PetscInt)icc->info.levels);CHKERRQ(ierr);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported for PCICC",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

/*MC
     PCICC - Incomplete Cholesky factorization preconditioners.

   Options Database Keys:
+  -pc_factor_levels <k> - number of levels of fill for ICC(k)
.  -pc_factor_in_place - only for ICC(0) with natural ordering, reuses the space of the matrix for
                      its factorization (overwrites original matrix)
.  -pc_factor_fill <nfill> - expected amount of fill in factored matrix compared to original matrix, nfill > 1
.  -pc_factor_mat_ordering_type <natural,nd,1wd,rcm,qmd> - set the row/column ordering of the factored matrix
.  -pc_factor_shift_nonzero <shift> - Sets shift amount or PETSC_DECIDE for the default
-  -pc_factor_shift_positive_definite [PETSC_TRUE/PETSC_FALSE] - Activate/Deactivate PCFactorSetShiftPd(); the value
   is optional with PETSC_TRUE being the default

   Level: beginner

  Concepts: incomplete Cholesky factorization

   Notes: Only implemented for some matrix formats. Not implemented in parallel

          For BAIJ matrices this implements a point block ICC.

          The Manteuffel shift is only implemented for matrices with block size 1

          By default, the Manteuffel is applied (for matrices with block size 1). Call PCFactorSetShiftPd(pc,PETSC_FALSE);
          to turn off the shift.


.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC, PCSOR, MatOrderingType,
           PCFactorSetZeroPivot(), PCFactorSetShiftNonzero(), PCFactorSetShiftPd(), 
           PCFactorSetFill(), PCFactorSetMatOrdering(), PCFactorSetReuseOrdering(), 
           PCFactorSetLevels(),PCFactorSetShiftNonzero(),PCFactorSetShiftPd(),

M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCreate_ICC"
PetscErrorCode PETSCKSP_DLLEXPORT PCCreate_ICC(PC pc)
{
  PetscErrorCode ierr;
  PC_ICC         *icc;

  PetscFunctionBegin;
  ierr = PetscNew(PC_ICC,&icc);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(pc,sizeof(PC_ICC));CHKERRQ(ierr);

  icc->fact	          = 0;
  ierr = PetscStrallocpy(MATORDERING_NATURAL,&icc->ordering);CHKERRQ(ierr);
  ierr = MatFactorInfoInitialize(&icc->info);CHKERRQ(ierr);
  icc->info.levels	  = 0;
  icc->info.fill          = 1.0;
  icc->implctx            = 0;

  icc->info.dtcol              = PETSC_DEFAULT;
  icc->info.shiftnz            = 0.0;
  icc->info.shiftpd            = 1.0; /* true */
  icc->info.shift_fraction     = 0.0;
  icc->info.zeropivot          = 1.e-12;
  pc->data	               = (void*)icc;

  pc->ops->apply	       = PCApply_ICC;
  pc->ops->setup               = PCSetup_ICC;
  pc->ops->destroy	       = PCDestroy_ICC;
  pc->ops->setfromoptions      = PCSetFromOptions_ICC;
  pc->ops->view                = PCView_ICC;
  pc->ops->getfactoredmatrix   = PCGetFactoredMatrix_ICC;
  pc->ops->applysymmetricleft  = PCApplySymmetricLeft_ICC;
  pc->ops->applysymmetricright = PCApplySymmetricRight_ICC;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetZeroPivot_C","PCFactorSetZeroPivot_ICC",
                    PCFactorSetZeroPivot_ICC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetShiftNonzero_C","PCFactorSetShiftNonzero_ICC",
                    PCFactorSetShiftNonzero_ICC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetShiftPd_C","PCFactorSetShiftPd_ICC",
                    PCFactorSetShiftPd_ICC);CHKERRQ(ierr);

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetLevels_C","PCFactorSetLevels_ICC",
                    PCFactorSetLevels_ICC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetFill_C","PCFactorSetFill_ICC",
                    PCFactorSetFill_ICC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetMatOrdering_C","PCFactorSetMatOrdering_ICC",
                    PCFactorSetMatOrdering_ICC);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END


