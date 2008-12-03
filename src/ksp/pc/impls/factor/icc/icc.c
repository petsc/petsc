#define PETSCKSP_DLL

#include "../src/ksp/pc/impls/factor/icc/icc.h"   /*I "petscpc.h" I*/

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
    ierr = MatGetFactor(pc->pmat,MAT_SOLVER_PETSC,MAT_FACTOR_ICC,& ((PC_Factor*)icc)->fact);CHKERRQ(ierr);
    ierr = MatICCFactorSymbolic(((PC_Factor*)icc)->fact,pc->pmat,perm,&((PC_Factor*)icc)->info);CHKERRQ(ierr);
  } else if (pc->flag != SAME_NONZERO_PATTERN) {
    ierr = MatDestroy(((PC_Factor*)icc)->fact);CHKERRQ(ierr);
    ierr = MatGetFactor(pc->pmat,MAT_SOLVER_PETSC,MAT_FACTOR_ICC,&((PC_Factor*)icc)->fact);CHKERRQ(ierr);
    ierr = MatICCFactorSymbolic(((PC_Factor*)icc)->fact,pc->pmat,perm,&((PC_Factor*)icc)->info);CHKERRQ(ierr);
  }
  ierr = MatGetInfo(((PC_Factor*)icc)->fact,MAT_LOCAL,&info);CHKERRQ(ierr);
  icc->actualfill = info.fill_ratio_needed;

  ierr = ISDestroy(cperm);CHKERRQ(ierr);
  ierr = ISDestroy(perm);CHKERRQ(ierr);
  ierr = MatCholeskyFactorNumeric(((PC_Factor*)icc)->fact,pc->pmat,&((PC_Factor*)icc)->info);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCDestroy_ICC"
static PetscErrorCode PCDestroy_ICC(PC pc)
{
  PC_ICC         *icc = (PC_ICC*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (((PC_Factor*)icc)->fact) {ierr = MatDestroy(((PC_Factor*)icc)->fact);CHKERRQ(ierr);}
  ierr = PetscStrfree(((PC_Factor*)icc)->ordering);CHKERRQ(ierr);
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
  char           tname[256];
  PetscTruth     flg;
  PetscErrorCode ierr;
  PetscFList     ordlist;

  PetscFunctionBegin;
  if (!MatOrderingRegisterAllCalled) {ierr = MatOrderingRegisterAll(PETSC_NULL);CHKERRQ(ierr);}
  ierr = PetscOptionsHead("ICC Options");CHKERRQ(ierr);
    ierr = PetscOptionsReal("-pc_factor_levels","levels of fill","PCFactorSetLevels",((PC_Factor*)icc)->info.levels,&((PC_Factor*)icc)->info.levels,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-pc_factor_fill","Expected fill in factorization","PCFactorSetFill",((PC_Factor*)icc)->info.fill,&((PC_Factor*)icc)->info.fill,&flg);CHKERRQ(ierr);
    ierr = MatGetOrderingList(&ordlist);CHKERRQ(ierr);
    ierr = PetscOptionsList("-pc_factor_mat_ordering_type","Reorder to reduce nonzeros in ICC","PCFactorSetMatOrderingType",ordlist,((PC_Factor*)icc)->ordering,tname,256,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PCFactorSetMatOrderingType(pc,tname);CHKERRQ(ierr);
    }
    ierr = PetscOptionsName("-pc_factor_shift_nonzero","Shift added to diagonal","PCFactorSetShiftNonzero",&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PCFactorSetShiftNonzero(pc,(PetscReal)PETSC_DECIDE);CHKERRQ(ierr);
    }
    ierr = PetscOptionsReal("-pc_factor_shift_nonzero","Shift added to diagonal","PCFactorSetShiftNonzero",((PC_Factor*)icc)->info.shiftnz,&((PC_Factor*)icc)->info.shiftnz,0);CHKERRQ(ierr);
    flg = (((PC_Factor*)icc)->info.shiftpd > 0.0) ? PETSC_TRUE : PETSC_FALSE;
    ierr = PetscOptionsTruth("-pc_factor_shift_positive_definite","Manteuffel shift applied to diagonal","PCFactorSetShiftPd",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
    ierr = PCFactorSetShiftPd(pc,flg);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-pc_factor_zeropivot","Pivot is considered zero if less than","PCFactorSetZeroPivot",((PC_Factor*)icc)->info.zeropivot,&((PC_Factor*)icc)->info.zeropivot,0);CHKERRQ(ierr);
 
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
    if (((PC_Factor*)icc)->info.levels == 1) {
        ierr = PetscViewerASCIIPrintf(viewer,"  ICC: %D level of fill\n",(PetscInt)((PC_Factor*)icc)->info.levels);CHKERRQ(ierr);
    } else {
        ierr = PetscViewerASCIIPrintf(viewer,"  ICC: %D levels of fill\n",(PetscInt)((PC_Factor*)icc)->info.levels);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  ICC: factor fill ratio allocated %G, ordering used %s\n",((PC_Factor*)icc)->info.fill,((PC_Factor*)icc)->ordering);CHKERRQ(ierr);
    if (((PC_Factor*)icc)->info.shiftpd) {ierr = PetscViewerASCIIPrintf(viewer,"  ICC: using Manteuffel shift\n");CHKERRQ(ierr);}
    if (((PC_Factor*)icc)->info.shiftnz) {ierr = PetscViewerASCIIPrintf(viewer,"  ICC: using diagonal shift to prevent zero pivot\n");CHKERRQ(ierr);}
    if (((PC_Factor*)icc)->fact) {
      ierr = PetscViewerASCIIPrintf(viewer,"  ICC: factor fill ratio needed %G\n",icc->actualfill);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"       Factored matrix follows\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr);
      ierr = MatView(((PC_Factor*)icc)->fact,viewer);CHKERRQ(ierr);
      ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
  } else if (isstring) {
    ierr = PetscViewerStringSPrintf(viewer," lvls=%D",(PetscInt)((PC_Factor*)icc)->info.levels);CHKERRQ(ierr);CHKERRQ(ierr);
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

   Notes: Only implemented for some matrix formats. Not implemented in parallel (for parallel use you 
             must use MATMPIROWBS, see MatCreateMPIRowbs(), this supports only ICC(0) and this is not recommended
             unless you really want a parallel ICC).

          For BAIJ matrices this implements a point block ICC.

          The Manteuffel shift is only implemented for matrices with block size 1

          By default, the Manteuffel is applied (for matrices with block size 1). Call PCFactorSetShiftPd(pc,PETSC_FALSE);
          to turn off the shift.

   References:
   Review article: APPROXIMATE AND INCOMPLETE FACTORIZATIONS, TONY F. CHAN AND HENK A. VAN DER VORST
      http://igitur-archive.library.uu.nl/math/2001-0621-115821/proc.pdf chapter in Parallel Numerical
      Algorithms, edited by D. Keyes, A. Semah, V. Venkatakrishnan, ICASE/LaRC Interdisciplinary Series in
      Science and Engineering, Kluwer, pp. 167--202.


.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC, PCSOR, MatOrderingType,
           PCFactorSetZeroPivot(), PCFactorSetShiftNonzero(), PCFactorSetShiftPd(), 
           PCFactorSetFill(), PCFactorSetMatOrderingType(), PCFactorSetReuseOrdering(), 
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
  ierr = PetscNewLog(pc,PC_ICC,&icc);CHKERRQ(ierr);

  ((PC_Factor*)icc)->fact	          = 0;
  ierr = PetscStrallocpy(MATORDERING_NATURAL,&((PC_Factor*)icc)->ordering);CHKERRQ(ierr);
  ierr = MatFactorInfoInitialize(&((PC_Factor*)icc)->info);CHKERRQ(ierr);
  ((PC_Factor*)icc)->info.levels	  = 0;
  ((PC_Factor*)icc)->info.fill          = 1.0;
  icc->implctx            = 0;

  ((PC_Factor*)icc)->info.dtcol              = PETSC_DEFAULT;
  ((PC_Factor*)icc)->info.shiftnz            = 0.0;
  ((PC_Factor*)icc)->info.shiftpd            = 1.0; /* true */
  ((PC_Factor*)icc)->info.zeropivot          = 1.e-12;
  pc->data	               = (void*)icc;

  pc->ops->apply	       = PCApply_ICC;
  pc->ops->setup               = PCSetup_ICC;
  pc->ops->destroy	       = PCDestroy_ICC;
  pc->ops->setfromoptions      = PCSetFromOptions_ICC;
  pc->ops->view                = PCView_ICC;
  pc->ops->getfactoredmatrix   = PCFactorGetMatrix_Factor;
  pc->ops->applysymmetricleft  = PCApplySymmetricLeft_ICC;
  pc->ops->applysymmetricright = PCApplySymmetricRight_ICC;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorGetMatSolverPackage_C","PCFactorGetMatSolverPackage_Factor",
                    PCFactorGetMatSolverPackage_Factor);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetZeroPivot_C","PCFactorSetZeroPivot_Factor",
                    PCFactorSetZeroPivot_Factor);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetShiftNonzero_C","PCFactorSetShiftNonzero_Factor",
                    PCFactorSetShiftNonzero_Factor);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetShiftPd_C","PCFactorSetShiftPd_Factor",
                    PCFactorSetShiftPd_Factor);CHKERRQ(ierr);

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetLevels_C","PCFactorSetLevels_Factor",
                    PCFactorSetLevels_Factor);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetFill_C","PCFactorSetFill_Factor",
                    PCFactorSetFill_Factor);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetMatOrderingType_C","PCFactorSetMatOrderingType_Factor",
                    PCFactorSetMatOrderingType_Factor);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END


