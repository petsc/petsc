#define PETSCKSP_DLL

#include "../src/ksp/pc/impls/factor/factor.h"     /*I "petscpc.h"  I*/

/* ------------------------------------------------------------------------------------------*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetZeroPivot_Factor"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetZeroPivot_Factor(PC pc,PetscReal z)
{
  PC_Factor *ilu = (PC_Factor*)pc->data;

  PetscFunctionBegin;
  ilu->info.zeropivot = z;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetShiftType_Factor"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetShiftType_Factor(PC pc,MatFactorShiftType shifttype)
{
  PC_Factor *dir = (PC_Factor*)pc->data;

  PetscFunctionBegin;
  if (shifttype == (MatFactorShiftType)PETSC_DECIDE){
    dir->info.shifttype = (PetscReal) MAT_SHIFT_NONE; 
  } else {
    dir->info.shifttype = (PetscReal) shifttype;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetShiftAmount_Factor"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetShiftAmount_Factor(PC pc,PetscReal shiftamount)
{
  PC_Factor *dir = (PC_Factor*)pc->data;

  PetscFunctionBegin;
  if (shiftamount == (PetscReal) PETSC_DECIDE){
    dir->info.shiftamount = 1.e-12; 
  } else {
    dir->info.shiftamount = shiftamount; 
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetDropTolerance_Factor"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetDropTolerance_Factor(PC pc,PetscReal dt,PetscReal dtcol,PetscInt dtcount)
{
  PC_Factor         *ilu = (PC_Factor*)pc->data;

  PetscFunctionBegin;
  if (pc->setupcalled && (!ilu->info.usedt || ((PC_Factor*)ilu)->info.dt != dt || ((PC_Factor*)ilu)->info.dtcol != dtcol || ((PC_Factor*)ilu)->info.dtcount != dtcount)) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Cannot change tolerance after use");
  }
  ilu->info.usedt   = PETSC_TRUE;
  ilu->info.dt      = dt;
  ilu->info.dtcol   = dtcol;
  ilu->info.dtcount = dtcount;
  ilu->info.fill    = PETSC_DEFAULT;
  PetscFunctionReturn(0);
}  
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetFill_Factor"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetFill_Factor(PC pc,PetscReal fill)
{
  PC_Factor *dir = (PC_Factor*)pc->data;

  PetscFunctionBegin;
  dir->info.fill = fill;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetMatOrderingType_Factor"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetMatOrderingType_Factor(PC pc,const MatOrderingType ordering)
{
  PC_Factor      *dir = (PC_Factor*)pc->data;
  PetscErrorCode ierr;
  PetscTruth     flg;
 
  PetscFunctionBegin;
  if (!pc->setupcalled) {
     ierr = PetscStrfree(dir->ordering);CHKERRQ(ierr);
     ierr = PetscStrallocpy(ordering,&dir->ordering);CHKERRQ(ierr);
  } else {
    ierr = PetscStrcmp(dir->ordering,ordering,&flg);CHKERRQ(ierr);
    if (!flg) {
      SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Cannot change ordering after use");
    }
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetLevels_Factor"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetLevels_Factor(PC pc,PetscInt levels)
{
  PC_Factor      *ilu = (PC_Factor*)pc->data;

  PetscFunctionBegin;
  if (!pc->setupcalled) {
    ilu->info.levels = levels;
  } else if (ilu->info.usedt || ilu->info.levels != levels) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Cannot change levels after use");
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetAllowDiagonalFill_Factor"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetAllowDiagonalFill_Factor(PC pc)
{
  PC_Factor *dir = (PC_Factor*)pc->data;

  PetscFunctionBegin;
  dir->info.diagonal_fill = 1;
  PetscFunctionReturn(0);
}
EXTERN_C_END


/* ------------------------------------------------------------------------------------------*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetPivotInBlocks_Factor"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetPivotInBlocks_Factor(PC pc,PetscTruth pivot)
{
  PC_Factor *dir = (PC_Factor*)pc->data;

  PetscFunctionBegin;
  dir->info.pivotinblocks = pivot ? 1.0 : 0.0;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFactorGetMatrix_Factor"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorGetMatrix_Factor(PC pc,Mat *mat)
{
  PC_Factor *ilu = (PC_Factor*)pc->data;

  PetscFunctionBegin;
  if (!ilu->fact) SETERRQ(PETSC_ERR_ORDER,"Matrix not yet factored; call after KSPSetUp() or PCSetUp()");
  *mat = ilu->fact;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetMatSolverPackage_Factor"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetMatSolverPackage_Factor(PC pc,const MatSolverPackage stype)
{
  PetscErrorCode ierr;
  PC_Factor      *lu = (PC_Factor*)pc->data;

  PetscFunctionBegin;
  if (lu->fact) {
    const MatSolverPackage ltype;
    PetscTruth             flg;
    ierr = MatFactorGetSolverPackage(lu->fact,&ltype);CHKERRQ(ierr);
    ierr = PetscStrcmp(stype,ltype,&flg);CHKERRQ(ierr);
    if (!flg) {
      SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Cannot change solver matrix package after PC has been setup or used");
    }
  }
  ierr = PetscStrfree(lu->solvertype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(stype,&lu->solvertype);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFactorGetMatSolverPackage_Factor"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorGetMatSolverPackage_Factor(PC pc,const MatSolverPackage *stype)
{
  PC_Factor      *lu = (PC_Factor*)pc->data;

  PetscFunctionBegin;
  *stype = lu->solvertype;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetColumnPivot_Factor"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetColumnPivot_Factor(PC pc,PetscReal dtcol)
{
  PC_Factor       *dir = (PC_Factor*)pc->data;

  PetscFunctionBegin;
  if (dtcol < 0.0 || dtcol > 1.0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Column pivot tolerance is %G must be between 0 and 1",dtcol);
  dir->info.dtcol = dtcol;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCSetFromOptions_Factor"
PetscErrorCode PETSCKSP_DLLEXPORT PCSetFromOptions_Factor(PC pc)
{
  PC_Factor       *factor = (PC_Factor*)pc->data;
  PetscErrorCode  ierr;
  PetscTruth      flg = PETSC_FALSE,set;
  char            tname[256], solvertype[64];
  PetscFList      ordlist;
  PetscEnum       etmp;

  PetscFunctionBegin;
  if (!MatOrderingRegisterAllCalled) {ierr = MatOrderingRegisterAll(PETSC_NULL);CHKERRQ(ierr);}
  ierr = PetscOptionsTruth("-pc_factor_in_place","Form factored matrix in the same memory as the matrix","PCFactorSetUseInPlace",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg) {
    ierr = PCFactorSetUseInPlace(pc);CHKERRQ(ierr);
  }
  ierr = PetscOptionsReal("-pc_factor_fill","Expected non-zeros in factored matrix","PCFactorSetFill",((PC_Factor*)factor)->info.fill,&((PC_Factor*)factor)->info.fill,0);CHKERRQ(ierr);

  ierr = PetscOptionsEnum("-pc_factor_shift_type","Shift added to diagonal","PCFactorSetShiftType",
                          MatFactorShiftTypes,(PetscEnum)(int)((PC_Factor*)factor)->info.shifttype,&etmp,&flg);CHKERRQ(ierr);
  if (flg) {
    ((PC_Factor*)factor)->info.shifttype = (PetscReal)(etmp);
  }
  ierr = PetscOptionsReal("-pc_factor_shift_amount","Shift added to diagonal","PCFactorSetShiftAmount",((PC_Factor*)factor)->info.shiftamount,&((PC_Factor*)factor)->info.shiftamount,0);CHKERRQ(ierr);
  
  ierr = PetscOptionsReal("-pc_factor_zeropivot","Pivot is considered zero if less than","PCFactorSetZeroPivot",((PC_Factor*)factor)->info.zeropivot,&((PC_Factor*)factor)->info.zeropivot,0);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-pc_factor_column_pivot","Column pivot tolerance (used only for some factorization)","PCFactorSetColumnPivot",((PC_Factor*)factor)->info.dtcol,&((PC_Factor*)factor)->info.dtcol,&flg);CHKERRQ(ierr);

  flg = ((PC_Factor*)factor)->info.pivotinblocks ? PETSC_TRUE : PETSC_FALSE;
  ierr = PetscOptionsTruth("-pc_factor_pivot_in_blocks","Pivot inside matrix dense blocks for BAIJ and SBAIJ","PCFactorSetPivotInBlocks",flg,&flg,&set);CHKERRQ(ierr);
  if (set) {
    ierr = PCFactorSetPivotInBlocks(pc,flg);CHKERRQ(ierr);
  }
  
  flg  = PETSC_FALSE;
  ierr = PetscOptionsTruth("-pc_factor_reuse_fill","Use fill from previous factorization","PCFactorSetReuseFill",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg) {
    ierr = PCFactorSetReuseFill(pc,PETSC_TRUE);CHKERRQ(ierr);
  }
  flg  = PETSC_FALSE;
  ierr = PetscOptionsTruth("-pc_factor_reuse_ordering","Reuse ordering from previous factorization","PCFactorSetReuseOrdering",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg) {
    ierr = PCFactorSetReuseOrdering(pc,PETSC_TRUE);CHKERRQ(ierr);
  }

  ierr = MatGetOrderingList(&ordlist);CHKERRQ(ierr);
  ierr = PetscOptionsList("-pc_factor_mat_ordering_type","Reordering to reduce nonzeros in factored matrix","PCFactorSetMatOrderingType",ordlist,((PC_Factor*)factor)->ordering,tname,256,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PCFactorSetMatOrderingType(pc,tname);CHKERRQ(ierr);
  }

  /* maybe should have MatGetSolverTypes(Mat,&list) like the ordering list */
  ierr = PetscOptionsString("-pc_factor_mat_solver_package","Specific direct solver to use","MatGetFactor",((PC_Factor*)factor)->solvertype,solvertype,64,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PCFactorSetMatSolverPackage(pc,solvertype);CHKERRQ(ierr);
  }   
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCView_Factor"
PetscErrorCode PCView_Factor(PC pc,PetscViewer viewer)
{
  PC_Factor       *factor = (PC_Factor*)pc->data;
  PetscErrorCode  ierr;
  PetscTruth      isstring,iascii;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_STRING,&isstring);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    if (factor->factortype == MAT_FACTOR_ILU || factor->factortype == MAT_FACTOR_ICC){
      if (factor->info.dt > 0) {
        ierr = PetscViewerASCIIPrintf(viewer,"  drop tolerance %G\n",factor->info.dt);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  max nonzeros per row %D\n",factor->info.dtcount);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  column permutation tolerance %G\n",(PetscInt)factor->info.dtcol);CHKERRQ(ierr);
      } else if (factor->info.levels == 1) {
        ierr = PetscViewerASCIIPrintf(viewer,"  %D level of fill\n",(PetscInt)factor->info.levels);CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIIPrintf(viewer,"  %D levels of fill\n",(PetscInt)factor->info.levels);CHKERRQ(ierr);
      }
    }
    
    ierr = PetscViewerASCIIPrintf(viewer,"  tolerance for zero pivot %G\n",factor->info.zeropivot);CHKERRQ(ierr);
    if (factor->info.shifttype==(PetscReal)MAT_SHIFT_POSITIVE_DEFINITE) {
      ierr = PetscViewerASCIIPrintf(viewer,"  using Manteuffel shift\n");CHKERRQ(ierr);
    }
    if (factor->info.shifttype==(PetscReal)MAT_SHIFT_NONZERO) {
      ierr = PetscViewerASCIIPrintf(viewer,"  using diagonal shift to prevent zero pivot\n");CHKERRQ(ierr);
    }
    if (factor->info.shifttype==(PetscReal)MAT_SHIFT_INBLOCKS) {
      ierr = PetscViewerASCIIPrintf(viewer,"  using diagonal shift on blocks to prevent zero pivot\n");CHKERRQ(ierr);
    }
    
    ierr = PetscViewerASCIIPrintf(viewer,"  matrix ordering: %s\n",factor->ordering);CHKERRQ(ierr);
    
    if (factor->fact) {
      MatInfo info;
      ierr = MatGetInfo(factor->fact,MAT_LOCAL,&info);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  factor fill ratio given %G, needed %G\n",info.fill_ratio_given,info.fill_ratio_needed);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"    Factored matrix follows:\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr);
      ierr = MatView(factor->fact,viewer);CHKERRQ(ierr);
      ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
    
  } else if (isstring) {
    MatFactorType t;
    ierr = MatGetFactorType(factor->fact,&t);CHKERRQ(ierr);
    if (t == MAT_FACTOR_ILU || t == MAT_FACTOR_ICC){
      ierr = PetscViewerStringSPrintf(viewer," lvls=%D,order=%s",(PetscInt)factor->info.levels,factor->ordering);CHKERRQ(ierr);CHKERRQ(ierr);
    }
  } else {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported for PC_Factor",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END
