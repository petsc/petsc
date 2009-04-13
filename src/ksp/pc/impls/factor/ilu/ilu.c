#define PETSCKSP_DLL

/*
   Defines a ILU factorization preconditioner for any Mat implementation
*/
#include "../src/ksp/pc/impls/factor/ilu/ilu.h"     /*I "petscpc.h"  I*/

/* ------------------------------------------------------------------------------------------*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetReuseFill_ILU"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetReuseFill_ILU(PC pc,PetscTruth flag)
{
  PC_ILU *lu = (PC_ILU*)pc->data;
  
  PetscFunctionBegin;
  lu->reusefill = flag;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFactorReorderForNonzeroDiagonal_ILU"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorReorderForNonzeroDiagonal_ILU(PC pc,PetscReal z)
{
  PC_ILU *ilu = (PC_ILU*)pc->data;

  PetscFunctionBegin;
  ilu->nonzerosalongdiagonal = PETSC_TRUE;                 
  if (z == PETSC_DECIDE) {
    ilu->nonzerosalongdiagonaltol = 1.e-10;
  } else {
    ilu->nonzerosalongdiagonaltol = z;
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "PCDestroy_ILU_Internal"
PetscErrorCode PCDestroy_ILU_Internal(PC pc)
{
  PC_ILU         *ilu = (PC_ILU*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!ilu->inplace && ((PC_Factor*)ilu)->fact) {ierr = MatDestroy(((PC_Factor*)ilu)->fact);CHKERRQ(ierr);}
  if (ilu->row && ilu->col && ilu->row != ilu->col) {ierr = ISDestroy(ilu->row);CHKERRQ(ierr);}
  if (ilu->col) {ierr = ISDestroy(ilu->col);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetUseDropTolerance_ILU"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetUseDropTolerance_ILU(PC pc,PetscReal dt,PetscReal dtcol,PetscInt dtcount)
{
  PC_ILU         *ilu = (PC_ILU*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (pc->setupcalled && (!ilu->usedt || ((PC_Factor*)ilu)->info.dt != dt || ((PC_Factor*)ilu)->info.dtcol != dtcol || ((PC_Factor*)ilu)->info.dtcount != dtcount)) {
    pc->setupcalled   = 0;
    ierr = PCDestroy_ILU_Internal(pc);CHKERRQ(ierr);
  }
  ilu->usedt                      = PETSC_TRUE;
  ((PC_Factor*)ilu)->info.dt      = dt;
  ((PC_Factor*)ilu)->info.dtcol   = dtcol;
  ((PC_Factor*)ilu)->info.dtcount = dtcount;
  ((PC_Factor*)ilu)->info.fill    = PETSC_DEFAULT;
  PetscFunctionReturn(0);
}  
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetReuseOrdering_ILU"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetReuseOrdering_ILU(PC pc,PetscTruth flag)
{
  PC_ILU *ilu = (PC_ILU*)pc->data;

  PetscFunctionBegin;
  ilu->reuseordering = flag;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetUseInPlace_ILU"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetUseInPlace_ILU(PC pc)
{
  PC_ILU *dir = (PC_ILU*)pc->data;

  PetscFunctionBegin;
  dir->inplace = PETSC_TRUE;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "PCSetFromOptions_ILU"
static PetscErrorCode PCSetFromOptions_ILU(PC pc)
{
  PetscErrorCode ierr;
  PetscInt       dtmax = 3,itmp;
  PetscTruth     flg,set;
  PetscReal      dt[3];
  char           tname[256];
  PC_ILU         *ilu = (PC_ILU*)pc->data;
  PetscFList     ordlist;
  PetscReal      tol;

  PetscFunctionBegin;
  if (!MatOrderingRegisterAllCalled) {ierr = MatOrderingRegisterAll(PETSC_NULL);CHKERRQ(ierr);}
  ierr = PetscOptionsHead("ILU Options");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-pc_factor_levels","levels of fill","PCFactorSetLevels",(PetscInt)((PC_Factor*)ilu)->info.levels,&itmp,&flg);CHKERRQ(ierr);
    if (flg) ((PC_Factor*)ilu)->info.levels = itmp;
    ierr = PetscOptionsTruth("-pc_factor_in_place","do factorization in place","PCFactorSetUseInPlace",ilu->inplace,&ilu->inplace,PETSC_NULL);CHKERRQ(ierr);
    flg  = PETSC_FALSE;
    ierr = PetscOptionsTruth("-pc_factor_diagonal_fill","Allow fill into empty diagonal entry","PCFactorSetAllowDiagonalFill",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
    ((PC_Factor*)ilu)->info.diagonal_fill = (double) flg;
    ierr = PetscOptionsTruth("-pc_factor_reuse_fill","Reuse fill ratio from previous factorization","PCFactorSetReuseFill",ilu->reusefill,&ilu->reusefill,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-pc_factor_reuse_ordering","Reuse previous reordering","PCFactorSetReuseOrdering",ilu->reuseordering,&ilu->reuseordering,PETSC_NULL);CHKERRQ(ierr);
    flg  = PETSC_FALSE;
    ierr = PetscOptionsTruth("-pc_factor_shift_nonzero","Shift added to diagonal","PCFactorSetShiftNonzero",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) {
      ierr = PCFactorSetShiftNonzero(pc,(PetscReal)PETSC_DECIDE);CHKERRQ(ierr);
    }
    ierr = PetscOptionsReal("-pc_factor_shift_nonzero","Shift added to diagonal","PCFactorSetShiftNonzero",((PC_Factor*)ilu)->info.shiftnz,&((PC_Factor*)ilu)->info.shiftnz,0);CHKERRQ(ierr);
    flg = (((PC_Factor*)ilu)->info.shiftpd > 0.0) ? PETSC_TRUE : PETSC_FALSE;
    ierr = PetscOptionsTruth("-pc_factor_shift_positive_definite","Manteuffel shift applied to diagonal","PCFactorSetShiftPd",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
    ierr = PCFactorSetShiftPd(pc,flg);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-pc_factor_zeropivot","Pivot is considered zero if less than","PCFactorSetZeroPivot",((PC_Factor*)ilu)->info.zeropivot,&((PC_Factor*)ilu)->info.zeropivot,0);CHKERRQ(ierr);

    dt[0] = ((PC_Factor*)ilu)->info.dt;
    dt[1] = ((PC_Factor*)ilu)->info.dtcol;
    dt[2] = ((PC_Factor*)ilu)->info.dtcount;
    ierr = PetscOptionsRealArray("-pc_factor_use_drop_tolerance","<dt,dtcol,maxrowcount>","PCFactorSetUseDropTolerance",dt,&dtmax,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PCFactorSetUseDropTolerance(pc,dt[0],dt[1],(PetscInt)dt[2]);CHKERRQ(ierr);
    }
    ierr = PetscOptionsReal("-pc_factor_fill","Expected fill in factorization","PCFactorSetFill",((PC_Factor*)ilu)->info.fill,&((PC_Factor*)ilu)->info.fill,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsName("-pc_factor_nonzeros_along_diagonal","Reorder to remove zeros from diagonal","PCFactorReorderForNonzeroDiagonal",&flg);CHKERRQ(ierr);
    if (flg) {
      tol = PETSC_DECIDE;
      ierr = PetscOptionsReal("-pc_factor_nonzeros_along_diagonal","Reorder to remove zeros from diagonal","PCFactorReorderForNonzeroDiagonal",ilu->nonzerosalongdiagonaltol,&tol,0);CHKERRQ(ierr);
      ierr = PCFactorReorderForNonzeroDiagonal(pc,tol);CHKERRQ(ierr);
    }

    ierr = MatGetOrderingList(&ordlist);CHKERRQ(ierr);
    ierr = PetscOptionsList("-pc_factor_mat_ordering_type","Reorder to reduce nonzeros in ILU","PCFactorSetMatOrderingType",ordlist,((PC_Factor*)ilu)->ordering,tname,256,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PCFactorSetMatOrderingType(pc,tname);CHKERRQ(ierr);
    }
    flg = ((PC_Factor*)ilu)->info.pivotinblocks ? PETSC_TRUE : PETSC_FALSE;
    ierr = PetscOptionsTruth("-pc_factor_pivot_in_blocks","Pivot inside matrix blocks for BAIJ and SBAIJ","PCFactorSetPivotInBlocks",flg,&flg,&set);CHKERRQ(ierr);
    if (set) {
      ierr = PCFactorSetPivotInBlocks(pc,flg);CHKERRQ(ierr);
    }
    flg  = PETSC_FALSE;
    ierr = PetscOptionsTruth("-pc_factor_shift_in_blocks","Shift added to diagonal of block","PCFactorSetShiftInBlocks",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) {
      ierr = PCFactorSetShiftInBlocks(pc,(PetscReal)PETSC_DECIDE);CHKERRQ(ierr);
    }
    ierr = PetscOptionsReal("-pc_factor_shift_in_blocks","Shift added to diagonal of block","PCFactorSetShiftInBlocks",((PC_Factor*)ilu)->info.shiftinblocks,&((PC_Factor*)ilu)->info.shiftinblocks,0);CHKERRQ(ierr);

  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCView_ILU"
static PetscErrorCode PCView_ILU(PC pc,PetscViewer viewer)
{
  PC_ILU         *ilu = (PC_ILU*)pc->data;
  PetscErrorCode ierr;
  PetscTruth     isstring,iascii;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_STRING,&isstring);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    if (ilu->usedt) {
        ierr = PetscViewerASCIIPrintf(viewer,"  ILU: drop tolerance %G\n",((PC_Factor*)ilu)->info.dt);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  ILU: max nonzeros per row %D\n",(PetscInt)((PC_Factor*)ilu)->info.dtcount);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  ILU: column permutation tolerance %G\n",((PC_Factor*)ilu)->info.dtcol);CHKERRQ(ierr);
    } else if (((PC_Factor*)ilu)->info.levels == 1) {
        ierr = PetscViewerASCIIPrintf(viewer,"  ILU: %D level of fill\n",(PetscInt)((PC_Factor*)ilu)->info.levels);CHKERRQ(ierr);
    } else {
        ierr = PetscViewerASCIIPrintf(viewer,"  ILU: %D levels of fill\n",(PetscInt)((PC_Factor*)ilu)->info.levels);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  ILU: factor fill ratio allocated %G\n",((PC_Factor*)ilu)->info.fill);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  ILU: tolerance for zero pivot %G\n",((PC_Factor*)ilu)->info.zeropivot);CHKERRQ(ierr);
    if (((PC_Factor*)ilu)->info.shiftpd) {ierr = PetscViewerASCIIPrintf(viewer,"  ILU: using Manteuffel shift\n");CHKERRQ(ierr);}
    if (((PC_Factor*)ilu)->info.shiftnz) {ierr = PetscViewerASCIIPrintf(viewer,"  ILU: using diagonal shift to prevent zero pivot\n");CHKERRQ(ierr);}
    if (((PC_Factor*)ilu)->info.shiftinblocks) {ierr = PetscViewerASCIIPrintf(viewer,"  ILU: using diagonal shift on blocks to prevent zero pivot\n");CHKERRQ(ierr);}
    if (ilu->inplace) {ierr = PetscViewerASCIIPrintf(viewer,"       in-place factorization\n");CHKERRQ(ierr);}
    else              {ierr = PetscViewerASCIIPrintf(viewer,"       out-of-place factorization\n");CHKERRQ(ierr);}
    ierr = PetscViewerASCIIPrintf(viewer,"       matrix ordering: %s\n",((PC_Factor*)ilu)->ordering);CHKERRQ(ierr);
    if (ilu->reusefill)     {ierr = PetscViewerASCIIPrintf(viewer,"       Reusing fill from past factorization\n");CHKERRQ(ierr);}
    if (ilu->reuseordering) {ierr = PetscViewerASCIIPrintf(viewer,"       Reusing reordering from past factorization\n");CHKERRQ(ierr);}
    if (((PC_Factor*)ilu)->fact) {
      ierr = PetscViewerASCIIPrintf(viewer,"  ILU: factor fill ratio needed %G\n",ilu->actualfill);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"       Factored matrix follows\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr);
      ierr = MatView(((PC_Factor*)ilu)->fact,viewer);CHKERRQ(ierr);
      ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
  } else if (isstring) {
    ierr = PetscViewerStringSPrintf(viewer," lvls=%D,order=%s",(PetscInt)((PC_Factor*)ilu)->info.levels,((PC_Factor*)ilu)->ordering);CHKERRQ(ierr);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported for PCILU",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSetUp_ILU"
static PetscErrorCode PCSetUp_ILU(PC pc)
{
  PetscErrorCode ierr;
  PC_ILU         *ilu = (PC_ILU*)pc->data;
  MatInfo        info;

  PetscFunctionBegin;
  if (ilu->inplace) {
    CHKMEMQ;
    if (!pc->setupcalled) {

      /* In-place factorization only makes sense with the natural ordering,
         so we only need to get the ordering once, even if nonzero structure changes */
      ierr = MatGetOrdering(pc->pmat,((PC_Factor*)ilu)->ordering,&ilu->row,&ilu->col);CHKERRQ(ierr);
      if (ilu->row) {ierr = PetscLogObjectParent(pc,ilu->row);CHKERRQ(ierr);}
      if (ilu->col) {ierr = PetscLogObjectParent(pc,ilu->col);CHKERRQ(ierr);}
    }

    /* In place ILU only makes sense with fill factor of 1.0 because 
       cannot have levels of fill */
    ((PC_Factor*)ilu)->info.fill          = 1.0;
    ((PC_Factor*)ilu)->info.diagonal_fill = 0;
    ierr = MatILUFactor(pc->pmat,ilu->row,ilu->col,&((PC_Factor*)ilu)->info);CHKERRQ(ierr);
    CHKMEMQ;
    ((PC_Factor*)ilu)->fact = pc->pmat;
  } else if (ilu->usedt) {
    if (!pc->setupcalled) {
      ierr = MatGetOrdering(pc->pmat,((PC_Factor*)ilu)->ordering,&ilu->row,&ilu->col);CHKERRQ(ierr);
    CHKMEMQ;
      if (ilu->row) {ierr = PetscLogObjectParent(pc,ilu->row);CHKERRQ(ierr);}
      if (ilu->col) {ierr = PetscLogObjectParent(pc,ilu->col);CHKERRQ(ierr);}
      ierr = MatILUDTFactor(pc->pmat,ilu->row,ilu->col,&((PC_Factor*)ilu)->info,&((PC_Factor*)ilu)->fact);CHKERRQ(ierr);
      ierr = PetscLogObjectParent(pc,((PC_Factor*)ilu)->fact);CHKERRQ(ierr);
    } else if (pc->flag != SAME_NONZERO_PATTERN) { 
    CHKMEMQ;
      ierr = MatDestroy(((PC_Factor*)ilu)->fact);CHKERRQ(ierr);
    CHKMEMQ;
      if (!ilu->reuseordering) {
        if (ilu->row) {ierr = ISDestroy(ilu->row);CHKERRQ(ierr);}
        if (ilu->col) {ierr = ISDestroy(ilu->col);CHKERRQ(ierr);}
        ierr = MatGetOrdering(pc->pmat,((PC_Factor*)ilu)->ordering,&ilu->row,&ilu->col);CHKERRQ(ierr);
        if (ilu->row) {ierr = PetscLogObjectParent(pc,ilu->row);CHKERRQ(ierr);}
        if (ilu->col) {ierr = PetscLogObjectParent(pc,ilu->col);CHKERRQ(ierr);}
      }
      ierr = MatILUDTFactor(pc->pmat,ilu->row,ilu->col,&((PC_Factor*)ilu)->info,&((PC_Factor*)ilu)->fact);CHKERRQ(ierr);
      ierr = PetscLogObjectParent(pc,((PC_Factor*)ilu)->fact);CHKERRQ(ierr);
    } else if (!ilu->reusefill) { 
      ierr = MatDestroy(((PC_Factor*)ilu)->fact);CHKERRQ(ierr);
      ierr = MatILUDTFactor(pc->pmat,ilu->row,ilu->col,&((PC_Factor*)ilu)->info,&((PC_Factor*)ilu)->fact);CHKERRQ(ierr);
      ierr = PetscLogObjectParent(pc,((PC_Factor*)ilu)->fact);CHKERRQ(ierr);
    } else {
      ierr = MatLUFactorNumeric(((PC_Factor*)ilu)->fact,pc->pmat,&((PC_Factor*)ilu)->info);CHKERRQ(ierr);
    }
  } else {
    if (!pc->setupcalled) {
      /* first time in so compute reordering and symbolic factorization */
      ierr = MatGetOrdering(pc->pmat,((PC_Factor*)ilu)->ordering,&ilu->row,&ilu->col);CHKERRQ(ierr);
      if (ilu->row) {ierr = PetscLogObjectParent(pc,ilu->row);CHKERRQ(ierr);}
      if (ilu->col) {ierr = PetscLogObjectParent(pc,ilu->col);CHKERRQ(ierr);}
      /*  Remove zeros along diagonal?     */
      if (ilu->nonzerosalongdiagonal) {
        ierr = MatReorderForNonzeroDiagonal(pc->pmat,ilu->nonzerosalongdiagonaltol,ilu->row,ilu->col);CHKERRQ(ierr);
      }
    CHKMEMQ;
      ierr = MatGetFactor(pc->pmat,MAT_SOLVER_PETSC,MAT_FACTOR_ILU,&((PC_Factor*)ilu)->fact);CHKERRQ(ierr);
    CHKMEMQ;
      ierr = MatILUFactorSymbolic(((PC_Factor*)ilu)->fact,pc->pmat,ilu->row,ilu->col,&((PC_Factor*)ilu)->info);CHKERRQ(ierr);
      ierr = MatGetInfo(((PC_Factor*)ilu)->fact,MAT_LOCAL,&info);CHKERRQ(ierr);
      ilu->actualfill = info.fill_ratio_needed;
      ierr = PetscLogObjectParent(pc,((PC_Factor*)ilu)->fact);CHKERRQ(ierr);
    } else if (pc->flag != SAME_NONZERO_PATTERN) { 
      if (!ilu->reuseordering) {
        /* compute a new ordering for the ILU */
        ierr = ISDestroy(ilu->row);CHKERRQ(ierr);
        ierr = ISDestroy(ilu->col);CHKERRQ(ierr);
        ierr = MatGetOrdering(pc->pmat,((PC_Factor*)ilu)->ordering,&ilu->row,&ilu->col);CHKERRQ(ierr);
        if (ilu->row) {ierr = PetscLogObjectParent(pc,ilu->row);CHKERRQ(ierr);}
        if (ilu->col) {ierr = PetscLogObjectParent(pc,ilu->col);CHKERRQ(ierr);}
        /*  Remove zeros along diagonal?     */
        if (ilu->nonzerosalongdiagonal) {
          ierr = MatReorderForNonzeroDiagonal(pc->pmat,ilu->nonzerosalongdiagonaltol,ilu->row,ilu->col);CHKERRQ(ierr);
        }
      }
      ierr = MatDestroy(((PC_Factor*)ilu)->fact);CHKERRQ(ierr);
      ierr = MatGetFactor(pc->pmat,MAT_SOLVER_PETSC,MAT_FACTOR_ILU,&((PC_Factor*)ilu)->fact);CHKERRQ(ierr);
      ierr = MatILUFactorSymbolic(((PC_Factor*)ilu)->fact,pc->pmat,ilu->row,ilu->col,&((PC_Factor*)ilu)->info);CHKERRQ(ierr);
      ierr = MatGetInfo(((PC_Factor*)ilu)->fact,MAT_LOCAL,&info);CHKERRQ(ierr);
      ilu->actualfill = info.fill_ratio_needed;
      ierr = PetscLogObjectParent(pc,((PC_Factor*)ilu)->fact);CHKERRQ(ierr);
    }
    CHKMEMQ;
    ierr = MatLUFactorNumeric(((PC_Factor*)ilu)->fact,pc->pmat,&((PC_Factor*)ilu)->info);CHKERRQ(ierr);
    CHKMEMQ;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCDestroy_ILU"
static PetscErrorCode PCDestroy_ILU(PC pc)
{
  PC_ILU         *ilu = (PC_ILU*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCDestroy_ILU_Internal(pc);CHKERRQ(ierr);
  ierr = PetscStrfree(((PC_Factor*)ilu)->ordering);CHKERRQ(ierr);
  ierr = PetscFree(ilu);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCApply_ILU"
static PetscErrorCode PCApply_ILU(PC pc,Vec x,Vec y)
{
  PC_ILU         *ilu = (PC_ILU*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSolve(((PC_Factor*)ilu)->fact,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCApplyTranspose_ILU"
static PetscErrorCode PCApplyTranspose_ILU(PC pc,Vec x,Vec y)
{
  PC_ILU         *ilu = (PC_ILU*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSolveTranspose(((PC_Factor*)ilu)->fact,x,y);CHKERRQ(ierr);
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
.  -pc_factor_use_drop_tolerance <dt,dtcol,maxrowcount> - use Saad's drop tolerance ILUdt
.  -pc_factor_fill <nfill> - expected amount of fill in factored matrix compared to original matrix, nfill > 1
.  -pc_factor_nonzeros_along_diagonal - reorder the matrix before factorization to remove zeros from the diagonal,
                                   this decreases the chance of getting a zero pivot
.  -pc_factor_mat_ordering_type <natural,nd,1wd,rcm,qmd> - set the row/column ordering of the factored matrix
.  -pc_factor_pivot_in_blocks - for block ILU(k) factorization, i.e. with BAIJ matrices with block size larger
                             than 1 the diagonal blocks are factored with partial pivoting (this increases the 
                             stability of the ILU factorization
.  -pc_factor_shift_in_blocks - adds a small diagonal to any block if it is singular during ILU factorization
.  -pc_factor_shift_nonzero <shift> - Sets shift amount or PETSC_DECIDE for the default
-  -pc_factor_shift_positive_definite true or false - Activate/Deactivate PCFactorSetShiftPd(); the value
   is optional with true being the default

   Level: beginner

  Concepts: incomplete factorization

   Notes: Only implemented for some matrix formats. (for parallel use you 
             must use MATMPIROWBS, see MatCreateMPIRowbs(), this supports only ILU(0) and this is not recommended
             unless you really want a parallel ILU).

          For BAIJ matrices this implements a point block ILU

   References:
   T. Dupont, R. Kendall, and H. Rachford. An approximate factorization procedure for solving
   self-adjoint elliptic difference equations. SIAM J. Numer. Anal., 5:559--573, 1968.

   T.A. Oliphant. An implicit numerical method for solving two-dimensional time-dependent dif-
   fusion problems. Quart. Appl. Math., 19:221--229, 1961.

   Review article: APPROXIMATE AND INCOMPLETE FACTORIZATIONS, TONY F. CHAN AND HENK A. VAN DER VORST
      http://igitur-archive.library.uu.nl/math/2001-0621-115821/proc.pdf chapter in Parallel Numerical
      Algorithms, edited by D. Keyes, A. Semah, V. Venkatakrishnan, ICASE/LaRC Interdisciplinary Series in
      Science and Engineering, Kluwer, pp. 167--202.


.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC, PCSOR, MatOrderingType,
           PCFactorSetZeroPivot(), PCFactorSetShiftNonzero(), PCFactorSetShiftPd(), PCFactorSetUseDropTolerance(),
           PCFactorSetFill(), PCFactorSetMatOrderingType(), PCFactorSetReuseOrdering(),
           PCFactorSetLevels(), PCFactorSetUseInPlace(), PCFactorSetAllowDiagonalFill(), PCFactorSetPivotInBlocks(),
           PCFactorSetShiftNonzero(),PCFactorSetShiftPd()

M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCreate_ILU"
PetscErrorCode PETSCKSP_DLLEXPORT PCCreate_ILU(PC pc)
{
  PetscErrorCode ierr;
  PC_ILU         *ilu;

  PetscFunctionBegin;
  ierr = PetscNewLog(pc,PC_ILU,&ilu);CHKERRQ(ierr);

  ((PC_Factor*)ilu)->fact                    = 0;
  ierr = MatFactorInfoInitialize(&((PC_Factor*)ilu)->info);CHKERRQ(ierr);
  ((PC_Factor*)ilu)->info.levels             = 0;
  ((PC_Factor*)ilu)->info.fill               = 1.0; 
  ilu->col                     = 0;
  ilu->row                     = 0;
  ilu->inplace                 = PETSC_FALSE;
  ierr = PetscStrallocpy(MATORDERING_NATURAL,&((PC_Factor*)ilu)->ordering);CHKERRQ(ierr);
  ilu->reuseordering           = PETSC_FALSE;
  ilu->usedt                   = PETSC_FALSE;
  ((PC_Factor*)ilu)->info.dt                 = PETSC_DEFAULT;
  ((PC_Factor*)ilu)->info.dtcount            = PETSC_DEFAULT;
  ((PC_Factor*)ilu)->info.dtcol              = PETSC_DEFAULT;
  ((PC_Factor*)ilu)->info.shiftnz            = 1.e-12;
  ((PC_Factor*)ilu)->info.shiftpd            = 0.0; /* false */
  ((PC_Factor*)ilu)->info.zeropivot          = 1.e-12;
  ((PC_Factor*)ilu)->info.pivotinblocks      = 1.0;
  ((PC_Factor*)ilu)->info.shiftinblocks      = 1.e-12;
  ilu->reusefill               = PETSC_FALSE;
  ((PC_Factor*)ilu)->info.diagonal_fill      = 0;
  pc->data                     = (void*)ilu;

  pc->ops->destroy             = PCDestroy_ILU;
  pc->ops->apply               = PCApply_ILU;
  pc->ops->applytranspose      = PCApplyTranspose_ILU;
  pc->ops->setup               = PCSetUp_ILU;
  pc->ops->setfromoptions      = PCSetFromOptions_ILU;
  pc->ops->getfactoredmatrix   = PCFactorGetMatrix_Factor;
  pc->ops->view                = PCView_ILU;
  pc->ops->applyrichardson     = 0;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetZeroPivot_C","PCFactorSetZeroPivot_Factor",
                    PCFactorSetZeroPivot_Factor);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetShiftNonzero_C","PCFactorSetShiftNonzero_Factor",
                    PCFactorSetShiftNonzero_Factor);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetShiftPd_C","PCFactorSetShiftPd_Factor",
                    PCFactorSetShiftPd_Factor);CHKERRQ(ierr);

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorGetMatSolverPackage_C","PCFactorGetMatSolverPackage_Factor",
                    PCFactorGetMatSolverPackage_Factor);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetUseDropTolerance_C","PCFactorSetUseDropTolerance_ILU",
                    PCFactorSetUseDropTolerance_ILU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetFill_C","PCFactorSetFill_Factor",
                    PCFactorSetFill_Factor);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetMatOrderingType_C","PCFactorSetMatOrderingType_Factor",
                    PCFactorSetMatOrderingType_Factor);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetReuseOrdering_C","PCFactorSetReuseOrdering_ILU",
                    PCFactorSetReuseOrdering_ILU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetReuseFill_C","PCFactorSetReuseFill_ILU",
                    PCFactorSetReuseFill_ILU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetLevels_C","PCFactorSetLevels_Factor",
                    PCFactorSetLevels_Factor);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetUseInPlace_C","PCFactorSetUseInPlace_ILU",
                    PCFactorSetUseInPlace_ILU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetAllowDiagonalFill_C","PCFactorSetAllowDiagonalFill_Factor",
                    PCFactorSetAllowDiagonalFill_Factor);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetPivotInBlocks_C","PCFactorSetPivotInBlocks_Factor",
                    PCFactorSetPivotInBlocks_Factor);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetShiftInBlocks_C","PCFactorSetShiftInBlocks_Factor",
                    PCFactorSetShiftInBlocks_Factor);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorReorderForNonzeroDiagonal_C","PCFactorReorderForNonzeroDiagonal_ILU",
                    PCFactorReorderForNonzeroDiagonal_ILU);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
