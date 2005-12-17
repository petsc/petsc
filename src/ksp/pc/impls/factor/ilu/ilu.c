#define PETSCKSP_DLL

/*
   Defines a ILU factorization preconditioner for any Mat implementation
*/
#include "private/pcimpl.h"                 /*I "petscpc.h"  I*/
#include "src/ksp/pc/impls/factor/ilu/ilu.h"
#include "src/mat/matimpl.h"

/* ------------------------------------------------------------------------------------------*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetReuseFill_ILU"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetReuseFill_ILU(PC pc,PetscTruth flag)
{
  PC_ILU *lu;
  
  PetscFunctionBegin;
  lu = (PC_ILU*)pc->data;
  lu->reusefill = flag;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetZeroPivot_ILU"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetZeroPivot_ILU(PC pc,PetscReal z)
{
  PC_ILU *ilu;

  PetscFunctionBegin;
  ilu                 = (PC_ILU*)pc->data;
  ilu->info.zeropivot = z;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetShiftNonzero_ILU"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetShiftNonzero_ILU(PC pc,PetscReal shift)
{
  PC_ILU *dir;

  PetscFunctionBegin;
  dir = (PC_ILU*)pc->data;
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
#define __FUNCT__ "PCFactorSetShiftPd_ILU"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetShiftPd_ILU(PC pc,PetscTruth shift)
{
  PC_ILU *dir;
 
  PetscFunctionBegin;
  dir = (PC_ILU*)pc->data;
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
  if (!ilu->inplace && ilu->fact) {ierr = MatDestroy(ilu->fact);CHKERRQ(ierr);}
  if (ilu->row && ilu->col && ilu->row != ilu->col) {ierr = ISDestroy(ilu->row);CHKERRQ(ierr);}
  if (ilu->col) {ierr = ISDestroy(ilu->col);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetUseDropTolerance_ILU"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetUseDropTolerance_ILU(PC pc,PetscReal dt,PetscReal dtcol,PetscInt dtcount)
{
  PC_ILU         *ilu;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ilu = (PC_ILU*)pc->data;
  if (pc->setupcalled && (!ilu->usedt || ilu->info.dt != dt || ilu->info.dtcol != dtcol || ilu->info.dtcount != dtcount)) {
    pc->setupcalled   = 0;
    ierr = PCDestroy_ILU_Internal(pc);CHKERRQ(ierr);
  }
  ilu->usedt        = PETSC_TRUE;
  ilu->info.dt      = dt;
  ilu->info.dtcol   = dtcol;
  ilu->info.dtcount = dtcount;
  ilu->info.fill    = PETSC_DEFAULT;
  PetscFunctionReturn(0);
}  
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetFill_ILU"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetFill_ILU(PC pc,PetscReal fill)
{
  PC_ILU *dir;

  PetscFunctionBegin;
  dir            = (PC_ILU*)pc->data;
  dir->info.fill = fill;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetMatOrdering_ILU"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetMatOrdering_ILU(PC pc,MatOrderingType ordering)
{
  PC_ILU         *dir = (PC_ILU*)pc->data;
  PetscErrorCode ierr;
  PetscTruth     flg;
 
  PetscFunctionBegin;
  if (!pc->setupcalled) {
     ierr = PetscStrfree(dir->ordering);CHKERRQ(ierr);
     ierr = PetscStrallocpy(ordering,&dir->ordering);CHKERRQ(ierr);
  } else {
    ierr = PetscStrcmp(dir->ordering,ordering,&flg);CHKERRQ(ierr);
    if (!flg) {
      pc->setupcalled = 0;
      ierr = PetscStrfree(dir->ordering);CHKERRQ(ierr);
      ierr = PetscStrallocpy(ordering,&dir->ordering);CHKERRQ(ierr);
      /* free the data structures, then create them again */
      ierr = PCDestroy_ILU_Internal(pc);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetReuseOrdering_ILU"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetReuseOrdering_ILU(PC pc,PetscTruth flag)
{
  PC_ILU *ilu;

  PetscFunctionBegin;
  ilu                = (PC_ILU*)pc->data;
  ilu->reuseordering = flag;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetLevels_ILU"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetLevels_ILU(PC pc,PetscInt levels)
{
  PC_ILU         *ilu;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ilu = (PC_ILU*)pc->data;

  if (!pc->setupcalled) {
    ilu->info.levels = levels;
  } else if (ilu->usedt || ilu->info.levels != levels) {
    ilu->info.levels = levels;
    pc->setupcalled  = 0;
    ilu->usedt       = PETSC_FALSE;
    ierr = PCDestroy_ILU_Internal(pc);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetUseInPlace_ILU"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetUseInPlace_ILU(PC pc)
{
  PC_ILU *dir;

  PetscFunctionBegin;
  dir          = (PC_ILU*)pc->data;
  dir->inplace = PETSC_TRUE;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetAllowDiagonalFill_ILU"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetAllowDiagonalFill_ILU(PC pc)
{
  PC_ILU *dir;

  PetscFunctionBegin;
  dir                 = (PC_ILU*)pc->data;
  dir->info.diagonal_fill = 1;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetUseDropTolerance"
/*@
   PCFactorSetUseDropTolerance - The preconditioner will use an ILU 
   based on a drop tolerance.

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
.  dt - the drop tolerance, try from 1.e-10 to .1
.  dtcol - tolerance for column pivot, good values [0.1 to 0.01]
-  maxrowcount - the max number of nonzeros allowed in a row, best value
                 depends on the number of nonzeros in row of original matrix

   Options Database Key:
.  -pc_factor_use_drop_tolerance <dt,dtcol,maxrowcount> - Sets drop tolerance

   Level: intermediate

    Notes:
      This uses the iludt() code of Saad's SPARSKIT package

.keywords: PC, levels, reordering, factorization, incomplete, ILU
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetUseDropTolerance(PC pc,PetscReal dt,PetscReal dtcol,PetscInt maxrowcount)
{
  PetscErrorCode ierr,(*f)(PC,PetscReal,PetscReal,PetscInt);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCFactorSetUseDropTolerance_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,dt,dtcol,maxrowcount);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}  

#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetLevels"
/*@
   PCFactorSetLevels - Sets the number of levels of fill to use.

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  levels - number of levels of fill

   Options Database Key:
.  -pc_factor_levels <levels> - Sets fill level

   Level: intermediate

.keywords: PC, levels, fill, factorization, incomplete, ILU
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetLevels(PC pc,PetscInt levels)
{
  PetscErrorCode ierr,(*f)(PC,PetscInt);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  if (levels < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"negative levels");
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCFactorSetLevels_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,levels);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetAllowDiagonalFill"
/*@
   PCFactorSetAllowDiagonalFill - Causes all diagonal matrix entries to be 
   treated as level 0 fill even if there is no non-zero location.

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context

   Options Database Key:
.  -pc_factor_diagonal_fill

   Notes:
   Does not apply with 0 fill.

   Level: intermediate

.keywords: PC, levels, fill, factorization, incomplete, ILU
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetAllowDiagonalFill(PC pc)
{
  PetscErrorCode ierr,(*f)(PC);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCFactorSetAllowDiagonalFill_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------------------------*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetPivotInBlocks_ILU"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetPivotInBlocks_ILU(PC pc,PetscTruth pivot)
{
  PC_ILU *dir = (PC_ILU*)pc->data;

  PetscFunctionBegin;
  dir->info.pivotinblocks = pivot ? 1.0 : 0.0;
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
  ierr = MatOrderingRegisterAll(PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsHead("ILU Options");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-pc_factor_levels","levels of fill","PCFactorSetLevels",(PetscInt)ilu->info.levels,&itmp,&flg);CHKERRQ(ierr);
    if (flg) ilu->info.levels = itmp;
    ierr = PetscOptionsName("-pc_factor_in_place","do factorization in place","PCFactorSetUseInPlace",&ilu->inplace);CHKERRQ(ierr);
    ierr = PetscOptionsName("-pc_factor_diagonal_fill","Allow fill into empty diagonal entry","PCFactorSetAllowDiagonalFill",&flg);CHKERRQ(ierr);
    ilu->info.diagonal_fill = (double) flg;
    ierr = PetscOptionsName("-pc_factor_reuse_fill","Reuse fill ratio from previous factorization","PCFactorSetReuseFill",&ilu->reusefill);CHKERRQ(ierr);
    ierr = PetscOptionsName("-pc_factor_reuse_ordering","Reuse previous reordering","PCFactorSetReuseOrdering",&ilu->reuseordering);CHKERRQ(ierr);

    ierr = PetscOptionsName("-pc_factor_shift_nonzero","Shift added to diagonal","PCFactorSetShiftNonzero",&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PCFactorSetShiftNonzero(pc,(PetscReal)PETSC_DECIDE);CHKERRQ(ierr);
    }
    ierr = PetscOptionsReal("-pc_factor_shift_nonzero","Shift added to diagonal","PCFactorSetShiftNonzero",ilu->info.shiftnz,&ilu->info.shiftnz,0);CHKERRQ(ierr);
    
    ierr = PetscOptionsName("-pc_factor_shift_positive_definite","Manteuffel shift applied to diagonal","PCFactorSetShiftPd",&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PetscOptionsInt("-pc_factor_shift_positive_definite","Manteuffel shift applied to diagonal","PCFactorSetShiftPd",(PetscInt)ilu->info.shiftpd,&itmp,&flg); CHKERRQ(ierr);
      if (flg && !itmp) {
	ierr = PCFactorSetShiftPd(pc,PETSC_FALSE);CHKERRQ(ierr);
      } else {
	ierr = PCFactorSetShiftPd(pc,PETSC_TRUE);CHKERRQ(ierr); 
      }
    }
    ierr = PetscOptionsReal("-pc_factor_zeropivot","Pivot is considered zero if less than","PCFactorSetZeroPivot",ilu->info.zeropivot,&ilu->info.zeropivot,0);CHKERRQ(ierr);

    dt[0] = ilu->info.dt;
    dt[1] = ilu->info.dtcol;
    dt[2] = ilu->info.dtcount;
    ierr = PetscOptionsRealArray("-pc_factor_use_drop_tolerance","<dt,dtcol,maxrowcount>","PCFactorSetUseDropTolerance",dt,&dtmax,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PCFactorSetUseDropTolerance(pc,dt[0],dt[1],(PetscInt)dt[2]);CHKERRQ(ierr);
    }
    ierr = PetscOptionsReal("-pc_factor_fill","Expected fill in factorization","PCFactorSetFill",ilu->info.fill,&ilu->info.fill,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsName("-pc_factor_nonzeros_along_diagonal","Reorder to remove zeros from diagonal","PCFactorReorderForNonzeroDiagonal",&flg);CHKERRQ(ierr);
    if (flg) {
      tol = PETSC_DECIDE;
      ierr = PetscOptionsReal("-pc_factor_nonzeros_along_diagonal","Reorder to remove zeros from diagonal","PCFactorReorderForNonzeroDiagonal",ilu->nonzerosalongdiagonaltol,&tol,0);CHKERRQ(ierr);
      ierr = PCFactorReorderForNonzeroDiagonal(pc,tol);CHKERRQ(ierr);
    }

    ierr = MatGetOrderingList(&ordlist);CHKERRQ(ierr);
    ierr = PetscOptionsList("-pc_factor_mat_ordering_type","Reorder to reduce nonzeros in ILU","PCFactorSetMatOrdering",ordlist,ilu->ordering,tname,256,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PCFactorSetMatOrdering(pc,tname);CHKERRQ(ierr);
    }
    flg = ilu->info.pivotinblocks ? PETSC_TRUE : PETSC_FALSE;
    ierr = PetscOptionsTruth("-pc_factor_pivot_in_blocks","Pivot inside matrix blocks for BAIJ and SBAIJ","PCFactorSetPivotInBlocks",flg,&flg,&set);CHKERRQ(ierr);
    if (set) {
      ierr = PCFactorSetPivotInBlocks(pc,flg);CHKERRQ(ierr);
    }
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
        ierr = PetscViewerASCIIPrintf(viewer,"  ILU: drop tolerance %G\n",ilu->info.dt);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  ILU: max nonzeros per row %D\n",(PetscInt)ilu->info.dtcount);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  ILU: column permutation tolerance %G\n",ilu->info.dtcol);CHKERRQ(ierr);
    } else if (ilu->info.levels == 1) {
        ierr = PetscViewerASCIIPrintf(viewer,"  ILU: %D level of fill\n",(PetscInt)ilu->info.levels);CHKERRQ(ierr);
    } else {
        ierr = PetscViewerASCIIPrintf(viewer,"  ILU: %D levels of fill\n",(PetscInt)ilu->info.levels);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  ILU: max fill ratio allocated %G\n",ilu->info.fill);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  ILU: tolerance for zero pivot %G\n",ilu->info.zeropivot);CHKERRQ(ierr);
    if (ilu->info.shiftpd) {ierr = PetscViewerASCIIPrintf(viewer,"  ILU: using Manteuffel shift\n");CHKERRQ(ierr);}
    if (ilu->inplace) {ierr = PetscViewerASCIIPrintf(viewer,"       in-place factorization\n");CHKERRQ(ierr);}
    else              {ierr = PetscViewerASCIIPrintf(viewer,"       out-of-place factorization\n");CHKERRQ(ierr);}
    ierr = PetscViewerASCIIPrintf(viewer,"       matrix ordering: %s\n",ilu->ordering);CHKERRQ(ierr);
    if (ilu->reusefill)     {ierr = PetscViewerASCIIPrintf(viewer,"       Reusing fill from past factorization\n");CHKERRQ(ierr);}
    if (ilu->reuseordering) {ierr = PetscViewerASCIIPrintf(viewer,"       Reusing reordering from past factorization\n");CHKERRQ(ierr);}
    if (ilu->fact) {
      ierr = PetscViewerASCIIPrintf(viewer,"       Factored matrix follows\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr);
      ierr = MatView(ilu->fact,viewer);CHKERRQ(ierr);
      ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
  } else if (isstring) {
    ierr = PetscViewerStringSPrintf(viewer," lvls=%D,order=%s",(PetscInt)ilu->info.levels,ilu->ordering);CHKERRQ(ierr);CHKERRQ(ierr);
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

  PetscFunctionBegin;
  if (ilu->inplace) {
    if (!pc->setupcalled) {

      /* In-place factorization only makes sense with the natural ordering,
         so we only need to get the ordering once, even if nonzero structure changes */
      ierr = MatGetOrdering(pc->pmat,ilu->ordering,&ilu->row,&ilu->col);CHKERRQ(ierr);
      if (ilu->row) {ierr = PetscLogObjectParent(pc,ilu->row);CHKERRQ(ierr);}
      if (ilu->col) {ierr = PetscLogObjectParent(pc,ilu->col);CHKERRQ(ierr);}
    }

    /* In place ILU only makes sense with fill factor of 1.0 because 
       cannot have levels of fill */
    ilu->info.fill          = 1.0;
    ilu->info.diagonal_fill = 0;
    ierr = MatILUFactor(pc->pmat,ilu->row,ilu->col,&ilu->info);CHKERRQ(ierr);
    ilu->fact = pc->pmat;
  } else if (ilu->usedt) {
    if (!pc->setupcalled) {
      ierr = MatGetOrdering(pc->pmat,ilu->ordering,&ilu->row,&ilu->col);CHKERRQ(ierr);
      if (ilu->row) {ierr = PetscLogObjectParent(pc,ilu->row);CHKERRQ(ierr);}
      if (ilu->col) {ierr = PetscLogObjectParent(pc,ilu->col);CHKERRQ(ierr);}
      ierr = MatILUDTFactor(pc->pmat,ilu->row,ilu->col,&ilu->info,&ilu->fact);CHKERRQ(ierr);
      ierr = PetscLogObjectParent(pc,ilu->fact);CHKERRQ(ierr);
    } else if (pc->flag != SAME_NONZERO_PATTERN) { 
      ierr = MatDestroy(ilu->fact);CHKERRQ(ierr);
      if (!ilu->reuseordering) {
        if (ilu->row) {ierr = ISDestroy(ilu->row);CHKERRQ(ierr);}
        if (ilu->col) {ierr = ISDestroy(ilu->col);CHKERRQ(ierr);}
        ierr = MatGetOrdering(pc->pmat,ilu->ordering,&ilu->row,&ilu->col);CHKERRQ(ierr);
        if (ilu->row) {ierr = PetscLogObjectParent(pc,ilu->row);CHKERRQ(ierr);}
        if (ilu->col) {ierr = PetscLogObjectParent(pc,ilu->col);CHKERRQ(ierr);}
      }
      ierr = MatILUDTFactor(pc->pmat,ilu->row,ilu->col,&ilu->info,&ilu->fact);CHKERRQ(ierr);
      ierr = PetscLogObjectParent(pc,ilu->fact);CHKERRQ(ierr);
    } else if (!ilu->reusefill) { 
      ierr = MatDestroy(ilu->fact);CHKERRQ(ierr);
      ierr = MatILUDTFactor(pc->pmat,ilu->row,ilu->col,&ilu->info,&ilu->fact);CHKERRQ(ierr);
      ierr = PetscLogObjectParent(pc,ilu->fact);CHKERRQ(ierr);
    } else {
      ierr = MatLUFactorNumeric(pc->pmat,&ilu->info,&ilu->fact);CHKERRQ(ierr);
    }
   } else {
    if (!pc->setupcalled) {
      /* first time in so compute reordering and symbolic factorization */
      ierr = MatGetOrdering(pc->pmat,ilu->ordering,&ilu->row,&ilu->col);CHKERRQ(ierr);
      if (ilu->row) {ierr = PetscLogObjectParent(pc,ilu->row);CHKERRQ(ierr);}
      if (ilu->col) {ierr = PetscLogObjectParent(pc,ilu->col);CHKERRQ(ierr);}
      /*  Remove zeros along diagonal?     */
      if (ilu->nonzerosalongdiagonal) {
        ierr = MatReorderForNonzeroDiagonal(pc->pmat,ilu->nonzerosalongdiagonaltol,ilu->row,ilu->col);CHKERRQ(ierr);
      }
      ierr = MatILUFactorSymbolic(pc->pmat,ilu->row,ilu->col,&ilu->info,&ilu->fact);CHKERRQ(ierr);
      ierr = PetscLogObjectParent(pc,ilu->fact);CHKERRQ(ierr);
    } else if (pc->flag != SAME_NONZERO_PATTERN) { 
      if (!ilu->reuseordering) {
        /* compute a new ordering for the ILU */
        ierr = ISDestroy(ilu->row);CHKERRQ(ierr);
        ierr = ISDestroy(ilu->col);CHKERRQ(ierr);
        ierr = MatGetOrdering(pc->pmat,ilu->ordering,&ilu->row,&ilu->col);CHKERRQ(ierr);
        if (ilu->row) {ierr = PetscLogObjectParent(pc,ilu->row);CHKERRQ(ierr);}
        if (ilu->col) {ierr = PetscLogObjectParent(pc,ilu->col);CHKERRQ(ierr);}
        /*  Remove zeros along diagonal?     */
        if (ilu->nonzerosalongdiagonal) {
          ierr = MatReorderForNonzeroDiagonal(pc->pmat,ilu->nonzerosalongdiagonaltol,ilu->row,ilu->col);CHKERRQ(ierr);
        }
      }
      ierr = MatDestroy(ilu->fact);CHKERRQ(ierr);
      ierr = MatILUFactorSymbolic(pc->pmat,ilu->row,ilu->col,&ilu->info,&ilu->fact);CHKERRQ(ierr);
      ierr = PetscLogObjectParent(pc,ilu->fact);CHKERRQ(ierr);
    }
    ierr = MatLUFactorNumeric(pc->pmat,&ilu->info,&ilu->fact);CHKERRQ(ierr);
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
  ierr = PetscStrfree(ilu->ordering);CHKERRQ(ierr);
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
  ierr = MatSolve(ilu->fact,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCApplyTranspose_ILU"
static PetscErrorCode PCApplyTranspose_ILU(PC pc,Vec x,Vec y)
{
  PC_ILU         *ilu = (PC_ILU*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSolveTranspose(ilu->fact,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCGetFactoredMatrix_ILU"
static PetscErrorCode PCGetFactoredMatrix_ILU(PC pc,Mat *mat)
{
  PC_ILU *ilu = (PC_ILU*)pc->data;

  PetscFunctionBegin;
  if (!ilu->fact) SETERRQ(PETSC_ERR_ORDER,"Matrix not yet factored; call after KSPSetUp() or PCSetUp()");
  *mat = ilu->fact;
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
.  -pc_factor_shift_nonzero <shift> - Sets shift amount or PETSC_DECIDE for the default
-  -pc_factor_shift_positive_definite [PETSC_TRUE/PETSC_FALSE] - Activate/Deactivate PCFactorSetShiftPd(); the value
   is optional with PETSC_TRUE being the default

   Level: beginner

  Concepts: incomplete factorization

   Notes: Only implemented for some matrix formats. Not implemented in parallel

          For BAIJ matrices this implements a point block ILU

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC, PCSOR, MatOrderingType,
           PCFactorSetZeroPivot(), PCFactorSetShiftNonzero(), PCFactorSetShiftPd(), PCFactorSetUseDropTolerance(),
           PCFactorSetFill(), PCFactorSetMatOrdering(), PCFactorSetReuseOrdering(),
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
  ierr = PetscNew(PC_ILU,&ilu);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(pc,sizeof(PC_ILU));CHKERRQ(ierr);

  ilu->fact                    = 0;
  ierr = MatFactorInfoInitialize(&ilu->info);CHKERRQ(ierr);
  ilu->info.levels             = 0;
  ilu->info.fill               = 1.0; 
  ilu->col                     = 0;
  ilu->row                     = 0;
  ilu->inplace                 = PETSC_FALSE;
  ierr = PetscStrallocpy(MATORDERING_NATURAL,&ilu->ordering);CHKERRQ(ierr);
  ilu->reuseordering           = PETSC_FALSE;
  ilu->usedt                   = PETSC_FALSE;
  ilu->info.dt                 = PETSC_DEFAULT;
  ilu->info.dtcount            = PETSC_DEFAULT;
  ilu->info.dtcol              = PETSC_DEFAULT;
  ilu->info.shiftnz            = 0.0;
  ilu->info.shiftpd            = 0.0; /* false */
  ilu->info.shift_fraction     = 0.0;
  ilu->info.zeropivot          = 1.e-12;
  ilu->info.pivotinblocks      = 1.0;
  ilu->reusefill               = PETSC_FALSE;
  ilu->info.diagonal_fill      = 0;
  pc->data                     = (void*)ilu;

  pc->ops->destroy             = PCDestroy_ILU;
  pc->ops->apply               = PCApply_ILU;
  pc->ops->applytranspose      = PCApplyTranspose_ILU;
  pc->ops->setup               = PCSetUp_ILU;
  pc->ops->setfromoptions      = PCSetFromOptions_ILU;
  pc->ops->getfactoredmatrix   = PCGetFactoredMatrix_ILU;
  pc->ops->view                = PCView_ILU;
  pc->ops->applyrichardson     = 0;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetZeroPivot_C","PCFactorSetZeroPivot_ILU",
                    PCFactorSetZeroPivot_ILU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetShiftNonzero_C","PCFactorSetShiftNonzero_ILU",
                    PCFactorSetShiftNonzero_ILU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetShiftPd_C","PCFactorSetShiftPd_ILU",
                    PCFactorSetShiftPd_ILU);CHKERRQ(ierr);

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetUseDropTolerance_C","PCFactorSetUseDropTolerance_ILU",
                    PCFactorSetUseDropTolerance_ILU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetFill_C","PCFactorSetFill_ILU",
                    PCFactorSetFill_ILU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetMatOrdering_C","PCFactorSetMatOrdering_ILU",
                    PCFactorSetMatOrdering_ILU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetReuseOrdering_C","PCFactorSetReuseOrdering_ILU",
                    PCFactorSetReuseOrdering_ILU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetReuseFill_C","PCFactorSetReuseFill_ILU",
                    PCFactorSetReuseFill_ILU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetLevels_C","PCFactorSetLevels_ILU",
                    PCFactorSetLevels_ILU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetUseInPlace_C","PCFactorSetUseInPlace_ILU",
                    PCFactorSetUseInPlace_ILU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetAllowDiagonalFill_C","PCFactorSetAllowDiagonalFill_ILU",
                    PCFactorSetAllowDiagonalFill_ILU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetPivotInBlocks_C","PCFactorSetPivotInBlocks_ILU",
                    PCFactorSetPivotInBlocks_ILU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorReorderForNonzeroDiagonal_C","PCFactorReorderForNonzeroDiagonal_ILU",
                    PCFactorReorderForNonzeroDiagonal_ILU);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
