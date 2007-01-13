#define PETSCKSP_DLL

/*
   Defines a direct factorization preconditioner for any Mat implementation
   Note: this need not be consided a preconditioner since it supplies
         a direct solver.
*/

#include "private/pcimpl.h"                /*I "petscpc.h" I*/
#include "src/ksp/pc/impls/factor/lu/lu.h"

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetZeroPivot_LU"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetZeroPivot_LU(PC pc,PetscReal z)
{
  PC_LU *lu;

  PetscFunctionBegin;
  lu                 = (PC_LU*)pc->data;
  lu->info.zeropivot = z;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetShiftNonzero_LU"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetShiftNonzero_LU(PC pc,PetscReal shift)
{
  PC_LU *dir;

  PetscFunctionBegin;
  dir = (PC_LU*)pc->data;
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
#define __FUNCT__ "PCFactorSetShiftPd_LU"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetShiftPd_LU(PC pc,PetscTruth shift)
{
  PC_LU *dir;
 
  PetscFunctionBegin;
  dir = (PC_LU*)pc->data;
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
#define __FUNCT__ "PCFactorReorderForNonzeroDiagonal_LU"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorReorderForNonzeroDiagonal_LU(PC pc,PetscReal z)
{
  PC_LU *lu = (PC_LU*)pc->data;

  PetscFunctionBegin;
  lu->nonzerosalongdiagonal = PETSC_TRUE;                 
  if (z == PETSC_DECIDE) {
    lu->nonzerosalongdiagonaltol = 1.e-10;
  } else {
    lu->nonzerosalongdiagonaltol = z;
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetReuseOrdering_LU"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetReuseOrdering_LU(PC pc,PetscTruth flag)
{
  PC_LU *lu;

  PetscFunctionBegin;
  lu                = (PC_LU*)pc->data;
  lu->reuseordering = flag;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetReuseFill_LU"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetReuseFill_LU(PC pc,PetscTruth flag)
{
  PC_LU *lu;

  PetscFunctionBegin;
  lu = (PC_LU*)pc->data;
  lu->reusefill = flag;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "PCSetFromOptions_LU"
static PetscErrorCode PCSetFromOptions_LU(PC pc)
{
  PC_LU          *lu = (PC_LU*)pc->data;
  PetscErrorCode ierr;
  PetscTruth     flg,set;
  char           tname[256];
  PetscFList     ordlist;
  PetscReal      tol;

  PetscFunctionBegin;
  ierr = MatOrderingRegisterAll(PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsHead("LU options");CHKERRQ(ierr);
    ierr = PetscOptionsName("-pc_factor_in_place","Form LU in the same memory as the matrix","PCFactorSetUseInPlace",&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PCFactorSetUseInPlace(pc);CHKERRQ(ierr);
    }
    ierr = PetscOptionsReal("-pc_factor_fill","Expected non-zeros in LU/non-zeros in matrix","PCFactorSetFill",lu->info.fill,&lu->info.fill,0);CHKERRQ(ierr);

    ierr = PetscOptionsName("-pc_factor_shift_nonzero","Shift added to diagonal","PCFactorSetShiftNonzero",&flg);CHKERRQ(ierr);
    if (flg) {
        ierr = PCFactorSetShiftNonzero(pc,(PetscReal) PETSC_DECIDE);CHKERRQ(ierr);
    }
    ierr = PetscOptionsReal("-pc_factor_shift_nonzero","Shift added to diagonal","PCFactorSetShiftNonzero",lu->info.shiftnz,&lu->info.shiftnz,0);CHKERRQ(ierr);
    ierr = PetscOptionsName("-pc_factor_shift_positive_definite","Manteuffel shift applied to diagonal","PCFactorSetShiftPd",&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PCFactorSetShiftPd(pc,PETSC_TRUE);CHKERRQ(ierr);
    }
    ierr = PetscOptionsReal("-pc_factor_zeropivot","Pivot is considered zero if less than","PCFactorSetZeroPivot",lu->info.zeropivot,&lu->info.zeropivot,0);CHKERRQ(ierr);

    ierr = PetscOptionsName("-pc_factor_reuse_fill","Use fill from previous factorization","PCFactorSetReuseFill",&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PCFactorSetReuseFill(pc,PETSC_TRUE);CHKERRQ(ierr);
    }
    ierr = PetscOptionsName("-pc_factor_reuse_ordering","Reuse ordering from previous factorization","PCFactorSetReuseOrdering",&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PCFactorSetReuseOrdering(pc,PETSC_TRUE);CHKERRQ(ierr);
    }

    ierr = MatGetOrderingList(&ordlist);CHKERRQ(ierr);
    ierr = PetscOptionsList("-pc_factor_mat_ordering_type","Reordering to reduce nonzeros in LU","PCFactorSetMatOrderingType",ordlist,lu->ordering,tname,256,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PCFactorSetMatOrderingType(pc,tname);CHKERRQ(ierr);
    }

    ierr = PetscOptionsName("-pc_factor_nonzeros_along_diagonal","Reorder to remove zeros from diagonal","PCFactorReorderForNonzeroDiagonal",&flg);CHKERRQ(ierr);
    if (flg) {
      tol = PETSC_DECIDE;
      ierr = PetscOptionsReal("-pc_factor_nonzeros_along_diagonal","Reorder to remove zeros from diagonal","PCFactorReorderForNonzeroDiagonal",lu->nonzerosalongdiagonaltol,&tol,0);CHKERRQ(ierr);
      ierr = PCFactorReorderForNonzeroDiagonal(pc,tol);CHKERRQ(ierr);
    }

    ierr = PetscOptionsReal("-pc_factor_pivoting","Pivoting tolerance (used only for some factorization)","PCFactorSetPivoting",lu->info.dtcol,&lu->info.dtcol,&flg);CHKERRQ(ierr);

    flg = lu->info.pivotinblocks ? PETSC_TRUE : PETSC_FALSE;
    ierr = PetscOptionsTruth("-pc_factor_pivot_in_blocks","Pivot inside matrix blocks for BAIJ and SBAIJ","PCFactorSetPivotInBlocks",flg,&flg,&set);CHKERRQ(ierr);
    if (set) {
      ierr = PCFactorSetPivotInBlocks(pc,flg);CHKERRQ(ierr);
    }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCView_LU"
static PetscErrorCode PCView_LU(PC pc,PetscViewer viewer)
{
  PC_LU          *lu = (PC_LU*)pc->data;
  PetscErrorCode ierr;
  PetscTruth     iascii,isstring;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_STRING,&isstring);CHKERRQ(ierr);
  if (iascii) {

    if (lu->inplace) {ierr = PetscViewerASCIIPrintf(viewer,"  LU: in-place factorization\n");CHKERRQ(ierr);}
    else             {ierr = PetscViewerASCIIPrintf(viewer,"  LU: out-of-place factorization\n");CHKERRQ(ierr);}
    ierr = PetscViewerASCIIPrintf(viewer,"    matrix ordering: %s\n",lu->ordering);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  LU: tolerance for zero pivot %G\n",lu->info.zeropivot);CHKERRQ(ierr);
    if (lu->info.shiftpd) {ierr = PetscViewerASCIIPrintf(viewer,"  LU: using Manteuffel shift\n");CHKERRQ(ierr);}
    if (lu->fact) {
      ierr = PetscViewerASCIIPrintf(viewer,"  LU: factor fill ratio needed %G\n",lu->actualfill);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"       Factored matrix follows\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr);
      ierr = MatView(lu->fact,viewer);CHKERRQ(ierr);
      ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
    if (lu->reusefill)    {ierr = PetscViewerASCIIPrintf(viewer,"       Reusing fill from past factorization\n");CHKERRQ(ierr);}
    if (lu->reuseordering) {ierr = PetscViewerASCIIPrintf(viewer,"       Reusing reordering from past factorization\n");CHKERRQ(ierr);}
  } else if (isstring) {
    ierr = PetscViewerStringSPrintf(viewer," order=%s",lu->ordering);CHKERRQ(ierr);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported for PCLU",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCGetFactoredMatrix_LU"
static PetscErrorCode PCGetFactoredMatrix_LU(PC pc,Mat *mat)
{
  PC_LU *dir = (PC_LU*)pc->data;

  PetscFunctionBegin;
  if (!dir->fact) SETERRQ(PETSC_ERR_ORDER,"Matrix not yet factored; call after KSPSetUp() or PCSetUp()");
  *mat = dir->fact;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSetUp_LU"
static PetscErrorCode PCSetUp_LU(PC pc)
{
  PetscErrorCode ierr;
  PC_LU          *dir = (PC_LU*)pc->data;

  PetscFunctionBegin;
  if (dir->reusefill && pc->setupcalled) dir->info.fill = dir->actualfill;

  if (dir->inplace) {
    if (dir->row && dir->col && dir->row != dir->col) {ierr = ISDestroy(dir->row);CHKERRQ(ierr);}
    if (dir->col) {ierr = ISDestroy(dir->col);CHKERRQ(ierr);}
    ierr = MatGetOrdering(pc->pmat,dir->ordering,&dir->row,&dir->col);CHKERRQ(ierr);
    if (dir->row) {
      ierr = PetscLogObjectParent(pc,dir->row);CHKERRQ(ierr); 
      ierr = PetscLogObjectParent(pc,dir->col);CHKERRQ(ierr);
    }
    ierr = MatLUFactor(pc->pmat,dir->row,dir->col,&dir->info);CHKERRQ(ierr);
    dir->fact = pc->pmat;
  } else {
    MatInfo info;
    if (!pc->setupcalled) {
      ierr = MatGetOrdering(pc->pmat,dir->ordering,&dir->row,&dir->col);CHKERRQ(ierr);
      if (dir->nonzerosalongdiagonal) {
        ierr = MatReorderForNonzeroDiagonal(pc->pmat,dir->nonzerosalongdiagonaltol,dir->row,dir->col);CHKERRQ(ierr);
      }
      if (dir->row) {
        ierr = PetscLogObjectParent(pc,dir->row);CHKERRQ(ierr); 
        ierr = PetscLogObjectParent(pc,dir->col);CHKERRQ(ierr);
      }
      ierr = MatLUFactorSymbolic(pc->pmat,dir->row,dir->col,&dir->info,&dir->fact);CHKERRQ(ierr);
      ierr = MatGetInfo(dir->fact,MAT_LOCAL,&info);CHKERRQ(ierr);
      dir->actualfill = info.fill_ratio_needed;
      ierr = PetscLogObjectParent(pc,dir->fact);CHKERRQ(ierr);
    } else if (pc->flag != SAME_NONZERO_PATTERN) {
      if (!dir->reuseordering) {
        if (dir->row && dir->col && dir->row != dir->col) {ierr = ISDestroy(dir->row);CHKERRQ(ierr);}
        if (dir->col) {ierr = ISDestroy(dir->col);CHKERRQ(ierr);}
        ierr = MatGetOrdering(pc->pmat,dir->ordering,&dir->row,&dir->col);CHKERRQ(ierr);
        if (dir->nonzerosalongdiagonal) {
          ierr = MatReorderForNonzeroDiagonal(pc->pmat,dir->nonzerosalongdiagonaltol,dir->row,dir->col);CHKERRQ(ierr);
        }
        if (dir->row) {
          ierr = PetscLogObjectParent(pc,dir->row);CHKERRQ(ierr);
          ierr = PetscLogObjectParent(pc,dir->col);CHKERRQ(ierr);
        }
      }
      ierr = MatDestroy(dir->fact);CHKERRQ(ierr);
      ierr = MatLUFactorSymbolic(pc->pmat,dir->row,dir->col,&dir->info,&dir->fact);CHKERRQ(ierr);
      ierr = MatGetInfo(dir->fact,MAT_LOCAL,&info);CHKERRQ(ierr);
      dir->actualfill = info.fill_ratio_needed;
      ierr = PetscLogObjectParent(pc,dir->fact);CHKERRQ(ierr);
    }
    ierr = MatLUFactorNumeric(pc->pmat,&dir->info,&dir->fact);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCDestroy_LU"
static PetscErrorCode PCDestroy_LU(PC pc)
{
  PC_LU          *dir = (PC_LU*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!dir->inplace && dir->fact) {ierr = MatDestroy(dir->fact);CHKERRQ(ierr);}
  if (dir->row && dir->col && dir->row != dir->col) {ierr = ISDestroy(dir->row);CHKERRQ(ierr);}
  if (dir->col) {ierr = ISDestroy(dir->col);CHKERRQ(ierr);}
  ierr = PetscStrfree(dir->ordering);CHKERRQ(ierr);
  ierr = PetscFree(dir);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCApply_LU"
static PetscErrorCode PCApply_LU(PC pc,Vec x,Vec y)
{
  PC_LU          *dir = (PC_LU*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (dir->inplace) {ierr = MatSolve(pc->pmat,x,y);CHKERRQ(ierr);}
  else              {ierr = MatSolve(dir->fact,x,y);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCApplyTranspose_LU"
static PetscErrorCode PCApplyTranspose_LU(PC pc,Vec x,Vec y)
{
  PC_LU          *dir = (PC_LU*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (dir->inplace) {ierr = MatSolveTranspose(pc->pmat,x,y);CHKERRQ(ierr);}
  else              {ierr = MatSolveTranspose(dir->fact,x,y);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/* -----------------------------------------------------------------------------------*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetFill_LU"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetFill_LU(PC pc,PetscReal fill)
{
  PC_LU *dir;

  PetscFunctionBegin;
  dir = (PC_LU*)pc->data;
  dir->info.fill = fill;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetUseInPlace_LU"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetUseInPlace_LU(PC pc)
{
  PC_LU *dir;

  PetscFunctionBegin;
  dir = (PC_LU*)pc->data;
  dir->inplace = PETSC_TRUE;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetMatOrderingType_LU"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetMatOrderingType_LU(PC pc,MatOrderingType ordering)
{
  PC_LU          *dir = (PC_LU*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscStrfree(dir->ordering);CHKERRQ(ierr);
  ierr = PetscStrallocpy(ordering,&dir->ordering);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetPivoting_LU"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetPivoting_LU(PC pc,PetscReal dtcol)
{
  PC_LU *dir = (PC_LU*)pc->data;

  PetscFunctionBegin;
  if (dtcol < 0.0 || dtcol > 1.0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Column pivot tolerance is %G must be between 0 and 1",dtcol);
  dir->info.dtcol = dtcol;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetPivotInBlocks_LU"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetPivotInBlocks_LU(PC pc,PetscTruth pivot)
{
  PC_LU *dir = (PC_LU*)pc->data;

  PetscFunctionBegin;
  dir->info.pivotinblocks = pivot ? 1.0 : 0.0;
  PetscFunctionReturn(0);
}
EXTERN_C_END

/* -----------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "PCFactorReorderForNonzeroDiagonal"
/*@
   PCFactorReorderForNonzeroDiagonal - reorders rows/columns of matrix to remove zeros from diagonal

   Collective on PC
   
   Input Parameters:
+  pc - the preconditioner context
-  tol - diagonal entries smaller than this in absolute value are considered zero

   Options Database Key:
.  -pc_factor_nonzeros_along_diagonal

   Level: intermediate

.keywords: PC, set, factorization, direct, fill

.seealso: PCFactorSetFill(), PCFactorSetShiftNonzero(), PCFactorSetZeroPivot(), MatReorderForNonzeroDiagonal()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorReorderForNonzeroDiagonal(PC pc,PetscReal rtol)
{
  PetscErrorCode ierr,(*f)(PC,PetscReal);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCFactorReorderForNonzeroDiagonal_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,rtol);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetFill"
/*@
   PCFactorSetFill - Indicate the amount of fill you expect in the factored matrix,
   fill = number nonzeros in factor/number nonzeros in original matrix.

   Collective on PC
   
   Input Parameters:
+  pc - the preconditioner context
-  fill - amount of expected fill

   Options Database Key:
.  -pc_factor_fill <fill> - Sets fill amount

   Level: intermediate

   Note:
   For sparse matrix factorizations it is difficult to predict how much 
   fill to expect. By running with the option -info PETSc will print the 
   actual amount of fill used; allowing you to set the value accurately for
   future runs. Default PETSc uses a value of 5.0

.keywords: PC, set, factorization, direct, fill

@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetFill(PC pc,PetscReal fill)
{
  PetscErrorCode ierr,(*f)(PC,PetscReal);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  if (fill < 1.0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Fill factor cannot be less then 1.0");
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCFactorSetFill_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,fill);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetUseInPlace"
/*@
   PCFactorSetUseInPlace - Tells the system to do an in-place factorization.
   For dense matrices, this enables the solution of much larger problems. 
   For sparse matrices the factorization cannot be done truly in-place 
   so this does not save memory during the factorization, but after the matrix
   is factored, the original unfactored matrix is freed, thus recovering that
   space.

   Collective on PC

   Input Parameters:
.  pc - the preconditioner context

   Options Database Key:
.  -pc_factor_in_place - Activates in-place factorization

   Notes:
   PCFactorSetUseInplace() can only be used with the KSP method KSPPREONLY or when 
   a different matrix is provided for the multiply and the preconditioner in 
   a call to KSPSetOperators().
   This is because the Krylov space methods require an application of the 
   matrix multiplication, which is not possible here because the matrix has 
   been factored in-place, replacing the original matrix.

   Level: intermediate

.keywords: PC, set, factorization, direct, inplace, in-place, LU

.seealso: PCILUSetUseInPlace()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetUseInPlace(PC pc)
{
  PetscErrorCode ierr,(*f)(PC);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCFactorSetUseInPlace_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetMatOrderingType"
/*@C
    PCFactorSetMatOrderingType - Sets the ordering routine (to reduce fill) to 
    be used in the LU factorization.

    Collective on PC

    Input Parameters:
+   pc - the preconditioner context
-   ordering - the matrix ordering name, for example, MATORDERING_ND or MATORDERING_RCM

    Options Database Key:
.   -pc_factor_mat_ordering_type <nd,rcm,...> - Sets ordering routine

    Level: intermediate

    Notes: nested dissection is used by default

@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetMatOrderingType(PC pc,MatOrderingType ordering)
{
  PetscErrorCode ierr,(*f)(PC,MatOrderingType);

  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCFactorSetMatOrderingType_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,ordering);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetPivoting"
/*@
    PCFactorSetPivoting - Determines when pivoting is done during LU. 
      For PETSc dense matrices column pivoting is always done, for PETSc sparse matrices
      it is never done. For the Matlab and SuperLU factorization this is used.

    Collective on PC

    Input Parameters:
+   pc - the preconditioner context
-   dtcol - 0.0 implies no pivoting, 1.0 complete pivoting (slower, requires more memory but more stable)

    Options Database Key:
.   -pc_factor_pivoting <dtcol>

    Level: intermediate

.seealso: PCILUSetMatOrdering(), PCFactorSetPivotInBlocks()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetPivoting(PC pc,PetscReal dtcol)
{
  PetscErrorCode ierr,(*f)(PC,PetscReal);

  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCFactorSetPivoting_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,dtcol);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetPivotInBlocks"
/*@
    PCFactorSetPivotInBlocks - Determines if pivoting is done while factoring each block
      with BAIJ or SBAIJ matrices

    Collective on PC

    Input Parameters:
+   pc - the preconditioner context
-   pivot - PETSC_TRUE or PETSC_FALSE

    Options Database Key:
.   -pc_factor_pivot_in_blocks <true,false>

    Level: intermediate

.seealso: PCILUSetMatOrdering(), PCFactorSetPivoting()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetPivotInBlocks(PC pc,PetscTruth pivot)
{
  PetscErrorCode ierr,(*f)(PC,PetscTruth);

  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCFactorSetPivotInBlocks_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,pivot);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------ */

/*MC
   PCLU - Uses a direct solver, based on LU factorization, as a preconditioner

   Options Database Keys:
+  -pc_factor_reuse_ordering - Activate PCFactorSetReuseOrdering()
.  -pc_factor_reuse_fill - Activates PCFactorSetReuseFill()
.  -pc_factor_fill <fill> - Sets fill amount
.  -pc_factor_in_place - Activates in-place factorization
.  -pc_factor_mat_ordering_type <nd,rcm,...> - Sets ordering routine
.  -pc_factor_pivot_in_blocks <true,false> - allow pivoting within the small blocks during factorization (may increase
                                         stability of factorization.
.  -pc_factor_shift_nonzero <shift> - Sets shift amount or PETSC_DECIDE for the default
-  -pc_factor_shift_positive_definite [PETSC_TRUE/PETSC_FALSE] - Activate/Deactivate PCFactorSetShiftPd(); the value
   is optional with PETSC_TRUE being the default

   Notes: Not all options work for all matrix formats
          Run with -help to see additional options for particular matrix formats or factorization
          algorithms

   Level: beginner

   Concepts: LU factorization, direct solver

   Notes: Usually this will compute an "exact" solution in one iteration and does 
          not need a Krylov method (i.e. you can use -ksp_type preonly, or 
          KSPSetType(ksp,KSPPREONLY) for the Krylov method

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC,
           PCILU, PCCHOLESKY, PCICC, PCFactorSetReuseOrdering(), PCFactorSetReuseFill(), PCGetFactoredMatrix(),
           PCFactorSetFill(), PCFactorSetUseInPlace(), PCFactorSetMatOrderingType(), PCFactorSetPivoting(),
           PCFactorSetPivotingInBlocks(),PCFactorSetShiftNonzero(),PCFactorSetShiftPd()
M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCreate_LU"
PetscErrorCode PETSCKSP_DLLEXPORT PCCreate_LU(PC pc)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PC_LU          *dir;

  PetscFunctionBegin;
  ierr = PetscNew(PC_LU,&dir);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(pc,sizeof(PC_LU));CHKERRQ(ierr);

  ierr = MatFactorInfoInitialize(&dir->info);CHKERRQ(ierr);
  dir->fact                  = 0;
  dir->inplace               = PETSC_FALSE;
  dir->nonzerosalongdiagonal = PETSC_FALSE;

  dir->info.fill           = 5.0;
  dir->info.dtcol          = 1.e-6; /* default to pivoting; this is only thing PETSc LU supports */
  dir->info.shiftnz        = 0.0;
  dir->info.zeropivot      = 1.e-12;
  dir->info.pivotinblocks  = 1.0;
  dir->info.shiftpd        = 0.0; /* false */
  dir->info.shift_fraction = 0.0;
  dir->col                 = 0;
  dir->row                 = 0;
  ierr = MPI_Comm_size(pc->comm,&size);CHKERRQ(ierr);
  if (size == 1) {
    ierr = PetscStrallocpy(MATORDERING_ND,&dir->ordering);CHKERRQ(ierr);
  } else {
    ierr = PetscStrallocpy(MATORDERING_NATURAL,&dir->ordering);CHKERRQ(ierr);
  }
  dir->reusefill        = PETSC_FALSE;
  dir->reuseordering    = PETSC_FALSE;
  pc->data              = (void*)dir;

  pc->ops->destroy           = PCDestroy_LU;
  pc->ops->apply             = PCApply_LU;
  pc->ops->applytranspose    = PCApplyTranspose_LU;
  pc->ops->setup             = PCSetUp_LU;
  pc->ops->setfromoptions    = PCSetFromOptions_LU;
  pc->ops->view              = PCView_LU;
  pc->ops->applyrichardson   = 0;
  pc->ops->getfactoredmatrix = PCGetFactoredMatrix_LU;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetZeroPivot_C","PCFactorSetZeroPivot_LU",
                    PCFactorSetZeroPivot_LU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetShiftNonzero_C","PCFactorSetShiftNonzero_LU",
                    PCFactorSetShiftNonzero_LU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetShiftPd_C","PCFactorSetShiftPd_LU",
                    PCFactorSetShiftPd_LU);CHKERRQ(ierr);

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetFill_C","PCFactorSetFill_LU",
                    PCFactorSetFill_LU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetUseInPlace_C","PCFactorSetUseInPlace_LU",
                    PCFactorSetUseInPlace_LU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetMatOrderingType_C","PCFactorSetMatOrderingType_LU",
                    PCFactorSetMatOrderingType_LU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetReuseOrdering_C","PCFactorSetReuseOrdering_LU",
                    PCFactorSetReuseOrdering_LU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetReuseFill_C","PCFactorSetReuseFill_LU",
                    PCFactorSetReuseFill_LU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetPivoting_C","PCFactorSetPivoting_LU",
                    PCFactorSetPivoting_LU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetPivotInBlocks_C","PCFactorSetPivotInBlocks_LU",
                    PCFactorSetPivotInBlocks_LU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorReorderForNonzeroDiagonal_C","PCFactorReorderForNonzeroDiagonal_LU",
                    PCFactorReorderForNonzeroDiagonal_LU);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
