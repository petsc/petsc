/*
   Defines a direct factorization preconditioner for any Mat implementation
   Note: this need not be consided a preconditioner since it supplies
         a direct solver.
*/

#include "src/ksp/pc/pcimpl.h"                /*I "petscpc.h" I*/
#include "src/ksp/pc/impls/factor/lu/lu.h"

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetZeroPivot_LU"
PetscErrorCode PCFactorSetZeroPivot_LU(PC pc,PetscReal z)
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
PetscErrorCode PCFactorSetShiftNonzero_LU(PC pc,PetscReal shift)
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
PetscErrorCode PCFactorSetShiftPd_LU(PC pc,PetscTruth shift)
{
  PC_LU *dir;
 
  PetscFunctionBegin;
  dir = (PC_LU*)pc->data;
  dir->info.shiftpd = shift;
  if (shift) dir->info.shift_fraction = 0.0;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCLUReorderForNonzeroDiagonal_LU"
PetscErrorCode PCLUReorderForNonzeroDiagonal_LU(PC pc,PetscReal z)
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
#define __FUNCT__ "PCLUSetReuseOrdering_LU"
PetscErrorCode PCLUSetReuseOrdering_LU(PC pc,PetscTruth flag)
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
#define __FUNCT__ "PCLUSetReuseFill_LU"
PetscErrorCode PCLUSetReuseFill_LU(PC pc,PetscTruth flag)
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
    ierr = PetscOptionsName("-pc_lu_in_place","Form LU in the same memory as the matrix","PCLUSetUseInPlace",&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PCLUSetUseInPlace(pc);CHKERRQ(ierr);
    }
    ierr = PetscOptionsReal("-pc_lu_fill","Expected non-zeros in LU/non-zeros in matrix","PCLUSetFill",lu->info.fill,&lu->info.fill,0);CHKERRQ(ierr);

    ierr = PetscOptionsName("-pc_factor_shiftnonzero","Shift added to diagonal","PCFactorSetShiftNonzero",&flg);CHKERRQ(ierr);
    if (flg) {
        ierr = PCFactorSetShiftNonzero(pc,(PetscReal) PETSC_DECIDE);CHKERRQ(ierr);
    }
    ierr = PetscOptionsReal("-pc_factor_shiftnonzero","Shift added to diagonal","PCFactorSetShiftNonzero",lu->info.shiftnz,&lu->info.shiftnz,0);CHKERRQ(ierr);
    ierr = PetscOptionsName("-pc_factor_shiftpd","Manteuffel shift applied to diagonal","PCFactorSetShiftPd",&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PCFactorSetShiftPd(pc,PETSC_TRUE);CHKERRQ(ierr);
    }
    ierr = PetscOptionsReal("-pc_factor_zeropivot","Pivot is considered zero if less than","PCFactorSetZeroPivot",lu->info.zeropivot,&lu->info.zeropivot,0);CHKERRQ(ierr);

    ierr = PetscOptionsName("-pc_lu_reuse_fill","Use fill from previous factorization","PCLUSetReuseFill",&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PCLUSetReuseFill(pc,PETSC_TRUE);CHKERRQ(ierr);
    }
    ierr = PetscOptionsName("-pc_lu_reuse_ordering","Reuse ordering from previous factorization","PCLUSetReuseOrdering",&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PCLUSetReuseOrdering(pc,PETSC_TRUE);CHKERRQ(ierr);
    }

    ierr = MatGetOrderingList(&ordlist);CHKERRQ(ierr);
    ierr = PetscOptionsList("-pc_lu_mat_ordering_type","Reordering to reduce nonzeros in LU","PCLUSetMatOrdering",ordlist,lu->ordering,tname,256,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PCLUSetMatOrdering(pc,tname);CHKERRQ(ierr);
    }

    ierr = PetscOptionsName("-pc_lu_nonzeros_along_diagonal","Reorder to remove zeros from diagonal","PCLUReorderForNonzeroDiagonal",&flg);CHKERRQ(ierr);
    if (flg) {
      tol = PETSC_DECIDE;
      ierr = PetscOptionsReal("-pc_lu_nonzeros_along_diagonal","Reorder to remove zeros from diagonal","PCLUReorderForNonzeroDiagonal",lu->nonzerosalongdiagonaltol,&tol,0);CHKERRQ(ierr);
      ierr = PCLUReorderForNonzeroDiagonal(pc,tol);CHKERRQ(ierr);
    }

    ierr = PetscOptionsReal("-pc_lu_pivoting","Pivoting tolerance (used only for some factorization)","PCLUSetPivoting",lu->info.dtcol,&lu->info.dtcol,&flg);CHKERRQ(ierr);

    flg = lu->info.pivotinblocks ? PETSC_TRUE : PETSC_FALSE;
    ierr = PetscOptionsLogical("-pc_lu_pivot_in_blocks","Pivot inside matrix blocks for BAIJ and SBAIJ","PCLUSetPivotInBlocks",flg,&flg,&set);CHKERRQ(ierr);
    if (set) {
      ierr = PCLUSetPivotInBlocks(pc,flg);CHKERRQ(ierr);
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
    MatInfo info;

    if (lu->inplace) {ierr = PetscViewerASCIIPrintf(viewer,"  LU: in-place factorization\n");CHKERRQ(ierr);}
    else             {ierr = PetscViewerASCIIPrintf(viewer,"  LU: out-of-place factorization\n");CHKERRQ(ierr);}
    ierr = PetscViewerASCIIPrintf(viewer,"    matrix ordering: %s\n",lu->ordering);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  LU: tolerance for zero pivot %g\n",lu->info.zeropivot);CHKERRQ(ierr);
    if (lu->info.shiftpd) {ierr = PetscViewerASCIIPrintf(viewer,"  LU: using Manteuffel shift\n");CHKERRQ(ierr);}
    if (lu->fact) {
      ierr = MatGetInfo(lu->fact,MAT_LOCAL,&info);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"    LU nonzeros %g\n",info.nz_used);CHKERRQ(ierr);
      ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_FACTOR_INFO);CHKERRQ(ierr);
      ierr = MatView(lu->fact,viewer);CHKERRQ(ierr);
      ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
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
    if (dir->row) {PetscLogObjectParent(pc,dir->row); PetscLogObjectParent(pc,dir->col);}
    ierr = MatLUFactor(pc->pmat,dir->row,dir->col,&dir->info);CHKERRQ(ierr);
    dir->fact = pc->pmat;
  } else {
    MatInfo info;
    if (!pc->setupcalled) {
      ierr = MatGetOrdering(pc->pmat,dir->ordering,&dir->row,&dir->col);CHKERRQ(ierr);
      if (dir->nonzerosalongdiagonal) {
        ierr = MatReorderForNonzeroDiagonal(pc->pmat,dir->nonzerosalongdiagonaltol,dir->row,dir->col);CHKERRQ(ierr);
      }
      if (dir->row) {PetscLogObjectParent(pc,dir->row); PetscLogObjectParent(pc,dir->col);}
      ierr = MatLUFactorSymbolic(pc->pmat,dir->row,dir->col,&dir->info,&dir->fact);CHKERRQ(ierr);
      ierr = MatGetInfo(dir->fact,MAT_LOCAL,&info);CHKERRQ(ierr);
      dir->actualfill = info.fill_ratio_needed;
      PetscLogObjectParent(pc,dir->fact);
    } else if (pc->flag != SAME_NONZERO_PATTERN) {
      if (!dir->reuseordering) {
        if (dir->row && dir->col && dir->row != dir->col) {ierr = ISDestroy(dir->row);CHKERRQ(ierr);}
        if (dir->col) {ierr = ISDestroy(dir->col);CHKERRQ(ierr);}
        ierr = MatGetOrdering(pc->pmat,dir->ordering,&dir->row,&dir->col);CHKERRQ(ierr);
        if (dir->nonzerosalongdiagonal) {
         ierr = MatReorderForNonzeroDiagonal(pc->pmat,dir->nonzerosalongdiagonaltol,dir->row,dir->col);CHKERRQ(ierr);
        }
        if (dir->row) {PetscLogObjectParent(pc,dir->row); PetscLogObjectParent(pc,dir->col);}
      }
      ierr = MatDestroy(dir->fact);CHKERRQ(ierr);
      ierr = MatLUFactorSymbolic(pc->pmat,dir->row,dir->col,&dir->info,&dir->fact);CHKERRQ(ierr);
      ierr = MatGetInfo(dir->fact,MAT_LOCAL,&info);CHKERRQ(ierr);
      dir->actualfill = info.fill_ratio_needed;
      PetscLogObjectParent(pc,dir->fact);
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
#define __FUNCT__ "PCLUSetFill_LU"
PetscErrorCode PCLUSetFill_LU(PC pc,PetscReal fill)
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
#define __FUNCT__ "PCLUSetUseInPlace_LU"
PetscErrorCode PCLUSetUseInPlace_LU(PC pc)
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
#define __FUNCT__ "PCLUSetMatOrdering_LU"
PetscErrorCode PCLUSetMatOrdering_LU(PC pc,MatOrderingType ordering)
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
#define __FUNCT__ "PCLUSetPivoting_LU"
PetscErrorCode PCLUSetPivoting_LU(PC pc,PetscReal dtcol)
{
  PC_LU *dir = (PC_LU*)pc->data;

  PetscFunctionBegin;
  if (dtcol < 0.0 || dtcol > 1.0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Column pivot tolerance is %g must be between 0 and 1",dtcol);
  dir->info.dtcol = dtcol;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCLUSetPivotInBlocks_LU"
PetscErrorCode PCLUSetPivotInBlocks_LU(PC pc,PetscTruth pivot)
{
  PC_LU *dir = (PC_LU*)pc->data;

  PetscFunctionBegin;
  dir->info.pivotinblocks = pivot ? 1.0 : 0.0;
  PetscFunctionReturn(0);
}
EXTERN_C_END

/* -----------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "PCLUReorderForNonzeroDiagonal"
/*@
   PCLUReorderForNonzeroDiagonal - reorders rows/columns of matrix to remove zeros from diagonal

   Collective on PC
   
   Input Parameters:
+  pc - the preconditioner context
-  tol - diagonal entries smaller than this in absolute value are considered zero

   Options Database Key:
.  -pc_lu_nonzeros_along_diagonal

   Level: intermediate

.keywords: PC, set, factorization, direct, fill

.seealso: PCLUSetFill(), PCFactorSetShiftNonzero(), PCFactorSetZeroPivot(), MatReorderForNonzeroDiagonal()
@*/
PetscErrorCode PCLUReorderForNonzeroDiagonal(PC pc,PetscReal rtol)
{
  PetscErrorCode ierr,(*f)(PC,PetscReal);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCLUReorderForNonzeroDiagonal_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,rtol);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCLUSetReuseOrdering"
/*@
   PCLUSetReuseOrdering - When similar matrices are factored, this
   causes the ordering computed in the first factor to be used for all
   following factors; applies to both fill and drop tolerance LUs.

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  flag - PETSC_TRUE to reuse else PETSC_FALSE

   Options Database Key:
.  -pc_lu_reuse_ordering - Activate PCLUSetReuseOrdering()

   Level: intermediate

.keywords: PC, levels, reordering, factorization, incomplete, LU

.seealso: PCLUSetReuseFill(), PCILUSetReuseOrdering(), PCILUDTSetReuseFill()
@*/
PetscErrorCode PCLUSetReuseOrdering(PC pc,PetscTruth flag)
{
  PetscErrorCode ierr,(*f)(PC,PetscTruth);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCLUSetReuseOrdering_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,flag);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCLUSetReuseFill"
/*@
   PCLUSetReuseFill - When matrices with same nonzero structure are LU factored,
   this causes later ones to use the fill computed in the initial factorization.

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  flag - PETSC_TRUE to reuse else PETSC_FALSE

   Options Database Key:
.  -pc_lu_reuse_fill - Activates PCLUSetReuseFill()

   Level: intermediate

.keywords: PC, levels, reordering, factorization, incomplete, LU

.seealso: PCILUSetReuseOrdering(), PCLUSetReuseOrdering(), PCILUDTSetReuseFill()
@*/
PetscErrorCode PCLUSetReuseFill(PC pc,PetscTruth flag)
{
  PetscErrorCode ierr,(*f)(PC,PetscTruth);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCLUSetReuseFill_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,flag);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCLUSetFill"
/*@
   PCLUSetFill - Indicate the amount of fill you expect in the factored matrix,
   fill = number nonzeros in factor/number nonzeros in original matrix.

   Collective on PC
   
   Input Parameters:
+  pc - the preconditioner context
-  fill - amount of expected fill

   Options Database Key:
.  -pc_lu_fill <fill> - Sets fill amount

   Level: intermediate

   Note:
   For sparse matrix factorizations it is difficult to predict how much 
   fill to expect. By running with the option -log_info PETSc will print the 
   actual amount of fill used; allowing you to set the value accurately for
   future runs. Default PETSc uses a value of 5.0

.keywords: PC, set, factorization, direct, fill

.seealso: PCILUSetFill()
@*/
PetscErrorCode PCLUSetFill(PC pc,PetscReal fill)
{
  PetscErrorCode ierr,(*f)(PC,PetscReal);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  if (fill < 1.0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Fill factor cannot be less then 1.0");
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCLUSetFill_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,fill);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCLUSetUseInPlace"
/*@
   PCLUSetUseInPlace - Tells the system to do an in-place factorization.
   For dense matrices, this enables the solution of much larger problems. 
   For sparse matrices the factorization cannot be done truly in-place 
   so this does not save memory during the factorization, but after the matrix
   is factored, the original unfactored matrix is freed, thus recovering that
   space.

   Collective on PC

   Input Parameters:
.  pc - the preconditioner context

   Options Database Key:
.  -pc_lu_in_place - Activates in-place factorization

   Notes:
   PCLUSetUseInplace() can only be used with the KSP method KSPPREONLY or when 
   a different matrix is provided for the multiply and the preconditioner in 
   a call to KSPSetOperators().
   This is because the Krylov space methods require an application of the 
   matrix multiplication, which is not possible here because the matrix has 
   been factored in-place, replacing the original matrix.

   Level: intermediate

.keywords: PC, set, factorization, direct, inplace, in-place, LU

.seealso: PCILUSetUseInPlace()
@*/
PetscErrorCode PCLUSetUseInPlace(PC pc)
{
  PetscErrorCode ierr,(*f)(PC);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCLUSetUseInPlace_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCLUSetMatOrdering"
/*@C
    PCLUSetMatOrdering - Sets the ordering routine (to reduce fill) to 
    be used in the LU factorization.

    Collective on PC

    Input Parameters:
+   pc - the preconditioner context
-   ordering - the matrix ordering name, for example, MATORDERING_ND or MATORDERING_RCM

    Options Database Key:
.   -pc_lu_mat_ordering_type <nd,rcm,...> - Sets ordering routine

    Level: intermediate

    Notes: nested dissection is used by default

.seealso: PCILUSetMatOrdering()
@*/
PetscErrorCode PCLUSetMatOrdering(PC pc,MatOrderingType ordering)
{
  PetscErrorCode ierr,(*f)(PC,MatOrderingType);

  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCLUSetMatOrdering_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,ordering);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCLUSetPivoting"
/*@
    PCLUSetPivoting - Determines when pivoting is done during LU. 
      For PETSc dense matrices column pivoting is always done, for PETSc sparse matrices
      it is never done. For the Matlab and SuperLU factorization this is used.

    Collective on PC

    Input Parameters:
+   pc - the preconditioner context
-   dtcol - 0.0 implies no pivoting, 1.0 complete pivoting (slower, requires more memory but more stable)

    Options Database Key:
.   -pc_lu_pivoting <dtcol>

    Level: intermediate

.seealso: PCILUSetMatOrdering(), PCLUSetPivotInBlocks()
@*/
PetscErrorCode PCLUSetPivoting(PC pc,PetscReal dtcol)
{
  PetscErrorCode ierr,(*f)(PC,PetscReal);

  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCLUSetPivoting_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,dtcol);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCLUSetPivotInBlocks"
/*@
    PCLUSetPivotInBlocks - Determines if pivoting is done while factoring each block
      with BAIJ or SBAIJ matrices

    Collective on PC

    Input Parameters:
+   pc - the preconditioner context
-   pivot - PETSC_TRUE or PETSC_FALSE

    Options Database Key:
.   -pc_lu_pivot_in_blocks <true,false>

    Level: intermediate

.seealso: PCILUSetMatOrdering(), PCLUSetPivoting()
@*/
PetscErrorCode PCLUSetPivotInBlocks(PC pc,PetscTruth pivot)
{
  PetscErrorCode ierr,(*f)(PC,PetscTruth);

  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCLUSetPivotInBlocks_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,pivot);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------ */

/*MC
   PCLU - Uses a direct solver, based on LU factorization, as a preconditioner

   Options Database Keys:
+  -pc_lu_reuse_ordering - Activate PCLUSetReuseOrdering()
.  -pc_lu_reuse_fill - Activates PCLUSetReuseFill()
.  -pc_lu_fill <fill> - Sets fill amount
.  -pc_lu_in_place - Activates in-place factorization
.  -pc_lu_mat_ordering_type <nd,rcm,...> - Sets ordering routine
-  -pc_lu_pivot_in_blocks <true,false> - allow pivoting within the small blocks during factorization (may increase
                                         stability of factorization.

   Notes: Not all options work for all matrix formats
          Run with -help to see additional options for particular matrix formats or factorization
          algorithms

   Level: beginner

   Concepts: LU factorization, direct solver

   Notes: Usually this will compute an "exact" solution in one iteration and does 
          not need a Krylov method (i.e. you can use -ksp_type preonly, or 
          KSPSetType(ksp,KSPPREONLY) for the Krylov method

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC,
           PCILU, PCCHOLESKY, PCICC, PCLUSetReuseOrdering(), PCLUSetReuseFill(), PCGetFactoredMatrix(),
           PCLUSetFill(), PCLUSetUseInPlace(), PCLUSetMatOrdering(), PCFactorSetPivoting(),
           PCLUSetPivotingInBlocks()
M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCreate_LU"
PetscErrorCode PCCreate_LU(PC pc)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PC_LU          *dir;

  PetscFunctionBegin;
  ierr = PetscNew(PC_LU,&dir);CHKERRQ(ierr);
  PetscLogObjectMemory(pc,sizeof(PC_LU));

  ierr = MatFactorInfoInitialize(&dir->info);CHKERRQ(ierr);
  dir->fact                  = 0;
  dir->inplace               = PETSC_FALSE;
  dir->nonzerosalongdiagonal = PETSC_FALSE;

  dir->info.fill           = 5.0;
  dir->info.dtcol          = 1.e-6; /* default to pivoting; this is only thing PETSc LU supports */
  dir->info.shiftnz        = 0.0;
  dir->info.zeropivot      = 1.e-12;
  dir->info.pivotinblocks  = 1.0;
  dir->info.shiftpd        = PETSC_FALSE;
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

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCLUSetFill_C","PCLUSetFill_LU",
                    PCLUSetFill_LU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCLUSetUseInPlace_C","PCLUSetUseInPlace_LU",
                    PCLUSetUseInPlace_LU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCLUSetMatOrdering_C","PCLUSetMatOrdering_LU",
                    PCLUSetMatOrdering_LU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCLUSetReuseOrdering_C","PCLUSetReuseOrdering_LU",
                    PCLUSetReuseOrdering_LU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCLUSetReuseFill_C","PCLUSetReuseFill_LU",
                    PCLUSetReuseFill_LU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCLUSetPivoting_C","PCLUSetPivoting_LU",
                    PCLUSetPivoting_LU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCLUSetPivotInBlocks_C","PCLUSetPivotInBlocks_LU",
                    PCLUSetPivotInBlocks_LU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCLUReorderForNonzeroDiagonal_C","PCLUReorderForNonzeroDiagonal_LU",
                    PCLUReorderForNonzeroDiagonal_LU);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
