
/*
   Defines a direct factorization preconditioner for any Mat implementation
   Note: this need not be consided a preconditioner since it supplies
         a direct solver.
*/
#include "src/ksp/pc/pcimpl.h"                /*I "petscpc.h" I*/

typedef struct {
  Mat             fact;             /* factored matrix */
  PetscReal       actualfill;       /* actual fill in factor */
  PetscTruth      inplace;          /* flag indicating in-place factorization */
  IS              row,col;          /* index sets used for reordering */
  MatOrderingType ordering;         /* matrix ordering */
  PetscTruth      reuseordering;    /* reuses previous reordering computed */
  PetscTruth      reusefill;        /* reuse fill from previous Cholesky */
  MatFactorInfo   info;
} PC_Cholesky;

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCholeskySetReuseOrdering_Cholesky"
PetscErrorCode PCCholeskySetReuseOrdering_Cholesky(PC pc,PetscTruth flag)
{
  PC_Cholesky *lu;
  
  PetscFunctionBegin;
  lu               = (PC_Cholesky*)pc->data;
  lu->reuseordering = flag;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCholeskySetReuseFill_Cholesky"
PetscErrorCode PCCholeskySetReuseFill_Cholesky(PC pc,PetscTruth flag)
{
  PC_Cholesky *lu;
  
  PetscFunctionBegin;
  lu = (PC_Cholesky*)pc->data;
  lu->reusefill = flag;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "PCSetFromOptions_Cholesky"
static PetscErrorCode PCSetFromOptions_Cholesky(PC pc)
{
  PC_Cholesky    *lu = (PC_Cholesky*)pc->data;
  PetscErrorCode ierr;
  PetscTruth     flg;
  char           tname[256];
  PetscFList     ordlist;
  
  PetscFunctionBegin;
  ierr = MatOrderingRegisterAll(PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsHead("Cholesky options");CHKERRQ(ierr);
  ierr = PetscOptionsName("-pc_cholesky_in_place","Form Cholesky in the same memory as the matrix","PCCholeskySetUseInPlace",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PCCholeskySetUseInPlace(pc);CHKERRQ(ierr);
  }
  ierr = PetscOptionsReal("-pc_cholesky_fill","Expected non-zeros in Cholesky/non-zeros in matrix","PCCholeskySetFill",lu->info.fill,&lu->info.fill,0);CHKERRQ(ierr);
  
  ierr = PetscOptionsName("-pc_cholesky_reuse_fill","Use fill from previous factorization","PCCholeskySetReuseFill",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PCCholeskySetReuseFill(pc,PETSC_TRUE);CHKERRQ(ierr);
  }
  ierr = PetscOptionsName("-pc_cholesky_reuse_ordering","Reuse ordering from previous factorization","PCCholeskySetReuseOrdering",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PCCholeskySetReuseOrdering(pc,PETSC_TRUE);CHKERRQ(ierr);
  }
  
  ierr = MatGetOrderingList(&ordlist);CHKERRQ(ierr);
  ierr = PetscOptionsList("-pc_cholesky_mat_ordering_type","Reordering to reduce nonzeros in Cholesky","PCCholeskySetMatOrdering",ordlist,lu->ordering,tname,256,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PCCholeskySetMatOrdering(pc,tname);CHKERRQ(ierr);
  }
  ierr = PetscOptionsName("-pc_cholesky_damping","Damping added to diagonal","PCCholestkySetDamping",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PCCholeskySetDamping(pc,(PetscReal) PETSC_DECIDE);CHKERRQ(ierr);
  }
  ierr = PetscOptionsReal("-pc_cholesky_damping","Damping added to diagonal","PCCholeskySetDamping",lu->info.damping,&lu->info.damping,0);CHKERRQ(ierr);
  ierr = PetscOptionsName("-pc_cholesky_shift","Manteuffel shift applied to diagonal","PCCholeskySetShift",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PCCholeskySetShift(pc,PETSC_TRUE);CHKERRQ(ierr);
  }

  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCView_Cholesky"
static PetscErrorCode PCView_Cholesky(PC pc,PetscViewer viewer)
{
  PC_Cholesky    *lu = (PC_Cholesky*)pc->data;
  PetscErrorCode ierr;
  PetscTruth     iascii,isstring;
  
  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_STRING,&isstring);CHKERRQ(ierr);
  if (iascii) {
    MatInfo info;
    
    if (lu->inplace) {ierr = PetscViewerASCIIPrintf(viewer,"  Cholesky: in-place factorization\n");CHKERRQ(ierr);}
    else             {ierr = PetscViewerASCIIPrintf(viewer,"  Cholesky: out-of-place factorization\n");CHKERRQ(ierr);}
    ierr = PetscViewerASCIIPrintf(viewer,"    matrix ordering: %s\n",lu->ordering);CHKERRQ(ierr);
    if (lu->fact) {
      ierr = MatGetInfo(lu->fact,MAT_LOCAL,&info);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"    Cholesky nonzeros %g\n",info.nz_used);CHKERRQ(ierr);
      ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_FACTOR_INFO);CHKERRQ(ierr);
      ierr = MatView(lu->fact,viewer);CHKERRQ(ierr);
      ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    }
    if (lu->reusefill)    {ierr = PetscViewerASCIIPrintf(viewer,"       Reusing fill from past factorization\n");CHKERRQ(ierr);}
    if (lu->reuseordering) {ierr = PetscViewerASCIIPrintf(viewer,"       Reusing reordering from past factorization\n");CHKERRQ(ierr);}
  } else if (isstring) {
    ierr = PetscViewerStringSPrintf(viewer," order=%s",lu->ordering);CHKERRQ(ierr);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported for PCCholesky",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCGetFactoredMatrix_Cholesky"
static PetscErrorCode PCGetFactoredMatrix_Cholesky(PC pc,Mat *mat)
{
  PC_Cholesky *dir = (PC_Cholesky*)pc->data;
  
  PetscFunctionBegin;
  if (!dir->fact) SETERRQ(PETSC_ERR_ORDER,"Matrix not yet factored; call after KSPSetUp() or PCSetUp()");
  *mat = dir->fact;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSetUp_Cholesky"
static PetscErrorCode PCSetUp_Cholesky(PC pc)
{
  PetscErrorCode ierr;
  PetscTruth     flg;
  PC_Cholesky    *dir = (PC_Cholesky*)pc->data;

  PetscFunctionBegin;
  if (dir->reusefill && pc->setupcalled) dir->info.fill = dir->actualfill;
  
  if (dir->inplace) {
    if (dir->row && dir->col && (dir->row != dir->col)) {
      ierr = ISDestroy(dir->row);CHKERRQ(ierr);
      dir->row = 0;
    }
    if (dir->col) {
      ierr = ISDestroy(dir->col);CHKERRQ(ierr);
      dir->col = 0;
    }
    ierr = MatGetOrdering(pc->pmat,dir->ordering,&dir->row,&dir->col);CHKERRQ(ierr);
    if (dir->col && (dir->row != dir->col)) {  /* only use row ordering for SBAIJ */
      ierr = ISDestroy(dir->col);CHKERRQ(ierr);
      dir->col=0;
    }
    if (dir->row) {PetscLogObjectParent(pc,dir->row);}
    ierr = MatCholeskyFactor(pc->pmat,dir->row,&dir->info);CHKERRQ(ierr);
    dir->fact = pc->pmat;
  } else {
    MatInfo info;
    if (!pc->setupcalled) {
      ierr = MatGetOrdering(pc->pmat,dir->ordering,&dir->row,&dir->col);CHKERRQ(ierr);
      if (dir->col && (dir->row != dir->col)) {  /* only use row ordering for SBAIJ */
        ierr = ISDestroy(dir->col);CHKERRQ(ierr); 
        dir->col=0; 
      }
      ierr = PetscOptionsHasName(pc->prefix,"-pc_cholesky_nonzeros_along_diagonal",&flg);CHKERRQ(ierr);
      if (flg) {
        PetscReal tol = 1.e-10;
        ierr = PetscOptionsGetReal(pc->prefix,"-pc_cholesky_nonzeros_along_diagonal",&tol,PETSC_NULL);CHKERRQ(ierr);
        ierr = MatReorderForNonzeroDiagonal(pc->pmat,tol,dir->row,dir->row);CHKERRQ(ierr);
      }
      if (dir->row) {PetscLogObjectParent(pc,dir->row);}
      ierr = MatCholeskyFactorSymbolic(pc->pmat,dir->row,&dir->info,&dir->fact);CHKERRQ(ierr);
      ierr = MatGetInfo(dir->fact,MAT_LOCAL,&info);CHKERRQ(ierr);
      dir->actualfill = info.fill_ratio_needed;
      PetscLogObjectParent(pc,dir->fact);
    } else if (pc->flag != SAME_NONZERO_PATTERN) {
      if (!dir->reuseordering) {
        if (dir->row && dir->col && (dir->row != dir->col)) {
          ierr = ISDestroy(dir->row);CHKERRQ(ierr);
          dir->row = 0;
        }
        if (dir->col) {
          ierr = ISDestroy(dir->col);CHKERRQ(ierr);
          dir->col =0;
        }
        ierr = MatGetOrdering(pc->pmat,dir->ordering,&dir->row,&dir->col);CHKERRQ(ierr);
        if (dir->col && (dir->row != dir->col)) {  /* only use row ordering for SBAIJ */
          ierr = ISDestroy(dir->col);CHKERRQ(ierr);
          dir->col=0;
        }
        ierr = PetscOptionsHasName(pc->prefix,"-pc_cholesky_nonzeros_along_diagonal",&flg);CHKERRQ(ierr);
        if (flg) {
          PetscReal tol = 1.e-10;
          ierr = PetscOptionsGetReal(pc->prefix,"-pc_cholesky_nonzeros_along_diagonal",&tol,PETSC_NULL);CHKERRQ(ierr);
          ierr = MatReorderForNonzeroDiagonal(pc->pmat,tol,dir->row,dir->row);CHKERRQ(ierr);
        }
        if (dir->row) {PetscLogObjectParent(pc,dir->row);}
      }
      ierr = MatDestroy(dir->fact);CHKERRQ(ierr);
      ierr = MatCholeskyFactorSymbolic(pc->pmat,dir->row,&dir->info,&dir->fact);CHKERRQ(ierr);
      ierr = MatGetInfo(dir->fact,MAT_LOCAL,&info);CHKERRQ(ierr);
      dir->actualfill = info.fill_ratio_needed;
      PetscLogObjectParent(pc,dir->fact);
    }
    ierr = MatCholeskyFactorNumeric(pc->pmat,&dir->info,&dir->fact);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCDestroy_Cholesky"
static PetscErrorCode PCDestroy_Cholesky(PC pc)
{
  PC_Cholesky    *dir = (PC_Cholesky*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!dir->inplace && dir->fact) {ierr = MatDestroy(dir->fact);CHKERRQ(ierr);}
  if (dir->row) {ierr = ISDestroy(dir->row);CHKERRQ(ierr);}
  if (dir->col) {ierr = ISDestroy(dir->col);CHKERRQ(ierr);}
  ierr = PetscStrfree(dir->ordering);CHKERRQ(ierr);
  ierr = PetscFree(dir);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCApply_Cholesky"
static PetscErrorCode PCApply_Cholesky(PC pc,Vec x,Vec y)
{
  PC_Cholesky    *dir = (PC_Cholesky*)pc->data;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  if (dir->inplace) {ierr = MatSolve(pc->pmat,x,y);CHKERRQ(ierr);}
  else              {ierr = MatSolve(dir->fact,x,y);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCApplyTranspose_Cholesky"
static PetscErrorCode PCApplyTranspose_Cholesky(PC pc,Vec x,Vec y)
{
  PC_Cholesky    *dir = (PC_Cholesky*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (dir->inplace) {ierr = MatSolveTranspose(pc->pmat,x,y);CHKERRQ(ierr);}
  else              {ierr = MatSolveTranspose(dir->fact,x,y);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/* -----------------------------------------------------------------------------------*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCholeskySetFill_Cholesky"
PetscErrorCode PCCholeskySetFill_Cholesky(PC pc,PetscReal fill)
{
  PC_Cholesky *dir;
  
  PetscFunctionBegin;
  dir = (PC_Cholesky*)pc->data;
  dir->info.fill = fill;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCholeskySetDamping_Cholesky"
PetscErrorCode PCCholeskySetDamping_Cholesky(PC pc,PetscReal damping)
{
  PC_Cholesky *dir;
  
  PetscFunctionBegin;
  dir = (PC_Cholesky*)pc->data;
  if (damping == (PetscReal) PETSC_DECIDE) {
    dir->info.damping = 1.e-12;
  } else {
    dir->info.damping = damping;
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCholeskySetShift_Cholesky"
PetscErrorCode PCCholeskySetShift_Cholesky(PC pc,PetscTruth shift)
{
  PC_Cholesky *dir;
  
  PetscFunctionBegin;
  dir = (PC_Cholesky*)pc->data;
  dir->info.shift = shift;
  if (shift) dir->info.shift_fraction = 0.0;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCholeskySetUseInPlace_Cholesky"
PetscErrorCode PCCholeskySetUseInPlace_Cholesky(PC pc)
{
  PC_Cholesky *dir;

  PetscFunctionBegin;
  dir = (PC_Cholesky*)pc->data;
  dir->inplace = PETSC_TRUE;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCholeskySetMatOrdering_Cholesky"
PetscErrorCode PCCholeskySetMatOrdering_Cholesky(PC pc,MatOrderingType ordering)
{
  PC_Cholesky    *dir = (PC_Cholesky*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscStrfree(dir->ordering);CHKERRQ(ierr);
  ierr = PetscStrallocpy(ordering,&dir->ordering);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

/* -----------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "PCCholeskySetReuseOrdering"
/*@
   PCCholeskySetReuseOrdering - When similar matrices are factored, this
   causes the ordering computed in the first factor to be used for all
   following factors.

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  flag - PETSC_TRUE to reuse else PETSC_FALSE

   Options Database Key:
.  -pc_cholesky_reuse_ordering - Activate PCCholeskySetReuseOrdering()

   Level: intermediate

.keywords: PC, levels, reordering, factorization, incomplete, LU

.seealso: PCCholeskySetReuseFill(), PCICholeskySetReuseOrdering(), PCICholeskyDTSetReuseFill()
@*/
PetscErrorCode PCCholeskySetReuseOrdering(PC pc,PetscTruth flag)
{
  PetscErrorCode ierr,(*f)(PC,PetscTruth);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCCholeskySetReuseOrdering_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,flag);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCCholeskySetReuseFill"
/*@
   PCCholeskySetReuseFill - When matrices with same nonzero structure are Cholesky factored,
   this causes later ones to use the fill computed in the initial factorization.

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  flag - PETSC_TRUE to reuse else PETSC_FALSE

   Options Database Key:
.  -pc_cholesky_reuse_fill - Activates PCCholeskySetReuseFill()

   Level: intermediate

.keywords: PC, levels, reordering, factorization, incomplete, Cholesky

.seealso: PCICholeskySetReuseOrdering(), PCCholeskySetReuseOrdering(), PCICholeskyDTSetReuseFill()
@*/
PetscErrorCode PCCholeskySetReuseFill(PC pc,PetscTruth flag)
{
  PetscErrorCode ierr,(*f)(PC,PetscTruth);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,2);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCCholeskySetReuseFill_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,flag);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCCholeskySetFill"
/*@
   PCCholeskySetFill - Indicates the amount of fill you expect in the factored matrix,
   fill = number nonzeros in factor/number nonzeros in original matrix.

   Collective on PC
   
   Input Parameters:
+  pc - the preconditioner context
-  fill - amount of expected fill

   Options Database Key:
.  -pc_cholesky_fill <fill> - Sets fill amount

   Level: intermediate

   Note:
   For sparse matrix factorizations it is difficult to predict how much 
   fill to expect. By running with the option -log_info PETSc will print the 
   actual amount of fill used; allowing you to set the value accurately for
   future runs. Default PETSc uses a value of 5.0

.keywords: PC, set, factorization, direct, fill

.seealso: PCILUSetFill()
@*/
PetscErrorCode PCCholeskySetFill(PC pc,PetscReal fill)
{
  PetscErrorCode ierr,(*f)(PC,PetscReal);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  if (fill < 1.0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Fill factor cannot be less then 1.0");
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCCholeskySetFill_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,fill);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCCholeskySetDamping"
/*@
   PCCholeskySetDamping - Adds this quantity to the diagonal of the matrix during the 
   Cholesky numerical factorization.

   Collective on PC
   
   Input Parameters:
+  pc - the preconditioner context
-  damping - amount of damping

   Options Database Key:
.  -pc_cholesky_damping <damping> - Sets damping amount

   Level: intermediate

.keywords: PC, set, factorization, direct, fill

.seealso: PCCholeskySetFill(), PCILUSetDamping()
@*/
PetscErrorCode PCCholeskySetDamping(PC pc,PetscReal damping)
{
  PetscErrorCode ierr,(*f)(PC,PetscReal);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCCholeskySetDamping_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,damping);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCCholeskySetShift"
/*@
   PCCholeskySetShift - specify whether to use Manteuffel shifting of Cholesky.
   If an Cholesky factorisation breaks down because of nonpositive pivots,
   adding sufficient identity to the diagonal will remedy this.
   Setting this causes a bisection method to find the minimum shift that
   will lead to a well-defined Cholesky.

   Input parameters:
+  pc - the preconditioner context
-  shifting - PETSC_TRUE to set shift else PETSC_FALSE

   Options Database Key:
.  -pc_ilu_shift - Activate PCCholeskySetShift()

   Level: intermediate

.keywords: PC, indefinite, factorization, incomplete, Cholesky

.seealso: PCILUSetShift()
@*/
PetscErrorCode PCCholeskySetShift(PC pc,PetscTruth shift)
{
  PetscErrorCode ierr,(*f)(PC,PetscTruth);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCCholeskySetShift_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,shift);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCCholeskySetUseInPlace"
/*@
   PCCholeskySetUseInPlace - Tells the system to do an in-place factorization.
   For dense matrices, this enables the solution of much larger problems. 
   For sparse matrices the factorization cannot be done truly in-place 
   so this does not save memory during the factorization, but after the matrix
   is factored, the original unfactored matrix is freed, thus recovering that
   space.

   Collective on PC

   Input Parameters:
.  pc - the preconditioner context

   Options Database Key:
.  -pc_cholesky_in_place - Activates in-place factorization

   Notes:
   PCCholeskySetUseInplace() can only be used with the KSP method KSPPREONLY or when 
   a different matrix is provided for the multiply and the preconditioner in 
   a call to KSPSetOperators().
   This is because the Krylov space methods require an application of the 
   matrix multiplication, which is not possible here because the matrix has 
   been factored in-place, replacing the original matrix.

   Level: intermediate

.keywords: PC, set, factorization, direct, inplace, in-place, Cholesky

.seealso: PCICholeskySetUseInPlace()
@*/
PetscErrorCode PCCholeskySetUseInPlace(PC pc)
{
  PetscErrorCode ierr,(*f)(PC);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCCholeskySetUseInPlace_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCCholeskySetMatOrdering"
/*@
    PCCholeskySetMatOrdering - Sets the ordering routine (to reduce fill) to 
    be used it the Cholesky factorization.

    Collective on PC

    Input Parameters:
+   pc - the preconditioner context
-   ordering - the matrix ordering name, for example, MATORDERING_ND or MATORDERING_RCM

    Options Database Key:
.   -pc_cholesky_mat_ordering_type <nd,rcm,...> - Sets ordering routine

    Level: intermediate

.seealso: PCICholeskySetMatOrdering()
@*/
PetscErrorCode PCCholeskySetMatOrdering(PC pc,MatOrderingType ordering)
{
  PetscErrorCode ierr,(*f)(PC,MatOrderingType);

  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCCholeskySetMatOrdering_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,ordering);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

/*MC
   PCCholesky - Uses a direct solver, based on Cholesky factorization, as a preconditioner

   Options Database Keys:
+  -pc_cholesky_reuse_ordering - Activate PCLUSetReuseOrdering()
.  -pc_cholesky_reuse_fill - Activates PCLUSetReuseFill()
.  -pc_cholesky_fill <fill> - Sets fill amount
.  -pc_cholesky_damping <damping> - Sets damping amount
.  -pc_cholesky_shift - Activates Manteuffel shift
.  -pc_cholesky_in_place - Activates in-place factorization
-  -pc_cholesky_mat_ordering_type <nd,rcm,...> - Sets ordering routine

   Notes: Not all options work for all matrix formats

   Level: beginner

   Concepts: Cholesky factorization, direct solver

   Notes: Usually this will compute an "exact" solution in one iteration and does 
          not need a Krylov method (i.e. you can use -ksp_type preonly, or 
          KSPSetType(ksp,KSPPREONLY) for the Krylov method

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC,
           PCILU, PCLU, PCICC, PCCholeskySetReuseOrdering(), PCCholeskySetReuseFill(), PCGetFactoredMatrix(),
           PCCholeskySetFill(), PCCholeskySetDamping(), PCCholeskySetShift(),
	   PCCholeskySetUseInPlace(), PCCholeskySetMatOrdering()

M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCreate_Cholesky"
PetscErrorCode PCCreate_Cholesky(PC pc)
{
  PetscErrorCode ierr;
  PC_Cholesky    *dir;

  PetscFunctionBegin;
  ierr = PetscNew(PC_Cholesky,&dir);CHKERRQ(ierr);
  PetscLogObjectMemory(pc,sizeof(PC_Cholesky));

  dir->fact                   = 0;
  dir->inplace                = PETSC_FALSE;
  ierr = MatFactorInfoInitialize(&dir->info);CHKERRQ(ierr);
  dir->info.fill              = 5.0;
  dir->info.damping           = 0.0;
  dir->info.shift             = PETSC_FALSE;
  dir->info.shift_fraction    = 0.0;
  dir->info.pivotinblocks     = 1.0;
  dir->col                    = 0;
  dir->row                    = 0;
  ierr = PetscStrallocpy(MATORDERING_NATURAL,&dir->ordering);CHKERRQ(ierr);
  dir->reusefill        = PETSC_FALSE;
  dir->reuseordering    = PETSC_FALSE;
  pc->data              = (void*)dir;

  pc->ops->destroy           = PCDestroy_Cholesky;
  pc->ops->apply             = PCApply_Cholesky;
  pc->ops->applytranspose    = PCApplyTranspose_Cholesky;
  pc->ops->setup             = PCSetUp_Cholesky;
  pc->ops->setfromoptions    = PCSetFromOptions_Cholesky;
  pc->ops->view              = PCView_Cholesky;
  pc->ops->applyrichardson   = 0;
  pc->ops->getfactoredmatrix = PCGetFactoredMatrix_Cholesky;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCCholeskySetFill_C","PCCholeskySetFill_Cholesky",
                    PCCholeskySetFill_Cholesky);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCCholeskySetDamping_C","PCCholeskySetDamping_Cholesky",
                    PCCholeskySetDamping_Cholesky);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCCholeskySetShift_C","PCCholeskySetShift_Cholesky",
                    PCCholeskySetShift_Cholesky);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCCholeskySetUseInPlace_C","PCCholeskySetUseInPlace_Cholesky",
                    PCCholeskySetUseInPlace_Cholesky);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCCholeskySetMatOrdering_C","PCCholeskySetMatOrdering_Cholesky",
                    PCCholeskySetMatOrdering_Cholesky);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCCholeskySetReuseOrdering_C","PCCholeskySetReuseOrdering_Cholesky",
                    PCCholeskySetReuseOrdering_Cholesky);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCCholeskySetReuseFill_C","PCCholeskySetReuseFill_Cholesky",
                    PCCholeskySetReuseFill_Cholesky);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
