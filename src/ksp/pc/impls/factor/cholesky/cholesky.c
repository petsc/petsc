#define PETSCKSP_DLL

/*
   Defines a direct factorization preconditioner for any Mat implementation
   Note: this need not be consided a preconditioner since it supplies
         a direct solver.
*/
#include "private/pcimpl.h"                /*I "petscpc.h" I*/

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
#define __FUNCT__ "PCFactorSetZeroPivot_Cholesky"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetZeroPivot_Cholesky(PC pc,PetscReal z)
{
  PC_Cholesky *ch;

  PetscFunctionBegin;
  ch                 = (PC_Cholesky*)pc->data;
  ch->info.zeropivot = z;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetShiftNonzero_Cholesky"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetShiftNonzero_Cholesky(PC pc,PetscReal shift)
{
  PC_Cholesky *dir;

  PetscFunctionBegin;
  dir = (PC_Cholesky*)pc->data;
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
#define __FUNCT__ "PCFactorSetShiftPd_Cholesky"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetShiftPd_Cholesky(PC pc,PetscTruth shift)
{
  PC_Cholesky *dir;
 
  PetscFunctionBegin;
  dir = (PC_Cholesky*)pc->data;
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
#define __FUNCT__ "PCFactorSetReuseOrdering_Cholesky"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetReuseOrdering_Cholesky(PC pc,PetscTruth flag)
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
#define __FUNCT__ "PCFactorSetReuseFill_Cholesky"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetReuseFill_Cholesky(PC pc,PetscTruth flag)
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
  ierr = PetscOptionsName("-pc_factor_in_place","Form Cholesky in the same memory as the matrix","PCFactorSetUseInPlace",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PCFactorSetUseInPlace(pc);CHKERRQ(ierr);
  }
  ierr = PetscOptionsReal("-pc_factor_fill","Expected non-zeros in Cholesky/non-zeros in matrix","PCFactorSetFill",lu->info.fill,&lu->info.fill,0);CHKERRQ(ierr);
  
  ierr = PetscOptionsName("-pc_factor_reuse_fill","Use fill from previous factorization","PCFactorSetReuseFill",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PCFactorSetReuseFill(pc,PETSC_TRUE);CHKERRQ(ierr);
  }
  ierr = PetscOptionsName("-pc_factor_reuse_ordering","Reuse ordering from previous factorization","PCFactorSetReuseOrdering",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PCFactorSetReuseOrdering(pc,PETSC_TRUE);CHKERRQ(ierr);
  }
  
  ierr = MatGetOrderingList(&ordlist);CHKERRQ(ierr);
  ierr = PetscOptionsList("-pc_factor_mat_ordering_type","Reordering to reduce nonzeros in Cholesky","PCFactorSetMatOrdering",ordlist,lu->ordering,tname,256,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PCFactorSetMatOrdering(pc,tname);CHKERRQ(ierr);
  }
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
      ierr = PetscViewerASCIIPrintf(viewer,"    Cholesky nonzeros %G\n",info.nz_used);CHKERRQ(ierr);
      ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_FACTOR_INFO);CHKERRQ(ierr);
      ierr = MatView(lu->fact,viewer);CHKERRQ(ierr);
      ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    }
    if (lu->reusefill)    {ierr = PetscViewerASCIIPrintf(viewer,"       Reusing fill from past factorization\n");CHKERRQ(ierr);}
    if (lu->reuseordering) {ierr = PetscViewerASCIIPrintf(viewer,"       Reusing reordering from past factorization\n");CHKERRQ(ierr);}
  } else if (isstring) {
    ierr = PetscViewerStringSPrintf(viewer," order=%s",lu->ordering);CHKERRQ(ierr);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported for PCCHOLESKY",((PetscObject)viewer)->type_name);
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
    if (dir->row) {ierr = PetscLogObjectParent(pc,dir->row);CHKERRQ(ierr);}
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
      ierr = PetscOptionsHasName(pc->prefix,"-pc_factor_nonzeros_along_diagonal",&flg);CHKERRQ(ierr);
      if (flg) {
        PetscReal tol = 1.e-10;
        ierr = PetscOptionsGetReal(pc->prefix,"-pc_factor_nonzeros_along_diagonal",&tol,PETSC_NULL);CHKERRQ(ierr);
        ierr = MatReorderForNonzeroDiagonal(pc->pmat,tol,dir->row,dir->row);CHKERRQ(ierr);
      }
      if (dir->row) {ierr = PetscLogObjectParent(pc,dir->row);CHKERRQ(ierr);}
      ierr = MatCholeskyFactorSymbolic(pc->pmat,dir->row,&dir->info,&dir->fact);CHKERRQ(ierr);
      ierr = MatGetInfo(dir->fact,MAT_LOCAL,&info);CHKERRQ(ierr);
      dir->actualfill = info.fill_ratio_needed;
      ierr = PetscLogObjectParent(pc,dir->fact);CHKERRQ(ierr);
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
        ierr = PetscOptionsHasName(pc->prefix,"-pc_factor_nonzeros_along_diagonal",&flg);CHKERRQ(ierr);
        if (flg) {
          PetscReal tol = 1.e-10;
          ierr = PetscOptionsGetReal(pc->prefix,"-pc_factor_nonzeros_along_diagonal",&tol,PETSC_NULL);CHKERRQ(ierr);
          ierr = MatReorderForNonzeroDiagonal(pc->pmat,tol,dir->row,dir->row);CHKERRQ(ierr);
        }
        if (dir->row) {ierr = PetscLogObjectParent(pc,dir->row);CHKERRQ(ierr);}
      }
      ierr = MatDestroy(dir->fact);CHKERRQ(ierr);
      ierr = MatCholeskyFactorSymbolic(pc->pmat,dir->row,&dir->info,&dir->fact);CHKERRQ(ierr);
      ierr = MatGetInfo(dir->fact,MAT_LOCAL,&info);CHKERRQ(ierr);
      dir->actualfill = info.fill_ratio_needed;
      ierr = PetscLogObjectParent(pc,dir->fact);CHKERRQ(ierr);
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
#define __FUNCT__ "PCFactorSetFill_Cholesky"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetFill_Cholesky(PC pc,PetscReal fill)
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
#define __FUNCT__ "PCFactorSetUseInPlace_Cholesky"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetUseInPlace_Cholesky(PC pc)
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
#define __FUNCT__ "PCFactorSetMatOrdering_Cholesky"
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetMatOrdering_Cholesky(PC pc,MatOrderingType ordering)
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
#define __FUNCT__ "PCFactorSetReuseOrdering"
/*@
   PCFactorSetReuseOrdering - When similar matrices are factored, this
   causes the ordering computed in the first factor to be used for all
   following factors.

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  flag - PETSC_TRUE to reuse else PETSC_FALSE

   Options Database Key:
.  -pc_factor_reuse_ordering - Activate PCFactorSetReuseOrdering()

   Level: intermediate

.keywords: PC, levels, reordering, factorization, incomplete, LU

.seealso: PCFactorSetReuseFill()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetReuseOrdering(PC pc,PetscTruth flag)
{
  PetscErrorCode ierr,(*f)(PC,PetscTruth);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCFactorSetReuseOrdering_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,flag);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCFactorSetReuseFill"
/*@
   PCFactorSetReuseFill - When matrices with same different nonzero structure are factored,
   this causes later ones to use the fill ratio computed in the initial factorization.

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  flag - PETSC_TRUE to reuse else PETSC_FALSE

   Options Database Key:
.  -pc_factor_reuse_fill - Activates PCFactorSetReuseFill()

   Level: intermediate

.keywords: PC, levels, reordering, factorization, incomplete, Cholesky

.seealso: PCFactorSetReuseOrdering()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCFactorSetReuseFill(PC pc,PetscTruth flag)
{
  PetscErrorCode ierr,(*f)(PC,PetscTruth);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,2);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCFactorSetReuseFill_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,flag);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

/*MC
   PCCholesky - Uses a direct solver, based on Cholesky factorization, as a preconditioner

   Options Database Keys:
+  -pc_factor_reuse_ordering - Activate PCFactorSetReuseOrdering()
.  -pc_factor_reuse_fill - Activates PCFactorSetReuseFill()
.  -pc_factor_fill <fill> - Sets fill amount
.  -pc_factor_in_place - Activates in-place factorization
.  -pc_factor_mat_ordering_type <nd,rcm,...> - Sets ordering routine
.  -pc_factor_shift_nonzero <shift> - Sets shift amount or PETSC_DECIDE for the default
-  -pc_factor_shift_positive_definite [PETSC_TRUE/PETSC_FALSE] - Activate/Deactivate PCFactorSetShiftPd(); the value
   is optional with PETSC_TRUE being the default

   Notes: Not all options work for all matrix formats

   Level: beginner

   Concepts: Cholesky factorization, direct solver

   Notes: Usually this will compute an "exact" solution in one iteration and does 
          not need a Krylov method (i.e. you can use -ksp_type preonly, or 
          KSPSetType(ksp,KSPPREONLY) for the Krylov method

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC,
           PCILU, PCLU, PCICC, PCFactorSetReuseOrdering(), PCFactorSetReuseFill(), PCGetFactoredMatrix(),
           PCFactorSetFill(), PCFactorSetShiftNonzero(), PCFactorSetShiftPd(),
	   PCFactorSetUseInPlace(), PCFactorSetMatOrdering(),PCFactorSetShiftNonzero(),PCFactorSetShiftPd()

M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCreate_Cholesky"
PetscErrorCode PETSCKSP_DLLEXPORT PCCreate_Cholesky(PC pc)
{
  PetscErrorCode ierr;
  PC_Cholesky    *dir;

  PetscFunctionBegin;
  ierr = PetscNew(PC_Cholesky,&dir);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(pc,sizeof(PC_Cholesky));CHKERRQ(ierr);

  dir->fact                   = 0;
  dir->inplace                = PETSC_FALSE;
  ierr = MatFactorInfoInitialize(&dir->info);CHKERRQ(ierr);
  dir->info.fill              = 5.0;
  dir->info.shiftnz           = 0.0;
  dir->info.shiftpd           = 0.0; /* false */
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

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetZeroPivot_C","PCFactorSetZeroPivot_Cholesky",
                    PCFactorSetZeroPivot_Cholesky);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetShiftNonzero_C","PCFactorSetShiftNonzero_Cholesky",
                    PCFactorSetShiftNonzero_Cholesky);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetShiftPd_C","PCFactorSetShiftPd_Cholesky",
                    PCFactorSetShiftPd_Cholesky);CHKERRQ(ierr);

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetFill_C","PCFactorSetFill_Cholesky",
                    PCFactorSetFill_Cholesky);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetUseInPlace_C","PCFactorSetUseInPlace_Cholesky",
                    PCFactorSetUseInPlace_Cholesky);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetMatOrdering_C","PCFactorSetMatOrdering_Cholesky",
                    PCFactorSetMatOrdering_Cholesky);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetReuseOrdering_C","PCFactorSetReuseOrdering_Cholesky",
                    PCFactorSetReuseOrdering_Cholesky);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCFactorSetReuseFill_C","PCFactorSetReuseFill_Cholesky",
                    PCFactorSetReuseFill_Cholesky);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
