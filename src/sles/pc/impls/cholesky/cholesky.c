/*$Id: cholesky.c,v 1.2 2000/09/25 17:23:06 bsmith Exp curfman $*/
/*
   Defines a direct factorization preconditioner for any Mat implementation
   Note: this need not be consided a preconditioner since it supplies
         a direct solver.
*/
#include "src/sles/pc/pcimpl.h"                /*I "petscpc.h" I*/

typedef struct {
  Mat             fact;             /* factored matrix */
  PetscReal       actualfill;       /* actual fill in factor */
  int             inplace;          /* flag indicating in-place factorization */
  IS              row,col;          /* index sets used for reordering */
  MatOrderingType ordering;         /* matrix ordering */
  PetscTruth      reuseordering;    /* reuses previous reordering computed */
  PetscTruth      reusefill;        /* reuse fill from previous Cholesky */
  MatCholeskyInfo info;
} PC_Cholesky;

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ /*<a name="PCCholeskySetReuseOrdering_Cholesky"></a>*/"PCCholeskySetReuseOrdering_Cholesky"
int PCCholeskySetReuseOrdering_Cholesky(PC pc,PetscTruth flag)
{
  PC_Cholesky *lu;

  PetscFunctionBegin;
  lu               = (PC_Cholesky*)pc->data;
  lu->reuseordering = flag;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ /*<a name="PCCholeskySetReuseFill_Cholesky"></a>*/"PCCholeskySetReuseFill_Cholesky"
int PCCholeskySetReuseFill_Cholesky(PC pc,PetscTruth flag)
{
  PC_Cholesky *lu;

  PetscFunctionBegin;
  lu = (PC_Cholesky*)pc->data;
  lu->reusefill = flag;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNC__  
#define __FUNC__ /*<a name="PCSetFromOptions_Cholesky"></a>*/"PCSetFromOptions_Cholesky"
static int PCSetFromOptions_Cholesky(PC pc)
{
  PC_Cholesky      *lu = (PC_Cholesky*)pc->data;
  int        ierr;
  PetscTruth flg;
  char       tname[256];

  PetscFunctionBegin;
  if (!MatOrderingRegisterAllCalled) {
    ierr = MatOrderingRegisterAll(PETSC_NULL);CHKERRQ(ierr);
  }
  ierr = OptionsHead("Cholesky options");CHKERRQ(ierr);
    ierr = OptionsName("-pc_cholesky_in_place","Form Cholesky in the same memory as the matrix","PCCholeskySetUseInPlace",&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PCCholeskySetUseInPlace(pc);CHKERRQ(ierr);
    }
    ierr = OptionsDouble("-pc_cholesky_fill","Expected non-zeros in Cholesky/non-zeros in matrix","PCCholeskySetFill",lu->info.fill,&lu->info.fill,0);CHKERRQ(ierr);

    ierr = OptionsName("-pc_cholesky_reuse_fill","Use fill from previous factorization","PCCholeskySetReuseFill",&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PCCholeskySetReuseFill(pc,PETSC_TRUE);CHKERRQ(ierr);
    }
    ierr = OptionsName("-pc_cholesky_reuse_ordering","Reuse ordering from previous factorization","PCCholeskySetReuseOrdering",&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PCCholeskySetReuseOrdering(pc,PETSC_TRUE);CHKERRQ(ierr);
    }

    ierr = OptionsList("-pc_cholesky_mat_ordering_type","Reordering to reduce nonzeros in Cholesky","PCCholeskySetMatOrdering",MatOrderingList,lu->ordering,tname,256,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PCCholeskySetMatOrdering(pc,tname);CHKERRQ(ierr);
    }
    ierr = OptionsDouble("-pc_cholesky_nonzeros_along_diagonal","Reorder to remove zeros from diagonal","MatReorderForNonzeroDiagonal",0.0,0,0);CHKERRQ(ierr);

  ierr = OptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="PCView_Cholesky"></a>*/"PCView_Cholesky"
static int PCView_Cholesky(PC pc,Viewer viewer)
{
  PC_Cholesky      *lu = (PC_Cholesky*)pc->data;
  int        ierr;
  PetscTruth isascii,isstring;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,ASCII_VIEWER,&isascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,STRING_VIEWER,&isstring);CHKERRQ(ierr);
  if (isascii) {
    MatInfo info;

    if (lu->inplace) {ierr = ViewerASCIIPrintf(viewer,"  Cholesky: in-place factorization\n");CHKERRQ(ierr);}
    else             {ierr = ViewerASCIIPrintf(viewer,"  Cholesky: out-of-place factorization\n");CHKERRQ(ierr);}
    ierr = ViewerASCIIPrintf(viewer,"    matrix ordering: %s\n",lu->ordering);CHKERRQ(ierr);
    if (lu->fact) {
      ierr = MatGetInfo(lu->fact,MAT_LOCAL,&info);CHKERRQ(ierr);
      ierr = ViewerASCIIPrintf(viewer,"    Cholesky nonzeros %g\n",info.nz_used);CHKERRQ(ierr);
    }
    if (lu->reusefill)    {ierr = ViewerASCIIPrintf(viewer,"       Reusing fill from past factorization\n");CHKERRQ(ierr);}
    if (lu->reuseordering) {ierr = ViewerASCIIPrintf(viewer,"       Reusing reordering from past factorization\n");CHKERRQ(ierr);}
  } else if (isstring) {
    ierr = ViewerStringSPrintf(viewer," order=%s",lu->ordering);CHKERRQ(ierr);CHKERRQ(ierr);
  } else {
    SETERRQ1(1,1,"Viewer type %s not supported for PCCholesky",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="PCGetFactoredMatrix_Cholesky"></a>*/"PCGetFactoredMatrix_Cholesky"
static int PCGetFactoredMatrix_Cholesky(PC pc,Mat *mat)
{
  PC_Cholesky *dir = (PC_Cholesky*)pc->data;

  PetscFunctionBegin;
  if (!dir->fact) SETERRQ(1,1,"Matrix not yet factored; call after SLESSetUp() or PCSetUp()");
  *mat = dir->fact;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="PCSetUp_Cholesky"></a>*/"PCSetUp_Cholesky"
static int PCSetUp_Cholesky(PC pc)
{
  int        ierr;
  PetscTruth flg;
  PC_Cholesky      *dir = (PC_Cholesky*)pc->data;

  PetscFunctionBegin;
  if (dir->reusefill && pc->setupcalled) dir->info.fill = dir->actualfill;

  if (dir->inplace) {
    if (dir->row && dir->col && dir->row != dir->col) {ierr = ISDestroy(dir->row);CHKERRQ(ierr);}
    if (dir->col) {ierr = ISDestroy(dir->col);CHKERRQ(ierr);}
    ierr = MatGetOrdering(pc->pmat,dir->ordering,&dir->row,&dir->col);CHKERRQ(ierr);
    if (dir->row) {PLogObjectParent(pc,dir->row); PLogObjectParent(pc,dir->col);}
    ierr = MatCholeskyFactor(pc->pmat,dir->row,dir->info.fill);CHKERRQ(ierr);
    dir->fact = pc->pmat;
  } else {
    MatInfo info;
    if (!pc->setupcalled) {
      ierr = MatGetOrdering(pc->pmat,dir->ordering,&dir->row,&dir->col);CHKERRQ(ierr);
      ierr = OptionsHasName(pc->prefix,"-pc_cholesky_nonzeros_along_diagonal",&flg);CHKERRQ(ierr);
      if (flg) {
        PetscReal tol = 1.e-10;
        ierr = OptionsGetDouble(pc->prefix,"-pc_cholesky_nonzeros_along_diagonal",&tol,PETSC_NULL);CHKERRQ(ierr);
        ierr = MatReorderForNonzeroDiagonal(pc->pmat,tol,dir->row,dir->col);CHKERRQ(ierr);
      }
      if (dir->row) {PLogObjectParent(pc,dir->row); PLogObjectParent(pc,dir->col);}
      ierr = MatCholeskyFactorSymbolic(pc->pmat,dir->row,dir->info.fill,&dir->fact);CHKERRQ(ierr);
      ierr = MatGetInfo(dir->fact,MAT_LOCAL,&info);CHKERRQ(ierr);
      dir->actualfill = info.fill_ratio_needed;
      PLogObjectParent(pc,dir->fact);
    } else if (pc->flag != SAME_NONZERO_PATTERN) {
      if (!dir->reuseordering) {
        if (dir->row && dir->col && dir->row != dir->col) {ierr = ISDestroy(dir->row);CHKERRQ(ierr);}
        if (dir->col) {ierr = ISDestroy(dir->col);CHKERRQ(ierr);}
        ierr = MatGetOrdering(pc->pmat,dir->ordering,&dir->row,&dir->col);CHKERRQ(ierr);
        ierr = OptionsHasName(pc->prefix,"-pc_cholesky_nonzeros_along_diagonal",&flg);CHKERRQ(ierr);
        if (flg) {
          PetscReal tol = 1.e-10;
          ierr = OptionsGetDouble(pc->prefix,"-pc_cholesky_nonzeros_along_diagonal",&tol,PETSC_NULL);CHKERRQ(ierr);
          ierr = MatReorderForNonzeroDiagonal(pc->pmat,tol,dir->row,dir->col);CHKERRQ(ierr);
        }
        if (dir->row) {PLogObjectParent(pc,dir->row); PLogObjectParent(pc,dir->col);}
      }
      ierr = MatDestroy(dir->fact);CHKERRQ(ierr);
      ierr = MatCholeskyFactorSymbolic(pc->pmat,dir->row,dir->info.fill,&dir->fact);CHKERRQ(ierr);
      ierr = MatGetInfo(dir->fact,MAT_LOCAL,&info);CHKERRQ(ierr);
      dir->actualfill = info.fill_ratio_needed;
      PLogObjectParent(pc,dir->fact);
    }
    ierr = MatCholeskyFactorNumeric(pc->pmat,&dir->fact);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="PCDestroy_Cholesky"></a>*/"PCDestroy_Cholesky"
static int PCDestroy_Cholesky(PC pc)
{
  PC_Cholesky *dir = (PC_Cholesky*)pc->data;
  int   ierr;

  PetscFunctionBegin;
  if (!dir->inplace && dir->fact) {ierr = MatDestroy(dir->fact);CHKERRQ(ierr);}
  if (dir->row && dir->col && dir->row != dir->col) {ierr = ISDestroy(dir->row);CHKERRQ(ierr);}
  if (dir->col) {ierr = ISDestroy(dir->col);CHKERRQ(ierr);}
  ierr = PetscStrfree(dir->ordering);CHKERRQ(ierr);
  ierr = PetscFree(dir);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="PCApply_Cholesky"></a>*/"PCApply_Cholesky"
static int PCApply_Cholesky(PC pc,Vec x,Vec y)
{
  PC_Cholesky *dir = (PC_Cholesky*)pc->data;
  int   ierr;

  PetscFunctionBegin;
  if (dir->inplace) {ierr = MatSolve(pc->pmat,x,y);CHKERRQ(ierr);}
  else              {ierr = MatSolve(dir->fact,x,y);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="PCApplyTranspose_Cholesky"></a>*/"PCApplyTranspose_Cholesky"
static int PCApplyTranspose_Cholesky(PC pc,Vec x,Vec y)
{
  PC_Cholesky *dir = (PC_Cholesky*)pc->data;
  int   ierr;

  PetscFunctionBegin;
  if (dir->inplace) {ierr = MatSolveTranspose(pc->pmat,x,y);CHKERRQ(ierr);}
  else              {ierr = MatSolveTranspose(dir->fact,x,y);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/* -----------------------------------------------------------------------------------*/

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ /*<a name="PCCholeskySetFill_Cholesky"></a>*/"PCCholeskySetFill_Cholesky"
int PCCholeskySetFill_Cholesky(PC pc,PetscReal fill)
{
  PC_Cholesky *dir;

  PetscFunctionBegin;
  dir = (PC_Cholesky*)pc->data;
  dir->info.fill = fill;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ /*<a name="PCCholeskySetDamping_Cholesky"></a>*/"PCCholeskySetDamping_Cholesky"
int PCCholeskySetDamping_Cholesky(PC pc,PetscReal damping)
{
  PC_Cholesky *dir;

  PetscFunctionBegin;
  dir = (PC_Cholesky*)pc->data;
  dir->info.damping = damping;
  dir->info.damp    = 1.0;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ /*<a name="PCCholeskySetUseInPlace_Cholesky"></a>*/"PCCholeskySetUseInPlace_Cholesky"
int PCCholeskySetUseInPlace_Cholesky(PC pc)
{
  PC_Cholesky *dir;

  PetscFunctionBegin;
  dir = (PC_Cholesky*)pc->data;
  dir->inplace = 1;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ /*<a name="PCCholeskySetMatOrdering_Cholesky"></a>*/"PCCholeskySetMatOrdering_Cholesky"
int PCCholeskySetMatOrdering_Cholesky(PC pc,MatOrderingType ordering)
{
  PC_Cholesky *dir = (PC_Cholesky*)pc->data;
  int   ierr;

  PetscFunctionBegin;
  ierr = PetscStrfree(dir->ordering);CHKERRQ(ierr);
  ierr = PetscStrallocpy(ordering,&dir->ordering);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

/* -----------------------------------------------------------------------------------*/

#undef __FUNC__  
#define __FUNC__ /*<a name="PCCholeskySetReuseOrdering"></a>*/"PCCholeskySetReuseOrdering"
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
int PCCholeskySetReuseOrdering(PC pc,PetscTruth flag)
{
  int ierr,(*f)(PC,PetscTruth);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCCholeskySetReuseOrdering_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,flag);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="PCCholeskySetReuseFill"></a>*/"PCCholeskySetReuseFill"
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
int PCCholeskySetReuseFill(PC pc,PetscTruth flag)
{
  int ierr,(*f)(PC,PetscTruth);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCCholeskySetReuseFill_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,flag);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="PCCholeskySetFill"></a>*/"PCCholeskySetFill"
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
int PCCholeskySetFill(PC pc,PetscReal fill)
{
  int ierr,(*f)(PC,PetscReal);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  if (fill < 1.0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,1,"Fill factor cannot be less then 1.0");
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCCholeskySetFill_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,fill);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="PCCholeskySetDamping"></a>*/"PCCholeskySetDamping"
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

.seealso: PCICholeskySetFill(), PCILUSetDamp()
@*/
int PCCholeskySetDamping(PC pc,PetscReal damping)
{
  int ierr,(*f)(PC,PetscReal);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCCholeskySetDamping_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,damping);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="PCCholeskySetUseInPlace"></a>*/"PCCholeskySetUseInPlace"
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
   a call to SLESSetOperators().
   This is because the Krylov space methods require an application of the 
   matrix multiplication, which is not possible here because the matrix has 
   been factored in-place, replacing the original matrix.

   Level: intermediate

.keywords: PC, set, factorization, direct, inplace, in-place, Cholesky

.seealso: PCICholeskySetUseInPlace()
@*/
int PCCholeskySetUseInPlace(PC pc)
{
  int ierr,(*f)(PC);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCCholeskySetUseInPlace_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="PCCholeskySetMatOrdering"></a>*/"PCCholeskySetMatOrdering"
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
int PCCholeskySetMatOrdering(PC pc,MatOrderingType ordering)
{
  int ierr,(*f)(PC,MatOrderingType);

  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCCholeskySetMatOrdering_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,ordering);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------ */

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ /*<a name="PCCreate_Cholesky"></a>*/"PCCreate_Cholesky"
int PCCreate_Cholesky(PC pc)
{
  int   ierr;
  PC_Cholesky *dir     = PetscNew(PC_Cholesky);CHKPTRQ(dir);

  PetscFunctionBegin;
  PLogObjectMemory(pc,sizeof(PC_Cholesky));

  dir->fact             = 0;
  dir->inplace          = 0;
  dir->info.fill        = 5.0;
  dir->info.damping     = 0.0;
  dir->info.damp        = 0.0;
  dir->col              = 0;
  dir->row              = 0;
  ierr = PetscStrallocpy(MATORDERING_ND,&dir->ordering);CHKERRQ(ierr);
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
