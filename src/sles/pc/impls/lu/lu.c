/*$Id: lu.c,v 1.136 2000/08/24 22:42:31 bsmith Exp bsmith $*/
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
  PetscTruth      reusefill;        /* reuse fill from previous LU */
  MatLUInfo       info;
} PC_LU;


EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ /*<a name="PCLUSetReuseOrdering_LU"></a>*/"PCLUSetReuseOrdering_LU"
int PCLUSetReuseOrdering_LU(PC pc,PetscTruth flag)
{
  PC_LU *lu;

  PetscFunctionBegin;
  lu               = (PC_LU*)pc->data;
  lu->reuseordering = flag;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ /*<a name="PCLUSetReuseFill_LU"></a>*/"PCLUSetReuseFill_LU"
int PCLUSetReuseFill_LU(PC pc,PetscTruth flag)
{
  PC_LU *lu;

  PetscFunctionBegin;
  lu = (PC_LU*)pc->data;
  lu->reusefill = flag;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNC__  
#define __FUNC__ /*<a name="PCSetFromOptions_LU"></a>*/"PCSetFromOptions_LU"
static int PCSetFromOptions_LU(PC pc)
{
  PC_LU      *lu = (PC_LU*)pc->data;
  int        ierr;
  PetscTruth flg;
  char       tname[256];

  PetscFunctionBegin;
  if (!MatOrderingRegisterAllCalled) {
    ierr = MatOrderingRegisterAll(PETSC_NULL);CHKERRQ(ierr);
  }
  ierr = OptionsHead("LU options");CHKERRQ(ierr);
    ierr = OptionsName("-pc_lu_in_place","Form LU in the same memory as the matrix","PCLUSetUseInPlace",&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PCLUSetUseInPlace(pc);CHKERRQ(ierr);
    }
    ierr = OptionsDouble("-pc_lu_fill","Expected non-zeros in LU/non-zeros in matrix","PCLUSetFill",lu->info.fill,&lu->info.fill,0);CHKERRQ(ierr);

    ierr = OptionsHasName(pc->prefix,"-pc_lu_damping",&flg);CHKERRQ(ierr);
    if (flg) {
        ierr = PCLUSetDamping(pc,0.0);CHKERRQ(ierr);
    }
    ierr = OptionsDouble("-pc_lu_damping","Damping added to diagonal","PCLUSetDamping",lu->info.damping,&lu->info.damping,0);CHKERRQ(ierr);

    ierr = OptionsName("-pc_lu_reuse_fill","Use fill from previous factorization","PCLUSetReuseFill",&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PCLUSetReuseFill(pc,PETSC_TRUE);CHKERRQ(ierr);
    }
    ierr = OptionsName("-pc_lu_reuse_ordering","Reuse ordering from previous factorization","PCLUSetReuseOrdering",&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PCLUSetReuseOrdering(pc,PETSC_TRUE);CHKERRQ(ierr);
    }

    ierr = OptionsList("-pc_lu_mat_ordering_type","Reordering to reduce nonzeros in LU","PCLUSetMatOrdering",MatOrderingList,lu->ordering,tname,256,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PCLUSetMatOrdering(pc,tname);CHKERRQ(ierr);
    }
    ierr = OptionsDouble("-pc_lu_nonzeros_along_diagonal","Reorder to remove zeros from diagonal","MatReorderForNonzeroDiagonal",0.0,0,0);CHKERRQ(ierr);

    ierr = OptionsDouble("-pc_lu_column_pivoting","Column pivoting tolerance (not used)","PCLUSetColumnPivoting",lu->info.dtcol,&lu->info.dtcol,&flg);CHKERRQ(ierr);
  ierr = OptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="PCView_LU"></a>*/"PCView_LU"
static int PCView_LU(PC pc,Viewer viewer)
{
  PC_LU      *lu = (PC_LU*)pc->data;
  int        ierr;
  PetscTruth isascii,isstring;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,ASCII_VIEWER,&isascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,STRING_VIEWER,&isstring);CHKERRQ(ierr);
  if (isascii) {
    MatInfo info;

    if (lu->inplace) {ierr = ViewerASCIIPrintf(viewer,"  LU: in-place factorization\n");CHKERRQ(ierr);}
    else             {ierr = ViewerASCIIPrintf(viewer,"  LU: out-of-place factorization\n");CHKERRQ(ierr);}
    ierr = ViewerASCIIPrintf(viewer,"    matrix ordering: %s\n",lu->ordering);CHKERRQ(ierr);
    if (lu->fact) {
      ierr = MatGetInfo(lu->fact,MAT_LOCAL,&info);CHKERRQ(ierr);
      ierr = ViewerASCIIPrintf(viewer,"    LU nonzeros %g\n",info.nz_used);CHKERRQ(ierr);
    }
    if (lu->reusefill)    {ierr = ViewerASCIIPrintf(viewer,"       Reusing fill from past factorization\n");CHKERRQ(ierr);}
    if (lu->reuseordering) {ierr = ViewerASCIIPrintf(viewer,"       Reusing reordering from past factorization\n");CHKERRQ(ierr);}
  } else if (isstring) {
    ierr = ViewerStringSPrintf(viewer," order=%s",lu->ordering);CHKERRQ(ierr);CHKERRQ(ierr);
  } else {
    SETERRQ1(1,1,"Viewer type %s not supported for PCLU",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="PCGetFactoredMatrix_LU"></a>*/"PCGetFactoredMatrix_LU"
static int PCGetFactoredMatrix_LU(PC pc,Mat *mat)
{
  PC_LU *dir = (PC_LU*)pc->data;

  PetscFunctionBegin;
  if (!dir->fact) SETERRQ(1,1,"Matrix not yet factored; call after SLESSetUp() or PCSetUp()");
  *mat = dir->fact;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="PCSetUp_LU"></a>*/"PCSetUp_LU"
static int PCSetUp_LU(PC pc)
{
  int        ierr;
  PetscTruth flg;
  PC_LU      *dir = (PC_LU*)pc->data;

  PetscFunctionBegin;
  if (dir->reusefill && pc->setupcalled) dir->info.fill = dir->actualfill;

  if (dir->inplace) {
    if (dir->row && dir->col && dir->row != dir->col) {ierr = ISDestroy(dir->row);CHKERRQ(ierr);}
    if (dir->col) {ierr = ISDestroy(dir->col);CHKERRQ(ierr);}
    ierr = MatGetOrdering(pc->pmat,dir->ordering,&dir->row,&dir->col);CHKERRQ(ierr);
    if (dir->row) {PLogObjectParent(pc,dir->row); PLogObjectParent(pc,dir->col);}
    ierr = MatLUFactor(pc->pmat,dir->row,dir->col,&dir->info);CHKERRQ(ierr);
    dir->fact = pc->pmat;
  } else {
    MatInfo info;
    if (!pc->setupcalled) {
      ierr = MatGetOrdering(pc->pmat,dir->ordering,&dir->row,&dir->col);CHKERRQ(ierr);
      ierr = OptionsHasName(pc->prefix,"-pc_lu_nonzeros_along_diagonal",&flg);CHKERRQ(ierr);
      if (flg) {
        PetscReal tol = 1.e-10;
        ierr = OptionsGetDouble(pc->prefix,"-pc_lu_nonzeros_along_diagonal",&tol,PETSC_NULL);CHKERRQ(ierr);
        ierr = MatReorderForNonzeroDiagonal(pc->pmat,tol,dir->row,dir->col);CHKERRQ(ierr);
      }
      if (dir->row) {PLogObjectParent(pc,dir->row); PLogObjectParent(pc,dir->col);}
      ierr = MatLUFactorSymbolic(pc->pmat,dir->row,dir->col,&dir->info,&dir->fact);CHKERRQ(ierr);
      ierr = MatGetInfo(dir->fact,MAT_LOCAL,&info);CHKERRQ(ierr);
      dir->actualfill = info.fill_ratio_needed;
      PLogObjectParent(pc,dir->fact);
    } else if (pc->flag != SAME_NONZERO_PATTERN) {
      if (!dir->reuseordering) {
        if (dir->row && dir->col && dir->row != dir->col) {ierr = ISDestroy(dir->row);CHKERRQ(ierr);}
        if (dir->col) {ierr = ISDestroy(dir->col);CHKERRQ(ierr);}
        ierr = MatGetOrdering(pc->pmat,dir->ordering,&dir->row,&dir->col);CHKERRQ(ierr);
        ierr = OptionsHasName(pc->prefix,"-pc_lu_nonzeros_along_diagonal",&flg);CHKERRQ(ierr);
        if (flg) {
          PetscReal tol = 1.e-10;
          ierr = OptionsGetDouble(pc->prefix,"-pc_lu_nonzeros_along_diagonal",&tol,PETSC_NULL);CHKERRQ(ierr);
          ierr = MatReorderForNonzeroDiagonal(pc->pmat,tol,dir->row,dir->col);CHKERRQ(ierr);
        }
        if (dir->row) {PLogObjectParent(pc,dir->row); PLogObjectParent(pc,dir->col);}
      }
      ierr = MatDestroy(dir->fact);CHKERRQ(ierr);
      ierr = MatLUFactorSymbolic(pc->pmat,dir->row,dir->col,&dir->info,&dir->fact);CHKERRQ(ierr);
      ierr = MatGetInfo(dir->fact,MAT_LOCAL,&info);CHKERRQ(ierr);
      dir->actualfill = info.fill_ratio_needed;
      PLogObjectParent(pc,dir->fact);
    }
    ierr = MatLUFactorNumeric(pc->pmat,&dir->fact);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="PCDestroy_LU"></a>*/"PCDestroy_LU"
static int PCDestroy_LU(PC pc)
{
  PC_LU *dir = (PC_LU*)pc->data;
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
#define __FUNC__ /*<a name="PCApply_LU"></a>*/"PCApply_LU"
static int PCApply_LU(PC pc,Vec x,Vec y)
{
  PC_LU *dir = (PC_LU*)pc->data;
  int   ierr;

  PetscFunctionBegin;
  if (dir->inplace) {ierr = MatSolve(pc->pmat,x,y);CHKERRQ(ierr);}
  else              {ierr = MatSolve(dir->fact,x,y);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="PCApplyTranspose_LU"></a>*/"PCApplyTranspose_LU"
static int PCApplyTranspose_LU(PC pc,Vec x,Vec y)
{
  PC_LU *dir = (PC_LU*)pc->data;
  int   ierr;

  PetscFunctionBegin;
  if (dir->inplace) {ierr = MatSolveTranspose(pc->pmat,x,y);CHKERRQ(ierr);}
  else              {ierr = MatSolveTranspose(dir->fact,x,y);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/* -----------------------------------------------------------------------------------*/

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ /*<a name="PCLUSetFill_LU"></a>*/"PCLUSetFill_LU"
int PCLUSetFill_LU(PC pc,PetscReal fill)
{
  PC_LU *dir;

  PetscFunctionBegin;
  dir = (PC_LU*)pc->data;
  dir->info.fill = fill;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ /*<a name="PCLUSetDamping_LU"></a>*/"PCLUSetDamping_LU"
int PCLUSetDamping_LU(PC pc,PetscReal damping)
{
  PC_LU *dir;

  PetscFunctionBegin;
  dir = (PC_LU*)pc->data;
  dir->info.damping = damping;
  dir->info.damp    = 1.0;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ /*<a name="PCLUSetUseInPlace_LU"></a>*/"PCLUSetUseInPlace_LU"
int PCLUSetUseInPlace_LU(PC pc)
{
  PC_LU *dir;

  PetscFunctionBegin;
  dir = (PC_LU*)pc->data;
  dir->inplace = 1;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ /*<a name="PCLUSetMatOrdering_LU"></a>*/"PCLUSetMatOrdering_LU"
int PCLUSetMatOrdering_LU(PC pc,MatOrderingType ordering)
{
  PC_LU *dir = (PC_LU*)pc->data;
  int   ierr;

  PetscFunctionBegin;
  ierr = PetscStrfree(dir->ordering);CHKERRQ(ierr);
  ierr = PetscStrallocpy(ordering,&dir->ordering);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ /*<a name="PCLUSetColumnPivoting_LU"></a>*/"PCLUSetColumnPivoting_LU"
int PCLUSetColumnPivoting_LU(PC pc,PetscReal dtcol)
{
  PC_LU *dir = (PC_LU*)pc->data;

  PetscFunctionBegin;
  if (dtcol < 0.0 || dtcol > 1.0) SETERRQ1(1,1,"Column pivot tolerance is %g must be between 0 and 1",dtcol);
  dir->info.dtcol = dtcol;
  PetscFunctionReturn(0);
}
EXTERN_C_END

/* -----------------------------------------------------------------------------------*/

#undef __FUNC__  
#define __FUNC__ /*<a name="PCLUSetReuseOrdering"></a>*/"PCLUSetReuseOrdering"
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
int PCLUSetReuseOrdering(PC pc,PetscTruth flag)
{
  int ierr,(*f)(PC,PetscTruth);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCLUSetReuseOrdering_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,flag);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="PCLUSetReuseFill"></a>*/"PCLUSetReuseFill"
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
int PCLUSetReuseFill(PC pc,PetscTruth flag)
{
  int ierr,(*f)(PC,PetscTruth);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCLUSetReuseFill_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,flag);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="PCLUSetFill"></a>*/"PCLUSetFill"
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
int PCLUSetFill(PC pc,PetscReal fill)
{
  int ierr,(*f)(PC,PetscReal);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  if (fill < 1.0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,1,"Fill factor cannot be less then 1.0");
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCLUSetFill_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,fill);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="PCLUSetDamping"></a>*/"PCLUSetDamping"
/*@
   PCLUSetDamping - adds this quantity to the diagonal of the matrix during the 
     LU numerical factorization

   Collective on PC
   
   Input Parameters:
+  pc - the preconditioner context
-  damping - amount of damping

   Options Database Key:
.  -pc_lu_damping <damping> - Sets damping amount

   Level: intermediate

.keywords: PC, set, factorization, direct, fill

.seealso: PCILUSetFill(), PCILUSetDamp()
@*/
int PCLUSetDamping(PC pc,PetscReal damping)
{
  int ierr,(*f)(PC,PetscReal);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCLUSetDamping_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,damping);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="PCLUSetUseInPlace"></a>*/"PCLUSetUseInPlace"
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
   a call to SLESSetOperators().
   This is because the Krylov space methods require an application of the 
   matrix multiplication, which is not possible here because the matrix has 
   been factored in-place, replacing the original matrix.

   Level: intermediate

.keywords: PC, set, factorization, direct, inplace, in-place, LU

.seealso: PCILUSetUseInPlace()
@*/
int PCLUSetUseInPlace(PC pc)
{
  int ierr,(*f)(PC);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCLUSetUseInPlace_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="PCLUSetMatOrdering"></a>*/"PCLUSetMatOrdering"
/*@
    PCLUSetMatOrdering - Sets the ordering routine (to reduce fill) to 
    be used it the LU factorization.

    Collective on PC

    Input Parameters:
+   pc - the preconditioner context
-   ordering - the matrix ordering name, for example, MATORDERING_ND or MATORDERING_RCM

    Options Database Key:
.   -pc_lu_mat_ordering_type <nd,rcm,...> - Sets ordering routine

    Level: intermediate

.seealso: PCILUSetMatOrdering()
@*/
int PCLUSetMatOrdering(PC pc,MatOrderingType ordering)
{
  int ierr,(*f)(PC,MatOrderingType);

  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCLUSetMatOrdering_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,ordering);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="PCLUSetColumnPivoting"></a>*/"PCLUSetColumnPivoting"
/*@
    PCLUSetColumnPivoting - Determines when column pivoting is done during LU. 
      For PETSc dense matrices column pivoting is always done, for PETSc sparse matrices
      it is never done. For the Matlab factorization this is used.

    Collective on PC

    Input Parameters:
+   pc - the preconditioner context
-   dtcol - 0.0 implies no pivoting, 1.0 complete column pivoting (slower, requires more memory but more stable)

    Options Database Key:
.   -pc_lu_column_pivoting - dttol

    Level: intermediate

.seealso: PCILUSetMatOrdering()
@*/
int PCLUSetColumnPivoting(PC pc,PetscReal dtcol)
{
  int ierr,(*f)(PC,PetscReal);

  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCLUSetColumnPivoting_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,dtcol);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------ */

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ /*<a name="PCCreate_LU"></a>*/"PCCreate_LU"
int PCCreate_LU(PC pc)
{
  int   ierr;
  PC_LU *dir     = PetscNew(PC_LU);CHKPTRQ(dir);

  PetscFunctionBegin;
  PLogObjectMemory(pc,sizeof(PC_LU));

  dir->fact             = 0;
  dir->inplace          = 0;
  dir->info.fill        = 5.0;
  dir->info.dtcol       = 0.0; /* default to no pivoting; this is only thing PETSc LU supports */
  dir->info.damping     = 0.0;
  dir->info.damp        = 0.0;
  dir->col              = 0;
  dir->row              = 0;
  ierr = PetscStrallocpy(MATORDERING_ND,&dir->ordering);CHKERRQ(ierr);
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

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCLUSetFill_C","PCLUSetFill_LU",
                    PCLUSetFill_LU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCLUSetDamping_C","PCLUSetDamping_LU",
                    PCLUSetDamping_LU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCLUSetUseInPlace_C","PCLUSetUseInPlace_LU",
                    PCLUSetUseInPlace_LU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCLUSetMatOrdering_C","PCLUSetMatOrdering_LU",
                    PCLUSetMatOrdering_LU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCLUSetReuseOrdering_C","PCLUSetReuseOrdering_LU",
                    PCLUSetReuseOrdering_LU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCLUSetReuseFill_C","PCLUSetReuseFill_LU",
                    PCLUSetReuseFill_LU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCLUSetColumnPivoting_C","PCLUSetColumnPivoting_LU",
                    PCLUSetColumnPivoting_LU);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
