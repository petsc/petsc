#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: lu.c,v 1.120 1999/10/13 20:37:54 bsmith Exp bsmith $";
#endif
/*
   Defines a direct factorization preconditioner for any Mat implementation
   Note: this need not be consided a preconditioner since it supplies
         a direct solver.
*/
#include "src/sles/pc/pcimpl.h"                /*I "pc.h" I*/

typedef struct {
  Mat             fact;             /* factored matrix */
  double          fill, actualfill; /* expected and actual fill in factor */
  int             inplace;          /* flag indicating in-place factorization */
  IS              row, col;         /* index sets used for reordering */
  MatOrderingType ordering;         /* matrix ordering */
  int             reuseorering;     /* reuses previous reordering computed */
  int             reusefill;        /* reuse fill from previous LU */
} PC_LU;


EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "PCLUSetReuseOrdering_LU"
int PCLUSetReuseOrdering_LU(PC pc,PetscTruth flag)
{
  PC_LU *lu;

  PetscFunctionBegin;
  lu               = (PC_LU *) pc->data;
  lu->reuseorering = (int) flag;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "PCLUSetReuseFill_LU"
int PCLUSetReuseFill_LU(PC pc,PetscTruth flag)
{
  PC_LU *lu;

  PetscFunctionBegin;
  lu = (PC_LU *) pc->data;
  lu->reusefill = (int) flag;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNC__  
#define __FUNC__ "PCSetFromOptions_LU"
static int PCSetFromOptions_LU(PC pc)
{
  int    ierr,flg;
  double fill;

  PetscFunctionBegin;
  ierr = OptionsHasName(pc->prefix,"-pc_lu_in_place",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PCLUSetUseInPlace(pc);CHKERRQ(ierr);
  }
  ierr = OptionsGetDouble(pc->prefix,"-pc_lu_fill",&fill,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PCLUSetFill(pc,fill);CHKERRQ(ierr);
  }
  ierr = OptionsHasName(pc->prefix,"-pc_lu_reuse_fill",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PCLUSetReuseFill(pc,PETSC_TRUE);CHKERRQ(ierr);
  }
  ierr = OptionsHasName(pc->prefix,"-pc_lu_reuse_ordering",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PCLUSetReuseOrdering(pc,PETSC_TRUE);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCPrintHelp_LU"
static int PCPrintHelp_LU(PC pc,char *p)
{
  int ierr;

  PetscFunctionBegin;
  ierr = (*PetscHelpPrintf)(pc->comm," Options for PCLU preconditioner:\n");CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(pc->comm," %spc_lu_in_place: do factorization in place\n",p);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(pc->comm," %spc_lu_fill <fill>: expected fill in factor\n",p);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(pc->comm," -mat_order <name>: ordering to reduce fill",p);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(pc->comm," (nd,natural,1wd,rcm,qmd)\n");CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(pc->comm," %spc_lu_nonzeros_along_diagonal <tol>: changes column ordering\n",p);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(pc->comm,"    to reduce the change of obtaining zero pivot during LU.\n");CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(pc->comm,"    If <tol> not given defaults to 1.e-10.\n");CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(pc->comm," %spc_lu_reuse_ordering:                          \n",p);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(pc->comm," %spc_lu_reuse_fill:                             \n",p);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCView_LU"
static int PCView_LU(PC pc,Viewer viewer)
{
  PC_LU      *lu = (PC_LU *) pc->data;
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
    if (lu->reuseorering) {ierr = ViewerASCIIPrintf(viewer,"       Reusing reordering from past factorization\n");CHKERRQ(ierr);}
  } else if (isstring) {
    ierr = ViewerStringSPrintf(viewer," order=%s",lu->ordering);CHKERRQ(ierr);CHKERRQ(ierr);
  } else {
    SETERRQ1(1,1,"Viewer type %s not supported for PCLU",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCGetFactoredMatrix_LU"
static int PCGetFactoredMatrix_LU(PC pc,Mat *mat)
{
  PC_LU *dir = (PC_LU *) pc->data;

  PetscFunctionBegin;
  if (!dir->fact) SETERRQ(1,1,"Matrix not yet factored; call after SLESSetUp() or PCSetUp()");
  *mat = dir->fact;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCSetUp_LU"
static int PCSetUp_LU(PC pc)
{
  int         ierr,flg;
  PC_LU       *dir = (PC_LU *) pc->data;

  PetscFunctionBegin;
  if (dir->reusefill && pc->setupcalled) dir->fill = dir->actualfill;

  if (dir->inplace) {
    if (dir->row && dir->col && dir->row != dir->col) {ierr = ISDestroy(dir->row);CHKERRQ(ierr);}
    if (dir->col) {ierr = ISDestroy(dir->col);CHKERRQ(ierr);}
    ierr = MatGetOrdering(pc->pmat,dir->ordering,&dir->row,&dir->col);CHKERRQ(ierr);
    if (dir->row) {PLogObjectParent(pc,dir->row); PLogObjectParent(pc,dir->col);}
    ierr = MatLUFactor(pc->pmat,dir->row,dir->col,dir->fill);CHKERRQ(ierr);
    dir->fact = pc->pmat;
  } else {
    MatInfo info;
    if (!pc->setupcalled) {
      ierr = MatGetOrdering(pc->pmat,dir->ordering,&dir->row,&dir->col);CHKERRQ(ierr);
      ierr = OptionsHasName(pc->prefix,"-pc_lu_nonzeros_along_diagonal",&flg);CHKERRQ(ierr);
      if (flg) {
        double tol = 1.e-10;
        ierr = OptionsGetDouble(pc->prefix,"-pc_lu_nonzeros_along_diagonal",&tol,&flg);CHKERRQ(ierr);
        ierr = MatReorderForNonzeroDiagonal(pc->pmat,tol,dir->row,dir->col);CHKERRQ(ierr);
      }
      if (dir->row) {PLogObjectParent(pc,dir->row); PLogObjectParent(pc,dir->col);}
      ierr = MatLUFactorSymbolic(pc->pmat,dir->row,dir->col,dir->fill,&dir->fact);CHKERRQ(ierr);
      ierr = MatGetInfo(dir->fact,MAT_LOCAL,&info);CHKERRQ(ierr);
      dir->actualfill = info.fill_ratio_needed;
      PLogObjectParent(pc,dir->fact);
    } else if (pc->flag != SAME_NONZERO_PATTERN) {
      if (!dir->reuseorering) {
        if (dir->row && dir->col && dir->row != dir->col) {ierr = ISDestroy(dir->row);CHKERRQ(ierr);}
        if (dir->col) {ierr = ISDestroy(dir->col);CHKERRQ(ierr);}
        ierr = MatGetOrdering(pc->pmat,dir->ordering,&dir->row,&dir->col);CHKERRQ(ierr);
        ierr = OptionsHasName(pc->prefix,"-pc_lu_nonzeros_along_diagonal",&flg);CHKERRQ(ierr);
        if (flg) {
          double tol = 1.e-10;
          ierr = OptionsGetDouble(pc->prefix,"-pc_lu_nonzeros_along_diagonal",&tol,&flg);CHKERRQ(ierr);
          ierr = MatReorderForNonzeroDiagonal(pc->pmat,tol,dir->row,dir->col);CHKERRQ(ierr);
        }
        if (dir->row) {PLogObjectParent(pc,dir->row); PLogObjectParent(pc,dir->col);}
      }
      ierr = MatDestroy(dir->fact);CHKERRQ(ierr);
      ierr = MatLUFactorSymbolic(pc->pmat,dir->row,dir->col,dir->fill,&dir->fact);CHKERRQ(ierr);
      ierr = MatGetInfo(dir->fact,MAT_LOCAL,&info);CHKERRQ(ierr);
      dir->actualfill = info.fill_ratio_needed;
      PLogObjectParent(pc,dir->fact);
    }
    ierr = MatLUFactorNumeric(pc->pmat,&dir->fact);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCDestroy_LU"
static int PCDestroy_LU(PC pc)
{
  PC_LU *dir = (PC_LU*) pc->data;
  int   ierr;

  PetscFunctionBegin;
  if (!dir->inplace && dir->fact) {ierr = MatDestroy(dir->fact);CHKERRQ(ierr);}
  if (dir->row && dir->col && dir->row != dir->col) {ierr = ISDestroy(dir->row);CHKERRQ(ierr);}
  if (dir->col) {ierr = ISDestroy(dir->col);CHKERRQ(ierr);}
  ierr = PetscFree(dir); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCApply_LU"
static int PCApply_LU(PC pc,Vec x,Vec y)
{
  PC_LU *dir = (PC_LU *) pc->data;
  int   ierr;

  PetscFunctionBegin;
  if (dir->inplace) {ierr = MatSolve(pc->pmat,x,y);CHKERRQ(ierr);}
  else              {ierr = MatSolve(dir->fact,x,y);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCApply_LU"
static int PCApplyTrans_LU(PC pc,Vec x,Vec y)
{
  PC_LU *dir = (PC_LU *) pc->data;
  int   ierr;

  PetscFunctionBegin;
  if (dir->inplace) {ierr = MatSolveTrans(pc->pmat,x,y);CHKERRQ(ierr);}
  else              {ierr = MatSolveTrans(dir->fact,x,y);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/* -----------------------------------------------------------------------------------*/

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "PCLUSetFill_LU"
int PCLUSetFill_LU(PC pc,double fill)
{
  PC_LU *dir;

  PetscFunctionBegin;
  dir = (PC_LU *) pc->data;
  dir->fill = fill;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "PCLUSetUseInPlace_LU"
int PCLUSetUseInPlace_LU(PC pc)
{
  PC_LU *dir;

  PetscFunctionBegin;
  dir = (PC_LU *) pc->data;
  dir->inplace = 1;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "PCLUSetMatOrdering_LU"
int PCLUSetMatOrdering_LU(PC pc, MatOrderingType ordering)
{
  PC_LU *dir = (PC_LU *) pc->data;

  PetscFunctionBegin;
  dir->ordering = ordering;
  PetscFunctionReturn(0);
}
EXTERN_C_END

/* -----------------------------------------------------------------------------------*/

#undef __FUNC__  
#define __FUNC__ "PCLUSetReuseOrdering"
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

.seealso: PCLUSetReuseFill(), PCILUSetReuseOrdering(), PCILUSetReuseFill()
@*/
int PCLUSetReuseOrdering(PC pc,PetscTruth flag)
{
  int ierr, (*f)(PC,PetscTruth);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCLUSetReuseOrdering_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,flag);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCLUSetReuseFill"
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

.seealso: PCILUSetReuseOrdering(), PCLUSetReuseOrdering(), PCILUSetReuseFill()
@*/
int PCLUSetReuseFill(PC pc,PetscTruth flag)
{
  int ierr, (*f)(PC,PetscTruth);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCLUSetReuseFill_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,flag);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCLUSetFill"
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
int PCLUSetFill(PC pc,double fill)
{
  int ierr, (*f)(PC,double);

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
#define __FUNC__ "PCLUSetUseInPlace"
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
  int ierr, (*f)(PC);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCLUSetUseInPlace_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCLUSetMatOrdering"
/*@
    PCLUSetMatOrdering - Sets the ordering routine (to reduce fill) to 
    be used it the LU factorization.

    Collective on PC

    Input Parameters:
+   pc - the preconditioner context
-   ordering - the matrix ordering name, for example, MATORDERING_ND or MATORDERING_RCM

    Options Database Key:
.   -mat_order <nd,rcm,...> - Sets ordering routine

    Level: intermediate

.seealso: PCILUSetMatOrdering()
@*/
int PCLUSetMatOrdering(PC pc, MatOrderingType ordering)
{
  int ierr, (*f)(PC,MatOrderingType);

  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCLUSetMatOrdering_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,ordering);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------ */

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "PCCreate_LU"
int PCCreate_LU(PC pc)
{
  int   ierr;
  PC_LU *dir     = PetscNew(PC_LU);CHKPTRQ(dir);

  PetscFunctionBegin;
  PLogObjectMemory(pc,sizeof(PC_LU));

  dir->fact             = 0;
  dir->inplace          = 0;
  dir->fill             = 5.0;
  dir->col              = 0;
  dir->row              = 0;
  dir->ordering         = MATORDERING_ND;
  dir->reusefill        = 0;
  dir->reuseorering     = 0;
  pc->data              = (void *) dir;

  pc->ops->destroy           = PCDestroy_LU;
  pc->ops->apply             = PCApply_LU;
  pc->ops->applytrans        = PCApplyTrans_LU;
  pc->ops->setup             = PCSetUp_LU;
  pc->ops->setfromoptions    = PCSetFromOptions_LU;
  pc->ops->printhelp         = PCPrintHelp_LU;
  pc->ops->view              = PCView_LU;
  pc->ops->applyrichardson   = 0;
  pc->ops->getfactoredmatrix = PCGetFactoredMatrix_LU;

  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCLUSetFill_C","PCLUSetFill_LU",
                    (void*)PCLUSetFill_LU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCLUSetUseInPlace_C","PCLUSetUseInPlace_LU",
                    (void*)PCLUSetUseInPlace_LU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCLUSetMatOrdering_C","PCLUSetMatOrdering_LU",
                    (void*)PCLUSetMatOrdering_LU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCLUSetReuseOrdering_C","PCLUSetReuseOrdering_LU",
                    (void*)PCLUSetReuseOrdering_LU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCLUSetReuseFill_C","PCLUSetReuseFill_LU",
                    (void*)PCLUSetReuseFill_LU);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
