#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ilu.c,v 1.93 1997/10/09 15:34:23 bsmith Exp bsmith $";
#endif
/*
   Defines a ILU factorization preconditioner for any Mat implementation
*/
#include "src/pc/pcimpl.h"                 /*I "pc.h"  I*/
#include "src/pc/impls/ilu/ilu.h"
#include "src/mat/matimpl.h"
#include "pinclude/pviewer.h"

extern int PCSetUp_ILU_MPIRowbs(PC);

static int (*setups[])(PC) = {0,
                              0,
                              0,
                              0,
#if defined(HAVE_BLOCKSOLVE) && !defined(__cplusplus)
                              PCSetUp_ILU_MPIRowbs,
#else
                              0,
#endif
                              0,
                              0,
                              0,   
                              0,
                              0,0,0,0,0};

#undef __FUNC__  
#define __FUNC__ "PCILUSetUseDropTolerance"
/*@
   PCILUSetUseDropTolerance - The preconditioner will use an ILU 
   based on a drop tolerance.

   Input Parameters:
.  pc - the preconditioner context
.  dt - the drop tolerance
.  dtcount - the max number of nonzeros allowed in a row?

   Options Database Key:
$  -pc_ilu_use_drop_tolerance <dt,dtcount>

   Note:
   This routine is NOT currently supported!

.keywords: PC, levels, reordering, factorization, incomplete, ILU
@*/
int PCILUSetUseDropTolerance(PC pc,double dt,int dtcount)
{
  PC_ILU *ilu;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  if (pc->type != PCILU) PetscFunctionReturn(0);
  ilu = (PC_ILU *) pc->data;
  ilu->usedt    = 1;
  ilu->dt       = dt;
  ilu->dtcount  = dtcount;
  PetscFunctionReturn(0);
}  

#undef __FUNC__  
#define __FUNC__ "PCILUSetFill"
/*@
   PCILUSetFill - Indicate the amount of fill you expect in the factored matrix,
       fill = number nonzeros in factor/number nonzeros in original matrix.

   Input Parameters:
.  pc - the preconditioner context
.  fill - amount of expected fill

$  -pc_ilu_fill <fill>

   Note:
    For sparse matrix factorizations it is difficult to predict how much 
  fill to expect. By running with the option -log_info PETSc will print the 
  actual amount of fill used; allowing you to set the value accurately for
  future runs. Bt default PETSc uses a value of 1.0

.keywords: PC, set, factorization, direct, fill

.seealso: PCLUSetFill()
@*/
int PCILUSetFill(PC pc,double fill)
{
  PC_ILU *dir;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  dir = (PC_ILU *) pc->data;
  if (pc->type != PCILU) PetscFunctionReturn(0);
  if (fill < 1.0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,1,"Fill factor cannot be less then 1.0");
  dir->fill = fill;
  PetscFunctionReturn(0);
}

/*@
     PCILUSetMatReordering - Sets the ordering routine (to reduce fill) to 
         be used it the ILU factorization.

    Input Parameters:
.   pc - the preconditioner context
.   ordering - the matrix ordering name, for example, ORDER_ND or ORDER_RCM

   Options Database:
.   -mat_order <nd,rcm,...>

.seealso: PCLUSetMatReordering()
@*/
int PCILUSetMatReordering(PC pc, MatReordering ordering)
{
  PC_ILU *dir = (PC_ILU *) pc->data;

  PetscFunctionBegin;
  if (pc->type != PCILU) PetscFunctionReturn(0);
  
  dir->ordering = ordering;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCILUSetReuseReordering"
/*@
   PCILUSetReuseReordering - When similar matrices are factored, this
   causes the ordering computed in the first factor to be used for all
   following factors; applies to both fill and drop tolerance ILUs.

   Input Parameters:
.  pc - the preconditioner context
.  flag - PETSC_TRUE to reuse else PETSC_FALSE

   Options Database Key:
$  -pc_ilu_reuse_reordering

.keywords: PC, levels, reordering, factorization, incomplete, ILU

.seealso: PCILUSetReuseFill()
@*/
int PCILUSetReuseReordering(PC pc,PetscTruth flag)
{
  PC_ILU *ilu;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  if (pc->type != PCILU) PetscFunctionReturn(0);
  ilu = (PC_ILU *) pc->data;
  ilu->reusereordering = (int) flag;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCILUSetReuseFill"
/*@
   PCILUSetReuseFill - When matrices with same nonzero structure are ILUDT factored,
     this causes later ones to use the fill computed in the initial factorization.

   Input Parameters:
.  pc - the preconditioner context
.  flag - PETSC_TRUE to reuse else PETSC_FALSE

   Options Database Key:
$  -pc_ilu_reuse_fill

.keywords: PC, levels, reordering, factorization, incomplete, ILU

.seealso: PCILUSetReuseReordering()
@*/
int PCILUSetReuseFill(PC pc,PetscTruth flag)
{
  PC_ILU *ilu;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  if (pc->type != PCILU) PetscFunctionReturn(0);
  ilu = (PC_ILU *) pc->data;
  ilu->reusefill = (int) flag;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCILUSetLevels"
/*@
   PCILUSetLevels - Sets the number of levels of fill to use.

   Input Parameters:
.  pc - the preconditioner context
.  levels - number of levels of fill

   Options Database Key:
$  -pc_ilu_levels <levels>

.keywords: PC, levels, fill, factorization, incomplete, ILU
@*/
int PCILUSetLevels(PC pc,int levels)
{
  PC_ILU *ilu;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  if (pc->type != PCILU) PetscFunctionReturn(0);
  if (levels < 0) SETERRQ(1,0,"negative levels");
  ilu = (PC_ILU *) pc->data;
  if (pc->type != PCILU) PetscFunctionReturn(0);
  ilu->levels = levels;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCILUSetUseInPlace"
/*@
   PCILUSetUseInPlace - Tells the system to do an in-place incomplete factorization.

   Input Parameters:
.  pc - the preconditioner context

   Options Database Key:
$  -pc_ilu_in_place

   Note:
   PCILUSetUseInPlace() is intended for use with matrix-free variants of
   Krylov methods, or when a different matrices are employed for the linear
   system and preconditioner.  Do NOT use this option if the linear system
   matrix also serves as the preconditioning matrix, since the factored
   matrix would then overwrite the original matrix. 

.keywords: PC, set, factorization, inplace, in-place, ILU

.seealso:  PCLUSetUseInPlace()
@*/
int PCILUSetUseInPlace(PC pc)
{
  PC_ILU *dir;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  dir = (PC_ILU *) pc->data;
  if (pc->type != PCILU) PetscFunctionReturn(0);
  dir->inplace = 1;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCSetFromOptions_ILU"
static int PCSetFromOptions_ILU(PC pc)
{
  int         levels,ierr,flg,dtmax = 2;
  double      dt[2],fill;

  PetscFunctionBegin;
  ierr = OptionsGetInt(pc->prefix,"-pc_ilu_levels",&levels,&flg); CHKERRQ(ierr);
  if (flg) {
    ierr = PCILUSetLevels(pc,levels); CHKERRQ(ierr);
  }
  ierr = OptionsHasName(pc->prefix,"-pc_ilu_in_place",&flg); CHKERRQ(ierr);
  if (flg) {
    ierr = PCILUSetUseInPlace(pc); CHKERRQ(ierr);
  }
  ierr = OptionsHasName(pc->prefix,"-pc_ilu_reuse_fill",&flg); CHKERRQ(ierr);
  if (flg) {
    ierr = PCILUSetReuseFill(pc,PETSC_TRUE); CHKERRQ(ierr);
  }
  ierr = OptionsHasName(pc->prefix,"-pc_ilu_reuse_reordering",&flg); CHKERRQ(ierr);
  if (flg) {
    ierr = PCILUSetReuseReordering(pc,PETSC_TRUE); CHKERRQ(ierr);
  }
  ierr = OptionsGetDoubleArray(pc->prefix,"-pc_ilu_use_drop_tolerance",dt,&dtmax,&flg);
         CHKERRQ(ierr);
  if (flg) {
    if (dtmax != 2) {
      SETERRQ(1,0,"Bad args to -pc_ilu_use_drop_tolerance");
    }
    ierr = PCILUSetUseDropTolerance(pc,dt[0],(int)dt[1]); CHKERRQ(ierr);
  }
  ierr = OptionsGetDouble(pc->prefix,"-pc_ilu_fill",&fill,&flg); CHKERRQ(ierr);
  if (flg) {
    ierr = PCILUSetFill(pc,fill); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCPrintHelp_ILU"
static int PCPrintHelp_ILU(PC pc,char *p)
{
  PetscFunctionBegin;
  PetscPrintf(pc->comm," Options for PCILU preconditioner:\n");
  PetscPrintf(pc->comm," -mat_order <name>: ordering to reduce fill",p);
  PetscPrintf(pc->comm," (nd,natural,1wd,rcm,qmd)\n");
  PetscPrintf(pc->comm," %spc_ilu_levels <levels>: levels of fill\n",p);
  PetscPrintf(pc->comm," %spc_ilu_fill <fill>: expected fill in factorization\n",p);
  PetscPrintf(pc->comm," %spc_ilu_in_place: do factorization in place\n",p);
  PetscPrintf(pc->comm," %spc_ilu_factorpointwise: do NOT use block factorization\n",p);
  PetscPrintf(pc->comm," %spc_ilu_use_drop_tolerance <dt,maxrowcount>: \n",p);
  PetscPrintf(pc->comm," %spc_ilu_reuse_reordering:                          \n",p);
  PetscPrintf(pc->comm," %spc_ilu_reuse_fill:                             \n",p);
  PetscPrintf(pc->comm," %spc_ilu_nonzeros_along_diagonal: <tol> changes column ordering\n",p);
  PetscPrintf(pc->comm,"    to reduce the chance of obtaining zero pivot during ILU.\n");
  PetscPrintf(pc->comm,"    If <tol> not given, defaults to 1.e-10.\n"); 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCView_ILU"
static int PCView_ILU(PetscObject obj,Viewer viewer)
{
  PC         pc = (PC)obj;
  FILE       *fd;
  PC_ILU     *ilu = (PC_ILU *) pc->data;
  int        ierr;
  char       *order;
  ViewerType vtype;
 
  PetscFunctionBegin;
  MatReorderingGetName(ilu->ordering,&order);
  ViewerGetType(viewer,&vtype);
  if (vtype  == ASCII_FILE_VIEWER || vtype == ASCII_FILES_VIEWER) {
    ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
    if (ilu->levels == 1)
      PetscFPrintf(pc->comm,fd,"    ILU: %d level of fill\n",ilu->levels);
    else
      PetscFPrintf(pc->comm,fd,"    ILU: %d levels of fill\n",ilu->levels);
    if (ilu->inplace) PetscFPrintf(pc->comm,fd,"         in-place factorization\n");
    else PetscFPrintf(pc->comm,fd,"         out-of-place factorization\n");
    PetscFPrintf(pc->comm,fd,"         matrix ordering: %s\n",order);
  }
  else if (vtype == STRING_VIEWER) {
    ViewerStringSPrintf(viewer," lvls=%d,order=%s",ilu->levels,order);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCSetUp_ILU"
static int PCSetUp_ILU(PC pc)
{
  int         ierr,flg;
  PC_ILU      *ilu = (PC_ILU *) pc->data;

  PetscFunctionBegin;
  if (ilu->inplace) {
    if (!pc->setupcalled) {
      /* In-place factorization only makes sense with the natural ordering,
         so we only need to get the ordering once, even if nonzero structure changes */
      ierr = MatGetReorderingTypeFromOptions(0,&ilu->ordering); CHKERRQ(ierr);
      ierr = MatGetReordering(pc->pmat,ilu->ordering,&ilu->row,&ilu->col); CHKERRQ(ierr);
      if (ilu->row) PLogObjectParent(pc,ilu->row);
      if (ilu->col) PLogObjectParent(pc,ilu->col);
    }
    if (setups[pc->pmat->type]) {
      ierr = (*setups[pc->pmat->type])(pc);
    }
    /* In place ILU only makes sense with fill factor of 1.0 because 
       cannot have levels of fill */
    ierr = MatILUFactor(pc->pmat,ilu->row,ilu->col,1.0,ilu->levels); CHKERRQ(ierr);
    ilu->fact = pc->pmat;
  }
  else if (ilu->usedt) {
    if (!pc->setupcalled) {
      ierr = MatGetReorderingTypeFromOptions(0,&ilu->ordering); CHKERRQ(ierr);
      ierr = MatGetReordering(pc->pmat,ilu->ordering,&ilu->row,&ilu->col);CHKERRQ(ierr);
      if (ilu->row) PLogObjectParent(pc,ilu->row); 
      if (ilu->col) PLogObjectParent(pc,ilu->col);
      ierr = MatILUDTFactor(pc->pmat,ilu->dt,ilu->dtcount,ilu->row,ilu->col,
                                   &ilu->fact);CHKERRQ(ierr);
      PLogObjectParent(pc,ilu->fact);
    } else if (pc->flag != SAME_NONZERO_PATTERN) { 
      ierr = MatDestroy(ilu->fact); CHKERRQ(ierr);
      if (!ilu->reusereordering) {
        if (ilu->row) {ierr = ISDestroy(ilu->row); CHKERRQ(ierr);}
        if (ilu->col) {ierr = ISDestroy(ilu->col); CHKERRQ(ierr);}
        ierr = MatGetReordering(pc->pmat,ilu->ordering,&ilu->row,&ilu->col);CHKERRQ(ierr);
        if (ilu->row) PLogObjectParent(pc,ilu->row);
        if (ilu->col) PLogObjectParent(pc,ilu->col);
      }
      ierr = MatILUDTFactor(pc->pmat,ilu->dt,ilu->dtcount,ilu->row,ilu->col,
                                   &ilu->fact);CHKERRQ(ierr);
      PLogObjectParent(pc,ilu->fact);
    } else if (!ilu->reusefill) { 
      ierr = MatDestroy(ilu->fact); CHKERRQ(ierr);
      ierr = MatILUDTFactor(pc->pmat,ilu->dt,ilu->dtcount,ilu->row,ilu->col,
                                   &ilu->fact);CHKERRQ(ierr);
      PLogObjectParent(pc,ilu->fact);
    } else {
      ierr = MatLUFactorNumeric(pc->pmat,&ilu->fact); CHKERRQ(ierr);
    }
  } else {
    if (!pc->setupcalled) {
      /* first time in so compute reordering and symbolic factorization */
      ierr = MatGetReorderingTypeFromOptions(0,&ilu->ordering); CHKERRQ(ierr);
      ierr = MatGetReordering(pc->pmat,ilu->ordering,&ilu->row,&ilu->col);CHKERRQ(ierr);
      if (ilu->row) PLogObjectParent(pc,ilu->row);
      if (ilu->col) PLogObjectParent(pc,ilu->col);
      /*  Remove zeros along diagonal?     */
      ierr = OptionsHasName(pc->prefix,"-pc_ilu_nonzeros_along_diagonal",&flg);CHKERRQ(ierr);
      if (flg) {
        double ntol = 1.e-10;
        ierr = OptionsGetDouble(pc->prefix,"-pc_ilu_nonzeros_along_diagonal",&ntol,&flg);CHKERRQ(ierr);
        ierr = MatReorderForNonzeroDiagonal(pc->pmat,ntol,ilu->row,ilu->col);CHKERRQ(ierr);
      }
      if (setups[pc->pmat->type]) {
        ierr = (*setups[pc->pmat->type])(pc);
      }
      ierr = MatILUFactorSymbolic(pc->pmat,ilu->row,ilu->col,ilu->fill,ilu->levels,
                                &ilu->fact); CHKERRQ(ierr);
      PLogObjectParent(pc,ilu->fact);
    }
    else if (pc->flag != SAME_NONZERO_PATTERN) { 
      if (!ilu->reusereordering) {
        /* compute a new ordering for the ILU */
        ISDestroy(ilu->row); ISDestroy(ilu->col);
        ierr = MatGetReordering(pc->pmat,ilu->ordering,&ilu->row,&ilu->col); CHKERRQ(ierr);
        if (ilu->row) PLogObjectParent(pc,ilu->row);
        if (ilu->col) PLogObjectParent(pc,ilu->col);
        /*  Remove zeros along diagonal?     */
        ierr = OptionsHasName(pc->prefix,"-pc_ilu_nonzeros_along_diagonal",&flg);CHKERRQ(ierr);
        if (flg) {
          double ntol = 1.e-10;
          ierr = OptionsGetDouble(pc->prefix,"-pc_ilu_nonzeros_along_diagonal",&ntol,&flg);CHKERRQ(ierr);
          ierr = MatReorderForNonzeroDiagonal(pc->pmat,ntol,ilu->row,ilu->col);CHKERRQ(ierr);
        }
      }
      ierr = MatDestroy(ilu->fact); CHKERRQ(ierr);
      ierr = MatILUFactorSymbolic(pc->pmat,ilu->row,ilu->col,ilu->fill,ilu->levels,
                                  &ilu->fact); CHKERRQ(ierr);
      PLogObjectParent(pc,ilu->fact);
    }
    ierr = MatLUFactorNumeric(pc->pmat,&ilu->fact); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCDestroy_ILU"
static int PCDestroy_ILU(PetscObject obj)
{
  PC     pc   = (PC) obj;
  PC_ILU *ilu = (PC_ILU*) pc->data;

  PetscFunctionBegin;
  if (!ilu->inplace && ilu->fact) MatDestroy(ilu->fact);
  if (ilu->row && ilu->col && ilu->row != ilu->col) ISDestroy(ilu->row);
  if (ilu->col) ISDestroy(ilu->col);
  PetscFree(ilu); 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCApply_ILU"
static int PCApply_ILU(PC pc,Vec x,Vec y)
{
  PC_ILU *ilu = (PC_ILU *) pc->data;
  int    ierr;

  PetscFunctionBegin;
  ierr = MatSolve(ilu->fact,x,y); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCGetFactoredMatrix_ILU"
static int PCGetFactoredMatrix_ILU(PC pc,Mat *mat)
{
  PC_ILU *ilu = (PC_ILU *) pc->data;

  PetscFunctionBegin;
  *mat = ilu->fact;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCCreate_ILU"
int PCCreate_ILU(PC pc)
{
  PC_ILU           *ilu = PetscNew(PC_ILU); CHKPTRQ(ilu);

  PetscFunctionBegin;
  PLogObjectMemory(pc,sizeof(PC_ILU));

  ilu->fact             = 0;
  ilu->levels           = 0;
  ilu->fill             = 1.0;
  ilu->col              = 0;
  ilu->row              = 0;
  ilu->inplace          = 0;
  ilu->ordering         = ORDER_NATURAL;
  ilu->reusereordering  = 0;
  ilu->usedt            = 0;
  ilu->dt               = 0.0;
  ilu->dtcount          = 0;
  ilu->reusefill        = 0;
  pc->destroy           = PCDestroy_ILU;
  pc->apply             = PCApply_ILU;
  pc->setup             = PCSetUp_ILU;
  pc->type              = PCILU;
  pc->data              = (void *) ilu;
  pc->setfrom           = PCSetFromOptions_ILU;
  pc->printhelp         = PCPrintHelp_ILU;
  pc->getfactoredmatrix = PCGetFactoredMatrix_ILU;
  pc->view              = PCView_ILU;
  pc->applyrich         = 0;
  PetscFunctionReturn(0);
}
