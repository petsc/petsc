/*$Id: ilu.c,v 1.139 1999/12/17 04:24:59 bsmith Exp bsmith $*/
/*
   Defines a ILU factorization preconditioner for any Mat implementation
*/
#include "src/sles/pc/pcimpl.h"                 /*I "pc.h"  I*/
#include "src/sles/pc/impls/ilu/ilu.h"
#include "src/mat/matimpl.h"

/* ------------------------------------------------------------------------------------------*/
EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "PCILUSetUseDropTolerance_ILU"
int PCILUSetUseDropTolerance_ILU(PC pc,double dt,double dtcol,int dtcount)
{
  PC_ILU *ilu;

  PetscFunctionBegin;
  ilu = (PC_ILU *) pc->data;
  ilu->usedt         = PETSC_TRUE;
  ilu->info.dt       = dt;
  ilu->info.dtcol    = dtcol;
  ilu->info.dtcount  = dtcount;
  ilu->info.fill     = PETSC_DEFAULT;
  PetscFunctionReturn(0);
}  
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "PCILUSetFill_ILU"
int PCILUSetFill_ILU(PC pc,double fill)
{
  PC_ILU *dir;

  PetscFunctionBegin;
  dir            = (PC_ILU *) pc->data;
  dir->info.fill = fill;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "PCILUSetMatOrdering_ILU"
int PCILUSetMatOrdering_ILU(PC pc, MatOrderingType ordering)
{
  PC_ILU *dir = (PC_ILU *) pc->data;
  int    ierr;
 
  PetscFunctionBegin;
  ierr = PetscStrfree(dir->ordering);CHKERRQ(ierr);
  ierr = PetscStrallocpy(ordering,&dir->ordering);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "PCILUSetReuseOrdering_ILU"
int PCILUSetReuseOrdering_ILU(PC pc,PetscTruth flag)
{
  PC_ILU *ilu;

  PetscFunctionBegin;
  ilu                = (PC_ILU *) pc->data;
  ilu->reuseordering = (int) flag;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "PCILUSetReuseFill_ILU"
int PCILUSetReuseFill_ILU(PC pc,PetscTruth flag)
{
  PC_ILU *ilu;

  PetscFunctionBegin;
  ilu = (PC_ILU *) pc->data;
  ilu->reusefill = (int) flag;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "PCILUSetLevels_ILU"
int PCILUSetLevels_ILU(PC pc,int levels)
{
  PC_ILU *ilu;

  PetscFunctionBegin;
  ilu = (PC_ILU *) pc->data;
  ilu->info.levels = levels;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "PCILUSetUseInPlace_ILU"
int PCILUSetUseInPlace_ILU(PC pc)
{
  PC_ILU *dir;

  PetscFunctionBegin;
  dir          = (PC_ILU *) pc->data;
  dir->inplace = 1;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "PCILUSetAllowDiagonalFill"
int PCILUSetAllowDiagonalFill_ILU(PC pc)
{
  PC_ILU *dir;

  PetscFunctionBegin;
  dir                 = (PC_ILU *) pc->data;
  dir->info.diagonal_fill = 1;
  PetscFunctionReturn(0);
}
EXTERN_C_END

/* ------------------------------------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "PCILUSetUseDropTolerance"
/*@
   PCILUSetUseDropTolerance - The preconditioner will use an ILU 
   based on a drop tolerance.

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
.  dt - the drop tolerance
.  dtcol - tolerance for column pivot
-  dtcount - the max number of nonzeros allowed in a row

   Options Database Key:
.  -pc_ilu_use_drop_tolerance <dt,dtcount> - Sets drop tolerance

   Level: intermediate

    Notes:
      This uses the iludt() code of Saad's SPARSKIT package

.keywords: PC, levels, reordering, factorization, incomplete, ILU
@*/
int PCILUSetUseDropTolerance(PC pc,double dt,double dtcol,int dtcount)
{
  int ierr, (*f)(PC,double,double,int);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCILUSetUseDropTolerance_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,dt,dtcol,dtcount);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}  

#undef __FUNC__  
#define __FUNC__ "PCILUSetFill"
/*@
   PCILUSetFill - Indicate the amount of fill you expect in the factored matrix,
   where fill = number nonzeros in factor/number nonzeros in original matrix.

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  fill - amount of expected fill

   Options Database Key:
$  -pc_ilu_fill <fill>

   Note:
   For sparse matrix factorizations it is difficult to predict how much 
   fill to expect. By running with the option -log_info PETSc will print the 
   actual amount of fill used; allowing you to set the value accurately for
   future runs. But default PETSc uses a value of 1.0

   Level: intermediate

.keywords: PC, set, factorization, direct, fill

.seealso: PCLUSetFill()
@*/
int PCILUSetFill(PC pc,double fill)
{
  int ierr, (*f)(PC,double);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  if (fill < 1.0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,1,"Fill factor cannot be less than 1.0");
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCILUSetFill_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,fill);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCILUSetMatOrdering"
/*@
    PCILUSetMatOrdering - Sets the ordering routine (to reduce fill) to 
    be used it the ILU factorization.

    Collective on PC

    Input Parameters:
+   pc - the preconditioner context
-   ordering - the matrix ordering name, for example, MATORDERING_ND or MATORDERING_RCM

    Options Database Key:
.   -pc_ilu_mat_ordering_type <nd,rcm,...> - Sets ordering routine

    Level: intermediate

.seealso: PCLUSetMatOrdering()

.keywords: PC, ILU, set, matrix, reordering

@*/
int PCILUSetMatOrdering(PC pc, MatOrderingType ordering)
{
  int ierr, (*f)(PC,MatOrderingType);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCILUSetMatOrdering_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,ordering);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCILUSetReuseOrdering"
/*@
   PCILUSetReuseOrdering - When similar matrices are factored, this
   causes the ordering computed in the first factor to be used for all
   following factors; applies to both fill and drop tolerance ILUs.

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  flag - PETSC_TRUE to reuse else PETSC_FALSE

   Options Database Key:
.  -pc_ilu_reuse_ordering - Activate PCILUSetReuseOrdering()

   Level: intermediate

.keywords: PC, levels, reordering, factorization, incomplete, ILU

.seealso: PCILUSetReuseFill(), PCLUSetReuseOrdering(), PCLUSetReuseFill()
@*/
int PCILUSetReuseOrdering(PC pc,PetscTruth flag)
{
  int ierr, (*f)(PC,PetscTruth);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCILUSetReuseOrdering_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,flag);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCILUSetReuseFill"
/*@
   PCILUSetReuseFill - When matrices with same nonzero structure are ILUDT factored,
   this causes later ones to use the fill computed in the initial factorization.

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  flag - PETSC_TRUE to reuse else PETSC_FALSE

   Options Database Key:
.  -pc_ilu_reuse_fill - Activates PCILUSetReuseFill()

   Level: intermediate

.keywords: PC, levels, reordering, factorization, incomplete, ILU

.seealso: PCILUSetReuseOrdering(), PCLUSetReuseOrdering(), PCLUSetReuseFill()
@*/
int PCILUSetReuseFill(PC pc,PetscTruth flag)
{
  int ierr, (*f)(PC,PetscTruth);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCILUSetReuseFill_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,flag);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCILUSetLevels"
/*@
   PCILUSetLevels - Sets the number of levels of fill to use.

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  levels - number of levels of fill

   Options Database Key:
.  -pc_ilu_levels <levels> - Sets fill level

   Level: intermediate

.keywords: PC, levels, fill, factorization, incomplete, ILU
@*/
int PCILUSetLevels(PC pc,int levels)
{
  int ierr, (*f)(PC,int);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  if (levels < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"negative levels");
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCILUSetLevels_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,levels);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCILUSetAllowDiagonalFill"
/*@
   PCILUSetAllowDiagonalFill - Causes all diagonal matrix entries to be 
   treated as level 0 fill even if there is no non-zero location.

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context

   Options Database Key:
.  -pc_ilu_diagonal_fill

   Notes:
   Does not apply with 0 fill.

   Level: intermediate

.keywords: PC, levels, fill, factorization, incomplete, ILU
@*/
int PCILUSetAllowDiagonalFill(PC pc)
{
  int ierr, (*f)(PC);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCILUSetAllowDiagonalFill_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCILUSetUseInPlace"
/*@
   PCILUSetUseInPlace - Tells the system to do an in-place incomplete factorization.
   Collective on PC

   Input Parameters:
.  pc - the preconditioner context

   Options Database Key:
.  -pc_ilu_in_place - Activates in-place factorization

   Notes:
   PCILUSetUseInPlace() is intended for use with matrix-free variants of
   Krylov methods, or when a different matrices are employed for the linear
   system and preconditioner, or with ASM preconditioning.  Do NOT use 
   this option if the linear system
   matrix also serves as the preconditioning matrix, since the factored
   matrix would then overwrite the original matrix. 

   Only works well with ILU(0).

   Level: intermediate

.keywords: PC, set, factorization, inplace, in-place, ILU

.seealso:  PCLUSetUseInPlace()
@*/
int PCILUSetUseInPlace(PC pc)
{
  int ierr, (*f)(PC);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCILUSetUseInPlace_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------------------------*/

#undef __FUNC__  
#define __FUNC__ "PCSetFromOptions_ILU"
static int PCSetFromOptions_ILU(PC pc)
{
  int        levels,ierr,dtmax = 3;
  PetscTruth flg;
  double     dt[3],fill;
  char       tname[256];

  PetscFunctionBegin;
  ierr = OptionsGetInt(pc->prefix,"-pc_ilu_levels",&levels,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PCILUSetLevels(pc,levels);CHKERRQ(ierr);
  }
  ierr = OptionsHasName(pc->prefix,"-pc_ilu_in_place",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PCILUSetUseInPlace(pc);CHKERRQ(ierr);
  }
  ierr = OptionsHasName(pc->prefix,"-pc_ilu_diagonal_fill",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PCILUSetAllowDiagonalFill(pc);CHKERRQ(ierr);
  }
  ierr = OptionsHasName(pc->prefix,"-pc_ilu_reuse_fill",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PCILUSetReuseFill(pc,PETSC_TRUE);CHKERRQ(ierr);
  }
  ierr = OptionsHasName(pc->prefix,"-pc_ilu_reuse_ordering",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PCILUSetReuseOrdering(pc,PETSC_TRUE);CHKERRQ(ierr);
  }
  dt[0] = PETSC_DEFAULT;
  dt[1] = PETSC_DEFAULT;
  dt[2] = PETSC_DEFAULT;
  ierr = OptionsGetDoubleArray(pc->prefix,"-pc_ilu_use_drop_tolerance",dt,&dtmax,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PCILUSetUseDropTolerance(pc,dt[0],dt[1],(int)dt[2]);CHKERRQ(ierr);
  }
  ierr = OptionsGetDouble(pc->prefix,"-pc_ilu_fill",&fill,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PCILUSetFill(pc,fill);CHKERRQ(ierr);
  }
  ierr = OptionsGetString(pc->prefix,"-pc_ilu_mat_ordering_type",tname,256,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PCILUSetMatOrdering(pc,tname);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCPrintHelp_ILU"
static int PCPrintHelp_ILU(PC pc,char *p)
{
  int ierr;

  PetscFunctionBegin;
  ierr = (*PetscHelpPrintf)(pc->comm," Options for PCILU preconditioner:\n");CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(pc->comm," -pc_ilu_mat_ordering_type <name>: ordering to reduce fill",p);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(pc->comm," (nd,natural,1wd,rcm,qmd)\n");CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(pc->comm," %spc_ilu_levels <levels>: levels of fill\n",p);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(pc->comm," %spc_ilu_fill <fill>: expected fill in factorization\n",p);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(pc->comm," %spc_ilu_in_place: do factorization in place\n",p);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(pc->comm," %spc_ilu_factorpointwise: do NOT use block factorization\n",p);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(pc->comm," %spc_ilu_use_drop_tolerance <dt,dtcol,maxrowcount>:\n",p);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(pc->comm," The ILUDT code is based on Yousef Saad's SPARSEKIT2 code\n",p);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(pc->comm," %spc_ilu_reuse_ordering:                          \n",p);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(pc->comm," %spc_ilu_reuse_fill:                             \n",p);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(pc->comm," %spc_ilu_nonzeros_along_diagonal: <tol> changes column ordering\n",p);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(pc->comm,"    to reduce the chance of obtaining zero pivot during ILU.\n");CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(pc->comm,"    If <tol> not given, defaults to 1.e-10.\n");CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCView_ILU"
static int PCView_ILU(PC pc,Viewer viewer)
{
  PC_ILU     *ilu = (PC_ILU *) pc->data;
  int        ierr;
  PetscTruth isstring,isascii;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,STRING_VIEWER,&isstring);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,ASCII_VIEWER,&isascii);CHKERRQ(ierr);
  if (isascii) {
    if (ilu->usedt) {
        ierr = ViewerASCIIPrintf(viewer,"  ILU: drop tolerance %g\n",ilu->info.dt);CHKERRQ(ierr);
        ierr = ViewerASCIIPrintf(viewer,"  ILU: max nonzeros per row %d\n",(int)ilu->info.dtcount);CHKERRQ(ierr);
        ierr = ViewerASCIIPrintf(viewer,"  ILU: column permutation tolerance %g\n",ilu->info.dtcol);CHKERRQ(ierr);
    } else if (ilu->info.levels == 1) {
        ierr = ViewerASCIIPrintf(viewer,"  ILU: %d level of fill\n",(int)ilu->info.levels);CHKERRQ(ierr);
    } else {
        ierr = ViewerASCIIPrintf(viewer,"  ILU: %d levels of fill\n",(int)ilu->info.levels);CHKERRQ(ierr);
    }
    ierr = ViewerASCIIPrintf(viewer,"  ILU: max fill ratio allocated %g\n",ilu->info.fill);CHKERRQ(ierr);
    if (ilu->inplace) {ierr = ViewerASCIIPrintf(viewer,"       in-place factorization\n");CHKERRQ(ierr);}
    else              {ierr = ViewerASCIIPrintf(viewer,"       out-of-place factorization\n");CHKERRQ(ierr);}
    ierr = ViewerASCIIPrintf(viewer,"       matrix ordering: %s\n",ilu->ordering);CHKERRQ(ierr);
    if (ilu->reusefill)     {ierr = ViewerASCIIPrintf(viewer,"       Reusing fill from past factorization\n");CHKERRQ(ierr);}
    if (ilu->reuseordering) {ierr = ViewerASCIIPrintf(viewer,"       Reusing reordering from past factorization\n");CHKERRQ(ierr);}
    ierr = ViewerASCIIPrintf(viewer,"       Factored matrix follows\n");CHKERRQ(ierr);
    ierr = ViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = ViewerPushFormat(viewer,VIEWER_FORMAT_ASCII_INFO,0);CHKERRQ(ierr);
    ierr = MatView(ilu->fact,viewer);CHKERRQ(ierr);
    ierr = ViewerPopFormat(viewer);CHKERRQ(ierr);
    ierr = ViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  } else if (isstring) {
    ierr = ViewerStringSPrintf(viewer," lvls=%d,order=%s",ilu->info.levels,ilu->ordering);CHKERRQ(ierr);CHKERRQ(ierr);
  } else {
    SETERRQ1(1,1,"Viewer type %s not supported for PCILU",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCSetUp_ILU"
static int PCSetUp_ILU(PC pc)
{
  int        ierr;
  PetscTruth flg;
  PC_ILU     *ilu = (PC_ILU *) pc->data;

  PetscFunctionBegin;
  if (ilu->inplace) {
    if (!pc->setupcalled) {

      /* In-place factorization only makes sense with the natural ordering,
         so we only need to get the ordering once, even if nonzero structure changes */
      ierr = MatGetOrdering(pc->pmat,ilu->ordering,&ilu->row,&ilu->col);CHKERRQ(ierr);
      if (ilu->row) PLogObjectParent(pc,ilu->row);
      if (ilu->col) PLogObjectParent(pc,ilu->col);
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
      if (ilu->row) PLogObjectParent(pc,ilu->row); 
      if (ilu->col) PLogObjectParent(pc,ilu->col);
      ierr = MatILUDTFactor(pc->pmat,&ilu->info,ilu->row,ilu->col,&ilu->fact);CHKERRQ(ierr);
      PLogObjectParent(pc,ilu->fact);
    } else if (pc->flag != SAME_NONZERO_PATTERN) { 
      ierr = MatDestroy(ilu->fact);CHKERRQ(ierr);
      if (!ilu->reuseordering) {
        if (ilu->row) {ierr = ISDestroy(ilu->row);CHKERRQ(ierr);}
        if (ilu->col) {ierr = ISDestroy(ilu->col);CHKERRQ(ierr);}
        ierr = MatGetOrdering(pc->pmat,ilu->ordering,&ilu->row,&ilu->col);CHKERRQ(ierr);
        if (ilu->row) PLogObjectParent(pc,ilu->row);
        if (ilu->col) PLogObjectParent(pc,ilu->col);
      }
      ierr = MatILUDTFactor(pc->pmat,&ilu->info,ilu->row,ilu->col,&ilu->fact);CHKERRQ(ierr);
      PLogObjectParent(pc,ilu->fact);
    } else if (!ilu->reusefill) { 
      ierr = MatDestroy(ilu->fact);CHKERRQ(ierr);
      ierr = MatILUDTFactor(pc->pmat,&ilu->info,ilu->row,ilu->col,&ilu->fact);CHKERRQ(ierr);
      PLogObjectParent(pc,ilu->fact);
    } else {
      ierr = MatLUFactorNumeric(pc->pmat,&ilu->fact);CHKERRQ(ierr);
    }
  } else {
    if (!pc->setupcalled) {
      /* first time in so compute reordering and symbolic factorization */
      ierr = MatGetOrdering(pc->pmat,ilu->ordering,&ilu->row,&ilu->col);CHKERRQ(ierr);
      if (ilu->row) PLogObjectParent(pc,ilu->row);
      if (ilu->col) PLogObjectParent(pc,ilu->col);
      /*  Remove zeros along diagonal?     */
      ierr = OptionsHasName(pc->prefix,"-pc_ilu_nonzeros_along_diagonal",&flg);CHKERRQ(ierr);
      if (flg) {
        double ntol = 1.e-10;
        ierr = OptionsGetDouble(pc->prefix,"-pc_ilu_nonzeros_along_diagonal",&ntol,PETSC_NULL);CHKERRQ(ierr);
        ierr = MatReorderForNonzeroDiagonal(pc->pmat,ntol,ilu->row,ilu->col);CHKERRQ(ierr);
      }

      ierr = MatILUFactorSymbolic(pc->pmat,ilu->row,ilu->col,&ilu->info,&ilu->fact);CHKERRQ(ierr);
      PLogObjectParent(pc,ilu->fact);
    } else if (pc->flag != SAME_NONZERO_PATTERN) { 
      if (!ilu->reuseordering) {
        /* compute a new ordering for the ILU */
        ierr = ISDestroy(ilu->row);CHKERRQ(ierr);
        ierr = ISDestroy(ilu->col);CHKERRQ(ierr);
        ierr = MatGetOrdering(pc->pmat,ilu->ordering,&ilu->row,&ilu->col);CHKERRQ(ierr);
        if (ilu->row) PLogObjectParent(pc,ilu->row);
        if (ilu->col) PLogObjectParent(pc,ilu->col);
        /*  Remove zeros along diagonal?     */
        ierr = OptionsHasName(pc->prefix,"-pc_ilu_nonzeros_along_diagonal",&flg);CHKERRQ(ierr);
        if (flg) {
          double ntol = 1.e-10;
          ierr = OptionsGetDouble(pc->prefix,"-pc_ilu_nonzeros_along_diagonal",&ntol,PETSC_NULL);CHKERRQ(ierr);
          ierr = MatReorderForNonzeroDiagonal(pc->pmat,ntol,ilu->row,ilu->col);CHKERRQ(ierr);
        }
      }
      ierr = MatDestroy(ilu->fact);CHKERRQ(ierr);
      ierr = MatILUFactorSymbolic(pc->pmat,ilu->row,ilu->col,&ilu->info,&ilu->fact);CHKERRQ(ierr);
      PLogObjectParent(pc,ilu->fact);
    }
    ierr = MatLUFactorNumeric(pc->pmat,&ilu->fact);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCDestroy_ILU"
static int PCDestroy_ILU(PC pc)
{
  PC_ILU *ilu = (PC_ILU*) pc->data;
  int    ierr;

  PetscFunctionBegin;
  if (!ilu->inplace && ilu->fact) {ierr = MatDestroy(ilu->fact);CHKERRQ(ierr);}
  if (ilu->row && ilu->col && ilu->row != ilu->col) {ierr = ISDestroy(ilu->row);CHKERRQ(ierr);}
  if (ilu->col) {ierr = ISDestroy(ilu->col);CHKERRQ(ierr);}
  ierr = PetscStrfree(ilu->ordering);CHKERRQ(ierr);
  ierr = PetscFree(ilu);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCApply_ILU"
static int PCApply_ILU(PC pc,Vec x,Vec y)
{
  PC_ILU *ilu = (PC_ILU *) pc->data;
  int    ierr;

  PetscFunctionBegin;
  ierr = MatSolve(ilu->fact,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCApplyTranspose_ILU"
static int PCApplyTranspose_ILU(PC pc,Vec x,Vec y)
{
  PC_ILU *ilu = (PC_ILU *) pc->data;
  int    ierr;

  PetscFunctionBegin;
  ierr = MatSolveTranspose(ilu->fact,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCGetFactoredMatrix_ILU"
static int PCGetFactoredMatrix_ILU(PC pc,Mat *mat)
{
  PC_ILU *ilu = (PC_ILU *) pc->data;

  PetscFunctionBegin;
  if (!ilu->fact) SETERRQ(1,1,"Matrix not yet factored; call after SLESSetUp() or PCSetUp()");
  *mat = ilu->fact;
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "PCCreate_ILU"
int PCCreate_ILU(PC pc)
{
  int    ierr;
  PC_ILU *ilu = PetscNew(PC_ILU);CHKPTRQ(ilu);

  PetscFunctionBegin;
  PLogObjectMemory(pc,sizeof(PC_ILU));

  ilu->fact               = 0;
  ilu->info.levels        = 0;
  ilu->info.fill          = 1.0; 
  ilu->col                = 0;
  ilu->row                = 0;
  ilu->inplace            = 0;
  ierr = PetscStrallocpy(MATORDERING_NATURAL,&ilu->ordering);CHKERRQ(ierr);
  ilu->reuseordering      = 0;
  ilu->usedt              = PETSC_FALSE;
  ilu->info.dt            = PETSC_DEFAULT;
  ilu->info.dtcount       = PETSC_DEFAULT;
  ilu->info.dtcol         = PETSC_DEFAULT;
  ilu->reusefill          = PETSC_DEFAULT;
  ilu->info.diagonal_fill = 0;
  pc->data                = (void *) ilu;

  pc->ops->destroy           = PCDestroy_ILU;
  pc->ops->apply             = PCApply_ILU;
  pc->ops->applytranspose    = PCApplyTranspose_ILU;
  pc->ops->setup             = PCSetUp_ILU;
  pc->ops->setfromoptions    = PCSetFromOptions_ILU;
  pc->ops->printhelp         = PCPrintHelp_ILU;
  pc->ops->getfactoredmatrix = PCGetFactoredMatrix_ILU;
  pc->ops->view              = PCView_ILU;
  pc->ops->applyrichardson   = 0;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCILUSetUseDropTolerance_C","PCILUSetUseDropTolerance_ILU",
                    (void*)PCILUSetUseDropTolerance_ILU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCILUSetFill_C","PCILUSetFill_ILU",
                    (void*)PCILUSetFill_ILU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCILUSetMatOrdering_C","PCILUSetMatOrdering_ILU",
                    (void*)PCILUSetMatOrdering_ILU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCILUSetReuseOrdering_C","PCILUSetReuseOrdering_ILU",
                    (void*)PCILUSetReuseOrdering_ILU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCILUSetReuseFill_C","PCILUSetReuseFill_ILU",
                    (void*)PCILUSetReuseFill_ILU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCILUSetLevels_C","PCILUSetLevels_ILU",
                    (void*)PCILUSetLevels_ILU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCILUSetUseInPlace_C","PCILUSetUseInPlace_ILU",
                    (void*)PCILUSetUseInPlace_ILU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCILUSetAllowDiagonalFill_C","PCILUSetAllowDiagonalFill_ILU",
                    (void*)PCILUSetAllowDiagonalFill_ILU);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END
