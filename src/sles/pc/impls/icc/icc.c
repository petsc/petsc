/*$Id: icc.c,v 1.82 2001/08/06 21:16:31 bsmith Exp $*/
/*
   Defines a Cholesky factorization preconditioner for any Mat implementation.
  Presently only provided for MPIRowbs format (i.e. BlockSolve).
*/

#include "src/sles/pc/impls/icc/icc.h"   /*I "petscpc.h" I*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCICCSetMatOrdering_ICC"
int PCICCSetMatOrdering_ICC(PC pc,MatOrderingType ordering)
{
  PC_ICC *dir = (PC_ICC*)pc->data;
  int    ierr;
 
  PetscFunctionBegin;
  ierr = PetscStrfree(dir->ordering);CHKERRQ(ierr);
  ierr = PetscStrallocpy(ordering,&dir->ordering);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCICCSetDamping_ICC"
int PCICCSetDamping_ICC(PC pc,PetscReal damping)
{
  PC_ICC *dir;

  PetscFunctionBegin;
  dir = (PC_ICC*)pc->data;
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
#define __FUNCT__ "PCICCSetSetZeroPivot_ICC"
int PCICCSetZeroPivot_ICC(PC pc,PetscReal z)
{
  PC_ICC *lu;

  PetscFunctionBegin;
  lu                 = (PC_ICC*)pc->data;
  lu->info.zeropivot = z;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCICCSetFill_ICC"
int PCICCSetFill_ICC(PC pc,PetscReal fill)
{
  PC_ICC *dir;

  PetscFunctionBegin;
  dir            = (PC_ICC*)pc->data;
  dir->info.fill = fill;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCICCSetLevels_ICC"
int PCICCSetLevels_ICC(PC pc,int levels)
{
  PC_ICC *icc;

  PetscFunctionBegin;
  icc = (PC_ICC*)pc->data;
  icc->info.levels = levels;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "PCICCSetMatOrdering"
/*@
    PCICCSetMatOrdering - Sets the ordering routine (to reduce fill) to 
    be used it the ICC factorization.

    Collective on PC

    Input Parameters:
+   pc - the preconditioner context
-   ordering - the matrix ordering name, for example, MATORDERING_ND or MATORDERING_RCM

    Options Database Key:
.   -pc_icc_mat_ordering_type <nd,rcm,...> - Sets ordering routine

    Level: intermediate

.seealso: PCLUSetMatOrdering()

.keywords: PC, ICC, set, matrix, reordering

@*/
int PCICCSetMatOrdering(PC pc,MatOrderingType ordering)
{
  int ierr,(*f)(PC,MatOrderingType);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCICCSetMatOrdering_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,ordering);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCICCSetLevels"
/*@
   PCICCSetLevels - Sets the number of levels of fill to use.

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  levels - number of levels of fill

   Options Database Key:
.  -pc_icc_levels <levels> - Sets fill level

   Level: intermediate

   Concepts: ICC^setting levels of fill

@*/
int PCICCSetLevels(PC pc,int levels)
{
  int ierr,(*f)(PC,int);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  if (levels < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"negative levels");
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCICCSetLevels_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,levels);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCICCSetFill"
/*@
   PCICCSetFill - Indicate the amount of fill you expect in the factored matrix,
   where fill = number nonzeros in factor/number nonzeros in original matrix.

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  fill - amount of expected fill

   Options Database Key:
$  -pc_icc_fill <fill>

   Note:
   For sparse matrix factorizations it is difficult to predict how much 
   fill to expect. By running with the option -log_info PETSc will print the 
   actual amount of fill used; allowing you to set the value accurately for
   future runs. But default PETSc uses a value of 1.0

   Level: intermediate

.keywords: PC, set, factorization, direct, fill

.seealso: PCLUSetFill()
@*/
int PCICCSetFill(PC pc,PetscReal fill)
{
  int ierr,(*f)(PC,PetscReal);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  if (fill < 1.0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Fill factor cannot be less than 1.0");
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCICCSetFill_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,fill);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCICCSetDamping"
/*@
   PCICCSetDamping - adds this quantity to the diagonal of the matrix during the 
     ICC numerical factorization

   Collective on PC
   
   Input Parameters:
+  pc - the preconditioner context
-  damping - amount of damping

   Options Database Key:
.  -pc_icc_damping <damping> - Sets damping amount or PETSC_DECIDE for the default

   Note: If 0.0 is given, then no damping is used. If a diagonal element is classified as a zero
         pivot, then the damping is doubled until this is alleviated.

   Level: intermediate

.keywords: PC, set, factorization, direct, fill

.seealso: PCICCSetFill(), PCLUSetDamp()
@*/
int PCICCSetDamping(PC pc,PetscReal damping)
{
  int ierr,(*f)(PC,PetscReal);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCICCSetDamping_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,damping);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCICCSetZeroPivot"
/*@
   PCICCSetZeroPivot - Sets the size at which smaller pivots are declared to be zero

   Collective on PC
   
   Input Parameters:
+  pc - the preconditioner context
-  zero - all pivots smaller than this will be considered zero

   Options Database Key:
.  -pc_ilu_zeropivot <zero> - Sets the zero pivot size

   Level: intermediate

.keywords: PC, set, factorization, direct, fill

.seealso: PCICCSetFill(), PCLUSetDamp(), PCLUSetZeroPivot()
@*/
int PCICCSetZeroPivot(PC pc,PetscReal zero)
{
  int ierr,(*f)(PC,PetscReal);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCICCSetZeroPivot_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,zero);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSetup_ICC"
static int PCSetup_ICC(PC pc)
{
  PC_ICC *icc = (PC_ICC*)pc->data;
  IS     perm,cperm;
  int    ierr;

  PetscFunctionBegin;
  ierr = MatGetOrdering(pc->pmat,icc->ordering,&perm,&cperm);CHKERRQ(ierr);

  if (!pc->setupcalled) {
    ierr = MatICCFactorSymbolic(pc->pmat,perm,&icc->info,&icc->fact);CHKERRQ(ierr);
  } else if (pc->flag != SAME_NONZERO_PATTERN) {
    ierr = MatDestroy(icc->fact);CHKERRQ(ierr);
    ierr = MatICCFactorSymbolic(pc->pmat,perm,&icc->info,&icc->fact);CHKERRQ(ierr);
  }
  ierr = ISDestroy(cperm);CHKERRQ(ierr);
  ierr = ISDestroy(perm);CHKERRQ(ierr);
  ierr = MatCholeskyFactorNumeric(pc->pmat,&icc->fact);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCDestroy_ICC"
static int PCDestroy_ICC(PC pc)
{
  PC_ICC *icc = (PC_ICC*)pc->data;
  int    ierr;

  PetscFunctionBegin;
  if (icc->fact) {ierr = MatDestroy(icc->fact);CHKERRQ(ierr);}
  ierr = PetscStrfree(icc->ordering);CHKERRQ(ierr);
  ierr = PetscFree(icc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCApply_ICC"
static int PCApply_ICC(PC pc,Vec x,Vec y)
{
  PC_ICC *icc = (PC_ICC*)pc->data;
  int    ierr;

  PetscFunctionBegin;
  ierr = MatSolve(icc->fact,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);  
}

#undef __FUNCT__  
#define __FUNCT__ "PCApplySymmetricLeft_ICC"
static int PCApplySymmetricLeft_ICC(PC pc,Vec x,Vec y)
{
  int    ierr;
  PC_ICC *icc = (PC_ICC*)pc->data;

  PetscFunctionBegin;
  ierr = MatForwardSolve(icc->fact,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCApplySymmetricRight_ICC"
static int PCApplySymmetricRight_ICC(PC pc,Vec x,Vec y)
{
  int    ierr;
  PC_ICC *icc = (PC_ICC*)pc->data;

  PetscFunctionBegin;
  ierr = MatBackwardSolve(icc->fact,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCGetFactoredMatrix_ICC"
static int PCGetFactoredMatrix_ICC(PC pc,Mat *mat)
{
  PC_ICC *icc = (PC_ICC*)pc->data;

  PetscFunctionBegin;
  *mat = icc->fact;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSetFromOptions_ICC"
static int PCSetFromOptions_ICC(PC pc)
{
  PC_ICC     *icc = (PC_ICC*)pc->data;
  char       tname[256];
  PetscTruth flg;
  int        ierr;
  PetscFList ordlist;

  PetscFunctionBegin;
  ierr = MatOrderingRegisterAll(PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsHead("ICC Options");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-pc_icc_levels","levels of fill","PCICCSetLevels",icc->info.levels,&icc->info.levels,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-pc_icc_fill","Expected fill in factorization","PCICCSetFill",icc->info.fill,&icc->info.fill,&flg);CHKERRQ(ierr);
    ierr = MatGetOrderingList(&ordlist);CHKERRQ(ierr);
    ierr = PetscOptionsList("-pc_icc_mat_ordering_type","Reorder to reduce nonzeros in ICC","PCICCSetMatOrdering",ordlist,icc->ordering,tname,256,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PCICCSetMatOrdering(pc,tname);CHKERRQ(ierr);
    }
    ierr = PetscOptionsName("-pc_icc_damping","Damping added to diagonal","PCICCSetDamping",&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PCICCSetDamping(pc,(PetscReal) PETSC_DECIDE);CHKERRQ(ierr);
    }
    ierr = PetscOptionsReal("-pc_icc_damping","Damping added to diagonal","PCICCSetDamping",icc->info.damping,&icc->info.damping,0);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-pc_icc_zeropivot","Pivot is considered zero if less than","PCICCSetSetZeroPivot",icc->info.zeropivot,&icc->info.zeropivot,0);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCView_ICC"
static int PCView_ICC(PC pc,PetscViewer viewer)
{
  PC_ICC     *icc = (PC_ICC*)pc->data;
  int        ierr;
  PetscTruth isstring,isascii;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_STRING,&isstring);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    if (icc->info.levels == 1) {
        ierr = PetscViewerASCIIPrintf(viewer,"  ICC: %d level of fill\n",icc->info.levels);CHKERRQ(ierr);
    } else {
        ierr = PetscViewerASCIIPrintf(viewer,"  ICC: %d levels of fill\n",icc->info.levels);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  ICC: max fill ratio allocated %g\n",icc->info.fill);CHKERRQ(ierr);
  } else if (isstring) {
    ierr = PetscViewerStringSPrintf(viewer," lvls=%d",icc->info.levels);CHKERRQ(ierr);CHKERRQ(ierr);
  } else {
    SETERRQ1(1,"Viewer type %s not supported for PCICC",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCreate_ICC"
int PCCreate_ICC(PC pc)
{
  int    ierr;
  PC_ICC *icc;

  PetscFunctionBegin;
  ierr = PetscNew(PC_ICC,&icc);CHKERRQ(ierr);
  PetscLogObjectMemory(pc,sizeof(PC_ICC));

  icc->fact	          = 0;
  ierr = PetscStrallocpy(MATORDERING_NATURAL,&icc->ordering);CHKERRQ(ierr);
  icc->info.levels	  = 0;
  icc->info.fill          = 1.0;
  icc->implctx            = 0;

  icc->info.dtcol         = PETSC_DEFAULT;
  icc->info.damping       = 0.0;
  icc->info.zeropivot     = 1.e-12;
  pc->data	          = (void*)icc;

  pc->ops->apply	       = PCApply_ICC;
  pc->ops->setup               = PCSetup_ICC;
  pc->ops->destroy	       = PCDestroy_ICC;
  pc->ops->setfromoptions      = PCSetFromOptions_ICC;
  pc->ops->view                = PCView_ICC;
  pc->ops->getfactoredmatrix   = PCGetFactoredMatrix_ICC;
  pc->ops->applysymmetricleft  = PCApplySymmetricLeft_ICC;
  pc->ops->applysymmetricright = PCApplySymmetricRight_ICC;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCICCSetLevels_C","PCICCSetLevels_ICC",
                    PCICCSetLevels_ICC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCICCSetFill_C","PCICCSetFill_ICC",
                    PCICCSetFill_ICC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCICCSetDamping_C","PCICCSetDamping_ICC",
                    PCICCSetDamping_ICC);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCICCSetMatOrdering_C","PCICCSetMatOrdering_ICC",
                    PCICCSetMatOrdering_ICC);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END


