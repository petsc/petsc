/*$Id: icc.c,v 1.68 2000/05/05 22:17:16 balay Exp bsmith $*/
/*
   Defines a Cholesky factorization preconditioner for any Mat implementation.
  Presently only provided for MPIRowbs format (i.e. BlockSolve).
*/

#include "src/sles/pc/impls/icc/icc.h"   /*I "petscpc.h" I*/

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PCSetup_ICC"
static int PCSetup_ICC(PC pc)
{
  PC_ICC *icc = (PC_ICC*)pc->data;
  IS     perm;
  int    ierr;

  PetscFunctionBegin;
  /* Currently no orderings are supported!
  ierr = MatGetOrdering(pc->pmat,icc->ordering,&perm,&perm);CHKERRQ(ierr); */
  perm = 0;

  if (!pc->setupcalled) {
    ierr = MatIncompleteCholeskyFactorSymbolic(pc->pmat,perm,1.0,icc->levels,&icc->fact);CHKERRQ(ierr);
  } else if (pc->flag != SAME_NONZERO_PATTERN) {
    ierr = MatDestroy(icc->fact);CHKERRQ(ierr);
    ierr = MatIncompleteCholeskyFactorSymbolic(pc->pmat,perm,1.0,icc->levels,&icc->fact);CHKERRQ(ierr);
  }
  ierr = MatCholeskyFactorNumeric(pc->pmat,&icc->fact);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PCDestroy_ICC"
static int PCDestroy_ICC(PC pc)
{
  PC_ICC *icc = (PC_ICC*)pc->data;
  int    ierr;

  PetscFunctionBegin;
  if (icc->fact) {ierr = MatDestroy(icc->fact);CHKERRQ(ierr);}
  ierr = PetscFree(icc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PCApply_ICC"
static int PCApply_ICC(PC pc,Vec x,Vec y)
{
  PC_ICC *icc = (PC_ICC*)pc->data;
  int    ierr;

  PetscFunctionBegin;
  ierr = MatSolve(icc->fact,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);  
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PCApplySymmetricLeft_ICC"
static int PCApplySymmetricLeft_ICC(PC pc,Vec x,Vec y)
{
  int    ierr;
  PC_ICC *icc = (PC_ICC*)pc->data;

  PetscFunctionBegin;
  ierr = MatForwardSolve(icc->fact,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PCApplySymmetricRight_ICC"
static int PCApplySymmetricRight_ICC(PC pc,Vec x,Vec y)
{
  int    ierr;
  PC_ICC *icc = (PC_ICC*)pc->data;

  PetscFunctionBegin;
  ierr = MatBackwardSolve(icc->fact,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PCGetFactoredMatrix_ICC"
static int PCGetFactoredMatrix_ICC(PC pc,Mat *mat)
{
  PC_ICC *icc = (PC_ICC*)pc->data;

  PetscFunctionBegin;
  *mat = icc->fact;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PCSetFromOptions_ICC"
static int PCSetFromOptions_ICC(PC pc)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PCCreate_ICC"
int PCCreate_ICC(PC pc)
{
  PC_ICC      *icc = PetscNew(PC_ICC);CHKPTRQ(icc);

  PetscFunctionBegin;
  PLogObjectMemory(pc,sizeof(PC_ICC));

  icc->fact	          = 0;
  icc->ordering           = MATORDERING_ND;
  icc->levels	          = 0;
  icc->implctx            = 0;
  pc->data	          = (void*)icc;

  pc->ops->apply	       = PCApply_ICC;
  pc->ops->setup               = PCSetup_ICC;
  pc->ops->destroy	       = PCDestroy_ICC;
  pc->ops->setfromoptions      = PCSetFromOptions_ICC;
  pc->ops->view                = 0;
  pc->ops->getfactoredmatrix   = PCGetFactoredMatrix_ICC;
  pc->ops->applysymmetricleft  = PCApplySymmetricLeft_ICC;
  pc->ops->applysymmetricright = PCApplySymmetricRight_ICC;
  PetscFunctionReturn(0);
}
EXTERN_C_END


