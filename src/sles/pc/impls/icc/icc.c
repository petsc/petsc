#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: icc.c,v 1.58 1999/04/16 16:08:28 bsmith Exp bsmith $ ";
#endif
/*
   Defines a Cholesky factorization preconditioner for any Mat implementation.
  Presently only provided for MPIRowbs format (i.e. BlockSolve).
*/

#include "src/sles/pc/impls/icc/icc.h"   /*I "pc.h" I*/

#undef __FUNC__  
#define __FUNC__ "PCSetup_ICC"
static int PCSetup_ICC(PC pc)
{
  PC_ICC *icc = (PC_ICC *) pc->data;
  IS     perm;
  int    ierr;

  PetscFunctionBegin;
  /* Currently no orderings are supported!
  ierr = MatGetOrdering(pc->pmat,icc->ordering,&perm,&perm); CHKERRQ(ierr); */
  perm = 0;

  if (!pc->setupcalled) {
    ierr = MatIncompleteCholeskyFactorSymbolic(pc->pmat,perm,1.0,icc->levels,&icc->fact);CHKERRQ(ierr);
  } else if (pc->flag != SAME_NONZERO_PATTERN) {
    ierr = MatDestroy(icc->fact); CHKERRQ(ierr);
    ierr = MatIncompleteCholeskyFactorSymbolic(pc->pmat,perm,1.0,icc->levels,&icc->fact);CHKERRQ(ierr);
  }
  ierr = MatCholeskyFactorNumeric(pc->pmat,&icc->fact); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCDestroy_ICC"
static int PCDestroy_ICC(PC pc)
{
  PC_ICC *icc = (PC_ICC *) pc->data;

  PetscFunctionBegin;
  MatDestroy(icc->fact);
  PetscFree(icc);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCApply_ICC"
static int PCApply_ICC(PC pc,Vec x,Vec y)
{
  PC_ICC *icc = (PC_ICC *) pc->data;
  int    ierr;

  PetscFunctionBegin;
  ierr = MatSolve(icc->fact,x,y); CHKERRQ(ierr);
  PetscFunctionReturn(0);  
}

#undef __FUNC__  
#define __FUNC__ "PCApplySymmetricLeft_ICC"
static int PCApplySymmetricLeft_ICC(PC pc,Vec x,Vec y)
{
  int    ierr;
  PC_ICC *icc = (PC_ICC *) pc->data;

  PetscFunctionBegin;
  ierr = MatForwardSolve(icc->fact,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCApplySymmetricRight_ICC"
static int PCApplySymmetricRight_ICC(PC pc,Vec x,Vec y)
{
  int    ierr;
  PC_ICC *icc = (PC_ICC *) pc->data;

  PetscFunctionBegin;
  ierr = MatBackwardSolve(icc->fact,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCPrintHelp_ICC"
static int PCPrintHelp_ICC(PC pc,char *p)
{
  int ierr;

  PetscFunctionBegin;
  ierr = (*PetscHelpPrintf)(pc->comm," Options for PCICC preconditioner:\n");CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(pc->comm,"  %spc_icc_factorpointwise: Do NOT use block factorization \n",p);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(pc->comm,"    (Note: This only applies to the MATMPIROWBS matrix format;\n");CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(pc->comm,"    all others currently only support point factorization.\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCGetFactoredMatrix_ICC"
static int PCGetFactoredMatrix_ICC(PC pc,Mat *mat)
{
  PC_ICC *icc = (PC_ICC *) pc->data;

  PetscFunctionBegin;
  *mat = icc->fact;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCSetFromOptions_ICC"
static int PCSetFromOptions_ICC(PC pc)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "PCCreate_ICC"
int PCCreate_ICC(PC pc)
{
  PC_ICC      *icc = PetscNew(PC_ICC); CHKPTRQ(icc);

  PetscFunctionBegin;
  PLogObjectMemory(pc,sizeof(PC_ICC));

  icc->fact	          = 0;
  icc->ordering           = MATORDERING_ND;
  icc->levels	          = 0;
  icc->implctx            = 0;
  pc->apply	          = PCApply_ICC;
  pc->setup               = PCSetup_ICC;
  pc->destroy	          = PCDestroy_ICC;
  pc->setfromoptions      = PCSetFromOptions_ICC;
  pc->printhelp           = PCPrintHelp_ICC;
  pc->view                = 0;
  pc->getfactoredmatrix   = PCGetFactoredMatrix_ICC;
  pc->data	          = (void *) icc;
  pc->applysymmetricleft  = PCApplySymmetricLeft_ICC;
  pc->applysymmetricright = PCApplySymmetricRight_ICC;
  PetscFunctionReturn(0);
}
EXTERN_C_END


