/*$Id: jacobi.c,v 1.75 2001/08/07 03:03:32 balay Exp $*/

#include "src/ksp/pc/pcimpl.h"   /*I "petscpc.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "PCApply_Mat"
static int PCApply_Mat(PC pc,Vec x,Vec y)
{
  int       ierr;

  PetscFunctionBegin;
  ierr = MatMult(pc->pmat,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCApplyTranspose_Mat"
static int PCApplyTranspose_Mat(PC pc,Vec x,Vec y)
{
  int       ierr;

  PetscFunctionBegin;
  ierr = MatMultTranspose(pc->pmat,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCDestroy_Mat"
static int PCDestroy_Mat(PC pc)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

/*MC
     PCMAT - A preconditioner obtained by multiplying by the preconditioner matrix supplied
             in PCSetOperators() or KSPSetOperators()

   Notes:  This one is a little strange. One rarely has an explict matrix that approximates the
         inverse of the matrix they wish to solve for.

   Level: intermediate

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC,
           PCSHELL

M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCreate_Mat"
int PCCreate_Mat(PC pc)
{
  PetscFunctionBegin;
  pc->ops->apply               = PCApply_Mat;
  pc->ops->applytranspose      = PCApplyTranspose_Mat;
  pc->ops->setup               = 0;
  pc->ops->destroy             = PCDestroy_Mat;
  pc->ops->setfromoptions      = 0;
  pc->ops->view                = 0;
  pc->ops->applyrichardson     = 0;
  pc->ops->applysymmetricleft  = 0;
  pc->ops->applysymmetricright = 0;
  PetscFunctionReturn(0);
}
EXTERN_C_END

