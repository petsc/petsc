
/*
    Identity preconditioner, simply copies vector x to y.
*/
#include <petsc-private/pcimpl.h>          /*I "petscpc.h" I*/

#undef __FUNCT__
#define __FUNCT__ "PCApply_None"
PetscErrorCode PCApply_None(PC pc,Vec x,Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCopy(x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
     PCNONE - This is used when you wish to employ a nonpreconditioned
             Krylov method.

   Level: beginner

  Concepts: preconditioners

  Notes: This is implemented by a VecCopy()

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC
M*/

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PCCreate_None"
PetscErrorCode  PCCreate_None(PC pc)
{
  PetscFunctionBegin;
  pc->ops->apply               = PCApply_None;
  pc->ops->applytranspose      = PCApply_None;
  pc->ops->destroy             = 0;
  pc->ops->setup               = 0;
  pc->ops->view                = 0;
  pc->ops->applysymmetricleft  = PCApply_None;
  pc->ops->applysymmetricright = PCApply_None;

  pc->data                     = 0;
  PetscFunctionReturn(0);
}
EXTERN_C_END
