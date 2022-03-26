
/*
    Identity preconditioner, simply copies vector x to y.
*/
#include <petsc/private/pcimpl.h>          /*I "petscpc.h" I*/

PetscErrorCode PCApply_None(PC pc,Vec x,Vec y)
{
  PetscFunctionBegin;
  PetscCall(VecCopy(x,y));
  PetscFunctionReturn(0);
}

PetscErrorCode PCMatApply_None(PC pc,Mat X,Mat Y)
{
  PetscFunctionBegin;
  PetscCall(MatCopy(X,Y,SAME_NONZERO_PATTERN));
  PetscFunctionReturn(0);
}

/*MC
     PCNONE - This is used when you wish to employ a nonpreconditioned
             Krylov method.

   Level: beginner

  Notes:
    This is implemented by a VecCopy()

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC
M*/

PETSC_EXTERN PetscErrorCode PCCreate_None(PC pc)
{
  PetscFunctionBegin;
  pc->ops->apply               = PCApply_None;
  pc->ops->matapply            = PCMatApply_None;
  pc->ops->applytranspose      = PCApply_None;
  pc->ops->destroy             = NULL;
  pc->ops->setup               = NULL;
  pc->ops->view                = NULL;
  pc->ops->applysymmetricleft  = PCApply_None;
  pc->ops->applysymmetricright = PCApply_None;

  pc->data = NULL;
  PetscFunctionReturn(0);
}
