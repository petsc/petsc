/*$Id: none.c,v 1.23 1999/05/04 20:33:59 balay Exp bsmith $*/
/*
    Identity preconditioner, simply copies vector x to y.
*/
#include "src/sles/pc/pcimpl.h"          /*I "pc.h" I*/

#undef __FUNC__  
#define __FUNC__ "PCApply_None"
int PCApply_None(PC pc,Vec x,Vec y)
{
  int ierr;

  PetscFunctionBegin;
  ierr = VecCopy(x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "PCCreate_None"
int PCCreate_None(PC pc)
{
  PetscFunctionBegin;
  pc->ops->apply               = PCApply_None;
  pc->ops->applytrans          = PCApply_None;
  pc->ops->destroy             = 0;
  pc->ops->setup               = 0;
  pc->ops->view                = 0;
  pc->ops->applysymmetricleft  = PCApply_None;
  pc->ops->applysymmetricright = PCApply_None;

  pc->data                     = 0;
  PetscFunctionReturn(0);
}
EXTERN_C_END
