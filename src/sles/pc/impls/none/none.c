#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: none.c,v 1.20 1998/10/19 22:17:21 bsmith Exp bsmith $";
#endif
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
  ierr = VecCopy(x,y); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "PCCreate_None"
int PCCreate_None(PC pc)
{
  PetscFunctionBegin;
  pc->apply               = PCApply_None;
  pc->applytrans          = PCApply_None;
  pc->destroy             = 0;
  pc->setup               = 0;
  pc->data                = 0;
  pc->view                = 0;
  pc->applysymmetricleft  = PCApply_None;
  pc->applysymmetricright = PCApply_None;
  PetscFunctionReturn(0);
}
EXTERN_C_END
