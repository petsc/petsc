#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: none.c,v 1.18 1998/03/06 00:13:33 bsmith Exp bsmith $";
#endif
/*
    Identity preconditioner, simply copies vector x to y.
*/
#include "src/pc/pcimpl.h"          /*I "pc.h" I*/

#undef __FUNC__  
#define __FUNC__ "PCApply_None"
int PCApply_None(PC pc,Vec x,Vec y)
{
  int ierr;

  PetscFunctionBegin;
  ierr = VecCopy(x,y); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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
