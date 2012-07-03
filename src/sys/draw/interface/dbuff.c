
/*
       Provides the calling sequences for all the basic PetscDraw routines.
*/
#include <../src/sys/draw/drawimpl.h>  /*I "petscdraw.h" I*/

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawIsNull" 
/*@
   PetscDrawIsNull - Returns PETSC_TRUE if draw is a null draw object.

   Not collective

   Input Parameter:
.  draw - the draw context

   Output Parameter:
.  yes - PETSC_TRUE if it is a null draw object; otherwise PETSC_FALSE

   Level: advanced

@*/
PetscErrorCode  PetscDrawIsNull(PetscDraw draw,PetscBool  *yes)
{
  PetscErrorCode ierr;
  PetscBool  isdrawnull;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  PetscValidIntPointer(yes,2);
  ierr = PetscObjectTypeCompare((PetscObject)draw,PETSC_DRAW_NULL,&isdrawnull);CHKERRQ(ierr);
  if (isdrawnull) *yes = PETSC_TRUE;
  else            *yes = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawSetDoubleBuffer" 
/*@
   PetscDrawSetDoubleBuffer - Sets a window to be double buffered. 

   Logically Collective on PetscDraw

   Input Parameter:
.  draw - the drawing context

   Level: intermediate

   Concepts: drawing^double buffer
   Concepts: graphics^double buffer
   Concepts: double buffer

@*/
PetscErrorCode  PetscDrawSetDoubleBuffer(PetscDraw draw)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  if (draw->ops->setdoublebuffer) {
    ierr = (*draw->ops->setdoublebuffer)(draw);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
