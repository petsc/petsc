/*$Id: dlinegw.c,v 1.30 2000/09/28 23:02:25 bsmith Exp bsmith $*/
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "src/sys/src/draw/drawimpl.h"  /*I "petscdraw.h" I*/

#undef __FUNC__  
#define __FUNC__ /*<a name="DrawLineGetWidth"></a>*/"DrawLineGetWidth" 
/*@
   DrawLineGetWidth - Gets the line width for future draws.  The width is
   relative to the user coordinates of the window; 0.0 denotes the natural
   width; 1.0 denotes the interior viewport. 

   Not collective

   Input Parameter:
.  draw - the drawing context

   Output Parameter:
.  width - the width in user coordinates

   Level: advanced

   Notes:
   Not currently implemented.

   Concepts: line^width

.seealso:  DrawLineSetWidth()
@*/
int DrawLineGetWidth(Draw draw,PetscReal *width)
{
  int        ierr;
  PetscTruth isdrawnull;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  ierr = PetscTypeCompare((PetscObject)draw,DRAW_NULL,&isdrawnull);CHKERRQ(ierr);
  if (isdrawnull) PetscFunctionReturn(0);
  if (!draw->ops->linegetwidth) SETERRQ(PETSC_ERR_SUP,0);
  ierr = (*draw->ops->linegetwidth)(draw,width);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

