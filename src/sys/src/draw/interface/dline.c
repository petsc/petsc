/*$Id: dline.c,v 1.28 2000/07/10 03:38:37 bsmith Exp bsmith $*/
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "src/sys/src/draw/drawimpl.h"  /*I "petscdraw.h" I*/
  
#undef __FUNC__  
#define __FUNC__ /*<a name="DrawLine"></a>*/"DrawLine" 
/*@
   DrawLine - Draws a line onto a drawable.

   Not collective

   Input Parameters:
+  draw - the drawing context
.  xl,yl,xr,yr - the coordinates of the line endpoints
-  cl - the colors of the endpoints

   Level: beginner

   Concepts: line^drawing
   Concepts: drawing^line

@*/
int DrawLine(Draw draw,PetscReal xl,PetscReal yl,PetscReal xr,PetscReal yr,int cl)
{
  int        ierr;
  PetscTruth isdrawnull;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  ierr = PetscTypeCompare((PetscObject)draw,DRAW_NULL,&isdrawnull);CHKERRQ(ierr);
  if (isdrawnull) PetscFunctionReturn(0);
  ierr = (*draw->ops->line)(draw,xl,yl,xr,yr,cl);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="DrawIsNull"></a>*/"DrawIsNull" 
/*@
   DrawIsNull - Returns PETSC_TRUE if draw is a null draw object.

   Not collective

   Input Parameter:
.  draw - the draw context

   Output Parameter:
.  yes - PETSC_TRUE if it is a null draw object; otherwise PETSC_FALSE

   Level: advanced

@*/
int DrawIsNull(Draw draw,PetscTruth *yes)
{
  int        ierr;
  PetscTruth isdrawnull;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,DRAW_COOKIE);
  ierr = PetscTypeCompare((PetscObject)draw,DRAW_NULL,&isdrawnull);CHKERRQ(ierr);
  if (isdrawnull) *yes = PETSC_TRUE;
  else            *yes = PETSC_FALSE;
  PetscFunctionReturn(0);
}
