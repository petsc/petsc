#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: drawregall.c,v 1.6 1999/03/17 23:21:11 bsmith Exp bsmith $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "src/sys/src/draw/drawimpl.h"  /*I "draw.h" I*/

EXTERN_C_BEGIN
extern int DrawCreate_X(Draw);
extern int DrawCreate_Null(Draw);
EXTERN_C_END
  
#undef __FUNC__  
#define __FUNC__ "DrawRegisterAll"
/*@C
  DrawRegisterAll - Registers all of the graphics methods in the Draw package.

  Not Collective

  Level: developer

.keywords: Draw, register, all

.seealso:  DrawRegisterDestroy()
@*/
int DrawRegisterAll(char *path)
{
  int ierr;

  PetscFunctionBegin;
  
#if defined(PETSC_HAVE_X11)
  ierr = DrawRegister(DRAW_X,     path,"DrawCreate_X",     DrawCreate_X);CHKERRQ(ierr);
#endif
  ierr = DrawRegister(DRAW_NULL,  path,"DrawCreate_Null",  DrawCreate_Null);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

