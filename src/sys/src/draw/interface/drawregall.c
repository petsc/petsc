#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: drawregall.c,v 1.1 1999/01/11 04:51:26 bsmith Exp bsmith $";
#endif
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "src/draw/drawimpl.h"  /*I "draw.h" I*/

EXTERN_C_BEGIN
extern int DrawCreate_X(Draw);
extern int DrawCreate_Null(Draw);
extern int DrawCreate_VRML(Draw);
EXTERN_C_END
  
/*
    This is used by DrawSetType() to make sure that at least one 
    DrawRegisterAll() is called. In general, if there is more than one
    DLL, then DrawRegisterAll() may be called several times.
*/
extern int DrawRegisterAllCalled;

#undef __FUNC__  
#define __FUNC__ "DrawRegisterAll"
/*@C
  DrawRegisterAll - Registers all of the graphics methods in the Draw package.

  Not Collective

.keywords: Draw, register, all

.seealso:  DrawRegisterDestroy()
@*/
int DrawRegisterAll(char *path)
{
  int ierr;

  PetscFunctionBegin;
  DrawRegisterAllCalled = 1;
  
#if defined(HAVE_X11)
  ierr = DrawRegister(DRAW_X,     path,"DrawCreate_X",     DrawCreate_X);CHKERRQ(ierr);
#endif
  ierr = DrawRegister(DRAW_NULL,  path,"DrawCreate_Null",  DrawCreate_Null);CHKERRQ(ierr);
  ierr = DrawRegister(DRAW_VRML,  path,"DrawCreate_VRML",  DrawCreate_VRML);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

