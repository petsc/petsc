/*$Id: drawregall.c,v 1.15 2000/06/07 17:36:20 balay Exp balay $*/
/*
       Provides the calling sequences for all the basic Draw routines.
*/
#include "src/sys/src/draw/drawimpl.h"  /*I "petscdraw.h" I*/

EXTERN_C_BEGIN
EXTERN int DrawCreate_X(Draw);
EXTERN int DrawCreate_PS(Draw);
EXTERN int DrawCreate_Null(Draw);
EXTERN int DrawCreate_Win32(Draw);
EXTERN_C_END
  
#undef __FUNC__  
#define __FUNC__ /*<a name="DrawRegisterAll"></a>*/"DrawRegisterAll" 
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
  ierr = DrawRegisterDynamic(DRAW_X,     path,"DrawCreate_X",     DrawCreate_X);CHKERRQ(ierr);
#elif defined (PARCH_win32)
  ierr = DrawRegisterDynamic(DRAW_WIN32, path,"DrawCreate_Win32", DrawCreate_Win32);CHKERRQ(ierr);
#endif
  ierr = DrawRegisterDynamic(DRAW_NULL,  path,"DrawCreate_Null",  DrawCreate_Null);CHKERRQ(ierr);
  ierr = DrawRegisterDynamic(DRAW_PS,    path,"DrawCreate_PS",    DrawCreate_PS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

