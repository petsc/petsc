/*$Id: viewregall.c,v 1.8 1999/05/12 03:26:24 bsmith Exp bsmith $*/

#include "src/sys/src/viewer/viewerimpl.h"  /*I "petsc.h" I*/  

EXTERN_C_BEGIN
extern int ViewerCreate_Socket(Viewer);
extern int ViewerCreate_ASCII(Viewer);
extern int ViewerCreate_Binary(Viewer);
extern int ViewerCreate_String(Viewer);
extern int ViewerCreate_Draw(Viewer);
extern int ViewerCreate_AMS(Viewer);
EXTERN_C_END
  
#undef __FUNC__  
#define __FUNC__ "ViewerRegisterAll"
/*@C
  ViewerRegisterAll - Registers all of the graphics methods in the Viewer package.

  Not Collective

   Level: developer

.keywords: Viewer, register, all

.seealso:  ViewerRegisterDestroy()
@*/
int ViewerRegisterAll(char *path)
{
  int ierr;

  PetscFunctionBegin;
  
  ierr = ViewerRegister(ASCII_VIEWER,    path,"ViewerCreate_ASCII",      ViewerCreate_ASCII);CHKERRQ(ierr);
  ierr = ViewerRegister(BINARY_VIEWER,   path,"ViewerCreate_Binary",     ViewerCreate_Binary);CHKERRQ(ierr);
  ierr = ViewerRegister(STRING_VIEWER,   path,"ViewerCreate_String",     ViewerCreate_String);CHKERRQ(ierr);
  ierr = ViewerRegister(DRAW_VIEWER,     path,"ViewerCreate_Draw",       ViewerCreate_Draw);CHKERRQ(ierr);
  ierr = ViewerRegister(SOCKET_VIEWER,   path,"ViewerCreate_Socket",     ViewerCreate_Socket);CHKERRQ(ierr);
#if defined(PETSC_HAVE_AMS)
  ierr = ViewerRegister(AMS_VIEWER,      path,"ViewerCreate_AMS",        ViewerCreate_AMS);CHKERRQ(ierr); 
#endif
  PetscFunctionReturn(0);
}

