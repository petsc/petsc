#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: viewregall.c,v 1.4 1999/01/19 16:27:39 bsmith Exp bsmith $";
#endif

#include "src/sys/src/viewer/viewerimpl.h"  /*I "petsc.h" I*/  

EXTERN_C_BEGIN
extern int ViewerCreate_Socket(Viewer);
extern int ViewerCreate_ASCII(Viewer);
extern int ViewerCreate_Binary(Viewer);
extern int ViewerCreate_String(Viewer);
extern int ViewerCreate_Draw(Viewer);
extern int ViewerCreate_AMS(Viewer);
EXTERN_C_END
  
/*
    This is used by ViewerSetType() to make sure that at least one 
    ViewerRegisterAll() is called. In general, if there is more than one
    DLL, then ViewerRegisterAll() may be called several times.
*/
extern int ViewerRegisterAllCalled;

#undef __FUNC__  
#define __FUNC__ "ViewerRegisterAll"
/*@C
  ViewerRegisterAll - Registers all of the graphics methods in the Viewer package.

  Not Collective

.keywords: Viewer, register, all

.seealso:  ViewerRegisterDestroy()
@*/
int ViewerRegisterAll(char *path)
{
  int ierr;

  PetscFunctionBegin;
  ViewerRegisterAllCalled = 1;
  
  ierr = ViewerRegister(ASCII_VIEWER,    path,"ViewerCreate_ASCII",      ViewerCreate_ASCII);CHKERRQ(ierr);
  ierr = ViewerRegister(BINARY_VIEWER,   path,"ViewerCreate_Binary",     ViewerCreate_Binary);CHKERRQ(ierr);
  ierr = ViewerRegister(STRING_VIEWER,   path,"ViewerCreate_String",     ViewerCreate_String);CHKERRQ(ierr);
  ierr = ViewerRegister(DRAW_VIEWER,     path,"ViewerCreate_Draw",       ViewerCreate_Draw);CHKERRQ(ierr);
  ierr = ViewerRegister(SOCKET_VIEWER,   path,"ViewerCreate_Socket",     ViewerCreate_Socket);CHKERRQ(ierr);
#if defined(HAVE_AMS)
  ierr = ViewerRegister(AMS_VIEWER,      path,"ViewerCreate_AMS",        ViewerCreate_AMS);CHKERRQ(ierr); 
#endif
  PetscFunctionReturn(0);
}

