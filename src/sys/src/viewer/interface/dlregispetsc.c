#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dlregispetsc.c,v 1.6 1999/10/13 20:36:28 bsmith Exp bsmith $";
#endif

#include "petsc.h"

  
EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "DLLibraryRegister"
/*
  DLLibraryRegister - This function is called when the dynamic library it is in is opened.

  This one registers all the draw and viewer objects.

  Input Parameter:
  path - library path
 */
int DLLibraryRegister(char *path)
{
  int ierr;

  ierr = PetscInitializeNoArguments(); if (ierr) return 1;

  /* this follows the Initialize() to make sure PETSc was setup first */
  PetscFunctionBegin;
  /*
      If we got here then PETSc was properly loaded
  */
  ierr = DrawRegisterAll(path);CHKERRQ(ierr);
  ierr = ViewerRegisterAll(path);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

/* --------------------------------------------------------------------------*/
static char *contents = "PETSc Graphics and Viewer libraries. \n\
     ASCII, Binary, Sockets, X-windows, ...\n";

#include "src/sys/src/utils/dlregis.h"

#if !defined(PETSC_USE_DYNAMIC_LIBRARIES)
#undef __FUNC__  
#define __FUNC__ "DLLibraryRegister_Petsc"
int DLLibraryRegister_Petsc(char *path)
{
  int ierr;

  PetscFunctionBegin;
  ierr = DLLibraryRegister(path);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif







