/*$Id: dlregispetsc.c,v 1.9 1999/11/05 14:43:47 bsmith Exp bsmith $*/

#include "petsc.h"

  
EXTERN_C_BEGIN
#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"DLLibraryRegister" 
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
#define  __FUNC__ /*<a name=""></a>*/"DLLibraryRegister_Petsc" 
int DLLibraryRegister_Petsc(char *path)
{
  int ierr;

  PetscFunctionBegin;
  ierr = DrawRegisterAll(path);CHKERRQ(ierr);
  ierr = ViewerRegisterAll(path);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif







