/*$Id: dlregis.c,v 1.10 2000/05/05 22:16:35 balay Exp bsmith $*/

#include "petscmat.h"

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "PetscDLLibraryRegister"
/*
  PetscDLLibraryRegister - This function is called when the dynamic library it is in is opened.

  This one registers all the matrix partitioners that are in the basic PETSc libpetscmat
  library.

  Input Parameter:
  path - library path
 */
int PetscDLLibraryRegister(char *path)
{
  int ierr;

  ierr = PetscInitializeNoArguments(); if (ierr) return 1;
  PetscFunctionBegin;
  /*
      If we got here then PETSc was properly loaded
  */
  ierr = MatPartitioningRegisterAll(path);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

/* --------------------------------------------------------------------------*/
static char *contents = "PETSc matrix library. \n Partitioners ";

#include "src/sys/src/utils/dlregis.h"


