/*$Id: dlregis.c,v 1.7 1999/10/13 20:37:01 bsmith Exp bsmith $*/

#include "vec.h"

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "DLLibraryRegister"
/*
  DLLibraryRegister - This function is called when the dynamic library it is in is opened.

  This one registers all the vector types that are in the basic PETSc libpetscvec
  library.

  Input Parameter:
  path - library path
 */
int DLLibraryRegister(char *path)
{
  int ierr;

  ierr = PetscInitializeNoArguments(); if (ierr) return 1;

  PetscFunctionBegin;
  /*
      If we got here then PETSc was properly loaded
  */
  ierr = VecRegisterAll(path);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

/* --------------------------------------------------------------------------*/
static char *contents = "PETSc vector library. \n\
     PETSc#VecSeq, PETSc#VecMPI, PETSc#VecShared ...\n";

#include "src/sys/src/utils/dlregis.h"

