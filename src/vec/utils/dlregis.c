/*$Id: dlregis.c,v 1.12 2000/05/05 22:14:53 balay Exp bsmith $*/

#include "petscvec.h"

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "PetscDLLibraryRegister"
/*
  PetscDLLibraryRegister - This function is called when the dynamic library it is in is opened.

  This one registers all the vector types that are in the basic PETSc libpetscvec
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
  ierr = VecRegisterAll(path);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

/* --------------------------------------------------------------------------*/
static char *contents = "PETSc vector library. \n\
     PETSc#VecSeq, PETSc#VecMPI, PETSc#VecShared ...\n";

#include "src/sys/src/utils/dlregis.h"

