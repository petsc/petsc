#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dlregis.c,v 1.2 1998/01/17 17:38:11 bsmith Exp bsmith $";
#endif

#include "ts.h"

#undef __FUNC__  
#define __FUNC__ "DLLibraryRegister"
/*
  DLLibraryRegister - This function is called when the dynamic library it is in is opened.

       This one registers all the TS methods that are in the basic PETSc libpetscts
    library.

@*/
int DLLibraryRegister(char *path)
{
  int ierr;

  PetscFunctionBegin;
  ierr = TSRegisterAll(path); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
