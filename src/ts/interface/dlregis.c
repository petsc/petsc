#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dlregis.c,v 1.1 1998/01/15 22:44:51 bsmith Exp bsmith $";
#endif

#include "ts.h"

#undef __FUNC__  
#define __FUNC__ "DLRegisterLibrary"
/*
  DLRegisterLibrary - This function is called when the dynamic library it is in is opened.

       This one registers all the TS methods that are in the basic PETSc libpetscts
    library.

@*/
int DLRegisterLibrary()
{
  int ierr;

  PetscFunctionBegin;
  ierr = TSRegisterAll(); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
