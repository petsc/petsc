#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dlregis.c,v 1.1 1998/01/15 02:45:15 bsmith Exp bsmith $";
#endif

#include "sles.h"

#undef __FUNC__  
#define __FUNC__ "DLRegisterLibrary"
/*
  DLRegisterLibrary - This function is called when the dynamic library it is in is opened.

       This one registers all the KSP methods that are in the basic PETSc libpetscsles
    library.

@*/
int DLRegisterLibrary()
{
  int ierr;

  PetscFunctionBegin;
  ierr = KSPRegisterAll(); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
