#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dlregis.c,v 1.3 1998/03/06 00:17:24 bsmith Exp bsmith $";
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

  ierr = PetscInitializeNoArguments(); if (ierr) return 1;

  /*
      If we got here then PETSc was properly loaded
  */
  ierr = TSRegisterAll(path); CHKERRQ(ierr);
  return(0);
}
