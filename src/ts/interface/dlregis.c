#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dlregis.c,v 1.7 1998/04/22 19:13:34 curfman Exp bsmith $";
#endif

#include "ts.h"


#undef __FUNC__  
#define __FUNC__ "DLLibraryRegister"
/*
  DLLibraryRegister - This function is called when the dynamic library it is in is opened.

  This one registers all the TS methods that are in the basic PETSc libpetscts library.

  Input Parameter:
  path - library path
 */
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

/* --------------------------------------------------------------------------*/
static char *contents = "PETSc timestepping library. Contains:\n\
     Euler\n\
     Backward Euler\n\
     PVODE interface\n";

static char *authors = PETSC_AUTHOR_INFO;
static char *version = PETSC_VERSION_NUMBER;

/* --------------------------------------------------------------------------*/
int DLLibraryInfo(char *path,char *type,char **mess) 
{ 
  if (!PetscStrcmp(type,"Contents"))     *mess = contents;
  else if (!PetscStrcmp(type,"Authors")) *mess = authors;
  else if (!PetscStrcmp(type,"Version")) *mess = version;
  else *mess = 0;

  return 0;
}
