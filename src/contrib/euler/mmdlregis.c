#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dlregis.c,v 1.8 1998/04/22 19:13:41 curfman Exp $";
#endif

#include "sles.h"

#undef __FUNC__  
#define __FUNC__ "DLLibraryRegister"
/*
  DLLibraryRegister - This function is called when the dynamic library it is in is opened.

  This one registers all the MM methods.

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
  ierr = MMRegisterAll(path); CHKERRQ(ierr);
  return 0;
}

/* --------------------------------------------------------------------------*/
static char *contents = "Multi-model library. Contains:\n\
     Euler, Full Potential, Hybrid, ...\n";

static char *authors = PETSC_AUTHOR_INFO;
static char *version = PETSC_VERSION_NUMBER;

/* --------------------------------------------------------------------------*/
char *DLLibraryInfo(char *path,char *type) 
{ 
  char *mess = contents;

  if (!PetscStrcmp(type,"Contents"))     mess = contents;
  else if (!PetscStrcmp(type,"Authors")) mess = authors;
  else if (!PetscStrcmp(type,"Version")) mess = version;
  else mess = 0;

  return mess;
}

