#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dlregis.c,v 1.1 1999/04/02 00:11:24 bsmith Exp bsmith $";
#endif

#include "mat.h"

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "DLLibraryRegister"
/*
  DLLibraryRegister - This function is called when the dynamic library it is in is opened.

  This one registers all the matrix partitioners that are in the basic PETSc libpetscmat
  library.

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
  ierr = MatPartitioningRegisterAll(path); CHKERRQ(ierr);
  return 0;
}
EXTERN_C_END

/* --------------------------------------------------------------------------*/
static char *contents = "PETSc matrix library. \n Partitioners ";

static char *authors = PETSC_AUTHOR_INFO;
static char *version = PETSC_VERSION_NUMBER;

/* --------------------------------------------------------------------------*/
EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "DLLibraryInfo"
int DLLibraryInfo(char *path,char *type,char **mess) 
{ 
  if (!PetscStrcmp(type,"Contents"))     *mess = contents;
  else if (!PetscStrcmp(type,"Authors")) *mess = authors;
  else if (!PetscStrcmp(type,"Version")) *mess = version;
  else *mess = 0;

  return 0;
}
EXTERN_C_END


