
#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dlregis.c,v 1.6 1999/09/20 19:53:16 bsmith Exp bsmith $";
#endif

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

static char *authors = PETSC_AUTHOR_INFO;
static char *version = PETSC_VERSION_NUMBER;

/* --------------------------------------------------------------------------*/
EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "DLLibraryInfo"
int DLLibraryInfo(char *path,char *type,char **mess) 
{ 
  int iscon,isaut,isver;

  PetscFunctionBegin; 

  iscon = !PetscStrcmp(type,"Contents");
  isaut = !PetscStrcmp(type,"Authors");
  isver = !PetscStrcmp(type,"Version");
  if (iscon)      *mess = contents;
  else if (isaut) *mess = authors;
  else if (isver) *mess = version;
  else            *mess = 0;
  PetscFunctionReturn(0);
}
EXTERN_C_END
