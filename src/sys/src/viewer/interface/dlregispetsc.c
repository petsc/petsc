#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dlregispetsc.c,v 1.4 1999/09/20 18:36:40 bsmith Exp bsmith $";
#endif

#include "petsc.h"

#undef __FUNC__  
#define __FUNC__ "DLLibraryRegister_Petsc"
/*
  DLLibraryRegister_Petsc - This function is called when the dynamic library it is in is opened.

  This one registers all the KSP and PC methods that are in the basic PETSc libpetscsles
  library.

  Input Parameter:
  path - library path
 */
int DLLibraryRegister_Petsc(char *path)
{
  int ierr;

  ierr = PetscInitializeNoArguments(); if (ierr) return 1;

  /* this follows the Initialize() to make sure PETSc was setup first */
  PetscFunctionBegin;
  /*
      If we got here then PETSc was properly loaded
  */
  ierr = DrawRegisterAll(path);CHKERRQ(ierr);
  ierr = ViewerRegisterAll(path);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------*/
static char *contents = "PETSc Graphics and Viewer libraries. \n\
     ASCII, Binary, Sockets, X-windows, ...\n";

static char *authors = PETSC_AUTHOR_INFO;
static char *version = PETSC_VERSION_NUMBER;

/* --------------------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "DLLibraryInfo_Petsc"
int DLLibraryInfo_Petsc(char *path,char *type,char **mess) 
{
  int ierr,iscon,isaut,isver;

  PetscFunctionBegin; 

  iscon = !PetscStrcmp(type,"Contents");CHKERRQ(ierr);
  isaut = !PetscStrcmp(type,"Authors");CHKERRQ(ierr);
  isver = !PetscStrcmp(type,"Version");CHKERRQ(ierr);
  if (iscon)      *mess = contents;
  else if (isaut) *mess = authors;
  else if (isver) *mess = version;
  else            *mess = 0;

  PetscFunctionReturn(0);
}










