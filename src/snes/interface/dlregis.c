#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dlregis.c,v 1.3 1998/03/20 22:52:20 bsmith Exp curfman $";
#endif

#include "snes.h"

#undef __FUNC__  
#define __FUNC__ "DLLibraryRegister"
/*
  DLLibraryRegister - This function is called when the dynamic library it is in is opened.

  This registers all of the SNES methods that are in the basic PETSc libpetscsnes library.

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
  ierr = SNESRegisterAll(path); CHKERRQ(ierr);
  return 0;
}

/* --------------------------------------------------------------------------*/
static char *contents = "PETSc nonlinear solver library\n\
  Contains:\n\
     line search\n\
     trust region\n";

static char *authors = "The PETSc Team\n\
  Satish Balay, Bill Gropp, Lois Curfman McInnes, Barry Smith\n";

static char *version = "2.0.21\n";

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

