#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dlregis.c,v 1.1 1998/02/14 02:33:16 bsmith Exp bsmith $";
#endif

#include "snes.h"

#undef __FUNC__  
#define __FUNC__ "DLLibraryRegister"
/*
  DLLibraryRegister - This function is called when the dynamic library it is in is opened.

       This one registers all the KSP methods that are in the basic PETSc libpetscsles
    library.

@*/
int DLLibraryRegister(char *path)
{
  int ierr;

  PetscFunctionBegin;
  ierr = SNESRegisterAll(path); CHKERRQ(ierr);
  PetscFunctionReturn(0);
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

