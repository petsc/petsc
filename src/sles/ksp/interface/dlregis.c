#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dlregis.c,v 1.4 1998/03/20 22:46:50 bsmith Exp curfman $";
#endif

#include "sles.h"

#undef __FUNC__  
#define __FUNC__ "DLLibraryRegister"
/*
  DLLibraryRegister - This function is called when the dynamic library it is in is opened.

  This one registers all the KSP and PC methods that are in the basic PETSc libpetscsles
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
  ierr = KSPRegisterAll(path); CHKERRQ(ierr);
  ierr = PCRegisterAll(path); CHKERRQ(ierr);
  return 0;
}

/* --------------------------------------------------------------------------*/
static char *contents = "PETSc Krylov Subspace and Preconditioner library\n\
  Contains:\n\
     GMRES, PCG, Bi-CG-stab, ...\n\
     Jacobi, ILU, Block Jacobi, LU, Additive Schwarz, ...\n";

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

