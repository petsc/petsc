#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dlregis.c,v 1.2 1998/01/17 17:36:08 bsmith Exp bsmith $";
#endif

#include "sles.h"

#undef __FUNC__  
#define __FUNC__ "DLLibraryRegister"
/*
  DLLibraryRegister - This function is called when the dynamic library it is in is opened.

       This one registers all the KSP and PC methods that are in the basic PETSc libpetscsles
    library.

@*/
int DLLibraryRegister(char *path)
{
  int ierr;

  PetscFunctionBegin;
  ierr = KSPRegisterAll(path); CHKERRQ(ierr);
  ierr = PCRegisterAll(path); CHKERRQ(ierr);
  PetscFunctionReturn(0);
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

