#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dlregis.c,v 1.11 1999/05/04 20:34:35 balay Exp bsmith $";
#endif

#include "sles.h"

EXTERN_C_BEGIN
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

  PetscFunctionBegin;
  /*
      If we got here then PETSc was properly loaded
  */
  ierr = KSPRegisterAll(path);CHKERRQ(ierr);
  ierr = PCRegisterAll(path);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

/* --------------------------------------------------------------------------*/
static char *contents = "PETSc Krylov subspace method and preconditioner library.\n\
     GMRES, PCG, Bi-CG-stab, ...\n\
     Jacobi, ILU, Block Jacobi, LU, Additive Schwarz, ...\n";

static char *authors = PETSC_AUTHOR_INFO;
static char *version = PETSC_VERSION_NUMBER;

/* --------------------------------------------------------------------------*/
EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "DLLibraryInfo"
int DLLibraryInfo(char *path,char *type,char **mess) 
{ 
  PetscFunctionBegin;
  if (!PetscStrcmp(type,"Contents"))     *mess = contents;
  else if (!PetscStrcmp(type,"Authors")) *mess = authors;
  else if (!PetscStrcmp(type,"Version")) *mess = version;
  else *mess = 0;
  PetscFunctionReturn(0);
}
EXTERN_C_END
