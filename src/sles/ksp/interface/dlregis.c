/*$Id: dlregis.c,v 1.16 2000/04/09 04:37:45 bsmith Exp bsmith $*/

#include "sles.h"

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"DLLibraryRegister"
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

#include "src/sys/src/utils/dlregis.h"
