/*$Id: dlregispetsc.c,v 1.14 2001/03/23 23:20:05 balay Exp $*/

#include "petsc.h"
#ifdef PETSC_HAVE_PLAPACK
  #include "PLA.h"
#endif

#undef __FUNCT__  
#define __FUNCT__ "PetscPLAPACKInitializePackage" 
/*@C
  PetscPLAPACKInitializePackage - This function initializes everything in the Petsc interface to PLAPACK. It is
  called from PetscDLLibraryRegister() when using dynamic libraries, and on the call to PetscInitialize()
  when using static libraries.

  Input Parameter:
  path - The dynamic library path, or PETSC_NULL

  Level: developer

.keywords: Petsc, initialize, package, PLAPACK
.seealso: PetscInitializePackage(), PetscInitialize()
@*/
int PetscPLAPACKInitializePackage(char *path) {
#ifdef PETSC_HAVE_PLAPACK
  MPI_Comm comm;
  int      initPLA;
  int      ierr;

  PetscFunctionBegin;
  ierr = PLA_Initialized(&initPLA);                                                                       CHKERRQ(ierr);
  if (!initPLA) {
    ierr = PLA_Comm_1D_to_2D_ratio(PETSC_COMM_WORLD, 1.0, &comm);                                         CHKERRQ(ierr);
    ierr = PLA_Init(comm);                                                                                CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
#else
  PetscFunctionBegin;
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__  
#define __FUNCT__ "PetscPLAPACKFinalizePackage" 
/*@C
  PetscPLAPACKFinalizePackage - This function destroys everything in the Petsc interface to PLAPACK. It is
  called from PetscFinalize().

  Level: developer

.keywords: Petsc, destroy, package, PLAPACK
.seealso: PetscFinalize()
@*/
int PetscPLAPACKFinalizePackage(void) {
#ifdef PETSC_HAVE_PLAPACK
  int ierr;

  PetscFunctionBegin;
  ierr = PLA_Finalize();                                                                                  CHKERRQ(ierr);
  PetscFunctionReturn(0);
#else
  PetscFunctionBegin;
  PetscFunctionReturn(0);
#endif
}
