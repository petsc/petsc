#include "petsc.h"
EXTERN_C_BEGIN

#undef  __FUNCT__ 
#define __FUNCT__ "TestIIACall"
PetscErrorCode PETSC_DLLEXPORT TestIIACall(PetscFwk component, const char* message) {
  MPI_Comm comm = ((PetscObject)component)->comm;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscPrintf(comm, "%s: running '%s'\n", __FUNCT__, message); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* TestIIACall() */

#undef  __FUNCT__ 
#define __FUNCT__ "TestIIBCall"
PetscErrorCode PETSC_DLLEXPORT TestIIBCall(PetscFwk component, const char* message) {
  MPI_Comm comm = ((PetscObject)component)->comm;
  PetscErrorCode ierr;
  PetscTruth init;
  PetscFunctionBegin;
  ierr = PetscPrintf(comm, "%s: running '%s'\n", __FUNCT__, message); CHKERRQ(ierr);
  ierr = PetscStrcmp(message, "initialize", &init); CHKERRQ(ierr);
  if(init) {
    PetscFwk fwk;
    ierr = PetscObjectQuery((PetscObject)component, "visitor", (PetscObject*)(&fwk)); CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "%s: registering dependence: %s --> TestIIA\n", __FUNCT__, ((PetscObject)component)->name); CHKERRQ(ierr);
    ierr = PetscFwkRegisterDependence(fwk, ((PetscObject)component)->name, "TestIIA"); CHKERRQ(ierr);    
  }
  PetscFunctionReturn(0);
}/* TestIIBCall() */

#undef  __FUNCT__ 
#define __FUNCT__ "TestIICCall"
PetscErrorCode PETSC_DLLEXPORT TestIICCall(PetscFwk component, const char* message) {
  MPI_Comm comm = ((PetscObject)component)->comm;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscPrintf(comm, "%s: running '%s'\n", __FUNCT__, message); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* TestIICCall() */

EXTERN_C_END
