#include "petsc.h"

EXTERN_C_BEGIN

#undef  __FUNCT__ 
#define __FUNCT__ "TestIACall"
PetscErrorCode  TestIACall(PetscFwk component, const char* message) {
  MPI_Comm comm = ((PetscObject)component)->comm;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscPrintf(comm, "%s: running '%s'\n", __FUNCT__, message); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* TestIACall() */

#undef  __FUNCT__ 
#define __FUNCT__ "TestIBCall"
PetscErrorCode  TestIBCall(PetscFwk component, const char* message) {
  MPI_Comm comm = ((PetscObject)component)->comm;
  PetscErrorCode ierr;
  PetscBool  init;
  PetscFunctionBegin;
  ierr = PetscPrintf(comm, "%s: running '%s'\n", __FUNCT__, message); CHKERRQ(ierr);
  ierr = PetscStrcmp(message, "initialize", &init); CHKERRQ(ierr);
  if(init) {
    PetscFwk fwk;
    ierr = PetscFwkGetParent(component, &fwk); CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "%s: registering dependence: %s --> TestIA\n", __FUNCT__, ((PetscObject)component)->name); CHKERRQ(ierr);
    ierr = PetscFwkRegisterDependence(fwk, ((PetscObject)component)->name, "TestIA"); CHKERRQ(ierr);    
  }
  PetscFunctionReturn(0);
}/* TestIBCall() */

#undef  __FUNCT__ 
#define __FUNCT__ "TestICCall"
PetscErrorCode  TestICCall(PetscFwk component, const char* message) {
  MPI_Comm comm = ((PetscObject)component)->comm;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscPrintf(comm, "%s: running '%s'\n", __FUNCT__, message); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* TestICCall() */

EXTERN_C_END

