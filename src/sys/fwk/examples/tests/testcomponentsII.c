#include <petsc.h>
EXTERN_C_BEGIN

#undef  __FUNCT__ 
#define __FUNCT__ "TestIIACall"
PetscErrorCode  TestIIACall(PetscFwk component, const char* message) {
  MPI_Comm comm = ((PetscObject)component)->comm;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscPrintf(comm, "%s: running '%s'\n", __FUNCT__, message); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* TestIIACall() */

#undef  __FUNCT__ 
#define __FUNCT__ "TestIIBCall"
PetscErrorCode  TestIIBCall(PetscFwk component, const char* message) {
  MPI_Comm comm = ((PetscObject)component)->comm;
  PetscErrorCode ierr;
  PetscBool  init;
  PetscFunctionBegin;
  ierr = PetscPrintf(comm, "%s: running '%s'\n", __FUNCT__, message); CHKERRQ(ierr);
  ierr = PetscStrcmp(message, "initialize", &init); CHKERRQ(ierr);
  if(init) {
    PetscFwk fwk;
    ierr = PetscFwkGetParent(component, &fwk); CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "%s: registering dependence: %s --> TestIIA\n", __FUNCT__, ((PetscObject)component)->name); CHKERRQ(ierr);
    ierr = PetscFwkRegisterDependence(fwk, ((PetscObject)component)->name, "TestIIA"); CHKERRQ(ierr);    
  }
  PetscFunctionReturn(0);
}/* TestIIBCall() */

#undef  __FUNCT__ 
#define __FUNCT__ "TestIICCall"
PetscErrorCode  TestIICCall(PetscFwk component, const char* message) {
  MPI_Comm comm = ((PetscObject)component)->comm;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscPrintf(comm, "%s: running '%s'\n", __FUNCT__, message); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* TestIICCall() */

EXTERN_C_END
