#include <petsc.h>

EXTERN_C_BEGIN

#undef  __FUNCT__ 
#define __FUNCT__ "TestIACall"
PetscErrorCode  TestIACall(PetscShell component, const char* message) {
  MPI_Comm comm = ((PetscObject)component)->comm;
  PetscErrorCode ierr;
  PetscBool  init;
  PetscFunctionBegin;
  ierr = PetscPrintf(comm, "%s: running '%s'\n", __FUNCT__, message); CHKERRQ(ierr);
  ierr = PetscStrcmp(message, "initialize", &init); CHKERRQ(ierr);
  if (init) {
    PetscShell shell;
    ierr = PetscShellGetVisitor(component, &shell); CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "%s: registering dependence: TestIB --> %s\n", __FUNCT__, ((PetscObject)component)->name); CHKERRQ(ierr);
    ierr = PetscShellRegisterDependence(shell, "TestIB", ((PetscObject)component)->name); CHKERRQ(ierr);    
  }
  PetscFunctionReturn(0);
}/* TestIACall() */

#undef  __FUNCT__ 
#define __FUNCT__ "TestIBCall"
PetscErrorCode  TestIBCall(PetscShell component, const char* message) {
  MPI_Comm comm = ((PetscObject)component)->comm;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscPrintf(comm, "%s: running '%s'\n", __FUNCT__, message); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* TestIBCall() */

#undef  __FUNCT__ 
#define __FUNCT__ "TestICCall"
PetscErrorCode  TestICCall(PetscShell component, const char* message) {
  MPI_Comm comm = ((PetscObject)component)->comm;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscPrintf(comm, "%s: running '%s'\n", __FUNCT__, message); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* TestICCall() */

EXTERN_C_END

