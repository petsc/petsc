#include <petsc.h>
EXTERN_C_BEGIN

#undef  __FUNCT__ 
#define __FUNCT__ "TestIIACall"
PetscErrorCode  TestIIACall(PetscShell component, const char* message) {
  MPI_Comm comm = ((PetscObject)component)->comm;
  PetscErrorCode ierr;
  PetscBool  init;
  PetscFunctionBegin;
  ierr = PetscPrintf(comm, "%s: running '%s'\n", __FUNCT__, message); CHKERRQ(ierr);
  ierr = PetscStrcmp(message, "initialize", &init); CHKERRQ(ierr);
  if(init) {
    PetscShell shell;
    ierr = PetscShellGetVisitor(component, &shell); CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "%s: registering dependence: TestIIB --> %s\n", __FUNCT__, ((PetscObject)component)->name); CHKERRQ(ierr);
    ierr = PetscShellRegisterDependence(shell, "TestIIB", ((PetscObject)component)->name); CHKERRQ(ierr);    
  }
  PetscFunctionReturn(0);
}/* TestIIACall() */

#undef  __FUNCT__ 
#define __FUNCT__ "TestIIBCall"
PetscErrorCode  TestIIBCall(PetscShell component, const char* message) {
  MPI_Comm comm = ((PetscObject)component)->comm;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscPrintf(comm, "%s: running '%s'\n", __FUNCT__, message); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* TestIIBCall() */

#undef  __FUNCT__ 
#define __FUNCT__ "TestIICCall"
PetscErrorCode  TestIICCall(PetscShell component, const char* message) {
  MPI_Comm comm = ((PetscObject)component)->comm;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscPrintf(comm, "%s: running '%s'\n", __FUNCT__, message); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* TestIICCall() */

EXTERN_C_END
