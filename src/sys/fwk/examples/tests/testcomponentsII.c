#include "petscfwk.h"
EXTERN_C_BEGIN
#undef  __FUNCT__ 
#define __FUNCT__ "PetscFwkComponentConfigureTestIIA"
PetscErrorCode PETSC_DLLEXPORT PetscFwkComponentConfigureTestIIA(PetscFwk fwk, PetscInt state, PetscObject *component) {
  MPI_Comm       comm;
  PetscContainer container;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)fwk, &comm); CHKERRQ(ierr);
  if(!*component) {
    ierr = PetscContainerCreate(comm, &container); CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)container, "TestIIA"); CHKERRQ(ierr);
    *component = (PetscObject)container;
    ierr = PetscPrintf(comm, "%s: created component\n", __FUNCT__); CHKERRQ(ierr);
  }
  else {
    container = *((PetscContainer*)component);
    ierr = PetscPrintf(comm, "%s: configuring to state %d\n", __FUNCT__, state); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}/* PetscFwkComponentConfigureTestIIA() */

#undef  __FUNCT__ 
#define __FUNCT__ "PetscFwkComponentConfigureTestIIB"
PetscErrorCode PETSC_DLLEXPORT PetscFwkComponentConfigureTestIIB(PetscFwk fwk, PetscInt state, PetscObject *component) {
  MPI_Comm       comm;
  PetscContainer container;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)fwk, &comm); CHKERRQ(ierr);
  if(!*component) {
    ierr = PetscContainerCreate(comm, &container); CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)container, "TestIIB"); CHKERRQ(ierr);
    *component = (PetscObject)container;
    ierr = PetscPrintf(comm, "%s: created component\n", __FUNCT__); CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "%s: registering dependency: TestIIB --> TestIIA\n", __FUNCT__); CHKERRQ(ierr);
    ierr = PetscFwkRegisterDependence(fwk, 
                                      "${PETSC_DIR}/${PETSC_ARCH}/lib/libtestcomponents.a:TestIIB",
                                      "${PETSC_DIR}/${PETSC_ARCH}/lib/libtestcomponents:TestIIA"); CHKERRQ(ierr);
  }
  else {
    container = *((PetscContainer*)component);
    ierr = PetscPrintf(comm, "%s: configuring to state %d\n", __FUNCT__, state); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}/* PetscFwkComponentConfigureTestIIB() */

#undef  __FUNCT__ 
#define __FUNCT__ "PetscFwkComponentConfigureTestIIC"
PetscErrorCode PETSC_DLLEXPORT PetscFwkComponentConfigureTestIIC(PetscFwk fwk, PetscInt state, PetscObject *component) {
  MPI_Comm       comm;
  PetscContainer container;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)fwk, &comm); CHKERRQ(ierr);
  if(!*component) {
    ierr = PetscContainerCreate(comm, &container); CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)container, "TestIIB"); CHKERRQ(ierr);
    *component = (PetscObject)container;
    ierr = PetscPrintf(comm, "%s: created component\n", __FUNCT__); CHKERRQ(ierr);
  }
  else {
    container = *((PetscContainer*)component);
    ierr = PetscPrintf(comm, "%s: configuring to state %d\n", __FUNCT__, state); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}/* PetscFwkComponentConfigureTestIIC() */
EXTERN_C_END
