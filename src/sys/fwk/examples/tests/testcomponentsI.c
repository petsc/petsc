#include "petscfwk.h"

EXTERN_C_BEGIN
#undef  __FUNCT__ 
#define __FUNCT__ "PetscFwkComponentConfigureTestIA"
PetscErrorCode PETSC_DLLEXPORT PetscFwkComponentConfigureTestIA(PetscFwk fwk, PetscInt state, PetscObject *component) {
  MPI_Comm       comm;
  PetscContainer container;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)fwk, &comm); CHKERRQ(ierr);
  if(!*component) {
    ierr = PetscContainerCreate(comm, &container); CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)container, "TestIA"); CHKERRQ(ierr);
    *component = (PetscObject)container;
    ierr = PetscPrintf(comm, "%s: created component\n", __FUNCT__); CHKERRQ(ierr);
  }
  else {
    container = *((PetscContainer*)component);
    ierr = PetscPrintf(comm, "%s: configuring to state %d\n", __FUNCT__, state); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}/* PetscFwkComponentConfigureTestIA() */

#undef  __FUNCT__ 
#define __FUNCT__ "PetscFwkComponentConfigureTestIB"
PetscErrorCode PETSC_DLLEXPORT PetscFwkComponentConfigureTestIB(PetscFwk fwk, PetscInt state, PetscObject *component) {
  MPI_Comm       comm;
  PetscContainer container;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)fwk, &comm); CHKERRQ(ierr);
  if(!*component) {
    ierr = PetscContainerCreate(comm, &container); CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)container, "TestIB"); CHKERRQ(ierr);
    *component = (PetscObject)container;
    ierr = PetscPrintf(comm, "%s: created component\n", __FUNCT__); CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "%s: registering dependence: TestIB --> TestIA\n", __FUNCT__); CHKERRQ(ierr);
    ierr = PetscFwkRegisterDependence(fwk, 
                                      "${PETSC_DIR}/${PETSC_ARCH}/lib/libtestcomponents.a:TestIB",
                                      "${PETSC_DIR}/${PETSC_ARCH}/lib/libtestcomponents:TestIA"); CHKERRQ(ierr);
  }
  else {
    container = *((PetscContainer*)component);
    ierr = PetscPrintf(comm, "%s: configuring to state %d\n", __FUNCT__, state); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}/* PetscFwkComponentConfigureTestIB() */
#undef  __FUNCT__ 
#define __FUNCT__ "PetscFwkComponentConfigureTestIC"
PetscErrorCode PETSC_DLLEXPORT PetscFwkComponentConfigureTestIC(PetscFwk fwk, PetscInt state, PetscObject *component) {
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
}/* PetscFwkComponentConfigureTestIC() */
EXTERN_C_END

