#include "petsc.h"
EXTERN_C_BEGIN
#undef  __FUNCT__ 
#define __FUNCT__ "PetscFwkComponentConfigureTestIIA"
PetscErrorCode PETSC_DLLEXPORT PetscFwkComponentConfigureTestIIA(PetscFwk fwk, const char* key, const char* conf, PetscObject *component) {
  MPI_Comm       comm;
  PetscContainer container;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)fwk, &comm); CHKERRQ(ierr);
  if(!*component) {
    ierr = PetscContainerCreate(comm, &container); CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)container, "TestIIA"); CHKERRQ(ierr);
    *component = (PetscObject)container;
    ierr = PetscPrintf(comm, "%s: created component %s\n", __FUNCT__, key); CHKERRQ(ierr);
  }
  else {
    container = *((PetscContainer*)component);
    ierr = PetscPrintf(comm, "%s: using configuration %s\n", __FUNCT__, conf); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}/* PetscFwkComponentConfigureTestIIA() */

#undef  __FUNCT__ 
#define __FUNCT__ "PetscFwkComponentConfigureTestIIB"
PetscErrorCode PETSC_DLLEXPORT PetscFwkComponentConfigureTestIIB(PetscFwk fwk, const char* key, const char* conf, PetscObject *component) {
  MPI_Comm       comm;
  PetscContainer container;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)fwk, &comm); CHKERRQ(ierr);
  if(!*component) {
    ierr = PetscContainerCreate(comm, &container); CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)container, "TestIIB"); CHKERRQ(ierr);
    *component = (PetscObject)container;
    ierr = PetscPrintf(comm, "%s: created component %s\n", __FUNCT__, key); CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "%s: registering dependency: %s --> IIA\n", __FUNCT__, key); CHKERRQ(ierr);
    ierr = PetscFwkRegisterDependence(fwk, key, "IIA"); CHKERRQ(ierr);
  }
  else {
    container = *((PetscContainer*)component);
    ierr = PetscPrintf(comm, "%s: using configuration %s\n", __FUNCT__, conf); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}/* PetscFwkComponentConfigureTestIIB() */

#undef  __FUNCT__ 
#define __FUNCT__ "PetscFwkComponentConfigureTestIIC"
PetscErrorCode PETSC_DLLEXPORT PetscFwkComponentConfigureTestIIC(PetscFwk fwk, const char* key, const char* conf, PetscObject *component) {
  MPI_Comm       comm;
  PetscContainer container;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)fwk, &comm); CHKERRQ(ierr);
  if(!*component) {
    ierr = PetscContainerCreate(comm, &container); CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)container, "TestIIC"); CHKERRQ(ierr);
    *component = (PetscObject)container;
    ierr = PetscPrintf(comm, "%s: created component %s\n", __FUNCT__, key); CHKERRQ(ierr);
  }
  else {
    container = *((PetscContainer*)component);
    ierr = PetscPrintf(comm, "%s: using configuration %s\n", __FUNCT__, conf); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}/* PetscFwkComponentConfigureTestIIC() */
EXTERN_C_END
