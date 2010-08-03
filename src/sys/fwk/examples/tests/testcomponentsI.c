#include "petsc.h"

EXTERN_C_BEGIN
#undef  __FUNCT__ 
#define __FUNCT__ "PetscFwkComponentConfigureTestIA"
PetscErrorCode PETSC_DLLEXPORT PetscFwkComponentConfigureTestIA(PetscFwk fwk, const char* key, const char *conf, PetscObject *component) {
  MPI_Comm       comm;
  PetscContainer container;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)fwk, &comm); CHKERRQ(ierr);
  if(!*component) {
    ierr = PetscContainerCreate(comm, &container); CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)container, "TestIA"); CHKERRQ(ierr);
    *component = (PetscObject)container;
    ierr = PetscPrintf(comm, "%s: created component %s\n", __FUNCT__, key); CHKERRQ(ierr);
  }
  else {
    container = *((PetscContainer*)component);
    ierr = PetscPrintf(comm, "%s: using configuration %s\n", __FUNCT__, conf); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}/* PetscFwkComponentConfigureTestIA() */

#undef  __FUNCT__ 
#define __FUNCT__ "PetscFwkComponentConfigureTestIB"
PetscErrorCode PETSC_DLLEXPORT PetscFwkComponentConfigureTestIB(PetscFwk fwk, const char* key, const char* conf, PetscObject *component) {
  MPI_Comm       comm;
  PetscContainer container;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)fwk, &comm); CHKERRQ(ierr);
  if(!*component) {
    ierr = PetscContainerCreate(comm, &container); CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)container, "TestIB"); CHKERRQ(ierr);
    *component = (PetscObject)container;
    ierr = PetscPrintf(comm, "%s: created component %s\n", __FUNCT__, key); CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "%s: registering dependence: %s --> IA\n", __FUNCT__, key); CHKERRQ(ierr);
    ierr = PetscFwkRegisterDependence(fwk, key, "IA"); CHKERRQ(ierr);
  }
  else {
    container = *((PetscContainer*)component);
    ierr = PetscPrintf(comm, "%s: using configuration %s\n", __FUNCT__, conf); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}/* PetscFwkComponentConfigureTestIB() */

#undef  __FUNCT__ 
#define __FUNCT__ "PetscFwkComponentConfigureTestIC"
PetscErrorCode PETSC_DLLEXPORT PetscFwkComponentConfigureTestIC(PetscFwk fwk, const char* key, const char *conf, PetscObject *component) {
  MPI_Comm       comm;
  PetscContainer container;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)fwk, &comm); CHKERRQ(ierr);
  if(!*component) {
    ierr = PetscContainerCreate(comm, &container); CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)container, "TestIC"); CHKERRQ(ierr);
    *component = (PetscObject)container;
    ierr = PetscPrintf(comm, "%s: created component %s\n", __FUNCT__, key); CHKERRQ(ierr);
  }
  else {
    container = *((PetscContainer*)component);
    ierr = PetscPrintf(comm, "%s: using configuration %s\n", __FUNCT__, conf); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}/* PetscFwkComponentConfigureTestIC() */
EXTERN_C_END

