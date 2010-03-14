#include "petscfwk.h"

#undef  __FUNCT__ 
#define __FUNCT__ "main"
PetscErrorCode main(int argc, char *argv[]) {
  PetscFwk       fwk;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, PETSC_NULL, PETSC_NULL); CHKERRQ(ierr);
  ierr = PetscFwkCreate(PETSC_COMM_WORLD, &fwk); CHKERRQ(ierr);
  ierr = PetscFwkRegisterDependence(fwk, 
                         "${PETSC_DIR}/${PETSC_ARCH}/lib/libtestcomponents.a:TestIIB",
                         "${PETSC_DIR}/${PETSC_ARCH}/lib/libtestcomponents.a:TestIB"); 
  CHKERRQ(ierr);
  ierr = PetscFwkRegisterComponent(fwk, 
                                   "${PETSC_DIR}/${PETSC_ARCH}/lib/libtestcomponents.so:TestIC");
  CHKERRQ(ierr);
  ierr = PetscFwkRegisterDependence(fwk, 
                         "${PETSC_DIR}/${PETSC_ARCH}/lib/libtestcomponents.so:TestIIA",
                         "${PETSC_DIR}/${PETSC_ARCH}/lib/libtestcomponents.so:TestIA");
  CHKERRQ(ierr);
  ierr = PetscFwkRegisterComponent(fwk, 
                                   "${PETSC_DIR}/${PETSC_ARCH}/lib/libtestcomponents.so:TestIIC");
  CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "Component initialization order:\n"); CHKERRQ(ierr);
  ierr = PetscFwkViewConfigurationOrder(fwk, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
  ierr = PetscFwkConfigure(fwk, 1); CHKERRQ(ierr);
  ierr = PetscFinalize();
  PetscFunctionReturn(0);
}/* main() */

