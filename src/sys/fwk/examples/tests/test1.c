#include "petsc.h"

#undef  __FUNCT__ 
#define __FUNCT__ "main"
PetscErrorCode main(int argc, char *argv[]) {
  PetscFwk       fwk;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, PETSC_NULL, PETSC_NULL); CHKERRQ(ierr);
  ierr = PetscFwkCreate(PETSC_COMM_WORLD, &fwk); CHKERRQ(ierr);
  ierr = PetscFwkRegisterDependence(fwk, "IIB", "IB"); CHKERRQ(ierr);
  ierr = PetscFwkRegisterComponent(fwk, "IC", "${PETSC_DIR}/${PETSC_ARCH}/lib/libtestcomponents.so:TestIC"); CHKERRQ(ierr);
  ierr = PetscFwkRegisterDependence(fwk,"IIA", "IA"); CHKERRQ(ierr); 
  ierr = PetscFwkRegisterComponent(fwk, "IIC", "${PETSC_DIR}/${PETSC_ARCH}/lib/libtestcomponents.so:TestIIC");
  CHKERRQ(ierr);
  /*
  ierr = PetscFwkRegisterComponent(fwk, "IIIA", "./testcomponents.py:TestIIIA");
  */
  CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "Component initialization order:\n"); CHKERRQ(ierr);
  ierr = PetscFwkView(fwk, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
  ierr = PetscFwkConfigure(fwk, PETSC_NULL); CHKERRQ(ierr);
  ierr = PetscFinalize();
  PetscFunctionReturn(0);
}/* main() */

