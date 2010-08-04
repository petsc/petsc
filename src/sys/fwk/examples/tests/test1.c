#include "petsc.h"

#undef  __FUNCT__ 
#define __FUNCT__ "main"
PetscErrorCode main(int argc, char *argv[]) {
  PetscFwk       fwk;
  char           *conf;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, PETSC_NULL, PETSC_NULL); CHKERRQ(ierr);
  ierr = PetscFwkCreate(PETSC_COMM_WORLD, &fwk); CHKERRQ(ierr);
  ierr = PetscFwkRegisterComponent(fwk, "IA", "${PETSC_DIR}/${PETSC_ARCH}/lib/libtestcomponents.so:TestIA");   CHKERRQ(ierr);
  ierr = PetscFwkRegisterComponent(fwk, "IB", "${PETSC_DIR}/${PETSC_ARCH}/lib/libtestcomponents.so:TestIB");   CHKERRQ(ierr);
  ierr = PetscFwkRegisterComponent(fwk, "IIA", "${PETSC_DIR}/${PETSC_ARCH}/lib/libtestcomponents.so:TestIIA"); CHKERRQ(ierr);
  ierr = PetscFwkRegisterComponent(fwk, "IIB", "${PETSC_DIR}/${PETSC_ARCH}/lib/libtestcomponents.so:TestIIB"); CHKERRQ(ierr);
  ierr = PetscFwkRegisterComponent(fwk, "IC", "${PETSC_DIR}/${PETSC_ARCH}/lib/libtestcomponents.so:TestIC");    CHKERRQ(ierr);
  ierr = PetscFwkRegisterComponent(fwk, "IIC", "${PETSC_DIR}/${PETSC_ARCH}/lib/libtestcomponents.so:TestIIC");  CHKERRQ(ierr);
  ierr = PetscFwkRegisterDependence(fwk, "IIB", "IB"); CHKERRQ(ierr);
  ierr = PetscFwkRegisterDependence(fwk, "IIA", "IA"); CHKERRQ(ierr); 

  CHKERRQ(ierr);
  /*
  ierr = PetscFwkRegisterComponent(fwk, "IIIA", "./testcomponents.py:TestIIIA");
  */
  CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "Component initialization order:\n"); CHKERRQ(ierr);
  ierr = PetscFwkView(fwk, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
  conf = PETSC_NULL;
  ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "Configuring to configuration %s:\n", conf); CHKERRQ(ierr);  
  ierr = PetscFwkConfigure(fwk, conf); CHKERRQ(ierr);
  conf = "testConfiguration";
  ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "Configuring to configuration %s:\n", conf); CHKERRQ(ierr);  
  ierr = PetscFwkConfigure(fwk, conf); CHKERRQ(ierr);
  ierr = PetscFinalize();
  PetscFunctionReturn(0);
}/* main() */

