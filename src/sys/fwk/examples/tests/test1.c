#include "petsc.h"

#undef  __FUNCT__ 
#define __FUNCT__ "TestACall"
PetscErrorCode  TestACall(PetscFwk component, const char* message) {
  MPI_Comm comm = ((PetscObject)component)->comm;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscPrintf(comm, "%s: running '%s'\n", __FUNCT__, message); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* TestACall() */

#undef  __FUNCT__ 
#define __FUNCT__ "TestBInitialize"
PetscErrorCode  TestBInitialize(PetscFwk component, const char* message) {
  MPI_Comm comm = ((PetscObject)component)->comm;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscPrintf(comm, "%s: initializing\n", __FUNCT__); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* TestBInitialize() */

#undef  __FUNCT__ 
#define __FUNCT__ "TestBCall"
PetscErrorCode  TestBCall(PetscFwk component, const char* message) {
  MPI_Comm comm = ((PetscObject)component)->comm;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscPrintf(comm, "%s: running '%s'\n", __FUNCT__, message); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* TestBCall() */

#undef  __FUNCT__ 
#define __FUNCT__ "TestCInitialize"
PetscErrorCode  TestCInitialize(PetscFwk component, const char* message) {
  MPI_Comm comm = ((PetscObject)component)->comm;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscPrintf(comm, "%s: initializing\n", __FUNCT__); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* TestCInitialize() */

#undef  __FUNCT__ 
#define __FUNCT__ "TestCCall"
PetscErrorCode  TestCCall(PetscFwk component, const char* message) {
  MPI_Comm comm = ((PetscObject)component)->comm;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscPrintf(comm, "%s: running '%s'\n", __FUNCT__, message); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* TestCCall() */

#undef  __FUNCT__ 
#define __FUNCT__ "main"
PetscErrorCode main(int argc, char *argv[]) {
  PetscFwk       fwk, a = PETSC_NULL, b = PETSC_NULL;
  const char    *conf;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, PETSC_NULL, PETSC_NULL); CHKERRQ(ierr);
  ierr = PetscFwkCreate(PETSC_COMM_WORLD, &fwk); CHKERRQ(ierr);
  ierr = PetscFwkRegisterComponentURL(fwk, "TestIA", "${PETSC_DIR}/${PETSC_ARCH}/lib/libtestcomponents.so:TestIA");   CHKERRQ(ierr);
  ierr = PetscFwkRegisterComponentURL(fwk, "TestIB", "${PETSC_DIR}/${PETSC_ARCH}/lib/libtestcomponents.so:TestIB");   CHKERRQ(ierr);
  ierr = PetscFwkRegisterComponentURL(fwk, "TestIIA", "${PETSC_DIR}/${PETSC_ARCH}/lib/libtestcomponents.so:TestIIA"); CHKERRQ(ierr);
  ierr = PetscFwkRegisterComponentURL(fwk, "TestIIB", "${PETSC_DIR}/${PETSC_ARCH}/lib/libtestcomponents.so:TestIIB"); CHKERRQ(ierr);
  ierr = PetscFwkRegisterComponentURL(fwk, "TestIC", "${PETSC_DIR}/${PETSC_ARCH}/lib/libtestcomponents.so:TestIC");    CHKERRQ(ierr);
  ierr = PetscFwkRegisterComponentURL(fwk, "TestIIC", "${PETSC_DIR}/${PETSC_ARCH}/lib/libtestcomponents.so:TestIIC");  CHKERRQ(ierr);
  ierr = PetscFwkRegisterDependence(fwk, "TestIIB", "TestIB"); CHKERRQ(ierr);
  ierr = PetscFwkRegisterDependence(fwk, "TestIIA", "TestIA"); CHKERRQ(ierr); 
  /**/
  ierr = PetscFwkRegisterComponent(fwk, "TestA"); CHKERRQ(ierr);
  ierr = PetscFwkGetComponent(fwk, "TestA", &a, PETSC_NULL);  CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)a, "call", "call", (void (*)(void))TestACall); CHKERRQ(ierr);
  /**/
  ierr = PetscFwkRegisterComponent(fwk, "TestB"); CHKERRQ(ierr);
  ierr = PetscFwkGetComponent(fwk, "TestB", &b, PETSC_NULL);  CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)b, "initialize", "initialize", (void (*)(void))TestBInitialize); CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)b, "call", "call", (void (*)(void))TestBCall); CHKERRQ(ierr);
  /**/
  /* The following line requires that this executable export symbols TestCXXX as dynamic. Uncomment, when you are sure your linker cooperates. */
  /* ierr = PetscFwkRegisterComponentURL(fwk, "TestC", "TestC");  CHKERRQ(ierr); */
  /**/
  ierr = PetscFwkRegisterComponentURL(fwk, "TestIIIA", "./testcomponents.py:TestIIIA"); CHKERRQ(ierr);

  CHKERRQ(ierr);
  ierr = PetscFwkRegisterComponentURL(fwk, "TestIIIA", "./testcomponents.py:TestIIIA");

  CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "Viewing top-level framework:\n"); CHKERRQ(ierr);
  ierr = PetscFwkView(fwk, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
  conf = "initialize";
  ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "Visiting with message '%s':\n", conf); CHKERRQ(ierr);  
  ierr = PetscFwkVisit(fwk, conf); CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "Viewing top-level framework:\n"); CHKERRQ(ierr);
  ierr = PetscFwkView(fwk, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
  conf = "configure";
  ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "Visiting with message '%s':\n", conf); CHKERRQ(ierr);  
  ierr = PetscFwkVisit(fwk, conf); CHKERRQ(ierr);
  ierr = PetscFinalize();
  PetscFunctionReturn(0);
}/* main() */

