static const char help[] = "Tests PetscShell usage\n";
#include <petsc.h>

#undef  __FUNCT__ 
#define __FUNCT__ "TestACall"
PetscErrorCode  TestACall(PetscShell component, const char* message) {
  MPI_Comm comm = ((PetscObject)component)->comm;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscPrintf(comm, "%s: running '%s'\n", __FUNCT__, message); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* TestACall() */

#undef  __FUNCT__ 
#define __FUNCT__ "TestBInitialize"
PetscErrorCode  TestBInitialize(PetscShell component, const char* message) {
  MPI_Comm comm = ((PetscObject)component)->comm;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscPrintf(comm, "%s: initializing\n", __FUNCT__); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* TestBInitialize() */

#undef  __FUNCT__ 
#define __FUNCT__ "TestBCall"
PetscErrorCode  TestBCall(PetscShell component, const char* message) {
  MPI_Comm comm = ((PetscObject)component)->comm;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscPrintf(comm, "%s: running '%s'\n", __FUNCT__, message); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* TestBCall() */

#undef  __FUNCT__ 
#define __FUNCT__ "TestCInitialize"
PetscErrorCode  TestCInitialize(PetscShell component, const char* message) {
  MPI_Comm comm = ((PetscObject)component)->comm;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscPrintf(comm, "%s: initializing\n", __FUNCT__); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* TestCInitialize() */

#undef  __FUNCT__ 
#define __FUNCT__ "TestCCall"
PetscErrorCode  TestCCall(PetscShell component, const char* message) {
  MPI_Comm comm = ((PetscObject)component)->comm;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscPrintf(comm, "%s: running '%s'\n", __FUNCT__, message); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}/* TestCCall() */

#undef  __FUNCT__ 
#define __FUNCT__ "main"
PetscErrorCode main(int argc, char *argv[]) {
  PetscShell       shell, a = PETSC_NULL, b = PETSC_NULL;
  const char    *conf;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, PETSC_NULL, help); CHKERRQ(ierr);
  ierr = PetscShellCreate(PETSC_COMM_WORLD, &shell); CHKERRQ(ierr);
  ierr = PetscShellRegisterComponentURL(shell, "TestIA", "${PETSC_DIR}/${PETSC_ARCH}/lib/libtestcomponents.so:TestIA");   CHKERRQ(ierr);
  ierr = PetscShellRegisterComponentURL(shell, "TestIB", "${PETSC_DIR}/${PETSC_ARCH}/lib/libtestcomponents.so:TestIB");   CHKERRQ(ierr);
  ierr = PetscShellRegisterComponentURL(shell, "TestIIA", "${PETSC_DIR}/${PETSC_ARCH}/lib/libtestcomponents.so:TestIIA"); CHKERRQ(ierr);
  ierr = PetscShellRegisterComponentURL(shell, "TestIIB", "${PETSC_DIR}/${PETSC_ARCH}/lib/libtestcomponents.so:TestIIB"); CHKERRQ(ierr);
  ierr = PetscShellRegisterComponentURL(shell, "TestIC", "${PETSC_DIR}/${PETSC_ARCH}/lib/libtestcomponents.so:TestIC");    CHKERRQ(ierr);
  ierr = PetscShellRegisterComponentURL(shell, "TestIIC", "${PETSC_DIR}/${PETSC_ARCH}/lib/libtestcomponents.so:TestIIC");  CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Registering dependence: TestIIB --> TestIB\n"); CHKERRQ(ierr);
  ierr = PetscShellRegisterDependence(shell, "TestIIB", "TestIB"); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Registering dependence: TestIIA --> TestIA\n"); CHKERRQ(ierr);
  ierr = PetscShellRegisterDependence(shell, "TestIIA", "TestIA"); CHKERRQ(ierr); 
  /**/
  ierr = PetscShellCreate(((PetscObject)shell)->comm, &a);        CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)a, "call", "call", (void (*)(void))TestACall); CHKERRQ(ierr);
  ierr = PetscShellRegisterComponentShell(shell, "TestA", a); CHKERRQ(ierr);
  /**/
  b = PETSC_NULL;
  ierr = PetscShellRegisterComponentShell(shell, "TestB", b); CHKERRQ(ierr);
  ierr = PetscShellGetComponent(shell, "TestB", &b, PETSC_NULL);  CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)b, "initialize", "initialize", (void (*)(void))TestBInitialize); CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)b, "call", "call", (void (*)(void))TestBCall); CHKERRQ(ierr);
  /**/
  /* The following line requires that this executable export symbols TestCXXX as dynamic. Uncomment, when you are sure your linker cooperates. */
  /* ierr = PetscShellRegisterComponentURL(shell, "TestC", "TestC");  CHKERRQ(ierr); */
  /**/
  ierr = PetscShellRegisterComponentURL(shell, "TestIIIA", "./testcomponentsIII.py:TestIIIA"); CHKERRQ(ierr);

  CHKERRQ(ierr);
  ierr = PetscShellRegisterComponentURL(shell, "TestIIIA", "./testcomponentsIII.py:TestIIIA");

  CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "Viewing top-level shell:\n"); CHKERRQ(ierr);
  ierr = PetscShellView(shell, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
  conf = "initialize";
  ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "Visiting with message '%s':\n", conf); CHKERRQ(ierr);  
  ierr = PetscShellVisit(shell, conf); CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "Viewing top-level shell:\n"); CHKERRQ(ierr);
  ierr = PetscShellView(shell, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
  conf = "configure";
  ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, "Visiting with message '%s':\n", conf); CHKERRQ(ierr);  
  ierr = PetscShellVisit(shell, conf); CHKERRQ(ierr);
  ierr = PetscFinalize();
  PetscFunctionReturn(0);
}/* main() */

