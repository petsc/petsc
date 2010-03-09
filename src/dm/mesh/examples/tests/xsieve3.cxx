static char help[] = "XSieve Basic Slice test.\n\n";

#include <petscsys.h>
#include "xsieveTest.hh"


typedef ALE::Test::XSieveTester::default_xsieve_type     xsieve_type;


#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help);CHKERRQ(ierr);
  {
    ALE::ArgDB argDB(ALE::Test::XSieveTester().argDB, argc, argv);
#ifdef ALE_USE_DEBUGGING
    // Set debugging options
    ALE::Xdebug   = argDB["debug"];
    ALE::Xcodebug = argDB["codebug"];
#endif
    ALE::Obj<xsieve_type> xsieveFork = ALE::Test::XSieveTester::createForkXSieve(PETSC_COMM_SELF, argDB);
    ierr = ALE::Test::XSieveTester::SliceBasicTest<xsieve_type>(xsieveFork, argDB, "Fork XSieve");CHKERRQ(ierr);
    ALE::Obj<xsieve_type> xsieveHat = ALE::Test::XSieveTester::createHatXSieve(PETSC_COMM_SELF, argDB);
    ierr = ALE::Test::XSieveTester::SliceBasicTest<xsieve_type>(xsieveHat, argDB, "Hat XSieve");CHKERRQ(ierr);
  }
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
