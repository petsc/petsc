static char help[] = "XSieve Boundary Test.\n\n";

#include <petscsys.h>
#include "xsieveTest.hh"



typedef ALE::Test::XSieveTester::symmetric_xsieve_type     xsieve_type;



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
    //
    ALE::Obj<xsieve_type> xsieveTree = ALE::Test::XSieveTester::createTreeXSieve(PETSC_COMM_SELF, argDB);
    ierr = ALE::Test::XSieveTester::BoundaryTest<xsieve_type>(xsieveTree, argDB, "Tree");CHKERRQ(ierr);
  }
  ierr = PetscFinalize();
  PetscFunctionReturn(0);
}
