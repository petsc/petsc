static char help[] = "Sifter Basic Ordering Tests.\n\n";

#include <petsc.h>
#include "xsifterTest.hh"



typedef ALE::Test::XSifterTester::default_xsifter_type     xsifter_type;



#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help); CHKERRQ(ierr);
  {
    ALE::Component::ArgDB argDB(ALE::Test::XSifterTester().argDB, argc, argv);
#ifdef ALE_USE_DEBUGGING
    // Set debugging options
    ALE::Xdebug   = argDB["debug"];
    ALE::Xcodebug = argDB["codebug"];
#endif
    ALE::Obj<xsifter_type> xsifterFork = ALE::Test::XSifterTester::createForkXSifter(PETSC_COMM_SELF, argDB);
    ierr = ALE::Test::XSifterTester::BasicTest<xsifter_type>(xsifterFork, argDB, "Fork"); CHKERRQ(ierr);
    ALE::Obj<xsifter_type> xsifterHat = ALE::Test::XSifterTester::createHatXSifter(PETSC_COMM_SELF, argDB);
    ierr = ALE::Test::XSifterTester::BasicTest<xsifter_type>(xsifterHat, argDB, "Hat"); CHKERRQ(ierr);
  }
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
