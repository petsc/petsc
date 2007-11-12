static char help[] = "Sieve Basic Tests: Sifter functionality.\n\n";

#include <petsc.h>
#include "xsieveTest.hh"



typedef ALE::Test::XSieveTester::default_xsieve_type     xsieve_type;



#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help); CHKERRQ(ierr);
  {
    ALE::Component::ArgDB argDB(ALE::Test::XSieveTester().argDB, argc, argv);
    ALE::Obj<xsieve_type> xsieveTree = ALE::Test::XSieveTester::createTreeXSieve(PETSC_COMM_SELF, argDB);
    ierr = ALE::Test::XSifterTester::BasicTest<xsieve_type>(xsieveTree, argDB, "Tree"); CHKERRQ(ierr);
    ierr = ALE::Test::XSifterTester::BaseTest<xsieve_type>(xsieveTree, argDB, "Tree"); CHKERRQ(ierr);
    ierr = ALE::Test::XSifterTester::ConeTest<xsieve_type>(xsieveTree, argDB, "Tree"); CHKERRQ(ierr);
  }
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
