#include <petscsnes.h>

#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/extensions/HelperMacros.h>

#include <iostream>
#include <fstream>

class FunctionTestBratu : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(FunctionTestBratu);

  CPPUNIT_TEST(testBratuUnitSquare);

  CPPUNIT_TEST_SUITE_END();
public:
  // Typedefs
protected:
  ALE::Obj<ALE::Problem::Bratu> _problem;
  int                           _debug; // The debugging level
  PetscInt                      _iters; // The number of test repetitions
public:
  PetscErrorCode processOptions() {
    PetscErrorCode ierr;

    this->_debug = 0;
    this->_iters = 1;

    PetscFunctionBegin;
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "Options for interval section stress test", "ISection");CHKERRQ(ierr);
      ierr = PetscOptionsInt("-debug", "The debugging level", "isection.c", this->_debug, &this->_debug, PETSC_NULL);CHKERRQ(ierr);
      ierr = PetscOptionsInt("-iterations", "The number of test repetitions", "isection.c", this->_iters, &this->_iters, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
    PetscFunctionReturn(0);
  };

  /// Setup data.
  void setUp(void) {
    this->processOptions();
    this->_problem = new ALE::Problem::Bratu(PETSC_COMM_WORLD, this->_debug);
  };

  /// Tear down data.
  void tearDown(void) {};

  void testBratuUnitSquare(void) {
    ///this->checkAnswer(answerStruct, "BratuUnitSquare");
  };
};

#undef __FUNCT__
#define __FUNCT__ "RegisterBratuFunctionSuite"
PetscErrorCode RegisterBratuFunctionSuite() {
  CPPUNIT_TEST_SUITE_REGISTRATION(FunctionTestBratu);
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RegisterBratuStressSuite"
PetscErrorCode RegisterBratuStressSuite() {
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RegisterBratuConvergenceSuite"
PetscErrorCode RegisterBratuConvergenceSuite() {
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
