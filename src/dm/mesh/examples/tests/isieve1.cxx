#include <petsc.h>
#include <ISieve.hh>
#include "unitTests.hh"

#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/extensions/HelperMacros.h>

class FunctionTestISieve : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(FunctionTestISieve);

  CPPUNIT_TEST(testBase);

  CPPUNIT_TEST_SUITE_END();
public:
  typedef ALE::IFSieve<int> sieve_type;
protected:
  ALE::Obj<sieve_type> _sieve;
  int                  _debug; // The debugging level
  PetscInt             _iters; // The number of test repetitions
  PetscInt             _size;  // The interval size
public:
  PetscErrorCode processOptions() {
    PetscErrorCode ierr;

    this->_debug = 0;
    this->_iters = 1;
    this->_size  = 10;

    PetscFunctionBegin;
    ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "Options for interval section stress test", "ISection");CHKERRQ(ierr);
      ierr = PetscOptionsInt("-debug", "The debugging level", "isection.c", this->_debug, &this->_debug, PETSC_NULL);CHKERRQ(ierr);
      ierr = PetscOptionsInt("-iterations", "The number of test repetitions", "isection.c", this->_iters, &this->_iters, PETSC_NULL);CHKERRQ(ierr);
      ierr = PetscOptionsInt("-size", "The interval size", "isection.c", this->_size, &this->_size, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
    PetscFunctionReturn(0);
  };

  /// Setup data.
  void setUp(void) {
    this->processOptions();
    this->_sieve = new sieve_type(PETSC_COMM_WORLD, 0, this->_size*3+1, this->_debug);
    try {
      ALE::Test::SifterBuilder::createHatISifter<sieve_type>(PETSC_COMM_WORLD, *this->_sieve, this->_size, this->_debug);
    } catch (ALE::Exception e) {
      std::cerr << "ERROR: " << e << std::endl;
    }
  };

  /// Tear down data.
  void tearDown(void) {};

  void testBase(void) {
    this->_sieve->view("Test Sieve");
  };
};

#undef __FUNCT__
#define __FUNCT__ "RegisterISieveFunctionSuite"
PetscErrorCode RegisterISieveFunctionSuite() {
  CPPUNIT_TEST_SUITE_REGISTRATION(FunctionTestISieve);
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
