#define ALE_MEM_LOGGING

#include <set>
#include <ALE.hh>
#include "unitTests.hh"

#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/extensions/HelperMacros.h>

typedef int point_type;

typedef struct {
  int debug;    // The debugging level
  int iters;    // The number of test repetitions
  int numItems; // The number of items per container
} Options;

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
static PetscErrorCode ProcessOptions(MPI_Comm comm, Options *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->debug    = 0;
  options->iters    = 100;
  options->numItems = 1000;

  ierr = PetscOptionsBegin(comm, "", "Options for stl memory test", "Sieve");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-debug", "The debugging level", "stl1.c", options->debug, &options->debug, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-iterations", "The number of test repetitions", "stl1.c", options->iters, &options->iters, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-items", "The number of items per container", "stl1.c", options->numItems, &options->numItems, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

class MemoryTestSTL : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(MemoryTestSTL);

  CPPUNIT_TEST(testSet);

  CPPUNIT_TEST_SUITE_END();
protected:
  Options _options;
public :
  /// Setup data.
  void setUp(void) {
    ProcessOptions(PETSC_COMM_WORLD, &this->_options);
  };

  /// Tear down data.
  void tearDown(void) {};

  /// Test cone().
  void testSet(void) {
    ALE::MemoryLogger& logger = ALE::MemoryLogger::singleton();

    logger.stagePush("Set");
    for(int i = 0; i < this->_options.iters; ++i) {
      std::set<int, std::less<int>, ALE::malloc_allocator<int> > s;

      for(int c = 0; c < this->_options.numItems; ++c) {
        s.insert(c);
      }
    }
    logger.stagePop();
    CPPUNIT_ASSERT_EQUAL_MESSAGE("Invalid number of allocations", this->_options.numItems*this->_options.iters, logger.getNumAllocations("Set"));
    CPPUNIT_ASSERT_EQUAL_MESSAGE("Invalid number of deallocations", this->_options.numItems*this->_options.iters, logger.getNumDeallocations("Set"));
    CPPUNIT_ASSERT_EQUAL_MESSAGE("Invalid number of bytes allocated", 20*this->_options.numItems*this->_options.iters, logger.getAllocationTotal("Set"));
    CPPUNIT_ASSERT_EQUAL_MESSAGE("Invalid number of bytes deallocated", 20*this->_options.numItems*this->_options.iters, logger.getDeallocationTotal("Set"));
  };
};


#undef __FUNCT__
#define __FUNCT__ "RegisterSTLMemorySuite"
PetscErrorCode RegisterSTLMemorySuite() {
  CPPUNIT_TEST_SUITE_REGISTRATION (MemoryTestSTL);
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
