#define ALE_MEM_LOGGING
static char help[] = "Sieve Package Correctness and Performance Unit Tests.\n\n";

#include <petscsys.h>

#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/BriefTestProgressListener.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestResultCollector.h>
#include <cppunit/TestRunner.h>
#include <cppunit/TextOutputter.h>

extern PetscErrorCode RegisterSTLMemorySuite();
extern PetscErrorCode RegisterSifterStressSuite();
extern PetscErrorCode RegisterSieveFunctionSuite();
extern PetscErrorCode RegisterSieveStressSuite();
extern PetscErrorCode RegisterISieveFunctionSuite();
extern PetscErrorCode RegisterISieveMemorySuite();
extern PetscErrorCode RegisterSectionStressSuite();
extern PetscErrorCode RegisterISectionStressSuite();
extern PetscErrorCode RegisterIMeshFunctionSuite();
extern PetscErrorCode RegisterIMeshStressSuite();
extern PetscErrorCode RegisterIMeshMemorySuite();
extern PetscErrorCode RegisterDistributionFunctionSuite();
extern PetscErrorCode RegisterIDistributionFunctionSuite();

typedef struct {
  PetscBool  function;      // Run the functionality tests
  PetscBool  stress;        // Run the stress tests
  PetscBool  memory;        // Run the memory tests
  PetscBool  stl;           // Run the STL tests
  PetscBool  sifter;        // Run the Sifter tests
  PetscBool  sieve;         // Run the Sieve tests
  PetscBool  isieve;        // Run the ISieve tests
  PetscBool  section;       // Run the Section tests
  PetscBool  isection;      // Run the ISection tests
  PetscBool  imesh;         // Run the IMesh tests
  PetscBool  distribution;  // Run the Distribution tests
  PetscBool  idistribution; // Run the IDistribution tests
} Options;

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, Options *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->function      = PETSC_FALSE;
  options->stress        = PETSC_FALSE;
  options->memory        = PETSC_FALSE;
  options->stl           = PETSC_FALSE;
  options->sifter        = PETSC_FALSE;
  options->sieve         = PETSC_FALSE;
  options->isieve        = PETSC_FALSE;
  options->section       = PETSC_FALSE;
  options->isection      = PETSC_FALSE;
  options->imesh         = PETSC_FALSE;
  options->distribution  = PETSC_FALSE;
  options->idistribution = PETSC_FALSE;

  ierr = PetscOptionsBegin(comm, "", "Options for the Sieve package tests", "Sieve");CHKERRQ(ierr);
    ierr = PetscOptionsBool("-function", "Run functionality tests", "unitTests", options->function, &options->function, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-stress", "Run stress tests", "unitTests", options->stress, &options->stress, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-memory", "Run memory tests", "unitTests", options->memory, &options->memory, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-stl", "Run STL tests", "unitTests", options->stl, &options->stl, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-sifter", "Run Sifter tests", "unitTests", options->sifter, &options->sifter, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-sieve", "Run Sieve tests", "unitTests", options->sieve, &options->sieve, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-isieve", "Run ISieve tests", "unitTests", options->isieve, &options->isieve, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-section", "Run Section tests", "unitTests", options->section, &options->section, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-isection", "Run ISection tests", "unitTests", options->isection, &options->isection, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-imesh", "Run IMesh tests", "unitTests", options->imesh, &options->imesh, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-distribution", "Run Distribution tests", "unitTests", options->distribution, &options->distribution, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-idistribution", "Run IDistribution tests", "unitTests", options->idistribution, &options->idistribution, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RegisterSuites"
PetscErrorCode RegisterSuites(Options *options) {
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (options->stl) {
    if (options->memory)   {ierr = RegisterSTLMemorySuite();CHKERRQ(ierr);}
  }
  if (options->sifter) {
    if (options->stress)   {ierr = RegisterSifterStressSuite();CHKERRQ(ierr);}
  }
  if (options->sieve) {
    if (options->function) {ierr = RegisterSieveFunctionSuite();CHKERRQ(ierr);}
    if (options->stress)   {ierr = RegisterSieveStressSuite();CHKERRQ(ierr);}
  }
  if (options->isieve) {
    if (options->function) {ierr = RegisterISieveFunctionSuite();CHKERRQ(ierr);}
    if (options->memory)   {ierr = RegisterISieveMemorySuite();CHKERRQ(ierr);}
  }
  if (options->section) {
    if (options->stress)   {ierr = RegisterSectionStressSuite();CHKERRQ(ierr);}
  }
  if (options->isection) {
    if (options->stress)   {ierr = RegisterISectionStressSuite();CHKERRQ(ierr);}
  }
  if (options->imesh) {
    if (options->function) {ierr = RegisterIMeshFunctionSuite();CHKERRQ(ierr);}
    if (options->stress)   {ierr = RegisterIMeshStressSuite();CHKERRQ(ierr);}
    if (options->memory)   {ierr = RegisterIMeshMemorySuite();CHKERRQ(ierr);}
  }
  if (options->distribution) {
    if (options->function) {ierr = RegisterDistributionFunctionSuite();CHKERRQ(ierr);}
  }
  if (options->idistribution) {
    if (options->function) {ierr = RegisterIDistributionFunctionSuite();CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RunUnitTests"
PetscErrorCode RunUnitTests()
{ // main
  CppUnit::TestResultCollector result;

  PetscFunctionBegin;
  try {
    // Create event manager and test controller
    CppUnit::TestResult controller;

    // Add listener to collect test results
    controller.addListener(&result);

    // Add listener to show progress as tests run
    CppUnit::BriefTestProgressListener progress;
    controller.addListener(&progress);

    // Add top suite to test runner
    CppUnit::TestRunner runner;
    runner.addTest(CppUnit::TestFactoryRegistry::getRegistry().makeTest());
    runner.run(controller);

    // Print tests
    CppUnit::TextOutputter outputter(&result, std::cerr);
    outputter.write();
  } catch (...) {
    abort();
  }

  PetscFunctionReturn(result.wasSuccessful() ? 0 : 1);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  Options        options;
  PetscErrorCode ierr, result;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help);CHKERRQ(ierr);
  ierr = PetscLogBegin();CHKERRQ(ierr);
  ierr = ProcessOptions(PETSC_COMM_WORLD, &options);CHKERRQ(ierr);
  ierr = RegisterSuites(&options);CHKERRQ(ierr);
  result = RunUnitTests();
  ierr = PetscFinalize();
  PetscFunctionReturn(result);
}
