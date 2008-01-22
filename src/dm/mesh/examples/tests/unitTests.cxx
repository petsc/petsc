
static char help[] = "Sieve Package Correctness and Performance Unit Tests.\n\n";

#include <petsc.h>

#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/BriefTestProgressListener.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestResultCollector.h>
#include <cppunit/TestRunner.h>
#include <cppunit/TextOutputter.h>

extern PetscErrorCode RegisterSifterSuite();
extern PetscErrorCode RegisterSieveSuite();
extern PetscErrorCode RegisterSectionSuite();
extern PetscErrorCode RegisterISectionSuite();

typedef struct {
  PetscTruth sifter;   // Run the Sifter tests
  PetscTruth sieve;    // Run the Sieve tests
  PetscTruth section;  // Run the Section tests
  PetscTruth isection; // Run the ISection tests
} Options;

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, Options *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->sifter   = PETSC_FALSE;
  options->sieve    = PETSC_FALSE;
  options->section  = PETSC_FALSE;
  options->isection = PETSC_FALSE;

  ierr = PetscOptionsBegin(comm, "", "Options for the Sieve package tests", "Sieve");CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-sifter", "Run Sifter tests", "unitTests", options->sifter, &options->sifter, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-sieve", "Run Sieve tests", "unitTests", options->sieve, &options->sieve, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-section", "Run Section tests", "unitTests", options->section, &options->section, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-isection", "Run ISection tests", "unitTests", options->isection, &options->isection, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RegisterSuites"
PetscErrorCode RegisterSuites(Options *options) {
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (options->sifter)   {ierr = RegisterSifterSuite();CHKERRQ(ierr);}
  if (options->sieve)    {ierr = RegisterSieveSuite();CHKERRQ(ierr);}
  if (options->section)  {ierr = RegisterSectionSuite();CHKERRQ(ierr);}
  if (options->isection) {ierr = RegisterISectionSuite();CHKERRQ(ierr);}
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
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(result);
}
