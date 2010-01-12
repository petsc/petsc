
static char help[] = "Sieve Package Correctness and Performance Unit Tests.\n\n";

#include <petscsnes.h>
#include <petscdmmg.h>
#include <petscmesh.hh>
#include "bratu_quadrature.h"

#include <iostream>
#include <fstream>

#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/BriefTestProgressListener.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestResultCollector.h>
#include <cppunit/TestRunner.h>
#include <cppunit/TextOutputter.h>

#include <sieve/problem/Functions.hh>
#include "bratu1.hh"
#include "laplaceBEM1.hh"

typedef struct {
  PetscTruth function;      // Run the functionality tests
  PetscTruth stress;        // Run the stress tests
  PetscTruth convergence;   // Run the convergence tests
  PetscTruth bratu;         // Run the Bratu problem tests
  PetscTruth laplaceBEM;    // Run the Laplace BEM problem tests
} Options;

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, Options *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->function      = PETSC_FALSE;
  options->stress        = PETSC_FALSE;
  options->convergence   = PETSC_FALSE;
  options->bratu         = PETSC_FALSE;
  options->laplaceBEM    = PETSC_FALSE;

  ierr = PetscOptionsBegin(comm, "", "Options for the Sieve package tests", "Sieve");CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-function", "Run functionality tests", "unitTests", options->function, &options->function, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-stress", "Run stress tests", "unitTests", options->stress, &options->stress, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-convergence", "Run convergence tests", "unitTests", options->convergence, &options->convergence, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-bratu", "Run Bratu tests", "unitTests", options->bratu, &options->bratu, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-laplace_bem", "Run Laplace BEM tests", "unitTests", options->laplaceBEM, &options->laplaceBEM, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RegisterSuites"
PetscErrorCode RegisterSuites(Options *options) {
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (options->bratu) {
    if (options->function)    {ierr = RegisterBratuFunctionSuite();CHKERRQ(ierr);}
    if (options->stress)      {ierr = RegisterBratuStressSuite();CHKERRQ(ierr);}
    if (options->convergence) {ierr = RegisterBratuConvergenceSuite();CHKERRQ(ierr);}
  }
  if (options->laplaceBEM) {
    if (options->function)    {ierr = RegisterLaplaceBEMFunctionSuite();CHKERRQ(ierr);}
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
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(result);
}
