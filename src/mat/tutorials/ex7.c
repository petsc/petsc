static char help[] = "Example use of PetscInfo() as a configurable informative logging or warning tool\n";

/*T
   Concepts: introduction to logging techniques and introspection in PETSc;
   Processors: n
T*/

# include <petscsys.h>
# include <petscmat.h>
# include <petscvec.h>

int main(int argc, char **argv)
{
  PetscErrorCode  ierr;
  Mat             A, Aself;
  Vec             b, bself;
#if defined(PETSC_USE_INFO)
  PetscInt        testarg = 1234;
#endif
  int             numClasses;
  PetscClassId    testMatClassid, testVecClassid, testSysClassid;
  PetscBool       isEnabled = PETSC_FALSE, invert = PETSC_FALSE;
  char            *testClassesStr, *filename;
  const char      *testMatClassname, *testVecClassname;
  char            **testClassesStrArr;
  FILE            *infoFile;

  ierr = PetscInitialize(&argc, &argv,(char *) 0, help);if (ierr) return ierr;

  /*
     Examples on how to call PetscInfo() using different objects with or without arguments, and different communicators.
      - Until PetscInfoDestroy() is called all PetscInfo() behaviour is goverened by command line options, which
        are processed during PetscInitialize().
  */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD, &A));
  CHKERRQ(VecCreate(PETSC_COMM_WORLD, &b));

  CHKERRQ(PetscInfo(A, "Mat info on PETSC_COMM_WORLD with no arguments\n"));
  CHKERRQ(PetscInfo(A, "Mat info on PETSC_COMM_WORLD with 1 argument equal to 1234: %" PetscInt_FMT "\n", testarg));
  CHKERRQ(PetscInfo(b, "Vec info on PETSC_COMM_WORLD with no arguments\n"));
  CHKERRQ(PetscInfo(b, "Vec info on PETSC_COMM_WORLD with 1 argument equal to 1234: %" PetscInt_FMT "\n", testarg));
  CHKERRQ(PetscInfo(NULL, "Sys info on PETSC_COMM_WORLD with no arguments\n"));
  CHKERRQ(PetscInfo(NULL, "Sys info on PETSC_COMM_WORLD with 1 argument equal to 1234: %" PetscInt_FMT "\n", testarg));

  CHKERRQ(MatCreate(PETSC_COMM_SELF, &Aself));
  CHKERRQ(VecCreate(PETSC_COMM_SELF, &bself));

  CHKERRQ(PetscInfo(Aself, "Mat info on PETSC_COMM_SELF with no arguments\n"));
  CHKERRQ(PetscInfo(Aself, "Mat info on PETSC_COMM_SELF with 1 argument equal to 1234: %" PetscInt_FMT "\n", testarg));
  CHKERRQ(PetscInfo(bself, "Vec info on PETSC_COMM_SELF with no arguments\n"));
  CHKERRQ(PetscInfo(bself, "Vec info on PETSC_COMM_SELF with 1 argument equal to 1234: %" PetscInt_FMT "\n", testarg));
  CHKERRQ(PetscInfo(NULL, "Sys info on PETSC_COMM_SELF with no arguments\n"));
  CHKERRQ(PetscInfo(NULL, "Sys info on PETSC_COMM_SELF with 1 argument equal to 1234: %" PetscInt_FMT "\n", testarg));

  CHKERRQ(MatDestroy(&Aself));
  CHKERRQ(VecDestroy(&bself));
  /*
     First retrieve some basic information regarding the classes for which we want to filter
  */
  CHKERRQ(PetscObjectGetClassId((PetscObject) A, &testMatClassid));
  CHKERRQ(PetscObjectGetClassId((PetscObject) b, &testVecClassid));
  /* Sys class has PetscClassId = PETSC_SMALLEST_CLASSID */
  testSysClassid = PETSC_SMALLEST_CLASSID;
  CHKERRQ(PetscObjectGetClassName((PetscObject) A, &testMatClassname));
  CHKERRQ(PetscObjectGetClassName((PetscObject) b, &testVecClassname));

  /*
     Examples on how to use individual PetscInfo() commands.
  */
  CHKERRQ(PetscInfoEnabled(testMatClassid, &isEnabled));
  if (isEnabled) CHKERRQ(PetscInfo(A, "Mat info is enabled\n"));
  CHKERRQ(PetscInfoEnabled(testVecClassid, &isEnabled));
  if (isEnabled) CHKERRQ(PetscInfo(b, "Vec info is enabled\n"));
  CHKERRQ(PetscInfoEnabled(testSysClassid, &isEnabled));
  if (isEnabled) CHKERRQ(PetscInfo(NULL, "Sys info is enabled\n"));

  /* Retrieve filename to append later entries to */
  CHKERRQ(PetscInfoGetFile(&filename, &infoFile));

  /*
     Destroy existing PetscInfo() configuration and reset all internal flags to default values. This allows the user to change filters
     midway through a program.
  */
  CHKERRQ(PetscInfoDestroy());

  /*
     Test if existing filters are reset.
      - Note these should NEVER print.
  */
  CHKERRQ(PetscInfoEnabled(testMatClassid, &isEnabled));
  if (isEnabled) CHKERRQ(PetscInfo(A, "Mat info is enabled after PetscInfoDestroy\n"));
  CHKERRQ(PetscInfoEnabled(testVecClassid, &isEnabled));
  if (isEnabled) CHKERRQ(PetscInfo(b, "Vec info is enabled after PetscInfoDestroy\n"));
  CHKERRQ(PetscInfoEnabled(testSysClassid, &isEnabled));
  if (isEnabled) CHKERRQ(PetscInfo(NULL, "Sys info is enabled after PetscInfoDestroy\n"));

  /*
     Reactivate PetscInfo() printing in one of two ways.
      - First we must reactivate PetscInfo() printing as a whole.
      - Keep in mind that by default ALL classes are allowed to print if PetscInfo() is enabled, so we deactivate
        relevant classes first to demonstrate activation functionality.
  */
  CHKERRQ(PetscInfoAllow(PETSC_TRUE));
  CHKERRQ(PetscInfoSetFile(filename, "a"));
  CHKERRQ(PetscInfoDeactivateClass(testMatClassid));
  CHKERRQ(PetscInfoDeactivateClass(testVecClassid));
  CHKERRQ(PetscInfoDeactivateClass(testSysClassid));

  /*
     Activate PetscInfo() on a per-class basis
  */
  CHKERRQ(PetscInfoActivateClass(testMatClassid));
  CHKERRQ(PetscInfo(A, "Mat info is enabled again through PetscInfoActivateClass\n"));
  CHKERRQ(PetscInfoDeactivateClass(testMatClassid));
  CHKERRQ(PetscInfoActivateClass(testVecClassid));
  CHKERRQ(PetscInfo(b, "Vec info is enabled again through PetscInfoActivateClass\n"));
  CHKERRQ(PetscInfoDeactivateClass(testVecClassid));
  CHKERRQ(PetscInfoActivateClass(testSysClassid));
  CHKERRQ(PetscInfo(NULL, "Sys info is enabled again through PetscInfoActivateClass\n"));
  CHKERRQ(PetscInfoDeactivateClass(testVecClassid));

  /*
     Activate PetscInfo() by specifying specific classnames to activate
  */
  CHKERRQ(PetscStrallocpy("mat,vec,sys", &testClassesStr));
  CHKERRQ(PetscStrToArray((const char *)testClassesStr, ',', &numClasses, &testClassesStrArr));
  CHKERRQ(PetscInfoSetClasses(invert, (PetscInt) numClasses, (const char *const *) testClassesStrArr));
  CHKERRQ(PetscInfoProcessClass(testMatClassname, 1, &testMatClassid));
  CHKERRQ(PetscInfoProcessClass(testVecClassname, 1, &testVecClassid));
  CHKERRQ(PetscInfoProcessClass("sys", 1, &testSysClassid));

  CHKERRQ(PetscInfo(A, "Mat info is enabled again through PetscInfoSetClasses\n"));
  CHKERRQ(PetscInfo(b, "Vec info is enabled again through PetscInfoSetClasses\n"));
  CHKERRQ(PetscInfo(NULL, "Sys info is enabled again through PetscInfoSetClasses\n"));

  CHKERRQ(PetscStrToArrayDestroy(numClasses, testClassesStrArr));
  CHKERRQ(PetscFree(testClassesStr));

  /*
     Activate PetscInfo() with an inverted filter selection.
      - Inverting our selection of filters enables PetscInfo() for all classes EXCEPT those specified.
      - Note we must reset PetscInfo() internal flags with PetscInfoDestroy() as invoking PetscInfoProcessClass() locks filters in place.
  */
  CHKERRQ(PetscInfoDestroy());
  CHKERRQ(PetscInfoAllow(PETSC_TRUE));
  CHKERRQ(PetscInfoSetFile(filename, "a"));
  CHKERRQ(PetscStrallocpy("vec,sys", &testClassesStr));
  CHKERRQ(PetscStrToArray((const char *)testClassesStr, ',', &numClasses, &testClassesStrArr));
  invert = PETSC_TRUE;
  CHKERRQ(PetscInfoSetClasses(invert, (PetscInt) numClasses, (const char *const *) testClassesStrArr));
  CHKERRQ(PetscInfoProcessClass(testMatClassname, 1, &testMatClassid));
  CHKERRQ(PetscInfoProcessClass(testVecClassname, 1, &testVecClassid));
  CHKERRQ(PetscInfoProcessClass("sys", 1, &testSysClassid));

  /*
     Here only the Mat() call will successfully print.
  */
  CHKERRQ(PetscInfo(A, "Mat info is enabled again through inverted PetscInfoSetClasses\n"));
  CHKERRQ(PetscInfo(b, "Vec info is enabled again through PetscInfoSetClasses\n"));
  CHKERRQ(PetscInfo(NULL, "Sys info is enabled again through PetscInfoSetClasses\n"));

  CHKERRQ(PetscStrToArrayDestroy(numClasses, testClassesStrArr));
  CHKERRQ(PetscFree(testClassesStr));
  CHKERRQ(PetscFree(filename));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(VecDestroy(&b));
  CHKERRQ(PetscFinalize());
  return ierr;
}

/*TEST

   test:
      requires: defined(PETSC_USE_INFO)
      suffix: 1
      args: -info
      filter: grep -h -ve Running -ve communicator -ve MPI_Comm -ve OpenMP -ve PetscGetHostName -ve PetscDetermineInitialFPTrap -ve libpetscbamg -ve PetscDeviceContext -ve PetscDeviceType

   test:
      requires: defined(PETSC_USE_INFO)
      suffix: 2
      args: -info ex7info.2
      filter: grep -h -ve Running -ve communicator -ve MPI_Comm -ve OpenMP -ve PetscGetHostName -ve PetscDetermineInitialFPTrap -ve libpetscbamg -ve PetscDeviceContext -ve PetscDeviceType "ex7info.2.0"

   test:
      requires: defined(PETSC_USE_INFO)
      suffix: 3
      nsize: 2
      args: -info ex7info.3
      filter: grep -h -ve Running -ve communicator -ve MPI_Comm -ve OpenMP -ve PetscGetHostName  -ve PetscDetermineInitialFPTrap -ve libpetscbamg -ve PetscDeviceContext -ve PetscDeviceType "ex7info.3.0" | sort -b

   test:
      requires: defined(PETSC_USE_INFO)
      suffix: 4
      args: -info :mat,vec:
      filter: grep -h -ve Running -ve communicator -ve MPI_Comm -ve OpenMP -ve PetscGetHostName -ve PetscDetermineInitialFPTrap

   test:
      requires: defined(PETSC_USE_INFO)
      suffix: 5
      args: -info :~sys:
      filter: grep -h  -ve PetscDetermineInitialFPTrap

   test:
      requires: defined(PETSC_USE_INFO)
      suffix: 6
      nsize: 2
      args: -info ex7info.6:mat:self
      filter: grep -h "ex7info.6.0" | sort -b

   test:
      requires: defined(PETSC_USE_INFO)
      suffix: 7
      nsize: 2
      args: -info ex7info.7:mat:~self
      filter: grep -h "ex7info.7.0" | sort -b

TEST*/
