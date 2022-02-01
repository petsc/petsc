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
  ierr = MatCreate(PETSC_COMM_WORLD, &A);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD, &b);CHKERRQ(ierr);

  ierr = PetscInfo(A, "Mat info on PETSC_COMM_WORLD with no arguments\n");CHKERRQ(ierr);
  ierr = PetscInfo1(A, "Mat info on PETSC_COMM_WORLD with 1 argument equal to 1234: %" PetscInt_FMT "\n", testarg);CHKERRQ(ierr);
  ierr = PetscInfo(b, "Vec info on PETSC_COMM_WORLD with no arguments\n");CHKERRQ(ierr);
  ierr = PetscInfo1(b, "Vec info on PETSC_COMM_WORLD with 1 argument equal to 1234: %" PetscInt_FMT "\n", testarg);CHKERRQ(ierr);
  ierr = PetscInfo(NULL, "Sys info on PETSC_COMM_WORLD with no arguments\n");CHKERRQ(ierr);
  ierr = PetscInfo1(NULL, "Sys info on PETSC_COMM_WORLD with 1 argument equal to 1234: %" PetscInt_FMT "\n", testarg);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_SELF, &Aself);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_SELF, &bself);CHKERRQ(ierr);

  ierr = PetscInfo(Aself, "Mat info on PETSC_COMM_SELF with no arguments\n");CHKERRQ(ierr);
  ierr = PetscInfo1(Aself, "Mat info on PETSC_COMM_SELF with 1 argument equal to 1234: %" PetscInt_FMT "\n", testarg);CHKERRQ(ierr);
  ierr = PetscInfo(bself, "Vec info on PETSC_COMM_SELF with no arguments\n");CHKERRQ(ierr);
  ierr = PetscInfo1(bself, "Vec info on PETSC_COMM_SELF with 1 argument equal to 1234: %" PetscInt_FMT "\n", testarg);CHKERRQ(ierr);
  ierr = PetscInfo(NULL, "Sys info on PETSC_COMM_SELF with no arguments\n");CHKERRQ(ierr);
  ierr = PetscInfo1(NULL, "Sys info on PETSC_COMM_SELF with 1 argument equal to 1234: %" PetscInt_FMT "\n", testarg);CHKERRQ(ierr);

  ierr = MatDestroy(&Aself);CHKERRQ(ierr);
  ierr = VecDestroy(&bself);CHKERRQ(ierr);
  /*
     First retrieve some basic information regarding the classes for which we want to filter
  */
  ierr = PetscObjectGetClassId((PetscObject) A, &testMatClassid);CHKERRQ(ierr);
  ierr = PetscObjectGetClassId((PetscObject) b, &testVecClassid);CHKERRQ(ierr);
  /* Sys class has PetscClassId = PETSC_SMALLEST_CLASSID */
  testSysClassid = PETSC_SMALLEST_CLASSID;
  ierr = PetscObjectGetClassName((PetscObject) A, &testMatClassname);CHKERRQ(ierr);
  ierr = PetscObjectGetClassName((PetscObject) b, &testVecClassname);CHKERRQ(ierr);

  /*
     Examples on how to use individual PetscInfo() commands.
  */
  ierr = PetscInfoEnabled(testMatClassid, &isEnabled);CHKERRQ(ierr);
  if (isEnabled) { ierr = PetscInfo(A, "Mat info is enabled\n");CHKERRQ(ierr);}
  ierr = PetscInfoEnabled(testVecClassid, &isEnabled);CHKERRQ(ierr);
  if (isEnabled) { ierr = PetscInfo(b, "Vec info is enabled\n");CHKERRQ(ierr);}
  ierr = PetscInfoEnabled(testSysClassid, &isEnabled);CHKERRQ(ierr);
  if (isEnabled) { ierr = PetscInfo(NULL, "Sys info is enabled\n");CHKERRQ(ierr);}

  /* Retrieve filename to append later entries to */
  ierr = PetscInfoGetFile(&filename, &infoFile);CHKERRQ(ierr);

  /*
     Destroy existing PetscInfo() configuration and reset all internal flags to default values. This allows the user to change filters
     midway through a program.
  */
  ierr = PetscInfoDestroy();CHKERRQ(ierr);

  /*
     Test if existing filters are reset.
      - Note these should NEVER print.
  */
  ierr = PetscInfoEnabled(testMatClassid, &isEnabled);CHKERRQ(ierr);
  if (isEnabled) { ierr = PetscInfo(A, "Mat info is enabled after PetscInfoDestroy\n");CHKERRQ(ierr);}
  ierr = PetscInfoEnabled(testVecClassid, &isEnabled);CHKERRQ(ierr);
  if (isEnabled) { ierr = PetscInfo(b, "Vec info is enabled after PetscInfoDestroy\n");CHKERRQ(ierr);}
  ierr = PetscInfoEnabled(testSysClassid, &isEnabled);CHKERRQ(ierr);
  if (isEnabled) { ierr = PetscInfo(NULL, "Sys info is enabled after PetscInfoDestroy\n");CHKERRQ(ierr);}

  /*
     Reactivate PetscInfo() printing in one of two ways.
      - First we must reactivate PetscInfo() printing as a whole.
      - Keep in mind that by default ALL classes are allowed to print if PetscInfo() is enabled, so we deactivate
        relevant classes first to demonstrate activation functionality.
  */
  ierr = PetscInfoAllow(PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscInfoSetFile(filename, "a");CHKERRQ(ierr);
  ierr = PetscInfoDeactivateClass(testMatClassid);CHKERRQ(ierr);
  ierr = PetscInfoDeactivateClass(testVecClassid);CHKERRQ(ierr);
  ierr = PetscInfoDeactivateClass(testSysClassid);CHKERRQ(ierr);

  /*
     Activate PetscInfo() on a per-class basis
  */
  ierr = PetscInfoActivateClass(testMatClassid);CHKERRQ(ierr);
  ierr = PetscInfo(A, "Mat info is enabled again through PetscInfoActivateClass\n");CHKERRQ(ierr);
  ierr = PetscInfoDeactivateClass(testMatClassid);CHKERRQ(ierr);
  ierr = PetscInfoActivateClass(testVecClassid);CHKERRQ(ierr);
  ierr = PetscInfo(b, "Vec info is enabled again through PetscInfoActivateClass\n");CHKERRQ(ierr);
  ierr = PetscInfoDeactivateClass(testVecClassid);CHKERRQ(ierr);
  ierr = PetscInfoActivateClass(testSysClassid);CHKERRQ(ierr);
  ierr = PetscInfo(NULL, "Sys info is enabled again through PetscInfoActivateClass\n");CHKERRQ(ierr);
  ierr = PetscInfoDeactivateClass(testVecClassid);CHKERRQ(ierr);

  /*
     Activate PetscInfo() by specifying specific classnames to activate
  */
  ierr = PetscStrallocpy("mat,vec,sys", &testClassesStr);CHKERRQ(ierr);
  ierr = PetscStrToArray((const char *)testClassesStr, ',', &numClasses, &testClassesStrArr);CHKERRQ(ierr);
  ierr = PetscInfoSetClasses(invert, (PetscInt) numClasses, (const char *const *) testClassesStrArr);CHKERRQ(ierr);
  ierr = PetscInfoProcessClass(testMatClassname, 1, &testMatClassid);CHKERRQ(ierr);
  ierr = PetscInfoProcessClass(testVecClassname, 1, &testVecClassid);CHKERRQ(ierr);
  ierr = PetscInfoProcessClass("sys", 1, &testSysClassid);CHKERRQ(ierr);

  ierr = PetscInfo(A, "Mat info is enabled again through PetscInfoSetClasses\n");CHKERRQ(ierr);
  ierr = PetscInfo(b, "Vec info is enabled again through PetscInfoSetClasses\n");CHKERRQ(ierr);
  ierr = PetscInfo(NULL, "Sys info is enabled again through PetscInfoSetClasses\n");CHKERRQ(ierr);

  ierr = PetscStrToArrayDestroy(numClasses, testClassesStrArr);CHKERRQ(ierr);
  ierr = PetscFree(testClassesStr);CHKERRQ(ierr);

  /*
     Activate PetscInfo() with an inverted filter selection.
      - Inverting our selection of filters enables PetscInfo() for all classes EXCEPT those specified.
      - Note we must reset PetscInfo() internal flags with PetscInfoDestroy() as invoking PetscInfoProcessClass() locks filters in place.
  */
  ierr = PetscInfoDestroy();CHKERRQ(ierr);
  ierr = PetscInfoAllow(PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscInfoSetFile(filename, "a");CHKERRQ(ierr);
  ierr = PetscStrallocpy("vec,sys", &testClassesStr);CHKERRQ(ierr);
  ierr = PetscStrToArray((const char *)testClassesStr, ',', &numClasses, &testClassesStrArr);CHKERRQ(ierr);
  invert = PETSC_TRUE;
  ierr = PetscInfoSetClasses(invert, (PetscInt) numClasses, (const char *const *) testClassesStrArr);CHKERRQ(ierr);
  ierr = PetscInfoProcessClass(testMatClassname, 1, &testMatClassid);CHKERRQ(ierr);
  ierr = PetscInfoProcessClass(testVecClassname, 1, &testVecClassid);CHKERRQ(ierr);
  ierr = PetscInfoProcessClass("sys", 1, &testSysClassid);CHKERRQ(ierr);

  /*
     Here only the Mat() call will successfully print.
  */
  ierr = PetscInfo(A, "Mat info is enabled again through inverted PetscInfoSetClasses\n");CHKERRQ(ierr);
  ierr = PetscInfo(b, "Vec info is enabled again through PetscInfoSetClasses\n");CHKERRQ(ierr);
  ierr = PetscInfo(NULL, "Sys info is enabled again through PetscInfoSetClasses\n");CHKERRQ(ierr);

  ierr = PetscStrToArrayDestroy(numClasses, testClassesStrArr);CHKERRQ(ierr);
  ierr = PetscFree(testClassesStr);CHKERRQ(ierr);
  ierr = PetscFree(filename);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
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
