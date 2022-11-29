static char help[] = "Example use of PetscInfo() as a configurable informative logging or warning tool\n";

#include <petscsys.h>
#include <petscmat.h>
#include <petscvec.h>

int main(int argc, char **argv)
{
  Mat A, Aself;
  Vec b, bself;
#if defined(PETSC_USE_INFO)
  PetscInt testarg = 1234;
#endif
  int          numClasses;
  PetscClassId testMatClassid, testVecClassid, testSysClassid;
  PetscBool    isEnabled = PETSC_FALSE, invert = PETSC_FALSE;
  char        *testClassesStr, *filename;
  const char  *testMatClassname, *testVecClassname;
  char       **testClassesStrArr;
  FILE        *infoFile;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));

  /*
     Examples on how to call PetscInfo() using different objects with or without arguments, and different communicators.
      - Until PetscInfoDestroy() is called all PetscInfo() behaviour is goverened by command line options, which
        are processed during PetscInitialize().
  */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(VecCreate(PETSC_COMM_WORLD, &b));

  PetscCall(PetscInfo(A, "Mat info on PETSC_COMM_WORLD with no arguments\n"));
  PetscCall(PetscInfo(A, "Mat info on PETSC_COMM_WORLD with 1 argument equal to 1234: %" PetscInt_FMT "\n", testarg));
  PetscCall(PetscInfo(b, "Vec info on PETSC_COMM_WORLD with no arguments\n"));
  PetscCall(PetscInfo(b, "Vec info on PETSC_COMM_WORLD with 1 argument equal to 1234: %" PetscInt_FMT "\n", testarg));
  PetscCall(PetscInfo(NULL, "Sys info on PETSC_COMM_WORLD with no arguments\n"));
  PetscCall(PetscInfo(NULL, "Sys info on PETSC_COMM_WORLD with 1 argument equal to 1234: %" PetscInt_FMT "\n", testarg));

  PetscCall(MatCreate(PETSC_COMM_SELF, &Aself));
  PetscCall(VecCreate(PETSC_COMM_SELF, &bself));

  PetscCall(PetscInfo(Aself, "Mat info on PETSC_COMM_SELF with no arguments\n"));
  PetscCall(PetscInfo(Aself, "Mat info on PETSC_COMM_SELF with 1 argument equal to 1234: %" PetscInt_FMT "\n", testarg));
  PetscCall(PetscInfo(bself, "Vec info on PETSC_COMM_SELF with no arguments\n"));
  PetscCall(PetscInfo(bself, "Vec info on PETSC_COMM_SELF with 1 argument equal to 1234: %" PetscInt_FMT "\n", testarg));
  PetscCall(PetscInfo(NULL, "Sys info on PETSC_COMM_SELF with no arguments\n"));
  PetscCall(PetscInfo(NULL, "Sys info on PETSC_COMM_SELF with 1 argument equal to 1234: %" PetscInt_FMT "\n", testarg));

  PetscCall(MatDestroy(&Aself));
  PetscCall(VecDestroy(&bself));
  /*
     First retrieve some basic information regarding the classes for which we want to filter
  */
  PetscCall(PetscObjectGetClassId((PetscObject)A, &testMatClassid));
  PetscCall(PetscObjectGetClassId((PetscObject)b, &testVecClassid));
  /* Sys class has PetscClassId = PETSC_SMALLEST_CLASSID */
  testSysClassid = PETSC_SMALLEST_CLASSID;
  PetscCall(PetscObjectGetClassName((PetscObject)A, &testMatClassname));
  PetscCall(PetscObjectGetClassName((PetscObject)b, &testVecClassname));

  /*
     Examples on how to use individual PetscInfo() commands.
  */
  PetscCall(PetscInfoEnabled(testMatClassid, &isEnabled));
  if (isEnabled) PetscCall(PetscInfo(A, "Mat info is enabled\n"));
  PetscCall(PetscInfoEnabled(testVecClassid, &isEnabled));
  if (isEnabled) PetscCall(PetscInfo(b, "Vec info is enabled\n"));
  PetscCall(PetscInfoEnabled(testSysClassid, &isEnabled));
  if (isEnabled) PetscCall(PetscInfo(NULL, "Sys info is enabled\n"));

  /* Retrieve filename to append later entries to */
  PetscCall(PetscInfoGetFile(&filename, &infoFile));

  /*
     Destroy existing PetscInfo() configuration and reset all internal flags to default values. This allows the user to change filters
     midway through a program.
  */
  PetscCall(PetscInfoDestroy());

  /*
     Test if existing filters are reset.
      - Note these should NEVER print.
  */
  PetscCall(PetscInfoEnabled(testMatClassid, &isEnabled));
  if (isEnabled) PetscCall(PetscInfo(A, "Mat info is enabled after PetscInfoDestroy\n"));
  PetscCall(PetscInfoEnabled(testVecClassid, &isEnabled));
  if (isEnabled) PetscCall(PetscInfo(b, "Vec info is enabled after PetscInfoDestroy\n"));
  PetscCall(PetscInfoEnabled(testSysClassid, &isEnabled));
  if (isEnabled) PetscCall(PetscInfo(NULL, "Sys info is enabled after PetscInfoDestroy\n"));

  /*
     Reactivate PetscInfo() printing in one of two ways.
      - First we must reactivate PetscInfo() printing as a whole.
      - Keep in mind that by default ALL classes are allowed to print if PetscInfo() is enabled, so we deactivate
        relevant classes first to demonstrate activation functionality.
  */
  PetscCall(PetscInfoAllow(PETSC_TRUE));
  PetscCall(PetscInfoSetFile(filename, "a"));
  PetscCall(PetscInfoDeactivateClass(testMatClassid));
  PetscCall(PetscInfoDeactivateClass(testVecClassid));
  PetscCall(PetscInfoDeactivateClass(testSysClassid));

  /*
     Activate PetscInfo() on a per-class basis
  */
  PetscCall(PetscInfoActivateClass(testMatClassid));
  PetscCall(PetscInfo(A, "Mat info is enabled again through PetscInfoActivateClass\n"));
  PetscCall(PetscInfoDeactivateClass(testMatClassid));
  PetscCall(PetscInfoActivateClass(testVecClassid));
  PetscCall(PetscInfo(b, "Vec info is enabled again through PetscInfoActivateClass\n"));
  PetscCall(PetscInfoDeactivateClass(testVecClassid));
  PetscCall(PetscInfoActivateClass(testSysClassid));
  PetscCall(PetscInfo(NULL, "Sys info is enabled again through PetscInfoActivateClass\n"));
  PetscCall(PetscInfoDeactivateClass(testVecClassid));

  /*
     Activate PetscInfo() by specifying specific classnames to activate
  */
  PetscCall(PetscStrallocpy("mat,vec,sys", &testClassesStr));
  PetscCall(PetscStrToArray((const char *)testClassesStr, ',', &numClasses, &testClassesStrArr));
  PetscCall(PetscInfoSetClasses(invert, (PetscInt)numClasses, (const char *const *)testClassesStrArr));
  PetscCall(PetscInfoProcessClass(testMatClassname, 1, &testMatClassid));
  PetscCall(PetscInfoProcessClass(testVecClassname, 1, &testVecClassid));
  PetscCall(PetscInfoProcessClass("sys", 1, &testSysClassid));

  PetscCall(PetscInfo(A, "Mat info is enabled again through PetscInfoSetClasses\n"));
  PetscCall(PetscInfo(b, "Vec info is enabled again through PetscInfoSetClasses\n"));
  PetscCall(PetscInfo(NULL, "Sys info is enabled again through PetscInfoSetClasses\n"));

  PetscCall(PetscStrToArrayDestroy(numClasses, testClassesStrArr));
  PetscCall(PetscFree(testClassesStr));

  /*
     Activate PetscInfo() with an inverted filter selection.
      - Inverting our selection of filters enables PetscInfo() for all classes EXCEPT those specified.
      - Note we must reset PetscInfo() internal flags with PetscInfoDestroy() as invoking PetscInfoProcessClass() locks filters in place.
  */
  PetscCall(PetscInfoDestroy());
  PetscCall(PetscInfoAllow(PETSC_TRUE));
  PetscCall(PetscInfoSetFile(filename, "a"));
  PetscCall(PetscStrallocpy("vec,sys", &testClassesStr));
  PetscCall(PetscStrToArray((const char *)testClassesStr, ',', &numClasses, &testClassesStrArr));
  invert = PETSC_TRUE;
  PetscCall(PetscInfoSetClasses(invert, (PetscInt)numClasses, (const char *const *)testClassesStrArr));
  PetscCall(PetscInfoProcessClass(testMatClassname, 1, &testMatClassid));
  PetscCall(PetscInfoProcessClass(testVecClassname, 1, &testVecClassid));
  PetscCall(PetscInfoProcessClass("sys", 1, &testSysClassid));

  /*
     Here only the Mat() call will successfully print.
  */
  PetscCall(PetscInfo(A, "Mat info is enabled again through inverted PetscInfoSetClasses\n"));
  PetscCall(PetscInfo(b, "Vec info is enabled again through PetscInfoSetClasses\n"));
  PetscCall(PetscInfo(NULL, "Sys info is enabled again through PetscInfoSetClasses\n"));

  PetscCall(PetscStrToArrayDestroy(numClasses, testClassesStrArr));
  PetscCall(PetscFree(testClassesStr));
  PetscCall(PetscFree(filename));
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&b));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      requires: defined(PETSC_USE_INFO)
      suffix: 1
      args: -info
      filter: grep -h -ve Running -ve communicator -ve MPI_Comm -ve OpenMP -ve PetscGetHostName -ve PetscDetermineInitialFPTrap -ve libpetscbamg -ve PetscDeviceContext -ve PetscDeviceType -ve PetscDeviceInitializeTypeFromOptions_Private

   test:
      requires: defined(PETSC_USE_INFO)
      suffix: 2
      args: -info ex7info.2
      filter: grep -h -ve Running -ve communicator -ve MPI_Comm -ve OpenMP -ve PetscGetHostName -ve PetscDetermineInitialFPTrap -ve libpetscbamg -ve PetscDeviceContext -ve PetscDeviceType -ve PetscDeviceInitializeTypeFromOptions_Private "ex7info.2.0"

   test:
      requires: defined(PETSC_USE_INFO)
      suffix: 3
      nsize: 2
      args: -info ex7info.3
      filter: grep -h -ve Running -ve communicator -ve MPI_Comm -ve OpenMP -ve PetscGetHostName  -ve PetscDetermineInitialFPTrap -ve libpetscbamg -ve PetscDeviceContext -ve PetscDeviceType -ve PetscDeviceInitializeTypeFromOptions_Private "ex7info.3.0" | sort -b

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
