static char help[] = "Test PetscSFFCompose against some corner cases \n\n";

#include <petscsf.h>

int main(int argc, char **argv)
{
  PetscMPIInt  size;
  PetscSF      sfA0, sfA1, sfA2, sfB;
  PetscInt     nroots, nleaves;
  PetscInt    *ilocalA0, *ilocalA1, *ilocalA2, *ilocalB;
  PetscSFNode *iremoteA0, *iremoteA1, *iremoteA2, *iremoteB;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_USER, "Only coded for one MPI process");
  PetscCall(PetscSFCreate(PETSC_COMM_WORLD, &sfA0));
  PetscCall(PetscSFCreate(PETSC_COMM_WORLD, &sfA1));
  PetscCall(PetscSFCreate(PETSC_COMM_WORLD, &sfA2));
  PetscCall(PetscSFCreate(PETSC_COMM_WORLD, &sfB));
  /* sfA0 */
  nroots  = 1;
  nleaves = 0;
  PetscCall(PetscMalloc1(nleaves, &ilocalA0));
  PetscCall(PetscMalloc1(nleaves, &iremoteA0));
  PetscCall(PetscSFSetGraph(sfA0, nroots, nleaves, ilocalA0, PETSC_OWN_POINTER, iremoteA0, PETSC_OWN_POINTER));
  PetscCall(PetscSFSetUp(sfA0));
  PetscCall(PetscObjectSetName((PetscObject)sfA0, "sfA0"));
  PetscCall(PetscSFView(sfA0, NULL));
  /* sfA1 */
  nroots  = 1;
  nleaves = 1;
  PetscCall(PetscMalloc1(nleaves, &ilocalA1));
  PetscCall(PetscMalloc1(nleaves, &iremoteA1));
  ilocalA1[0]        = 1;
  iremoteA1[0].rank  = 0;
  iremoteA1[0].index = 0;
  PetscCall(PetscSFSetGraph(sfA1, nroots, nleaves, ilocalA1, PETSC_OWN_POINTER, iremoteA1, PETSC_OWN_POINTER));
  PetscCall(PetscSFSetUp(sfA1));
  PetscCall(PetscObjectSetName((PetscObject)sfA1, "sfA1"));
  PetscCall(PetscSFView(sfA1, NULL));
  /* sfA2 */
  nroots  = 1;
  nleaves = 1;
  PetscCall(PetscMalloc1(nleaves, &ilocalA2));
  PetscCall(PetscMalloc1(nleaves, &iremoteA2));
  ilocalA2[0]        = 0;
  iremoteA2[0].rank  = 0;
  iremoteA2[0].index = 0;
  PetscCall(PetscSFSetGraph(sfA2, nroots, nleaves, ilocalA2, PETSC_OWN_POINTER, iremoteA2, PETSC_OWN_POINTER));
  PetscCall(PetscSFSetUp(sfA2));
  PetscCall(PetscObjectSetName((PetscObject)sfA2, "sfA2"));
  PetscCall(PetscSFView(sfA2, NULL));
  /* sfB */
  nroots  = 2;
  nleaves = 2;
  PetscCall(PetscMalloc1(nleaves, &ilocalB));
  PetscCall(PetscMalloc1(nleaves, &iremoteB));
  ilocalB[0]        = 100;
  iremoteB[0].rank  = 0;
  iremoteB[0].index = 0;
  ilocalB[1]        = 101;
  iremoteB[1].rank  = 0;
  iremoteB[1].index = 1;
  PetscCall(PetscSFSetGraph(sfB, nroots, nleaves, ilocalB, PETSC_OWN_POINTER, iremoteB, PETSC_OWN_POINTER));
  PetscCall(PetscSFSetUp(sfB));
  PetscCall(PetscObjectSetName((PetscObject)sfB, "sfB"));
  PetscCall(PetscSFView(sfB, NULL));
  /* Test 0 */
  {
    PetscSF sfC;

    PetscCall(PetscSFCompose(sfA0, sfB, &sfC));
    PetscCall(PetscObjectSetName((PetscObject)sfC, "PetscSFCompose(sfA0, sfB)"));
    PetscCall(PetscSFView(sfC, NULL));
    PetscCall(PetscSFDestroy(&sfC));
  }
  /* Test 1 */
  {
    PetscSF sfC;

    PetscCall(PetscSFCompose(sfA1, sfB, &sfC));
    PetscCall(PetscObjectSetName((PetscObject)sfC, "PetscSFCompose(sfA1, sfB)"));
    PetscCall(PetscSFView(sfC, NULL));
    PetscCall(PetscSFDestroy(&sfC));
  }
  /* Test 2 */
  {
    PetscSF sfC;

    PetscCall(PetscSFCompose(sfA2, sfB, &sfC));
    PetscCall(PetscObjectSetName((PetscObject)sfC, "PetscSFCompose(sfA2, sfB)"));
    PetscCall(PetscSFView(sfC, NULL));
    PetscCall(PetscSFDestroy(&sfC));
  }
  PetscCall(PetscSFDestroy(&sfA0));
  PetscCall(PetscSFDestroy(&sfA1));
  PetscCall(PetscSFDestroy(&sfA2));
  PetscCall(PetscSFDestroy(&sfB));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
     suffix: 0

TEST*/
