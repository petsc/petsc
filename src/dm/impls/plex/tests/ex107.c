static char help[] = "Tests DMPlexPointQueue growth when its entries wrap around the end of its storage\n\n";

#include <petscdmplex.h>

static PetscErrorCode TestQueue(PetscInt size)
{
  DMPlexPointQueue queue;
  PetscInt         p;

  PetscFunctionBeginUser;
  PetscCall(DMPlexPointQueueCreate(size, &queue));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "Initial size %" PetscInt_FMT ":", size));
  for (PetscInt i = 0; i < 4; ++i) PetscCall(DMPlexPointQueueEnqueue(queue, i));
  for (PetscInt i = 0; i < 2; ++i) {
    PetscCall(DMPlexPointQueueDequeue(queue, &p));
    PetscCall(PetscPrintf(PETSC_COMM_SELF, " %" PetscInt_FMT, p));
  }
  // 2 and 3 are at positions 2 and 3 of a storage of size 4, so 4 and 5 wrap to positions 0 and 1 and 6 makes the full queue grow
  for (PetscInt i = 4; i < 7; ++i) PetscCall(DMPlexPointQueueEnqueue(queue, i));
  while (!DMPlexPointQueueEmpty(queue)) {
    PetscCall(DMPlexPointQueueDequeue(queue, &p));
    PetscCall(PetscPrintf(PETSC_COMM_SELF, " %" PetscInt_FMT, p));
  }
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "\n"));
  PetscCall(DMPlexPointQueueDestroy(&queue));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(TestQueue(0));
  PetscCall(TestQueue(4));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:

TEST*/
