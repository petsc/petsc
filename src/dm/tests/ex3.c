
static char help[] = "Solves the 1-dimensional wave equation.\n\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petscdraw.h>

int main(int argc, char **argv)
{
  PetscMPIInt  rank, size;
  PetscInt     M = 60, time_steps = 100, localsize, j, i, mybase, myend, width, xbase, *localnodes = NULL;
  DM           da;
  PetscViewer  viewer, viewer_private;
  PetscDraw    draw;
  Vec          local, global;
  PetscScalar *localptr, *globalptr;
  PetscReal    a, h, k;
  PetscBool    flg = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-M", &M, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-time", &time_steps, NULL));
  /*
      Test putting two nodes on each processor, exact last processor gets the rest
  */
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-distribute", &flg, NULL));
  if (flg) {
    PetscCall(PetscMalloc1(size, &localnodes));
    for (i = 0; i < size - 1; i++) localnodes[i] = 2;
    localnodes[size - 1] = M - 2 * (size - 1);
  }

  /* Set up the array */
  PetscCall(DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, M, 1, 1, localnodes, &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(PetscFree(localnodes));
  PetscCall(DMCreateGlobalVector(da, &global));
  PetscCall(DMCreateLocalVector(da, &local));

  /* Set up display to show combined wave graph */
  PetscCall(PetscViewerDrawOpen(PETSC_COMM_WORLD, 0, "Entire Solution", 20, 480, 800, 200, &viewer));
  PetscCall(PetscViewerDrawGetDraw(viewer, 0, &draw));
  PetscCall(PetscDrawSetDoubleBuffer(draw));

  /* determine starting point of each processor */
  PetscCall(VecGetOwnershipRange(global, &mybase, &myend));

  /* set up display to show my portion of the wave */
  xbase = (int)((mybase) * ((800.0 - 4.0 * size) / M) + 4.0 * rank);
  width = (int)((myend - mybase) * 800. / M);
  PetscCall(PetscViewerDrawOpen(PETSC_COMM_SELF, 0, "Local Portion of Solution", xbase, 200, width, 200, &viewer_private));
  PetscCall(PetscViewerDrawGetDraw(viewer_private, 0, &draw));
  PetscCall(PetscDrawSetDoubleBuffer(draw));

  /* Initialize the array */
  PetscCall(VecGetLocalSize(local, &localsize));
  PetscCall(VecGetArray(global, &globalptr));

  for (i = 1; i < localsize - 1; i++) {
    j                = (i - 1) + mybase;
    globalptr[i - 1] = PetscSinReal((PETSC_PI * j * 6) / ((PetscReal)M) + 1.2 * PetscSinReal((PETSC_PI * j * 2) / ((PetscReal)M))) * 2;
  }

  PetscCall(VecRestoreArray(global, &globalptr));

  /* Assign Parameters */
  a = 1.0;
  h = 1.0 / M;
  k = h;

  for (j = 0; j < time_steps; j++) {
    /* Global to Local */
    PetscCall(DMGlobalToLocalBegin(da, global, INSERT_VALUES, local));
    PetscCall(DMGlobalToLocalEnd(da, global, INSERT_VALUES, local));

    /*Extract local array */
    PetscCall(VecGetArray(local, &localptr));
    PetscCall(VecGetArray(global, &globalptr));

    /* Update Locally - Make array of new values */
    /* Note: I don't do anything for the first and last entry */
    for (i = 1; i < localsize - 1; i++) globalptr[i - 1] = .5 * (localptr[i + 1] + localptr[i - 1]) - (k / (2.0 * a * h)) * (localptr[i + 1] - localptr[i - 1]);
    PetscCall(VecRestoreArray(global, &globalptr));
    PetscCall(VecRestoreArray(local, &localptr));

    /* View my part of Wave */
    PetscCall(VecView(global, viewer_private));

    /* View global Wave */
    PetscCall(VecView(global, viewer));
  }

  PetscCall(DMDestroy(&da));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(PetscViewerDestroy(&viewer_private));
  PetscCall(VecDestroy(&local));
  PetscCall(VecDestroy(&global));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

    test:
      nsize: 3
      args: -time 50 -nox
      requires: x

TEST*/
