
static char help[] = "\n\n";

/*
     Demonstrates using DM_BOUNDARY_GHOSTED how to handle a rotated boundary conditions where one edge
    is connected to its immediate neighbor

    Consider the domain (with natural numbering)

     6   7   8
     3   4   5
     0   1   2

    The ghost points along the bottom (directly below the three columns above) should be 0 3 and 6
    while the ghost points along the left side should be 0 1 2

    Note that the ghosted local vectors extend in both the x and y directions so, for example if we have a
    single MPI process the ghosted vector has (in the original natural numbering)

     x  x  x  x  x
     2  6  7  8  x
     1  3  4  5  x
     0  0  1  2  x
     x  0  3  6  x

    where x indicates a location that is not updated by the communication and should be used.

    For this to make sense the number of grid points in the x and y directions must be the same

    This ghost point mapping was suggested by: Wenbo Zhao <zhaowenbo.npic@gmail.com>
*/

#include <petscdm.h>
#include <petscdmda.h>

int main(int argc, char **argv)
{
  PetscInt     M = 6;
  DM           da;
  Vec          local, global, natural;
  PetscInt     i, start, end, *ifrom, x, y, xm, ym;
  PetscScalar *xnatural;
  IS           from, to;
  AO           ao;
  VecScatter   scatter1, scatter2;
  PetscViewer  subviewer;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));

  /* Create distributed array and get vectors */
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED, DMDA_STENCIL_STAR, M, M, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &da));
  PetscCall(DMSetUp(da));
  PetscCall(DMCreateGlobalVector(da, &global));
  PetscCall(DMCreateLocalVector(da, &local));

  /* construct global to local scatter for the left side of the domain to the ghost on the bottom */
  PetscCall(DMDAGetCorners(da, &x, &y, NULL, &xm, &ym, NULL));
  if (!y) { /* only processes on the bottom of the domain fill up the ghost locations */
    PetscCall(ISCreateStride(PETSC_COMM_SELF, xm, 1, 1, &to));
  } else {
    PetscCall(ISCreateStride(PETSC_COMM_SELF, 0, 0, 0, &to));
  }
  PetscCall(PetscMalloc1(xm, &ifrom));
  for (i = x; i < x + xm; i++) ifrom[i - x] = M * i;
  PetscCall(DMDAGetAO(da, &ao));
  PetscCall(AOApplicationToPetsc(ao, xm, ifrom));
  if (!y) {
    PetscCall(ISCreateGeneral(PETSC_COMM_WORLD, xm, ifrom, PETSC_OWN_POINTER, &from));
  } else {
    PetscCall(PetscFree(ifrom));
    PetscCall(ISCreateGeneral(PETSC_COMM_WORLD, 0, NULL, PETSC_COPY_VALUES, &from));
  }
  PetscCall(VecScatterCreate(global, from, local, to, &scatter1));
  PetscCall(ISDestroy(&to));
  PetscCall(ISDestroy(&from));

  /* construct global to local scatter for the bottom side of the domain to the ghost on the right */
  if (!x) { /* only processes on the left side of the domain fill up the ghost locations */
    PetscCall(ISCreateStride(PETSC_COMM_SELF, ym, xm + 2, xm + 2, &to));
  } else {
    PetscCall(ISCreateStride(PETSC_COMM_SELF, 0, 0, 0, &to));
  }
  PetscCall(PetscMalloc1(ym, &ifrom));
  for (i = y; i < y + ym; i++) ifrom[i - y] = i;
  PetscCall(DMDAGetAO(da, &ao));
  PetscCall(AOApplicationToPetsc(ao, ym, ifrom));
  if (!x) {
    PetscCall(ISCreateGeneral(PETSC_COMM_WORLD, ym, ifrom, PETSC_OWN_POINTER, &from));
  } else {
    PetscCall(PetscFree(ifrom));
    PetscCall(ISCreateGeneral(PETSC_COMM_WORLD, 0, NULL, PETSC_COPY_VALUES, &from));
  }
  PetscCall(VecScatterCreate(global, from, local, to, &scatter2));
  PetscCall(ISDestroy(&to));
  PetscCall(ISDestroy(&from));

  /*
     fill the global vector with the natural global numbering for each local entry
     this is only done for testing purposes since it is easy to see if the scatter worked correctly
  */
  PetscCall(DMDACreateNaturalVector(da, &natural));
  PetscCall(VecGetOwnershipRange(natural, &start, &end));
  PetscCall(VecGetArray(natural, &xnatural));
  for (i = start; i < end; i++) xnatural[i - start] = i;
  PetscCall(VecRestoreArray(natural, &xnatural));
  PetscCall(DMDANaturalToGlobalBegin(da, natural, INSERT_VALUES, global));
  PetscCall(DMDANaturalToGlobalEnd(da, natural, INSERT_VALUES, global));
  PetscCall(VecDestroy(&natural));

  /* scatter from global to local */
  PetscCall(VecScatterBegin(scatter1, global, local, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(scatter1, global, local, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterBegin(scatter2, global, local, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(scatter2, global, local, INSERT_VALUES, SCATTER_FORWARD));
  /*
     normally here you would also call
  PetscCall(DMGlobalToLocalBegin(da,global,INSERT_VALUES,local));
  PetscCall(DMGlobalToLocalEnd(da,global,INSERT_VALUES,local));
    to update all the interior ghost cells between neighboring processes.
    We don't do it here since this is only a test of "special" ghost points.
  */

  /* view each local ghosted vector */
  PetscCall(PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD, PETSC_COMM_SELF, &subviewer));
  PetscCall(VecView(local, subviewer));
  PetscCall(PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD, PETSC_COMM_SELF, &subviewer));

  PetscCall(VecScatterDestroy(&scatter1));
  PetscCall(VecScatterDestroy(&scatter2));
  PetscCall(VecDestroy(&local));
  PetscCall(VecDestroy(&global));
  PetscCall(DMDestroy(&da));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

   test:
      suffix: 2
      nsize: 2

   test:
      suffix: 4
      nsize: 4

   test:
      suffix: 9
      nsize: 9

TEST*/
