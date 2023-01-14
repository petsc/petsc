static char help[] = "Test PetscSFSetGraphFromCoordinates()\n\n";

#include <petscsf.h>

int main(int argc, char **argv)
{
  PetscSF     sf;
  MPI_Comm    comm;
  PetscMPIInt rank, size;
  PetscInt    height = 2, width = 3, nroots = height, nleaves, dim = 2;
  PetscReal  *rootcoords, *leafcoords;
  PetscViewer viewer;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &size));

  nleaves = (width - (rank == 0) - (rank == size - 1)) * height;
  PetscCall(PetscMalloc2(nroots * dim, &rootcoords, nleaves * dim, &leafcoords));
  for (PetscInt i = 0; i < height; i++) {
    rootcoords[i * dim + 0] = 0.1 * rank;
    rootcoords[i * dim + 1] = 1. * i;
    for (PetscInt j = 0, l = 0; j < width; j++) {
      if (rank + j - 1 < 0 || rank + j - 1 >= size) continue;
      leafcoords[(i * nleaves / height + l) * dim + 0] = 0.1 * (rank + j - 1);
      leafcoords[(i * nleaves / height + l) * dim + 1] = 1. * i;
      l++;
    }
  }
  viewer = PETSC_VIEWER_STDOUT_WORLD;
  PetscCall(PetscPrintf(comm, "Roots by rank\n"));
  PetscCall(PetscRealView(nroots * dim, rootcoords, viewer));
  PetscCall(PetscPrintf(comm, "Leaves by rank\n"));
  PetscCall(PetscRealView(nleaves * dim, leafcoords, viewer));

  PetscCall(PetscSFCreate(comm, &sf));
  PetscCall(PetscSFSetGraphFromCoordinates(sf, nroots, nleaves, dim, 1e-10, rootcoords, leafcoords));

  PetscCall(PetscSFViewFromOptions(sf, NULL, "-sf_view"));
  PetscCall(PetscFree2(rootcoords, leafcoords));
  PetscCall(PetscSFDestroy(&sf));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
  test:
    suffix: 1
    nsize: 3
    args: -sf_view
TEST*/
