static char help[] = "Tests for DMPlexMarkBoundaryFaces()\n\n";

#include <petscdmplex.h>
#include <petscsf.h>

typedef struct {
  PetscInt overlap; /* The overlap size used when partitioning */
} AppCtx;

PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscFunctionBegin;
  options->overlap = 0;

  PetscOptionsBegin(comm, "", "Options for DMPlexMarkBoundaryFaces() problem", "DMPLEX");
  PetscCall(PetscOptionsBoundedInt("-overlap", "The overlap size used", "ex70.c", options->overlap, &options->overlap, NULL, 0));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscFunctionBeginUser;
  {
    const PetscInt faces[2] = {2, 2};

    PetscCall(DMPlexCreateBoxMesh(comm, 2, PETSC_TRUE, faces, NULL, NULL, NULL, PETSC_TRUE, 0, PETSC_TRUE, dm));
  }
  {
    PetscPartitioner part;
    PetscInt        *sizes  = NULL;
    PetscInt        *points = NULL;
    PetscMPIInt      rank, size;

    PetscCallMPI(MPI_Comm_rank(comm, &rank));
    PetscCallMPI(MPI_Comm_size(comm, &size));
    if (rank == 0) {
      PetscInt sizes1[2]  = {4, 4};
      PetscInt points1[8] = {3, 5, 6, 7, 0, 1, 2, 4};

      PetscCall(PetscMalloc2(2, &sizes, 8, &points));
      PetscCall(PetscArraycpy(sizes, sizes1, 2));
      PetscCall(PetscArraycpy(points, points1, 8));
    }
    PetscCall(DMPlexGetPartitioner(*dm, &part));
    PetscCall(PetscPartitionerSetType(part, PETSCPARTITIONERSHELL));
    PetscCall(PetscPartitionerShellSetPartition(part, size, sizes, points));
    PetscCall(PetscFree2(sizes, points));
  }
  {
    DM dmDist = NULL;

    PetscCall(DMSetAdjacency(*dm, -1, PETSC_FALSE, PETSC_TRUE));
    PetscCall(DMPlexDistribute(*dm, user->overlap, NULL, &dmDist));
    if (dmDist) {
      PetscCall(DMDestroy(dm));
      *dm = dmDist;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM          dm;
  DMLabel     extLabel;
  MPI_Comm    comm;
  PetscMPIInt size;
  AppCtx      user;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  if (size != 2) {
    PetscCall(PetscPrintf(comm, "This example is specifically designed for comm size == 2.\n"));
    PetscCall(PetscFinalize());
    return 0;
  }
  PetscCall(ProcessOptions(comm, &user));
  PetscCall(CreateMesh(comm, &user, &dm));
  PetscCall(DMCreateLabel(dm, "exterior_facets"));
  PetscCall(DMGetLabel(dm, "exterior_facets", &extLabel));
  PetscCall(DMPlexMarkBoundaryFaces(dm, 1, extLabel));
  PetscCall(PetscObjectSetName((PetscObject)dm, "Example_DM"));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    nsize: 2
    requires: triangle
    args: -overlap {{0 1}separate output} -dm_view ascii::ascii_info_detail

TEST*/
