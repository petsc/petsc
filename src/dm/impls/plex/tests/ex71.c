static char help[] = "Tests for submesh creation\n\n";

#include <petscdmplex.h>
#include <petscsf.h>

/* Submesh of a 2 x 2 mesh using 3 processes.

Local numbering on each rank:

     (6)(16)-(7)(17)-(8)       5--14---6--15---7      (10)(20)(11)(21)(12)
      |       |       |        |       |       |        |       |       |
    (18) (1)(19) (2)(20)      16   0  17   1  18      (22) (2)(23) (3)(24)
      |       |       |        |       |       |        |       |       |
     (5)(15)(11)(22)(12)       4--13-(11)(22)(12)      (9)(19)--6--14---7
      |       |       |        |       |       |        |       |       |
     14   0 (23) (3)(24)     (20) (2)(23) (3)(24)     (18) (1) 15   0  16
      |       |       |        |       |       |        |       |       |
      4--13--(9)(21)(10)      (8)(19)-(9)(21)(10)      (8)(17)--4--13---5

           mesh_0                   mesh_1                   mesh_2

where () represents ghost points. We extract the left 2 cells.
With sanitize_submesh = PETSC_FALSE, we get:

     (4)(11)-(5)               3---9---4
      |       |                |       |
    (12) (1)(13)              10   0  11
      |       |                |       |
     (3)(10) (7)               2---8---7
      |       |                |       |
      9   0 (14)             (13) (1) 14
      |       |                |       |
      2---8--(6)              (5)(12)--6

On the other hand, with sanitize_submesh = PETSC_TRUE, we get:

     (4)(11)-(5)               3---9---4
      |       |                |       |
    (12) (1)(13)              10   0  11
      |       |                |       |
     (3)(10) (7)               2---8---7
      |       |                |       |
      9   0  14              (13) (1)(14)
      |       |                |       |
      2---8---6               (5)(12)-(6)

        submesh_0                submesh_1               submesh_2

as points 15 and 4 of mesh_2 are in the closure of a submesh cell owned by rank 0 (point 0 of submesh_0),
and not in the closure of any submesh cell owned by rank 1.

*/

typedef struct {
  PetscBool ignoreLabelHalo; /* Ignore filter values in the halo. */
  PetscBool sanitizeSubmesh; /* Sanitize submesh. */
} AppCtx;

PetscErrorCode ProcessOptions(AppCtx *options)
{
  PetscFunctionBegin;
  options->ignoreLabelHalo = PETSC_FALSE;
  options->sanitizeSubmesh = PETSC_FALSE;

  PetscOptionsBegin(PETSC_COMM_SELF, "", "Filtering Problem Options", "DMPLEX");
  PetscCall(PetscOptionsBool("-ignore_label_halo", "Ignore filter values in the halo", "ex80.c", options->ignoreLabelHalo, &options->ignoreLabelHalo, NULL));
  PetscCall(PetscOptionsBool("-sanitize_submesh", "Sanitize submesh", "ex80.c", options->sanitizeSubmesh, &options->sanitizeSubmesh, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}
int main(int argc, char **argv)
{
  DM             dm, subdm;
  PetscSF        ownershipTransferSF;
  DMLabel        filter;
  const PetscInt filterValue = 1;
  MPI_Comm       comm;
  PetscMPIInt    size, rank;
  AppCtx         user;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(&user));
  comm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  if (size != 3) {
    PetscCall(PetscPrintf(comm, "This example is specifically designed for comm size == 3.\n"));
    PetscCall(PetscFinalize());
    return 0;
  }
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  {
    DM             pdm;
    const PetscInt faces[2] = {2, 2};
    PetscInt       overlap  = 1;

    PetscCall(DMPlexCreateBoxMesh(comm, 2, PETSC_FALSE, faces, NULL, NULL, NULL, PETSC_TRUE, 0, PETSC_TRUE, &dm));
    {
      PetscPartitioner part;
      PetscInt        *sizes  = NULL;
      PetscInt        *points = NULL;

      if (rank == 0) {
        PetscInt sizes1[3]  = {1, 2, 1};
        PetscInt points1[4] = {0, 2, 3, 1};

        PetscCall(PetscMalloc2(3, &sizes, 4, &points));
        PetscCall(PetscArraycpy(sizes, sizes1, 3));
        PetscCall(PetscArraycpy(points, points1, 4));
      }
      PetscCall(DMPlexGetPartitioner(dm, &part));
      PetscCall(PetscPartitionerSetType(part, PETSCPARTITIONERSHELL));
      PetscCall(PetscPartitionerShellSetPartition(part, size, sizes, points));
      PetscCall(PetscFree2(sizes, points));
    }
    PetscCall(DMSetAdjacency(dm, -1, PETSC_FALSE, PETSC_TRUE));
    PetscCall(DMPlexDistribute(dm, overlap, NULL, &pdm));
    if (pdm) {
      PetscCall(DMDestroy(&dm));
      dm = pdm;
    }
  }
  PetscCall(DMLabelCreate(PETSC_COMM_SELF, "filter", &filter));
  switch (rank) {
  case 0:
    PetscCall(DMLabelSetValue(filter, 0, filterValue));
    PetscCall(DMLabelSetValue(filter, 1, filterValue));
    break;
  case 1:
    PetscCall(DMLabelSetValue(filter, 0, filterValue));
    PetscCall(DMLabelSetValue(filter, 2, filterValue));
    break;
  case 2:
    break;
  }
  PetscCall(PetscObjectSetName((PetscObject)dm, "Example_DM"));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));
  PetscCall(DMPlexFilter(dm, filter, filterValue, user.ignoreLabelHalo, user.sanitizeSubmesh, &ownershipTransferSF, &subdm));
  PetscCall(DMLabelDestroy(&filter));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscObjectSetName((PetscObject)subdm, "Example_SubDM"));
  PetscCall(DMViewFromOptions(subdm, NULL, "-dm_view"));
  PetscCall(DMDestroy(&subdm));
  PetscCall(PetscObjectSetName((PetscObject)ownershipTransferSF, "Example_Ownership_Transfer_SF"));
  PetscCall(PetscSFView(ownershipTransferSF, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscSFDestroy(&ownershipTransferSF));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    nsize: 3
    args: -dm_view ascii::ascii_info_detail

    test:
      suffix: 0
      args:

    test:
      suffix: 1
      args: -sanitize_submesh

    test:
      suffix: 2
      args: -ignore_label_halo

TEST*/
