static char help[] = "Orient a mesh in parallel\n\n";

#include <petscdmplex.h>

typedef struct {
  /* Domain and mesh definition */
  PetscBool testPartition; /* Use a fixed partitioning for testing */
  PetscInt  testNum;       /* Labels the different test partitions */
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->testPartition = PETSC_TRUE;
  options->testNum       = 0;

  ierr = PetscOptionsBegin(comm, "", "Meshing Problem Options", "DMPLEX");CHKERRQ(ierr);
  CHKERRQ(PetscOptionsBool("-test_partition", "Use a fixed partition for testing", "ex13.c", options->testPartition, &options->testPartition, NULL));
  CHKERRQ(PetscOptionsBoundedInt("-test_num", "The test partition number", "ex13.c", options->testNum, &options->testNum, NULL,0));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  DM             dmDist = NULL;
  PetscBool      simplex;
  PetscInt       dim;

  PetscFunctionBeginUser;
  CHKERRQ(DMCreate(comm, dm));
  CHKERRQ(DMSetType(*dm, DMPLEX));
  CHKERRQ(DMPlexDistributeSetDefault(*dm, PETSC_FALSE));
  CHKERRQ(DMSetFromOptions(*dm));
  CHKERRQ(DMGetDimension(*dm, &dim));
  CHKERRQ(DMPlexIsSimplex(*dm, &simplex));
  if (user->testPartition) {
    PetscPartitioner part;
    PetscInt        *sizes  = NULL;
    PetscInt        *points = NULL;
    PetscMPIInt      rank, size;

    CHKERRMPI(MPI_Comm_rank(comm, &rank));
    CHKERRMPI(MPI_Comm_size(comm, &size));
    if (rank == 0) {
      if (dim == 2 && simplex && size == 2) {
        switch (user->testNum) {
        case 0: {
          PetscInt triSizes_p2[2]  = {4, 4};
          PetscInt triPoints_p2[8] = {3, 5, 6, 7, 0, 1, 2, 4};

          CHKERRQ(PetscMalloc2(2, &sizes, 8, &points));
          CHKERRQ(PetscArraycpy(sizes,  triSizes_p2, 2));
          CHKERRQ(PetscArraycpy(points, triPoints_p2, 8));
          break;}
        case 1: {
          PetscInt triSizes_p2[2]  = {6, 2};
          PetscInt triPoints_p2[8] = {1, 2, 3, 4, 6, 7, 0, 5};

          CHKERRQ(PetscMalloc2(2, &sizes, 8, &points));
          CHKERRQ(PetscArraycpy(sizes,  triSizes_p2, 2));
          CHKERRQ(PetscArraycpy(points, triPoints_p2, 8));
          break;}
        default:
          SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Could not find matching test number %d for triangular mesh on 2 procs", user->testNum);
        }
      } else if (dim == 2 && simplex && size == 3) {
        PetscInt triSizes_p3[3]  = {3, 3, 2};
        PetscInt triPoints_p3[8] = {1, 2, 4, 3, 6, 7, 0, 5};

        CHKERRQ(PetscMalloc2(3, &sizes, 8, &points));
        CHKERRQ(PetscArraycpy(sizes,  triSizes_p3, 3));
        CHKERRQ(PetscArraycpy(points, triPoints_p3, 8));
      } else if (dim == 2 && !simplex && size == 2) {
        PetscInt quadSizes_p2[2]  = {2, 2};
        PetscInt quadPoints_p2[4] = {2, 3, 0, 1};

        CHKERRQ(PetscMalloc2(2, &sizes, 4, &points));
        CHKERRQ(PetscArraycpy(sizes,  quadSizes_p2, 2));
        CHKERRQ(PetscArraycpy(points, quadPoints_p2, 4));
      } else SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Could not find matching test partition");
    }
    CHKERRQ(DMPlexGetPartitioner(*dm, &part));
    CHKERRQ(PetscPartitionerSetType(part, PETSCPARTITIONERSHELL));
    CHKERRQ(PetscPartitionerShellSetPartition(part, size, sizes, points));
    CHKERRQ(PetscFree2(sizes, points));
  }
  CHKERRQ(DMPlexDistribute(*dm, 0, NULL, &dmDist));
  if (dmDist) {
    CHKERRQ(DMDestroy(dm));
    *dm  = dmDist;
  }
  CHKERRQ(PetscObjectSetName((PetscObject) *dm, simplex ? "Simplicial Mesh" : "Tensor Product Mesh"));
  CHKERRQ(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(0);
}

static PetscErrorCode ScrambleOrientation(DM dm, AppCtx *user)
{
  PetscInt       h, cStart, cEnd, c;

  PetscFunctionBeginUser;
  CHKERRQ(DMPlexGetVTKCellHeight(dm, &h));
  CHKERRQ(DMPlexGetHeightStratum(dm, h, &cStart, &cEnd));
  for (c = cStart; c < cEnd; ++c) {
    /* Could use PetscRand instead */
    if (c%2) CHKERRQ(DMPlexOrientPoint(dm, c, -1));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TestOrientation(DM dm, AppCtx *user)
{
  PetscFunctionBeginUser;
  CHKERRQ(ScrambleOrientation(dm, user));
  CHKERRQ(DMPlexOrient(dm));
  CHKERRQ(DMViewFromOptions(dm, NULL, "-oriented_dm_view"));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  AppCtx         user; /* user-defined work context */
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  CHKERRQ(ProcessOptions(PETSC_COMM_WORLD, &user));
  CHKERRQ(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
  CHKERRQ(TestOrientation(dm, &user));
  CHKERRQ(DMDestroy(&dm));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST
  testset:
    requires: triangle
    args: -dm_coord_space 0 -dm_view ascii::ascii_info_detail -oriented_dm_view ascii::ascii_info_detail -orientation_view

    test:
      suffix: 0
      args: -test_partition 0
    test:
      suffix: 1
      nsize: 2
    test:
      suffix: 2
      nsize: 2
      args: -test_num 1
    test:
      suffix: 3
      nsize: 3
      args: -orientation_view_synchronized

TEST*/
