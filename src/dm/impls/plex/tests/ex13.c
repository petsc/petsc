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
  ierr = PetscOptionsBool("-test_partition", "Use a fixed partition for testing", "ex13.c", options->testPartition, &options->testPartition, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBoundedInt("-test_num", "The test partition number", "ex13.c", options->testNum, &options->testNum, NULL,0);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  DM             dmDist = NULL;
  PetscBool      simplex;
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMGetDimension(*dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexIsSimplex(*dm, &simplex);CHKERRQ(ierr);
  if (user->testPartition) {
    PetscPartitioner part;
    PetscInt        *sizes  = NULL;
    PetscInt        *points = NULL;
    PetscMPIInt      rank, size;

    ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
    ierr = MPI_Comm_size(comm, &size);CHKERRMPI(ierr);
    if (rank == 0) {
      if (dim == 2 && simplex && size == 2) {
        switch (user->testNum) {
        case 0: {
          PetscInt triSizes_p2[2]  = {4, 4};
          PetscInt triPoints_p2[8] = {3, 5, 6, 7, 0, 1, 2, 4};

          ierr = PetscMalloc2(2, &sizes, 8, &points);CHKERRQ(ierr);
          ierr = PetscArraycpy(sizes,  triSizes_p2, 2);CHKERRQ(ierr);
          ierr = PetscArraycpy(points, triPoints_p2, 8);CHKERRQ(ierr);
          break;}
        case 1: {
          PetscInt triSizes_p2[2]  = {6, 2};
          PetscInt triPoints_p2[8] = {1, 2, 3, 4, 6, 7, 0, 5};

          ierr = PetscMalloc2(2, &sizes, 8, &points);CHKERRQ(ierr);
          ierr = PetscArraycpy(sizes,  triSizes_p2, 2);CHKERRQ(ierr);
          ierr = PetscArraycpy(points, triPoints_p2, 8);CHKERRQ(ierr);
          break;}
        default:
          SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Could not find matching test number %d for triangular mesh on 2 procs", user->testNum);
        }
      } else if (dim == 2 && simplex && size == 3) {
        PetscInt triSizes_p3[3]  = {3, 3, 2};
        PetscInt triPoints_p3[8] = {1, 2, 4, 3, 6, 7, 0, 5};

        ierr = PetscMalloc2(3, &sizes, 8, &points);CHKERRQ(ierr);
        ierr = PetscArraycpy(sizes,  triSizes_p3, 3);CHKERRQ(ierr);
        ierr = PetscArraycpy(points, triPoints_p3, 8);CHKERRQ(ierr);
      } else if (dim == 2 && !simplex && size == 2) {
        PetscInt quadSizes_p2[2]  = {2, 2};
        PetscInt quadPoints_p2[4] = {2, 3, 0, 1};

        ierr = PetscMalloc2(2, &sizes, 4, &points);CHKERRQ(ierr);
        ierr = PetscArraycpy(sizes,  quadSizes_p2, 2);CHKERRQ(ierr);
        ierr = PetscArraycpy(points, quadPoints_p2, 4);CHKERRQ(ierr);
      } else SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Could not find matching test partition");
    }
    ierr = DMPlexGetPartitioner(*dm, &part);CHKERRQ(ierr);
    ierr = PetscPartitionerSetType(part, PETSCPARTITIONERSHELL);CHKERRQ(ierr);
    ierr = PetscPartitionerShellSetPartition(part, size, sizes, points);CHKERRQ(ierr);
    ierr = PetscFree2(sizes, points);CHKERRQ(ierr);
  }
  ierr = DMPlexDistribute(*dm, 0, NULL, &dmDist);CHKERRQ(ierr);
  if (dmDist) {
    ierr = DMDestroy(dm);CHKERRQ(ierr);
    *dm  = dmDist;
  }
  ierr = PetscObjectSetName((PetscObject) *dm, simplex ? "Simplicial Mesh" : "Tensor Product Mesh");CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode ScrambleOrientation(DM dm, AppCtx *user)
{
  PetscInt       h, cStart, cEnd, c;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMPlexGetVTKCellHeight(dm, &h);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, h, &cStart, &cEnd);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    /* Could use PetscRand instead */
    if (c%2) {ierr = DMPlexOrientPoint(dm, c, -1);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TestOrientation(DM dm, AppCtx *user)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = ScrambleOrientation(dm, user);CHKERRQ(ierr);
  ierr = DMPlexOrient(dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm, NULL, "-oriented_dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  AppCtx         user; /* user-defined work context */
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &dm);CHKERRQ(ierr);
  ierr = TestOrientation(dm, &user);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
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
