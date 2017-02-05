static char help[] = "Partition a mesh in parallel, perhaps with overlap\n\n";

#include <petscdmplex.h>

typedef struct {
  /* Domain and mesh definition */
  PetscInt  dim;                          /* The topological mesh dimension */
  PetscBool cellSimplex;                  /* Use simplices or hexes */
  char      filename[PETSC_MAX_PATH_LEN]; /* Import mesh from file */
  PetscInt  overlap;                      /* The cell overlap to use during partitioning */
  PetscBool testPartition;                /* Use a fixed partitioning for testing */
  PetscBool testRedundant;                /* Use a redundant partitioning for testing */
  PetscBool loadBalance;                  /* Load balance via a second distribute step */
} AppCtx;

PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->dim           = 2;
  options->cellSimplex   = PETSC_TRUE;
  options->filename[0]   = '\0';
  options->overlap       = 0;
  options->testPartition = PETSC_FALSE;
  options->testRedundant = PETSC_FALSE;
  options->loadBalance   = PETSC_FALSE;

  ierr = PetscOptionsBegin(comm, "", "Meshing Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex12.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-cell_simplex", "Use simplices if true, otherwise hexes", "ex12.c", options->cellSimplex, &options->cellSimplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-filename", "The mesh file", "ex12.c", options->filename, options->filename, PETSC_MAX_PATH_LEN, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-overlap", "The cell overlap for partitioning", "ex12.c", options->overlap, &options->overlap, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_partition", "Use a fixed partition for testing", "ex12.c", options->testPartition, &options->testPartition, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_redundant", "Use a redundant partition for testing", "ex12.c", options->testRedundant, &options->testRedundant, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-load_balance", "Perform parallel load balancing in a second distribution step", "ex12.c", options->loadBalance, &options->loadBalance, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
};

PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  DM             distMesh        = NULL;
  PetscInt       dim             = user->dim;
  PetscBool      cellSimplex     = user->cellSimplex;
  const char    *filename        = user->filename;
  PetscInt       triSizes_n2[2]  = {4, 4};
  PetscInt       triPoints_n2[8] = {0, 1, 4, 6, 2, 3, 5, 7};
  PetscInt       triSizes_n3[3]  = {3, 2, 3};
  PetscInt       triPoints_n3[8] = {3, 5, 6, 1, 7, 0, 2, 4};
  PetscInt       triSizes_n4[4]  = {2, 2, 2, 2};
  PetscInt       triPoints_n4[8] = {0, 7, 1, 5, 2, 3, 4, 6};
  PetscInt       triSizes_n8[8]  = {1, 1, 1, 1, 1, 1, 1, 1};
  PetscInt       triPoints_n8[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  PetscInt       quadSizes[2]    = {2, 2};
  PetscInt       quadPoints[4]   = {2, 3, 0, 1};
  PetscInt       overlap         = user->overlap >= 0 ? user->overlap : 0;
  size_t         len;
  PetscMPIInt    rank, numProcs;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &numProcs);CHKERRQ(ierr);
  ierr = PetscStrlen(filename, &len);CHKERRQ(ierr);
  if (len) {
    const char *extGmsh = ".msh";
    PetscBool   isGmsh;

    ierr = PetscStrncmp(&filename[PetscMax(0,len-4)], extGmsh, 4, &isGmsh);CHKERRQ(ierr);
    if (isGmsh) {
      PetscViewer viewer;

      ierr = PetscViewerCreate(comm, &viewer);CHKERRQ(ierr);
      ierr = PetscViewerSetType(viewer, PETSCVIEWERASCII);CHKERRQ(ierr);
      ierr = PetscViewerFileSetMode(viewer, FILE_MODE_READ);CHKERRQ(ierr);
      ierr = PetscViewerFileSetName(viewer, filename);CHKERRQ(ierr);
      ierr = DMPlexCreateGmsh(comm, viewer, PETSC_TRUE, dm);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    } else {
      ierr = DMPlexCreateCGNSFromFile(comm, filename, PETSC_TRUE, dm);CHKERRQ(ierr);
    }
  } else if (cellSimplex) {
    ierr = DMPlexCreateBoxMesh(comm, dim, dim == 2 ? 2 : 1, PETSC_TRUE, dm);CHKERRQ(ierr);
  } else {
    const PetscInt cells[3] = {2, 2, 2};

    ierr = DMPlexCreateHexBoxMesh(comm, dim, cells, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, dm);CHKERRQ(ierr);
  }
  if (!user->testRedundant) {
    if (user->testPartition) {
      const PetscInt  *sizes = NULL;
      const PetscInt  *points = NULL;
      PetscPartitioner part;

      if (!rank) {
        if (dim == 2 && cellSimplex && numProcs == 2) {
          sizes = triSizes_n2; points = triPoints_n2;
        } else if (dim == 2 && cellSimplex && numProcs == 3) {
          sizes = triSizes_n3; points = triPoints_n3;
        } else if (dim == 2 && cellSimplex && numProcs == 4) {
          sizes = triSizes_n4; points = triPoints_n4;
        } else if (dim == 2 && cellSimplex && numProcs == 8) {
          sizes = triSizes_n8; points = triPoints_n8;
        } else if (dim == 2 && !cellSimplex && numProcs == 2) {
          sizes = quadSizes; points = quadPoints;
        }
      }
      ierr = DMPlexGetPartitioner(*dm, &part);CHKERRQ(ierr);
      ierr = PetscPartitionerSetType(part, PETSCPARTITIONERSHELL);CHKERRQ(ierr);
      ierr = PetscPartitionerShellSetPartition(part, numProcs, sizes, points);CHKERRQ(ierr);
    }
    ierr = DMPlexDistribute(*dm, overlap, NULL, &distMesh);CHKERRQ(ierr);
  }
  else {
    ierr = DMPlexGetRedundantDM(*dm,&distMesh);CHKERRQ(ierr);
  }
  if (distMesh) {
    ierr = DMDestroy(dm);CHKERRQ(ierr);
    *dm  = distMesh;
  }
  if (user->loadBalance) {
    PetscPartitioner part;
    PetscInt         reSizes_n2[2]  = {2, 2};
    PetscInt         rePoints_n2[4] = {2, 3, 0, 1};
    if (rank) {rePoints_n2[0] = 1; rePoints_n2[1] = 2, rePoints_n2[2] = 0, rePoints_n2[3] = 3;}

    ierr = DMPlexGetPartitioner(*dm, &part);CHKERRQ(ierr);
    ierr = PetscPartitionerSetType(part, PETSCPARTITIONERSHELL);CHKERRQ(ierr);
    ierr = PetscPartitionerShellSetPartition(part, numProcs, reSizes_n2, rePoints_n2);CHKERRQ(ierr);

    ierr = DMPlexDistribute(*dm, overlap, NULL, &distMesh);CHKERRQ(ierr);
    if (distMesh) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = distMesh;
    }
  }
  ierr = PetscObjectSetName((PetscObject) *dm, cellSimplex ? "Simplicial Mesh" : "Tensor Product Mesh");CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
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
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST
  # Parallel, no overlap tests 0-2
  test:
    suffix: 0
    requires: triangle
    args: -dm_view ascii:mesh.tex:ascii_latex
  test:
    suffix: 1
    requires: triangle
    nsize: 3
    args: -test_partition -dm_view ::ascii_info_detail
  test:
    suffix: 2
    requires: triangle
    nsize: 8
    args: -test_partition -dm_view ::ascii_info_detail
  # Parallel, level-1 overlap tests 3-4
  test:
    suffix: 3
    requires: triangle
    nsize: 3
    args: -test_partition -overlap 1 -dm_view ::ascii_info_detail
  test:
    suffix: 4
    requires: triangle
    nsize: 8
    args: -test_partition -overlap 1 -dm_view ::ascii_info_detail
  # Parallel, level-2 overlap test 5
  test:
    suffix: 5
    requires: triangle
    nsize: 8
    args: -test_partition -overlap 2 -dm_view ::ascii_info_detail
  # Parallel load balancing, test 6-7
  test:
    suffix: 6
    requires: triangle
    nsize: 2
    args: -test_partition -overlap 1 -dm_view ::ascii_info_detail
  test:
    suffix: 7
    requires: triangle
    nsize: 2
    args: -test_partition -overlap 1 -load_balance -dm_view ::ascii_info_detail
  # Parallel redundant copying, test 8
  test:
    suffix: 8
    requires: triangle
    nsize: 2
    args: -test_redundant -dm_view ::ascii_info_detail

TEST*/
