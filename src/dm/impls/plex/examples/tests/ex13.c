static char help[] = "Orient a mesh in parallel\n\n";

#include <petscdmplex.h>

typedef struct {
  /* Domain and mesh definition */
  PetscInt  dim;                          /* The topological mesh dimension */
  PetscBool cellSimplex;                  /* Use simplices or hexes */
  char      filename[PETSC_MAX_PATH_LEN]; /* Import mesh from file */
  PetscBool testPartition;                /* Use a fixed partitioning for testing */
  PetscInt  testNum;                      /* Labels the different test partitions */
} AppCtx;

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->dim           = 2;
  options->cellSimplex   = PETSC_TRUE;
  options->filename[0]   = '\0';
  options->testPartition = PETSC_TRUE;
  options->testNum       = 0;

  ierr = PetscOptionsBegin(comm, "", "Meshing Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex13.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-cell_simplex", "Use simplices if true, otherwise hexes", "ex13.c", options->cellSimplex, &options->cellSimplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-filename", "The mesh file", "ex13.c", options->filename, options->filename, PETSC_MAX_PATH_LEN, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_partition", "Use a fixed partition for testing", "ex13.c", options->testPartition, &options->testPartition, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-test_num", "The test partition number", "ex13.c", options->testNum, &options->testNum, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
};

#undef __FUNCT__
#define __FUNCT__ "CreateMesh"
static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  DM             dmDist      = NULL;
  PetscInt       dim         = user->dim;
  PetscBool      cellSimplex = user->cellSimplex;
  const char    *filename    = user->filename;
  const PetscInt cells[3]    = {2, 2, 2};
  size_t         len;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscStrlen(filename, &len);CHKERRQ(ierr);
  if (len)              {ierr = DMPlexCreateFromFile(comm, filename, PETSC_TRUE, dm);CHKERRQ(ierr);}
  else if (cellSimplex) {ierr = DMPlexCreateBoxMesh(comm, dim, PETSC_TRUE, dm);CHKERRQ(ierr);}
  else                  {ierr = DMPlexCreateHexBoxMesh(comm, dim, cells, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE, dm);CHKERRQ(ierr);}
  if (user->testPartition) {
    PetscPartitioner part;
    const PetscInt  *sizes  = NULL;
    const PetscInt  *points = NULL;
    PetscMPIInt      rank, numProcs;

    ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
    ierr = MPI_Comm_size(comm, &numProcs);CHKERRQ(ierr);
    if (!rank) {
      if (dim == 2 && cellSimplex && numProcs == 2) {
        switch (user->testNum) {
        case 0: {
          PetscInt triSizes_p2[2]  = {4, 4};
          PetscInt triPoints_p2[8] = {3, 5, 6, 7, 0, 1, 2, 4};

          ierr = PetscMalloc2(2, &sizes, 8, &points);CHKERRQ(ierr);
          ierr = PetscMemcpy(sizes,  triSizes_p2, 2 * sizeof(PetscInt));CHKERRQ(ierr);
          ierr = PetscMemcpy(points, triPoints_p2, 8 * sizeof(PetscInt));CHKERRQ(ierr);break;}
        case 1: {
          PetscInt triSizes_p2[2]  = {6, 2};
          PetscInt triPoints_p2[8] = {1, 2, 3, 4, 6, 7, 0, 5};

          ierr = PetscMalloc2(2, &sizes, 8, &points);CHKERRQ(ierr);
          ierr = PetscMemcpy(sizes,  triSizes_p2, 2 * sizeof(PetscInt));CHKERRQ(ierr);
          ierr = PetscMemcpy(points, triPoints_p2, 8 * sizeof(PetscInt));CHKERRQ(ierr);break;}
        default:
          SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Could not find matching test number %d for triangular mesh on 2 procs", user->testNum);
        }
      } else if (dim == 2 && cellSimplex && numProcs == 3) {
        PetscInt triSizes_p3[3]  = {3, 3, 2};
        PetscInt triPoints_p3[8] = {1, 2, 4, 3, 6, 7, 0, 5};

        ierr = PetscMalloc2(3, &sizes, 8, &points);CHKERRQ(ierr);
        ierr = PetscMemcpy(sizes,  triSizes_p3, 3 * sizeof(PetscInt));CHKERRQ(ierr);
        ierr = PetscMemcpy(points, triPoints_p3, 8 * sizeof(PetscInt));CHKERRQ(ierr);
      } else if (dim == 2 && !cellSimplex && numProcs == 2) {
        PetscInt quadSizes_p2[2]  = {2, 2};
        PetscInt quadPoints_p2[4] = {2, 3, 0, 1};

        ierr = PetscMalloc2(2, &sizes, 4, &points);CHKERRQ(ierr);
        ierr = PetscMemcpy(sizes,  quadSizes_p2, 2 * sizeof(PetscInt));CHKERRQ(ierr);
        ierr = PetscMemcpy(points, quadPoints_p2, 4 * sizeof(PetscInt));CHKERRQ(ierr);
      } else SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Could not find matching test partition");
    }
    ierr = DMPlexGetPartitioner(*dm, &part);CHKERRQ(ierr);
    ierr = PetscPartitionerSetType(part, PETSCPARTITIONERSHELL);CHKERRQ(ierr);
    ierr = PetscPartitionerShellSetPartition(part, numProcs, sizes, points);CHKERRQ(ierr);
    ierr = PetscFree2(sizes, points);CHKERRQ(ierr);
  }
  ierr = DMPlexDistribute(*dm, 0, NULL, &dmDist);CHKERRQ(ierr);
  if (dmDist) {
    ierr = DMDestroy(dm);CHKERRQ(ierr);
    *dm  = dmDist;
  }
  ierr = PetscObjectSetName((PetscObject) *dm, cellSimplex ? "Simplicial Mesh" : "Tensor Product Mesh");CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ScrambleOrientation"
static PetscErrorCode ScrambleOrientation(DM dm, AppCtx *user)
{
  PetscInt       h, cStart, cEnd, c;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMPlexGetVTKCellHeight(dm, &h);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, h, &cStart, &cEnd);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    /* Could use PetscRand instead */
    if (c%2) {ierr = DMPlexReverseCell(dm, c);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TestOrientation"
static PetscErrorCode TestOrientation(DM dm, AppCtx *user)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = ScrambleOrientation(dm, user);CHKERRQ(ierr);
  ierr = DMPlexOrient(dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm, NULL, "-oriented_dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  DM             dm;
  AppCtx         user; /* user-defined work context */
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);CHKERRQ(ierr);
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &dm);CHKERRQ(ierr);
  ierr = TestOrientation(dm, &user);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
