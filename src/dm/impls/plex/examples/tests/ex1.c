static char help[] = "Run C version of TetGen to construct and refine a mesh\n\n";

#include <petscdmplex.h>

typedef struct {
  DM            dm;                /* REQUIRED in order to use SNES evaluation functions */
  PetscInt      debug;             /* The debugging level */
  PetscLogEvent createMeshEvent;
  /* Domain and mesh definition */
  PetscInt      dim;                          /* The topological mesh dimension */
  PetscBool     interpolate;                  /* Generate intermediate mesh elements */
  PetscReal     refinementLimit;              /* The largest allowable cell volume */
  PetscBool     cellSimplex;                  /* Use simplices or hexes */
  char          filename[PETSC_MAX_PATH_LEN]; /* Import mesh from file */
  PetscBool     testPartition;                /* Use a fixed partitioning for testing */
} AppCtx;

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->debug             = 0;
  options->dim               = 2;
  options->interpolate       = PETSC_FALSE;
  options->refinementLimit   = 0.0;
  options->cellSimplex       = PETSC_TRUE;
  options->filename[0]       = '\0';
  options->testPartition     = PETSC_FALSE;

  ierr = PetscOptionsBegin(comm, "", "Meshing Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-debug", "The debugging level", "ex1.c", options->debug, &options->debug, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex1.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-interpolate", "Generate intermediate mesh elements", "ex1.c", options->interpolate, &options->interpolate, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-refinement_limit", "The largest allowable cell volume", "ex1.c", options->refinementLimit, &options->refinementLimit, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-cell_simplex", "Use simplices if true, otherwise hexes", "ex1.c", options->cellSimplex, &options->cellSimplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-filename", "The mesh file", "ex7.c", options->filename, options->filename, PETSC_MAX_PATH_LEN, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_partition", "Use a fixed partition for testing", "ex1.c", options->testPartition, &options->testPartition, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  ierr = PetscLogEventRegister("CreateMesh",          DM_CLASSID,   &options->createMeshEvent);CHKERRQ(ierr);
  PetscFunctionReturn(0);
};

#undef __FUNCT__
#define __FUNCT__ "CreateMesh"
PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscInt       dim             = user->dim;
  PetscBool      interpolate     = user->interpolate;
  PetscReal      refinementLimit = user->refinementLimit;
  PetscBool      cellSimplex     = user->cellSimplex;
  const char    *filename        = user->filename;
  PetscInt       triSizes[2]     = {4, 4};
  PetscInt       triPoints[8]    = {3, 5, 6, 7, 0, 1, 2, 4};
  PetscInt       quadSizes[2]    = {2, 2};
  PetscInt       quadPoints[8]   = {2, 3, 0, 1};
  const PetscInt cells[3]        = {2, 2, 2};
  size_t         len;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(user->createMeshEvent,0,0,0,0);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = PetscStrlen(filename, &len);CHKERRQ(ierr);
  if (len)              {ierr = DMPlexCreateFromFile(comm, filename, interpolate, dm);CHKERRQ(ierr);}
  else if (cellSimplex) {ierr = DMPlexCreateBoxMesh(comm, dim, interpolate, dm);CHKERRQ(ierr);}
  else                  {ierr = DMPlexCreateHexBoxMesh(comm, dim, cells, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE, dm);CHKERRQ(ierr);}
  {
    DM refinedMesh     = NULL;
    DM distributedMesh = NULL;

    /* Refine mesh using a volume constraint */
    ierr = DMPlexSetRefinementUniform(*dm, PETSC_FALSE);CHKERRQ(ierr);
    ierr = DMPlexSetRefinementLimit(*dm, refinementLimit);CHKERRQ(ierr);
    ierr = DMRefine(*dm, comm, &refinedMesh);CHKERRQ(ierr);
    if (refinedMesh) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = refinedMesh;
    }
    if (user->testPartition) {
      const PetscInt  *sizes  = !rank ? dim == 2 ? cellSimplex ? triSizes  : quadSizes  : NULL : NULL;
      const PetscInt  *points = !rank ? dim == 2 ? cellSimplex ? triPoints : quadPoints : NULL : NULL;
      PetscPartitioner part;

      ierr = DMPlexGetPartitioner(*dm, &part);CHKERRQ(ierr);
      ierr = PetscPartitionerSetType(part, PETSCPARTITIONERSHELL);CHKERRQ(ierr);
      ierr = PetscPartitionerShellSetPartition(part, 2, sizes, points);CHKERRQ(ierr);
    }
    /* Distribute mesh over processes */
    ierr = DMPlexDistribute(*dm, 0, NULL, &distributedMesh);CHKERRQ(ierr);
    if (distributedMesh) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = distributedMesh;
    } else {
      ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
    }
  }
  ierr = PetscObjectSetName((PetscObject) *dm, "Simplicial Mesh");CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  ierr = PetscLogEventEnd(user->createMeshEvent,0,0,0,0);CHKERRQ(ierr);
  user->dm = *dm;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  AppCtx         user;                 /* user-defined work context */
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);CHKERRQ(ierr);
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &user.dm);CHKERRQ(ierr);
  ierr = DMDestroy(&user.dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
