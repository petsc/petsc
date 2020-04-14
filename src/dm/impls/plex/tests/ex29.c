static char help[] = "Test scalable partitioning on distributed meshes\n\n";

#include <petscdmplex.h>

enum {STAGE_LOAD, STAGE_DISTRIBUTE, STAGE_REFINE, STAGE_OVERLAP};

typedef struct {
  PetscLogEvent createMeshEvent;
  PetscLogStage stages[4];
  /* Domain and mesh definition */
  PetscInt  dim;                             /* The topological mesh dimension */
  PetscInt  faces[3];                        /* Number of faces per dimension */
  PetscBool simplex;                         /* Use simplices or hexes */
  char      filename[PETSC_MAX_PATH_LEN];    /* Import mesh from file */
  PetscInt  overlap;                         /* The cell overlap to use during partitioning */
} AppCtx;

PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscInt        dim;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  options->dim         = 2;
  options->simplex     = PETSC_TRUE;
  options->filename[0] = '\0';
  options->overlap     = PETSC_FALSE;
  options->faces[0]    = 1;
  options->faces[1]    = 1;
  options->faces[2]    = 1;

  ierr = PetscOptionsBegin(comm, "", "Meshing Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsRangeInt("-dim", "The topological mesh dimension", "ex29.c", options->dim, &options->dim, NULL,1,3);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-simplex", "Use simplices if true, otherwise hexes", "ex29.c", options->simplex, &options->simplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-filename", "The mesh file", "", options->filename, options->filename, PETSC_MAX_PATH_LEN, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBoundedInt("-overlap", "The cell overlap for partitioning", "ex29.c", options->overlap, &options->overlap, NULL,0);CHKERRQ(ierr);
  dim = options->dim;
  ierr = PetscOptionsIntArray("-faces", "Number of faces per dimension", "ex29.c", options->faces, &dim, NULL);CHKERRQ(ierr);
  if (dim) options->dim = dim;
  ierr = PetscOptionsEnd();

  ierr = PetscLogEventRegister("CreateMesh", DM_CLASSID, &options->createMeshEvent);CHKERRQ(ierr);
  ierr = PetscLogStageRegister("MeshLoad",       &options->stages[STAGE_LOAD]);CHKERRQ(ierr);
  ierr = PetscLogStageRegister("MeshDistribute", &options->stages[STAGE_DISTRIBUTE]);CHKERRQ(ierr);
  ierr = PetscLogStageRegister("MeshRefine",     &options->stages[STAGE_REFINE]);CHKERRQ(ierr);
  ierr = PetscLogStageRegister("MeshOverlap",    &options->stages[STAGE_OVERLAP]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscInt       dim      = user->dim;
  PetscInt      *faces    = user->faces;
  PetscBool      simplex  = user->simplex;
  const char    *filename = user->filename;
  size_t         len;
  PetscMPIInt    rank, size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(user->createMeshEvent,0,0,0,0);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  ierr = PetscStrlen(filename, &len);CHKERRQ(ierr);
  ierr = PetscLogStagePush(user->stages[STAGE_LOAD]);CHKERRQ(ierr);
  if (len) {
    ierr = DMPlexCreateFromFile(comm, filename, PETSC_TRUE, dm);CHKERRQ(ierr);
  } else {
    ierr = DMPlexCreateBoxMesh(comm, dim, simplex, faces, NULL, NULL, NULL, PETSC_TRUE, dm);CHKERRQ(ierr);
  }
  ierr = PetscLogStagePop();CHKERRQ(ierr);
  {
    DM               pdm = NULL;
    PetscPartitioner part;

    ierr = DMPlexGetPartitioner(*dm, &part);CHKERRQ(ierr);
    ierr = PetscPartitionerSetFromOptions(part);CHKERRQ(ierr);
    /* Distribute mesh over processes */
    ierr = PetscLogStagePush(user->stages[STAGE_DISTRIBUTE]);CHKERRQ(ierr);
    ierr = DMPlexDistribute(*dm, 0, NULL, &pdm);CHKERRQ(ierr);
    if (pdm) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = pdm;
    }
    ierr = PetscLogStagePop();CHKERRQ(ierr);
  }
  ierr = PetscLogStagePush(user->stages[STAGE_REFINE]);CHKERRQ(ierr);
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = PetscLogStagePop();CHKERRQ(ierr);
  if (user->overlap) {
    DM odm = NULL;
    /* Add the level-1 overlap to refined mesh */
    ierr = PetscLogStagePush(user->stages[STAGE_OVERLAP]);CHKERRQ(ierr);
    ierr = DMPlexDistributeOverlap(*dm, 1, NULL, &odm);CHKERRQ(ierr);
    if (odm) {
      ierr = DMView(odm, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm = odm;
    }
    ierr = PetscLogStagePop();CHKERRQ(ierr);
  }
  if (simplex){ierr = PetscObjectSetName((PetscObject) *dm, "Simplicial Mesh");CHKERRQ(ierr);}
  else{ierr = PetscObjectSetName((PetscObject) *dm, "Tensor Product Mesh");CHKERRQ(ierr);}
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  ierr = PetscLogEventEnd(user->createMeshEvent,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm, pdm;
  AppCtx         user;                 /* user-defined work context */
  PetscErrorCode ierr;
  PetscPartitioner part;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &dm);CHKERRQ(ierr);
  ierr = DMPlexGetPartitioner(dm, &part);CHKERRQ(ierr);
  ierr = PetscPartitionerSetFromOptions(part);CHKERRQ(ierr);
  ierr = DMPlexDistribute(dm, user.overlap, NULL, &pdm);CHKERRQ(ierr);
  if (pdm) {ierr = DMViewFromOptions(pdm, NULL, "-pdm_view");CHKERRQ(ierr);}
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = DMDestroy(&pdm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: 0
    requires: ctetgen
    args: -dim 3 -dm_refine 2 -petscpartitioner_type simple -dm_view
  test:
    suffix: 1
    args: -dim 3 -simplex 0 -dm_refine 2 -petscpartitioner_type simple -dm_view
  test:
    suffix: quad_0
    nsize: 2
    args: -dim 3 -simplex 0 -dm_refine 2 -petscpartitioner_type simple -dm_view -pdm_view

TEST*/
