static char help[] = "Tests DMPlexCreateColoring().\n\n";

#include <petscdmplex.h>

typedef struct {
  PetscInt depth;
  PetscInt distance;
} AppCtx;

PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscFunctionBegin;
  options->depth    = 0;
  options->distance = 1;
  PetscOptionsBegin(comm, "", "DMPlexCreateColoring() Test Options", "DMPLEX");
  PetscCall(PetscOptionsInt("-depth", "Stratum depth defining the nodes in the connectivity graph", "ex104.c", options->depth, &options->depth, NULL));
  PetscCall(PetscOptionsInt("-distance", "Coloring distance", "ex104.c", options->distance, &options->distance, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  DM       pdm     = NULL;
  PetscInt overlap = user->distance;
  PetscInt dim;

  PetscFunctionBegin;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMPlexDistributeSetDefault(*dm, PETSC_TRUE));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMGetDimension(*dm, &dim));
  if (user->depth == dim) {
    PetscCall(DMSetBasicAdjacency(*dm, PETSC_TRUE, PETSC_FALSE));
  } else {
    PetscCall(DMSetBasicAdjacency(*dm, PETSC_FALSE, PETSC_TRUE));
  }
  {
    PetscPartitioner part;
    PetscCall(DMPlexSetOptionsPrefix(*dm, "lb_"));
    PetscCall(DMPlexGetPartitioner(*dm, &part));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)part, "lb_"));
    PetscCall(PetscPartitionerSetFromOptions(part));
  }
  PetscCall(DMPlexDistribute(*dm, overlap, NULL, &pdm));
  if (pdm) {
    PetscCall(DMDestroy(dm));
    *dm = pdm;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM         dm;
  AppCtx     user;
  PetscInt   ncolors  = 0;
  IS        *iscolors = NULL;
  ISColoring coloring = NULL;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  /* Create a BoxMesh */
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
  /* Color the DMPlex */
  PetscCall(DMPlexCreateColoring(dm, user.depth, user.distance, &coloring));
  PetscCall(ISColoringGetIS(coloring, PETSC_USE_POINTER, &ncolors, &iscolors));
  for (PetscInt c = 0; c < ncolors; c++) {
    PetscCall(ISViewFromOptions(iscolors[c], NULL, "-iscoloring_view"));
  }
  PetscCall(ISColoringRestoreIS(coloring, PETSC_USE_POINTER, &iscolors));
  PetscCall(ISColoringDestroy(&coloring));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    nsize: {{1 2}separate output}
    args: -depth {{0 1 2}separate output} -distance {{1 2}separate output} -iscoloring_view -dm_coord_space 0 -dm_plex_simplex 0 -dm_plex_box_faces 4,4 -petscpartitioner_type simple

TEST*/
