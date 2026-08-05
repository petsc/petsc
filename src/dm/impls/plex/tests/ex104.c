static char help[] = "Tests DMPlexCreateColoring() and DMPlexCreateColoringLabel().\n\n";

#include <petscdmplex.h>

typedef struct {
  PetscInt depth;
  PetscInt distance;
  PetscInt markCells;
} AppCtx;

PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscFunctionBegin;
  options->depth     = 0;
  options->distance  = 1;
  options->markCells = 0;
  PetscOptionsBegin(comm, "", "DMPlexCreateColoring() Test Options", "DMPLEX");
  PetscCall(PetscOptionsInt("-depth", "Stratum depth defining the nodes in the connectivity graph", "ex104.c", options->depth, &options->depth, NULL));
  PetscCall(PetscOptionsInt("-distance", "Coloring distance", "ex104.c", options->distance, &options->distance, NULL));
  PetscCall(PetscOptionsInt("-mark_cells", "Color only the points in the closure of this many cells, instead of the whole stratum", "ex104.c", options->markCells, &options->markCells, NULL));
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

/* Select the closure of the first markCells owned cells. */
static PetscErrorCode CreateActiveLabel(DM dm, AppCtx *user, DMLabel *label)
{
  PetscInt cStart, cEnd;

  PetscFunctionBeginUser;
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMLabelCreate(PetscObjectComm((PetscObject)dm), "active", label));
  for (PetscInt c = cStart; c < PetscMin(cStart + user->markCells, cEnd); ++c) PetscCall(DMLabelSetValue(*label, c, 1));
  PetscCall(DMPlexLabelComplete(dm, *label));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM         dm;
  DMLabel    active = NULL;
  AppCtx     user;
  PetscInt   ncolors = 0, maxcolors = 0;
  IS        *iscolors = NULL;
  ISColoring coloring = NULL;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  /* Create a BoxMesh */
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
  /* Color the DMPlex */
  if (user.markCells > 0) PetscCall(CreateActiveLabel(dm, &user, &active));
  if (active == NULL) PetscCall(DMPlexCreateColoring(dm, user.depth, user.distance, &coloring));
  else PetscCall(DMPlexCreateColoringLabel(dm, user.depth, user.distance, active, 1, &coloring));
  PetscCall(ISColoringGetIS(coloring, PETSC_USE_POINTER, &ncolors, &iscolors));
  /* Report the largest color count across processes. */
  PetscCallMPI(MPIU_Allreduce(&ncolors, &maxcolors, 1, MPIU_INT, MPI_MAX, PETSC_COMM_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Number of colors: %" PetscInt_FMT "\n", maxcolors));
  for (PetscInt c = 0; c < ncolors; c++) {
    PetscCall(ISViewFromOptions(iscolors[c], NULL, "-iscoloring_view"));
  }
  PetscCall(ISColoringRestoreIS(coloring, PETSC_USE_POINTER, &iscolors));
  PetscCall(ISColoringDestroy(&coloring));
  PetscCall(DMLabelDestroy(&active));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    nsize: {{1 2}separate output}
    args: -depth {{0 1 2}separate output} -distance {{1 2}separate output} -iscoloring_view -dm_coord_space 0 -dm_plex_simplex 0 -dm_plex_box_faces 4,4 -petscpartitioner_type simple

  # Lexical weighting uses the four-color vertex pattern; the other cases exercise fixed and reordered point order.
  testset:
    nsize: 1
    args: -depth 0 -distance 1 -dm_coord_space 0 -dm_plex_simplex 0 -dm_plex_box_faces 16,16
    test:
      suffix: grid_lexical
      output_file: output/ex104_grid.out
    test:
      suffix: grid_natural
      args: -dm_plex_coloring_ordering_type natural
      output_file: output/ex104_grid.out
    test:
      suffix: grid_ordering
      args: -dm_plex_coloring_ordering_type {{rcm nd}separate output}

  # Color only the closure of a few cells.
  test:
    suffix: label
    nsize: {{1 2}separate output}
    args: -depth 0 -distance 1 -mark_cells 3 -iscoloring_view -dm_coord_space 0 -dm_plex_simplex 0 -dm_plex_box_faces 8,8 -petscpartitioner_type simple

TEST*/
