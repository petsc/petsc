static char help[] = "Tests DMPlexCreateColoring() and DMPlexCreateColoringLabel().\n\n";

#include <petscdmplex.h>
#include <petsc/private/hashseti.h>

typedef struct {
  PetscInt  depth;
  PetscInt  distance;
  PetscInt  markCells;
  PetscBool femAdjacency;
} AppCtx;

PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscFunctionBegin;
  options->depth        = 0;
  options->distance     = 1;
  options->markCells    = 0;
  options->femAdjacency = PETSC_FALSE;
  PetscOptionsBegin(comm, "", "DMPlexCreateColoring() Test Options", "DMPLEX");
  PetscCall(PetscOptionsInt("-depth", "Stratum depth defining the nodes in the connectivity graph", "ex104.c", options->depth, &options->depth, NULL));
  PetscCall(PetscOptionsInt("-distance", "How far through the mesh a point reaches", "ex104.c", options->distance, &options->distance, NULL));
  PetscCall(PetscOptionsInt("-mark_cells", "Color only the points in the closure of this many cells, instead of the whole stratum", "ex104.c", options->markCells, &options->markCells, NULL));
  PetscCall(PetscOptionsBool("-fem_adjacency", "Use the finite-element adjacency at the cell stratum too, as a patch coloring needs", "ex104.c", options->femAdjacency, &options->femAdjacency, NULL));
  PetscOptionsEnd();
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

/*
  Check that no two points of a color are within distance hops in the mesh. Cross-rank conflicts are not checked.
*/
static PetscErrorCode CheckColoring(DM dm, PetscInt distance, PetscInt ncolors, IS iscolors[])
{
  PetscHSetI ht, nbr;
  PetscInt  *pts  = NULL;
  PetscInt   npts = 0, index = 0;

  PetscFunctionBeginUser;
  PetscCall(PetscHSetICreate(&ht));
  PetscCall(PetscHSetICreate(&nbr));
  for (PetscInt c = 0; c < ncolors; ++c) {
    const PetscInt *color;
    PetscInt        n;

    PetscCall(PetscHSetIClear(ht));
    PetscCall(ISGetLocalSize(iscolors[c], &n));
    PetscCall(ISGetIndices(iscolors[c], &color));
    for (PetscInt k = 0; k < n; ++k) PetscCall(PetscHSetIAdd(ht, color[k]));
    for (PetscInt k = 0; k < n; ++k) {
      /* Grow the neighborhood of this point one hop at a time */
      PetscCall(PetscHSetIClear(nbr));
      PetscCall(PetscHSetIAdd(nbr, color[k]));
      for (PetscInt r = 0; r < distance; ++r) {
        PetscCall(PetscHSetIGetSize(nbr, &npts));
        PetscCall(PetscMalloc1(npts, &pts));
        index = 0;
        PetscCall(PetscHSetIGetElems(nbr, &index, pts));
        for (PetscInt m = 0; m < npts; ++m) {
          PetscInt  nadj = PETSC_DETERMINE;
          PetscInt *adj  = NULL;

          PetscCall(DMPlexGetAdjacency(dm, pts[m], &nadj, &adj));
          for (PetscInt a = 0; a < nadj; ++a) PetscCall(PetscHSetIAdd(nbr, adj[a]));
          PetscCall(PetscFree(adj));
        }
        PetscCall(PetscFree(pts));
      }
      PetscCall(PetscHSetIGetSize(nbr, &npts));
      PetscCall(PetscMalloc1(npts, &pts));
      index = 0;
      PetscCall(PetscHSetIGetElems(nbr, &index, pts));
      for (PetscInt m = 0; m < npts; ++m) {
        PetscBool has;

        if (pts[m] == color[k]) continue;
        PetscCall(PetscHSetIHas(ht, pts[m], &has));
        PetscCheck(has == PETSC_FALSE, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Points %" PetscInt_FMT " and %" PetscInt_FMT " are within %" PetscInt_FMT " of each other but share color %" PetscInt_FMT, color[k], pts[m], distance, c);
      }
      PetscCall(PetscFree(pts));
    }
    PetscCall(ISRestoreIndices(iscolors[c], &color));
  }
  PetscCall(PetscHSetIDestroy(&nbr));
  PetscCall(PetscHSetIDestroy(&ht));
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
  /* A cell's finite-element adjacency does not reach another cell; use cone adjacency unless requested. */
  if (user->depth == dim && user->femAdjacency == PETSC_FALSE) PetscCall(DMSetBasicAdjacency(*dm, PETSC_TRUE, PETSC_FALSE));
  else PetscCall(DMSetBasicAdjacency(*dm, PETSC_FALSE, PETSC_TRUE));
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
  DMLabel    active = NULL;
  AppCtx     user;
  PetscInt   ncolors = 0, maxcolors = 0;
  IS        *iscolors = NULL;
  ISColoring coloring = NULL;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
  if (user.markCells > 0) PetscCall(CreateActiveLabel(dm, &user, &active));
  if (active == NULL) PetscCall(DMPlexCreateColoring(dm, user.depth, user.distance, &coloring));
  else PetscCall(DMPlexCreateColoringLabel(dm, user.depth, user.distance, active, 1, &coloring));
  PetscCall(ISColoringGetIS(coloring, PETSC_USE_POINTER, &ncolors, &iscolors));
  /* Report the largest color count across processes. */
  PetscCallMPI(MPIU_Allreduce(&ncolors, &maxcolors, 1, MPIU_INT, MPI_MAX, PETSC_COMM_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Number of colors: %" PetscInt_FMT "\n", maxcolors));
  PetscCall(CheckColoring(dm, user.distance, ncolors, iscolors));
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

  # Local coloring tests the induced graph on each rank; the reported count is the maximum across ranks.
  test:
    suffix: local
    nsize: {{1 2}separate output}
    args: -depth 0 -distance 1 -dm_plex_coloring_local -dm_coord_space 0 -dm_plex_simplex 0 -dm_plex_box_faces 4,4 -petscpartitioner_type simple

  # Color only the closure of a few cells.
  test:
    suffix: label
    nsize: {{1 2}separate output}
    args: -depth 0 -distance 1 -mark_cells 3 -iscoloring_view -dm_coord_space 0 -dm_plex_simplex 0 -dm_plex_box_faces 8,8 -petscpartitioner_type simple

  # Finite-element cell adjacency gives an edgeless distance-one graph; distance two tests Vanka patches.
  test:
    suffix: cell_fem
    nsize: {{1 2}separate output}
    args: -depth 2 -fem_adjacency -distance {{1 2}separate output} -dm_coord_space 0 -dm_plex_simplex 0 -dm_plex_box_faces 4,4 -petscpartitioner_type simple

TEST*/
