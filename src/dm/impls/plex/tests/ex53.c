const char help[] = "Test distribution with overlap using DMForest";

#include <petscdmplex.h>
#include <petscdmforest.h>

int main(int argc, char **argv)
{
  DM       forest, plex;
  Vec      CoordVec;
  MPI_Comm comm;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscCall(DMCreate(comm, &forest));
  PetscCall(PetscObjectSetName((PetscObject)forest, "forest"));
  PetscCall(DMSetType(forest, DMP8EST));
  PetscCall(DMSetBasicAdjacency(forest, PETSC_TRUE, PETSC_TRUE));
  {
    DM dm_base;

    PetscCall(DMCreate(comm, &dm_base));
    PetscCall(DMSetType(dm_base, DMPLEX));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)dm_base, "base_"));
    PetscCall(DMSetFromOptions(dm_base));
    PetscCall(DMViewFromOptions(dm_base, NULL, "-dm_view"));
    PetscCall(DMCopyFields(dm_base, PETSC_DETERMINE, PETSC_DETERMINE, forest));
    PetscCall(DMForestSetBaseDM(forest, dm_base));
    PetscCall(DMDestroy(&dm_base));
  }
  PetscCall(DMForestSetPartitionOverlap(forest, 1));
  PetscCall(DMSetFromOptions(forest));
  PetscCall(DMSetUp(forest));
  PetscCall(DMViewFromOptions(forest, NULL, "-dm_forest_view"));

  PetscCall(DMConvert(forest, DMPLEX, &plex));
  PetscCall(DMGetCoordinates(plex, &CoordVec));
  PetscCall(DMViewFromOptions(forest, NULL, "-dm_plex_view"));

  PetscCall(DMDestroy(&plex));
  PetscCall(DMDestroy(&forest));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    requires: p4est
    args: -base_dm_plex_dim 3 -base_dm_plex_simplex 0 -base_dm_plex_box_faces 3,3,3 -base_dm_distribute_overlap 1 \
          -base_dm_plex_adj_cone true -base_dm_plex_adj_closure true \
          -base_dm_view -dm_forest_view -dm_plex_view

    test:
      suffix: 0

    test:
      suffix: 1
      nsize: 3
      args: -petscpartitioner_type simple

TEST*/
