const char help[] = "Test Berend's example";

#include <petscdmplex.h>
#include <petscdmforest.h>

int main(int argc, char **argv)
{
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  MPI_Comm       comm            = PETSC_COMM_WORLD;
  PetscInt       dim             = 3;
  PetscInt       cells_per_dir[] = {3, 3, 3};
  PetscReal      dir_min[]       = {0.0, 0.0, 0.0};
  PetscReal      dir_max[]       = {1.0, 1.0, 1.0};
  PetscInt       overlap         = 1;
  DMBoundaryType bcs[]           = {DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE};
  DM             forest, plex;
  Vec            CoordVec;

  PetscCall(DMCreate(comm, &forest));
  PetscCall(DMSetType(forest, DMP8EST));
  PetscCall(DMSetBasicAdjacency(forest, PETSC_TRUE, PETSC_TRUE));
  {
    DM dm_base, pdm, idm;
    PetscCall(DMPlexCreateBoxMesh(comm, dim, PETSC_FALSE /* simplex */
                                  ,
                                  cells_per_dir, dir_min, dir_max, bcs, PETSC_TRUE /* interpolate */
                                  ,
                                  &dm_base));
    PetscCall(DMSetBasicAdjacency(dm_base, PETSC_TRUE, PETSC_TRUE));
    PetscCall(DMPlexDistribute(dm_base, overlap, NULL, &pdm));
    if (pdm) {
      PetscCall(DMDestroy(&dm_base));
      dm_base = pdm;
    }
    PetscCall(DMPlexInterpolate(dm_base, &idm));
    PetscCall(DMDestroy(&dm_base));
    dm_base = idm;
    PetscCall(DMLocalizeCoordinates(dm_base));
    PetscCall(DMPlexDistributeSetDefault(dm_base, PETSC_FALSE));
    PetscCall(DMSetFromOptions(dm_base));
    PetscCall(DMViewFromOptions(dm_base, NULL, "-dm_base_view"));
    PetscCall(DMCopyFields(dm_base, forest));
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

  test:
    suffix: 0

TEST*/
