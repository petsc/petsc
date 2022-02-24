static char help[] = "Create a Plex Schwarz P surface with quads\n\n";

#include <petscdmplex.h>

int main(int argc, char **argv)
{
  DM             dm;
  PetscInt       extent[3] = {1,1,1}, refine = 0, layers = 0, three;
  PetscReal      thickness = 0.;
  PetscBool      distribute = PETSC_TRUE;
  DMBoundaryType periodic[3] = {DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE};
  DMPlexTPSType  tps_type = DMPLEX_TPS_SCHWARZ_P;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Schwarz P Example", NULL);CHKERRQ(ierr);
  CHKERRQ(PetscOptionsIntArray("-extent", "Number of replicas for each of three dimensions", NULL, extent, (three=3, &three), NULL));
  CHKERRQ(PetscOptionsInt("-refine", "Number of refinements", NULL, refine, &refine, NULL));
  CHKERRQ(PetscOptionsEnumArray("-periodic", "Periodicity in each of three dimensions", NULL, DMBoundaryTypes, (PetscEnum*)periodic, (three=3, &three), NULL));
  CHKERRQ(PetscOptionsBool("-distribute", "Distribute TPS manifold prior to refinement and extrusion", NULL, distribute, &distribute, NULL));
  CHKERRQ(PetscOptionsInt("-layers", "Number of layers in volumetric extrusion (or zero to not extrude)", NULL, layers, &layers, NULL));
  CHKERRQ(PetscOptionsReal("-thickness", "Thickness of volumetric extrusion", NULL, thickness, &thickness, NULL));
  CHKERRQ(PetscOptionsEnum("-tps_type", "Type of triply-periodic surface", NULL, DMPlexTPSTypes, (PetscEnum)tps_type, (PetscEnum*)&tps_type, NULL));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  CHKERRQ(DMPlexCreateTPSMesh(PETSC_COMM_WORLD, tps_type, extent, periodic, distribute, refine, layers, thickness, &dm));
  CHKERRQ(PetscObjectSetName((PetscObject)dm, "TPS"));
  CHKERRQ(DMViewFromOptions(dm, NULL, "-dm_view"));
  CHKERRQ(DMDestroy(&dm));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: 0
    args: -extent 1,2,3 -dm_view -refine 0
  test:
    suffix: 1
    args: -extent 2,3,1 -dm_view -refine 1

  test:
    suffix: gyroid_0
    args: -extent 1,2,3 -dm_view -refine 0 -tps_type gyroid
  test:
    suffix: gyroid_1
    args: -extent 2,3,1 -dm_view -refine 1 -tps_type gyroid
  test:
    suffix: extrude_0
    args: -extent 2,3,1 -dm_view -refine 0 -layers 3 -thickness .2
  test:
    suffix: extrude_1_dist
    nsize: 2
    args: -extent 2,1,1 -dm_view -refine 1 -layers 3 -thickness .2

TEST*/
