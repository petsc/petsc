static char help[] = "Create and view a forest mesh\n\n";

#include <petscdmforest.h>
#include <petscdmplex.h>
#include <petscoptions.h>

int main(int argc, char **argv)
{
  DM             dm;
  char           typeString[256] = {'\0'};
  PetscViewer    viewer          = NULL;
  PetscBool      conv = PETSC_FALSE;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  CHKERRQ(DMCreate(PETSC_COMM_WORLD, &dm));
  CHKERRQ(PetscStrncpy(typeString,DMFOREST,256));
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"DM Forest example options",NULL);CHKERRQ(ierr);
  CHKERRQ(PetscOptionsString("-dm_type","The type of the dm",NULL,DMFOREST,typeString,sizeof(typeString),NULL));
  CHKERRQ(PetscOptionsBool("-test_convert","Test conversion to DMPLEX",NULL,conv,&conv,NULL));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  CHKERRQ(DMSetType(dm,(DMType) typeString));
  CHKERRQ(DMSetFromOptions(dm));
  CHKERRQ(DMSetUp(dm));
  CHKERRQ(DMViewFromOptions(dm,NULL,"-dm_view"));
  CHKERRQ(PetscViewerDestroy(&viewer));
  if (conv) {
    DM dmConv;

    CHKERRQ(DMConvert(dm,DMPLEX,&dmConv));
    CHKERRQ(DMLocalizeCoordinates(dmConv));
    CHKERRQ(DMViewFromOptions(dmConv,NULL,"-dm_conv_view"));
    CHKERRQ(DMPlexCheckCellShape(dmConv,PETSC_FALSE,PETSC_DETERMINE));
    CHKERRQ(DMDestroy(&dmConv));
  }
  CHKERRQ(DMDestroy(&dm));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

      test:
        output_file: output/ex1_moebius.out
        suffix: p4est_topology_moebius
        nsize: 3
        args: -dm_type p4est -dm_forest_topology moebius -dm_view vtk:moebius.vtu
        requires: p4est !complex

      test:
        output_file: output/ex1_moebius.out
        suffix: p4est_topology_moebius_convert
        nsize: 3
        args: -dm_type p4est -dm_forest_topology moebius -test_convert -dm_conv_view vtk:moebiusconv.vtu
        requires: p4est !complex

      test:
        output_file: output/ex1_shell.out
        suffix: p4est_topology_shell
        nsize: 3
        args: -dm_type p8est -dm_forest_topology shell -dm_view vtk:shell.vtu
        requires: p4est !complex

      test:
        TODO: broken
        output_file: output/ex1_shell.out
        suffix: p4est_topology_shell_convert
        nsize: 3
        args: -dm_type p8est -dm_forest_topology shell -test_convert -dm_conv_view vtk:shellconv.vtu
        requires: p4est !complex

      test:
        TODO: broken
        output_file: output/ex1_sphere.out
        suffix: p4est_topology_sphere_convert
        nsize: 3
        args: -dm_type p8est -dm_forest_topology sphere -dm_view vtk:sphere.vtu  -dm_forest_initial_refinement 1 -dm_forest_maximum_refinement 1 -test_convert -dm_conv_view vtk:sphereconv.vtu
        requires: p4est !complex

      test:
        output_file: output/ex1_brick.out
        suffix: p4est_topology_brick
        nsize: 3
        args: -dm_type p8est -dm_forest_topology brick -dm_p4est_brick_size 2,3,5 -dm_view vtk:brick.vtu
        requires: p4est !complex

      test:
        output_file: output/ex1_brick_periodic_glvis.out
        suffix: p4est_topology_brick_periodic_glvis
        args: -dm_type p8est -dm_forest_topology brick -dm_p4est_brick_size 3,4,5 -dm_p4est_brick_periodicity 0,1,0 -test_convert -dm_conv_view glvis:
        requires: p4est

      test:
        output_file: output/ex1_brick.out
        suffix: p4est_topology_brick_periodic_2d
        nsize: 3
        args: -dm_type p4est -dm_forest_topology brick -dm_p4est_brick_size 5,6 -dm_p4est_brick_periodicity 1,0 -test_convert -dm_forest_initial_refinement 0 -dm_forest_maximum_refinement 2 -dm_p4est_refine_pattern hash
        requires: p4est

      test:
        output_file: output/ex1_brick.out
        suffix: p4est_topology_brick_periodic_3d
        nsize: 3
        args: -dm_type p8est -dm_forest_topology brick -dm_p4est_brick_size 5,6,1 -dm_p4est_brick_periodicity 0,1,0 -test_convert -dm_forest_initial_refinement 0 -dm_forest_maximum_refinement 2 -dm_p4est_refine_pattern hash
        requires: p4est

TEST*/
