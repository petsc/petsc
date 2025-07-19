static char help[] = "PETSc Annual Meeting 2025: Meshing Tutorial.\n\n\n";

#include <petsc.h>

static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm)
{
  PetscFunctionBeginUser;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM dm;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &dm));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return PETSC_SUCCESS;
}

/*TEST

  # Draw a square with X, use with -draw_pause -1
  test:
    suffix: 0
    requires: triangle x
    args: -dm_view draw -draw_pause 0
    output_file: output/empty.out

  # Draw a square with PyVista
  test:
    suffix: 1
    requires: triangle pyvista
    args: -dm_view pyvista
    output_file: output/empty.out

  # Refine the square
  test:
    suffix: 2
    requires: triangle pyvista
    args: -dm_view pyvista -dm_refine 1
    output_file: output/empty.out

  # Refine the square three times
  test:
    suffix: 3
    requires: triangle pyvista
    args: -dm_view pyvista -dm_refine 3
    output_file: output/empty.out

  # Refine the cube three times
  test:
    suffix: 4
    requires: ctetgen pyvista
    args: -dm_view pyvista -dm_plex_dim 3 -dm_refine 3
    output_file: output/empty.out

  # Draw a sphere with PyVista (all we get is an icosahedron)
  test:
    suffix: 5
    requires: pyvista
    args: -dm_view pyvista -dm_plex_shape sphere
    output_file: output/empty.out

  # Refine the sphere three times
  test:
    suffix: 6
    requires: pyvista
    args: -dm_view pyvista -dm_plex_shape sphere -dm_refine 3
    output_file: output/empty.out

  # Show the 3-sphere
  test:
    suffix: 7
    args: -dm_view -dm_plex_shape sphere -dm_plex_dim 3

  # Extrude the sphere
  test:
    suffix: 8
    requires: pyvista
    args: -dm_view pyvista -dm_plex_shape sphere -dm_refine 1 \
          -dm_plex_transform_type extrude -dm_plex_transform_extrude_layers 3 \
            -dm_plex_transform_extrude_use_tensor 0 -dm_plex_transform_extrude_thickness 0.3
    output_file: output/empty.out

  # Extrude the sphere with cutaway
  test:
    suffix: 9
    requires: pyvista
    args: -dm_view pyvista -view_pyvista_clip 1.,0.,0. -dm_plex_shape sphere -dm_refine 1 \
          -dm_plex_transform_type extrude -dm_plex_transform_extrude_layers 3 \
            -dm_plex_transform_extrude_use_tensor 0 -dm_plex_transform_extrude_thickness 0.3
    output_file: output/empty.out

  # Extrude the refined sphere
  test:
    suffix: 10
    requires: pyvista
    args: -dm_view pyvista -view_pyvista_clip 1.,0.,0. -dm_plex_shape sphere -dm_refine 3 \
          -dm_plex_option_phases ext_ \
            -ext_dm_refine 1 -ext_dm_plex_transform_type extrude -ext_dm_plex_transform_extrude_layers 3 \
              -ext_dm_plex_transform_extrude_use_tensor 0 -ext_dm_plex_transform_extrude_thickness 0.5
    output_file: output/empty.out

  # Extrude the refined sphere
  test:
    suffix: 11
    requires: pyvista
    args: -dm_view pyvista -view_pyvista_clip 1.,0.,0. -dm_plex_shape sphere -dm_refine 3 \
          -dm_plex_option_phases ext_,ref_ \
            -ext_dm_refine 1 -ext_dm_plex_transform_type extrude -ext_dm_plex_transform_extrude_layers 3 \
              -ext_dm_plex_transform_extrude_use_tensor 0 -ext_dm_plex_transform_extrude_thickness 0.5 \
            -ref_dm_refine 1
    output_file: output/empty.out

  # Refine the Schwartz surface
  test:
    suffix: 12
    requires: pyvista
    args: -dm_view pyvista -dm_plex_shape schwarz_p -dm_plex_tps_extent 3,2 -dm_plex_tps_refine 2
    output_file: output/empty.out

  # Refine and extrude the Schwartz surface with given thickness
  test:
    suffix: 13
    requires: pyvista
    args: -dm_view pyvista -dm_plex_shape schwarz_p -dm_plex_tps_extent 3,2,2 -dm_plex_tps_refine 3 \
            -dm_plex_tps_thickness 0.4 -dm_plex_tps_layers 3
    output_file: output/empty.out

  # Filter the square
  test:
    suffix: 14
    requires: triangle pyvista
    args: -dm_view pyvista -dm_refine 1 \
          -dm_plex_transform_type transform_filter -dm_plex_transform_active filter_cells -dm_plex_label_filter_cells 0,1,2
    output_file: output/empty.out

  # Filter a cap on the sphere
  test:
    suffix: 15
    requires: pyvista
    args: -dm_view pyvista -dm_plex_shape sphere -dm_refine 4 \
          -dm_plex_option_phases filt_ \
            -filt_dm_refine 1 -filt_dm_plex_transform_type transform_filter \
              -filt_dm_plex_transform_active filter_cells -dm_plex_label_filter_cells 0
    output_file: output/empty.out

  # Filter the sphere minus a cap
  test:
    suffix: 16
    requires: pyvista
    args: -dm_view pyvista -dm_plex_shape sphere -dm_refine 4 \
          -dm_plex_option_phases filt_ \
            -filt_dm_refine 1 -filt_dm_plex_transform_type transform_filter \
              -filt_dm_plex_transform_active filter_cells \
              -dm_plex_label_filter_cells 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18
    output_file: output/empty.out

  # Convert sphere to quads
  test:
    suffix: 17
    requires: pyvista
    args: -dm_view pyvista -dm_plex_shape sphere -dm_refine 3 \
          -dm_plex_option_phases conv_ -conv_dm_refine 1 -conv_dm_plex_transform_type refine_tobox
    output_file: output/empty.out

  # Load and refine the nozzle
  test:
    suffix: 18
    requires: pyvista egads datafilespath
    args: -dm_view pyvista -dm_plex_filename /Users/knepley/PETSc4/petsc/petsc-dev/share/petsc/datafiles/meshes/nozzle.stp -dm_refine 3
    output_file: output/empty.out

  # Load and refine the nozzle, and convert to quads
  test:
    suffix: 19
    requires: pyvista egads datafilespath
    args: -dm_view pyvista -dm_plex_filename /Users/knepley/PETSc4/petsc/petsc-dev/share/petsc/datafiles/meshes/nozzle.stp -dm_refine 3 \
          -dm_plex_option_phases conv_ -conv_dm_refine 1 -conv_dm_plex_transform_type refine_tobox
    output_file: output/empty.out

TEST*/
