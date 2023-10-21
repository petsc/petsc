static const char help[] = "Test of CAD functionality";

#include <petscdmplexegads.h>

static PetscErrorCode ComputeVolume(DM dm)
{
  DMLabel     bodyLabel, faceLabel, edgeLabel;
  double      surface = 0., volume = 0., vol;
  PetscInt    dim, pStart, pEnd, pid;
  const char *name;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  if (dim < 2) PetscFunctionReturn(0);
  PetscCall(DMGetCoordinatesLocalSetUp(dm));
  PetscCall(DMGetLabel(dm, "EGADS Body ID", &bodyLabel));
  PetscCall(DMGetLabel(dm, "EGADS Face ID", &faceLabel));
  PetscCall(DMGetLabel(dm, "EGADS Edge ID", &edgeLabel));

  PetscCall(DMPlexGetHeightStratum(dm, 0, &pStart, &pEnd));
  for (PetscInt p = pStart; p < pEnd; ++p) {
    PetscCall(DMLabelGetValue(dim == 2 ? faceLabel : bodyLabel, p, &pid));
    if (pid >= 0) {
      PetscCall(DMPlexComputeCellGeometryFVM(dm, p, &vol, NULL, NULL));
      volume += vol;
    }
  }
  PetscCall(DMPlexGetHeightStratum(dm, 1, &pStart, &pEnd));
  for (PetscInt p = pStart; p < pEnd; ++p) {
    PetscCall(DMLabelGetValue(dim == 2 ? edgeLabel : faceLabel, p, &pid));
    if (pid >= 0) {
      PetscCall(DMPlexComputeCellGeometryFVM(dm, p, &vol, NULL, NULL));
      surface += vol;
    }
  }

  PetscCall(PetscObjectGetName((PetscObject)dm, &name));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm), "DM %s: Surface Area = %.6e Volume = %.6e\n", name ? name : "", surface, volume));
  PetscFunctionReturn(0);
}

int main(int argc, char *argv[])
{
  DM dm;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(DMCreate(PETSC_COMM_WORLD, &dm));
  PetscCall(DMSetType(dm, DMPLEX));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));

  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)dm, "ref1_"));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));

  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)dm, "inf1_"));
  PetscCall(DMPlexInflateToGeomModel(dm, PETSC_TRUE));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));

  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)dm, "ref2_"));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));

  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)dm, "inf2_"));
  PetscCall(DMPlexInflateToGeomModel(dm, PETSC_TRUE));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));

  PetscCall(ComputeVolume(dm));
  PetscCall(DMDestroy(&dm));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  build:
    requires: egads tetgen

  testset:
    requires: datafilespath
    args: -bd_dm_distribute 0 -bd_dm_plex_name "CAD Surface" -bd_dm_plex_check_all -dm_plex_geom_print_model \
          -bd_dm_view -bd_dm_plex_view_labels "EGADS Body ID","EGADS Face ID","EGADS Edge ID" \
          -bd_dm_generator tetgen -dm_plex_name "CAD Mesh" -dm_view hdf5:sphere_volume.h5 \
          -ref1_dm_refine 1 -ref1_dm_view hdf5:sphere_volume.h5 \
          -inf1_dm_view hdf5:sphere_volume_inflate1.h5 \
          -ref2_dm_refine 1 -ref2_dm_view hdf5:sphere_volume.h5 \
          -inf2_dm_view hdf5:sphere_volume_inflate2.h5

    test:
      suffix: sphere_egadslite
      args: -dm_plex_boundary_filename ${DATAFILESPATH}/meshes/cad/sphere_example.egadslite

    test:
      suffix: sphere_egadslite_tess
      TODO: broken
      args: -dm_plex_boundary_filename ${DATAFILESPATH}/meshes/cad/sphere_example.egadslite \
            -dm_plex_geom_tess_model 1

    test:
      suffix: sphere_egads
      args: -dm_plex_boundary_filename ${DATAFILESPATH}/meshes/cad/sphere_example.egads

    test:
      suffix: sphere_egads_tess
      TODO: broken
      args: -dm_plex_boundary_filename ${DATAFILESPATH}/meshes/cad/sphere_example.egads \
            -dm_plex_geom_tess_model 1

    test:
      suffix: cylinder_egads
      args: -dm_plex_boundary_filename ${DATAFILESPATH}/meshes/cad/cylinder_example.egads

    test:
      suffix: cylinder_igs
      args: -dm_plex_boundary_filename ${DATAFILESPATH}/meshes/cad/cylinder_example.igs

    test:
      suffix: cylinder_igs_tess
      TODO: broken
      args: -dm_plex_boundary_filename ${DATAFILESPATH}/meshes/cad/cylinder_example.igs \
            -dm_plex_geom_tess_model 1 -ref1_dm_refine 0 -ref2_dm_refine 0 -bd_dm_plex_view_labels

    test:
      suffix: nozzle_stp
      args: -dm_plex_boundary_filename ${DATAFILESPATH}/meshes/cad/nozzle_example.stp

    test:
      suffix: nozzle_stp_tess
      TODO: broken
      args: -dm_plex_boundary_filename ${DATAFILESPATH}/meshes/cad/nozzle_example.stp \
            -dm_plex_geom_tess_model 1 -ref1_dm_refine 0 -ref2_dm_refine 0 -bd_dm_plex_view_labels

    test:
      suffix: nozzle_igs
      args: -dm_plex_boundary_filename ${DATAFILESPATH}/meshes/cad/nozzle_example.igs

    test:
      suffix: nozzle_igs_tess
      TODO: broken
      args: -dm_plex_boundary_filename ${DATAFILESPATH}/meshes/cad/nozzle_example.igs \
            -dm_plex_geom_tess_model 1 -ref1_dm_refine 0 -ref2_dm_refine 0 -bd_dm_plex_view_labels

TEST*/
