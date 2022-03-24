static char help[] = "Load and save the mesh and fields to HDF5 and ExodusII\n\n";

#include <petscdmplex.h>
#include <petscviewerhdf5.h>
#include <petscsf.h>

typedef struct {
  PetscBool compare;                      /* Compare the meshes using DMPlexEqual() */
  PetscBool compare_labels;               /* Compare labels in the meshes using DMCompareLabels() */
  PetscBool distribute;                   /* Distribute the mesh */
  PetscBool interpolate;                  /* Generate intermediate mesh elements */
  char      filename[PETSC_MAX_PATH_LEN]; /* Mesh filename */
  char      meshname[PETSC_MAX_PATH_LEN]; /* Mesh name */
  PetscViewerFormat format;               /* Format to write and read */
  PetscBool second_write_read;            /* Write and read for the 2nd time */
  PetscBool use_low_level_functions;      /* Use low level functions for viewing and loading */
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->compare = PETSC_FALSE;
  options->compare_labels = PETSC_FALSE;
  options->distribute = PETSC_TRUE;
  options->interpolate = PETSC_FALSE;
  options->filename[0] = '\0';
  options->meshname[0] = '\0';
  options->format = PETSC_VIEWER_DEFAULT;
  options->second_write_read = PETSC_FALSE;
  options->use_low_level_functions = PETSC_FALSE;

  ierr = PetscOptionsBegin(comm, "", "Meshing Problem Options", "DMPLEX");CHKERRQ(ierr);
  CHKERRQ(PetscOptionsBool("-compare", "Compare the meshes using DMPlexEqual()", "ex55.c", options->compare, &options->compare, NULL));
  CHKERRQ(PetscOptionsBool("-compare_labels", "Compare labels in the meshes using DMCompareLabels()", "ex55.c", options->compare_labels, &options->compare_labels, NULL));
  CHKERRQ(PetscOptionsBool("-distribute", "Distribute the mesh", "ex55.c", options->distribute, &options->distribute, NULL));
  CHKERRQ(PetscOptionsBool("-interpolate", "Generate intermediate mesh elements", "ex55.c", options->interpolate, &options->interpolate, NULL));
  CHKERRQ(PetscOptionsString("-filename", "The mesh file", "ex55.c", options->filename, options->filename, sizeof(options->filename), NULL));
  CHKERRQ(PetscOptionsString("-meshname", "The mesh file", "ex55.c", options->meshname, options->meshname, sizeof(options->meshname), NULL));
  CHKERRQ(PetscOptionsEnum("-format", "Format to write and read", "ex55.c", PetscViewerFormats, (PetscEnum)options->format, (PetscEnum*)&options->format, NULL));
  CHKERRQ(PetscOptionsBool("-second_write_read", "Write and read for the 2nd time", "ex55.c", options->second_write_read, &options->second_write_read, NULL));
  CHKERRQ(PetscOptionsBool("-use_low_level_functions", "Use low level functions for viewing and loading", "ex55.c", options->use_low_level_functions, &options->use_low_level_functions, NULL));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
};

static PetscErrorCode DMPlexWriteAndReadHDF5(DM dm, const char filename[], const char prefix[], AppCtx user, DM *dm_new)
{
  DM             dmnew;
  const char     savedName[]  = "Mesh";
  const char     loadedName[] = "Mesh_new";
  PetscViewer    v;

  PetscFunctionBeginUser;
  CHKERRQ(PetscViewerHDF5Open(PetscObjectComm((PetscObject) dm), filename, FILE_MODE_WRITE, &v));
  CHKERRQ(PetscViewerPushFormat(v, user.format));
  CHKERRQ(PetscObjectSetName((PetscObject) dm, savedName));
  if (user.use_low_level_functions) {
    CHKERRQ(DMPlexTopologyView(dm, v));
    CHKERRQ(DMPlexCoordinatesView(dm, v));
    CHKERRQ(DMPlexLabelsView(dm, v));
  } else {
    CHKERRQ(DMView(dm, v));
  }

  CHKERRQ(PetscViewerFileSetMode(v, FILE_MODE_READ));
  CHKERRQ(DMCreate(PETSC_COMM_WORLD, &dmnew));
  CHKERRQ(DMSetType(dmnew, DMPLEX));
  CHKERRQ(PetscObjectSetName((PetscObject) dmnew, savedName));
  CHKERRQ(DMSetOptionsPrefix(dmnew, prefix));
  if (user.use_low_level_functions) {
    PetscSF  sfXC;

    CHKERRQ(DMPlexTopologyLoad(dmnew, v, &sfXC));
    CHKERRQ(DMPlexCoordinatesLoad(dmnew, v, sfXC));
    CHKERRQ(DMPlexLabelsLoad(dmnew, v, sfXC));
    CHKERRQ(PetscSFDestroy(&sfXC));
  } else {
    CHKERRQ(DMLoad(dmnew, v));
  }
  CHKERRQ(PetscObjectSetName((PetscObject)dmnew,loadedName));

  CHKERRQ(PetscViewerPopFormat(v));
  CHKERRQ(PetscViewerDestroy(&v));
  *dm_new = dmnew;
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm, dmnew;
  PetscPartitioner part;
  AppCtx         user;
  PetscBool      flg;

  CHKERRQ(PetscInitialize(&argc, &argv, NULL,help));
  CHKERRQ(ProcessOptions(PETSC_COMM_WORLD, &user));
  CHKERRQ(DMPlexCreateFromFile(PETSC_COMM_WORLD, user.filename, user.meshname, user.interpolate, &dm));
  CHKERRQ(DMSetOptionsPrefix(dm,"orig_"));
  CHKERRQ(DMViewFromOptions(dm, NULL, "-dm_view"));

  if (user.distribute) {
    DM dmdist;

    CHKERRQ(DMPlexGetPartitioner(dm, &part));
    CHKERRQ(PetscPartitionerSetFromOptions(part));
    CHKERRQ(DMPlexDistribute(dm, 0, NULL, &dmdist));
    if (dmdist) {
      CHKERRQ(DMDestroy(&dm));
      dm   = dmdist;
    }
  }

  CHKERRQ(DMSetOptionsPrefix(dm,NULL));
  CHKERRQ(DMPlexDistributeSetDefault(dm, PETSC_FALSE));
  CHKERRQ(DMSetFromOptions(dm));
  CHKERRQ(DMViewFromOptions(dm, NULL, "-dm_view"));

  CHKERRQ(DMPlexWriteAndReadHDF5(dm, "dmdist.h5", "new_", user, &dmnew));

  if (user.second_write_read) {
    CHKERRQ(DMDestroy(&dm));
    dm = dmnew;
    CHKERRQ(DMPlexWriteAndReadHDF5(dm, "dmdist.h5", "new_", user, &dmnew));
  }

  CHKERRQ(DMViewFromOptions(dmnew, NULL, "-dm_view"));
  /* TODO: Is it still true? */
  /* The NATIVE format for coordiante viewing is killing parallel output, since we have a local vector. Map it to global, and it will work. */

  /* This currently makes sense only for sequential meshes. */
  if (user.compare) {
    CHKERRQ(DMPlexEqual(dmnew, dm, &flg));
    PetscCheck(flg,PETSC_COMM_WORLD, PETSC_ERR_ARG_INCOMP, "DMs are not equal");
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"DMs equal\n"));
  }
  if (user.compare_labels) {
    CHKERRQ(DMCompareLabels(dmnew, dm, NULL, NULL));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"DMLabels equal\n"));
  }

  CHKERRQ(DMDestroy(&dm));
  CHKERRQ(DMDestroy(&dmnew));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST
  build:
    requires: hdf5
  # Idempotence of saving/loading
  #   Have to replace Exodus file, which is creating uninterpolated edges
  test:
    suffix: 0
    TODO: broken
    requires: exodusii
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/Rect-tri3.exo -dm_view ascii::ascii_info_detail
    args: -format hdf5_petsc -compare
  test:
    suffix: 1
    TODO: broken
    requires: exodusii parmetis !defined(PETSC_USE_64BIT_INDICES)
    nsize: 2
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/Rect-tri3.exo -dm_view ascii::ascii_info_detail
    args: -petscpartitioner_type parmetis
    args: -format hdf5_petsc -new_dm_view ascii::ascii_info_detail
  testset:
    requires: exodusii
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/blockcylinder-50.exo
    args: -petscpartitioner_type simple
    args: -dm_view ascii::ascii_info_detail
    args: -new_dm_view ascii::ascii_info_detail
    test:
      suffix: 2
      nsize: {{1 2 4 8}separate output}
      args: -format {{default hdf5_petsc}separate output}
      args: -interpolate {{0 1}separate output}
    test:
      suffix: 2a
      nsize: {{1 2 4 8}separate output}
      args: -format {{hdf5_xdmf hdf5_viz}separate output}
  test:
    suffix: 3
    requires: exodusii
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/blockcylinder-50.exo -compare -compare_labels

  # Load HDF5 file in XDMF format in parallel, write, read dm1, write, read dm2, and compare dm1 and dm2
  testset:
    suffix: 4
    requires: !complex
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/blockcylinder-50.h5 -dm_plex_create_from_hdf5_xdmf
    args: -distribute 0 -second_write_read -compare
    test:
      suffix: hdf5_petsc
      nsize: {{1 2}}
      args: -format hdf5_petsc -compare_labels
    test:
      suffix: hdf5_xdmf
      nsize: {{1 3 8}}
      args: -format hdf5_xdmf

  # Use low level functions, DMPlexTopologyView()/Load(), DMPlexCoordinatesView()/Load(), and DMPlexLabelsView()/Load()
  # Output must be the same as ex55_2_nsize-2_format-hdf5_petsc_interpolate-0.out
  test:
    suffix: 5
    requires: exodusii
    nsize: 2
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/blockcylinder-50.exo
    args: -petscpartitioner_type simple
    args: -dm_view ascii::ascii_info_detail
    args: -new_dm_view ascii::ascii_info_detail
    args: -format hdf5_petsc -use_low_level_functions

  testset:
    suffix: 6
    requires: hdf5 !complex datafilespath
    nsize: {{1 3}}
    args: -dm_plex_check_symmetry -dm_plex_check_skeleton -dm_plex_check_geometry
    args: -filename ${DATAFILESPATH}/meshes/cube-hexahedra-refined.h5 -dm_plex_create_from_hdf5_xdmf -dm_plex_hdf5_topology_path /cells -dm_plex_hdf5_geometry_path /coordinates
    args: -format hdf5_petsc -second_write_read -compare -compare_labels
    args: -interpolate {{0 1}} -distribute {{0 1}} -petscpartitioner_type simple

  testset:
    # the same data and settings as dm_impls_plex_tests-ex18_9%
    suffix: 9
    requires: hdf5 !complex datafilespath
    nsize: {{1 2 4}}
    args: -dm_plex_check_symmetry -dm_plex_check_skeleton -dm_plex_check_geometry
    args: -filename ${DATAFILESPATH}/meshes/cube-hexahedra-refined.h5 -dm_plex_create_from_hdf5_xdmf -dm_plex_hdf5_topology_path /cells -dm_plex_hdf5_geometry_path /coordinates
    args: -format hdf5_xdmf -second_write_read -compare
    test:
      suffix: hdf5_seqload
      args: -distribute -petscpartitioner_type simple
      args: -interpolate {{0 1}}
      args: -dm_plex_hdf5_force_sequential
    test:
      suffix: hdf5_seqload_metis
      requires: parmetis
      args: -distribute -petscpartitioner_type parmetis
      args: -interpolate 1
      args: -dm_plex_hdf5_force_sequential
    test:
      suffix: hdf5
      args: -interpolate 1 -petscpartitioner_type simple
    test:
      suffix: hdf5_repart
      requires: parmetis
      args: -distribute -petscpartitioner_type parmetis
      args: -interpolate 1
    test:
      TODO: Parallel partitioning of uninterpolated meshes not supported
      suffix: hdf5_repart_ppu
      requires: parmetis
      args: -distribute -petscpartitioner_type parmetis
      args: -interpolate 0

  # reproduce PetscSFView() crash - fixed, left as regression test
  test:
    suffix: new_dm_view
    requires: exodusii
    nsize: 2
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/TwoQuads.exo -new_dm_view ascii:ex5_new_dm_view.log:ascii_info_detail

  # test backward compatibility of petsc_hdf5 format
  testset:
    suffix: 10-v3.16.0-v1.0.0
    requires: hdf5 !complex datafilespath
    args: -dm_plex_check_all -compare -compare_labels
    args: -dm_plex_view_hdf5_storage_version {{1.0.0 2.0.0}} -use_low_level_functions {{0 1}}
    test:
      suffix: a
      args: -filename ${DATAFILESPATH}/meshes/hdf5-petsc/petsc-v3.16.0/v1.0.0/annulus-20.h5
    test:
      suffix: b
      TODO: broken
      args: -filename ${DATAFILESPATH}/meshes/hdf5-petsc/petsc-v3.16.0/v1.0.0/barycentricallyrefinedcube.h5
    test:
      suffix: c
      args: -filename ${DATAFILESPATH}/meshes/hdf5-petsc/petsc-v3.16.0/v1.0.0/blockcylinder-50.h5
    test:
      suffix: d
      args: -filename ${DATAFILESPATH}/meshes/hdf5-petsc/petsc-v3.16.0/v1.0.0/cube-hexahedra-refined.h5
    test:
      suffix: e
      args: -filename ${DATAFILESPATH}/meshes/hdf5-petsc/petsc-v3.16.0/v1.0.0/hybrid_hexwedge.h5
    test:
      suffix: f
      args: -filename ${DATAFILESPATH}/meshes/hdf5-petsc/petsc-v3.16.0/v1.0.0/square.h5

TEST*/
