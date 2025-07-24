static char help[] = "Load and save the mesh and fields to HDF5 and ExodusII\n\n";

#include <petsc/private/dmpleximpl.h>
#include <petscviewerhdf5.h>
#include <petscsf.h>

typedef struct {
  PetscBool         compare;                    /* Compare the meshes using DMPlexEqual() */
  PetscBool         compare_labels;             /* Compare labels in the meshes using DMCompareLabels() */
  PetscBool         distribute;                 /* Distribute the mesh */
  PetscBool         field;                      /* Layout a field over the mesh */
  PetscBool         reorder;                    /* Reorder the points in the section */
  char              ofname[PETSC_MAX_PATH_LEN]; /* Output mesh filename */
  PetscViewerFormat format;                     /* Format to write and read */
  PetscBool         second_write_read;          /* Write and read for the 2nd time */
  PetscBool         use_low_level_functions;    /* Use low level functions for viewing and loading */
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscFunctionBeginUser;
  options->compare                 = PETSC_FALSE;
  options->compare_labels          = PETSC_FALSE;
  options->distribute              = PETSC_TRUE;
  options->field                   = PETSC_FALSE;
  options->reorder                 = PETSC_FALSE;
  options->format                  = PETSC_VIEWER_DEFAULT;
  options->second_write_read       = PETSC_FALSE;
  options->use_low_level_functions = PETSC_FALSE;
  PetscCall(PetscStrncpy(options->ofname, "ex55.h5", sizeof(options->ofname)));

  PetscOptionsBegin(comm, "", "Meshing Problem Options", "DMPLEX");
  PetscCall(PetscOptionsBool("-compare", "Compare the meshes using DMPlexEqual()", "ex55.c", options->compare, &options->compare, NULL));
  PetscCall(PetscOptionsBool("-compare_labels", "Compare labels in the meshes using DMCompareLabels()", "ex55.c", options->compare_labels, &options->compare_labels, NULL));
  PetscCall(PetscOptionsBool("-distribute", "Distribute the mesh", "ex55.c", options->distribute, &options->distribute, NULL));
  PetscCall(PetscOptionsBool("-field", "Layout a field over the mesh", "ex55.c", options->field, &options->field, NULL));
  PetscCall(PetscOptionsBool("-reorder", "Reorder the points in the section", "ex55.c", options->reorder, &options->reorder, NULL));
  PetscCall(PetscOptionsString("-ofilename", "The output mesh file", "ex55.c", options->ofname, options->ofname, sizeof(options->ofname), NULL));
  PetscCall(PetscOptionsEnum("-format", "Format to write and read", "ex55.c", PetscViewerFormats, (PetscEnum)options->format, (PetscEnum *)&options->format, NULL));
  PetscCall(PetscOptionsBool("-second_write_read", "Write and read for the 2nd time", "ex55.c", options->second_write_read, &options->second_write_read, NULL));
  PetscCall(PetscOptionsBool("-use_low_level_functions", "Use low level functions for viewing and loading", "ex55.c", options->use_low_level_functions, &options->use_low_level_functions, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm)
{
  PetscFunctionBeginUser;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetOptionsPrefix(*dm, "orig_"));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateDiscretization(DM dm)
{
  PetscFE        fe;
  DMPolytopeType ct;
  PetscInt       dim, cStart;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, NULL));
  PetscCall(DMPlexGetCellType(dm, cStart, &ct));
  PetscCall(PetscFECreateLagrangeByCell(PETSC_COMM_SELF, dim, 1, ct, 1, PETSC_DETERMINE, &fe));
  PetscCall(DMSetField(dm, 0, NULL, (PetscObject)fe));
  PetscCall(PetscFEDestroy(&fe));
  PetscCall(DMCreateDS(dm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckDistributed(DM dm, PetscBool expectedDistributed)
{
  PetscMPIInt size;
  PetscBool   distributed;
  const char  YES[] = "DISTRIBUTED";
  const char  NO[]  = "NOT DISTRIBUTED";

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)dm), &size));
  if (size < 2) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(DMPlexIsDistributed(dm, &distributed));
  PetscCheck(distributed == expectedDistributed, PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Expected DM being %s but actually is %s", expectedDistributed ? YES : NO, distributed ? YES : NO);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckInterpolated(DM dm, PetscBool expectedInterpolated)
{
  DMPlexInterpolatedFlag iflg;
  PetscBool              interpolated;
  const char             YES[] = "INTERPOLATED";
  const char             NO[]  = "NOT INTERPOLATED";

  PetscFunctionBeginUser;
  PetscCall(DMPlexIsInterpolatedCollective(dm, &iflg));
  interpolated = (PetscBool)(iflg == DMPLEX_INTERPOLATED_FULL);
  PetscCheck(interpolated == expectedInterpolated, PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Expected DM being %s but actually is %s", expectedInterpolated ? YES : NO, interpolated ? YES : NO);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckDistributedInterpolated(DM dm, PetscBool expectedInterpolated, PetscViewer v, AppCtx *user)
{
  PetscViewerFormat format;
  PetscBool         distributed, interpolated = expectedInterpolated;

  PetscFunctionBeginUser;
  PetscCall(PetscViewerGetFormat(v, &format));
  switch (format) {
  case PETSC_VIEWER_HDF5_XDMF:
  case PETSC_VIEWER_HDF5_VIZ: {
    distributed  = PETSC_TRUE;
    interpolated = PETSC_FALSE;
  }; break;
  case PETSC_VIEWER_HDF5_PETSC:
  case PETSC_VIEWER_DEFAULT:
  case PETSC_VIEWER_NATIVE: {
    DMPlexStorageVersion version;

    PetscCall(PetscViewerHDF5GetDMPlexStorageVersionReading(v, &version));
    distributed = (PetscBool)(version->major >= 3);
  }; break;
  default:
    distributed = PETSC_FALSE;
  }
  PetscCall(CheckDistributed(dm, distributed));
  PetscCall(CheckInterpolated(dm, interpolated));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexWriteAndReadHDF5(DM dm, Vec vec, const char filename[], const char prefix[], PetscBool expectedInterpolated, AppCtx *user, DM *dm_new, Vec *v_new)
{
  DM          dmnew;
  Vec         vnew         = NULL;
  const char  savedName[]  = "Mesh";
  const char  loadedName[] = "Mesh_new";
  PetscViewer v;

  PetscFunctionBeginUser;
  PetscCall(PetscViewerHDF5Open(PetscObjectComm((PetscObject)dm), filename, FILE_MODE_WRITE, &v));
  PetscCall(PetscViewerPushFormat(v, user->format));
  PetscCall(PetscObjectSetName((PetscObject)dm, savedName));
  if (user->use_low_level_functions) {
    PetscCall(DMPlexTopologyView(dm, v));
    PetscCall(DMPlexCoordinatesView(dm, v));
    PetscCall(DMPlexLabelsView(dm, v));
  } else {
    PetscCall(DMView(dm, v));
    if (vec) PetscCall(VecView(vec, v));
  }
  PetscCall(PetscViewerFileSetMode(v, FILE_MODE_READ));
  PetscCall(DMCreate(PETSC_COMM_WORLD, &dmnew));
  PetscCall(DMSetType(dmnew, DMPLEX));
  PetscCall(DMPlexDistributeSetDefault(dmnew, PETSC_FALSE));
  PetscCall(PetscObjectSetName((PetscObject)dmnew, savedName));
  PetscCall(DMSetOptionsPrefix(dmnew, prefix));
  if (user->use_low_level_functions) {
    PetscSF sfXC;

    PetscCall(DMPlexTopologyLoad(dmnew, v, &sfXC));
    PetscCall(DMPlexCoordinatesLoad(dmnew, v, sfXC));
    PetscCall(DMPlexLabelsLoad(dmnew, v, sfXC));
    PetscCall(PetscSFDestroy(&sfXC));
  } else {
    PetscCall(DMLoad(dmnew, v));
    if (vec) {
      PetscCall(CreateDiscretization(dmnew));
      PetscCall(DMCreateGlobalVector(dmnew, &vnew));
      PetscCall(PetscObjectSetName((PetscObject)vnew, "solution"));
      PetscCall(VecLoad(vnew, v));
    }
  }
  DMLabel celltypes;
  PetscCall(DMPlexGetCellTypeLabel(dmnew, &celltypes));
  PetscCall(CheckDistributedInterpolated(dmnew, expectedInterpolated, v, user));
  PetscCall(PetscObjectSetName((PetscObject)dmnew, loadedName));
  PetscCall(PetscViewerPopFormat(v));
  PetscCall(PetscViewerDestroy(&v));
  *dm_new = dmnew;
  *v_new  = vnew;
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM               dm, dmnew;
  Vec              v = NULL, vnew = NULL;
  PetscPartitioner part;
  AppCtx           user;
  PetscBool        interpolated = PETSC_TRUE, flg;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &dm));
  PetscCall(PetscOptionsGetBool(NULL, "orig_", "-dm_plex_interpolate", &interpolated, NULL));
  PetscCall(CheckInterpolated(dm, interpolated));

  if (user.distribute) {
    DM dmdist;

    PetscCall(DMPlexGetPartitioner(dm, &part));
    PetscCall(PetscPartitionerSetType(part, PETSCPARTITIONERSIMPLE));
    PetscCall(PetscPartitionerSetFromOptions(part));
    PetscCall(DMPlexDistribute(dm, 0, NULL, &dmdist));
    if (dmdist) {
      PetscCall(DMDestroy(&dm));
      dm = dmdist;
      PetscCall(CheckDistributed(dm, PETSC_TRUE));
      PetscCall(CheckInterpolated(dm, interpolated));
    }
  }
  if (user.field) {
    PetscSection gs;
    PetscScalar *a;
    PetscInt     pStart, pEnd, rStart;

    PetscCall(CreateDiscretization(dm));

    PetscCall(DMCreateGlobalVector(dm, &v));
    PetscCall(PetscObjectSetName((PetscObject)v, "solution"));
    PetscCall(DMGetGlobalSection(dm, &gs));
    PetscCall(PetscSectionGetChart(gs, &pStart, &pEnd));
    PetscCall(VecGetOwnershipRange(v, &rStart, NULL));
    PetscCall(VecGetArrayWrite(v, &a));
    for (PetscInt p = pStart; p < pEnd; ++p) {
      PetscInt dof, off;

      PetscCall(PetscSectionGetDof(gs, p, &dof));
      PetscCall(PetscSectionGetOffset(gs, p, &off));
      if (off < 0) continue;
      for (PetscInt d = 0; d < dof; ++d) a[off + d] = p;
    }
    PetscCall(VecRestoreArrayWrite(v, &a));
  }

  PetscCall(DMSetOptionsPrefix(dm, NULL));
  PetscCall(DMPlexDistributeSetDefault(dm, PETSC_FALSE));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));

  PetscCall(DMPlexWriteAndReadHDF5(dm, v, user.ofname, "new_", interpolated, &user, &dmnew, &vnew));

  if (user.second_write_read) {
    PetscCall(DMDestroy(&dm));
    dm = dmnew;
    PetscCall(DMPlexWriteAndReadHDF5(dm, v, user.ofname, "new_", interpolated, &user, &dmnew, &vnew));
  }

  PetscCall(DMViewFromOptions(dmnew, NULL, "-dm_view"));

  /* This currently makes sense only for sequential meshes. */
  if (user.compare) {
    PetscCall(DMPlexEqual(dmnew, dm, &flg));
    PetscCheck(flg, PETSC_COMM_WORLD, PETSC_ERR_ARG_INCOMP, "DMs are not equal");
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "DMs equal\n"));
    if (v) {
      PetscCall(VecEqual(vnew, v, &flg));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Vecs %s\n", flg ? "equal" : "are not equal"));
      PetscCall(VecViewFromOptions(v, NULL, "-old_vec_view"));
      PetscCall(VecViewFromOptions(vnew, NULL, "-new_vec_view"));
    }
  }
  if (user.compare_labels) {
    PetscCall(DMCompareLabels(dmnew, dm, NULL, NULL));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "DMLabels equal\n"));
  }

  PetscCall(VecDestroy(&v));
  PetscCall(DMDestroy(&dm));
  PetscCall(VecDestroy(&vnew));
  PetscCall(DMDestroy(&dmnew));
  PetscCall(PetscFinalize());
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
    args: -orig_dm_plex_filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/blockcylinder-50.exo
    output_file: output/empty.out
    test:
      suffix: 2
      nsize: {{1 2 4 8}separate output}
      args: -format {{default hdf5_petsc}separate output}
      args: -dm_plex_view_hdf5_storage_version {{1.0.0 2.0.0 3.0.0}}
      args: -orig_dm_plex_interpolate {{0 1}separate output}
    test:
      suffix: 2a
      nsize: {{1 2 4 8}separate output}
      args: -format {{hdf5_xdmf hdf5_viz}separate output}

  test:
    suffix: 3
    requires: exodusii
    args: -orig_dm_plex_filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/blockcylinder-50.exo -compare -compare_labels
    args: -dm_plex_view_hdf5_storage_version {{1.0.0 2.0.0 3.0.0}}

  # Load HDF5 file in XDMF format in parallel, write, read dm1, write, read dm2, and compare dm1 and dm2
  testset:
    suffix: 4
    requires: !complex
    args: -orig_dm_plex_filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/blockcylinder-50.h5 -dm_plex_create_from_hdf5_xdmf
    args: -distribute 0 -second_write_read -compare
    test:
      suffix: hdf5_petsc
      nsize: {{1 2}}
      args: -format hdf5_petsc -compare_labels
      args: -dm_plex_view_hdf5_storage_version {{1.0.0 2.0.0 3.0.0}}
    test:
      suffix: hdf5_xdmf
      nsize: {{1 3 8}}
      args: -format hdf5_xdmf

  # Use low level functions, DMPlexTopologyView()/Load(), DMPlexCoordinatesView()/Load(), and DMPlexLabelsView()/Load()
  # TODO: The output is very long so keeping just 1.0.0 version. This test should be redesigned or removed.
  test:
    suffix: 5
    requires: exodusii
    nsize: 2
    args: -orig_dm_plex_filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/blockcylinder-50.exo
    args: -orig_dm_plex_interpolate 0 -orig_dm_distribute 0
    args: -dm_view ascii::ascii_info_detail
    args: -new_dm_view ascii::ascii_info_detail
    args: -format hdf5_petsc -use_low_level_functions {{0 1}}
    args: -dm_plex_view_hdf5_storage_version 1.0.0

  testset:
    suffix: 6
    requires: hdf5 !complex datafilespath
    nsize: {{1 3}}
    args: -dm_plex_check_symmetry -dm_plex_check_skeleton -dm_plex_check_geometry
    args: -orig_dm_plex_filename ${DATAFILESPATH}/meshes/cube-hexahedra-refined.h5 -dm_plex_create_from_hdf5_xdmf -dm_plex_hdf5_topology_path /cells -dm_plex_hdf5_geometry_path /coordinates
    args: -orig_dm_distribute 0
    args: -format hdf5_petsc -second_write_read -compare -compare_labels
    args: -orig_dm_plex_interpolate {{0 1}} -distribute {{0 1}}
    args: -dm_plex_view_hdf5_storage_version {{1.0.0 2.0.0 3.0.0}}

  testset:
    # the same data and settings as dm_impls_plex_tests-ex18_9%
    suffix: 9
    requires: hdf5 !complex datafilespath
    nsize: {{1 2 4}}
    args: -dm_plex_check_symmetry -dm_plex_check_skeleton -dm_plex_check_geometry
    args: -orig_dm_plex_filename ${DATAFILESPATH}/meshes/cube-hexahedra-refined.h5 -dm_plex_create_from_hdf5_xdmf -dm_plex_hdf5_topology_path /cells -dm_plex_hdf5_geometry_path /coordinates
    args: -orig_dm_distribute 0
    args: -format {{hdf5_petsc hdf5_xdmf}} -second_write_read -compare
    args: -dm_plex_view_hdf5_storage_version 3.0.0
    test:
      suffix: hdf5_seqload
      args: -distribute
      args: -orig_dm_plex_interpolate {{0 1}}
      args: -dm_plex_hdf5_force_sequential
    test:
      suffix: hdf5_seqload_metis
      requires: parmetis
      args: -distribute -petscpartitioner_type parmetis
      args: -orig_dm_plex_interpolate 1
      args: -dm_plex_hdf5_force_sequential
    test:
      suffix: hdf5
      args: -orig_dm_plex_interpolate 1
    test:
      suffix: hdf5_repart
      requires: parmetis
      args: -distribute -petscpartitioner_type parmetis
      args: -orig_dm_plex_interpolate 1
    test:
      TODO: Parallel partitioning of uninterpolated meshes not supported
      suffix: hdf5_repart_ppu
      requires: parmetis
      args: -distribute -petscpartitioner_type parmetis
      args: -orig_dm_plex_interpolate 0

  # reproduce PetscSFView() crash - fixed, left as regression test
  test:
    suffix: new_dm_view
    requires: exodusii
    nsize: 2
    args: -orig_dm_plex_filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/TwoQuads.exo -new_dm_view ascii:ex5_new_dm_view.log:ascii_info_detail
    args: -dm_plex_view_hdf5_storage_version {{1.0.0 2.0.0 3.0.0}}
    output_file: output/empty.out

  # test backward compatibility with petsc_hdf5 format version 1.0.0, serial idempotence
  testset:
    suffix: 10-v3.16.0-v1.0.0
    requires: hdf5 !complex datafilespath
    args: -dm_plex_check_all -compare -compare_labels
    args: -dm_plex_view_hdf5_storage_version {{1.0.0 2.0.0 3.0.0}} -use_low_level_functions {{0 1}}
    test:
      suffix: a
      args: -orig_dm_plex_filename ${DATAFILESPATH}/meshes/hdf5-petsc/petsc-v3.16.0/v1.0.0/annulus-20.h5
    test:
      suffix: b
      TODO: broken
      args: -orig_dm_plex_filename ${DATAFILESPATH}/meshes/hdf5-petsc/petsc-v3.16.0/v1.0.0/barycentricallyrefinedcube.h5
    test:
      suffix: c
      args: -orig_dm_plex_filename ${DATAFILESPATH}/meshes/hdf5-petsc/petsc-v3.16.0/v1.0.0/blockcylinder-50.h5
    test:
      suffix: d
      args: -orig_dm_plex_filename ${DATAFILESPATH}/meshes/hdf5-petsc/petsc-v3.16.0/v1.0.0/cube-hexahedra-refined.h5
    test:
      suffix: e
      args: -orig_dm_plex_filename ${DATAFILESPATH}/meshes/hdf5-petsc/petsc-v3.16.0/v1.0.0/hybrid_hexwedge.h5
    test:
      suffix: f
      args: -orig_dm_plex_filename ${DATAFILESPATH}/meshes/hdf5-petsc/petsc-v3.16.0/v1.0.0/square.h5

  # test permuted sections with petsc_hdf5 format version 1.0.0
  testset:
    suffix: 11
    requires: hdf5 triangle
    args: -field
    args: -dm_plex_check_all -compare -compare_labels
    args: -orig_dm_plex_box_faces 3,3 -dm_plex_view_hdf5_storage_version 1.0.0
    args: -orig_dm_reorder_section -orig_dm_reorder_section_type reverse

    test:
      suffix: serial
    test:
      suffix: serial_no_perm
      args: -orig_dm_ignore_perm_output

TEST*/
