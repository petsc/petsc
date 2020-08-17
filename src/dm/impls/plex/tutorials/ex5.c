static char help[] = "Load and save the mesh and fields to HDF5 and ExodusII\n\n";

#include <petscdmplex.h>
#include <petscviewerhdf5.h>
#include <petscsf.h>

typedef struct {
  PetscBool compare;                      /* Compare the meshes using DMPlexEqual() */
  PetscBool distribute;                   /* Distribute the mesh */
  PetscBool interpolate;                  /* Generate intermediate mesh elements */
  char      filename[PETSC_MAX_PATH_LEN]; /* Mesh filename */
  PetscViewerFormat format;               /* Format to write and read */
  PetscBool second_write_read;            /* Write and read for the 2nd time */
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->compare = PETSC_FALSE;
  options->distribute = PETSC_TRUE;
  options->interpolate = PETSC_FALSE;
  options->filename[0] = '\0';
  options->format = PETSC_VIEWER_DEFAULT;
  options->second_write_read = PETSC_FALSE;

  ierr = PetscOptionsBegin(comm, "", "Meshing Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-compare", "Compare the meshes using DMPlexEqual()", "ex5.c", options->compare, &options->compare, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-distribute", "Distribute the mesh", "ex5.c", options->distribute, &options->distribute, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-interpolate", "Generate intermediate mesh elements", "ex5.c", options->interpolate, &options->interpolate, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-filename", "The mesh file", "ex5.c", options->filename, options->filename, sizeof(options->filename), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-format", "Format to write and read", "ex5.c", PetscViewerFormats, (PetscEnum)options->format, (PetscEnum*)&options->format, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-second_write_read", "Write and read for the 2nd time", "ex5.c", options->second_write_read, &options->second_write_read, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
};

static PetscErrorCode DMPlexWriteAndReadHDF5(DM dm, const char filename[], PetscViewerFormat format, const char prefix[], DM *dm_new)
{
  DM             dmnew;
  PetscViewer    v;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscViewerHDF5Open(PetscObjectComm((PetscObject) dm), filename, FILE_MODE_WRITE, &v);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(v, format);CHKERRQ(ierr);
  ierr = DMView(dm, v);CHKERRQ(ierr);

  ierr = PetscViewerFileSetMode(v, FILE_MODE_READ);CHKERRQ(ierr);
  ierr = DMCreate(PETSC_COMM_WORLD, &dmnew);CHKERRQ(ierr);
  ierr = DMSetType(dmnew, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetOptionsPrefix(dmnew, prefix);CHKERRQ(ierr);
  ierr = DMLoad(dmnew, v);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)dmnew,"Mesh_new");CHKERRQ(ierr);

  ierr = PetscViewerPopFormat(v);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&v);CHKERRQ(ierr);
  *dm_new = dmnew;
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm, dmnew;
  PetscPartitioner part;
  AppCtx         user;
  PetscBool      flg;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = DMPlexCreateFromFile(PETSC_COMM_WORLD, user.filename, user.interpolate, &dm);CHKERRQ(ierr);
  ierr = DMSetOptionsPrefix(dm,"orig_");CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);

  if (user.distribute) {
    DM dmdist;

    ierr = DMPlexGetPartitioner(dm, &part);CHKERRQ(ierr);
    ierr = PetscPartitionerSetFromOptions(part);CHKERRQ(ierr);
    ierr = DMPlexDistribute(dm, 0, NULL, &dmdist);CHKERRQ(ierr);
    if (dmdist) {
      ierr = DMDestroy(&dm);CHKERRQ(ierr);
      dm   = dmdist;
    }
  }

  ierr = DMSetOptionsPrefix(dm,NULL);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);

  ierr = DMPlexWriteAndReadHDF5(dm, "dmdist.h5", user.format, "new_", &dmnew);CHKERRQ(ierr);

  if (user.second_write_read) {
    ierr = DMDestroy(&dm);CHKERRQ(ierr);
    dm = dmnew;
    ierr = DMPlexWriteAndReadHDF5(dm, "dmdist.h5", user.format, "new_", &dmnew);CHKERRQ(ierr);
  }

  ierr = DMViewFromOptions(dmnew, NULL, "-dm_view");CHKERRQ(ierr);
  /* TODO: Is it still true? */
  /* The NATIVE format for coordiante viewing is killing parallel output, since we have a local vector. Map it to global, and it will work. */

  /* This currently makes sense only for sequential meshes. */
  if (user.compare) {
    ierr = DMPlexEqual(dmnew, dm, &flg);CHKERRQ(ierr);
    if (flg) {ierr = PetscPrintf(PETSC_COMM_WORLD,"DMs equal\n");CHKERRQ(ierr);}
    else     {ierr = PetscPrintf(PETSC_COMM_WORLD,"DMs are not equal\n");CHKERRQ(ierr);}
  }

  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = DMDestroy(&dmnew);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST
  build:
    requires: hdf5
  # Idempotence of saving/loading
  #   Have to replace Exodus file, which is creating uninterpolated edges
  test:
    suffix: 0
    requires: exodusii broken
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/Rect-tri3.exo -dm_view ascii::ascii_info_detail
    args: -format hdf5_petsc -compare
  test:
    suffix: 1
    requires: exodusii parmetis !define(PETSC_USE_64BIT_INDICES) broken
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
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/blockcylinder-50.exo -compare

  # Load HDF5 file in XDMF format in parallel, write, read dm1, write, read dm2, and compare dm1 and dm2
  test:
    suffix: 4
    requires: !complex
    nsize: {{1 2 3 4 8}}
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/blockcylinder-50.h5
    args: -dm_plex_create_from_hdf5_xdmf -distribute 0 -format hdf5_xdmf -second_write_read -compare

  testset:
    # the same data and settings as dm_impls_plex_tests-ex18_9%
    requires: hdf5 !complex datafilespath
    #TODO DMPlexCheckPointSF() fails for nsize 4
    nsize: {{1 2}}
    args: -dm_plex_check_symmetry -dm_plex_check_skeleton -dm_plex_check_geometry
    args: -filename ${DATAFILESPATH}/meshes/cube-hexahedra-refined.h5 -dm_plex_create_from_hdf5_xdmf -dm_plex_hdf5_topology_path /cells -dm_plex_hdf5_geometry_path /coordinates
    args: -format hdf5_xdmf -second_write_read -compare
    test:
      suffix: 9_hdf5_seqload
      args: -distribute -petscpartitioner_type simple
      args: -interpolate {{0 1}}
      args: -dm_plex_hdf5_force_sequential
    test:
      suffix: 9_hdf5_seqload_metis
      requires: parmetis
      args: -distribute -petscpartitioner_type parmetis
      args: -interpolate 1
      args: -dm_plex_hdf5_force_sequential
    test:
      suffix: 9_hdf5
      args: -interpolate 1
    test:
      suffix: 9_hdf5_repart
      requires: parmetis
      args: -distribute -petscpartitioner_type parmetis
      args: -interpolate 1
    test:
      TODO: Parallel partitioning of uninterpolated meshes not supported
      suffix: 9_hdf5_repart_ppu
      requires: parmetis
      args: -distribute -petscpartitioner_type parmetis
      args: -interpolate 0

  # reproduce PetscSFView() crash - fixed, left as regression test
  test:
    suffix: new_dm_view
    requires: exodusii
    nsize: 2
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/TwoQuads.exo -new_dm_view ascii:ex5_new_dm_view.log:ascii_info_detail
TEST*/
