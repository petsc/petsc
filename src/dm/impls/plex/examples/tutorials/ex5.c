static char help[] = "Load and save the mesh and fields to HDF5 and ExodusII\n\n";

#include <petscdmplex.h>
#include <petscviewerhdf5.h>
#include <petscsf.h>

typedef struct {
  PetscBool compare;                      /* Compare the meshes using DMPlexEqual() */
  PetscBool distribute;                   /* Distribute the mesh */
  PetscBool interpolate;                  /* Generate intermediate mesh elements */
  char      filename[PETSC_MAX_PATH_LEN]; /* Mesh filename */
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->compare = PETSC_FALSE;
  options->distribute = PETSC_TRUE;
  options->interpolate = PETSC_FALSE;
  options->filename[0] = '\0';

  ierr = PetscOptionsBegin(comm, "", "Meshing Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-compare", "Compare the meshes using DMPlexEqual()", "ex5.c", options->compare, &options->compare, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-distribute", "Distribute the mesh", "ex5.c", options->distribute, &options->distribute, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-interpolate", "Generate intermediate mesh elements", "ex5.c", options->interpolate, &options->interpolate, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-filename", "The mesh file", "ex5.c", options->filename, options->filename, PETSC_MAX_PATH_LEN, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
};

int main(int argc, char **argv)
{
  DM             dm, dmnew;
  PetscPartitioner part;
  AppCtx         user;
  PetscViewer    v;
  PetscViewerFormat f;
  PetscBool      flg;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = DMPlexCreateFromFile(PETSC_COMM_WORLD, user.filename, user.interpolate, &dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm, NULL, "-orig_dm_view");CHKERRQ(ierr);

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

  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);

  f = PETSC_VIEWER_DEFAULT;
  ierr = PetscOptionsGetEnum(NULL, NULL, "-format", PetscViewerFormats, (PetscEnum*)&f, NULL);
  ierr = PetscViewerHDF5Open(PetscObjectComm((PetscObject) dm), "dmdist.h5", FILE_MODE_WRITE, &v);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(v, f);CHKERRQ(ierr);
  ierr = DMView(dm, v);CHKERRQ(ierr);

  ierr = PetscViewerFileSetMode(v, FILE_MODE_READ);CHKERRQ(ierr);
  ierr = DMCreate(PetscObjectComm((PetscObject) dm), &dmnew);CHKERRQ(ierr);
  ierr = DMSetType(dmnew, DMPLEX);CHKERRQ(ierr);
  ierr = DMLoad(dmnew, v);CHKERRQ(ierr);

  ierr = PetscViewerPopFormat(v);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&v);CHKERRQ(ierr);

  ierr = DMViewFromOptions(dmnew, NULL, "-new_dm_view");CHKERRQ(ierr);
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
    requires: hdf5 exodusii
  # Idempotence of saving/loading
  test:
    suffix: 0
    requires: hdf5 exodusii
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/Rect-tri3.exo -dm_view ascii::ascii_info_detail
    args: -format hdf5_petsc -compare
  test:
    suffix: 1
    requires: hdf5 exodusii parmetis
    nsize: 2
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/Rect-tri3.exo -dm_view ascii::ascii_info_detail
    args: -petscpartitioner_type parmetis
    args: -format hdf5_petsc -new_dm_view ascii::ascii_info_detail 
  test:
    suffix: 2
    requires: hdf5 exodusii chaco
    nsize: {{1 2 4 8}separate output}
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/blockcylinder-50.exo
    args: -petscpartitioner_type chaco
    args: -new_dm_view ascii::ascii_info_detail
    args: -format {{default hdf5_petsc}separate output}
    args: -interpolate {{0 1}separate output}
  test:
    suffix: 2a
    requires: hdf5 exodusii chaco
    nsize: {{1 2 4 8}separate output}
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/blockcylinder-50.exo
    args: -petscpartitioner_type chaco
    args: -new_dm_view ascii::ascii_info_detail
    args: -format {{hdf5_xdmf hdf5_viz}separate output}

  # reproduce PetscSFView() crash - fixed, left as regression test
  test:
    suffix: new_dm_view
    requires: hdf5 exodusii
    nsize: 2
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/TwoQuads.exo -new_dm_view ascii::ascii_info_detail
TEST*/
