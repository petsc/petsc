static char help[] = "Tests save/load plex with distribution in HDF5.\n\n";

#include <petscdmplex.h>
#include <petscsf.h>
#include <petsclayouthdf5.h>

typedef struct {
  char fname[PETSC_MAX_PATH_LEN]; /* Output mesh filename */
} AppCtx;

PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscBool flg;

  PetscFunctionBegin;
  options->fname[0] = '\0';
  PetscOptionsBegin(comm, "", "DMPlex View/Load Test Options", "DMPLEX");
  PetscCall(PetscOptionsString("-fname", "The output mesh file", "ex51.c", options->fname, options->fname, sizeof(options->fname), &flg));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  const char        exampleDMPlexName[]       = "exampleDMPlex";
  const char        exampleDistributionName[] = "exampleDistribution";
  PetscViewerFormat format                    = PETSC_VIEWER_HDF5_PETSC;
  AppCtx            user;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  /* Save */
  {
    DM          dm;
    PetscViewer viewer;

    PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD, user.fname, FILE_MODE_WRITE, &viewer));
    /* Save exampleDMPlex */
    {
      DM               pdm;
      PetscPartitioner part;
      const PetscInt   faces[2] = {6, 1};
      PetscSF          sf;
      PetscInt         overlap = 1;

      PetscCall(DMPlexCreateBoxMesh(PETSC_COMM_WORLD, 2, PETSC_FALSE, faces, NULL, NULL, NULL, PETSC_TRUE, &dm));
      PetscCall(DMPlexGetPartitioner(dm, &part));
      PetscCall(PetscPartitionerSetFromOptions(part));
      PetscCall(DMPlexDistribute(dm, overlap, &sf, &pdm));
      if (pdm) {
        PetscCall(DMDestroy(&dm));
        dm = pdm;
      }
      PetscCall(PetscSFDestroy(&sf));
      PetscCall(PetscObjectSetName((PetscObject)dm, exampleDMPlexName));
      PetscCall(DMPlexDistributionSetName(dm, exampleDistributionName));
      PetscCall(PetscViewerPushFormat(viewer, format));
      PetscCall(DMPlexTopologyView(dm, viewer));
      PetscCall(DMPlexLabelsView(dm, viewer));
      PetscCall(PetscViewerPopFormat(viewer));
    }
    /* Save coordinates */
    PetscCall(PetscViewerPushFormat(viewer, format));
    PetscCall(DMPlexCoordinatesView(dm, viewer));
    PetscCall(PetscViewerPopFormat(viewer));
    PetscCall(PetscObjectSetName((PetscObject)dm, "Save: DM"));
    PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));
    PetscCall(DMDestroy(&dm));
    PetscCall(PetscViewerDestroy(&viewer));
  }
  /* Load */
  {
    DM          dm;
    PetscSF     sfXC;
    PetscViewer viewer;

    PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD, user.fname, FILE_MODE_READ, &viewer));
    /* Load exampleDMPlex */
    PetscCall(DMCreate(PETSC_COMM_WORLD, &dm));
    PetscCall(DMSetType(dm, DMPLEX));
    PetscCall(PetscObjectSetName((PetscObject)dm, exampleDMPlexName));
    PetscCall(DMPlexDistributionSetName(dm, exampleDistributionName));
    /* sfXC: X -> C                         */
    /* X: set of globalPointNumbers, [0, N) */
    /* C: loaded in-memory plex             */
    PetscCall(PetscViewerPushFormat(viewer, format));
    PetscCall(DMPlexTopologyLoad(dm, viewer, &sfXC));
    PetscCall(PetscViewerPopFormat(viewer));
    /* Do not distribute (Already distributed just like the saved plex) */
    /* Load labels */
    PetscCall(DMPlexLabelsLoad(dm, viewer, sfXC));
    /* Load coordinates */
    PetscCall(PetscViewerPushFormat(viewer, format));
    PetscCall(DMPlexCoordinatesLoad(dm, viewer, sfXC));
    PetscCall(PetscViewerPopFormat(viewer));
    PetscCall(DMSetFromOptions(dm));
    /* Print the exact same plex as the saved one */
    PetscCall(PetscObjectSetName((PetscObject)dm, "Load: DM"));
    PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));
    PetscCall(PetscSFDestroy(&sfXC));
    PetscCall(DMDestroy(&dm));
    PetscCall(PetscViewerDestroy(&viewer));
  }
  /* Finalize */
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  build:
    requires: hdf5
    requires: !complex
  test:
    suffix: 0
    requires: parmetis
    nsize: {{1 2 4}separate output}
    args: -fname ex51_dump.h5 -dm_view ascii::ascii_info_detail
    args: -petscpartitioner_type parmetis
    args: -dm_plex_view_hdf5_storage_version 2.1.0

TEST*/
