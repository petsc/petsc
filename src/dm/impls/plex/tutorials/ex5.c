static char help[] = "Demonstrate HDF5 parallel load-save-reload cycle\n\n";

#include <petscdmplex.h>
#include <petscviewerhdf5.h>
#define EX "ex5.c"

typedef struct {
  char              infile[PETSC_MAX_PATH_LEN];  /* Input mesh filename */
  char              outfile[PETSC_MAX_PATH_LEN]; /* Dump/reload mesh filename */
  PetscViewerFormat informat;                    /* Input mesh format */
  PetscViewerFormat outformat;                   /* Dump/reload mesh format */
  PetscBool         heterogeneous;               /* Test save on N / load on M */
  PetscInt          ntimes;                      /* How many times do the cycle */
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscBool flg;

  PetscFunctionBeginUser;
  options->infile[0]     = '\0';
  options->outfile[0]    = '\0';
  options->informat      = PETSC_VIEWER_HDF5_XDMF;
  options->outformat     = PETSC_VIEWER_HDF5_XDMF;
  options->heterogeneous = PETSC_FALSE;
  options->ntimes        = 2;
  PetscOptionsBegin(comm, "", "Meshing Problem Options", "DMPLEX");
  PetscCall(PetscOptionsString("-infile", "The input mesh file", EX, options->infile, options->infile, sizeof(options->infile), &flg));
  PetscCheck(flg, comm, PETSC_ERR_USER_INPUT, "-infile needs to be specified");
  PetscCall(PetscOptionsString("-outfile", "The output mesh file (by default it's the same as infile)", EX, options->outfile, options->outfile, sizeof(options->outfile), &flg));
  PetscCheck(flg, comm, PETSC_ERR_USER_INPUT, "-outfile needs to be specified");
  PetscCall(PetscOptionsEnum("-informat", "Input mesh format", EX, PetscViewerFormats, (PetscEnum)options->informat, (PetscEnum *)&options->informat, NULL));
  PetscCall(PetscOptionsEnum("-outformat", "Dump/reload mesh format", EX, PetscViewerFormats, (PetscEnum)options->outformat, (PetscEnum *)&options->outformat, NULL));
  PetscCall(PetscOptionsBool("-heterogeneous", "Test save on N / load on M", EX, options->heterogeneous, &options->heterogeneous, NULL));
  PetscCall(PetscOptionsInt("-ntimes", "How many times do the cycle", EX, options->ntimes, &options->ntimes, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
};

//TODO test DMLabel I/O (not yet working for PETSC_VIEWER_HDF5_XDMF)
int main(int argc, char **argv)
{
  AppCtx            user;
  MPI_Comm          comm;
  PetscMPIInt       gsize, grank, mycolor;
  PetscInt          i;
  PetscBool         flg;
  const char        exampleDMPlexName[] = "DMPlex Object";
  const char       *infilename;
  PetscViewerFormat informat;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &gsize));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &grank));

  for (i = 0; i < user.ntimes; i++) {
    if (i == 0) {
      /* Use infile/informat for the initial load */
      infilename = user.infile;
      informat   = user.informat;
    } else {
      /* Use outfile/outformat for all I/O except the very initial load */
      infilename = user.outfile;
      informat   = user.outformat;
    }

    if (user.heterogeneous) {
      mycolor = (PetscMPIInt)(grank > user.ntimes - i);
    } else {
      mycolor = (PetscMPIInt)0;
    }
    PetscCallMPI(MPI_Comm_split(PETSC_COMM_WORLD, mycolor, grank, &comm));

    if (mycolor == 0) {
      /* Load/Save only on processes with mycolor == 0 */
      DM          dm;
      PetscViewer v;

      PetscCall(PetscPrintf(comm, "Begin cycle %" PetscInt_FMT "\n", i));

      /* Load data from XDMF into dm in parallel */
      /* We could also use
          PetscCall(DMPlexCreateFromFile(PETSC_COMM_WORLD, user.filename, "ex5_plex", PETSC_TRUE, &dm));
        This currently support a few more formats than DMLoad().
      */
      PetscCall(PetscViewerHDF5Open(comm, infilename, FILE_MODE_READ, &v));
      PetscCall(PetscViewerPushFormat(v, informat));
      PetscCall(DMCreate(comm, &dm));
      PetscCall(DMSetType(dm, DMPLEX));
      PetscCall(PetscObjectSetName((PetscObject)dm, exampleDMPlexName));
      PetscCall(DMSetOptionsPrefix(dm, "loaded_"));
      PetscCall(DMLoad(dm, v));
      PetscCall(DMPlexDistributeSetDefault(dm, PETSC_FALSE));
      PetscCall(DMSetFromOptions(dm));
      PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));
      PetscCall(PetscViewerPopFormat(v));
      PetscCall(PetscViewerDestroy(&v));

      /* We just test/demonstrate DM is indeed distributed - unneeded in the application code */
      PetscCall(DMPlexIsDistributed(dm, &flg));
      PetscCall(PetscPrintf(comm, "Loaded mesh distributed? %s\n", PetscBools[flg]));

      /* Interpolate */
      PetscCall(DMSetOptionsPrefix(dm, "interpolated_"));
      PetscCall(DMSetFromOptions(dm));
      PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));

      /* Redistribute */
      PetscCall(DMSetOptionsPrefix(dm, "redistributed_"));
      PetscCall(DMSetFromOptions(dm));
      PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));

      /* Save redistributed dm to XDMF in parallel and destroy it */
      PetscCall(PetscViewerHDF5Open(comm, user.outfile, FILE_MODE_WRITE, &v));
      PetscCall(PetscViewerPushFormat(v, user.outformat));
      PetscCall(PetscObjectSetName((PetscObject)dm, exampleDMPlexName));
      PetscCall(DMView(dm, v));
      PetscCall(PetscViewerPopFormat(v));
      PetscCall(PetscViewerDestroy(&v));
      PetscCall(DMDestroy(&dm));

      PetscCall(PetscPrintf(comm, "End   cycle %" PetscInt_FMT "\n--------\n", i));
    }
    PetscCallMPI(MPI_Comm_free(&comm));
    PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
  }

  /* Final clean-up */
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
  build:
    requires: hdf5
  testset:
    suffix: 0
    requires: !complex
    nsize: 4
    args: -infile ${wPETSC_DIR}/share/petsc/datafiles/meshes/blockcylinder-50.h5 -informat hdf5_xdmf
    args: -outfile ex5_dump.h5 -outformat {{hdf5_xdmf hdf5_petsc}separate output}
    args: -ntimes 3 -interpolated_dm_plex_interpolate_pre -redistributed_dm_distribute
    args: -loaded_dm_view -interpolated_dm_view -redistributed_dm_view
    test:
      # this partitioner should not shuffle anything, it should yield the same partititioning as the XDMF reader - added just for testing
      suffix: simple
      args: -petscpartitioner_type simple
    test:
      suffix: parmetis
      requires: parmetis
      args: -petscpartitioner_type parmetis
    test:
      suffix: ptscotch
      requires: ptscotch
      args: -petscpartitioner_type ptscotch

  testset:
    suffix: 1
    requires: !complex
    nsize: 4
    args: -infile ${wPETSC_DIR}/share/petsc/datafiles/meshes/blockcylinder-50.h5 -informat hdf5_xdmf
    args: -outfile ex5_dump.h5 -outformat {{hdf5_xdmf hdf5_petsc}separate output}
    args: -ntimes 3 -interpolated_dm_plex_interpolate_pre -redistributed_dm_distribute
    args: -heterogeneous True
    args: -loaded_dm_view -interpolated_dm_view -redistributed_dm_view
    test:
      suffix: simple
      args: -petscpartitioner_type simple
    test:
      suffix: parmetis
      requires: parmetis
      args: -petscpartitioner_type parmetis
    test:
      suffix: ptscotch
      requires: ptscotch
      args: -petscpartitioner_type ptscotch

TEST*/
