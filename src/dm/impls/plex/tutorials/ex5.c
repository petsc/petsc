static char help[] = "Demonstrate HDF5/XDMF load-save-reload cycle\n\n";

#include <petscdmplex.h>
#include <petscviewerhdf5.h>
#define EX "ex5.c"

typedef struct {
  char              infile[PETSC_MAX_PATH_LEN];  /* Input mesh filename */
  char              outfile[PETSC_MAX_PATH_LEN]; /* Dump/reload mesh filename */
  PetscViewerFormat informat;                    /* Input mesh format */
  PetscViewerFormat outformat;                   /* Dump/reload mesh format */
  PetscBool         redistribute;                /* Redistribute the mesh */
  PetscBool         heterogeneous;               /* Test save on N / load on M */
  PetscInt          ntimes;                      /* How many times do the cycle */
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->infile[0]     = '\0';
  options->outfile[0]    = '\0';
  options->informat      = PETSC_VIEWER_HDF5_XDMF;
  options->outformat     = PETSC_VIEWER_HDF5_XDMF;
  options->redistribute  = PETSC_TRUE;
  options->heterogeneous = PETSC_FALSE;
  options->ntimes        = 2;
  ierr = PetscOptionsBegin(comm, "", "Meshing Problem Options", "DMPLEX");CHKERRQ(ierr);
  CHKERRQ(PetscOptionsString("-infile", "The input mesh file", EX, options->infile, options->infile, sizeof(options->infile), &flg));
  PetscCheckFalse(!flg,comm, PETSC_ERR_USER_INPUT, "-infile needs to be specified");
  CHKERRQ(PetscOptionsString("-outfile", "The output mesh file (by default it's the same as infile)", EX, options->outfile, options->outfile, sizeof(options->outfile), &flg));
  PetscCheckFalse(!flg,comm, PETSC_ERR_USER_INPUT, "-outfile needs to be specified");
  CHKERRQ(PetscOptionsEnum("-informat", "Input mesh format", EX, PetscViewerFormats, (PetscEnum)options->informat, (PetscEnum*)&options->informat, NULL));
  CHKERRQ(PetscOptionsEnum("-outformat", "Dump/reload mesh format", EX, PetscViewerFormats, (PetscEnum)options->outformat, (PetscEnum*)&options->outformat, NULL));
  CHKERRQ(PetscOptionsBool("-redistribute", "Redistribute the mesh", EX, options->redistribute, &options->redistribute, NULL));
  CHKERRQ(PetscOptionsBool("-heterogeneous", "Test save on N / load on M", EX, options->heterogeneous, &options->heterogeneous, NULL));
  CHKERRQ(PetscOptionsInt("-ntimes", "How many times do the cycle", EX, options->ntimes, &options->ntimes, NULL));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
};

//TODO test DMLabel I/O (not yet working for PETSC_VIEWER_HDF5_XDMF)
int main(int argc, char **argv)
{
  AppCtx            user;
  MPI_Comm          comm;
  PetscMPIInt       gsize, grank, mycolor;
  PetscInt          i;
  PetscBool         flg;
  PetscErrorCode    ierr;
  const char        exampleDMPlexName[] = "DMPlex Object";
  const char        *infilename;
  PetscViewerFormat informat;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  CHKERRQ(ProcessOptions(PETSC_COMM_WORLD, &user));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&gsize));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&grank));

  for (i=0; i<user.ntimes; i++) {
    if (i==0) {
      /* Use infile/informat for the initial load */
      infilename = user.infile;
      informat   = user.informat;
    } else {
      /* Use outfile/outformat for all I/O except the very initial load */
      infilename = user.outfile;
      informat   = user.outformat;
    }

    if (user.heterogeneous) {
      mycolor = (PetscMPIInt)(grank > user.ntimes-i);
    } else {
      mycolor = (PetscMPIInt)0;
      /* comm = PETSC_COMM_WORLD; */
    }
    CHKERRMPI(MPI_Comm_split(PETSC_COMM_WORLD,mycolor,grank,&comm));

    if (mycolor == 0) {
      /* Load/Save only on processes with mycolor == 0 */
      DM                dm;
      PetscPartitioner  part;
      PetscViewer       v;

      CHKERRQ(PetscPrintf(comm, "Begin cycle %D\n",i));

      /* Load data from XDMF into dm in parallel */
      /* We could also use
          CHKERRQ(DMPlexCreateFromFile(PETSC_COMM_WORLD, user.filename, "ex5_plex", PETSC_TRUE, &dm));
        This currently support a few more formats than DMLoad().
      */
      CHKERRQ(PetscViewerHDF5Open(comm, infilename, FILE_MODE_READ, &v));
      CHKERRQ(PetscViewerPushFormat(v, informat));
      CHKERRQ(DMCreate(comm, &dm));
      CHKERRQ(DMSetType(dm, DMPLEX));
      CHKERRQ(PetscObjectSetName((PetscObject) dm, exampleDMPlexName));
      CHKERRQ(DMSetOptionsPrefix(dm,"loaded_"));
      CHKERRQ(DMLoad(dm, v));
      CHKERRQ(DMPlexDistributeSetDefault(dm, PETSC_FALSE));
      CHKERRQ(DMSetFromOptions(dm));
      CHKERRQ(DMViewFromOptions(dm, NULL, "-dm_view"));
      CHKERRQ(PetscViewerPopFormat(v));
      CHKERRQ(PetscViewerDestroy(&v));

      /* We just test/demonstrate DM is indeed distributed - unneeded in the application code */
      CHKERRQ(DMPlexIsDistributed(dm, &flg));
      CHKERRQ(PetscPrintf(comm, "Loaded mesh distributed? %s\n", PetscBools[flg]));

      /* Interpolate */
      //TODO we want to be able to do this from options in DMSetFromOptions() probably
      //TODO we want to be able to do this in-place
      {
        DM idm;

        CHKERRQ(DMPlexInterpolate(dm, &idm));
        CHKERRQ(DMDestroy(&dm));
        dm   = idm;
          CHKERRQ(DMSetOptionsPrefix(dm,"interpolated_"));
          CHKERRQ(DMSetFromOptions(dm));
          CHKERRQ(DMViewFromOptions(dm, NULL, "-dm_view"));
      }

      /* Redistribute */
      //TODO we want to be able to do this from options in DMSetFromOptions() probably
      if (user.redistribute) {
        DM dmdist;

        CHKERRQ(DMPlexGetPartitioner(dm, &part));
        CHKERRQ(PetscPartitionerSetFromOptions(part));
        CHKERRQ(DMPlexDistribute(dm, 0, NULL, &dmdist));
        //TODO we want to be able to do this in-place
        if (dmdist) {
          CHKERRQ(DMDestroy(&dm));
          dm   = dmdist;
          CHKERRQ(DMSetOptionsPrefix(dm,"redistributed_"));
          CHKERRQ(DMSetFromOptions(dm));
          CHKERRQ(DMViewFromOptions(dm, NULL, "-dm_view"));
        }
      }

      /* Save redistributed dm to XDMF in parallel and destroy it */
      CHKERRQ(PetscViewerHDF5Open(comm, user.outfile, FILE_MODE_WRITE, &v));
      CHKERRQ(PetscViewerPushFormat(v, user.outformat));
      CHKERRQ(PetscObjectSetName((PetscObject) dm, exampleDMPlexName));
      CHKERRQ(DMView(dm, v));
      CHKERRQ(PetscViewerPopFormat(v));
      CHKERRQ(PetscViewerDestroy(&v));
      CHKERRQ(DMDestroy(&dm));

      CHKERRQ(PetscPrintf(comm, "End   cycle %D\n--------\n",i));
    }
    CHKERRMPI(MPI_Comm_free(&comm));
    CHKERRMPI(MPI_Barrier(PETSC_COMM_WORLD));
  }

  /* Final clean-up */
  ierr = PetscFinalize();
  return ierr;
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
    args: -ntimes 3
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
    args: -ntimes 3
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
