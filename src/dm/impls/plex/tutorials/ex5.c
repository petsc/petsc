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
  PetscInt          ntimes;                      /* How many times do the cycle */
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->infile[0]    = '\0';
  options->outfile[0]   = '\0';
  options->informat     = PETSC_VIEWER_HDF5_XDMF;
  options->outformat    = PETSC_VIEWER_HDF5_XDMF;
  options->redistribute = PETSC_TRUE;
  options->ntimes       = 2;
  ierr = PetscOptionsBegin(comm, "", "Meshing Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsString("-infile", "The input mesh file", EX, options->infile, options->infile, sizeof(options->infile), &flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(comm, PETSC_ERR_USER_INPUT, "-infile needs to be specified");
  ierr = PetscOptionsString("-outfile", "The output mesh file (by default it's the same as infile)", EX, options->outfile, options->outfile, sizeof(options->outfile), &flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(comm, PETSC_ERR_USER_INPUT, "-outfile needs to be specified");
  ierr = PetscOptionsEnum("-informat", "Input mesh format", EX, PetscViewerFormats, (PetscEnum)options->informat, (PetscEnum*)&options->informat, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-outformat", "Dump/reload mesh format", EX, PetscViewerFormats, (PetscEnum)options->outformat, (PetscEnum*)&options->outformat, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-redistribute", "Redistribute the mesh", EX, options->redistribute, &options->redistribute, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-ntimes", "How many times do the cycle", EX, options->ntimes, &options->ntimes, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
};

//TODO test DMLabel I/O (not yet working for PETSC_VIEWER_HDF5_XDMF)
int main(int argc, char **argv)
{
  AppCtx            user;
  PetscInt          i;
  PetscBool         flg;
  MPI_Comm          comm;
  PetscMPIInt       size;
  PetscErrorCode    ierr;
  const char        *filename;
  PetscViewerFormat format;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  ierr = ProcessOptions(comm, &user);CHKERRQ(ierr);

  /* Use infile for the initial load */
  filename = user.infile;
  format   = user.informat;

  for (i=0; i<user.ntimes; i++) {
    DM                dm;
    PetscPartitioner  part;
    PetscViewer       v;

    ierr = PetscPrintf(comm, "Begin cycle %D\n",i);CHKERRQ(ierr);

    /* Load data from XDMF into dm in parallel */
    /* We could also use
        ierr = DMPlexCreateFromFile(PETSC_COMM_WORLD, user.filename, PETSC_TRUE, &dm);CHKERRQ(ierr);
      This currently support a few more formats than DMLoad().
    */
    ierr = PetscViewerHDF5Open(comm, filename, FILE_MODE_READ, &v);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(v, format);CHKERRQ(ierr);
    ierr = DMCreate(comm, &dm);CHKERRQ(ierr);
    ierr = DMSetType(dm, DMPLEX);CHKERRQ(ierr);
    ierr = DMSetOptionsPrefix(dm,"loaded_");CHKERRQ(ierr);
    ierr = DMLoad(dm, v);CHKERRQ(ierr);
    ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
    ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(v);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&v);CHKERRQ(ierr);

    /* Use outfile for all I/O except the very initial load */
    filename = user.outfile;
    format   = user.outformat;

    /* We just test/demonstrate DM is indeed distributed - unneeded in the application code */
    ierr = DMPlexIsDistributed(dm, &flg);CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "Loaded mesh distributed? %s\n", PetscBools[flg]);

    /* Interpolate */
    //TODO we want to be able to do this from options in DMSetFromOptions() probably
    //TODO we want to be able to do this in-place
    {
      DM idm;

      ierr = DMPlexInterpolate(dm, &idm);CHKERRQ(ierr);
      ierr = DMDestroy(&dm);CHKERRQ(ierr);
      dm   = idm;
        ierr = DMSetOptionsPrefix(dm,"interpolated_");CHKERRQ(ierr);
        ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
        ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);
    }

    /* Redistribute */
    //TODO we want to be able to do this from options in DMSetFromOptions() probably
    if (user.redistribute) {
      DM dmdist;

      ierr = DMPlexGetPartitioner(dm, &part);CHKERRQ(ierr);
      ierr = PetscPartitionerSetFromOptions(part);CHKERRQ(ierr);
      ierr = DMPlexDistribute(dm, 0, NULL, &dmdist);CHKERRQ(ierr);
      //TODO we want to be able to do this in-place
      if (dmdist) {
        ierr = DMDestroy(&dm);CHKERRQ(ierr);
        dm   = dmdist;
        ierr = DMSetOptionsPrefix(dm,"redistributed_");CHKERRQ(ierr);
        ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
        ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);
      }
    }

    /* Save redistributed dm to XDMF in parallel and destroy it */
    ierr = PetscViewerHDF5Open(comm, user.outfile, FILE_MODE_WRITE, &v);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(v, format);CHKERRQ(ierr);
    ierr = DMView(dm, v);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(v);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&v);CHKERRQ(ierr);
    ierr = DMDestroy(&dm);CHKERRQ(ierr);

    ierr = PetscPrintf(comm, "End   cycle %D\n--------\n",i);CHKERRQ(ierr);
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
    nsize: {{2 4}}
    args: -infile ${wPETSC_DIR}/share/petsc/datafiles/meshes/blockcylinder-50.h5 -informat hdf5_xdmf
    args: -outfile ex5_dump.h5 -outformat {{hdf5_xdmf hdf5_petsc}separate output}
    args: -ntimes 3
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
TEST*/
