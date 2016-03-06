static char help[] = "Load and save the mesh and fields to HDF5 and ExodusII\n\n";

#include <petscdmplex.h>
#include <petscviewerhdf5.h>

typedef struct {
  PetscBool interpolate;                  /* Generate intermediate mesh elements */
  char      filename[PETSC_MAX_PATH_LEN]; /* Mesh filename */
} AppCtx;

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->interpolate = PETSC_FALSE;
  options->filename[0] = '\0';

  ierr = PetscOptionsBegin(comm, "", "Meshing Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-interpolate", "Generate intermediate mesh elements", "ex5.c", options->interpolate, &options->interpolate, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-filename", "The mesh file", "ex5.c", options->filename, options->filename, PETSC_MAX_PATH_LEN, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
};

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  DM             dm, dmdist, dmnew;
  AppCtx         user;
  PetscViewer    v;
  PetscSF        pointSF;
  PetscBool      flg;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);CHKERRQ(ierr);
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = DMPlexCreateFromFile(PETSC_COMM_WORLD, user.filename, user.interpolate, &dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm, NULL, "-orig_dm_view");CHKERRQ(ierr);
  ierr = DMPlexDistribute(dm, 0, &pointSF, &dmdist);CHKERRQ(ierr);
  if (dmdist) {
    ierr = DMDestroy(&dm);CHKERRQ(ierr);
    dm   = dmdist;
  }
  ierr = PetscSFDestroy(&pointSF);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);

  ierr = PetscViewerHDF5Open(PetscObjectComm((PetscObject) dm), "dmdist.h5", FILE_MODE_WRITE, &v); CHKERRQ(ierr);
  ierr = DMView(dm, v);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&v);CHKERRQ(ierr);

  ierr = DMCreate(PetscObjectComm((PetscObject) dm), &dmnew);CHKERRQ(ierr);
  ierr = DMSetType(dmnew, DMPLEX);CHKERRQ(ierr);
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, "dmdist.h5", FILE_MODE_READ, &v);CHKERRQ(ierr);
  ierr = DMLoad(dmnew, v);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&v);CHKERRQ(ierr);
  ierr = DMViewFromOptions(dmnew, NULL, "-new_dm_view");CHKERRQ(ierr);

  /* The NATIVE format for coordiante viewing is killing parallel output, since we have a local vector. Map it to global, and it will work. */
  ierr = DMPlexEqual(dmnew, dm, &flg);CHKERRQ(ierr);
  if (flg) {ierr = PetscPrintf(PETSC_COMM_WORLD,"DMs equal\n");CHKERRQ(ierr);}
  else     {ierr = PetscPrintf(PETSC_COMM_WORLD,"DMs are not equal\n");CHKERRQ(ierr);}

  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = DMDestroy(&dmnew);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
