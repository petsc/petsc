static char help[] = "Make a 2D grid of patches and view them\n\n";

#include <petscdmpatch.h>

typedef struct {
  PetscInt   debug;     /* The debugging level */
  MatStencil patchSize; /* Size of patches */
  MatStencil gridSize;  /* Size of patch grid */
} AppCtx;

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options) {
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->debug = 0;
  options->patchSize.i = options->patchSize.j = options->patchSize.k = 2;
  options->gridSize.i  = options->gridSize.j  = options->gridSize.k  = 2;

  ierr = PetscOptionsBegin(comm, "", "Patch Test Options", "DMPATCH");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-debug", "The debugging level", "ex1.c", options->debug, &options->debug, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
};

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  DM             dm;
  AppCtx         user;                 /* user-defined work context */
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, PETSC_NULL, help);CHKERRQ(ierr);
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = DMPatchCreateGrid(PETSC_COMM_WORLD, 2, user.patchSize, user.gridSize, &dm);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) dm, "Patch Mesh");CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMSetUp(dm);CHKERRQ(ierr);
  ierr = DMView(dm, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
