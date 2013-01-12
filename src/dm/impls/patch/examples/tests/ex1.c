static char help[] = "Make a 2D grid of patches and view them\n\n";

/*
Serial Test
Parallel Test where all zooms are serials
Parallel Test where zooms are parallel

Return DMPatch from Zoom
Override refine from DMPatch to split cells
 */
#include <petscdmpatch.h>

typedef struct {
  PetscInt   debug;     /* The debugging level */
  MatStencil patchSize; /* Size of patches */
  MatStencil gridSize;  /* Size of patch grid */
} AppCtx;

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options) {
  PetscInt       patchSize, gridSize;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->debug = 0;
  patchSize      = 0;
  gridSize       = 0;

  ierr = PetscOptionsBegin(comm, "", "Patch Test Options", "DMPATCH");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-debug", "The debugging level", "ex1.c", options->debug, &options->debug, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-patch_size", "The patch size in each dimension", "ex1.c", patchSize, &patchSize, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-grid_size", "The grid size in each dimension", "ex1.c", gridSize, &gridSize, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  options->patchSize.i = options->patchSize.j = options->patchSize.k = patchSize;
  options->gridSize.i  = options->gridSize.j  = options->gridSize.k  = gridSize;
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
  ierr = DMPatchSolve(dm);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
