#include <petscsnes.h>
#include <petscdmda.h>

int main(int argc, char *argv[]) {
  DM              da, daX, daY;
  DMDALocalInfo     info;
  MPI_Comm        commX, commY;
  Vec             basisX, basisY;
  PetscScalar   **arrayX, **arrayY;
  const PetscInt *lx, *ly;
  PetscInt        M = 3, N = 3;
  PetscInt        p = 1;
  PetscInt        numGP = 3;
  PetscInt        dof = 2*(p+1)*numGP;
  PetscMPIInt     rank, subsize, subrank;
  PetscErrorCode  ierr;

  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRQ(ierr);
  /* Create 2D DMDA */
  ierr = DMDACreate2d(PETSC_COMM_WORLD, DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE, DMDA_STENCIL_STAR, M, N, PETSC_DECIDE, PETSC_DECIDE, 1, 1, PETSC_NULL, PETSC_NULL, &da);CHKERRQ(ierr);
  /* Create 1D DMDAs along two directions */
  ierr = DMDAGetOwnershipRanges(da, &lx, &ly, PETSC_NULL);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da, &info);CHKERRQ(ierr);
  ierr = DMDAGetProcessorSubsets(da, DMDA_X, &commX);CHKERRQ(ierr);
  ierr = DMDAGetProcessorSubsets(da, DMDA_Y, &commY);CHKERRQ(ierr);
  ierr = MPI_Comm_size(commX, &subsize);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(commX, &subrank);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF, "[%d]X subrank: %d subsize: %d\n", rank, subrank, subsize);
  ierr = MPI_Comm_size(commY, &subsize);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(commY, &subrank);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF, "[%d]Y subrank: %d subsize: %d\n", rank, subrank, subsize);
  ierr = DMDACreate1d(commX, DMDA_BOUNDARY_NONE, M, dof, 1, lx, &daX);CHKERRQ(ierr);
  ierr = DMDACreate1d(commY, DMDA_BOUNDARY_NONE, N, dof, 1, ly, &daY);CHKERRQ(ierr);
  /* Create 1D vectors for basis functions */
  ierr = DMGetGlobalVector(daX, &basisX);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(daY, &basisY);CHKERRQ(ierr);
  /* Extract basis functions */
  ierr = DMDAVecGetArrayDOF(daX, basisX, &arrayX);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayDOF(daY, basisY, &arrayY);CHKERRQ(ierr);
  //arrayX[i][ndof];
  //arrayY[j][ndof];
  ierr = DMDAVecRestoreArrayDOF(daX, basisX, &arrayX);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayDOF(daY, basisY, &arrayY);CHKERRQ(ierr);
  /* Return basis vectors */
  ierr = DMRestoreGlobalVector(daX, &basisX);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(daY, &basisY);CHKERRQ(ierr);
  /* Cleanup */
  ierr = DMDestroy(&daX);CHKERRQ(ierr);
  ierr = DMDestroy(&daY);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
