static char help[] = "Tests DMDAVecGetArrayDOF()\n";
#include <petscdm.h>
#include <petscdmda.h>

int main(int argc, char *argv[])
{
  DM             da, daX, daY;
  DMDALocalInfo  info;
  MPI_Comm       commX, commY;
  Vec            basisX, basisY;
  PetscScalar    **arrayX, **arrayY;
  const PetscInt *lx, *ly;
  PetscInt       M     = 3, N = 3;
  PetscInt       p     = 1;
  PetscInt       numGP = 3;
  PetscInt       dof   = 2*(p+1)*numGP;
  PetscMPIInt    rank, subsize, subrank;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRMPI(ierr);
  /* Create 2D DMDA */
  ierr = DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,M,N,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  /* Create 1D DMDAs along two directions. */
  ierr = DMDAGetOwnershipRanges(da, &lx, &ly, NULL);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da, &info);CHKERRQ(ierr);
  /* Partitioning in the X direction makes a subcomm extending in the Y direction and vice-versa. */
  ierr = DMDAGetProcessorSubsets(da, DM_X, &commY);CHKERRQ(ierr);
  ierr = DMDAGetProcessorSubsets(da, DM_Y, &commX);CHKERRQ(ierr);
  ierr = MPI_Comm_size(commX, &subsize);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(commX, &subrank);CHKERRMPI(ierr);
  ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "[%d]X subrank: %d subsize: %d\n", rank, subrank, subsize);CHKERRQ(ierr);
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);CHKERRQ(ierr);
  ierr = MPI_Comm_size(commY, &subsize);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(commY, &subrank);CHKERRMPI(ierr);
  ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "[%d]Y subrank: %d subsize: %d\n", rank, subrank, subsize);CHKERRQ(ierr);
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);CHKERRQ(ierr);
  ierr = DMDACreate1d(commX, DM_BOUNDARY_NONE, info.mx, dof, 1, lx, &daX);CHKERRQ(ierr);
  ierr = DMSetUp(daX);CHKERRQ(ierr);
  ierr = DMDACreate1d(commY, DM_BOUNDARY_NONE, info.my, dof, 1, ly, &daY);CHKERRQ(ierr);
  ierr = DMSetUp(daY);CHKERRQ(ierr);
  /* Create 1D vectors for basis functions */
  ierr = DMGetGlobalVector(daX, &basisX);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(daY, &basisY);CHKERRQ(ierr);
  /* Extract basis functions */
  ierr = DMDAVecGetArrayDOF(daX, basisX, &arrayX);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayDOF(daY, basisY, &arrayY);CHKERRQ(ierr);
  /*arrayX[i][ndof]; */
  /*arrayY[j][ndof]; */
  ierr = DMDAVecRestoreArrayDOF(daX, basisX, &arrayX);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayDOF(daY, basisY, &arrayY);CHKERRQ(ierr);
  /* Return basis vectors */
  ierr = DMRestoreGlobalVector(daX, &basisX);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(daY, &basisY);CHKERRQ(ierr);
  /* Cleanup */
  ierr = MPI_Comm_free(&commX);CHKERRMPI(ierr);
  ierr = MPI_Comm_free(&commY);CHKERRMPI(ierr);
  ierr = DMDestroy(&daX);CHKERRQ(ierr);
  ierr = DMDestroy(&daY);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}


/*TEST

   test:
      nsize: 2

TEST*/
