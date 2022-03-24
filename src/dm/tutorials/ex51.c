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

  CHKERRQ(PetscInitialize(&argc,&argv,0,help));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  /* Create 2D DMDA */
  CHKERRQ(DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,M,N,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&da));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetUp(da));
  /* Create 1D DMDAs along two directions. */
  CHKERRQ(DMDAGetOwnershipRanges(da, &lx, &ly, NULL));
  CHKERRQ(DMDAGetLocalInfo(da, &info));
  /* Partitioning in the X direction makes a subcomm extending in the Y direction and vice-versa. */
  CHKERRQ(DMDAGetProcessorSubsets(da, DM_X, &commY));
  CHKERRQ(DMDAGetProcessorSubsets(da, DM_Y, &commX));
  CHKERRMPI(MPI_Comm_size(commX, &subsize));
  CHKERRMPI(MPI_Comm_rank(commX, &subrank));
  CHKERRQ(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "[%d]X subrank: %d subsize: %d\n", rank, subrank, subsize));
  CHKERRQ(PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT));
  CHKERRMPI(MPI_Comm_size(commY, &subsize));
  CHKERRMPI(MPI_Comm_rank(commY, &subrank));
  CHKERRQ(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "[%d]Y subrank: %d subsize: %d\n", rank, subrank, subsize));
  CHKERRQ(PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT));
  CHKERRQ(DMDACreate1d(commX, DM_BOUNDARY_NONE, info.mx, dof, 1, lx, &daX));
  CHKERRQ(DMSetUp(daX));
  CHKERRQ(DMDACreate1d(commY, DM_BOUNDARY_NONE, info.my, dof, 1, ly, &daY));
  CHKERRQ(DMSetUp(daY));
  /* Create 1D vectors for basis functions */
  CHKERRQ(DMGetGlobalVector(daX, &basisX));
  CHKERRQ(DMGetGlobalVector(daY, &basisY));
  /* Extract basis functions */
  CHKERRQ(DMDAVecGetArrayDOF(daX, basisX, &arrayX));
  CHKERRQ(DMDAVecGetArrayDOF(daY, basisY, &arrayY));
  /*arrayX[i][ndof]; */
  /*arrayY[j][ndof]; */
  CHKERRQ(DMDAVecRestoreArrayDOF(daX, basisX, &arrayX));
  CHKERRQ(DMDAVecRestoreArrayDOF(daY, basisY, &arrayY));
  /* Return basis vectors */
  CHKERRQ(DMRestoreGlobalVector(daX, &basisX));
  CHKERRQ(DMRestoreGlobalVector(daY, &basisY));
  /* Cleanup */
  CHKERRMPI(MPI_Comm_free(&commX));
  CHKERRMPI(MPI_Comm_free(&commY));
  CHKERRQ(DMDestroy(&daX));
  CHKERRQ(DMDestroy(&daY));
  CHKERRQ(DMDestroy(&da));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 2

TEST*/
