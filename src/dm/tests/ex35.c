
static char help[] = "MatLoad test for loading matrices that are created by DMCreateMatrix() and\n\
                      stored in binary via MatView_MPI_DA.MatView_MPI_DA stores the matrix\n\
                      in natural ordering. Hence MatLoad() has to read the matrix first in\n\
                      natural ordering and then permute it back to the application ordering.This\n\
                      example is used for testing the subroutine MatLoad_MPI_DA\n\n";

#include <petscdm.h>
#include <petscdmda.h>

int main(int argc,char **argv)
{
  PetscInt       X = 10,Y = 8,Z=8;
  PetscErrorCode ierr;
  DM             da;
  PetscViewer    viewer;
  Mat            A;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"temp.dat",FILE_MODE_WRITE,&viewer));

  /* Read options */
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-X",&X,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-Y",&Y,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-Z",&Z,NULL));

  /* Create distributed array and get vectors */
  CHKERRQ(DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,X,Y,Z,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,NULL,&da));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetUp(da));
  CHKERRQ(DMSetMatType(da,MATMPIAIJ));
  CHKERRQ(DMCreateMatrix(da,&A));
  CHKERRQ(MatShift(A,X));
  CHKERRQ(MatView(A,viewer));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(PetscViewerDestroy(&viewer));

  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"temp.dat",FILE_MODE_READ,&viewer));
  CHKERRQ(DMCreateMatrix(da,&A));
  CHKERRQ(MatLoad(A,viewer));

  /* Free memory */
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(PetscViewerDestroy(&viewer));
  CHKERRQ(DMDestroy(&da));
  ierr = PetscFinalize();
  return ierr;
}
