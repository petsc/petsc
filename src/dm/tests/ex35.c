
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
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"temp.dat",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);

  /* Read options */
  ierr = PetscOptionsGetInt(NULL,NULL,"-X",&X,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-Y",&Y,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-Z",&Z,NULL);CHKERRQ(ierr);

  /* Create distributed array and get vectors */
  ierr = DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,X,Y,Z,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,NULL,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = DMSetMatType(da,MATMPIAIJ);CHKERRQ(ierr);
  ierr = DMCreateMatrix(da,&A);CHKERRQ(ierr);
  ierr = MatShift(A,X);CHKERRQ(ierr);
  ierr = MatView(A,viewer);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"temp.dat",FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = DMCreateMatrix(da,&A);CHKERRQ(ierr);
  ierr = MatLoad(A,viewer);CHKERRQ(ierr);

  /* Free memory */
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
