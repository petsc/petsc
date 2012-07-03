
static char help[] = "MatLoad test for loading matrices that are created by DMCreateMatrix and\n\
                      stored in binary via MatView_MPI_DA.MatView_MPI_DA stores the matrix\n\
                      in natural ordering. Hence MatLoad() has to read the matrix first in\n\
                      natural ordering and then permute it back to the application ordering.This\n\
                      example is used for testing the subroutine MatLoad_MPI_DA\n\n";

#include <petscdmda.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscInt       X = 10,Y = 8,Z=8;
  PetscErrorCode ierr;
  DM             da;
  PetscViewer    viewer;
  Mat            A;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr); 
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"temp.dat",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);

  /* Read options */
  ierr = PetscOptionsGetInt(PETSC_NULL,"-X",&X,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-Y",&Y,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-Z",&Z,PETSC_NULL);CHKERRQ(ierr);

  /* Create distributed array and get vectors */
  ierr = DMDACreate3d(PETSC_COMM_WORLD,DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,DMDA_STENCIL_STAR,
                    X,Y,Z,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,1,1,PETSC_NULL,PETSC_NULL,PETSC_NULL,&da);CHKERRQ(ierr);
  ierr = DMCreateMatrix(da,MATMPIAIJ,&A);CHKERRQ(ierr);
  ierr = MatShift(A,X);CHKERRQ(ierr);
  ierr = MatView(A,viewer);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"temp.dat",FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = DMCreateMatrix(da,MATMPIAIJ,&A);CHKERRQ(ierr);
  ierr = MatLoad(A,viewer);CHKERRQ(ierr);

  /* Free memory */
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
 
