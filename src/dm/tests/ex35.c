
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
  DM             da;
  PetscViewer    viewer;
  Mat            A;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"temp.dat",FILE_MODE_WRITE,&viewer));

  /* Read options */
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-X",&X,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-Y",&Y,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-Z",&Z,NULL));

  /* Create distributed array and get vectors */
  PetscCall(DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,X,Y,Z,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,NULL,&da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMSetMatType(da,MATMPIAIJ));
  PetscCall(DMCreateMatrix(da,&A));
  PetscCall(MatShift(A,X));
  PetscCall(MatView(A,viewer));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"temp.dat",FILE_MODE_READ,&viewer));
  PetscCall(DMCreateMatrix(da,&A));
  PetscCall(MatLoad(A,viewer));

  /* Free memory */
  PetscCall(MatDestroy(&A));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(DMDestroy(&da));
  PetscCall(PetscFinalize());
  return 0;
}
