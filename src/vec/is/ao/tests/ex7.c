
static char help[] = "Demonstrates constructing an application ordering.\n\n";

#include <petscao.h>
#include <petscviewer.h>

int main(int argc,char **argv)
{
  PetscInt       n = 5;
  PetscErrorCode ierr;
  PetscMPIInt    rank,size;
  IS             ispetsc,isapp;
  AO             ao;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  /* create the index sets */
  CHKERRQ(ISCreateStride(PETSC_COMM_WORLD,n,rank,size,&ispetsc));
  CHKERRQ(ISCreateStride(PETSC_COMM_WORLD,n,n*rank,1,&isapp));

  /* create the application ordering */
  CHKERRQ(AOCreateBasicIS(isapp,ispetsc,&ao));

  CHKERRQ(AOView(ao,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(ISView(ispetsc,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(ISView(isapp,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(AOPetscToApplicationIS(ao,ispetsc));
  CHKERRQ(ISView(isapp,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(ISView(ispetsc,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(ISDestroy(&ispetsc));
  CHKERRQ(ISDestroy(&isapp));

  CHKERRQ(AODestroy(&ao));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      nsize: 2

TEST*/
