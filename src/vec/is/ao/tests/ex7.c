
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
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);

  /* create the index sets */
  ierr = ISCreateStride(PETSC_COMM_WORLD,n,rank,size,&ispetsc);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_WORLD,n,n*rank,1,&isapp);CHKERRQ(ierr);

  /* create the application ordering */
  ierr = AOCreateBasicIS(isapp,ispetsc,&ao);CHKERRQ(ierr);


  ierr = AOView(ao,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = ISView(ispetsc,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = ISView(isapp,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = AOPetscToApplicationIS(ao,ispetsc);CHKERRQ(ierr);
  ierr = ISView(isapp,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = ISView(ispetsc,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);


  ierr = ISDestroy(&ispetsc);CHKERRQ(ierr);
  ierr = ISDestroy(&isapp);CHKERRQ(ierr);

  ierr = AODestroy(&ao);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      nsize: 2

TEST*/
