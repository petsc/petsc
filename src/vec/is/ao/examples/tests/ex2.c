
static char help[] = "Tests application ordering.\n\n";

#include <petscsys.h>
#include <petscao.h>
#include <petscviewer.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank,size;
  PetscInt       n,*ispetsc,*isapp,start,N,i;
  AO             ao;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr); n = rank + 2;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);

  /* create the orderings */
  ierr = PetscMalloc2(n,&ispetsc,n,&isapp);CHKERRQ(ierr);

  ierr   = MPI_Scan(&n,&start,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr   = MPI_Allreduce(&n,&N,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD);CHKERRQ(ierr);
  start -= n;

  for (i=0; i<n; i++) {
    ispetsc[i] = start + i;
    isapp[i]   = N - start - i - 1;
  }

  /* create the application ordering */
  ierr = AOCreateBasic(PETSC_COMM_WORLD,n,isapp,ispetsc,&ao);CHKERRQ(ierr);
  ierr = AOView(ao,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* check the mapping */
  ierr = AOPetscToApplication(ao,n,ispetsc);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    if (ispetsc[i] != isapp[i]) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"[%d] Problem with mapping %D to %D\n",rank,i,ispetsc[i]);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree2(ispetsc,isapp);CHKERRQ(ierr);

  ierr = AODestroy(&ao);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}





/*TEST

   test:

   test:
      suffix: 2
      nsize: 2

   test:
      suffix: 3
      nsize: 3

TEST*/
