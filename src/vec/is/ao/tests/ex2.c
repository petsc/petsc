
static char help[] = "Tests application ordering.\n\n";

#include <petscsys.h>
#include <petscao.h>
#include <petscviewer.h>

int main(int argc,char **argv)
{
  PetscMPIInt    rank,size;
  PetscInt       n,*ispetsc,*isapp,start,N,i;
  AO             ao;

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));n = rank + 2;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  /* create the orderings */
  CHKERRQ(PetscMalloc2(n,&ispetsc,n,&isapp));

  CHKERRMPI(MPI_Scan(&n,&start,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD));
  CHKERRMPI(MPI_Allreduce(&n,&N,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD));
  start -= n;

  for (i=0; i<n; i++) {
    ispetsc[i] = start + i;
    isapp[i]   = N - start - i - 1;
  }

  /* create the application ordering */
  CHKERRQ(AOCreateBasic(PETSC_COMM_WORLD,n,isapp,ispetsc,&ao));
  CHKERRQ(AOView(ao,PETSC_VIEWER_STDOUT_WORLD));

  /* check the mapping */
  CHKERRQ(AOPetscToApplication(ao,n,ispetsc));
  for (i=0; i<n; i++) {
    if (ispetsc[i] != isapp[i]) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"[%d] Problem with mapping %" PetscInt_FMT " to %" PetscInt_FMT "\n",rank,i,ispetsc[i]));
    }
  }
  CHKERRQ(PetscFree2(ispetsc,isapp));

  CHKERRQ(AODestroy(&ao));
  CHKERRQ(PetscFinalize());
  return 0;
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
