
static char help[] = "Tests application ordering.\n\n";

#include <petscsys.h>
#include <petscao.h>
#include <petscviewer.h>

int main(int argc,char **argv)
{
  PetscMPIInt    rank,size;
  PetscInt       n,*ispetsc,*isapp,start,N,i;
  AO             ao;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));n = rank + 2;
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  /* create the orderings */
  PetscCall(PetscMalloc2(n,&ispetsc,n,&isapp));

  PetscCallMPI(MPI_Scan(&n,&start,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&n,&N,1,MPIU_INT,MPI_SUM,PETSC_COMM_WORLD));
  start -= n;

  for (i=0; i<n; i++) {
    ispetsc[i] = start + i;
    isapp[i]   = N - start - i - 1;
  }

  /* create the application ordering */
  PetscCall(AOCreateBasic(PETSC_COMM_WORLD,n,isapp,ispetsc,&ao));
  PetscCall(AOView(ao,PETSC_VIEWER_STDOUT_WORLD));

  /* check the mapping */
  PetscCall(AOPetscToApplication(ao,n,ispetsc));
  for (i=0; i<n; i++) {
    if (ispetsc[i] != isapp[i]) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"[%d] Problem with mapping %" PetscInt_FMT " to %" PetscInt_FMT "\n",rank,i,ispetsc[i]));
    }
  }
  PetscCall(PetscFree2(ispetsc,isapp));

  PetscCall(AODestroy(&ao));
  PetscCall(PetscFinalize());
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
