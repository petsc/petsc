
static char help[] = "Demonstrates constructing an application ordering.\n\n";

#include <petscao.h>
#include <petscviewer.h>

int main(int argc,char **argv)
{
  PetscInt       n = 5;
  PetscMPIInt    rank,size;
  IS             ispetsc,isapp;
  AO             ao;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  /* create the index sets */
  PetscCall(ISCreateStride(PETSC_COMM_WORLD,n,rank,size,&ispetsc));
  PetscCall(ISCreateStride(PETSC_COMM_WORLD,n,n*rank,1,&isapp));

  /* create the application ordering */
  PetscCall(AOCreateBasicIS(isapp,ispetsc,&ao));

  PetscCall(AOView(ao,PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(ISView(ispetsc,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(ISView(isapp,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(AOPetscToApplicationIS(ao,ispetsc));
  PetscCall(ISView(isapp,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(ISView(ispetsc,PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(ISDestroy(&ispetsc));
  PetscCall(ISDestroy(&isapp));

  PetscCall(AODestroy(&ao));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 2

TEST*/
