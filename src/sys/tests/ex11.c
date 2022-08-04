
static char help[] = "Tests PetscSynchronizedPrintf() and PetscSynchronizedFPrintf().\n\n";

#include <petscsys.h>

int main(int argc,char **argv)
{
  PetscMPIInt    rank;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"Greetings from %d\n",rank));
  PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT));

  PetscCall(PetscSynchronizedFPrintf(PETSC_COMM_WORLD,PETSC_STDOUT,"Greetings again from %d\n",rank));
  PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 3

TEST*/
