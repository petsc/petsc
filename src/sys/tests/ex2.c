
static char help[] = "Tests the signal handler.\n";

#include <petscsys.h>

int CreateError(int n)
{
  PetscReal      *x = 0;
  if (!n) {x[0] = 100.; return 0;}
  PetscCall(CreateError(n-1));
  return 0;
}

int main(int argc,char **argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscFPrintf(PETSC_COMM_WORLD,stdout,"Demonstrates how PETSc can trap error interrupts\n"));
  PetscCall(PetscFPrintf(PETSC_COMM_WORLD,stdout,"The error below is contrived to test the code!\n"));
  PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT));
  PetscCall(CreateError(5));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
     args: -error_output_stdout
     filter: egrep "(Caught signal number 11 SEGV|Caught signal number 4 Illegal)" | wc -l
     TODO:  Does not always produce exactly expected output on all systems for all runs

TEST*/
