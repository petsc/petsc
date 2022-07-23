
static char help[] = "Tests timing PetscSortInt().\n\n";

#include <petscsys.h>

int main(int argc,char **argv)
{
  PetscInt       i,n = 1000,*values;
#if defined(PETSC_USE_LOG)
  PetscLogEvent  event;
#endif
  PetscRandom    rand;
  PetscReal      value;
  PetscBool      values_view=PETSC_FALSE;
  PetscMPIInt    rank;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscOptionsGetBool(NULL,0,"-values_view",&values_view,NULL));

  PetscCall(PetscRandomCreate(PETSC_COMM_SELF,&rand));
  PetscCall(PetscRandomSetFromOptions(rand));

  PetscCall(PetscMalloc1(n,&values));
  for (i=0; i<n; i++) {
    PetscCall(PetscRandomGetValueReal(rand,&value));
    values[i] = (PetscInt)(n*value + 2.0);
  }
  PetscCall(PetscSortInt(n,values));

  PetscCall(PetscLogEventRegister("Sort",0,&event));
  PetscCall(PetscLogEventBegin(event,0,0,0,0));

  for (i=0; i<n; i++) {
    PetscCall(PetscRandomGetValueReal(rand,&value));
    values[i] = (PetscInt)(n*value + 2.0);
  }
  PetscCall(PetscSortInt(n,values));
  PetscCall(PetscLogEventEnd(event,0,0,0,0));

  for (i=1; i<n; i++) {
    PetscCheck(values[i] >= values[i-1],PETSC_COMM_SELF,PETSC_ERR_PLIB,"Values not sorted");
    if (values_view && rank == 0) PetscCall(PetscPrintf(PETSC_COMM_SELF,"%" PetscInt_FMT " %" PetscInt_FMT "\n",i,values[i]));
  }
  PetscCall(PetscFree(values));
  PetscCall(PetscRandomDestroy(&rand));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      args: -values_view

TEST*/
