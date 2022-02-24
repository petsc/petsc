
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
  PetscErrorCode ierr;
  PetscBool      values_view=PETSC_FALSE;
  PetscMPIInt    rank;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,0,"-values_view",&values_view,NULL));

  CHKERRQ(PetscRandomCreate(PETSC_COMM_SELF,&rand));
  CHKERRQ(PetscRandomSetFromOptions(rand));

  CHKERRQ(PetscMalloc1(n,&values));
  for (i=0; i<n; i++) {
    CHKERRQ(PetscRandomGetValueReal(rand,&value));
    values[i] = (PetscInt)(n*value + 2.0);
  }
  CHKERRQ(PetscSortInt(n,values));

  CHKERRQ(PetscLogEventRegister("Sort",0,&event));
  CHKERRQ(PetscLogEventBegin(event,0,0,0,0));

  for (i=0; i<n; i++) {
    CHKERRQ(PetscRandomGetValueReal(rand,&value));
    values[i] = (PetscInt)(n*value + 2.0);
  }
  CHKERRQ(PetscSortInt(n,values));
  CHKERRQ(PetscLogEventEnd(event,0,0,0,0));

  for (i=1; i<n; i++) {
    PetscCheckFalse(values[i] < values[i-1],PETSC_COMM_SELF,PETSC_ERR_PLIB,"Values not sorted");
    if (values_view && rank == 0) CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"%" PetscInt_FMT " %" PetscInt_FMT "\n",i,values[i]));
  }
  CHKERRQ(PetscFree(values));
  CHKERRQ(PetscRandomDestroy(&rand));

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      args: -values_view

TEST*/
