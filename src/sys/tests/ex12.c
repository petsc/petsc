
static char help[] = "Tests timing PetscSortInt().\n\n";

#include <petscsys.h>

int main(int argc,char **argv)
{
  PetscInt       i,n = 1000,*values;
  int            event;
  PetscRandom    rand;
  PetscReal      value;
  PetscErrorCode ierr;
  PetscBool      values_view=PETSC_FALSE;
  PetscMPIInt    rank;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,0,"-values_view",&values_view,NULL);CHKERRQ(ierr);

  ierr = PetscRandomCreate(PETSC_COMM_SELF,&rand);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rand);CHKERRQ(ierr);

  ierr = PetscMalloc1(n,&values);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    ierr      = PetscRandomGetValueReal(rand,&value);CHKERRQ(ierr);
    values[i] = (PetscInt)(n*value + 2.0);
  }
  ierr = PetscSortInt(n,values);CHKERRQ(ierr);

  ierr = PetscLogEventRegister("Sort",0,&event);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(event,0,0,0,0);CHKERRQ(ierr);

  for (i=0; i<n; i++) {
    ierr      = PetscRandomGetValueReal(rand,&value);CHKERRQ(ierr);
    values[i] = (PetscInt)(n*value + 2.0);
  }
  ierr = PetscSortInt(n,values);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(event,0,0,0,0);CHKERRQ(ierr);

  for (i=1; i<n; i++) {
    if (values[i] < values[i-1]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Values not sorted");
    if (values_view && !rank) {ierr = PetscPrintf(PETSC_COMM_SELF,"%D %D\n",i,values[i]);CHKERRQ(ierr);}
  }
  ierr = PetscFree(values);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}



/*TEST

   test:
      args: -values_view

TEST*/
