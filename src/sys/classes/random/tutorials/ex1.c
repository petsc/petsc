
static char help[] = "Tests PetscRandom functions.\n\n";

#include <petscsys.h>

/* Usage:
   mpiexec -n <np> ./ex1 -n <num_of_random_numbers> -random_type <type> -log_view
                         -view_randomvalues <view_rank>
                         -random_view ascii -random_view :filename
*/

int main(int argc,char **argv)
{
  PetscInt       i,n = 1000,*values;
  PetscRandom    rnd;
  PetscScalar    value,avg = 0.0;
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  PetscInt       view_rank=-1;
#if defined(PETSC_USE_LOG)
  PetscLogEvent  event;
#endif

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-view_randomvalues",&view_rank,NULL));

  CHKERRQ(PetscRandomCreate(PETSC_COMM_WORLD,&rnd));
  /* force imaginary part of random number to always be zero; thus obtain reproducible results with real and complex numbers */
  CHKERRQ(PetscRandomSetInterval(rnd,0.0,1.0));
  CHKERRQ(PetscRandomSetFromOptions(rnd));

  CHKERRQ(PetscMalloc1(n,&values));
  for (i=0; i<n; i++) {
    CHKERRQ(PetscRandomGetValue(rnd,&value));
    avg += value;
    if (view_rank == (PetscInt)rank) {
      CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"[%d] value[%" PetscInt_FMT "] = %6.4e\n",rank,i,(double)PetscRealPart(value)));
    }
    values[i] = (PetscInt)(n*PetscRealPart(value) + 2.0);
  }
  avg = avg/((PetscReal)n);
  if (view_rank == (PetscInt)rank) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"[%d] Average value %6.4e\n",rank,(double)PetscRealPart(avg)));
  }

  CHKERRQ(PetscSortInt(n,values));

  CHKERRQ(PetscLogEventRegister("Sort",0,&event));
  CHKERRQ(PetscLogEventBegin(event,0,0,0,0));

  CHKERRQ(PetscRandomSeed(rnd));
  for (i=0; i<n; i++) {
    CHKERRQ(PetscRandomGetValue(rnd,&value));
    values[i] = (PetscInt)(n*PetscRealPart(value) + 2.0);
    /* printf("value[%d] = %g\n",i,value); */
  }
  CHKERRQ(PetscSortInt(n,values));
  CHKERRQ(PetscLogEventEnd(event,0,0,0,0));

  for (i=1; i<n; i++) {
    PetscCheckFalse(values[i] < values[i-1],PETSC_COMM_SELF,PETSC_ERR_PLIB,"Values not sorted");
  }
  CHKERRQ(PetscFree(values));
  CHKERRQ(PetscRandomDestroy(&rnd));

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

   test:
      suffix: 2
      nsize: 2
      output_file: output/ex1_1.out

   test:
      suffix: 3
      args: -view_randomvalues 0

TEST*/
