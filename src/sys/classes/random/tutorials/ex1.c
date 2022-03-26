
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
  PetscMPIInt    rank;
  PetscInt       view_rank=-1;
#if defined(PETSC_USE_LOG)
  PetscLogEvent  event;
#endif

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-view_randomvalues",&view_rank,NULL));

  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD,&rnd));
  /* force imaginary part of random number to always be zero; thus obtain reproducible results with real and complex numbers */
  PetscCall(PetscRandomSetInterval(rnd,0.0,1.0));
  PetscCall(PetscRandomSetFromOptions(rnd));

  PetscCall(PetscMalloc1(n,&values));
  for (i=0; i<n; i++) {
    PetscCall(PetscRandomGetValue(rnd,&value));
    avg += value;
    if (view_rank == (PetscInt)rank) {
      PetscCall(PetscPrintf(PETSC_COMM_SELF,"[%d] value[%" PetscInt_FMT "] = %6.4e\n",rank,i,(double)PetscRealPart(value)));
    }
    values[i] = (PetscInt)(n*PetscRealPart(value) + 2.0);
  }
  avg = avg/((PetscReal)n);
  if (view_rank == (PetscInt)rank) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"[%d] Average value %6.4e\n",rank,(double)PetscRealPart(avg)));
  }

  PetscCall(PetscSortInt(n,values));

  PetscCall(PetscLogEventRegister("Sort",0,&event));
  PetscCall(PetscLogEventBegin(event,0,0,0,0));

  PetscCall(PetscRandomSeed(rnd));
  for (i=0; i<n; i++) {
    PetscCall(PetscRandomGetValue(rnd,&value));
    values[i] = (PetscInt)(n*PetscRealPart(value) + 2.0);
    /* printf("value[%d] = %g\n",i,value); */
  }
  PetscCall(PetscSortInt(n,values));
  PetscCall(PetscLogEventEnd(event,0,0,0,0));

  for (i=1; i<n; i++) {
    PetscCheckFalse(values[i] < values[i-1],PETSC_COMM_SELF,PETSC_ERR_PLIB,"Values not sorted");
  }
  PetscCall(PetscFree(values));
  PetscCall(PetscRandomDestroy(&rnd));

  PetscCall(PetscFinalize());
  return 0;
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
