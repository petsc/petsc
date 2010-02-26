
static char help[] = "Tests PetscRandom functions.\n\n";

#include "petscsys.h"

/* Usage: 
   mpiexec -n <np> ./ex1 -n <num_of_random_numbers> -random_type <type> -log_summary
                         -view_randomvalues <view_rank> 
                         -random_view ascii -random_view_file <filename>
*/

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscInt       i,n = 1000,*values;
  PetscRandom    rnd;
  PetscScalar    value;
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  PetscInt       view_rank=-1;
#if defined (PETSC_USE_LOG)
  PetscLogEvent  event;
#endif

  PetscInitialize(&argc,&argv,(char *)0,help);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-view_randomvalues",&view_rank,PETSC_NULL);CHKERRQ(ierr);
  
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rnd);CHKERRQ(ierr);
#if defined(PETSC_HAVE_DRAND48)
  ierr = PetscRandomSetType(rnd,PETSCRAND48);CHKERRQ(ierr);
#elif defined(PETSC_HAVE_RAND)
  ierr = PetscRandomSetType(rnd,PETSCRAND);CHKERRQ(ierr);
#endif
  ierr = PetscRandomSetFromOptions(rnd);CHKERRQ(ierr); 

  ierr = PetscMalloc(n*sizeof(PetscInt),&values);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    ierr = PetscRandomGetValue(rnd,&value);CHKERRQ(ierr);
    if (view_rank == (PetscInt)rank) {
      ierr = PetscPrintf(PETSC_COMM_SELF,"[%D] value[%d] = %g\n",rank,i,PetscRealPart(value));CHKERRQ(ierr);
    }
    values[i] = (PetscInt)(n*PetscRealPart(value) + 2.0);
  }
  ierr = PetscSortInt(n,values);CHKERRQ(ierr);

  ierr = PetscLogEventRegister("Sort",0,&event);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(event,0,0,0,0);CHKERRQ(ierr);
 
  ierr = PetscRandomSeed(rnd);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    ierr = PetscRandomGetValue(rnd,&value);CHKERRQ(ierr);
    values[i] = (PetscInt)(n*PetscRealPart(value) + 2.0);
    /* printf("value[%d] = %g\n",i,value); */
  }
  ierr = PetscSortInt(n,values);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(event,0,0,0,0);CHKERRQ(ierr);

  for (i=1; i<n; i++) {
    if (values[i] < values[i-1]) SETERRQ(1,"Values not sorted");
  }
  ierr = PetscFree(values);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(rnd);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
 
