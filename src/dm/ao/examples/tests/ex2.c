
static char help[] = "Tests application ordering.\n\n";

#include "petsc.h"
#include "petscao.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank,size;
  PetscInt       n,*ispetsc,*isapp,start,N,i;
  AO             ao;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr); 
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr); n = rank + 2;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);

  /* create the orderings */
  ierr = PetscMalloc(2*n*sizeof(PetscInt),&ispetsc);CHKERRQ(ierr);
  isapp   = ispetsc + n;

  ierr = MPI_Scan(&n,&start,1,MPI_INT,MPI_SUM,PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&n,&N,1,MPI_INT,MPI_SUM,PETSC_COMM_WORLD);CHKERRQ(ierr);
  start -= n;

  for (i=0; i<n; i++) {  
    ispetsc[i] = start + i;
    isapp[i]   = N - start - i - 1;
  }

  /* create the application ordering */
  ierr = AOCreateBasic(PETSC_COMM_WORLD,n,isapp,ispetsc,&ao);CHKERRQ(ierr);

  ierr = AOView(ao,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* check the mapping */
  ierr = AOPetscToApplication(ao,n,ispetsc);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    if (ispetsc[i] != isapp[i]) {
      fprintf(stdout,"[%d] Problem with mapping %d to %d\n",rank,(int)i,(int)ispetsc[i]);
    }
  }

  ierr = PetscFree(ispetsc);CHKERRQ(ierr);

  ierr = AODestroy(ao);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
 


