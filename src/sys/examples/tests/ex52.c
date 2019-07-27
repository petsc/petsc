static char help[] = "A benchmark for testing PetscSortInt() and PetscSortIntWithArrayPair()\n\
  The array is filled with random numbers, but one can control average duplicates for each unique integer with the -d option.\n\
  Usage:\n\
   mpirun -n 1 ./ex52 -n <length of the array to sort>, default=100 \n\
                      -r <repeat times for each sort>, default=10 \n\
                      -d <average duplicates for each unique integer>, default=1, i.e., no duplicates \n\n";

#include <petscsys.h>
#include <petsctime.h>
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       i,l,n=100,r=10,d=1;
  PetscInt       *X,*Y,*Z;
  PetscReal      val;
  PetscRandom    rdm;
  PetscLogDouble time;
  PetscMPIInt    size;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"This is a uniprocessor example only!");

  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-r",&r,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-d",&d,NULL);CHKERRQ(ierr);
  if (n<1 || r<1 || d<1 || d>n) SETERRQ3(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Wrong input n=%D,r=%D,d=%d. They must be >=1 and n>=d\n",n,r,d);

  ierr = PetscCalloc3(n,&X,n,&Y,n,&Z);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_SELF,&rdm);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rdm);CHKERRQ(ierr);

  time = 0.0;
  for (l=0; l<r; l++) { /* r loops */
    for (i=0; i<n; i++) { /* Init X[] */
      ierr = PetscRandomGetValueReal(rdm,&val);CHKERRQ(ierr);
      X[i] = val*PETSC_MAX_INT;
      if (d > 1) X[i] = X[i] % (n/d);
    }

    ierr = PetscTimeSubtract(&time);CHKERRQ(ierr);
    ierr = PetscSortInt(n,X);CHKERRQ(ierr);
    ierr = PetscTimeAdd(&time);CHKERRQ(ierr);

    for (i=0; i<n-1; i++) {if (X[i] > X[i+1]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"PetscSortInt() produced wrong results!");}
  }
  ierr = PetscPrintf(PETSC_COMM_SELF,"PetscSortInt()              with %D integers, %D duplicate(s) per unique value took %g seconds\n",n,d,time/r);CHKERRQ(ierr);

  time = 0.0;
  for (l=0; l<r; l++) { /* r loops */
    for (i=0; i<n; i++) { /* Init X[] */
      ierr = PetscRandomGetValueReal(rdm,&val);CHKERRQ(ierr);
      X[i] = val*PETSC_MAX_INT;
      if (d > 1) X[i] = X[i] % (n/d);
    }

    ierr = PetscTimeSubtract(&time);CHKERRQ(ierr);
    ierr = PetscSortIntWithArrayPair(n,X,Y,Z);CHKERRQ(ierr);
    ierr = PetscTimeAdd(&time);CHKERRQ(ierr);

    for (i=0; i<n-1; i++) {if (X[i] > X[i+1]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"PetscSortInt() produced wrong results!");}
  }
  ierr = PetscPrintf(PETSC_COMM_SELF,"PetscSortIntWithArrayPair() with %D integers, %D duplicate(s) per unique value took %g seconds\n",n,d,time/r);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"SUCCEEDED\n");CHKERRQ(ierr);

  ierr = PetscRandomDestroy(&rdm);CHKERRQ(ierr);
  ierr = PetscFree3(X,Y,Z);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      args: -n 1000 -r 1 -d 1
      # Do not need to output timing results for test
      filter: grep -v "per unique value took"

TEST*/
