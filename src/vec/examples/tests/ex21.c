
static char help[] = "Tests VecMax() with index.\n\
  -n <length> : vector length\n\n";

#include "petscvec.h"
#include "petscsys.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       n = 5,idx;
  PetscReal      value;
  Vec            x;
  PetscRandom    rdm;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr); 
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);

  /* create vector */
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);

  ierr = PetscRandomCreate(PETSC_COMM_WORLD,RANDOM_DEFAULT_REAL,&rdm);CHKERRQ(ierr);
  ierr = VecSetRandom(rdm,x);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(rdm);CHKERRQ(ierr);

  ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = VecMax(x,&idx,&value);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Maximum value %g index %D\n",value,idx);CHKERRQ(ierr);
  ierr = VecMin(x,&idx,&value);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Minimum value %g index %D\n",value,idx);CHKERRQ(ierr);

  ierr = VecDestroy(x);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
 
