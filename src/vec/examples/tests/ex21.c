/*$Id: ex21.c,v 1.8 2001/01/22 23:03:19 bsmith Exp balay $*/

static char help[] = "Tests VecMax() with index\
  -n <length> : vector length\n\n";

#include "petscvec.h"
#include "petscsys.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int           n = 5,ierr,idx;
  Scalar        value;
  Vec           x;
  PetscRandom   rand;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr); 
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);

  /* create vector */
  ierr = VecCreateMPI(PETSC_COMM_WORLD,PETSC_DECIDE,n,&x);CHKERRQ(ierr);

  ierr = PetscRandomCreate(PETSC_COMM_WORLD,RANDOM_DEFAULT_REAL,&rand);CHKERRQ(ierr);
  ierr = VecSetRandom(rand,x);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(rand);CHKERRQ(ierr);

  ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = VecMax(x,&idx,&value);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Maximum value %g index %d\n",value,idx);CHKERRQ(ierr);
  ierr = VecMin(x,&idx,&value);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Minimum value %g index %d\n",value,idx);CHKERRQ(ierr);

  ierr = VecDestroy(x);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
 
