/*$Id: ex3.c,v 1.48 2001/01/15 21:45:13 bsmith Exp bsmith $*/

static char help[] = "Tests parallel vector assembly.  Input arguments are\n\
  -n <length> : local vector length\n\n";

#include "petscvec.h"
#include "petscsys.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int          n = 5,ierr,size,rank;
  Scalar       one = 1.0,two = 2.0,three = 3.0;
  Vec          x,y;
  int          idx;

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);
  if (n < 5) n = 5;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  if (size < 2) SETERRQ(1,"Must be run with at least two processors");

  /* create two vector */
  ierr = VecCreateSeq(PETSC_COMM_SELF,n,&x);CHKERRQ(ierr);
  ierr = VecCreateMPI(PETSC_COMM_WORLD,n,PETSC_DECIDE,&y);CHKERRQ(ierr);
  ierr = VecSet(&one,x);CHKERRQ(ierr);
  ierr = VecSet(&two,y);CHKERRQ(ierr);

  if (rank == 1) {
    idx = 2; ierr = VecSetValues(y,1,&idx,&three,INSERT_VALUES);CHKERRQ(ierr);
    idx = 0; ierr = VecSetValues(y,1,&idx,&two,INSERT_VALUES);CHKERRQ(ierr); 
    idx = 0; ierr = VecSetValues(y,1,&idx,&one,INSERT_VALUES);CHKERRQ(ierr); 
  }
  else {
    idx = 7; ierr = VecSetValues(y,1,&idx,&three,INSERT_VALUES);CHKERRQ(ierr); 
  } 
  ierr = VecAssemblyBegin(y);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(y);CHKERRQ(ierr);

  ierr = VecView(y,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = VecDestroy(x);CHKERRQ(ierr);
  ierr = VecDestroy(y);CHKERRQ(ierr);

  PetscFinalize();
  return 0;
}
 
