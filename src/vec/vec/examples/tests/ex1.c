
static char help[] = "Tests repeated VecSetType().\n\n";

#include <petscvec.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       n = 5;
  PetscScalar    one = 1.0,two = 2.0;
  Vec            x,y;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr); 

  /* create vector */
  ierr = VecCreate(PETSC_COMM_SELF,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,n,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetType(x,"mpi");CHKERRQ(ierr);
  ierr = VecSetType(x,"seq");CHKERRQ(ierr);
  ierr = VecDuplicate(x,&y);CHKERRQ(ierr);
  ierr = VecSetType(x,"mpi");CHKERRQ(ierr);

  ierr = VecSet(x,one);CHKERRQ(ierr);
  ierr = VecSet(y,two);CHKERRQ(ierr);

  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return 0;
}
 
