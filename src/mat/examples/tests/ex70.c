
static char help[] = "Tests Vec/MatSetValues() with negative row and column indices.\n\n"; 

#include "petscmat.h"
#include "petscpc.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat            C; 
  PetscInt       i[2],j[2];
  PetscErrorCode ierr;
  PetscScalar    v[] = {1.0,2.0,3.0,4.0};
  Vec            x;

  PetscInitialize(&argc,&args,(char *)0,help);

  ierr = MatCreate(PETSC_COMM_WORLD,&C);CHKERRQ(ierr);
  ierr = MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,3,3);CHKERRQ(ierr);
  ierr = MatSetFromOptions(C);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_WORLD,3,&x);CHKERRQ(ierr);
  
  i[0] = 1; i[1] = -1; j[0] = 1; j[1] = 2;
  ierr = MatSetValues(C,2,i,2,j,v,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatSetValues(C,2,j,2,i,v,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecSetValues(x,2,i,v,INSERT_VALUES);CHKERRQ(ierr);
  
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatView(C,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = VecDestroy(x);CHKERRQ(ierr);
  ierr = MatDestroy(C);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}

 
