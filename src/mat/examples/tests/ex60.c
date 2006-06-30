
static char help[] = "Tests MatGetColumnVector().";

#include "petscmat.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat            C;
  PetscInt       i,j,m = 3,n = 2,Ii,J,col = 0;
  PetscMPIInt    size,rank;
  PetscErrorCode ierr;
  PetscScalar    v;
  Vec            yy;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-col",&col,PETSC_NULL);CHKERRQ(ierr);

  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  n = 2*size;

  /* create the matrix for the five point stencil, YET AGAIN*/
  ierr = MatCreateMPIAIJ(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,
         m*n,m*n,5,PETSC_NULL,5,PETSC_NULL,&C);CHKERRQ(ierr);
  for (i=0; i<m; i++) { 
    for (j=2*rank; j<2*rank+2; j++) {
      v = -1.0;  Ii = j + n*i;
      if (i>0)   {J = Ii - n; ierr = MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (i<m-1) {J = Ii + n; ierr = MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (j>0)   {J = Ii - 1; ierr = MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (j<n-1) {J = Ii + 1; ierr = MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      v = 4.0; ierr = MatSetValues(C,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatView(C,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&yy);CHKERRQ(ierr);
  ierr = VecSetSizes(yy,PETSC_DECIDE,m*n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(yy);CHKERRQ(ierr);

  ierr = MatGetColumnVector(C,yy,col);CHKERRQ(ierr);

  ierr = VecView(yy,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = VecDestroy(yy);CHKERRQ(ierr);
  ierr = MatDestroy(C);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
