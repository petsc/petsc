
static char help[] = "Passes a sparse matrix to Matlab.\n\n";

#include "petscmat.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  int         ierr,m = 4,n = 5,i,j,I,J;
  PetscScalar one = 1.0,v;
  Vec         x;
  Mat         A;
  PetscViewer viewer;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-m",&m,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);

  ierr = PetscViewerSocketOpen(PETSC_COMM_WORLD,"eagle",-1,&viewer);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n,&A);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);

  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      v = -1.0;  I = j + n*i;
      if (i>0)   {J = I - n; ierr = MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (i<m-1) {J = I + n; ierr = MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (j>0)   {J = I - 1; ierr = MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (j<n-1) {J = I + 1; ierr = MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      v = 4.0; ierr = MatSetValues(A,1,&I,1,&I,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatView(A,viewer);CHKERRQ(ierr);

  ierr = VecCreateSeq(PETSC_COMM_SELF,m,&x);CHKERRQ(ierr);
  ierr = VecSet(&one,x);CHKERRQ(ierr);
  ierr = VecView(x,viewer);CHKERRQ(ierr);
  
  ierr = PetscSleep(30);CHKERRQ(ierr);
  ierr = PetscObjectDestroy((PetscObject)viewer);CHKERRQ(ierr);

  ierr = VecDestroy(x);CHKERRQ(ierr);
  ierr = MatDestroy(A);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
    


