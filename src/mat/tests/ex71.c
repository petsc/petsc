
static char help[] = "Passes a sparse matrix to MATLAB.\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  PetscErrorCode ierr;
  PetscInt       m   = 4,n = 5,i,j,II,J;
  PetscScalar    one = 1.0,v;
  Vec            x;
  Mat            A;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      v = -1.0;  II = j + n*i;
      if (i>0)   {J = II - n; ierr = MatSetValues(A,1,&II,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (i<m-1) {J = II + n; ierr = MatSetValues(A,1,&II,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (j>0)   {J = II - 1; ierr = MatSetValues(A,1,&II,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (j<n-1) {J = II + 1; ierr = MatSetValues(A,1,&II,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      v = 4.0; ierr = MatSetValues(A,1,&II,1,&II,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
#if defined(PETSC_USE_SOCKET_VIEWER)
  ierr = MatView(A,PETSC_VIEWER_SOCKET_WORLD);CHKERRQ(ierr);
#endif
  ierr = VecCreateSeq(PETSC_COMM_SELF,m,&x);CHKERRQ(ierr);
  ierr = VecSet(x,one);CHKERRQ(ierr);
#if defined(PETSC_USE_SOCKET_VIEWER)
  ierr = VecView(x,PETSC_VIEWER_SOCKET_WORLD);CHKERRQ(ierr);
#endif

  ierr = PetscSleep(30);CHKERRQ(ierr);

  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
     TODO: cannot test with socket viewer

TEST*/
