
static char help[] = "Passes a sparse matrix to MATLAB.\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  PetscInt       m   = 4,n = 5,i,j,II,J;
  PetscScalar    one = 1.0,v;
  Vec            x;
  Mat            A;

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));

  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      v = -1.0;  II = j + n*i;
      if (i>0)   {J = II - n; CHKERRQ(MatSetValues(A,1,&II,1,&J,&v,INSERT_VALUES));}
      if (i<m-1) {J = II + n; CHKERRQ(MatSetValues(A,1,&II,1,&J,&v,INSERT_VALUES));}
      if (j>0)   {J = II - 1; CHKERRQ(MatSetValues(A,1,&II,1,&J,&v,INSERT_VALUES));}
      if (j<n-1) {J = II + 1; CHKERRQ(MatSetValues(A,1,&II,1,&J,&v,INSERT_VALUES));}
      v = 4.0; CHKERRQ(MatSetValues(A,1,&II,1,&II,&v,INSERT_VALUES));
    }
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
#if defined(PETSC_USE_SOCKET_VIEWER)
  CHKERRQ(MatView(A,PETSC_VIEWER_SOCKET_WORLD));
#endif
  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,m,&x));
  CHKERRQ(VecSet(x,one));
#if defined(PETSC_USE_SOCKET_VIEWER)
  CHKERRQ(VecView(x,PETSC_VIEWER_SOCKET_WORLD));
#endif

  CHKERRQ(PetscSleep(30));

  CHKERRQ(VecDestroy(&x));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:
     TODO: cannot test with socket viewer

TEST*/
