
static char help[] = "Passes a sparse matrix to MATLAB.\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  PetscInt       m   = 4,n = 5,i,j,II,J;
  PetscScalar    one = 1.0,v;
  Vec            x;
  Mat            A;

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));

  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));

  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      v = -1.0;  II = j + n*i;
      if (i>0)   {J = II - n; PetscCall(MatSetValues(A,1,&II,1,&J,&v,INSERT_VALUES));}
      if (i<m-1) {J = II + n; PetscCall(MatSetValues(A,1,&II,1,&J,&v,INSERT_VALUES));}
      if (j>0)   {J = II - 1; PetscCall(MatSetValues(A,1,&II,1,&J,&v,INSERT_VALUES));}
      if (j<n-1) {J = II + 1; PetscCall(MatSetValues(A,1,&II,1,&J,&v,INSERT_VALUES));}
      v = 4.0; PetscCall(MatSetValues(A,1,&II,1,&II,&v,INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
#if defined(PETSC_USE_SOCKET_VIEWER)
  PetscCall(MatView(A,PETSC_VIEWER_SOCKET_WORLD));
#endif
  PetscCall(VecCreateSeq(PETSC_COMM_SELF,m,&x));
  PetscCall(VecSet(x,one));
#if defined(PETSC_USE_SOCKET_VIEWER)
  PetscCall(VecView(x,PETSC_VIEWER_SOCKET_WORLD));
#endif

  PetscCall(PetscSleep(30));

  PetscCall(VecDestroy(&x));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
     TODO: cannot test with socket viewer

TEST*/
