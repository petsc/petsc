
static char help[] = "Tests MatTranspose()\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            C,A;
  PetscScalar    v;
  PetscInt       i,j,m = 4,n = 4,Ii,J,Istart,Iend;
  PetscMPIInt    rank,size;
  PetscErrorCode ierr;
  PetscBool      equal=PETSC_FALSE;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  n    = m;

  CHKERRQ(MatCreateAIJ(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n,5,NULL,5,NULL,&C));

  /* create the symmetric matrix for the five point stencil */
  CHKERRQ(MatGetOwnershipRange(C,&Istart,&Iend));
  for (Ii=Istart; Ii<Iend; Ii++) {
    v = -1.0; i = Ii/n; j = Ii - i*n;
    if (i>0)   {J = Ii - n; CHKERRQ(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));}
    if (i<m-1) {J = Ii + n; CHKERRQ(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));}
    if (j>0)   {J = Ii - 1; CHKERRQ(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));}
    if (j<n-1) {J = Ii + 1; CHKERRQ(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));}
    v = 4.0; CHKERRQ(MatSetValues(C,1,&Ii,1,&Ii,&v,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));

  CHKERRQ(MatTranspose(C,MAT_INITIAL_MATRIX,&A));

  CHKERRQ(MatEqual(C,A,&equal));
  PetscCheck(equal,PETSC_COMM_SELF,PETSC_ERR_SUP,"C != C^T");

  CHKERRQ(MatDestroy(&C));
  CHKERRQ(MatDestroy(&A));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

TEST*/
