
static char help[] = "Test MatAXPY and SUBSET_NONZERO_PATTERN [-different] [-skip]\n by default subset pattern is used \n\n";

/* A test contributed by Jose E. Roman, Oct. 2014 */

#include <petscmat.h>

int main(int argc,char **args)
{
  PetscErrorCode ierr;
  Mat            A,B,C;
  PetscBool      different=PETSC_FALSE,skip=PETSC_FALSE;
  PetscInt       m0,m1,n=128,i;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-different",&different,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-skip",&skip,NULL));
  /*
     Create matrices
     A = tridiag(1,-2,1) and B = diag(7);
  */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&B));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,n,n));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetFromOptions(B));
  CHKERRQ(MatSetUp(A));
  CHKERRQ(MatSetUp(B));
  CHKERRQ(MatGetOwnershipRange(A,&m0,&m1));
  for (i=m0;i<m1;i++) {
    if (i>0) CHKERRQ(MatSetValue(A,i,i-1,-1.0,INSERT_VALUES));
    if (i<n-1) CHKERRQ(MatSetValue(A,i,i+1,-1.0,INSERT_VALUES));
    CHKERRQ(MatSetValue(A,i,i,2.0,INSERT_VALUES));
    CHKERRQ(MatSetValue(B,i,i,7.0,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));

  CHKERRQ(MatDuplicate(A,MAT_COPY_VALUES,&C));
  /* Add B */
  CHKERRQ(MatAXPY(C,1.0,B,(different)?DIFFERENT_NONZERO_PATTERN:SUBSET_NONZERO_PATTERN));
  /* Add A */
  if (!skip) CHKERRQ(MatAXPY(C,1.0,A,SUBSET_NONZERO_PATTERN));

  /*
     Free memory
  */
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(MatDestroy(&C));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      nsize: 4
      output_file: output/ex172.out

   test:
      suffix: 2
      nsize: 4
      args: -different
      output_file: output/ex172.out

   test:
      suffix: 3
      nsize: 4
      args: -skip
      output_file: output/ex172.out

   test:
      suffix: 4
      nsize: 4
      args: -different -skip
      output_file: output/ex172.out

   test:
      suffix: baij
      args: -mat_type baij> ex172.tmp 2>&1
      output_file: output/ex172.out

   test:
      suffix: mpibaij
      nsize: 4
      args: -mat_type baij> ex172.tmp 2>&1
      output_file: output/ex172.out

   test:
      suffix: mpisbaij
      nsize: 4
      args: -mat_type sbaij> ex172.tmp 2>&1
      output_file: output/ex172.out

   test:
      suffix: sbaij
      args: -mat_type sbaij> ex172.tmp 2>&1
      output_file: output/ex172.out

TEST*/
