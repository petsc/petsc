
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
  ierr = PetscOptionsGetBool(NULL,NULL,"-different",&different,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-skip",&skip,NULL);CHKERRQ(ierr);
  /*
     Create matrices
     A = tridiag(1,-2,1) and B = diag(7);
  */
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetFromOptions(B);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);
  ierr = MatSetUp(B);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&m0,&m1);CHKERRQ(ierr);
  for (i=m0;i<m1;i++) {
    if (i>0) { ierr = MatSetValue(A,i,i-1,-1.0,INSERT_VALUES);CHKERRQ(ierr); }
    if (i<n-1) { ierr = MatSetValue(A,i,i+1,-1.0,INSERT_VALUES);CHKERRQ(ierr); }
    ierr = MatSetValue(A,i,i,2.0,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatSetValue(B,i,i,7.0,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatDuplicate(A,MAT_COPY_VALUES,&C);CHKERRQ(ierr);
  /* Add B */
  ierr = MatAXPY(C,1.0,B,(different)?DIFFERENT_NONZERO_PATTERN:SUBSET_NONZERO_PATTERN);CHKERRQ(ierr);
  /* Add A */
  if (!skip) { ierr = MatAXPY(C,1.0,A,SUBSET_NONZERO_PATTERN);CHKERRQ(ierr); }

  /*
     Free memory
  */
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
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
