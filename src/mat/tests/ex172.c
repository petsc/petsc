
static char help[] = "Test MatAXPY and SUBSET_NONZERO_PATTERN [-different] [-skip]\n by default subset pattern is used \n\n";

/* A test contributed by Jose E. Roman, Oct. 2014 */

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            A,B,C;
  PetscBool      different=PETSC_FALSE,skip=PETSC_FALSE;
  PetscInt       m0,m1,n=128,i;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-different",&different,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-skip",&skip,NULL));
  /*
     Create matrices
     A = tridiag(1,-2,1) and B = diag(7);
  */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatCreate(PETSC_COMM_WORLD,&B));
  PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n));
  PetscCall(MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,n,n));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetFromOptions(B));
  PetscCall(MatSetUp(A));
  PetscCall(MatSetUp(B));
  PetscCall(MatGetOwnershipRange(A,&m0,&m1));
  for (i=m0;i<m1;i++) {
    if (i>0) PetscCall(MatSetValue(A,i,i-1,-1.0,INSERT_VALUES));
    if (i<n-1) PetscCall(MatSetValue(A,i,i+1,-1.0,INSERT_VALUES));
    PetscCall(MatSetValue(A,i,i,2.0,INSERT_VALUES));
    PetscCall(MatSetValue(B,i,i,7.0,INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));

  PetscCall(MatDuplicate(A,MAT_COPY_VALUES,&C));
  /* Add B */
  PetscCall(MatAXPY(C,1.0,B,(different)?DIFFERENT_NONZERO_PATTERN:SUBSET_NONZERO_PATTERN));
  /* Add A */
  if (!skip) PetscCall(MatAXPY(C,1.0,A,SUBSET_NONZERO_PATTERN));

  /*
     Free memory
  */
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&C));
  PetscCall(PetscFinalize());
  return 0;
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
