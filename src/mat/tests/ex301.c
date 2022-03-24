
static char help[] = "Tests for bugs in A->offloadmask consistency for GPU matrices\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            A;
  PetscInt       i,j,rstart,rend,m = 3;
  PetscScalar    one = 1.0,zero = 0.0,negativeone = -1.0;
  PetscReal      norm;
  Vec            x,y;

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));

  for (i=0; i<2; i++) {
    /* Create the matrix and set it to contain explicit zero entries on the diagonal. */
    CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
    CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m*m,m*m));
    CHKERRQ(MatSetFromOptions(A));
    CHKERRQ(MatSetUp(A));
    CHKERRQ(MatGetOwnershipRange(A,&rstart,&rend));
    CHKERRQ(MatCreateVecs(A,&x,&y));
    CHKERRQ(VecSet(x,one));
    CHKERRQ(VecSet(y,zero));
    CHKERRQ(MatDiagonalSet(A,y,INSERT_VALUES));

    /* Now set A to be the identity using various approaches.
     * Note that there may be other approaches that should be added here. */
    switch (i) {
    case 0:
      CHKERRQ(MatDiagonalSet(A,x,INSERT_VALUES));
      break;
    case 1:
      for (j=rstart; j<rend; j++) {
        CHKERRQ(MatSetValue(A,j,j,one,INSERT_VALUES));
      }
      CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
      CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
      break;
    case 2:
      for (j=rstart; j<rend; j++) {
        CHKERRQ(MatSetValuesRow(A,j,&one));
      }
      CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
      CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
    default:
      break;
    }

    /* Compute y <- A*x and verify that the difference between y and x is negligible, as it should be since A is the identity. */
    CHKERRQ(MatMult(A,x,y));
    CHKERRQ(VecAXPY(y,negativeone,x));
    CHKERRQ(VecNorm(y,NORM_2,&norm));
    if (norm > PETSC_SQRT_MACHINE_EPSILON) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Test %" PetscInt_FMT ": Norm of error is %g, but should be near 0.\n",i,(double)norm));
    }

    CHKERRQ(MatDestroy(&A));
    CHKERRQ(VecDestroy(&x));
    CHKERRQ(VecDestroy(&y));
  }

  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:
      suffix: aijviennacl_1
      nsize: 1
      args: -mat_type aijviennacl
      requires: viennacl

   test:
      suffix: aijviennacl_2
      nsize: 2
      args: -mat_type aijviennacl
      requires: viennacl

   test:
      suffix: aijcusparse_1
      nsize: 1
      args: -mat_type aijcusparse
      requires: cuda

   test:
      suffix: aijcusparse_2
      nsize: 2
      args: -mat_type aijcusparse
      requires: cuda
TEST*/
