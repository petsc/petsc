
static char help[] = "Tests for bugs in A->offloadmask consistency for GPU matrices\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            A;
  PetscInt       i,j,rstart,rend,m = 3;
  PetscScalar    one = 1.0,zero = 0.0,negativeone = -1.0;
  PetscReal      norm;
  Vec            x,y;

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));

  for (i=0; i<2; i++) {
    /* Create the matrix and set it to contain explicit zero entries on the diagonal. */
    PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
    PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m*m,m*m));
    PetscCall(MatSetFromOptions(A));
    PetscCall(MatSetUp(A));
    PetscCall(MatGetOwnershipRange(A,&rstart,&rend));
    PetscCall(MatCreateVecs(A,&x,&y));
    PetscCall(VecSet(x,one));
    PetscCall(VecSet(y,zero));
    PetscCall(MatDiagonalSet(A,y,INSERT_VALUES));

    /* Now set A to be the identity using various approaches.
     * Note that there may be other approaches that should be added here. */
    switch (i) {
    case 0:
      PetscCall(MatDiagonalSet(A,x,INSERT_VALUES));
      break;
    case 1:
      for (j=rstart; j<rend; j++) {
        PetscCall(MatSetValue(A,j,j,one,INSERT_VALUES));
      }
      PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
      break;
    case 2:
      for (j=rstart; j<rend; j++) {
        PetscCall(MatSetValuesRow(A,j,&one));
      }
      PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
    default:
      break;
    }

    /* Compute y <- A*x and verify that the difference between y and x is negligible, as it should be since A is the identity. */
    PetscCall(MatMult(A,x,y));
    PetscCall(VecAXPY(y,negativeone,x));
    PetscCall(VecNorm(y,NORM_2,&norm));
    if (norm > PETSC_SQRT_MACHINE_EPSILON) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Test %" PetscInt_FMT ": Norm of error is %g, but should be near 0.\n",i,(double)norm));
    }

    PetscCall(MatDestroy(&A));
    PetscCall(VecDestroy(&x));
    PetscCall(VecDestroy(&y));
  }

  PetscCall(PetscFinalize());
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
