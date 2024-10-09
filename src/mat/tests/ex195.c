/*
 * ex195.c
 *
 *  Created on: Aug 24, 2015
 *      Author: Fande Kong <fdkong.jd@gmail.com>
 */

static char help[] = " Demonstrate the use of MatConvert_Nest_AIJ\n";

#include <petscmat.h>

int main(int argc, char **args)
{
  Mat         A1, A2, A3, A4, A5, B, C, C1, nest;
  Mat         aij;
  MPI_Comm    comm;
  PetscInt    m, M, n, istart, iend, ii, i, J, j, K = 10;
  PetscScalar v;
  PetscMPIInt size;
  PetscBool   equal, test = PETSC_FALSE, test_null = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_size(comm, &size));

  /*
     Assemble the matrix for the five point stencil, YET AGAIN
  */
  PetscCall(MatCreate(comm, &A1));
  m = 2, n = 2;
  PetscCall(MatSetSizes(A1, PETSC_DECIDE, PETSC_DECIDE, m * n, m * n));
  PetscCall(MatSetFromOptions(A1));
  PetscCall(MatSetUp(A1));
  PetscCall(MatGetOwnershipRange(A1, &istart, &iend));
  for (ii = istart; ii < iend; ii++) {
    v = -1.0;
    i = ii / n;
    j = ii - i * n;
    if (i > 0) {
      J = ii - n;
      PetscCall(MatSetValues(A1, 1, &ii, 1, &J, &v, INSERT_VALUES));
    }
    if (i < m - 1) {
      J = ii + n;
      PetscCall(MatSetValues(A1, 1, &ii, 1, &J, &v, INSERT_VALUES));
    }
    if (j > 0) {
      J = ii - 1;
      PetscCall(MatSetValues(A1, 1, &ii, 1, &J, &v, INSERT_VALUES));
    }
    if (j < n - 1) {
      J = ii + 1;
      PetscCall(MatSetValues(A1, 1, &ii, 1, &J, &v, INSERT_VALUES));
    }
    v = 4.0;
    PetscCall(MatSetValues(A1, 1, &ii, 1, &ii, &v, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A1, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A1, MAT_FINAL_ASSEMBLY));
  PetscCall(MatView(A1, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(MatDuplicate(A1, MAT_COPY_VALUES, &A2));
  PetscCall(MatDuplicate(A1, MAT_COPY_VALUES, &A3));
  PetscCall(MatDuplicate(A1, MAT_COPY_VALUES, &A4));

  /* create a nest matrix */
  PetscCall(MatCreate(comm, &nest));
  PetscCall(MatSetType(nest, MATNEST));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_null", &test_null, NULL));
  if (test_null) {
    IS       is[2];
    PetscInt st;

    PetscCall(MatGetLocalSize(A1, &m, NULL));
    PetscCall(MatGetOwnershipRange(A1, &st, NULL));
    PetscCall(ISCreateStride(comm, m, st, 2, &is[0]));
    PetscCall(ISCreateStride(comm, m, st + 1, 2, &is[1]));
    PetscCall(MatNestSetSubMats(nest, 2, is, 2, is, NULL));
    PetscCall(ISDestroy(&is[0]));
    PetscCall(ISDestroy(&is[1]));
  } else {
    Mat mata[] = {A1, A2, A3, A4};

    PetscCall(MatNestSetSubMats(nest, 2, NULL, 2, NULL, mata));
  }
  PetscCall(MatSetUp(nest));

  /* test matrix product error messages */
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_nest*nest", &test, NULL));
  if (test) PetscCall(MatMatMult(nest, nest, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &C));

  PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_matproductsetfromoptions", &test, NULL));
  if (test) {
    PetscCall(MatProductCreate(nest, nest, NULL, &C));
    PetscCall(MatProductSetType(C, MATPRODUCT_AB));
    PetscCall(MatProductSymbolic(C));
  }

  PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_matproductsymbolic", &test, NULL));
  if (test) {
    PetscCall(MatProductCreate(nest, nest, NULL, &C));
    PetscCall(MatProductSetType(C, MATPRODUCT_AB));
    PetscCall(MatProductSetFromOptions(C));
    PetscCall(MatProductSymbolic(C));
  }

  PetscCall(MatConvert(nest, MATAIJ, MAT_INITIAL_MATRIX, &aij));
  PetscCall(MatView(aij, PETSC_VIEWER_STDOUT_WORLD));

  /* create a dense matrix */
  PetscCall(MatGetSize(nest, &M, NULL));
  PetscCall(MatGetLocalSize(nest, &m, NULL));
  PetscCall(MatCreateDense(comm, m, PETSC_DECIDE, M, K, NULL, &B));
  PetscCall(MatSetRandom(B, NULL));

  /* C = nest*B_dense */
  PetscCall(MatMatMult(nest, B, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &C));
  PetscCall(MatMatMult(nest, B, MAT_REUSE_MATRIX, PETSC_DETERMINE, &C));
  PetscCall(MatMatMultEqual(nest, B, C, 10, &equal));
  PetscCheck(equal, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Error in C != nest*B_dense");

  /* Test B = nest*C, reuse C and B with MatProductCreateWithMat() */
  /* C has been obtained from nest*B. Clear internal data structures related to factors to prevent circular references */
  PetscCall(MatProductClear(C));
  PetscCall(MatProductCreateWithMat(nest, C, NULL, B));
  PetscCall(MatProductSetType(B, MATPRODUCT_AB));
  PetscCall(MatProductSetFromOptions(B));
  PetscCall(MatProductSymbolic(B));
  PetscCall(MatProductNumeric(B));
  PetscCall(MatMatMultEqual(nest, C, B, 10, &equal));
  PetscCheck(equal, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Error in B != nest*C_dense");
  PetscCall(MatConvert(nest, MATAIJ, MAT_INPLACE_MATRIX, &nest));
  PetscCall(MatEqual(nest, aij, &equal));
  PetscCheck(equal, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Error in aij != nest");

  /* Test with virtual block */
  PetscCall(MatCreateTranspose(A1, &A5)); /* A1 is symmetric */
  PetscCall(MatNestSetSubMat(nest, 0, 0, A5));
  PetscCall(MatMatMult(nest, B, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &C1));
  PetscCall(MatMatMult(nest, B, MAT_REUSE_MATRIX, PETSC_DETERMINE, &C1));
  PetscCall(MatMatMultEqual(nest, B, C1, 10, &equal));
  PetscCheck(equal, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Error in C1 != C");
  PetscCall(MatDestroy(&C1));
  PetscCall(MatDestroy(&A5));

  PetscCall(MatDestroy(&nest));
  PetscCall(MatDestroy(&C));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&aij));
  PetscCall(MatDestroy(&A1));
  PetscCall(MatDestroy(&A2));
  PetscCall(MatDestroy(&A3));
  PetscCall(MatDestroy(&A4));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      diff_args: -j
      nsize: 2
      suffix: 1

   test:
      diff_args: -j
      nsize: 2
      suffix: 1_null
      args: -test_null

   test:
      diff_args: -j
      suffix: 2

   test:
      diff_args: -j
      suffix: 2_null
      args: -test_null

TEST*/
