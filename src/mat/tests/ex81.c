
static char help[] = "Tests MatOption MAT_FORCE_DIAGONAL_ENTRIES.\n\n";

#include <petscmat.h>

int main(int argc, char **args)
{
  Mat         A, B;
  Vec         diag;
  PetscInt    i, n = 10, col[3], test;
  PetscScalar v[3];

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));

  /* Create A which has empty 0-th row and column */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n));
  PetscCall(MatSetType(A, MATAIJ));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));

  v[0] = -1.;
  v[1] = 2.;
  v[2] = -1.;
  for (i = 2; i < n - 1; i++) {
    col[0] = i - 1;
    col[1] = i;
    col[2] = i + 1;
    PetscCall(MatSetValues(A, 1, &i, 3, col, v, INSERT_VALUES));
  }
  i      = 1;
  col[0] = 1;
  col[1] = 2;
  PetscCall(MatSetValues(A, 1, &i, 2, col, v + 1, INSERT_VALUES));
  i      = n - 1;
  col[0] = n - 2;
  col[1] = n - 1;
  PetscCall(MatSetValues(A, 1, &i, 2, col, v, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  for (test = 0; test < 2; test++) {
    PetscCall(MatProductCreate(A, A, NULL, &B));

    if (test == 0) {
      /* Compute B = A*A; B misses 0-th diagonal */
      PetscCall(MatProductSetType(B, MATPRODUCT_AB));
      PetscCall(MatSetOptionsPrefix(B, "AB_"));
    } else {
      /* Compute B = A^t*A; B misses 0-th diagonal */
      PetscCall(MatProductSetType(B, MATPRODUCT_AtB));
      PetscCall(MatSetOptionsPrefix(B, "AtB_"));
    }

    /* Force allocate missing diagonal entries of B */
    PetscCall(MatSetOption(B, MAT_FORCE_DIAGONAL_ENTRIES, PETSC_TRUE));
    PetscCall(MatProductSetFromOptions(B));

    PetscCall(MatProductSymbolic(B));
    PetscCall(MatProductNumeric(B));

    PetscCall(MatSetOption(B, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE));

    /* Insert entries to diagonal of B */
    PetscCall(MatCreateVecs(B, NULL, &diag));
    PetscCall(MatGetDiagonal(B, diag));
    PetscCall(VecSetValue(diag, 0, 100.0, INSERT_VALUES));
    PetscCall(VecAssemblyBegin(diag));
    PetscCall(VecAssemblyEnd(diag));

    PetscCall(MatDiagonalSet(B, diag, INSERT_VALUES));
    if (test == 1) PetscCall(MatView(B, PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(MatDestroy(&B));
    PetscCall(VecDestroy(&diag));
  }

  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
     output_file: output/ex81_1.out

   test:
     suffix: 2
     args: -AtB_mat_product_algorithm at*b
     output_file: output/ex81_1.out

   test:
     suffix: 3
     args: -AtB_mat_product_algorithm outerproduct
     output_file: output/ex81_1.out

   test:
     suffix: 4
     nsize: 3
     args: -AtB_mat_product_algorithm nonscalable
     output_file: output/ex81_3.out

   test:
     suffix: 5
     nsize: 3
     args: -AtB_mat_product_algorithm scalable
     output_file: output/ex81_3.out

   test:
     suffix: 6
     nsize: 3
     args: -AtB_mat_product_algorithm at*b
     output_file: output/ex81_3.out

TEST*/
