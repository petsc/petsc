static char help[] = "Test MatConvert() with a MATNEST with scaled and shifted MATTRANSPOSEVIRTUAL blocks.\n\n";

#include <petscmat.h>

/*
   This example builds the matrix

        H = [  R                    C
              alpha C^H + beta I    gamma R^T + delta I ],

   where R is Hermitian and C is complex symmetric. In particular, R and C have the
   following Toeplitz structure:

        R = pentadiag{a,b,c,conj(b),conj(a)}
        C = tridiag{b,d,b}

   where a,b,d are complex scalars, and c is real.
*/

int main(int argc, char **argv)
{
  Mat         block[4], H, R, C, M;
  PetscScalar a, b, c, d;
  PetscInt    n = 13, Istart, Iend, i;
  PetscBool   flg;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));

  a = PetscCMPLX(-0.1, 0.2);
  b = PetscCMPLX(1.0, 0.5);
  c = 4.5;
  d = PetscCMPLX(2.0, 0.2);

  PetscCall(MatCreate(PETSC_COMM_WORLD, &R));
  PetscCall(MatSetSizes(R, PETSC_DECIDE, PETSC_DECIDE, n, n));
  PetscCall(MatSetFromOptions(R));

  PetscCall(MatCreate(PETSC_COMM_WORLD, &C));
  PetscCall(MatSetSizes(C, PETSC_DECIDE, PETSC_DECIDE, n, n));
  PetscCall(MatSetFromOptions(C));

  PetscCall(MatGetOwnershipRange(R, &Istart, &Iend));
  for (i = Istart; i < Iend; i++) {
    if (i > 1) PetscCall(MatSetValue(R, i, i - 2, a, INSERT_VALUES));
    if (i > 0) PetscCall(MatSetValue(R, i, i - 1, b, INSERT_VALUES));
    PetscCall(MatSetValue(R, i, i, c, INSERT_VALUES));
    if (i < n - 1) PetscCall(MatSetValue(R, i, i + 1, PetscConj(b), INSERT_VALUES));
    if (i < n - 2) PetscCall(MatSetValue(R, i, i + 2, PetscConj(a), INSERT_VALUES));
  }

  PetscCall(MatGetOwnershipRange(C, &Istart, &Iend));
  for (i = Istart; i < Iend; i++) {
    if (i > 0) PetscCall(MatSetValue(C, i, i - 1, b, INSERT_VALUES));
    PetscCall(MatSetValue(C, i, i, d, INSERT_VALUES));
    if (i < n - 1) PetscCall(MatSetValue(C, i, i + 1, b, INSERT_VALUES));
  }

  PetscCall(MatAssemblyBegin(R, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(R, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY));

  block[0] = R;
  block[1] = C;

  PetscCall(MatCreateHermitianTranspose(C, &block[2]));
  PetscCall(MatScale(block[2], PetscConj(b)));
  PetscCall(MatShift(block[2], d));
  PetscCall(MatCreateTranspose(R, &block[3]));
  PetscCall(MatScale(block[3], PetscConj(d)));
  PetscCall(MatShift(block[3], b));
  PetscCall(MatCreateNest(PetscObjectComm((PetscObject)R), 2, NULL, 2, NULL, block, &H));
  PetscCall(MatDestroy(&block[2]));
  PetscCall(MatDestroy(&block[3]));

  PetscCall(MatConvert(H, MATAIJ, MAT_INITIAL_MATRIX, &M));
  PetscCall(MatMultEqual(H, M, 20, &flg));
  PetscCheck(flg, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "MatNest != MatAIJ");

  PetscCall(MatDestroy(&R));
  PetscCall(MatDestroy(&C));
  PetscCall(MatDestroy(&H));
  PetscCall(MatDestroy(&M));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: complex

   test:
      output_file: output/empty.out
      nsize: {{1 4}}

TEST*/
