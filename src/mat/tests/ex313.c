static const char
  help[] = "Test MatNorm() on dense matrices, including submatrix views where the leading dimension exceeds the local row count.\nThe matrix under test is built with -mat_type, or through MatCreateDenseFromVecType() with -from_vec_type.\n\n";

// Contributed by: Steven Dargaville

#include <petscmat.h>

/*
   Entries are small signed integers, so every norm is an exactly representable sum. The magnitude
   1 + (i % 5) + 2 * (j % 3) lies between 1 and 9 and varies with both the row and the column, so
   the largest column sum and the largest row sum are attained at nontrivial positions. The sign
   alternates in a checkerboard, so the absolute values taken by NORM_1 and NORM_INFINITY matter,
   and complex builds add an imaginary part, so the modulus differs from |re| + |im| and
   absolute-value shortcuts on the device are caught
*/
static PetscScalar Entry(PetscInt i, PetscInt j)
{
  PetscReal   magnitude = (PetscReal)(1 + (i % 5) + 2 * (j % 3));
  PetscReal   sign      = ((i + j) % 2) ? -1.0 : 1.0;
  PetscScalar v         = sign * magnitude;

#if PetscDefined(USE_COMPLEX)
  v += PETSC_i * (PetscScalar)(i - 2 * j);
#endif
  return v;
}

/*
   Compare the norms of the matrix under test against those of a reference matrix holding the same entries
*/
static PetscErrorCode CheckNorms(Mat A, Mat B)
{
  const NormType types[] = {NORM_1, NORM_FROBENIUS, NORM_INFINITY};
  PetscReal      na, nb;

  PetscFunctionBeginUser;
  for (PetscInt t = 0; t < 3; t++) {
    PetscCall(MatNorm(A, types[t], &na));
    PetscCall(MatNorm(B, types[t], &nb));
    PetscCheck(PetscAbsReal(na - nb) <= PETSC_SMALL + PETSC_SMALL * PetscAbsReal(nb), PetscObjectComm((PetscObject)A), PETSC_ERR_PLIB, "NORM_%s mismatch: %g (tested) != %g (reference)", NormTypes[types[t]], (double)na, (double)nb);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  Mat       A, B, C, Asub, Bsub;
  PetscInt  m = 12, n = 7, rstart, rend;
  char      vtype[64];
  PetscBool from_vec_type = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-m", &m, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-from_vec_type", vtype, sizeof(vtype), &from_vec_type));
  PetscCheck(m > 2 && n > 2, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Need m > 2 and n > 2, got %" PetscInt_FMT " and %" PetscInt_FMT, m, n);

  /* the matrix under test; MatCreateDenseFromVecType() is the construction route Kokkos codes use,
     and it yields MATDENSECUDA or MATDENSEHIP on a device Kokkos build and MATDENSE on a host
     Kokkos build */
  if (from_vec_type) PetscCall(MatCreateDenseFromVecType(PETSC_COMM_WORLD, vtype, PETSC_DECIDE, PETSC_DECIDE, m, n, PETSC_DECIDE, NULL, &A));
  else {
    PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
    PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, m, n));
    PetscCall(MatSetType(A, MATDENSE));
    PetscCall(MatSetFromOptions(A));
    PetscCall(MatSetUp(A));
  }

  /* the reference matrix, always on the host */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &B));
  PetscCall(MatSetSizes(B, PETSC_DECIDE, PETSC_DECIDE, m, n));
  PetscCall(MatSetType(B, MATDENSE));
  PetscCall(MatSetUp(B));

  PetscCall(MatGetOwnershipRange(A, &rstart, &rend));
  for (PetscInt i = rstart; i < rend; i++) {
    for (PetscInt j = 0; j < n; j++) {
      PetscCall(MatSetValue(A, i, j, Entry(i, j), INSERT_VALUES));
      PetscCall(MatSetValue(B, i, j, Entry(i, j), INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));

  PetscCall(CheckNorms(A, B));

  /* the entries of the interior block, in fresh contiguous storage where the leading dimension
     equals the number of local rows, so it is unaffected by any leading dimension handling */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &C));
  PetscCall(MatSetSizes(C, PETSC_DECIDE, PETSC_DECIDE, m - 2, n - 2));
  PetscCall(MatSetType(C, MATDENSE));
  PetscCall(MatSetUp(C));
  PetscCall(MatGetOwnershipRange(C, &rstart, &rend));
  for (PetscInt i = rstart; i < rend; i++) {
    for (PetscInt j = 0; j < n - 2; j++) PetscCall(MatSetValue(C, i, j, Entry(i + 1, j + 1), INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY));

  /* an interior view keeps the leading dimension of its parent, so the leading dimension exceeds the number of local rows */
  PetscCall(MatDenseGetSubMatrix(A, 1, m - 1, 1, n - 1, &Asub));
  PetscCall(MatDenseGetSubMatrix(B, 1, m - 1, 1, n - 1, &Bsub));
  PetscCall(CheckNorms(Asub, C));
  PetscCall(CheckNorms(Bsub, C));
  PetscCall(CheckNorms(Asub, Bsub));
  PetscCall(MatDenseRestoreSubMatrix(A, &Asub));
  PetscCall(MatDenseRestoreSubMatrix(B, &Bsub));

  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&C));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    nsize: {{1 2}}
    output_file: output/empty.out

    test:
      suffix: cpu

    test:
      suffix: cuda
      requires: cuda
      args: -mat_type densecuda

    test:
      suffix: hip
      requires: hip
      args: -mat_type densehip

    test:
      suffix: from_vec_kokkos
      requires: kokkos_kernels !sycl
      args: -from_vec_type kokkos

TEST*/
