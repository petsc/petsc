static char help[] = "Tests block sparse-dense matrix products.\n\n";

#include <petscmat.h>

static PetscErrorCode CheckEqual(Mat A, Mat B)
{
  Mat       D;
  PetscReal norm;

  PetscFunctionBegin;
  PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &D));
  PetscCall(MatAXPY(D, -1.0, B, SAME_NONZERO_PATTERN));
  PetscCall(MatNorm(D, NORM_FROBENIUS, &norm));
  PetscCheck(norm <= 100.0 * PETSC_MACHINE_EPSILON, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Matrix product error %g", (double)norm);
  PetscCall(MatDestroy(&D));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  Mat         A, Alibxsmm = NULL, AlibxsmmDuplicate = NULL, Aref, B, Cbaij, Clibxsmm = NULL, Cref;
  MatType     type, actual;
  PetscInt    rstart, rend, M, ncols = 5, bs = 2;
  PetscMPIInt size;
  PetscBool   match, testlibxsmm = PETSC_FALSE;
  PetscScalar value;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_libxsmm", &testlibxsmm, NULL));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  M = 4 * size;
  PetscCall(MatCreateBAIJ(PETSC_COMM_WORLD, bs, PETSC_DECIDE, PETSC_DECIDE, M, M, 3, NULL, 3, NULL, &A));
  PetscCall(MatGetOwnershipRange(A, &rstart, &rend));
  for (PetscInt i = rstart; i < rend; i++) {
    PetscInt brow = i / bs;

    for (PetscInt bcol = PetscMax(brow - 1, 0); bcol <= PetscMin(brow + 1, M / bs - 1); bcol++) {
      for (PetscInt j = bcol * bs; j < (bcol + 1) * bs; j++) {
        value = 1.0 + i + 0.25 * j;
        PetscCall(MatSetValue(A, i, j, value, INSERT_VALUES));
      }
    }
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  PetscCall(MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, M, ncols, NULL, &B));
  PetscCall(MatGetOwnershipRange(B, &rstart, &rend));
  for (PetscInt i = rstart; i < rend; i++) {
    for (PetscInt j = 0; j < ncols; j++) {
      value = 0.5 + 0.125 * i - 0.25 * j;
      PetscCall(MatSetValue(B, i, j, value, INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));

  type = size == 1 ? MATSEQBAIJ : MATMPIBAIJ;
  PetscCall(PetscObjectTypeCompare((PetscObject)A, type, &match));
  PetscCall(MatGetType(A, &actual));
  PetscCheck(match, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Unexpected BAIJ matrix type %s", actual);
  PetscCall(MatConvert(A, MATAIJ, MAT_INITIAL_MATRIX, &Aref));
  if (testlibxsmm) {
    PetscCall(MatConvert(A, MATBAIJLIBXSMM, MAT_INITIAL_MATRIX, &Alibxsmm));
    type = size == 1 ? MATSEQBAIJLIBXSMM : MATMPIBAIJLIBXSMM;
    PetscCall(PetscObjectTypeCompare((PetscObject)Alibxsmm, type, &match));
    PetscCall(MatGetType(Alibxsmm, &actual));
    PetscCheck(match, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Unexpected converted matrix type %s", actual);
    PetscCall(MatDuplicate(Alibxsmm, MAT_COPY_VALUES, &AlibxsmmDuplicate));
    PetscCall(PetscObjectTypeCompare((PetscObject)AlibxsmmDuplicate, type, &match));
    PetscCall(MatGetType(AlibxsmmDuplicate, &actual));
    PetscCheck(match, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Unexpected duplicated matrix type %s", actual);
  }

  PetscCall(MatMatMult(Aref, B, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &Cref));
  PetscCall(MatMatMult(A, B, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &Cbaij));
  PetscCall(CheckEqual(Cbaij, Cref));
  if (testlibxsmm) {
    PetscCall(MatMatMult(AlibxsmmDuplicate, B, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &Clibxsmm));
    PetscCall(CheckEqual(Clibxsmm, Cref));
  }
  PetscCall(MatScale(B, -0.5));
  PetscCall(MatMatMult(Aref, B, MAT_REUSE_MATRIX, PETSC_DETERMINE, &Cref));
  PetscCall(MatMatMult(A, B, MAT_REUSE_MATRIX, PETSC_DETERMINE, &Cbaij));
  PetscCall(CheckEqual(Cbaij, Cref));
  if (testlibxsmm) {
    PetscCall(MatMatMult(AlibxsmmDuplicate, B, MAT_REUSE_MATRIX, PETSC_DETERMINE, &Clibxsmm));
    PetscCall(CheckEqual(Clibxsmm, Cref));
  }

  if (testlibxsmm) {
    type = size == 1 ? MATSEQBAIJ : MATMPIBAIJ;
    PetscCall(MatConvert(Alibxsmm, type, MAT_INPLACE_MATRIX, &Alibxsmm));
    PetscCall(PetscObjectTypeCompare((PetscObject)Alibxsmm, type, &match));
    PetscCall(MatGetType(Alibxsmm, &actual));
    PetscCheck(match, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Unexpected reverted matrix type %s", actual);
  }

  PetscCall(MatDestroy(&Clibxsmm));
  PetscCall(MatDestroy(&Cbaij));
  PetscCall(MatDestroy(&Cref));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&Aref));
  PetscCall(MatDestroy(&AlibxsmmDuplicate));
  PetscCall(MatDestroy(&Alibxsmm));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    output_file: output/empty.out

    test:
      suffix: baij_seq

    test:
      suffix: baij_mpi
      nsize: 2

    test:
      suffix: baij_mpi_batch
      nsize: 2
      args: -matproduct_batch_size 3

  testset:
    requires: libxsmm !complex
    args: -test_libxsmm
    output_file: output/empty.out

    test:
      suffix: libxsmm_seq

    test:
      suffix: libxsmm_mpi
      nsize: 2

    test:
      suffix: libxsmm_mpi_batch
      nsize: 2
      args: -matproduct_batch_size 3

TEST*/
