static const char help[] = "Test MatPtAP with MATDIAGONAL and MATCONSTANTDIAGONAL\n";

#include <petscmat.h>

/* KOKKOS: Following two cases will fail, as MatDiagonalScale_{Seq,MPI}AIJKOKKOS does
   not support CPU diagonal vector against AIJ KOKKOS.
   -a_mat_type diagonal -p_mat_type aijkokkos -a_mat_vec_type standard
   -a_mat_type aijkokkos -p_mat_type diagonal -p_mat_vec_type standard */
static PetscErrorCode CreateTestMatrix(MPI_Comm comm, const char prefix[], PetscInt m, PetscInt n, Mat *M)
{
  PetscFunctionBeginUser;
  PetscCall(MatCreate(comm, M));
  PetscCall(MatSetSizes(*M, PETSC_DECIDE, PETSC_DECIDE, m, n));
  PetscCall(MatSetType(*M, MATAIJ));
  PetscCall(MatSetOptionsPrefix(*M, prefix));
  PetscCall(MatSetFromOptions(*M));
  PetscCall(MatSeqAIJSetPreallocation(*M, n, NULL));
  PetscCall(MatMPIAIJSetPreallocation(*M, n, NULL, n, NULL));
  PetscCall(MatSetUp(*M));
  PetscCall(MatSetRandom(*M, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Check correctness of C = P^T A P via mat-vec
   For AIJ-type P matrix, check if factorization is possible */
static PetscErrorCode VerifyPtAP(MPI_Comm comm, Mat A, Mat P, Mat C)
{
  PetscBool flg, can_chol;

  PetscFunctionBeginUser;
  PetscCall(MatPtAPMultEqual(A, P, C, 10, &flg));
  PetscCheck(flg, comm, PETSC_ERR_PLIB, "MatPtAPMultEqual() failed");

  PetscCall(MatGetFactorAvailable(C, MATSOLVERPETSC, MAT_FACTOR_CHOLESKY, &can_chol));
  if (can_chol) {
    Mat           Cshift, fact;
    IS            perm, cperm;
    MatFactorInfo info;
    PetscReal     norm;

    PetscCall(MatDuplicate(C, MAT_COPY_VALUES, &Cshift));
    PetscCall(MatNorm(Cshift, NORM_INFINITY, &norm));
    PetscCall(MatShift(Cshift, 2.0 * norm + 1.0));
    PetscCall(MatGetFactor(Cshift, MATSOLVERPETSC, MAT_FACTOR_CHOLESKY, &fact));
    PetscCall(MatGetOrdering(Cshift, MATORDERINGNATURAL, &perm, &cperm));
    PetscCall(MatFactorInfoInitialize(&info));
    info.fill = 5.0;
    PetscCall(MatCholeskyFactorSymbolic(fact, Cshift, perm, &info));
    PetscCall(MatCholeskyFactorNumeric(fact, Cshift, &info));
    PetscCall(ISDestroy(&perm));
    PetscCall(ISDestroy(&cperm));
    PetscCall(MatDestroy(&fact));
    PetscCall(MatDestroy(&Cshift));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  Mat      A, P, C;
  MPI_Comm comm;
  PetscInt m = 10, n = 8;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;

  PetscOptionsBegin(comm, "", help, "none");
  PetscCall(PetscOptionsInt("-m", "m size", "", m, &m, NULL));
  PetscCall(PetscOptionsInt("-n", "n size", "", n, &n, NULL));
  PetscOptionsEnd();

  PetscCall(CreateTestMatrix(comm, "a_", m, m, &A));
  PetscCall(CreateTestMatrix(comm, "p_", m, n, &P));

  /* Initial PtAP */
  PetscCall(MatPtAP(A, P, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &C));
  PetscCall(VerifyPtAP(comm, A, P, C));

  /* Reuse with modified A */
  PetscCall(MatScale(A, 2.0));
  PetscCall(MatPtAP(A, P, MAT_REUSE_MATRIX, PETSC_DETERMINE, &C));
  PetscCall(VerifyPtAP(comm, A, P, C));

  /* Reuse with modified P */
  PetscCall(MatScale(P, 0.5));
  PetscCall(MatPtAP(A, P, MAT_REUSE_MATRIX, PETSC_DETERMINE, &C));
  PetscCall(VerifyPtAP(comm, A, P, C));

  /* Reuse with modified A */
  PetscCall(MatScale(A, 1.1));
  PetscCall(MatPtAP(A, P, MAT_REUSE_MATRIX, PETSC_DETERMINE, &C));
  PetscCall(VerifyPtAP(comm, A, P, C));

  /* Reuse with modified P */
  PetscCall(MatScale(P, 3.7));
  PetscCall(MatPtAP(A, P, MAT_REUSE_MATRIX, PETSC_DETERMINE, &C));
  PetscCall(VerifyPtAP(comm, A, P, C));

  /* Modify both A and P */
  PetscCall(MatScale(A, 0.23));
  PetscCall(MatScale(P, 1.43));
  PetscCall(MatPtAP(A, P, MAT_REUSE_MATRIX, PETSC_DETERMINE, &C));
  PetscCall(VerifyPtAP(comm, A, P, C));

  PetscCall(MatDestroy(&C));
  PetscCall(MatDestroy(&P));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: a_diag_cpu
    nsize: {{1 2}}
    args: -a_mat_type diagonal -p_mat_type {{aij dense}}
    output_file: output/empty.out

  test:
    suffix: p_diag_cpu
    nsize: {{1 2}}
    args: -a_mat_type {{diagonal aij dense}} -p_mat_type diagonal -n 10 -m 10
    output_file: output/empty.out

  test:
    suffix: diag_diag_kokkos
    nsize: {{1 2}}
    requires: kokkos_kernels
    args: -a_mat_type diagonal -p_mat_type diagonal -a_mat_vec_type kokkos -p_mat_vec_type {{kokkos standard}} -n 10 -m 10
    output_file: output/empty.out

  test:
    suffix: diag_standard_diag_kokkos
    nsize: {{1 2}}
    requires: kokkos_kernels
    args: -a_mat_type diagonal -p_mat_type diagonal -a_mat_vec_type standard -p_mat_vec_type kokkos -n 10 -m 10
    output_file: output/empty.out

  test:
    suffix: diag_diag_cuda
    nsize: {{1 2}}
    requires: cuda
    args: -a_mat_type diagonal -p_mat_type diagonal -a_mat_vec_type cuda -p_mat_vec_type {{cuda standard}} -n 10 -m 10
    output_file: output/empty.out

  test:
    suffix: diag_standard_diag_cuda
    nsize: {{1 2}}
    requires: cuda
    args: -a_mat_type diagonal -p_mat_type diagonal -a_mat_vec_type standard -p_mat_vec_type cuda -n 10 -m 10
    output_file: output/empty.out

  test:
    suffix: diag_cuda_mat
    nsize: {{1 2}}
    requires: cuda
    args: -a_mat_type diagonal -p_mat_type {{aijcusparse densecuda}} -a_mat_vec_type cuda
    output_file: output/empty.out

  test:
    suffix: cuda_mat_diag
    nsize: {{1 2}}
    requires: cuda
    args: -a_mat_type {{aijcusparse densecuda}} -p_mat_type diagonal -p_mat_vec_type cuda -n 10 -m 10
    output_file: output/empty.out

  test:
    suffix: diag_diag_hip
    nsize: {{1 2}}
    requires: hip
    args: -a_mat_type diagonal -p_mat_type diagonal -a_mat_vec_type hip -p_mat_vec_type {{hip standard}} -n 10 -m 10
    output_file: output/empty.out

  test:
    suffix: diag_standard_diag_hip
    nsize: {{1 2}}
    requires: hip
    args: -a_mat_type diagonal -p_mat_type diagonal -a_mat_vec_type standard -p_mat_vec_type hip -n 10 -m 10
    output_file: output/empty.out

  test:
    suffix: diag_hip_mat
    nsize: {{1 2}}
    requires: hip
    args: -a_mat_type diagonal -p_mat_type {{aijhipsparse densehip}} -a_mat_vec_type hip
    output_file: output/empty.out

  test:
    suffix: hip_mat_diag
    nsize: {{1 2}}
    requires: hip
    args: -a_mat_type {{aijhipsparse densehip}} -p_mat_type diagonal -p_mat_vec_type hip -n 10 -m 10
    output_file: output/empty.out

  test:
    suffix: a_cdiag_p_dense_cpu
    nsize: {{1 2}}
    args: -a_mat_type constantdiagonal -p_mat_type {{aij dense}}
    output_file: output/empty.out

  test:
    suffix: a_cdiag_p_diag_cpu
    nsize: {{1 2}}
    args: -a_mat_type constantdiagonal -p_mat_type diagonal -n 10 -m 10
    output_file: output/empty.out

  test:
    suffix: p_cdiag_cpu
    nsize: {{1 2}}
    args: -a_mat_type {{constantdiagonal diagonal aij dense}} -p_mat_type constantdiagonal -n 10 -m 10
    output_file: output/empty.out

  test:
    suffix: cdiag_diag_cuda
    nsize: {{1 2}}
    requires: cuda
    args: -a_mat_type constantdiagonal -p_mat_type diagonal -p_mat_vec_type cuda -n 10 -m 10
    output_file: output/empty.out

  test:
    suffix: diag_cuda_cdiag
    nsize: {{1 2}}
    requires: cuda
    args: -a_mat_type diagonal -a_mat_vec_type cuda -p_mat_type constantdiagonal -n 10 -m 10
    output_file: output/empty.out

  test:
    suffix: cdiag_cuda_mat
    nsize: {{1 2}}
    requires: cuda
    args: -a_mat_type constantdiagonal -p_mat_type {{aijcusparse densecuda}}
    output_file: output/empty.out

  test:
    suffix: cuda_mat_cdiag
    nsize: {{1 2}}
    requires: cuda
    args: -a_mat_type {{aijcusparse densecuda}} -p_mat_type constantdiagonal -n 10 -m 10
    output_file: output/empty.out

  test:
    suffix: cdiag_diag_kokkos
    nsize: {{1 2}}
    requires: kokkos_kernels
    args: -a_mat_type constantdiagonal -p_mat_type diagonal -p_mat_vec_type kokkos -n 10 -m 10
    output_file: output/empty.out

  test:
    suffix: diag_kokkos_cdiag
    nsize: {{1 2}}
    requires: kokkos_kernels
    args: -a_mat_type diagonal -a_mat_vec_type kokkos -p_mat_type constantdiagonal -n 10 -m 10
    output_file: output/empty.out

  test:
    suffix: cdiag_diag_hip
    nsize: {{1 2}}
    requires: hip
    args: -a_mat_type constantdiagonal -p_mat_type diagonal -p_mat_vec_type hip -n 10 -m 10
    output_file: output/empty.out

  test:
    suffix: diag_hip_cdiag
    nsize: {{1 2}}
    requires: hip
    args: -a_mat_type diagonal -a_mat_vec_type hip -p_mat_type constantdiagonal -n 10 -m 10
    output_file: output/empty.out

  test:
    suffix: cdiag_hip_mat
    nsize: {{1 2}}
    requires: hip
    args: -a_mat_type constantdiagonal -p_mat_type {{aijhipsparse densehip}}
    output_file: output/empty.out

  test:
    suffix: hip_mat_cdiag
    nsize: {{1 2}}
    requires: hip
    args: -a_mat_type {{aijhipsparse densehip}} -p_mat_type constantdiagonal -n 10 -m 10
    output_file: output/empty.out

TEST*/
