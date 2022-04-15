static char help[] = "Working out corner cases of the ASM preconditioner.\n\n";

#include <petscksp.h>

int main(int argc,char **args)
{
  KSP            ksp;
  PC             pc;
  Mat            A;
  Vec            u, x, b;
  PetscReal      error;
  PetscMPIInt    rank, size, sized;
  PetscInt       M = 8, N = 8, m, n, rstart, rend, r;
  PetscBool      userSubdomains = PETSC_FALSE;

  PetscCall(PetscInitialize(&argc, &args, NULL,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL, "-M", &M, NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL, "-N", &N, NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL, "-user_subdomains", &userSubdomains, NULL));
  /* Do parallel decomposition */
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  sized = (PetscMPIInt) PetscSqrtReal((PetscReal) size);
  PetscCheck(PetscSqr(sized) == size,PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "This test may only be run on a number of processes which is a perfect square, not %d", (int) size);
  PetscCheck((M % sized) == 0,PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "The number of x-vertices %" PetscInt_FMT " does not divide the number of x-processes %d", M, (int) sized);
  PetscCheck((N % sized) == 0,PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "The number of y-vertices %" PetscInt_FMT " does not divide the number of y-processes %d", N, (int) sized);
  /* Assemble the matrix for the five point stencil, YET AGAIN
       Every other process will be empty */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  m    = (sized > 1) ? (rank % 2) ? 0 : 2*M/sized : M;
  n    = N/sized;
  PetscCall(MatSetSizes(A, m*n, m*n, M*N, M*N));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));
  PetscCall(MatGetOwnershipRange(A, &rstart, &rend));
  for (r = rstart; r < rend; ++r) {
    const PetscScalar diag = 4.0, offdiag = -1.0;
    const PetscInt    i    = r/N;
    const PetscInt    j    = r - i*N;
    PetscInt          c;

    if (i > 0)   {c = r - n; PetscCall(MatSetValues(A, 1, &r, 1, &c, &offdiag, INSERT_VALUES));}
    if (i < M-1) {c = r + n; PetscCall(MatSetValues(A, 1, &r, 1, &c, &offdiag, INSERT_VALUES));}
    if (j > 0)   {c = r - 1; PetscCall(MatSetValues(A, 1, &r, 1, &c, &offdiag, INSERT_VALUES));}
    if (j < N-1) {c = r + 1; PetscCall(MatSetValues(A, 1, &r, 1, &c, &offdiag, INSERT_VALUES));}
    PetscCall(MatSetValues(A, 1, &r, 1, &r, &diag, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  /* Setup Solve */
  PetscCall(MatCreateVecs(A, &x, &b));
  PetscCall(VecDuplicate(x, &u));
  PetscCall(VecSet(u, 1.0));
  PetscCall(MatMult(A, u, b));
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCASM));
  /* Setup ASM by hand */
  if (userSubdomains) {
    IS        is;
    PetscInt *rows;

    /* Use no overlap for now */
    PetscCall(PetscMalloc1(rend-rstart, &rows));
    for (r = rstart; r < rend; ++r) rows[r-rstart] = r;
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF, rend-rstart, rows, PETSC_OWN_POINTER, &is));
    PetscCall(PCASMSetLocalSubdomains(pc, 1, &is, &is));
    PetscCall(ISDestroy(&is));
  }
  PetscCall(KSPSetFromOptions(ksp));
  /* Solve and Compare */
  PetscCall(KSPSolve(ksp, b, x));
  PetscCall(VecAXPY(x, -1.0, u));
  PetscCall(VecNorm(x, NORM_INFINITY, &error));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Infinity norm of the error: %g\n", (double) error));
  /* Cleanup */
  PetscCall(KSPDestroy(&ksp));
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 0
      args: -ksp_view

   test:
      requires: !sycl kokkos_kernels
      suffix: 0_kokkos
      args: -ksp_view -mat_type aijkokkos

   test:
      requires: cuda
      suffix: 0_cuda
      args: -ksp_view -mat_type aijcusparse -sub_pc_factor_mat_solver_type cusparse

   test:
      suffix: 1
      nsize: 4
      args: -ksp_view

   test:
      requires: !sycl kokkos_kernels
      suffix: 1_kokkos
      nsize: 4
      args: -ksp_view -mat_type aijkokkos

   test:
      requires: cuda
      suffix: 1_cuda
      nsize: 4
      args: -ksp_view -mat_type aijcusparse -sub_pc_factor_mat_solver_type cusparse

   test:
      suffix: 2
      nsize: 4
      args: -user_subdomains -ksp_view

   test:
      requires: !sycl kokkos_kernels
      suffix: 2_kokkos
      nsize: 4
      args: -user_subdomains -ksp_view -mat_type aijkokkos

   test:
      requires: cuda
      suffix: 2_cuda
      nsize: 4
      args: -user_subdomains -ksp_view -mat_type aijcusparse -sub_pc_factor_mat_solver_type cusparse

TEST*/
