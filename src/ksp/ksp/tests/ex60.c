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
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &args, NULL,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL, "-M", &M, NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL, "-N", &N, NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL, "-user_subdomains", &userSubdomains, NULL));
  /* Do parallel decomposition */
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  sized = (PetscMPIInt) PetscSqrtReal((PetscReal) size);
  PetscCheckFalse(PetscSqr(sized) != size,PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "This test may only be run on a number of processes which is a perfect square, not %d", (int) size);
  PetscCheckFalse(M % sized,PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "The number of x-vertices %D does not divide the number of x-processes %d", M, (int) sized);
  PetscCheckFalse(N % sized,PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "The number of y-vertices %D does not divide the number of y-processes %d", N, (int) sized);
  /* Assemble the matrix for the five point stencil, YET AGAIN
       Every other process will be empty */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD, &A));
  m    = (sized > 1) ? (rank % 2) ? 0 : 2*M/sized : M;
  n    = N/sized;
  CHKERRQ(MatSetSizes(A, m*n, m*n, M*N, M*N));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));
  CHKERRQ(MatGetOwnershipRange(A, &rstart, &rend));
  for (r = rstart; r < rend; ++r) {
    const PetscScalar diag = 4.0, offdiag = -1.0;
    const PetscInt    i    = r/N;
    const PetscInt    j    = r - i*N;
    PetscInt          c;

    if (i > 0)   {c = r - n; CHKERRQ(MatSetValues(A, 1, &r, 1, &c, &offdiag, INSERT_VALUES));}
    if (i < M-1) {c = r + n; CHKERRQ(MatSetValues(A, 1, &r, 1, &c, &offdiag, INSERT_VALUES));}
    if (j > 0)   {c = r - 1; CHKERRQ(MatSetValues(A, 1, &r, 1, &c, &offdiag, INSERT_VALUES));}
    if (j < N-1) {c = r + 1; CHKERRQ(MatSetValues(A, 1, &r, 1, &c, &offdiag, INSERT_VALUES));}
    CHKERRQ(MatSetValues(A, 1, &r, 1, &r, &diag, INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  /* Setup Solve */
  CHKERRQ(MatCreateVecs(A, &x, &b));
  CHKERRQ(VecDuplicate(x, &u));
  CHKERRQ(VecSet(u, 1.0));
  CHKERRQ(MatMult(A, u, b));
  CHKERRQ(KSPCreate(PETSC_COMM_WORLD, &ksp));
  CHKERRQ(KSPSetOperators(ksp, A, A));
  CHKERRQ(KSPGetPC(ksp, &pc));
  CHKERRQ(PCSetType(pc, PCASM));
  /* Setup ASM by hand */
  if (userSubdomains) {
    IS        is;
    PetscInt *rows;

    /* Use no overlap for now */
    CHKERRQ(PetscMalloc1(rend-rstart, &rows));
    for (r = rstart; r < rend; ++r) rows[r-rstart] = r;
    CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF, rend-rstart, rows, PETSC_OWN_POINTER, &is));
    CHKERRQ(PCASMSetLocalSubdomains(pc, 1, &is, &is));
    CHKERRQ(ISDestroy(&is));
  }
  CHKERRQ(KSPSetFromOptions(ksp));
  /* Solve and Compare */
  CHKERRQ(KSPSolve(ksp, b, x));
  CHKERRQ(VecAXPY(x, -1.0, u));
  CHKERRQ(VecNorm(x, NORM_INFINITY, &error));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "Infinity norm of the error: %g\n", (double) error));
  /* Cleanup */
  CHKERRQ(KSPDestroy(&ksp));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(VecDestroy(&u));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&b));
  ierr = PetscFinalize();
  return ierr;
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
