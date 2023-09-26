static char help[] = "Solve a 2D 5-point stencil in parallel with Kokkos batched KSP and ASM solvers.\n\
Input parameters include:\n\
  -n                : number of mesh points in x direction\n\
  -m                : number of mesh points in y direction\n\
  -num_local_blocks : number of local sub domains for block jacobi\n\n";

/*
  Include "petscksp.h" so that we can use KSP solvers.
*/
#include <petscksp.h>
#include <petscmat.h>

int main(int argc, char **args)
{
  Vec         x, b, u;           /* approx solution, RHS, exact solution */
  Mat         A, Pmat, Aseq, AA; /* linear system matrix */
  KSP         ksp;               /* linear solver context */
  PetscReal   norm, norm0;       /* norm of solution error */
  PetscInt    i, j, Ii, J, Istart, Iend, n = 7, m = 8, its, nblocks = 2;
  PetscBool   flg, ismpi;
  PetscScalar v;
  PetscMPIInt size, rank;
  IS         *loc_blocks = NULL;
  PC          pc;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-m", &m, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-num_local_blocks", &nblocks, NULL));
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Compute the matrix and right-hand-side vector that define
         the linear system, Ax = b.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n * m, n * m));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSeqAIJSetPreallocation(A, 5, NULL));
  PetscCall(MatMPIAIJSetPreallocation(A, 5, NULL, 3, NULL));
  /*
     Currently, all PETSc parallel matrix formats are partitioned by
     contiguous chunks of rows across the processors.  Determine which
     rows of the matrix are locally owned.
  */
  PetscCall(MatGetOwnershipRange(A, &Istart, &Iend));
  /*
    Set matrix elements for the 2-D, five-point stencil.
    */
  for (Ii = Istart; Ii < Iend; Ii++) {
    v = -1.0;
    i = Ii / n;
    j = Ii - i * n;
    if (i > 0) {
      J = Ii - n;
      PetscCall(MatSetValues(A, 1, &Ii, 1, &J, &v, ADD_VALUES));
    }
    if (i < m - 1) {
      J = Ii + n;
      PetscCall(MatSetValues(A, 1, &Ii, 1, &J, &v, ADD_VALUES));
    }
    if (j > 0) {
      J = Ii - 1;
      PetscCall(MatSetValues(A, 1, &Ii, 1, &J, &v, ADD_VALUES));
    }
    if (j < n - 1) {
      J = Ii + 1;
      PetscCall(MatSetValues(A, 1, &Ii, 1, &J, &v, ADD_VALUES));
    }
    v = 4.0;
    PetscCall(MatSetValues(A, 1, &Ii, 1, &Ii, &v, ADD_VALUES));
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Setup ASM solver and batched KSP solver data
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /* make explicit block matrix for batch solver */
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)A, MATMPIAIJ, &ismpi));
  if (!ismpi) {
    Aseq = A;
  } else {
    PetscCall(MatMPIAIJGetSeqAIJ(A, &Aseq, NULL, NULL));
  }
  PetscCall(PCASMCreateSubdomains(Aseq, nblocks, &loc_blocks)); // A
  Mat nest, arrray[10000];
  for (Ii = 0; Ii < 10000; Ii++) arrray[Ii] = NULL;
  for (PetscInt bid = 0; bid < nblocks; bid++) {
    Mat matblock;
    PetscCall(MatCreateSubMatrix(Aseq, loc_blocks[bid], loc_blocks[bid], MAT_INITIAL_MATRIX, &matblock));
    //PetscCall(MatViewFromOptions(matblock, NULL, "-view_b"));
    arrray[bid * nblocks + bid] = matblock;
  }
  PetscCall(MatCreate(PETSC_COMM_SELF, &nest));
  PetscCall(MatSetFromOptions(nest));
  PetscCall(MatSetType(nest, MATNEST));
  PetscCall(MatNestSetSubMats(nest, nblocks, NULL, nblocks, NULL, arrray));
  PetscCall(MatSetUp(nest));
  PetscCall(MatConvert(nest, MATAIJKOKKOS, MAT_INITIAL_MATRIX, &AA));
  PetscCall(MatDestroy(&nest));
  for (PetscInt bid = 0; bid < nblocks; bid++) PetscCall(MatDestroy(&arrray[bid * nblocks + bid]));
  if (ismpi) {
    Mat AAseq;
    PetscCall(MatCreate(PETSC_COMM_WORLD, &Pmat));
    PetscCall(MatSetSizes(Pmat, Iend - Istart, Iend - Istart, n * m, n * m));
    PetscCall(MatSetFromOptions(Pmat));
    PetscCall(MatSeqAIJSetPreallocation(Pmat, 5, NULL));
    PetscCall(MatMPIAIJSetPreallocation(Pmat, 5, NULL, 3, NULL));
    PetscCall(MatAssemblyBegin(Pmat, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(Pmat, MAT_FINAL_ASSEMBLY));
    PetscCall(MatMPIAIJGetSeqAIJ(Pmat, &AAseq, NULL, NULL));
    PetscCheck(AAseq, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "No A mat");
    PetscCall(MatAXPY(AAseq, 1.0, AA, DIFFERENT_NONZERO_PATTERN));
    PetscCall(MatDestroy(&AA));
  } else {
    Pmat = AA;
  }
  PetscCall(MatViewFromOptions(Pmat, NULL, "-view_p"));
  PetscCall(MatViewFromOptions(A, NULL, "-view_a"));

  /* A is symmetric. Set symmetric flag to enable ICC/Cholesky preconditioner */
  PetscCall(MatSetOption(A, MAT_SYMMETRIC, PETSC_TRUE));
  PetscCall(MatCreateVecs(A, &u, &b));
  PetscCall(MatCreateVecs(A, &x, NULL));
  /*
     Set exact solution; then compute right-hand-side vector.
     By default we use an exact solution of a vector with all
     elements of 1.0;
  */
  PetscCall(VecSet(u, 1.0));
  PetscCall(MatMult(A, u, b));
  /*
     View the exact solution vector if desired
  */
  flg = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-view_exact_sol", &flg, NULL));
  if (flg) PetscCall(VecView(u, PETSC_VIEWER_STDOUT_WORLD));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, Pmat));
  PetscCall(KSPSetFromOptions(ksp));
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Setup ASM solver
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PetscObjectTypeCompare((PetscObject)pc, PCASM, &flg));
  if (flg && nblocks > 0) {
    for (PetscInt bid = 0, gid0 = Istart; bid < nblocks; bid++) {
      PetscInt nn;
      IS       new_loc_blocks;
      PetscCall(ISGetSize(loc_blocks[bid], &nn)); // size only
      PetscCall(ISCreateStride(PETSC_COMM_SELF, nn, gid0, 1, &new_loc_blocks));
      PetscCall(ISDestroy(&loc_blocks[bid]));
      loc_blocks[bid] = new_loc_blocks;
      gid0 += nn; // start of next block
    }
    PetscCall(PCASMSetLocalSubdomains(pc, nblocks, loc_blocks, NULL));
  }
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(KSPSolve(ksp, b, x));
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Check the solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(VecAXPY(x, -1.0, u));
  PetscCall(VecNorm(x, NORM_2, &norm));
  PetscCall(VecNorm(b, NORM_2, &norm0));
  PetscCall(KSPGetIterationNumber(ksp, &its));
  /*
     Print convergence information.
  */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Relative norm of error %g iterations %" PetscInt_FMT "\n", (double)norm / norm0, its));
  /*
    cleanup
  */
  PetscCall(KSPDestroy(&ksp));
  if (loc_blocks) {
    if (0) {
      for (PetscInt bid = 0; bid < nblocks; bid++) PetscCall(ISDestroy(&loc_blocks[bid]));
      PetscCall(PetscFree(loc_blocks));
    } else {
      PetscCall(PCASMDestroySubdomains(nblocks, loc_blocks, NULL));
    }
  }
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&Pmat));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
  build:
    requires: kokkos_kernels
  testset:
    requires: parmetis
    args: -ksp_converged_reason -ksp_norm_type unpreconditioned -ksp_rtol 1e-4 -m 37 -n 23 -num_local_blocks 4
    nsize: 4
    output_file: output/ex19_0.out
    test:
      suffix: batch
      args: -ksp_type cg -pc_type bjkokkos -pc_bjkokkos_ksp_max_it 60 -pc_bjkokkos_ksp_type tfqmr -pc_bjkokkos_pc_type jacobi -pc_bjkokkos_ksp_rtol 1e-3 -mat_type aijkokkos
    test:
      suffix: asm
      args: -ksp_type cg -pc_type asm -sub_pc_type jacobi -sub_ksp_type tfqmr -sub_ksp_rtol 1e-3
    test:
      suffix: batch_bicg
      args:  -ksp_type cg -pc_type bjkokkos -pc_bjkokkos_ksp_max_it 60 -pc_bjkokkos_ksp_type bicg -pc_bjkokkos_pc_type jacobi -pc_bjkokkos_ksp_rtol 1e-3 -mat_type aijkokkos

  test:
    nsize: 4
    suffix: no_metis_batch
    args: -ksp_converged_reason -ksp_norm_type unpreconditioned -ksp_rtol 1e-6 -m 37 -n 23 -num_local_blocks 4 -ksp_type cg -pc_type bjkokkos -pc_bjkokkos_ksp_max_it 60 -pc_bjkokkos_ksp_type tfqmr -pc_bjkokkos_pc_type jacobi -pc_bjkokkos_ksp_rtol 1e-3 -mat_type aijkokkos

  test:
    nsize: 1
    suffix: serial_batch
    args: -ksp_monitor -ksp_converged_reason -ksp_norm_type unpreconditioned -ksp_rtol 1e-4 -m 37 -n 23 -num_local_blocks 16 -ksp_type cg -pc_type bjkokkos -pc_bjkokkos_ksp_max_it 60 -pc_bjkokkos_ksp_type tfqmr -pc_bjkokkos_pc_type jacobi -pc_bjkokkos_ksp_rtol 1e-6 -mat_type aijkokkos -pc_bjkokkos_ksp_converged_reason

 TEST*/
