
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
  Vec         x, b, u; /* approx solution, RHS, exact solution */
  Mat         A, Pmat; /* linear system matrix */
  KSP         ksp;     /* linear solver context */
  PetscReal   norm;    /* norm of solution error */
  PetscInt    i, j, Ii, J, Istart, Iend, n = 7, m = 8, its, nblocks = 2;
  PetscBool   flg;
  PetscScalar v;
  PetscMPIInt size, rank;
  IS         *is_loc = NULL;
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
  // MatDuplicate keeps the zero
  PetscCall(MatCreate(PETSC_COMM_WORLD, &Pmat));
  PetscCall(MatSetSizes(Pmat, PETSC_DECIDE, PETSC_DECIDE, n * m, n * m));
  PetscCall(MatSetFromOptions(Pmat));
  PetscCall(MatSeqAIJSetPreallocation(Pmat, 5, NULL));
  PetscCall(MatMPIAIJSetPreallocation(Pmat, 5, NULL, 3, NULL));
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
                Setup ASM solver and batched KSP solver data
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(PCASMCreateSubdomains(A, nblocks, &is_loc));
  {
    MatScalar *AA;
    PetscInt  *AJ, maxcols = 0, ncols;
    for (PetscInt row = Istart; row < Iend; row++) {
      PetscCall(MatGetRow(A, row, &ncols, NULL, NULL));
      if (ncols > maxcols) maxcols = ncols;
      PetscCall(MatRestoreRow(A, row, &ncols, NULL, NULL));
    }
    PetscCall(PetscMalloc2(maxcols, &AA, maxcols, &AJ));
    /* make explicit block matrix for batch solver */
    //if (rank==1) PetscCall(PetscPrintf(PETSC_COMM_SELF, "[%d] nblocks = %d\n", rank, nblocks));
    for (PetscInt bid = 0; bid < nblocks; bid++) {
      IS blk_is = is_loc[bid];
      //if (rank==1) PetscCall(ISView(blk_is, PETSC_VIEWER_STDOUT_SELF));
      const PetscInt *subdom, *cols;
      PetscInt        n, ncol_row, jj;
      PetscCall(ISGetIndices(blk_is, &subdom));
      PetscCall(ISGetSize(blk_is, &n));
      //if (rank==1) PetscCall(PetscPrintf(PETSC_COMM_SELF, "\t[%d] n[%d] = %d\n",rank,bid,n));
      for (PetscInt ii = 0; ii < n; ii++) {
        const MatScalar *vals;
        //if (rank==1) PetscCall(PetscPrintf(PETSC_COMM_SELF, "\t\t[%d] subdom[%d] = %d\n",rank,ii,subdom[ii]));
        PetscInt rowB = subdom[ii]; // global
        PetscCall(MatGetRow(A, rowB, &ncols, &cols, &vals));
        for (jj = ncol_row = 0; jj < ncols; jj++) {
          PetscInt idx, colj = cols[jj];
          PetscCall(ISLocate(blk_is, colj, &idx));
          if (idx >= 0) {
            AJ[ncol_row] = cols[jj];
            AA[ncol_row] = vals[jj];
            ncol_row++;
          }
        }
        PetscCall(MatRestoreRow(A, rowB, &ncols, &cols, &vals));
        PetscCall(MatSetValues(Pmat, 1, &rowB, ncol_row, AJ, AA, INSERT_VALUES));
      }
      PetscCall(ISRestoreIndices(blk_is, &subdom));
    }
    PetscCall(PetscFree2(AA, AJ));
    PetscCall(MatAssemblyBegin(Pmat, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(Pmat, MAT_FINAL_ASSEMBLY));
    PetscCall(MatViewFromOptions(Pmat, NULL, "-view_c"));
  }
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
  if (flg && nblocks > 0) { PetscCall(PCASMSetLocalSubdomains(pc, nblocks, is_loc, NULL)); }
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(KSPSolve(ksp, b, x));
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Check the solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(VecAXPY(x, -1.0, u));
  PetscCall(VecNorm(x, NORM_2, &norm));
  PetscCall(KSPGetIterationNumber(ksp, &its));
  /*
     Print convergence information.
  */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Norm of error %g iterations %" PetscInt_FMT "\n", (double)norm, its));
  /*
    cleanup
  */
  PetscCall(KSPDestroy(&ksp));
  if (is_loc) {
    if (0) {
      for (PetscInt bid = 0; bid < nblocks; bid++) PetscCall(ISDestroy(&is_loc[bid]));
      PetscCall(PetscFree(is_loc));
    } else {
      PetscCall(PCASMDestroySubdomains(nblocks, is_loc, NULL));
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
    requires: parmetis kokkos_kernels
  testset:
    args: -ksp_converged_reason -ksp_norm_type unpreconditioned -ksp_rtol 1e-4 -m 37 -n 23 -num_local_blocks 4
    nsize: 4
    output_file: output/ex19_0.out
    test:
      suffix: batch
      args: -ksp_type cg -pc_type bjkokkos -pc_bjkokkos_ksp_max_it 60 -pc_bjkokkos_ksp_rtol 1e-1 -pc_bjkokkos_ksp_type tfqmr -pc_bjkokkos_pc_type jacobi -pc_bjkokkos_ksp_rtol 1e-3 -mat_type aijkokkos
    test:
      suffix: asm
      args: -ksp_type cg -pc_type asm -sub_pc_type jacobi -sub_ksp_type tfqmr -sub_ksp_rtol 1e-3

 TEST*/
