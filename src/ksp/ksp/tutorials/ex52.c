
static char help[] = "Solves a linear system in parallel with KSP. Modified from ex2.c \n\
                      Illustrate how to use external packages MUMPS, SUPERLU and STRUMPACK \n\
Input parameters include:\n\
  -random_exact_sol : use a random exact solution vector\n\
  -view_exact_sol   : write exact solution vector to stdout\n\
  -m <mesh_x>       : number of mesh points in x-direction\n\
  -n <mesh_y>       : number of mesh points in y-direction\n\n";

#include <petscksp.h>

#if defined(PETSC_HAVE_MUMPS)
/* Subroutine contributed by Varun Hiremath */
PetscErrorCode printMumpsMemoryInfo(Mat F)
{
  PetscInt maxMem, sumMem;

  PetscFunctionBeginUser;
  PetscCall(MatMumpsGetInfog(F, 16, &maxMem));
  PetscCall(MatMumpsGetInfog(F, 17, &sumMem));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n MUMPS INFOG(16) :: Max memory in MB = %" PetscInt_FMT, maxMem));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n MUMPS INFOG(17) :: Sum memory in MB = %" PetscInt_FMT "\n", sumMem));
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

int main(int argc, char **args)
{
  Vec         x, b, u; /* approx solution, RHS, exact solution */
  Mat         A, F;
  KSP         ksp; /* linear solver context */
  PC          pc;
  PetscRandom rctx; /* random number generator context */
  PetscReal   norm; /* norm of solution error */
  PetscInt    i, j, Ii, J, Istart, Iend, m = 8, n = 7, its;
  PetscBool   flg = PETSC_FALSE, flg_ilu = PETSC_FALSE, flg_ch = PETSC_FALSE;
#if defined(PETSC_HAVE_MUMPS)
  PetscBool flg_mumps = PETSC_FALSE, flg_mumps_ch = PETSC_FALSE;
#endif
#if defined(PETSC_HAVE_SUPERLU) || defined(PETSC_HAVE_SUPERLU_DIST)
  PetscBool flg_superlu = PETSC_FALSE;
#endif
#if defined(PETSC_HAVE_STRUMPACK)
  PetscBool flg_strumpack = PETSC_FALSE;
#endif
  PetscScalar v;
  PetscMPIInt rank, size;
#if defined(PETSC_USE_LOG)
  PetscLogStage stage;
#endif

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-m", &m, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Compute the matrix and right-hand-side vector that define
         the linear system, Ax = b.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, m * n, m * n));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatMPIAIJSetPreallocation(A, 5, NULL, 5, NULL));
  PetscCall(MatSeqAIJSetPreallocation(A, 5, NULL));
  PetscCall(MatSetUp(A));

  /*
     Currently, all PETSc parallel matrix formats are partitioned by
     contiguous chunks of rows across the processors.  Determine which
     rows of the matrix are locally owned.
  */
  PetscCall(MatGetOwnershipRange(A, &Istart, &Iend));

  /*
     Set matrix elements for the 2-D, five-point stencil in parallel.
      - Each processor needs to insert only elements that it owns
        locally (but any non-local elements will be sent to the
        appropriate processor during matrix assembly).
      - Always specify global rows and columns of matrix entries.

     Note: this uses the less common natural ordering that orders first
     all the unknowns for x = h then for x = 2h etc; Hence you see J = Ii +- n
     instead of J = I +- m as you might expect. The more standard ordering
     would first do all variables for y = h, then y = 2h etc.

   */
  PetscCall(PetscLogStageRegister("Assembly", &stage));
  PetscCall(PetscLogStagePush(stage));
  for (Ii = Istart; Ii < Iend; Ii++) {
    v = -1.0;
    i = Ii / n;
    j = Ii - i * n;
    if (i > 0) {
      J = Ii - n;
      PetscCall(MatSetValues(A, 1, &Ii, 1, &J, &v, INSERT_VALUES));
    }
    if (i < m - 1) {
      J = Ii + n;
      PetscCall(MatSetValues(A, 1, &Ii, 1, &J, &v, INSERT_VALUES));
    }
    if (j > 0) {
      J = Ii - 1;
      PetscCall(MatSetValues(A, 1, &Ii, 1, &J, &v, INSERT_VALUES));
    }
    if (j < n - 1) {
      J = Ii + 1;
      PetscCall(MatSetValues(A, 1, &Ii, 1, &J, &v, INSERT_VALUES));
    }
    v = 4.0;
    PetscCall(MatSetValues(A, 1, &Ii, 1, &Ii, &v, INSERT_VALUES));
  }

  /*
     Assemble matrix, using the 2-step process:
       MatAssemblyBegin(), MatAssemblyEnd()
     Computations can be done while messages are in transition
     by placing code between these two statements.
  */
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(PetscLogStagePop());

  /* A is symmetric. Set symmetric flag to enable ICC/Cholesky preconditioner */
  PetscCall(MatSetOption(A, MAT_SYMMETRIC, PETSC_TRUE));

  /* Create parallel vectors */
  PetscCall(MatCreateVecs(A, &u, &b));
  PetscCall(VecDuplicate(u, &x));

  /*
     Set exact solution; then compute right-hand-side vector.
     By default we use an exact solution of a vector with all
     elements of 1.0;  Alternatively, using the runtime option
     -random_sol forms a solution vector with random components.
  */
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-random_exact_sol", &flg, NULL));
  if (flg) {
    PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rctx));
    PetscCall(PetscRandomSetFromOptions(rctx));
    PetscCall(VecSetRandom(u, rctx));
    PetscCall(PetscRandomDestroy(&rctx));
  } else {
    PetscCall(VecSet(u, 1.0));
  }
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

  /*
     Create linear solver context
  */
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));

  /*
    Example of how to use external package MUMPS
    Note: runtime options
          '-ksp_type preonly -pc_type lu -pc_factor_mat_solver_type mumps -mat_mumps_icntl_7 3 -mat_mumps_icntl_1 0.0'
          are equivalent to these procedural calls
  */
#if defined(PETSC_HAVE_MUMPS)
  flg_mumps    = PETSC_FALSE;
  flg_mumps_ch = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-use_mumps_lu", &flg_mumps, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-use_mumps_ch", &flg_mumps_ch, NULL));
  if (flg_mumps || flg_mumps_ch) {
    PetscCall(KSPSetType(ksp, KSPPREONLY));
    PetscInt  ival, icntl;
    PetscReal val;
    PetscCall(KSPGetPC(ksp, &pc));
    if (flg_mumps) {
      PetscCall(PCSetType(pc, PCLU));
    } else if (flg_mumps_ch) {
      PetscCall(MatSetOption(A, MAT_SPD, PETSC_TRUE)); /* set MUMPS id%SYM=1 */
      PetscCall(PCSetType(pc, PCCHOLESKY));
    }
    PetscCall(PCFactorSetMatSolverType(pc, MATSOLVERMUMPS));
    PetscCall(PCFactorSetUpMatSolverType(pc)); /* call MatGetFactor() to create F */
    PetscCall(PCFactorGetMatrix(pc, &F));
    PetscCall(MatMumpsSetIcntl(F, 24, 1));
    PetscCall(MatMumpsGetIcntl(F, 24, &ival));
    PetscCheck(ival == 1, PetscObjectComm((PetscObject)F), PETSC_ERR_LIB, "ICNTL(24) = %" PetscInt_FMT " (!= 1)", ival);
    PetscCall(MatMumpsSetCntl(F, 3, 1e-6));
    PetscCall(MatMumpsGetCntl(F, 3, &val));
    PetscCheck(PetscEqualReal(val, 1e-6), PetscObjectComm((PetscObject)F), PETSC_ERR_LIB, "CNTL(3) = %g (!= %g)", (double)val, 1e-6);
    if (flg_mumps) {
      /* Zero the first and last rows in the rank, they should then show up in corresponding null pivot rows output via
         MatMumpsGetNullPivots */
      flg = PETSC_FALSE;
      PetscCall(PetscOptionsGetBool(NULL, NULL, "-zero_first_and_last_rows", &flg, NULL));
      if (flg) {
        PetscInt rows[2];
        rows[0] = Istart;   /* first row of the rank */
        rows[1] = Iend - 1; /* last row of the rank */
        PetscCall(MatZeroRows(A, 2, rows, 0.0, NULL, NULL));
      }
      /* Get memory estimates from MUMPS' MatLUFactorSymbolic(), e.g. INFOG(16), INFOG(17).
         KSPSetUp() below will do nothing inside MatLUFactorSymbolic() */
      MatFactorInfo info;
      PetscCall(MatLUFactorSymbolic(F, A, NULL, NULL, &info));
      flg = PETSC_FALSE;
      PetscCall(PetscOptionsGetBool(NULL, NULL, "-print_mumps_memory", &flg, NULL));
      if (flg) PetscCall(printMumpsMemoryInfo(F));
    }

    /* sequential ordering */
    icntl = 7;
    ival  = 2;
    PetscCall(MatMumpsSetIcntl(F, icntl, ival));

    /* threshold for row pivot detection */
    PetscCall(MatMumpsGetIcntl(F, 24, &ival));
    PetscCheck(ival == 1, PetscObjectComm((PetscObject)F), PETSC_ERR_LIB, "ICNTL(24) = %" PetscInt_FMT " (!= 1)", ival);
    icntl = 3;
    PetscCall(MatMumpsGetCntl(F, icntl, &val));
    PetscCheck(PetscEqualReal(val, 1e-6), PetscObjectComm((PetscObject)F), PETSC_ERR_LIB, "CNTL(3) = %g (!= %g)", (double)val, 1e-6);

    /* compute determinant of A */
    PetscCall(MatMumpsSetIcntl(F, 33, 1));
  }
#endif

  /*
    Example of how to use external package SuperLU
    Note: runtime options
          '-ksp_type preonly -pc_type ilu -pc_factor_mat_solver_type superlu -mat_superlu_ilu_droptol 1.e-8'
          are equivalent to these procedual calls
  */
#if defined(PETSC_HAVE_SUPERLU) || defined(PETSC_HAVE_SUPERLU_DIST)
  flg_ilu     = PETSC_FALSE;
  flg_superlu = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-use_superlu_lu", &flg_superlu, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-use_superlu_ilu", &flg_ilu, NULL));
  if (flg_superlu || flg_ilu) {
    PetscCall(KSPSetType(ksp, KSPPREONLY));
    PetscCall(KSPGetPC(ksp, &pc));
    if (flg_superlu) PetscCall(PCSetType(pc, PCLU));
    else if (flg_ilu) PetscCall(PCSetType(pc, PCILU));
    if (size == 1) {
  #if !defined(PETSC_HAVE_SUPERLU)
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "This test requires SUPERLU");
  #else
      PetscCall(PCFactorSetMatSolverType(pc, MATSOLVERSUPERLU));
  #endif
    } else {
  #if !defined(PETSC_HAVE_SUPERLU_DIST)
      SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "This test requires SUPERLU_DIST");
  #else
      PetscCall(PCFactorSetMatSolverType(pc, MATSOLVERSUPERLU_DIST));
  #endif
    }
    PetscCall(PCFactorSetUpMatSolverType(pc)); /* call MatGetFactor() to create F */
    PetscCall(PCFactorGetMatrix(pc, &F));
  #if defined(PETSC_HAVE_SUPERLU)
    if (size == 1) PetscCall(MatSuperluSetILUDropTol(F, 1.e-8));
  #endif
  }
#endif

  /*
    Example of how to use external package STRUMPACK
    Note: runtime options
          '-pc_type lu/ilu \
           -pc_factor_mat_solver_type strumpack \
           -mat_strumpack_reordering METIS \
           -mat_strumpack_colperm 0 \
           -mat_strumpack_hss_rel_tol 1.e-3 \
           -mat_strumpack_hss_min_sep_size 50 \
           -mat_strumpack_max_rank 100 \
           -mat_strumpack_leaf_size 4'
       are equivalent to these procedural calls

    We refer to the STRUMPACK-sparse manual, section 5, for more info on
    how to tune the preconditioner.
  */
#if defined(PETSC_HAVE_STRUMPACK)
  flg_ilu       = PETSC_FALSE;
  flg_strumpack = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-use_strumpack_lu", &flg_strumpack, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-use_strumpack_ilu", &flg_ilu, NULL));
  if (flg_strumpack || flg_ilu) {
    PetscCall(KSPSetType(ksp, KSPPREONLY));
    PetscCall(KSPGetPC(ksp, &pc));
    if (flg_strumpack) PetscCall(PCSetType(pc, PCLU));
    else if (flg_ilu) PetscCall(PCSetType(pc, PCILU));
  #if !defined(PETSC_HAVE_STRUMPACK)
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "This test requires STRUMPACK");
  #endif
    PetscCall(PCFactorSetMatSolverType(pc, MATSOLVERSTRUMPACK));
    PetscCall(PCFactorSetUpMatSolverType(pc)); /* call MatGetFactor() to create F */
    PetscCall(PCFactorGetMatrix(pc, &F));
  #if defined(PETSC_HAVE_STRUMPACK)
    /* Set the fill-reducing reordering.                              */
    PetscCall(MatSTRUMPACKSetReordering(F, MAT_STRUMPACK_METIS));
    /* Since this is a simple discretization, the diagonal is always  */
    /* nonzero, and there is no need for the extra MC64 permutation.  */
    PetscCall(MatSTRUMPACKSetColPerm(F, PETSC_FALSE));
    /* The compression tolerance used when doing low-rank compression */
    /* in the preconditioner. This is problem specific!               */
    PetscCall(MatSTRUMPACKSetHSSRelTol(F, 1.e-3));
    /* Set minimum matrix size for HSS compression to 15 in order to  */
    /* demonstrate preconditioner on small problems. For performance  */
    /* a value of say 500 is better.                                  */
    PetscCall(MatSTRUMPACKSetHSSMinSepSize(F, 15));
    /* You can further limit the fill in the preconditioner by        */
    /* setting a maximum rank                                         */
    PetscCall(MatSTRUMPACKSetHSSMaxRank(F, 100));
    /* Set the size of the diagonal blocks (the leafs) in the HSS     */
    /* approximation. The default value should be better for real     */
    /* problems. This is mostly for illustration on a small problem.  */
    PetscCall(MatSTRUMPACKSetHSSLeafSize(F, 4));
  #endif
  }
#endif

  /*
    Example of how to use procedural calls that are equivalent to
          '-ksp_type preonly -pc_type lu/ilu -pc_factor_mat_solver_type petsc'
  */
  flg     = PETSC_FALSE;
  flg_ilu = PETSC_FALSE;
  flg_ch  = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-use_petsc_lu", &flg, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-use_petsc_ilu", &flg_ilu, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-use_petsc_ch", &flg_ch, NULL));
  if (flg || flg_ilu || flg_ch) {
    Vec diag;

    PetscCall(KSPSetType(ksp, KSPPREONLY));
    PetscCall(KSPGetPC(ksp, &pc));
    if (flg) PetscCall(PCSetType(pc, PCLU));
    else if (flg_ilu) PetscCall(PCSetType(pc, PCILU));
    else if (flg_ch) PetscCall(PCSetType(pc, PCCHOLESKY));
    PetscCall(PCFactorSetMatSolverType(pc, MATSOLVERPETSC));
    PetscCall(PCFactorSetUpMatSolverType(pc)); /* call MatGetFactor() to create F */
    PetscCall(PCFactorGetMatrix(pc, &F));

    /* Test MatGetDiagonal() */
    PetscCall(KSPSetUp(ksp));
    PetscCall(VecDuplicate(x, &diag));
    PetscCall(MatGetDiagonal(F, diag));
    /* PetscCall(VecView(diag,PETSC_VIEWER_STDOUT_WORLD)); */
    PetscCall(VecDestroy(&diag));
  }

  PetscCall(KSPSetFromOptions(ksp));

  /* Get info from matrix factors */
  PetscCall(KSPSetUp(ksp));

#if defined(PETSC_HAVE_MUMPS)
  if (flg_mumps || flg_mumps_ch) {
    PetscInt  icntl, infog34, num_null_pivots, *null_pivots;
    PetscReal cntl, rinfo12, rinfo13;
    icntl = 3;
    PetscCall(MatMumpsGetCntl(F, icntl, &cntl));

    /* compute determinant and check for any null pivots*/
    if (rank == 0) {
      PetscCall(MatMumpsGetInfog(F, 34, &infog34));
      PetscCall(MatMumpsGetRinfog(F, 12, &rinfo12));
      PetscCall(MatMumpsGetRinfog(F, 13, &rinfo13));
      PetscCall(MatMumpsGetNullPivots(F, &num_null_pivots, &null_pivots));
      PetscCall(PetscPrintf(PETSC_COMM_SELF, "  Mumps row pivot threshold = %g\n", cntl));
      PetscCall(PetscPrintf(PETSC_COMM_SELF, "  Mumps determinant = (%g, %g) * 2^%" PetscInt_FMT " \n", (double)rinfo12, (double)rinfo13, infog34));
      if (num_null_pivots > 0) {
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "  Mumps num of null pivots detected = %" PetscInt_FMT "\n", num_null_pivots));
        PetscCall(PetscSortInt(num_null_pivots, null_pivots)); /* just make the printf deterministic */
        for (j = 0; j < num_null_pivots; j++) PetscCall(PetscPrintf(PETSC_COMM_SELF, "  Mumps row with null pivots is = %" PetscInt_FMT "\n", null_pivots[j]));
      }
      PetscCall(PetscFree(null_pivots));
    }
  }
#endif

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(KSPSolve(ksp, b, x));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Check solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(VecAXPY(x, -1.0, u));
  PetscCall(VecNorm(x, NORM_2, &norm));
  PetscCall(KSPGetIterationNumber(ksp, &its));

  /*
     Print convergence information.  PetscPrintf() produces a single
     print statement from all processes that share a communicator.
     An alternative is PetscFPrintf(), which prints to a file.
  */
  if (norm < 1.e-12) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Norm of error < 1.e-12 iterations %" PetscInt_FMT "\n", its));
  } else {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Norm of error %g iterations %" PetscInt_FMT "\n", (double)norm, its));
  }

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));

  /*
     Always call PetscFinalize() before exiting a program.  This routine
       - finalizes the PETSc libraries as well as MPI
       - provides summary and diagnostic information if certain runtime
         options are chosen (e.g., -log_view).
  */
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      args: -use_petsc_lu
      output_file: output/ex52_2.out

   test:
      suffix: mumps
      nsize: 3
      requires: mumps
      args: -use_mumps_lu
      output_file: output/ex52_1.out

   test:
      suffix: mumps_2
      nsize: 3
      requires: mumps
      args: -use_mumps_ch
      output_file: output/ex52_1.out

   test:
      suffix: mumps_3
      nsize: 3
      requires: mumps
      args: -use_mumps_ch -mat_type sbaij
      output_file: output/ex52_1.out

   test:
      suffix: mumps_4
      nsize: 3
      requires: mumps !complex !single
      args: -use_mumps_lu -m 50 -n 50 -print_mumps_memory
      output_file: output/ex52_4.out

   test:
      suffix: mumps_5
      nsize: 3
      requires: mumps !complex !single
      args: -use_mumps_lu -m 50 -n 50 -zero_first_and_last_rows
      output_file: output/ex52_5.out

   test:
      suffix: mumps_omp_2
      nsize: 4
      requires: mumps hwloc openmp pthread defined(PETSC_HAVE_MPI_PROCESS_SHARED_MEMORY)
      args: -use_mumps_lu -mat_mumps_use_omp_threads 2
      output_file: output/ex52_1.out

   test:
      suffix: mumps_omp_3
      nsize: 4
      requires: mumps hwloc openmp pthread defined(PETSC_HAVE_MPI_PROCESS_SHARED_MEMORY)
      args: -use_mumps_ch -mat_mumps_use_omp_threads 3
      # Ignore the warning since we are intentionally testing the imbalanced case
      filter: grep -v "Warning: number of OpenMP threads"
      output_file: output/ex52_1.out

   test:
      suffix: mumps_omp_4
      nsize: 4
      requires: mumps hwloc openmp pthread defined(PETSC_HAVE_MPI_PROCESS_SHARED_MEMORY)
      # let petsc guess a proper number for threads
      args: -use_mumps_ch -mat_type sbaij -mat_mumps_use_omp_threads
      output_file: output/ex52_1.out

   testset:
      suffix: strumpack_2
      nsize: {{1 2}}
      requires: strumpack
      args: -use_strumpack_lu
      output_file: output/ex52_3.out

      test:
        suffix: aij
        args: -mat_type aij

      test:
        requires: kokkos_kernels
        suffix: kok
        args: -mat_type aijkokkos

      test:
        requires: cuda
        suffix: cuda
        args: -mat_type aijcusparse

      test:
        requires: hip
        suffix: hip
        args: -mat_type aijhipsparse

   test:
      suffix: strumpack_ilu
      nsize: {{1 2}}
      requires: strumpack
      args: -use_strumpack_ilu
      output_file: output/ex52_3.out

   testset:
      suffix: superlu_dist
      nsize: {{1 2}}
      requires: superlu superlu_dist
      args: -use_superlu_lu
      output_file: output/ex52_2.out

      test:
        suffix: aij
        args: -mat_type aij

      test:
        requires: kokkos_kernels
        suffix: kok
        args: -mat_type aijkokkos

      test:
        requires: cuda
        suffix: cuda
        args: -mat_type aijcusparse

      test:
        requires: hip
        suffix: hip
        args: -mat_type aijhipsparse

   test:
      suffix: superlu_ilu
      requires: superlu superlu_dist
      args: -use_superlu_ilu
      output_file: output/ex52_2.out

TEST*/
