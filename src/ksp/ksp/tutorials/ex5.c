
static char help[] = "Solves two linear systems in parallel with KSP.  The code\n\
illustrates repeated solution of linear systems with the same preconditioner\n\
method but different matrices (having the same nonzero structure).  The code\n\
also uses multiple profiling stages.  Input arguments are\n\
  -m <size> : problem size\n\
  -mat_nonsym : use nonsymmetric matrix (default is symmetric)\n\n";

/*
  Include "petscksp.h" so that we can use KSP solvers.  Note that this file
  automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h - vectors
     petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
*/
#include <petscksp.h>

int main(int argc, char **args)
{
  KSP         ksp;         /* linear solver context */
  Mat         C;           /* matrix */
  Vec         x, u, b;     /* approx solution, RHS, exact solution */
  PetscReal   norm, bnorm; /* norm of solution error */
  PetscScalar v, none = -1.0;
  PetscInt    Ii, J, ldim, low, high, iglobal, Istart, Iend;
  PetscInt    i, j, m = 3, n = 2, its;
  PetscMPIInt size, rank;
  PetscBool   mat_nonsymmetric = PETSC_FALSE;
  PetscBool   testnewC = PETSC_FALSE, testscaledMat = PETSC_FALSE;
#if defined(PETSC_USE_LOG)
  PetscLogStage stages[2];
#endif

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-m", &m, NULL));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  n = 2 * size;

  /*
     Set flag if we are doing a nonsymmetric problem; the default is symmetric.
  */
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-mat_nonsym", &mat_nonsymmetric, NULL));

  PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_scaledMat", &testscaledMat, NULL));

  /*
     Register two stages for separate profiling of the two linear solves.
     Use the runtime option -log_view for a printout of performance
     statistics at the program's conlusion.
  */
  PetscCall(PetscLogStageRegister("Original Solve", &stages[0]));
  PetscCall(PetscLogStageRegister("Second Solve", &stages[1]));

  /* -------------- Stage 0: Solve Original System ---------------------- */
  /*
     Indicate to PETSc profiling that we're beginning the first stage
  */
  PetscCall(PetscLogStagePush(stages[0]));

  /*
     Create parallel matrix, specifying only its global dimensions.
     When using MatCreate(), the matrix format can be specified at
     runtime. Also, the parallel partitioning of the matrix is
     determined by PETSc at runtime.
  */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &C));
  PetscCall(MatSetSizes(C, PETSC_DECIDE, PETSC_DECIDE, m * n, m * n));
  PetscCall(MatSetFromOptions(C));
  PetscCall(MatSetUp(C));

  /*
     Currently, all PETSc parallel matrix formats are partitioned by
     contiguous chunks of rows across the processors.  Determine which
     rows of the matrix are locally owned.
  */
  PetscCall(MatGetOwnershipRange(C, &Istart, &Iend));

  /*
     Set matrix entries matrix in parallel.
      - Each processor needs to insert only elements that it owns
        locally (but any non-local elements will be sent to the
        appropriate processor during matrix assembly).
      - Always specify global row and columns of matrix entries.
  */
  for (Ii = Istart; Ii < Iend; Ii++) {
    v = -1.0;
    i = Ii / n;
    j = Ii - i * n;
    if (i > 0) {
      J = Ii - n;
      PetscCall(MatSetValues(C, 1, &Ii, 1, &J, &v, ADD_VALUES));
    }
    if (i < m - 1) {
      J = Ii + n;
      PetscCall(MatSetValues(C, 1, &Ii, 1, &J, &v, ADD_VALUES));
    }
    if (j > 0) {
      J = Ii - 1;
      PetscCall(MatSetValues(C, 1, &Ii, 1, &J, &v, ADD_VALUES));
    }
    if (j < n - 1) {
      J = Ii + 1;
      PetscCall(MatSetValues(C, 1, &Ii, 1, &J, &v, ADD_VALUES));
    }
    v = 4.0;
    PetscCall(MatSetValues(C, 1, &Ii, 1, &Ii, &v, ADD_VALUES));
  }

  /*
     Make the matrix nonsymmetric if desired
  */
  if (mat_nonsymmetric) {
    for (Ii = Istart; Ii < Iend; Ii++) {
      v = -1.5;
      i = Ii / n;
      if (i > 1) {
        J = Ii - n - 1;
        PetscCall(MatSetValues(C, 1, &Ii, 1, &J, &v, ADD_VALUES));
      }
    }
  } else {
    PetscCall(MatSetOption(C, MAT_SYMMETRIC, PETSC_TRUE));
    PetscCall(MatSetOption(C, MAT_SYMMETRY_ETERNAL, PETSC_TRUE));
  }

  /*
     Assemble matrix, using the 2-step process:
       MatAssemblyBegin(), MatAssemblyEnd()
     Computations can be done while messages are in transition
     by placing code between these two statements.
  */
  PetscCall(MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY));

  /*
     Create parallel vectors.
      - When using VecSetSizes(), we specify only the vector's global
        dimension; the parallel partitioning is determined at runtime.
      - Note: We form 1 vector from scratch and then duplicate as needed.
  */
  PetscCall(VecCreate(PETSC_COMM_WORLD, &u));
  PetscCall(VecSetSizes(u, PETSC_DECIDE, m * n));
  PetscCall(VecSetFromOptions(u));
  PetscCall(VecDuplicate(u, &b));
  PetscCall(VecDuplicate(b, &x));

  /*
     Currently, all parallel PETSc vectors are partitioned by
     contiguous chunks across the processors.  Determine which
     range of entries are locally owned.
  */
  PetscCall(VecGetOwnershipRange(x, &low, &high));

  /*
    Set elements within the exact solution vector in parallel.
     - Each processor needs to insert only elements that it owns
       locally (but any non-local entries will be sent to the
       appropriate processor during vector assembly).
     - Always specify global locations of vector entries.
  */
  PetscCall(VecGetLocalSize(x, &ldim));
  for (i = 0; i < ldim; i++) {
    iglobal = i + low;
    v       = (PetscScalar)(i + 100 * rank);
    PetscCall(VecSetValues(u, 1, &iglobal, &v, INSERT_VALUES));
  }

  /*
     Assemble vector, using the 2-step process:
       VecAssemblyBegin(), VecAssemblyEnd()
     Computations can be done while messages are in transition,
     by placing code between these two statements.
  */
  PetscCall(VecAssemblyBegin(u));
  PetscCall(VecAssemblyEnd(u));

  /*
     Compute right-hand-side vector
  */
  PetscCall(MatMult(C, u, b));

  /*
    Create linear solver context
  */
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));

  /*
     Set operators. Here the matrix that defines the linear system
     also serves as the preconditioning matrix.
  */
  PetscCall(KSPSetOperators(ksp, C, C));

  /*
     Set runtime options (e.g., -ksp_type <type> -pc_type <type>)
  */
  PetscCall(KSPSetFromOptions(ksp));

  /*
     Solve linear system.  Here we explicitly call KSPSetUp() for more
     detailed performance monitoring of certain preconditioners, such
     as ICC and ILU.  This call is optional, as KSPSetUp() will
     automatically be called within KSPSolve() if it hasn't been
     called already.
  */
  PetscCall(KSPSetUp(ksp));
  PetscCall(KSPSolve(ksp, b, x));

  /*
     Check the residual
  */
  PetscCall(VecAXPY(x, none, u));
  PetscCall(VecNorm(x, NORM_2, &norm));
  PetscCall(VecNorm(b, NORM_2, &bnorm));
  PetscCall(KSPGetIterationNumber(ksp, &its));
  if (!testscaledMat || norm / bnorm > 1.e-5) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Relative norm of the residual %g, Iterations %" PetscInt_FMT "\n", (double)norm / (double)bnorm, its));

  /* -------------- Stage 1: Solve Second System ---------------------- */
  /*
     Solve another linear system with the same method.  We reuse the KSP
     context, matrix and vector data structures, and hence save the
     overhead of creating new ones.

     Indicate to PETSc profiling that we're concluding the first
     stage with PetscLogStagePop(), and beginning the second stage with
     PetscLogStagePush().
  */
  PetscCall(PetscLogStagePop());
  PetscCall(PetscLogStagePush(stages[1]));

  /*
     Initialize all matrix entries to zero.  MatZeroEntries() retains the
     nonzero structure of the matrix for sparse formats.
  */
  PetscCall(MatZeroEntries(C));

  /*
     Assemble matrix again.  Note that we retain the same matrix data
     structure and the same nonzero pattern; we just change the values
     of the matrix entries.
  */
  for (i = 0; i < m; i++) {
    for (j = 2 * rank; j < 2 * rank + 2; j++) {
      v  = -1.0;
      Ii = j + n * i;
      if (i > 0) {
        J = Ii - n;
        PetscCall(MatSetValues(C, 1, &Ii, 1, &J, &v, ADD_VALUES));
      }
      if (i < m - 1) {
        J = Ii + n;
        PetscCall(MatSetValues(C, 1, &Ii, 1, &J, &v, ADD_VALUES));
      }
      if (j > 0) {
        J = Ii - 1;
        PetscCall(MatSetValues(C, 1, &Ii, 1, &J, &v, ADD_VALUES));
      }
      if (j < n - 1) {
        J = Ii + 1;
        PetscCall(MatSetValues(C, 1, &Ii, 1, &J, &v, ADD_VALUES));
      }
      v = 6.0;
      PetscCall(MatSetValues(C, 1, &Ii, 1, &Ii, &v, ADD_VALUES));
    }
  }
  if (mat_nonsymmetric) {
    for (Ii = Istart; Ii < Iend; Ii++) {
      v = -1.5;
      i = Ii / n;
      if (i > 1) {
        J = Ii - n - 1;
        PetscCall(MatSetValues(C, 1, &Ii, 1, &J, &v, ADD_VALUES));
      }
    }
  }
  PetscCall(MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY));

  if (testscaledMat) {
    PetscRandom rctx;

    /* Scale a(0,0) and a(M-1,M-1) */
    if (rank == 0) {
      v  = 6.0 * 0.00001;
      Ii = 0;
      J  = 0;
      PetscCall(MatSetValues(C, 1, &Ii, 1, &J, &v, INSERT_VALUES));
    } else if (rank == size - 1) {
      v  = 6.0 * 0.00001;
      Ii = m * n - 1;
      J  = m * n - 1;
      PetscCall(MatSetValues(C, 1, &Ii, 1, &J, &v, INSERT_VALUES));
    }
    PetscCall(MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY));

    /* Compute a new right-hand-side vector */
    PetscCall(VecDestroy(&u));
    PetscCall(VecCreate(PETSC_COMM_WORLD, &u));
    PetscCall(VecSetSizes(u, PETSC_DECIDE, m * n));
    PetscCall(VecSetFromOptions(u));

    PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rctx));
    PetscCall(PetscRandomSetFromOptions(rctx));
    PetscCall(VecSetRandom(u, rctx));
    PetscCall(PetscRandomDestroy(&rctx));
    PetscCall(VecAssemblyBegin(u));
    PetscCall(VecAssemblyEnd(u));
  }

  PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_newMat", &testnewC, NULL));
  if (testnewC) {
    /*
     User may use a new matrix C with same nonzero pattern, e.g.
      ./ex5 -ksp_monitor -mat_type sbaij -pc_type cholesky -pc_factor_mat_solver_type mumps -test_newMat
    */
    Mat Ctmp;
    PetscCall(MatDuplicate(C, MAT_COPY_VALUES, &Ctmp));
    PetscCall(MatDestroy(&C));
    PetscCall(MatDuplicate(Ctmp, MAT_COPY_VALUES, &C));
    PetscCall(MatDestroy(&Ctmp));
  }

  PetscCall(MatMult(C, u, b));

  /*
     Set operators. Here the matrix that defines the linear system
     also serves as the preconditioning matrix.
  */
  PetscCall(KSPSetOperators(ksp, C, C));

  /*
     Solve linear system
  */
  PetscCall(KSPSetUp(ksp));
  PetscCall(KSPSolve(ksp, b, x));

  /*
     Check the residual
  */
  PetscCall(VecAXPY(x, none, u));
  PetscCall(VecNorm(x, NORM_2, &norm));
  PetscCall(KSPGetIterationNumber(ksp, &its));
  if (!testscaledMat || norm / bnorm > PETSC_SMALL) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Relative norm of the residual %g, Iterations %" PetscInt_FMT "\n", (double)norm / (double)bnorm, its));

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&C));

  /*
     Indicate to PETSc profiling that we're concluding the second stage
  */
  PetscCall(PetscLogStagePop());

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      args: -pc_type jacobi -ksp_monitor_short -ksp_gmres_cgs_refinement_type refine_always

   test:
      suffix: 2
      nsize: 2
      args: -pc_type jacobi -ksp_monitor_short -ksp_gmres_cgs_refinement_type refine_always -ksp_rtol .000001

   test:
      suffix: 5
      nsize: 2
      args: -ksp_gmres_cgs_refinement_type refine_always -ksp_monitor draw::draw_lg -ksp_monitor_true_residual draw::draw_lg

   test:
      suffix: asm
      nsize: 4
      args: -pc_type asm

   test:
      suffix: asm_baij
      nsize: 4
      args: -pc_type asm -mat_type baij
      output_file: output/ex5_asm.out

   test:
      suffix: redundant_0
      args: -m 1000 -pc_type redundant -pc_redundant_number 1 -redundant_ksp_type gmres -redundant_pc_type jacobi

   test:
      suffix: redundant_1
      nsize: 5
      args: -pc_type redundant -pc_redundant_number 1 -redundant_ksp_type gmres -redundant_pc_type jacobi

   test:
      suffix: redundant_2
      nsize: 5
      args: -pc_type redundant -pc_redundant_number 3 -redundant_ksp_type gmres -redundant_pc_type jacobi

   test:
      suffix: redundant_3
      nsize: 5
      args: -pc_type redundant -pc_redundant_number 5 -redundant_ksp_type gmres -redundant_pc_type jacobi

   test:
      suffix: redundant_4
      nsize: 5
      args: -pc_type redundant -pc_redundant_number 3 -redundant_ksp_type gmres -redundant_pc_type jacobi -psubcomm_type interlaced

   test:
      suffix: superlu_dist
      nsize: 15
      requires: superlu_dist
      args: -pc_type lu -pc_factor_mat_solver_type superlu_dist -mat_superlu_dist_equil false -m 150 -mat_superlu_dist_r 3 -mat_superlu_dist_c 5 -test_scaledMat

   test:
      suffix: superlu_dist_2
      nsize: 15
      requires: superlu_dist
      args: -pc_type lu -pc_factor_mat_solver_type superlu_dist -mat_superlu_dist_equil false -m 150 -mat_superlu_dist_r 3 -mat_superlu_dist_c 5 -test_scaledMat -mat_superlu_dist_fact SamePattern_SameRowPerm
      output_file: output/ex5_superlu_dist.out

   test:
      suffix: superlu_dist_3
      nsize: 15
      requires: superlu_dist
      args: -pc_type lu -pc_factor_mat_solver_type superlu_dist -mat_superlu_dist_equil false -m 500 -mat_superlu_dist_r 3 -mat_superlu_dist_c 5 -test_scaledMat -mat_superlu_dist_fact DOFACT
      output_file: output/ex5_superlu_dist.out

   test:
      suffix: superlu_dist_0
      nsize: 1
      requires: superlu_dist
      args: -pc_type lu -pc_factor_mat_solver_type superlu_dist -test_scaledMat
      output_file: output/ex5_superlu_dist.out

TEST*/
