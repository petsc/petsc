
static char help[] = "Solves a linear system in parallel with KSP. \n\
Contributed by Jose E. Roman, SLEPc developer, for testing repeated call of KSPSetOperators(), 2014 \n\n";

#include <petscksp.h>
int main(int argc, char **args)
{
  Vec         x, b, u; /* approx solution, RHS, exact solution */
  Mat         A;       /* linear system matrix */
  KSP         ksp;     /* linear solver context */
  PetscRandom rctx;    /* random number generator context */
  PetscInt    i, j, Ii, J, Istart, Iend, m = 8, n = 7;
  PetscBool   flg = PETSC_FALSE;
  PetscScalar v;
  PC          pc;
  PetscInt    in;
  Mat         F, B;
  PetscBool   solve = PETSC_FALSE, sameA = PETSC_FALSE, setfromoptions_first = PETSC_FALSE;
#if defined(PETSC_USE_LOG)
  PetscLogStage stage;
#endif
#if !defined(PETSC_HAVE_MUMPS)
  PetscMPIInt size;
#endif

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
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

  PetscCall(MatGetOwnershipRange(A, &Istart, &Iend));

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
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(PetscLogStagePop());

  /* A is symmetric. Set symmetric flag to enable ICC/Cholesky preconditioner */
  PetscCall(MatSetOption(A, MAT_SYMMETRIC, PETSC_TRUE));

  /* Create parallel vectors. */
  PetscCall(VecCreate(PETSC_COMM_WORLD, &u));
  PetscCall(VecSetSizes(u, PETSC_DECIDE, m * n));
  PetscCall(VecSetFromOptions(u));
  PetscCall(VecDuplicate(u, &b));
  PetscCall(VecDuplicate(b, &x));

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

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /* Create linear solver context */
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));

  /* Set operators. */
  PetscCall(KSPSetOperators(ksp, A, A));

  PetscCall(KSPSetTolerances(ksp, 1.e-2 / ((m + 1) * (n + 1)), PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));

  PetscCall(PetscOptionsGetBool(NULL, NULL, "-setfromoptions_first", &setfromoptions_first, NULL));
  if (setfromoptions_first) {
    /* code path for changing from KSPLSQR to KSPREONLY */
    PetscCall(KSPSetFromOptions(ksp));
  }
  PetscCall(KSPSetType(ksp, KSPPREONLY));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCCHOLESKY));
#if defined(PETSC_HAVE_MUMPS)
  #if defined(PETSC_USE_COMPLEX)
  SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "Spectrum slicing with MUMPS is not available for complex scalars");
  #endif
  PetscCall(PCFactorSetMatSolverType(pc, MATSOLVERMUMPS));
  /*
     must use runtime option '-mat_mumps_icntl_13 1' (turn off ScaLAPACK for
     matrix inertia), currently there is no better way of setting this in program
  */
  PetscCall(PetscOptionsInsertString(NULL, "-mat_mumps_icntl_13 1"));
#else
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_SUP, "Configure with MUMPS if you want to run this example in parallel");
#endif

  if (!setfromoptions_first) {
    /* when -setfromoptions_first is true, do not call KSPSetFromOptions() again and stick to KSPPREONLY */
    PetscCall(KSPSetFromOptions(ksp));
  }

  /* get inertia */
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-solve", &solve, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-sameA", &sameA, NULL));
  PetscCall(KSPSetUp(ksp));
  PetscCall(PCFactorGetMatrix(pc, &F));
  PetscCall(MatGetInertia(F, &in, NULL, NULL));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "INERTIA=%" PetscInt_FMT "\n", in));
  if (solve) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Solving the intermediate KSP\n"));
    PetscCall(KSPSolve(ksp, b, x));
  } else PetscCall(PetscPrintf(PETSC_COMM_WORLD, "NOT Solving the intermediate KSP\n"));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &B));
  if (sameA) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Setting A\n"));
    PetscCall(MatAXPY(A, 1.1, B, DIFFERENT_NONZERO_PATTERN));
    PetscCall(KSPSetOperators(ksp, A, A));
  } else {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Setting B\n"));
    PetscCall(MatAXPY(B, 1.1, A, DIFFERENT_NONZERO_PATTERN));
    PetscCall(KSPSetOperators(ksp, B, B));
  }
  PetscCall(KSPSetUp(ksp));
  PetscCall(PCFactorGetMatrix(pc, &F));
  PetscCall(MatGetInertia(F, &in, NULL, NULL));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "INERTIA=%" PetscInt_FMT "\n", in));
  PetscCall(KSPSolve(ksp, b, x));
  PetscCall(MatDestroy(&B));

  /* Free work space.*/
  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

    build:
      requires: !complex

    test:
      args:

    test:
      suffix: 2
      args: -sameA

    test:
      suffix: 3
      args: -ksp_lsqr_monitor -ksp_type lsqr -setfromoptions_first {{0 1}separate output}

TEST*/
