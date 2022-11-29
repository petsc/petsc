
static char help[] = "The solution of 2 different linear systems with different linear solvers.\n\
Also, this example illustrates the repeated\n\
solution of linear systems, while reusing matrix, vector, and solver data\n\
structures throughout the process.  Note the various stages of event logging.\n\n";

/*
  Include "petscksp.h" so that we can use KSP solvers.  Note that this file
  automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h - vectors
     petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
*/
#include <petscksp.h>

/*
   Declare user-defined routines
*/
extern PetscErrorCode CheckError(Vec, Vec, Vec, PetscInt, PetscReal, PetscLogEvent);
extern PetscErrorCode MyKSPMonitor(KSP, PetscInt, PetscReal, void *);

int main(int argc, char **args)
{
  Vec           x1, b1, x2, b2; /* solution and RHS vectors for systems #1 and #2 */
  Vec           u;              /* exact solution vector */
  Mat           C1, C2;         /* matrices for systems #1 and #2 */
  KSP           ksp1, ksp2;     /* KSP contexts for systems #1 and #2 */
  PetscInt      ntimes = 3;     /* number of times to solve the linear systems */
  PetscLogEvent CHECK_ERROR;    /* event number for error checking */
  PetscInt      ldim, low, high, iglobal, Istart, Iend, Istart2, Iend2;
  PetscInt      Ii, J, i, j, m = 3, n = 2, its, t;
  PetscBool     flg = PETSC_FALSE, unsym = PETSC_TRUE;
  PetscScalar   v;
  PetscMPIInt   rank, size;
#if defined(PETSC_USE_LOG)
  PetscLogStage stages[3];
#endif

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-m", &m, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-t", &ntimes, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-unsym", &unsym, NULL));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  n = 2 * size;

  /*
     Register various stages for profiling
  */
  PetscCall(PetscLogStageRegister("Prelim setup", &stages[0]));
  PetscCall(PetscLogStageRegister("Linear System 1", &stages[1]));
  PetscCall(PetscLogStageRegister("Linear System 2", &stages[2]));

  /*
     Register a user-defined event for profiling (error checking).
  */
  CHECK_ERROR = 0;
  PetscCall(PetscLogEventRegister("Check Error", KSP_CLASSID, &CHECK_ERROR));

  /* - - - - - - - - - - - - Stage 0: - - - - - - - - - - - - - -
                        Preliminary Setup
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(PetscLogStagePush(stages[0]));

  /*
     Create data structures for first linear system.
      - Create parallel matrix, specifying only its global dimensions.
        When using MatCreate(), the matrix format can be specified at
        runtime. Also, the parallel partitioning of the matrix is
        determined by PETSc at runtime.
      - Create parallel vectors.
        - When using VecSetSizes(), we specify only the vector's global
          dimension; the parallel partitioning is determined at runtime.
        - Note: We form 1 vector from scratch and then duplicate as needed.
  */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &C1));
  PetscCall(MatSetSizes(C1, PETSC_DECIDE, PETSC_DECIDE, m * n, m * n));
  PetscCall(MatSetFromOptions(C1));
  PetscCall(MatSetUp(C1));
  PetscCall(MatGetOwnershipRange(C1, &Istart, &Iend));
  PetscCall(VecCreate(PETSC_COMM_WORLD, &u));
  PetscCall(VecSetSizes(u, PETSC_DECIDE, m * n));
  PetscCall(VecSetFromOptions(u));
  PetscCall(VecDuplicate(u, &b1));
  PetscCall(VecDuplicate(u, &x1));

  /*
     Create first linear solver context.
     Set runtime options (e.g., -pc_type <type>).
     Note that the first linear system uses the default option
     names, while the second linear system uses a different
     options prefix.
  */
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp1));
  PetscCall(KSPSetFromOptions(ksp1));

  /*
     Set user-defined monitoring routine for first linear system.
  */
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-my_ksp_monitor", &flg, NULL));
  if (flg) PetscCall(KSPMonitorSet(ksp1, MyKSPMonitor, NULL, 0));

  /*
     Create data structures for second linear system.
  */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &C2));
  PetscCall(MatSetSizes(C2, PETSC_DECIDE, PETSC_DECIDE, m * n, m * n));
  PetscCall(MatSetFromOptions(C2));
  PetscCall(MatSetUp(C2));
  PetscCall(MatGetOwnershipRange(C2, &Istart2, &Iend2));
  PetscCall(VecDuplicate(u, &b2));
  PetscCall(VecDuplicate(u, &x2));

  /*
     Create second linear solver context
  */
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp2));

  /*
     Set different options prefix for second linear system.
     Set runtime options (e.g., -s2_pc_type <type>)
  */
  PetscCall(KSPAppendOptionsPrefix(ksp2, "s2_"));
  PetscCall(KSPSetFromOptions(ksp2));

  /*
     Assemble exact solution vector in parallel.  Note that each
     processor needs to set only its local part of the vector.
  */
  PetscCall(VecGetLocalSize(u, &ldim));
  PetscCall(VecGetOwnershipRange(u, &low, &high));
  for (i = 0; i < ldim; i++) {
    iglobal = i + low;
    v       = (PetscScalar)(i + 100 * rank);
    PetscCall(VecSetValues(u, 1, &iglobal, &v, ADD_VALUES));
  }
  PetscCall(VecAssemblyBegin(u));
  PetscCall(VecAssemblyEnd(u));

  /*
     Log the number of flops for computing vector entries
  */
  PetscCall(PetscLogFlops(2.0 * ldim));

  /*
     End curent profiling stage
  */
  PetscCall(PetscLogStagePop());

  /* --------------------------------------------------------------
                        Linear solver loop:
      Solve 2 different linear systems several times in succession
     -------------------------------------------------------------- */

  for (t = 0; t < ntimes; t++) {
    /* - - - - - - - - - - - - Stage 1: - - - - - - - - - - - - - -
                 Assemble and solve first linear system
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    /*
       Begin profiling stage #1
    */
    PetscCall(PetscLogStagePush(stages[1]));

    /*
       Initialize all matrix entries to zero.  MatZeroEntries() retains
       the nonzero structure of the matrix for sparse formats.
    */
    if (t > 0) PetscCall(MatZeroEntries(C1));

    /*
       Set matrix entries in parallel.  Also, log the number of flops
       for computing matrix entries.
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
        PetscCall(MatSetValues(C1, 1, &Ii, 1, &J, &v, ADD_VALUES));
      }
      if (i < m - 1) {
        J = Ii + n;
        PetscCall(MatSetValues(C1, 1, &Ii, 1, &J, &v, ADD_VALUES));
      }
      if (j > 0) {
        J = Ii - 1;
        PetscCall(MatSetValues(C1, 1, &Ii, 1, &J, &v, ADD_VALUES));
      }
      if (j < n - 1) {
        J = Ii + 1;
        PetscCall(MatSetValues(C1, 1, &Ii, 1, &J, &v, ADD_VALUES));
      }
      v = 4.0;
      PetscCall(MatSetValues(C1, 1, &Ii, 1, &Ii, &v, ADD_VALUES));
    }
    if (unsym) {
      for (Ii = Istart; Ii < Iend; Ii++) { /* Make matrix nonsymmetric */
        v = -1.0 * (t + 0.5);
        i = Ii / n;
        if (i > 0) {
          J = Ii - n;
          PetscCall(MatSetValues(C1, 1, &Ii, 1, &J, &v, ADD_VALUES));
        }
      }
      PetscCall(PetscLogFlops(2.0 * (Iend - Istart)));
    }

    /*
       Assemble matrix, using the 2-step process:
         MatAssemblyBegin(), MatAssemblyEnd()
       Computations can be done while messages are in transition
       by placing code between these two statements.
    */
    PetscCall(MatAssemblyBegin(C1, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(C1, MAT_FINAL_ASSEMBLY));

    /*
       Indicate same nonzero structure of successive linear system matrices
    */
    PetscCall(MatSetOption(C1, MAT_NEW_NONZERO_LOCATIONS, PETSC_TRUE));

    /*
       Compute right-hand-side vector
    */
    PetscCall(MatMult(C1, u, b1));

    /*
       Set operators. Here the matrix that defines the linear system
       also serves as the preconditioning matrix.
    */
    PetscCall(KSPSetOperators(ksp1, C1, C1));

    /*
       Use the previous solution of linear system #1 as the initial
       guess for the next solve of linear system #1.  The user MUST
       call KSPSetInitialGuessNonzero() in indicate use of an initial
       guess vector; otherwise, an initial guess of zero is used.
    */
    if (t > 0) PetscCall(KSPSetInitialGuessNonzero(ksp1, PETSC_TRUE));

    /*
       Solve the first linear system.  Here we explicitly call
       KSPSetUp() for more detailed performance monitoring of
       certain preconditioners, such as ICC and ILU.  This call
       is optional, ase KSPSetUp() will automatically be called
       within KSPSolve() if it hasn't been called already.
    */
    PetscCall(KSPSetUp(ksp1));
    PetscCall(KSPSolve(ksp1, b1, x1));
    PetscCall(KSPGetIterationNumber(ksp1, &its));

    /*
       Check error of solution to first linear system
    */
    PetscCall(CheckError(u, x1, b1, its, 1.e-4, CHECK_ERROR));

    /* - - - - - - - - - - - - Stage 2: - - - - - - - - - - - - - -
                 Assemble and solve second linear system
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    /*
       Conclude profiling stage #1; begin profiling stage #2
    */
    PetscCall(PetscLogStagePop());
    PetscCall(PetscLogStagePush(stages[2]));

    /*
       Initialize all matrix entries to zero
    */
    if (t > 0) PetscCall(MatZeroEntries(C2));

    /*
       Assemble matrix in parallel. Also, log the number of flops
       for computing matrix entries.
        - To illustrate the features of parallel matrix assembly, we
          intentionally set the values differently from the way in
          which the matrix is distributed across the processors.  Each
          entry that is not owned locally will be sent to the appropriate
          processor during MatAssemblyBegin() and MatAssemblyEnd().
        - For best efficiency the user should strive to set as many
          entries locally as possible.
     */
    for (i = 0; i < m; i++) {
      for (j = 2 * rank; j < 2 * rank + 2; j++) {
        v  = -1.0;
        Ii = j + n * i;
        if (i > 0) {
          J = Ii - n;
          PetscCall(MatSetValues(C2, 1, &Ii, 1, &J, &v, ADD_VALUES));
        }
        if (i < m - 1) {
          J = Ii + n;
          PetscCall(MatSetValues(C2, 1, &Ii, 1, &J, &v, ADD_VALUES));
        }
        if (j > 0) {
          J = Ii - 1;
          PetscCall(MatSetValues(C2, 1, &Ii, 1, &J, &v, ADD_VALUES));
        }
        if (j < n - 1) {
          J = Ii + 1;
          PetscCall(MatSetValues(C2, 1, &Ii, 1, &J, &v, ADD_VALUES));
        }
        v = 6.0 + t * 0.5;
        PetscCall(MatSetValues(C2, 1, &Ii, 1, &Ii, &v, ADD_VALUES));
      }
    }
    if (unsym) {
      for (Ii = Istart2; Ii < Iend2; Ii++) { /* Make matrix nonsymmetric */
        v = -1.0 * (t + 0.5);
        i = Ii / n;
        if (i > 0) {
          J = Ii - n;
          PetscCall(MatSetValues(C2, 1, &Ii, 1, &J, &v, ADD_VALUES));
        }
      }
    }
    PetscCall(MatAssemblyBegin(C2, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(C2, MAT_FINAL_ASSEMBLY));
    PetscCall(PetscLogFlops(2.0 * (Iend - Istart)));

    /*
       Indicate same nonzero structure of successive linear system matrices
    */
    PetscCall(MatSetOption(C2, MAT_NEW_NONZERO_LOCATIONS, PETSC_FALSE));

    /*
       Compute right-hand-side vector
    */
    PetscCall(MatMult(C2, u, b2));

    /*
       Set operators. Here the matrix that defines the linear system
       also serves as the preconditioning matrix.  Indicate same nonzero
       structure of successive preconditioner matrices by setting flag
       SAME_NONZERO_PATTERN.
    */
    PetscCall(KSPSetOperators(ksp2, C2, C2));

    /*
       Solve the second linear system
    */
    PetscCall(KSPSetUp(ksp2));
    PetscCall(KSPSolve(ksp2, b2, x2));
    PetscCall(KSPGetIterationNumber(ksp2, &its));

    /*
       Check error of solution to second linear system
    */
    PetscCall(CheckError(u, x2, b2, its, 1.e-4, CHECK_ERROR));

    /*
       Conclude profiling stage #2
    */
    PetscCall(PetscLogStagePop());
  }
  /* --------------------------------------------------------------
                       End of linear solver loop
     -------------------------------------------------------------- */

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  PetscCall(KSPDestroy(&ksp1));
  PetscCall(KSPDestroy(&ksp2));
  PetscCall(VecDestroy(&x1));
  PetscCall(VecDestroy(&x2));
  PetscCall(VecDestroy(&b1));
  PetscCall(VecDestroy(&b2));
  PetscCall(MatDestroy(&C1));
  PetscCall(MatDestroy(&C2));
  PetscCall(VecDestroy(&u));

  PetscCall(PetscFinalize());
  return 0;
}
/* ------------------------------------------------------------- */
/*
    CheckError - Checks the error of the solution.

    Input Parameters:
    u - exact solution
    x - approximate solution
    b - work vector
    its - number of iterations for convergence
    tol - tolerance
    CHECK_ERROR - the event number for error checking
                  (for use with profiling)

    Notes:
    In order to profile this section of code separately from the
    rest of the program, we register it as an "event" with
    PetscLogEventRegister() in the main program.  Then, we indicate
    the start and end of this event by respectively calling
        PetscLogEventBegin(CHECK_ERROR,u,x,b,0);
        PetscLogEventEnd(CHECK_ERROR,u,x,b,0);
    Here, we specify the objects most closely associated with
    the event (the vectors u,x,b).  Such information is optional;
    we could instead just use 0 instead for all objects.
*/
PetscErrorCode CheckError(Vec u, Vec x, Vec b, PetscInt its, PetscReal tol, PetscLogEvent CHECK_ERROR)
{
  PetscScalar none = -1.0;
  PetscReal   norm;

  PetscCall(PetscLogEventBegin(CHECK_ERROR, u, x, b, 0));

  /*
     Compute error of the solution, using b as a work vector.
  */
  PetscCall(VecCopy(x, b));
  PetscCall(VecAXPY(b, none, u));
  PetscCall(VecNorm(b, NORM_2, &norm));
  if (norm > tol) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Norm of error %g, Iterations %" PetscInt_FMT "\n", (double)norm, its));
  PetscCall(PetscLogEventEnd(CHECK_ERROR, u, x, b, 0));
  return 0;
}
/* ------------------------------------------------------------- */
/*
   MyKSPMonitor - This is a user-defined routine for monitoring
   the KSP iterative solvers.

   Input Parameters:
     ksp   - iterative context
     n     - iteration number
     rnorm - 2-norm (preconditioned) residual value (may be estimated)
     dummy - optional user-defined monitor context (unused here)
*/
PetscErrorCode MyKSPMonitor(KSP ksp, PetscInt n, PetscReal rnorm, void *dummy)
{
  Vec x;

  /*
     Build the solution vector
  */
  PetscCall(KSPBuildSolution(ksp, NULL, &x));

  /*
     Write the solution vector and residual norm to stdout.
      - PetscPrintf() handles output for multiprocessor jobs
        by printing from only one processor in the communicator.
      - The parallel viewer PETSC_VIEWER_STDOUT_WORLD handles
        data from multiple processors so that the output
        is not jumbled.
  */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "iteration %" PetscInt_FMT " solution vector:\n", n));
  PetscCall(VecView(x, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "iteration %" PetscInt_FMT " KSP Residual norm %14.12e \n", n, (double)rnorm));
  return 0;
}

/*TEST

   test:
      args: -t 2 -pc_type jacobi -ksp_monitor_short -ksp_type gmres -ksp_gmres_cgs_refinement_type refine_always -s2_ksp_type bcgs -s2_pc_type jacobi -s2_ksp_monitor_short

   test:
      requires: hpddm
      suffix: hpddm
      args: -t 2 -pc_type jacobi -ksp_monitor_short -ksp_type {{gmres hpddm}} -s2_ksp_type {{gmres hpddm}} -s2_pc_type jacobi -s2_ksp_monitor_short

   test:
      requires: hpddm
      suffix: hpddm_2
      args: -t 2 -pc_type jacobi -ksp_monitor_short -ksp_type gmres -s2_ksp_type hpddm -s2_ksp_hpddm_type gcrodr -s2_ksp_hpddm_recycle 10 -s2_pc_type jacobi -s2_ksp_monitor_short

   testset:
      requires: hpddm
      output_file: output/ex9_hpddm_cg.out
      args: -unsym 0 -t 2 -pc_type jacobi -ksp_monitor_short -s2_pc_type jacobi -s2_ksp_monitor_short -ksp_rtol 1.e-2 -s2_ksp_rtol 1.e-2
      test:
         suffix: hpddm_cg_p_p
         args: -ksp_type cg -s2_ksp_type cg
      test:
         suffix: hpddm_cg_p_h
         args: -ksp_type cg -s2_ksp_type hpddm -s2_ksp_hpddm_type {{cg bcg bfbcg}shared output}
      test:
         suffix: hpddm_cg_h_h
         args: -ksp_type hpddm -ksp_hpddm_type cg -s2_ksp_type hpddm -s2_ksp_hpddm_type {{cg bcg bfbcg}shared output}

TEST*/
