
static char help[] = "Solves a tridiagonal linear system with KSP. \n\
                      Modified from ex1.c to illustrate reuse of preconditioner \n\
                      Written as requested by [petsc-maint #63875] \n\n";

#include <petscksp.h>

int main(int argc, char **args)
{
  Vec         x, x2, b, u;                              /* approx solution, RHS, exact solution */
  Mat         A;                                        /* linear system matrix */
  KSP         ksp;                                      /* linear solver context */
  PC          pc;                                       /* preconditioner context */
  PetscReal   norm, tol = 100. * PETSC_MACHINE_EPSILON; /* norm of solution error */
  PetscInt    i, n      = 10, col[3], its;
  PetscMPIInt rank, size;
  PetscScalar one = 1.0, value[3];

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));

  /* Create vectors.*/
  PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
  PetscCall(PetscObjectSetName((PetscObject)x, "Solution"));
  PetscCall(VecSetSizes(x, PETSC_DECIDE, n));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecDuplicate(x, &b));
  PetscCall(VecDuplicate(x, &u));
  PetscCall(VecDuplicate(x, &x2));

  /* Create matrix. Only proc[0] sets values - not efficient for parallel processing!
     See ex23.c for efficient parallel assembly matrix */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));

  if (rank == 0) {
    value[0] = -1.0;
    value[1] = 2.0;
    value[2] = -1.0;
    for (i = 1; i < n - 1; i++) {
      col[0] = i - 1;
      col[1] = i;
      col[2] = i + 1;
      PetscCall(MatSetValues(A, 1, &i, 3, col, value, INSERT_VALUES));
    }
    i      = n - 1;
    col[0] = n - 2;
    col[1] = n - 1;
    PetscCall(MatSetValues(A, 1, &i, 2, col, value, INSERT_VALUES));
    i        = 0;
    col[0]   = 0;
    col[1]   = 1;
    value[0] = 2.0;
    value[1] = -1.0;
    PetscCall(MatSetValues(A, 1, &i, 2, col, value, INSERT_VALUES));

    i        = 0;
    col[0]   = n - 1;
    value[0] = 0.5; /* make A non-symmetric */
    PetscCall(MatSetValues(A, 1, &i, 1, col, value, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  /* Set exact solution */
  PetscCall(VecSet(u, one));

  /* Create linear solver context */
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCLU));
#if defined(PETSC_HAVE_MUMPS)
  if (size > 1) PetscCall(PCFactorSetMatSolverType(pc, MATSOLVERMUMPS));
#endif
  PetscCall(KSPSetFromOptions(ksp));

  /* 1. Solve linear system A x = b */
  PetscCall(MatMult(A, u, b));
  PetscCall(KSPSolve(ksp, b, x));

  /* Check the error */
  PetscCall(VecAXPY(x, -1.0, u));
  PetscCall(VecNorm(x, NORM_2, &norm));
  PetscCall(KSPGetIterationNumber(ksp, &its));
  if (norm > tol) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "1. Norm of error for Ax=b: %g, Iterations %" PetscInt_FMT "\n", (double)norm, its));

  /* 2. Solve linear system A^T x = b*/
  PetscCall(MatMultTranspose(A, u, b));
  PetscCall(KSPSolveTranspose(ksp, b, x2));

  /* Check the error */
  PetscCall(VecAXPY(x2, -1.0, u));
  PetscCall(VecNorm(x2, NORM_2, &norm));
  PetscCall(KSPGetIterationNumber(ksp, &its));
  if (norm > tol) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "2. Norm of error for A^T x=b: %g, Iterations %" PetscInt_FMT "\n", (double)norm, its));

  /* 3. Change A and solve A x = b with an iterative solver using A=LU as a preconditioner*/
  if (rank == 0) {
    i        = 0;
    col[0]   = n - 1;
    value[0] = 1.e-2;
    PetscCall(MatSetValues(A, 1, &i, 1, col, value, ADD_VALUES));
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  PetscCall(MatMult(A, u, b));
  PetscCall(KSPSolve(ksp, b, x));

  /* Check the error */
  PetscCall(VecAXPY(x, -1.0, u));
  PetscCall(VecNorm(x, NORM_2, &norm));
  PetscCall(KSPGetIterationNumber(ksp, &its));
  if (norm > tol) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "3. Norm of error for (A+Delta) x=b: %g, Iterations %" PetscInt_FMT "\n", (double)norm, its));

  /* Free work space. */
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&x2));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));
  PetscCall(KSPDestroy(&ksp));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      requires: mumps

   test:
      suffix: 2
      nsize: 2
      requires: mumps
      output_file: output/ex53.out

TEST*/
