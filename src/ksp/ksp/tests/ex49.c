
static char help[] = "Tests SeqSBAIJ factorizations for different block sizes\n\n";

#include <petscksp.h>

int main(int argc, char **args)
{
  Vec         x, b, u;
  Mat         A, A2;
  KSP         ksp;
  PetscRandom rctx;
  PetscReal   norm;
  PetscInt    i, j, k, l, n = 27, its, bs = 2, Ii, J;
  PetscBool   test_hermitian = PETSC_FALSE, convert = PETSC_FALSE;
  PetscScalar v;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-bs", &bs, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-herm", &test_hermitian, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-conv", &convert, NULL));

  PetscCall(MatCreate(PETSC_COMM_SELF, &A));
  PetscCall(MatSetSizes(A, n * bs, n * bs, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(MatSetBlockSize(A, bs));
  PetscCall(MatSetType(A, MATSEQSBAIJ));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSeqSBAIJSetPreallocation(A, bs, n, NULL));
  PetscCall(MatSeqBAIJSetPreallocation(A, bs, n, NULL));
  PetscCall(MatSeqAIJSetPreallocation(A, n * bs, NULL));
  PetscCall(MatMPIAIJSetPreallocation(A, n * bs, NULL, n * bs, NULL));

  PetscCall(PetscRandomCreate(PETSC_COMM_SELF, &rctx));
  for (i = 0; i < n; i++) {
    for (j = i; j < n; j++) {
      PetscCall(PetscRandomGetValue(rctx, &v));
      if (PetscRealPart(v) < .1 || i == j) {
        for (k = 0; k < bs; k++) {
          for (l = 0; l < bs; l++) {
            Ii = i * bs + k;
            J  = j * bs + l;
            PetscCall(PetscRandomGetValue(rctx, &v));
            if (Ii == J) v = PetscRealPart(v + 3 * n * bs);
            PetscCall(MatSetValue(A, Ii, J, v, INSERT_VALUES));
            if (test_hermitian) {
              PetscCall(MatSetValue(A, J, Ii, PetscConj(v), INSERT_VALUES));
            } else {
              PetscCall(MatSetValue(A, J, Ii, v, INSERT_VALUES));
            }
          }
        }
      }
    }
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  /* With complex numbers:
     - PETSc cholesky does not support hermitian matrices
     - CHOLMOD only supports hermitian matrices
     - SUPERLU_DIST seems supporting both
  */
  if (test_hermitian) PetscCall(MatSetOption(A, MAT_HERMITIAN, PETSC_TRUE));

  {
    Mat M;
    PetscCall(MatComputeOperator(A, MATAIJ, &M));
    PetscCall(MatViewFromOptions(M, NULL, "-expl_view"));
    PetscCall(MatDestroy(&M));
  }

  A2 = NULL;
  if (convert) PetscCall(MatConvert(A, MATAIJ, MAT_INITIAL_MATRIX, &A2));

  PetscCall(VecCreate(PETSC_COMM_SELF, &u));
  PetscCall(VecSetSizes(u, PETSC_DECIDE, n * bs));
  PetscCall(VecSetFromOptions(u));
  PetscCall(VecDuplicate(u, &b));
  PetscCall(VecDuplicate(b, &x));

  PetscCall(VecSet(u, 1.0));
  PetscCall(MatMult(A, u, b));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create linear solver context
  */
  PetscCall(KSPCreate(PETSC_COMM_SELF, &ksp));

  /*
     Set operators.
  */
  PetscCall(KSPSetOperators(ksp, A2 ? A2 : A, A));

  PetscCall(KSPSetFromOptions(ksp));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(KSPSolve(ksp, b, x));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Check solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Check the error
  */
  PetscCall(VecAXPY(x, -1.0, u));
  PetscCall(VecNorm(x, NORM_2, &norm));
  PetscCall(KSPGetIterationNumber(ksp, &its));

  /*
     Print convergence information.  PetscPrintf() produces a single
     print statement from all processes that share a communicator.
     An alternative is PetscFPrintf(), which prints to a file.
  */
  if (norm > 100 * PETSC_SMALL) PetscCall(PetscPrintf(PETSC_COMM_SELF, "Norm of residual %g iterations %" PetscInt_FMT " bs %" PetscInt_FMT "\n", (double)norm, its, bs));

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&A2));
  PetscCall(PetscRandomDestroy(&rctx));

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
      args: -mat_type {{aij baij sbaij}} -bs {{1 2 3 4 5 6 7 8 9 10 11 12}} -pc_type cholesky -herm 0 -conv {{0 1}}

   test:
      nsize: {{1 4}}
      suffix: cholmod
      requires: suitesparse
      args: -mat_type {{aij sbaij}} -bs 1 -pc_type cholesky -pc_factor_mat_solver_type cholmod -herm -conv {{0 1}}

   test:
      nsize: {{1 4}}
      suffix: superlu_dist
      requires: superlu_dist
      output_file: output/ex49_cholmod.out
      args: -mat_type mpiaij -bs 3 -pc_type cholesky -pc_factor_mat_solver_type superlu_dist -herm {{0 1}} -conv {{0 1}}

   test:
      suffix: mkl_pardiso
      requires: mkl_pardiso
      output_file: output/ex49_1.out
      args: -bs {{1 3}} -pc_type cholesky -pc_factor_mat_solver_type mkl_pardiso

   test:
      suffix: cg
      requires: complex
      output_file: output/ex49_cg.out
      args: -herm -ksp_cg_type hermitian -mat_type aij -ksp_type cg -pc_type jacobi -ksp_rtol 4e-07

   test:
      suffix: pipecg2
      requires: complex
      output_file: output/ex49_pipecg2.out
      args: -herm -mat_type aij -ksp_type pipecg2 -pc_type jacobi -ksp_rtol 4e-07 -ksp_norm_type {{preconditioned unpreconditioned natural}}

TEST*/
