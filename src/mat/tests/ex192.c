static char help[] = "Tests MatSolve() and MatMatSolve() with MUMPS or MKL_PARDISO sequential solvers in Schur complement mode.\n\
Example: mpiexec -n 1 ./ex192 -f <matrix binary file> -nrhs 4 -symmetric_solve -hermitian_solve -schur_ratio 0.3\n\n";

#include <petscmat.h>

int main(int argc, char **args)
{
  Mat         A, RHS, C, F, X, S;
  Vec         u, x, b;
  Vec         xschur, bschur, uschur;
  IS          is_schur;
  PetscMPIInt size;
  PetscInt    isolver = 0, size_schur, m, n, nfact, nsolve, nrhs;
  PetscReal   norm, tol = PETSC_SQRT_MACHINE_EPSILON;
  PetscRandom rand;
  PetscBool   data_provided, herm, symm, use_lu, cuda = PETSC_FALSE;
  PetscReal   sratio = 5.1 / 12.;
  PetscViewer fd; /* viewer */
  char        solver[256];
  char        file[PETSC_MAX_PATH_LEN]; /* input file name */

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This is a uniprocessor test");
  /* Determine which type of solver we want to test for */
  herm = PETSC_FALSE;
  symm = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-symmetric_solve", &symm, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-hermitian_solve", &herm, NULL));
  if (herm) symm = PETSC_TRUE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-cuda_solve", &cuda, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-tol", &tol, NULL));

  /* Determine file from which we read the matrix A */
  PetscCall(PetscOptionsGetString(NULL, NULL, "-f", file, sizeof(file), &data_provided));
  if (!data_provided) { /* get matrices from PETSc distribution */
    PetscCall(PetscStrncpy(file, "${PETSC_DIR}/share/petsc/datafiles/matrices/", sizeof(file)));
    if (symm) {
#if defined(PETSC_USE_COMPLEX)
      PetscCall(PetscStrlcat(file, "hpd-complex-", sizeof(file)));
#else
      PetscCall(PetscStrlcat(file, "spd-real-", sizeof(file)));
#endif
    } else {
#if defined(PETSC_USE_COMPLEX)
      PetscCall(PetscStrlcat(file, "nh-complex-", sizeof(file)));
#else
      PetscCall(PetscStrlcat(file, "ns-real-", sizeof(file)));
#endif
    }
#if defined(PETSC_USE_64BIT_INDICES)
    PetscCall(PetscStrlcat(file, "int64-", sizeof(file)));
#else
    PetscCall(PetscStrlcat(file, "int32-", sizeof(file)));
#endif
#if defined(PETSC_USE_REAL_SINGLE)
    PetscCall(PetscStrlcat(file, "float32", sizeof(file)));
#else
    PetscCall(PetscStrlcat(file, "float64", sizeof(file)));
#endif
  }
  /* Load matrix A */
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, file, FILE_MODE_READ, &fd));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatLoad(A, fd));
  PetscCall(PetscViewerDestroy(&fd));
  PetscCall(MatGetSize(A, &m, &n));
  PetscCheck(m == n, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "This example is not intended for rectangular matrices (%" PetscInt_FMT ", %" PetscInt_FMT ")", m, n);

  /* Create dense matrix C and X; C holds true solution with identical columns */
  nrhs = 2;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nrhs", &nrhs, NULL));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &C));
  PetscCall(MatSetSizes(C, m, PETSC_DECIDE, PETSC_DECIDE, nrhs));
  PetscCall(MatSetType(C, MATDENSE));
  PetscCall(MatSetFromOptions(C));
  PetscCall(MatSetUp(C));

  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rand));
  PetscCall(PetscRandomSetFromOptions(rand));
  PetscCall(MatSetRandom(C, rand));
  PetscCall(MatDuplicate(C, MAT_DO_NOT_COPY_VALUES, &X));

  /* Create vectors */
  PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
  PetscCall(VecSetSizes(x, n, PETSC_DECIDE));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecDuplicate(x, &b));
  PetscCall(VecDuplicate(x, &u)); /* save the true solution */

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-solver", &isolver, NULL));
  switch (isolver) {
#if defined(PETSC_HAVE_MUMPS)
  case 0:
    PetscCall(PetscStrcpy(solver, MATSOLVERMUMPS));
    break;
#endif
#if defined(PETSC_HAVE_MKL_PARDISO)
  case 1:
    PetscCall(PetscStrcpy(solver, MATSOLVERMKL_PARDISO));
    break;
#endif
  default:
    PetscCall(PetscStrcpy(solver, MATSOLVERPETSC));
    break;
  }

#if defined(PETSC_USE_COMPLEX)
  if (isolver == 0 && symm && !data_provided) { /* MUMPS (5.0.0) does not have support for hermitian matrices, so make them symmetric */
    PetscScalar im  = PetscSqrtScalar((PetscScalar)-1.);
    PetscScalar val = -1.0;
    val             = val + im;
    PetscCall(MatSetValue(A, 1, 0, val, INSERT_VALUES));
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  }
#endif

  PetscCall(PetscOptionsGetReal(NULL, NULL, "-schur_ratio", &sratio, NULL));
  PetscCheck(sratio >= 0. && sratio <= 1., PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Invalid ratio for schur degrees of freedom %g", (double)sratio);
  size_schur = (PetscInt)(sratio * m);

  PetscCall(PetscPrintf(PETSC_COMM_SELF, "Solving with %s: nrhs %" PetscInt_FMT ", sym %d, herm %d, size schur %" PetscInt_FMT ", size mat %" PetscInt_FMT "\n", solver, nrhs, symm, herm, size_schur, m));

  /* Test LU/Cholesky Factorization */
  use_lu = PETSC_FALSE;
  if (!symm) use_lu = PETSC_TRUE;
#if defined(PETSC_USE_COMPLEX)
  if (isolver == 1) use_lu = PETSC_TRUE;
#endif
  if (cuda && symm && !herm) use_lu = PETSC_TRUE;

  if (herm && !use_lu) { /* test also conversion routines inside the solver packages */
    PetscCall(MatSetOption(A, MAT_SYMMETRIC, PETSC_TRUE));
    PetscCall(MatConvert(A, MATSEQSBAIJ, MAT_INPLACE_MATRIX, &A));
  }

  if (use_lu) {
    PetscCall(MatGetFactor(A, solver, MAT_FACTOR_LU, &F));
  } else {
    if (herm) {
      PetscCall(MatSetOption(A, MAT_SPD, PETSC_TRUE));
    } else {
      PetscCall(MatSetOption(A, MAT_SYMMETRIC, PETSC_TRUE));
      PetscCall(MatSetOption(A, MAT_SPD, PETSC_FALSE));
    }
    PetscCall(MatGetFactor(A, solver, MAT_FACTOR_CHOLESKY, &F));
  }
  PetscCall(ISCreateStride(PETSC_COMM_SELF, size_schur, m - size_schur, 1, &is_schur));
  PetscCall(MatFactorSetSchurIS(F, is_schur));

  PetscCall(ISDestroy(&is_schur));
  if (use_lu) {
    PetscCall(MatLUFactorSymbolic(F, A, NULL, NULL, NULL));
  } else {
    PetscCall(MatCholeskyFactorSymbolic(F, A, NULL, NULL));
  }

  for (nfact = 0; nfact < 3; nfact++) {
    Mat AD;

    if (!nfact) {
      PetscCall(VecSetRandom(x, rand));
      if (symm && herm) PetscCall(VecAbs(x));
      PetscCall(MatDiagonalSet(A, x, ADD_VALUES));
    }
    if (use_lu) {
      PetscCall(MatLUFactorNumeric(F, A, NULL));
    } else {
      PetscCall(MatCholeskyFactorNumeric(F, A, NULL));
    }
    if (cuda) {
      PetscCall(MatFactorGetSchurComplement(F, &S, NULL));
      PetscCall(MatSetType(S, MATSEQDENSECUDA));
      PetscCall(MatCreateVecs(S, &xschur, &bschur));
      PetscCall(MatFactorRestoreSchurComplement(F, &S, MAT_FACTOR_SCHUR_UNFACTORED));
    }
    PetscCall(MatFactorCreateSchurComplement(F, &S, NULL));
    if (!cuda) PetscCall(MatCreateVecs(S, &xschur, &bschur));
    PetscCall(VecDuplicate(xschur, &uschur));
    if (nfact == 1 && (!cuda || (herm && symm))) PetscCall(MatFactorInvertSchurComplement(F));
    for (nsolve = 0; nsolve < 2; nsolve++) {
      PetscCall(VecSetRandom(x, rand));
      PetscCall(VecCopy(x, u));

      if (nsolve) {
        PetscCall(MatMult(A, x, b));
        PetscCall(MatSolve(F, b, x));
      } else {
        PetscCall(MatMultTranspose(A, x, b));
        PetscCall(MatSolveTranspose(F, b, x));
      }
      /* Check the error */
      PetscCall(VecAXPY(u, -1.0, x)); /* u <- (-1.0)x + u */
      PetscCall(VecNorm(u, NORM_2, &norm));
      if (norm > tol) {
        PetscReal resi;
        if (nsolve) {
          PetscCall(MatMult(A, x, u)); /* u = A*x */
        } else {
          PetscCall(MatMultTranspose(A, x, u)); /* u = A*x */
        }
        PetscCall(VecAXPY(u, -1.0, b)); /* u <- (-1.0)b + u */
        PetscCall(VecNorm(u, NORM_2, &resi));
        if (nsolve) {
          PetscCall(PetscPrintf(PETSC_COMM_SELF, "(f %" PetscInt_FMT ", s %" PetscInt_FMT ") MatSolve error: Norm of error %g, residual %g\n", nfact, nsolve, (double)norm, (double)resi));
        } else {
          PetscCall(PetscPrintf(PETSC_COMM_SELF, "(f %" PetscInt_FMT ", s %" PetscInt_FMT ") MatSolveTranspose error: Norm of error %g, residual %f\n", nfact, nsolve, (double)norm, (double)resi));
        }
      }
      PetscCall(VecSetRandom(xschur, rand));
      PetscCall(VecCopy(xschur, uschur));
      if (nsolve) {
        PetscCall(MatMult(S, xschur, bschur));
        PetscCall(MatFactorSolveSchurComplement(F, bschur, xschur));
      } else {
        PetscCall(MatMultTranspose(S, xschur, bschur));
        PetscCall(MatFactorSolveSchurComplementTranspose(F, bschur, xschur));
      }
      /* Check the error */
      PetscCall(VecAXPY(uschur, -1.0, xschur)); /* u <- (-1.0)x + u */
      PetscCall(VecNorm(uschur, NORM_2, &norm));
      if (norm > tol) {
        PetscReal resi;
        if (nsolve) {
          PetscCall(MatMult(S, xschur, uschur)); /* u = A*x */
        } else {
          PetscCall(MatMultTranspose(S, xschur, uschur)); /* u = A*x */
        }
        PetscCall(VecAXPY(uschur, -1.0, bschur)); /* u <- (-1.0)b + u */
        PetscCall(VecNorm(uschur, NORM_2, &resi));
        if (nsolve) {
          PetscCall(PetscPrintf(PETSC_COMM_SELF, "(f %" PetscInt_FMT ", s %" PetscInt_FMT ") MatFactorSolveSchurComplement error: Norm of error %g, residual %g\n", nfact, nsolve, (double)norm, (double)resi));
        } else {
          PetscCall(PetscPrintf(PETSC_COMM_SELF, "(f %" PetscInt_FMT ", s %" PetscInt_FMT ") MatFactorSolveSchurComplementTranspose error: Norm of error %g, residual %f\n", nfact, nsolve, (double)norm, (double)resi));
        }
      }
    }
    PetscCall(MatConvert(A, MATSEQAIJ, MAT_INITIAL_MATRIX, &AD));
    if (!nfact) {
      PetscCall(MatMatMult(AD, C, MAT_INITIAL_MATRIX, 2.0, &RHS));
    } else {
      PetscCall(MatMatMult(AD, C, MAT_REUSE_MATRIX, 2.0, &RHS));
    }
    PetscCall(MatDestroy(&AD));
    for (nsolve = 0; nsolve < 2; nsolve++) {
      PetscCall(MatMatSolve(F, RHS, X));

      /* Check the error */
      PetscCall(MatAXPY(X, -1.0, C, SAME_NONZERO_PATTERN));
      PetscCall(MatNorm(X, NORM_FROBENIUS, &norm));
      if (norm > tol) PetscCall(PetscPrintf(PETSC_COMM_SELF, "(f %" PetscInt_FMT ", s %" PetscInt_FMT ") MatMatSolve: Norm of error %g\n", nfact, nsolve, (double)norm));
    }
    if (isolver == 0) {
      Mat spRHS, spRHST, RHST;

      PetscCall(MatTranspose(RHS, MAT_INITIAL_MATRIX, &RHST));
      PetscCall(MatConvert(RHST, MATSEQAIJ, MAT_INITIAL_MATRIX, &spRHST));
      PetscCall(MatCreateTranspose(spRHST, &spRHS));
      for (nsolve = 0; nsolve < 2; nsolve++) {
        PetscCall(MatMatSolve(F, spRHS, X));

        /* Check the error */
        PetscCall(MatAXPY(X, -1.0, C, SAME_NONZERO_PATTERN));
        PetscCall(MatNorm(X, NORM_FROBENIUS, &norm));
        if (norm > tol) PetscCall(PetscPrintf(PETSC_COMM_SELF, "(f %" PetscInt_FMT ", s %" PetscInt_FMT ") sparse MatMatSolve: Norm of error %g\n", nfact, nsolve, (double)norm));
      }
      PetscCall(MatDestroy(&spRHST));
      PetscCall(MatDestroy(&spRHS));
      PetscCall(MatDestroy(&RHST));
    }
    PetscCall(MatDestroy(&S));
    PetscCall(VecDestroy(&xschur));
    PetscCall(VecDestroy(&bschur));
    PetscCall(VecDestroy(&uschur));
  }
  /* Free data structures */
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&C));
  PetscCall(MatDestroy(&F));
  PetscCall(MatDestroy(&X));
  PetscCall(MatDestroy(&RHS));
  PetscCall(PetscRandomDestroy(&rand));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&u));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   testset:
     requires: mkl_pardiso double !complex
     args: -solver 1

     test:
       suffix: mkl_pardiso
     test:
       requires: cuda
       suffix: mkl_pardiso_cuda
       args: -cuda_solve
       output_file: output/ex192_mkl_pardiso.out
     test:
       suffix: mkl_pardiso_1
       args: -symmetric_solve
       output_file: output/ex192_mkl_pardiso_1.out
     test:
       requires: cuda
       suffix: mkl_pardiso_cuda_1
       args: -symmetric_solve -cuda_solve
       output_file: output/ex192_mkl_pardiso_1.out
     test:
       suffix: mkl_pardiso_3
       args: -symmetric_solve -hermitian_solve
       output_file: output/ex192_mkl_pardiso_3.out
     test:
       requires: cuda defined(PETSC_HAVE_CUSOLVERDNDPOTRI)
       suffix: mkl_pardiso_cuda_3
       args: -symmetric_solve -hermitian_solve -cuda_solve
       output_file: output/ex192_mkl_pardiso_3.out

   testset:
     requires: mumps double !complex
     args: -solver 0

     test:
       suffix: mumps
     test:
       requires: cuda
       suffix: mumps_cuda
       args: -cuda_solve
       output_file: output/ex192_mumps.out
     test:
       suffix: mumps_2
       args: -symmetric_solve
       output_file: output/ex192_mumps_2.out
     test:
       requires: cuda
       suffix: mumps_cuda_2
       args: -symmetric_solve -cuda_solve
       output_file: output/ex192_mumps_2.out
     test:
       suffix: mumps_3
       args: -symmetric_solve -hermitian_solve
       output_file: output/ex192_mumps_3.out
     test:
       requires: cuda defined(PETSC_HAVE_CUSOLVERDNDPOTRI)
       suffix: mumps_cuda_3
       args: -symmetric_solve -hermitian_solve -cuda_solve
       output_file: output/ex192_mumps_3.out

TEST*/
