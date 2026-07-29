static char help[] = "Solve multiple shifted linear systems.\n\nInput arguments are:\n\
  -m size                   - problem size\n\
  -nshift nshift            - number of shifts\n\
  -mass (true|false)        - test with a mass matrix M\n\
  -explicitmat (true|false) - build the nested matrix explicitly\n\
  -cmplx (true|false)       - test with complex shifts\n\n";

#include <petscksp.h>

/* element stiffness for Laplacian */
PetscErrorCode FormElementStiffness(PetscReal H, PetscScalar *Ke)
{
  PetscFunctionBeginUser;
  Ke[0]  = H / 6.0;
  Ke[1]  = -.125 * H;
  Ke[2]  = H / 12.0;
  Ke[3]  = -.125 * H;
  Ke[4]  = -.125 * H;
  Ke[5]  = H / 6.0;
  Ke[6]  = -.125 * H;
  Ke[7]  = H / 12.0;
  Ke[8]  = H / 12.0;
  Ke[9]  = -.125 * H;
  Ke[10] = H / 6.0;
  Ke[11] = -.125 * H;
  Ke[12] = -.125 * H;
  Ke[13] = H / 12.0;
  Ke[14] = -.125 * H;
  Ke[15] = H / 6.0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FormElementRhs(PetscScalar x, PetscScalar y, PetscReal H, PetscScalar *r)
{
  PetscFunctionBeginUser;
  r[0] = 0.;
  r[1] = 0.;
  r[2] = 0.;
  r[3] = 0.0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* compute the norm ||(K+alpha*M)*x+beta*M*y-b|| using two workspace vectors r and w */
PetscErrorCode ComputeResidualNorm(Mat K, PetscScalar alpha, Mat M, Vec x, Vec b, Vec r, Vec w, PetscScalar beta, Vec y, PetscReal *norm)
{
  PetscFunctionBeginUser;
  PetscCall(MatMult(K, x, r));
  if (M) {
    PetscCall(MatMult(M, x, w));
    PetscCall(VecAXPY(r, alpha, w));
  } else PetscCall(VecAXPY(r, alpha, x));
  if (y) {
    if (M) {
      PetscCall(MatMult(M, y, w));
      PetscCall(VecAXPY(r, beta, w));
    } else PetscCall(VecAXPY(r, beta, y));
  }
  if (b) PetscCall(VecAXPY(r, -1.0, b));
  PetscCall(VecNorm(r, NORM_2, norm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* print a message with the shift and the residual error */
PetscErrorCode PrintError(PetscScalar sigma, PetscScalar sigmai, PetscReal norm, PetscReal normb, PetscBool conjpair, PetscReal tol)
{
  char msg[PETSC_MAX_PATH_LEN];

  PetscFunctionBeginUser;
  if (norm / normb < 10 * tol) PetscCall(PetscSNPrintf(msg, sizeof(msg), "Relative residual error < 10 * %g", (double)tol));
  else PetscCall(PetscSNPrintf(msg, sizeof(msg), "Relative residual error %g", (double)(norm / normb)));
#if PetscDefined(USE_COMPLEX)
  if (PetscImaginaryPart(sigma) == 0.0) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "sigma=%g: %s\n", (double)PetscRealPart(sigma), msg));
  else PetscCall(PetscPrintf(PETSC_COMM_WORLD, "sigma=%g%+gi: %s\n", (double)PetscRealPart(sigma), (double)PetscImaginaryPart(sigma), msg));
#else
  if (!conjpair) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "sigma=%g: %s\n", (double)sigma, msg));
  else {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "sigma=%g+%gi: %s\n", (double)sigma, (double)sigmai, msg));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "sigma=%g-%gi: %s\n", (double)sigma, (double)sigmai, msg));
  }
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **args)
{
  Vec          b, x, r, b_nest, x_nest, w = NULL;
  Mat          A, K, M = NULL;
  KSP          ksp;
  PetscInt     N, m2, idx[4], count, m = 80, nshift = 4, start, end, ostart, oend;
  PetscInt    *rows;
  PetscReal    h, norm, normb, rtol;
  PetscScalar  Ke[16], v[4], xx, yy, lend = -8e-4, rend = -2e-4;
  PetscScalar *sigma = NULL, *sigma_imaginary = NULL;
  PetscBool    mass = PETSC_FALSE, explicitmat = PETSC_FALSE, cmplx = PETSC_FALSE;
  PetscMPIInt  rank, size;
#if !PetscDefined(USE_COMPLEX)
  Vec       xi;
  PetscReal normi;
#endif

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-mass", &mass, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-explicitmat", &explicitmat, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-cmplx", &cmplx, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-m", &m, NULL));
  N  = (m + 1) * (m + 1);
  m2 = m * m;
  h  = 1.0 / m;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nshift", &nshift, NULL));
  PetscCheck(nshift > 0, PETSC_COMM_WORLD, PETSC_ERR_USER, "nshift should be at least 1");
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
              Build the matrices and right-hand side
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create stiffness matrix
  */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &K));
  PetscCall(MatSetSizes(K, PETSC_DECIDE, PETSC_DECIDE, N, N));
  PetscCall(MatSetFromOptions(K));
  PetscCall(MatSeqAIJSetPreallocation(K, 9, NULL));
  PetscCall(MatMPIAIJSetPreallocation(K, 9, NULL, 8, NULL));
  start = rank * (m2 / size) + ((m2 % size) < rank ? (m2 % size) : rank);
  end   = start + m2 / size + ((m2 % size) > rank);

  PetscCall(FormElementStiffness(h * h, Ke));
  for (PetscInt i = start; i < end; i++) {
    /* node numbers for the four corners of element */
    idx[0] = (m + 1) * (i / m) + (i % m);
    idx[1] = idx[0] + 1;
    idx[2] = idx[1] + m + 1;
    idx[3] = idx[2] - 1;
    PetscCall(MatSetValues(K, 4, idx, 4, idx, Ke, ADD_VALUES));
  }
  PetscCall(MatAssemblyBegin(K, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(K, MAT_FINAL_ASSEMBLY));

  if (mass) {
    /*
        Create mass matrix
     */
    PetscCall(MatCreate(PETSC_COMM_WORLD, &M));
    PetscCall(MatSetSizes(M, PETSC_DECIDE, PETSC_DECIDE, N, N));
    PetscCall(MatSetFromOptions(M));
    PetscCall(MatGetOwnershipRange(M, &ostart, &oend));
    for (PetscInt i = ostart; i < oend; i++) PetscCall(MatSetValue(M, i, i, 3.0, INSERT_VALUES));
    PetscCall(MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY));
  }

  /*
     Assemble right-hand side
  */
  PetscCall(MatCreateVecs(K, &r, &b));
  if (M) PetscCall(VecDuplicate(b, &w));
  for (PetscInt i = start; i < end; i++) {
    /* location of lower-left corner of element */
    xx = h * (i % m);
    yy = h * (i / m);
    /* node numbers for the four corners of element */
    idx[0] = (m + 1) * (i / m) + (i % m);
    idx[1] = idx[0] + 1;
    idx[2] = idx[1] + m + 1;
    idx[3] = idx[2] - 1;
    PetscCall(FormElementRhs(xx, yy, h * h, v));
    PetscCall(VecSetValues(b, 4, idx, v, ADD_VALUES));
  }
  PetscCall(VecAssemblyBegin(b));
  PetscCall(VecAssemblyEnd(b));

  /*
     Modify matrices and right-hand side for Dirichlet boundary conditions
  */
  PetscCall(PetscMalloc1(4 * m, &rows));
  for (PetscInt i = 0; i < m + 1; i++) {
    rows[i]             = i;               /* bottom */
    rows[3 * m - 1 + i] = m * (m + 1) + i; /* top */
  }
  count = m + 1; /* left side */
  for (PetscInt i = m + 1; i < m * (m + 1); i += m + 1) rows[count++] = i;
  count = 2 * m; /* right side */
  for (PetscInt i = 2 * m + 1; i < m * (m + 1); i += m + 1) rows[count++] = i;
  for (PetscInt i = 0; i < 4 * m; i++) {
    yy = h * (rows[i] / (m + 1));
    PetscCall(VecSetValues(b, 1, &rows[i], &yy, INSERT_VALUES));
  }
  PetscCall(MatZeroRows(K, 4 * m, rows, 1.0, NULL, NULL));
  if (mass) PetscCall(MatZeroRows(M, 4 * m, rows, 2.0, NULL, NULL));
  PetscCall(PetscFree(rows));

  PetscCall(VecAssemblyBegin(b));
  PetscCall(VecAssemblyEnd(b));
  PetscCall(VecNorm(b, NORM_2, &normb));

  /*
     Generate shifts
  */
  PetscCall(PetscCalloc2(nshift, &sigma, nshift, &sigma_imaginary));
  for (PetscInt i = 0; i < nshift; i++) {
    sigma[i] = lend + (rend - lend) / (nshift + 1) * (i + 1);
    if (cmplx) {
#if PetscDefined(USE_COMPLEX)
      sigma[i] += 1e-4 * (i + 1) * PETSC_i;
#else
      if (i == 0 && nshift > 1) {
        sigma[i + 1]           = sigma[i];
        sigma_imaginary[i]     = 1e-4;
        sigma_imaginary[i + 1] = -1e-4;
        i++;
      }
#endif
    }
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the shifted linear systems
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(MatCreateNestFromMultipleShifts(K, nshift, sigma, sigma_imaginary, M, explicitmat, SUBSET_NONZERO_PATTERN, &A));
  PetscCall(MatCreateVecNestFromMultipleShifts(A, b, &b_nest));
  PetscCall(MatCreateVecNestFromMultipleShifts(A, NULL, &x_nest));

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPGetTolerances(ksp, &rtol, NULL, NULL, NULL));
  PetscCall(KSPSolve(ksp, b_nest, x_nest));

  /*
     Check residual norm for each shifted linear system
  */
  for (PetscInt i = 0; i < nshift; i++) {
    PetscCall(VecNestGetSubVec(x_nest, i, &x));
#if PetscDefined(USE_COMPLEX)
    PetscCall(ComputeResidualNorm(K, sigma[i], M, x, b, r, w, 0.0, NULL, &norm));
    PetscCall(PrintError(sigma[i], 0.0, norm, normb, PETSC_FALSE, rtol));
#else
    if (sigma_imaginary[i] == 0.0) {
      PetscCall(ComputeResidualNorm(K, sigma[i], M, x, b, r, w, 0.0, NULL, &norm));
      PetscCall(PrintError(sigma[i], 0.0, norm, normb, PETSC_FALSE, rtol));
    } else {
      PetscCall(VecNestGetSubVec(x_nest, i + 1, &xi));
      PetscCall(ComputeResidualNorm(K, sigma[i], M, x, b, r, w, -sigma_imaginary[i], xi, &norm));
      PetscCall(ComputeResidualNorm(K, sigma[i], M, xi, NULL, r, w, sigma_imaginary[i], x, &normi));
      norm = PetscHypotReal(norm, normi);
      PetscCall(PrintError(sigma[i], sigma_imaginary[i], norm, normb, PETSC_TRUE, rtol));
      i++;
    }
#endif
  }

  /*
     Clean up
  */
  PetscCall(KSPDestroy(&ksp));
  PetscCall(MatDestroy(&K));
  PetscCall(MatDestroy(&M));
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&b_nest));
  PetscCall(VecDestroy(&x_nest));
  PetscCall(VecDestroy(&w));
  PetscCall(VecDestroy(&r));
  PetscCall(VecDestroy(&b));
  PetscCall(PetscFree2(sigma, sigma_imaginary));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   testset:
      nsize: 2
      args: -ksp_type {{gmres eksm}} -explicitmat {{0 1}}
      output_file: output/ex90_1.out
      test:
         suffix: 1
      test:
         suffix: 2
         args: -mass

   test:
      args: -ksp_type {{gmres eksm}} -explicitmat {{0 1}} -cmplx
      suffix: 3
      requires: !complex

TEST*/
