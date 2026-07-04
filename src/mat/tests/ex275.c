static char help[] = "Tests MatNormApproximate() on the Brusselator matrix.\n\n"
                     "The command line options are:\n"
                     "  -n <n>, where <n> = block dimension of the 2x2 block matrix.\n"
                     "  -L <L>, where <L> = bifurcation parameter.\n"
                     "  -alpha <alpha>, -beta <beta>, -delta1 <delta1>,  -delta2 <delta2>,\n"
                     "       where <alpha> <beta> <delta1> <delta2> = model parameters.\n\n";

#include <petscmat.h>

/*
   The Brusselator matrix is

        A = [ tau1*T+(beta-1)*I     alpha^2*I
                  -beta*I        tau2*T-alpha^2*I ],

   where

        T = tridiag{1,-2,1}
        h = 1/(n+1)
        tau1 = delta1/(h*L)^2
        tau2 = delta2/(h*L)^2
 */

int main(int argc, char **argv)
{
  Mat         A, T1, T2, D1, D2, mats[4], Ae;
  PetscScalar alpha, beta, tau1, tau2, delta1, delta2, L, h;
  PetscReal   norm[3];
  PetscInt    N = 30, i, Istart, Iend, samples = PETSC_DECIDE;
  PetscBool   test_fwd = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &N, NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Generate the matrix
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  alpha  = 2.0;
  beta   = 5.45;
  delta1 = 0.008;
  delta2 = 0.004;
  L      = 0.51302;

  PetscCall(PetscOptionsGetScalar(NULL, NULL, "-L", &L, NULL));
  PetscCall(PetscOptionsGetScalar(NULL, NULL, "-alpha", &alpha, NULL));
  PetscCall(PetscOptionsGetScalar(NULL, NULL, "-beta", &beta, NULL));
  PetscCall(PetscOptionsGetScalar(NULL, NULL, "-delta1", &delta1, NULL));
  PetscCall(PetscOptionsGetScalar(NULL, NULL, "-delta2", &delta2, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-samples", &samples, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_fwd", &test_fwd, NULL));

  h    = 1.0 / (PetscReal)(N + 1);
  tau1 = delta1 / ((h * L) * (h * L));
  tau2 = delta2 / ((h * L) * (h * L));

  /* Create matrices T1, T2 */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &T1));
  PetscCall(MatSetSizes(T1, PETSC_DECIDE, PETSC_DECIDE, N, N));
  PetscCall(MatSetFromOptions(T1));
  PetscCall(MatGetOwnershipRange(T1, &Istart, &Iend));
  for (i = Istart; i < Iend; i++) {
    if (i > 0) PetscCall(MatSetValue(T1, i, i - 1, 1.0, INSERT_VALUES));
    if (i < N - 1) PetscCall(MatSetValue(T1, i, i + 1, 1.0, INSERT_VALUES));
    PetscCall(MatSetValue(T1, i, i, -2.0, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(T1, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(T1, MAT_FINAL_ASSEMBLY));

  PetscCall(MatDuplicate(T1, MAT_COPY_VALUES, &T2));
  PetscCall(MatScale(T1, tau1));
  PetscCall(MatShift(T1, beta - 1.0));
  PetscCall(MatScale(T2, tau2));
  PetscCall(MatShift(T2, -alpha * alpha));

  /* Create matrices D1, D2 */
  PetscCall(MatCreateConstantDiagonal(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, N, N, alpha * alpha, &D1));
  PetscCall(MatCreateConstantDiagonal(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, N, N, -beta, &D2));

  /* Create the nest matrix */
  mats[0] = T1;
  mats[1] = D1;
  mats[2] = D2;
  mats[3] = T2;
  PetscCall(MatCreateNest(PETSC_COMM_WORLD, 2, NULL, 2, NULL, mats, &A));
  if (test_fwd) PetscCall(MatSetOperation(A, MATOP_MULT_TRANSPOSE, NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Estimate the norm
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(MatComputeOperator(A, MATDENSE, &Ae));
  PetscCall(MatViewFromOptions(Ae, NULL, "-view_explicit"));

  PetscCall(MatNormApproximate(A, NORM_1, samples, norm));
  PetscCall(MatNormApproximate(A, NORM_INFINITY, samples, norm + 1));
  PetscCall(MatNormApproximate(A, NORM_2, samples, norm + 2));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nBrusselator matrix, n=%" PetscInt_FMT " - estimated 1-norm = %g - estimated infinity-norm = %g - estimated 2-norm = %g\n\n", N, (double)norm[0], (double)norm[1], (double)norm[2]));

  PetscCall(MatNorm(Ae, NORM_1, norm));
  PetscCall(MatNorm(Ae, NORM_INFINITY, norm + 1));
  PetscCall(MatNorm(Ae, NORM_2, norm + 2));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nBrusselator matrix, n=%" PetscInt_FMT " - exact 1-norm = %g - exact infinity-norm = %g - exact 2-norm = %g\n\n", N, (double)norm[0], (double)norm[1], (double)norm[2]));

  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&Ae));
  PetscCall(MatDestroy(&T1));
  PetscCall(MatDestroy(&T2));
  PetscCall(MatDestroy(&D1));
  PetscCall(MatDestroy(&D2));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
     suffix: 1
     args: -test_fwd {{false true}shared output}
     output_file: output/ex275_1.out

TEST*/
