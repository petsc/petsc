static char help[] = "Tests MatForwardSolve and MatBackwardSolve for LU and Cholesky decompositions using C/Pardiso.\n\n";

#include <petscdmda.h>
#include <petscmat.h>

int main(int argc, char **args)
{
  DM            da;
  DMDALocalInfo info;
  Mat           A, F;
  Vec           x, y, b, ytmp;
  IS            rowis, colis;
  PetscInt      i, j, k, n = 5;
  PetscBool     CHOL = PETSC_FALSE;
  MatStencil    row, cols[5];
  PetscScalar   vals[5];
  PetscReal     norm2, tol = 100. * PETSC_MACHINE_EPSILON;
  PetscRandom   rdm;
  MatFactorInfo finfo;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));

  PetscCall(DMDACreate2d(MPI_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, n, n, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &da));
  PetscCall(DMSetUp(da));

  PetscCall(DMCreateMatrix(da, &A));
  PetscCall(PetscOptionsHasName(NULL, NULL, "-chol", &CHOL));
  if (CHOL) PetscCall(MatSetType(A, MATSBAIJ));
  else PetscCall(MatSetType(A, MATBAIJ));
  PetscCall(MatSetUp(A));

  PetscCall(DMDAGetLocalInfo(da, &info));
  for (j = info.ys; j < info.ys + info.ym; j++) {
    for (i = info.xs; i < info.xs + info.xm; i++) {
      row.j = j;
      row.i = i;

      k = 0;
      if (j != 0) {
        cols[k].j = j - 1;
        cols[k].i = i;
        vals[k]   = -1;
        ++k;
      }
      if (i != 0) {
        cols[k].j = j;
        cols[k].i = i - 1;
        vals[k]   = -1;
        ++k;
      }
      cols[k].j = j;
      cols[k].i = i;
      vals[k]   = 4;
      ++k;
      if (j != info.my - 1) {
        cols[k].j = j + 1;
        cols[k].i = i;
        vals[k]   = -1;
        ++k;
      }
      if (i != info.mx - 1) {
        cols[k].j = j;
        cols[k].i = i + 1;
        vals[k]   = -1;
        ++k;
      }
      PetscCall(MatSetValuesStencil(A, 1, &row, k, cols, vals, INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  if (CHOL) PetscCall(MatSetOption(A, MAT_SPD, PETSC_TRUE));

  /* Create vectors for error checking */
  PetscCall(MatCreateVecs(A, &x, &b));
  PetscCall(VecDuplicate(x, &y));
  PetscCall(VecDuplicate(x, &ytmp));
  PetscCall(PetscRandomCreate(PETSC_COMM_SELF, &rdm));
  PetscCall(PetscRandomSetFromOptions(rdm));
  PetscCall(VecSetRandom(x, rdm));
  PetscCall(MatMult(A, x, b));

  PetscCall(MatGetOrdering(A, MATORDERINGNATURAL, &rowis, &colis));
  if (CHOL) {
    PetscCall(PetscPrintf(MPI_COMM_WORLD, "Test Cholesky...\n"));
    PetscCall(MatGetFactor(A, MATSOLVERMKL_CPARDISO, MAT_FACTOR_CHOLESKY, &F));
    PetscCall(MatCholeskyFactorSymbolic(F, A, rowis, &finfo));
    PetscCall(MatCholeskyFactorNumeric(F, A, &finfo));
  } else {
    PetscCall(PetscPrintf(MPI_COMM_WORLD, "Test LU...\n"));
    PetscCall(MatGetFactor(A, MATSOLVERMKL_CPARDISO, MAT_FACTOR_LU, &F));
    PetscCall(MatLUFactorSymbolic(F, A, rowis, colis, &finfo));
    PetscCall(MatLUFactorNumeric(F, A, &finfo));
  }

  PetscCall(PetscPrintf(MPI_COMM_WORLD, "Test MatForwardSolve...\n"));
  PetscCall(MatForwardSolve(F, b, ytmp));
  PetscCall(PetscPrintf(MPI_COMM_WORLD, "Test MatBackwardSolve...\n"));
  PetscCall(MatBackwardSolve(F, ytmp, y));
  PetscCall(VecAXPY(y, -1.0, x));
  PetscCall(VecNorm(y, NORM_2, &norm2));
  if (norm2 > tol) PetscCall(PetscPrintf(PETSC_COMM_SELF, "MatForwardSolve and BackwardSolve: Norm of error=%g\n", (double)norm2));

  /* Free data structures */
  PetscCall(ISDestroy(&rowis));
  PetscCall(ISDestroy(&colis));
  PetscCall(MatDestroy(&F));
  PetscCall(MatDestroy(&A));
  PetscCall(DMDestroy(&da));
  PetscCall(PetscRandomDestroy(&rdm));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&ytmp));
  PetscCall(VecDestroy(&b));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      requires: mkl_cpardiso
      nsize: 4

   test:
      suffix: 2
      requires: !complex mkl_cpardiso
      nsize: 4
      args: -chol

TEST*/
