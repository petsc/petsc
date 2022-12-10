static const char help[] = "Test parallel assembly of SBAIJ matrices\n\n";

#include <petscmat.h>

PetscErrorCode Assemble(MPI_Comm comm, PetscInt bs, MatType mtype)
{
  const PetscInt    rc[]   = {0, 1, 2, 3};
  const PetscScalar vals[] = {100, 2,  3,  4,  5,  600, 7,  8,  9,  100, 11, 1200, 13, 14, 15, 1600, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 2800, 29, 30, 31, 32,
                              33,  34, 35, 36, 37, 38,  39, 40, 41, 42,  43, 44,   45, 46, 47, 48,   49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 49, 60,   61, 62, 63, 64};
  Mat               A;
#if defined(PETSC_HAVE_MUMPS) || defined(PETSC_HAVE_MKL_CPARDISO)
  Mat           F;
  MatSolverType stype = MATSOLVERPETSC;
  PetscRandom   rdm;
  Vec           b, x, y;
  PetscInt      i, j;
  PetscReal     norm2, tol = 100 * PETSC_SMALL;
  PetscBool     issbaij;
#endif
  PetscViewer viewer;

  PetscFunctionBegin;
  PetscCall(MatCreate(comm, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, 4 * bs, 4 * bs));
  PetscCall(MatSetType(A, mtype));
  PetscCall(MatMPIBAIJSetPreallocation(A, bs, 2, NULL, 2, NULL));
  PetscCall(MatMPISBAIJSetPreallocation(A, bs, 2, NULL, 2, NULL));
  PetscCall(MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));
  /* All processes contribute a global matrix */
  PetscCall(MatSetValuesBlocked(A, 4, rc, 4, rc, vals, ADD_VALUES));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(PetscPrintf(comm, "Matrix %s(%" PetscInt_FMT ")\n", mtype, bs));
  PetscCall(PetscViewerASCIIGetStdout(comm, &viewer));
  PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_INFO_DETAIL));
  PetscCall(MatView(A, viewer));
  PetscCall(PetscViewerPopFormat(viewer));
  PetscCall(MatView(A, viewer));
#if defined(PETSC_HAVE_MUMPS) || defined(PETSC_HAVE_MKL_CPARDISO)
  PetscCall(PetscStrcmp(mtype, MATMPISBAIJ, &issbaij));
  if (!issbaij) PetscCall(MatShift(A, 10));
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rdm));
  PetscCall(PetscRandomSetFromOptions(rdm));
  PetscCall(MatCreateVecs(A, &x, &y));
  PetscCall(VecDuplicate(x, &b));
  for (j = 0; j < 2; j++) {
  #if defined(PETSC_HAVE_MUMPS)
    if (j == 0) stype = MATSOLVERMUMPS;
  #else
    if (j == 0) continue;
  #endif
  #if defined(PETSC_HAVE_MKL_CPARDISO)
    if (j == 1) stype = MATSOLVERMKL_CPARDISO;
  #else
    if (j == 1) continue;
  #endif
    if (issbaij) {
      PetscCall(MatGetFactor(A, stype, MAT_FACTOR_CHOLESKY, &F));
      PetscCall(MatCholeskyFactorSymbolic(F, A, NULL, NULL));
      PetscCall(MatCholeskyFactorNumeric(F, A, NULL));
    } else {
      PetscCall(MatGetFactor(A, stype, MAT_FACTOR_LU, &F));
      PetscCall(MatLUFactorSymbolic(F, A, NULL, NULL, NULL));
      PetscCall(MatLUFactorNumeric(F, A, NULL));
    }
    for (i = 0; i < 10; i++) {
      PetscCall(VecSetRandom(b, rdm));
      PetscCall(MatSolve(F, b, y));
      /* Check the error */
      PetscCall(MatMult(A, y, x));
      PetscCall(VecAXPY(x, -1.0, b));
      PetscCall(VecNorm(x, NORM_2, &norm2));
      if (norm2 > tol) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Error:MatSolve(), norm2: %g\n", (double)norm2));
    }
    PetscCall(MatDestroy(&F));
  }
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&b));
  PetscCall(PetscRandomDestroy(&rdm));
#endif
  PetscCall(MatDestroy(&A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char *argv[])
{
  MPI_Comm    comm;
  PetscMPIInt size;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCheck(size == 2, comm, PETSC_ERR_USER, "This example must be run with exactly two processes");
  PetscCall(Assemble(comm, 2, MATMPIBAIJ));
  PetscCall(Assemble(comm, 2, MATMPISBAIJ));
  PetscCall(Assemble(comm, 1, MATMPIBAIJ));
  PetscCall(Assemble(comm, 1, MATMPISBAIJ));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 2
      args: -mat_ignore_lower_triangular
      filter: sed -e "s~mem [0-9]*~mem~g"

TEST*/
