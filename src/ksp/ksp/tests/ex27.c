
static char help[] = "Reads a PETSc matrix and vector from a file and solves a linear system.\n\
Test MatMatSolve().  Input parameters include\n\
  -f <input_file> : file to load \n\n";

/*
  Usage:
     ex27 -f0 <mat_binaryfile>
*/

#include <petscksp.h>
extern PetscErrorCode PCShellApply_Matinv(PC, Vec, Vec);

int main(int argc, char **args)
{
  KSP         ksp;
  Mat         A, B, F, X;
  Vec         x, b, u;                     /* approx solution, RHS, exact solution */
  PetscViewer fd;                          /* viewer */
  char        file[1][PETSC_MAX_PATH_LEN]; /* input file name */
  PetscBool   flg;
  PetscInt    M, N, i, its;
  PetscReal   norm;
  PetscScalar val = 1.0;
  PetscMPIInt size;
  PC          pc;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This is a uniprocessor example only!");

  /* Read matrix and right-hand-side vector */
  PetscCall(PetscOptionsGetString(NULL, NULL, "-f", file[0], sizeof(file[0]), &flg));
  PetscCheck(flg, PETSC_COMM_WORLD, PETSC_ERR_USER, "Must indicate binary file with the -f option");

  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, file[0], FILE_MODE_READ, &fd));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetType(A, MATAIJ));
  PetscCall(MatLoad(A, fd));
  PetscCall(VecCreate(PETSC_COMM_WORLD, &b));
  PetscCall(VecLoad(b, fd));
  PetscCall(PetscViewerDestroy(&fd));

  /*
     If the loaded matrix is larger than the vector (due to being padded
     to match the block size of the system), then create a new padded vector.
  */
  {
    PetscInt     m, n, j, mvec, start, end, indx;
    Vec          tmp;
    PetscScalar *bold;

    /* Create a new vector b by padding the old one */
    PetscCall(MatGetLocalSize(A, &m, &n));
    PetscCheck(m == n, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "This example is not intended for rectangular matrices (%" PetscInt_FMT ", %" PetscInt_FMT ")", m, n);
    PetscCall(VecCreate(PETSC_COMM_WORLD, &tmp));
    PetscCall(VecSetSizes(tmp, m, PETSC_DECIDE));
    PetscCall(VecSetFromOptions(tmp));
    PetscCall(VecGetOwnershipRange(b, &start, &end));
    PetscCall(VecGetLocalSize(b, &mvec));
    PetscCall(VecGetArray(b, &bold));
    for (j = 0; j < mvec; j++) {
      indx = start + j;
      PetscCall(VecSetValues(tmp, 1, &indx, bold + j, INSERT_VALUES));
    }
    PetscCall(VecRestoreArray(b, &bold));
    PetscCall(VecDestroy(&b));
    PetscCall(VecAssemblyBegin(tmp));
    PetscCall(VecAssemblyEnd(tmp));
    b = tmp;
  }
  PetscCall(VecDuplicate(b, &x));
  PetscCall(VecDuplicate(b, &u));
  PetscCall(VecSet(x, 0.0));

  /* Create dense matric B and X. Set B as an identity matrix */
  PetscCall(MatGetSize(A, &M, &N));
  PetscCall(MatCreate(MPI_COMM_SELF, &B));
  PetscCall(MatSetSizes(B, M, N, M, N));
  PetscCall(MatSetType(B, MATSEQDENSE));
  PetscCall(MatSeqDenseSetPreallocation(B, NULL));
  for (i = 0; i < M; i++) PetscCall(MatSetValues(B, 1, &i, 1, &i, &val, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));

  PetscCall(MatDuplicate(B, MAT_DO_NOT_COPY_VALUES, &X));

  /* Compute X=inv(A) by MatMatSolve() */
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCLU));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSetUp(ksp));
  PetscCall(PCFactorGetMatrix(pc, &F));
  PetscCall(MatMatSolve(F, B, X));
  PetscCall(MatDestroy(&B));

  /* Now, set X=inv(A) as a preconditioner */
  PetscCall(PCSetType(pc, PCSHELL));
  PetscCall(PCShellSetContext(pc, X));
  PetscCall(PCShellSetApply(pc, PCShellApply_Matinv));
  PetscCall(KSPSetFromOptions(ksp));

  /* Solve preconditioned system A*x = b */
  PetscCall(KSPSolve(ksp, b, x));
  PetscCall(KSPGetIterationNumber(ksp, &its));

  /* Check error */
  PetscCall(MatMult(A, x, u));
  PetscCall(VecAXPY(u, -1.0, b));
  PetscCall(VecNorm(u, NORM_2, &norm));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Number of iterations = %3" PetscInt_FMT "\n", its));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Residual norm %g\n", (double)norm));

  /* Free work space.  */
  PetscCall(MatDestroy(&X));
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&x));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(PetscFinalize());
  return 0;
}

PetscErrorCode PCShellApply_Matinv(PC pc, Vec xin, Vec xout)
{
  Mat X;

  PetscFunctionBeginUser;
  PetscCall(PCShellGetContext(pc, &X));
  PetscCall(MatMult(X, xin, xout));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*TEST

    test:
      args: -f ${DATAFILESPATH}/matrices/small
      requires: datafilespath !complex double !defined(PETSC_USE_64BIT_INDICES)
      output_file: output/ex27.out

TEST*/
