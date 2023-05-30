const char help[] = "Test PCMatSetApplyOperation() and PCMatGetApplyOperation()";

#include <petscpc.h>

static PetscErrorCode TestVecEquality(Vec x, Vec y)
{
  Vec       diff;
  PetscReal err, scale;

  PetscFunctionBegin;
  PetscCall(VecDuplicate(x, &diff));
  PetscCall(VecCopy(x, diff));
  PetscCall(VecAXPY(diff, -1.0, y));
  PetscCall(VecNorm(diff, NORM_INFINITY, &err));
  PetscCall(VecNorm(x, NORM_INFINITY, &scale));
  PetscCheck(err <= PETSC_SMALL * scale, PetscObjectComm((PetscObject)x), PETSC_ERR_PLIB, "PC operation does not match Mat operation");
  PetscCall(VecDestroy(&diff));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestMatEquality(Mat x, Mat y)
{
  Mat       diff;
  PetscReal err, scale;
  PetscInt  m, n;

  PetscFunctionBegin;
  PetscCall(MatGetSize(x, &m, &n));
  PetscCall(MatDuplicate(x, MAT_COPY_VALUES, &diff));
  PetscCall(MatAXPY(diff, -1.0, y, SAME_NONZERO_PATTERN));
  PetscCall(MatNorm(diff, NORM_FROBENIUS, &err));
  PetscCall(MatNorm(x, NORM_FROBENIUS, &scale));
  PetscCheck(err < PETSC_SMALL * m * n * scale, PetscObjectComm((PetscObject)x), PETSC_ERR_PLIB, "PC operation does not match Mat operation");
  PetscCall(MatDestroy(&diff));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Test PCMat with a given operation versus a Matrix that encode the same operation relative to the original matrix
static PetscErrorCode TestPCMatVersusMat(PC pc, Mat A, PetscRandom rand, MatOperation op)
{
  Vec          b, x, x2;
  Mat          B, X, X2;
  MatOperation op2;

  PetscFunctionBegin;
  PetscCall(PCMatSetApplyOperation(pc, op));
  PetscCall(PCView(pc, PETSC_VIEWER_STDOUT_SELF));
  PetscCall(PCMatGetApplyOperation(pc, &op2));
  PetscCheck(op == op2, PetscObjectComm((PetscObject)pc), PETSC_ERR_PLIB, "Input and output MatOperation differ");

  PetscCall(MatCreateVecs(A, &b, &x));
  PetscCall(VecDuplicate(x, &x2));
  PetscCall(VecSetRandom(b, rand));

  PetscCall(MatDuplicate(A, MAT_DO_NOT_COPY_VALUES, &B));
  PetscCall(MatDuplicate(A, MAT_DO_NOT_COPY_VALUES, &X2));
  PetscCall(MatSetRandom(B, rand));

  PetscCall(MatMult(A, b, x));
  PetscCall(PCApply(pc, b, x2));
  PetscCall(TestVecEquality(x, x2));

  PetscCall(MatMultTranspose(A, b, x));
  PetscCall(PCApplyTranspose(pc, b, x2));
  PetscCall(TestVecEquality(x, x2));

  PetscCall(MatMatMult(A, B, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &X));
  PetscCall(PCMatApply(pc, B, X2));
  PetscCall(TestMatEquality(X, X2));

  PetscCall(MatDestroy(&X2));
  PetscCall(MatDestroy(&X));
  PetscCall(MatDestroy(&B));
  PetscCall(VecDestroy(&x2));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscInt    n = 10;
  Mat         A, AT, AH, II, Ainv, AinvT;
  MPI_Comm    comm;
  PC          pc;
  PetscRandom rand;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_SELF;

  PetscCall(PetscRandomCreate(comm, &rand));
  if (PetscDefined(USE_COMPLEX)) {
    PetscScalar i = PetscSqrtScalar(-1.0);
    PetscCall(PetscRandomSetInterval(rand, -1.0 - i, 1.0 + i));
  } else {
    PetscCall(PetscRandomSetInterval(rand, -1.0, 1.0));
  }

  PetscCall(MatCreateSeqDense(comm, n, n, NULL, &A));
  PetscCall(MatSetRandom(A, rand));

  PetscCall(PCCreate(comm, &pc));
  PetscCall(PCSetType(pc, PCMAT));
  PetscCall(PCSetOperators(pc, A, A));
  PetscCall(PCSetUp(pc));

  MatOperation default_op;
  PetscCall(PCMatGetApplyOperation(pc, &default_op));
  PetscCheck(default_op == MATOP_MULT, comm, PETSC_ERR_PLIB, "Default operation has changed");

  // Test setting an invalid operation
  PetscCall(PetscPushErrorHandler(PetscReturnErrorHandler, NULL));
  PetscErrorCode ierr = PCMatSetApplyOperation(pc, MATOP_SET_VALUES);
  PetscCall(PetscPopErrorHandler());
  PetscCheck(ierr == PETSC_ERR_ARG_INCOMP, comm, PETSC_ERR_PLIB, "Wrong error message for unsupported MatOperation");

  PetscCall(TestPCMatVersusMat(pc, A, rand, MATOP_MULT));

  PetscCall(MatTranspose(A, MAT_INITIAL_MATRIX, &AT));
  PetscCall(TestPCMatVersusMat(pc, AT, rand, MATOP_MULT_TRANSPOSE));

  PetscCall(MatHermitianTranspose(A, MAT_INITIAL_MATRIX, &AH));
  PetscCall(TestPCMatVersusMat(pc, AH, rand, MATOP_MULT_HERMITIAN_TRANSPOSE));

  PetscCall(MatDuplicate(A, MAT_DO_NOT_COPY_VALUES, &II));
  PetscCall(MatDuplicate(A, MAT_DO_NOT_COPY_VALUES, &Ainv));
  PetscCall(MatZeroEntries(II));
  PetscCall(MatShift(II, 1.0));
  PetscCall(MatLUFactor(A, NULL, NULL, NULL));
  PetscCall(MatMatSolve(A, II, Ainv));
  PetscCall(PCSetOperators(pc, A, A));
  PetscCall(TestPCMatVersusMat(pc, Ainv, rand, MATOP_SOLVE));

  PetscCall(MatTranspose(Ainv, MAT_INITIAL_MATRIX, &AinvT));
  PetscCall(TestPCMatVersusMat(pc, AinvT, rand, MATOP_SOLVE_TRANSPOSE));

  PetscCall(PCDestroy(&pc));
  PetscCall(MatDestroy(&AinvT));
  PetscCall(MatDestroy(&Ainv));
  PetscCall(MatDestroy(&II));
  PetscCall(MatDestroy(&AH));
  PetscCall(MatDestroy(&AT));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscRandomDestroy(&rand));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0

TEST*/
