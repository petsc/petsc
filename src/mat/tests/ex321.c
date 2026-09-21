static char help[] = "Tests that the assembled views of a MATIS follow a change made through MatISGetLocalMat().\n\n";

#include <petscmat.h>

// Prints the diagonal of A, one line per process.
static PetscErrorCode DiagonalView(Mat A, const char *label)
{
  Vec                d;
  PetscInt           n;
  const PetscScalar *vals;

  PetscFunctionBeginUser;
  PetscCall(MatCreateVecs(A, NULL, &d));
  PetscCall(MatGetDiagonal(A, d));
  PetscCall(VecGetLocalSize(d, &n));
  PetscCall(VecGetArrayRead(d, &vals));
  PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "%s", label));
  for (PetscInt i = 0; i < n; i++) PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, " %g", (double)PetscRealPart(vals[i])));
  PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "\n"));
  PetscCall(VecRestoreArrayRead(d, &vals));
  PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT));
  PetscCall(VecDestroy(&d));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckViews(Mat A, PetscBool subfirst)
{
  Mat       B, dA, dB, *sub;
  MatState  before, after;
  IS        rows;
  PetscInt  start, end;
  PetscBool equal;

  PetscFunctionBeginUser;
  PetscCall(MatConvert(A, MATAIJ, MAT_INITIAL_MATRIX, &B));
  PetscCall(MatGetDiagonalBlock(B, &dB));
  PetscCall(MatGetOwnershipRange(A, &start, &end));
  PetscCall(ISCreateStride(PETSC_COMM_SELF, end - start, start, 1, &rows));
  if (!subfirst) PetscCall(MatGetDiagonalBlock(A, &dA));
  PetscCall(MatCreateSubMatrices(A, 1, &rows, &rows, MAT_INITIAL_MATRIX, &sub));
  if (subfirst) PetscCall(MatGetDiagonalBlock(A, &dA));
  PetscCall(MatMultEqual(dA, dB, 3, &equal));
  PetscCheck(equal, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Stale diagonal block");
  PetscCall(MatMultEqual(sub[0], dB, 3, &equal));
  PetscCheck(equal, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Stale assembled submatrix");
  PetscCall(MatGetState(dA, &before));
  PetscCall(MatGetDiagonalBlock(A, &dA));
  PetscCall(MatGetState(dA, &after));
  PetscCall(MatStateCompare(before, after, &equal));
  PetscCheck(equal, PETSC_COMM_SELF, PETSC_ERR_PLIB, "An unchanged diagonal block was rebuilt");
  PetscCall(MatDestroySubMatrices(1, &sub));
  PetscCall(ISDestroy(&rows));
  PetscCall(MatDestroy(&B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestState(Mat A)
{
  Mat         lA, B, C, D;
  MatState    before, after, lbefore, lafter;
  PetscInt    row = 0, n;
  PetscMPIInt rank;
  PetscBool   same;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)A), &rank));
  PetscCall(CheckViews(A, PETSC_FALSE));
  PetscCall(MatGetState(A, &before));
  PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &B));
  PetscCall(MatAXPY(A, 0.5, B, SAME_NONZERO_PATTERN));
  PetscCall(MatGetState(A, &after));
  PetscCheck(after.state > before.state && after.nonzerostate == before.nonzerostate, PETSC_COMM_SELF, PETSC_ERR_PLIB, "MatAXPY() did not propagate a numerical change");
  PetscCall(CheckViews(A, PETSC_FALSE));
  before = after;

  PetscCall(MatCopy(B, A, SAME_NONZERO_PATTERN));
  PetscCall(MatGetState(A, &after));
  PetscCheck(after.state > before.state && after.nonzerostate == before.nonzerostate, PETSC_COMM_SELF, PETSC_ERR_PLIB, "MatCopy() did not propagate a numerical change");
  PetscCall(MatMultEqual(A, B, 3, &same));
  PetscCheck(same, PetscObjectComm((PetscObject)A), PETSC_ERR_PLIB, "MatCopy() with the same nonzero pattern produced incorrect values");
  PetscCall(CheckViews(A, PETSC_TRUE));
  PetscCall(MatDestroy(&B));
  before = after;

  if (!rank) {
    PetscCall(MatISGetLocalMat(A, &lA));
    PetscCall(MatScale(lA, 2.0));
    PetscCall(MatISRestoreLocalMat(A, &lA));
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatGetState(A, &after));
  PetscCheck(after.state > before.state && after.nonzerostate == before.nonzerostate, PETSC_COMM_SELF, PETSC_ERR_PLIB, "A numerical change was not propagated correctly");
  PetscCall(CheckViews(A, PETSC_TRUE));

  PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &B));
  PetscCall(MatConvert(B, MATAIJ, MAT_INITIAL_MATRIX, &C));
  PetscCall(MatSetOption(C, MAT_KEEP_NONZERO_PATTERN, PETSC_FALSE));
  PetscCall(MatZeroRows(C, rank ? 0 : 1, &row, 1.0, NULL, NULL));
  before = after;
  if (!rank) {
    PetscCall(MatISGetLocalMat(A, &lA));
    PetscCall(MatSetOption(lA, MAT_KEEP_NONZERO_PATTERN, PETSC_FALSE));
    PetscCall(MatZeroRows(lA, 1, &row, 1.0, NULL, NULL));
    PetscCall(MatISRestoreLocalMat(A, &lA));
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatGetState(A, &after));
  PetscCheck(after.state > before.state && after.nonzerostate > before.nonzerostate, PETSC_COMM_SELF, PETSC_ERR_PLIB, "A structural change on one rank was not propagated");
  PetscCall(MatMultEqual(A, C, 3, &same));
  PetscCheck(same, PetscObjectComm((PetscObject)A), PETSC_ERR_PLIB, "Zeroing a local row without keeping its nonzero pattern produced incorrect values");
  PetscCall(CheckViews(A, PETSC_FALSE));
  PetscCall(MatDestroy(&C));

  // Restore the dropped entries through MatCopy(), with both views already cached.
  before = after;
  PetscCall(MatCopy(B, A, DIFFERENT_NONZERO_PATTERN));
  PetscCall(MatGetState(A, &after));
  PetscCheck(after.state > before.state && after.nonzerostate > before.nonzerostate, PETSC_COMM_SELF, PETSC_ERR_PLIB, "MatCopy() did not propagate a structural change");
  PetscCall(MatMultEqual(A, B, 3, &same));
  PetscCheck(same, PetscObjectComm((PetscObject)A), PETSC_ERR_PLIB, "MatCopy() with different nonzero patterns produced incorrect values");
  PetscCall(CheckViews(A, PETSC_TRUE));

  PetscCall(MatConvert(A, MATAIJ, MAT_INITIAL_MATRIX, &C));
  PetscCall(MatSetOption(C, MAT_KEEP_NONZERO_PATTERN, PETSC_FALSE));
  PetscCall(MatZeroRows(C, rank ? 0 : 1, &row, 1.0, NULL, NULL));
  PetscCall(MatSetOption(A, MAT_KEEP_NONZERO_PATTERN, PETSC_FALSE));
  before = after;
  PetscCall(MatZeroRows(A, rank ? 0 : 1, &row, 1.0, NULL, NULL));
  PetscCall(MatGetState(A, &after));
  PetscCheck(after.state > before.state && after.nonzerostate > before.nonzerostate, PETSC_COMM_SELF, PETSC_ERR_PLIB, "MatZeroRows() did not propagate a structural change");
  PetscCall(MatMultEqual(A, C, 3, &same));
  PetscCheck(same, PetscObjectComm((PetscObject)A), PETSC_ERR_PLIB, "MatZeroRows() without keeping the nonzero pattern produced incorrect values");
  PetscCall(CheckViews(A, PETSC_FALSE));

  // MatAXPY() must add the missing entries and invalidate both cached views.
  PetscCall(MatConvert(B, MATAIJ, MAT_INITIAL_MATRIX, &D));
  PetscCall(MatAXPY(C, 0.5, D, DIFFERENT_NONZERO_PATTERN));
  before = after;
  PetscCall(MatAXPY(A, 0.5, B, DIFFERENT_NONZERO_PATTERN));
  PetscCall(MatGetState(A, &after));
  PetscCheck(after.state > before.state && after.nonzerostate > before.nonzerostate, PETSC_COMM_SELF, PETSC_ERR_PLIB, "MatAXPY() did not propagate a structural change");
  PetscCall(MatMultEqual(A, C, 3, &same));
  PetscCheck(same, PetscObjectComm((PetscObject)A), PETSC_ERR_PLIB, "MatAXPY() with different nonzero patterns produced incorrect values");
  PetscCall(CheckViews(A, PETSC_TRUE));
  PetscCall(MatDestroy(&D));
  PetscCall(MatDestroy(&C));
  PetscCall(MatDestroy(&B));

  // Replace the local matrix on one rank, retaining its type and nonzero pattern.
  for (PetscInt i = 0; i < 2; i++) {
    PetscCall(MatGetState(A, &before));
    if (!rank) {
      PetscCall(MatISGetLocalMat(A, &lA));
      PetscCall(MatDuplicate(lA, MAT_COPY_VALUES, &B));
      PetscCall(MatISRestoreLocalMat(A, &lA));
      PetscCall(MatScale(B, 2.0));
      PetscCall(MatISSetLocalMat(A, B));
      PetscCall(MatDestroy(&B));
    }
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatGetState(A, &after));
    PetscCheck(after.state > before.state && after.nonzerostate > before.nonzerostate, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Local matrix replacement was not propagated");
    PetscCall(CheckViews(A, (PetscBool)i));
  }

  // Replacing an empty local matrix must be detected even when its nonzero state is unchanged.
  for (PetscInt i = 0; i < 2; i++) {
    PetscCall(MatGetState(A, &before));
    if (!rank) {
      PetscCall(MatISGetLocalMat(A, &lA));
      PetscCall(MatGetState(lA, &lbefore));
      PetscCall(MatGetSize(lA, &n, NULL));
      PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, n, n, 0, NULL, &B));
      PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
      PetscCall(MatGetState(B, &lafter));
      if (i) PetscCheck(lbefore.id != lafter.id && lbefore.nonzerostate == lafter.nonzerostate, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Expected different local matrices with equal nonzero states");
      PetscCall(MatISRestoreLocalMat(A, &lA));
      PetscCall(MatISSetLocalMat(A, B));
      PetscCall(MatDestroy(&B));
    }
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatGetState(A, &after));
    PetscCheck(after.state > before.state && after.nonzerostate > before.nonzerostate, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Local matrix replacement was not propagated");
    PetscCall(CheckViews(A, PETSC_TRUE));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestReuse(Mat A)
{
  Mat       C;
  PetscBool equal;

  PetscFunctionBeginUser;
  PetscCall(MatConvert(A, MATAIJ, MAT_INITIAL_MATRIX, &C));
  PetscCall(MatConvert(A, MATAIJ, MAT_REUSE_MATRIX, &C));
  PetscCall(MatMultEqual(A, C, 3, &equal));
  PetscCheck(equal, PetscObjectComm((PetscObject)A), PETSC_ERR_PLIB, "Reuse with the same source failed");
  PetscCall(MatDestroy(&C));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **args)
{
  Mat                    A, Aij, dA, lA;
  ISLocalToGlobalMapping l2g;
  PetscInt              *gidx;
  PetscInt               row = 1, nl = 4, N;
  PetscMPIInt            rank, size;
  PetscBool              teststate = PETSC_FALSE;
  const PetscScalar      elem[]    = {2.0, -1.0, -1.0, 2.0};

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_state", &teststate, NULL));

  // Subdomain r holds the nl nodes r*(nl-1) ... r*(nl-1)+nl-1, so consecutive subdomains share a node.
  N = size * (nl - 1) + 1;
  PetscCall(PetscMalloc1(nl, &gidx));
  for (PetscInt i = 0; i < nl; i++) gidx[i] = rank * (nl - 1) + i;
  PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD, 1, nl, gidx, PETSC_OWN_POINTER, &l2g));
  PetscCall(MatCreateIS(PETSC_COMM_WORLD, 1, PETSC_DECIDE, PETSC_DECIDE, N, N, l2g, l2g, &A));
  PetscCall(MatSetFromOptions(A));
  PetscCall(ISLocalToGlobalMappingDestroy(&l2g));
  PetscCall(MatISSetPreallocation(A, 3, NULL, 0, NULL));
  for (PetscInt e = 0; e < nl - 1; e++) {
    const PetscInt erows[] = {e, e + 1};

    PetscCall(MatSetValuesLocal(A, 2, erows, 2, erows, elem, ADD_VALUES));
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  // Assembled views built before the change, the way a solver caches them.
  PetscCall(MatConvert(A, MATAIJ, MAT_INITIAL_MATRIX, &Aij));
  PetscCall(MatGetDiagonalBlock(A, &dA));

  // Impose a unit diagonal on one row of every local matrix, keeping the nonzero pattern.
  PetscCall(MatISGetLocalMat(A, &lA));
  PetscCall(MatSetOption(lA, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE));
  PetscCall(MatZeroRows(lA, 1, &row, 1.0, NULL, NULL));
  PetscCall(MatISRestoreLocalMat(A, &lA));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  PetscCall(MatISGetLocalMat(A, &lA));
  PetscCall(DiagonalView(lA, "local (Neumann):"));
  PetscCall(MatISRestoreLocalMat(A, &lA));
  PetscCall(MatConvert(A, MATAIJ, MAT_REUSE_MATRIX, &Aij));
  PetscCall(DiagonalView(Aij, "assembled (MAT_REUSE_MATRIX):"));
  PetscCall(MatDestroy(&Aij));
  PetscCall(MatConvert(A, MATAIJ, MAT_INITIAL_MATRIX, &Aij));
  PetscCall(DiagonalView(Aij, "assembled (MAT_INITIAL_MATRIX):"));
  PetscCall(MatGetDiagonalBlock(A, &dA));
  PetscCall(DiagonalView(dA, "diagonal block:"));

  if (teststate) {
    PetscCall(TestReuse(A));
    PetscCall(TestState(A));
  }

  PetscCall(MatDestroy(&Aij));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 2
      diff_args: -j

   test:
      suffix: state
      nsize: 2
      args: -test_state -mat_is_keepassembled {{0 1}}
      output_file: output/ex321_1.out
      diff_args: -j

TEST*/
