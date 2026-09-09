static char help[] = "Tests that MATSELL honors MAT_NEW_NONZERO_LOCATIONS and MAT_IGNORE_ZERO_ENTRIES.\n\n";

/*
  MatSetValues() must silently ignore an entry that would create a new nonzero location once
  MAT_NEW_NONZERO_LOCATIONS is false, and must ignore a zero value once MAT_IGNORE_ZERO_ENTRIES is
  true. On one process this exercises MatSetValues_SeqSELL(); on two it exercises
  MatSetValues_MPISELL(), probing the diagonal block, an off-diagonal column that garray already
  knows, and an off-diagonal column it does not. A MATAIJ matrix receives the same calls and is
  used as the reference for the resulting values.

  MAT_IGNORE_ZERO_ENTRIES leaves nothing visible behind, so it is probed indirectly: the ignored
  zero is followed by a nonzero write to the same location with MAT_NEW_NONZERO_LOCATIONS false. If
  the zero had wrongly created the location, that second write would land and the values would differ.

  MAT_IGNORE_ZERO_ENTRIES does not apply to the diagonal, which a zero must still create. Two
  diagonal locations are left out of the initial structure so the same indirect probe can be run
  there with the opposite expectation, once with ADD_VALUES and once with INSERT_VALUES, since the
  guards for the two insert modes are separate.

  The last probe repeats the off-diagonal one after MatDisAssemble_MPISELL() has replaced the
  off-diagonal block, which happens once a new off-diagonal column arrives while new locations are
  still allowed. The replacement block must still honor the option.

  CheckDiagonalInOffDiagonalBlock() covers the one case where the two rules meet. The diagonal the
  exception protects is the one MatInvertDiagonalForSOR_SeqSELL() reads, which lives in the
  diagonal block, so only that block exempts it. A matrix whose row and column layouts differ can
  place a global (i,i) in the off-diagonal block instead, where it must be dropped like any other
  zero, and dropped the same way whether or not garray already knows the column.

  CheckOffDiagonalInDiagonalBlock() covers the reverse, which the same layout produces on the other
  rank: a global (i,j) with i != j that lands at a block-local (r,r) of the diagonal block. Whether
  the exemption applies is a question about the global indices, so this zero must be dropped too.
*/

#include <petscmat.h>

static PetscErrorCode CheckEntry(Mat A, Mat B, PetscInt row, PetscInt col, const char *where)
{
  PetscScalar va, vb;

  PetscFunctionBeginUser;
  PetscCall(MatGetValues(A, 1, &row, 1, &col, &va));
  PetscCall(MatGetValues(B, 1, &row, 1, &col, &vb));
  PetscCheck(va == vb, PETSC_COMM_SELF, PETSC_ERR_PLIB, "MATSELL has %g at (%" PetscInt_FMT ",%" PetscInt_FMT ") but MATAIJ has %g: %s", (double)PetscRealPart(va), row, col, (double)PetscRealPart(vb), where);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* set the same entry in both matrices, then reassemble both */
static PetscErrorCode SetAndAssemble(Mat A, Mat B, PetscInt row, PetscInt col, PetscScalar value, InsertMode addv)
{
  PetscFunctionBeginUser;
  PetscCall(MatSetValues(A, 1, &row, 1, &col, &value, addv));
  PetscCall(MatSetValues(B, 1, &row, 1, &col, &value, addv));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* only the owning rank sets the entry, but the assembly is collective */
static PetscErrorCode SetOnOwnerAndAssemble(Mat A, Mat B, PetscBool owner, PetscInt row, PetscInt col, PetscScalar value)
{
  PetscFunctionBeginUser;
  if (owner) {
    PetscCall(MatSetValues(A, 1, &row, 1, &col, &value, INSERT_VALUES));
    PetscCall(MatSetValues(B, 1, &row, 1, &col, &value, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* An 8 by 8 MATSELL and MATAIJ pair over two ranks whose row and column layouts differ: rank 0 owns
   rows 0-3 but only columns 0-1, and rank 1 owns rows 4-7 and columns 2-7. A global (2,2) therefore
   belongs to rank 0's off-diagonal block, while a global (4,2) reaches rank 1 as block-local (0,0)
   of its diagonal block. Neither location is filled here, so both are free for the probes below. */
static PetscErrorCode CreateMismatchedLayoutPair(Mat *A, Mat *B)
{
  PetscInt    i, rstart, rend, ncol, col;
  PetscMPIInt rank;
  PetscScalar value = 1.0;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  ncol = rank ? 6 : 2;

  PetscCall(MatCreate(PETSC_COMM_WORLD, A));
  PetscCall(MatSetSizes(*A, 4, ncol, 8, 8));
  PetscCall(MatSetType(*A, MATSELL));
  PetscCall(MatSetFromOptions(*A));
  PetscCall(MatSeqSELLSetPreallocation(*A, 8, NULL));
  PetscCall(MatMPISELLSetPreallocation(*A, 8, NULL, 8, NULL));

  PetscCall(MatCreate(PETSC_COMM_WORLD, B));
  PetscCall(MatSetSizes(*B, 4, ncol, 8, 8));
  PetscCall(MatSetType(*B, MATAIJ));
  PetscCall(MatSeqAIJSetPreallocation(*B, 8, NULL));
  PetscCall(MatMPIAIJSetPreallocation(*B, 8, NULL, 8, NULL));

  PetscCall(MatGetOwnershipRange(*A, &rstart, &rend));
  for (i = rstart; i < rend; i++) {
    col = (i + 4) % 8;
    PetscCall(MatSetValues(*A, 1, &i, 1, &col, &value, INSERT_VALUES));
    PetscCall(MatSetValues(*B, 1, &i, 1, &col, &value, INSERT_VALUES));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* A zero at a global (i,i) that the column layout puts in the off-diagonal block must be dropped,
   not exempted. ingarray selects which of the two suppression paths in MatSetValues_MPISELL() the
   probe reaches; both must reach the same answer, and it must be the MATAIJ answer. */
static PetscErrorCode CheckDiagonalInOffDiagonalBlock(PetscBool ingarray)
{
  Mat         A, B;
  PetscInt    i, probe = 2;
  PetscMPIInt rank;
  PetscBool   owner;
  PetscScalar value = 1.0;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  owner = (PetscBool)(rank == 0);
  PetscCall(CreateMismatchedLayoutPair(&A, &B));
  if (ingarray && owner) {
    i = 0;
    PetscCall(MatSetValues(A, 1, &i, 1, &probe, &value, INSERT_VALUES));
    PetscCall(MatSetValues(B, 1, &i, 1, &probe, &value, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));

  /* new locations stay allowed, or nonew would suppress the insert before the zero test is reached */
  PetscCall(MatSetOption(A, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE));
  PetscCall(MatSetOption(B, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE));
  PetscCall(SetOnOwnerAndAssemble(A, B, owner, probe, probe, 0.0));
  PetscCall(MatSetOption(A, MAT_IGNORE_ZERO_ENTRIES, PETSC_FALSE));
  PetscCall(MatSetOption(B, MAT_IGNORE_ZERO_ENTRIES, PETSC_FALSE));
  PetscCall(MatSetOption(A, MAT_NEW_NONZERO_LOCATIONS, PETSC_FALSE));
  PetscCall(MatSetOption(B, MAT_NEW_NONZERO_LOCATIONS, PETSC_FALSE));
  PetscCall(SetOnOwnerAndAssemble(A, B, owner, probe, probe, 5.0));
  if (owner) PetscCall(CheckEntry(A, B, probe, probe, ingarray ? "global diagonal in the off-diagonal block, column in garray" : "global diagonal in the off-diagonal block, column not in garray"));

  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* The mirror case. A global (4,2) is not on the diagonal, but rank 1 owns rows from 4 and columns
   from 2, so it arrives at block-local (0,0) of the diagonal block. The exemption belongs to the
   global diagonal, the one MatInvertDiagonalForSOR_SeqAIJ() and MatInvertDiagonalForSOR_SeqSELL()
   read, so a zero here must be dropped like any other. */
static PetscErrorCode CheckOffDiagonalInDiagonalBlock(void)
{
  Mat         A, B;
  PetscInt    prow = 4, pcol = 2;
  PetscMPIInt rank;
  PetscBool   owner;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  owner = (PetscBool)(rank == 1);
  PetscCall(CreateMismatchedLayoutPair(&A, &B));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));

  /* new locations stay allowed, or nonew would suppress the insert before the zero test is reached */
  PetscCall(MatSetOption(A, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE));
  PetscCall(MatSetOption(B, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE));
  PetscCall(SetOnOwnerAndAssemble(A, B, owner, prow, pcol, 0.0));
  PetscCall(MatSetOption(A, MAT_IGNORE_ZERO_ENTRIES, PETSC_FALSE));
  PetscCall(MatSetOption(B, MAT_IGNORE_ZERO_ENTRIES, PETSC_FALSE));
  PetscCall(MatSetOption(A, MAT_NEW_NONZERO_LOCATIONS, PETSC_FALSE));
  PetscCall(MatSetOption(B, MAT_NEW_NONZERO_LOCATIONS, PETSC_FALSE));
  PetscCall(SetOnOwnerAndAssemble(A, B, owner, prow, pcol, 5.0));
  if (owner) PetscCall(CheckEntry(A, B, prow, pcol, "global off-diagonal at a block-local diagonal position"));

  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **args)
{
  Mat         A, B;
  PetscInt    i, rstart, rend, col, known, unknown, rebuilt, dadd, dins;
  PetscMPIInt rank, size;
  PetscScalar value = 1.0;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCheck(size == 1 || size == 2, PETSC_COMM_WORLD, PETSC_ERR_USER, "This test requires 1 or 2 processes");

  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, 4, 4, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(MatSetType(A, MATSELL));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSeqSELLSetPreallocation(A, 4, NULL));
  PetscCall(MatMPISELLSetPreallocation(A, 4, NULL, 4, NULL));

  PetscCall(MatCreate(PETSC_COMM_WORLD, &B));
  PetscCall(MatSetSizes(B, 4, 4, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(MatSetType(B, MATAIJ));
  PetscCall(MatSeqAIJSetPreallocation(B, 4, NULL));
  PetscCall(MatMPIAIJSetPreallocation(B, 4, NULL, 4, NULL));

  PetscCall(MatGetOwnershipRange(A, &rstart, &rend));
  /* known enters garray during the first assembly; unknown never does */
  known   = rank ? 0 : 4;
  unknown = rank ? 1 : 5;
  /* rebuilt is the off-diagonal column whose arrival forces MatDisAssemble_MPISELL() */
  rebuilt = rank ? 2 : 6;
  /* the diagonal is filled except at these two rows, which the zero-value probes below must create */
  dadd = rstart + 2;
  dins = rstart + 3;

  for (i = rstart; i < rend; i++) {
    if (i == dadd || i == dins) continue;
    PetscCall(MatSetValues(A, 1, &i, 1, &i, &value, INSERT_VALUES));
    PetscCall(MatSetValues(B, 1, &i, 1, &i, &value, INSERT_VALUES));
  }
  if (size > 1) {
    PetscCall(MatSetValues(A, 1, &rstart, 1, &known, &value, INSERT_VALUES));
    PetscCall(MatSetValues(B, 1, &rstart, 1, &known, &value, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));

  /* from here on, a new nonzero location must be ignored rather than created */
  PetscCall(MatSetOption(A, MAT_NEW_NONZERO_LOCATIONS, PETSC_FALSE));
  PetscCall(MatSetOption(B, MAT_NEW_NONZERO_LOCATIONS, PETSC_FALSE));

  i = rstart + 1;
  PetscCall(SetAndAssemble(A, B, i, rstart + 2, 7.0, INSERT_VALUES)); /* diagonal block */
  PetscCall(CheckEntry(A, B, i, rstart + 2, "new location in the diagonal block"));
  if (size > 1) {
    PetscCall(SetAndAssemble(A, B, i, known, 7.0, INSERT_VALUES)); /* off-diagonal, column in garray */
    PetscCall(CheckEntry(A, B, i, known, "new location in an off-diagonal column garray knows"));
    PetscCall(SetAndAssemble(A, B, i, unknown, 7.0, INSERT_VALUES)); /* off-diagonal, column not in garray */
    PetscCall(CheckEntry(A, B, i, unknown, "new location in an off-diagonal column garray does not know"));
  }

  /* an existing location must still be writable */
  PetscCall(SetAndAssemble(A, B, i, i, 9.0, INSERT_VALUES));
  PetscCall(CheckEntry(A, B, i, i, "overwrite of an existing diagonal entry"));

  /* a zero value must not create a location either */
  PetscCall(MatSetOption(A, MAT_NEW_NONZERO_LOCATIONS, PETSC_TRUE));
  PetscCall(MatSetOption(B, MAT_NEW_NONZERO_LOCATIONS, PETSC_TRUE));
  PetscCall(MatSetOption(A, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE));
  PetscCall(MatSetOption(B, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE));
  col = rstart + 3;
  PetscCall(SetAndAssemble(A, B, i, col, 0.0, INSERT_VALUES));
  PetscCall(MatSetOption(A, MAT_NEW_NONZERO_LOCATIONS, PETSC_FALSE));
  PetscCall(MatSetOption(B, MAT_NEW_NONZERO_LOCATIONS, PETSC_FALSE));
  PetscCall(SetAndAssemble(A, B, i, col, 5.0, INSERT_VALUES));
  PetscCall(CheckEntry(A, B, i, col, "location that an ignored zero must not have created"));

  /* but a zero on the diagonal must create its location, for both insert modes */
  PetscCall(MatSetOption(A, MAT_NEW_NONZERO_LOCATIONS, PETSC_TRUE));
  PetscCall(MatSetOption(B, MAT_NEW_NONZERO_LOCATIONS, PETSC_TRUE));
  PetscCall(SetAndAssemble(A, B, dadd, dadd, 0.0, ADD_VALUES));
  PetscCall(SetAndAssemble(A, B, dins, dins, 0.0, INSERT_VALUES));
  PetscCall(MatSetOption(A, MAT_NEW_NONZERO_LOCATIONS, PETSC_FALSE));
  PetscCall(MatSetOption(B, MAT_NEW_NONZERO_LOCATIONS, PETSC_FALSE));
  PetscCall(SetAndAssemble(A, B, dadd, dadd, 5.0, INSERT_VALUES));
  PetscCall(CheckEntry(A, B, dadd, dadd, "diagonal location that a zero added under MAT_IGNORE_ZERO_ENTRIES must have created"));
  PetscCall(SetAndAssemble(A, B, dins, dins, 5.0, INSERT_VALUES));
  PetscCall(CheckEntry(A, B, dins, dins, "diagonal location that a zero inserted under MAT_IGNORE_ZERO_ENTRIES must have created"));

  /* the same off-diagonal probe, but against the block MatDisAssemble_MPISELL() builds. New
     locations must be allowed for the zero to reach the rebuilt block at all, since a column that
     garray does not know is what triggers the disassembly. */
  if (size > 1) {
    PetscCall(MatSetOption(A, MAT_NEW_NONZERO_LOCATIONS, PETSC_TRUE));
    PetscCall(MatSetOption(B, MAT_NEW_NONZERO_LOCATIONS, PETSC_TRUE));
    PetscCall(SetAndAssemble(A, B, i, rebuilt, 0.0, INSERT_VALUES));
    PetscCall(MatSetOption(A, MAT_NEW_NONZERO_LOCATIONS, PETSC_FALSE));
    PetscCall(MatSetOption(B, MAT_NEW_NONZERO_LOCATIONS, PETSC_FALSE));
    PetscCall(SetAndAssemble(A, B, i, rebuilt, 5.0, INSERT_VALUES));
    PetscCall(CheckEntry(A, B, i, rebuilt, "off-diagonal location that an ignored zero must not have created in the rebuilt block"));
  }

  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));

  if (size > 1) {
    PetscCall(CheckDiagonalInOffDiagonalBlock(PETSC_TRUE));
    PetscCall(CheckDiagonalInOffDiagonalBlock(PETSC_FALSE));
    PetscCall(CheckOffDiagonalInDiagonalBlock());
  }
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   testset:
      output_file: output/empty.out

      test:
         suffix: 1
         nsize: {{1 2}}
         args: -mat_type sell

      test:
         suffix: cuda
         nsize: {{1 2}}
         requires: cuda !complex
         args: -mat_type sellcuda

      test:
         suffix: hip
         nsize: {{1 2}}
         requires: hip !complex
         args: -mat_type sellhip

TEST*/
