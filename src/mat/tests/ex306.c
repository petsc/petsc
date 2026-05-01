static char help[] = "Regression test for the MatSOR_SeqAIJ_Inode() block-diagonal cache.\n\n\
   This test catches the typo in MatInvertDiagonalForSOR_SeqAIJ_Inode() introduced\n\
   in MR !8797 where the cache validity check at inode.c:2432 reads a->idiagState\n\
   (the point-SOR cache token) instead of a->inode.ibdiagState (the inode block-SOR\n\
   cache token).  With the bug, the inverted block diagonal is recomputed on every\n\
   MatSOR() call, producing a runtime regression but not a correctness failure --\n\
   so we observe the cache invariant directly by poisoning the cached buffer and\n\
   checking whether it survives a second MatSOR() call on an unchanged matrix.\n\n";

#include <petscmat.h>

/* The bug lives in private inode-cache fields.  Including the private header is
   the cleanest way to test the cache invariant; this is the established practice
   for tests that target a specific implementation detail rather than public API. */
#include <../src/mat/impls/aij/seq/aij.h>

int main(int argc, char **args)
{
  Mat         A;
  Vec         b, x_pristine, x_poisoned;
  Mat_SeqAIJ *a;
  PetscInt    nblock = 20, blocksize = 3, n;
  PetscBool   same;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  n = nblock * blocksize;

  PetscCall(MatCreate(PETSC_COMM_SELF, &A));
  PetscCall(MatSetSizes(A, n, n, n, n));
  PetscCall(MatSetType(A, MATSEQAIJ));
  PetscCall(MatSeqAIJSetPreallocation(A, 3 * blocksize, NULL));

  /* Block-tridiagonal matrix: each row has one diagonal entry of value 4.0
     and -1.0 entries at every other column in its own block and the two
     neighbouring blocks.  All `blocksize` rows of a block share the same column
     pattern, so MatSeqAIJCheckInode() forms `nblock` inodes of size `blocksize`. */
  for (PetscInt Ib = 0; Ib < nblock; Ib++) {
    PetscInt    Jlo = PetscMax(Ib - 1, 0), Jhi = PetscMin(Ib + 1, nblock - 1);
    PetscInt    ncols = (Jhi - Jlo + 1) * blocksize, cols[9];
    PetscScalar vals[9];

    for (PetscInt Jb = Jlo, k = 0; Jb <= Jhi; Jb++)
      for (PetscInt jj = 0; jj < blocksize; jj++, k++) cols[k] = Jb * blocksize + jj;

    for (PetscInt ii = 0; ii < blocksize; ii++) {
      PetscInt row = Ib * blocksize + ii;

      for (PetscInt k = 0; k < ncols; k++) vals[k] = (cols[k] == row) ? 4.0 : -1.0;
      PetscCall(MatSetValues(A, 1, &row, ncols, cols, vals, INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  PetscCall(MatCreateVecs(A, &b, &x_pristine));
  PetscCall(VecDuplicate(x_pristine, &x_poisoned));
  PetscCall(VecSet(b, 1.0));

  /* First MatSOR() call: triggers MatInvertDiagonalForSOR_SeqAIJ_Inode() and
     populates a->inode.ibdiag with correctly inverted blocks. */
  PetscCall(MatSOR(A, b, 1.0, SOR_FORWARD_SWEEP, 0.0, 1, 1, x_pristine));

  /* Confirm we actually reached the inode SOR path -- otherwise the test is
     vacuous and would silently pass against a buggy build. */
  a = (Mat_SeqAIJ *)A->data;
  PetscCheck(a->inode.use && a->inode.size_csr && a->inode.node_count > 0 && a->inode.ibdiag != NULL, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Inode SOR path was not exercised (use=%d node_count=%" PetscInt_FMT " ibdiag=%p); test cannot detect the bug.",
             (int)a->inode.use, a->inode.node_count, (void *)a->inode.ibdiag);

  /* Poison the cached inverted block diagonal.  A is unchanged, so its
     PetscObject state does not advance and the cache *should* be honoured by
     the next MatSOR() call.  A buggy implementation that rebuilds ibdiag on
     every call will overwrite our zeros with the correct inverse and recover
     x_pristine.  A correct implementation will use the zeros and produce a
     materially different x_poisoned. */
  for (PetscInt i = 0; i < a->inode.bdiagsize; i++) a->inode.ibdiag[i] = 0.0;

  PetscCall(MatSOR(A, b, 1.0, SOR_FORWARD_SWEEP, 0.0, 1, 1, x_poisoned));

  PetscCall(VecEqual(x_pristine, x_poisoned, &same));
  PetscCheck(!same, PETSC_COMM_SELF, PETSC_ERR_PLIB, "MatSOR_SeqAIJ_Inode() rebuilt its cached block diagonal despite an unchanged matrix state. The cache validity check in MatInvertDiagonalForSOR_SeqAIJ_Inode() is comparing the wrong PetscObjectState field (regression of MR !8797).");

  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&x_pristine));
  PetscCall(VecDestroy(&x_poisoned));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
     suffix: 0
     args: -mat_no_inode false
     output_file: output/empty.out

TEST*/
