
#include <../src/mat/impls/baij/seq/baij.h>

PETSC_INTERN PetscErrorCode MatConvert_SeqBAIJ_SeqAIJ(Mat A, MatType newtype, MatReuse reuse, Mat *newmat)
{
  Mat          B;
  Mat_SeqAIJ  *b;
  PetscBool    roworiented;
  Mat_SeqBAIJ *a  = (Mat_SeqBAIJ *)A->data;
  PetscInt     bs = A->rmap->bs, *ai = a->i, *aj = a->j, n = A->rmap->N / bs, i, j, k;
  PetscInt    *rowlengths, *rows, *cols, maxlen            = 0, ncols;
  MatScalar   *aa = a->a;

  PetscFunctionBegin;
  if (reuse == MAT_REUSE_MATRIX) {
    B = *newmat;
    for (i = 0; i < n; i++) maxlen = PetscMax(maxlen, (ai[i + 1] - ai[i]));
  } else {
    PetscCall(PetscMalloc1(n * bs, &rowlengths));
    for (i = 0; i < n; i++) {
      maxlen = PetscMax(maxlen, (ai[i + 1] - ai[i]));
      for (j = 0; j < bs; j++) rowlengths[i * bs + j] = bs * (ai[i + 1] - ai[i]);
    }
    PetscCall(MatCreate(PetscObjectComm((PetscObject)A), &B));
    PetscCall(MatSetType(B, MATSEQAIJ));
    PetscCall(MatSetSizes(B, A->rmap->n, A->cmap->n, A->rmap->N, A->cmap->N));
    PetscCall(MatSetBlockSizes(B, A->rmap->bs, A->cmap->bs));
    PetscCall(MatSeqAIJSetPreallocation(B, 0, rowlengths));
    PetscCall(PetscFree(rowlengths));
  }
  b           = (Mat_SeqAIJ *)B->data;
  roworiented = b->roworiented;

  PetscCall(MatSetOption(B, MAT_ROW_ORIENTED, PETSC_FALSE));
  PetscCall(PetscMalloc1(bs, &rows));
  PetscCall(PetscMalloc1(bs * maxlen, &cols));
  for (i = 0; i < n; i++) {
    for (j = 0; j < bs; j++) rows[j] = i * bs + j;
    ncols = ai[i + 1] - ai[i];
    for (k = 0; k < ncols; k++) {
      for (j = 0; j < bs; j++) cols[k * bs + j] = bs * (*aj) + j;
      aj++;
    }
    PetscCall(MatSetValues(B, bs, rows, bs * ncols, cols, aa, INSERT_VALUES));
    aa += ncols * bs * bs;
  }
  PetscCall(PetscFree(cols));
  PetscCall(PetscFree(rows));
  PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatSetOption(B, MAT_ROW_ORIENTED, roworiented));

  if (reuse == MAT_INPLACE_MATRIX) {
    PetscCall(MatHeaderReplace(A, &B));
  } else *newmat = B;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#include <../src/mat/impls/aij/seq/aij.h>

PETSC_INTERN PetscErrorCode MatConvert_SeqAIJ_SeqBAIJ_Preallocate(Mat A, PetscInt **nnz)
{
  PetscInt    m, n, bs = PetscAbs(A->rmap->bs);
  PetscInt   *ii;
  Mat_SeqAIJ *Aa = (Mat_SeqAIJ *)A->data;

  PetscFunctionBegin;
  PetscCall(MatGetSize(A, &m, &n));
  PetscCall(PetscCalloc1(m / bs, nnz));
  PetscCall(PetscMalloc1(bs, &ii));

  /* loop over all potential blocks in the matrix to see if the potential block has a nonzero */
  for (PetscInt i = 0; i < m / bs; i++) {
    for (PetscInt k = 0; k < bs; k++) ii[k] = Aa->i[i * bs + k];
    for (PetscInt j = 0; j < n / bs; j++) {
      for (PetscInt k = 0; k < bs; k++) {
        for (; ii[k] < Aa->i[i * bs + k + 1] && Aa->j[ii[k]] < (j + 1) * bs; ii[k]++) {
          if (Aa->j[ii[k]] >= j * bs) {
            /* found a nonzero in the potential block so allocate for that block and jump to check the next potential block */
            (*nnz)[i]++;
            goto theend;
          }
        }
      }
    theend:;
    }
  }
  PetscCall(PetscFree(ii));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatConvert_SeqAIJ_SeqBAIJ(Mat A, MatType newtype, MatReuse reuse, Mat *newmat)
{
  Mat          B;
  Mat_SeqAIJ  *a = (Mat_SeqAIJ *)A->data;
  Mat_SeqBAIJ *b;
  PetscInt     m = A->rmap->N, n = A->cmap->N, *rowlengths, bs = PetscAbs(A->rmap->bs);

  PetscFunctionBegin;
  if (reuse != MAT_REUSE_MATRIX) {
    PetscCall(MatConvert_SeqAIJ_SeqBAIJ_Preallocate(A, &rowlengths));
    PetscCall(MatCreate(PetscObjectComm((PetscObject)A), &B));
    PetscCall(MatSetSizes(B, m, n, m, n));
    PetscCall(MatSetType(B, MATSEQBAIJ));
    PetscCall(MatSeqBAIJSetPreallocation(B, bs, 0, rowlengths));
    PetscCall(PetscFree(rowlengths));
  } else B = *newmat;

  if (bs == 1) {
    b = (Mat_SeqBAIJ *)(B->data);

    PetscCall(PetscArraycpy(b->i, a->i, m + 1));
    PetscCall(PetscArraycpy(b->ilen, a->ilen, m));
    PetscCall(PetscArraycpy(b->j, a->j, a->nz));
    PetscCall(PetscArraycpy(b->a, a->a, a->nz));

    PetscCall(MatSetOption(B, MAT_ROW_ORIENTED, PETSC_TRUE));
    PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
  } else {
    /* reuse may not be equal to MAT_REUSE_MATRIX, but the basic converter will reallocate or replace newmat if this value is not used */
    /* if reuse is equal to MAT_INITIAL_MATRIX, it has been appropriately preallocated before                                          */
    /*                      MAT_INPLACE_MATRIX, it will be replaced with MatHeaderReplace below                                        */
    PetscCall(MatConvert_Basic(A, newtype, MAT_REUSE_MATRIX, &B));
  }

  if (reuse == MAT_INPLACE_MATRIX) {
    PetscCall(MatHeaderReplace(A, &B));
  } else *newmat = B;
  PetscFunctionReturn(PETSC_SUCCESS);
}
