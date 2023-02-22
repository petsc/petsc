
/*
  Defines transpose routines for SeqAIJ matrices.
*/

#include <../src/mat/impls/aij/seq/aij.h>

/*
   The symbolic and full transpose versions share several similar code blocks but the macros to reuse the code would be confusing and ugly
*/
PetscErrorCode MatTransposeSymbolic_SeqAIJ(Mat A, Mat *B)
{
  PetscInt    i, j, anzj;
  Mat         At;
  Mat_SeqAIJ *a  = (Mat_SeqAIJ *)A->data, *at;
  PetscInt    an = A->cmap->N, am = A->rmap->N;
  PetscInt   *ati, *atj, *atfill, *ai = a->i, *aj = a->j;

  PetscFunctionBegin;
  /* Allocate space for symbolic transpose info and work array */
  PetscCall(PetscCalloc1(an + 1, &ati));
  PetscCall(PetscMalloc1(ai[am], &atj));

  /* Walk through aj and count ## of non-zeros in each row of A^T. */
  /* Note: offset by 1 for fast conversion into csr format. */
  for (i = 0; i < ai[am]; i++) ati[aj[i] + 1] += 1;
  /* Form ati for csr format of A^T. */
  for (i = 0; i < an; i++) ati[i + 1] += ati[i];

  /* Copy ati into atfill so we have locations of the next free space in atj */
  PetscCall(PetscMalloc1(an, &atfill));
  PetscCall(PetscArraycpy(atfill, ati, an));

  /* Walk through A row-wise and mark nonzero entries of A^T. */
  for (i = 0; i < am; i++) {
    anzj = ai[i + 1] - ai[i];
    for (j = 0; j < anzj; j++) {
      atj[atfill[*aj]] = i;
      atfill[*aj++] += 1;
    }
  }
  PetscCall(PetscFree(atfill));

  PetscCall(MatCreateSeqAIJWithArrays(PetscObjectComm((PetscObject)A), an, am, ati, atj, NULL, &At));
  PetscCall(MatSetBlockSizes(At, PetscAbs(A->cmap->bs), PetscAbs(A->rmap->bs)));
  PetscCall(MatSetType(At, ((PetscObject)A)->type_name));
  at          = (Mat_SeqAIJ *)At->data;
  at->free_a  = PETSC_FALSE;
  at->free_ij = PETSC_TRUE;
  at->nonew   = 0;
  at->maxnz   = ati[an];
  *B          = At;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatTranspose_SeqAIJ(Mat A, MatReuse reuse, Mat *B)
{
  PetscInt         i, j, anzj;
  Mat              At;
  Mat_SeqAIJ      *a  = (Mat_SeqAIJ *)A->data, *at;
  PetscInt         an = A->cmap->N, am = A->rmap->N;
  PetscInt        *ati, *atj, *atfill, *ai = a->i, *aj = a->j;
  MatScalar       *ata;
  const MatScalar *aa, *av;
  PetscContainer   rB;
  MatParentState  *rb;
  PetscBool        nonzerochange = PETSC_FALSE;

  PetscFunctionBegin;
  if (reuse == MAT_REUSE_MATRIX) {
    PetscCall(PetscObjectQuery((PetscObject)*B, "MatTransposeParent", (PetscObject *)&rB));
    PetscCheck(rB, PetscObjectComm((PetscObject)*B), PETSC_ERR_ARG_WRONG, "Reuse matrix used was not generated from call to MatTranspose()");
    PetscCall(PetscContainerGetPointer(rB, (void **)&rb));
    if (rb->nonzerostate != A->nonzerostate) nonzerochange = PETSC_TRUE;
  }

  PetscCall(MatSeqAIJGetArrayRead(A, &av));
  aa = av;
  if (reuse == MAT_INITIAL_MATRIX || reuse == MAT_INPLACE_MATRIX || nonzerochange) {
    /* Allocate space for symbolic transpose info and work array */
    PetscCall(PetscCalloc1(an + 1, &ati));
    PetscCall(PetscMalloc1(ai[am], &atj));
    /* Walk through aj and count ## of non-zeros in each row of A^T. */
    /* Note: offset by 1 for fast conversion into csr format. */
    for (i = 0; i < ai[am]; i++) ati[aj[i] + 1] += 1;
    /* Form ati for csr format of A^T. */
    for (i = 0; i < an; i++) ati[i + 1] += ati[i];
    PetscCall(PetscMalloc1(ai[am], &ata));
  } else {
    Mat_SeqAIJ *sub_B = (Mat_SeqAIJ *)(*B)->data;
    ati               = sub_B->i;
    atj               = sub_B->j;
    ata               = sub_B->a;
    At                = *B;
  }

  /* Copy ati into atfill so we have locations of the next free space in atj */
  PetscCall(PetscMalloc1(an, &atfill));
  PetscCall(PetscArraycpy(atfill, ati, an));

  /* Walk through A row-wise and mark nonzero entries of A^T. */
  for (i = 0; i < am; i++) {
    anzj = ai[i + 1] - ai[i];
    for (j = 0; j < anzj; j++) {
      atj[atfill[*aj]] = i;
      ata[atfill[*aj]] = *aa++;
      atfill[*aj++] += 1;
    }
  }
  PetscCall(PetscFree(atfill));
  PetscCall(MatSeqAIJRestoreArrayRead(A, &av));
  if (reuse == MAT_REUSE_MATRIX) PetscCall(PetscObjectStateIncrease((PetscObject)(*B)));

  if (reuse == MAT_INITIAL_MATRIX || reuse == MAT_INPLACE_MATRIX || nonzerochange) {
    PetscCall(MatCreateSeqAIJWithArrays(PetscObjectComm((PetscObject)A), an, am, ati, atj, ata, &At));
    PetscCall(MatSetBlockSizes(At, PetscAbs(A->cmap->bs), PetscAbs(A->rmap->bs)));
    PetscCall(MatSetType(At, ((PetscObject)A)->type_name));
    at          = (Mat_SeqAIJ *)(At->data);
    at->free_a  = PETSC_TRUE;
    at->free_ij = PETSC_TRUE;
    at->nonew   = 0;
    at->maxnz   = ati[an];
  }

  if (reuse == MAT_INITIAL_MATRIX || (reuse == MAT_REUSE_MATRIX && !nonzerochange)) {
    *B = At;
  } else if (nonzerochange) {
    PetscCall(MatHeaderMerge(*B, &At));
    PetscCall(MatTransposeSetPrecursor(A, *B));
  } else if (reuse == MAT_INPLACE_MATRIX) {
    PetscCall(MatHeaderMerge(A, &At));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   Get symbolic matrix structure of a submatrix of A, A[rstart:rend,:],
*/
PetscErrorCode MatGetSymbolicTransposeReduced_SeqAIJ(Mat A, PetscInt rstart, PetscInt rend, PetscInt *Ati[], PetscInt *Atj[])
{
  PetscInt    i, j, anzj;
  Mat_SeqAIJ *a  = (Mat_SeqAIJ *)A->data;
  PetscInt    an = A->cmap->N;
  PetscInt   *ati, *atj, *atfill, *ai = a->i, *aj = a->j, am = ai[rend] - ai[rstart];

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(MAT_Getsymtransreduced, A, 0, 0, 0));

  /* Allocate space for symbolic transpose info and work array */
  PetscCall(PetscCalloc1(an + 1, &ati));
  PetscCall(PetscMalloc1(am + 1, &atj));

  /* Walk through aj and count ## of non-zeros in each row of A^T. */
  /* Note: offset by 1 for fast conversion into csr format. */
  for (i = ai[rstart]; i < ai[rend]; i++) ati[aj[i] + 1] += 1;
  /* Form ati for csr format of A^T. */
  for (i = 0; i < an; i++) ati[i + 1] += ati[i];

  /* Copy ati into atfill so we have locations of the next free space in atj */
  PetscCall(PetscMalloc1(an + 1, &atfill));
  PetscCall(PetscArraycpy(atfill, ati, an));

  /* Walk through A row-wise and mark nonzero entries of A^T. */
  aj = aj + ai[rstart];
  for (i = rstart; i < rend; i++) {
    anzj = ai[i + 1] - ai[i];
    for (j = 0; j < anzj; j++) {
      atj[atfill[*aj]] = i - rstart;
      atfill[*aj++] += 1;
    }
  }
  PetscCall(PetscFree(atfill));
  *Ati = ati;
  *Atj = atj;

  PetscCall(PetscLogEventEnd(MAT_Getsymtransreduced, A, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
    Returns the i and j arrays for a symbolic transpose, this is used internally within SeqAIJ code when the full
    symbolic matrix (which can be obtained with MatTransposeSymbolic() is not needed. MatRestoreSymbolicTranspose_SeqAIJ() should be used to free the arrays.
*/
PetscErrorCode MatGetSymbolicTranspose_SeqAIJ(Mat A, PetscInt *Ati[], PetscInt *Atj[])
{
  PetscFunctionBegin;
  PetscCall(MatGetSymbolicTransposeReduced_SeqAIJ(A, 0, A->rmap->N, Ati, Atj));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatRestoreSymbolicTranspose_SeqAIJ(Mat A, PetscInt *ati[], PetscInt *atj[])
{
  PetscFunctionBegin;
  PetscCall(PetscFree(*ati));
  PetscCall(PetscFree(*atj));
  PetscFunctionReturn(PETSC_SUCCESS);
}
