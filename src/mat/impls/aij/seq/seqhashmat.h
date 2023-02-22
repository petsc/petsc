/*
   used by SEQAIJ, BAIJ and SBAIJ to reduce code duplication

     define TYPE to AIJ BAIJ or SBAIJ
            TYPE_BS_ON for BAIJ and SBAIJ

*/
static PetscErrorCode MatAssemblyEnd_Seq_Hash(Mat A, MatAssemblyType type)
{
  PetscConcat(Mat_Seq, TYPE) *a = (PetscConcat(Mat_Seq, TYPE) *)A->data;
  PetscHashIter  hi;
  PetscHashIJKey key;
  PetscScalar    value, *values;
  PetscInt       m, n, *cols, *rowstarts;
#if defined(TYPE_BS_ON)
  PetscInt bs;
#endif

  PetscFunctionBegin;
#if defined(TYPE_BS_ON)
  PetscCall(MatGetBlockSize(A, &bs));
  if (bs > 1) PetscCall(PetscHSetIJDestroy(&a->bht));
#endif
  A->preallocated = PETSC_FALSE; /* this was set to true for the MatSetValues_Hash() to work */

  PetscCall(PetscMemcpy(&A->ops, &a->cops, sizeof(*(A->ops))));
  A->hash_active = PETSC_FALSE;

  /* move values from hash format to matrix type format */
  PetscCall(MatGetSize(A, &m, NULL));
#if defined(TYPE_BS_ON)
  if (bs > 1) PetscCall(PetscConcat(PetscConcat(MatSeq, TYPE), SetPreallocation)(A, bs, PETSC_DETERMINE, a->bdnz));
  else PetscCall(PetscConcat(PetscConcat(MatSeq, TYPE), SetPreallocation)(A, 1, PETSC_DETERMINE, a->dnz));
#else
  PetscCall(MatSeqAIJSetPreallocation(A, PETSC_DETERMINE, a->dnz));
#endif
  PetscCall(MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  PetscCall(PetscHMapIJVGetSize(a->ht, &n));
  PetscCall(PetscMalloc3(n, &cols, m + 1, &rowstarts, n, &values));
  rowstarts[0] = 0;
  for (PetscInt i = 0; i < m; i++) rowstarts[i + 1] = rowstarts[i] + a->dnz[i];

  PetscHashIterBegin(a->ht, hi);
  while (!PetscHashIterAtEnd(a->ht, hi)) {
    PetscHashIterGetKey(a->ht, hi, key);
    PetscHashIterGetVal(a->ht, hi, value);
    cols[rowstarts[key.i]]     = key.j;
    values[rowstarts[key.i]++] = value;
    PetscHashIterNext(a->ht, hi);
  }
  PetscCall(PetscHMapIJVDestroy(&a->ht));

  for (PetscInt i = 0, start = 0; i < m; i++) {
    PetscCall(MatSetValues(A, 1, &i, a->dnz[i], &cols[start], &values[start], A->insertmode));
    start += a->dnz[i];
  }
  PetscCall(PetscFree3(cols, rowstarts, values));
  PetscCall(PetscFree(a->dnz));
#if defined(TYPE_BS_ON)
  if (bs > 1) PetscCall(PetscFree(a->bdnz));
#endif
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDestroy_Seq_Hash(Mat A)
{
  PetscConcat(Mat_Seq, TYPE) *a = (PetscConcat(Mat_Seq, TYPE) *)A->data;
#if defined(TYPE_BS_ON)
  PetscInt bs;
#endif

  PetscFunctionBegin;
  PetscCall(PetscHMapIJVDestroy(&a->ht));
  PetscCall(PetscFree(a->dnz));
#if defined(TYPE_BS_ON)
  PetscCall(MatGetBlockSize(A, &bs));
  if (bs > 1) {
    PetscCall(PetscFree(a->bdnz));
    PetscCall(PetscHSetIJDestroy(&a->bht));
  }
#endif
  PetscCall((*a->cops.destroy)(A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatZeroEntries_Seq_Hash(Mat A)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetRandom_Seq_Hash(Mat A, PetscRandom r)
{
  PetscFunctionBegin;
  SETERRQ(PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_WRONGSTATE, "Must set preallocation first");
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetUp_Seq_Hash(Mat A)
{
  PetscConcat(Mat_Seq, TYPE) *a = (PetscConcat(Mat_Seq, TYPE) *)A->data;
  PetscInt m;
#if defined(TYPE_BS_ON)
  PetscInt bs;
#endif

  PetscFunctionBegin;
  PetscCall(PetscInfo(A, "Using hash-based MatSetValues() for MATSEQAIJ because no preallocation provided\n"));
  PetscCall(PetscLayoutSetUp(A->rmap));
  PetscCall(PetscLayoutSetUp(A->cmap));
  if (A->rmap->bs < 1) A->rmap->bs = 1;
  if (A->cmap->bs < 1) A->cmap->bs = 1;

  PetscCall(MatGetLocalSize(A, &m, NULL));
  PetscCall(PetscHMapIJVCreate(&a->ht));
  PetscCall(PetscCalloc1(m, &a->dnz));
#if defined(TYPE_BS_ON)
  PetscCall(MatGetBlockSize(A, &bs));
  if (bs > 1) {
    PetscCall(PetscCalloc1(m / bs, &a->bdnz));
    PetscCall(PetscHSetIJCreate(&a->bht));
  }
#endif

  /* keep a record of the operations so they can be reset when the hash handling is complete */
  PetscCall(PetscMemcpy(&a->cops, &A->ops, sizeof(*(A->ops))));

  A->ops->assemblybegin = NULL;
  A->ops->assemblyend   = MatAssemblyEnd_Seq_Hash;
  A->ops->destroy       = MatDestroy_Seq_Hash;
  A->ops->zeroentries   = MatZeroEntries_Seq_Hash;
  A->ops->setrandom     = MatSetRandom_Seq_Hash;
#if defined(TYPE_BS_ON)
  if (bs > 1) A->ops->setvalues = MatSetValues_Seq_Hash_BS;
  else
#endif
    A->ops->setvalues = MatSetValues_Seq_Hash;
  A->ops->setvaluesblocked = NULL;

  A->preallocated = PETSC_TRUE;
  A->hash_active  = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}
