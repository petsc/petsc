/*
   used by SEQAIJ, BAIJ and SBAIJ to reduce code duplication

     define TYPE to AIJ BAIJ or SBAIJ
            TYPE_BS_ON for BAIJ and SBAIJ and bs > 1
            TYPE_SBAIJ for SBAIJ

*/
static PetscErrorCode PetscConcat(MatSetValues_Seq_Hash, TYPE_BS)(Mat A, PetscInt m, const PetscInt *rows, PetscInt n, const PetscInt *cols, const PetscScalar *values, InsertMode addv)
{
  PetscConcat(Mat_Seq, TYPE) *a = (PetscConcat(Mat_Seq, TYPE) *)A->data;
#if defined(TYPE_BS_ON)
  PetscInt bs;
#endif

  PetscFunctionBegin;
#if defined(TYPE_BS_ON)
  PetscCall(MatGetBlockSize(A, &bs));
#endif
  for (PetscInt r = 0; r < m; ++r) {
    PetscHashIJKey key;
    PetscBool      missing;
    PetscScalar    value;
#if defined(TYPE_BS_ON)
    PetscHashIJKey bkey;
#endif

    key.i = rows[r];
#if defined(TYPE_BS_ON)
    bkey.i = key.i / bs;
#endif
    if (key.i < 0) continue;
    for (PetscInt c = 0; c < n; ++c) {
      key.j = cols[c];
#if defined(TYPE_BS_ON)
      bkey.j = key.j / bs;
  #if defined(TYPE_SBAIJ)
      if (bkey.j < bkey.i) continue;
  #else
      if (key.j < 0) continue;
  #endif
#else
  #if defined(TYPE_SBAIJ)
      if (key.j < key.i) continue;
  #else
      if (key.j < 0) continue;
  #endif
#endif
      value = values ? ((a->roworiented) ? values[r * n + c] : values[r + m * c]) : 0;
      switch (addv) {
      case INSERT_VALUES:
        PetscCall(PetscHMapIJVQuerySet(a->ht, key, value, &missing));
        break;
      case ADD_VALUES:
        PetscCall(PetscHMapIJVQueryAdd(a->ht, key, value, &missing));
        break;
      default:
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "InsertMode not supported");
      }
      if (missing) ++a->dnz[key.i];
#if defined(TYPE_BS_ON)
      PetscCall(PetscHSetIJQueryAdd(a->bht, bkey, &missing));
      if (missing) ++a->bdnz[bkey.i];
#endif
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
