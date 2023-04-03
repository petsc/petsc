/*
   used by MPIAIJ, BAIJ and SBAIJ to reduce code duplication

     define TYPE to AIJ BAIJ or SBAIJ
            TYPE_SBAIJ for SBAIJ matrix

*/

static PetscErrorCode MatSetValues_MPI_Hash(Mat A, PetscInt m, const PetscInt *rows, PetscInt n, const PetscInt *cols, const PetscScalar *values, InsertMode addv)
{
  PetscConcat(Mat_MPI, TYPE) *a = (PetscConcat(Mat_MPI, TYPE) *)A->data;
  PetscInt rStart, rEnd, cStart, cEnd;
#if defined(TYPE_SBAIJ)
  PetscInt bs;
#endif

  PetscFunctionBegin;
  PetscCall(MatGetOwnershipRange(A, &rStart, &rEnd));
  PetscCall(MatGetOwnershipRangeColumn(A, &cStart, &cEnd));
#if defined(TYPE_SBAIJ)
  PetscCall(MatGetBlockSize(A, &bs));
#endif
  for (PetscInt r = 0; r < m; ++r) {
    PetscScalar value;
    if (rows[r] < 0) continue;
    if (rows[r] < rStart || rows[r] >= rEnd) {
      if (a->roworiented) {
        PetscCall(MatStashValuesRow_Private(&A->stash, rows[r], n, cols, values + r * n, PETSC_FALSE));
      } else {
        PetscCall(MatStashValuesCol_Private(&A->stash, rows[r], n, cols, values + r, m, PETSC_FALSE));
      }
    } else {
      for (PetscInt c = 0; c < n; ++c) {
#if defined(TYPE_SBAIJ)
        if (cols[c] / bs < rows[r] / bs) continue;
#else
        if (cols[c] < 0) continue;
#endif
        value = values ? ((a->roworiented) ? values[r * n + c] : values[r + m * c]) : 0;
        if (cols[c] >= cStart && cols[c] < cEnd) PetscCall(MatSetValue(a->A, rows[r] - rStart, cols[c] - cStart, value, addv));
        else PetscCall(MatSetValue(a->B, rows[r] - rStart, cols[c], value, addv));
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatAssemblyBegin_MPI_Hash(Mat A, PETSC_UNUSED MatAssemblyType type)
{
  PetscInt nstash, reallocs;

  PetscFunctionBegin;
  PetscCall(MatStashScatterBegin_Private(A, &A->stash, A->rmap->range));
  PetscCall(MatStashGetInfo_Private(&A->stash, &nstash, &reallocs));
  PetscCall(PetscInfo(A, "Stash has %" PetscInt_FMT " entries, uses %" PetscInt_FMT " mallocs.\n", nstash, reallocs));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatAssemblyEnd_MPI_Hash(Mat A, MatAssemblyType type)
{
  PetscConcat(Mat_MPI, TYPE) *a = (PetscConcat(Mat_MPI, TYPE) *)A->data;
  PetscMPIInt  n;
  PetscScalar *val;
  PetscInt    *row, *col;
  PetscInt     j, ncols, flg, rstart;

  PetscFunctionBegin;
  while (1) {
    PetscCall(MatStashScatterGetMesg_Private(&A->stash, &n, &row, &col, &val, &flg));
    if (!flg) break;

    for (PetscInt i = 0; i < n;) {
      /* Now identify the consecutive vals belonging to the same row */
      for (j = i, rstart = row[j]; j < n; j++) {
        if (row[j] != rstart) break;
      }
      if (j < n) ncols = j - i;
      else ncols = n - i;
      /* Now assemble all these values with a single function call */
      PetscCall(MatSetValues_MPI_Hash(A, 1, row + i, ncols, col + i, val + i, A->insertmode));
      i = j;
    }
  }
  PetscCall(MatStashScatterEnd_Private(&A->stash));
  if (type != MAT_FINAL_ASSEMBLY) PetscFunctionReturn(PETSC_SUCCESS);

  A->insertmode = NOT_SET_VALUES; /* this was set by the previous calls to MatSetValues() */

  PetscCall(PetscMemcpy(&A->ops, &a->cops, sizeof(*(A->ops))));
  A->hash_active = PETSC_FALSE;

  PetscCall(MatAssemblyBegin(a->A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(a->A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(a->B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(a->B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDestroy_MPI_Hash(Mat A)
{
  PetscConcat(Mat_MPI, TYPE) *a = (PetscConcat(Mat_MPI, TYPE) *)A->data;

  PetscFunctionBegin;
  PetscCall(MatStashDestroy_Private(&A->stash));
  PetscCall(MatDestroy(&a->A));
  PetscCall(MatDestroy(&a->B));
  PetscCall((*a->cops.destroy)(A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatZeroEntries_MPI_Hash(PETSC_UNUSED Mat A)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetRandom_MPI_Hash(Mat A, PETSC_UNUSED PetscRandom r)
{
  SETERRQ(PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_WRONGSTATE, "Must set preallocation first");
}

static PetscErrorCode MatSetUp_MPI_Hash(Mat A)
{
  PetscConcat(Mat_MPI, TYPE) *a = (PetscConcat(Mat_MPI, TYPE) *)A->data;
  PetscMPIInt size;
#if !defined(TYPE_AIJ)
  PetscInt bs;
#endif

  PetscFunctionBegin;
  PetscCall(PetscInfo(A, "Using hash-based MatSetValues() for MATMPISBAIJ because no preallocation provided\n"));
  PetscCall(PetscLayoutSetUp(A->rmap));
  PetscCall(PetscLayoutSetUp(A->cmap));
  if (A->rmap->bs < 1) A->rmap->bs = 1;
  if (A->cmap->bs < 1) A->cmap->bs = 1;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)A), &size));

#if !defined(TYPE_AIJ)
  PetscCall(MatGetBlockSize(A, &bs));
  /* these values are set in MatMPISBAIJSetPreallocation() */
  a->bs2 = bs * bs;
  a->mbs = A->rmap->n / bs;
  a->nbs = A->cmap->n / bs;
  a->Mbs = A->rmap->N / bs;
  a->Nbs = A->cmap->N / bs;

  for (PetscInt i = 0; i <= a->size; i++) a->rangebs[i] = A->rmap->range[i] / bs;
  a->rstartbs = A->rmap->rstart / bs;
  a->rendbs   = A->rmap->rend / bs;
  a->cstartbs = A->cmap->rstart / bs;
  a->cendbs   = A->cmap->rend / bs;
  PetscCall(MatStashCreate_Private(PetscObjectComm((PetscObject)A), A->rmap->bs, &A->bstash));
#endif

  PetscCall(MatCreate(PETSC_COMM_SELF, &a->A));
  PetscCall(MatSetSizes(a->A, A->rmap->n, A->cmap->n, A->rmap->n, A->cmap->n));
  PetscCall(MatSetBlockSizesFromMats(a->A, A, A));
#if defined(SUB_TYPE_CUSPARSE)
  PetscCall(MatSetType(a->A, MATSEQAIJCUSPARSE));
#else
  PetscCall(MatSetType(a->A, PetscConcat(MATSEQ, TYPE)));
#endif
  PetscCall(MatSetUp(a->A));

  PetscCall(MatCreate(PETSC_COMM_SELF, &a->B));
  PetscCall(MatSetSizes(a->B, A->rmap->n, size > 1 ? A->cmap->N : 0, A->rmap->n, size > 1 ? A->cmap->N : 0));
  PetscCall(MatSetBlockSizesFromMats(a->B, A, A));
#if defined(TYPE_SBAIJ)
  PetscCall(MatSetType(a->B, MATSEQBAIJ));
#else
  #if defined(SUB_TYPE_CUSPARSE)
  PetscCall(MatSetType(a->B, MATSEQAIJCUSPARSE));
  #else
  PetscCall(MatSetType(a->B, PetscConcat(MATSEQ, TYPE)));
  #endif
#endif
  PetscCall(MatSetUp(a->B));

  /* keep a record of the operations so they can be reset when the hash handling is complete */
  PetscCall(PetscMemcpy(&a->cops, &A->ops, sizeof(*(A->ops))));

  A->ops->assemblybegin    = MatAssemblyBegin_MPI_Hash;
  A->ops->assemblyend      = MatAssemblyEnd_MPI_Hash;
  A->ops->setvalues        = MatSetValues_MPI_Hash;
  A->ops->destroy          = MatDestroy_MPI_Hash;
  A->ops->zeroentries      = MatZeroEntries_MPI_Hash;
  A->ops->setrandom        = MatSetRandom_MPI_Hash;
  A->ops->setvaluesblocked = NULL;

  A->preallocated = PETSC_TRUE;
  A->hash_active  = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}
