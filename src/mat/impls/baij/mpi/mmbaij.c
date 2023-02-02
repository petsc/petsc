
/*
   Support for the parallel BAIJ matrix vector multiply
*/
#include <../src/mat/impls/baij/mpi/mpibaij.h>
#include <petsc/private/isimpl.h> /* needed because accesses data structure of ISLocalToGlobalMapping directly */

PetscErrorCode MatSetUpMultiply_MPIBAIJ(Mat mat)
{
  Mat_MPIBAIJ *baij = (Mat_MPIBAIJ *)mat->data;
  Mat_SeqBAIJ *B    = (Mat_SeqBAIJ *)(baij->B->data);
  PetscInt     i, j, *aj = B->j, ec = 0, *garray;
  PetscInt     bs = mat->rmap->bs, *stmp;
  IS           from, to;
  Vec          gvec;
#if defined(PETSC_USE_CTABLE)
  PetscHMapI    gid1_lid1 = NULL;
  PetscHashIter tpos;
  PetscInt      gid, lid;
#else
  PetscInt Nbs = baij->Nbs, *indices;
#endif

  PetscFunctionBegin;
#if defined(PETSC_USE_CTABLE)
  /* use a table - Mark Adams */
  PetscCall(PetscHMapICreateWithSize(B->mbs, &gid1_lid1));
  for (i = 0; i < B->mbs; i++) {
    for (j = 0; j < B->ilen[i]; j++) {
      PetscInt data, gid1 = aj[B->i[i] + j] + 1;
      PetscCall(PetscHMapIGetWithDefault(gid1_lid1, gid1, 0, &data));
      if (!data) {
        /* one based table */
        PetscCall(PetscHMapISet(gid1_lid1, gid1, ++ec));
      }
    }
  }
  /* form array of columns we need */
  PetscCall(PetscMalloc1(ec, &garray));
  PetscHashIterBegin(gid1_lid1, tpos);
  while (!PetscHashIterAtEnd(gid1_lid1, tpos)) {
    PetscHashIterGetKey(gid1_lid1, tpos, gid);
    PetscHashIterGetVal(gid1_lid1, tpos, lid);
    PetscHashIterNext(gid1_lid1, tpos);
    gid--;
    lid--;
    garray[lid] = gid;
  }
  PetscCall(PetscSortInt(ec, garray));
  PetscCall(PetscHMapIClear(gid1_lid1));
  for (i = 0; i < ec; i++) PetscCall(PetscHMapISet(gid1_lid1, garray[i] + 1, i + 1));
  /* compact out the extra columns in B */
  for (i = 0; i < B->mbs; i++) {
    for (j = 0; j < B->ilen[i]; j++) {
      PetscInt gid1 = aj[B->i[i] + j] + 1;
      PetscCall(PetscHMapIGetWithDefault(gid1_lid1, gid1, 0, &lid));
      lid--;
      aj[B->i[i] + j] = lid;
    }
  }
  B->nbs = ec;
  PetscCall(PetscLayoutDestroy(&baij->B->cmap));
  PetscCall(PetscLayoutCreateFromSizes(PetscObjectComm((PetscObject)baij->B), ec * mat->rmap->bs, ec * mat->rmap->bs, mat->rmap->bs, &baij->B->cmap));
  PetscCall(PetscHMapIDestroy(&gid1_lid1));
#else
  /* Make an array as long as the number of columns */
  /* mark those columns that are in baij->B */
  PetscCall(PetscCalloc1(Nbs, &indices));
  for (i = 0; i < B->mbs; i++) {
    for (j = 0; j < B->ilen[i]; j++) {
      if (!indices[aj[B->i[i] + j]]) ec++;
      indices[aj[B->i[i] + j]] = 1;
    }
  }

  /* form array of columns we need */
  PetscCall(PetscMalloc1(ec, &garray));
  ec = 0;
  for (i = 0; i < Nbs; i++) {
    if (indices[i]) garray[ec++] = i;
  }

  /* make indices now point into garray */
  for (i = 0; i < ec; i++) indices[garray[i]] = i;

  /* compact out the extra columns in B */
  for (i = 0; i < B->mbs; i++) {
    for (j = 0; j < B->ilen[i]; j++) aj[B->i[i] + j] = indices[aj[B->i[i] + j]];
  }
  B->nbs = ec;
  PetscCall(PetscLayoutDestroy(&baij->B->cmap));
  PetscCall(PetscLayoutCreateFromSizes(PetscObjectComm((PetscObject)baij->B), ec * mat->rmap->bs, ec * mat->rmap->bs, mat->rmap->bs, &baij->B->cmap));
  PetscCall(PetscFree(indices));
#endif

  /* create local vector that is used to scatter into */
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, ec * bs, &baij->lvec));

  /* create two temporary index sets for building scatter-gather */
  PetscCall(ISCreateBlock(PETSC_COMM_SELF, bs, ec, garray, PETSC_COPY_VALUES, &from));

  PetscCall(PetscMalloc1(ec, &stmp));
  for (i = 0; i < ec; i++) stmp[i] = i;
  PetscCall(ISCreateBlock(PETSC_COMM_SELF, bs, ec, stmp, PETSC_OWN_POINTER, &to));

  /* create temporary global vector to generate scatter context */
  PetscCall(VecCreateMPIWithArray(PetscObjectComm((PetscObject)mat), 1, mat->cmap->n, mat->cmap->N, NULL, &gvec));

  PetscCall(VecScatterCreate(gvec, from, baij->lvec, to, &baij->Mvctx));
  PetscCall(VecScatterViewFromOptions(baij->Mvctx, (PetscObject)mat, "-matmult_vecscatter_view"));

  baij->garray = garray;

  PetscCall(ISDestroy(&from));
  PetscCall(ISDestroy(&to));
  PetscCall(VecDestroy(&gvec));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
     Takes the local part of an already assembled MPIBAIJ matrix
   and disassembles it. This is to allow new nonzeros into the matrix
   that require more communication in the matrix vector multiply.
   Thus certain data-structures must be rebuilt.

   Kind of slow! But that's what application programmers get when
   they are sloppy.
*/
PetscErrorCode MatDisAssemble_MPIBAIJ(Mat A)
{
  Mat_MPIBAIJ *baij  = (Mat_MPIBAIJ *)A->data;
  Mat          B     = baij->B, Bnew;
  Mat_SeqBAIJ *Bbaij = (Mat_SeqBAIJ *)B->data;
  PetscInt     i, j, mbs = Bbaij->mbs, n = A->cmap->N, col, *garray = baij->garray;
  PetscInt     bs2 = baij->bs2, *nz, m = A->rmap->n;
  MatScalar   *a = Bbaij->a;
  MatScalar   *atmp;

  PetscFunctionBegin;
  /* free stuff related to matrix-vec multiply */
  PetscCall(VecDestroy(&baij->lvec));
  baij->lvec = NULL;
  PetscCall(VecScatterDestroy(&baij->Mvctx));
  baij->Mvctx = NULL;
  if (baij->colmap) {
#if defined(PETSC_USE_CTABLE)
    PetscCall(PetscHMapIDestroy(&baij->colmap));
#else
    PetscCall(PetscFree(baij->colmap));
#endif
  }

  /* make sure that B is assembled so we can access its values */
  PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));

  /* invent new B and copy stuff over */
  PetscCall(PetscMalloc1(mbs, &nz));
  for (i = 0; i < mbs; i++) nz[i] = Bbaij->i[i + 1] - Bbaij->i[i];
  PetscCall(MatCreate(PetscObjectComm((PetscObject)B), &Bnew));
  PetscCall(MatSetSizes(Bnew, m, n, m, n));
  PetscCall(MatSetType(Bnew, ((PetscObject)B)->type_name));
  PetscCall(MatSeqBAIJSetPreallocation(Bnew, B->rmap->bs, 0, nz));
  if (Bbaij->nonew >= 0) { /* Inherit insertion error options (if positive). */
    ((Mat_SeqBAIJ *)Bnew->data)->nonew = Bbaij->nonew;
  }

  PetscCall(MatSetOption(Bnew, MAT_ROW_ORIENTED, PETSC_FALSE));
  /*
   Ensure that B's nonzerostate is monotonically increasing.
   Or should this follow the MatSetValuesBlocked() loop to preserve B's nonzerstate across a MatDisAssemble() call?
   */
  Bnew->nonzerostate = B->nonzerostate;

  for (i = 0; i < mbs; i++) {
    for (j = Bbaij->i[i]; j < Bbaij->i[i + 1]; j++) {
      col  = garray[Bbaij->j[j]];
      atmp = a + j * bs2;
      PetscCall(MatSetValuesBlocked_SeqBAIJ(Bnew, 1, &i, 1, &col, atmp, B->insertmode));
    }
  }
  PetscCall(MatSetOption(Bnew, MAT_ROW_ORIENTED, PETSC_TRUE));

  PetscCall(PetscFree(nz));
  PetscCall(PetscFree(baij->garray));
  PetscCall(MatDestroy(&B));

  baij->B          = Bnew;
  A->was_assembled = PETSC_FALSE;
  A->assembled     = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*      ugly stuff added for Glenn someday we should fix this up */

static PetscInt *uglyrmapd = NULL, *uglyrmapo = NULL; /* mapping from the local ordering to the "diagonal" and "off-diagonal" parts of the local matrix */
static Vec       uglydd = NULL, uglyoo = NULL;        /* work vectors used to scale the two parts of the local matrix */

PetscErrorCode MatMPIBAIJDiagonalScaleLocalSetUp(Mat inA, Vec scale)
{
  Mat_MPIBAIJ *ina = (Mat_MPIBAIJ *)inA->data; /*access private part of matrix */
  Mat_SeqBAIJ *B   = (Mat_SeqBAIJ *)ina->B->data;
  PetscInt     bs = inA->rmap->bs, i, n, nt, j, cstart, cend, no, *garray = ina->garray, *lindices;
  PetscInt    *r_rmapd, *r_rmapo;

  PetscFunctionBegin;
  PetscCall(MatGetOwnershipRange(inA, &cstart, &cend));
  PetscCall(MatGetSize(ina->A, NULL, &n));
  PetscCall(PetscCalloc1(inA->rmap->mapping->n + 1, &r_rmapd));
  nt = 0;
  for (i = 0; i < inA->rmap->mapping->n; i++) {
    if (inA->rmap->mapping->indices[i] * bs >= cstart && inA->rmap->mapping->indices[i] * bs < cend) {
      nt++;
      r_rmapd[i] = inA->rmap->mapping->indices[i] + 1;
    }
  }
  PetscCheck(nt * bs == n, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Hmm nt*bs %" PetscInt_FMT " n %" PetscInt_FMT, nt * bs, n);
  PetscCall(PetscMalloc1(n + 1, &uglyrmapd));
  for (i = 0; i < inA->rmap->mapping->n; i++) {
    if (r_rmapd[i]) {
      for (j = 0; j < bs; j++) uglyrmapd[(r_rmapd[i] - 1) * bs + j - cstart] = i * bs + j;
    }
  }
  PetscCall(PetscFree(r_rmapd));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, n, &uglydd));

  PetscCall(PetscCalloc1(ina->Nbs + 1, &lindices));
  for (i = 0; i < B->nbs; i++) lindices[garray[i]] = i + 1;
  no = inA->rmap->mapping->n - nt;
  PetscCall(PetscCalloc1(inA->rmap->mapping->n + 1, &r_rmapo));
  nt = 0;
  for (i = 0; i < inA->rmap->mapping->n; i++) {
    if (lindices[inA->rmap->mapping->indices[i]]) {
      nt++;
      r_rmapo[i] = lindices[inA->rmap->mapping->indices[i]];
    }
  }
  PetscCheck(nt <= no, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Hmm nt %" PetscInt_FMT " no %" PetscInt_FMT, nt, n);
  PetscCall(PetscFree(lindices));
  PetscCall(PetscMalloc1(nt * bs + 1, &uglyrmapo));
  for (i = 0; i < inA->rmap->mapping->n; i++) {
    if (r_rmapo[i]) {
      for (j = 0; j < bs; j++) uglyrmapo[(r_rmapo[i] - 1) * bs + j] = i * bs + j;
    }
  }
  PetscCall(PetscFree(r_rmapo));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, nt * bs, &uglyoo));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatMPIBAIJDiagonalScaleLocal(Mat A, Vec scale)
{
  /* This routine should really be abandoned as it duplicates MatDiagonalScaleLocal */

  PetscFunctionBegin;
  PetscTryMethod(A, "MatDiagonalScaleLocal_C", (Mat, Vec), (A, scale));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatDiagonalScaleLocal_MPIBAIJ(Mat A, Vec scale)
{
  Mat_MPIBAIJ       *a = (Mat_MPIBAIJ *)A->data; /*access private part of matrix */
  PetscInt           n, i;
  PetscScalar       *d, *o;
  const PetscScalar *s;

  PetscFunctionBegin;
  if (!uglyrmapd) PetscCall(MatMPIBAIJDiagonalScaleLocalSetUp(A, scale));

  PetscCall(VecGetArrayRead(scale, &s));

  PetscCall(VecGetLocalSize(uglydd, &n));
  PetscCall(VecGetArray(uglydd, &d));
  for (i = 0; i < n; i++) { d[i] = s[uglyrmapd[i]]; /* copy "diagonal" (true local) portion of scale into dd vector */ }
  PetscCall(VecRestoreArray(uglydd, &d));
  /* column scale "diagonal" portion of local matrix */
  PetscCall(MatDiagonalScale(a->A, NULL, uglydd));

  PetscCall(VecGetLocalSize(uglyoo, &n));
  PetscCall(VecGetArray(uglyoo, &o));
  for (i = 0; i < n; i++) { o[i] = s[uglyrmapo[i]]; /* copy "off-diagonal" portion of scale into oo vector */ }
  PetscCall(VecRestoreArrayRead(scale, &s));
  PetscCall(VecRestoreArray(uglyoo, &o));
  /* column scale "off-diagonal" portion of local matrix */
  PetscCall(MatDiagonalScale(a->B, NULL, uglyoo));
  PetscFunctionReturn(PETSC_SUCCESS);
}
