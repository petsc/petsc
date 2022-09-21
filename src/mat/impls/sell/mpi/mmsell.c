/*
   Support for the parallel SELL matrix vector multiply
*/
#include <../src/mat/impls/sell/mpi/mpisell.h>
#include <petsc/private/isimpl.h> /* needed because accesses data structure of ISLocalToGlobalMapping directly */

/*
   Takes the local part of an already assembled MPISELL matrix
   and disassembles it. This is to allow new nonzeros into the matrix
   that require more communication in the matrix vector multiply.
   Thus certain data-structures must be rebuilt.

   Kind of slow! But that's what application programmers get when
   they are sloppy.
*/
PetscErrorCode MatDisAssemble_MPISELL(Mat A)
{
  Mat_MPISELL *sell  = (Mat_MPISELL *)A->data;
  Mat          B     = sell->B, Bnew;
  Mat_SeqSELL *Bsell = (Mat_SeqSELL *)B->data;
  PetscInt     i, j, totalslices, N = A->cmap->N, row;
  PetscBool    isnonzero;

  PetscFunctionBegin;
  /* free stuff related to matrix-vec multiply */
  PetscCall(VecDestroy(&sell->lvec));
  PetscCall(VecScatterDestroy(&sell->Mvctx));
  if (sell->colmap) {
#if defined(PETSC_USE_CTABLE)
    PetscCall(PetscTableDestroy(&sell->colmap));
#else
    PetscCall(PetscFree(sell->colmap));
#endif
  }

  /* make sure that B is assembled so we can access its values */
  PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));

  /* invent new B and copy stuff over */
  PetscCall(MatCreate(PETSC_COMM_SELF, &Bnew));
  PetscCall(MatSetSizes(Bnew, B->rmap->n, N, B->rmap->n, N));
  PetscCall(MatSetBlockSizesFromMats(Bnew, A, A));
  PetscCall(MatSetType(Bnew, ((PetscObject)B)->type_name));
  PetscCall(MatSeqSELLSetPreallocation(Bnew, 0, Bsell->rlen));
  if (Bsell->nonew >= 0) { /* Inherit insertion error options (if positive). */
    ((Mat_SeqSELL *)Bnew->data)->nonew = Bsell->nonew;
  }

  /*
   Ensure that B's nonzerostate is monotonically increasing.
   Or should this follow the MatSetValues() loop to preserve B's nonzerstate across a MatDisAssemble() call?
   */
  Bnew->nonzerostate = B->nonzerostate;

  totalslices = B->rmap->n / 8 + ((B->rmap->n & 0x07) ? 1 : 0); /* floor(n/8) */
  for (i = 0; i < totalslices; i++) {                           /* loop over slices */
    for (j = Bsell->sliidx[i], row = 0; j < Bsell->sliidx[i + 1]; j++, row = ((row + 1) & 0x07)) {
      isnonzero = (PetscBool)((j - Bsell->sliidx[i]) / 8 < Bsell->rlen[8 * i + row]);
      if (isnonzero) PetscCall(MatSetValue(Bnew, 8 * i + row, sell->garray[Bsell->colidx[j]], Bsell->val[j], B->insertmode));
    }
  }

  PetscCall(PetscFree(sell->garray));
  PetscCall(MatDestroy(&B));

  sell->B          = Bnew;
  A->was_assembled = PETSC_FALSE;
  A->assembled     = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetUpMultiply_MPISELL(Mat mat)
{
  Mat_MPISELL *sell = (Mat_MPISELL *)mat->data;
  Mat_SeqSELL *B    = (Mat_SeqSELL *)(sell->B->data);
  PetscInt     i, j, *bcolidx = B->colidx, ec = 0, *garray, totalslices;
  IS           from, to;
  Vec          gvec;
  PetscBool    isnonzero;
#if defined(PETSC_USE_CTABLE)
  PetscTable         gid1_lid1;
  PetscTablePosition tpos;
  PetscInt           gid, lid;
#else
  PetscInt N = mat->cmap->N, *indices;
#endif

  PetscFunctionBegin;
  totalslices = sell->B->rmap->n / 8 + ((sell->B->rmap->n & 0x07) ? 1 : 0); /* floor(n/8) */

  /* ec counts the number of columns that contain nonzeros */
#if defined(PETSC_USE_CTABLE)
  /* use a table */
  PetscCall(PetscTableCreate(sell->B->rmap->n, mat->cmap->N + 1, &gid1_lid1));
  for (i = 0; i < totalslices; i++) { /* loop over slices */
    for (j = B->sliidx[i]; j < B->sliidx[i + 1]; j++) {
      isnonzero = (PetscBool)((j - B->sliidx[i]) / 8 < B->rlen[(i << 3) + (j & 0x07)]);
      if (isnonzero) { /* check the mask bit */
        PetscInt data, gid1 = bcolidx[j] + 1;
        PetscCall(PetscTableFind(gid1_lid1, gid1, &data));
        if (!data) {
          /* one based table */
          PetscCall(PetscTableAdd(gid1_lid1, gid1, ++ec, INSERT_VALUES));
        }
      }
    }
  }

  /* form array of columns we need */
  PetscCall(PetscMalloc1(ec, &garray));
  PetscCall(PetscTableGetHeadPosition(gid1_lid1, &tpos));
  while (tpos) {
    PetscCall(PetscTableGetNext(gid1_lid1, &tpos, &gid, &lid));
    gid--;
    lid--;
    garray[lid] = gid;
  }
  PetscCall(PetscSortInt(ec, garray)); /* sort, and rebuild */
  PetscCall(PetscTableRemoveAll(gid1_lid1));
  for (i = 0; i < ec; i++) PetscCall(PetscTableAdd(gid1_lid1, garray[i] + 1, i + 1, INSERT_VALUES));

  /* compact out the extra columns in B */
  for (i = 0; i < totalslices; i++) { /* loop over slices */
    for (j = B->sliidx[i]; j < B->sliidx[i + 1]; j++) {
      isnonzero = (PetscBool)((j - B->sliidx[i]) / 8 < B->rlen[(i << 3) + (j & 0x07)]);
      if (isnonzero) {
        PetscInt gid1 = bcolidx[j] + 1;
        PetscCall(PetscTableFind(gid1_lid1, gid1, &lid));
        lid--;
        bcolidx[j] = lid;
      }
    }
  }
  PetscCall(PetscLayoutDestroy(&sell->B->cmap));
  PetscCall(PetscLayoutCreateFromSizes(PetscObjectComm((PetscObject)sell->B), ec, ec, 1, &sell->B->cmap));
  PetscCall(PetscTableDestroy(&gid1_lid1));
#else
  /* Make an array as long as the number of columns */
  PetscCall(PetscCalloc1(N, &indices));
  /* mark those columns that are in sell->B */
  for (i = 0; i < totalslices; i++) { /* loop over slices */
    for (j = B->sliidx[i]; j < B->sliidx[i + 1]; j++) {
      isnonzero = (PetscBool)((j - B->sliidx[i]) / 8 < B->rlen[(i << 3) + (j & 0x07)]);
      if (isnonzero) {
        if (!indices[bcolidx[j]]) ec++;
        indices[bcolidx[j]] = 1;
      }
    }
  }

  /* form array of columns we need */
  PetscCall(PetscMalloc1(ec, &garray));
  ec = 0;
  for (i = 0; i < N; i++) {
    if (indices[i]) garray[ec++] = i;
  }

  /* make indices now point into garray */
  for (i = 0; i < ec; i++) indices[garray[i]] = i;

  /* compact out the extra columns in B */
  for (i = 0; i < totalslices; i++) { /* loop over slices */
    for (j = B->sliidx[i]; j < B->sliidx[i + 1]; j++) {
      isnonzero = (PetscBool)((j - B->sliidx[i]) / 8 < B->rlen[(i << 3) + (j & 0x07)]);
      if (isnonzero) bcolidx[j] = indices[bcolidx[j]];
    }
  }
  PetscCall(PetscLayoutDestroy(&sell->B->cmap));
  PetscCall(PetscLayoutCreateFromSizes(PetscObjectComm((PetscObject)sell->B), ec, ec, 1, &sell->B->cmap));
  PetscCall(PetscFree(indices));
#endif
  /* create local vector that is used to scatter into */
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, ec, &sell->lvec));
  /* create two temporary Index sets for build scatter gather */
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, ec, garray, PETSC_COPY_VALUES, &from));
  PetscCall(ISCreateStride(PETSC_COMM_SELF, ec, 0, 1, &to));

  /* create temporary global vector to generate scatter context */
  /* This does not allocate the array's memory so is efficient */
  PetscCall(VecCreateMPIWithArray(PetscObjectComm((PetscObject)mat), 1, mat->cmap->n, mat->cmap->N, NULL, &gvec));

  /* generate the scatter context */
  PetscCall(VecScatterCreate(gvec, from, sell->lvec, to, &sell->Mvctx));
  PetscCall(VecScatterViewFromOptions(sell->Mvctx, (PetscObject)mat, "-matmult_vecscatter_view"));

  sell->garray = garray;

  PetscCall(ISDestroy(&from));
  PetscCall(ISDestroy(&to));
  PetscCall(VecDestroy(&gvec));
  PetscFunctionReturn(0);
}

/*      ugly stuff added for Glenn someday we should fix this up */
static PetscInt *auglyrmapd = NULL, *auglyrmapo = NULL; /* mapping from the local ordering to the "diagonal" and "off-diagonal" parts of the local matrix */
static Vec       auglydd = NULL, auglyoo = NULL;        /* work vectors used to scale the two parts of the local matrix */

PetscErrorCode MatMPISELLDiagonalScaleLocalSetUp(Mat inA, Vec scale)
{
  Mat_MPISELL *ina = (Mat_MPISELL *)inA->data; /*access private part of matrix */
  PetscInt     i, n, nt, cstart, cend, no, *garray = ina->garray, *lindices;
  PetscInt    *r_rmapd, *r_rmapo;

  PetscFunctionBegin;
  PetscCall(MatGetOwnershipRange(inA, &cstart, &cend));
  PetscCall(MatGetSize(ina->A, NULL, &n));
  PetscCall(PetscCalloc1(inA->rmap->mapping->n + 1, &r_rmapd));
  nt = 0;
  for (i = 0; i < inA->rmap->mapping->n; i++) {
    if (inA->rmap->mapping->indices[i] >= cstart && inA->rmap->mapping->indices[i] < cend) {
      nt++;
      r_rmapd[i] = inA->rmap->mapping->indices[i] + 1;
    }
  }
  PetscCheck(nt == n, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Hmm nt %" PetscInt_FMT " n %" PetscInt_FMT, nt, n);
  PetscCall(PetscMalloc1(n + 1, &auglyrmapd));
  for (i = 0; i < inA->rmap->mapping->n; i++) {
    if (r_rmapd[i]) auglyrmapd[(r_rmapd[i] - 1) - cstart] = i;
  }
  PetscCall(PetscFree(r_rmapd));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, n, &auglydd));
  PetscCall(PetscCalloc1(inA->cmap->N + 1, &lindices));
  for (i = 0; i < ina->B->cmap->n; i++) lindices[garray[i]] = i + 1;
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
  PetscCall(PetscMalloc1(nt + 1, &auglyrmapo));
  for (i = 0; i < inA->rmap->mapping->n; i++) {
    if (r_rmapo[i]) auglyrmapo[(r_rmapo[i] - 1)] = i;
  }
  PetscCall(PetscFree(r_rmapo));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, nt, &auglyoo));
  PetscFunctionReturn(0);
}

PetscErrorCode MatDiagonalScaleLocal_MPISELL(Mat A, Vec scale)
{
  Mat_MPISELL       *a = (Mat_MPISELL *)A->data; /*access private part of matrix */
  PetscInt           n, i;
  PetscScalar       *d, *o;
  const PetscScalar *s;

  PetscFunctionBegin;
  if (!auglyrmapd) PetscCall(MatMPISELLDiagonalScaleLocalSetUp(A, scale));
  PetscCall(VecGetArrayRead(scale, &s));
  PetscCall(VecGetLocalSize(auglydd, &n));
  PetscCall(VecGetArray(auglydd, &d));
  for (i = 0; i < n; i++) { d[i] = s[auglyrmapd[i]]; /* copy "diagonal" (true local) portion of scale into dd vector */ }
  PetscCall(VecRestoreArray(auglydd, &d));
  /* column scale "diagonal" portion of local matrix */
  PetscCall(MatDiagonalScale(a->A, NULL, auglydd));
  PetscCall(VecGetLocalSize(auglyoo, &n));
  PetscCall(VecGetArray(auglyoo, &o));
  for (i = 0; i < n; i++) { o[i] = s[auglyrmapo[i]]; /* copy "off-diagonal" portion of scale into oo vector */ }
  PetscCall(VecRestoreArrayRead(scale, &s));
  PetscCall(VecRestoreArray(auglyoo, &o));
  /* column scale "off-diagonal" portion of local matrix */
  PetscCall(MatDiagonalScale(a->B, NULL, auglyoo));
  PetscFunctionReturn(0);
}
