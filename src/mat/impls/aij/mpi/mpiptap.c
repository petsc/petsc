
/*
  Defines projective product routines where A is a MPIAIJ matrix
          C = P^T * A * P
*/

#include <../src/mat/impls/aij/seq/aij.h> /*I "petscmat.h" I*/
#include <../src/mat/utils/freespace.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <petscbt.h>
#include <petsctime.h>
#include <petsc/private/hashmapiv.h>
#include <petsc/private/hashseti.h>
#include <petscsf.h>

PetscErrorCode MatView_MPIAIJ_PtAP(Mat A, PetscViewer viewer)
{
  PetscBool         iascii;
  PetscViewerFormat format;
  Mat_APMPI        *ptap;

  PetscFunctionBegin;
  MatCheckProduct(A, 1);
  ptap = (Mat_APMPI *)A->product->data;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    PetscCall(PetscViewerGetFormat(viewer, &format));
    if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
      if (ptap->algType == 0) {
        PetscCall(PetscViewerASCIIPrintf(viewer, "using scalable MatPtAP() implementation\n"));
      } else if (ptap->algType == 1) {
        PetscCall(PetscViewerASCIIPrintf(viewer, "using nonscalable MatPtAP() implementation\n"));
      } else if (ptap->algType == 2) {
        PetscCall(PetscViewerASCIIPrintf(viewer, "using allatonce MatPtAP() implementation\n"));
      } else if (ptap->algType == 3) {
        PetscCall(PetscViewerASCIIPrintf(viewer, "using merged allatonce MatPtAP() implementation\n"));
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatDestroy_MPIAIJ_PtAP(void *data)
{
  Mat_APMPI           *ptap = (Mat_APMPI *)data;
  Mat_Merge_SeqsToMPI *merge;

  PetscFunctionBegin;
  PetscCall(PetscFree2(ptap->startsj_s, ptap->startsj_r));
  PetscCall(PetscFree(ptap->bufa));
  PetscCall(MatDestroy(&ptap->P_loc));
  PetscCall(MatDestroy(&ptap->P_oth));
  PetscCall(MatDestroy(&ptap->A_loc)); /* used by MatTransposeMatMult() */
  PetscCall(MatDestroy(&ptap->Rd));
  PetscCall(MatDestroy(&ptap->Ro));
  if (ptap->AP_loc) { /* used by alg_rap */
    Mat_SeqAIJ *ap = (Mat_SeqAIJ *)(ptap->AP_loc)->data;
    PetscCall(PetscFree(ap->i));
    PetscCall(PetscFree2(ap->j, ap->a));
    PetscCall(MatDestroy(&ptap->AP_loc));
  } else { /* used by alg_ptap */
    PetscCall(PetscFree(ptap->api));
    PetscCall(PetscFree(ptap->apj));
  }
  PetscCall(MatDestroy(&ptap->C_loc));
  PetscCall(MatDestroy(&ptap->C_oth));
  if (ptap->apa) PetscCall(PetscFree(ptap->apa));

  PetscCall(MatDestroy(&ptap->Pt));

  merge = ptap->merge;
  if (merge) { /* used by alg_ptap */
    PetscCall(PetscFree(merge->id_r));
    PetscCall(PetscFree(merge->len_s));
    PetscCall(PetscFree(merge->len_r));
    PetscCall(PetscFree(merge->bi));
    PetscCall(PetscFree(merge->bj));
    PetscCall(PetscFree(merge->buf_ri[0]));
    PetscCall(PetscFree(merge->buf_ri));
    PetscCall(PetscFree(merge->buf_rj[0]));
    PetscCall(PetscFree(merge->buf_rj));
    PetscCall(PetscFree(merge->coi));
    PetscCall(PetscFree(merge->coj));
    PetscCall(PetscFree(merge->owners_co));
    PetscCall(PetscLayoutDestroy(&merge->rowmap));
    PetscCall(PetscFree(ptap->merge));
  }
  PetscCall(ISLocalToGlobalMappingDestroy(&ptap->ltog));

  PetscCall(PetscSFDestroy(&ptap->sf));
  PetscCall(PetscFree(ptap->c_othi));
  PetscCall(PetscFree(ptap->c_rmti));
  PetscCall(PetscFree(ptap));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatPtAPNumeric_MPIAIJ_MPIAIJ_scalable(Mat A, Mat P, Mat C)
{
  Mat_MPIAIJ        *a = (Mat_MPIAIJ *)A->data, *p = (Mat_MPIAIJ *)P->data;
  Mat_SeqAIJ        *ad = (Mat_SeqAIJ *)(a->A)->data, *ao = (Mat_SeqAIJ *)(a->B)->data;
  Mat_SeqAIJ        *ap, *p_loc, *p_oth = NULL, *c_seq;
  Mat_APMPI         *ptap;
  Mat                AP_loc, C_loc, C_oth;
  PetscInt           i, rstart, rend, cm, ncols, row, *api, *apj, am = A->rmap->n, apnz, nout;
  PetscScalar       *apa;
  const PetscInt    *cols;
  const PetscScalar *vals;

  PetscFunctionBegin;
  MatCheckProduct(C, 3);
  ptap = (Mat_APMPI *)C->product->data;
  PetscCheck(ptap, PetscObjectComm((PetscObject)C), PETSC_ERR_ARG_WRONGSTATE, "PtAP cannot be computed. Missing data");
  PetscCheck(ptap->AP_loc, PetscObjectComm((PetscObject)C), PETSC_ERR_ARG_WRONGSTATE, "PtAP cannot be reused. Do not call MatProductClear()");

  PetscCall(MatZeroEntries(C));

  /* 1) get R = Pd^T,Ro = Po^T */
  if (ptap->reuse == MAT_REUSE_MATRIX) {
    PetscCall(MatTranspose(p->A, MAT_REUSE_MATRIX, &ptap->Rd));
    PetscCall(MatTranspose(p->B, MAT_REUSE_MATRIX, &ptap->Ro));
  }

  /* 2) get AP_loc */
  AP_loc = ptap->AP_loc;
  ap     = (Mat_SeqAIJ *)AP_loc->data;

  /* 2-1) get P_oth = ptap->P_oth  and P_loc = ptap->P_loc */
  if (ptap->reuse == MAT_REUSE_MATRIX) {
    /* P_oth and P_loc are obtained in MatPtASymbolic() when reuse == MAT_INITIAL_MATRIX */
    PetscCall(MatGetBrowsOfAoCols_MPIAIJ(A, P, MAT_REUSE_MATRIX, &ptap->startsj_s, &ptap->startsj_r, &ptap->bufa, &ptap->P_oth));
    PetscCall(MatMPIAIJGetLocalMat(P, MAT_REUSE_MATRIX, &ptap->P_loc));
  }

  /* 2-2) compute numeric A_loc*P - dominating part */
  /* get data from symbolic products */
  p_loc = (Mat_SeqAIJ *)(ptap->P_loc)->data;
  if (ptap->P_oth) p_oth = (Mat_SeqAIJ *)(ptap->P_oth)->data;

  api = ap->i;
  apj = ap->j;
  PetscCall(ISLocalToGlobalMappingApply(ptap->ltog, api[AP_loc->rmap->n], apj, apj));
  for (i = 0; i < am; i++) {
    /* AP[i,:] = A[i,:]*P = Ad*P_loc Ao*P_oth */
    apnz = api[i + 1] - api[i];
    apa  = ap->a + api[i];
    PetscCall(PetscArrayzero(apa, apnz));
    AProw_scalable(i, ad, ao, p_loc, p_oth, api, apj, apa);
  }
  PetscCall(ISGlobalToLocalMappingApply(ptap->ltog, IS_GTOLM_DROP, api[AP_loc->rmap->n], apj, &nout, apj));
  PetscCheck(api[AP_loc->rmap->n] == nout, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Incorrect mapping %" PetscInt_FMT " != %" PetscInt_FMT, api[AP_loc->rmap->n], nout);

  /* 3) C_loc = Rd*AP_loc, C_oth = Ro*AP_loc */
  /* Always use scalable version since we are in the MPI scalable version */
  PetscCall(MatMatMultNumeric_SeqAIJ_SeqAIJ_Scalable(ptap->Rd, AP_loc, ptap->C_loc));
  PetscCall(MatMatMultNumeric_SeqAIJ_SeqAIJ_Scalable(ptap->Ro, AP_loc, ptap->C_oth));

  C_loc = ptap->C_loc;
  C_oth = ptap->C_oth;

  /* add C_loc and Co to to C */
  PetscCall(MatGetOwnershipRange(C, &rstart, &rend));

  /* C_loc -> C */
  cm    = C_loc->rmap->N;
  c_seq = (Mat_SeqAIJ *)C_loc->data;
  cols  = c_seq->j;
  vals  = c_seq->a;
  PetscCall(ISLocalToGlobalMappingApply(ptap->ltog, c_seq->i[C_loc->rmap->n], c_seq->j, c_seq->j));

  /* The (fast) MatSetValues_MPIAIJ_CopyFromCSRFormat function can only be used when C->was_assembled is PETSC_FALSE and */
  /* when there are no off-processor parts.  */
  /* If was_assembled is true, then the statement aj[rowstart_diag+dnz_row] = mat_j[col] - cstart; in MatSetValues_MPIAIJ_CopyFromCSRFormat */
  /* is no longer true. Then the more complex function MatSetValues_MPIAIJ() has to be used, where the column index is looked up from */
  /* a table, and other, more complex stuff has to be done. */
  if (C->assembled) {
    C->was_assembled = PETSC_TRUE;
    C->assembled     = PETSC_FALSE;
  }
  if (C->was_assembled) {
    for (i = 0; i < cm; i++) {
      ncols = c_seq->i[i + 1] - c_seq->i[i];
      row   = rstart + i;
      PetscCall(MatSetValues_MPIAIJ(C, 1, &row, ncols, cols, vals, ADD_VALUES));
      cols += ncols;
      vals += ncols;
    }
  } else {
    PetscCall(MatSetValues_MPIAIJ_CopyFromCSRFormat(C, c_seq->j, c_seq->i, c_seq->a));
  }
  PetscCall(ISGlobalToLocalMappingApply(ptap->ltog, IS_GTOLM_DROP, c_seq->i[C_loc->rmap->n], c_seq->j, &nout, c_seq->j));
  PetscCheck(c_seq->i[C_loc->rmap->n] == nout, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Incorrect mapping %" PetscInt_FMT " != %" PetscInt_FMT, c_seq->i[C_loc->rmap->n], nout);

  /* Co -> C, off-processor part */
  cm    = C_oth->rmap->N;
  c_seq = (Mat_SeqAIJ *)C_oth->data;
  cols  = c_seq->j;
  vals  = c_seq->a;
  PetscCall(ISLocalToGlobalMappingApply(ptap->ltog, c_seq->i[C_oth->rmap->n], c_seq->j, c_seq->j));
  for (i = 0; i < cm; i++) {
    ncols = c_seq->i[i + 1] - c_seq->i[i];
    row   = p->garray[i];
    PetscCall(MatSetValues(C, 1, &row, ncols, cols, vals, ADD_VALUES));
    cols += ncols;
    vals += ncols;
  }
  PetscCall(MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY));

  ptap->reuse = MAT_REUSE_MATRIX;

  PetscCall(ISGlobalToLocalMappingApply(ptap->ltog, IS_GTOLM_DROP, c_seq->i[C_oth->rmap->n], c_seq->j, &nout, c_seq->j));
  PetscCheck(c_seq->i[C_oth->rmap->n] == nout, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Incorrect mapping %" PetscInt_FMT " != %" PetscInt_FMT, c_seq->i[C_loc->rmap->n], nout);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatPtAPSymbolic_MPIAIJ_MPIAIJ_scalable(Mat A, Mat P, PetscReal fill, Mat Cmpi)
{
  Mat_APMPI               *ptap;
  Mat_MPIAIJ              *a = (Mat_MPIAIJ *)A->data, *p = (Mat_MPIAIJ *)P->data;
  MPI_Comm                 comm;
  PetscMPIInt              size, rank;
  Mat                      P_loc, P_oth;
  PetscFreeSpaceList       free_space = NULL, current_space = NULL;
  PetscInt                 am = A->rmap->n, pm = P->rmap->n, pN = P->cmap->N, pn = P->cmap->n;
  PetscInt                *lnk, i, k, pnz, row, nsend;
  PetscMPIInt              tagi, tagj, *len_si, *len_s, *len_ri, nrecv;
  PETSC_UNUSED PetscMPIInt icompleted = 0;
  PetscInt               **buf_rj, **buf_ri, **buf_ri_k;
  const PetscInt          *owners;
  PetscInt                 len, proc, *dnz, *onz, nzi, nspacedouble;
  PetscInt                 nrows, *buf_s, *buf_si, *buf_si_i, **nextrow, **nextci;
  MPI_Request             *swaits, *rwaits;
  MPI_Status              *sstatus, rstatus;
  PetscLayout              rowmap;
  PetscInt                *owners_co, *coi, *coj; /* i and j array of (p->B)^T*A*P - used in the communication */
  PetscMPIInt             *len_r, *id_r;          /* array of length of comm->size, store send/recv matrix values */
  PetscInt                *api, *apj, *Jptr, apnz, *prmap = p->garray, con, j, Crmax, *aj, *ai, *pi, nout;
  Mat_SeqAIJ              *p_loc, *p_oth = NULL, *ad = (Mat_SeqAIJ *)(a->A)->data, *ao = NULL, *c_loc, *c_oth;
  PetscScalar             *apv;
  PetscHMapI               ta;
  MatType                  mtype;
  const char              *prefix;
#if defined(PETSC_USE_INFO)
  PetscReal apfill;
#endif

  PetscFunctionBegin;
  MatCheckProduct(Cmpi, 4);
  PetscCheck(!Cmpi->product->data, PetscObjectComm((PetscObject)Cmpi), PETSC_ERR_PLIB, "Product data not empty");
  PetscCall(PetscObjectGetComm((PetscObject)A, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  if (size > 1) ao = (Mat_SeqAIJ *)(a->B)->data;

  /* create symbolic parallel matrix Cmpi */
  PetscCall(MatGetType(A, &mtype));
  PetscCall(MatSetType(Cmpi, mtype));

  /* create struct Mat_APMPI and attached it to C later */
  PetscCall(PetscNew(&ptap));
  ptap->reuse   = MAT_INITIAL_MATRIX;
  ptap->algType = 0;

  /* get P_oth by taking rows of P (= non-zero cols of local A) from other processors */
  PetscCall(MatGetBrowsOfAoCols_MPIAIJ(A, P, MAT_INITIAL_MATRIX, &ptap->startsj_s, &ptap->startsj_r, &ptap->bufa, &P_oth));
  /* get P_loc by taking all local rows of P */
  PetscCall(MatMPIAIJGetLocalMat(P, MAT_INITIAL_MATRIX, &P_loc));

  ptap->P_loc = P_loc;
  ptap->P_oth = P_oth;

  /* (0) compute Rd = Pd^T, Ro = Po^T  */
  PetscCall(MatTranspose(p->A, MAT_INITIAL_MATRIX, &ptap->Rd));
  PetscCall(MatTranspose(p->B, MAT_INITIAL_MATRIX, &ptap->Ro));

  /* (1) compute symbolic AP = A_loc*P = Ad*P_loc + Ao*P_oth (api,apj) */
  p_loc = (Mat_SeqAIJ *)P_loc->data;
  if (P_oth) p_oth = (Mat_SeqAIJ *)P_oth->data;

  /* create and initialize a linked list */
  PetscCall(PetscHMapICreateWithSize(pn, &ta)); /* for compute AP_loc and Cmpi */
  MatRowMergeMax_SeqAIJ(p_loc, P_loc->rmap->N, ta);
  MatRowMergeMax_SeqAIJ(p_oth, P_oth->rmap->N, ta);
  PetscCall(PetscHMapIGetSize(ta, &Crmax)); /* Crmax = nnz(sum of Prows) */

  PetscCall(PetscLLCondensedCreate_Scalable(Crmax, &lnk));

  /* Initial FreeSpace size is fill*(nnz(A) + nnz(P)) */
  if (ao) {
    PetscCall(PetscFreeSpaceGet(PetscRealIntMultTruncate(fill, PetscIntSumTruncate(ad->i[am], PetscIntSumTruncate(ao->i[am], p_loc->i[pm]))), &free_space));
  } else {
    PetscCall(PetscFreeSpaceGet(PetscRealIntMultTruncate(fill, PetscIntSumTruncate(ad->i[am], p_loc->i[pm])), &free_space));
  }
  current_space = free_space;
  nspacedouble  = 0;

  PetscCall(PetscMalloc1(am + 1, &api));
  api[0] = 0;
  for (i = 0; i < am; i++) {
    /* diagonal portion: Ad[i,:]*P */
    ai  = ad->i;
    pi  = p_loc->i;
    nzi = ai[i + 1] - ai[i];
    aj  = ad->j + ai[i];
    for (j = 0; j < nzi; j++) {
      row  = aj[j];
      pnz  = pi[row + 1] - pi[row];
      Jptr = p_loc->j + pi[row];
      /* add non-zero cols of P into the sorted linked list lnk */
      PetscCall(PetscLLCondensedAddSorted_Scalable(pnz, Jptr, lnk));
    }
    /* off-diagonal portion: Ao[i,:]*P */
    if (ao) {
      ai  = ao->i;
      pi  = p_oth->i;
      nzi = ai[i + 1] - ai[i];
      aj  = ao->j + ai[i];
      for (j = 0; j < nzi; j++) {
        row  = aj[j];
        pnz  = pi[row + 1] - pi[row];
        Jptr = p_oth->j + pi[row];
        PetscCall(PetscLLCondensedAddSorted_Scalable(pnz, Jptr, lnk));
      }
    }
    apnz       = lnk[0];
    api[i + 1] = api[i] + apnz;

    /* if free space is not available, double the total space in the list */
    if (current_space->local_remaining < apnz) {
      PetscCall(PetscFreeSpaceGet(PetscIntSumTruncate(apnz, current_space->total_array_size), &current_space));
      nspacedouble++;
    }

    /* Copy data into free space, then initialize lnk */
    PetscCall(PetscLLCondensedClean_Scalable(apnz, current_space->array, lnk));

    current_space->array += apnz;
    current_space->local_used += apnz;
    current_space->local_remaining -= apnz;
  }
  /* Allocate space for apj and apv, initialize apj, and */
  /* destroy list of free space and other temporary array(s) */
  PetscCall(PetscCalloc2(api[am], &apj, api[am], &apv));
  PetscCall(PetscFreeSpaceContiguous(&free_space, apj));
  PetscCall(PetscLLCondensedDestroy_Scalable(lnk));

  /* Create AP_loc for reuse */
  PetscCall(MatCreateSeqAIJWithArrays(PETSC_COMM_SELF, am, pN, api, apj, apv, &ptap->AP_loc));
  PetscCall(MatSeqAIJCompactOutExtraColumns_SeqAIJ(ptap->AP_loc, &ptap->ltog));

#if defined(PETSC_USE_INFO)
  if (ao) {
    apfill = (PetscReal)api[am] / (ad->i[am] + ao->i[am] + p_loc->i[pm] + 1);
  } else {
    apfill = (PetscReal)api[am] / (ad->i[am] + p_loc->i[pm] + 1);
  }
  ptap->AP_loc->info.mallocs           = nspacedouble;
  ptap->AP_loc->info.fill_ratio_given  = fill;
  ptap->AP_loc->info.fill_ratio_needed = apfill;

  if (api[am]) {
    PetscCall(PetscInfo(ptap->AP_loc, "Scalable algorithm, AP_loc reallocs %" PetscInt_FMT "; Fill ratio: given %g needed %g.\n", nspacedouble, (double)fill, (double)apfill));
    PetscCall(PetscInfo(ptap->AP_loc, "Use MatPtAP(A,B,MatReuse,%g,&C) for best AP_loc performance.;\n", (double)apfill));
  } else {
    PetscCall(PetscInfo(ptap->AP_loc, "Scalable algorithm, AP_loc is empty \n"));
  }
#endif

  /* (2-1) compute symbolic Co = Ro*AP_loc  */
  PetscCall(MatProductCreate(ptap->Ro, ptap->AP_loc, NULL, &ptap->C_oth));
  PetscCall(MatGetOptionsPrefix(A, &prefix));
  PetscCall(MatSetOptionsPrefix(ptap->C_oth, prefix));
  PetscCall(MatAppendOptionsPrefix(ptap->C_oth, "inner_offdiag_"));

  PetscCall(MatProductSetType(ptap->C_oth, MATPRODUCT_AB));
  PetscCall(MatProductSetAlgorithm(ptap->C_oth, "sorted"));
  PetscCall(MatProductSetFill(ptap->C_oth, fill));
  PetscCall(MatProductSetFromOptions(ptap->C_oth));
  PetscCall(MatProductSymbolic(ptap->C_oth));

  /* (3) send coj of C_oth to other processors  */
  /* determine row ownership */
  PetscCall(PetscLayoutCreate(comm, &rowmap));
  PetscCall(PetscLayoutSetLocalSize(rowmap, pn));
  PetscCall(PetscLayoutSetBlockSize(rowmap, 1));
  PetscCall(PetscLayoutSetUp(rowmap));
  PetscCall(PetscLayoutGetRanges(rowmap, &owners));

  /* determine the number of messages to send, their lengths */
  PetscCall(PetscMalloc4(size, &len_s, size, &len_si, size, &sstatus, size + 2, &owners_co));
  PetscCall(PetscArrayzero(len_s, size));
  PetscCall(PetscArrayzero(len_si, size));

  c_oth = (Mat_SeqAIJ *)ptap->C_oth->data;
  coi   = c_oth->i;
  coj   = c_oth->j;
  con   = ptap->C_oth->rmap->n;
  proc  = 0;
  PetscCall(ISLocalToGlobalMappingApply(ptap->ltog, coi[con], coj, coj));
  for (i = 0; i < con; i++) {
    while (prmap[i] >= owners[proc + 1]) proc++;
    len_si[proc]++;                     /* num of rows in Co(=Pt*AP) to be sent to [proc] */
    len_s[proc] += coi[i + 1] - coi[i]; /* num of nonzeros in Co to be sent to [proc] */
  }

  len          = 0; /* max length of buf_si[], see (4) */
  owners_co[0] = 0;
  nsend        = 0;
  for (proc = 0; proc < size; proc++) {
    owners_co[proc + 1] = owners_co[proc] + len_si[proc];
    if (len_s[proc]) {
      nsend++;
      len_si[proc] = 2 * (len_si[proc] + 1); /* length of buf_si to be sent to [proc] */
      len += len_si[proc];
    }
  }

  /* determine the number and length of messages to receive for coi and coj  */
  PetscCall(PetscGatherNumberOfMessages(comm, NULL, len_s, &nrecv));
  PetscCall(PetscGatherMessageLengths2(comm, nsend, nrecv, len_s, len_si, &id_r, &len_r, &len_ri));

  /* post the Irecv and Isend of coj */
  PetscCall(PetscCommGetNewTag(comm, &tagj));
  PetscCall(PetscPostIrecvInt(comm, tagj, nrecv, id_r, len_r, &buf_rj, &rwaits));
  PetscCall(PetscMalloc1(nsend + 1, &swaits));
  for (proc = 0, k = 0; proc < size; proc++) {
    if (!len_s[proc]) continue;
    i = owners_co[proc];
    PetscCallMPI(MPI_Isend(coj + coi[i], len_s[proc], MPIU_INT, proc, tagj, comm, swaits + k));
    k++;
  }

  /* (2-2) compute symbolic C_loc = Rd*AP_loc */
  PetscCall(MatProductCreate(ptap->Rd, ptap->AP_loc, NULL, &ptap->C_loc));
  PetscCall(MatProductSetType(ptap->C_loc, MATPRODUCT_AB));
  PetscCall(MatProductSetAlgorithm(ptap->C_loc, "default"));
  PetscCall(MatProductSetFill(ptap->C_loc, fill));

  PetscCall(MatSetOptionsPrefix(ptap->C_loc, prefix));
  PetscCall(MatAppendOptionsPrefix(ptap->C_loc, "inner_diag_"));

  PetscCall(MatProductSetFromOptions(ptap->C_loc));
  PetscCall(MatProductSymbolic(ptap->C_loc));

  c_loc = (Mat_SeqAIJ *)ptap->C_loc->data;
  PetscCall(ISLocalToGlobalMappingApply(ptap->ltog, c_loc->i[ptap->C_loc->rmap->n], c_loc->j, c_loc->j));

  /* receives coj are complete */
  for (i = 0; i < nrecv; i++) PetscCallMPI(MPI_Waitany(nrecv, rwaits, &icompleted, &rstatus));
  PetscCall(PetscFree(rwaits));
  if (nsend) PetscCallMPI(MPI_Waitall(nsend, swaits, sstatus));

  /* add received column indices into ta to update Crmax */
  for (k = 0; k < nrecv; k++) { /* k-th received message */
    Jptr = buf_rj[k];
    for (j = 0; j < len_r[k]; j++) PetscCall(PetscHMapISet(ta, *(Jptr + j) + 1, 1));
  }
  PetscCall(PetscHMapIGetSize(ta, &Crmax));
  PetscCall(PetscHMapIDestroy(&ta));

  /* (4) send and recv coi */
  PetscCall(PetscCommGetNewTag(comm, &tagi));
  PetscCall(PetscPostIrecvInt(comm, tagi, nrecv, id_r, len_ri, &buf_ri, &rwaits));
  PetscCall(PetscMalloc1(len + 1, &buf_s));
  buf_si = buf_s; /* points to the beginning of k-th msg to be sent */
  for (proc = 0, k = 0; proc < size; proc++) {
    if (!len_s[proc]) continue;
    /* form outgoing message for i-structure:
         buf_si[0]:                 nrows to be sent
               [1:nrows]:           row index (global)
               [nrows+1:2*nrows+1]: i-structure index
    */
    nrows       = len_si[proc] / 2 - 1; /* num of rows in Co to be sent to [proc] */
    buf_si_i    = buf_si + nrows + 1;
    buf_si[0]   = nrows;
    buf_si_i[0] = 0;
    nrows       = 0;
    for (i = owners_co[proc]; i < owners_co[proc + 1]; i++) {
      nzi                 = coi[i + 1] - coi[i];
      buf_si_i[nrows + 1] = buf_si_i[nrows] + nzi;   /* i-structure */
      buf_si[nrows + 1]   = prmap[i] - owners[proc]; /* local row index */
      nrows++;
    }
    PetscCallMPI(MPI_Isend(buf_si, len_si[proc], MPIU_INT, proc, tagi, comm, swaits + k));
    k++;
    buf_si += len_si[proc];
  }
  for (i = 0; i < nrecv; i++) PetscCallMPI(MPI_Waitany(nrecv, rwaits, &icompleted, &rstatus));
  PetscCall(PetscFree(rwaits));
  if (nsend) PetscCallMPI(MPI_Waitall(nsend, swaits, sstatus));

  PetscCall(PetscFree4(len_s, len_si, sstatus, owners_co));
  PetscCall(PetscFree(len_ri));
  PetscCall(PetscFree(swaits));
  PetscCall(PetscFree(buf_s));

  /* (5) compute the local portion of Cmpi      */
  /* set initial free space to be Crmax, sufficient for holding nozeros in each row of Cmpi */
  PetscCall(PetscFreeSpaceGet(Crmax, &free_space));
  current_space = free_space;

  PetscCall(PetscMalloc3(nrecv, &buf_ri_k, nrecv, &nextrow, nrecv, &nextci));
  for (k = 0; k < nrecv; k++) {
    buf_ri_k[k] = buf_ri[k]; /* beginning of k-th recved i-structure */
    nrows       = *buf_ri_k[k];
    nextrow[k]  = buf_ri_k[k] + 1;           /* next row number of k-th recved i-structure */
    nextci[k]   = buf_ri_k[k] + (nrows + 1); /* points to the next i-structure of k-th recved i-structure  */
  }

  MatPreallocateBegin(comm, pn, pn, dnz, onz);
  PetscCall(PetscLLCondensedCreate_Scalable(Crmax, &lnk));
  for (i = 0; i < pn; i++) {
    /* add C_loc into Cmpi */
    nzi  = c_loc->i[i + 1] - c_loc->i[i];
    Jptr = c_loc->j + c_loc->i[i];
    PetscCall(PetscLLCondensedAddSorted_Scalable(nzi, Jptr, lnk));

    /* add received col data into lnk */
    for (k = 0; k < nrecv; k++) { /* k-th received message */
      if (i == *nextrow[k]) {     /* i-th row */
        nzi  = *(nextci[k] + 1) - *nextci[k];
        Jptr = buf_rj[k] + *nextci[k];
        PetscCall(PetscLLCondensedAddSorted_Scalable(nzi, Jptr, lnk));
        nextrow[k]++;
        nextci[k]++;
      }
    }
    nzi = lnk[0];

    /* copy data into free space, then initialize lnk */
    PetscCall(PetscLLCondensedClean_Scalable(nzi, current_space->array, lnk));
    PetscCall(MatPreallocateSet(i + owners[rank], nzi, current_space->array, dnz, onz));
  }
  PetscCall(PetscFree3(buf_ri_k, nextrow, nextci));
  PetscCall(PetscLLCondensedDestroy_Scalable(lnk));
  PetscCall(PetscFreeSpaceDestroy(free_space));

  /* local sizes and preallocation */
  PetscCall(MatSetSizes(Cmpi, pn, pn, PETSC_DETERMINE, PETSC_DETERMINE));
  if (P->cmap->bs > 0) {
    PetscCall(PetscLayoutSetBlockSize(Cmpi->rmap, P->cmap->bs));
    PetscCall(PetscLayoutSetBlockSize(Cmpi->cmap, P->cmap->bs));
  }
  PetscCall(MatMPIAIJSetPreallocation(Cmpi, 0, dnz, 0, onz));
  MatPreallocateEnd(dnz, onz);

  /* members in merge */
  PetscCall(PetscFree(id_r));
  PetscCall(PetscFree(len_r));
  PetscCall(PetscFree(buf_ri[0]));
  PetscCall(PetscFree(buf_ri));
  PetscCall(PetscFree(buf_rj[0]));
  PetscCall(PetscFree(buf_rj));
  PetscCall(PetscLayoutDestroy(&rowmap));

  nout = 0;
  PetscCall(ISGlobalToLocalMappingApply(ptap->ltog, IS_GTOLM_DROP, c_oth->i[ptap->C_oth->rmap->n], c_oth->j, &nout, c_oth->j));
  PetscCheck(c_oth->i[ptap->C_oth->rmap->n] == nout, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Incorrect mapping %" PetscInt_FMT " != %" PetscInt_FMT, c_oth->i[ptap->C_oth->rmap->n], nout);
  PetscCall(ISGlobalToLocalMappingApply(ptap->ltog, IS_GTOLM_DROP, c_loc->i[ptap->C_loc->rmap->n], c_loc->j, &nout, c_loc->j));
  PetscCheck(c_loc->i[ptap->C_loc->rmap->n] == nout, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Incorrect mapping %" PetscInt_FMT " != %" PetscInt_FMT, c_loc->i[ptap->C_loc->rmap->n], nout);

  /* attach the supporting struct to Cmpi for reuse */
  Cmpi->product->data    = ptap;
  Cmpi->product->view    = MatView_MPIAIJ_PtAP;
  Cmpi->product->destroy = MatDestroy_MPIAIJ_PtAP;

  /* Cmpi is not ready for use - assembly will be done by MatPtAPNumeric() */
  Cmpi->assembled        = PETSC_FALSE;
  Cmpi->ops->ptapnumeric = MatPtAPNumeric_MPIAIJ_MPIAIJ_scalable;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode MatPtAPSymbolicComputeOneRowOfAP_private(Mat A, Mat P, Mat P_oth, const PetscInt *map, PetscInt dof, PetscInt i, PetscHSetI dht, PetscHSetI oht)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ *)A->data, *p = (Mat_MPIAIJ *)P->data;
  Mat_SeqAIJ *ad = (Mat_SeqAIJ *)(a->A)->data, *ao = (Mat_SeqAIJ *)(a->B)->data, *p_oth = (Mat_SeqAIJ *)P_oth->data, *pd = (Mat_SeqAIJ *)p->A->data, *po = (Mat_SeqAIJ *)p->B->data;
  PetscInt   *ai, nzi, j, *aj, row, col, *pi, *pj, pnz, nzpi, *p_othcols, k;
  PetscInt    pcstart, pcend, column, offset;

  PetscFunctionBegin;
  pcstart = P->cmap->rstart;
  pcstart *= dof;
  pcend = P->cmap->rend;
  pcend *= dof;
  /* diagonal portion: Ad[i,:]*P */
  ai  = ad->i;
  nzi = ai[i + 1] - ai[i];
  aj  = ad->j + ai[i];
  for (j = 0; j < nzi; j++) {
    row    = aj[j];
    offset = row % dof;
    row /= dof;
    nzpi = pd->i[row + 1] - pd->i[row];
    pj   = pd->j + pd->i[row];
    for (k = 0; k < nzpi; k++) PetscCall(PetscHSetIAdd(dht, pj[k] * dof + offset + pcstart));
  }
  /* off diag P */
  for (j = 0; j < nzi; j++) {
    row    = aj[j];
    offset = row % dof;
    row /= dof;
    nzpi = po->i[row + 1] - po->i[row];
    pj   = po->j + po->i[row];
    for (k = 0; k < nzpi; k++) PetscCall(PetscHSetIAdd(oht, p->garray[pj[k]] * dof + offset));
  }

  /* off diagonal part: Ao[i, :]*P_oth */
  if (ao) {
    ai  = ao->i;
    pi  = p_oth->i;
    nzi = ai[i + 1] - ai[i];
    aj  = ao->j + ai[i];
    for (j = 0; j < nzi; j++) {
      row       = aj[j];
      offset    = a->garray[row] % dof;
      row       = map[row];
      pnz       = pi[row + 1] - pi[row];
      p_othcols = p_oth->j + pi[row];
      for (col = 0; col < pnz; col++) {
        column = p_othcols[col] * dof + offset;
        if (column >= pcstart && column < pcend) {
          PetscCall(PetscHSetIAdd(dht, column));
        } else {
          PetscCall(PetscHSetIAdd(oht, column));
        }
      }
    }
  } /* end if (ao) */
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode MatPtAPNumericComputeOneRowOfAP_private(Mat A, Mat P, Mat P_oth, const PetscInt *map, PetscInt dof, PetscInt i, PetscHMapIV hmap)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ *)A->data, *p = (Mat_MPIAIJ *)P->data;
  Mat_SeqAIJ *ad = (Mat_SeqAIJ *)(a->A)->data, *ao = (Mat_SeqAIJ *)(a->B)->data, *p_oth = (Mat_SeqAIJ *)P_oth->data, *pd = (Mat_SeqAIJ *)p->A->data, *po = (Mat_SeqAIJ *)p->B->data;
  PetscInt   *ai, nzi, j, *aj, row, col, *pi, pnz, *p_othcols, pcstart, *pj, k, nzpi, offset;
  PetscScalar ra, *aa, *pa;

  PetscFunctionBegin;
  pcstart = P->cmap->rstart;
  pcstart *= dof;

  /* diagonal portion: Ad[i,:]*P */
  ai  = ad->i;
  nzi = ai[i + 1] - ai[i];
  aj  = ad->j + ai[i];
  aa  = ad->a + ai[i];
  for (j = 0; j < nzi; j++) {
    ra     = aa[j];
    row    = aj[j];
    offset = row % dof;
    row /= dof;
    nzpi = pd->i[row + 1] - pd->i[row];
    pj   = pd->j + pd->i[row];
    pa   = pd->a + pd->i[row];
    for (k = 0; k < nzpi; k++) PetscCall(PetscHMapIVAddValue(hmap, pj[k] * dof + offset + pcstart, ra * pa[k]));
    PetscCall(PetscLogFlops(2.0 * nzpi));
  }
  for (j = 0; j < nzi; j++) {
    ra     = aa[j];
    row    = aj[j];
    offset = row % dof;
    row /= dof;
    nzpi = po->i[row + 1] - po->i[row];
    pj   = po->j + po->i[row];
    pa   = po->a + po->i[row];
    for (k = 0; k < nzpi; k++) PetscCall(PetscHMapIVAddValue(hmap, p->garray[pj[k]] * dof + offset, ra * pa[k]));
    PetscCall(PetscLogFlops(2.0 * nzpi));
  }

  /* off diagonal part: Ao[i, :]*P_oth */
  if (ao) {
    ai  = ao->i;
    pi  = p_oth->i;
    nzi = ai[i + 1] - ai[i];
    aj  = ao->j + ai[i];
    aa  = ao->a + ai[i];
    for (j = 0; j < nzi; j++) {
      row       = aj[j];
      offset    = a->garray[row] % dof;
      row       = map[row];
      ra        = aa[j];
      pnz       = pi[row + 1] - pi[row];
      p_othcols = p_oth->j + pi[row];
      pa        = p_oth->a + pi[row];
      for (col = 0; col < pnz; col++) PetscCall(PetscHMapIVAddValue(hmap, p_othcols[col] * dof + offset, ra * pa[col]));
      PetscCall(PetscLogFlops(2.0 * pnz));
    }
  } /* end if (ao) */

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatGetBrowsOfAcols_MPIXAIJ(Mat, Mat, PetscInt dof, MatReuse, Mat *);

PetscErrorCode MatPtAPNumeric_MPIAIJ_MPIXAIJ_allatonce(Mat A, Mat P, PetscInt dof, Mat C)
{
  Mat_MPIAIJ     *p = (Mat_MPIAIJ *)P->data, *c = (Mat_MPIAIJ *)C->data;
  Mat_SeqAIJ     *cd, *co, *po = (Mat_SeqAIJ *)p->B->data, *pd = (Mat_SeqAIJ *)p->A->data;
  Mat_APMPI      *ptap;
  PetscHMapIV     hmap;
  PetscInt        i, j, jj, kk, nzi, *c_rmtj, voff, *c_othj, pn, pon, pcstart, pcend, ccstart, ccend, row, am, *poj, *pdj, *apindices, cmaxr, *c_rmtc, *c_rmtjj, *dcc, *occ, loc;
  PetscScalar    *c_rmta, *c_otha, *poa, *pda, *apvalues, *apvaluestmp, *c_rmtaa;
  PetscInt        offset, ii, pocol;
  const PetscInt *mappingindices;
  IS              map;

  PetscFunctionBegin;
  MatCheckProduct(C, 4);
  ptap = (Mat_APMPI *)C->product->data;
  PetscCheck(ptap, PetscObjectComm((PetscObject)C), PETSC_ERR_ARG_WRONGSTATE, "PtAP cannot be computed. Missing data");
  PetscCheck(ptap->P_oth, PetscObjectComm((PetscObject)C), PETSC_ERR_ARG_WRONGSTATE, "PtAP cannot be reused. Do not call MatProductClear()");

  PetscCall(MatZeroEntries(C));

  /* Get P_oth = ptap->P_oth  and P_loc = ptap->P_loc */
  if (ptap->reuse == MAT_REUSE_MATRIX) {
    /* P_oth and P_loc are obtained in MatPtASymbolic() when reuse == MAT_INITIAL_MATRIX */
    PetscCall(MatGetBrowsOfAcols_MPIXAIJ(A, P, dof, MAT_REUSE_MATRIX, &ptap->P_oth));
  }
  PetscCall(PetscObjectQuery((PetscObject)ptap->P_oth, "aoffdiagtopothmapping", (PetscObject *)&map));

  PetscCall(MatGetLocalSize(p->B, NULL, &pon));
  pon *= dof;
  PetscCall(PetscCalloc2(ptap->c_rmti[pon], &c_rmtj, ptap->c_rmti[pon], &c_rmta));
  PetscCall(MatGetLocalSize(A, &am, NULL));
  cmaxr = 0;
  for (i = 0; i < pon; i++) cmaxr = PetscMax(cmaxr, ptap->c_rmti[i + 1] - ptap->c_rmti[i]);
  PetscCall(PetscCalloc4(cmaxr, &apindices, cmaxr, &apvalues, cmaxr, &apvaluestmp, pon, &c_rmtc));
  PetscCall(PetscHMapIVCreateWithSize(cmaxr, &hmap));
  PetscCall(ISGetIndices(map, &mappingindices));
  for (i = 0; i < am && pon; i++) {
    PetscCall(PetscHMapIVClear(hmap));
    offset = i % dof;
    ii     = i / dof;
    nzi    = po->i[ii + 1] - po->i[ii];
    if (!nzi) continue;
    PetscCall(MatPtAPNumericComputeOneRowOfAP_private(A, P, ptap->P_oth, mappingindices, dof, i, hmap));
    voff = 0;
    PetscCall(PetscHMapIVGetPairs(hmap, &voff, apindices, apvalues));
    if (!voff) continue;

    /* Form C(ii, :) */
    poj = po->j + po->i[ii];
    poa = po->a + po->i[ii];
    for (j = 0; j < nzi; j++) {
      pocol   = poj[j] * dof + offset;
      c_rmtjj = c_rmtj + ptap->c_rmti[pocol];
      c_rmtaa = c_rmta + ptap->c_rmti[pocol];
      for (jj = 0; jj < voff; jj++) {
        apvaluestmp[jj] = apvalues[jj] * poa[j];
        /* If the row is empty */
        if (!c_rmtc[pocol]) {
          c_rmtjj[jj] = apindices[jj];
          c_rmtaa[jj] = apvaluestmp[jj];
          c_rmtc[pocol]++;
        } else {
          PetscCall(PetscFindInt(apindices[jj], c_rmtc[pocol], c_rmtjj, &loc));
          if (loc >= 0) { /* hit */
            c_rmtaa[loc] += apvaluestmp[jj];
            PetscCall(PetscLogFlops(1.0));
          } else { /* new element */
            loc = -(loc + 1);
            /* Move data backward */
            for (kk = c_rmtc[pocol]; kk > loc; kk--) {
              c_rmtjj[kk] = c_rmtjj[kk - 1];
              c_rmtaa[kk] = c_rmtaa[kk - 1];
            } /* End kk */
            c_rmtjj[loc] = apindices[jj];
            c_rmtaa[loc] = apvaluestmp[jj];
            c_rmtc[pocol]++;
          }
        }
        PetscCall(PetscLogFlops(voff));
      } /* End jj */
    }   /* End j */
  }     /* End i */

  PetscCall(PetscFree4(apindices, apvalues, apvaluestmp, c_rmtc));
  PetscCall(PetscHMapIVDestroy(&hmap));

  PetscCall(MatGetLocalSize(P, NULL, &pn));
  pn *= dof;
  PetscCall(PetscCalloc2(ptap->c_othi[pn], &c_othj, ptap->c_othi[pn], &c_otha));

  PetscCall(PetscSFReduceBegin(ptap->sf, MPIU_INT, c_rmtj, c_othj, MPI_REPLACE));
  PetscCall(PetscSFReduceBegin(ptap->sf, MPIU_SCALAR, c_rmta, c_otha, MPI_REPLACE));
  PetscCall(MatGetOwnershipRangeColumn(P, &pcstart, &pcend));
  pcstart = pcstart * dof;
  pcend   = pcend * dof;
  cd      = (Mat_SeqAIJ *)(c->A)->data;
  co      = (Mat_SeqAIJ *)(c->B)->data;

  cmaxr = 0;
  for (i = 0; i < pn; i++) cmaxr = PetscMax(cmaxr, (cd->i[i + 1] - cd->i[i]) + (co->i[i + 1] - co->i[i]));
  PetscCall(PetscCalloc5(cmaxr, &apindices, cmaxr, &apvalues, cmaxr, &apvaluestmp, pn, &dcc, pn, &occ));
  PetscCall(PetscHMapIVCreateWithSize(cmaxr, &hmap));
  for (i = 0; i < am && pn; i++) {
    PetscCall(PetscHMapIVClear(hmap));
    offset = i % dof;
    ii     = i / dof;
    nzi    = pd->i[ii + 1] - pd->i[ii];
    if (!nzi) continue;
    PetscCall(MatPtAPNumericComputeOneRowOfAP_private(A, P, ptap->P_oth, mappingindices, dof, i, hmap));
    voff = 0;
    PetscCall(PetscHMapIVGetPairs(hmap, &voff, apindices, apvalues));
    if (!voff) continue;
    /* Form C(ii, :) */
    pdj = pd->j + pd->i[ii];
    pda = pd->a + pd->i[ii];
    for (j = 0; j < nzi; j++) {
      row = pcstart + pdj[j] * dof + offset;
      for (jj = 0; jj < voff; jj++) apvaluestmp[jj] = apvalues[jj] * pda[j];
      PetscCall(PetscLogFlops(voff));
      PetscCall(MatSetValues(C, 1, &row, voff, apindices, apvaluestmp, ADD_VALUES));
    }
  }
  PetscCall(ISRestoreIndices(map, &mappingindices));
  PetscCall(MatGetOwnershipRangeColumn(C, &ccstart, &ccend));
  PetscCall(PetscFree5(apindices, apvalues, apvaluestmp, dcc, occ));
  PetscCall(PetscHMapIVDestroy(&hmap));
  PetscCall(PetscSFReduceEnd(ptap->sf, MPIU_INT, c_rmtj, c_othj, MPI_REPLACE));
  PetscCall(PetscSFReduceEnd(ptap->sf, MPIU_SCALAR, c_rmta, c_otha, MPI_REPLACE));
  PetscCall(PetscFree2(c_rmtj, c_rmta));

  /* Add contributions from remote */
  for (i = 0; i < pn; i++) {
    row = i + pcstart;
    PetscCall(MatSetValues(C, 1, &row, ptap->c_othi[i + 1] - ptap->c_othi[i], c_othj + ptap->c_othi[i], c_otha + ptap->c_othi[i], ADD_VALUES));
  }
  PetscCall(PetscFree2(c_othj, c_otha));

  PetscCall(MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY));

  ptap->reuse = MAT_REUSE_MATRIX;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatPtAPNumeric_MPIAIJ_MPIAIJ_allatonce(Mat A, Mat P, Mat C)
{
  PetscFunctionBegin;

  PetscCall(MatPtAPNumeric_MPIAIJ_MPIXAIJ_allatonce(A, P, 1, C));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatPtAPNumeric_MPIAIJ_MPIXAIJ_allatonce_merged(Mat A, Mat P, PetscInt dof, Mat C)
{
  Mat_MPIAIJ     *p = (Mat_MPIAIJ *)P->data, *c = (Mat_MPIAIJ *)C->data;
  Mat_SeqAIJ     *cd, *co, *po = (Mat_SeqAIJ *)p->B->data, *pd = (Mat_SeqAIJ *)p->A->data;
  Mat_APMPI      *ptap;
  PetscHMapIV     hmap;
  PetscInt        i, j, jj, kk, nzi, dnzi, *c_rmtj, voff, *c_othj, pn, pon, pcstart, pcend, row, am, *poj, *pdj, *apindices, cmaxr, *c_rmtc, *c_rmtjj, loc;
  PetscScalar    *c_rmta, *c_otha, *poa, *pda, *apvalues, *apvaluestmp, *c_rmtaa;
  PetscInt        offset, ii, pocol;
  const PetscInt *mappingindices;
  IS              map;

  PetscFunctionBegin;
  MatCheckProduct(C, 4);
  ptap = (Mat_APMPI *)C->product->data;
  PetscCheck(ptap, PetscObjectComm((PetscObject)C), PETSC_ERR_ARG_WRONGSTATE, "PtAP cannot be computed. Missing data");
  PetscCheck(ptap->P_oth, PetscObjectComm((PetscObject)C), PETSC_ERR_ARG_WRONGSTATE, "PtAP cannot be reused. Do not call MatProductClear()");

  PetscCall(MatZeroEntries(C));

  /* Get P_oth = ptap->P_oth  and P_loc = ptap->P_loc */
  if (ptap->reuse == MAT_REUSE_MATRIX) {
    /* P_oth and P_loc are obtained in MatPtASymbolic() when reuse == MAT_INITIAL_MATRIX */
    PetscCall(MatGetBrowsOfAcols_MPIXAIJ(A, P, dof, MAT_REUSE_MATRIX, &ptap->P_oth));
  }
  PetscCall(PetscObjectQuery((PetscObject)ptap->P_oth, "aoffdiagtopothmapping", (PetscObject *)&map));
  PetscCall(MatGetLocalSize(p->B, NULL, &pon));
  pon *= dof;
  PetscCall(MatGetLocalSize(P, NULL, &pn));
  pn *= dof;

  PetscCall(PetscCalloc2(ptap->c_rmti[pon], &c_rmtj, ptap->c_rmti[pon], &c_rmta));
  PetscCall(MatGetLocalSize(A, &am, NULL));
  PetscCall(MatGetOwnershipRangeColumn(P, &pcstart, &pcend));
  pcstart *= dof;
  pcend *= dof;
  cmaxr = 0;
  for (i = 0; i < pon; i++) cmaxr = PetscMax(cmaxr, ptap->c_rmti[i + 1] - ptap->c_rmti[i]);
  cd = (Mat_SeqAIJ *)(c->A)->data;
  co = (Mat_SeqAIJ *)(c->B)->data;
  for (i = 0; i < pn; i++) cmaxr = PetscMax(cmaxr, (cd->i[i + 1] - cd->i[i]) + (co->i[i + 1] - co->i[i]));
  PetscCall(PetscCalloc4(cmaxr, &apindices, cmaxr, &apvalues, cmaxr, &apvaluestmp, pon, &c_rmtc));
  PetscCall(PetscHMapIVCreateWithSize(cmaxr, &hmap));
  PetscCall(ISGetIndices(map, &mappingindices));
  for (i = 0; i < am && (pon || pn); i++) {
    PetscCall(PetscHMapIVClear(hmap));
    offset = i % dof;
    ii     = i / dof;
    nzi    = po->i[ii + 1] - po->i[ii];
    dnzi   = pd->i[ii + 1] - pd->i[ii];
    if (!nzi && !dnzi) continue;
    PetscCall(MatPtAPNumericComputeOneRowOfAP_private(A, P, ptap->P_oth, mappingindices, dof, i, hmap));
    voff = 0;
    PetscCall(PetscHMapIVGetPairs(hmap, &voff, apindices, apvalues));
    if (!voff) continue;

    /* Form remote C(ii, :) */
    poj = po->j + po->i[ii];
    poa = po->a + po->i[ii];
    for (j = 0; j < nzi; j++) {
      pocol   = poj[j] * dof + offset;
      c_rmtjj = c_rmtj + ptap->c_rmti[pocol];
      c_rmtaa = c_rmta + ptap->c_rmti[pocol];
      for (jj = 0; jj < voff; jj++) {
        apvaluestmp[jj] = apvalues[jj] * poa[j];
        /* If the row is empty */
        if (!c_rmtc[pocol]) {
          c_rmtjj[jj] = apindices[jj];
          c_rmtaa[jj] = apvaluestmp[jj];
          c_rmtc[pocol]++;
        } else {
          PetscCall(PetscFindInt(apindices[jj], c_rmtc[pocol], c_rmtjj, &loc));
          if (loc >= 0) { /* hit */
            c_rmtaa[loc] += apvaluestmp[jj];
            PetscCall(PetscLogFlops(1.0));
          } else { /* new element */
            loc = -(loc + 1);
            /* Move data backward */
            for (kk = c_rmtc[pocol]; kk > loc; kk--) {
              c_rmtjj[kk] = c_rmtjj[kk - 1];
              c_rmtaa[kk] = c_rmtaa[kk - 1];
            } /* End kk */
            c_rmtjj[loc] = apindices[jj];
            c_rmtaa[loc] = apvaluestmp[jj];
            c_rmtc[pocol]++;
          }
        }
      } /* End jj */
      PetscCall(PetscLogFlops(voff));
    } /* End j */

    /* Form local C(ii, :) */
    pdj = pd->j + pd->i[ii];
    pda = pd->a + pd->i[ii];
    for (j = 0; j < dnzi; j++) {
      row = pcstart + pdj[j] * dof + offset;
      for (jj = 0; jj < voff; jj++) apvaluestmp[jj] = apvalues[jj] * pda[j]; /* End kk */
      PetscCall(PetscLogFlops(voff));
      PetscCall(MatSetValues(C, 1, &row, voff, apindices, apvaluestmp, ADD_VALUES));
    } /* End j */
  }   /* End i */

  PetscCall(ISRestoreIndices(map, &mappingindices));
  PetscCall(PetscFree4(apindices, apvalues, apvaluestmp, c_rmtc));
  PetscCall(PetscHMapIVDestroy(&hmap));
  PetscCall(PetscCalloc2(ptap->c_othi[pn], &c_othj, ptap->c_othi[pn], &c_otha));

  PetscCall(PetscSFReduceBegin(ptap->sf, MPIU_INT, c_rmtj, c_othj, MPI_REPLACE));
  PetscCall(PetscSFReduceBegin(ptap->sf, MPIU_SCALAR, c_rmta, c_otha, MPI_REPLACE));
  PetscCall(PetscSFReduceEnd(ptap->sf, MPIU_INT, c_rmtj, c_othj, MPI_REPLACE));
  PetscCall(PetscSFReduceEnd(ptap->sf, MPIU_SCALAR, c_rmta, c_otha, MPI_REPLACE));
  PetscCall(PetscFree2(c_rmtj, c_rmta));

  /* Add contributions from remote */
  for (i = 0; i < pn; i++) {
    row = i + pcstart;
    PetscCall(MatSetValues(C, 1, &row, ptap->c_othi[i + 1] - ptap->c_othi[i], c_othj + ptap->c_othi[i], c_otha + ptap->c_othi[i], ADD_VALUES));
  }
  PetscCall(PetscFree2(c_othj, c_otha));

  PetscCall(MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY));

  ptap->reuse = MAT_REUSE_MATRIX;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatPtAPNumeric_MPIAIJ_MPIAIJ_allatonce_merged(Mat A, Mat P, Mat C)
{
  PetscFunctionBegin;

  PetscCall(MatPtAPNumeric_MPIAIJ_MPIXAIJ_allatonce_merged(A, P, 1, C));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* TODO: move algorithm selection to MatProductSetFromOptions */
PetscErrorCode MatPtAPSymbolic_MPIAIJ_MPIXAIJ_allatonce(Mat A, Mat P, PetscInt dof, PetscReal fill, Mat Cmpi)
{
  Mat_APMPI      *ptap;
  Mat_MPIAIJ     *p = (Mat_MPIAIJ *)P->data;
  MPI_Comm        comm;
  Mat_SeqAIJ     *pd = (Mat_SeqAIJ *)p->A->data, *po = (Mat_SeqAIJ *)p->B->data;
  MatType         mtype;
  PetscSF         sf;
  PetscSFNode    *iremote;
  PetscInt        rootspacesize, *rootspace, *rootspaceoffsets, nleaves;
  const PetscInt *rootdegrees;
  PetscHSetI      ht, oht, *hta, *hto;
  PetscInt        pn, pon, *c_rmtc, i, j, nzi, htsize, htosize, *c_rmtj, off, *c_othj, rcvncols, sendncols, *c_rmtoffsets;
  PetscInt        lidx, *rdj, col, pcstart, pcend, *dnz, *onz, am, arstart, arend, *poj, *pdj;
  PetscInt        nalg = 2, alg = 0, offset, ii;
  PetscMPIInt     owner;
  const PetscInt *mappingindices;
  PetscBool       flg;
  const char     *algTypes[2] = {"overlapping", "merged"};
  IS              map;

  PetscFunctionBegin;
  MatCheckProduct(Cmpi, 5);
  PetscCheck(!Cmpi->product->data, PetscObjectComm((PetscObject)Cmpi), PETSC_ERR_PLIB, "Product data not empty");
  PetscCall(PetscObjectGetComm((PetscObject)A, &comm));

  /* Create symbolic parallel matrix Cmpi */
  PetscCall(MatGetLocalSize(P, NULL, &pn));
  pn *= dof;
  PetscCall(MatGetType(A, &mtype));
  PetscCall(MatSetType(Cmpi, mtype));
  PetscCall(MatSetSizes(Cmpi, pn, pn, PETSC_DETERMINE, PETSC_DETERMINE));

  PetscCall(PetscNew(&ptap));
  ptap->reuse   = MAT_INITIAL_MATRIX;
  ptap->algType = 2;

  /* Get P_oth by taking rows of P (= non-zero cols of local A) from other processors */
  PetscCall(MatGetBrowsOfAcols_MPIXAIJ(A, P, dof, MAT_INITIAL_MATRIX, &ptap->P_oth));
  PetscCall(PetscObjectQuery((PetscObject)ptap->P_oth, "aoffdiagtopothmapping", (PetscObject *)&map));
  /* This equals to the number of offdiag columns in P */
  PetscCall(MatGetLocalSize(p->B, NULL, &pon));
  pon *= dof;
  /* offsets */
  PetscCall(PetscMalloc1(pon + 1, &ptap->c_rmti));
  /* The number of columns we will send to remote ranks */
  PetscCall(PetscMalloc1(pon, &c_rmtc));
  PetscCall(PetscMalloc1(pon, &hta));
  for (i = 0; i < pon; i++) PetscCall(PetscHSetICreate(&hta[i]));
  PetscCall(MatGetLocalSize(A, &am, NULL));
  PetscCall(MatGetOwnershipRange(A, &arstart, &arend));
  /* Create hash table to merge all columns for C(i, :) */
  PetscCall(PetscHSetICreate(&ht));

  PetscCall(ISGetIndices(map, &mappingindices));
  ptap->c_rmti[0] = 0;
  /* 2) Pass 1: calculate the size for C_rmt (a matrix need to be sent to other processors)  */
  for (i = 0; i < am && pon; i++) {
    /* Form one row of AP */
    PetscCall(PetscHSetIClear(ht));
    offset = i % dof;
    ii     = i / dof;
    /* If the off diag is empty, we should not do any calculation */
    nzi = po->i[ii + 1] - po->i[ii];
    if (!nzi) continue;

    PetscCall(MatPtAPSymbolicComputeOneRowOfAP_private(A, P, ptap->P_oth, mappingindices, dof, i, ht, ht));
    PetscCall(PetscHSetIGetSize(ht, &htsize));
    /* If AP is empty, just continue */
    if (!htsize) continue;
    /* Form C(ii, :) */
    poj = po->j + po->i[ii];
    for (j = 0; j < nzi; j++) PetscCall(PetscHSetIUpdate(hta[poj[j] * dof + offset], ht));
  }

  for (i = 0; i < pon; i++) {
    PetscCall(PetscHSetIGetSize(hta[i], &htsize));
    ptap->c_rmti[i + 1] = ptap->c_rmti[i] + htsize;
    c_rmtc[i]           = htsize;
  }

  PetscCall(PetscMalloc1(ptap->c_rmti[pon], &c_rmtj));

  for (i = 0; i < pon; i++) {
    off = 0;
    PetscCall(PetscHSetIGetElems(hta[i], &off, c_rmtj + ptap->c_rmti[i]));
    PetscCall(PetscHSetIDestroy(&hta[i]));
  }
  PetscCall(PetscFree(hta));

  PetscCall(PetscMalloc1(pon, &iremote));
  for (i = 0; i < pon; i++) {
    owner  = 0;
    lidx   = 0;
    offset = i % dof;
    ii     = i / dof;
    PetscCall(PetscLayoutFindOwnerIndex(P->cmap, p->garray[ii], &owner, &lidx));
    iremote[i].index = lidx * dof + offset;
    iremote[i].rank  = owner;
  }

  PetscCall(PetscSFCreate(comm, &sf));
  PetscCall(PetscSFSetGraph(sf, pn, pon, NULL, PETSC_OWN_POINTER, iremote, PETSC_OWN_POINTER));
  /* Reorder ranks properly so that the data handled by gather and scatter have the same order */
  PetscCall(PetscSFSetRankOrder(sf, PETSC_TRUE));
  PetscCall(PetscSFSetFromOptions(sf));
  PetscCall(PetscSFSetUp(sf));
  /* How many neighbors have contributions to my rows? */
  PetscCall(PetscSFComputeDegreeBegin(sf, &rootdegrees));
  PetscCall(PetscSFComputeDegreeEnd(sf, &rootdegrees));
  rootspacesize = 0;
  for (i = 0; i < pn; i++) rootspacesize += rootdegrees[i];
  PetscCall(PetscMalloc1(rootspacesize, &rootspace));
  PetscCall(PetscMalloc1(rootspacesize + 1, &rootspaceoffsets));
  /* Get information from leaves
   * Number of columns other people contribute to my rows
   * */
  PetscCall(PetscSFGatherBegin(sf, MPIU_INT, c_rmtc, rootspace));
  PetscCall(PetscSFGatherEnd(sf, MPIU_INT, c_rmtc, rootspace));
  PetscCall(PetscFree(c_rmtc));
  PetscCall(PetscCalloc1(pn + 1, &ptap->c_othi));
  /* The number of columns is received for each row */
  ptap->c_othi[0]     = 0;
  rootspacesize       = 0;
  rootspaceoffsets[0] = 0;
  for (i = 0; i < pn; i++) {
    rcvncols = 0;
    for (j = 0; j < rootdegrees[i]; j++) {
      rcvncols += rootspace[rootspacesize];
      rootspaceoffsets[rootspacesize + 1] = rootspaceoffsets[rootspacesize] + rootspace[rootspacesize];
      rootspacesize++;
    }
    ptap->c_othi[i + 1] = ptap->c_othi[i] + rcvncols;
  }
  PetscCall(PetscFree(rootspace));

  PetscCall(PetscMalloc1(pon, &c_rmtoffsets));
  PetscCall(PetscSFScatterBegin(sf, MPIU_INT, rootspaceoffsets, c_rmtoffsets));
  PetscCall(PetscSFScatterEnd(sf, MPIU_INT, rootspaceoffsets, c_rmtoffsets));
  PetscCall(PetscSFDestroy(&sf));
  PetscCall(PetscFree(rootspaceoffsets));

  PetscCall(PetscCalloc1(ptap->c_rmti[pon], &iremote));
  nleaves = 0;
  for (i = 0; i < pon; i++) {
    owner = 0;
    ii    = i / dof;
    PetscCall(PetscLayoutFindOwnerIndex(P->cmap, p->garray[ii], &owner, NULL));
    sendncols = ptap->c_rmti[i + 1] - ptap->c_rmti[i];
    for (j = 0; j < sendncols; j++) {
      iremote[nleaves].rank    = owner;
      iremote[nleaves++].index = c_rmtoffsets[i] + j;
    }
  }
  PetscCall(PetscFree(c_rmtoffsets));
  PetscCall(PetscCalloc1(ptap->c_othi[pn], &c_othj));

  PetscCall(PetscSFCreate(comm, &ptap->sf));
  PetscCall(PetscSFSetGraph(ptap->sf, ptap->c_othi[pn], nleaves, NULL, PETSC_OWN_POINTER, iremote, PETSC_OWN_POINTER));
  PetscCall(PetscSFSetFromOptions(ptap->sf));
  /* One to one map */
  PetscCall(PetscSFReduceBegin(ptap->sf, MPIU_INT, c_rmtj, c_othj, MPI_REPLACE));

  PetscCall(PetscMalloc2(pn, &dnz, pn, &onz));
  PetscCall(PetscHSetICreate(&oht));
  PetscCall(MatGetOwnershipRangeColumn(P, &pcstart, &pcend));
  pcstart *= dof;
  pcend *= dof;
  PetscCall(PetscMalloc2(pn, &hta, pn, &hto));
  for (i = 0; i < pn; i++) {
    PetscCall(PetscHSetICreate(&hta[i]));
    PetscCall(PetscHSetICreate(&hto[i]));
  }
  /* Work on local part */
  /* 4) Pass 1: Estimate memory for C_loc */
  for (i = 0; i < am && pn; i++) {
    PetscCall(PetscHSetIClear(ht));
    PetscCall(PetscHSetIClear(oht));
    offset = i % dof;
    ii     = i / dof;
    nzi    = pd->i[ii + 1] - pd->i[ii];
    if (!nzi) continue;

    PetscCall(MatPtAPSymbolicComputeOneRowOfAP_private(A, P, ptap->P_oth, mappingindices, dof, i, ht, oht));
    PetscCall(PetscHSetIGetSize(ht, &htsize));
    PetscCall(PetscHSetIGetSize(oht, &htosize));
    if (!(htsize + htosize)) continue;
    /* Form C(ii, :) */
    pdj = pd->j + pd->i[ii];
    for (j = 0; j < nzi; j++) {
      PetscCall(PetscHSetIUpdate(hta[pdj[j] * dof + offset], ht));
      PetscCall(PetscHSetIUpdate(hto[pdj[j] * dof + offset], oht));
    }
  }

  PetscCall(ISRestoreIndices(map, &mappingindices));

  PetscCall(PetscHSetIDestroy(&ht));
  PetscCall(PetscHSetIDestroy(&oht));

  /* Get remote data */
  PetscCall(PetscSFReduceEnd(ptap->sf, MPIU_INT, c_rmtj, c_othj, MPI_REPLACE));
  PetscCall(PetscFree(c_rmtj));

  for (i = 0; i < pn; i++) {
    nzi = ptap->c_othi[i + 1] - ptap->c_othi[i];
    rdj = c_othj + ptap->c_othi[i];
    for (j = 0; j < nzi; j++) {
      col = rdj[j];
      /* diag part */
      if (col >= pcstart && col < pcend) {
        PetscCall(PetscHSetIAdd(hta[i], col));
      } else { /* off diag */
        PetscCall(PetscHSetIAdd(hto[i], col));
      }
    }
    PetscCall(PetscHSetIGetSize(hta[i], &htsize));
    dnz[i] = htsize;
    PetscCall(PetscHSetIDestroy(&hta[i]));
    PetscCall(PetscHSetIGetSize(hto[i], &htsize));
    onz[i] = htsize;
    PetscCall(PetscHSetIDestroy(&hto[i]));
  }

  PetscCall(PetscFree2(hta, hto));
  PetscCall(PetscFree(c_othj));

  /* local sizes and preallocation */
  PetscCall(MatSetSizes(Cmpi, pn, pn, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(MatSetBlockSizes(Cmpi, dof > 1 ? dof : P->cmap->bs, dof > 1 ? dof : P->cmap->bs));
  PetscCall(MatMPIAIJSetPreallocation(Cmpi, 0, dnz, 0, onz));
  PetscCall(MatSetUp(Cmpi));
  PetscCall(PetscFree2(dnz, onz));

  /* attach the supporting struct to Cmpi for reuse */
  Cmpi->product->data    = ptap;
  Cmpi->product->destroy = MatDestroy_MPIAIJ_PtAP;
  Cmpi->product->view    = MatView_MPIAIJ_PtAP;

  /* Cmpi is not ready for use - assembly will be done by MatPtAPNumeric() */
  Cmpi->assembled = PETSC_FALSE;
  /* pick an algorithm */
  PetscOptionsBegin(PetscObjectComm((PetscObject)A), ((PetscObject)A)->prefix, "MatPtAP", "Mat");
  alg = 0;
  PetscCall(PetscOptionsEList("-matptap_allatonce_via", "PtAP allatonce numeric approach", "MatPtAP", algTypes, nalg, algTypes[alg], &alg, &flg));
  PetscOptionsEnd();
  switch (alg) {
  case 0:
    Cmpi->ops->ptapnumeric = MatPtAPNumeric_MPIAIJ_MPIAIJ_allatonce;
    break;
  case 1:
    Cmpi->ops->ptapnumeric = MatPtAPNumeric_MPIAIJ_MPIAIJ_allatonce_merged;
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, " Unsupported allatonce numerical algorithm ");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatPtAPSymbolic_MPIAIJ_MPIAIJ_allatonce(Mat A, Mat P, PetscReal fill, Mat C)
{
  PetscFunctionBegin;
  PetscCall(MatPtAPSymbolic_MPIAIJ_MPIXAIJ_allatonce(A, P, 1, fill, C));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatPtAPSymbolic_MPIAIJ_MPIXAIJ_allatonce_merged(Mat A, Mat P, PetscInt dof, PetscReal fill, Mat Cmpi)
{
  Mat_APMPI      *ptap;
  Mat_MPIAIJ     *p = (Mat_MPIAIJ *)P->data;
  MPI_Comm        comm;
  Mat_SeqAIJ     *pd = (Mat_SeqAIJ *)p->A->data, *po = (Mat_SeqAIJ *)p->B->data;
  MatType         mtype;
  PetscSF         sf;
  PetscSFNode    *iremote;
  PetscInt        rootspacesize, *rootspace, *rootspaceoffsets, nleaves;
  const PetscInt *rootdegrees;
  PetscHSetI      ht, oht, *hta, *hto, *htd;
  PetscInt        pn, pon, *c_rmtc, i, j, nzi, dnzi, htsize, htosize, *c_rmtj, off, *c_othj, rcvncols, sendncols, *c_rmtoffsets;
  PetscInt        lidx, *rdj, col, pcstart, pcend, *dnz, *onz, am, arstart, arend, *poj, *pdj;
  PetscInt        nalg = 2, alg = 0, offset, ii;
  PetscMPIInt     owner;
  PetscBool       flg;
  const char     *algTypes[2] = {"merged", "overlapping"};
  const PetscInt *mappingindices;
  IS              map;

  PetscFunctionBegin;
  MatCheckProduct(Cmpi, 5);
  PetscCheck(!Cmpi->product->data, PetscObjectComm((PetscObject)Cmpi), PETSC_ERR_PLIB, "Product data not empty");
  PetscCall(PetscObjectGetComm((PetscObject)A, &comm));

  /* Create symbolic parallel matrix Cmpi */
  PetscCall(MatGetLocalSize(P, NULL, &pn));
  pn *= dof;
  PetscCall(MatGetType(A, &mtype));
  PetscCall(MatSetType(Cmpi, mtype));
  PetscCall(MatSetSizes(Cmpi, pn, pn, PETSC_DETERMINE, PETSC_DETERMINE));

  PetscCall(PetscNew(&ptap));
  ptap->reuse   = MAT_INITIAL_MATRIX;
  ptap->algType = 3;

  /* 0) Get P_oth by taking rows of P (= non-zero cols of local A) from other processors */
  PetscCall(MatGetBrowsOfAcols_MPIXAIJ(A, P, dof, MAT_INITIAL_MATRIX, &ptap->P_oth));
  PetscCall(PetscObjectQuery((PetscObject)ptap->P_oth, "aoffdiagtopothmapping", (PetscObject *)&map));

  /* This equals to the number of offdiag columns in P */
  PetscCall(MatGetLocalSize(p->B, NULL, &pon));
  pon *= dof;
  /* offsets */
  PetscCall(PetscMalloc1(pon + 1, &ptap->c_rmti));
  /* The number of columns we will send to remote ranks */
  PetscCall(PetscMalloc1(pon, &c_rmtc));
  PetscCall(PetscMalloc1(pon, &hta));
  for (i = 0; i < pon; i++) PetscCall(PetscHSetICreate(&hta[i]));
  PetscCall(MatGetLocalSize(A, &am, NULL));
  PetscCall(MatGetOwnershipRange(A, &arstart, &arend));
  /* Create hash table to merge all columns for C(i, :) */
  PetscCall(PetscHSetICreate(&ht));
  PetscCall(PetscHSetICreate(&oht));
  PetscCall(PetscMalloc2(pn, &htd, pn, &hto));
  for (i = 0; i < pn; i++) {
    PetscCall(PetscHSetICreate(&htd[i]));
    PetscCall(PetscHSetICreate(&hto[i]));
  }

  PetscCall(ISGetIndices(map, &mappingindices));
  ptap->c_rmti[0] = 0;
  /* 2) Pass 1: calculate the size for C_rmt (a matrix need to be sent to other processors)  */
  for (i = 0; i < am && (pon || pn); i++) {
    /* Form one row of AP */
    PetscCall(PetscHSetIClear(ht));
    PetscCall(PetscHSetIClear(oht));
    offset = i % dof;
    ii     = i / dof;
    /* If the off diag is empty, we should not do any calculation */
    nzi  = po->i[ii + 1] - po->i[ii];
    dnzi = pd->i[ii + 1] - pd->i[ii];
    if (!nzi && !dnzi) continue;

    PetscCall(MatPtAPSymbolicComputeOneRowOfAP_private(A, P, ptap->P_oth, mappingindices, dof, i, ht, oht));
    PetscCall(PetscHSetIGetSize(ht, &htsize));
    PetscCall(PetscHSetIGetSize(oht, &htosize));
    /* If AP is empty, just continue */
    if (!(htsize + htosize)) continue;

    /* Form remote C(ii, :) */
    poj = po->j + po->i[ii];
    for (j = 0; j < nzi; j++) {
      PetscCall(PetscHSetIUpdate(hta[poj[j] * dof + offset], ht));
      PetscCall(PetscHSetIUpdate(hta[poj[j] * dof + offset], oht));
    }

    /* Form local C(ii, :) */
    pdj = pd->j + pd->i[ii];
    for (j = 0; j < dnzi; j++) {
      PetscCall(PetscHSetIUpdate(htd[pdj[j] * dof + offset], ht));
      PetscCall(PetscHSetIUpdate(hto[pdj[j] * dof + offset], oht));
    }
  }

  PetscCall(ISRestoreIndices(map, &mappingindices));

  PetscCall(PetscHSetIDestroy(&ht));
  PetscCall(PetscHSetIDestroy(&oht));

  for (i = 0; i < pon; i++) {
    PetscCall(PetscHSetIGetSize(hta[i], &htsize));
    ptap->c_rmti[i + 1] = ptap->c_rmti[i] + htsize;
    c_rmtc[i]           = htsize;
  }

  PetscCall(PetscMalloc1(ptap->c_rmti[pon], &c_rmtj));

  for (i = 0; i < pon; i++) {
    off = 0;
    PetscCall(PetscHSetIGetElems(hta[i], &off, c_rmtj + ptap->c_rmti[i]));
    PetscCall(PetscHSetIDestroy(&hta[i]));
  }
  PetscCall(PetscFree(hta));

  PetscCall(PetscMalloc1(pon, &iremote));
  for (i = 0; i < pon; i++) {
    owner  = 0;
    lidx   = 0;
    offset = i % dof;
    ii     = i / dof;
    PetscCall(PetscLayoutFindOwnerIndex(P->cmap, p->garray[ii], &owner, &lidx));
    iremote[i].index = lidx * dof + offset;
    iremote[i].rank  = owner;
  }

  PetscCall(PetscSFCreate(comm, &sf));
  PetscCall(PetscSFSetGraph(sf, pn, pon, NULL, PETSC_OWN_POINTER, iremote, PETSC_OWN_POINTER));
  /* Reorder ranks properly so that the data handled by gather and scatter have the same order */
  PetscCall(PetscSFSetRankOrder(sf, PETSC_TRUE));
  PetscCall(PetscSFSetFromOptions(sf));
  PetscCall(PetscSFSetUp(sf));
  /* How many neighbors have contributions to my rows? */
  PetscCall(PetscSFComputeDegreeBegin(sf, &rootdegrees));
  PetscCall(PetscSFComputeDegreeEnd(sf, &rootdegrees));
  rootspacesize = 0;
  for (i = 0; i < pn; i++) rootspacesize += rootdegrees[i];
  PetscCall(PetscMalloc1(rootspacesize, &rootspace));
  PetscCall(PetscMalloc1(rootspacesize + 1, &rootspaceoffsets));
  /* Get information from leaves
   * Number of columns other people contribute to my rows
   * */
  PetscCall(PetscSFGatherBegin(sf, MPIU_INT, c_rmtc, rootspace));
  PetscCall(PetscSFGatherEnd(sf, MPIU_INT, c_rmtc, rootspace));
  PetscCall(PetscFree(c_rmtc));
  PetscCall(PetscMalloc1(pn + 1, &ptap->c_othi));
  /* The number of columns is received for each row */
  ptap->c_othi[0]     = 0;
  rootspacesize       = 0;
  rootspaceoffsets[0] = 0;
  for (i = 0; i < pn; i++) {
    rcvncols = 0;
    for (j = 0; j < rootdegrees[i]; j++) {
      rcvncols += rootspace[rootspacesize];
      rootspaceoffsets[rootspacesize + 1] = rootspaceoffsets[rootspacesize] + rootspace[rootspacesize];
      rootspacesize++;
    }
    ptap->c_othi[i + 1] = ptap->c_othi[i] + rcvncols;
  }
  PetscCall(PetscFree(rootspace));

  PetscCall(PetscMalloc1(pon, &c_rmtoffsets));
  PetscCall(PetscSFScatterBegin(sf, MPIU_INT, rootspaceoffsets, c_rmtoffsets));
  PetscCall(PetscSFScatterEnd(sf, MPIU_INT, rootspaceoffsets, c_rmtoffsets));
  PetscCall(PetscSFDestroy(&sf));
  PetscCall(PetscFree(rootspaceoffsets));

  PetscCall(PetscCalloc1(ptap->c_rmti[pon], &iremote));
  nleaves = 0;
  for (i = 0; i < pon; i++) {
    owner = 0;
    ii    = i / dof;
    PetscCall(PetscLayoutFindOwnerIndex(P->cmap, p->garray[ii], &owner, NULL));
    sendncols = ptap->c_rmti[i + 1] - ptap->c_rmti[i];
    for (j = 0; j < sendncols; j++) {
      iremote[nleaves].rank    = owner;
      iremote[nleaves++].index = c_rmtoffsets[i] + j;
    }
  }
  PetscCall(PetscFree(c_rmtoffsets));
  PetscCall(PetscCalloc1(ptap->c_othi[pn], &c_othj));

  PetscCall(PetscSFCreate(comm, &ptap->sf));
  PetscCall(PetscSFSetGraph(ptap->sf, ptap->c_othi[pn], nleaves, NULL, PETSC_OWN_POINTER, iremote, PETSC_OWN_POINTER));
  PetscCall(PetscSFSetFromOptions(ptap->sf));
  /* One to one map */
  PetscCall(PetscSFReduceBegin(ptap->sf, MPIU_INT, c_rmtj, c_othj, MPI_REPLACE));
  /* Get remote data */
  PetscCall(PetscSFReduceEnd(ptap->sf, MPIU_INT, c_rmtj, c_othj, MPI_REPLACE));
  PetscCall(PetscFree(c_rmtj));
  PetscCall(PetscMalloc2(pn, &dnz, pn, &onz));
  PetscCall(MatGetOwnershipRangeColumn(P, &pcstart, &pcend));
  pcstart *= dof;
  pcend *= dof;
  for (i = 0; i < pn; i++) {
    nzi = ptap->c_othi[i + 1] - ptap->c_othi[i];
    rdj = c_othj + ptap->c_othi[i];
    for (j = 0; j < nzi; j++) {
      col = rdj[j];
      /* diag part */
      if (col >= pcstart && col < pcend) {
        PetscCall(PetscHSetIAdd(htd[i], col));
      } else { /* off diag */
        PetscCall(PetscHSetIAdd(hto[i], col));
      }
    }
    PetscCall(PetscHSetIGetSize(htd[i], &htsize));
    dnz[i] = htsize;
    PetscCall(PetscHSetIDestroy(&htd[i]));
    PetscCall(PetscHSetIGetSize(hto[i], &htsize));
    onz[i] = htsize;
    PetscCall(PetscHSetIDestroy(&hto[i]));
  }

  PetscCall(PetscFree2(htd, hto));
  PetscCall(PetscFree(c_othj));

  /* local sizes and preallocation */
  PetscCall(MatSetSizes(Cmpi, pn, pn, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(MatSetBlockSizes(Cmpi, dof > 1 ? dof : P->cmap->bs, dof > 1 ? dof : P->cmap->bs));
  PetscCall(MatMPIAIJSetPreallocation(Cmpi, 0, dnz, 0, onz));
  PetscCall(PetscFree2(dnz, onz));

  /* attach the supporting struct to Cmpi for reuse */
  Cmpi->product->data    = ptap;
  Cmpi->product->destroy = MatDestroy_MPIAIJ_PtAP;
  Cmpi->product->view    = MatView_MPIAIJ_PtAP;

  /* Cmpi is not ready for use - assembly will be done by MatPtAPNumeric() */
  Cmpi->assembled = PETSC_FALSE;
  /* pick an algorithm */
  PetscOptionsBegin(PetscObjectComm((PetscObject)A), ((PetscObject)A)->prefix, "MatPtAP", "Mat");
  alg = 0;
  PetscCall(PetscOptionsEList("-matptap_allatonce_via", "PtAP allatonce numeric approach", "MatPtAP", algTypes, nalg, algTypes[alg], &alg, &flg));
  PetscOptionsEnd();
  switch (alg) {
  case 0:
    Cmpi->ops->ptapnumeric = MatPtAPNumeric_MPIAIJ_MPIAIJ_allatonce_merged;
    break;
  case 1:
    Cmpi->ops->ptapnumeric = MatPtAPNumeric_MPIAIJ_MPIAIJ_allatonce;
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, " Unsupported allatonce numerical algorithm ");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatPtAPSymbolic_MPIAIJ_MPIAIJ_allatonce_merged(Mat A, Mat P, PetscReal fill, Mat C)
{
  PetscFunctionBegin;
  PetscCall(MatPtAPSymbolic_MPIAIJ_MPIXAIJ_allatonce_merged(A, P, 1, fill, C));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatPtAPSymbolic_MPIAIJ_MPIAIJ(Mat A, Mat P, PetscReal fill, Mat Cmpi)
{
  Mat_APMPI               *ptap;
  Mat_MPIAIJ              *a = (Mat_MPIAIJ *)A->data, *p = (Mat_MPIAIJ *)P->data;
  MPI_Comm                 comm;
  PetscMPIInt              size, rank;
  PetscFreeSpaceList       free_space = NULL, current_space = NULL;
  PetscInt                 am = A->rmap->n, pm = P->rmap->n, pN = P->cmap->N, pn = P->cmap->n;
  PetscInt                *lnk, i, k, pnz, row, nsend;
  PetscBT                  lnkbt;
  PetscMPIInt              tagi, tagj, *len_si, *len_s, *len_ri, nrecv;
  PETSC_UNUSED PetscMPIInt icompleted = 0;
  PetscInt               **buf_rj, **buf_ri, **buf_ri_k;
  PetscInt                 len, proc, *dnz, *onz, *owners, nzi, nspacedouble;
  PetscInt                 nrows, *buf_s, *buf_si, *buf_si_i, **nextrow, **nextci;
  MPI_Request             *swaits, *rwaits;
  MPI_Status              *sstatus, rstatus;
  PetscLayout              rowmap;
  PetscInt                *owners_co, *coi, *coj; /* i and j array of (p->B)^T*A*P - used in the communication */
  PetscMPIInt             *len_r, *id_r;          /* array of length of comm->size, store send/recv matrix values */
  PetscInt                *api, *apj, *Jptr, apnz, *prmap = p->garray, con, j, ap_rmax = 0, Crmax, *aj, *ai, *pi;
  Mat_SeqAIJ              *p_loc, *p_oth = NULL, *ad = (Mat_SeqAIJ *)(a->A)->data, *ao = NULL, *c_loc, *c_oth;
  PetscScalar             *apv;
  PetscHMapI               ta;
  MatType                  mtype;
  const char              *prefix;
#if defined(PETSC_USE_INFO)
  PetscReal apfill;
#endif

  PetscFunctionBegin;
  MatCheckProduct(Cmpi, 4);
  PetscCheck(!Cmpi->product->data, PetscObjectComm((PetscObject)Cmpi), PETSC_ERR_PLIB, "Product data not empty");
  PetscCall(PetscObjectGetComm((PetscObject)A, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  if (size > 1) ao = (Mat_SeqAIJ *)(a->B)->data;

  /* create symbolic parallel matrix Cmpi */
  PetscCall(MatGetType(A, &mtype));
  PetscCall(MatSetType(Cmpi, mtype));

  /* Do dense axpy in MatPtAPNumeric_MPIAIJ_MPIAIJ() */
  Cmpi->ops->ptapnumeric = MatPtAPNumeric_MPIAIJ_MPIAIJ;

  /* create struct Mat_APMPI and attached it to C later */
  PetscCall(PetscNew(&ptap));
  ptap->reuse   = MAT_INITIAL_MATRIX;
  ptap->algType = 1;

  /* get P_oth by taking rows of P (= non-zero cols of local A) from other processors */
  PetscCall(MatGetBrowsOfAoCols_MPIAIJ(A, P, MAT_INITIAL_MATRIX, &ptap->startsj_s, &ptap->startsj_r, &ptap->bufa, &ptap->P_oth));
  /* get P_loc by taking all local rows of P */
  PetscCall(MatMPIAIJGetLocalMat(P, MAT_INITIAL_MATRIX, &ptap->P_loc));

  /* (0) compute Rd = Pd^T, Ro = Po^T  */
  PetscCall(MatTranspose(p->A, MAT_INITIAL_MATRIX, &ptap->Rd));
  PetscCall(MatTranspose(p->B, MAT_INITIAL_MATRIX, &ptap->Ro));

  /* (1) compute symbolic AP = A_loc*P = Ad*P_loc + Ao*P_oth (api,apj) */
  p_loc = (Mat_SeqAIJ *)(ptap->P_loc)->data;
  if (ptap->P_oth) p_oth = (Mat_SeqAIJ *)(ptap->P_oth)->data;

  /* create and initialize a linked list */
  PetscCall(PetscHMapICreateWithSize(pn, &ta)); /* for compute AP_loc and Cmpi */
  MatRowMergeMax_SeqAIJ(p_loc, ptap->P_loc->rmap->N, ta);
  MatRowMergeMax_SeqAIJ(p_oth, ptap->P_oth->rmap->N, ta);
  PetscCall(PetscHMapIGetSize(ta, &Crmax)); /* Crmax = nnz(sum of Prows) */
  /* printf("[%d] est %d, Crmax %d; pN %d\n",rank,5*(p_loc->rmax+p_oth->rmax + (PetscInt)(1.e-2*pN)),Crmax,pN); */

  PetscCall(PetscLLCondensedCreate(Crmax, pN, &lnk, &lnkbt));

  /* Initial FreeSpace size is fill*(nnz(A) + nnz(P)) */
  if (ao) {
    PetscCall(PetscFreeSpaceGet(PetscRealIntMultTruncate(fill, PetscIntSumTruncate(ad->i[am], PetscIntSumTruncate(ao->i[am], p_loc->i[pm]))), &free_space));
  } else {
    PetscCall(PetscFreeSpaceGet(PetscRealIntMultTruncate(fill, PetscIntSumTruncate(ad->i[am], p_loc->i[pm])), &free_space));
  }
  current_space = free_space;
  nspacedouble  = 0;

  PetscCall(PetscMalloc1(am + 1, &api));
  api[0] = 0;
  for (i = 0; i < am; i++) {
    /* diagonal portion: Ad[i,:]*P */
    ai  = ad->i;
    pi  = p_loc->i;
    nzi = ai[i + 1] - ai[i];
    aj  = ad->j + ai[i];
    for (j = 0; j < nzi; j++) {
      row  = aj[j];
      pnz  = pi[row + 1] - pi[row];
      Jptr = p_loc->j + pi[row];
      /* add non-zero cols of P into the sorted linked list lnk */
      PetscCall(PetscLLCondensedAddSorted(pnz, Jptr, lnk, lnkbt));
    }
    /* off-diagonal portion: Ao[i,:]*P */
    if (ao) {
      ai  = ao->i;
      pi  = p_oth->i;
      nzi = ai[i + 1] - ai[i];
      aj  = ao->j + ai[i];
      for (j = 0; j < nzi; j++) {
        row  = aj[j];
        pnz  = pi[row + 1] - pi[row];
        Jptr = p_oth->j + pi[row];
        PetscCall(PetscLLCondensedAddSorted(pnz, Jptr, lnk, lnkbt));
      }
    }
    apnz       = lnk[0];
    api[i + 1] = api[i] + apnz;
    if (ap_rmax < apnz) ap_rmax = apnz;

    /* if free space is not available, double the total space in the list */
    if (current_space->local_remaining < apnz) {
      PetscCall(PetscFreeSpaceGet(PetscIntSumTruncate(apnz, current_space->total_array_size), &current_space));
      nspacedouble++;
    }

    /* Copy data into free space, then initialize lnk */
    PetscCall(PetscLLCondensedClean(pN, apnz, current_space->array, lnk, lnkbt));

    current_space->array += apnz;
    current_space->local_used += apnz;
    current_space->local_remaining -= apnz;
  }
  /* Allocate space for apj and apv, initialize apj, and */
  /* destroy list of free space and other temporary array(s) */
  PetscCall(PetscMalloc2(api[am], &apj, api[am], &apv));
  PetscCall(PetscFreeSpaceContiguous(&free_space, apj));
  PetscCall(PetscLLDestroy(lnk, lnkbt));

  /* Create AP_loc for reuse */
  PetscCall(MatCreateSeqAIJWithArrays(PETSC_COMM_SELF, am, pN, api, apj, apv, &ptap->AP_loc));
  PetscCall(MatSetType(ptap->AP_loc, ((PetscObject)p->A)->type_name));
#if defined(PETSC_USE_INFO)
  if (ao) {
    apfill = (PetscReal)api[am] / (ad->i[am] + ao->i[am] + p_loc->i[pm] + 1);
  } else {
    apfill = (PetscReal)api[am] / (ad->i[am] + p_loc->i[pm] + 1);
  }
  ptap->AP_loc->info.mallocs           = nspacedouble;
  ptap->AP_loc->info.fill_ratio_given  = fill;
  ptap->AP_loc->info.fill_ratio_needed = apfill;

  if (api[am]) {
    PetscCall(PetscInfo(ptap->AP_loc, "Nonscalable algorithm, AP_loc reallocs %" PetscInt_FMT "; Fill ratio: given %g needed %g.\n", nspacedouble, (double)fill, (double)apfill));
    PetscCall(PetscInfo(ptap->AP_loc, "Use MatPtAP(A,B,MatReuse,%g,&C) for best AP_loc performance.;\n", (double)apfill));
  } else {
    PetscCall(PetscInfo(ptap->AP_loc, "Nonscalable algorithm, AP_loc is empty \n"));
  }
#endif

  /* (2-1) compute symbolic Co = Ro*AP_loc  */
  PetscCall(MatGetOptionsPrefix(A, &prefix));
  PetscCall(MatSetOptionsPrefix(ptap->Ro, prefix));
  PetscCall(MatAppendOptionsPrefix(ptap->Ro, "inner_offdiag_"));
  PetscCall(MatProductCreate(ptap->Ro, ptap->AP_loc, NULL, &ptap->C_oth));
  PetscCall(MatGetOptionsPrefix(Cmpi, &prefix));
  PetscCall(MatSetOptionsPrefix(ptap->C_oth, prefix));
  PetscCall(MatAppendOptionsPrefix(ptap->C_oth, "inner_C_oth_"));
  PetscCall(MatProductSetType(ptap->C_oth, MATPRODUCT_AB));
  PetscCall(MatProductSetAlgorithm(ptap->C_oth, "default"));
  PetscCall(MatProductSetFill(ptap->C_oth, fill));
  PetscCall(MatProductSetFromOptions(ptap->C_oth));
  PetscCall(MatProductSymbolic(ptap->C_oth));

  /* (3) send coj of C_oth to other processors  */
  /* determine row ownership */
  PetscCall(PetscLayoutCreate(comm, &rowmap));
  rowmap->n  = pn;
  rowmap->bs = 1;
  PetscCall(PetscLayoutSetUp(rowmap));
  owners = rowmap->range;

  /* determine the number of messages to send, their lengths */
  PetscCall(PetscMalloc4(size, &len_s, size, &len_si, size, &sstatus, size + 2, &owners_co));
  PetscCall(PetscArrayzero(len_s, size));
  PetscCall(PetscArrayzero(len_si, size));

  c_oth = (Mat_SeqAIJ *)ptap->C_oth->data;
  coi   = c_oth->i;
  coj   = c_oth->j;
  con   = ptap->C_oth->rmap->n;
  proc  = 0;
  for (i = 0; i < con; i++) {
    while (prmap[i] >= owners[proc + 1]) proc++;
    len_si[proc]++;                     /* num of rows in Co(=Pt*AP) to be sent to [proc] */
    len_s[proc] += coi[i + 1] - coi[i]; /* num of nonzeros in Co to be sent to [proc] */
  }

  len          = 0; /* max length of buf_si[], see (4) */
  owners_co[0] = 0;
  nsend        = 0;
  for (proc = 0; proc < size; proc++) {
    owners_co[proc + 1] = owners_co[proc] + len_si[proc];
    if (len_s[proc]) {
      nsend++;
      len_si[proc] = 2 * (len_si[proc] + 1); /* length of buf_si to be sent to [proc] */
      len += len_si[proc];
    }
  }

  /* determine the number and length of messages to receive for coi and coj  */
  PetscCall(PetscGatherNumberOfMessages(comm, NULL, len_s, &nrecv));
  PetscCall(PetscGatherMessageLengths2(comm, nsend, nrecv, len_s, len_si, &id_r, &len_r, &len_ri));

  /* post the Irecv and Isend of coj */
  PetscCall(PetscCommGetNewTag(comm, &tagj));
  PetscCall(PetscPostIrecvInt(comm, tagj, nrecv, id_r, len_r, &buf_rj, &rwaits));
  PetscCall(PetscMalloc1(nsend + 1, &swaits));
  for (proc = 0, k = 0; proc < size; proc++) {
    if (!len_s[proc]) continue;
    i = owners_co[proc];
    PetscCallMPI(MPI_Isend(coj + coi[i], len_s[proc], MPIU_INT, proc, tagj, comm, swaits + k));
    k++;
  }

  /* (2-2) compute symbolic C_loc = Rd*AP_loc */
  PetscCall(MatSetOptionsPrefix(ptap->Rd, prefix));
  PetscCall(MatAppendOptionsPrefix(ptap->Rd, "inner_diag_"));
  PetscCall(MatProductCreate(ptap->Rd, ptap->AP_loc, NULL, &ptap->C_loc));
  PetscCall(MatGetOptionsPrefix(Cmpi, &prefix));
  PetscCall(MatSetOptionsPrefix(ptap->C_loc, prefix));
  PetscCall(MatAppendOptionsPrefix(ptap->C_loc, "inner_C_loc_"));
  PetscCall(MatProductSetType(ptap->C_loc, MATPRODUCT_AB));
  PetscCall(MatProductSetAlgorithm(ptap->C_loc, "default"));
  PetscCall(MatProductSetFill(ptap->C_loc, fill));
  PetscCall(MatProductSetFromOptions(ptap->C_loc));
  PetscCall(MatProductSymbolic(ptap->C_loc));

  c_loc = (Mat_SeqAIJ *)ptap->C_loc->data;

  /* receives coj are complete */
  for (i = 0; i < nrecv; i++) PetscCallMPI(MPI_Waitany(nrecv, rwaits, &icompleted, &rstatus));
  PetscCall(PetscFree(rwaits));
  if (nsend) PetscCallMPI(MPI_Waitall(nsend, swaits, sstatus));

  /* add received column indices into ta to update Crmax */
  for (k = 0; k < nrecv; k++) { /* k-th received message */
    Jptr = buf_rj[k];
    for (j = 0; j < len_r[k]; j++) PetscCall(PetscHMapISet(ta, *(Jptr + j) + 1, 1));
  }
  PetscCall(PetscHMapIGetSize(ta, &Crmax));
  PetscCall(PetscHMapIDestroy(&ta));

  /* (4) send and recv coi */
  PetscCall(PetscCommGetNewTag(comm, &tagi));
  PetscCall(PetscPostIrecvInt(comm, tagi, nrecv, id_r, len_ri, &buf_ri, &rwaits));
  PetscCall(PetscMalloc1(len + 1, &buf_s));
  buf_si = buf_s; /* points to the beginning of k-th msg to be sent */
  for (proc = 0, k = 0; proc < size; proc++) {
    if (!len_s[proc]) continue;
    /* form outgoing message for i-structure:
         buf_si[0]:                 nrows to be sent
               [1:nrows]:           row index (global)
               [nrows+1:2*nrows+1]: i-structure index
    */
    nrows       = len_si[proc] / 2 - 1; /* num of rows in Co to be sent to [proc] */
    buf_si_i    = buf_si + nrows + 1;
    buf_si[0]   = nrows;
    buf_si_i[0] = 0;
    nrows       = 0;
    for (i = owners_co[proc]; i < owners_co[proc + 1]; i++) {
      nzi                 = coi[i + 1] - coi[i];
      buf_si_i[nrows + 1] = buf_si_i[nrows] + nzi;   /* i-structure */
      buf_si[nrows + 1]   = prmap[i] - owners[proc]; /* local row index */
      nrows++;
    }
    PetscCallMPI(MPI_Isend(buf_si, len_si[proc], MPIU_INT, proc, tagi, comm, swaits + k));
    k++;
    buf_si += len_si[proc];
  }
  for (i = 0; i < nrecv; i++) PetscCallMPI(MPI_Waitany(nrecv, rwaits, &icompleted, &rstatus));
  PetscCall(PetscFree(rwaits));
  if (nsend) PetscCallMPI(MPI_Waitall(nsend, swaits, sstatus));

  PetscCall(PetscFree4(len_s, len_si, sstatus, owners_co));
  PetscCall(PetscFree(len_ri));
  PetscCall(PetscFree(swaits));
  PetscCall(PetscFree(buf_s));

  /* (5) compute the local portion of Cmpi      */
  /* set initial free space to be Crmax, sufficient for holding nozeros in each row of Cmpi */
  PetscCall(PetscFreeSpaceGet(Crmax, &free_space));
  current_space = free_space;

  PetscCall(PetscMalloc3(nrecv, &buf_ri_k, nrecv, &nextrow, nrecv, &nextci));
  for (k = 0; k < nrecv; k++) {
    buf_ri_k[k] = buf_ri[k]; /* beginning of k-th recved i-structure */
    nrows       = *buf_ri_k[k];
    nextrow[k]  = buf_ri_k[k] + 1;           /* next row number of k-th recved i-structure */
    nextci[k]   = buf_ri_k[k] + (nrows + 1); /* points to the next i-structure of k-th recved i-structure  */
  }

  MatPreallocateBegin(comm, pn, pn, dnz, onz);
  PetscCall(PetscLLCondensedCreate(Crmax, pN, &lnk, &lnkbt));
  for (i = 0; i < pn; i++) {
    /* add C_loc into Cmpi */
    nzi  = c_loc->i[i + 1] - c_loc->i[i];
    Jptr = c_loc->j + c_loc->i[i];
    PetscCall(PetscLLCondensedAddSorted(nzi, Jptr, lnk, lnkbt));

    /* add received col data into lnk */
    for (k = 0; k < nrecv; k++) { /* k-th received message */
      if (i == *nextrow[k]) {     /* i-th row */
        nzi  = *(nextci[k] + 1) - *nextci[k];
        Jptr = buf_rj[k] + *nextci[k];
        PetscCall(PetscLLCondensedAddSorted(nzi, Jptr, lnk, lnkbt));
        nextrow[k]++;
        nextci[k]++;
      }
    }
    nzi = lnk[0];

    /* copy data into free space, then initialize lnk */
    PetscCall(PetscLLCondensedClean(pN, nzi, current_space->array, lnk, lnkbt));
    PetscCall(MatPreallocateSet(i + owners[rank], nzi, current_space->array, dnz, onz));
  }
  PetscCall(PetscFree3(buf_ri_k, nextrow, nextci));
  PetscCall(PetscLLDestroy(lnk, lnkbt));
  PetscCall(PetscFreeSpaceDestroy(free_space));

  /* local sizes and preallocation */
  PetscCall(MatSetSizes(Cmpi, pn, pn, PETSC_DETERMINE, PETSC_DETERMINE));
  if (P->cmap->bs > 0) {
    PetscCall(PetscLayoutSetBlockSize(Cmpi->rmap, P->cmap->bs));
    PetscCall(PetscLayoutSetBlockSize(Cmpi->cmap, P->cmap->bs));
  }
  PetscCall(MatMPIAIJSetPreallocation(Cmpi, 0, dnz, 0, onz));
  MatPreallocateEnd(dnz, onz);

  /* members in merge */
  PetscCall(PetscFree(id_r));
  PetscCall(PetscFree(len_r));
  PetscCall(PetscFree(buf_ri[0]));
  PetscCall(PetscFree(buf_ri));
  PetscCall(PetscFree(buf_rj[0]));
  PetscCall(PetscFree(buf_rj));
  PetscCall(PetscLayoutDestroy(&rowmap));

  PetscCall(PetscCalloc1(pN, &ptap->apa));

  /* attach the supporting struct to Cmpi for reuse */
  Cmpi->product->data    = ptap;
  Cmpi->product->destroy = MatDestroy_MPIAIJ_PtAP;
  Cmpi->product->view    = MatView_MPIAIJ_PtAP;

  /* Cmpi is not ready for use - assembly will be done by MatPtAPNumeric() */
  Cmpi->assembled = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatPtAPNumeric_MPIAIJ_MPIAIJ(Mat A, Mat P, Mat C)
{
  Mat_MPIAIJ        *a = (Mat_MPIAIJ *)A->data, *p = (Mat_MPIAIJ *)P->data;
  Mat_SeqAIJ        *ad = (Mat_SeqAIJ *)(a->A)->data, *ao = (Mat_SeqAIJ *)(a->B)->data;
  Mat_SeqAIJ        *ap, *p_loc, *p_oth = NULL, *c_seq;
  Mat_APMPI         *ptap;
  Mat                AP_loc, C_loc, C_oth;
  PetscInt           i, rstart, rend, cm, ncols, row;
  PetscInt          *api, *apj, am = A->rmap->n, j, col, apnz;
  PetscScalar       *apa;
  const PetscInt    *cols;
  const PetscScalar *vals;

  PetscFunctionBegin;
  MatCheckProduct(C, 3);
  ptap = (Mat_APMPI *)C->product->data;
  PetscCheck(ptap, PetscObjectComm((PetscObject)C), PETSC_ERR_ARG_WRONGSTATE, "PtAP cannot be computed. Missing data");
  PetscCheck(ptap->AP_loc, PetscObjectComm((PetscObject)C), PETSC_ERR_ARG_WRONGSTATE, "PtAP cannot be reused. Do not call MatProductClear()");

  PetscCall(MatZeroEntries(C));
  /* 1) get R = Pd^T,Ro = Po^T */
  if (ptap->reuse == MAT_REUSE_MATRIX) {
    PetscCall(MatTranspose(p->A, MAT_REUSE_MATRIX, &ptap->Rd));
    PetscCall(MatTranspose(p->B, MAT_REUSE_MATRIX, &ptap->Ro));
  }

  /* 2) get AP_loc */
  AP_loc = ptap->AP_loc;
  ap     = (Mat_SeqAIJ *)AP_loc->data;

  /* 2-1) get P_oth = ptap->P_oth  and P_loc = ptap->P_loc */
  if (ptap->reuse == MAT_REUSE_MATRIX) {
    /* P_oth and P_loc are obtained in MatPtASymbolic() when reuse == MAT_INITIAL_MATRIX */
    PetscCall(MatGetBrowsOfAoCols_MPIAIJ(A, P, MAT_REUSE_MATRIX, &ptap->startsj_s, &ptap->startsj_r, &ptap->bufa, &ptap->P_oth));
    PetscCall(MatMPIAIJGetLocalMat(P, MAT_REUSE_MATRIX, &ptap->P_loc));
  }

  /* 2-2) compute numeric A_loc*P - dominating part */
  /* get data from symbolic products */
  p_loc = (Mat_SeqAIJ *)(ptap->P_loc)->data;
  if (ptap->P_oth) p_oth = (Mat_SeqAIJ *)(ptap->P_oth)->data;
  apa = ptap->apa;
  api = ap->i;
  apj = ap->j;
  for (i = 0; i < am; i++) {
    /* AP[i,:] = A[i,:]*P = Ad*P_loc Ao*P_oth */
    AProw_nonscalable(i, ad, ao, p_loc, p_oth, apa);
    apnz = api[i + 1] - api[i];
    for (j = 0; j < apnz; j++) {
      col                 = apj[j + api[i]];
      ap->a[j + ap->i[i]] = apa[col];
      apa[col]            = 0.0;
    }
  }
  /* We have modified the contents of local matrix AP_loc and must increase its ObjectState, since we are not doing AssemblyBegin/End on it. */
  PetscCall(PetscObjectStateIncrease((PetscObject)AP_loc));

  /* 3) C_loc = Rd*AP_loc, C_oth = Ro*AP_loc */
  PetscCall(MatProductNumeric(ptap->C_loc));
  PetscCall(MatProductNumeric(ptap->C_oth));
  C_loc = ptap->C_loc;
  C_oth = ptap->C_oth;

  /* add C_loc and Co to to C */
  PetscCall(MatGetOwnershipRange(C, &rstart, &rend));

  /* C_loc -> C */
  cm    = C_loc->rmap->N;
  c_seq = (Mat_SeqAIJ *)C_loc->data;
  cols  = c_seq->j;
  vals  = c_seq->a;

  /* The (fast) MatSetValues_MPIAIJ_CopyFromCSRFormat function can only be used when C->was_assembled is PETSC_FALSE and */
  /* when there are no off-processor parts.  */
  /* If was_assembled is true, then the statement aj[rowstart_diag+dnz_row] = mat_j[col] - cstart; in MatSetValues_MPIAIJ_CopyFromCSRFormat */
  /* is no longer true. Then the more complex function MatSetValues_MPIAIJ() has to be used, where the column index is looked up from */
  /* a table, and other, more complex stuff has to be done. */
  if (C->assembled) {
    C->was_assembled = PETSC_TRUE;
    C->assembled     = PETSC_FALSE;
  }
  if (C->was_assembled) {
    for (i = 0; i < cm; i++) {
      ncols = c_seq->i[i + 1] - c_seq->i[i];
      row   = rstart + i;
      PetscCall(MatSetValues_MPIAIJ(C, 1, &row, ncols, cols, vals, ADD_VALUES));
      cols += ncols;
      vals += ncols;
    }
  } else {
    PetscCall(MatSetValues_MPIAIJ_CopyFromCSRFormat(C, c_seq->j, c_seq->i, c_seq->a));
  }

  /* Co -> C, off-processor part */
  cm    = C_oth->rmap->N;
  c_seq = (Mat_SeqAIJ *)C_oth->data;
  cols  = c_seq->j;
  vals  = c_seq->a;
  for (i = 0; i < cm; i++) {
    ncols = c_seq->i[i + 1] - c_seq->i[i];
    row   = p->garray[i];
    PetscCall(MatSetValues(C, 1, &row, ncols, cols, vals, ADD_VALUES));
    cols += ncols;
    vals += ncols;
  }

  PetscCall(MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY));

  ptap->reuse = MAT_REUSE_MATRIX;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatProductSymbolic_PtAP_MPIAIJ_MPIAIJ(Mat C)
{
  Mat_Product        *product = C->product;
  Mat                 A = product->A, P = product->B;
  MatProductAlgorithm alg  = product->alg;
  PetscReal           fill = product->fill;
  PetscBool           flg;

  PetscFunctionBegin;
  /* scalable: do R=P^T locally, then C=R*A*P */
  PetscCall(PetscStrcmp(alg, "scalable", &flg));
  if (flg) {
    PetscCall(MatPtAPSymbolic_MPIAIJ_MPIAIJ_scalable(A, P, product->fill, C));
    C->ops->productnumeric = MatProductNumeric_PtAP;
    goto next;
  }

  /* nonscalable: do R=P^T locally, then C=R*A*P */
  PetscCall(PetscStrcmp(alg, "nonscalable", &flg));
  if (flg) {
    PetscCall(MatPtAPSymbolic_MPIAIJ_MPIAIJ(A, P, fill, C));
    goto next;
  }

  /* allatonce */
  PetscCall(PetscStrcmp(alg, "allatonce", &flg));
  if (flg) {
    PetscCall(MatPtAPSymbolic_MPIAIJ_MPIAIJ_allatonce(A, P, fill, C));
    goto next;
  }

  /* allatonce_merged */
  PetscCall(PetscStrcmp(alg, "allatonce_merged", &flg));
  if (flg) {
    PetscCall(MatPtAPSymbolic_MPIAIJ_MPIAIJ_allatonce_merged(A, P, fill, C));
    goto next;
  }

  /* backend general code */
  PetscCall(PetscStrcmp(alg, "backend", &flg));
  if (flg) {
    PetscCall(MatProductSymbolic_MPIAIJBACKEND(C));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* hypre */
#if defined(PETSC_HAVE_HYPRE)
  PetscCall(PetscStrcmp(alg, "hypre", &flg));
  if (flg) {
    PetscCall(MatPtAPSymbolic_AIJ_AIJ_wHYPRE(A, P, fill, C));
    C->ops->productnumeric = MatProductNumeric_PtAP;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
#endif
  SETERRQ(PetscObjectComm((PetscObject)C), PETSC_ERR_SUP, "Mat Product Algorithm is not supported");

next:
  C->ops->productnumeric = MatProductNumeric_PtAP;
  PetscFunctionReturn(PETSC_SUCCESS);
}
