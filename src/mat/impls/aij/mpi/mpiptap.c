
/*
  Defines projective product routines where A is a MPIAIJ matrix
          C = P^T * A * P
*/

#include <../src/mat/impls/aij/seq/aij.h>   /*I "petscmat.h" I*/
#include <../src/mat/utils/freespace.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <petscbt.h>
#include <petsctime.h>
#include <petsc/private/hashmapiv.h>
#include <petsc/private/hashseti.h>
#include <petscsf.h>

PetscErrorCode MatView_MPIAIJ_PtAP(Mat A,PetscViewer viewer)
{
  PetscBool         iascii;
  PetscViewerFormat format;
  Mat_APMPI         *ptap;

  PetscFunctionBegin;
  MatCheckProduct(A,1);
  ptap = (Mat_APMPI*)A->product->data;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    CHKERRQ(PetscViewerGetFormat(viewer,&format));
    if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
      if (ptap->algType == 0) {
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"using scalable MatPtAP() implementation\n"));
      } else if (ptap->algType == 1) {
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"using nonscalable MatPtAP() implementation\n"));
      } else if (ptap->algType == 2) {
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"using allatonce MatPtAP() implementation\n"));
      } else if (ptap->algType == 3) {
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"using merged allatonce MatPtAP() implementation\n"));
      }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_MPIAIJ_PtAP(void *data)
{
  Mat_APMPI           *ptap = (Mat_APMPI*)data;
  Mat_Merge_SeqsToMPI *merge;

  PetscFunctionBegin;
  CHKERRQ(PetscFree2(ptap->startsj_s,ptap->startsj_r));
  CHKERRQ(PetscFree(ptap->bufa));
  CHKERRQ(MatDestroy(&ptap->P_loc));
  CHKERRQ(MatDestroy(&ptap->P_oth));
  CHKERRQ(MatDestroy(&ptap->A_loc)); /* used by MatTransposeMatMult() */
  CHKERRQ(MatDestroy(&ptap->Rd));
  CHKERRQ(MatDestroy(&ptap->Ro));
  if (ptap->AP_loc) { /* used by alg_rap */
    Mat_SeqAIJ *ap = (Mat_SeqAIJ*)(ptap->AP_loc)->data;
    CHKERRQ(PetscFree(ap->i));
    CHKERRQ(PetscFree2(ap->j,ap->a));
    CHKERRQ(MatDestroy(&ptap->AP_loc));
  } else { /* used by alg_ptap */
    CHKERRQ(PetscFree(ptap->api));
    CHKERRQ(PetscFree(ptap->apj));
  }
  CHKERRQ(MatDestroy(&ptap->C_loc));
  CHKERRQ(MatDestroy(&ptap->C_oth));
  if (ptap->apa) CHKERRQ(PetscFree(ptap->apa));

  CHKERRQ(MatDestroy(&ptap->Pt));

  merge = ptap->merge;
  if (merge) { /* used by alg_ptap */
    CHKERRQ(PetscFree(merge->id_r));
    CHKERRQ(PetscFree(merge->len_s));
    CHKERRQ(PetscFree(merge->len_r));
    CHKERRQ(PetscFree(merge->bi));
    CHKERRQ(PetscFree(merge->bj));
    CHKERRQ(PetscFree(merge->buf_ri[0]));
    CHKERRQ(PetscFree(merge->buf_ri));
    CHKERRQ(PetscFree(merge->buf_rj[0]));
    CHKERRQ(PetscFree(merge->buf_rj));
    CHKERRQ(PetscFree(merge->coi));
    CHKERRQ(PetscFree(merge->coj));
    CHKERRQ(PetscFree(merge->owners_co));
    CHKERRQ(PetscLayoutDestroy(&merge->rowmap));
    CHKERRQ(PetscFree(ptap->merge));
  }
  CHKERRQ(ISLocalToGlobalMappingDestroy(&ptap->ltog));

  CHKERRQ(PetscSFDestroy(&ptap->sf));
  CHKERRQ(PetscFree(ptap->c_othi));
  CHKERRQ(PetscFree(ptap->c_rmti));
  CHKERRQ(PetscFree(ptap));
  PetscFunctionReturn(0);
}

PetscErrorCode MatPtAPNumeric_MPIAIJ_MPIAIJ_scalable(Mat A,Mat P,Mat C)
{
  Mat_MPIAIJ        *a=(Mat_MPIAIJ*)A->data,*p=(Mat_MPIAIJ*)P->data;
  Mat_SeqAIJ        *ad=(Mat_SeqAIJ*)(a->A)->data,*ao=(Mat_SeqAIJ*)(a->B)->data;
  Mat_SeqAIJ        *ap,*p_loc,*p_oth=NULL,*c_seq;
  Mat_APMPI         *ptap;
  Mat               AP_loc,C_loc,C_oth;
  PetscInt          i,rstart,rend,cm,ncols,row,*api,*apj,am = A->rmap->n,apnz,nout;
  PetscScalar       *apa;
  const PetscInt    *cols;
  const PetscScalar *vals;

  PetscFunctionBegin;
  MatCheckProduct(C,3);
  ptap = (Mat_APMPI*)C->product->data;
  PetscCheckFalse(!ptap,PetscObjectComm((PetscObject)C),PETSC_ERR_ARG_WRONGSTATE,"PtAP cannot be computed. Missing data");
  PetscCheckFalse(!ptap->AP_loc,PetscObjectComm((PetscObject)C),PETSC_ERR_ARG_WRONGSTATE,"PtAP cannot be reused. Do not call MatProductClear()");

  CHKERRQ(MatZeroEntries(C));

  /* 1) get R = Pd^T,Ro = Po^T */
  if (ptap->reuse == MAT_REUSE_MATRIX) {
    CHKERRQ(MatTranspose(p->A,MAT_REUSE_MATRIX,&ptap->Rd));
    CHKERRQ(MatTranspose(p->B,MAT_REUSE_MATRIX,&ptap->Ro));
  }

  /* 2) get AP_loc */
  AP_loc = ptap->AP_loc;
  ap = (Mat_SeqAIJ*)AP_loc->data;

  /* 2-1) get P_oth = ptap->P_oth  and P_loc = ptap->P_loc */
  /*-----------------------------------------------------*/
  if (ptap->reuse == MAT_REUSE_MATRIX) {
    /* P_oth and P_loc are obtained in MatPtASymbolic() when reuse == MAT_INITIAL_MATRIX */
    CHKERRQ(MatGetBrowsOfAoCols_MPIAIJ(A,P,MAT_REUSE_MATRIX,&ptap->startsj_s,&ptap->startsj_r,&ptap->bufa,&ptap->P_oth));
    CHKERRQ(MatMPIAIJGetLocalMat(P,MAT_REUSE_MATRIX,&ptap->P_loc));
  }

  /* 2-2) compute numeric A_loc*P - dominating part */
  /* ---------------------------------------------- */
  /* get data from symbolic products */
  p_loc = (Mat_SeqAIJ*)(ptap->P_loc)->data;
  if (ptap->P_oth) p_oth = (Mat_SeqAIJ*)(ptap->P_oth)->data;

  api   = ap->i;
  apj   = ap->j;
  CHKERRQ(ISLocalToGlobalMappingApply(ptap->ltog,api[AP_loc->rmap->n],apj,apj));
  for (i=0; i<am; i++) {
    /* AP[i,:] = A[i,:]*P = Ad*P_loc Ao*P_oth */
    apnz = api[i+1] - api[i];
    apa = ap->a + api[i];
    CHKERRQ(PetscArrayzero(apa,apnz));
    AProw_scalable(i,ad,ao,p_loc,p_oth,api,apj,apa);
  }
  CHKERRQ(ISGlobalToLocalMappingApply(ptap->ltog,IS_GTOLM_DROP,api[AP_loc->rmap->n],apj,&nout,apj));
  PetscCheckFalse(api[AP_loc->rmap->n] != nout,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Incorrect mapping %" PetscInt_FMT " != %" PetscInt_FMT,api[AP_loc->rmap->n],nout);

  /* 3) C_loc = Rd*AP_loc, C_oth = Ro*AP_loc */
  /* Always use scalable version since we are in the MPI scalable version */
  CHKERRQ(MatMatMultNumeric_SeqAIJ_SeqAIJ_Scalable(ptap->Rd,AP_loc,ptap->C_loc));
  CHKERRQ(MatMatMultNumeric_SeqAIJ_SeqAIJ_Scalable(ptap->Ro,AP_loc,ptap->C_oth));

  C_loc = ptap->C_loc;
  C_oth = ptap->C_oth;

  /* add C_loc and Co to to C */
  CHKERRQ(MatGetOwnershipRange(C,&rstart,&rend));

  /* C_loc -> C */
  cm    = C_loc->rmap->N;
  c_seq = (Mat_SeqAIJ*)C_loc->data;
  cols = c_seq->j;
  vals = c_seq->a;
  CHKERRQ(ISLocalToGlobalMappingApply(ptap->ltog,c_seq->i[C_loc->rmap->n],c_seq->j,c_seq->j));

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
    for (i=0; i<cm; i++) {
      ncols = c_seq->i[i+1] - c_seq->i[i];
      row = rstart + i;
      CHKERRQ(MatSetValues_MPIAIJ(C,1,&row,ncols,cols,vals,ADD_VALUES));
      cols += ncols; vals += ncols;
    }
  } else {
    CHKERRQ(MatSetValues_MPIAIJ_CopyFromCSRFormat(C,c_seq->j,c_seq->i,c_seq->a));
  }
  CHKERRQ(ISGlobalToLocalMappingApply(ptap->ltog,IS_GTOLM_DROP,c_seq->i[C_loc->rmap->n],c_seq->j,&nout,c_seq->j));
  PetscCheckFalse(c_seq->i[C_loc->rmap->n] != nout,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Incorrect mapping %" PetscInt_FMT " != %" PetscInt_FMT,c_seq->i[C_loc->rmap->n],nout);

  /* Co -> C, off-processor part */
  cm = C_oth->rmap->N;
  c_seq = (Mat_SeqAIJ*)C_oth->data;
  cols = c_seq->j;
  vals = c_seq->a;
  CHKERRQ(ISLocalToGlobalMappingApply(ptap->ltog,c_seq->i[C_oth->rmap->n],c_seq->j,c_seq->j));
  for (i=0; i<cm; i++) {
    ncols = c_seq->i[i+1] - c_seq->i[i];
    row = p->garray[i];
    CHKERRQ(MatSetValues(C,1,&row,ncols,cols,vals,ADD_VALUES));
    cols += ncols; vals += ncols;
  }
  CHKERRQ(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));

  ptap->reuse = MAT_REUSE_MATRIX;

  CHKERRQ(ISGlobalToLocalMappingApply(ptap->ltog,IS_GTOLM_DROP,c_seq->i[C_oth->rmap->n],c_seq->j,&nout,c_seq->j));
  PetscCheckFalse(c_seq->i[C_oth->rmap->n] != nout,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Incorrect mapping %" PetscInt_FMT " != %" PetscInt_FMT,c_seq->i[C_loc->rmap->n],nout);
  PetscFunctionReturn(0);
}

PetscErrorCode MatPtAPSymbolic_MPIAIJ_MPIAIJ_scalable(Mat A,Mat P,PetscReal fill,Mat Cmpi)
{
  PetscErrorCode      ierr;
  Mat_APMPI           *ptap;
  Mat_MPIAIJ          *a=(Mat_MPIAIJ*)A->data,*p=(Mat_MPIAIJ*)P->data;
  MPI_Comm            comm;
  PetscMPIInt         size,rank;
  Mat                 P_loc,P_oth;
  PetscFreeSpaceList  free_space=NULL,current_space=NULL;
  PetscInt            am=A->rmap->n,pm=P->rmap->n,pN=P->cmap->N,pn=P->cmap->n;
  PetscInt            *lnk,i,k,pnz,row,nsend;
  PetscMPIInt         tagi,tagj,*len_si,*len_s,*len_ri,nrecv;
  PETSC_UNUSED PetscMPIInt icompleted=0;
  PetscInt            **buf_rj,**buf_ri,**buf_ri_k;
  const PetscInt      *owners;
  PetscInt            len,proc,*dnz,*onz,nzi,nspacedouble;
  PetscInt            nrows,*buf_s,*buf_si,*buf_si_i,**nextrow,**nextci;
  MPI_Request         *swaits,*rwaits;
  MPI_Status          *sstatus,rstatus;
  PetscLayout         rowmap;
  PetscInt            *owners_co,*coi,*coj;    /* i and j array of (p->B)^T*A*P - used in the communication */
  PetscMPIInt         *len_r,*id_r;    /* array of length of comm->size, store send/recv matrix values */
  PetscInt            *api,*apj,*Jptr,apnz,*prmap=p->garray,con,j,Crmax,*aj,*ai,*pi,nout;
  Mat_SeqAIJ          *p_loc,*p_oth=NULL,*ad=(Mat_SeqAIJ*)(a->A)->data,*ao=NULL,*c_loc,*c_oth;
  PetscScalar         *apv;
  PetscTable          ta;
  MatType             mtype;
  const char          *prefix;
#if defined(PETSC_USE_INFO)
  PetscReal           apfill;
#endif

  PetscFunctionBegin;
  MatCheckProduct(Cmpi,4);
  PetscCheckFalse(Cmpi->product->data,PetscObjectComm((PetscObject)Cmpi),PETSC_ERR_PLIB,"Product data not empty");
  CHKERRQ(PetscObjectGetComm((PetscObject)A,&comm));
  CHKERRMPI(MPI_Comm_size(comm,&size));
  CHKERRMPI(MPI_Comm_rank(comm,&rank));

  if (size > 1) ao = (Mat_SeqAIJ*)(a->B)->data;

  /* create symbolic parallel matrix Cmpi */
  CHKERRQ(MatGetType(A,&mtype));
  CHKERRQ(MatSetType(Cmpi,mtype));

  /* create struct Mat_APMPI and attached it to C later */
  CHKERRQ(PetscNew(&ptap));
  ptap->reuse = MAT_INITIAL_MATRIX;
  ptap->algType = 0;

  /* get P_oth by taking rows of P (= non-zero cols of local A) from other processors */
  CHKERRQ(MatGetBrowsOfAoCols_MPIAIJ(A,P,MAT_INITIAL_MATRIX,&ptap->startsj_s,&ptap->startsj_r,&ptap->bufa,&P_oth));
  /* get P_loc by taking all local rows of P */
  CHKERRQ(MatMPIAIJGetLocalMat(P,MAT_INITIAL_MATRIX,&P_loc));

  ptap->P_loc = P_loc;
  ptap->P_oth = P_oth;

  /* (0) compute Rd = Pd^T, Ro = Po^T  */
  /* --------------------------------- */
  CHKERRQ(MatTranspose(p->A,MAT_INITIAL_MATRIX,&ptap->Rd));
  CHKERRQ(MatTranspose(p->B,MAT_INITIAL_MATRIX,&ptap->Ro));

  /* (1) compute symbolic AP = A_loc*P = Ad*P_loc + Ao*P_oth (api,apj) */
  /* ----------------------------------------------------------------- */
  p_loc  = (Mat_SeqAIJ*)P_loc->data;
  if (P_oth) p_oth = (Mat_SeqAIJ*)P_oth->data;

  /* create and initialize a linked list */
  CHKERRQ(PetscTableCreate(pn,pN,&ta)); /* for compute AP_loc and Cmpi */
  MatRowMergeMax_SeqAIJ(p_loc,P_loc->rmap->N,ta);
  MatRowMergeMax_SeqAIJ(p_oth,P_oth->rmap->N,ta);
  CHKERRQ(PetscTableGetCount(ta,&Crmax)); /* Crmax = nnz(sum of Prows) */

  CHKERRQ(PetscLLCondensedCreate_Scalable(Crmax,&lnk));

  /* Initial FreeSpace size is fill*(nnz(A) + nnz(P)) */
  if (ao) {
    CHKERRQ(PetscFreeSpaceGet(PetscRealIntMultTruncate(fill,PetscIntSumTruncate(ad->i[am],PetscIntSumTruncate(ao->i[am],p_loc->i[pm]))),&free_space));
  } else {
    CHKERRQ(PetscFreeSpaceGet(PetscRealIntMultTruncate(fill,PetscIntSumTruncate(ad->i[am],p_loc->i[pm])),&free_space));
  }
  current_space = free_space;
  nspacedouble  = 0;

  CHKERRQ(PetscMalloc1(am+1,&api));
  api[0] = 0;
  for (i=0; i<am; i++) {
    /* diagonal portion: Ad[i,:]*P */
    ai = ad->i; pi = p_loc->i;
    nzi = ai[i+1] - ai[i];
    aj  = ad->j + ai[i];
    for (j=0; j<nzi; j++) {
      row  = aj[j];
      pnz  = pi[row+1] - pi[row];
      Jptr = p_loc->j + pi[row];
      /* add non-zero cols of P into the sorted linked list lnk */
      CHKERRQ(PetscLLCondensedAddSorted_Scalable(pnz,Jptr,lnk));
    }
    /* off-diagonal portion: Ao[i,:]*P */
    if (ao) {
      ai = ao->i; pi = p_oth->i;
      nzi = ai[i+1] - ai[i];
      aj  = ao->j + ai[i];
      for (j=0; j<nzi; j++) {
        row  = aj[j];
        pnz  = pi[row+1] - pi[row];
        Jptr = p_oth->j + pi[row];
        CHKERRQ(PetscLLCondensedAddSorted_Scalable(pnz,Jptr,lnk));
      }
    }
    apnz     = lnk[0];
    api[i+1] = api[i] + apnz;

    /* if free space is not available, double the total space in the list */
    if (current_space->local_remaining<apnz) {
      CHKERRQ(PetscFreeSpaceGet(PetscIntSumTruncate(apnz,current_space->total_array_size),&current_space));
      nspacedouble++;
    }

    /* Copy data into free space, then initialize lnk */
    CHKERRQ(PetscLLCondensedClean_Scalable(apnz,current_space->array,lnk));

    current_space->array           += apnz;
    current_space->local_used      += apnz;
    current_space->local_remaining -= apnz;
  }
  /* Allocate space for apj and apv, initialize apj, and */
  /* destroy list of free space and other temporary array(s) */
  CHKERRQ(PetscCalloc2(api[am],&apj,api[am],&apv));
  CHKERRQ(PetscFreeSpaceContiguous(&free_space,apj));
  CHKERRQ(PetscLLCondensedDestroy_Scalable(lnk));

  /* Create AP_loc for reuse */
  CHKERRQ(MatCreateSeqAIJWithArrays(PETSC_COMM_SELF,am,pN,api,apj,apv,&ptap->AP_loc));
  CHKERRQ(MatSeqAIJCompactOutExtraColumns_SeqAIJ(ptap->AP_loc, &ptap->ltog));

#if defined(PETSC_USE_INFO)
  if (ao) {
    apfill = (PetscReal)api[am]/(ad->i[am]+ao->i[am]+p_loc->i[pm]+1);
  } else {
    apfill = (PetscReal)api[am]/(ad->i[am]+p_loc->i[pm]+1);
  }
  ptap->AP_loc->info.mallocs           = nspacedouble;
  ptap->AP_loc->info.fill_ratio_given  = fill;
  ptap->AP_loc->info.fill_ratio_needed = apfill;

  if (api[am]) {
    CHKERRQ(PetscInfo(ptap->AP_loc,"Scalable algorithm, AP_loc reallocs %" PetscInt_FMT "; Fill ratio: given %g needed %g.\n",nspacedouble,(double)fill,(double)apfill));
    CHKERRQ(PetscInfo(ptap->AP_loc,"Use MatPtAP(A,B,MatReuse,%g,&C) for best AP_loc performance.;\n",(double)apfill));
  } else {
    CHKERRQ(PetscInfo(ptap->AP_loc,"Scalable algorithm, AP_loc is empty \n"));
  }
#endif

  /* (2-1) compute symbolic Co = Ro*AP_loc  */
  /* -------------------------------------- */
  CHKERRQ(MatProductCreate(ptap->Ro,ptap->AP_loc,NULL,&ptap->C_oth));
  CHKERRQ(MatGetOptionsPrefix(A,&prefix));
  CHKERRQ(MatSetOptionsPrefix(ptap->C_oth,prefix));
  CHKERRQ(MatAppendOptionsPrefix(ptap->C_oth,"inner_offdiag_"));

  CHKERRQ(MatProductSetType(ptap->C_oth,MATPRODUCT_AB));
  CHKERRQ(MatProductSetAlgorithm(ptap->C_oth,"sorted"));
  CHKERRQ(MatProductSetFill(ptap->C_oth,fill));
  CHKERRQ(MatProductSetFromOptions(ptap->C_oth));
  CHKERRQ(MatProductSymbolic(ptap->C_oth));

  /* (3) send coj of C_oth to other processors  */
  /* ------------------------------------------ */
  /* determine row ownership */
  CHKERRQ(PetscLayoutCreate(comm,&rowmap));
  CHKERRQ(PetscLayoutSetLocalSize(rowmap, pn));
  CHKERRQ(PetscLayoutSetBlockSize(rowmap, 1));
  CHKERRQ(PetscLayoutSetUp(rowmap));
  CHKERRQ(PetscLayoutGetRanges(rowmap,&owners));

  /* determine the number of messages to send, their lengths */
  CHKERRQ(PetscMalloc4(size,&len_s,size,&len_si,size,&sstatus,size+2,&owners_co));
  CHKERRQ(PetscArrayzero(len_s,size));
  CHKERRQ(PetscArrayzero(len_si,size));

  c_oth = (Mat_SeqAIJ*)ptap->C_oth->data;
  coi   = c_oth->i; coj = c_oth->j;
  con   = ptap->C_oth->rmap->n;
  proc  = 0;
  CHKERRQ(ISLocalToGlobalMappingApply(ptap->ltog,coi[con],coj,coj));
  for (i=0; i<con; i++) {
    while (prmap[i] >= owners[proc+1]) proc++;
    len_si[proc]++;               /* num of rows in Co(=Pt*AP) to be sent to [proc] */
    len_s[proc] += coi[i+1] - coi[i]; /* num of nonzeros in Co to be sent to [proc] */
  }

  len          = 0; /* max length of buf_si[], see (4) */
  owners_co[0] = 0;
  nsend        = 0;
  for (proc=0; proc<size; proc++) {
    owners_co[proc+1] = owners_co[proc] + len_si[proc];
    if (len_s[proc]) {
      nsend++;
      len_si[proc] = 2*(len_si[proc] + 1); /* length of buf_si to be sent to [proc] */
      len         += len_si[proc];
    }
  }

  /* determine the number and length of messages to receive for coi and coj  */
  CHKERRQ(PetscGatherNumberOfMessages(comm,NULL,len_s,&nrecv));
  CHKERRQ(PetscGatherMessageLengths2(comm,nsend,nrecv,len_s,len_si,&id_r,&len_r,&len_ri));

  /* post the Irecv and Isend of coj */
  CHKERRQ(PetscCommGetNewTag(comm,&tagj));
  CHKERRQ(PetscPostIrecvInt(comm,tagj,nrecv,id_r,len_r,&buf_rj,&rwaits));
  CHKERRQ(PetscMalloc1(nsend+1,&swaits));
  for (proc=0, k=0; proc<size; proc++) {
    if (!len_s[proc]) continue;
    i    = owners_co[proc];
    CHKERRMPI(MPI_Isend(coj+coi[i],len_s[proc],MPIU_INT,proc,tagj,comm,swaits+k));
    k++;
  }

  /* (2-2) compute symbolic C_loc = Rd*AP_loc */
  /* ---------------------------------------- */
  CHKERRQ(MatProductCreate(ptap->Rd,ptap->AP_loc,NULL,&ptap->C_loc));
  CHKERRQ(MatProductSetType(ptap->C_loc,MATPRODUCT_AB));
  CHKERRQ(MatProductSetAlgorithm(ptap->C_loc,"default"));
  CHKERRQ(MatProductSetFill(ptap->C_loc,fill));

  CHKERRQ(MatSetOptionsPrefix(ptap->C_loc,prefix));
  CHKERRQ(MatAppendOptionsPrefix(ptap->C_loc,"inner_diag_"));

  CHKERRQ(MatProductSetFromOptions(ptap->C_loc));
  CHKERRQ(MatProductSymbolic(ptap->C_loc));

  c_loc = (Mat_SeqAIJ*)ptap->C_loc->data;
  CHKERRQ(ISLocalToGlobalMappingApply(ptap->ltog,c_loc->i[ptap->C_loc->rmap->n],c_loc->j,c_loc->j));

  /* receives coj are complete */
  for (i=0; i<nrecv; i++) {
    CHKERRMPI(MPI_Waitany(nrecv,rwaits,&icompleted,&rstatus));
  }
  CHKERRQ(PetscFree(rwaits));
  if (nsend) CHKERRMPI(MPI_Waitall(nsend,swaits,sstatus));

  /* add received column indices into ta to update Crmax */
  for (k=0; k<nrecv; k++) {/* k-th received message */
    Jptr = buf_rj[k];
    for (j=0; j<len_r[k]; j++) {
      CHKERRQ(PetscTableAdd(ta,*(Jptr+j)+1,1,INSERT_VALUES));
    }
  }
  CHKERRQ(PetscTableGetCount(ta,&Crmax));
  CHKERRQ(PetscTableDestroy(&ta));

  /* (4) send and recv coi */
  /*-----------------------*/
  CHKERRQ(PetscCommGetNewTag(comm,&tagi));
  CHKERRQ(PetscPostIrecvInt(comm,tagi,nrecv,id_r,len_ri,&buf_ri,&rwaits));
  CHKERRQ(PetscMalloc1(len+1,&buf_s));
  buf_si = buf_s;  /* points to the beginning of k-th msg to be sent */
  for (proc=0,k=0; proc<size; proc++) {
    if (!len_s[proc]) continue;
    /* form outgoing message for i-structure:
         buf_si[0]:                 nrows to be sent
               [1:nrows]:           row index (global)
               [nrows+1:2*nrows+1]: i-structure index
    */
    /*-------------------------------------------*/
    nrows       = len_si[proc]/2 - 1; /* num of rows in Co to be sent to [proc] */
    buf_si_i    = buf_si + nrows+1;
    buf_si[0]   = nrows;
    buf_si_i[0] = 0;
    nrows       = 0;
    for (i=owners_co[proc]; i<owners_co[proc+1]; i++) {
      nzi = coi[i+1] - coi[i];
      buf_si_i[nrows+1] = buf_si_i[nrows] + nzi;  /* i-structure */
      buf_si[nrows+1]   = prmap[i] -owners[proc]; /* local row index */
      nrows++;
    }
    CHKERRMPI(MPI_Isend(buf_si,len_si[proc],MPIU_INT,proc,tagi,comm,swaits+k));
    k++;
    buf_si += len_si[proc];
  }
  for (i=0; i<nrecv; i++) {
    CHKERRMPI(MPI_Waitany(nrecv,rwaits,&icompleted,&rstatus));
  }
  CHKERRQ(PetscFree(rwaits));
  if (nsend) CHKERRMPI(MPI_Waitall(nsend,swaits,sstatus));

  CHKERRQ(PetscFree4(len_s,len_si,sstatus,owners_co));
  CHKERRQ(PetscFree(len_ri));
  CHKERRQ(PetscFree(swaits));
  CHKERRQ(PetscFree(buf_s));

  /* (5) compute the local portion of Cmpi      */
  /* ------------------------------------------ */
  /* set initial free space to be Crmax, sufficient for holding nozeros in each row of Cmpi */
  CHKERRQ(PetscFreeSpaceGet(Crmax,&free_space));
  current_space = free_space;

  CHKERRQ(PetscMalloc3(nrecv,&buf_ri_k,nrecv,&nextrow,nrecv,&nextci));
  for (k=0; k<nrecv; k++) {
    buf_ri_k[k] = buf_ri[k]; /* beginning of k-th recved i-structure */
    nrows       = *buf_ri_k[k];
    nextrow[k]  = buf_ri_k[k] + 1;  /* next row number of k-th recved i-structure */
    nextci[k]   = buf_ri_k[k] + (nrows + 1); /* points to the next i-structure of k-th recved i-structure  */
  }

  ierr = MatPreallocateInitialize(comm,pn,pn,dnz,onz);CHKERRQ(ierr);
  CHKERRQ(PetscLLCondensedCreate_Scalable(Crmax,&lnk));
  for (i=0; i<pn; i++) {
    /* add C_loc into Cmpi */
    nzi  = c_loc->i[i+1] - c_loc->i[i];
    Jptr = c_loc->j + c_loc->i[i];
    CHKERRQ(PetscLLCondensedAddSorted_Scalable(nzi,Jptr,lnk));

    /* add received col data into lnk */
    for (k=0; k<nrecv; k++) { /* k-th received message */
      if (i == *nextrow[k]) { /* i-th row */
        nzi  = *(nextci[k]+1) - *nextci[k];
        Jptr = buf_rj[k] + *nextci[k];
        CHKERRQ(PetscLLCondensedAddSorted_Scalable(nzi,Jptr,lnk));
        nextrow[k]++; nextci[k]++;
      }
    }
    nzi = lnk[0];

    /* copy data into free space, then initialize lnk */
    CHKERRQ(PetscLLCondensedClean_Scalable(nzi,current_space->array,lnk));
    CHKERRQ(MatPreallocateSet(i+owners[rank],nzi,current_space->array,dnz,onz));
  }
  CHKERRQ(PetscFree3(buf_ri_k,nextrow,nextci));
  CHKERRQ(PetscLLCondensedDestroy_Scalable(lnk));
  CHKERRQ(PetscFreeSpaceDestroy(free_space));

  /* local sizes and preallocation */
  CHKERRQ(MatSetSizes(Cmpi,pn,pn,PETSC_DETERMINE,PETSC_DETERMINE));
  if (P->cmap->bs > 0) {
    CHKERRQ(PetscLayoutSetBlockSize(Cmpi->rmap,P->cmap->bs));
    CHKERRQ(PetscLayoutSetBlockSize(Cmpi->cmap,P->cmap->bs));
  }
  CHKERRQ(MatMPIAIJSetPreallocation(Cmpi,0,dnz,0,onz));
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);

  /* members in merge */
  CHKERRQ(PetscFree(id_r));
  CHKERRQ(PetscFree(len_r));
  CHKERRQ(PetscFree(buf_ri[0]));
  CHKERRQ(PetscFree(buf_ri));
  CHKERRQ(PetscFree(buf_rj[0]));
  CHKERRQ(PetscFree(buf_rj));
  CHKERRQ(PetscLayoutDestroy(&rowmap));

  nout = 0;
  CHKERRQ(ISGlobalToLocalMappingApply(ptap->ltog,IS_GTOLM_DROP,c_oth->i[ptap->C_oth->rmap->n],c_oth->j,&nout,c_oth->j));
  PetscCheckFalse(c_oth->i[ptap->C_oth->rmap->n] != nout,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Incorrect mapping %" PetscInt_FMT " != %" PetscInt_FMT,c_oth->i[ptap->C_oth->rmap->n],nout);
  CHKERRQ(ISGlobalToLocalMappingApply(ptap->ltog,IS_GTOLM_DROP,c_loc->i[ptap->C_loc->rmap->n],c_loc->j,&nout,c_loc->j));
  PetscCheckFalse(c_loc->i[ptap->C_loc->rmap->n] != nout,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Incorrect mapping %" PetscInt_FMT " != %" PetscInt_FMT,c_loc->i[ptap->C_loc->rmap->n],nout);

  /* attach the supporting struct to Cmpi for reuse */
  Cmpi->product->data    = ptap;
  Cmpi->product->view    = MatView_MPIAIJ_PtAP;
  Cmpi->product->destroy = MatDestroy_MPIAIJ_PtAP;

  /* Cmpi is not ready for use - assembly will be done by MatPtAPNumeric() */
  Cmpi->assembled        = PETSC_FALSE;
  Cmpi->ops->ptapnumeric = MatPtAPNumeric_MPIAIJ_MPIAIJ_scalable;
  PetscFunctionReturn(0);
}

static inline PetscErrorCode MatPtAPSymbolicComputeOneRowOfAP_private(Mat A,Mat P,Mat P_oth,const PetscInt *map,PetscInt dof,PetscInt i,PetscHSetI dht,PetscHSetI oht)
{
  Mat_MPIAIJ           *a=(Mat_MPIAIJ*)A->data,*p=(Mat_MPIAIJ*)P->data;
  Mat_SeqAIJ           *ad=(Mat_SeqAIJ*)(a->A)->data,*ao=(Mat_SeqAIJ*)(a->B)->data,*p_oth=(Mat_SeqAIJ*)P_oth->data,*pd=(Mat_SeqAIJ*)p->A->data,*po=(Mat_SeqAIJ*)p->B->data;
  PetscInt             *ai,nzi,j,*aj,row,col,*pi,*pj,pnz,nzpi,*p_othcols,k;
  PetscInt             pcstart,pcend,column,offset;

  PetscFunctionBegin;
  pcstart = P->cmap->rstart;
  pcstart *= dof;
  pcend   = P->cmap->rend;
  pcend   *= dof;
  /* diagonal portion: Ad[i,:]*P */
  ai = ad->i;
  nzi = ai[i+1] - ai[i];
  aj  = ad->j + ai[i];
  for (j=0; j<nzi; j++) {
    row  = aj[j];
    offset = row%dof;
    row   /= dof;
    nzpi = pd->i[row+1] - pd->i[row];
    pj  = pd->j + pd->i[row];
    for (k=0; k<nzpi; k++) {
      CHKERRQ(PetscHSetIAdd(dht,pj[k]*dof+offset+pcstart));
    }
  }
  /* off diag P */
  for (j=0; j<nzi; j++) {
    row  = aj[j];
    offset = row%dof;
    row   /= dof;
    nzpi = po->i[row+1] - po->i[row];
    pj  = po->j + po->i[row];
    for (k=0; k<nzpi; k++) {
      CHKERRQ(PetscHSetIAdd(oht,p->garray[pj[k]]*dof+offset));
    }
  }

  /* off diagonal part: Ao[i, :]*P_oth */
  if (ao) {
    ai = ao->i;
    pi = p_oth->i;
    nzi = ai[i+1] - ai[i];
    aj  = ao->j + ai[i];
    for (j=0; j<nzi; j++) {
      row  = aj[j];
      offset = a->garray[row]%dof;
      row  = map[row];
      pnz  = pi[row+1] - pi[row];
      p_othcols = p_oth->j + pi[row];
      for (col=0; col<pnz; col++) {
        column = p_othcols[col] * dof + offset;
        if (column>=pcstart && column<pcend) {
          CHKERRQ(PetscHSetIAdd(dht,column));
        } else {
          CHKERRQ(PetscHSetIAdd(oht,column));
        }
      }
    }
  } /* end if (ao) */
  PetscFunctionReturn(0);
}

static inline PetscErrorCode MatPtAPNumericComputeOneRowOfAP_private(Mat A,Mat P,Mat P_oth,const PetscInt *map,PetscInt dof,PetscInt i,PetscHMapIV hmap)
{
  Mat_MPIAIJ           *a=(Mat_MPIAIJ*)A->data,*p=(Mat_MPIAIJ*)P->data;
  Mat_SeqAIJ           *ad=(Mat_SeqAIJ*)(a->A)->data,*ao=(Mat_SeqAIJ*)(a->B)->data,*p_oth=(Mat_SeqAIJ*)P_oth->data,*pd=(Mat_SeqAIJ*)p->A->data,*po=(Mat_SeqAIJ*)p->B->data;
  PetscInt             *ai,nzi,j,*aj,row,col,*pi,pnz,*p_othcols,pcstart,*pj,k,nzpi,offset;
  PetscScalar          ra,*aa,*pa;

  PetscFunctionBegin;
  pcstart = P->cmap->rstart;
  pcstart *= dof;

  /* diagonal portion: Ad[i,:]*P */
  ai  = ad->i;
  nzi = ai[i+1] - ai[i];
  aj  = ad->j + ai[i];
  aa  = ad->a + ai[i];
  for (j=0; j<nzi; j++) {
    ra   = aa[j];
    row  = aj[j];
    offset = row%dof;
    row    /= dof;
    nzpi = pd->i[row+1] - pd->i[row];
    pj = pd->j + pd->i[row];
    pa = pd->a + pd->i[row];
    for (k=0; k<nzpi; k++) {
      CHKERRQ(PetscHMapIVAddValue(hmap,pj[k]*dof+offset+pcstart,ra*pa[k]));
    }
    CHKERRQ(PetscLogFlops(2.0*nzpi));
  }
  for (j=0; j<nzi; j++) {
    ra   = aa[j];
    row  = aj[j];
    offset = row%dof;
    row   /= dof;
    nzpi = po->i[row+1] - po->i[row];
    pj = po->j + po->i[row];
    pa = po->a + po->i[row];
    for (k=0; k<nzpi; k++) {
      CHKERRQ(PetscHMapIVAddValue(hmap,p->garray[pj[k]]*dof+offset,ra*pa[k]));
    }
    CHKERRQ(PetscLogFlops(2.0*nzpi));
  }

  /* off diagonal part: Ao[i, :]*P_oth */
  if (ao) {
    ai = ao->i;
    pi = p_oth->i;
    nzi = ai[i+1] - ai[i];
    aj  = ao->j + ai[i];
    aa  = ao->a + ai[i];
    for (j=0; j<nzi; j++) {
      row  = aj[j];
      offset = a->garray[row]%dof;
      row    = map[row];
      ra   = aa[j];
      pnz  = pi[row+1] - pi[row];
      p_othcols = p_oth->j + pi[row];
      pa   = p_oth->a + pi[row];
      for (col=0; col<pnz; col++) {
        CHKERRQ(PetscHMapIVAddValue(hmap,p_othcols[col]*dof+offset,ra*pa[col]));
      }
      CHKERRQ(PetscLogFlops(2.0*pnz));
    }
  } /* end if (ao) */

  PetscFunctionReturn(0);
}

PetscErrorCode MatGetBrowsOfAcols_MPIXAIJ(Mat,Mat,PetscInt dof,MatReuse,Mat*);

PetscErrorCode MatPtAPNumeric_MPIAIJ_MPIXAIJ_allatonce(Mat A,Mat P,PetscInt dof,Mat C)
{
  Mat_MPIAIJ        *p=(Mat_MPIAIJ*)P->data,*c=(Mat_MPIAIJ*)C->data;
  Mat_SeqAIJ        *cd,*co,*po=(Mat_SeqAIJ*)p->B->data,*pd=(Mat_SeqAIJ*)p->A->data;
  Mat_APMPI         *ptap;
  PetscHMapIV       hmap;
  PetscInt          i,j,jj,kk,nzi,*c_rmtj,voff,*c_othj,pn,pon,pcstart,pcend,ccstart,ccend,row,am,*poj,*pdj,*apindices,cmaxr,*c_rmtc,*c_rmtjj,*dcc,*occ,loc;
  PetscScalar       *c_rmta,*c_otha,*poa,*pda,*apvalues,*apvaluestmp,*c_rmtaa;
  PetscInt          offset,ii,pocol;
  const PetscInt    *mappingindices;
  IS                map;

  PetscFunctionBegin;
  MatCheckProduct(C,4);
  ptap = (Mat_APMPI*)C->product->data;
  PetscCheckFalse(!ptap,PetscObjectComm((PetscObject)C),PETSC_ERR_ARG_WRONGSTATE,"PtAP cannot be computed. Missing data");
  PetscCheckFalse(!ptap->P_oth,PetscObjectComm((PetscObject)C),PETSC_ERR_ARG_WRONGSTATE,"PtAP cannot be reused. Do not call MatProductClear()");

  CHKERRQ(MatZeroEntries(C));

  /* Get P_oth = ptap->P_oth  and P_loc = ptap->P_loc */
  /*-----------------------------------------------------*/
  if (ptap->reuse == MAT_REUSE_MATRIX) {
    /* P_oth and P_loc are obtained in MatPtASymbolic() when reuse == MAT_INITIAL_MATRIX */
    CHKERRQ(MatGetBrowsOfAcols_MPIXAIJ(A,P,dof,MAT_REUSE_MATRIX,&ptap->P_oth));
  }
  CHKERRQ(PetscObjectQuery((PetscObject)ptap->P_oth,"aoffdiagtopothmapping",(PetscObject*)&map));

  CHKERRQ(MatGetLocalSize(p->B,NULL,&pon));
  pon *= dof;
  CHKERRQ(PetscCalloc2(ptap->c_rmti[pon],&c_rmtj,ptap->c_rmti[pon],&c_rmta));
  CHKERRQ(MatGetLocalSize(A,&am,NULL));
  cmaxr = 0;
  for (i=0; i<pon; i++) {
    cmaxr = PetscMax(cmaxr,ptap->c_rmti[i+1]-ptap->c_rmti[i]);
  }
  CHKERRQ(PetscCalloc4(cmaxr,&apindices,cmaxr,&apvalues,cmaxr,&apvaluestmp,pon,&c_rmtc));
  CHKERRQ(PetscHMapIVCreate(&hmap));
  CHKERRQ(PetscHMapIVResize(hmap,cmaxr));
  CHKERRQ(ISGetIndices(map,&mappingindices));
  for (i=0; i<am && pon; i++) {
    CHKERRQ(PetscHMapIVClear(hmap));
    offset = i%dof;
    ii     = i/dof;
    nzi = po->i[ii+1] - po->i[ii];
    if (!nzi) continue;
    CHKERRQ(MatPtAPNumericComputeOneRowOfAP_private(A,P,ptap->P_oth,mappingindices,dof,i,hmap));
    voff = 0;
    CHKERRQ(PetscHMapIVGetPairs(hmap,&voff,apindices,apvalues));
    if (!voff) continue;

    /* Form C(ii, :) */
    poj = po->j + po->i[ii];
    poa = po->a + po->i[ii];
    for (j=0; j<nzi; j++) {
      pocol = poj[j]*dof+offset;
      c_rmtjj = c_rmtj + ptap->c_rmti[pocol];
      c_rmtaa = c_rmta + ptap->c_rmti[pocol];
      for (jj=0; jj<voff; jj++) {
        apvaluestmp[jj] = apvalues[jj]*poa[j];
        /*If the row is empty */
        if (!c_rmtc[pocol]) {
          c_rmtjj[jj] = apindices[jj];
          c_rmtaa[jj] = apvaluestmp[jj];
          c_rmtc[pocol]++;
        } else {
          CHKERRQ(PetscFindInt(apindices[jj],c_rmtc[pocol],c_rmtjj,&loc));
          if (loc>=0){ /* hit */
            c_rmtaa[loc] += apvaluestmp[jj];
            CHKERRQ(PetscLogFlops(1.0));
          } else { /* new element */
            loc = -(loc+1);
            /* Move data backward */
            for (kk=c_rmtc[pocol]; kk>loc; kk--) {
              c_rmtjj[kk] = c_rmtjj[kk-1];
              c_rmtaa[kk] = c_rmtaa[kk-1];
            }/* End kk */
            c_rmtjj[loc] = apindices[jj];
            c_rmtaa[loc] = apvaluestmp[jj];
            c_rmtc[pocol]++;
          }
        }
        CHKERRQ(PetscLogFlops(voff));
      } /* End jj */
    } /* End j */
  } /* End i */

  CHKERRQ(PetscFree4(apindices,apvalues,apvaluestmp,c_rmtc));
  CHKERRQ(PetscHMapIVDestroy(&hmap));

  CHKERRQ(MatGetLocalSize(P,NULL,&pn));
  pn *= dof;
  CHKERRQ(PetscCalloc2(ptap->c_othi[pn],&c_othj,ptap->c_othi[pn],&c_otha));

  CHKERRQ(PetscSFReduceBegin(ptap->sf,MPIU_INT,c_rmtj,c_othj,MPI_REPLACE));
  CHKERRQ(PetscSFReduceBegin(ptap->sf,MPIU_SCALAR,c_rmta,c_otha,MPI_REPLACE));
  CHKERRQ(MatGetOwnershipRangeColumn(P,&pcstart,&pcend));
  pcstart = pcstart*dof;
  pcend   = pcend*dof;
  cd = (Mat_SeqAIJ*)(c->A)->data;
  co = (Mat_SeqAIJ*)(c->B)->data;

  cmaxr = 0;
  for (i=0; i<pn; i++) {
    cmaxr = PetscMax(cmaxr,(cd->i[i+1]-cd->i[i])+(co->i[i+1]-co->i[i]));
  }
  CHKERRQ(PetscCalloc5(cmaxr,&apindices,cmaxr,&apvalues,cmaxr,&apvaluestmp,pn,&dcc,pn,&occ));
  CHKERRQ(PetscHMapIVCreate(&hmap));
  CHKERRQ(PetscHMapIVResize(hmap,cmaxr));
  for (i=0; i<am && pn; i++) {
    CHKERRQ(PetscHMapIVClear(hmap));
    offset = i%dof;
    ii     = i/dof;
    nzi = pd->i[ii+1] - pd->i[ii];
    if (!nzi) continue;
    CHKERRQ(MatPtAPNumericComputeOneRowOfAP_private(A,P,ptap->P_oth,mappingindices,dof,i,hmap));
    voff = 0;
    CHKERRQ(PetscHMapIVGetPairs(hmap,&voff,apindices,apvalues));
    if (!voff) continue;
    /* Form C(ii, :) */
    pdj = pd->j + pd->i[ii];
    pda = pd->a + pd->i[ii];
    for (j=0; j<nzi; j++) {
      row = pcstart + pdj[j] * dof + offset;
      for (jj=0; jj<voff; jj++) {
        apvaluestmp[jj] = apvalues[jj]*pda[j];
      }
      CHKERRQ(PetscLogFlops(voff));
      CHKERRQ(MatSetValues(C,1,&row,voff,apindices,apvaluestmp,ADD_VALUES));
    }
  }
  CHKERRQ(ISRestoreIndices(map,&mappingindices));
  CHKERRQ(MatGetOwnershipRangeColumn(C,&ccstart,&ccend));
  CHKERRQ(PetscFree5(apindices,apvalues,apvaluestmp,dcc,occ));
  CHKERRQ(PetscHMapIVDestroy(&hmap));
  CHKERRQ(PetscSFReduceEnd(ptap->sf,MPIU_INT,c_rmtj,c_othj,MPI_REPLACE));
  CHKERRQ(PetscSFReduceEnd(ptap->sf,MPIU_SCALAR,c_rmta,c_otha,MPI_REPLACE));
  CHKERRQ(PetscFree2(c_rmtj,c_rmta));

  /* Add contributions from remote */
  for (i = 0; i < pn; i++) {
    row = i + pcstart;
    CHKERRQ(MatSetValues(C,1,&row,ptap->c_othi[i+1]-ptap->c_othi[i],c_othj+ptap->c_othi[i],c_otha+ptap->c_othi[i],ADD_VALUES));
  }
  CHKERRQ(PetscFree2(c_othj,c_otha));

  CHKERRQ(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));

  ptap->reuse = MAT_REUSE_MATRIX;
  PetscFunctionReturn(0);
}

PetscErrorCode MatPtAPNumeric_MPIAIJ_MPIAIJ_allatonce(Mat A,Mat P,Mat C)
{
  PetscFunctionBegin;

  CHKERRQ(MatPtAPNumeric_MPIAIJ_MPIXAIJ_allatonce(A,P,1,C));
  PetscFunctionReturn(0);
}

PetscErrorCode MatPtAPNumeric_MPIAIJ_MPIXAIJ_allatonce_merged(Mat A,Mat P,PetscInt dof,Mat C)
{
  Mat_MPIAIJ        *p=(Mat_MPIAIJ*)P->data,*c=(Mat_MPIAIJ*)C->data;
  Mat_SeqAIJ        *cd,*co,*po=(Mat_SeqAIJ*)p->B->data,*pd=(Mat_SeqAIJ*)p->A->data;
  Mat_APMPI         *ptap;
  PetscHMapIV       hmap;
  PetscInt          i,j,jj,kk,nzi,dnzi,*c_rmtj,voff,*c_othj,pn,pon,pcstart,pcend,row,am,*poj,*pdj,*apindices,cmaxr,*c_rmtc,*c_rmtjj,loc;
  PetscScalar       *c_rmta,*c_otha,*poa,*pda,*apvalues,*apvaluestmp,*c_rmtaa;
  PetscInt          offset,ii,pocol;
  const PetscInt    *mappingindices;
  IS                map;

  PetscFunctionBegin;
  MatCheckProduct(C,4);
  ptap = (Mat_APMPI*)C->product->data;
  PetscCheckFalse(!ptap,PetscObjectComm((PetscObject)C),PETSC_ERR_ARG_WRONGSTATE,"PtAP cannot be computed. Missing data");
  PetscCheckFalse(!ptap->P_oth,PetscObjectComm((PetscObject)C),PETSC_ERR_ARG_WRONGSTATE,"PtAP cannot be reused. Do not call MatProductClear()");

  CHKERRQ(MatZeroEntries(C));

  /* Get P_oth = ptap->P_oth  and P_loc = ptap->P_loc */
  /*-----------------------------------------------------*/
  if (ptap->reuse == MAT_REUSE_MATRIX) {
    /* P_oth and P_loc are obtained in MatPtASymbolic() when reuse == MAT_INITIAL_MATRIX */
    CHKERRQ(MatGetBrowsOfAcols_MPIXAIJ(A,P,dof,MAT_REUSE_MATRIX,&ptap->P_oth));
  }
  CHKERRQ(PetscObjectQuery((PetscObject)ptap->P_oth,"aoffdiagtopothmapping",(PetscObject*)&map));
  CHKERRQ(MatGetLocalSize(p->B,NULL,&pon));
  pon *= dof;
  CHKERRQ(MatGetLocalSize(P,NULL,&pn));
  pn  *= dof;

  CHKERRQ(PetscCalloc2(ptap->c_rmti[pon],&c_rmtj,ptap->c_rmti[pon],&c_rmta));
  CHKERRQ(MatGetLocalSize(A,&am,NULL));
  CHKERRQ(MatGetOwnershipRangeColumn(P,&pcstart,&pcend));
  pcstart *= dof;
  pcend   *= dof;
  cmaxr = 0;
  for (i=0; i<pon; i++) {
    cmaxr = PetscMax(cmaxr,ptap->c_rmti[i+1]-ptap->c_rmti[i]);
  }
  cd = (Mat_SeqAIJ*)(c->A)->data;
  co = (Mat_SeqAIJ*)(c->B)->data;
  for (i=0; i<pn; i++) {
    cmaxr = PetscMax(cmaxr,(cd->i[i+1]-cd->i[i])+(co->i[i+1]-co->i[i]));
  }
  CHKERRQ(PetscCalloc4(cmaxr,&apindices,cmaxr,&apvalues,cmaxr,&apvaluestmp,pon,&c_rmtc));
  CHKERRQ(PetscHMapIVCreate(&hmap));
  CHKERRQ(PetscHMapIVResize(hmap,cmaxr));
  CHKERRQ(ISGetIndices(map,&mappingindices));
  for (i=0; i<am && (pon || pn); i++) {
    CHKERRQ(PetscHMapIVClear(hmap));
    offset = i%dof;
    ii     = i/dof;
    nzi  = po->i[ii+1] - po->i[ii];
    dnzi = pd->i[ii+1] - pd->i[ii];
    if (!nzi && !dnzi) continue;
    CHKERRQ(MatPtAPNumericComputeOneRowOfAP_private(A,P,ptap->P_oth,mappingindices,dof,i,hmap));
    voff = 0;
    CHKERRQ(PetscHMapIVGetPairs(hmap,&voff,apindices,apvalues));
    if (!voff) continue;

    /* Form remote C(ii, :) */
    poj = po->j + po->i[ii];
    poa = po->a + po->i[ii];
    for (j=0; j<nzi; j++) {
      pocol = poj[j]*dof+offset;
      c_rmtjj = c_rmtj + ptap->c_rmti[pocol];
      c_rmtaa = c_rmta + ptap->c_rmti[pocol];
      for (jj=0; jj<voff; jj++) {
        apvaluestmp[jj] = apvalues[jj]*poa[j];
        /*If the row is empty */
        if (!c_rmtc[pocol]) {
          c_rmtjj[jj] = apindices[jj];
          c_rmtaa[jj] = apvaluestmp[jj];
          c_rmtc[pocol]++;
        } else {
          CHKERRQ(PetscFindInt(apindices[jj],c_rmtc[pocol],c_rmtjj,&loc));
          if (loc>=0){ /* hit */
            c_rmtaa[loc] += apvaluestmp[jj];
            CHKERRQ(PetscLogFlops(1.0));
          } else { /* new element */
            loc = -(loc+1);
            /* Move data backward */
            for (kk=c_rmtc[pocol]; kk>loc; kk--) {
              c_rmtjj[kk] = c_rmtjj[kk-1];
              c_rmtaa[kk] = c_rmtaa[kk-1];
            }/* End kk */
            c_rmtjj[loc] = apindices[jj];
            c_rmtaa[loc] = apvaluestmp[jj];
            c_rmtc[pocol]++;
          }
        }
      } /* End jj */
      CHKERRQ(PetscLogFlops(voff));
    } /* End j */

    /* Form local C(ii, :) */
    pdj = pd->j + pd->i[ii];
    pda = pd->a + pd->i[ii];
    for (j=0; j<dnzi; j++) {
      row = pcstart + pdj[j] * dof + offset;
      for (jj=0; jj<voff; jj++) {
        apvaluestmp[jj] = apvalues[jj]*pda[j];
      }/* End kk */
      CHKERRQ(PetscLogFlops(voff));
      CHKERRQ(MatSetValues(C,1,&row,voff,apindices,apvaluestmp,ADD_VALUES));
    }/* End j */
  } /* End i */

  CHKERRQ(ISRestoreIndices(map,&mappingindices));
  CHKERRQ(PetscFree4(apindices,apvalues,apvaluestmp,c_rmtc));
  CHKERRQ(PetscHMapIVDestroy(&hmap));
  CHKERRQ(PetscCalloc2(ptap->c_othi[pn],&c_othj,ptap->c_othi[pn],&c_otha));

  CHKERRQ(PetscSFReduceBegin(ptap->sf,MPIU_INT,c_rmtj,c_othj,MPI_REPLACE));
  CHKERRQ(PetscSFReduceBegin(ptap->sf,MPIU_SCALAR,c_rmta,c_otha,MPI_REPLACE));
  CHKERRQ(PetscSFReduceEnd(ptap->sf,MPIU_INT,c_rmtj,c_othj,MPI_REPLACE));
  CHKERRQ(PetscSFReduceEnd(ptap->sf,MPIU_SCALAR,c_rmta,c_otha,MPI_REPLACE));
  CHKERRQ(PetscFree2(c_rmtj,c_rmta));

  /* Add contributions from remote */
  for (i = 0; i < pn; i++) {
    row = i + pcstart;
    CHKERRQ(MatSetValues(C,1,&row,ptap->c_othi[i+1]-ptap->c_othi[i],c_othj+ptap->c_othi[i],c_otha+ptap->c_othi[i],ADD_VALUES));
  }
  CHKERRQ(PetscFree2(c_othj,c_otha));

  CHKERRQ(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));

  ptap->reuse = MAT_REUSE_MATRIX;
  PetscFunctionReturn(0);
}

PetscErrorCode MatPtAPNumeric_MPIAIJ_MPIAIJ_allatonce_merged(Mat A,Mat P,Mat C)
{
  PetscFunctionBegin;

  CHKERRQ(MatPtAPNumeric_MPIAIJ_MPIXAIJ_allatonce_merged(A,P,1,C));
  PetscFunctionReturn(0);
}

/* TODO: move algorithm selection to MatProductSetFromOptions */
PetscErrorCode MatPtAPSymbolic_MPIAIJ_MPIXAIJ_allatonce(Mat A,Mat P,PetscInt dof,PetscReal fill,Mat Cmpi)
{
  Mat_APMPI           *ptap;
  Mat_MPIAIJ          *p=(Mat_MPIAIJ*)P->data;
  MPI_Comm            comm;
  Mat_SeqAIJ          *pd=(Mat_SeqAIJ*)p->A->data,*po=(Mat_SeqAIJ*)p->B->data;
  MatType             mtype;
  PetscSF             sf;
  PetscSFNode         *iremote;
  PetscInt            rootspacesize,*rootspace,*rootspaceoffsets,nleaves;
  const PetscInt      *rootdegrees;
  PetscHSetI          ht,oht,*hta,*hto;
  PetscInt            pn,pon,*c_rmtc,i,j,nzi,htsize,htosize,*c_rmtj,off,*c_othj,rcvncols,sendncols,*c_rmtoffsets;
  PetscInt            lidx,*rdj,col,pcstart,pcend,*dnz,*onz,am,arstart,arend,*poj,*pdj;
  PetscInt            nalg=2,alg=0,offset,ii;
  PetscMPIInt         owner;
  const PetscInt      *mappingindices;
  PetscBool           flg;
  const char          *algTypes[2] = {"overlapping","merged"};
  IS                  map;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  MatCheckProduct(Cmpi,5);
  PetscCheckFalse(Cmpi->product->data,PetscObjectComm((PetscObject)Cmpi),PETSC_ERR_PLIB,"Product data not empty");
  CHKERRQ(PetscObjectGetComm((PetscObject)A,&comm));

  /* Create symbolic parallel matrix Cmpi */
  CHKERRQ(MatGetLocalSize(P,NULL,&pn));
  pn *= dof;
  CHKERRQ(MatGetType(A,&mtype));
  CHKERRQ(MatSetType(Cmpi,mtype));
  CHKERRQ(MatSetSizes(Cmpi,pn,pn,PETSC_DETERMINE,PETSC_DETERMINE));

  CHKERRQ(PetscNew(&ptap));
  ptap->reuse = MAT_INITIAL_MATRIX;
  ptap->algType = 2;

  /* Get P_oth by taking rows of P (= non-zero cols of local A) from other processors */
  CHKERRQ(MatGetBrowsOfAcols_MPIXAIJ(A,P,dof,MAT_INITIAL_MATRIX,&ptap->P_oth));
  CHKERRQ(PetscObjectQuery((PetscObject)ptap->P_oth,"aoffdiagtopothmapping",(PetscObject*)&map));
  /* This equals to the number of offdiag columns in P */
  CHKERRQ(MatGetLocalSize(p->B,NULL,&pon));
  pon *= dof;
  /* offsets */
  CHKERRQ(PetscMalloc1(pon+1,&ptap->c_rmti));
  /* The number of columns we will send to remote ranks */
  CHKERRQ(PetscMalloc1(pon,&c_rmtc));
  CHKERRQ(PetscMalloc1(pon,&hta));
  for (i=0; i<pon; i++) {
    CHKERRQ(PetscHSetICreate(&hta[i]));
  }
  CHKERRQ(MatGetLocalSize(A,&am,NULL));
  CHKERRQ(MatGetOwnershipRange(A,&arstart,&arend));
  /* Create hash table to merge all columns for C(i, :) */
  CHKERRQ(PetscHSetICreate(&ht));

  CHKERRQ(ISGetIndices(map,&mappingindices));
  ptap->c_rmti[0] = 0;
  /* 2) Pass 1: calculate the size for C_rmt (a matrix need to be sent to other processors)  */
  for (i=0; i<am && pon; i++) {
    /* Form one row of AP */
    CHKERRQ(PetscHSetIClear(ht));
    offset = i%dof;
    ii     = i/dof;
    /* If the off diag is empty, we should not do any calculation */
    nzi = po->i[ii+1] - po->i[ii];
    if (!nzi) continue;

    CHKERRQ(MatPtAPSymbolicComputeOneRowOfAP_private(A,P,ptap->P_oth,mappingindices,dof,i,ht,ht));
    CHKERRQ(PetscHSetIGetSize(ht,&htsize));
    /* If AP is empty, just continue */
    if (!htsize) continue;
    /* Form C(ii, :) */
    poj = po->j + po->i[ii];
    for (j=0; j<nzi; j++) {
      CHKERRQ(PetscHSetIUpdate(hta[poj[j]*dof+offset],ht));
    }
  }

  for (i=0; i<pon; i++) {
    CHKERRQ(PetscHSetIGetSize(hta[i],&htsize));
    ptap->c_rmti[i+1] = ptap->c_rmti[i] + htsize;
    c_rmtc[i] = htsize;
  }

  CHKERRQ(PetscMalloc1(ptap->c_rmti[pon],&c_rmtj));

  for (i=0; i<pon; i++) {
    off = 0;
    CHKERRQ(PetscHSetIGetElems(hta[i],&off,c_rmtj+ptap->c_rmti[i]));
    CHKERRQ(PetscHSetIDestroy(&hta[i]));
  }
  CHKERRQ(PetscFree(hta));

  CHKERRQ(PetscMalloc1(pon,&iremote));
  for (i=0; i<pon; i++) {
    owner = 0; lidx = 0;
    offset = i%dof;
    ii     = i/dof;
    CHKERRQ(PetscLayoutFindOwnerIndex(P->cmap,p->garray[ii],&owner,&lidx));
    iremote[i].index = lidx*dof + offset;
    iremote[i].rank  = owner;
  }

  CHKERRQ(PetscSFCreate(comm,&sf));
  CHKERRQ(PetscSFSetGraph(sf,pn,pon,NULL,PETSC_OWN_POINTER,iremote,PETSC_OWN_POINTER));
  /* Reorder ranks properly so that the data handled by gather and scatter have the same order */
  CHKERRQ(PetscSFSetRankOrder(sf,PETSC_TRUE));
  CHKERRQ(PetscSFSetFromOptions(sf));
  CHKERRQ(PetscSFSetUp(sf));
  /* How many neighbors have contributions to my rows? */
  CHKERRQ(PetscSFComputeDegreeBegin(sf,&rootdegrees));
  CHKERRQ(PetscSFComputeDegreeEnd(sf,&rootdegrees));
  rootspacesize = 0;
  for (i = 0; i < pn; i++) {
    rootspacesize += rootdegrees[i];
  }
  CHKERRQ(PetscMalloc1(rootspacesize,&rootspace));
  CHKERRQ(PetscMalloc1(rootspacesize+1,&rootspaceoffsets));
  /* Get information from leaves
   * Number of columns other people contribute to my rows
   * */
  CHKERRQ(PetscSFGatherBegin(sf,MPIU_INT,c_rmtc,rootspace));
  CHKERRQ(PetscSFGatherEnd(sf,MPIU_INT,c_rmtc,rootspace));
  CHKERRQ(PetscFree(c_rmtc));
  CHKERRQ(PetscCalloc1(pn+1,&ptap->c_othi));
  /* The number of columns is received for each row */
  ptap->c_othi[0] = 0;
  rootspacesize = 0;
  rootspaceoffsets[0] = 0;
  for (i = 0; i < pn; i++) {
    rcvncols = 0;
    for (j = 0; j<rootdegrees[i]; j++) {
      rcvncols += rootspace[rootspacesize];
      rootspaceoffsets[rootspacesize+1] = rootspaceoffsets[rootspacesize] + rootspace[rootspacesize];
      rootspacesize++;
    }
    ptap->c_othi[i+1] = ptap->c_othi[i] + rcvncols;
  }
  CHKERRQ(PetscFree(rootspace));

  CHKERRQ(PetscMalloc1(pon,&c_rmtoffsets));
  CHKERRQ(PetscSFScatterBegin(sf,MPIU_INT,rootspaceoffsets,c_rmtoffsets));
  CHKERRQ(PetscSFScatterEnd(sf,MPIU_INT,rootspaceoffsets,c_rmtoffsets));
  CHKERRQ(PetscSFDestroy(&sf));
  CHKERRQ(PetscFree(rootspaceoffsets));

  CHKERRQ(PetscCalloc1(ptap->c_rmti[pon],&iremote));
  nleaves = 0;
  for (i = 0; i<pon; i++) {
    owner = 0;
    ii = i/dof;
    CHKERRQ(PetscLayoutFindOwnerIndex(P->cmap,p->garray[ii],&owner,NULL));
    sendncols = ptap->c_rmti[i+1] - ptap->c_rmti[i];
    for (j=0; j<sendncols; j++) {
      iremote[nleaves].rank = owner;
      iremote[nleaves++].index = c_rmtoffsets[i] + j;
    }
  }
  CHKERRQ(PetscFree(c_rmtoffsets));
  CHKERRQ(PetscCalloc1(ptap->c_othi[pn],&c_othj));

  CHKERRQ(PetscSFCreate(comm,&ptap->sf));
  CHKERRQ(PetscSFSetGraph(ptap->sf,ptap->c_othi[pn],nleaves,NULL,PETSC_OWN_POINTER,iremote,PETSC_OWN_POINTER));
  CHKERRQ(PetscSFSetFromOptions(ptap->sf));
  /* One to one map */
  CHKERRQ(PetscSFReduceBegin(ptap->sf,MPIU_INT,c_rmtj,c_othj,MPI_REPLACE));

  CHKERRQ(PetscMalloc2(pn,&dnz,pn,&onz));
  CHKERRQ(PetscHSetICreate(&oht));
  CHKERRQ(MatGetOwnershipRangeColumn(P,&pcstart,&pcend));
  pcstart *= dof;
  pcend   *= dof;
  CHKERRQ(PetscMalloc2(pn,&hta,pn,&hto));
  for (i=0; i<pn; i++) {
    CHKERRQ(PetscHSetICreate(&hta[i]));
    CHKERRQ(PetscHSetICreate(&hto[i]));
  }
  /* Work on local part */
  /* 4) Pass 1: Estimate memory for C_loc */
  for (i=0; i<am && pn; i++) {
    CHKERRQ(PetscHSetIClear(ht));
    CHKERRQ(PetscHSetIClear(oht));
    offset = i%dof;
    ii     = i/dof;
    nzi = pd->i[ii+1] - pd->i[ii];
    if (!nzi) continue;

    CHKERRQ(MatPtAPSymbolicComputeOneRowOfAP_private(A,P,ptap->P_oth,mappingindices,dof,i,ht,oht));
    CHKERRQ(PetscHSetIGetSize(ht,&htsize));
    CHKERRQ(PetscHSetIGetSize(oht,&htosize));
    if (!(htsize+htosize)) continue;
    /* Form C(ii, :) */
    pdj = pd->j + pd->i[ii];
    for (j=0; j<nzi; j++) {
      CHKERRQ(PetscHSetIUpdate(hta[pdj[j]*dof+offset],ht));
      CHKERRQ(PetscHSetIUpdate(hto[pdj[j]*dof+offset],oht));
    }
  }

  CHKERRQ(ISRestoreIndices(map,&mappingindices));

  CHKERRQ(PetscHSetIDestroy(&ht));
  CHKERRQ(PetscHSetIDestroy(&oht));

  /* Get remote data */
  CHKERRQ(PetscSFReduceEnd(ptap->sf,MPIU_INT,c_rmtj,c_othj,MPI_REPLACE));
  CHKERRQ(PetscFree(c_rmtj));

  for (i = 0; i < pn; i++) {
    nzi = ptap->c_othi[i+1] - ptap->c_othi[i];
    rdj = c_othj + ptap->c_othi[i];
    for (j = 0; j < nzi; j++) {
      col = rdj[j];
      /* diag part */
      if (col>=pcstart && col<pcend) {
        CHKERRQ(PetscHSetIAdd(hta[i],col));
      } else { /* off diag */
        CHKERRQ(PetscHSetIAdd(hto[i],col));
      }
    }
    CHKERRQ(PetscHSetIGetSize(hta[i],&htsize));
    dnz[i] = htsize;
    CHKERRQ(PetscHSetIDestroy(&hta[i]));
    CHKERRQ(PetscHSetIGetSize(hto[i],&htsize));
    onz[i] = htsize;
    CHKERRQ(PetscHSetIDestroy(&hto[i]));
  }

  CHKERRQ(PetscFree2(hta,hto));
  CHKERRQ(PetscFree(c_othj));

  /* local sizes and preallocation */
  CHKERRQ(MatSetSizes(Cmpi,pn,pn,PETSC_DETERMINE,PETSC_DETERMINE));
  CHKERRQ(MatSetBlockSizes(Cmpi,dof>1? dof: P->cmap->bs,dof>1? dof: P->cmap->bs));
  CHKERRQ(MatMPIAIJSetPreallocation(Cmpi,0,dnz,0,onz));
  CHKERRQ(MatSetUp(Cmpi));
  CHKERRQ(PetscFree2(dnz,onz));

  /* attach the supporting struct to Cmpi for reuse */
  Cmpi->product->data    = ptap;
  Cmpi->product->destroy = MatDestroy_MPIAIJ_PtAP;
  Cmpi->product->view    = MatView_MPIAIJ_PtAP;

  /* Cmpi is not ready for use - assembly will be done by MatPtAPNumeric() */
  Cmpi->assembled = PETSC_FALSE;
  /* pick an algorithm */
  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)A),((PetscObject)A)->prefix,"MatPtAP","Mat");CHKERRQ(ierr);
  alg = 0;
  CHKERRQ(PetscOptionsEList("-matptap_allatonce_via","PtAP allatonce numeric approach","MatPtAP",algTypes,nalg,algTypes[alg],&alg,&flg));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  switch (alg) {
    case 0:
      Cmpi->ops->ptapnumeric = MatPtAPNumeric_MPIAIJ_MPIAIJ_allatonce;
      break;
    case 1:
      Cmpi->ops->ptapnumeric = MatPtAPNumeric_MPIAIJ_MPIAIJ_allatonce_merged;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG," Unsupported allatonce numerical algorithm ");
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatPtAPSymbolic_MPIAIJ_MPIAIJ_allatonce(Mat A,Mat P,PetscReal fill,Mat C)
{
  PetscFunctionBegin;
  CHKERRQ(MatPtAPSymbolic_MPIAIJ_MPIXAIJ_allatonce(A,P,1,fill,C));
  PetscFunctionReturn(0);
}

PetscErrorCode MatPtAPSymbolic_MPIAIJ_MPIXAIJ_allatonce_merged(Mat A,Mat P,PetscInt dof,PetscReal fill,Mat Cmpi)
{
  Mat_APMPI           *ptap;
  Mat_MPIAIJ          *p=(Mat_MPIAIJ*)P->data;
  MPI_Comm            comm;
  Mat_SeqAIJ          *pd=(Mat_SeqAIJ*)p->A->data,*po=(Mat_SeqAIJ*)p->B->data;
  MatType             mtype;
  PetscSF             sf;
  PetscSFNode         *iremote;
  PetscInt            rootspacesize,*rootspace,*rootspaceoffsets,nleaves;
  const PetscInt      *rootdegrees;
  PetscHSetI          ht,oht,*hta,*hto,*htd;
  PetscInt            pn,pon,*c_rmtc,i,j,nzi,dnzi,htsize,htosize,*c_rmtj,off,*c_othj,rcvncols,sendncols,*c_rmtoffsets;
  PetscInt            lidx,*rdj,col,pcstart,pcend,*dnz,*onz,am,arstart,arend,*poj,*pdj;
  PetscInt            nalg=2,alg=0,offset,ii;
  PetscMPIInt         owner;
  PetscBool           flg;
  const char          *algTypes[2] = {"merged","overlapping"};
  const PetscInt      *mappingindices;
  IS                  map;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  MatCheckProduct(Cmpi,5);
  PetscCheckFalse(Cmpi->product->data,PetscObjectComm((PetscObject)Cmpi),PETSC_ERR_PLIB,"Product data not empty");
  CHKERRQ(PetscObjectGetComm((PetscObject)A,&comm));

  /* Create symbolic parallel matrix Cmpi */
  CHKERRQ(MatGetLocalSize(P,NULL,&pn));
  pn *= dof;
  CHKERRQ(MatGetType(A,&mtype));
  CHKERRQ(MatSetType(Cmpi,mtype));
  CHKERRQ(MatSetSizes(Cmpi,pn,pn,PETSC_DETERMINE,PETSC_DETERMINE));

  CHKERRQ(PetscNew(&ptap));
  ptap->reuse = MAT_INITIAL_MATRIX;
  ptap->algType = 3;

  /* 0) Get P_oth by taking rows of P (= non-zero cols of local A) from other processors */
  CHKERRQ(MatGetBrowsOfAcols_MPIXAIJ(A,P,dof,MAT_INITIAL_MATRIX,&ptap->P_oth));
  CHKERRQ(PetscObjectQuery((PetscObject)ptap->P_oth,"aoffdiagtopothmapping",(PetscObject*)&map));

  /* This equals to the number of offdiag columns in P */
  CHKERRQ(MatGetLocalSize(p->B,NULL,&pon));
  pon *= dof;
  /* offsets */
  CHKERRQ(PetscMalloc1(pon+1,&ptap->c_rmti));
  /* The number of columns we will send to remote ranks */
  CHKERRQ(PetscMalloc1(pon,&c_rmtc));
  CHKERRQ(PetscMalloc1(pon,&hta));
  for (i=0; i<pon; i++) {
    CHKERRQ(PetscHSetICreate(&hta[i]));
  }
  CHKERRQ(MatGetLocalSize(A,&am,NULL));
  CHKERRQ(MatGetOwnershipRange(A,&arstart,&arend));
  /* Create hash table to merge all columns for C(i, :) */
  CHKERRQ(PetscHSetICreate(&ht));
  CHKERRQ(PetscHSetICreate(&oht));
  CHKERRQ(PetscMalloc2(pn,&htd,pn,&hto));
  for (i=0; i<pn; i++) {
    CHKERRQ(PetscHSetICreate(&htd[i]));
    CHKERRQ(PetscHSetICreate(&hto[i]));
  }

  CHKERRQ(ISGetIndices(map,&mappingindices));
  ptap->c_rmti[0] = 0;
  /* 2) Pass 1: calculate the size for C_rmt (a matrix need to be sent to other processors)  */
  for (i=0; i<am && (pon || pn); i++) {
    /* Form one row of AP */
    CHKERRQ(PetscHSetIClear(ht));
    CHKERRQ(PetscHSetIClear(oht));
    offset = i%dof;
    ii     = i/dof;
    /* If the off diag is empty, we should not do any calculation */
    nzi = po->i[ii+1] - po->i[ii];
    dnzi = pd->i[ii+1] - pd->i[ii];
    if (!nzi && !dnzi) continue;

    CHKERRQ(MatPtAPSymbolicComputeOneRowOfAP_private(A,P,ptap->P_oth,mappingindices,dof,i,ht,oht));
    CHKERRQ(PetscHSetIGetSize(ht,&htsize));
    CHKERRQ(PetscHSetIGetSize(oht,&htosize));
    /* If AP is empty, just continue */
    if (!(htsize+htosize)) continue;

    /* Form remote C(ii, :) */
    poj = po->j + po->i[ii];
    for (j=0; j<nzi; j++) {
      CHKERRQ(PetscHSetIUpdate(hta[poj[j]*dof+offset],ht));
      CHKERRQ(PetscHSetIUpdate(hta[poj[j]*dof+offset],oht));
    }

    /* Form local C(ii, :) */
    pdj = pd->j + pd->i[ii];
    for (j=0; j<dnzi; j++) {
      CHKERRQ(PetscHSetIUpdate(htd[pdj[j]*dof+offset],ht));
      CHKERRQ(PetscHSetIUpdate(hto[pdj[j]*dof+offset],oht));
    }
  }

  CHKERRQ(ISRestoreIndices(map,&mappingindices));

  CHKERRQ(PetscHSetIDestroy(&ht));
  CHKERRQ(PetscHSetIDestroy(&oht));

  for (i=0; i<pon; i++) {
    CHKERRQ(PetscHSetIGetSize(hta[i],&htsize));
    ptap->c_rmti[i+1] = ptap->c_rmti[i] + htsize;
    c_rmtc[i] = htsize;
  }

  CHKERRQ(PetscMalloc1(ptap->c_rmti[pon],&c_rmtj));

  for (i=0; i<pon; i++) {
    off = 0;
    CHKERRQ(PetscHSetIGetElems(hta[i],&off,c_rmtj+ptap->c_rmti[i]));
    CHKERRQ(PetscHSetIDestroy(&hta[i]));
  }
  CHKERRQ(PetscFree(hta));

  CHKERRQ(PetscMalloc1(pon,&iremote));
  for (i=0; i<pon; i++) {
    owner = 0; lidx = 0;
    offset = i%dof;
    ii     = i/dof;
    CHKERRQ(PetscLayoutFindOwnerIndex(P->cmap,p->garray[ii],&owner,&lidx));
    iremote[i].index = lidx*dof+offset;
    iremote[i].rank  = owner;
  }

  CHKERRQ(PetscSFCreate(comm,&sf));
  CHKERRQ(PetscSFSetGraph(sf,pn,pon,NULL,PETSC_OWN_POINTER,iremote,PETSC_OWN_POINTER));
  /* Reorder ranks properly so that the data handled by gather and scatter have the same order */
  CHKERRQ(PetscSFSetRankOrder(sf,PETSC_TRUE));
  CHKERRQ(PetscSFSetFromOptions(sf));
  CHKERRQ(PetscSFSetUp(sf));
  /* How many neighbors have contributions to my rows? */
  CHKERRQ(PetscSFComputeDegreeBegin(sf,&rootdegrees));
  CHKERRQ(PetscSFComputeDegreeEnd(sf,&rootdegrees));
  rootspacesize = 0;
  for (i = 0; i < pn; i++) {
    rootspacesize += rootdegrees[i];
  }
  CHKERRQ(PetscMalloc1(rootspacesize,&rootspace));
  CHKERRQ(PetscMalloc1(rootspacesize+1,&rootspaceoffsets));
  /* Get information from leaves
   * Number of columns other people contribute to my rows
   * */
  CHKERRQ(PetscSFGatherBegin(sf,MPIU_INT,c_rmtc,rootspace));
  CHKERRQ(PetscSFGatherEnd(sf,MPIU_INT,c_rmtc,rootspace));
  CHKERRQ(PetscFree(c_rmtc));
  CHKERRQ(PetscMalloc1(pn+1,&ptap->c_othi));
  /* The number of columns is received for each row */
  ptap->c_othi[0]     = 0;
  rootspacesize       = 0;
  rootspaceoffsets[0] = 0;
  for (i = 0; i < pn; i++) {
    rcvncols = 0;
    for (j = 0; j<rootdegrees[i]; j++) {
      rcvncols += rootspace[rootspacesize];
      rootspaceoffsets[rootspacesize+1] = rootspaceoffsets[rootspacesize] + rootspace[rootspacesize];
      rootspacesize++;
    }
    ptap->c_othi[i+1] = ptap->c_othi[i] + rcvncols;
  }
  CHKERRQ(PetscFree(rootspace));

  CHKERRQ(PetscMalloc1(pon,&c_rmtoffsets));
  CHKERRQ(PetscSFScatterBegin(sf,MPIU_INT,rootspaceoffsets,c_rmtoffsets));
  CHKERRQ(PetscSFScatterEnd(sf,MPIU_INT,rootspaceoffsets,c_rmtoffsets));
  CHKERRQ(PetscSFDestroy(&sf));
  CHKERRQ(PetscFree(rootspaceoffsets));

  CHKERRQ(PetscCalloc1(ptap->c_rmti[pon],&iremote));
  nleaves = 0;
  for (i = 0; i<pon; i++) {
    owner = 0;
    ii    = i/dof;
    CHKERRQ(PetscLayoutFindOwnerIndex(P->cmap,p->garray[ii],&owner,NULL));
    sendncols = ptap->c_rmti[i+1] - ptap->c_rmti[i];
    for (j=0; j<sendncols; j++) {
      iremote[nleaves].rank    = owner;
      iremote[nleaves++].index = c_rmtoffsets[i] + j;
    }
  }
  CHKERRQ(PetscFree(c_rmtoffsets));
  CHKERRQ(PetscCalloc1(ptap->c_othi[pn],&c_othj));

  CHKERRQ(PetscSFCreate(comm,&ptap->sf));
  CHKERRQ(PetscSFSetGraph(ptap->sf,ptap->c_othi[pn],nleaves,NULL,PETSC_OWN_POINTER,iremote,PETSC_OWN_POINTER));
  CHKERRQ(PetscSFSetFromOptions(ptap->sf));
  /* One to one map */
  CHKERRQ(PetscSFReduceBegin(ptap->sf,MPIU_INT,c_rmtj,c_othj,MPI_REPLACE));
  /* Get remote data */
  CHKERRQ(PetscSFReduceEnd(ptap->sf,MPIU_INT,c_rmtj,c_othj,MPI_REPLACE));
  CHKERRQ(PetscFree(c_rmtj));
  CHKERRQ(PetscMalloc2(pn,&dnz,pn,&onz));
  CHKERRQ(MatGetOwnershipRangeColumn(P,&pcstart,&pcend));
  pcstart *= dof;
  pcend   *= dof;
  for (i = 0; i < pn; i++) {
    nzi = ptap->c_othi[i+1] - ptap->c_othi[i];
    rdj = c_othj + ptap->c_othi[i];
    for (j = 0; j < nzi; j++) {
      col =  rdj[j];
      /* diag part */
      if (col>=pcstart && col<pcend) {
        CHKERRQ(PetscHSetIAdd(htd[i],col));
      } else { /* off diag */
        CHKERRQ(PetscHSetIAdd(hto[i],col));
      }
    }
    CHKERRQ(PetscHSetIGetSize(htd[i],&htsize));
    dnz[i] = htsize;
    CHKERRQ(PetscHSetIDestroy(&htd[i]));
    CHKERRQ(PetscHSetIGetSize(hto[i],&htsize));
    onz[i] = htsize;
    CHKERRQ(PetscHSetIDestroy(&hto[i]));
  }

  CHKERRQ(PetscFree2(htd,hto));
  CHKERRQ(PetscFree(c_othj));

  /* local sizes and preallocation */
  CHKERRQ(MatSetSizes(Cmpi,pn,pn,PETSC_DETERMINE,PETSC_DETERMINE));
  CHKERRQ(MatSetBlockSizes(Cmpi, dof>1? dof: P->cmap->bs,dof>1? dof: P->cmap->bs));
  CHKERRQ(MatMPIAIJSetPreallocation(Cmpi,0,dnz,0,onz));
  CHKERRQ(PetscFree2(dnz,onz));

  /* attach the supporting struct to Cmpi for reuse */
  Cmpi->product->data    = ptap;
  Cmpi->product->destroy = MatDestroy_MPIAIJ_PtAP;
  Cmpi->product->view    = MatView_MPIAIJ_PtAP;

  /* Cmpi is not ready for use - assembly will be done by MatPtAPNumeric() */
  Cmpi->assembled = PETSC_FALSE;
  /* pick an algorithm */
  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)A),((PetscObject)A)->prefix,"MatPtAP","Mat");CHKERRQ(ierr);
  alg = 0;
  CHKERRQ(PetscOptionsEList("-matptap_allatonce_via","PtAP allatonce numeric approach","MatPtAP",algTypes,nalg,algTypes[alg],&alg,&flg));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  switch (alg) {
    case 0:
      Cmpi->ops->ptapnumeric = MatPtAPNumeric_MPIAIJ_MPIAIJ_allatonce_merged;
      break;
    case 1:
      Cmpi->ops->ptapnumeric = MatPtAPNumeric_MPIAIJ_MPIAIJ_allatonce;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG," Unsupported allatonce numerical algorithm ");
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatPtAPSymbolic_MPIAIJ_MPIAIJ_allatonce_merged(Mat A,Mat P,PetscReal fill,Mat C)
{
  PetscFunctionBegin;
  CHKERRQ(MatPtAPSymbolic_MPIAIJ_MPIXAIJ_allatonce_merged(A,P,1,fill,C));
  PetscFunctionReturn(0);
}

PetscErrorCode MatPtAPSymbolic_MPIAIJ_MPIAIJ(Mat A,Mat P,PetscReal fill,Mat Cmpi)
{
  PetscErrorCode      ierr;
  Mat_APMPI           *ptap;
  Mat_MPIAIJ          *a=(Mat_MPIAIJ*)A->data,*p=(Mat_MPIAIJ*)P->data;
  MPI_Comm            comm;
  PetscMPIInt         size,rank;
  PetscFreeSpaceList  free_space=NULL,current_space=NULL;
  PetscInt            am=A->rmap->n,pm=P->rmap->n,pN=P->cmap->N,pn=P->cmap->n;
  PetscInt            *lnk,i,k,pnz,row,nsend;
  PetscBT             lnkbt;
  PetscMPIInt         tagi,tagj,*len_si,*len_s,*len_ri,nrecv;
  PETSC_UNUSED PetscMPIInt icompleted=0;
  PetscInt            **buf_rj,**buf_ri,**buf_ri_k;
  PetscInt            len,proc,*dnz,*onz,*owners,nzi,nspacedouble;
  PetscInt            nrows,*buf_s,*buf_si,*buf_si_i,**nextrow,**nextci;
  MPI_Request         *swaits,*rwaits;
  MPI_Status          *sstatus,rstatus;
  PetscLayout         rowmap;
  PetscInt            *owners_co,*coi,*coj;    /* i and j array of (p->B)^T*A*P - used in the communication */
  PetscMPIInt         *len_r,*id_r;    /* array of length of comm->size, store send/recv matrix values */
  PetscInt            *api,*apj,*Jptr,apnz,*prmap=p->garray,con,j,ap_rmax=0,Crmax,*aj,*ai,*pi;
  Mat_SeqAIJ          *p_loc,*p_oth=NULL,*ad=(Mat_SeqAIJ*)(a->A)->data,*ao=NULL,*c_loc,*c_oth;
  PetscScalar         *apv;
  PetscTable          ta;
  MatType             mtype;
  const char          *prefix;
#if defined(PETSC_USE_INFO)
  PetscReal           apfill;
#endif

  PetscFunctionBegin;
  MatCheckProduct(Cmpi,4);
  PetscCheckFalse(Cmpi->product->data,PetscObjectComm((PetscObject)Cmpi),PETSC_ERR_PLIB,"Product data not empty");
  CHKERRQ(PetscObjectGetComm((PetscObject)A,&comm));
  CHKERRMPI(MPI_Comm_size(comm,&size));
  CHKERRMPI(MPI_Comm_rank(comm,&rank));

  if (size > 1) ao = (Mat_SeqAIJ*)(a->B)->data;

  /* create symbolic parallel matrix Cmpi */
  CHKERRQ(MatGetType(A,&mtype));
  CHKERRQ(MatSetType(Cmpi,mtype));

  /* Do dense axpy in MatPtAPNumeric_MPIAIJ_MPIAIJ() */
  Cmpi->ops->ptapnumeric = MatPtAPNumeric_MPIAIJ_MPIAIJ;

  /* create struct Mat_APMPI and attached it to C later */
  CHKERRQ(PetscNew(&ptap));
  ptap->reuse = MAT_INITIAL_MATRIX;
  ptap->algType = 1;

  /* get P_oth by taking rows of P (= non-zero cols of local A) from other processors */
  CHKERRQ(MatGetBrowsOfAoCols_MPIAIJ(A,P,MAT_INITIAL_MATRIX,&ptap->startsj_s,&ptap->startsj_r,&ptap->bufa,&ptap->P_oth));
  /* get P_loc by taking all local rows of P */
  CHKERRQ(MatMPIAIJGetLocalMat(P,MAT_INITIAL_MATRIX,&ptap->P_loc));

  /* (0) compute Rd = Pd^T, Ro = Po^T  */
  /* --------------------------------- */
  CHKERRQ(MatTranspose(p->A,MAT_INITIAL_MATRIX,&ptap->Rd));
  CHKERRQ(MatTranspose(p->B,MAT_INITIAL_MATRIX,&ptap->Ro));

  /* (1) compute symbolic AP = A_loc*P = Ad*P_loc + Ao*P_oth (api,apj) */
  /* ----------------------------------------------------------------- */
  p_loc  = (Mat_SeqAIJ*)(ptap->P_loc)->data;
  if (ptap->P_oth) p_oth  = (Mat_SeqAIJ*)(ptap->P_oth)->data;

  /* create and initialize a linked list */
  CHKERRQ(PetscTableCreate(pn,pN,&ta)); /* for compute AP_loc and Cmpi */
  MatRowMergeMax_SeqAIJ(p_loc,ptap->P_loc->rmap->N,ta);
  MatRowMergeMax_SeqAIJ(p_oth,ptap->P_oth->rmap->N,ta);
  CHKERRQ(PetscTableGetCount(ta,&Crmax)); /* Crmax = nnz(sum of Prows) */
  /* printf("[%d] est %d, Crmax %d; pN %d\n",rank,5*(p_loc->rmax+p_oth->rmax + (PetscInt)(1.e-2*pN)),Crmax,pN); */

  CHKERRQ(PetscLLCondensedCreate(Crmax,pN,&lnk,&lnkbt));

  /* Initial FreeSpace size is fill*(nnz(A) + nnz(P)) */
  if (ao) {
    CHKERRQ(PetscFreeSpaceGet(PetscRealIntMultTruncate(fill,PetscIntSumTruncate(ad->i[am],PetscIntSumTruncate(ao->i[am],p_loc->i[pm]))),&free_space));
  } else {
    CHKERRQ(PetscFreeSpaceGet(PetscRealIntMultTruncate(fill,PetscIntSumTruncate(ad->i[am],p_loc->i[pm])),&free_space));
  }
  current_space = free_space;
  nspacedouble  = 0;

  CHKERRQ(PetscMalloc1(am+1,&api));
  api[0] = 0;
  for (i=0; i<am; i++) {
    /* diagonal portion: Ad[i,:]*P */
    ai = ad->i; pi = p_loc->i;
    nzi = ai[i+1] - ai[i];
    aj  = ad->j + ai[i];
    for (j=0; j<nzi; j++) {
      row  = aj[j];
      pnz  = pi[row+1] - pi[row];
      Jptr = p_loc->j + pi[row];
      /* add non-zero cols of P into the sorted linked list lnk */
      CHKERRQ(PetscLLCondensedAddSorted(pnz,Jptr,lnk,lnkbt));
    }
    /* off-diagonal portion: Ao[i,:]*P */
    if (ao) {
      ai = ao->i; pi = p_oth->i;
      nzi = ai[i+1] - ai[i];
      aj  = ao->j + ai[i];
      for (j=0; j<nzi; j++) {
        row  = aj[j];
        pnz  = pi[row+1] - pi[row];
        Jptr = p_oth->j + pi[row];
        CHKERRQ(PetscLLCondensedAddSorted(pnz,Jptr,lnk,lnkbt));
      }
    }
    apnz     = lnk[0];
    api[i+1] = api[i] + apnz;
    if (ap_rmax < apnz) ap_rmax = apnz;

    /* if free space is not available, double the total space in the list */
    if (current_space->local_remaining<apnz) {
      CHKERRQ(PetscFreeSpaceGet(PetscIntSumTruncate(apnz,current_space->total_array_size),&current_space));
      nspacedouble++;
    }

    /* Copy data into free space, then initialize lnk */
    CHKERRQ(PetscLLCondensedClean(pN,apnz,current_space->array,lnk,lnkbt));

    current_space->array           += apnz;
    current_space->local_used      += apnz;
    current_space->local_remaining -= apnz;
  }
  /* Allocate space for apj and apv, initialize apj, and */
  /* destroy list of free space and other temporary array(s) */
  CHKERRQ(PetscMalloc2(api[am],&apj,api[am],&apv));
  CHKERRQ(PetscFreeSpaceContiguous(&free_space,apj));
  CHKERRQ(PetscLLDestroy(lnk,lnkbt));

  /* Create AP_loc for reuse */
  CHKERRQ(MatCreateSeqAIJWithArrays(PETSC_COMM_SELF,am,pN,api,apj,apv,&ptap->AP_loc));
  CHKERRQ(MatSetType(ptap->AP_loc,((PetscObject)p->A)->type_name));
#if defined(PETSC_USE_INFO)
  if (ao) {
    apfill = (PetscReal)api[am]/(ad->i[am]+ao->i[am]+p_loc->i[pm]+1);
  } else {
    apfill = (PetscReal)api[am]/(ad->i[am]+p_loc->i[pm]+1);
  }
  ptap->AP_loc->info.mallocs           = nspacedouble;
  ptap->AP_loc->info.fill_ratio_given  = fill;
  ptap->AP_loc->info.fill_ratio_needed = apfill;

  if (api[am]) {
    CHKERRQ(PetscInfo(ptap->AP_loc,"Nonscalable algorithm, AP_loc reallocs %" PetscInt_FMT "; Fill ratio: given %g needed %g.\n",nspacedouble,(double)fill,(double)apfill));
    CHKERRQ(PetscInfo(ptap->AP_loc,"Use MatPtAP(A,B,MatReuse,%g,&C) for best AP_loc performance.;\n",(double)apfill));
  } else {
    CHKERRQ(PetscInfo(ptap->AP_loc,"Nonscalable algorithm, AP_loc is empty \n"));
  }
#endif

  /* (2-1) compute symbolic Co = Ro*AP_loc  */
  /* ------------------------------------ */
  CHKERRQ(MatGetOptionsPrefix(A,&prefix));
  CHKERRQ(MatSetOptionsPrefix(ptap->Ro,prefix));
  CHKERRQ(MatAppendOptionsPrefix(ptap->Ro,"inner_offdiag_"));
  CHKERRQ(MatProductCreate(ptap->Ro,ptap->AP_loc,NULL,&ptap->C_oth));
  CHKERRQ(MatGetOptionsPrefix(Cmpi,&prefix));
  CHKERRQ(MatSetOptionsPrefix(ptap->C_oth,prefix));
  CHKERRQ(MatAppendOptionsPrefix(ptap->C_oth,"inner_C_oth_"));
  CHKERRQ(MatProductSetType(ptap->C_oth,MATPRODUCT_AB));
  CHKERRQ(MatProductSetAlgorithm(ptap->C_oth,"default"));
  CHKERRQ(MatProductSetFill(ptap->C_oth,fill));
  CHKERRQ(MatProductSetFromOptions(ptap->C_oth));
  CHKERRQ(MatProductSymbolic(ptap->C_oth));

  /* (3) send coj of C_oth to other processors  */
  /* ------------------------------------------ */
  /* determine row ownership */
  CHKERRQ(PetscLayoutCreate(comm,&rowmap));
  rowmap->n  = pn;
  rowmap->bs = 1;
  CHKERRQ(PetscLayoutSetUp(rowmap));
  owners = rowmap->range;

  /* determine the number of messages to send, their lengths */
  CHKERRQ(PetscMalloc4(size,&len_s,size,&len_si,size,&sstatus,size+2,&owners_co));
  CHKERRQ(PetscArrayzero(len_s,size));
  CHKERRQ(PetscArrayzero(len_si,size));

  c_oth = (Mat_SeqAIJ*)ptap->C_oth->data;
  coi   = c_oth->i; coj = c_oth->j;
  con   = ptap->C_oth->rmap->n;
  proc  = 0;
  for (i=0; i<con; i++) {
    while (prmap[i] >= owners[proc+1]) proc++;
    len_si[proc]++;               /* num of rows in Co(=Pt*AP) to be sent to [proc] */
    len_s[proc] += coi[i+1] - coi[i]; /* num of nonzeros in Co to be sent to [proc] */
  }

  len          = 0; /* max length of buf_si[], see (4) */
  owners_co[0] = 0;
  nsend        = 0;
  for (proc=0; proc<size; proc++) {
    owners_co[proc+1] = owners_co[proc] + len_si[proc];
    if (len_s[proc]) {
      nsend++;
      len_si[proc] = 2*(len_si[proc] + 1); /* length of buf_si to be sent to [proc] */
      len         += len_si[proc];
    }
  }

  /* determine the number and length of messages to receive for coi and coj  */
  CHKERRQ(PetscGatherNumberOfMessages(comm,NULL,len_s,&nrecv));
  CHKERRQ(PetscGatherMessageLengths2(comm,nsend,nrecv,len_s,len_si,&id_r,&len_r,&len_ri));

  /* post the Irecv and Isend of coj */
  CHKERRQ(PetscCommGetNewTag(comm,&tagj));
  CHKERRQ(PetscPostIrecvInt(comm,tagj,nrecv,id_r,len_r,&buf_rj,&rwaits));
  CHKERRQ(PetscMalloc1(nsend+1,&swaits));
  for (proc=0, k=0; proc<size; proc++) {
    if (!len_s[proc]) continue;
    i    = owners_co[proc];
    CHKERRMPI(MPI_Isend(coj+coi[i],len_s[proc],MPIU_INT,proc,tagj,comm,swaits+k));
    k++;
  }

  /* (2-2) compute symbolic C_loc = Rd*AP_loc */
  /* ---------------------------------------- */
  CHKERRQ(MatSetOptionsPrefix(ptap->Rd,prefix));
  CHKERRQ(MatAppendOptionsPrefix(ptap->Rd,"inner_diag_"));
  CHKERRQ(MatProductCreate(ptap->Rd,ptap->AP_loc,NULL,&ptap->C_loc));
  CHKERRQ(MatGetOptionsPrefix(Cmpi,&prefix));
  CHKERRQ(MatSetOptionsPrefix(ptap->C_loc,prefix));
  CHKERRQ(MatAppendOptionsPrefix(ptap->C_loc,"inner_C_loc_"));
  CHKERRQ(MatProductSetType(ptap->C_loc,MATPRODUCT_AB));
  CHKERRQ(MatProductSetAlgorithm(ptap->C_loc,"default"));
  CHKERRQ(MatProductSetFill(ptap->C_loc,fill));
  CHKERRQ(MatProductSetFromOptions(ptap->C_loc));
  CHKERRQ(MatProductSymbolic(ptap->C_loc));

  c_loc = (Mat_SeqAIJ*)ptap->C_loc->data;

  /* receives coj are complete */
  for (i=0; i<nrecv; i++) {
    CHKERRMPI(MPI_Waitany(nrecv,rwaits,&icompleted,&rstatus));
  }
  CHKERRQ(PetscFree(rwaits));
  if (nsend) CHKERRMPI(MPI_Waitall(nsend,swaits,sstatus));

  /* add received column indices into ta to update Crmax */
  for (k=0; k<nrecv; k++) {/* k-th received message */
    Jptr = buf_rj[k];
    for (j=0; j<len_r[k]; j++) {
      CHKERRQ(PetscTableAdd(ta,*(Jptr+j)+1,1,INSERT_VALUES));
    }
  }
  CHKERRQ(PetscTableGetCount(ta,&Crmax));
  CHKERRQ(PetscTableDestroy(&ta));

  /* (4) send and recv coi */
  /*-----------------------*/
  CHKERRQ(PetscCommGetNewTag(comm,&tagi));
  CHKERRQ(PetscPostIrecvInt(comm,tagi,nrecv,id_r,len_ri,&buf_ri,&rwaits));
  CHKERRQ(PetscMalloc1(len+1,&buf_s));
  buf_si = buf_s;  /* points to the beginning of k-th msg to be sent */
  for (proc=0,k=0; proc<size; proc++) {
    if (!len_s[proc]) continue;
    /* form outgoing message for i-structure:
         buf_si[0]:                 nrows to be sent
               [1:nrows]:           row index (global)
               [nrows+1:2*nrows+1]: i-structure index
    */
    /*-------------------------------------------*/
    nrows       = len_si[proc]/2 - 1; /* num of rows in Co to be sent to [proc] */
    buf_si_i    = buf_si + nrows+1;
    buf_si[0]   = nrows;
    buf_si_i[0] = 0;
    nrows       = 0;
    for (i=owners_co[proc]; i<owners_co[proc+1]; i++) {
      nzi = coi[i+1] - coi[i];
      buf_si_i[nrows+1] = buf_si_i[nrows] + nzi;  /* i-structure */
      buf_si[nrows+1]   = prmap[i] -owners[proc]; /* local row index */
      nrows++;
    }
    CHKERRMPI(MPI_Isend(buf_si,len_si[proc],MPIU_INT,proc,tagi,comm,swaits+k));
    k++;
    buf_si += len_si[proc];
  }
  for (i=0; i<nrecv; i++) {
    CHKERRMPI(MPI_Waitany(nrecv,rwaits,&icompleted,&rstatus));
  }
  CHKERRQ(PetscFree(rwaits));
  if (nsend) CHKERRMPI(MPI_Waitall(nsend,swaits,sstatus));

  CHKERRQ(PetscFree4(len_s,len_si,sstatus,owners_co));
  CHKERRQ(PetscFree(len_ri));
  CHKERRQ(PetscFree(swaits));
  CHKERRQ(PetscFree(buf_s));

  /* (5) compute the local portion of Cmpi      */
  /* ------------------------------------------ */
  /* set initial free space to be Crmax, sufficient for holding nozeros in each row of Cmpi */
  CHKERRQ(PetscFreeSpaceGet(Crmax,&free_space));
  current_space = free_space;

  CHKERRQ(PetscMalloc3(nrecv,&buf_ri_k,nrecv,&nextrow,nrecv,&nextci));
  for (k=0; k<nrecv; k++) {
    buf_ri_k[k] = buf_ri[k]; /* beginning of k-th recved i-structure */
    nrows       = *buf_ri_k[k];
    nextrow[k]  = buf_ri_k[k] + 1;  /* next row number of k-th recved i-structure */
    nextci[k]   = buf_ri_k[k] + (nrows + 1); /* points to the next i-structure of k-th recved i-structure  */
  }

  ierr = MatPreallocateInitialize(comm,pn,pn,dnz,onz);CHKERRQ(ierr);
  CHKERRQ(PetscLLCondensedCreate(Crmax,pN,&lnk,&lnkbt));
  for (i=0; i<pn; i++) {
    /* add C_loc into Cmpi */
    nzi  = c_loc->i[i+1] - c_loc->i[i];
    Jptr = c_loc->j + c_loc->i[i];
    CHKERRQ(PetscLLCondensedAddSorted(nzi,Jptr,lnk,lnkbt));

    /* add received col data into lnk */
    for (k=0; k<nrecv; k++) { /* k-th received message */
      if (i == *nextrow[k]) { /* i-th row */
        nzi  = *(nextci[k]+1) - *nextci[k];
        Jptr = buf_rj[k] + *nextci[k];
        CHKERRQ(PetscLLCondensedAddSorted(nzi,Jptr,lnk,lnkbt));
        nextrow[k]++; nextci[k]++;
      }
    }
    nzi = lnk[0];

    /* copy data into free space, then initialize lnk */
    CHKERRQ(PetscLLCondensedClean(pN,nzi,current_space->array,lnk,lnkbt));
    CHKERRQ(MatPreallocateSet(i+owners[rank],nzi,current_space->array,dnz,onz));
  }
  CHKERRQ(PetscFree3(buf_ri_k,nextrow,nextci));
  CHKERRQ(PetscLLDestroy(lnk,lnkbt));
  CHKERRQ(PetscFreeSpaceDestroy(free_space));

  /* local sizes and preallocation */
  CHKERRQ(MatSetSizes(Cmpi,pn,pn,PETSC_DETERMINE,PETSC_DETERMINE));
  if (P->cmap->bs > 0) {
    CHKERRQ(PetscLayoutSetBlockSize(Cmpi->rmap,P->cmap->bs));
    CHKERRQ(PetscLayoutSetBlockSize(Cmpi->cmap,P->cmap->bs));
  }
  CHKERRQ(MatMPIAIJSetPreallocation(Cmpi,0,dnz,0,onz));
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);

  /* members in merge */
  CHKERRQ(PetscFree(id_r));
  CHKERRQ(PetscFree(len_r));
  CHKERRQ(PetscFree(buf_ri[0]));
  CHKERRQ(PetscFree(buf_ri));
  CHKERRQ(PetscFree(buf_rj[0]));
  CHKERRQ(PetscFree(buf_rj));
  CHKERRQ(PetscLayoutDestroy(&rowmap));

  CHKERRQ(PetscCalloc1(pN,&ptap->apa));

  /* attach the supporting struct to Cmpi for reuse */
  Cmpi->product->data    = ptap;
  Cmpi->product->destroy = MatDestroy_MPIAIJ_PtAP;
  Cmpi->product->view    = MatView_MPIAIJ_PtAP;

  /* Cmpi is not ready for use - assembly will be done by MatPtAPNumeric() */
  Cmpi->assembled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode MatPtAPNumeric_MPIAIJ_MPIAIJ(Mat A,Mat P,Mat C)
{
  Mat_MPIAIJ        *a=(Mat_MPIAIJ*)A->data,*p=(Mat_MPIAIJ*)P->data;
  Mat_SeqAIJ        *ad=(Mat_SeqAIJ*)(a->A)->data,*ao=(Mat_SeqAIJ*)(a->B)->data;
  Mat_SeqAIJ        *ap,*p_loc,*p_oth=NULL,*c_seq;
  Mat_APMPI         *ptap;
  Mat               AP_loc,C_loc,C_oth;
  PetscInt          i,rstart,rend,cm,ncols,row;
  PetscInt          *api,*apj,am = A->rmap->n,j,col,apnz;
  PetscScalar       *apa;
  const PetscInt    *cols;
  const PetscScalar *vals;

  PetscFunctionBegin;
  MatCheckProduct(C,3);
  ptap = (Mat_APMPI*)C->product->data;
  PetscCheckFalse(!ptap,PetscObjectComm((PetscObject)C),PETSC_ERR_ARG_WRONGSTATE,"PtAP cannot be computed. Missing data");
  PetscCheckFalse(!ptap->AP_loc,PetscObjectComm((PetscObject)C),PETSC_ERR_ARG_WRONGSTATE,"PtAP cannot be reused. Do not call MatProductClear()");

  CHKERRQ(MatZeroEntries(C));
  /* 1) get R = Pd^T,Ro = Po^T */
  if (ptap->reuse == MAT_REUSE_MATRIX) {
    CHKERRQ(MatTranspose(p->A,MAT_REUSE_MATRIX,&ptap->Rd));
    CHKERRQ(MatTranspose(p->B,MAT_REUSE_MATRIX,&ptap->Ro));
  }

  /* 2) get AP_loc */
  AP_loc = ptap->AP_loc;
  ap = (Mat_SeqAIJ*)AP_loc->data;

  /* 2-1) get P_oth = ptap->P_oth  and P_loc = ptap->P_loc */
  /*-----------------------------------------------------*/
  if (ptap->reuse == MAT_REUSE_MATRIX) {
    /* P_oth and P_loc are obtained in MatPtASymbolic() when reuse == MAT_INITIAL_MATRIX */
    CHKERRQ(MatGetBrowsOfAoCols_MPIAIJ(A,P,MAT_REUSE_MATRIX,&ptap->startsj_s,&ptap->startsj_r,&ptap->bufa,&ptap->P_oth));
    CHKERRQ(MatMPIAIJGetLocalMat(P,MAT_REUSE_MATRIX,&ptap->P_loc));
  }

  /* 2-2) compute numeric A_loc*P - dominating part */
  /* ---------------------------------------------- */
  /* get data from symbolic products */
  p_loc = (Mat_SeqAIJ*)(ptap->P_loc)->data;
  if (ptap->P_oth) {
    p_oth = (Mat_SeqAIJ*)(ptap->P_oth)->data;
  }
  apa   = ptap->apa;
  api   = ap->i;
  apj   = ap->j;
  for (i=0; i<am; i++) {
    /* AP[i,:] = A[i,:]*P = Ad*P_loc Ao*P_oth */
    AProw_nonscalable(i,ad,ao,p_loc,p_oth,apa);
    apnz = api[i+1] - api[i];
    for (j=0; j<apnz; j++) {
      col = apj[j+api[i]];
      ap->a[j+ap->i[i]] = apa[col];
      apa[col] = 0.0;
    }
  }
  /* We have modified the contents of local matrix AP_loc and must increase its ObjectState, since we are not doing AssemblyBegin/End on it. */
  CHKERRQ(PetscObjectStateIncrease((PetscObject)AP_loc));

  /* 3) C_loc = Rd*AP_loc, C_oth = Ro*AP_loc */
  CHKERRQ(MatProductNumeric(ptap->C_loc));
  CHKERRQ(MatProductNumeric(ptap->C_oth));
  C_loc = ptap->C_loc;
  C_oth = ptap->C_oth;

  /* add C_loc and Co to to C */
  CHKERRQ(MatGetOwnershipRange(C,&rstart,&rend));

  /* C_loc -> C */
  cm    = C_loc->rmap->N;
  c_seq = (Mat_SeqAIJ*)C_loc->data;
  cols = c_seq->j;
  vals = c_seq->a;

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
    for (i=0; i<cm; i++) {
      ncols = c_seq->i[i+1] - c_seq->i[i];
      row = rstart + i;
      CHKERRQ(MatSetValues_MPIAIJ(C,1,&row,ncols,cols,vals,ADD_VALUES));
      cols += ncols; vals += ncols;
    }
  } else {
    CHKERRQ(MatSetValues_MPIAIJ_CopyFromCSRFormat(C,c_seq->j,c_seq->i,c_seq->a));
  }

  /* Co -> C, off-processor part */
  cm = C_oth->rmap->N;
  c_seq = (Mat_SeqAIJ*)C_oth->data;
  cols = c_seq->j;
  vals = c_seq->a;
  for (i=0; i<cm; i++) {
    ncols = c_seq->i[i+1] - c_seq->i[i];
    row = p->garray[i];
    CHKERRQ(MatSetValues(C,1,&row,ncols,cols,vals,ADD_VALUES));
    cols += ncols; vals += ncols;
  }

  CHKERRQ(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));

  ptap->reuse = MAT_REUSE_MATRIX;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatProductSymbolic_PtAP_MPIAIJ_MPIAIJ(Mat C)
{
  Mat_Product         *product = C->product;
  Mat                 A=product->A,P=product->B;
  MatProductAlgorithm alg=product->alg;
  PetscReal           fill=product->fill;
  PetscBool           flg;

  PetscFunctionBegin;
  /* scalable: do R=P^T locally, then C=R*A*P */
  CHKERRQ(PetscStrcmp(alg,"scalable",&flg));
  if (flg) {
    CHKERRQ(MatPtAPSymbolic_MPIAIJ_MPIAIJ_scalable(A,P,product->fill,C));
    C->ops->productnumeric = MatProductNumeric_PtAP;
    goto next;
  }

  /* nonscalable: do R=P^T locally, then C=R*A*P */
  CHKERRQ(PetscStrcmp(alg,"nonscalable",&flg));
  if (flg) {
    CHKERRQ(MatPtAPSymbolic_MPIAIJ_MPIAIJ(A,P,fill,C));
    goto next;
  }

  /* allatonce */
  CHKERRQ(PetscStrcmp(alg,"allatonce",&flg));
  if (flg) {
    CHKERRQ(MatPtAPSymbolic_MPIAIJ_MPIAIJ_allatonce(A,P,fill,C));
    goto next;
  }

  /* allatonce_merged */
  CHKERRQ(PetscStrcmp(alg,"allatonce_merged",&flg));
  if (flg) {
    CHKERRQ(MatPtAPSymbolic_MPIAIJ_MPIAIJ_allatonce_merged(A,P,fill,C));
    goto next;
  }

  /* backend general code */
  CHKERRQ(PetscStrcmp(alg,"backend",&flg));
  if (flg) {
    CHKERRQ(MatProductSymbolic_MPIAIJBACKEND(C));
    PetscFunctionReturn(0);
  }

  /* hypre */
#if defined(PETSC_HAVE_HYPRE)
  CHKERRQ(PetscStrcmp(alg,"hypre",&flg));
  if (flg) {
    CHKERRQ(MatPtAPSymbolic_AIJ_AIJ_wHYPRE(A,P,fill,C));
    C->ops->productnumeric = MatProductNumeric_PtAP;
    PetscFunctionReturn(0);
  }
#endif
  SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_SUP,"Mat Product Algorithm is not supported");

next:
  C->ops->productnumeric = MatProductNumeric_PtAP;
  PetscFunctionReturn(0);
}
