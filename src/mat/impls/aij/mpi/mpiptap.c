
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
  PetscErrorCode    ierr;
  PetscBool         iascii;
  PetscViewerFormat format;
  Mat_APMPI         *ptap;

  PetscFunctionBegin;
  MatCheckProduct(A,1);
  ptap = (Mat_APMPI*)A->product->data;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
      if (ptap->algType == 0) {
        ierr = PetscViewerASCIIPrintf(viewer,"using scalable MatPtAP() implementation\n");CHKERRQ(ierr);
      } else if (ptap->algType == 1) {
        ierr = PetscViewerASCIIPrintf(viewer,"using nonscalable MatPtAP() implementation\n");CHKERRQ(ierr);
      } else if (ptap->algType == 2) {
        ierr = PetscViewerASCIIPrintf(viewer,"using allatonce MatPtAP() implementation\n");CHKERRQ(ierr);
      } else if (ptap->algType == 3) {
        ierr = PetscViewerASCIIPrintf(viewer,"using merged allatonce MatPtAP() implementation\n");CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_MPIAIJ_PtAP(void *data)
{
  PetscErrorCode      ierr;
  Mat_APMPI           *ptap = (Mat_APMPI*)data;
  Mat_Merge_SeqsToMPI *merge;

  PetscFunctionBegin;
  ierr = PetscFree2(ptap->startsj_s,ptap->startsj_r);CHKERRQ(ierr);
  ierr = PetscFree(ptap->bufa);CHKERRQ(ierr);
  ierr = MatDestroy(&ptap->P_loc);CHKERRQ(ierr);
  ierr = MatDestroy(&ptap->P_oth);CHKERRQ(ierr);
  ierr = MatDestroy(&ptap->A_loc);CHKERRQ(ierr); /* used by MatTransposeMatMult() */
  ierr = MatDestroy(&ptap->Rd);CHKERRQ(ierr);
  ierr = MatDestroy(&ptap->Ro);CHKERRQ(ierr);
  if (ptap->AP_loc) { /* used by alg_rap */
    Mat_SeqAIJ *ap = (Mat_SeqAIJ*)(ptap->AP_loc)->data;
    ierr = PetscFree(ap->i);CHKERRQ(ierr);
    ierr = PetscFree2(ap->j,ap->a);CHKERRQ(ierr);
    ierr = MatDestroy(&ptap->AP_loc);CHKERRQ(ierr);
  } else { /* used by alg_ptap */
    ierr = PetscFree(ptap->api);CHKERRQ(ierr);
    ierr = PetscFree(ptap->apj);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&ptap->C_loc);CHKERRQ(ierr);
  ierr = MatDestroy(&ptap->C_oth);CHKERRQ(ierr);
  if (ptap->apa) {ierr = PetscFree(ptap->apa);CHKERRQ(ierr);}

  ierr = MatDestroy(&ptap->Pt);CHKERRQ(ierr);

  merge = ptap->merge;
  if (merge) { /* used by alg_ptap */
    ierr = PetscFree(merge->id_r);CHKERRQ(ierr);
    ierr = PetscFree(merge->len_s);CHKERRQ(ierr);
    ierr = PetscFree(merge->len_r);CHKERRQ(ierr);
    ierr = PetscFree(merge->bi);CHKERRQ(ierr);
    ierr = PetscFree(merge->bj);CHKERRQ(ierr);
    ierr = PetscFree(merge->buf_ri[0]);CHKERRQ(ierr);
    ierr = PetscFree(merge->buf_ri);CHKERRQ(ierr);
    ierr = PetscFree(merge->buf_rj[0]);CHKERRQ(ierr);
    ierr = PetscFree(merge->buf_rj);CHKERRQ(ierr);
    ierr = PetscFree(merge->coi);CHKERRQ(ierr);
    ierr = PetscFree(merge->coj);CHKERRQ(ierr);
    ierr = PetscFree(merge->owners_co);CHKERRQ(ierr);
    ierr = PetscLayoutDestroy(&merge->rowmap);CHKERRQ(ierr);
    ierr = PetscFree(ptap->merge);CHKERRQ(ierr);
  }
  ierr = ISLocalToGlobalMappingDestroy(&ptap->ltog);CHKERRQ(ierr);

  ierr = PetscSFDestroy(&ptap->sf);CHKERRQ(ierr);
  ierr = PetscFree(ptap->c_othi);CHKERRQ(ierr);
  ierr = PetscFree(ptap->c_rmti);CHKERRQ(ierr);
  ierr = PetscFree(ptap);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatPtAPNumeric_MPIAIJ_MPIAIJ_scalable(Mat A,Mat P,Mat C)
{
  PetscErrorCode    ierr;
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
  if (!ptap) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_ARG_WRONGSTATE,"PtAP cannot be computed. Missing data");
  if (!ptap->AP_loc) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_ARG_WRONGSTATE,"PtAP cannot be reused. Do not call MatProductClear()");

  ierr = MatZeroEntries(C);CHKERRQ(ierr);

  /* 1) get R = Pd^T,Ro = Po^T */
  if (ptap->reuse == MAT_REUSE_MATRIX) {
    ierr = MatTranspose(p->A,MAT_REUSE_MATRIX,&ptap->Rd);CHKERRQ(ierr);
    ierr = MatTranspose(p->B,MAT_REUSE_MATRIX,&ptap->Ro);CHKERRQ(ierr);
  }

  /* 2) get AP_loc */
  AP_loc = ptap->AP_loc;
  ap = (Mat_SeqAIJ*)AP_loc->data;

  /* 2-1) get P_oth = ptap->P_oth  and P_loc = ptap->P_loc */
  /*-----------------------------------------------------*/
  if (ptap->reuse == MAT_REUSE_MATRIX) {
    /* P_oth and P_loc are obtained in MatPtASymbolic() when reuse == MAT_INITIAL_MATRIX */
    ierr = MatGetBrowsOfAoCols_MPIAIJ(A,P,MAT_REUSE_MATRIX,&ptap->startsj_s,&ptap->startsj_r,&ptap->bufa,&ptap->P_oth);CHKERRQ(ierr);
    ierr = MatMPIAIJGetLocalMat(P,MAT_REUSE_MATRIX,&ptap->P_loc);CHKERRQ(ierr);
  }

  /* 2-2) compute numeric A_loc*P - dominating part */
  /* ---------------------------------------------- */
  /* get data from symbolic products */
  p_loc = (Mat_SeqAIJ*)(ptap->P_loc)->data;
  if (ptap->P_oth) p_oth = (Mat_SeqAIJ*)(ptap->P_oth)->data;

  api   = ap->i;
  apj   = ap->j;
  ierr = ISLocalToGlobalMappingApply(ptap->ltog,api[AP_loc->rmap->n],apj,apj);CHKERRQ(ierr);
  for (i=0; i<am; i++) {
    /* AP[i,:] = A[i,:]*P = Ad*P_loc Ao*P_oth */
    apnz = api[i+1] - api[i];
    apa = ap->a + api[i];
    ierr = PetscArrayzero(apa,apnz);CHKERRQ(ierr);
    AProw_scalable(i,ad,ao,p_loc,p_oth,api,apj,apa);
  }
  ierr = ISGlobalToLocalMappingApply(ptap->ltog,IS_GTOLM_DROP,api[AP_loc->rmap->n],apj,&nout,apj);CHKERRQ(ierr);
  if (api[AP_loc->rmap->n] != nout) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Incorrect mapping %" PetscInt_FMT " != %" PetscInt_FMT,api[AP_loc->rmap->n],nout);

  /* 3) C_loc = Rd*AP_loc, C_oth = Ro*AP_loc */
  /* Always use scalable version since we are in the MPI scalable version */
  ierr = MatMatMultNumeric_SeqAIJ_SeqAIJ_Scalable(ptap->Rd,AP_loc,ptap->C_loc);CHKERRQ(ierr);
  ierr = MatMatMultNumeric_SeqAIJ_SeqAIJ_Scalable(ptap->Ro,AP_loc,ptap->C_oth);CHKERRQ(ierr);

  C_loc = ptap->C_loc;
  C_oth = ptap->C_oth;

  /* add C_loc and Co to to C */
  ierr = MatGetOwnershipRange(C,&rstart,&rend);CHKERRQ(ierr);

  /* C_loc -> C */
  cm    = C_loc->rmap->N;
  c_seq = (Mat_SeqAIJ*)C_loc->data;
  cols = c_seq->j;
  vals = c_seq->a;
  ierr = ISLocalToGlobalMappingApply(ptap->ltog,c_seq->i[C_loc->rmap->n],c_seq->j,c_seq->j);CHKERRQ(ierr);

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
      ierr = MatSetValues_MPIAIJ(C,1,&row,ncols,cols,vals,ADD_VALUES);CHKERRQ(ierr);
      cols += ncols; vals += ncols;
    }
  } else {
    ierr = MatSetValues_MPIAIJ_CopyFromCSRFormat(C,c_seq->j,c_seq->i,c_seq->a);CHKERRQ(ierr);
  }
  ierr = ISGlobalToLocalMappingApply(ptap->ltog,IS_GTOLM_DROP,c_seq->i[C_loc->rmap->n],c_seq->j,&nout,c_seq->j);CHKERRQ(ierr);
  if (c_seq->i[C_loc->rmap->n] != nout) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Incorrect mapping %" PetscInt_FMT " != %" PetscInt_FMT,c_seq->i[C_loc->rmap->n],nout);

  /* Co -> C, off-processor part */
  cm = C_oth->rmap->N;
  c_seq = (Mat_SeqAIJ*)C_oth->data;
  cols = c_seq->j;
  vals = c_seq->a;
  ierr = ISLocalToGlobalMappingApply(ptap->ltog,c_seq->i[C_oth->rmap->n],c_seq->j,c_seq->j);CHKERRQ(ierr);
  for (i=0; i<cm; i++) {
    ncols = c_seq->i[i+1] - c_seq->i[i];
    row = p->garray[i];
    ierr = MatSetValues(C,1,&row,ncols,cols,vals,ADD_VALUES);CHKERRQ(ierr);
    cols += ncols; vals += ncols;
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ptap->reuse = MAT_REUSE_MATRIX;

  ierr = ISGlobalToLocalMappingApply(ptap->ltog,IS_GTOLM_DROP,c_seq->i[C_oth->rmap->n],c_seq->j,&nout,c_seq->j);CHKERRQ(ierr);
  if (c_seq->i[C_oth->rmap->n] != nout) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Incorrect mapping %" PetscInt_FMT " != %" PetscInt_FMT,c_seq->i[C_loc->rmap->n],nout);
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
  if (Cmpi->product->data) SETERRQ(PetscObjectComm((PetscObject)Cmpi),PETSC_ERR_PLIB,"Product data not empty");
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);

  if (size > 1) ao = (Mat_SeqAIJ*)(a->B)->data;

  /* create symbolic parallel matrix Cmpi */
  ierr = MatGetType(A,&mtype);CHKERRQ(ierr);
  ierr = MatSetType(Cmpi,mtype);CHKERRQ(ierr);

  /* create struct Mat_APMPI and attached it to C later */
  ierr        = PetscNew(&ptap);CHKERRQ(ierr);
  ptap->reuse = MAT_INITIAL_MATRIX;
  ptap->algType = 0;

  /* get P_oth by taking rows of P (= non-zero cols of local A) from other processors */
  ierr = MatGetBrowsOfAoCols_MPIAIJ(A,P,MAT_INITIAL_MATRIX,&ptap->startsj_s,&ptap->startsj_r,&ptap->bufa,&P_oth);CHKERRQ(ierr);
  /* get P_loc by taking all local rows of P */
  ierr = MatMPIAIJGetLocalMat(P,MAT_INITIAL_MATRIX,&P_loc);CHKERRQ(ierr);

  ptap->P_loc = P_loc;
  ptap->P_oth = P_oth;

  /* (0) compute Rd = Pd^T, Ro = Po^T  */
  /* --------------------------------- */
  ierr = MatTranspose(p->A,MAT_INITIAL_MATRIX,&ptap->Rd);CHKERRQ(ierr);
  ierr = MatTranspose(p->B,MAT_INITIAL_MATRIX,&ptap->Ro);CHKERRQ(ierr);

  /* (1) compute symbolic AP = A_loc*P = Ad*P_loc + Ao*P_oth (api,apj) */
  /* ----------------------------------------------------------------- */
  p_loc  = (Mat_SeqAIJ*)P_loc->data;
  if (P_oth) p_oth = (Mat_SeqAIJ*)P_oth->data;

  /* create and initialize a linked list */
  ierr = PetscTableCreate(pn,pN,&ta);CHKERRQ(ierr); /* for compute AP_loc and Cmpi */
  MatRowMergeMax_SeqAIJ(p_loc,P_loc->rmap->N,ta);
  MatRowMergeMax_SeqAIJ(p_oth,P_oth->rmap->N,ta);
  ierr = PetscTableGetCount(ta,&Crmax);CHKERRQ(ierr); /* Crmax = nnz(sum of Prows) */

  ierr = PetscLLCondensedCreate_Scalable(Crmax,&lnk);CHKERRQ(ierr);

  /* Initial FreeSpace size is fill*(nnz(A) + nnz(P)) */
  if (ao) {
    ierr = PetscFreeSpaceGet(PetscRealIntMultTruncate(fill,PetscIntSumTruncate(ad->i[am],PetscIntSumTruncate(ao->i[am],p_loc->i[pm]))),&free_space);CHKERRQ(ierr);
  } else {
    ierr = PetscFreeSpaceGet(PetscRealIntMultTruncate(fill,PetscIntSumTruncate(ad->i[am],p_loc->i[pm])),&free_space);CHKERRQ(ierr);
  }
  current_space = free_space;
  nspacedouble  = 0;

  ierr   = PetscMalloc1(am+1,&api);CHKERRQ(ierr);
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
      ierr = PetscLLCondensedAddSorted_Scalable(pnz,Jptr,lnk);CHKERRQ(ierr);
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
        ierr = PetscLLCondensedAddSorted_Scalable(pnz,Jptr,lnk);CHKERRQ(ierr);
      }
    }
    apnz     = lnk[0];
    api[i+1] = api[i] + apnz;

    /* if free space is not available, double the total space in the list */
    if (current_space->local_remaining<apnz) {
      ierr = PetscFreeSpaceGet(PetscIntSumTruncate(apnz,current_space->total_array_size),&current_space);CHKERRQ(ierr);
      nspacedouble++;
    }

    /* Copy data into free space, then initialize lnk */
    ierr = PetscLLCondensedClean_Scalable(apnz,current_space->array,lnk);CHKERRQ(ierr);

    current_space->array           += apnz;
    current_space->local_used      += apnz;
    current_space->local_remaining -= apnz;
  }
  /* Allocate space for apj and apv, initialize apj, and */
  /* destroy list of free space and other temporary array(s) */
  ierr = PetscCalloc2(api[am],&apj,api[am],&apv);CHKERRQ(ierr);
  ierr = PetscFreeSpaceContiguous(&free_space,apj);CHKERRQ(ierr);
  ierr = PetscLLCondensedDestroy_Scalable(lnk);CHKERRQ(ierr);

  /* Create AP_loc for reuse */
  ierr = MatCreateSeqAIJWithArrays(PETSC_COMM_SELF,am,pN,api,apj,apv,&ptap->AP_loc);CHKERRQ(ierr);
  ierr = MatSeqAIJCompactOutExtraColumns_SeqAIJ(ptap->AP_loc, &ptap->ltog);CHKERRQ(ierr);

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
    ierr = PetscInfo3(ptap->AP_loc,"Scalable algorithm, AP_loc reallocs %" PetscInt_FMT "; Fill ratio: given %g needed %g.\n",nspacedouble,(double)fill,(double)apfill);CHKERRQ(ierr);
    ierr = PetscInfo1(ptap->AP_loc,"Use MatPtAP(A,B,MatReuse,%g,&C) for best AP_loc performance.;\n",(double)apfill);CHKERRQ(ierr);
  } else {
    ierr = PetscInfo(ptap->AP_loc,"Scalable algorithm, AP_loc is empty \n");CHKERRQ(ierr);
  }
#endif

  /* (2-1) compute symbolic Co = Ro*AP_loc  */
  /* -------------------------------------- */
  ierr = MatProductCreate(ptap->Ro,ptap->AP_loc,NULL,&ptap->C_oth);CHKERRQ(ierr);
  ierr = MatGetOptionsPrefix(A,&prefix);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(ptap->C_oth,prefix);CHKERRQ(ierr);
  ierr = MatAppendOptionsPrefix(ptap->C_oth,"inner_offdiag_");CHKERRQ(ierr);

  ierr = MatProductSetType(ptap->C_oth,MATPRODUCT_AB);CHKERRQ(ierr);
  ierr = MatProductSetAlgorithm(ptap->C_oth,"sorted");CHKERRQ(ierr);
  ierr = MatProductSetFill(ptap->C_oth,fill);CHKERRQ(ierr);
  ierr = MatProductSetFromOptions(ptap->C_oth);CHKERRQ(ierr);
  ierr = MatProductSymbolic(ptap->C_oth);CHKERRQ(ierr);

  /* (3) send coj of C_oth to other processors  */
  /* ------------------------------------------ */
  /* determine row ownership */
  ierr = PetscLayoutCreate(comm,&rowmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetLocalSize(rowmap, pn);CHKERRQ(ierr);
  ierr = PetscLayoutSetBlockSize(rowmap, 1);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(rowmap);CHKERRQ(ierr);
  ierr = PetscLayoutGetRanges(rowmap,&owners);CHKERRQ(ierr);

  /* determine the number of messages to send, their lengths */
  ierr = PetscMalloc4(size,&len_s,size,&len_si,size,&sstatus,size+2,&owners_co);CHKERRQ(ierr);
  ierr = PetscArrayzero(len_s,size);CHKERRQ(ierr);
  ierr = PetscArrayzero(len_si,size);CHKERRQ(ierr);

  c_oth = (Mat_SeqAIJ*)ptap->C_oth->data;
  coi   = c_oth->i; coj = c_oth->j;
  con   = ptap->C_oth->rmap->n;
  proc  = 0;
  ierr = ISLocalToGlobalMappingApply(ptap->ltog,coi[con],coj,coj);CHKERRQ(ierr);
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
  ierr = PetscGatherNumberOfMessages(comm,NULL,len_s,&nrecv);CHKERRQ(ierr);
  ierr = PetscGatherMessageLengths2(comm,nsend,nrecv,len_s,len_si,&id_r,&len_r,&len_ri);CHKERRQ(ierr);

  /* post the Irecv and Isend of coj */
  ierr = PetscCommGetNewTag(comm,&tagj);CHKERRQ(ierr);
  ierr = PetscPostIrecvInt(comm,tagj,nrecv,id_r,len_r,&buf_rj,&rwaits);CHKERRQ(ierr);
  ierr = PetscMalloc1(nsend+1,&swaits);CHKERRQ(ierr);
  for (proc=0, k=0; proc<size; proc++) {
    if (!len_s[proc]) continue;
    i    = owners_co[proc];
    ierr = MPI_Isend(coj+coi[i],len_s[proc],MPIU_INT,proc,tagj,comm,swaits+k);CHKERRMPI(ierr);
    k++;
  }

  /* (2-2) compute symbolic C_loc = Rd*AP_loc */
  /* ---------------------------------------- */
  ierr = MatProductCreate(ptap->Rd,ptap->AP_loc,NULL,&ptap->C_loc);CHKERRQ(ierr);
  ierr = MatProductSetType(ptap->C_loc,MATPRODUCT_AB);CHKERRQ(ierr);
  ierr = MatProductSetAlgorithm(ptap->C_loc,"default");CHKERRQ(ierr);
  ierr = MatProductSetFill(ptap->C_loc,fill);CHKERRQ(ierr);

  ierr = MatSetOptionsPrefix(ptap->C_loc,prefix);CHKERRQ(ierr);
  ierr = MatAppendOptionsPrefix(ptap->C_loc,"inner_diag_");CHKERRQ(ierr);

  ierr = MatProductSetFromOptions(ptap->C_loc);CHKERRQ(ierr);
  ierr = MatProductSymbolic(ptap->C_loc);CHKERRQ(ierr);

  c_loc = (Mat_SeqAIJ*)ptap->C_loc->data;
  ierr = ISLocalToGlobalMappingApply(ptap->ltog,c_loc->i[ptap->C_loc->rmap->n],c_loc->j,c_loc->j);CHKERRQ(ierr);

  /* receives coj are complete */
  for (i=0; i<nrecv; i++) {
    ierr = MPI_Waitany(nrecv,rwaits,&icompleted,&rstatus);CHKERRMPI(ierr);
  }
  ierr = PetscFree(rwaits);CHKERRQ(ierr);
  if (nsend) {ierr = MPI_Waitall(nsend,swaits,sstatus);CHKERRMPI(ierr);}

  /* add received column indices into ta to update Crmax */
  for (k=0; k<nrecv; k++) {/* k-th received message */
    Jptr = buf_rj[k];
    for (j=0; j<len_r[k]; j++) {
      ierr = PetscTableAdd(ta,*(Jptr+j)+1,1,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = PetscTableGetCount(ta,&Crmax);CHKERRQ(ierr);
  ierr = PetscTableDestroy(&ta);CHKERRQ(ierr);

  /* (4) send and recv coi */
  /*-----------------------*/
  ierr   = PetscCommGetNewTag(comm,&tagi);CHKERRQ(ierr);
  ierr   = PetscPostIrecvInt(comm,tagi,nrecv,id_r,len_ri,&buf_ri,&rwaits);CHKERRQ(ierr);
  ierr   = PetscMalloc1(len+1,&buf_s);CHKERRQ(ierr);
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
    ierr = MPI_Isend(buf_si,len_si[proc],MPIU_INT,proc,tagi,comm,swaits+k);CHKERRMPI(ierr);
    k++;
    buf_si += len_si[proc];
  }
  for (i=0; i<nrecv; i++) {
    ierr = MPI_Waitany(nrecv,rwaits,&icompleted,&rstatus);CHKERRMPI(ierr);
  }
  ierr = PetscFree(rwaits);CHKERRQ(ierr);
  if (nsend) {ierr = MPI_Waitall(nsend,swaits,sstatus);CHKERRMPI(ierr);}

  ierr = PetscFree4(len_s,len_si,sstatus,owners_co);CHKERRQ(ierr);
  ierr = PetscFree(len_ri);CHKERRQ(ierr);
  ierr = PetscFree(swaits);CHKERRQ(ierr);
  ierr = PetscFree(buf_s);CHKERRQ(ierr);

  /* (5) compute the local portion of Cmpi      */
  /* ------------------------------------------ */
  /* set initial free space to be Crmax, sufficient for holding nozeros in each row of Cmpi */
  ierr          = PetscFreeSpaceGet(Crmax,&free_space);CHKERRQ(ierr);
  current_space = free_space;

  ierr = PetscMalloc3(nrecv,&buf_ri_k,nrecv,&nextrow,nrecv,&nextci);CHKERRQ(ierr);
  for (k=0; k<nrecv; k++) {
    buf_ri_k[k] = buf_ri[k]; /* beginning of k-th recved i-structure */
    nrows       = *buf_ri_k[k];
    nextrow[k]  = buf_ri_k[k] + 1;  /* next row number of k-th recved i-structure */
    nextci[k]   = buf_ri_k[k] + (nrows + 1); /* points to the next i-structure of k-th recved i-structure  */
  }

  ierr = MatPreallocateInitialize(comm,pn,pn,dnz,onz);CHKERRQ(ierr);
  ierr = PetscLLCondensedCreate_Scalable(Crmax,&lnk);CHKERRQ(ierr);
  for (i=0; i<pn; i++) {
    /* add C_loc into Cmpi */
    nzi  = c_loc->i[i+1] - c_loc->i[i];
    Jptr = c_loc->j + c_loc->i[i];
    ierr = PetscLLCondensedAddSorted_Scalable(nzi,Jptr,lnk);CHKERRQ(ierr);

    /* add received col data into lnk */
    for (k=0; k<nrecv; k++) { /* k-th received message */
      if (i == *nextrow[k]) { /* i-th row */
        nzi  = *(nextci[k]+1) - *nextci[k];
        Jptr = buf_rj[k] + *nextci[k];
        ierr = PetscLLCondensedAddSorted_Scalable(nzi,Jptr,lnk);CHKERRQ(ierr);
        nextrow[k]++; nextci[k]++;
      }
    }
    nzi = lnk[0];

    /* copy data into free space, then initialize lnk */
    ierr = PetscLLCondensedClean_Scalable(nzi,current_space->array,lnk);CHKERRQ(ierr);
    ierr = MatPreallocateSet(i+owners[rank],nzi,current_space->array,dnz,onz);CHKERRQ(ierr);
  }
  ierr = PetscFree3(buf_ri_k,nextrow,nextci);CHKERRQ(ierr);
  ierr = PetscLLCondensedDestroy_Scalable(lnk);CHKERRQ(ierr);
  ierr = PetscFreeSpaceDestroy(free_space);CHKERRQ(ierr);

  /* local sizes and preallocation */
  ierr = MatSetSizes(Cmpi,pn,pn,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  if (P->cmap->bs > 0) {
    ierr = PetscLayoutSetBlockSize(Cmpi->rmap,P->cmap->bs);CHKERRQ(ierr);
    ierr = PetscLayoutSetBlockSize(Cmpi->cmap,P->cmap->bs);CHKERRQ(ierr);
  }
  ierr = MatMPIAIJSetPreallocation(Cmpi,0,dnz,0,onz);CHKERRQ(ierr);
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);

  /* members in merge */
  ierr = PetscFree(id_r);CHKERRQ(ierr);
  ierr = PetscFree(len_r);CHKERRQ(ierr);
  ierr = PetscFree(buf_ri[0]);CHKERRQ(ierr);
  ierr = PetscFree(buf_ri);CHKERRQ(ierr);
  ierr = PetscFree(buf_rj[0]);CHKERRQ(ierr);
  ierr = PetscFree(buf_rj);CHKERRQ(ierr);
  ierr = PetscLayoutDestroy(&rowmap);CHKERRQ(ierr);

  nout = 0;
  ierr = ISGlobalToLocalMappingApply(ptap->ltog,IS_GTOLM_DROP,c_oth->i[ptap->C_oth->rmap->n],c_oth->j,&nout,c_oth->j);CHKERRQ(ierr);
  if (c_oth->i[ptap->C_oth->rmap->n] != nout) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Incorrect mapping %" PetscInt_FMT " != %" PetscInt_FMT,c_oth->i[ptap->C_oth->rmap->n],nout);
  ierr = ISGlobalToLocalMappingApply(ptap->ltog,IS_GTOLM_DROP,c_loc->i[ptap->C_loc->rmap->n],c_loc->j,&nout,c_loc->j);CHKERRQ(ierr);
  if (c_loc->i[ptap->C_loc->rmap->n] != nout) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Incorrect mapping %" PetscInt_FMT " != %" PetscInt_FMT,c_loc->i[ptap->C_loc->rmap->n],nout);

  /* attach the supporting struct to Cmpi for reuse */
  Cmpi->product->data    = ptap;
  Cmpi->product->view    = MatView_MPIAIJ_PtAP;
  Cmpi->product->destroy = MatDestroy_MPIAIJ_PtAP;

  /* Cmpi is not ready for use - assembly will be done by MatPtAPNumeric() */
  Cmpi->assembled        = PETSC_FALSE;
  Cmpi->ops->ptapnumeric = MatPtAPNumeric_MPIAIJ_MPIAIJ_scalable;
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode MatPtAPSymbolicComputeOneRowOfAP_private(Mat A,Mat P,Mat P_oth,const PetscInt *map,PetscInt dof,PetscInt i,PetscHSetI dht,PetscHSetI oht)
{
  Mat_MPIAIJ           *a=(Mat_MPIAIJ*)A->data,*p=(Mat_MPIAIJ*)P->data;
  Mat_SeqAIJ           *ad=(Mat_SeqAIJ*)(a->A)->data,*ao=(Mat_SeqAIJ*)(a->B)->data,*p_oth=(Mat_SeqAIJ*)P_oth->data,*pd=(Mat_SeqAIJ*)p->A->data,*po=(Mat_SeqAIJ*)p->B->data;
  PetscInt             *ai,nzi,j,*aj,row,col,*pi,*pj,pnz,nzpi,*p_othcols,k;
  PetscInt             pcstart,pcend,column,offset;
  PetscErrorCode       ierr;

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
      ierr = PetscHSetIAdd(dht,pj[k]*dof+offset+pcstart);CHKERRQ(ierr);
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
      ierr = PetscHSetIAdd(oht,p->garray[pj[k]]*dof+offset);CHKERRQ(ierr);
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
          ierr = PetscHSetIAdd(dht,column);CHKERRQ(ierr);
        } else {
          ierr = PetscHSetIAdd(oht,column);CHKERRQ(ierr);
        }
      }
    }
  } /* end if (ao) */
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode MatPtAPNumericComputeOneRowOfAP_private(Mat A,Mat P,Mat P_oth,const PetscInt *map,PetscInt dof,PetscInt i,PetscHMapIV hmap)
{
  Mat_MPIAIJ           *a=(Mat_MPIAIJ*)A->data,*p=(Mat_MPIAIJ*)P->data;
  Mat_SeqAIJ           *ad=(Mat_SeqAIJ*)(a->A)->data,*ao=(Mat_SeqAIJ*)(a->B)->data,*p_oth=(Mat_SeqAIJ*)P_oth->data,*pd=(Mat_SeqAIJ*)p->A->data,*po=(Mat_SeqAIJ*)p->B->data;
  PetscInt             *ai,nzi,j,*aj,row,col,*pi,pnz,*p_othcols,pcstart,*pj,k,nzpi,offset;
  PetscScalar          ra,*aa,*pa;
  PetscErrorCode       ierr;

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
      ierr = PetscHMapIVAddValue(hmap,pj[k]*dof+offset+pcstart,ra*pa[k]);CHKERRQ(ierr);
    }
    ierr = PetscLogFlops(2.0*nzpi);CHKERRQ(ierr);
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
      ierr = PetscHMapIVAddValue(hmap,p->garray[pj[k]]*dof+offset,ra*pa[k]);CHKERRQ(ierr);
    }
    ierr = PetscLogFlops(2.0*nzpi);CHKERRQ(ierr);
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
        ierr = PetscHMapIVAddValue(hmap,p_othcols[col]*dof+offset,ra*pa[col]);CHKERRQ(ierr);
      }
      ierr = PetscLogFlops(2.0*pnz);CHKERRQ(ierr);
    }
  } /* end if (ao) */

  PetscFunctionReturn(0);
}

PetscErrorCode MatGetBrowsOfAcols_MPIXAIJ(Mat,Mat,PetscInt dof,MatReuse,Mat*);

PetscErrorCode MatPtAPNumeric_MPIAIJ_MPIXAIJ_allatonce(Mat A,Mat P,PetscInt dof,Mat C)
{
  PetscErrorCode    ierr;
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
  if (!ptap) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_ARG_WRONGSTATE,"PtAP cannot be computed. Missing data");
  if (!ptap->P_oth) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_ARG_WRONGSTATE,"PtAP cannot be reused. Do not call MatProductClear()");

  ierr = MatZeroEntries(C);CHKERRQ(ierr);

  /* Get P_oth = ptap->P_oth  and P_loc = ptap->P_loc */
  /*-----------------------------------------------------*/
  if (ptap->reuse == MAT_REUSE_MATRIX) {
    /* P_oth and P_loc are obtained in MatPtASymbolic() when reuse == MAT_INITIAL_MATRIX */
    ierr =  MatGetBrowsOfAcols_MPIXAIJ(A,P,dof,MAT_REUSE_MATRIX,&ptap->P_oth);CHKERRQ(ierr);
  }
  ierr = PetscObjectQuery((PetscObject)ptap->P_oth,"aoffdiagtopothmapping",(PetscObject*)&map);CHKERRQ(ierr);

  ierr = MatGetLocalSize(p->B,NULL,&pon);CHKERRQ(ierr);
  pon *= dof;
  ierr = PetscCalloc2(ptap->c_rmti[pon],&c_rmtj,ptap->c_rmti[pon],&c_rmta);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&am,NULL);CHKERRQ(ierr);
  cmaxr = 0;
  for (i=0; i<pon; i++) {
    cmaxr = PetscMax(cmaxr,ptap->c_rmti[i+1]-ptap->c_rmti[i]);
  }
  ierr = PetscCalloc4(cmaxr,&apindices,cmaxr,&apvalues,cmaxr,&apvaluestmp,pon,&c_rmtc);CHKERRQ(ierr);
  ierr = PetscHMapIVCreate(&hmap);CHKERRQ(ierr);
  ierr = PetscHMapIVResize(hmap,cmaxr);CHKERRQ(ierr);
  ierr = ISGetIndices(map,&mappingindices);CHKERRQ(ierr);
  for (i=0; i<am && pon; i++) {
    ierr = PetscHMapIVClear(hmap);CHKERRQ(ierr);
    offset = i%dof;
    ii     = i/dof;
    nzi = po->i[ii+1] - po->i[ii];
    if (!nzi) continue;
    ierr = MatPtAPNumericComputeOneRowOfAP_private(A,P,ptap->P_oth,mappingindices,dof,i,hmap);CHKERRQ(ierr);
    voff = 0;
    ierr = PetscHMapIVGetPairs(hmap,&voff,apindices,apvalues);CHKERRQ(ierr);
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
          ierr = PetscFindInt(apindices[jj],c_rmtc[pocol],c_rmtjj,&loc);CHKERRQ(ierr);
          if (loc>=0){ /* hit */
            c_rmtaa[loc] += apvaluestmp[jj];
            ierr = PetscLogFlops(1.0);CHKERRQ(ierr);
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
        ierr = PetscLogFlops(voff);CHKERRQ(ierr);
      } /* End jj */
    } /* End j */
  } /* End i */

  ierr = PetscFree4(apindices,apvalues,apvaluestmp,c_rmtc);CHKERRQ(ierr);
  ierr = PetscHMapIVDestroy(&hmap);CHKERRQ(ierr);

  ierr = MatGetLocalSize(P,NULL,&pn);CHKERRQ(ierr);
  pn *= dof;
  ierr = PetscCalloc2(ptap->c_othi[pn],&c_othj,ptap->c_othi[pn],&c_otha);CHKERRQ(ierr);

  ierr = PetscSFReduceBegin(ptap->sf,MPIU_INT,c_rmtj,c_othj,MPI_REPLACE);CHKERRQ(ierr);
  ierr = PetscSFReduceBegin(ptap->sf,MPIU_SCALAR,c_rmta,c_otha,MPI_REPLACE);CHKERRQ(ierr);
  ierr = MatGetOwnershipRangeColumn(P,&pcstart,&pcend);CHKERRQ(ierr);
  pcstart = pcstart*dof;
  pcend   = pcend*dof;
  cd = (Mat_SeqAIJ*)(c->A)->data;
  co = (Mat_SeqAIJ*)(c->B)->data;

  cmaxr = 0;
  for (i=0; i<pn; i++) {
    cmaxr = PetscMax(cmaxr,(cd->i[i+1]-cd->i[i])+(co->i[i+1]-co->i[i]));
  }
  ierr = PetscCalloc5(cmaxr,&apindices,cmaxr,&apvalues,cmaxr,&apvaluestmp,pn,&dcc,pn,&occ);CHKERRQ(ierr);
  ierr = PetscHMapIVCreate(&hmap);CHKERRQ(ierr);
  ierr = PetscHMapIVResize(hmap,cmaxr);CHKERRQ(ierr);
  for (i=0; i<am && pn; i++) {
    ierr = PetscHMapIVClear(hmap);CHKERRQ(ierr);
    offset = i%dof;
    ii     = i/dof;
    nzi = pd->i[ii+1] - pd->i[ii];
    if (!nzi) continue;
    ierr = MatPtAPNumericComputeOneRowOfAP_private(A,P,ptap->P_oth,mappingindices,dof,i,hmap);CHKERRQ(ierr);
    voff = 0;
    ierr = PetscHMapIVGetPairs(hmap,&voff,apindices,apvalues);CHKERRQ(ierr);
    if (!voff) continue;
    /* Form C(ii, :) */
    pdj = pd->j + pd->i[ii];
    pda = pd->a + pd->i[ii];
    for (j=0; j<nzi; j++) {
      row = pcstart + pdj[j] * dof + offset;
      for (jj=0; jj<voff; jj++) {
        apvaluestmp[jj] = apvalues[jj]*pda[j];
      }
      ierr = PetscLogFlops(voff);CHKERRQ(ierr);
      ierr = MatSetValues(C,1,&row,voff,apindices,apvaluestmp,ADD_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = ISRestoreIndices(map,&mappingindices);CHKERRQ(ierr);
  ierr = MatGetOwnershipRangeColumn(C,&ccstart,&ccend);CHKERRQ(ierr);
  ierr = PetscFree5(apindices,apvalues,apvaluestmp,dcc,occ);CHKERRQ(ierr);
  ierr = PetscHMapIVDestroy(&hmap);CHKERRQ(ierr);
  ierr = PetscSFReduceEnd(ptap->sf,MPIU_INT,c_rmtj,c_othj,MPI_REPLACE);CHKERRQ(ierr);
  ierr = PetscSFReduceEnd(ptap->sf,MPIU_SCALAR,c_rmta,c_otha,MPI_REPLACE);CHKERRQ(ierr);
  ierr = PetscFree2(c_rmtj,c_rmta);CHKERRQ(ierr);

  /* Add contributions from remote */
  for (i = 0; i < pn; i++) {
    row = i + pcstart;
    ierr = MatSetValues(C,1,&row,ptap->c_othi[i+1]-ptap->c_othi[i],c_othj+ptap->c_othi[i],c_otha+ptap->c_othi[i],ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = PetscFree2(c_othj,c_otha);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ptap->reuse = MAT_REUSE_MATRIX;
  PetscFunctionReturn(0);
}

PetscErrorCode MatPtAPNumeric_MPIAIJ_MPIAIJ_allatonce(Mat A,Mat P,Mat C)
{
  PetscErrorCode      ierr;

  PetscFunctionBegin;

  ierr = MatPtAPNumeric_MPIAIJ_MPIXAIJ_allatonce(A,P,1,C);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatPtAPNumeric_MPIAIJ_MPIXAIJ_allatonce_merged(Mat A,Mat P,PetscInt dof,Mat C)
{
  PetscErrorCode    ierr;
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
  if (!ptap) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_ARG_WRONGSTATE,"PtAP cannot be computed. Missing data");
  if (!ptap->P_oth) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_ARG_WRONGSTATE,"PtAP cannot be reused. Do not call MatProductClear()");

  ierr = MatZeroEntries(C);CHKERRQ(ierr);

  /* Get P_oth = ptap->P_oth  and P_loc = ptap->P_loc */
  /*-----------------------------------------------------*/
  if (ptap->reuse == MAT_REUSE_MATRIX) {
    /* P_oth and P_loc are obtained in MatPtASymbolic() when reuse == MAT_INITIAL_MATRIX */
    ierr =  MatGetBrowsOfAcols_MPIXAIJ(A,P,dof,MAT_REUSE_MATRIX,&ptap->P_oth);CHKERRQ(ierr);
  }
  ierr = PetscObjectQuery((PetscObject)ptap->P_oth,"aoffdiagtopothmapping",(PetscObject*)&map);CHKERRQ(ierr);
  ierr = MatGetLocalSize(p->B,NULL,&pon);CHKERRQ(ierr);
  pon *= dof;
  ierr = MatGetLocalSize(P,NULL,&pn);CHKERRQ(ierr);
  pn  *= dof;

  ierr = PetscCalloc2(ptap->c_rmti[pon],&c_rmtj,ptap->c_rmti[pon],&c_rmta);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&am,NULL);CHKERRQ(ierr);
  ierr = MatGetOwnershipRangeColumn(P,&pcstart,&pcend);CHKERRQ(ierr);
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
  ierr = PetscCalloc4(cmaxr,&apindices,cmaxr,&apvalues,cmaxr,&apvaluestmp,pon,&c_rmtc);CHKERRQ(ierr);
  ierr = PetscHMapIVCreate(&hmap);CHKERRQ(ierr);
  ierr = PetscHMapIVResize(hmap,cmaxr);CHKERRQ(ierr);
  ierr = ISGetIndices(map,&mappingindices);CHKERRQ(ierr);
  for (i=0; i<am && (pon || pn); i++) {
    ierr = PetscHMapIVClear(hmap);CHKERRQ(ierr);
    offset = i%dof;
    ii     = i/dof;
    nzi  = po->i[ii+1] - po->i[ii];
    dnzi = pd->i[ii+1] - pd->i[ii];
    if (!nzi && !dnzi) continue;
    ierr = MatPtAPNumericComputeOneRowOfAP_private(A,P,ptap->P_oth,mappingindices,dof,i,hmap);CHKERRQ(ierr);
    voff = 0;
    ierr = PetscHMapIVGetPairs(hmap,&voff,apindices,apvalues);CHKERRQ(ierr);
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
          ierr = PetscFindInt(apindices[jj],c_rmtc[pocol],c_rmtjj,&loc);CHKERRQ(ierr);
          if (loc>=0){ /* hit */
            c_rmtaa[loc] += apvaluestmp[jj];
            ierr = PetscLogFlops(1.0);CHKERRQ(ierr);
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
      ierr = PetscLogFlops(voff);CHKERRQ(ierr);
    } /* End j */

    /* Form local C(ii, :) */
    pdj = pd->j + pd->i[ii];
    pda = pd->a + pd->i[ii];
    for (j=0; j<dnzi; j++) {
      row = pcstart + pdj[j] * dof + offset;
      for (jj=0; jj<voff; jj++) {
        apvaluestmp[jj] = apvalues[jj]*pda[j];
      }/* End kk */
      ierr = PetscLogFlops(voff);CHKERRQ(ierr);
      ierr = MatSetValues(C,1,&row,voff,apindices,apvaluestmp,ADD_VALUES);CHKERRQ(ierr);
    }/* End j */
  } /* End i */

  ierr = ISRestoreIndices(map,&mappingindices);CHKERRQ(ierr);
  ierr = PetscFree4(apindices,apvalues,apvaluestmp,c_rmtc);CHKERRQ(ierr);
  ierr = PetscHMapIVDestroy(&hmap);CHKERRQ(ierr);
  ierr = PetscCalloc2(ptap->c_othi[pn],&c_othj,ptap->c_othi[pn],&c_otha);CHKERRQ(ierr);

  ierr = PetscSFReduceBegin(ptap->sf,MPIU_INT,c_rmtj,c_othj,MPI_REPLACE);CHKERRQ(ierr);
  ierr = PetscSFReduceBegin(ptap->sf,MPIU_SCALAR,c_rmta,c_otha,MPI_REPLACE);CHKERRQ(ierr);
  ierr = PetscSFReduceEnd(ptap->sf,MPIU_INT,c_rmtj,c_othj,MPI_REPLACE);CHKERRQ(ierr);
  ierr = PetscSFReduceEnd(ptap->sf,MPIU_SCALAR,c_rmta,c_otha,MPI_REPLACE);CHKERRQ(ierr);
  ierr = PetscFree2(c_rmtj,c_rmta);CHKERRQ(ierr);

  /* Add contributions from remote */
  for (i = 0; i < pn; i++) {
    row = i + pcstart;
    ierr = MatSetValues(C,1,&row,ptap->c_othi[i+1]-ptap->c_othi[i],c_othj+ptap->c_othi[i],c_otha+ptap->c_othi[i],ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = PetscFree2(c_othj,c_otha);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ptap->reuse = MAT_REUSE_MATRIX;
  PetscFunctionReturn(0);
}

PetscErrorCode MatPtAPNumeric_MPIAIJ_MPIAIJ_allatonce_merged(Mat A,Mat P,Mat C)
{
  PetscErrorCode      ierr;

  PetscFunctionBegin;

  ierr = MatPtAPNumeric_MPIAIJ_MPIXAIJ_allatonce_merged(A,P,1,C);CHKERRQ(ierr);
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
  if (Cmpi->product->data) SETERRQ(PetscObjectComm((PetscObject)Cmpi),PETSC_ERR_PLIB,"Product data not empty");
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);

  /* Create symbolic parallel matrix Cmpi */
  ierr = MatGetLocalSize(P,NULL,&pn);CHKERRQ(ierr);
  pn *= dof;
  ierr = MatGetType(A,&mtype);CHKERRQ(ierr);
  ierr = MatSetType(Cmpi,mtype);CHKERRQ(ierr);
  ierr = MatSetSizes(Cmpi,pn,pn,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);

  ierr = PetscNew(&ptap);CHKERRQ(ierr);
  ptap->reuse = MAT_INITIAL_MATRIX;
  ptap->algType = 2;

  /* Get P_oth by taking rows of P (= non-zero cols of local A) from other processors */
  ierr = MatGetBrowsOfAcols_MPIXAIJ(A,P,dof,MAT_INITIAL_MATRIX,&ptap->P_oth);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject)ptap->P_oth,"aoffdiagtopothmapping",(PetscObject*)&map);CHKERRQ(ierr);
  /* This equals to the number of offdiag columns in P */
  ierr = MatGetLocalSize(p->B,NULL,&pon);CHKERRQ(ierr);
  pon *= dof;
  /* offsets */
  ierr = PetscMalloc1(pon+1,&ptap->c_rmti);CHKERRQ(ierr);
  /* The number of columns we will send to remote ranks */
  ierr = PetscMalloc1(pon,&c_rmtc);CHKERRQ(ierr);
  ierr = PetscMalloc1(pon,&hta);CHKERRQ(ierr);
  for (i=0; i<pon; i++) {
    ierr = PetscHSetICreate(&hta[i]);CHKERRQ(ierr);
  }
  ierr = MatGetLocalSize(A,&am,NULL);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&arstart,&arend);CHKERRQ(ierr);
  /* Create hash table to merge all columns for C(i, :) */
  ierr = PetscHSetICreate(&ht);CHKERRQ(ierr);

  ierr = ISGetIndices(map,&mappingindices);CHKERRQ(ierr);
  ptap->c_rmti[0] = 0;
  /* 2) Pass 1: calculate the size for C_rmt (a matrix need to be sent to other processors)  */
  for (i=0; i<am && pon; i++) {
    /* Form one row of AP */
    ierr = PetscHSetIClear(ht);CHKERRQ(ierr);
    offset = i%dof;
    ii     = i/dof;
    /* If the off diag is empty, we should not do any calculation */
    nzi = po->i[ii+1] - po->i[ii];
    if (!nzi) continue;

    ierr = MatPtAPSymbolicComputeOneRowOfAP_private(A,P,ptap->P_oth,mappingindices,dof,i,ht,ht);CHKERRQ(ierr);
    ierr = PetscHSetIGetSize(ht,&htsize);CHKERRQ(ierr);
    /* If AP is empty, just continue */
    if (!htsize) continue;
    /* Form C(ii, :) */
    poj = po->j + po->i[ii];
    for (j=0; j<nzi; j++) {
      ierr = PetscHSetIUpdate(hta[poj[j]*dof+offset],ht);CHKERRQ(ierr);
    }
  }

  for (i=0; i<pon; i++) {
    ierr = PetscHSetIGetSize(hta[i],&htsize);CHKERRQ(ierr);
    ptap->c_rmti[i+1] = ptap->c_rmti[i] + htsize;
    c_rmtc[i] = htsize;
  }

  ierr = PetscMalloc1(ptap->c_rmti[pon],&c_rmtj);CHKERRQ(ierr);

  for (i=0; i<pon; i++) {
    off = 0;
    ierr = PetscHSetIGetElems(hta[i],&off,c_rmtj+ptap->c_rmti[i]);CHKERRQ(ierr);
    ierr = PetscHSetIDestroy(&hta[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(hta);CHKERRQ(ierr);

  ierr = PetscMalloc1(pon,&iremote);CHKERRQ(ierr);
  for (i=0; i<pon; i++) {
    owner = 0; lidx = 0;
    offset = i%dof;
    ii     = i/dof;
    ierr = PetscLayoutFindOwnerIndex(P->cmap,p->garray[ii],&owner,&lidx);CHKERRQ(ierr);
    iremote[i].index = lidx*dof + offset;
    iremote[i].rank  = owner;
  }

  ierr = PetscSFCreate(comm,&sf);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(sf,pn,pon,NULL,PETSC_OWN_POINTER,iremote,PETSC_OWN_POINTER);CHKERRQ(ierr);
  /* Reorder ranks properly so that the data handled by gather and scatter have the same order */
  ierr = PetscSFSetRankOrder(sf,PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscSFSetFromOptions(sf);CHKERRQ(ierr);
  ierr = PetscSFSetUp(sf);CHKERRQ(ierr);
  /* How many neighbors have contributions to my rows? */
  ierr = PetscSFComputeDegreeBegin(sf,&rootdegrees);CHKERRQ(ierr);
  ierr = PetscSFComputeDegreeEnd(sf,&rootdegrees);CHKERRQ(ierr);
  rootspacesize = 0;
  for (i = 0; i < pn; i++) {
    rootspacesize += rootdegrees[i];
  }
  ierr = PetscMalloc1(rootspacesize,&rootspace);CHKERRQ(ierr);
  ierr = PetscMalloc1(rootspacesize+1,&rootspaceoffsets);CHKERRQ(ierr);
  /* Get information from leaves
   * Number of columns other people contribute to my rows
   * */
  ierr = PetscSFGatherBegin(sf,MPIU_INT,c_rmtc,rootspace);CHKERRQ(ierr);
  ierr = PetscSFGatherEnd(sf,MPIU_INT,c_rmtc,rootspace);CHKERRQ(ierr);
  ierr = PetscFree(c_rmtc);CHKERRQ(ierr);
  ierr = PetscCalloc1(pn+1,&ptap->c_othi);CHKERRQ(ierr);
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
  ierr = PetscFree(rootspace);CHKERRQ(ierr);

  ierr = PetscMalloc1(pon,&c_rmtoffsets);CHKERRQ(ierr);
  ierr = PetscSFScatterBegin(sf,MPIU_INT,rootspaceoffsets,c_rmtoffsets);CHKERRQ(ierr);
  ierr = PetscSFScatterEnd(sf,MPIU_INT,rootspaceoffsets,c_rmtoffsets);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);
  ierr = PetscFree(rootspaceoffsets);CHKERRQ(ierr);

  ierr = PetscCalloc1(ptap->c_rmti[pon],&iremote);CHKERRQ(ierr);
  nleaves = 0;
  for (i = 0; i<pon; i++) {
    owner = 0;
    ii = i/dof;
    ierr = PetscLayoutFindOwnerIndex(P->cmap,p->garray[ii],&owner,NULL);CHKERRQ(ierr);
    sendncols = ptap->c_rmti[i+1] - ptap->c_rmti[i];
    for (j=0; j<sendncols; j++) {
      iremote[nleaves].rank = owner;
      iremote[nleaves++].index = c_rmtoffsets[i] + j;
    }
  }
  ierr = PetscFree(c_rmtoffsets);CHKERRQ(ierr);
  ierr = PetscCalloc1(ptap->c_othi[pn],&c_othj);CHKERRQ(ierr);

  ierr = PetscSFCreate(comm,&ptap->sf);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(ptap->sf,ptap->c_othi[pn],nleaves,NULL,PETSC_OWN_POINTER,iremote,PETSC_OWN_POINTER);CHKERRQ(ierr);
  ierr = PetscSFSetFromOptions(ptap->sf);CHKERRQ(ierr);
  /* One to one map */
  ierr = PetscSFReduceBegin(ptap->sf,MPIU_INT,c_rmtj,c_othj,MPI_REPLACE);CHKERRQ(ierr);

  ierr = PetscMalloc2(pn,&dnz,pn,&onz);CHKERRQ(ierr);
  ierr = PetscHSetICreate(&oht);CHKERRQ(ierr);
  ierr = MatGetOwnershipRangeColumn(P,&pcstart,&pcend);CHKERRQ(ierr);
  pcstart *= dof;
  pcend   *= dof;
  ierr = PetscMalloc2(pn,&hta,pn,&hto);CHKERRQ(ierr);
  for (i=0; i<pn; i++) {
    ierr = PetscHSetICreate(&hta[i]);CHKERRQ(ierr);
    ierr = PetscHSetICreate(&hto[i]);CHKERRQ(ierr);
  }
  /* Work on local part */
  /* 4) Pass 1: Estimate memory for C_loc */
  for (i=0; i<am && pn; i++) {
    ierr = PetscHSetIClear(ht);CHKERRQ(ierr);
    ierr = PetscHSetIClear(oht);CHKERRQ(ierr);
    offset = i%dof;
    ii     = i/dof;
    nzi = pd->i[ii+1] - pd->i[ii];
    if (!nzi) continue;

    ierr = MatPtAPSymbolicComputeOneRowOfAP_private(A,P,ptap->P_oth,mappingindices,dof,i,ht,oht);CHKERRQ(ierr);
    ierr = PetscHSetIGetSize(ht,&htsize);CHKERRQ(ierr);
    ierr = PetscHSetIGetSize(oht,&htosize);CHKERRQ(ierr);
    if (!(htsize+htosize)) continue;
    /* Form C(ii, :) */
    pdj = pd->j + pd->i[ii];
    for (j=0; j<nzi; j++) {
      ierr = PetscHSetIUpdate(hta[pdj[j]*dof+offset],ht);CHKERRQ(ierr);
      ierr = PetscHSetIUpdate(hto[pdj[j]*dof+offset],oht);CHKERRQ(ierr);
    }
  }

  ierr = ISRestoreIndices(map,&mappingindices);CHKERRQ(ierr);

  ierr = PetscHSetIDestroy(&ht);CHKERRQ(ierr);
  ierr = PetscHSetIDestroy(&oht);CHKERRQ(ierr);

  /* Get remote data */
  ierr = PetscSFReduceEnd(ptap->sf,MPIU_INT,c_rmtj,c_othj,MPI_REPLACE);CHKERRQ(ierr);
  ierr = PetscFree(c_rmtj);CHKERRQ(ierr);

  for (i = 0; i < pn; i++) {
    nzi = ptap->c_othi[i+1] - ptap->c_othi[i];
    rdj = c_othj + ptap->c_othi[i];
    for (j = 0; j < nzi; j++) {
      col = rdj[j];
      /* diag part */
      if (col>=pcstart && col<pcend) {
        ierr = PetscHSetIAdd(hta[i],col);CHKERRQ(ierr);
      } else { /* off diag */
        ierr = PetscHSetIAdd(hto[i],col);CHKERRQ(ierr);
      }
    }
    ierr = PetscHSetIGetSize(hta[i],&htsize);CHKERRQ(ierr);
    dnz[i] = htsize;
    ierr = PetscHSetIDestroy(&hta[i]);CHKERRQ(ierr);
    ierr = PetscHSetIGetSize(hto[i],&htsize);CHKERRQ(ierr);
    onz[i] = htsize;
    ierr = PetscHSetIDestroy(&hto[i]);CHKERRQ(ierr);
  }

  ierr = PetscFree2(hta,hto);CHKERRQ(ierr);
  ierr = PetscFree(c_othj);CHKERRQ(ierr);

  /* local sizes and preallocation */
  ierr = MatSetSizes(Cmpi,pn,pn,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetBlockSizes(Cmpi,dof>1? dof: P->cmap->bs,dof>1? dof: P->cmap->bs);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(Cmpi,0,dnz,0,onz);CHKERRQ(ierr);
  ierr = MatSetUp(Cmpi);CHKERRQ(ierr);
  ierr = PetscFree2(dnz,onz);CHKERRQ(ierr);

  /* attach the supporting struct to Cmpi for reuse */
  Cmpi->product->data    = ptap;
  Cmpi->product->destroy = MatDestroy_MPIAIJ_PtAP;
  Cmpi->product->view    = MatView_MPIAIJ_PtAP;

  /* Cmpi is not ready for use - assembly will be done by MatPtAPNumeric() */
  Cmpi->assembled = PETSC_FALSE;
  /* pick an algorithm */
  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)A),((PetscObject)A)->prefix,"MatPtAP","Mat");CHKERRQ(ierr);
  alg = 0;
  ierr = PetscOptionsEList("-matptap_allatonce_via","PtAP allatonce numeric approach","MatPtAP",algTypes,nalg,algTypes[alg],&alg,&flg);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatPtAPSymbolic_MPIAIJ_MPIXAIJ_allatonce(A,P,1,fill,C);CHKERRQ(ierr);
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
  if (Cmpi->product->data) SETERRQ(PetscObjectComm((PetscObject)Cmpi),PETSC_ERR_PLIB,"Product data not empty");
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);

  /* Create symbolic parallel matrix Cmpi */
  ierr = MatGetLocalSize(P,NULL,&pn);CHKERRQ(ierr);
  pn *= dof;
  ierr = MatGetType(A,&mtype);CHKERRQ(ierr);
  ierr = MatSetType(Cmpi,mtype);CHKERRQ(ierr);
  ierr = MatSetSizes(Cmpi,pn,pn,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);

  ierr        = PetscNew(&ptap);CHKERRQ(ierr);
  ptap->reuse = MAT_INITIAL_MATRIX;
  ptap->algType = 3;

  /* 0) Get P_oth by taking rows of P (= non-zero cols of local A) from other processors */
  ierr = MatGetBrowsOfAcols_MPIXAIJ(A,P,dof,MAT_INITIAL_MATRIX,&ptap->P_oth);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject)ptap->P_oth,"aoffdiagtopothmapping",(PetscObject*)&map);CHKERRQ(ierr);

  /* This equals to the number of offdiag columns in P */
  ierr = MatGetLocalSize(p->B,NULL,&pon);CHKERRQ(ierr);
  pon *= dof;
  /* offsets */
  ierr = PetscMalloc1(pon+1,&ptap->c_rmti);CHKERRQ(ierr);
  /* The number of columns we will send to remote ranks */
  ierr = PetscMalloc1(pon,&c_rmtc);CHKERRQ(ierr);
  ierr = PetscMalloc1(pon,&hta);CHKERRQ(ierr);
  for (i=0; i<pon; i++) {
    ierr = PetscHSetICreate(&hta[i]);CHKERRQ(ierr);
  }
  ierr = MatGetLocalSize(A,&am,NULL);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&arstart,&arend);CHKERRQ(ierr);
  /* Create hash table to merge all columns for C(i, :) */
  ierr = PetscHSetICreate(&ht);CHKERRQ(ierr);
  ierr = PetscHSetICreate(&oht);CHKERRQ(ierr);
  ierr = PetscMalloc2(pn,&htd,pn,&hto);CHKERRQ(ierr);
  for (i=0; i<pn; i++) {
    ierr = PetscHSetICreate(&htd[i]);CHKERRQ(ierr);
    ierr = PetscHSetICreate(&hto[i]);CHKERRQ(ierr);
  }

  ierr = ISGetIndices(map,&mappingindices);CHKERRQ(ierr);
  ptap->c_rmti[0] = 0;
  /* 2) Pass 1: calculate the size for C_rmt (a matrix need to be sent to other processors)  */
  for (i=0; i<am && (pon || pn); i++) {
    /* Form one row of AP */
    ierr = PetscHSetIClear(ht);CHKERRQ(ierr);
    ierr = PetscHSetIClear(oht);CHKERRQ(ierr);
    offset = i%dof;
    ii     = i/dof;
    /* If the off diag is empty, we should not do any calculation */
    nzi = po->i[ii+1] - po->i[ii];
    dnzi = pd->i[ii+1] - pd->i[ii];
    if (!nzi && !dnzi) continue;

    ierr = MatPtAPSymbolicComputeOneRowOfAP_private(A,P,ptap->P_oth,mappingindices,dof,i,ht,oht);CHKERRQ(ierr);
    ierr = PetscHSetIGetSize(ht,&htsize);CHKERRQ(ierr);
    ierr = PetscHSetIGetSize(oht,&htosize);CHKERRQ(ierr);
    /* If AP is empty, just continue */
    if (!(htsize+htosize)) continue;

    /* Form remote C(ii, :) */
    poj = po->j + po->i[ii];
    for (j=0; j<nzi; j++) {
      ierr = PetscHSetIUpdate(hta[poj[j]*dof+offset],ht);CHKERRQ(ierr);
      ierr = PetscHSetIUpdate(hta[poj[j]*dof+offset],oht);CHKERRQ(ierr);
    }

    /* Form local C(ii, :) */
    pdj = pd->j + pd->i[ii];
    for (j=0; j<dnzi; j++) {
      ierr = PetscHSetIUpdate(htd[pdj[j]*dof+offset],ht);CHKERRQ(ierr);
      ierr = PetscHSetIUpdate(hto[pdj[j]*dof+offset],oht);CHKERRQ(ierr);
    }
  }

  ierr = ISRestoreIndices(map,&mappingindices);CHKERRQ(ierr);

  ierr = PetscHSetIDestroy(&ht);CHKERRQ(ierr);
  ierr = PetscHSetIDestroy(&oht);CHKERRQ(ierr);

  for (i=0; i<pon; i++) {
    ierr = PetscHSetIGetSize(hta[i],&htsize);CHKERRQ(ierr);
    ptap->c_rmti[i+1] = ptap->c_rmti[i] + htsize;
    c_rmtc[i] = htsize;
  }

  ierr = PetscMalloc1(ptap->c_rmti[pon],&c_rmtj);CHKERRQ(ierr);

  for (i=0; i<pon; i++) {
    off = 0;
    ierr = PetscHSetIGetElems(hta[i],&off,c_rmtj+ptap->c_rmti[i]);CHKERRQ(ierr);
    ierr = PetscHSetIDestroy(&hta[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(hta);CHKERRQ(ierr);

  ierr = PetscMalloc1(pon,&iremote);CHKERRQ(ierr);
  for (i=0; i<pon; i++) {
    owner = 0; lidx = 0;
    offset = i%dof;
    ii     = i/dof;
    ierr = PetscLayoutFindOwnerIndex(P->cmap,p->garray[ii],&owner,&lidx);CHKERRQ(ierr);
    iremote[i].index = lidx*dof+offset;
    iremote[i].rank  = owner;
  }

  ierr = PetscSFCreate(comm,&sf);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(sf,pn,pon,NULL,PETSC_OWN_POINTER,iremote,PETSC_OWN_POINTER);CHKERRQ(ierr);
  /* Reorder ranks properly so that the data handled by gather and scatter have the same order */
  ierr = PetscSFSetRankOrder(sf,PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscSFSetFromOptions(sf);CHKERRQ(ierr);
  ierr = PetscSFSetUp(sf);CHKERRQ(ierr);
  /* How many neighbors have contributions to my rows? */
  ierr = PetscSFComputeDegreeBegin(sf,&rootdegrees);CHKERRQ(ierr);
  ierr = PetscSFComputeDegreeEnd(sf,&rootdegrees);CHKERRQ(ierr);
  rootspacesize = 0;
  for (i = 0; i < pn; i++) {
    rootspacesize += rootdegrees[i];
  }
  ierr = PetscMalloc1(rootspacesize,&rootspace);CHKERRQ(ierr);
  ierr = PetscMalloc1(rootspacesize+1,&rootspaceoffsets);CHKERRQ(ierr);
  /* Get information from leaves
   * Number of columns other people contribute to my rows
   * */
  ierr = PetscSFGatherBegin(sf,MPIU_INT,c_rmtc,rootspace);CHKERRQ(ierr);
  ierr = PetscSFGatherEnd(sf,MPIU_INT,c_rmtc,rootspace);CHKERRQ(ierr);
  ierr = PetscFree(c_rmtc);CHKERRQ(ierr);
  ierr = PetscMalloc1(pn+1,&ptap->c_othi);CHKERRQ(ierr);
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
  ierr = PetscFree(rootspace);CHKERRQ(ierr);

  ierr = PetscMalloc1(pon,&c_rmtoffsets);CHKERRQ(ierr);
  ierr = PetscSFScatterBegin(sf,MPIU_INT,rootspaceoffsets,c_rmtoffsets);CHKERRQ(ierr);
  ierr = PetscSFScatterEnd(sf,MPIU_INT,rootspaceoffsets,c_rmtoffsets);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);
  ierr = PetscFree(rootspaceoffsets);CHKERRQ(ierr);

  ierr = PetscCalloc1(ptap->c_rmti[pon],&iremote);CHKERRQ(ierr);
  nleaves = 0;
  for (i = 0; i<pon; i++) {
    owner = 0;
    ii    = i/dof;
    ierr = PetscLayoutFindOwnerIndex(P->cmap,p->garray[ii],&owner,NULL);CHKERRQ(ierr);
    sendncols = ptap->c_rmti[i+1] - ptap->c_rmti[i];
    for (j=0; j<sendncols; j++) {
      iremote[nleaves].rank    = owner;
      iremote[nleaves++].index = c_rmtoffsets[i] + j;
    }
  }
  ierr = PetscFree(c_rmtoffsets);CHKERRQ(ierr);
  ierr = PetscCalloc1(ptap->c_othi[pn],&c_othj);CHKERRQ(ierr);

  ierr = PetscSFCreate(comm,&ptap->sf);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(ptap->sf,ptap->c_othi[pn],nleaves,NULL,PETSC_OWN_POINTER,iremote,PETSC_OWN_POINTER);CHKERRQ(ierr);
  ierr = PetscSFSetFromOptions(ptap->sf);CHKERRQ(ierr);
  /* One to one map */
  ierr = PetscSFReduceBegin(ptap->sf,MPIU_INT,c_rmtj,c_othj,MPI_REPLACE);CHKERRQ(ierr);
  /* Get remote data */
  ierr = PetscSFReduceEnd(ptap->sf,MPIU_INT,c_rmtj,c_othj,MPI_REPLACE);CHKERRQ(ierr);
  ierr = PetscFree(c_rmtj);CHKERRQ(ierr);
  ierr = PetscMalloc2(pn,&dnz,pn,&onz);CHKERRQ(ierr);
  ierr = MatGetOwnershipRangeColumn(P,&pcstart,&pcend);CHKERRQ(ierr);
  pcstart *= dof;
  pcend   *= dof;
  for (i = 0; i < pn; i++) {
    nzi = ptap->c_othi[i+1] - ptap->c_othi[i];
    rdj = c_othj + ptap->c_othi[i];
    for (j = 0; j < nzi; j++) {
      col =  rdj[j];
      /* diag part */
      if (col>=pcstart && col<pcend) {
        ierr = PetscHSetIAdd(htd[i],col);CHKERRQ(ierr);
      } else { /* off diag */
        ierr = PetscHSetIAdd(hto[i],col);CHKERRQ(ierr);
      }
    }
    ierr = PetscHSetIGetSize(htd[i],&htsize);CHKERRQ(ierr);
    dnz[i] = htsize;
    ierr = PetscHSetIDestroy(&htd[i]);CHKERRQ(ierr);
    ierr = PetscHSetIGetSize(hto[i],&htsize);CHKERRQ(ierr);
    onz[i] = htsize;
    ierr = PetscHSetIDestroy(&hto[i]);CHKERRQ(ierr);
  }

  ierr = PetscFree2(htd,hto);CHKERRQ(ierr);
  ierr = PetscFree(c_othj);CHKERRQ(ierr);

  /* local sizes and preallocation */
  ierr = MatSetSizes(Cmpi,pn,pn,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetBlockSizes(Cmpi, dof>1? dof: P->cmap->bs,dof>1? dof: P->cmap->bs);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(Cmpi,0,dnz,0,onz);CHKERRQ(ierr);
  ierr = PetscFree2(dnz,onz);CHKERRQ(ierr);

  /* attach the supporting struct to Cmpi for reuse */
  Cmpi->product->data    = ptap;
  Cmpi->product->destroy = MatDestroy_MPIAIJ_PtAP;
  Cmpi->product->view    = MatView_MPIAIJ_PtAP;

  /* Cmpi is not ready for use - assembly will be done by MatPtAPNumeric() */
  Cmpi->assembled = PETSC_FALSE;
  /* pick an algorithm */
  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)A),((PetscObject)A)->prefix,"MatPtAP","Mat");CHKERRQ(ierr);
  alg = 0;
  ierr = PetscOptionsEList("-matptap_allatonce_via","PtAP allatonce numeric approach","MatPtAP",algTypes,nalg,algTypes[alg],&alg,&flg);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatPtAPSymbolic_MPIAIJ_MPIXAIJ_allatonce_merged(A,P,1,fill,C);CHKERRQ(ierr);
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
  if (Cmpi->product->data) SETERRQ(PetscObjectComm((PetscObject)Cmpi),PETSC_ERR_PLIB,"Product data not empty");
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);

  if (size > 1) ao = (Mat_SeqAIJ*)(a->B)->data;

  /* create symbolic parallel matrix Cmpi */
  ierr = MatGetType(A,&mtype);CHKERRQ(ierr);
  ierr = MatSetType(Cmpi,mtype);CHKERRQ(ierr);

  /* Do dense axpy in MatPtAPNumeric_MPIAIJ_MPIAIJ() */
  Cmpi->ops->ptapnumeric = MatPtAPNumeric_MPIAIJ_MPIAIJ;

  /* create struct Mat_APMPI and attached it to C later */
  ierr        = PetscNew(&ptap);CHKERRQ(ierr);
  ptap->reuse = MAT_INITIAL_MATRIX;
  ptap->algType = 1;

  /* get P_oth by taking rows of P (= non-zero cols of local A) from other processors */
  ierr = MatGetBrowsOfAoCols_MPIAIJ(A,P,MAT_INITIAL_MATRIX,&ptap->startsj_s,&ptap->startsj_r,&ptap->bufa,&ptap->P_oth);CHKERRQ(ierr);
  /* get P_loc by taking all local rows of P */
  ierr = MatMPIAIJGetLocalMat(P,MAT_INITIAL_MATRIX,&ptap->P_loc);CHKERRQ(ierr);

  /* (0) compute Rd = Pd^T, Ro = Po^T  */
  /* --------------------------------- */
  ierr = MatTranspose(p->A,MAT_INITIAL_MATRIX,&ptap->Rd);CHKERRQ(ierr);
  ierr = MatTranspose(p->B,MAT_INITIAL_MATRIX,&ptap->Ro);CHKERRQ(ierr);

  /* (1) compute symbolic AP = A_loc*P = Ad*P_loc + Ao*P_oth (api,apj) */
  /* ----------------------------------------------------------------- */
  p_loc  = (Mat_SeqAIJ*)(ptap->P_loc)->data;
  if (ptap->P_oth) p_oth  = (Mat_SeqAIJ*)(ptap->P_oth)->data;

  /* create and initialize a linked list */
  ierr = PetscTableCreate(pn,pN,&ta);CHKERRQ(ierr); /* for compute AP_loc and Cmpi */
  MatRowMergeMax_SeqAIJ(p_loc,ptap->P_loc->rmap->N,ta);
  MatRowMergeMax_SeqAIJ(p_oth,ptap->P_oth->rmap->N,ta);
  ierr = PetscTableGetCount(ta,&Crmax);CHKERRQ(ierr); /* Crmax = nnz(sum of Prows) */
  /* printf("[%d] est %d, Crmax %d; pN %d\n",rank,5*(p_loc->rmax+p_oth->rmax + (PetscInt)(1.e-2*pN)),Crmax,pN); */

  ierr = PetscLLCondensedCreate(Crmax,pN,&lnk,&lnkbt);CHKERRQ(ierr);

  /* Initial FreeSpace size is fill*(nnz(A) + nnz(P)) */
  if (ao) {
    ierr = PetscFreeSpaceGet(PetscRealIntMultTruncate(fill,PetscIntSumTruncate(ad->i[am],PetscIntSumTruncate(ao->i[am],p_loc->i[pm]))),&free_space);CHKERRQ(ierr);
  } else {
    ierr = PetscFreeSpaceGet(PetscRealIntMultTruncate(fill,PetscIntSumTruncate(ad->i[am],p_loc->i[pm])),&free_space);CHKERRQ(ierr);
  }
  current_space = free_space;
  nspacedouble  = 0;

  ierr   = PetscMalloc1(am+1,&api);CHKERRQ(ierr);
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
      ierr = PetscLLCondensedAddSorted(pnz,Jptr,lnk,lnkbt);CHKERRQ(ierr);
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
        ierr = PetscLLCondensedAddSorted(pnz,Jptr,lnk,lnkbt);CHKERRQ(ierr);
      }
    }
    apnz     = lnk[0];
    api[i+1] = api[i] + apnz;
    if (ap_rmax < apnz) ap_rmax = apnz;

    /* if free space is not available, double the total space in the list */
    if (current_space->local_remaining<apnz) {
      ierr = PetscFreeSpaceGet(PetscIntSumTruncate(apnz,current_space->total_array_size),&current_space);CHKERRQ(ierr);
      nspacedouble++;
    }

    /* Copy data into free space, then initialize lnk */
    ierr = PetscLLCondensedClean(pN,apnz,current_space->array,lnk,lnkbt);CHKERRQ(ierr);

    current_space->array           += apnz;
    current_space->local_used      += apnz;
    current_space->local_remaining -= apnz;
  }
  /* Allocate space for apj and apv, initialize apj, and */
  /* destroy list of free space and other temporary array(s) */
  ierr = PetscMalloc2(api[am],&apj,api[am],&apv);CHKERRQ(ierr);
  ierr = PetscFreeSpaceContiguous(&free_space,apj);CHKERRQ(ierr);
  ierr = PetscLLDestroy(lnk,lnkbt);CHKERRQ(ierr);

  /* Create AP_loc for reuse */
  ierr = MatCreateSeqAIJWithArrays(PETSC_COMM_SELF,am,pN,api,apj,apv,&ptap->AP_loc);CHKERRQ(ierr);
  ierr = MatSetType(ptap->AP_loc,((PetscObject)p->A)->type_name);CHKERRQ(ierr);
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
    ierr = PetscInfo3(ptap->AP_loc,"Nonscalable algorithm, AP_loc reallocs %" PetscInt_FMT "; Fill ratio: given %g needed %g.\n",nspacedouble,(double)fill,(double)apfill);CHKERRQ(ierr);
    ierr = PetscInfo1(ptap->AP_loc,"Use MatPtAP(A,B,MatReuse,%g,&C) for best AP_loc performance.;\n",(double)apfill);CHKERRQ(ierr);
  } else {
    ierr = PetscInfo(ptap->AP_loc,"Nonscalable algorithm, AP_loc is empty \n");CHKERRQ(ierr);
  }
#endif

  /* (2-1) compute symbolic Co = Ro*AP_loc  */
  /* ------------------------------------ */
  ierr = MatGetOptionsPrefix(A,&prefix);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(ptap->Ro,prefix);CHKERRQ(ierr);
  ierr = MatAppendOptionsPrefix(ptap->Ro,"inner_offdiag_");CHKERRQ(ierr);
  ierr = MatProductCreate(ptap->Ro,ptap->AP_loc,NULL,&ptap->C_oth);CHKERRQ(ierr);
  ierr = MatGetOptionsPrefix(Cmpi,&prefix);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(ptap->C_oth,prefix);CHKERRQ(ierr);
  ierr = MatAppendOptionsPrefix(ptap->C_oth,"inner_C_oth_");CHKERRQ(ierr);
  ierr = MatProductSetType(ptap->C_oth,MATPRODUCT_AB);CHKERRQ(ierr);
  ierr = MatProductSetAlgorithm(ptap->C_oth,"default");CHKERRQ(ierr);
  ierr = MatProductSetFill(ptap->C_oth,fill);CHKERRQ(ierr);
  ierr = MatProductSetFromOptions(ptap->C_oth);CHKERRQ(ierr);
  ierr = MatProductSymbolic(ptap->C_oth);CHKERRQ(ierr);

  /* (3) send coj of C_oth to other processors  */
  /* ------------------------------------------ */
  /* determine row ownership */
  ierr = PetscLayoutCreate(comm,&rowmap);CHKERRQ(ierr);
  rowmap->n  = pn;
  rowmap->bs = 1;
  ierr   = PetscLayoutSetUp(rowmap);CHKERRQ(ierr);
  owners = rowmap->range;

  /* determine the number of messages to send, their lengths */
  ierr = PetscMalloc4(size,&len_s,size,&len_si,size,&sstatus,size+2,&owners_co);CHKERRQ(ierr);
  ierr = PetscArrayzero(len_s,size);CHKERRQ(ierr);
  ierr = PetscArrayzero(len_si,size);CHKERRQ(ierr);

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
  ierr = PetscGatherNumberOfMessages(comm,NULL,len_s,&nrecv);CHKERRQ(ierr);
  ierr = PetscGatherMessageLengths2(comm,nsend,nrecv,len_s,len_si,&id_r,&len_r,&len_ri);CHKERRQ(ierr);

  /* post the Irecv and Isend of coj */
  ierr = PetscCommGetNewTag(comm,&tagj);CHKERRQ(ierr);
  ierr = PetscPostIrecvInt(comm,tagj,nrecv,id_r,len_r,&buf_rj,&rwaits);CHKERRQ(ierr);
  ierr = PetscMalloc1(nsend+1,&swaits);CHKERRQ(ierr);
  for (proc=0, k=0; proc<size; proc++) {
    if (!len_s[proc]) continue;
    i    = owners_co[proc];
    ierr = MPI_Isend(coj+coi[i],len_s[proc],MPIU_INT,proc,tagj,comm,swaits+k);CHKERRMPI(ierr);
    k++;
  }

  /* (2-2) compute symbolic C_loc = Rd*AP_loc */
  /* ---------------------------------------- */
  ierr = MatSetOptionsPrefix(ptap->Rd,prefix);CHKERRQ(ierr);
  ierr = MatAppendOptionsPrefix(ptap->Rd,"inner_diag_");CHKERRQ(ierr);
  ierr = MatProductCreate(ptap->Rd,ptap->AP_loc,NULL,&ptap->C_loc);CHKERRQ(ierr);
  ierr = MatGetOptionsPrefix(Cmpi,&prefix);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(ptap->C_loc,prefix);CHKERRQ(ierr);
  ierr = MatAppendOptionsPrefix(ptap->C_loc,"inner_C_loc_");CHKERRQ(ierr);
  ierr = MatProductSetType(ptap->C_loc,MATPRODUCT_AB);CHKERRQ(ierr);
  ierr = MatProductSetAlgorithm(ptap->C_loc,"default");CHKERRQ(ierr);
  ierr = MatProductSetFill(ptap->C_loc,fill);CHKERRQ(ierr);
  ierr = MatProductSetFromOptions(ptap->C_loc);CHKERRQ(ierr);
  ierr = MatProductSymbolic(ptap->C_loc);CHKERRQ(ierr);

  c_loc = (Mat_SeqAIJ*)ptap->C_loc->data;

  /* receives coj are complete */
  for (i=0; i<nrecv; i++) {
    ierr = MPI_Waitany(nrecv,rwaits,&icompleted,&rstatus);CHKERRMPI(ierr);
  }
  ierr = PetscFree(rwaits);CHKERRQ(ierr);
  if (nsend) {ierr = MPI_Waitall(nsend,swaits,sstatus);CHKERRMPI(ierr);}

  /* add received column indices into ta to update Crmax */
  for (k=0; k<nrecv; k++) {/* k-th received message */
    Jptr = buf_rj[k];
    for (j=0; j<len_r[k]; j++) {
      ierr = PetscTableAdd(ta,*(Jptr+j)+1,1,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = PetscTableGetCount(ta,&Crmax);CHKERRQ(ierr);
  ierr = PetscTableDestroy(&ta);CHKERRQ(ierr);

  /* (4) send and recv coi */
  /*-----------------------*/
  ierr   = PetscCommGetNewTag(comm,&tagi);CHKERRQ(ierr);
  ierr   = PetscPostIrecvInt(comm,tagi,nrecv,id_r,len_ri,&buf_ri,&rwaits);CHKERRQ(ierr);
  ierr   = PetscMalloc1(len+1,&buf_s);CHKERRQ(ierr);
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
    ierr = MPI_Isend(buf_si,len_si[proc],MPIU_INT,proc,tagi,comm,swaits+k);CHKERRMPI(ierr);
    k++;
    buf_si += len_si[proc];
  }
  for (i=0; i<nrecv; i++) {
    ierr = MPI_Waitany(nrecv,rwaits,&icompleted,&rstatus);CHKERRMPI(ierr);
  }
  ierr = PetscFree(rwaits);CHKERRQ(ierr);
  if (nsend) {ierr = MPI_Waitall(nsend,swaits,sstatus);CHKERRMPI(ierr);}

  ierr = PetscFree4(len_s,len_si,sstatus,owners_co);CHKERRQ(ierr);
  ierr = PetscFree(len_ri);CHKERRQ(ierr);
  ierr = PetscFree(swaits);CHKERRQ(ierr);
  ierr = PetscFree(buf_s);CHKERRQ(ierr);

  /* (5) compute the local portion of Cmpi      */
  /* ------------------------------------------ */
  /* set initial free space to be Crmax, sufficient for holding nozeros in each row of Cmpi */
  ierr          = PetscFreeSpaceGet(Crmax,&free_space);CHKERRQ(ierr);
  current_space = free_space;

  ierr = PetscMalloc3(nrecv,&buf_ri_k,nrecv,&nextrow,nrecv,&nextci);CHKERRQ(ierr);
  for (k=0; k<nrecv; k++) {
    buf_ri_k[k] = buf_ri[k]; /* beginning of k-th recved i-structure */
    nrows       = *buf_ri_k[k];
    nextrow[k]  = buf_ri_k[k] + 1;  /* next row number of k-th recved i-structure */
    nextci[k]   = buf_ri_k[k] + (nrows + 1); /* points to the next i-structure of k-th recved i-structure  */
  }

  ierr = MatPreallocateInitialize(comm,pn,pn,dnz,onz);CHKERRQ(ierr);
  ierr = PetscLLCondensedCreate(Crmax,pN,&lnk,&lnkbt);CHKERRQ(ierr);
  for (i=0; i<pn; i++) {
    /* add C_loc into Cmpi */
    nzi  = c_loc->i[i+1] - c_loc->i[i];
    Jptr = c_loc->j + c_loc->i[i];
    ierr = PetscLLCondensedAddSorted(nzi,Jptr,lnk,lnkbt);CHKERRQ(ierr);

    /* add received col data into lnk */
    for (k=0; k<nrecv; k++) { /* k-th received message */
      if (i == *nextrow[k]) { /* i-th row */
        nzi  = *(nextci[k]+1) - *nextci[k];
        Jptr = buf_rj[k] + *nextci[k];
        ierr = PetscLLCondensedAddSorted(nzi,Jptr,lnk,lnkbt);CHKERRQ(ierr);
        nextrow[k]++; nextci[k]++;
      }
    }
    nzi = lnk[0];

    /* copy data into free space, then initialize lnk */
    ierr = PetscLLCondensedClean(pN,nzi,current_space->array,lnk,lnkbt);CHKERRQ(ierr);
    ierr = MatPreallocateSet(i+owners[rank],nzi,current_space->array,dnz,onz);CHKERRQ(ierr);
  }
  ierr = PetscFree3(buf_ri_k,nextrow,nextci);CHKERRQ(ierr);
  ierr = PetscLLDestroy(lnk,lnkbt);CHKERRQ(ierr);
  ierr = PetscFreeSpaceDestroy(free_space);CHKERRQ(ierr);

  /* local sizes and preallocation */
  ierr = MatSetSizes(Cmpi,pn,pn,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  if (P->cmap->bs > 0) {
    ierr = PetscLayoutSetBlockSize(Cmpi->rmap,P->cmap->bs);CHKERRQ(ierr);
    ierr = PetscLayoutSetBlockSize(Cmpi->cmap,P->cmap->bs);CHKERRQ(ierr);
  }
  ierr = MatMPIAIJSetPreallocation(Cmpi,0,dnz,0,onz);CHKERRQ(ierr);
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);

  /* members in merge */
  ierr = PetscFree(id_r);CHKERRQ(ierr);
  ierr = PetscFree(len_r);CHKERRQ(ierr);
  ierr = PetscFree(buf_ri[0]);CHKERRQ(ierr);
  ierr = PetscFree(buf_ri);CHKERRQ(ierr);
  ierr = PetscFree(buf_rj[0]);CHKERRQ(ierr);
  ierr = PetscFree(buf_rj);CHKERRQ(ierr);
  ierr = PetscLayoutDestroy(&rowmap);CHKERRQ(ierr);

  ierr = PetscCalloc1(pN,&ptap->apa);CHKERRQ(ierr);

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
  PetscErrorCode    ierr;
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
  if (!ptap) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_ARG_WRONGSTATE,"PtAP cannot be computed. Missing data");
  if (!ptap->AP_loc) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_ARG_WRONGSTATE,"PtAP cannot be reused. Do not call MatProductClear()");

  ierr = MatZeroEntries(C);CHKERRQ(ierr);
  /* 1) get R = Pd^T,Ro = Po^T */
  if (ptap->reuse == MAT_REUSE_MATRIX) {
    ierr = MatTranspose(p->A,MAT_REUSE_MATRIX,&ptap->Rd);CHKERRQ(ierr);
    ierr = MatTranspose(p->B,MAT_REUSE_MATRIX,&ptap->Ro);CHKERRQ(ierr);
  }

  /* 2) get AP_loc */
  AP_loc = ptap->AP_loc;
  ap = (Mat_SeqAIJ*)AP_loc->data;

  /* 2-1) get P_oth = ptap->P_oth  and P_loc = ptap->P_loc */
  /*-----------------------------------------------------*/
  if (ptap->reuse == MAT_REUSE_MATRIX) {
    /* P_oth and P_loc are obtained in MatPtASymbolic() when reuse == MAT_INITIAL_MATRIX */
    ierr = MatGetBrowsOfAoCols_MPIAIJ(A,P,MAT_REUSE_MATRIX,&ptap->startsj_s,&ptap->startsj_r,&ptap->bufa,&ptap->P_oth);CHKERRQ(ierr);
    ierr = MatMPIAIJGetLocalMat(P,MAT_REUSE_MATRIX,&ptap->P_loc);CHKERRQ(ierr);
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
  ierr = PetscObjectStateIncrease((PetscObject)AP_loc);CHKERRQ(ierr);

  /* 3) C_loc = Rd*AP_loc, C_oth = Ro*AP_loc */
  ierr = MatProductNumeric(ptap->C_loc);CHKERRQ(ierr);
  ierr = MatProductNumeric(ptap->C_oth);CHKERRQ(ierr);
  C_loc = ptap->C_loc;
  C_oth = ptap->C_oth;

  /* add C_loc and Co to to C */
  ierr = MatGetOwnershipRange(C,&rstart,&rend);CHKERRQ(ierr);

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
      ierr = MatSetValues_MPIAIJ(C,1,&row,ncols,cols,vals,ADD_VALUES);CHKERRQ(ierr);
      cols += ncols; vals += ncols;
    }
  } else {
    ierr = MatSetValues_MPIAIJ_CopyFromCSRFormat(C,c_seq->j,c_seq->i,c_seq->a);CHKERRQ(ierr);
  }

  /* Co -> C, off-processor part */
  cm = C_oth->rmap->N;
  c_seq = (Mat_SeqAIJ*)C_oth->data;
  cols = c_seq->j;
  vals = c_seq->a;
  for (i=0; i<cm; i++) {
    ncols = c_seq->i[i+1] - c_seq->i[i];
    row = p->garray[i];
    ierr = MatSetValues(C,1,&row,ncols,cols,vals,ADD_VALUES);CHKERRQ(ierr);
    cols += ncols; vals += ncols;
  }

  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ptap->reuse = MAT_REUSE_MATRIX;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatProductSymbolic_PtAP_MPIAIJ_MPIAIJ(Mat C)
{
  PetscErrorCode      ierr;
  Mat_Product         *product = C->product;
  Mat                 A=product->A,P=product->B;
  MatProductAlgorithm alg=product->alg;
  PetscReal           fill=product->fill;
  PetscBool           flg;

  PetscFunctionBegin;
  /* scalable: do R=P^T locally, then C=R*A*P */
  ierr = PetscStrcmp(alg,"scalable",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatPtAPSymbolic_MPIAIJ_MPIAIJ_scalable(A,P,product->fill,C);CHKERRQ(ierr);
    C->ops->productnumeric = MatProductNumeric_PtAP;
    goto next;
  }

  /* nonscalable: do R=P^T locally, then C=R*A*P */
  ierr = PetscStrcmp(alg,"nonscalable",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatPtAPSymbolic_MPIAIJ_MPIAIJ(A,P,fill,C);CHKERRQ(ierr);
    goto next;
  }

  /* allatonce */
  ierr = PetscStrcmp(alg,"allatonce",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatPtAPSymbolic_MPIAIJ_MPIAIJ_allatonce(A,P,fill,C);CHKERRQ(ierr);
    goto next;
  }

  /* allatonce_merged */
  ierr = PetscStrcmp(alg,"allatonce_merged",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatPtAPSymbolic_MPIAIJ_MPIAIJ_allatonce_merged(A,P,fill,C);CHKERRQ(ierr);
    goto next;
  }

  /* backend general code */
  ierr = PetscStrcmp(alg,"backend",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatProductSymbolic_MPIAIJBACKEND(C);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  /* hypre */
#if defined(PETSC_HAVE_HYPRE)
  ierr = PetscStrcmp(alg,"hypre",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatPtAPSymbolic_AIJ_AIJ_wHYPRE(A,P,fill,C);CHKERRQ(ierr);
    C->ops->productnumeric = MatProductNumeric_PtAP;
    PetscFunctionReturn(0);
  }
#endif
  SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_SUP,"Mat Product Algorithm is not supported");

next:
  C->ops->productnumeric = MatProductNumeric_PtAP;
  PetscFunctionReturn(0);
}
