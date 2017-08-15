
/*
  Defines projective product routines where A is a MPIAIJ matrix
          C = P^T * A * P
*/

#include <../src/mat/impls/aij/seq/aij.h>   /*I "petscmat.h" I*/
#include <../src/mat/utils/freespace.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <petscbt.h>
#include <petsctime.h>

/* #define PTAP_PROFILE */

PetscErrorCode MatView_MPIAIJ_PtAP(Mat A,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  Mat_MPIAIJ        *a=(Mat_MPIAIJ*)A->data;
  Mat_PtAPMPI       *ptap=a->ptap;
  PetscBool         iascii;
  PetscViewerFormat format;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
      if (ptap->algType == 0) {
        ierr = PetscViewerASCIIPrintf(viewer,"using scalable MatPtAP() implementation\n");CHKERRQ(ierr);
      } else if (ptap->algType == 1) {
        ierr = PetscViewerASCIIPrintf(viewer,"using nonscalable MatPtAP() implementation\n");CHKERRQ(ierr);
      }
    }
  }
  ierr = (ptap->view)(A,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_MPIAIJ_PtAP(Mat A)
{
  PetscErrorCode ierr;
  Mat_MPIAIJ     *a=(Mat_MPIAIJ*)A->data;
  Mat_PtAPMPI    *ptap=a->ptap;

  PetscFunctionBegin;
  if (ptap) {
    Mat_Merge_SeqsToMPI *merge=ptap->merge;
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
    
    ierr = ptap->destroy(A);CHKERRQ(ierr);
    ierr = PetscFree(ptap);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatDuplicate_MPIAIJ_MatPtAP(Mat A, MatDuplicateOption op, Mat *M)
{
  PetscErrorCode ierr;
  Mat_MPIAIJ     *a     = (Mat_MPIAIJ*)A->data;
  Mat_PtAPMPI    *ptap  = a->ptap;

  PetscFunctionBegin;
  ierr = (*ptap->duplicate)(A,op,M);CHKERRQ(ierr);
  (*M)->ops->destroy   = ptap->destroy;
  (*M)->ops->duplicate = ptap->duplicate;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatPtAP_MPIAIJ_MPIAIJ(Mat A,Mat P,MatReuse scall,PetscReal fill,Mat *C)
{
  PetscErrorCode ierr;
  PetscBool      rap=PETSC_TRUE; /* do R=P^T locally, then C=R*A*P */
  MPI_Comm       comm;

  PetscFunctionBegin;
  /* check if matrix local sizes are compatible */
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  if (A->rmap->rstart != P->rmap->rstart || A->rmap->rend != P->rmap->rend) SETERRQ4(comm,PETSC_ERR_ARG_SIZ,"Matrix local dimensions are incompatible, Arow (%D, %D) != Prow (%D,%D)",A->rmap->rstart,A->rmap->rend,P->rmap->rstart,P->rmap->rend);
  if (A->cmap->rstart != P->rmap->rstart || A->cmap->rend != P->rmap->rend) SETERRQ4(comm,PETSC_ERR_ARG_SIZ,"Matrix local dimensions are incompatible, Acol (%D, %D) != Prow (%D,%D)",A->cmap->rstart,A->cmap->rend,P->rmap->rstart,P->rmap->rend);

  ierr = PetscOptionsGetBool(NULL,NULL,"-matrap",&rap,NULL);CHKERRQ(ierr);
  if (scall == MAT_INITIAL_MATRIX) {
    ierr = PetscLogEventBegin(MAT_PtAPSymbolic,A,P,0,0);CHKERRQ(ierr);
    if (rap) { /* do R=P^T locally, then C=R*A*P */
      ierr = MatPtAPSymbolic_MPIAIJ_MPIAIJ(A,P,fill,C);CHKERRQ(ierr);
    } else {       /* do P^T*A*P */
      ierr = MatPtAPSymbolic_MPIAIJ_MPIAIJ_ptap(A,P,fill,C);CHKERRQ(ierr);
    }
    ierr = PetscLogEventEnd(MAT_PtAPSymbolic,A,P,0,0);CHKERRQ(ierr);
  }
  ierr = PetscLogEventBegin(MAT_PtAPNumeric,A,P,0,0);CHKERRQ(ierr);
  ierr = (*(*C)->ops->ptapnumeric)(A,P,*C);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_PtAPNumeric,A,P,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_HYPRE)
PETSC_INTERN PetscErrorCode MatPtAPSymbolic_AIJ_AIJ_wHYPRE(Mat,Mat,PetscReal,Mat*);
#endif

PetscErrorCode MatPtAPNumeric_MPIAIJ_MPIAIJ_scalable(Mat A,Mat P,Mat C)
{
  PetscErrorCode    ierr;
  Mat_MPIAIJ        *a=(Mat_MPIAIJ*)A->data,*p=(Mat_MPIAIJ*)P->data,*c=(Mat_MPIAIJ*)C->data;
  Mat_SeqAIJ        *ad=(Mat_SeqAIJ*)(a->A)->data,*ao=(Mat_SeqAIJ*)(a->B)->data;
  Mat_SeqAIJ        *ap,*p_loc,*p_oth,*c_seq;
  Mat_PtAPMPI       *ptap = c->ptap;
  Mat               AP_loc,C_loc,C_oth;
  PetscInt          i,rstart,rend,cm,ncols,row,*api,*apj,am = A->rmap->n,apnz;
  PetscScalar       *apa;
  const PetscInt    *cols;
  const PetscScalar *vals;

  PetscFunctionBegin;
  ierr = MatZeroEntries(C);CHKERRQ(ierr);

  /* 1) get R = Pd^T,Ro = Po^T */
  if (ptap->reuse == MAT_REUSE_MATRIX) {
    ierr = MatTranspose_SeqAIJ(p->A,MAT_REUSE_MATRIX,&ptap->Rd);CHKERRQ(ierr);
    ierr = MatTranspose_SeqAIJ(p->B,MAT_REUSE_MATRIX,&ptap->Ro);CHKERRQ(ierr);
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
  else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"ptap->P_oth is NULL. Cannot proceed!");

  api   = ap->i;
  apj   = ap->j;
  for (i=0; i<am; i++) {
    /* AP[i,:] = A[i,:]*P = Ad*P_loc Ao*P_oth */
    apnz = api[i+1] - api[i];
    apa = ap->a + api[i];
    ierr = PetscMemzero(apa,sizeof(PetscScalar)*apnz);CHKERRQ(ierr);
    AProw_scalable(i,ad,ao,p_loc,p_oth,api,apj,apa);
    ierr = PetscLogFlops(2.0*apnz);CHKERRQ(ierr);
  }

  /* 3) C_loc = Rd*AP_loc, C_oth = Ro*AP_loc */
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
  for (i=0; i<cm; i++) {
    ncols = c_seq->i[i+1] - c_seq->i[i];
    row = rstart + i;
    ierr = MatSetValues(C,1,&row,ncols,cols,vals,ADD_VALUES);CHKERRQ(ierr);
    cols += ncols; vals += ncols;
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

PetscErrorCode MatPtAPSymbolic_MPIAIJ_MPIAIJ_scalable(Mat A,Mat P,PetscReal fill,Mat *C)
{
  PetscErrorCode      ierr;
  Mat_PtAPMPI         *ptap;
  Mat_MPIAIJ          *a=(Mat_MPIAIJ*)A->data,*p=(Mat_MPIAIJ*)P->data,*c;
  MPI_Comm            comm;
  PetscMPIInt         size,rank;
  Mat                 Cmpi,P_loc,P_oth;
  PetscFreeSpaceList  free_space=NULL,current_space=NULL;
  PetscInt            am=A->rmap->n,pm=P->rmap->n,pN=P->cmap->N,pn=P->cmap->n;
  PetscInt            *lnk,i,k,pnz,row,nsend;
  PetscMPIInt         tagi,tagj,*len_si,*len_s,*len_ri,icompleted=0,nrecv;
  PetscInt            **buf_rj,**buf_ri,**buf_ri_k;
  PetscInt            len,proc,*dnz,*onz,*owners,nzi,nspacedouble;
  PetscInt            nrows,*buf_s,*buf_si,*buf_si_i,**nextrow,**nextci;
  MPI_Request         *swaits,*rwaits;
  MPI_Status          *sstatus,rstatus;
  PetscLayout         rowmap;
  PetscInt            *owners_co,*coi,*coj;    /* i and j array of (p->B)^T*A*P - used in the communication */
  PetscMPIInt         *len_r,*id_r;    /* array of length of comm->size, store send/recv matrix values */
  PetscInt            *api,*apj,*Jptr,apnz,*prmap=p->garray,con,j,Crmax,*aj,*ai,*pi;
  Mat_SeqAIJ          *p_loc,*p_oth=NULL,*ad=(Mat_SeqAIJ*)(a->A)->data,*ao=NULL,*c_loc,*c_oth;
  PetscScalar         *apv;
  PetscTable          ta;
#if defined(PETSC_USE_INFO)
  PetscReal           apfill;
#endif

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  if (size > 1) ao = (Mat_SeqAIJ*)(a->B)->data;

  /* create symbolic parallel matrix Cmpi */
  ierr = MatCreate(comm,&Cmpi);CHKERRQ(ierr);
  ierr = MatSetType(Cmpi,MATMPIAIJ);CHKERRQ(ierr);

  /* create struct Mat_PtAPMPI and attached it to C later */
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
  ierr = MatTranspose_SeqAIJ(p->A,MAT_INITIAL_MATRIX,&ptap->Rd);CHKERRQ(ierr);
  ierr = MatTranspose_SeqAIJ(p->B,MAT_INITIAL_MATRIX,&ptap->Ro);CHKERRQ(ierr);

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
  ierr = PetscMalloc2(api[am],&apj,api[am],&apv);CHKERRQ(ierr);
  ierr = PetscFreeSpaceContiguous(&free_space,apj);CHKERRQ(ierr);
  ierr = PetscLLCondensedDestroy_Scalable(lnk);CHKERRQ(ierr);

  /* Create AP_loc for reuse */
  ierr = MatCreateSeqAIJWithArrays(PETSC_COMM_SELF,am,pN,api,apj,apv,&ptap->AP_loc);CHKERRQ(ierr);

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
    ierr = PetscInfo3(ptap->AP_loc,"Scalable algorithm, AP_loc reallocs %D; Fill ratio: given %g needed %g.\n",nspacedouble,(double)fill,(double)apfill);CHKERRQ(ierr);
    ierr = PetscInfo1(ptap->AP_loc,"Use MatPtAP(A,B,MatReuse,%g,&C) for best AP_loc performance.;\n",(double)apfill);CHKERRQ(ierr);
  } else {
    ierr = PetscInfo(ptap->AP_loc,"Scalable algorithm, AP_loc is empty \n");CHKERRQ(ierr);
  }
#endif

  /* (2-1) compute symbolic Co = Ro*AP_loc  */
  /* ------------------------------------ */
  ierr = MatMatMultSymbolic_SeqAIJ_SeqAIJ(ptap->Ro,ptap->AP_loc,fill,&ptap->C_oth);CHKERRQ(ierr);

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
  ierr = PetscMemzero(len_s,size*sizeof(PetscMPIInt));CHKERRQ(ierr);
  ierr = PetscMemzero(len_si,size*sizeof(PetscMPIInt));CHKERRQ(ierr);

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
    ierr = MPI_Isend(coj+coi[i],len_s[proc],MPIU_INT,proc,tagj,comm,swaits+k);CHKERRQ(ierr);
    k++;
  }

  /* (2-2) compute symbolic C_loc = Rd*AP_loc */
  /* ---------------------------------------- */
  ierr = MatMatMultSymbolic_SeqAIJ_SeqAIJ(ptap->Rd,ptap->AP_loc,fill,&ptap->C_loc);CHKERRQ(ierr);
  c_loc = (Mat_SeqAIJ*)ptap->C_loc->data;

  /* receives coj are complete */
  for (i=0; i<nrecv; i++) {
    ierr = MPI_Waitany(nrecv,rwaits,&icompleted,&rstatus);CHKERRQ(ierr);
  }
  ierr = PetscFree(rwaits);CHKERRQ(ierr);
  if (nsend) {ierr = MPI_Waitall(nsend,swaits,sstatus);CHKERRQ(ierr);}

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
    ierr = MPI_Isend(buf_si,len_si[proc],MPIU_INT,proc,tagi,comm,swaits+k);CHKERRQ(ierr);
    k++;
    buf_si += len_si[proc];
  }
  for (i=0; i<nrecv; i++) {
    ierr = MPI_Waitany(nrecv,rwaits,&icompleted,&rstatus);CHKERRQ(ierr);
  }
  ierr = PetscFree(rwaits);CHKERRQ(ierr);
  if (nsend) {ierr = MPI_Waitall(nsend,swaits,sstatus);CHKERRQ(ierr);}

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
    nextci[k]   = buf_ri_k[k] + (nrows + 1); /* poins to the next i-structure of k-th recved i-structure  */
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
  ierr = MatSetBlockSizes(Cmpi,PetscAbs(P->cmap->bs),PetscAbs(P->cmap->bs));CHKERRQ(ierr);
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

  /* attach the supporting struct to Cmpi for reuse */
  c = (Mat_MPIAIJ*)Cmpi->data;
  c->ptap         = ptap;
  ptap->duplicate = Cmpi->ops->duplicate;
  ptap->destroy   = Cmpi->ops->destroy;
  ptap->view      = Cmpi->ops->view;

  /* Cmpi is not ready for use - assembly will be done by MatPtAPNumeric() */
  Cmpi->assembled        = PETSC_FALSE;
  Cmpi->ops->destroy     = MatDestroy_MPIAIJ_PtAP;
  Cmpi->ops->duplicate   = MatDuplicate_MPIAIJ_MatPtAP;
  Cmpi->ops->view        = MatView_MPIAIJ_PtAP;
  *C                     = Cmpi;
  PetscFunctionReturn(0);
}

PetscErrorCode MatPtAPSymbolic_MPIAIJ_MPIAIJ(Mat A,Mat P,PetscReal fill,Mat *C)
{
  PetscErrorCode      ierr;
  Mat_PtAPMPI         *ptap;
  Mat_MPIAIJ          *a=(Mat_MPIAIJ*)A->data,*p=(Mat_MPIAIJ*)P->data,*c;
  MPI_Comm            comm;
  PetscMPIInt         size,rank;
  Mat                 Cmpi;
  PetscFreeSpaceList  free_space=NULL,current_space=NULL;
  PetscInt            am=A->rmap->n,pm=P->rmap->n,pN=P->cmap->N,pn=P->cmap->n;
  PetscInt            *lnk,i,k,pnz,row,nsend;
  PetscBT             lnkbt;
  PetscMPIInt         tagi,tagj,*len_si,*len_s,*len_ri,icompleted=0,nrecv;
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
#if defined(PETSC_HAVE_HYPRE)
  const char          *algTypes[3] = {"scalable","nonscalable","hypre"};
  PetscInt            nalg = 3;
#else
  const char          *algTypes[2] = {"scalable","nonscalable"};
  PetscInt            nalg = 2;
#endif
  PetscInt            alg = 1; /* set default algorithm */
#if defined(PETSC_USE_INFO)
  PetscReal           apfill;
#endif
#if defined(PTAP_PROFILE)
  PetscLogDouble      t0,t1,t11,t12,t2,t3,t4;
#endif
  PetscBool           flg;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);

  /* pick an algorithm */
  ierr = PetscObjectOptionsBegin((PetscObject)A);CHKERRQ(ierr);
  PetscOptionsObject->alreadyprinted = PETSC_FALSE; /* a hack to ensure the option shows in '-help' */
  ierr = PetscOptionsEList("-matptap_via","Algorithmic approach","MatPtAP",algTypes,nalg,algTypes[1],&alg,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  if (!flg && pN > 100000) { /* may switch to scalable algorithm as default */
    MatInfo     Ainfo,Pinfo;
    PetscInt    nz_local;
    PetscBool   alg_scalable_loc=PETSC_FALSE,alg_scalable;

    ierr = MatGetInfo(A,MAT_LOCAL,&Ainfo);CHKERRQ(ierr);
    ierr = MatGetInfo(P,MAT_LOCAL,&Pinfo);CHKERRQ(ierr);
    nz_local = (PetscInt)(Ainfo.nz_allocated + Pinfo.nz_allocated);

    if (pN > fill*nz_local) alg_scalable_loc = PETSC_TRUE;
    ierr = MPIU_Allreduce(&alg_scalable_loc,&alg_scalable,1,MPIU_BOOL,MPI_LOR,comm);CHKERRQ(ierr);

    if (alg_scalable) {
      alg  = 0; /* scalable algorithm would 50% slower than nonscalable algorithm */
    }
  }

  if (alg == 0) {
    ierr = MatPtAPSymbolic_MPIAIJ_MPIAIJ_scalable(A,P,fill,C);CHKERRQ(ierr);
    (*C)->ops->ptapnumeric = MatPtAPNumeric_MPIAIJ_MPIAIJ_scalable;
    PetscFunctionReturn(0);

#if defined(PETSC_HAVE_HYPRE)
  } else if (alg == 2) {
    /* Use boomerAMGBuildCoarseOperator */
    ierr = MatPtAPSymbolic_AIJ_AIJ_wHYPRE(A,P,fill,C);CHKERRQ(ierr);
    PetscFunctionReturn(0);
#endif
  }

#if defined(PTAP_PROFILE)
  ierr = PetscTime(&t0);CHKERRQ(ierr);
#endif

  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  if (size > 1) ao = (Mat_SeqAIJ*)(a->B)->data;

  /* create symbolic parallel matrix Cmpi */
  ierr = MatCreate(comm,&Cmpi);CHKERRQ(ierr);
  ierr = MatSetType(Cmpi,MATMPIAIJ);CHKERRQ(ierr);

  /* Do dense axpy in MatPtAPNumeric_MPIAIJ_MPIAIJ() */
  Cmpi->ops->ptapnumeric = MatPtAPNumeric_MPIAIJ_MPIAIJ;

  /* create struct Mat_PtAPMPI and attached it to C later */
  ierr        = PetscNew(&ptap);CHKERRQ(ierr);
  ptap->reuse = MAT_INITIAL_MATRIX;
  ptap->algType = alg;

  /* get P_oth by taking rows of P (= non-zero cols of local A) from other processors */
  ierr = MatGetBrowsOfAoCols_MPIAIJ(A,P,MAT_INITIAL_MATRIX,&ptap->startsj_s,&ptap->startsj_r,&ptap->bufa,&ptap->P_oth);CHKERRQ(ierr);
  /* get P_loc by taking all local rows of P */
  ierr = MatMPIAIJGetLocalMat(P,MAT_INITIAL_MATRIX,&ptap->P_loc);CHKERRQ(ierr);

  /* (0) compute Rd = Pd^T, Ro = Po^T  */
  /* --------------------------------- */
  ierr = MatTranspose_SeqAIJ(p->A,MAT_INITIAL_MATRIX,&ptap->Rd);CHKERRQ(ierr);
  ierr = MatTranspose_SeqAIJ(p->B,MAT_INITIAL_MATRIX,&ptap->Ro);CHKERRQ(ierr);
#if defined(PTAP_PROFILE)
  ierr = PetscTime(&t11);CHKERRQ(ierr);
#endif

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
  ierr   = PetscMalloc2(api[am],&apj,api[am],&apv);CHKERRQ(ierr);
  ierr   = PetscFreeSpaceContiguous(&free_space,apj);CHKERRQ(ierr);
  ierr = PetscLLDestroy(lnk,lnkbt);CHKERRQ(ierr);

  /* Create AP_loc for reuse */
  ierr = MatCreateSeqAIJWithArrays(PETSC_COMM_SELF,am,pN,api,apj,apv,&ptap->AP_loc);CHKERRQ(ierr);

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
    ierr = PetscInfo3(ptap->AP_loc,"Nonscalable algorithm, AP_loc reallocs %D; Fill ratio: given %g needed %g.\n",nspacedouble,(double)fill,(double)apfill);CHKERRQ(ierr);
    ierr = PetscInfo1(ptap->AP_loc,"Use MatPtAP(A,B,MatReuse,%g,&C) for best AP_loc performance.;\n",(double)apfill);CHKERRQ(ierr);
  } else {
    ierr = PetscInfo(ptap->AP_loc,"Nonscalable algorithm, AP_loc is empty \n");CHKERRQ(ierr);
  }
#endif

#if defined(PTAP_PROFILE)
  ierr = PetscTime(&t12);CHKERRQ(ierr);
#endif

  /* (2-1) compute symbolic Co = Ro*AP_loc  */
  /* ------------------------------------ */
  ierr = MatMatMultSymbolic_SeqAIJ_SeqAIJ(ptap->Ro,ptap->AP_loc,fill,&ptap->C_oth);CHKERRQ(ierr);
#if defined(PTAP_PROFILE)
  ierr = PetscTime(&t1);CHKERRQ(ierr);
#endif

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
  ierr = PetscMemzero(len_s,size*sizeof(PetscMPIInt));CHKERRQ(ierr);
  ierr = PetscMemzero(len_si,size*sizeof(PetscMPIInt));CHKERRQ(ierr);

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
    ierr = MPI_Isend(coj+coi[i],len_s[proc],MPIU_INT,proc,tagj,comm,swaits+k);CHKERRQ(ierr);
    k++;
  }

  /* (2-2) compute symbolic C_loc = Rd*AP_loc */
  /* ---------------------------------------- */
  ierr = MatMatMultSymbolic_SeqAIJ_SeqAIJ(ptap->Rd,ptap->AP_loc,fill,&ptap->C_loc);CHKERRQ(ierr);
  c_loc = (Mat_SeqAIJ*)ptap->C_loc->data;

  /* receives coj are complete */
  for (i=0; i<nrecv; i++) {
    ierr = MPI_Waitany(nrecv,rwaits,&icompleted,&rstatus);CHKERRQ(ierr);
  }
  ierr = PetscFree(rwaits);CHKERRQ(ierr);
  if (nsend) {ierr = MPI_Waitall(nsend,swaits,sstatus);CHKERRQ(ierr);}

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
    ierr = MPI_Isend(buf_si,len_si[proc],MPIU_INT,proc,tagi,comm,swaits+k);CHKERRQ(ierr);
    k++;
    buf_si += len_si[proc];
  }
  for (i=0; i<nrecv; i++) {
    ierr = MPI_Waitany(nrecv,rwaits,&icompleted,&rstatus);CHKERRQ(ierr);
  }
  ierr = PetscFree(rwaits);CHKERRQ(ierr);
  if (nsend) {ierr = MPI_Waitall(nsend,swaits,sstatus);CHKERRQ(ierr);}

  ierr = PetscFree4(len_s,len_si,sstatus,owners_co);CHKERRQ(ierr);
  ierr = PetscFree(len_ri);CHKERRQ(ierr);
  ierr = PetscFree(swaits);CHKERRQ(ierr);
  ierr = PetscFree(buf_s);CHKERRQ(ierr);
#if defined(PTAP_PROFILE)
  ierr = PetscTime(&t2);CHKERRQ(ierr);
#endif
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
    nextci[k]   = buf_ri_k[k] + (nrows + 1); /* poins to the next i-structure of k-th recved i-structure  */
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
#if defined(PTAP_PROFILE)
  ierr = PetscTime(&t3);CHKERRQ(ierr);
#endif

  /* local sizes and preallocation */
  ierr = MatSetSizes(Cmpi,pn,pn,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetBlockSizes(Cmpi,PetscAbs(P->cmap->bs),PetscAbs(P->cmap->bs));CHKERRQ(ierr);
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

  /* attach the supporting struct to Cmpi for reuse */
  c = (Mat_MPIAIJ*)Cmpi->data;
  c->ptap         = ptap;
  ptap->duplicate = Cmpi->ops->duplicate;
  ptap->destroy   = Cmpi->ops->destroy;
  ptap->view      = Cmpi->ops->view;

  if (alg == 1) {
    ierr = PetscCalloc1(pN,&ptap->apa);CHKERRQ(ierr);
  }

  /* Cmpi is not ready for use - assembly will be done by MatPtAPNumeric() */
  Cmpi->assembled        = PETSC_FALSE;
  Cmpi->ops->destroy     = MatDestroy_MPIAIJ_PtAP;
  Cmpi->ops->duplicate   = MatDuplicate_MPIAIJ_MatPtAP;
  Cmpi->ops->view        = MatView_MPIAIJ_PtAP;
  *C                     = Cmpi;

#if defined(PTAP_PROFILE)
  ierr = PetscTime(&t4);CHKERRQ(ierr);
  if (rank == 1) {
    printf("PtAPSym: %g + %g + %g + %g + %g + %g = %g\n",t11-t0,t1-t11,t12-t11,t2-t2,t3-t2,t4-t3,t4-t0);
  }
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode MatPtAPNumeric_MPIAIJ_MPIAIJ(Mat A,Mat P,Mat C)
{
  PetscErrorCode    ierr;
  Mat_MPIAIJ        *a=(Mat_MPIAIJ*)A->data,*p=(Mat_MPIAIJ*)P->data,*c=(Mat_MPIAIJ*)C->data;
  Mat_SeqAIJ        *ad=(Mat_SeqAIJ*)(a->A)->data,*ao=(Mat_SeqAIJ*)(a->B)->data;
  Mat_SeqAIJ        *ap,*p_loc,*p_oth=NULL,*c_seq;
  Mat_PtAPMPI       *ptap = c->ptap;
  Mat               AP_loc,C_loc,C_oth;
  PetscInt          i,rstart,rend,cm,ncols,row;
  PetscInt          *api,*apj,am = A->rmap->n,j,col,apnz;
  PetscScalar       *apa;
  const PetscInt    *cols;
  const PetscScalar *vals;
#if defined(PTAP_PROFILE)
  PetscMPIInt       rank;
  MPI_Comm          comm;
  PetscLogDouble    t0,t1,t2,t3,t4,eR,eAP,eCseq,eCmpi;
#endif

  PetscFunctionBegin;
  ierr = MatZeroEntries(C);CHKERRQ(ierr);

  /* 1) get R = Pd^T,Ro = Po^T */
#if defined(PTAP_PROFILE)
  ierr = PetscTime(&t0);CHKERRQ(ierr);
#endif
  if (ptap->reuse == MAT_REUSE_MATRIX) {
    ierr = MatTranspose_SeqAIJ(p->A,MAT_REUSE_MATRIX,&ptap->Rd);CHKERRQ(ierr);
    ierr = MatTranspose_SeqAIJ(p->B,MAT_REUSE_MATRIX,&ptap->Ro);CHKERRQ(ierr);
  }
#if defined(PTAP_PROFILE)
  ierr = PetscTime(&t1);CHKERRQ(ierr);
  eR = t1 - t0;
#endif

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
    ierr = PetscLogFlops(2.0*apnz);CHKERRQ(ierr);
  }
#if defined(PTAP_PROFILE)
  ierr = PetscTime(&t2);CHKERRQ(ierr);
  eAP = t2 - t1;
#endif

  /* 3) C_loc = Rd*AP_loc, C_oth = Ro*AP_loc */
  ierr = ((ptap->C_loc)->ops->matmultnumeric)(ptap->Rd,AP_loc,ptap->C_loc);CHKERRQ(ierr);
  ierr = ((ptap->C_oth)->ops->matmultnumeric)(ptap->Ro,AP_loc,ptap->C_oth);CHKERRQ(ierr);
  C_loc = ptap->C_loc;
  C_oth = ptap->C_oth;
#if defined(PTAP_PROFILE)
  ierr = PetscTime(&t3);CHKERRQ(ierr);
  eCseq = t3 - t2;
#endif

  /* add C_loc and Co to to C */
  ierr = MatGetOwnershipRange(C,&rstart,&rend);CHKERRQ(ierr);

  /* C_loc -> C */
  cm    = C_loc->rmap->N;
  c_seq = (Mat_SeqAIJ*)C_loc->data;
  cols = c_seq->j;
  vals = c_seq->a;
  for (i=0; i<cm; i++) {
    ncols = c_seq->i[i+1] - c_seq->i[i];
    row = rstart + i;
    ierr = MatSetValues(C,1,&row,ncols,cols,vals,ADD_VALUES);CHKERRQ(ierr);
    cols += ncols; vals += ncols;
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
#if defined(PTAP_PROFILE)
  ierr = PetscTime(&t4);CHKERRQ(ierr);
  eCmpi = t4 - t3;

  ierr = PetscObjectGetComm((PetscObject)C,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (rank==1) {
    ierr = PetscPrintf(MPI_COMM_SELF," R %g, AP %g, Cseq %g, Cmpi %g = %g\n", eR,eAP,eCseq,eCmpi,eR+eAP+eCseq+eCmpi);CHKERRQ(ierr);
  }
#endif
  ptap->reuse = MAT_REUSE_MATRIX;
  PetscFunctionReturn(0);
}

PetscErrorCode MatPtAPSymbolic_MPIAIJ_MPIAIJ_ptap(Mat A,Mat P,PetscReal fill,Mat *C)
{
  PetscErrorCode      ierr;
  Mat                 Cmpi;
  Mat_PtAPMPI         *ptap;
  PetscFreeSpaceList  free_space=NULL,current_space=NULL;
  Mat_MPIAIJ          *a        =(Mat_MPIAIJ*)A->data,*p=(Mat_MPIAIJ*)P->data,*c;
  Mat_SeqAIJ          *ad       =(Mat_SeqAIJ*)(a->A)->data,*ao=(Mat_SeqAIJ*)(a->B)->data;
  Mat_SeqAIJ          *p_loc,*p_oth;
  PetscInt            *pi_loc,*pj_loc,*pi_oth,*pj_oth,*pdti,*pdtj,*poti,*potj,*ptJ;
  PetscInt            *adi=ad->i,*aj,*aoi=ao->i,nnz;
  PetscInt            *lnk,*owners_co,*coi,*coj,i,k,pnz,row;
  PetscInt            am=A->rmap->n,pN=P->cmap->N,pm=P->rmap->n,pn=P->cmap->n;
  PetscBT             lnkbt;
  MPI_Comm            comm;
  PetscMPIInt         size,rank,tagi,tagj,*len_si,*len_s,*len_ri,icompleted=0;
  PetscInt            **buf_rj,**buf_ri,**buf_ri_k;
  PetscInt            len,proc,*dnz,*onz,*owners;
  PetscInt            nzi,*pti,*ptj;
  PetscInt            nrows,*buf_s,*buf_si,*buf_si_i,**nextrow,**nextci;
  MPI_Request         *swaits,*rwaits;
  MPI_Status          *sstatus,rstatus;
  Mat_Merge_SeqsToMPI *merge;
  PetscInt            *api,*apj,*Jptr,apnz,*prmap=p->garray,pon,nspacedouble=0,j,ap_rmax=0;
  PetscReal           afill=1.0,afill_tmp;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  /* create struct Mat_PtAPMPI and attached it to C later */
  ierr        = PetscNew(&ptap);CHKERRQ(ierr);
  ierr        = PetscNew(&merge);CHKERRQ(ierr);
  ptap->merge = merge;
  ptap->reuse = MAT_INITIAL_MATRIX;

  /* get P_oth by taking rows of P (= non-zero cols of local A) from other processors */
  ierr = MatGetBrowsOfAoCols_MPIAIJ(A,P,MAT_INITIAL_MATRIX,&ptap->startsj_s,&ptap->startsj_r,&ptap->bufa,&ptap->P_oth);CHKERRQ(ierr);

  /* get P_loc by taking all local rows of P */
  ierr = MatMPIAIJGetLocalMat(P,MAT_INITIAL_MATRIX,&ptap->P_loc);CHKERRQ(ierr);

  p_loc  = (Mat_SeqAIJ*)(ptap->P_loc)->data;
  p_oth  = (Mat_SeqAIJ*)(ptap->P_oth)->data;
  pi_loc = p_loc->i; pj_loc = p_loc->j;
  pi_oth = p_oth->i; pj_oth = p_oth->j;

  /* (1) compute symbolic AP = A_loc*P = A_diag*P_loc + A_off*P_oth (api,apj) */
  /*--------------------------------------------------------------------------*/
  ierr   = PetscMalloc1(am+1,&api);CHKERRQ(ierr);
  api[0] = 0;

  /* create and initialize a linked list */
  ierr = PetscLLCondensedCreate(pN,pN,&lnk,&lnkbt);CHKERRQ(ierr);

  /* Initial FreeSpace size is fill*(nnz(A) + nnz(P)) -OOM for ex56, np=8k on Intrepid! */
  ierr = PetscFreeSpaceGet(PetscRealIntMultTruncate(fill,PetscIntSumTruncate(adi[am],PetscIntSumTruncate(aoi[am],pi_loc[pm]))),&free_space);CHKERRQ(ierr);
  current_space = free_space;

  for (i=0; i<am; i++) {
    /* diagonal portion of A */
    nzi = adi[i+1] - adi[i];
    aj  = ad->j + adi[i];
    for (j=0; j<nzi; j++) {
      row  = aj[j];
      pnz  = pi_loc[row+1] - pi_loc[row];
      Jptr = pj_loc + pi_loc[row];
      /* add non-zero cols of P into the sorted linked list lnk */
      ierr = PetscLLCondensedAddSorted(pnz,Jptr,lnk,lnkbt);CHKERRQ(ierr);
    }
    /* off-diagonal portion of A */
    nzi = aoi[i+1] - aoi[i];
    aj  = ao->j + aoi[i];
    for (j=0; j<nzi; j++) {
      row  = aj[j];
      pnz  = pi_oth[row+1] - pi_oth[row];
      Jptr = pj_oth + pi_oth[row];
      ierr = PetscLLCondensedAddSorted(pnz,Jptr,lnk,lnkbt);CHKERRQ(ierr);
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

  /* Allocate space for apj, initialize apj, and */
  /* destroy list of free space and other temporary array(s) */
  ierr      = PetscMalloc1(api[am]+1,&apj);CHKERRQ(ierr);
  ierr      = PetscFreeSpaceContiguous(&free_space,apj);CHKERRQ(ierr);
  afill_tmp = (PetscReal)api[am]/(adi[am]+aoi[am]+pi_loc[pm]+1);
  if (afill_tmp > afill) afill = afill_tmp;

  /* (2) determine symbolic Co=(p->B)^T*AP - send to others (coi,coj)*/
  /*-----------------------------------------------------------------*/
  ierr = MatGetSymbolicTranspose_SeqAIJ(p->B,&poti,&potj);CHKERRQ(ierr);

  /* then, compute symbolic Co = (p->B)^T*AP */
  pon    = (p->B)->cmap->n; /* total num of rows to be sent to other processors
                         >= (num of nonzero rows of C_seq) - pn */
  ierr   = PetscMalloc1(pon+1,&coi);CHKERRQ(ierr);
  coi[0] = 0;

  /* set initial free space to be fill*(nnz(p->B) + nnz(AP)) */
  nnz           = PetscRealIntMultTruncate(fill,PetscIntSumTruncate(poti[pon],api[am]));
  ierr          = PetscFreeSpaceGet(nnz,&free_space);CHKERRQ(ierr);
  current_space = free_space;

  for (i=0; i<pon; i++) {
    pnz = poti[i+1] - poti[i];
    ptJ = potj + poti[i];
    for (j=0; j<pnz; j++) {
      row  = ptJ[j]; /* row of AP == col of Pot */
      apnz = api[row+1] - api[row];
      Jptr = apj + api[row];
      /* add non-zero cols of AP into the sorted linked list lnk */
      ierr = PetscLLCondensedAddSorted(apnz,Jptr,lnk,lnkbt);CHKERRQ(ierr);
    }
    nnz = lnk[0];

    /* If free space is not available, double the total space in the list */
    if (current_space->local_remaining<nnz) {
      ierr = PetscFreeSpaceGet(PetscIntSumTruncate(nnz,current_space->total_array_size),&current_space);CHKERRQ(ierr);
      nspacedouble++;
    }

    /* Copy data into free space, and zero out denserows */
    ierr = PetscLLCondensedClean(pN,nnz,current_space->array,lnk,lnkbt);CHKERRQ(ierr);

    current_space->array           += nnz;
    current_space->local_used      += nnz;
    current_space->local_remaining -= nnz;

    coi[i+1] = coi[i] + nnz;
  }
  
  ierr      = PetscMalloc1(coi[pon],&coj);CHKERRQ(ierr);
  ierr      = PetscFreeSpaceContiguous(&free_space,coj);CHKERRQ(ierr);
  afill_tmp = (PetscReal)coi[pon]/(poti[pon] + api[am]+1);
  if (afill_tmp > afill) afill = afill_tmp;
  ierr = MatRestoreSymbolicTranspose_SeqAIJ(p->B,&poti,&potj);CHKERRQ(ierr);

  /* (3) send j-array (coj) of Co to other processors */
  /*--------------------------------------------------*/
  ierr = PetscCalloc1(size,&merge->len_s);CHKERRQ(ierr);
  len_s        = merge->len_s;
  merge->nsend = 0;


  /* determine row ownership */
  ierr = PetscLayoutCreate(comm,&merge->rowmap);CHKERRQ(ierr);
  merge->rowmap->n  = pn;
  merge->rowmap->bs = 1;

  ierr   = PetscLayoutSetUp(merge->rowmap);CHKERRQ(ierr);
  owners = merge->rowmap->range;

  /* determine the number of messages to send, their lengths */
  ierr = PetscMalloc2(size,&len_si,size,&sstatus);CHKERRQ(ierr);
  ierr = PetscMemzero(len_si,size*sizeof(PetscMPIInt));CHKERRQ(ierr);
  ierr = PetscMalloc1(size+2,&owners_co);CHKERRQ(ierr);

  proc = 0;
  for (i=0; i<pon; i++) {
    while (prmap[i] >= owners[proc+1]) proc++;
    len_si[proc]++;               /* num of rows in Co(=Pt*AP) to be sent to [proc] */
    len_s[proc] += coi[i+1] - coi[i]; /* num of nonzeros in Co to be sent to [proc] */
  }

  len          = 0; /* max length of buf_si[], see (4) */
  owners_co[0] = 0;
  for (proc=0; proc<size; proc++) {
    owners_co[proc+1] = owners_co[proc] + len_si[proc];
    if (len_s[proc]) {
      merge->nsend++;
      len_si[proc] = 2*(len_si[proc] + 1); /* length of buf_si to be sent to [proc] */
      len         += len_si[proc];
    }
  }

  /* determine the number and length of messages to receive for coi and coj  */
  ierr = PetscGatherNumberOfMessages(comm,NULL,len_s,&merge->nrecv);CHKERRQ(ierr);
  ierr = PetscGatherMessageLengths2(comm,merge->nsend,merge->nrecv,len_s,len_si,&merge->id_r,&merge->len_r,&len_ri);CHKERRQ(ierr);

  /* post the Irecv and Isend of coj */
  ierr = PetscCommGetNewTag(comm,&tagj);CHKERRQ(ierr);
  ierr = PetscPostIrecvInt(comm,tagj,merge->nrecv,merge->id_r,merge->len_r,&buf_rj,&rwaits);CHKERRQ(ierr);
  ierr = PetscMalloc1(merge->nsend+1,&swaits);CHKERRQ(ierr);
  for (proc=0, k=0; proc<size; proc++) {
    if (!len_s[proc]) continue;
    i    = owners_co[proc];
    ierr = MPI_Isend(coj+coi[i],len_s[proc],MPIU_INT,proc,tagj,comm,swaits+k);CHKERRQ(ierr);
    k++;
  }

  /* receives and sends of coj are complete */
  for (i=0; i<merge->nrecv; i++) {
    ierr = MPI_Waitany(merge->nrecv,rwaits,&icompleted,&rstatus);CHKERRQ(ierr);
  }
  ierr = PetscFree(rwaits);CHKERRQ(ierr);
  if (merge->nsend) {ierr = MPI_Waitall(merge->nsend,swaits,sstatus);CHKERRQ(ierr);}

  /* (4) send and recv coi */
  /*-----------------------*/
  ierr   = PetscCommGetNewTag(comm,&tagi);CHKERRQ(ierr);
  ierr   = PetscPostIrecvInt(comm,tagi,merge->nrecv,merge->id_r,len_ri,&buf_ri,&rwaits);CHKERRQ(ierr);
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
    ierr = MPI_Isend(buf_si,len_si[proc],MPIU_INT,proc,tagi,comm,swaits+k);CHKERRQ(ierr);
    k++;
    buf_si += len_si[proc];
  }
  i = merge->nrecv;
  while (i--) {
    ierr = MPI_Waitany(merge->nrecv,rwaits,&icompleted,&rstatus);CHKERRQ(ierr);
  }
  ierr = PetscFree(rwaits);CHKERRQ(ierr);
  if (merge->nsend) {ierr = MPI_Waitall(merge->nsend,swaits,sstatus);CHKERRQ(ierr);}

  ierr = PetscFree2(len_si,sstatus);CHKERRQ(ierr);
  ierr = PetscFree(len_ri);CHKERRQ(ierr);
  ierr = PetscFree(swaits);CHKERRQ(ierr);
  ierr = PetscFree(buf_s);CHKERRQ(ierr);

  /* (5) compute the local portion of C (mpi mat) */
  /*----------------------------------------------*/
  ierr = MatGetSymbolicTranspose_SeqAIJ(p->A,&pdti,&pdtj);CHKERRQ(ierr);

  /* allocate pti array and free space for accumulating nonzero column info */
  ierr   = PetscMalloc1(pn+1,&pti);CHKERRQ(ierr);
  pti[0] = 0;

  /* set initial free space to be fill*(nnz(P) + nnz(AP)) */
  nnz           = PetscRealIntMultTruncate(fill,PetscIntSumTruncate(pi_loc[pm],api[am]));
  ierr          = PetscFreeSpaceGet(nnz,&free_space);CHKERRQ(ierr);
  current_space = free_space;

  ierr = PetscMalloc3(merge->nrecv,&buf_ri_k,merge->nrecv,&nextrow,merge->nrecv,&nextci);CHKERRQ(ierr);
  for (k=0; k<merge->nrecv; k++) {
    buf_ri_k[k] = buf_ri[k]; /* beginning of k-th recved i-structure */
    nrows       = *buf_ri_k[k];
    nextrow[k]  = buf_ri_k[k] + 1;  /* next row number of k-th recved i-structure */
    nextci[k]   = buf_ri_k[k] + (nrows + 1); /* poins to the next i-structure of k-th recved i-structure  */
  }
  ierr = MatPreallocateInitialize(comm,pn,pn,dnz,onz);CHKERRQ(ierr);
  
  for (i=0; i<pn; i++) {
    /* add pdt[i,:]*AP into lnk */
    pnz = pdti[i+1] - pdti[i];
    ptJ = pdtj + pdti[i];
    for (j=0; j<pnz; j++) {
      row  = ptJ[j];  /* row of AP == col of Pt */
      apnz = api[row+1] - api[row];
      Jptr = apj + api[row];
      /* add non-zero cols of AP into the sorted linked list lnk */
      ierr = PetscLLCondensedAddSorted(apnz,Jptr,lnk,lnkbt);CHKERRQ(ierr);
    }

    /* add received col data into lnk */
    for (k=0; k<merge->nrecv; k++) { /* k-th received message */
      if (i == *nextrow[k]) { /* i-th row */
        nzi  = *(nextci[k]+1) - *nextci[k];
        Jptr = buf_rj[k] + *nextci[k];
        ierr = PetscLLCondensedAddSorted(nzi,Jptr,lnk,lnkbt);CHKERRQ(ierr);
        nextrow[k]++; nextci[k]++;
      }
    }
    nnz = lnk[0];

    /* if free space is not available, make more free space */
    if (current_space->local_remaining<nnz) {
      ierr = PetscFreeSpaceGet(PetscIntSumTruncate(nnz,current_space->total_array_size),&current_space);CHKERRQ(ierr);
      nspacedouble++;
    }
    /* copy data into free space, then initialize lnk */
    ierr = PetscLLCondensedClean(pN,nnz,current_space->array,lnk,lnkbt);CHKERRQ(ierr);
    ierr = MatPreallocateSet(i+owners[rank],nnz,current_space->array,dnz,onz);CHKERRQ(ierr);

    current_space->array           += nnz;
    current_space->local_used      += nnz;
    current_space->local_remaining -= nnz;

    pti[i+1] = pti[i] + nnz;
  }
  ierr = MatRestoreSymbolicTranspose_SeqAIJ(p->A,&pdti,&pdtj);CHKERRQ(ierr);
  ierr = PetscFree3(buf_ri_k,nextrow,nextci);CHKERRQ(ierr);

  ierr      = PetscMalloc1(pti[pn]+1,&ptj);CHKERRQ(ierr);
  ierr      = PetscFreeSpaceContiguous(&free_space,ptj);CHKERRQ(ierr);
  afill_tmp = (PetscReal)pti[pn]/(pi_loc[pm] + api[am]+1);
  if (afill_tmp > afill) afill = afill_tmp;
  ierr = PetscLLDestroy(lnk,lnkbt);CHKERRQ(ierr);

  /* (6) create symbolic parallel matrix Cmpi */
  /*------------------------------------------*/
  ierr = MatCreate(comm,&Cmpi);CHKERRQ(ierr);
  ierr = MatSetSizes(Cmpi,pn,pn,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetBlockSizes(Cmpi,PetscAbs(P->cmap->bs),PetscAbs(P->cmap->bs));CHKERRQ(ierr);
  ierr = MatSetType(Cmpi,MATMPIAIJ);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(Cmpi,0,dnz,0,onz);CHKERRQ(ierr);
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);

  merge->bi        = pti;      /* Cseq->i */
  merge->bj        = ptj;      /* Cseq->j */
  merge->coi       = coi;      /* Co->i   */
  merge->coj       = coj;      /* Co->j   */
  merge->buf_ri    = buf_ri;
  merge->buf_rj    = buf_rj;
  merge->owners_co = owners_co;
  merge->destroy   = Cmpi->ops->destroy;

  /* attach the supporting struct to Cmpi for reuse */
  c           = (Mat_MPIAIJ*)Cmpi->data;
  c->ptap     = ptap;
  ptap->api   = api;
  ptap->apj   = apj;
  ptap->duplicate = Cmpi->ops->duplicate;
  ptap->destroy   = Cmpi->ops->destroy;

  /* Cmpi is not ready for use - assembly will be done by MatPtAPNumeric() */
  Cmpi->assembled        = PETSC_FALSE;
  Cmpi->ops->destroy     = MatDestroy_MPIAIJ_PtAP;
  Cmpi->ops->duplicate   = MatDuplicate_MPIAIJ_MatPtAP; 
  Cmpi->ops->ptapnumeric = MatPtAPNumeric_MPIAIJ_MPIAIJ_ptap;
  *C                     = Cmpi;

  /* flag 'scalable' determines which implementations to be used:
       0: do dense axpy in MatPtAPNumeric() - fast, but requires storage of a nonscalable dense array apa;
       1: do sparse axpy in MatPtAPNumeric() - might slow, uses a sparse array apa */
  /* set default scalable */
  ptap->scalable = PETSC_FALSE; /* PETSC_TRUE; */

  ierr = PetscOptionsGetBool(((PetscObject)Cmpi)->options,((PetscObject)Cmpi)->prefix,"-matptap_scalable",&ptap->scalable,NULL);CHKERRQ(ierr);
  if (!ptap->scalable) {  /* Do dense axpy */
    ierr = PetscCalloc1(pN,&ptap->apa);CHKERRQ(ierr);
  } else {
    ierr = PetscCalloc1(ap_rmax+1,&ptap->apa);CHKERRQ(ierr);
  }

#if defined(PETSC_USE_INFO)
  if (pti[pn] != 0) {
    ierr = PetscInfo3(Cmpi,"Reallocs %D; Fill ratio: given %g needed %g.\n",nspacedouble,(double)fill,(double)afill);CHKERRQ(ierr);
    ierr = PetscInfo1(Cmpi,"Use MatPtAP(A,P,MatReuse,%g,&C) for best performance.\n",(double)afill);CHKERRQ(ierr);
  } else {
    ierr = PetscInfo(Cmpi,"Empty matrix product\n");CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode MatPtAPNumeric_MPIAIJ_MPIAIJ_ptap(Mat A,Mat P,Mat C)
{
  PetscErrorCode      ierr;
  Mat_MPIAIJ          *a =(Mat_MPIAIJ*)A->data,*p=(Mat_MPIAIJ*)P->data,*c=(Mat_MPIAIJ*)C->data;
  Mat_SeqAIJ          *ad=(Mat_SeqAIJ*)(a->A)->data,*ao=(Mat_SeqAIJ*)(a->B)->data;
  Mat_SeqAIJ          *pd=(Mat_SeqAIJ*)(p->A)->data,*po=(Mat_SeqAIJ*)(p->B)->data;
  Mat_SeqAIJ          *p_loc,*p_oth;
  Mat_PtAPMPI         *ptap;
  Mat_Merge_SeqsToMPI *merge;
  PetscInt            *adi=ad->i,*aoi=ao->i,*adj,*aoj,*apJ,nextp;
  PetscInt            *pi_loc,*pj_loc,*pi_oth,*pj_oth,*pJ,*pj;
  PetscInt            i,j,k,anz,pnz,apnz,nextap,row,*cj;
  MatScalar           *ada,*aoa,*apa,*pa,*ca,*pa_loc,*pa_oth,valtmp;
  PetscInt            am  =A->rmap->n,cm=C->rmap->n,pon=(p->B)->cmap->n;
  MPI_Comm            comm;
  PetscMPIInt         size,rank,taga,*len_s;
  PetscInt            *owners,proc,nrows,**buf_ri_k,**nextrow,**nextci;
  PetscInt            **buf_ri,**buf_rj;
  PetscInt            cnz=0,*bj_i,*bi,*bj,bnz,nextcj;  /* bi,bj,ba: local array of C(mpi mat) */
  MPI_Request         *s_waits,*r_waits;
  MPI_Status          *status;
  MatScalar           **abuf_r,*ba_i,*pA,*coa,*ba;
  PetscInt            *api,*apj,*coi,*coj;
  PetscInt            *poJ=po->j,*pdJ=pd->j,pcstart=P->cmap->rstart,pcend=P->cmap->rend;
  PetscBool           scalable;
#if defined(PTAP_PROFILE)
  PetscLogDouble t0,t1,t2,eP,t3,t4,et2_AP=0.0,ePtAP=0.0,t2_0,t2_1,t2_2;
#endif

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)C,&comm);CHKERRQ(ierr);
#if defined(PTAP_PROFILE)
  ierr = PetscTime(&t0);CHKERRQ(ierr);
#endif
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  ptap = c->ptap;
  if (!ptap) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_ARG_INCOMP,"MatPtAP() has not been called to create matrix C yet, cannot use MAT_REUSE_MATRIX");
  merge    = ptap->merge;
  apa      = ptap->apa;
  scalable = ptap->scalable;

  /* 1) get P_oth = ptap->P_oth  and P_loc = ptap->P_loc */
  /*-----------------------------------------------------*/
  if (ptap->reuse == MAT_INITIAL_MATRIX) {
    /* P_oth and P_loc are obtained in MatPtASymbolic(), skip calling MatGetBrowsOfAoCols() and MatMPIAIJGetLocalMat() */
    ptap->reuse = MAT_REUSE_MATRIX;
  } else { /* update numerical values of P_oth and P_loc */
    ierr = MatGetBrowsOfAoCols_MPIAIJ(A,P,MAT_REUSE_MATRIX,&ptap->startsj_s,&ptap->startsj_r,&ptap->bufa,&ptap->P_oth);CHKERRQ(ierr);
    ierr = MatMPIAIJGetLocalMat(P,MAT_REUSE_MATRIX,&ptap->P_loc);CHKERRQ(ierr);
  }
#if defined(PTAP_PROFILE)
  ierr = PetscTime(&t1);CHKERRQ(ierr);
  eP = t1-t0;
#endif
  /*
  printf("[%d] Ad: %d, %d; Ao: %d, %d; P_loc: %d, %d; P_oth %d, %d;\n",rank,
         a->A->rmap->N,a->A->cmap->N,a->B->rmap->N,a->B->cmap->N,
         ptap->P_loc->rmap->N,ptap->P_loc->cmap->N,
         ptap->P_oth->rmap->N,ptap->P_oth->cmap->N);
   */

  /* 2) compute numeric C_seq = P_loc^T*A_loc*P - dominating part */
  /*--------------------------------------------------------------*/
  /* get data from symbolic products */
  p_loc = (Mat_SeqAIJ*)(ptap->P_loc)->data;
  p_oth = (Mat_SeqAIJ*)(ptap->P_oth)->data;
  pi_loc=p_loc->i; pj_loc=p_loc->j; pa_loc=p_loc->a;
  pi_oth=p_oth->i; pj_oth=p_oth->j; pa_oth=p_oth->a;

  coi  = merge->coi; coj = merge->coj;
  ierr = PetscCalloc1(coi[pon]+1,&coa);CHKERRQ(ierr);

  bi     = merge->bi; bj = merge->bj;
  owners = merge->rowmap->range;
  ierr   = PetscCalloc1(bi[cm]+1,&ba);CHKERRQ(ierr);  /* ba: Cseq->a */

  api = ptap->api; apj = ptap->apj;

  if (!scalable) { /* Do dense axpy on apa (length of pN, stores A[i,:]*P) - nonscalable, but faster (could take 1/3 scalable time) */
    ierr = PetscInfo(C,"Using non-scalable dense axpy\n");CHKERRQ(ierr);
#if defined(PTAP_PROFILE)   
    ierr = PetscTime(&t1);CHKERRQ(ierr);
#endif
    for (i=0; i<am; i++) {
#if defined(PTAP_PROFILE)
      ierr = PetscTime(&t2_0);CHKERRQ(ierr);
#endif
      /* 2-a) form i-th sparse row of A_loc*P = Ad*P_loc + Ao*P_oth */
      /*------------------------------------------------------------*/
      apJ = apj + api[i];

      /* diagonal portion of A */
      anz = adi[i+1] - adi[i];
      adj = ad->j + adi[i];
      ada = ad->a + adi[i];
      for (j=0; j<anz; j++) {
        row = adj[j];
        pnz = pi_loc[row+1] - pi_loc[row];
        pj  = pj_loc + pi_loc[row];
        pa  = pa_loc + pi_loc[row];

        /* perform dense axpy */
        valtmp = ada[j];
        for (k=0; k<pnz; k++) {
          apa[pj[k]] += valtmp*pa[k];
        }
        ierr = PetscLogFlops(2.0*pnz);CHKERRQ(ierr);
      }

      /* off-diagonal portion of A */
      anz = aoi[i+1] - aoi[i];
      aoj = ao->j + aoi[i];
      aoa = ao->a + aoi[i];
      for (j=0; j<anz; j++) {
        row = aoj[j];
        pnz = pi_oth[row+1] - pi_oth[row];
        pj  = pj_oth + pi_oth[row];
        pa  = pa_oth + pi_oth[row];

        /* perform dense axpy */
        valtmp = aoa[j];
        for (k=0; k<pnz; k++) {
          apa[pj[k]] += valtmp*pa[k];
        }
        ierr = PetscLogFlops(2.0*pnz);CHKERRQ(ierr);
      }
#if defined(PTAP_PROFILE)
      ierr    = PetscTime(&t2_1);CHKERRQ(ierr);
      et2_AP += t2_1 - t2_0;
#endif

      /* 2-b) Compute Cseq = P_loc[i,:]^T*AP[i,:] using outer product */
      /*--------------------------------------------------------------*/
      apnz = api[i+1] - api[i];
      /* put the value into Co=(p->B)^T*AP (off-diagonal part, send to others) */
      pnz = po->i[i+1] - po->i[i];
      poJ = po->j + po->i[i];
      pA  = po->a + po->i[i];
      for (j=0; j<pnz; j++) {
        row = poJ[j];
        cnz = coi[row+1] - coi[row];
        cj  = coj + coi[row];
        ca  = coa + coi[row];
        /* perform dense axpy */
        valtmp = pA[j];
        for (k=0; k<cnz; k++) {
          ca[k] += valtmp*apa[cj[k]];
        }
        ierr = PetscLogFlops(2.0*cnz);CHKERRQ(ierr);
      }
      /* put the value into Cd (diagonal part) */
      pnz = pd->i[i+1] - pd->i[i];
      pdJ = pd->j + pd->i[i];
      pA  = pd->a + pd->i[i];
      for (j=0; j<pnz; j++) {
        row = pdJ[j];
        cnz = bi[row+1] - bi[row];
        cj  = bj + bi[row];
        ca  = ba + bi[row];
        /* perform dense axpy */
        valtmp = pA[j];
        for (k=0; k<cnz; k++) {
          ca[k] += valtmp*apa[cj[k]];
        }
        ierr = PetscLogFlops(2.0*cnz);CHKERRQ(ierr);
      }
  
      /* zero the current row of A*P */
      for (k=0; k<apnz; k++) apa[apJ[k]] = 0.0;
#if defined(PTAP_PROFILE)
      ierr      = PetscTime(&t2_2);CHKERRQ(ierr);
      ePtAP += t2_2 - t2_1;
#endif
    }
  } else { /* Do sparse axpy on apa (length of ap_rmax, stores A[i,:]*P) - scalable, but slower */
    ierr = PetscInfo(C,"Using scalable sparse axpy\n");CHKERRQ(ierr);
    /*-----------------------------------------------------------------------------------------*/
    pA=pa_loc;
    for (i=0; i<am; i++) {
#if defined(PTAP_PROFILE)
      ierr = PetscTime(&t2_0);CHKERRQ(ierr);
#endif
      /* form i-th sparse row of A*P */
      apnz = api[i+1] - api[i];
      apJ  = apj + api[i];
      /* diagonal portion of A */
      anz = adi[i+1] - adi[i];
      adj = ad->j + adi[i];
      ada = ad->a + adi[i];
      for (j=0; j<anz; j++) {
        row    = adj[j];
        pnz    = pi_loc[row+1] - pi_loc[row];
        pj     = pj_loc + pi_loc[row];
        pa     = pa_loc + pi_loc[row];
        valtmp = ada[j];
        nextp  = 0;
        for (k=0; nextp<pnz; k++) {
          if (apJ[k] == pj[nextp]) { /* col of AP == col of P */
            apa[k] += valtmp*pa[nextp++];
          }
        }
        ierr = PetscLogFlops(2.0*pnz);CHKERRQ(ierr);
      }
      /* off-diagonal portion of A */
      anz = aoi[i+1] - aoi[i];
      aoj = ao->j + aoi[i];
      aoa = ao->a + aoi[i];
      for (j=0; j<anz; j++) {
        row    = aoj[j];
        pnz    = pi_oth[row+1] - pi_oth[row];
        pj     = pj_oth + pi_oth[row];
        pa     = pa_oth + pi_oth[row];
        valtmp = aoa[j];
        nextp  = 0;
        for (k=0; nextp<pnz; k++) {
          if (apJ[k] == pj[nextp]) { /* col of AP == col of P */
            apa[k] += valtmp*pa[nextp++];
          }
        }
        ierr = PetscLogFlops(2.0*pnz);CHKERRQ(ierr);
      }
#if defined(PTAP_PROFILE)
      ierr    = PetscTime(&t2_1);CHKERRQ(ierr);
      et2_AP += t2_1 - t2_0;
#endif

      /* 2-b) Compute Cseq = P_loc[i,:]^T*AP[i,:] using outer product */
      /*--------------------------------------------------------------*/
      pnz = pi_loc[i+1] - pi_loc[i];
      pJ  = pj_loc + pi_loc[i];
      for (j=0; j<pnz; j++) {
        nextap = 0;
        row    = pJ[j]; /* global index */
        if (row < pcstart || row >=pcend) { /* put the value into Co */
          row = *poJ;
          cj  = coj + coi[row];
          ca  = coa + coi[row]; poJ++;
        } else {                            /* put the value into Cd */
          row = *pdJ;
          cj  = bj + bi[row];
          ca  = ba + bi[row]; pdJ++;
        }
        valtmp = pA[j];
        for (k=0; nextap<apnz; k++) {
          if (cj[k]==apJ[nextap]) ca[k] += valtmp*apa[nextap++];
        }
        ierr = PetscLogFlops(2.0*apnz);CHKERRQ(ierr);
      }
      pA += pnz;
      /* zero the current row info for A*P */
      ierr = PetscMemzero(apa,apnz*sizeof(MatScalar));CHKERRQ(ierr);
#if defined(PTAP_PROFILE)
      ierr      = PetscTime(&t2_2);CHKERRQ(ierr);
      ePtAP += t2_2 - t2_1;
#endif
    }
  }
#if defined(PTAP_PROFILE)
  ierr = PetscTime(&t2);CHKERRQ(ierr);
#endif

  /* 3) send and recv matrix values coa */
  /*------------------------------------*/
  buf_ri = merge->buf_ri;
  buf_rj = merge->buf_rj;
  len_s  = merge->len_s;
  ierr   = PetscCommGetNewTag(comm,&taga);CHKERRQ(ierr);
  ierr   = PetscPostIrecvScalar(comm,taga,merge->nrecv,merge->id_r,merge->len_r,&abuf_r,&r_waits);CHKERRQ(ierr);

  ierr = PetscMalloc2(merge->nsend+1,&s_waits,size,&status);CHKERRQ(ierr);
  for (proc=0,k=0; proc<size; proc++) {
    if (!len_s[proc]) continue;
    i    = merge->owners_co[proc];
    ierr = MPI_Isend(coa+coi[i],len_s[proc],MPIU_MATSCALAR,proc,taga,comm,s_waits+k);CHKERRQ(ierr);
    k++;
  }
  if (merge->nrecv) {ierr = MPI_Waitall(merge->nrecv,r_waits,status);CHKERRQ(ierr);}
  if (merge->nsend) {ierr = MPI_Waitall(merge->nsend,s_waits,status);CHKERRQ(ierr);}

  ierr = PetscFree2(s_waits,status);CHKERRQ(ierr);
  ierr = PetscFree(r_waits);CHKERRQ(ierr);
  ierr = PetscFree(coa);CHKERRQ(ierr);
#if defined(PTAP_PROFILE)
  ierr = PetscTime(&t3);CHKERRQ(ierr);
#endif

  /* 4) insert local Cseq and received values into Cmpi */
  /*------------------------------------------------------*/
  ierr = PetscMalloc3(merge->nrecv,&buf_ri_k,merge->nrecv,&nextrow,merge->nrecv,&nextci);CHKERRQ(ierr);
  for (k=0; k<merge->nrecv; k++) {
    buf_ri_k[k] = buf_ri[k]; /* beginning of k-th recved i-structure */
    nrows       = *(buf_ri_k[k]);
    nextrow[k]  = buf_ri_k[k]+1;  /* next row number of k-th recved i-structure */
    nextci[k]   = buf_ri_k[k] + (nrows + 1); /* poins to the next i-structure of k-th recved i-structure  */
  }

  for (i=0; i<cm; i++) {
    row  = owners[rank] + i; /* global row index of C_seq */
    bj_i = bj + bi[i];  /* col indices of the i-th row of C */
    ba_i = ba + bi[i];
    bnz  = bi[i+1] - bi[i];
    /* add received vals into ba */
    for (k=0; k<merge->nrecv; k++) { /* k-th received message */
      /* i-th row */
      if (i == *nextrow[k]) {
        cnz    = *(nextci[k]+1) - *nextci[k];
        cj     = buf_rj[k] + *(nextci[k]);
        ca     = abuf_r[k] + *(nextci[k]);
        nextcj = 0;
        for (j=0; nextcj<cnz; j++) {
          if (bj_i[j] == cj[nextcj]) { /* bcol == ccol */
            ba_i[j] += ca[nextcj++];
          }
        }
        nextrow[k]++; nextci[k]++;
        ierr = PetscLogFlops(2.0*cnz);CHKERRQ(ierr);
      }
    }
    ierr = MatSetValues(C,1,&row,bnz,bj_i,ba_i,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = PetscFree(ba);CHKERRQ(ierr);
  ierr = PetscFree(abuf_r[0]);CHKERRQ(ierr);
  ierr = PetscFree(abuf_r);CHKERRQ(ierr);
  ierr = PetscFree3(buf_ri_k,nextrow,nextci);CHKERRQ(ierr);
#if defined(PTAP_PROFILE)
  ierr = PetscTime(&t4);CHKERRQ(ierr);
  if (rank==1) {
    ierr = PetscPrintf(MPI_COMM_SELF,"  [%d] PtAPNum %g/P + %g/PtAP( %g/A*P + %g/Pt*AP ) + %g/comm + %g/Cloc = %g\n\n",rank,eP,t2-t1,et2_AP,ePtAP,t3-t2,t4-t3,t4-t0);CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}
