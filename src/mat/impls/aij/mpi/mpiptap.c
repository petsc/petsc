/*
  Defines projective product routines where A is a MPIAIJ matrix
          C = P^T * A * P
*/

#include "src/mat/impls/aij/seq/aij.h"   /*I "petscmat.h" I*/
#include "src/mat/utils/freespace.h"
#include "src/mat/impls/aij/mpi/mpiaij.h"
#include "petscbt.h"

EXTERN PetscErrorCode MatDestroy_MPIAIJ(Mat);
#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_MPIAIJ_PtAP"
PetscErrorCode MatDestroy_MPIAIJ_PtAP(Mat A)
{
  PetscErrorCode       ierr;
  Mat_MatMatMultMPI    *ptap; 
  PetscObjectContainer container;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)A,"MatPtAPMPI",(PetscObject *)&container);CHKERRQ(ierr);
  if (container) {
    ierr  = PetscObjectContainerGetPointer(container,(void **)&ptap);CHKERRQ(ierr); 
    ierr = MatDestroy(ptap->B_loc);CHKERRQ(ierr);
    ierr = MatDestroy(ptap->B_oth);CHKERRQ(ierr);
    ierr = ISDestroy(ptap->isrowb);CHKERRQ(ierr);
    ierr = ISDestroy(ptap->iscolb);CHKERRQ(ierr);
    if (ptap->abi) ierr = PetscFree(ptap->abi);CHKERRQ(ierr);
    if (ptap->abj) ierr = PetscFree(ptap->abj);CHKERRQ(ierr);
    
    ierr = PetscObjectContainerDestroy(container);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)A,"MatPtAPMPI",0);CHKERRQ(ierr);
  }
  ierr = PetscFree(ptap);CHKERRQ(ierr);

  ierr = MatDestroy_MPIAIJ(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPtAP_MPIAIJ_MPIAIJ"
PetscErrorCode MatPtAP_MPIAIJ_MPIAIJ(Mat A,Mat P,MatReuse scall,PetscReal fill,Mat *C) 
{
  PetscErrorCode       ierr;
  Mat_MatMatMultMPI    *ptap;
  PetscObjectContainer container;

  PetscFunctionBegin;
  if (scall == MAT_INITIAL_MATRIX){ 
    ierr = PetscLogEventBegin(MAT_PtAPSymbolic,A,P,0,0);CHKERRQ(ierr);
    ierr = PetscNew(Mat_MatMatMultMPI,&ptap);CHKERRQ(ierr);
    ptap->abi=PETSC_NULL; ptap->abj=PETSC_NULL; 
    ptap->abnz_max = 0; /* symbolic A*P is not done yet */

    /* get P_oth by taking rows of P (= non-zero cols of local A) from other processors */
    ierr = MatGetBrowsOfAoCols(A,P,scall,&ptap->isrowb,&ptap->iscolb,&ptap->brstart,&ptap->B_oth);CHKERRQ(ierr);
  
    /* get P_loc by taking all local rows of P */
    ierr = MatGetLocalMat(P,scall,&ptap->B_loc);CHKERRQ(ierr);

    /* attach the supporting struct to P for reuse */
    P->ops->destroy  = MatDestroy_MPIAIJ_PtAP;
    ierr = PetscObjectContainerCreate(PETSC_COMM_SELF,&container);CHKERRQ(ierr);
    ierr = PetscObjectContainerSetPointer(container,ptap);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)P,"MatPtAPMPI",(PetscObject)container);CHKERRQ(ierr);
  
    /* now, compute symbolic local P^T*A*P */
    ierr = MatPtAPSymbolic_MPIAIJ_MPIAIJ(A,P,fill,C);CHKERRQ(ierr);/* numeric product is computed as well */
    ierr = PetscLogEventEnd(MAT_PtAPSymbolic,A,P,0,0);CHKERRQ(ierr); 
  } else if (scall == MAT_REUSE_MATRIX){
    ierr = PetscObjectQuery((PetscObject)P,"MatPtAPMPI",(PetscObject *)&container);CHKERRQ(ierr);
    if (container) { 
      ierr  = PetscObjectContainerGetPointer(container,(void **)&ptap);CHKERRQ(ierr); 
    } else {
      SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "Matrix P does not posses an object container");
    } 
    
    /* get P_oth by taking rows of P (= non-zero cols of local A) from other processors */
    ierr = MatGetBrowsOfAoCols(A,P,scall,&ptap->isrowb,&ptap->iscolb,&ptap->brstart,&ptap->B_oth);CHKERRQ(ierr);
  
    /* get P_loc by taking all local rows of P */
    ierr = MatGetLocalMat(P,scall,&ptap->B_loc);CHKERRQ(ierr);

  } else {
    SETERRQ1(PETSC_ERR_ARG_WRONG,"Invalid MatReuse %d",scall);
  }
  /* now, compute numeric local P^T*A*P */
  ierr = PetscLogEventBegin(MAT_PtAPNumeric,A,P,0,0);CHKERRQ(ierr); 
  ierr = MatPtAPNumeric_MPIAIJ_MPIAIJ(A,P,*C);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_PtAPNumeric,A,P,0,0);CHKERRQ(ierr); 
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatPtAPSymbolic_MPIAIJ_MPIAIJ"
PetscErrorCode MatPtAPSymbolic_MPIAIJ_MPIAIJ(Mat A,Mat P,PetscReal fill,Mat *C)
{
  PetscErrorCode       ierr;
  Mat                  P_loc,P_oth;
  Mat_MatMatMultMPI    *ptap;
  PetscObjectContainer container;
  FreeSpaceList        free_space=PETSC_NULL,current_space=PETSC_NULL;
  Mat_MPIAIJ           *a=(Mat_MPIAIJ*)A->data,*p=(Mat_MPIAIJ*)P->data;
  Mat_SeqAIJ           *ad=(Mat_SeqAIJ*)(a->A)->data,*ao=(Mat_SeqAIJ*)(a->B)->data;
  Mat_SeqAIJ           *p_loc,*p_oth;
  PetscInt             *pi_loc,*pj_loc,*pi_oth,*pj_oth,*pti,*ptj,*ptJ,*pjj;
  PetscInt             *adi=ad->i,*adj=ad->j,*aoi=ao->i,*aoj=ao->j,nnz; 
  PetscInt             pm=P->m,pn=P->n,nlnk,*lnk,*ci,*cj,*cji; 
  PetscInt             aN=A->N,am=A->m,pN=P->N;
  PetscInt             i,j,k,ptnzi,arow,prow,pnzj,cnzi;
  PetscBT              lnkbt;
  PetscInt             prstart,prend;
  MPI_Comm             comm=A->comm;
  Mat                  B_mpi;
  PetscMPIInt          size,rank,tagi,tagj,*len_s,*len_si,*len_ri;
  PetscInt             **buf_rj,**buf_ri,**buf_ri_k;
  PetscInt             len,proc,*dnz,*onz,*owners;
  PetscInt             anzi,*bi,*bj,bnzi,nspacedouble=0; 
  PetscInt             nrows,*buf_s,*buf_si,*buf_si_i,**nextrow,**nextci;
  MPI_Request          *si_waits,*sj_waits,*ri_waits,*rj_waits; 
  MPI_Status           *status;
  Mat_Merge_SeqsToMPI  *merge;
  PetscInt             *apsymi,*apsymj,*apj,apnzj,*rmap=p->garray,tnrows;
  /*  PetscInt             tnzrows,pcstart=p->cstart,pcend=p->cend; */

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)P,"MatPtAPMPI",(PetscObject *)&container);CHKERRQ(ierr);
  if (container) { 
    ierr  = PetscObjectContainerGetPointer(container,(void **)&ptap);CHKERRQ(ierr); 
  } else {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "Matrix P does not posses an object container");
  } 

  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  /* compute symbolic C_seq = P_loc^T * A_loc * P */
  /*----------------------------------------------*/
  /* get data from P_loc and P_oth */
  P_loc=ptap->B_loc; P_oth=ptap->B_oth; prstart=ptap->brstart;
  p_loc=(Mat_SeqAIJ*)P_loc->data; p_oth=(Mat_SeqAIJ*)P_oth->data;
  pi_loc=p_loc->i; pj_loc=p_loc->j; pi_oth=p_oth->i; pj_oth=p_oth->j;
  prend = prstart + pm;

  /* first, compute symbolic AP = A_loc*P */
  ierr  = PetscMalloc((am+2)*sizeof(PetscInt),&apsymi);CHKERRQ(ierr);
  ptap->abi = apsymi;
  apsymi[0] = 0;

  /* create and initialize a linked list */
  nlnk = pN+1;
  ierr = PetscLLCreate(pN,pN,nlnk,lnk,lnkbt);CHKERRQ(ierr);

  /* Initial FreeSpace size is 2*nnz(A) */
  ierr = GetMoreSpace((PetscInt)(2*(adi[am]+aoi[am])),&free_space);CHKERRQ(ierr);
  current_space = free_space; 

  for (i=0;i<am;i++) {
    apnzj = 0;
    /* diagonal portion of A */
    anzi = adi[i+1] - adi[i];
    for (j=0; j<anzi; j++){
      prow = *adj++; 
      pnzj = pi_loc[prow+1] - pi_loc[prow];
      pjj  = pj_loc + pi_loc[prow];
      /* add non-zero cols of P into the sorted linked list lnk */
      ierr = PetscLLAdd(pnzj,pjj,pN,nlnk,lnk,lnkbt);CHKERRQ(ierr);
      apnzj += nlnk;
    }
    /* off-diagonal portion of A */
    anzi = aoi[i+1] - aoi[i];
    for (j=0; j<anzi; j++){   
      prow = *aoj++; 
      pnzj = pi_oth[prow+1] - pi_oth[prow];
      pjj  = pj_oth + pi_oth[prow];  
      ierr = PetscLLAdd(pnzj,pjj,pN,nlnk,lnk,lnkbt);CHKERRQ(ierr);
      apnzj += nlnk;
    }

    apsymi[i+1] = apsymi[i] + apnzj;
    if (ptap->abnz_max < apnzj) ptap->abnz_max = apnzj;

    /* if free space is not available, double the total space in the list */
    if (current_space->local_remaining<apnzj) {
      ierr = GetMoreSpace(current_space->total_array_size,&current_space);CHKERRQ(ierr);
    }

    /* Copy data into free space, then initialize lnk */
    ierr = PetscLLClean(pN,pN,apnzj,lnk,current_space->array,lnkbt);CHKERRQ(ierr); 
    current_space->array           += apnzj;
    current_space->local_used      += apnzj;
    current_space->local_remaining -= apnzj;
  }
  /* Allocate space for apsymj, initialize apsymj, and */
  /* destroy list of free space and other temporary array(s) */
  ierr = PetscMalloc((apsymi[am]+1)*sizeof(PetscInt),&ptap->abj);CHKERRQ(ierr);
  apsymj = ptap->abj;
  ierr = MakeSpaceContiguous(&free_space,ptap->abj);CHKERRQ(ierr);

  /* get ij structure of P_loc^T */
  ierr = MatGetSymbolicTranspose_SeqAIJ(P_loc,&pti,&ptj);CHKERRQ(ierr);

  /* then, compute symbolic C_seq = P_loc^T*AP */
  /* Allocate ci array, arrays for fill computation and */
  /* free space for accumulating nonzero column info */
  ierr = PetscMalloc((pN+1)*sizeof(PetscInt),&ci);CHKERRQ(ierr);
  ci[0] = 0;

  /* tnzrows = 0; */
  tnrows = (p->B)->N; /* total num of rows to be sent to other processors 
                         >= (num of nonzero rows of C_seq) - pn */


  /* Set initial free space to be nnz(A) scaled by fill*P->N/P->M. */
  /* This should be reasonable if sparsity of PtAP is similar to that of A. */
  nnz           = adi[am] + aoi[am];
  ierr          = GetMoreSpace((PetscInt)(fill*nnz*pN/aN+1),&free_space);
  current_space = free_space;

  /* determine symbolic info for each row of C: */
  for (i=0; i<pN; i++) {
    cnzi  = 0;
    ptnzi = pti[i+1] - pti[i];
    if (ptnzi){
    j     = ptnzi;
    ptJ   = ptj + pti[i+1];
    while (j){/* assume cols are almost in increasing order, starting from its end saves computation */
      j--; ptJ--;
      arow  = *ptJ; /* row of AP == col of Pt */
      apnzj = apsymi[arow+1] - apsymi[arow];
      apj   = apsymj + apsymi[arow];
      /* add non-zero cols of AP into the sorted linked list lnk */
      ierr = PetscLLAdd(apnzj,apj,pN,nlnk,lnk,lnkbt);CHKERRQ(ierr);
      cnzi += nlnk;
    }

    /* If free space is not available, double the total space in the list */
    if (current_space->local_remaining<cnzi) {
      ierr = GetMoreSpace(current_space->total_array_size,&current_space);CHKERRQ(ierr);
    }

    /* Copy data into free space, and zero out denserows */
    ierr = PetscLLClean(pN,pN,cnzi,lnk,current_space->array,lnkbt);CHKERRQ(ierr);
    current_space->array           += cnzi;
    current_space->local_used      += cnzi;
    current_space->local_remaining -= cnzi;
    /* tnzrows++; */
    } /* if (ptnzi) */
    ci[i+1] = ci[i] + cnzi;

  }
  /* printf("[%d] tnzrows-pn: %d,tnrows: %d, pN: %d; pcstart/end: %d, %d\n",rank,tnzrows-pn,tnrows,pN,pcstart,pcend);*/

  /* Clean up. */
  ierr = MatRestoreSymbolicTranspose_SeqAIJ(P_loc,&pti,&ptj);CHKERRQ(ierr);

  /* Allocate space for cj, initialize cj, and */
  /* destroy list of free space and other temporary array(s) */
  ierr = PetscMalloc((ci[pN]+1)*sizeof(PetscInt),&cj);CHKERRQ(ierr);
  ierr = MakeSpaceContiguous(&free_space,cj);CHKERRQ(ierr);
  ierr = PetscLLDestroy(lnk,lnkbt);CHKERRQ(ierr);

  /* add C_seq into mpi C              */
  /*-----------------------------------*/
  /* determine row ownership */
  ierr = PetscNew(Mat_Merge_SeqsToMPI,&merge);CHKERRQ(ierr);
  ierr = PetscMapCreate(comm,&merge->rowmap);CHKERRQ(ierr);
  ierr = PetscMapSetLocalSize(merge->rowmap,pn);CHKERRQ(ierr); 
  ierr = PetscMapSetType(merge->rowmap,MAP_MPI);CHKERRQ(ierr);
  ierr = PetscMapGetGlobalRange(merge->rowmap,&owners);CHKERRQ(ierr);

  /* determine the number of messages to send, their lengths */
  /*---------------------------------------------------------*/
  ierr = PetscMalloc(size*sizeof(PetscMPIInt),&len_si);CHKERRQ(ierr);
  ierr = PetscMemzero(len_si,size*sizeof(PetscMPIInt));CHKERRQ(ierr);
  ierr = PetscMalloc(size*sizeof(PetscMPIInt),&merge->len_s);CHKERRQ(ierr);
  len_s  = merge->len_s;
  len = 0;  /* max length of buf_si[] */
  merge->nsend = 0;
  
  proc = 0;
  for (i=0; i<tnrows; i++){
    while (rmap[i] >= owners[proc+1]) proc++;
    len_si[proc]++;
  }
  for (proc=0; proc<size; proc++){
    len_s[proc] = 0;
    if (len_si[proc]){
      merge->nsend++;
      len_si[proc] = 2*(len_si[proc] + 1);
      len_s[proc] = ci[owners[proc+1]] - ci[owners[proc]]; /* num of col indices to be sent to [proc] */
      len += len_si[proc]; 
    }
  }

  /* determine the number and length of messages to receive for ij-structure */
  /*-------------------------------------------------------------------------*/
  ierr = PetscGatherNumberOfMessages(comm,PETSC_NULL,len_s,&merge->nrecv);CHKERRQ(ierr);
  ierr = PetscGatherMessageLengths2(comm,merge->nsend,merge->nrecv,len_s,len_si,&merge->id_r,&merge->len_r,&len_ri);CHKERRQ(ierr);      

  /* post the Irecv of j-structure */
  /*-------------------------------*/
  ierr = PetscObjectGetNewTag((PetscObject)merge->rowmap,&tagj);CHKERRQ(ierr);
  ierr = PetscPostIrecvInt(comm,tagj,merge->nrecv,merge->id_r,merge->len_r,&buf_rj,&rj_waits);CHKERRQ(ierr);

  /* post the Isend of j-structure */
  /*--------------------------------*/
  ierr = PetscMalloc((2*merge->nsend+1)*sizeof(MPI_Request),&si_waits);CHKERRQ(ierr);
  sj_waits = si_waits + merge->nsend;

  for (proc=0, k=0; proc<size; proc++){  
    if (!len_s[proc]) continue;
    i = owners[proc];
    ierr = MPI_Isend(cj+ci[i],len_s[proc],MPIU_INT,proc,tagj,comm,sj_waits+k);CHKERRQ(ierr);
    k++;
  } 

  /* receives and sends of j-structure are complete */
  /*------------------------------------------------*/
  ierr = PetscMalloc(size*sizeof(MPI_Status),&status);CHKERRQ(ierr);
  ierr = MPI_Waitall(merge->nrecv,rj_waits,status);CHKERRQ(ierr);
  ierr = MPI_Waitall(merge->nsend,sj_waits,status);CHKERRQ(ierr);
  
  /* send and recv i-structure */
  /*---------------------------*/  
  ierr = PetscObjectGetNewTag((PetscObject)merge->rowmap,&tagi);CHKERRQ(ierr);
  ierr = PetscPostIrecvInt(comm,tagi,merge->nrecv,merge->id_r,len_ri,&buf_ri,&ri_waits);CHKERRQ(ierr);
    
  ierr = PetscMalloc((len+1)*sizeof(PetscInt),&buf_s);CHKERRQ(ierr); 
  buf_si = buf_s;  /* points to the beginning of k-th msg to be sent */
  for (proc=0,k=0; proc<size; proc++){  
    if (!len_s[proc]) continue;
    /* form outgoing message for i-structure: 
         buf_si[0]:                 nrows to be sent
               [1:nrows]:           row index (global)
               [nrows+1:2*nrows+1]: i-structure index
    */
    /*-------------------------------------------*/      
    nrows = len_si[proc]/2 - 1; 
    buf_si_i    = buf_si + nrows+1;
    buf_si[0]   = nrows;
    buf_si_i[0] = 0;
    nrows = 0;
    for (i=owners[proc]; i<owners[proc+1]; i++){
      anzi = ci[i+1] - ci[i];
      if (anzi) {
        buf_si_i[nrows+1] = buf_si_i[nrows] + anzi; /* i-structure */
        buf_si[nrows+1] = i-owners[proc]; /* local row index */
        nrows++;
      }
    }
    ierr = MPI_Isend(buf_si,len_si[proc],MPIU_INT,proc,tagi,comm,si_waits+k);CHKERRQ(ierr);
    k++;
    buf_si += len_si[proc];
  } 

  ierr = MPI_Waitall(merge->nrecv,ri_waits,status);CHKERRQ(ierr);
  ierr = MPI_Waitall(merge->nsend,si_waits,status);CHKERRQ(ierr);

  ierr = PetscLogInfo((PetscObject)A,"MatMerge_SeqsToMPI: nsend: %d, nrecv: %d\n",merge->nsend,merge->nrecv);CHKERRQ(ierr);
  for (i=0; i<merge->nrecv; i++){
    ierr = PetscLogInfo((PetscObject)A,"MatMerge_SeqsToMPI:   recv len_ri=%d, len_rj=%d from [%d]\n",len_ri[i],merge->len_r[i],merge->id_r[i]);CHKERRQ(ierr);
  }

  ierr = PetscFree(len_si);CHKERRQ(ierr);
  ierr = PetscFree(len_ri);CHKERRQ(ierr);
  ierr = PetscFree(rj_waits);CHKERRQ(ierr);
  ierr = PetscFree(si_waits);CHKERRQ(ierr);
  ierr = PetscFree(ri_waits);CHKERRQ(ierr);
  ierr = PetscFree(buf_s);CHKERRQ(ierr);
  ierr = PetscFree(status);CHKERRQ(ierr);

  /* compute a local seq matrix in each processor */
  /*----------------------------------------------*/
  /* allocate bi array and free space for accumulating nonzero column info */
  ierr = PetscMalloc((pn+1)*sizeof(PetscInt),&bi);CHKERRQ(ierr);
  bi[0] = 0;

  /* create and initialize a linked list */
  nlnk = pN+1;
  ierr = PetscLLCreate(pN,pN,nlnk,lnk,lnkbt);CHKERRQ(ierr);
  
  /* initial FreeSpace size is 2*(num of local nnz(C_seq)) */
  free_space=PETSC_NULL; current_space=PETSC_NULL;
  len = 0;
  len  = ci[owners[rank+1]] - ci[owners[rank]];
  ierr = GetMoreSpace((PetscInt)(2*len+1),&free_space);CHKERRQ(ierr);
  current_space = free_space;

  /* determine symbolic info for each local row */
  ierr = PetscMalloc((3*merge->nrecv+1)*sizeof(PetscInt**),&buf_ri_k);CHKERRQ(ierr);
  nextrow = buf_ri_k + merge->nrecv;
  nextci  = nextrow + merge->nrecv;
  for (k=0; k<merge->nrecv; k++){
    buf_ri_k[k] = buf_ri[k]; /* beginning of k-th recved i-structure */
    nrows = *buf_ri_k[k];
    nextrow[k]  = buf_ri_k[k] + 1;  /* next row number of k-th recved i-structure */
    nextci[k]   = buf_ri_k[k] + (nrows + 1);/* poins to the next i-structure of k-th recved i-structure  */
  }

  ierr = MatPreallocateInitialize(comm,pn,pn,dnz,onz);CHKERRQ(ierr);
  len = 0;  
  for (i=0; i<pn; i++) {
    bnzi   = 0;
    /* add local non-zero cols of this proc's C_seq into lnk */
    arow   = owners[rank] + i;
    anzi   = ci[arow+1] - ci[arow];
    cji    = cj + ci[arow]; 
    ierr = PetscLLAdd(anzi,cji,pN,nlnk,lnk,lnkbt);CHKERRQ(ierr);
    bnzi += nlnk;
    /* add received col data into lnk */
    for (k=0; k<merge->nrecv; k++){ /* k-th received message */
      if (i == *nextrow[k]) { /* i-th row */
        anzi = *(nextci[k]+1) - *nextci[k]; 
        cji  = buf_rj[k] + *nextci[k];
        ierr = PetscLLAdd(anzi,cji,pN,nlnk,lnk,lnkbt);CHKERRQ(ierr);
        bnzi += nlnk;
        nextrow[k]++; nextci[k]++;
      }
    }
    if (len < bnzi) len = bnzi;  /* =max(bnzi) */

    /* if free space is not available, make more free space */
    if (current_space->local_remaining<bnzi) {
      ierr = GetMoreSpace(current_space->total_array_size,&current_space);CHKERRQ(ierr);
      nspacedouble++;
    }
    /* copy data into free space, then initialize lnk */
    ierr = PetscLLClean(pN,pN,bnzi,lnk,current_space->array,lnkbt);CHKERRQ(ierr);
    ierr = MatPreallocateSet(i+owners[rank],bnzi,current_space->array,dnz,onz);CHKERRQ(ierr);

    current_space->array           += bnzi;
    current_space->local_used      += bnzi;
    current_space->local_remaining -= bnzi;
   
    bi[i+1] = bi[i] + bnzi;
  }
  
  ierr = PetscFree(buf_ri_k);CHKERRQ(ierr);

  ierr = PetscMalloc((bi[pn]+1)*sizeof(PetscInt),&bj);CHKERRQ(ierr);
  ierr = MakeSpaceContiguous(&free_space,bj);CHKERRQ(ierr);
  ierr = PetscLLDestroy(lnk,lnkbt);CHKERRQ(ierr);

  /* create symbolic parallel matrix B_mpi */
  /*---------------------------------------*/
  ierr = MatCreate(comm,pn,pn,PETSC_DETERMINE,PETSC_DETERMINE,&B_mpi);CHKERRQ(ierr);
  ierr = MatSetType(B_mpi,MATMPIAIJ);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(B_mpi,0,dnz,0,onz);CHKERRQ(ierr);
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);
  /* ierr = MatSetOption(B_mpi,MAT_COLUMNS_SORTED);CHKERRQ(ierr); -cause delay? */

  /* B_mpi is not ready for use - assembly will be done by MatMerge_SeqsToMPINumeric() */
  B_mpi->assembled     = PETSC_FALSE; 
  B_mpi->ops->destroy  = MatDestroy_MPIAIJ_SeqsToMPI;  
  merge->bi            = bi;
  merge->bj            = bj;
  merge->ci            = ci;
  merge->cj            = cj;
  merge->buf_ri        = buf_ri;
  merge->buf_rj        = buf_rj;
  merge->C_seq         = PETSC_NULL;

  /* attach the supporting struct to B_mpi for reuse */
  ierr = PetscObjectContainerCreate(PETSC_COMM_SELF,&container);CHKERRQ(ierr);
  ierr = PetscObjectContainerSetPointer(container,merge);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)B_mpi,"MatMergeSeqsToMPI",(PetscObject)container);CHKERRQ(ierr);
  *C = B_mpi;
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatPtAPNumeric_MPIAIJ_MPIAIJ"
PetscErrorCode MatPtAPNumeric_MPIAIJ_MPIAIJ(Mat A,Mat P,Mat C)
{
  PetscErrorCode       ierr;
  Mat_Merge_SeqsToMPI  *merge; 
  Mat_MatMatMultMPI    *ptap;
  PetscObjectContainer cont_merge,cont_ptap;

  PetscInt       flops=0;
  Mat_MPIAIJ     *a=(Mat_MPIAIJ*)A->data;
  Mat_SeqAIJ     *ad=(Mat_SeqAIJ*)(a->A)->data,*ao=(Mat_SeqAIJ*)(a->B)->data;
  Mat_SeqAIJ     *p_loc,*p_oth; 
  PetscInt       *adi=ad->i,*aoi=ao->i,*apj,cstart=a->cstart,cend=a->cend,col; 
  PetscInt       *pi_loc,*pj_loc,*pi_oth,*pj_oth,*pJ,*pjj;
  PetscInt       i,j,k,anzi,pnzi,apnzj,nextap,pnzj,prow,crow;
  PetscInt       *cjj;
  MatScalar      *ada=ad->a,*aoa=ao->a,*apa,*paj,*cseqa,*caj; 
  MatScalar      *pa_loc,*pA,*pa_oth;
  PetscInt       am=A->m,cN=C->N; 
  PetscInt       nextp,*adj=ad->j,*aoj=ao->j;
  MPI_Comm             comm=C->comm;
  PetscMPIInt          size,rank,taga,*len_s;
  PetscInt             *owners; 
  PetscInt             proc;
  PetscInt             **buf_ri,**buf_rj;  
  PetscInt             cseqnzi,*bj_i,*bi,*bj,cseqrow,bnzi,nextcseqj; /* bi, bj, ba: for C (mpi mat) */
  PetscInt             nrows,**buf_ri_k,**nextrow,**nextcseqi;
  MPI_Request          *s_waits,*r_waits; 
  MPI_Status           *status;
  MatScalar            **abuf_r,*ba_i;
  PetscInt             *cseqi,*cseqj;
  PetscInt             *cseqj_tmp;
  MatScalar            *cseqa_tmp;
  PetscInt             stages[2];
  PetscInt             *apsymi,*apsymj; 

  PetscFunctionBegin;
  ierr = PetscLogStageRegister(&stages[0],"NumPtAP_local");CHKERRQ(ierr);
  ierr = PetscLogStageRegister(&stages[1],"NumPtAP_Comm");CHKERRQ(ierr);

  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  ierr = PetscObjectQuery((PetscObject)C,"MatMergeSeqsToMPI",(PetscObject *)&cont_merge);CHKERRQ(ierr);
  if (cont_merge) { 
    ierr  = PetscObjectContainerGetPointer(cont_merge,(void **)&merge);CHKERRQ(ierr); 
  } else {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "Matrix C does not posses an object container");
  } 
  ierr = PetscObjectQuery((PetscObject)P,"MatPtAPMPI",(PetscObject *)&cont_ptap);CHKERRQ(ierr);
  if (cont_ptap) { 
    ierr  = PetscObjectContainerGetPointer(cont_ptap,(void **)&ptap);CHKERRQ(ierr); 
  } else {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "Matrix P does not posses an object container");
  } 
 
  /* get data from symbolic products */
  p_loc=(Mat_SeqAIJ*)(ptap->B_loc)->data;
  p_oth=(Mat_SeqAIJ*)(ptap->B_oth)->data;
  pi_loc=p_loc->i; pj_loc=p_loc->j; pJ=pj_loc; pa_loc=p_loc->a; pA=pa_loc; 
  pi_oth=p_oth->i; pj_oth=p_oth->j; pa_oth=p_oth->a;

  cseqi = merge->ci; cseqj=merge->cj;
  ierr  = PetscMalloc((cseqi[cN]+1)*sizeof(MatScalar),&cseqa);CHKERRQ(ierr);
  ierr  = PetscMemzero(cseqa,cseqi[cN]*sizeof(MatScalar));CHKERRQ(ierr);

  /* get data from symbolic A*P */ 
  ierr = PetscMalloc((ptap->abnz_max+1)*sizeof(MatScalar),&apa);CHKERRQ(ierr);
  ierr = PetscMemzero(apa,ptap->abnz_max*sizeof(MatScalar));CHKERRQ(ierr);

  /* compute numeric C_seq=P_loc^T*A_loc*P */
  ierr = PetscLogStagePush(stages[0]);CHKERRQ(ierr);
  apsymi = ptap->abi; apsymj = ptap->abj;
  for (i=0;i<am;i++) {
    /* form i-th sparse row of A*P */
    apnzj = apsymi[i+1] - apsymi[i];
    apj   = apsymj + apsymi[i];
    /* diagonal portion of A */
    anzi  = adi[i+1] - adi[i];
    for (j=0;j<anzi;j++) {
      prow = *adj; 
      adj++;
      pnzj = pi_loc[prow+1] - pi_loc[prow];
      pjj  = pj_loc + pi_loc[prow];
      paj  = pa_loc + pi_loc[prow];
      nextp = 0;
      for (k=0; nextp<pnzj; k++) {
        if (apj[k] == pjj[nextp]) { /* col of AP == col of P */
          apa[k] += (*ada)*paj[nextp++];
        }
      }
      flops += 2*pnzj;
      ada++;
    }
    /* off-diagonal portion of A */
    anzi  = aoi[i+1] - aoi[i];
    for (j=0;j<anzi;j++) {
      col = a->garray[*aoj];
      if (col < cstart){
        prow = *aoj;
      } else if (col >= cend){
        prow = *aoj; 
      } else {
        SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Off-diagonal portion of A has wrong column map");
      }
      aoj++;
      pnzj = pi_oth[prow+1] - pi_oth[prow];
      pjj  = pj_oth + pi_oth[prow];
      paj  = pa_oth + pi_oth[prow];
      nextp = 0;
      for (k=0; nextp<pnzj; k++) {
        if (apj[k] == pjj[nextp]) { /* col of AP == col of P */
          apa[k] += (*aoa)*paj[nextp++];
        }
      }
      flops += 2*pnzj;
      aoa++;
    }

    /* Compute P_loc[i,:]^T*AP[i,:] using outer product */
    pnzi = pi_loc[i+1] - pi_loc[i];
    for (j=0;j<pnzi;j++) {
      nextap = 0;
      crow   = *pJ++;  
      cjj    = cseqj + cseqi[crow];
      caj    = cseqa + cseqi[crow];
      /* Perform sparse axpy operation.  Note cjj includes apj. */
      for (k=0;nextap<apnzj;k++) {
        if (cjj[k]==apj[nextap]) caj[k] += (*pA)*apa[nextap++]; 
      }
      flops += 2*apnzj;
      pA++;
    }
    /* zero the current row info for A*P */
    ierr = PetscMemzero(apa,apnzj*sizeof(MatScalar));CHKERRQ(ierr);
  }

  ierr = PetscFree(apa);CHKERRQ(ierr);
  ierr = PetscLogStagePop();
  
  bi     = merge->bi;
  bj     = merge->bj;
  buf_ri = merge->buf_ri;
  buf_rj = merge->buf_rj;
  
  ierr   = PetscMalloc(size*sizeof(MPI_Status),&status);CHKERRQ(ierr);
  ierr   = PetscMapGetGlobalRange(merge->rowmap,&owners);CHKERRQ(ierr);
  
  /* send and recv matrix values */
  /*-----------------------------*/
  ierr = PetscLogStagePush(stages[1]);CHKERRQ(ierr);
  len_s  = merge->len_s;
  ierr = PetscObjectGetNewTag((PetscObject)merge->rowmap,&taga);CHKERRQ(ierr);
  ierr = PetscPostIrecvScalar(comm,taga,merge->nrecv,merge->id_r,merge->len_r,&abuf_r,&r_waits);CHKERRQ(ierr);

  ierr = PetscMalloc((merge->nsend+1)*sizeof(MPI_Request),&s_waits);CHKERRQ(ierr);
  for (proc=0,k=0; proc<size; proc++){  
    if (!len_s[proc]) continue;
    i = owners[proc];
    ierr = MPI_Isend(cseqa+cseqi[i],len_s[proc],MPIU_MATSCALAR,proc,taga,comm,s_waits+k);CHKERRQ(ierr);
    k++;
  } 

  ierr = MPI_Waitall(merge->nrecv,r_waits,status);CHKERRQ(ierr);
  ierr = MPI_Waitall(merge->nsend,s_waits,status);CHKERRQ(ierr);
  ierr = PetscFree(status);CHKERRQ(ierr);

  ierr = PetscFree(s_waits);CHKERRQ(ierr);
  ierr = PetscFree(r_waits);CHKERRQ(ierr);

  /* insert mat values of mpimat */
  /*----------------------------*/
  ierr = PetscMalloc(cN*sizeof(MatScalar),&ba_i);CHKERRQ(ierr);
  ierr = PetscMalloc((3*merge->nrecv+1)*sizeof(PetscInt**),&buf_ri_k);CHKERRQ(ierr);
  nextrow = buf_ri_k + merge->nrecv;
  nextcseqi  = nextrow + merge->nrecv;

  for (k=0; k<merge->nrecv; k++){
    buf_ri_k[k] = buf_ri[k]; /* beginning of k-th recved i-structure */
    nrows = *(buf_ri_k[k]);
    nextrow[k]  = buf_ri_k[k]+1;  /* next row number of k-th recved i-structure */
    nextcseqi[k]   = buf_ri_k[k] + (nrows + 1);/* poins to the next i-structure of k-th recved i-structure  */
  }

  /* set values of ba */
  for (i=0; i<C->m; i++) {
    cseqrow = owners[rank] + i; /* global row index of C_seq */
    bj_i = bj+bi[i];  /* col indices of the i-th row of C */
    bnzi = bi[i+1] - bi[i];
    ierr = PetscMemzero(ba_i,bnzi*sizeof(MatScalar));CHKERRQ(ierr);

    /* add local non-zero vals of this proc's C_seq into ba */
    cseqnzi = cseqi[cseqrow+1] - cseqi[cseqrow];
    cseqj_tmp = cseqj + cseqi[cseqrow]; 
    cseqa_tmp = cseqa + cseqi[cseqrow]; 
    nextcseqj = 0;
    for (j=0; nextcseqj<cseqnzi; j++){
      if (*(bj_i + j) == cseqj_tmp[nextcseqj]){ /* bcol == cseqcol */
        ba_i[j] += cseqa_tmp[nextcseqj++];
      }
    }

    /* add received vals into ba */
    for (k=0; k<merge->nrecv; k++){ /* k-th received message */
      /* i-th row */
      if (i == *nextrow[k]) {
        cseqnzi   = *(nextcseqi[k]+1) - *nextcseqi[k]; 
        cseqj_tmp = buf_rj[k] + *(nextcseqi[k]);
        cseqa_tmp = abuf_r[k] + *(nextcseqi[k]);
        nextcseqj = 0;
        for (j=0; nextcseqj<cseqnzi; j++){ 
          if (*(bj_i + j) == cseqj_tmp[nextcseqj]){ /* bcol == cseqcol */
            ba_i[j] += cseqa_tmp[nextcseqj++];
          }
        }
        nextrow[k]++; nextcseqi[k]++;
      } 
    }
    ierr = MatSetValues(C,1,&cseqrow,bnzi,bj_i,ba_i,INSERT_VALUES);CHKERRQ(ierr);
    flops += 2*bnzi;
  } 
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr); 
  ierr = PetscLogStagePop();CHKERRQ(ierr);

  ierr = PetscFree(cseqa);CHKERRQ(ierr);
  ierr = PetscFree(abuf_r);CHKERRQ(ierr);
  ierr = PetscFree(ba_i);CHKERRQ(ierr);
  ierr = PetscFree(buf_ri_k);CHKERRQ(ierr);
  ierr = PetscLogFlops(flops);CHKERRQ(ierr);
 
  PetscFunctionReturn(0);
}
