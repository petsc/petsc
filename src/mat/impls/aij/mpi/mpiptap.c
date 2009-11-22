#define PETSCMAT_DLL

/*
  Defines projective product routines where A is a MPIAIJ matrix
          C = P^T * A * P
*/

#include "../src/mat/impls/aij/seq/aij.h"   /*I "petscmat.h" I*/
#include "../src/mat/utils/freespace.h"
#include "../src/mat/impls/aij/mpi/mpiaij.h"
#include "petscbt.h"

EXTERN PetscErrorCode MatDestroy_MPIAIJ(Mat);
#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_MPIAIJ_MatPtAP"
PetscErrorCode PETSCMAT_DLLEXPORT MatDestroy_MPIAIJ_MatPtAP(Mat A)
{
  PetscErrorCode       ierr;
  Mat_Merge_SeqsToMPI  *merge; 
  PetscContainer       container;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)A,"MatMergeSeqsToMPI",(PetscObject *)&container);CHKERRQ(ierr);
  if (container) {
    ierr = PetscContainerGetPointer(container,(void **)&merge);CHKERRQ(ierr); 
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
    ierr = PetscLayoutDestroy(merge->rowmap);CHKERRQ(ierr);
    
    ierr = PetscContainerDestroy(container);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)A,"MatMergeSeqsToMPI",0);CHKERRQ(ierr);
  }
  ierr = merge->MatDestroy(A);CHKERRQ(ierr);
  ierr = PetscFree(merge);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDuplicate_MPIAIJ_MatPtAP"
PetscErrorCode MatDuplicate_MPIAIJ_MatPtAP(Mat A, MatDuplicateOption op, Mat *M) 
{
  PetscErrorCode       ierr;
  Mat_Merge_SeqsToMPI  *merge; 
  PetscContainer       container;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)A,"MatMergeSeqsToMPI",(PetscObject *)&container);CHKERRQ(ierr);
  if (container) {
    ierr  = PetscContainerGetPointer(container,(void **)&merge);CHKERRQ(ierr); 
  } else {
    SETERRQ(PETSC_ERR_PLIB,"Container does not exit");
  }
  ierr = (*merge->MatDuplicate)(A,op,M);CHKERRQ(ierr);
  (*M)->ops->destroy   = merge->MatDestroy;   /* =MatDestroy_MPIAIJ, *M doesn't duplicate A's container! */
  (*M)->ops->duplicate = merge->MatDuplicate; /* =MatDuplicate_ MPIAIJ */
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPtAPSymbolic_MPIAIJ"
PetscErrorCode MatPtAPSymbolic_MPIAIJ(Mat A,Mat P,PetscReal fill,Mat *C)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!P->ops->ptapsymbolic_mpiaij) {
    SETERRQ2(PETSC_ERR_SUP,"Not implemented for A=%s and P=%s",((PetscObject)A)->type_name,((PetscObject)P)->type_name);
  }
  ierr = (*P->ops->ptapsymbolic_mpiaij)(A,P,fill,C);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPtAPNumeric_MPIAIJ"
PetscErrorCode MatPtAPNumeric_MPIAIJ(Mat A,Mat P,Mat C)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!P->ops->ptapnumeric_mpiaij) {
    SETERRQ2(PETSC_ERR_SUP,"Not implemented for A=%s and P=%s",((PetscObject)A)->type_name,((PetscObject)P)->type_name);
  }
  ierr = (*P->ops->ptapnumeric_mpiaij)(A,P,C);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatPtAPSymbolic_MPIAIJ_MPIAIJ"
PetscErrorCode MatPtAPSymbolic_MPIAIJ_MPIAIJ(Mat A,Mat P,PetscReal fill,Mat *C)
{
  PetscErrorCode       ierr;
  Mat                  B_mpi; 
  Mat_MatMatMultMPI    *ap;
  PetscContainer       container;
  PetscFreeSpaceList   free_space=PETSC_NULL,current_space=PETSC_NULL;
  Mat_MPIAIJ           *a=(Mat_MPIAIJ*)A->data,*p=(Mat_MPIAIJ*)P->data;
  Mat_SeqAIJ           *ad=(Mat_SeqAIJ*)(a->A)->data,*ao=(Mat_SeqAIJ*)(a->B)->data;
  Mat_SeqAIJ           *p_loc,*p_oth;
  PetscInt             *pi_loc,*pj_loc,*pi_oth,*pj_oth,*pdti,*pdtj,*poti,*potj,*ptJ;
  PetscInt             *adi=ad->i,*adj=ad->j,*aoi=ao->i,*aoj=ao->j,nnz; 
  PetscInt             nlnk,*lnk,*owners_co,*coi,*coj,i,k,pnz,row;
  PetscInt             am=A->rmap->n,pN=P->cmap->N,pn=P->cmap->n;  
  PetscBT              lnkbt;
  MPI_Comm             comm=((PetscObject)A)->comm;
  PetscMPIInt          size,rank,tag,*len_si,*len_s,*len_ri; 
  PetscInt             **buf_rj,**buf_ri,**buf_ri_k;
  PetscInt             len,proc,*dnz,*onz,*owners;
  PetscInt             nzi,*bi,*bj; 
  PetscInt             nrows,*buf_s,*buf_si,*buf_si_i,**nextrow,**nextci;
  MPI_Request          *swaits,*rwaits; 
  MPI_Status           *sstatus,rstatus;
  Mat_Merge_SeqsToMPI  *merge;
  PetscInt             *api,*apj,*Jptr,apnz,*prmap=p->garray,pon,nspacedouble=0;
  PetscMPIInt          j;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  /* destroy the container 'Mat_MatMatMultMPI' in case that P is attached to it */
  ierr = PetscObjectQuery((PetscObject)P,"Mat_MatMatMultMPI",(PetscObject *)&container);CHKERRQ(ierr);
  if (container) { 
    /* reset functions */
    ierr = PetscContainerGetPointer(container,(void **)&ap);CHKERRQ(ierr);
    P->ops->destroy = ap->MatDestroy;
    P->ops->duplicate = ap->MatDuplicate;
    /* destroy container and contents */
    ierr = PetscContainerDestroy(container);CHKERRQ(ierr); 
    ierr = PetscObjectCompose((PetscObject)P,"Mat_MatMatMultMPI",0);CHKERRQ(ierr);
  }

  /* create the container 'Mat_MatMatMultMPI' and attach it to P */
  ierr = PetscNew(Mat_MatMatMultMPI,&ap);CHKERRQ(ierr);
  ap->abi=PETSC_NULL; ap->abj=PETSC_NULL; 
  ap->abnz_max = 0; 

  ierr = PetscContainerCreate(PETSC_COMM_SELF,&container);CHKERRQ(ierr);
  ierr = PetscContainerSetPointer(container,ap);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)P,"Mat_MatMatMultMPI",(PetscObject)container);CHKERRQ(ierr);
  ap->MatDestroy  = P->ops->destroy;
  P->ops->destroy = MatDestroy_MPIAIJ_MatMatMult;
  ap->reuse       = MAT_INITIAL_MATRIX;
  ierr = PetscContainerSetUserDestroy(container,PetscContainerDestroy_Mat_MatMatMultMPI);CHKERRQ(ierr);

  /* get P_oth by taking rows of P (= non-zero cols of local A) from other processors */
  ierr = MatGetBrowsOfAoCols(A,P,MAT_INITIAL_MATRIX,&ap->startsj,&ap->startsj_r,&ap->bufa,&ap->B_oth);CHKERRQ(ierr);
  /* get P_loc by taking all local rows of P */
  ierr = MatGetLocalMat(P,MAT_INITIAL_MATRIX,&ap->B_loc);CHKERRQ(ierr);

  p_loc = (Mat_SeqAIJ*)(ap->B_loc)->data; 
  p_oth = (Mat_SeqAIJ*)(ap->B_oth)->data;
  pi_loc = p_loc->i; pj_loc = p_loc->j; 
  pi_oth = p_oth->i; pj_oth = p_oth->j;

  /* first, compute symbolic AP = A_loc*P = A_diag*P_loc + A_off*P_oth */
  /*-------------------------------------------------------------------*/
  ierr  = PetscMalloc((am+2)*sizeof(PetscInt),&api);CHKERRQ(ierr);
  ap->abi = api;
  api[0] = 0;

  /* create and initialize a linked list */
  nlnk = pN+1;
  ierr = PetscLLCreate(pN,pN,nlnk,lnk,lnkbt);CHKERRQ(ierr);

  /* Initial FreeSpace size is fill*nnz(A) */
  ierr = PetscFreeSpaceGet((PetscInt)(fill*(adi[am]+aoi[am])),&free_space);CHKERRQ(ierr);
  current_space = free_space; 

  for (i=0;i<am;i++) {
    apnz = 0;
    /* diagonal portion of A */
    nzi = adi[i+1] - adi[i];
    for (j=0; j<nzi; j++){
      row = *adj++; 
      pnz = pi_loc[row+1] - pi_loc[row];
      Jptr  = pj_loc + pi_loc[row];
      /* add non-zero cols of P into the sorted linked list lnk */
      ierr = PetscLLAdd(pnz,Jptr,pN,nlnk,lnk,lnkbt);CHKERRQ(ierr);
      apnz += nlnk;
    }
    /* off-diagonal portion of A */
    nzi = aoi[i+1] - aoi[i];
    for (j=0; j<nzi; j++){   
      row = *aoj++; 
      pnz = pi_oth[row+1] - pi_oth[row];
      Jptr  = pj_oth + pi_oth[row];  
      ierr = PetscLLAdd(pnz,Jptr,pN,nlnk,lnk,lnkbt);CHKERRQ(ierr);
      apnz += nlnk;
    }

    api[i+1] = api[i] + apnz;
    if (ap->abnz_max < apnz) ap->abnz_max = apnz;

    /* if free space is not available, double the total space in the list */
    if (current_space->local_remaining<apnz) {
      ierr = PetscFreeSpaceGet(apnz+current_space->total_array_size,&current_space);CHKERRQ(ierr);
      nspacedouble++;
    }

    /* Copy data into free space, then initialize lnk */
    ierr = PetscLLClean(pN,pN,apnz,lnk,current_space->array,lnkbt);CHKERRQ(ierr); 
    current_space->array           += apnz;
    current_space->local_used      += apnz;
    current_space->local_remaining -= apnz;
  }
  /* Allocate space for apj, initialize apj, and */
  /* destroy list of free space and other temporary array(s) */
  ierr = PetscMalloc((api[am]+1)*sizeof(PetscInt),&ap->abj);CHKERRQ(ierr);
  apj = ap->abj;
  ierr = PetscFreeSpaceContiguous(&free_space,ap->abj);CHKERRQ(ierr);

  /* determine symbolic Co=(p->B)^T*AP - send to others */
  /*----------------------------------------------------*/
  ierr = MatGetSymbolicTranspose_SeqAIJ(p->B,&poti,&potj);CHKERRQ(ierr);

  /* then, compute symbolic Co = (p->B)^T*AP */
  pon = (p->B)->cmap->n; /* total num of rows to be sent to other processors 
                         >= (num of nonzero rows of C_seq) - pn */
  ierr = PetscMalloc((pon+1)*sizeof(PetscInt),&coi);CHKERRQ(ierr);
  coi[0] = 0;

  /* set initial free space to be 3*pon*max( nnz(AP) per row) */
  nnz           = 3*pon*ap->abnz_max + 1;
  ierr          = PetscFreeSpaceGet(nnz,&free_space);
  current_space = free_space;

  for (i=0; i<pon; i++) {
    nnz  = 0;
    pnz = poti[i+1] - poti[i];
    j     = pnz;
    ptJ   = potj + poti[i+1];
    while (j){/* assume cols are almost in increasing order, starting from its end saves computation */
      j--; ptJ--;
      row  = *ptJ; /* row of AP == col of Pot */
      apnz = api[row+1] - api[row];
      Jptr   = apj + api[row];
      /* add non-zero cols of AP into the sorted linked list lnk */
      ierr = PetscLLAdd(apnz,Jptr,pN,nlnk,lnk,lnkbt);CHKERRQ(ierr);
      nnz += nlnk;
    }

    /* If free space is not available, double the total space in the list */
    if (current_space->local_remaining<nnz) {
      ierr = PetscFreeSpaceGet(nnz+current_space->total_array_size,&current_space);CHKERRQ(ierr);
    }

    /* Copy data into free space, and zero out denserows */
    ierr = PetscLLClean(pN,pN,nnz,lnk,current_space->array,lnkbt);CHKERRQ(ierr);
    current_space->array           += nnz;
    current_space->local_used      += nnz;
    current_space->local_remaining -= nnz;
    coi[i+1] = coi[i] + nnz;
  }
  ierr = PetscMalloc((coi[pon]+1)*sizeof(PetscInt),&coj);CHKERRQ(ierr);
  ierr = PetscFreeSpaceContiguous(&free_space,coj);CHKERRQ(ierr);
  ierr = MatRestoreSymbolicTranspose_SeqAIJ(p->B,&poti,&potj);CHKERRQ(ierr);

  /* send j-array (coj) of Co to other processors */
  /*----------------------------------------------*/
  /* determine row ownership */
  ierr = PetscNew(Mat_Merge_SeqsToMPI,&merge);CHKERRQ(ierr);
  ierr = PetscLayoutCreate(comm,&merge->rowmap);CHKERRQ(ierr);
  merge->rowmap->n = pn;
  merge->rowmap->bs = 1;
  ierr = PetscLayoutSetUp(merge->rowmap);CHKERRQ(ierr);
  owners = merge->rowmap->range;

  /* determine the number of messages to send, their lengths */
  ierr = PetscMalloc(size*sizeof(PetscMPIInt),&len_si);CHKERRQ(ierr);
  ierr = PetscMemzero(len_si,size*sizeof(PetscMPIInt));CHKERRQ(ierr);
  ierr = PetscMalloc(size*sizeof(PetscMPIInt),&merge->len_s);CHKERRQ(ierr);
  len_s = merge->len_s;
  merge->nsend = 0;
  
  ierr = PetscMalloc((size+2)*sizeof(PetscInt),&owners_co);CHKERRQ(ierr);
  ierr = PetscMemzero(len_s,size*sizeof(PetscMPIInt));CHKERRQ(ierr);

  proc = 0;
  for (i=0; i<pon; i++){
    while (prmap[i] >= owners[proc+1]) proc++;
    len_si[proc]++;  /* num of rows in Co to be sent to [proc] */
    len_s[proc] += coi[i+1] - coi[i];
  }

  len   = 0;  /* max length of buf_si[] */
  owners_co[0] = 0;
  for (proc=0; proc<size; proc++){
    owners_co[proc+1] = owners_co[proc] + len_si[proc];
    if (len_si[proc]){ 
      merge->nsend++;
      len_si[proc] = 2*(len_si[proc] + 1);
      len += len_si[proc]; 
    }
  }

  /* determine the number and length of messages to receive for coi and coj  */
  ierr = PetscGatherNumberOfMessages(comm,PETSC_NULL,len_s,&merge->nrecv);CHKERRQ(ierr);
  ierr = PetscGatherMessageLengths2(comm,merge->nsend,merge->nrecv,len_s,len_si,&merge->id_r,&merge->len_r,&len_ri);CHKERRQ(ierr);      

  /* post the Irecv and Isend of coj */
  ierr = PetscCommGetNewTag(comm,&tag);CHKERRQ(ierr);
  ierr = PetscPostIrecvInt(comm,tag,merge->nrecv,merge->id_r,merge->len_r,&buf_rj,&rwaits);CHKERRQ(ierr);

  ierr = PetscMalloc((merge->nsend+1)*sizeof(MPI_Request),&swaits);CHKERRQ(ierr);

  for (proc=0, k=0; proc<size; proc++){  
    if (!len_s[proc]) continue;
    i = owners_co[proc];
    ierr = MPI_Isend(coj+coi[i],len_s[proc],MPIU_INT,proc,tag,comm,swaits+k);CHKERRQ(ierr);
    k++;
  } 

  /* receives and sends of coj are complete */
  ierr = PetscMalloc(size*sizeof(MPI_Status),&sstatus);CHKERRQ(ierr); 
  i = merge->nrecv;
  while (i--) {
    ierr = MPI_Waitany(merge->nrecv,rwaits,&j,&rstatus);CHKERRQ(ierr);
  }
  ierr = PetscFree(rwaits);CHKERRQ(ierr);
  if (merge->nsend) {ierr = MPI_Waitall(merge->nsend,swaits,sstatus);CHKERRQ(ierr);}
  
  /* send and recv coi */
  /*-------------------*/  
  ierr = PetscPostIrecvInt(comm,tag,merge->nrecv,merge->id_r,len_ri,&buf_ri,&rwaits);CHKERRQ(ierr);
    
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
    for (i=owners_co[proc]; i<owners_co[proc+1]; i++){
      nzi = coi[i+1] - coi[i];
      buf_si_i[nrows+1] = buf_si_i[nrows] + nzi; /* i-structure */
      buf_si[nrows+1] =prmap[i] -owners[proc]; /* local row index */
      nrows++;
    }
    ierr = MPI_Isend(buf_si,len_si[proc],MPIU_INT,proc,tag,comm,swaits+k);CHKERRQ(ierr);
    k++;
    buf_si += len_si[proc];
  } 
  i = merge->nrecv;
  while (i--) {
    ierr = MPI_Waitany(merge->nrecv,rwaits,&j,&rstatus);CHKERRQ(ierr);
  }
  ierr = PetscFree(rwaits);CHKERRQ(ierr);
  if (merge->nsend) {ierr = MPI_Waitall(merge->nsend,swaits,sstatus);CHKERRQ(ierr);}
  /*
  ierr = PetscInfo2(A,"nsend: %d, nrecv: %d\n",merge->nsend,merge->nrecv);CHKERRQ(ierr);
  for (i=0; i<merge->nrecv; i++){
    ierr = PetscInfo3(A,"recv len_ri=%d, len_rj=%d from [%d]\n",len_ri[i],merge->len_r[i],merge->id_r[i]);CHKERRQ(ierr);
  }
  */
  ierr = PetscFree(len_si);CHKERRQ(ierr);
  ierr = PetscFree(len_ri);CHKERRQ(ierr);
  ierr = PetscFree(swaits);CHKERRQ(ierr);
  ierr = PetscFree(sstatus);CHKERRQ(ierr);
  ierr = PetscFree(buf_s);CHKERRQ(ierr);

  /* compute the local portion of C (mpi mat) */
  /*------------------------------------------*/
  ierr = MatGetSymbolicTranspose_SeqAIJ(p->A,&pdti,&pdtj);CHKERRQ(ierr);

  /* allocate bi array and free space for accumulating nonzero column info */
  ierr = PetscMalloc((pn+1)*sizeof(PetscInt),&bi);CHKERRQ(ierr);
  bi[0] = 0;
  
  /* set initial free space to be 3*pn*max( nnz(AP) per row) */
  nnz           = 3*pn*ap->abnz_max + 1; 
  ierr          = PetscFreeSpaceGet(nnz,&free_space);
  current_space = free_space;

  ierr = PetscMalloc3(merge->nrecv,PetscInt**,&buf_ri_k,merge->nrecv,PetscInt*,&nextrow,merge->nrecv,PetscInt*,&nextci);CHKERRQ(ierr);
  for (k=0; k<merge->nrecv; k++){
    buf_ri_k[k] = buf_ri[k]; /* beginning of k-th recved i-structure */
    nrows       = *buf_ri_k[k];
    nextrow[k]  = buf_ri_k[k] + 1;  /* next row number of k-th recved i-structure */
    nextci[k]   = buf_ri_k[k] + (nrows + 1);/* poins to the next i-structure of k-th recved i-structure  */
  }
  ierr = MatPreallocateInitialize(comm,pn,pn,dnz,onz);CHKERRQ(ierr);
  for (i=0; i<pn; i++) {
    /* add pdt[i,:]*AP into lnk */
    nnz = 0;
    pnz  = pdti[i+1] - pdti[i];
    j    = pnz;
    ptJ  = pdtj + pdti[i+1];
    while (j){/* assume cols are almost in increasing order, starting from its end saves computation */
      j--; ptJ--;
      row  = *ptJ; /* row of AP == col of Pt */
      apnz = api[row+1] - api[row];
      Jptr   = apj + api[row];
      /* add non-zero cols of AP into the sorted linked list lnk */
      ierr = PetscLLAdd(apnz,Jptr,pN,nlnk,lnk,lnkbt);CHKERRQ(ierr);
      nnz += nlnk;
    }
    /* add received col data into lnk */
    for (k=0; k<merge->nrecv; k++){ /* k-th received message */
      if (i == *nextrow[k]) { /* i-th row */
        nzi = *(nextci[k]+1) - *nextci[k]; 
        Jptr  = buf_rj[k] + *nextci[k];
        ierr = PetscLLAdd(nzi,Jptr,pN,nlnk,lnk,lnkbt);CHKERRQ(ierr);
        nnz += nlnk;
        nextrow[k]++; nextci[k]++;
      }
    }

    /* if free space is not available, make more free space */
    if (current_space->local_remaining<nnz) {
      ierr = PetscFreeSpaceGet(nnz+current_space->total_array_size,&current_space);CHKERRQ(ierr);
    }
    /* copy data into free space, then initialize lnk */
    ierr = PetscLLClean(pN,pN,nnz,lnk,current_space->array,lnkbt);CHKERRQ(ierr);
    ierr = MatPreallocateSet(i+owners[rank],nnz,current_space->array,dnz,onz);CHKERRQ(ierr);
    current_space->array           += nnz;
    current_space->local_used      += nnz;
    current_space->local_remaining -= nnz;
    bi[i+1] = bi[i] + nnz;
  }
  ierr = MatRestoreSymbolicTranspose_SeqAIJ(p->A,&pdti,&pdtj);CHKERRQ(ierr);
  ierr = PetscFree3(buf_ri_k,nextrow,nextci);CHKERRQ(ierr);

  ierr = PetscMalloc((bi[pn]+1)*sizeof(PetscInt),&bj);CHKERRQ(ierr);
  ierr = PetscFreeSpaceContiguous(&free_space,bj);CHKERRQ(ierr);
  ierr = PetscLLDestroy(lnk,lnkbt);CHKERRQ(ierr);

  /* create symbolic parallel matrix B_mpi */
  /*---------------------------------------*/
  ierr = MatCreate(comm,&B_mpi);CHKERRQ(ierr);
  ierr = MatSetSizes(B_mpi,pn,pn,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetType(B_mpi,MATMPIAIJ);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(B_mpi,0,dnz,0,onz);CHKERRQ(ierr);
  ierr = MatPreallocateFinalize(dnz,onz);CHKERRQ(ierr);

  merge->bi            = bi;
  merge->bj            = bj;
  merge->coi           = coi;
  merge->coj           = coj;
  merge->buf_ri        = buf_ri;
  merge->buf_rj        = buf_rj;
  merge->owners_co     = owners_co;
  merge->MatDestroy    = B_mpi->ops->destroy;
  merge->MatDuplicate  = B_mpi->ops->duplicate;

  /* B_mpi is not ready for use - assembly will be done by MatPtAPNumeric() */
  B_mpi->assembled     = PETSC_FALSE; 
  B_mpi->ops->destroy  = MatDestroy_MPIAIJ_MatPtAP;  
  B_mpi->ops->duplicate = MatDuplicate_MPIAIJ_MatPtAP;

  /* attach the supporting struct to B_mpi for reuse */
  ierr = PetscContainerCreate(PETSC_COMM_SELF,&container);CHKERRQ(ierr);
  ierr = PetscContainerSetPointer(container,merge);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)B_mpi,"MatMergeSeqsToMPI",(PetscObject)container);CHKERRQ(ierr);
  *C = B_mpi;
#if defined(PETSC_USE_INFO)
  if (bi[pn] != 0) {
    PetscReal afill = ((PetscReal)bi[pn])/(adi[am]+aoi[am]);
    if (afill < 1.0) afill = 1.0;
    ierr = PetscInfo3(B_mpi,"Reallocs %D; Fill ratio: given %G needed %G when computing A*P.\n",nspacedouble,fill,afill);CHKERRQ(ierr);
    ierr = PetscInfo1(B_mpi,"Use MatPtAP(A,P,MatReuse,%G,&C) for best performance.\n",afill);CHKERRQ(ierr);
  } else {
    ierr = PetscInfo(B_mpi,"Empty matrix product\n");CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatPtAPNumeric_MPIAIJ_MPIAIJ"
PetscErrorCode MatPtAPNumeric_MPIAIJ_MPIAIJ(Mat A,Mat P,Mat C)
{
  PetscErrorCode       ierr;
  Mat_Merge_SeqsToMPI  *merge; 
  Mat_MatMatMultMPI    *ap;
  PetscContainer       cont_merge,cont_ptap;
  Mat_MPIAIJ           *a=(Mat_MPIAIJ*)A->data,*p=(Mat_MPIAIJ*)P->data;
  Mat_SeqAIJ           *ad=(Mat_SeqAIJ*)(a->A)->data,*ao=(Mat_SeqAIJ*)(a->B)->data;
  Mat_SeqAIJ           *pd=(Mat_SeqAIJ*)(p->A)->data,*po=(Mat_SeqAIJ*)(p->B)->data;
  Mat_SeqAIJ           *p_loc,*p_oth; 
  PetscInt             *adi=ad->i,*aoi=ao->i,*adj=ad->j,*aoj=ao->j,*apJ,nextp;
  PetscInt             *pi_loc,*pj_loc,*pi_oth,*pj_oth,*pJ,*pj;
  PetscInt             i,j,k,anz,pnz,apnz,nextap,row,*cj;
  MatScalar            *ada=ad->a,*aoa=ao->a,*apa,*pa,*ca,*pa_loc,*pa_oth;
  PetscInt             am=A->rmap->n,cm=C->rmap->n,pon=(p->B)->cmap->n; 
  MPI_Comm             comm=((PetscObject)C)->comm;
  PetscMPIInt          size,rank,taga,*len_s;
  PetscInt             *owners,proc,nrows,**buf_ri_k,**nextrow,**nextci;
  PetscInt             **buf_ri,**buf_rj;  
  PetscInt             cnz=0,*bj_i,*bi,*bj,bnz,nextcj; /* bi,bj,ba: local array of C(mpi mat) */
  MPI_Request          *s_waits,*r_waits; 
  MPI_Status           *status;
  MatScalar            **abuf_r,*ba_i,*pA,*coa,*ba; 
  PetscInt             *api,*apj,*coi,*coj; 
  PetscInt             *poJ=po->j,*pdJ=pd->j,pcstart=P->cmap->rstart,pcend=P->cmap->rend; 

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  ierr = PetscObjectQuery((PetscObject)C,"MatMergeSeqsToMPI",(PetscObject *)&cont_merge);CHKERRQ(ierr);
  if (cont_merge) { 
    ierr  = PetscContainerGetPointer(cont_merge,(void **)&merge);CHKERRQ(ierr); 
  } else {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "Matrix C does not posses an object container");
  } 

  ierr = PetscObjectQuery((PetscObject)P,"Mat_MatMatMultMPI",(PetscObject *)&cont_ptap);CHKERRQ(ierr);
  if (cont_ptap) { 
    ierr  = PetscContainerGetPointer(cont_ptap,(void **)&ap);CHKERRQ(ierr); 
    if (ap->reuse == MAT_INITIAL_MATRIX){
      ap->reuse = MAT_REUSE_MATRIX;
    } else { /* update numerical values of P_oth and P_loc */
      ierr = MatGetBrowsOfAoCols(A,P,MAT_REUSE_MATRIX,&ap->startsj,&ap->startsj_r,&ap->bufa,&ap->B_oth);CHKERRQ(ierr);  
      ierr = MatGetLocalMat(P,MAT_REUSE_MATRIX,&ap->B_loc);CHKERRQ(ierr);
    }
  } else {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "Matrix P does not posses an object container");
  } 

  /* get data from symbolic products */
  p_loc = (Mat_SeqAIJ*)(ap->B_loc)->data;
  p_oth = (Mat_SeqAIJ*)(ap->B_oth)->data;
  pi_loc=p_loc->i; pj_loc=p_loc->j; pJ=pj_loc; pa_loc=p_loc->a,pA=pa_loc;  
  pi_oth=p_oth->i; pj_oth=p_oth->j; pa_oth=p_oth->a;
  
  coi = merge->coi; coj = merge->coj;
  ierr = PetscMalloc((coi[pon]+1)*sizeof(MatScalar),&coa);CHKERRQ(ierr);
  ierr = PetscMemzero(coa,coi[pon]*sizeof(MatScalar));CHKERRQ(ierr);

  bi     = merge->bi; bj = merge->bj;
  owners = merge->rowmap->range;
  ierr   = PetscMalloc((bi[cm]+1)*sizeof(MatScalar),&ba);CHKERRQ(ierr);
  ierr   = PetscMemzero(ba,bi[cm]*sizeof(MatScalar));CHKERRQ(ierr);

  /* get data from symbolic A*P */ 
  ierr = PetscMalloc((ap->abnz_max+1)*sizeof(MatScalar),&apa);CHKERRQ(ierr);
  ierr = PetscMemzero(apa,ap->abnz_max*sizeof(MatScalar));CHKERRQ(ierr);

  /* compute numeric C_seq=P_loc^T*A_loc*P */
  api = ap->abi; apj = ap->abj;
  for (i=0;i<am;i++) {
    /* form i-th sparse row of A*P */
    apnz = api[i+1] - api[i];
    apJ  = apj + api[i];
    /* diagonal portion of A */
    anz  = adi[i+1] - adi[i];
    for (j=0;j<anz;j++) {
      row = *adj++; 
      pnz = pi_loc[row+1] - pi_loc[row];
      pj  = pj_loc + pi_loc[row];
      pa  = pa_loc + pi_loc[row];
      nextp = 0;
      for (k=0; nextp<pnz; k++) {
        if (apJ[k] == pj[nextp]) { /* col of AP == col of P */
          apa[k] += (*ada)*pa[nextp++];
        }
      }
      ierr = PetscLogFlops(2.0*pnz);CHKERRQ(ierr);
      ada++;
    }
    /* off-diagonal portion of A */
    anz  = aoi[i+1] - aoi[i];
    for (j=0; j<anz; j++) {
      row = *aoj++;
      pnz = pi_oth[row+1] - pi_oth[row];
      pj  = pj_oth + pi_oth[row];
      pa  = pa_oth + pi_oth[row];
      nextp = 0;
      for (k=0; nextp<pnz; k++) {
        if (apJ[k] == pj[nextp]) { /* col of AP == col of P */
          apa[k] += (*aoa)*pa[nextp++];
        }
      }
      ierr = PetscLogFlops(2.0*pnz);CHKERRQ(ierr);
      aoa++;
    }

    /* Compute P_loc[i,:]^T*AP[i,:] using outer product */
    pnz = pi_loc[i+1] - pi_loc[i];
    for (j=0; j<pnz; j++) {
      nextap = 0;
      row    = *pJ++; /* global index */
      if (row < pcstart || row >=pcend) { /* put the value into Co */
        cj  = coj + coi[*poJ]; 
        ca  = coa + coi[*poJ++];
      } else {                            /* put the value into Cd */
        cj   = bj + bi[*pdJ]; 
        ca   = ba + bi[*pdJ++];
      } 
      for (k=0; nextap<apnz; k++) {
        if (cj[k]==apJ[nextap]) ca[k] += (*pA)*apa[nextap++]; 
      }
      ierr = PetscLogFlops(2.0*apnz);CHKERRQ(ierr);
      pA++;
    }

    /* zero the current row info for A*P */
    ierr = PetscMemzero(apa,apnz*sizeof(MatScalar));CHKERRQ(ierr);
  }
  ierr = PetscFree(apa);CHKERRQ(ierr);
  
  /* send and recv matrix values */
  /*-----------------------------*/
  buf_ri = merge->buf_ri;
  buf_rj = merge->buf_rj;
  len_s  = merge->len_s;
  ierr = PetscCommGetNewTag(comm,&taga);CHKERRQ(ierr);
  ierr = PetscPostIrecvScalar(comm,taga,merge->nrecv,merge->id_r,merge->len_r,&abuf_r,&r_waits);CHKERRQ(ierr);

  ierr = PetscMalloc((merge->nsend+1)*sizeof(MPI_Request),&s_waits);CHKERRQ(ierr);
  for (proc=0,k=0; proc<size; proc++){  
    if (!len_s[proc]) continue;
    i = merge->owners_co[proc];
    ierr = MPI_Isend(coa+coi[i],len_s[proc],MPIU_MATSCALAR,proc,taga,comm,s_waits+k);CHKERRQ(ierr);
    k++;
  } 
  ierr = PetscMalloc(size*sizeof(MPI_Status),&status);CHKERRQ(ierr);
  if (merge->nrecv) {ierr = MPI_Waitall(merge->nrecv,r_waits,status);CHKERRQ(ierr);}
  if (merge->nsend) {ierr = MPI_Waitall(merge->nsend,s_waits,status);CHKERRQ(ierr);}
  ierr = PetscFree(status);CHKERRQ(ierr);

  ierr = PetscFree(s_waits);CHKERRQ(ierr);
  ierr = PetscFree(r_waits);CHKERRQ(ierr);
  ierr = PetscFree(coa);CHKERRQ(ierr);

  /* insert local and received values into C */
  /*-----------------------------------------*/
  ierr = PetscMalloc3(merge->nrecv,PetscInt**,&buf_ri_k,merge->nrecv,PetscInt*,&nextrow,merge->nrecv,PetscInt*,&nextci);CHKERRQ(ierr);

  for (k=0; k<merge->nrecv; k++){
    buf_ri_k[k] = buf_ri[k]; /* beginning of k-th recved i-structure */
    nrows       = *(buf_ri_k[k]);
    nextrow[k]  = buf_ri_k[k]+1;  /* next row number of k-th recved i-structure */
    nextci[k]   = buf_ri_k[k] + (nrows + 1);/* poins to the next i-structure of k-th recved i-structure  */
  }

  for (i=0; i<cm; i++) {
    row = owners[rank] + i; /* global row index of C_seq */
    bj_i = bj + bi[i];  /* col indices of the i-th row of C */
    ba_i = ba + bi[i]; 
    bnz  = bi[i+1] - bi[i];
    /* add received vals into ba */
    for (k=0; k<merge->nrecv; k++){ /* k-th received message */
      /* i-th row */
      if (i == *nextrow[k]) {
        cnz = *(nextci[k]+1) - *nextci[k]; 
        cj  = buf_rj[k] + *(nextci[k]);
        ca  = abuf_r[k] + *(nextci[k]);
        nextcj = 0;
        for (j=0; nextcj<cnz; j++){ 
          if (bj_i[j] == cj[nextcj]){ /* bcol == ccol */
            ba_i[j] += ca[nextcj++]; 
          }
        }
        nextrow[k]++; nextci[k]++;
      } 
    }
    ierr = MatSetValues(C,1,&row,bnz,bj_i,ba_i,INSERT_VALUES);CHKERRQ(ierr); 
    ierr = PetscLogFlops(2.0*cnz);CHKERRQ(ierr);
  } 
  ierr = MatSetBlockSize(C,1);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr); 

  ierr = PetscFree(ba);CHKERRQ(ierr);
  ierr = PetscFree(abuf_r[0]);CHKERRQ(ierr);
  ierr = PetscFree(abuf_r);CHKERRQ(ierr);
  ierr = PetscFree3(buf_ri_k,nextrow,nextci);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}
