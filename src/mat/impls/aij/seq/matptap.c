/*
  Defines projective product routines where A is a AIJ matrix
          C = P^T * A * P
*/

#include "src/mat/impls/aij/seq/aij.h"   /*I "petscmat.h" I*/
#include "src/mat/utils/freespace.h"
#include "src/mat/impls/aij/mpi/mpiaij.h"
#include "petscbt.h"

#undef __FUNCT__
#define __FUNCT__ "MatPtAP"
/*@
   MatPtAP - Creates the matrix projection C = P^T * A * P

   Collective on Mat

   Input Parameters:
+  A - the matrix
.  P - the projection matrix
.  scall - either MAT_INITIAL_MATRIX or MAT_REUSE_MATRIX
-  fill - expected fill as ratio of nnz(C)/(nnz(A) + nnz(P))

   Output Parameters:
.  C - the product matrix

   Notes:
   C will be created and must be destroyed by the user with MatDestroy().

   This routine is currently only implemented for pairs of SeqAIJ matrices and classes
   which inherit from SeqAIJ.  C will be of type MATSEQAIJ.

   Level: intermediate

.seealso: MatPtAPSymbolic(),MatPtAPNumeric(),MatMatMult()
@*/
PetscErrorCode MatPtAP(Mat A,Mat P,MatReuse scall,PetscReal fill,Mat *C) 
{
  PetscErrorCode ierr;
  PetscErrorCode (*fA)(Mat,Mat,MatReuse,PetscReal,Mat *);
  PetscErrorCode (*fP)(Mat,Mat,MatReuse,PetscReal,Mat *);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_COOKIE,1);
  PetscValidType(A,1);
  MatPreallocated(A);
  if (!A->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (A->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  PetscValidHeaderSpecific(P,MAT_COOKIE,2);
  PetscValidType(P,2);
  MatPreallocated(P);
  if (!P->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (P->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  PetscValidPointer(C,3);

  if (P->M!=A->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, %D != %D",P->M,A->N);

  if (fill <=0.0) SETERRQ1(PETSC_ERR_ARG_SIZ,"fill=%g must be > 0.0",fill);

  /* For now, we do not dispatch based on the type of A and P */
  /* When implementations like _SeqAIJ_MAIJ exist, attack the multiple dispatch problem. */  
  fA = A->ops->ptap;
  if (!fA) SETERRQ1(PETSC_ERR_SUP,"MatPtAP not supported for A of type %s",A->type_name);
  fP = P->ops->ptap;
  if (!fP) SETERRQ1(PETSC_ERR_SUP,"MatPtAP not supported for P of type %s",P->type_name);
  if (fP!=fA) SETERRQ2(PETSC_ERR_ARG_INCOMP,"MatPtAP requires A, %s, to be compatible with P, %s",A->type_name,P->type_name);

  ierr = PetscLogEventBegin(MAT_PtAP,A,P,0,0);CHKERRQ(ierr); 
  ierr = (*fA)(A,P,scall,fill,C);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_PtAP,A,P,0,0);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}

EXTERN PetscErrorCode MatPtAPSymbolic_MPIAIJ_MPIAIJ_Local(Mat,Mat,Mat,PetscInt,PetscReal,Mat*);
EXTERN PetscErrorCode MatPtAPNumeric_MPIAIJ_MPIAIJ_Local(Mat,Mat,Mat,PetscInt,Mat C);

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
    /* update P_oth */
    ierr = MatGetBrowsOfAoCols(A,P,scall,&ptap->isrowb,&ptap->iscolb,&ptap->brstart,&ptap->B_oth);CHKERRQ(ierr);

  } else {
    SETERRQ1(PETSC_ERR_ARG_WRONG,"Invalid MatReuse %D",scall);
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
  Mat                  C_seq;
  Mat_MatMatMultMPI    *ptap;
  PetscObjectContainer container;

  PetscFunctionBegin;
 
  ierr = PetscObjectQuery((PetscObject)P,"MatPtAPMPI",(PetscObject *)&container);CHKERRQ(ierr);
  if (container) { 
    ierr  = PetscObjectContainerGetPointer(container,(void **)&ptap);CHKERRQ(ierr); 
  } else {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "Matrix P does not posses an object container");
  } 
  /* compute C_seq = P_loc^T * A_loc * P_seq */
  ierr = MatPtAPSymbolic_MPIAIJ_MPIAIJ_Local(A,ptap->B_loc,ptap->B_oth,ptap->brstart,fill,&C_seq);CHKERRQ(ierr);
  
  /* add C_seq into mpi C */
  ierr = MatMerge_SeqsToMPISymbolic(A->comm,C_seq,P->n,P->n,C);CHKERRQ(ierr);

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
  PetscInt             flops=0;
  Mat_MPIAIJ           *a=(Mat_MPIAIJ*)A->data,*p=(Mat_MPIAIJ*)P->data,*c=(Mat_MPIAIJ*)C->data;
  Mat_SeqAIJ           *ad=(Mat_SeqAIJ*)(a->A)->data,*ao=(Mat_SeqAIJ*)(a->B)->data,
                       *pd=(Mat_SeqAIJ*)(p->A)->data,*po=(Mat_SeqAIJ*)(p->B)->data,
                       *cd=(Mat_SeqAIJ*)(c->A)->data,*co=(Mat_SeqAIJ*)(c->B)->data;
  Mat                  C_seq;
  Mat_SeqAIJ           *cseq,*p_oth; 
  PetscInt             *adi=ad->i,*adj=ad->j,*aoi=ao->i,*aoj=ao->j,*apj,*apjdense,cstart=a->cstart,cend=a->cend;
  PetscInt             *pi_oth,*pj_oth,*pJ_d=pd->j,*pJ_o=po->j,*pjj;
  PetscInt             i,j,k,nzi,pnzi,apnzj,nextap,pnzj,prow,crow;
  PetscInt             *cjj;
  MatScalar            *ada=ad->a,*aoa=ao->a,*apa,*paj,*cseqa,*caj; 
  MatScalar            *pA_d=pd->a,*pA_o=po->a,*pa_oth;
  PetscInt             am=A->m,cN=C->N,cm=C->m; 
  MPI_Comm             comm=C->comm;
  PetscMPIInt          size,rank,taga,*len_s;
  PetscInt             *owners; 
  PetscInt             proc;
  PetscInt             **buf_ri,**buf_rj;  
  PetscInt             cseqnzi,*bj_i,*bi,*bj,bnzi,nextcseqj; /* bi, bj, ba: for C (mpi mat) */
  PetscInt             nrows,**buf_ri_k,**nextrow,**nextcseqi;
  MPI_Request          *s_waits,*r_waits; 
  MPI_Status           *status;
  MatScalar            **abuf_r,*ba_i,*ca;
  PetscInt             *cseqi,*cseqj,col;
 

  PetscFunctionBegin;
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

  p_oth=(Mat_SeqAIJ*)(ptap->B_oth)->data;
  pi_oth=p_oth->i; pj_oth=p_oth->j; pa_oth=p_oth->a;

  C_seq=merge->C_seq;
  cseq=(Mat_SeqAIJ*)C_seq->data;
  cseqi=cseq->i; cseqj=cseq->j;
  cseqa=cseq->a;

  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = PetscMapGetGlobalRange(merge->rowmap,&owners);CHKERRQ(ierr);

  /* Allocate temporary array for storage of one row of A*P */
  ierr = PetscMalloc(cN*(sizeof(MatScalar)+2*sizeof(PetscInt)),&apa);CHKERRQ(ierr);
  ierr = PetscMemzero(apa,cN*(sizeof(MatScalar)+2*sizeof(PetscInt)));CHKERRQ(ierr);
  apj      = (PetscInt *)(apa + cN);
  apjdense = apj + cN;

  /* Allocate temporary MatScalar array for storage of one row of C */
  ierr = PetscMalloc(cN*(sizeof(MatScalar)),&ca);CHKERRQ(ierr);
  ierr = PetscMemzero(ca,cN*(sizeof(MatScalar)));CHKERRQ(ierr);

  /* Clear old values in C_Seq and C */
  ierr = PetscMemzero(cseqa,cseqi[cN]*sizeof(MatScalar));CHKERRQ(ierr);
  ierr = PetscMemzero(cd->a,cd->i[cm]*sizeof(MatScalar));CHKERRQ(ierr);
  ierr = PetscMemzero(co->a,co->i[cm]*sizeof(MatScalar));CHKERRQ(ierr);

  for (i=0;i<am;i++) {
    /* Form i-th sparse row of A*P */
     apnzj = 0;
    /* diagonal portion of A */
    nzi  = adi[i+1] - adi[i];
    for (j=0;j<nzi;j++) {
      prow = *adj++; 
      /* diagonal portion of P */
      pnzj = pd->i[prow+1] - pd->i[prow];
      pjj  = pd->j + pd->i[prow]; /* local col index of P */
      paj  = pd->a + pd->i[prow];
      for (k=0;k<pnzj;k++) {
        col = *pjj + p->cstart; pjj++; /* global col index of P */
        if (!apjdense[col]) {
          apjdense[col] = -1; 
          apj[apnzj++]  = col;
        }
        apa[col] += (*ada)*paj[k];
      }
      flops += 2*pnzj;
      /* off-diagonal portion of P */
      pnzj = po->i[prow+1] - po->i[prow];
      pjj  = po->j + po->i[prow]; /* local col index of P */
      paj  = po->a + po->i[prow];
      for (k=0;k<pnzj;k++) {
        col = p->garray[*pjj]; pjj++; /* global col index of P */
        if (!apjdense[col]) {
          apjdense[col] = -1; 
          apj[apnzj++]  = col;
        }
        apa[col] += (*ada)*paj[k];
      }
      flops += 2*pnzj;
      
      ada++;
    }
    /* off-diagonal portion of A */
    nzi  = aoi[i+1] - aoi[i];
    for (j=0;j<nzi;j++) {
      prow = *aoj++;
      pnzj = pi_oth[prow+1] - pi_oth[prow];
      pjj  = pj_oth + pi_oth[prow];
      paj  = pa_oth + pi_oth[prow];
      for (k=0;k<pnzj;k++) {
        if (!apjdense[pjj[k]]) {
          apjdense[pjj[k]] = -1; 
          apj[apnzj++]     = pjj[k];
        }
        apa[pjj[k]] += (*aoa)*paj[k];
      }
      flops += 2*pnzj;
      aoa++;
    }
    /* Sort the j index array for quick sparse axpy. */
    ierr = PetscSortInt(apnzj,apj);CHKERRQ(ierr);

    /* Compute P_loc[i,:]^T*AP[i,:] using outer product */
    /* diagonal portion of P */
    pnzi = pd->i[i+1] - pd->i[i];
    for (j=0;j<pnzi;j++) {
      crow = (*pJ_d++) + owners[rank]; 
      cjj    = cseqj + cseqi[crow];
      caj    = cseqa + cseqi[crow];
      /* add value into C */
      for (k=0; k<apnzj; k++) ca[k] = (*pA_d)*apa[apj[k]];
      ierr = MatSetValues(C,1,&crow,apnzj,apj,ca,ADD_VALUES);CHKERRQ(ierr);
      ierr = PetscMemzero(ca,apnzj*(sizeof(MatScalar)));CHKERRQ(ierr);
      pA_d++;
      flops += 2*apnzj;
    }

    /* off-diagonal portion of P */
    pnzi = po->i[i+1] - po->i[i];
    for (j=0;j<pnzi;j++) {
      crow   = p->garray[*pJ_o++]; 
      cjj    = cseqj + cseqi[crow];
      caj    = cseqa + cseqi[crow];
      /* add value into C_seq to be sent to other processors */
      nextap = 0;
      for (k=0;nextap<apnzj;k++) {
        if (cjj[k]==apj[nextap]) {
          caj[k] += (*pA_o)*apa[apj[nextap++]];
        }
      }
      flops += 2*apnzj;
      pA_o++;
    }

    /* Zero the current row info for A*P */
    for (j=0;j<apnzj;j++) {
      apa[apj[j]]      = 0.;
      apjdense[apj[j]] = 0;
    }
  }

  /* Assemble the final matrix and clean up */
  ierr = MatAssemblyBegin(C_seq,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C_seq,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscFree(apa);CHKERRQ(ierr);
  ierr = PetscLogFlops(flops);CHKERRQ(ierr);
  
  /*--------------------------------------------------------------*/
  /* send and recv matrix values */
  /*-----------------------------*/
  bi     = merge->bi;
  bj     = merge->bj;
  buf_ri = merge->buf_ri;
  buf_rj = merge->buf_rj;
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
  ierr   = PetscMalloc(size*sizeof(MPI_Status),&status);CHKERRQ(ierr);
  ierr = MPI_Waitall(merge->nrecv,r_waits,status);CHKERRQ(ierr);
  ierr = MPI_Waitall(merge->nsend,s_waits,status);CHKERRQ(ierr);
  ierr = PetscFree(status);CHKERRQ(ierr);

  ierr = PetscFree(s_waits);CHKERRQ(ierr);
  ierr = PetscFree(r_waits);CHKERRQ(ierr);

  /* insert mat values of mpimat */
  /*----------------------------*/
  ierr = PetscMalloc(cN*sizeof(MatScalar),&ba_i);CHKERRQ(ierr);
  ierr = PetscMemzero(ba_i,cN*sizeof(MatScalar));CHKERRQ(ierr); 
  ierr = PetscMalloc((3*merge->nrecv+1)*sizeof(PetscInt**),&buf_ri_k);CHKERRQ(ierr);
  nextrow = buf_ri_k + merge->nrecv;
  nextcseqi = nextrow + merge->nrecv;

  for (k=0; k<merge->nrecv; k++){
    buf_ri_k[k] = buf_ri[k]; /* beginning of k-th recved i-structure */
    nrows = *(buf_ri_k[k]);
    nextrow[k]  = buf_ri_k[k]+1;  /* next row number of k-th recved i-structure */
    nextcseqi[k]   = buf_ri_k[k] + (nrows + 1);/* poins to the next i-structure of k-th recved i-structure  */
  }

  /* add received local vals to C */
  for (i=0; i<cm; i++) {
    crow = owners[rank] + i; 
    bj_i = bj+bi[i];  /* col indices of the i-th row of C */
    bnzi = bi[i+1] - bi[i];
    nzi = 0;
    for (k=0; k<merge->nrecv; k++){ /* k-th received message */
      /* i-th row */
      if (i == *nextrow[k]) {
        cseqnzi = *(nextcseqi[k]+1) - *nextcseqi[k]; 
        cseqj   = buf_rj[k] + *(nextcseqi[k]);
        cseqa   = abuf_r[k] + *(nextcseqi[k]);
        nextcseqj = 0;
        for (j=0; nextcseqj<cseqnzi; j++){ 
          if (*(bj_i + j) == cseqj[nextcseqj]){ /* bcol == cseqcol */
            ba_i[j] += cseqa[nextcseqj++];
            nzi++;
          }
        }
        nextrow[k]++; nextcseqi[k]++;
      } 
    }
    if (nzi>0){
      ierr = MatSetValues(C,1,&crow,bnzi,bj_i,ba_i,ADD_VALUES);CHKERRQ(ierr);
      ierr = PetscMemzero(ba_i,bnzi*sizeof(MatScalar));CHKERRQ(ierr);
      flops += 2*nzi;
    }
  } 
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr); 

  ierr = PetscFree(abuf_r);CHKERRQ(ierr);
  ierr = PetscFree(ba_i);CHKERRQ(ierr);
  ierr = PetscFree(buf_ri_k);CHKERRQ(ierr);
  ierr = PetscFree(ca);CHKERRQ(ierr);
 
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPtAP_SeqAIJ_SeqAIJ"
PetscErrorCode MatPtAP_SeqAIJ_SeqAIJ(Mat A,Mat P,MatReuse scall,PetscReal fill,Mat *C) 
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (scall == MAT_INITIAL_MATRIX){
    ierr = PetscLogEventBegin(MAT_PtAPSymbolic,A,P,0,0);CHKERRQ(ierr);
    ierr = MatPtAPSymbolic_SeqAIJ_SeqAIJ(A,P,fill,C);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(MAT_PtAPSymbolic,A,P,0,0);CHKERRQ(ierr); 
  }
  ierr = PetscLogEventBegin(MAT_PtAPNumeric,A,P,0,0);CHKERRQ(ierr);
  ierr = MatPtAPNumeric_SeqAIJ_SeqAIJ(A,P,*C);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_PtAPNumeric,A,P,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   MatPtAPSymbolic - Creates the (i,j) structure of the matrix projection C = P^T * A * P

   Collective on Mat

   Input Parameters:
+  A - the matrix
-  P - the projection matrix

   Output Parameters:
.  C - the (i,j) structure of the product matrix

   Notes:
   C will be created and must be destroyed by the user with MatDestroy().

   This routine is currently only implemented for pairs of SeqAIJ matrices and classes
   which inherit from SeqAIJ.  C will be of type MATSEQAIJ.  The product is computed using
   this (i,j) structure by calling MatPtAPNumeric().

   Level: intermediate

.seealso: MatPtAP(),MatPtAPNumeric(),MatMatMultSymbolic()
*/
#undef __FUNCT__
#define __FUNCT__ "MatPtAPSymbolic"
PetscErrorCode MatPtAPSymbolic(Mat A,Mat P,PetscReal fill,Mat *C) 
{
  PetscErrorCode ierr;
  PetscErrorCode (*fA)(Mat,Mat,PetscReal,Mat*);
  PetscErrorCode (*fP)(Mat,Mat,PetscReal,Mat*);

  PetscFunctionBegin;

  PetscValidHeaderSpecific(A,MAT_COOKIE,1);
  PetscValidType(A,1);
  MatPreallocated(A);
  if (!A->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (A->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 

  PetscValidHeaderSpecific(P,MAT_COOKIE,2);
  PetscValidType(P,2);
  MatPreallocated(P);
  if (!P->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (P->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 

  PetscValidPointer(C,3);

  if (P->M!=A->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, %D != %D",P->M,A->N);
  if (A->M!=A->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Matrix 'A' must be square, %D != %D",A->M,A->N);

  /* For now, we do not dispatch based on the type of A and P */
  /* When implementations like _SeqAIJ_MAIJ exist, attack the multiple dispatch problem. */  
  fA = A->ops->ptapsymbolic;
  if (!fA) SETERRQ1(PETSC_ERR_SUP,"MatPtAPSymbolic not supported for A of type %s",A->type_name);
  fP = P->ops->ptapsymbolic;
  if (!fP) SETERRQ1(PETSC_ERR_SUP,"MatPtAPSymbolic not supported for P of type %s",P->type_name);
  if (fP!=fA) SETERRQ2(PETSC_ERR_ARG_INCOMP,"MatPtAPSymbolic requires A, %s, to be compatible with P, %s",A->type_name,P->type_name);

  ierr = PetscLogEventBegin(MAT_PtAPSymbolic,A,P,0,0);CHKERRQ(ierr); 
  ierr = (*fA)(A,P,fill,C);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_PtAPSymbolic,A,P,0,0);CHKERRQ(ierr); 
 
  PetscFunctionReturn(0);
}

typedef struct { 
  Mat    symAP;
} Mat_PtAPstruct;

EXTERN PetscErrorCode MatDestroy_SeqAIJ(Mat);

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_SeqAIJ_PtAP"
PetscErrorCode MatDestroy_SeqAIJ_PtAP(Mat A)
{
  PetscErrorCode    ierr;
  Mat_PtAPstruct    *ptap=(Mat_PtAPstruct*)A->spptr; 

  PetscFunctionBegin;
  ierr = MatDestroy(ptap->symAP);CHKERRQ(ierr);
  ierr = PetscFree(ptap);CHKERRQ(ierr);
  ierr = MatDestroy_SeqAIJ(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPtAPSymbolic_SeqAIJ_SeqAIJ"
PetscErrorCode MatPtAPSymbolic_SeqAIJ_SeqAIJ(Mat A,Mat P,PetscReal fill,Mat *C) 
{
  PetscErrorCode ierr;
  FreeSpaceList  free_space=PETSC_NULL,current_space=PETSC_NULL;
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data,*p = (Mat_SeqAIJ*)P->data,*c;
  PetscInt       *pti,*ptj,*ptJ,*ai=a->i,*aj=a->j,*ajj,*pi=p->i,*pj=p->j,*pjj;
  PetscInt       *ci,*cj,*ptadenserow,*ptasparserow,*ptaj;
  PetscInt       an=A->N,am=A->M,pn=P->N,pm=P->M;
  PetscInt       i,j,k,ptnzi,arow,anzj,ptanzi,prow,pnzj,cnzi,nlnk,*lnk;
  MatScalar      *ca;
  PetscBT        lnkbt;

  PetscFunctionBegin;
  /* Get ij structure of P^T */
  ierr = MatGetSymbolicTranspose_SeqAIJ(P,&pti,&ptj);CHKERRQ(ierr);
  ptJ=ptj;

  /* Allocate ci array, arrays for fill computation and */
  /* free space for accumulating nonzero column info */
  ierr = PetscMalloc((pn+1)*sizeof(PetscInt),&ci);CHKERRQ(ierr);
  ci[0] = 0;

  ierr = PetscMalloc((2*an+1)*sizeof(PetscInt),&ptadenserow);CHKERRQ(ierr);
  ierr = PetscMemzero(ptadenserow,(2*an+1)*sizeof(PetscInt));CHKERRQ(ierr);
  ptasparserow = ptadenserow  + an;

  /* create and initialize a linked list */
  nlnk = pn+1;
  ierr = PetscLLCreate(pn,pn,nlnk,lnk,lnkbt);CHKERRQ(ierr);

  /* Set initial free space to be nnz(A) scaled by aspect ratio of P. */
  /* This should be reasonable if sparsity of PtAP is similar to that of A. */
  ierr          = GetMoreSpace((ai[am]/pm)*pn,&free_space);
  current_space = free_space;

  /* Determine symbolic info for each row of C: */
  for (i=0;i<pn;i++) {
    ptnzi  = pti[i+1] - pti[i];
    ptanzi = 0;
    /* Determine symbolic row of PtA: */
    for (j=0;j<ptnzi;j++) {
      arow = *ptJ++;
      anzj = ai[arow+1] - ai[arow];
      ajj  = aj + ai[arow];
      for (k=0;k<anzj;k++) {
        if (!ptadenserow[ajj[k]]) {
          ptadenserow[ajj[k]]    = -1;
          ptasparserow[ptanzi++] = ajj[k];
        }
      }
    }
      /* Using symbolic info for row of PtA, determine symbolic info for row of C: */
    ptaj = ptasparserow;
    cnzi   = 0;
    for (j=0;j<ptanzi;j++) {
      prow = *ptaj++;
      pnzj = pi[prow+1] - pi[prow];
      pjj  = pj + pi[prow];
      /* add non-zero cols of P into the sorted linked list lnk */
      ierr = PetscLLAdd(pnzj,pjj,pn,nlnk,lnk,lnkbt);CHKERRQ(ierr);
      cnzi += nlnk;
    }
   
    /* If free space is not available, make more free space */
    /* Double the amount of total space in the list */
    if (current_space->local_remaining<cnzi) {
      ierr = GetMoreSpace(current_space->total_array_size,&current_space);CHKERRQ(ierr);
    }

    /* Copy data into free space, and zero out denserows */
    ierr = PetscLLClean(pn,pn,cnzi,lnk,current_space->array,lnkbt);CHKERRQ(ierr);
    current_space->array           += cnzi;
    current_space->local_used      += cnzi;
    current_space->local_remaining -= cnzi;
    
    for (j=0;j<ptanzi;j++) {
      ptadenserow[ptasparserow[j]] = 0;
    }
    /* Aside: Perhaps we should save the pta info for the numerical factorization. */
    /*        For now, we will recompute what is needed. */ 
    ci[i+1] = ci[i] + cnzi;
  }
  /* nnz is now stored in ci[ptm], column indices are in the list of free space */
  /* Allocate space for cj, initialize cj, and */
  /* destroy list of free space and other temporary array(s) */
  ierr = PetscMalloc((ci[pn]+1)*sizeof(PetscInt),&cj);CHKERRQ(ierr);
  ierr = MakeSpaceContiguous(&free_space,cj);CHKERRQ(ierr);
  ierr = PetscFree(ptadenserow);CHKERRQ(ierr);
  ierr = PetscLLDestroy(lnk,lnkbt);CHKERRQ(ierr);
  
  /* Allocate space for ca */
  ierr = PetscMalloc((ci[pn]+1)*sizeof(MatScalar),&ca);CHKERRQ(ierr);
  ierr = PetscMemzero(ca,(ci[pn]+1)*sizeof(MatScalar));CHKERRQ(ierr);
  
  /* put together the new matrix */
  ierr = MatCreateSeqAIJWithArrays(A->comm,pn,pn,ci,cj,ca,C);CHKERRQ(ierr);

  /* MatCreateSeqAIJWithArrays flags matrix so PETSc doesn't free the user's arrays. */
  /* Since these are PETSc arrays, change flags to free them as necessary. */
  c = (Mat_SeqAIJ *)((*C)->data);
  c->freedata = PETSC_TRUE;
  c->nonew    = 0;

  /* Clean up. */
  ierr = MatRestoreSymbolicTranspose_SeqAIJ(P,&pti,&ptj);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#include "src/mat/impls/maij/maij.h"
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatPtAPSymbolic_SeqAIJ_SeqMAIJ"
PetscErrorCode MatPtAPSymbolic_SeqAIJ_SeqMAIJ(Mat A,Mat PP,Mat *C) 
{
  /* This routine requires testing -- I don't think it works. */
  PetscErrorCode ierr;
  FreeSpaceList  free_space=PETSC_NULL,current_space=PETSC_NULL;
  Mat_SeqMAIJ    *pp=(Mat_SeqMAIJ*)PP->data;
  Mat            P=pp->AIJ;
  Mat_SeqAIJ     *a=(Mat_SeqAIJ*)A->data,*p=(Mat_SeqAIJ*)P->data,*c;
  PetscInt       *pti,*ptj,*ptJ,*ai=a->i,*aj=a->j,*ajj,*pi=p->i,*pj=p->j,*pjj;
  PetscInt       *ci,*cj,*denserow,*sparserow,*ptadenserow,*ptasparserow,*ptaj;
  PetscInt       an=A->N,am=A->M,pn=P->N,pm=P->M,ppdof=pp->dof;
  PetscInt       i,j,k,dof,pdof,ptnzi,arow,anzj,ptanzi,prow,pnzj,cnzi;
  MatScalar      *ca;

  PetscFunctionBegin;  
  /* Start timer */
  ierr = PetscLogEventBegin(MAT_PtAPSymbolic,A,PP,0,0);CHKERRQ(ierr);

  /* Get ij structure of P^T */
  ierr = MatGetSymbolicTranspose_SeqAIJ(P,&pti,&ptj);CHKERRQ(ierr);

  /* Allocate ci array, arrays for fill computation and */
  /* free space for accumulating nonzero column info */
  ierr = PetscMalloc((pn+1)*sizeof(PetscInt),&ci);CHKERRQ(ierr);
  ci[0] = 0;

  ierr = PetscMalloc((2*pn+2*an+1)*sizeof(PetscInt),&ptadenserow);CHKERRQ(ierr);
  ierr = PetscMemzero(ptadenserow,(2*pn+2*an+1)*sizeof(PetscInt));CHKERRQ(ierr);
  ptasparserow = ptadenserow  + an;
  denserow     = ptasparserow + an;
  sparserow    = denserow     + pn;

  /* Set initial free space to be nnz(A) scaled by aspect ratio of P. */
  /* This should be reasonable if sparsity of PtAP is similar to that of A. */
  ierr          = GetMoreSpace((ai[am]/pm)*pn,&free_space);
  current_space = free_space;

  /* Determine symbolic info for each row of C: */
  for (i=0;i<pn/ppdof;i++) {
    ptnzi  = pti[i+1] - pti[i];
    ptanzi = 0;
    ptJ    = ptj + pti[i];
    for (dof=0;dof<ppdof;dof++) {
    /* Determine symbolic row of PtA: */
      for (j=0;j<ptnzi;j++) {
        arow = ptJ[j] + dof;
        anzj = ai[arow+1] - ai[arow];
        ajj  = aj + ai[arow];
        for (k=0;k<anzj;k++) {
          if (!ptadenserow[ajj[k]]) {
            ptadenserow[ajj[k]]    = -1;
            ptasparserow[ptanzi++] = ajj[k];
          }
        }
      }
      /* Using symbolic info for row of PtA, determine symbolic info for row of C: */
      ptaj = ptasparserow;
      cnzi   = 0;
      for (j=0;j<ptanzi;j++) {
        pdof = *ptaj%dof;
        prow = (*ptaj++)/dof;
        pnzj = pi[prow+1] - pi[prow];
        pjj  = pj + pi[prow];
        for (k=0;k<pnzj;k++) {
          if (!denserow[pjj[k]+pdof]) {
            denserow[pjj[k]+pdof] = -1;
            sparserow[cnzi++]     = pjj[k]+pdof;
          }
        }
      }

      /* sort sparserow */
      ierr = PetscSortInt(cnzi,sparserow);CHKERRQ(ierr);
      
      /* If free space is not available, make more free space */
      /* Double the amount of total space in the list */
      if (current_space->local_remaining<cnzi) {
        ierr = GetMoreSpace(current_space->total_array_size,&current_space);CHKERRQ(ierr);
      }

      /* Copy data into free space, and zero out denserows */
      ierr = PetscMemcpy(current_space->array,sparserow,cnzi*sizeof(PetscInt));CHKERRQ(ierr);
      current_space->array           += cnzi;
      current_space->local_used      += cnzi;
      current_space->local_remaining -= cnzi;

      for (j=0;j<ptanzi;j++) {
        ptadenserow[ptasparserow[j]] = 0;
      }
      for (j=0;j<cnzi;j++) {
        denserow[sparserow[j]] = 0;
      }
      /* Aside: Perhaps we should save the pta info for the numerical factorization. */
      /*        For now, we will recompute what is needed. */ 
      ci[i+1+dof] = ci[i+dof] + cnzi;
    }
  }
  /* nnz is now stored in ci[ptm], column indices are in the list of free space */
  /* Allocate space for cj, initialize cj, and */
  /* destroy list of free space and other temporary array(s) */
  ierr = PetscMalloc((ci[pn]+1)*sizeof(PetscInt),&cj);CHKERRQ(ierr);
  ierr = MakeSpaceContiguous(&free_space,cj);CHKERRQ(ierr);
  ierr = PetscFree(ptadenserow);CHKERRQ(ierr);
    
  /* Allocate space for ca */
  ierr = PetscMalloc((ci[pn]+1)*sizeof(MatScalar),&ca);CHKERRQ(ierr);
  ierr = PetscMemzero(ca,(ci[pn]+1)*sizeof(MatScalar));CHKERRQ(ierr);
  
  /* put together the new matrix */
  ierr = MatCreateSeqAIJWithArrays(A->comm,pn,pn,ci,cj,ca,C);CHKERRQ(ierr);

  /* MatCreateSeqAIJWithArrays flags matrix so PETSc doesn't free the user's arrays. */
  /* Since these are PETSc arrays, change flags to free them as necessary. */
  c = (Mat_SeqAIJ *)((*C)->data);
  c->freedata = PETSC_TRUE;
  c->nonew    = 0;

  /* Clean up. */
  ierr = MatRestoreSymbolicTranspose_SeqAIJ(P,&pti,&ptj);CHKERRQ(ierr);

  ierr = PetscLogEventEnd(MAT_PtAPSymbolic,A,PP,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

/*
   MatPtAPNumeric - Computes the matrix projection C = P^T * A * P

   Collective on Mat

   Input Parameters:
+  A - the matrix
-  P - the projection matrix

   Output Parameters:
.  C - the product matrix

   Notes:
   C must have been created by calling MatPtAPSymbolic and must be destroyed by
   the user using MatDeatroy().

   This routine is currently only implemented for pairs of AIJ matrices and classes
   which inherit from AIJ.  C will be of type MATAIJ.

   Level: intermediate

.seealso: MatPtAP(),MatPtAPSymbolic(),MatMatMultNumeric()
*/
#undef __FUNCT__
#define __FUNCT__ "MatPtAPNumeric"
PetscErrorCode MatPtAPNumeric(Mat A,Mat P,Mat C) 
{
  PetscErrorCode ierr;
  PetscErrorCode (*fA)(Mat,Mat,Mat);
  PetscErrorCode (*fP)(Mat,Mat,Mat);

  PetscFunctionBegin;

  PetscValidHeaderSpecific(A,MAT_COOKIE,1);
  PetscValidType(A,1);
  MatPreallocated(A);
  if (!A->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (A->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 

  PetscValidHeaderSpecific(P,MAT_COOKIE,2);
  PetscValidType(P,2);
  MatPreallocated(P);
  if (!P->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (P->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 

  PetscValidHeaderSpecific(C,MAT_COOKIE,3);
  PetscValidType(C,3);
  MatPreallocated(C);
  if (!C->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (C->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 

  if (P->N!=C->M) SETERRQ2(PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, %D != %D",P->N,C->M);
  if (P->M!=A->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, %D != %D",P->M,A->N);
  if (A->M!=A->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Matrix 'A' must be square, %D != %D",A->M,A->N);
  if (P->N!=C->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, %D != %D",P->N,C->N);

  /* For now, we do not dispatch based on the type of A and P */
  /* When implementations like _SeqAIJ_MAIJ exist, attack the multiple dispatch problem. */  
  fA = A->ops->ptapnumeric;
  if (!fA) SETERRQ1(PETSC_ERR_SUP,"MatPtAPNumeric not supported for A of type %s",A->type_name);
  fP = P->ops->ptapnumeric;
  if (!fP) SETERRQ1(PETSC_ERR_SUP,"MatPtAPNumeric not supported for P of type %s",P->type_name);
  if (fP!=fA) SETERRQ2(PETSC_ERR_ARG_INCOMP,"MatPtAPNumeric requires A, %s, to be compatible with P, %s",A->type_name,P->type_name);

  ierr = PetscLogEventBegin(MAT_PtAPNumeric,A,P,0,0);CHKERRQ(ierr); 
  ierr = (*fA)(A,P,C);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_PtAPNumeric,A,P,0,0);CHKERRQ(ierr); 

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPtAPNumeric_SeqAIJ_SeqAIJ"
PetscErrorCode MatPtAPNumeric_SeqAIJ_SeqAIJ(Mat A,Mat P,Mat C) 
{
  PetscErrorCode ierr;
  PetscInt       flops=0;
  Mat_SeqAIJ     *a  = (Mat_SeqAIJ *) A->data;
  Mat_SeqAIJ     *p  = (Mat_SeqAIJ *) P->data;
  Mat_SeqAIJ     *c  = (Mat_SeqAIJ *) C->data;
  PetscInt       *ai=a->i,*aj=a->j,*apj,*apjdense,*pi=p->i,*pj=p->j,*pJ=p->j,*pjj;
  PetscInt       *ci=c->i,*cj=c->j,*cjj;
  PetscInt       am=A->M,cn=C->N,cm=C->M;
  PetscInt       i,j,k,anzi,pnzi,apnzj,nextap,pnzj,prow,crow;
  MatScalar      *aa=a->a,*apa,*pa=p->a,*pA=p->a,*paj,*ca=c->a,*caj;

  PetscFunctionBegin;
  /* Allocate temporary array for storage of one row of A*P */
  ierr = PetscMalloc(cn*(sizeof(MatScalar)+2*sizeof(PetscInt)),&apa);CHKERRQ(ierr);
  ierr = PetscMemzero(apa,cn*(sizeof(MatScalar)+2*sizeof(PetscInt)));CHKERRQ(ierr);

  apj      = (PetscInt *)(apa + cn);
  apjdense = apj + cn;

  /* Clear old values in C */
  ierr = PetscMemzero(ca,ci[cm]*sizeof(MatScalar));CHKERRQ(ierr);

  for (i=0;i<am;i++) {
    /* Form sparse row of A*P */
    anzi  = ai[i+1] - ai[i];
    apnzj = 0;
    for (j=0;j<anzi;j++) {
      prow = *aj++;
      pnzj = pi[prow+1] - pi[prow];
      pjj  = pj + pi[prow];
      paj  = pa + pi[prow];
      for (k=0;k<pnzj;k++) {
        if (!apjdense[pjj[k]]) {
          apjdense[pjj[k]] = -1; 
          apj[apnzj++]     = pjj[k];
        }
        apa[pjj[k]] += (*aa)*paj[k];
      }
      flops += 2*pnzj;
      aa++;
    }

    /* Sort the j index array for quick sparse axpy. */
    ierr = PetscSortInt(apnzj,apj);CHKERRQ(ierr);

    /* Compute P^T*A*P using outer product (P^T)[:,j]*(A*P)[j,:]. */
    pnzi = pi[i+1] - pi[i];
    for (j=0;j<pnzi;j++) {
      nextap = 0;
      crow   = *pJ++;
      cjj    = cj + ci[crow];
      caj    = ca + ci[crow];
      /* Perform sparse axpy operation.  Note cjj includes apj. */
      for (k=0;nextap<apnzj;k++) {
        if (cjj[k]==apj[nextap]) {
          caj[k] += (*pA)*apa[apj[nextap++]];
        }
      }
      flops += 2*apnzj;
      pA++;
    }

    /* Zero the current row info for A*P */
    for (j=0;j<apnzj;j++) {
      apa[apj[j]]      = 0.;
      apjdense[apj[j]] = 0;
    }
  }

  /* Assemble the final matrix and clean up */
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscFree(apa);CHKERRQ(ierr);
  ierr = PetscLogFlops(flops);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* Compute local C = P_loc^T * A * P - used by MatPtAP_MPIAIJ_MPIAIJ() */
static PetscEvent logkey_matptapnumeric_local = 0;
#undef __FUNCT__
#define __FUNCT__ "MatPtAPNumeric_MPIAIJ_MPIAIJ_Local"
PetscErrorCode MatPtAPNumeric_MPIAIJ_MPIAIJ_Local(Mat A,Mat P_loc,Mat P_oth,PetscInt prstart,Mat C) 
{
  PetscErrorCode ierr;
  PetscInt       flops=0;
  Mat_MPIAIJ     *a=(Mat_MPIAIJ*)A->data;
  Mat_SeqAIJ     *ad=(Mat_SeqAIJ*)(a->A)->data,*ao=(Mat_SeqAIJ*)(a->B)->data;
  Mat_SeqAIJ     *c  = (Mat_SeqAIJ *) C->data;
  Mat_SeqAIJ     *p_loc=(Mat_SeqAIJ*)P_loc->data,*p_oth=(Mat_SeqAIJ*)P_oth->data;
  PetscInt       *pi_loc=p_loc->i,*pj_loc=p_loc->j,*pi_oth=p_oth->i,*pj_oth=p_oth->j;
  PetscInt       *adi=ad->i,*adj=ad->j,*aoi=ao->i,*aoj=ao->j,*apj,*apjdense,cstart=a->cstart,cend=a->cend,col;
  PetscInt       *pJ=pj_loc,*pjj; 
  PetscInt       *ci=c->i,*cj=c->j,*cjj;
  PetscInt       am=A->m,cn=C->N,cm=C->M;
  PetscInt       i,j,k,anzi,pnzi,apnzj,nextap,pnzj,prow,crow;
  MatScalar      *ada=ad->a,*aoa=ao->a,*apa,*paj,*ca=c->a,*caj; 
  MatScalar      *pa_loc=p_loc->a,*pA=pa_loc,*pa_oth=p_oth->a;

  PetscFunctionBegin;
  if (!logkey_matptapnumeric_local) {
    ierr = PetscLogEventRegister(&logkey_matptapnumeric_local,"MatPtAPNumeric_MPIAIJ_MPIAIJ_Local",MAT_COOKIE);CHKERRQ(ierr);
  }
  ierr = PetscLogEventBegin(logkey_matptapnumeric_local,A,0,0,0);CHKERRQ(ierr);

  /* Allocate temporary array for storage of one row of A*P */
  ierr = PetscMalloc(cn*(sizeof(MatScalar)+2*sizeof(PetscInt)),&apa);CHKERRQ(ierr);
  ierr = PetscMemzero(apa,cn*(sizeof(MatScalar)+2*sizeof(PetscInt)));CHKERRQ(ierr);
  apj      = (PetscInt *)(apa + cn);
  apjdense = apj + cn;

  /* Clear old values in C */
  ierr = PetscMemzero(ca,ci[cm]*sizeof(MatScalar));CHKERRQ(ierr);

  for (i=0;i<am;i++) {
    /* Form i-th sparse row of A*P */
     apnzj = 0;
    /* diagonal portion of A */
    anzi  = adi[i+1] - adi[i];
    for (j=0;j<anzi;j++) {
      prow = *adj; 
      adj++;
      pnzj = pi_loc[prow+1] - pi_loc[prow];
      pjj  = pj_loc + pi_loc[prow];
      paj  = pa_loc + pi_loc[prow];
      for (k=0;k<pnzj;k++) {
        if (!apjdense[pjj[k]]) {
          apjdense[pjj[k]] = -1; 
          apj[apnzj++]     = pjj[k];
        }
        apa[pjj[k]] += (*ada)*paj[k];
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
      for (k=0;k<pnzj;k++) {
        if (!apjdense[pjj[k]]) {
          apjdense[pjj[k]] = -1; 
          apj[apnzj++]     = pjj[k];
        }
        apa[pjj[k]] += (*aoa)*paj[k];
      }
      flops += 2*pnzj;
      aoa++;
    }
    /* Sort the j index array for quick sparse axpy. */
    ierr = PetscSortInt(apnzj,apj);CHKERRQ(ierr);

    /* Compute P_loc[i,:]^T*AP[i,:] using outer product */
    pnzi = pi_loc[i+1] - pi_loc[i];
    for (j=0;j<pnzi;j++) {
      nextap = 0;
      crow   = *pJ++;  
      cjj    = cj + ci[crow];
      caj    = ca + ci[crow];
      /* Perform sparse axpy operation.  Note cjj includes apj. */
      for (k=0;nextap<apnzj;k++) {
        if (cjj[k]==apj[nextap]) {
          caj[k] += (*pA)*apa[apj[nextap++]];
        }
      }
      flops += 2*apnzj;
      pA++;
    }

    /* Zero the current row info for A*P */
    for (j=0;j<apnzj;j++) {
      apa[apj[j]]      = 0.;
      apjdense[apj[j]] = 0;
    }
  }

  /* Assemble the final matrix and clean up */
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscFree(apa);CHKERRQ(ierr);
  ierr = PetscLogFlops(flops);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(logkey_matptapnumeric_local,A,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
static PetscEvent logkey_matptapsymbolic_local = 0;
#undef __FUNCT__
#define __FUNCT__ "MatPtAPSymbolic_MPIAIJ_MPIAIJ_Local"
PetscErrorCode MatPtAPSymbolic_MPIAIJ_MPIAIJ_Local(Mat A,Mat P_loc,Mat P_oth,PetscInt prstart,PetscReal fill,Mat *C) 
{
  PetscErrorCode ierr;
  FreeSpaceList  free_space=PETSC_NULL,current_space=PETSC_NULL;
  Mat_MPIAIJ     *a=(Mat_MPIAIJ*)A->data;
  Mat_SeqAIJ     *ad=(Mat_SeqAIJ*)(a->A)->data,*ao=(Mat_SeqAIJ*)(a->B)->data,*c; 
  PetscInt       *pti,*ptj,*ptJ,*ajj,*pjj; 
  PetscInt       *adi=ad->i,*adj=ad->j,*aoi=ao->i,*aoj=ao->j,nnz,cstart=a->cstart,cend=a->cend,col;  
  PetscInt       *ci,*cj,*ptaj; 
  PetscInt       aN=A->N,am=A->m,pN=P_loc->N;
  PetscInt       i,j,k,ptnzi,arow,anzj,prow,pnzj,cnzi;
  PetscInt       pm=P_loc->m,nlnk,*lnk;
  MatScalar      *ca;
  PetscBT        lnkbt;
  PetscInt       prend,nprow_loc,nprow_oth;
  PetscInt       *ptadenserow_loc,*ptadenserow_oth,*ptasparserow_loc,*ptasparserow_oth;  
  Mat_SeqAIJ     *p_loc=(Mat_SeqAIJ*)P_loc->data,*p_oth=(Mat_SeqAIJ*)P_oth->data;
  PetscInt       *pi_loc=p_loc->i,*pj_loc=p_loc->j,*pi_oth=p_oth->i,*pj_oth=p_oth->j;

  PetscFunctionBegin;
  if (!logkey_matptapsymbolic_local) {
    ierr = PetscLogEventRegister(&logkey_matptapsymbolic_local,"MatPtAPSymbolic_MPIAIJ_MPIAIJ_Local",MAT_COOKIE);
  }
  ierr = PetscLogEventBegin(logkey_matptapsymbolic_local,A,P_loc,P_oth,0);CHKERRQ(ierr);

  prend = prstart + pm;

  /* get ij structure of P_loc^T */
  ierr = MatGetSymbolicTranspose_SeqAIJ(P_loc,&pti,&ptj);CHKERRQ(ierr);
  ptJ=ptj;

  /* Allocate ci array, arrays for fill computation and */
  /* free space for accumulating nonzero column info */
  ierr = PetscMalloc((pN+1)*sizeof(PetscInt),&ci);CHKERRQ(ierr);
  ci[0] = 0;

  ierr = PetscMalloc((4*aN+1)*sizeof(PetscInt),&ptadenserow_loc);CHKERRQ(ierr);
  ierr = PetscMemzero(ptadenserow_loc,(4*aN+1)*sizeof(PetscInt));CHKERRQ(ierr);
  ptasparserow_loc = ptadenserow_loc + aN;
  ptadenserow_oth  = ptasparserow_loc + aN;
  ptasparserow_oth = ptadenserow_oth + aN;

  /* create and initialize a linked list */
  nlnk = pN+1;
  ierr = PetscLLCreate(pN,pN,nlnk,lnk,lnkbt);CHKERRQ(ierr);

  /* Set initial free space to be nnz(A) scaled by fill*P->N/P->M. */
  /* This should be reasonable if sparsity of PtAP is similar to that of A. */
  nnz           = adi[am] + aoi[am];
  ierr          = GetMoreSpace((PetscInt)(fill*nnz*pN/aN+1),&free_space);
  current_space = free_space;

  /* determine symbolic info for each row of C: */
  for (i=0;i<pN;i++) {
    ptnzi  = pti[i+1] - pti[i];
    nprow_loc = 0; nprow_oth = 0;
    /* i-th row of symbolic P_loc^T*A_loc: */
    for (j=0;j<ptnzi;j++) {
      arow = *ptJ++;
      /* diagonal portion of A */
      anzj = adi[arow+1] - adi[arow];
      ajj  = adj + adi[arow];
      for (k=0;k<anzj;k++) {
        col = ajj[k]+prstart;
        if (!ptadenserow_loc[col]) {
          ptadenserow_loc[col]    = -1;
          ptasparserow_loc[nprow_loc++] = col;
        }
      }
      /* off-diagonal portion of A */
      anzj = aoi[arow+1] - aoi[arow];
      ajj  = aoj + aoi[arow];
      for (k=0;k<anzj;k++) {
        col = a->garray[ajj[k]];  /* global col */
        if (col < cstart){
          col = ajj[k];
        } else if (col >= cend){
          col = ajj[k] + pm;
        } else {
          SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Off-diagonal portion of A has wrong column map");
        }
        if (!ptadenserow_oth[col]) {
          ptadenserow_oth[col]    = -1;
          ptasparserow_oth[nprow_oth++] = col;
        }
      }
    }
    
    /* using symbolic info of local PtA, determine symbolic info for row of C: */
    cnzi   = 0;
    /* rows of P_loc */
    ptaj = ptasparserow_loc;
    for (j=0; j<nprow_loc; j++) {
      prow = *ptaj++; 
      prow -= prstart; /* rm */
      pnzj = pi_loc[prow+1] - pi_loc[prow];
      pjj  = pj_loc + pi_loc[prow];
      /* add non-zero cols of P into the sorted linked list lnk */
      ierr = PetscLLAdd(pnzj,pjj,pN,nlnk,lnk,lnkbt);CHKERRQ(ierr);
      cnzi += nlnk;
    }
    /* rows of P_oth */
    ptaj = ptasparserow_oth;
    for (j=0; j<nprow_oth; j++) {
      prow = *ptaj++; 
      if (prow >= prend) prow -= pm; /* rm */
      pnzj = pi_oth[prow+1] - pi_oth[prow];
      pjj  = pj_oth + pi_oth[prow];
      /* add non-zero cols of P into the sorted linked list lnk */
      ierr = PetscLLAdd(pnzj,pjj,pN,nlnk,lnk,lnkbt);CHKERRQ(ierr);
      cnzi += nlnk;
    }
   
    /* If free space is not available, make more free space */
    /* Double the amount of total space in the list */
    if (current_space->local_remaining<cnzi) {
      ierr = GetMoreSpace(current_space->total_array_size,&current_space);CHKERRQ(ierr);
    }

    /* Copy data into free space, and zero out denserows */
    ierr = PetscLLClean(pN,pN,cnzi,lnk,current_space->array,lnkbt);CHKERRQ(ierr);
    current_space->array           += cnzi;
    current_space->local_used      += cnzi;
    current_space->local_remaining -= cnzi;
   
    for (j=0;j<nprow_loc; j++) {
      ptadenserow_loc[ptasparserow_loc[j]] = 0;
    }
    for (j=0;j<nprow_oth; j++) {
      ptadenserow_oth[ptasparserow_oth[j]] = 0;
    }
    
    /* Aside: Perhaps we should save the pta info for the numerical factorization. */
    /*        For now, we will recompute what is needed. */ 
    ci[i+1] = ci[i] + cnzi;
  }
  /* nnz is now stored in ci[ptm], column indices are in the list of free space */
  /* Allocate space for cj, initialize cj, and */
  /* destroy list of free space and other temporary array(s) */
  ierr = PetscMalloc((ci[pN]+1)*sizeof(PetscInt),&cj);CHKERRQ(ierr);
  ierr = MakeSpaceContiguous(&free_space,cj);CHKERRQ(ierr);
  ierr = PetscFree(ptadenserow_loc);CHKERRQ(ierr);
  ierr = PetscLLDestroy(lnk,lnkbt);CHKERRQ(ierr);
  
  /* Allocate space for ca */
  ierr = PetscMalloc((ci[pN]+1)*sizeof(MatScalar),&ca);CHKERRQ(ierr);
  ierr = PetscMemzero(ca,(ci[pN]+1)*sizeof(MatScalar));CHKERRQ(ierr);
  
  /* put together the new matrix */
  ierr = MatCreateSeqAIJWithArrays(PETSC_COMM_SELF,pN,pN,ci,cj,ca,C);CHKERRQ(ierr);

  /* MatCreateSeqAIJWithArrays flags matrix so PETSc doesn't free the user's arrays. */
  /* Since these are PETSc arrays, change flags to free them as necessary. */
  c = (Mat_SeqAIJ *)((*C)->data);
  c->freedata = PETSC_TRUE;
  c->nonew    = 0;

  /* Clean up. */
  ierr = MatRestoreSymbolicTranspose_SeqAIJ(P_loc,&pti,&ptj);CHKERRQ(ierr);
  
  ierr = PetscLogEventEnd(logkey_matptapsymbolic_local,A,P_loc,P_oth,0);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}
