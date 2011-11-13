
/*
  Defines projective product routines where A is a SeqAIJ matrix
          C = P^T * A * P
*/

#include <../src/mat/impls/aij/seq/aij.h>   /*I "petscmat.h" I*/
#include <../src/mat/utils/freespace.h>
#include <petscbt.h>

#undef __FUNCT__
#define __FUNCT__ "MatPtAPSymbolic_SeqAIJ"
PetscErrorCode MatPtAPSymbolic_SeqAIJ(Mat A,Mat P,PetscReal fill,Mat *C)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!P->ops->ptapsymbolic_seqaij) SETERRQ2(((PetscObject)A)->comm,PETSC_ERR_SUP,"Not implemented for A=%s and P=%s",((PetscObject)A)->type_name,((PetscObject)P)->type_name);
  ierr = (*P->ops->ptapsymbolic_seqaij)(A,P,fill,C);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPtAPNumeric_SeqAIJ"
PetscErrorCode MatPtAPNumeric_SeqAIJ(Mat A,Mat P,Mat C)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!P->ops->ptapnumeric_seqaij) SETERRQ2(((PetscObject)A)->comm,PETSC_ERR_SUP,"Not implemented for A=%s and P=%s",((PetscObject)A)->type_name,((PetscObject)P)->type_name);
  ierr = (*P->ops->ptapnumeric_seqaij)(A,P,C);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscContainerDestroy_Mat_PtAP"
PetscErrorCode PetscContainerDestroy_Mat_PtAP(void *ptr)
{
  PetscErrorCode ierr;
  Mat_PtAP       *ptap=(Mat_PtAP*)ptr;

  PetscFunctionBegin;
  ierr = PetscFree(ptap->apa);CHKERRQ(ierr);
  ierr = PetscFree(ptap->api);CHKERRQ(ierr);
  ierr = PetscFree(ptap->apj);CHKERRQ(ierr);
  ierr = PetscFree(ptap);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_SeqAIJ_PtAP"
PetscErrorCode MatDestroy_SeqAIJ_PtAP(Mat A)
{
  PetscErrorCode ierr;
  PetscContainer container;
  Mat_PtAP       *ptap=PETSC_NULL;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)A,"Mat_PtAP",(PetscObject *)&container);CHKERRQ(ierr);
  if (!container) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Container does not exit");
  ierr = PetscContainerGetPointer(container,(void **)&ptap);CHKERRQ(ierr);
  A->ops->destroy = ptap->destroy;
  if (A->ops->destroy) {
    ierr = (*A->ops->destroy)(A);CHKERRQ(ierr);
  }
  ierr = PetscObjectCompose((PetscObject)A,"Mat_PtAP",0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPtAPSymbolic_SeqAIJ_SeqAIJ"
PetscErrorCode MatPtAPSymbolic_SeqAIJ_SeqAIJ(Mat A,Mat P,PetscReal fill,Mat *C) 
{
  PetscErrorCode     ierr;
  Mat_SeqAIJ         *a = (Mat_SeqAIJ*)A->data,*p = (Mat_SeqAIJ*)P->data,*c;
  PetscInt           *pti,*ptj,*ptJ,*ai=a->i,*aj=a->j,*pi=p->i,*pj=p->j,*api,*apj;
  PetscInt           *ci,*cj,ndouble_ap,ndouble_ptap;
  PetscInt           an=A->cmap->N,am=A->rmap->N,pn=P->cmap->N;
  MatScalar          *ca;
  Mat_PtAP           *ptap;
  PetscContainer     container;

  PetscFunctionBegin;
  /* Get ij structure of Pt = P^T */
  ierr = MatGetSymbolicTranspose_SeqAIJ(P,&pti,&ptj);CHKERRQ(ierr);
  ptJ=ptj;

  /* Get structure of AP = A*P */
  ierr = MatGetSymbolicMatMatMult_SeqAIJ_SeqAIJ(am,ai,aj,an,pn,pi,pj,fill,&api,&apj,&ndouble_ap);CHKERRQ(ierr);

  /* Get structure of C = Pt*AP */
  ierr = MatGetSymbolicMatMatMult_SeqAIJ_SeqAIJ(pn,pti,ptj,am,pn,api,apj,fill,&ci,&cj,&ndouble_ptap);CHKERRQ(ierr);
#if defined(MV)
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

  /* Set initial free space to be fill*nnz(A). */
  /* This should be reasonable if sparsity of PtAP is similar to that of A. */
  ierr          = PetscFreeSpaceGet((PetscInt)(fill*ai[am]),&free_space);
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
      ierr = PetscFreeSpaceGet(cnzi+current_space->total_array_size,&current_space);CHKERRQ(ierr);
      nspacedouble++;
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
  ierr = PetscFreeSpaceContiguous(&free_space,cj);CHKERRQ(ierr);
  ierr = PetscFree(ptadenserow);CHKERRQ(ierr);
  ierr = PetscLLDestroy(lnk,lnkbt);CHKERRQ(ierr);

#endif  
  
  /* Allocate space for ca */
  ierr = PetscMalloc((ci[pn]+1)*sizeof(MatScalar),&ca);CHKERRQ(ierr);
  ierr = PetscMemzero(ca,(ci[pn]+1)*sizeof(MatScalar));CHKERRQ(ierr);
  
  /* put together the new matrix */
  ierr = MatCreateSeqAIJWithArrays(((PetscObject)A)->comm,pn,pn,ci,cj,ca,C);CHKERRQ(ierr);

  /* MatCreateSeqAIJWithArrays flags matrix so PETSc doesn't free the user's arrays. */
  /* Since these are PETSc arrays, change flags to free them as necessary. */
  c          = (Mat_SeqAIJ *)(*C)->data;
  c->free_a  = PETSC_TRUE;
  c->free_ij = PETSC_TRUE;
  c->nonew   = 0;

  /* create a supporting struct for reuse by MatPtAPNumeric(), attach it to *C */
  ierr = PetscNew(Mat_PtAP,&ptap);CHKERRQ(ierr);

  /* attach the supporting struct to C */
  ierr = PetscContainerCreate(PETSC_COMM_SELF,&container);CHKERRQ(ierr);
  ierr = PetscContainerSetPointer(container,ptap);CHKERRQ(ierr);
  ierr = PetscContainerSetUserDestroy(container,PetscContainerDestroy_Mat_PtAP);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)(*C),"Mat_PtAP",(PetscObject)container);CHKERRQ(ierr);
  ierr = PetscContainerDestroy(&container);CHKERRQ(ierr);

  ptap->destroy      = (*C)->ops->destroy;
  (*C)->ops->destroy = MatDestroy_SeqAIJ_PtAP;

  /* Allocate temporary array for storage of one row of A*P */
  ierr = PetscMalloc((pn+1)*sizeof(PetscScalar),&ptap->apa);CHKERRQ(ierr);
  ierr = PetscMemzero(ptap->apa,(pn+1)*sizeof(MatScalar));CHKERRQ(ierr); 
  ptap->api = api;
  ptap->apj = apj;

  /* Clean up. */
  ierr = MatRestoreSymbolicTranspose_SeqAIJ(P,&pti,&ptj);CHKERRQ(ierr);
#if defined(PETSC_USE_INFO)
  if (ci[pn] != 0) {
    PetscReal apfill,ptapfill;
    apfill = ((PetscReal)api[am])/(ai[am]+pi[an]);
    if (apfill < 1.0) apfill = 1.0;
    ierr = PetscInfo3((*C),"A*P: Reallocs %D; Fill ratio: given %G needed %G.\n",ndouble_ap,fill,apfill);CHKERRQ(ierr);
    ptapfill = ((PetscReal)ci[pn])/(pi[an]+api[am]);
    if (ptapfill < 1.0) ptapfill = 1.0;
    ierr = PetscInfo3((*C),"Pt*AP: Reallocs %D; Fill ratio: given %G needed %G.\n",ndouble_ptap,fill,ptapfill);CHKERRQ(ierr);
    
    ierr = PetscInfo1((*C),"Use MatPtAP(A,P,MatReuse,%G,&C) for best performance.\n",PetscMax(apfill,ptapfill));CHKERRQ(ierr);
    ierr = PetscInfo4((*C),"nonzeros: A %D, P %D, A*P %D, C=PtAP %D\n",ai[am],pi[an],api[am],ci[pn]);CHKERRQ(ierr);
  } else {
    ierr = PetscInfo((*C),"Empty matrix product\n");CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPtAPNumeric_SeqAIJ_SeqAIJ"
PetscErrorCode MatPtAPNumeric_SeqAIJ_SeqAIJ(Mat A,Mat P,Mat C) 
{
  PetscErrorCode  ierr;
  Mat_SeqAIJ      *a  = (Mat_SeqAIJ *) A->data;
  Mat_SeqAIJ      *p  = (Mat_SeqAIJ *) P->data; 
  Mat_SeqAIJ      *c  = (Mat_SeqAIJ *) C->data;
  PetscInt        *ai=a->i,*aj=a->j,*pi=p->i,*pj=p->j,*ci=c->i,*cj=c->j;
  PetscScalar     *aa=a->a,*pa=p->a;
  PetscInt        *apj,*pcol,*cjj,cnz;
  PetscInt        am=A->rmap->N,cm=C->rmap->N;
  PetscInt        i,j,k,anz,apnz,pnz,prow,crow,apcol,nextap;
  PetscScalar     *apa,*pval,*ca=c->a,*caj;
  PetscBool       sparse_axpy=PETSC_FALSE;
  Mat_PtAP        *ptap;
  PetscContainer  container;

  PetscFunctionBegin;
  ierr = PetscOptionsGetBool(PETSC_NULL,"-matptap_spaxpy",&sparse_axpy,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject)C,"Mat_PtAP",(PetscObject *)&container);CHKERRQ(ierr);
  if (!container) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Container does not exit");
  ierr  = PetscContainerGetPointer(container,(void **)&ptap);CHKERRQ(ierr);

  /* Get temporary array for storage of one row of A*P */
  apa = ptap->apa;
  
  /* Clear old values in C */
  ierr = PetscMemzero(ca,ci[cm]*sizeof(MatScalar));CHKERRQ(ierr);

  for (i=0;i<am;i++) {
    /* Form sparse row of AP[i,:] = A[i,:]*P */
    anz  = ai[i+1] - ai[i];
    apnz = 0;
    for (j=0; j<anz; j++) {
      prow = aj[j];
      pnz  = pi[prow+1] - pi[prow];
      pcol = pj + pi[prow];
      pval = pa + pi[prow];
      for (k=0; k<pnz; k++) {
        apa[pcol[k]] += aa[j]*pval[k];
      }
      ierr = PetscLogFlops(2.0*pnz);CHKERRQ(ierr);
    }
    aj += anz; aa += anz;
    
    /* Compute P^T*A*P using outer product P[i,:]^T*AP[i,:]. */
    apj  = ptap->apj + ptap->api[i]; 
    apnz = ptap->api[i+1] - ptap->api[i];
    pnz  = pi[i+1] - pi[i];
    pcol = pj + pi[i];
    pval = pa + pi[i];
   
    if (sparse_axpy){  /* Perform sparse axpy */
      for (j=0; j<pnz; j++) {
        crow   = pcol[j]; 
        cjj    = cj + ci[crow];
        caj    = ca + ci[crow];
        nextap = 0;
        apcol  = apj[nextap];
        for (k=0; nextap<apnz; k++) {
          if (cjj[k] == apcol) {
            caj[k] += pval[j]*apa[apcol];
            apcol   = apj[++nextap];
          }
        }
        ierr = PetscLogFlops(2.0*apnz);CHKERRQ(ierr);
      }
    } else { /* Perform dense axpy */
      for (j=0; j<pnz; j++) {
        crow = pcol[j]; 
        cjj  = cj + ci[crow];
        caj  = ca + ci[crow];
        cnz  = ci[crow+1] - ci[crow];
        for (k=0; k<cnz; k++){
          caj[k] += pval[j]*apa[cjj[k]];
        }
        ierr = PetscLogFlops(2.0*cnz);CHKERRQ(ierr);
      }
    }

    /* Zero the current row info for A*P */
    for (j=0; j<apnz; j++) {
      apcol      = apj[j];
      apa[apcol] = 0.;
    }
  }

  /* Assemble the final matrix and clean up */
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
