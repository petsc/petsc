
/*
  Defines projective product routines where A is a SeqAIJ matrix
          C = P^T * A * P
*/

#include <../src/mat/impls/aij/seq/aij.h>   /*I "petscmat.h" I*/
#include <../src/mat/utils/freespace.h>
#include <petscbt.h>

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

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_SeqAIJ_PtAP"
PetscErrorCode MatDestroy_SeqAIJ_PtAP(Mat A)
{
  PetscErrorCode ierr;
  Mat_SeqAIJ     *a = (Mat_SeqAIJ *)A->data;
  Mat_PtAP       *ptap = a->ptap;

  PetscFunctionBegin;
  /* free ptap, then A */
  ierr = PetscFree(ptap->apa);CHKERRQ(ierr);
  ierr = PetscFree(ptap->api);CHKERRQ(ierr);
  ierr = PetscFree(ptap->apj);CHKERRQ(ierr);
  ierr = (ptap->destroy)(A);CHKERRQ(ierr);
  ierr = PetscFree(ptap);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPtAPSymbolic_SeqAIJ_SeqAIJ_SparseAxpy2"
PetscErrorCode MatPtAPSymbolic_SeqAIJ_SeqAIJ_SparseAxpy2(Mat A,Mat P,PetscReal fill,Mat *C)
{
  PetscErrorCode     ierr;
  PetscFreeSpaceList free_space=PETSC_NULL,current_space=PETSC_NULL;
  Mat_SeqAIJ         *a = (Mat_SeqAIJ*)A->data,*p = (Mat_SeqAIJ*)P->data,*c;
  PetscInt           *pti,*ptj,*ptJ,*ai=a->i,*aj=a->j,*ajj,*pi=p->i,*pj=p->j,*pjj;
  PetscInt           *ci,*cj,*ptadenserow,*ptasparserow,*ptaj,nspacedouble=0;
  PetscInt           an=A->cmap->N,am=A->rmap->N,pn=P->cmap->N;
  PetscInt           i,j,k,ptnzi,arow,anzj,ptanzi,prow,pnzj,cnzi,nlnk,*lnk;
  MatScalar          *ca;
  PetscBT            lnkbt;

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
      ierr = PetscLLAddSorted(pnzj,pjj,pn,nlnk,lnk,lnkbt);CHKERRQ(ierr);
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

  /* Allocate space for ca */
  ierr = PetscMalloc((ci[pn]+1)*sizeof(MatScalar),&ca);CHKERRQ(ierr);
  ierr = PetscMemzero(ca,(ci[pn]+1)*sizeof(MatScalar));CHKERRQ(ierr);

  /* put together the new matrix */
  ierr = MatCreateSeqAIJWithArrays(((PetscObject)A)->comm,pn,pn,ci,cj,ca,C);CHKERRQ(ierr);
  (*C)->rmap->bs = P->cmap->bs;
  (*C)->cmap->bs = P->cmap->bs;
PetscPrintf(PETSC_COMM_SELF,"************%s C.bs=%d,%d\n",__FUNCT__,(*C)->rmap->bs,(*C)->cmap->bs);
  /* MatCreateSeqAIJWithArrays flags matrix so PETSc doesn't free the user's arrays. */
  /* Since these are PETSc arrays, change flags to free them as necessary. */
  c = (Mat_SeqAIJ *)((*C)->data);
  c->free_a  = PETSC_TRUE;
  c->free_ij = PETSC_TRUE;
  c->nonew   = 0;
  A->ops->ptapnumeric = MatPtAPNumeric_SeqAIJ_SeqAIJ_SparseAxpy2; /* should use *C->ops until PtAP insterface is updated to double dispatch as MatMatMult() */

  /* Clean up. */
  ierr = MatRestoreSymbolicTranspose_SeqAIJ(P,&pti,&ptj);CHKERRQ(ierr);
#if defined(PETSC_USE_INFO)
  if (ci[pn] != 0) {
    PetscReal afill = ((PetscReal)ci[pn])/ai[am];
    if (afill < 1.0) afill = 1.0;
    ierr = PetscInfo3((*C),"Reallocs %D; Fill ratio: given %G needed %G.\n",nspacedouble,fill,afill);CHKERRQ(ierr);
    ierr = PetscInfo1((*C),"Use MatPtAP(A,P,MatReuse,%G,&C) for best performance.\n",afill);CHKERRQ(ierr);
  } else {
    ierr = PetscInfo((*C),"Empty matrix product\n");CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPtAPNumeric_SeqAIJ_SeqAIJ_SparseAxpy2"
PetscErrorCode MatPtAPNumeric_SeqAIJ_SeqAIJ_SparseAxpy2(Mat A,Mat P,Mat C)
{
  PetscErrorCode ierr;
  Mat_SeqAIJ     *a  = (Mat_SeqAIJ *) A->data;
  Mat_SeqAIJ     *p  = (Mat_SeqAIJ *) P->data;
  Mat_SeqAIJ     *c  = (Mat_SeqAIJ *) C->data;
  PetscInt       *ai=a->i,*aj=a->j,*apj,*apjdense,*pi=p->i,*pj=p->j,*pJ=p->j,*pjj;
  PetscInt       *ci=c->i,*cj=c->j,*cjj;
  PetscInt       am=A->rmap->N,cn=C->cmap->N,cm=C->rmap->N;
  PetscInt       i,j,k,anzi,pnzi,apnzj,nextap,pnzj,prow,crow;
  MatScalar      *aa=a->a,*apa,*pa=p->a,*pA=p->a,*paj,*ca=c->a,*caj;

  PetscFunctionBegin;
  /* Allocate temporary array for storage of one row of A*P */
  ierr = PetscMalloc(cn*(sizeof(MatScalar)+sizeof(PetscInt))+c->rmax*sizeof(PetscInt),&apa);CHKERRQ(ierr);

  apjdense = (PetscInt *)(apa + cn);
  apj      = apjdense + cn;
  ierr = PetscMemzero(apa,cn*(sizeof(MatScalar)+sizeof(PetscInt)));CHKERRQ(ierr);

  /* Clear old values in C */
  ierr = PetscMemzero(ca,ci[cm]*sizeof(MatScalar));CHKERRQ(ierr);

  for (i=0; i<am; i++) {
    /* Form sparse row of A*P */
    anzi  = ai[i+1] - ai[i];
    apnzj = 0;
    for (j=0; j<anzi; j++) {
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
      ierr = PetscLogFlops(2.0*pnzj);CHKERRQ(ierr);
      aa++;
    }

    /* Sort the j index array for quick sparse axpy. */
    /* Note: a array does not need sorting as it is in dense storage locations. */
    ierr = PetscSortInt(apnzj,apj);CHKERRQ(ierr);

    /* Compute P^T*A*P using outer product (P^T)[:,j]*(A*P)[j,:]. */
    pnzi = pi[i+1] - pi[i];
    for (j=0; j<pnzi; j++) {
      nextap = 0;
      crow   = *pJ++;
      cjj    = cj + ci[crow];
      caj    = ca + ci[crow];
      /* Perform sparse axpy operation.  Note cjj includes apj. */
      for (k=0;nextap<apnzj;k++) {
#if defined(PETSC_USE_DEBUG)
        if (k >= ci[crow+1] - ci[crow]) {
          SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"k too large k %d, crow %d",k,crow);
        }
#endif
        if (cjj[k]==apj[nextap]) {
          caj[k] += (*pA)*apa[apj[nextap++]];
        }
      }
      ierr = PetscLogFlops(2.0*apnzj);CHKERRQ(ierr);
      pA++;
    }

    /* Zero the current row info for A*P */
    for (j=0; j<apnzj; j++) {
      apa[apj[j]]      = 0.;
      apjdense[apj[j]] = 0;
    }
  }

  /* Assemble the final matrix and clean up */
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
PetscPrintf(PETSC_COMM_SELF,"************%s C.bs=%d,%d\n",__FUNCT__,C->rmap->bs,C->cmap->bs);
  ierr = PetscFree(apa);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPtAPSymbolic_SeqAIJ_SeqAIJ"
PetscErrorCode MatPtAPSymbolic_SeqAIJ_SeqAIJ(Mat A,Mat P,PetscReal fill,Mat *C)
{
  PetscErrorCode     ierr;
  Mat_SeqAIJ         *ap,*c;
  PetscInt           *api,*apj,*ci,pn=P->cmap->N,sparse_axpy=0;
  MatScalar          *ca;
  Mat_PtAP           *ptap;
  Mat                Pt,AP;

  PetscFunctionBegin;
  /* flag 'sparse_axpy' determines which implementations to be used:
       0: do dense axpy in MatPtAPNumeric() - fastest, but requires storage of struct A*P; (default)
       1: do one sparse axpy - uses same memory as sparse_axpy=0 and might execute less flops
          (apnz vs. cnz in the outerproduct), slower than case '0' when cnz is not too large than apnz;
       2: do two sparse axpy in MatPtAPNumeric() - slowest, does not store structure of A*P. */
  ierr = PetscOptionsGetInt(PETSC_NULL,"-matptap_sparseaxpy",&sparse_axpy,PETSC_NULL);CHKERRQ(ierr);
  if (sparse_axpy == 2){
    ierr = MatPtAPSymbolic_SeqAIJ_SeqAIJ_SparseAxpy2(A,P,fill,C);CHKERRQ(ierr);

    PetscFunctionReturn(0);
  }

  /* Get symbolic Pt = P^T */
  ierr = MatTransposeSymbolic_SeqAIJ(P,&Pt);CHKERRQ(ierr);

  /* Get symbolic AP = A*P */
  ierr = MatMatMultSymbolic_SeqAIJ_SeqAIJ(A,P,fill,&AP);CHKERRQ(ierr);

  ap          = (Mat_SeqAIJ*)AP->data;
  api         = ap->i;
  apj         = ap->j;
  ap->free_ij = PETSC_FALSE; /* api and apj are kept in struct ptap, cannot be destroyed with AP */

  /* Get C = Pt*AP */
  ierr = MatMatMultSymbolic_SeqAIJ_SeqAIJ(Pt,AP,fill,C);CHKERRQ(ierr);

  c  = (Mat_SeqAIJ*)(*C)->data;
  ci = c->i;
  ierr = PetscMalloc((ci[pn]+1)*sizeof(MatScalar),&ca);CHKERRQ(ierr);
  ierr = PetscMemzero(ca,(ci[pn]+1)*sizeof(MatScalar));CHKERRQ(ierr);
  c->a       = ca;
  c->free_a  = PETSC_TRUE;

  /* Create a supporting struct for reuse by MatPtAPNumeric() */
  ierr = PetscNew(Mat_PtAP,&ptap);CHKERRQ(ierr);
  c->ptap            = ptap;
  ptap->destroy      = (*C)->ops->destroy;
  (*C)->ops->destroy = MatDestroy_SeqAIJ_PtAP;

  /* Allocate temporary array for storage of one row of A*P */
  ierr = PetscMalloc((pn+1)*sizeof(PetscScalar),&ptap->apa);CHKERRQ(ierr);
  ierr = PetscMemzero(ptap->apa,(pn+1)*sizeof(PetscScalar));CHKERRQ(ierr);
  if (sparse_axpy == 1){
    A->ops->ptapnumeric = MatPtAPNumeric_SeqAIJ_SeqAIJ_SparseAxpy;
  } else {
    A->ops->ptapnumeric = MatPtAPNumeric_SeqAIJ_SeqAIJ;
  }
  ptap->api = api;
  ptap->apj = apj;

  /* Clean up. */
  ierr = MatDestroy(&Pt);CHKERRQ(ierr);
  ierr = MatDestroy(&AP);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* #define PROFILE_MatPtAPNumeric */
#undef __FUNCT__
#define __FUNCT__ "MatPtAPNumeric_SeqAIJ_SeqAIJ"
PetscErrorCode MatPtAPNumeric_SeqAIJ_SeqAIJ(Mat A,Mat P,Mat C)
{
  PetscErrorCode    ierr;
  Mat_SeqAIJ        *a  = (Mat_SeqAIJ *) A->data;
  Mat_SeqAIJ        *p  = (Mat_SeqAIJ *) P->data;
  Mat_SeqAIJ        *c  = (Mat_SeqAIJ *) C->data;
  const PetscInt    *ai=a->i,*aj=a->j,*pi=p->i,*pj=p->j,*ci=c->i,*cj=c->j;
  const PetscScalar *aa=a->a,*pa=p->a,*pval;
  const PetscInt    *apj,*pcol,*cjj;
  const PetscInt    am=A->rmap->N,cm=C->rmap->N;
  PetscInt          i,j,k,anz,apnz,pnz,prow,crow,cnz;
  PetscScalar       *apa,*ca=c->a,*caj,pvalj;
  Mat_PtAP          *ptap = c->ptap;
#if defined(PROFILE_MatPtAPNumeric)
  PetscLogDouble    t0,tf,time_Cseq0=0.0,time_Cseq1=0.0;
  PetscInt          flops0=0,flops1=0;
#endif

  PetscFunctionBegin;
  /* Get temporary array for storage of one row of A*P */
  apa = ptap->apa;

  /* Clear old values in C */
  ierr = PetscMemzero(ca,ci[cm]*sizeof(MatScalar));CHKERRQ(ierr);

  for (i=0;i<am;i++) {
    /* Form sparse row of AP[i,:] = A[i,:]*P */
#if defined(PROFILE_MatPtAPNumeric)
    ierr = PetscGetTime(&t0);CHKERRQ(ierr);
#endif
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
#if defined(PROFILE_MatPtAPNumeric)
      flops0 += 2.0*pnz;
#endif
    }
    aj += anz; aa += anz;
#if defined(PROFILE_MatPtAPNumeric)
    ierr = PetscGetTime(&tf);CHKERRQ(ierr);
    time_Cseq0 += tf - t0;
#endif

    /* Compute P^T*A*P using outer product P[i,:]^T*AP[i,:]. */
#if defined(PROFILE_MatPtAPNumeric)
    ierr = PetscGetTime(&t0);CHKERRQ(ierr);
#endif
    apj  = ptap->apj + ptap->api[i];
    apnz = ptap->api[i+1] - ptap->api[i];
    pnz  = pi[i+1] - pi[i];
    pcol = pj + pi[i];
    pval = pa + pi[i];

    /* Perform dense axpy */
    for (j=0; j<pnz; j++) {
      crow  = pcol[j];
      cjj   = cj + ci[crow];
      caj   = ca + ci[crow];
      pvalj = pval[j];
      cnz   = ci[crow+1] - ci[crow];
      for (k=0; k<cnz; k++){
        caj[k] += pvalj*apa[cjj[k]];
      }
      ierr = PetscLogFlops(2.0*cnz);CHKERRQ(ierr);
#if defined(PROFILE_MatPtAPNumeric)
      flops1 += 2.0*cnz;
#endif
    }
#if defined(PROFILE_MatPtAPNumeric)
    ierr = PetscGetTime(&tf);CHKERRQ(ierr);
    time_Cseq1 += tf - t0;
#endif

    /* Zero the current row info for A*P */
    for (j=0; j<apnz; j++) apa[apj[j]] = 0.0;
  }

  /* Assemble the final matrix and clean up */
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
#if defined(PROFILE_MatPtAPNumeric)
  printf("PtAPNumeric_SeqAIJ time %g + %g, flops %d %d\n",time_Cseq0,time_Cseq1,flops0,flops1);
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatPtAPNumeric_SeqAIJ_SeqAIJ_SparseAxpy"
PetscErrorCode MatPtAPNumeric_SeqAIJ_SeqAIJ_SparseAxpy(Mat A,Mat P,Mat C)
{
  PetscErrorCode    ierr;
  Mat_SeqAIJ        *a  = (Mat_SeqAIJ *) A->data;
  Mat_SeqAIJ        *p  = (Mat_SeqAIJ *) P->data;
  Mat_SeqAIJ        *c  = (Mat_SeqAIJ *) C->data;
  const PetscInt    *ai=a->i,*aj=a->j,*pi=p->i,*pj=p->j,*ci=c->i,*cj=c->j;
  const PetscScalar *aa=a->a,*pa=p->a,*pval;
  const PetscInt    *apj,*pcol,*cjj;
  const PetscInt    am=A->rmap->N,cm=C->rmap->N;
  PetscInt          i,j,k,anz,apnz,pnz,prow,crow,apcol,nextap;
  PetscScalar       *apa,*ca=c->a,*caj,pvalj;
  Mat_PtAP          *ptap = c->ptap;
#if defined(PROFILE_MatPtAPNumeric)
  PetscLogDouble    t0,tf,time_Cseq0=0.0,time_Cseq1=0.0;
  PetscInt          flops0=0,flops1=0;
#endif

  PetscFunctionBegin;
  /* Get temporary array for storage of one row of A*P */
  apa = ptap->apa;

  /* Clear old values in C */
  ierr = PetscMemzero(ca,ci[cm]*sizeof(MatScalar));CHKERRQ(ierr);

  for (i=0;i<am;i++) {
    /* Form sparse row of AP[i,:] = A[i,:]*P */
#if defined(PROFILE_MatPtAPNumeric)
    ierr = PetscGetTime(&t0);CHKERRQ(ierr);
#endif
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
#if defined(PROFILE_MatPtAPNumeric)
      flops0 += 2.0*pnz;
#endif
    }
    aj += anz; aa += anz;
#if defined(PROFILE_MatPtAPNumeric)
    ierr = PetscGetTime(&tf);CHKERRQ(ierr);
    time_Cseq0 += tf - t0;
#endif

    /* Compute P^T*A*P using outer product P[i,:]^T*AP[i,:]. */
#if defined(PROFILE_MatPtAPNumeric)
    ierr = PetscGetTime(&t0);CHKERRQ(ierr);
#endif
    apj  = ptap->apj + ptap->api[i];
    apnz = ptap->api[i+1] - ptap->api[i];
    pnz  = pi[i+1] - pi[i];
    pcol = pj + pi[i];
    pval = pa + pi[i];

    /* Perform sparse axpy */
    for (j=0; j<pnz; j++) {
      crow   = pcol[j];
      cjj    = cj + ci[crow];
      caj    = ca + ci[crow];
      pvalj  = pval[j];
      nextap = 1;
      apcol  = apj[nextap];
      for (k=0; nextap<apnz; k++) {
        if (cjj[k] == apcol) {
          caj[k] += pvalj*apa[apcol];
          apcol   = apj[nextap++];
        }
      }
      ierr = PetscLogFlops(2.0*apnz);CHKERRQ(ierr);
#if defined(PROFILE_MatPtAPNumeric)
      flops1 += 2.0*apnz;
#endif
    }
#if defined(PROFILE_MatPtAPNumeric)
    ierr = PetscGetTime(&tf);CHKERRQ(ierr);
    time_Cseq1 += tf - t0;
#endif

    /* Zero the current row info for A*P */
    for (j=0; j<apnz; j++) {
      apcol      = apj[j];
      apa[apcol] = 0.;
    }
  }

  /* Assemble the final matrix and clean up */
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
#if defined(PROFILE_MatPtAPNumeric)
  printf("MatPtAPNumeric_SeqAIJ_SeqAIJ_SparseAxpy time %g + %g, flops %d %d\n",time_Cseq0,time_Cseq1,flops0,flops1);
#endif
  PetscFunctionReturn(0);
}
