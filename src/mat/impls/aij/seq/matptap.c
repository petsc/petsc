
/*
  Defines projective product routines where A is a SeqAIJ matrix
          C = P^T * A * P
*/

#include <../src/mat/impls/aij/seq/aij.h>   /*I "petscmat.h" I*/
#include <../src/mat/utils/freespace.h>
#include <petscbt.h>
#include <petsctime.h>

#if defined(PETSC_HAVE_HYPRE)
PETSC_INTERN PetscErrorCode MatPtAPSymbolic_AIJ_AIJ_wHYPRE(Mat,Mat,PetscReal,Mat);
#endif

PetscErrorCode MatProductSymbolic_PtAP_SeqAIJ_SeqAIJ(Mat C)
{
  PetscErrorCode      ierr;
  Mat_Product         *product = C->product;
  Mat                 A=product->A,P=product->B;
  MatProductAlgorithm alg=product->alg;
  PetscReal           fill=product->fill;
  PetscBool           flg;
  Mat                 Pt;
  Mat_MatTransMatMult *atb;
  Mat_SeqAIJ          *c;

  PetscFunctionBegin;
  /* "scalable" */
  ierr = PetscStrcmp(alg,"scalable",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatPtAPSymbolic_SeqAIJ_SeqAIJ_SparseAxpy(A,P,fill,C);CHKERRQ(ierr);
    C->ops->productnumeric = MatProductNumeric_PtAP;
    PetscFunctionReturn(0);
  }

  /* "rap" */
  ierr = PetscStrcmp(alg,"rap",&flg);CHKERRQ(ierr);
  if (flg) { /* Set default algorithm */
    ierr = PetscNew(&atb);CHKERRQ(ierr);
    ierr = MatTranspose_SeqAIJ(P,MAT_INITIAL_MATRIX,&Pt);CHKERRQ(ierr);
    ierr = MatMatMatMultSymbolic_SeqAIJ_SeqAIJ_SeqAIJ(Pt,A,P,fill,C);CHKERRQ(ierr);

    c                      = (Mat_SeqAIJ*)C->data;
    c->atb                 = atb;
    atb->At                = Pt;
    atb->destroy           = C->ops->destroy;
    C->ops->destroy        = MatDestroy_SeqAIJ_MatTransMatMult;
    C->ops->ptapnumeric    = MatPtAPNumeric_SeqAIJ_SeqAIJ;
    C->ops->productnumeric = MatProductNumeric_PtAP;
    PetscFunctionReturn(0);
  }

  /* hypre */
#if defined(PETSC_HAVE_HYPRE)
  ierr = PetscStrcmp(alg,"hypre",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatPtAPSymbolic_AIJ_AIJ_wHYPRE(A,P,fill,C);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
#endif

  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"MatProductType is not supported");
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_SeqAIJ_PtAP(Mat A)
{
  PetscErrorCode ierr;
  Mat_SeqAIJ     *a  = (Mat_SeqAIJ*)A->data;
  Mat_AP         *ap = a->ap;

  PetscFunctionBegin;
  ierr = PetscFree(ap->apa);CHKERRQ(ierr);
  ierr = PetscFree(ap->api);CHKERRQ(ierr);
  ierr = PetscFree(ap->apj);CHKERRQ(ierr);
  ierr = (ap->destroy)(A);CHKERRQ(ierr);
  ierr = PetscFree(ap);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatPtAPSymbolic_SeqAIJ_SeqAIJ_SparseAxpy(Mat A,Mat P,PetscReal fill,Mat C)
{
  PetscErrorCode     ierr;
  PetscFreeSpaceList free_space=NULL,current_space=NULL;
  Mat_SeqAIJ         *a        = (Mat_SeqAIJ*)A->data,*p = (Mat_SeqAIJ*)P->data,*c;
  PetscInt           *pti,*ptj,*ptJ,*ai=a->i,*aj=a->j,*ajj,*pi=p->i,*pj=p->j,*pjj;
  PetscInt           *ci,*cj,*ptadenserow,*ptasparserow,*ptaj,nspacedouble=0;
  PetscInt           an=A->cmap->N,am=A->rmap->N,pn=P->cmap->N,pm=P->rmap->N;
  PetscInt           i,j,k,ptnzi,arow,anzj,ptanzi,prow,pnzj,cnzi,nlnk,*lnk;
  MatScalar          *ca;
  PetscBT            lnkbt;
  PetscReal          afill;

  PetscFunctionBegin;
  /* Get ij structure of P^T */
  ierr = MatGetSymbolicTranspose_SeqAIJ(P,&pti,&ptj);CHKERRQ(ierr);
  ptJ  = ptj;

  /* Allocate ci array, arrays for fill computation and */
  /* free space for accumulating nonzero column info */
  ierr  = PetscMalloc1(pn+1,&ci);CHKERRQ(ierr);
  ci[0] = 0;

  ierr         = PetscCalloc1(2*an+1,&ptadenserow);CHKERRQ(ierr);
  ptasparserow = ptadenserow  + an;

  /* create and initialize a linked list */
  nlnk = pn+1;
  ierr = PetscLLCreate(pn,pn,nlnk,lnk,lnkbt);CHKERRQ(ierr);

  /* Set initial free space to be fill*(nnz(A)+ nnz(P)) */
  ierr          = PetscFreeSpaceGet(PetscRealIntMultTruncate(fill,PetscIntSumTruncate(ai[am],pi[pm])),&free_space);CHKERRQ(ierr);
  current_space = free_space;

  /* Determine symbolic info for each row of C: */
  for (i=0; i<pn; i++) {
    ptnzi  = pti[i+1] - pti[i];
    ptanzi = 0;
    /* Determine symbolic row of PtA: */
    for (j=0; j<ptnzi; j++) {
      arow = *ptJ++;
      anzj = ai[arow+1] - ai[arow];
      ajj  = aj + ai[arow];
      for (k=0; k<anzj; k++) {
        if (!ptadenserow[ajj[k]]) {
          ptadenserow[ajj[k]]    = -1;
          ptasparserow[ptanzi++] = ajj[k];
        }
      }
    }
    /* Using symbolic info for row of PtA, determine symbolic info for row of C: */
    ptaj = ptasparserow;
    cnzi = 0;
    for (j=0; j<ptanzi; j++) {
      prow = *ptaj++;
      pnzj = pi[prow+1] - pi[prow];
      pjj  = pj + pi[prow];
      /* add non-zero cols of P into the sorted linked list lnk */
      ierr  = PetscLLAddSorted(pnzj,pjj,pn,nlnk,lnk,lnkbt);CHKERRQ(ierr);
      cnzi += nlnk;
    }

    /* If free space is not available, make more free space */
    /* Double the amount of total space in the list */
    if (current_space->local_remaining<cnzi) {
      ierr = PetscFreeSpaceGet(PetscIntSumTruncate(cnzi,current_space->total_array_size),&current_space);CHKERRQ(ierr);
      nspacedouble++;
    }

    /* Copy data into free space, and zero out denserows */
    ierr = PetscLLClean(pn,pn,cnzi,lnk,current_space->array,lnkbt);CHKERRQ(ierr);

    current_space->array           += cnzi;
    current_space->local_used      += cnzi;
    current_space->local_remaining -= cnzi;

    for (j=0; j<ptanzi; j++) ptadenserow[ptasparserow[j]] = 0;

    /* Aside: Perhaps we should save the pta info for the numerical factorization. */
    /*        For now, we will recompute what is needed. */
    ci[i+1] = ci[i] + cnzi;
  }
  /* nnz is now stored in ci[ptm], column indices are in the list of free space */
  /* Allocate space for cj, initialize cj, and */
  /* destroy list of free space and other temporary array(s) */
  ierr = PetscMalloc1(ci[pn]+1,&cj);CHKERRQ(ierr);
  ierr = PetscFreeSpaceContiguous(&free_space,cj);CHKERRQ(ierr);
  ierr = PetscFree(ptadenserow);CHKERRQ(ierr);
  ierr = PetscLLDestroy(lnk,lnkbt);CHKERRQ(ierr);

  ierr = PetscCalloc1(ci[pn]+1,&ca);CHKERRQ(ierr);

  /* put together the new matrix */
  ierr = MatSetSeqAIJWithArrays_private(PetscObjectComm((PetscObject)A),pn,pn,ci,cj,ca,((PetscObject)A)->type_name,C);CHKERRQ(ierr);
  ierr = MatSetBlockSizes(C,PetscAbs(P->cmap->bs),PetscAbs(P->cmap->bs));CHKERRQ(ierr);

  /* MatCreateSeqAIJWithArrays flags matrix so PETSc doesn't free the user's arrays. */
  /* Since these are PETSc arrays, change flags to free them as necessary. */
  c          = (Mat_SeqAIJ*)((C)->data);
  c->free_a  = PETSC_TRUE;
  c->free_ij = PETSC_TRUE;
  c->nonew   = 0;
  C->ops->ptapnumeric    = MatPtAPNumeric_SeqAIJ_SeqAIJ_SparseAxpy;

  /* set MatInfo */
  afill = (PetscReal)ci[pn]/(ai[am]+pi[pm] + 1.e-5);
  if (afill < 1.0) afill = 1.0;
  c->maxnz                     = ci[pn];
  c->nz                        = ci[pn];
  C->info.mallocs           = nspacedouble;
  C->info.fill_ratio_given  = fill;
  C->info.fill_ratio_needed = afill;

  /* Clean up. */
  ierr = MatRestoreSymbolicTranspose_SeqAIJ(P,&pti,&ptj);CHKERRQ(ierr);
#if defined(PETSC_USE_INFO)
  if (ci[pn] != 0) {
    ierr = PetscInfo3(C,"Reallocs %D; Fill ratio: given %g needed %g.\n",nspacedouble,(double)fill,(double)afill);CHKERRQ(ierr);
    ierr = PetscInfo1(C,"Use MatPtAP(A,P,MatReuse,%g,&C) for best performance.\n",(double)afill);CHKERRQ(ierr);
  } else {
    ierr = PetscInfo(C,"Empty matrix product\n");CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode MatPtAPNumeric_SeqAIJ_SeqAIJ_SparseAxpy(Mat A,Mat P,Mat C)
{
  PetscErrorCode ierr;
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*) A->data;
  Mat_SeqAIJ     *p = (Mat_SeqAIJ*) P->data;
  Mat_SeqAIJ     *c = (Mat_SeqAIJ*) C->data;
  PetscInt       *ai=a->i,*aj=a->j,*apj,*apjdense,*pi=p->i,*pj=p->j,*pJ=p->j,*pjj;
  PetscInt       *ci=c->i,*cj=c->j,*cjj;
  PetscInt       am =A->rmap->N,cn=C->cmap->N,cm=C->rmap->N;
  PetscInt       i,j,k,anzi,pnzi,apnzj,nextap,pnzj,prow,crow;
  MatScalar      *aa=a->a,*apa,*pa=p->a,*pA=p->a,*paj,*ca=c->a,*caj;

  PetscFunctionBegin;
  /* Allocate temporary array for storage of one row of A*P (cn: non-scalable) */
  ierr = PetscCalloc2(cn,&apa,cn,&apjdense);CHKERRQ(ierr);
  ierr = PetscMalloc1(cn,&apj);CHKERRQ(ierr);

  /* Clear old values in C */
  ierr = PetscArrayzero(ca,ci[cm]);CHKERRQ(ierr);

  for (i=0; i<am; i++) {
    /* Form sparse row of A*P */
    anzi  = ai[i+1] - ai[i];
    apnzj = 0;
    for (j=0; j<anzi; j++) {
      prow = *aj++;
      pnzj = pi[prow+1] - pi[prow];
      pjj  = pj + pi[prow];
      paj  = pa + pi[prow];
      for (k=0; k<pnzj; k++) {
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
      for (k=0; nextap<apnzj; k++) {
        if (PetscUnlikelyDebug(k >= ci[crow+1] - ci[crow])) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"k too large k %d, crow %d",k,crow);
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

  ierr = PetscFree2(apa,apjdense);CHKERRQ(ierr);
  ierr = PetscFree(apj);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatPtAPNumeric_SeqAIJ_SeqAIJ(Mat A,Mat P,Mat C)
{
  PetscErrorCode      ierr;
  Mat_SeqAIJ          *c = (Mat_SeqAIJ*)C->data;
  Mat_MatTransMatMult *atb = c->atb;
  Mat                 Pt = atb->At;

  PetscFunctionBegin;
  ierr = MatTranspose_SeqAIJ(P,MAT_REUSE_MATRIX,&Pt);CHKERRQ(ierr);
  ierr = (C->ops->matmatmultnumeric)(Pt,A,P,C);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
