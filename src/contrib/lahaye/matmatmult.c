/*$Id: matmatmult.c,v 1.15 2001/09/07 20:04:44 buschelm Exp $*/
/*
  Defines a matrix-matrix product for 2 SeqAIJ matrices
          C = A * B
*/

#include "src/mat/impls/aij/seq/aij.h"

typedef struct _Space *FreeSpaceList;
typedef struct _Space {
  FreeSpaceList more_space;
  int           *array;
  int           *array_head;
  int           total_array_size;
  int           local_used;
  int           local_remaining;
} FreeSpace;  

#undef __FUNCT__
#define __FUNCT__ "GetMoreSpace"
int GetMoreSpace(int size,FreeSpaceList *list) {
  FreeSpaceList a;
  int ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc(sizeof(FreeSpace),&a);CHKERRQ(ierr);
  ierr = PetscMalloc(size*sizeof(int),&(a->array_head));CHKERRQ(ierr);
  a->array            = a->array_head;
  a->local_remaining  = size;
  a->local_used       = 0;
  a->total_array_size = 0;
  a->more_space       = NULL;

  if (*list) {
    (*list)->more_space = a;
    a->total_array_size = (*list)->total_array_size;
  }

  a->total_array_size += size;
  *list               =  a;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MakeSpaceContiguous"
int MakeSpaceContiguous(int *space,FreeSpaceList *head) {
  FreeSpaceList a;
  int           ierr;

  PetscFunctionBegin;
  while ((*head)!=NULL) {
    a     =  (*head)->more_space;
    ierr  =  PetscMemcpy(space,(*head)->array_head,((*head)->local_used)*sizeof(int));CHKERRQ(ierr);
    space += (*head)->local_used;
    ierr  =  PetscFree((*head)->array_head);CHKERRQ(ierr);
    ierr  =  PetscFree(*head);CHKERRQ(ierr);
    *head =  a;
  }
  PetscFunctionReturn(0);
}

static int logkey_matmatmult_symbolic = 0;
static int logkey_matmatmult_numeric  = 0;

/*
     MatMatMult_SeqAIJ_SeqAIJ_Symbolic - Forms the symbolic product of two SeqAIJ matrices
           C=A*B;

     Note: C is assumed to be uninitialized.
           If this is not the case, Destroy C before calling this routine.
*/
#undef __FUNCT__  
#define __FUNCT__ "MatMatMult_SeqAIJ_SeqAIJ_Symbolic"
int MatMatMult_SeqAIJ_SeqAIJ_Symbolic(Mat A,Mat B,Mat *C)
{
  int            ierr;
  FreeSpaceList  free_space=PETSC_NULL,current_space=PETSC_NULL;
  Mat_SeqAIJ     *a=(Mat_SeqAIJ*)A->data,*b=(Mat_SeqAIJ*)B->data,*c;
  int            aishift=a->indexshift,bishift=b->indexshift;
  int            *ai=a->i,*aj=a->j,*bi=b->i,*bj=b->j,*bjj;
  int            *ci,*cj,*densefill,*sparsefill;
  int            an=A->N,am=A->M,bn=B->N,bm=B->M;
  int            i,j,k,anzi,brow,bnzj,cnzi;
  MatScalar      *ca;

  PetscFunctionBegin;
  /* some error checking which could be moved into interface layer */
  if (aishift || bishift) SETERRQ(PETSC_ERR_SUP,"Shifted matrix indices are not supported.");
  if (an!=bm) SETERRQ2(PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, %d != %d",an,bm);
  
  if (!logkey_matmatmult_symbolic) {
    ierr = PetscLogEventRegister(&logkey_matmatmult_symbolic,"MatMatMult_Symbolic",MAT_COOKIE);CHKERRQ(ierr);
  }
  ierr = PetscLogEventBegin(logkey_matmatmult_symbolic,A,B,0,0);CHKERRQ(ierr);

  /* Set up */
  /* Allocate ci array, arrays for fill computation and */
  /* free space for accumulating nonzero column info */
  ierr = PetscMalloc(((am+1)+1)*sizeof(int),&ci);CHKERRQ(ierr);
  ci[0] = 0;

  ierr = PetscMalloc((2*bn+1)*sizeof(int),&densefill);CHKERRQ(ierr);
  ierr = PetscMemzero(densefill,(2*bn+1)*sizeof(int));CHKERRQ(ierr);
  sparsefill = densefill + bn;

  /* Initial FreeSpace size is nnz(B)=bi[bm] */
  ierr          = GetMoreSpace(bi[bm],&free_space);CHKERRQ(ierr);
  current_space = free_space;

  /* Determine fill for each row: */
  for (i=0;i<am;i++) {
    anzi = ai[i+1] - ai[i];
    cnzi = 0;
    for (j=0;j<anzi;j++) {
      brow = *aj++;
      bnzj = bi[brow+1] - bi[brow];
      bjj  = bj + bi[brow];
      for (k=0;k<bnzj;k++) {
        /* If column is not marked, mark it in compressed and uncompressed locations. */
        /* For simplicity, leave uncompressed row unsorted until finished with row, */
        /* and increment nonzero count for this row. */
        if (!densefill[bjj[k]]) {
          densefill[bjj[k]]  = -1;
          sparsefill[cnzi++] = bjj[k];
        }
      }
    }

    /* sort sparsefill */
    ierr = PetscSortInt(cnzi,sparsefill);CHKERRQ(ierr);

    /* If free space is not available, make more free space */
    /* Double the amount of total space in the list */
    if (current_space->local_remaining<cnzi) {
      ierr = GetMoreSpace(current_space->total_array_size,&current_space);CHKERRQ(ierr);
    }

    /* Copy data into free space, and zero out densefill */
    ierr = PetscMemcpy(current_space->array,sparsefill,cnzi*sizeof(int));CHKERRQ(ierr);
    current_space->array           += cnzi;
    current_space->local_used      += cnzi;
    current_space->local_remaining -= cnzi;
    for (j=0;j<cnzi;j++) {
      densefill[sparsefill[j]] = 0;
    }
    ci[i+1] = ci[i] + cnzi;
  }

  /* nnz is now stored in ci[am], column indices are in the list of free space */
  /* Allocate space for cj, initialize cj, and */
  /* destroy list of free space and other temporary array(s) */
  ierr = PetscMalloc((ci[am]+1)*sizeof(int),&cj);CHKERRQ(ierr);
  ierr = MakeSpaceContiguous(cj,&free_space);CHKERRQ(ierr);
  ierr = PetscFree(densefill);CHKERRQ(ierr);
    
  /* Allocate space for ca */
  ierr = PetscMalloc((ci[am]+1)*sizeof(MatScalar),&ca);CHKERRQ(ierr);
  ierr = PetscMemzero(ca,(ci[am]+1)*sizeof(MatScalar));CHKERRQ(ierr);
  
  /* put together the new matrix */
  ierr = MatCreateSeqAIJWithArrays(A->comm,am,bn,ci,cj,ca,C);CHKERRQ(ierr);

  /* MatCreateSeqAIJWithArrays flags matrix so PETSc doesn't free the user's arrays. */
  /* These are PETSc arrays, so change flags so arrays can be deleted by PETSc */
  c = (Mat_SeqAIJ *)((*C)->data);
  c->freedata = PETSC_TRUE;
  c->nonew    = 0;

  ierr = PetscLogEventEnd(logkey_matmatmult_symbolic,A,B,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
     MatMatMult_SeqAIJ_SeqAIJ_Numeric - Forms the numeric product of two SeqAIJ matrices
           C=A*B;
     Note: C must have been created by calling MatMatMult_SeqAIJ_SeqAIJ_Symbolic.
*/
#undef __FUNCT__  
#define __FUNCT__ "MatMatMult_SeqAIJ_SeqAIJ_Numeric"
int MatMatMult_SeqAIJ_SeqAIJ_Numeric(Mat A,Mat B,Mat C)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *)A->data;
  Mat_SeqAIJ *b = (Mat_SeqAIJ *)B->data;
  Mat_SeqAIJ *c = (Mat_SeqAIJ *)C->data;
  int        aishift=a->indexshift,bishift=b->indexshift,cishift=c->indexshift;
  int        *ai=a->i,*aj=a->j,*bi=b->i,*bj=b->j,*bjj,*ci=c->i,*cj=c->j;
  int        an=A->N,am=A->M,bn=B->N,bm=B->M,cn=C->N,cm=C->M;
  int        ierr,i,j,k,anzi,bnzi,cnzi,brow,flops;
  MatScalar  *aa=a->a,*ba=b->a,*baj,*ca=c->a,*temp;

  PetscFunctionBegin;  

  /* This error checking should be unnecessary if the symbolic was performed */ 
  if (aishift || bishift || cishift) SETERRQ(PETSC_ERR_SUP,"Shifted matrix indices are not supported.");
  if (am!=cm) SETERRQ2(PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, %d != %d",am,cm);
  if (an!=bm) SETERRQ2(PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, %d != %d",an,bm);
  if (bn!=cn) SETERRQ2(PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, %d != %d",bn,cn);

  if (!logkey_matmatmult_numeric) {
    ierr = PetscLogEventRegister(&logkey_matmatmult_numeric,"MatMatMult_Numeric",MAT_COOKIE);CHKERRQ(ierr);
  }
  ierr = PetscLogEventBegin(logkey_matmatmult_numeric,A,B,C,0);CHKERRQ(ierr);
  flops = 0;
  /* Allocate temp accumulation space to avoid searching for nonzero columns in C */
  ierr = PetscMalloc((cn+1)*sizeof(MatScalar),&temp);CHKERRQ(ierr);
  ierr = PetscMemzero(temp,cn*sizeof(MatScalar));CHKERRQ(ierr);
  /* Traverse A row-wise. */
  /* Build the ith row in C by summing over nonzero columns in A, */
  /* the rows of B corresponding to nonzeros of A. */
  for (i=0;i<am;i++) {
    anzi = ai[i+1] - ai[i];
    for (j=0;j<anzi;j++) {
      brow = *aj++;
      bnzi = bi[brow+1] - bi[brow];
      bjj  = bj + bi[brow];
      baj  = ba + bi[brow];
      for (k=0;k<bnzi;k++) {
        temp[bjj[k]] += (*aa)*baj[k];
      }
      flops += 2*bnzi;
      aa++;
    }
    /* Store row back into C, and re-zero temp */
    cnzi = ci[i+1] - ci[i];
    for (j=0;j<cnzi;j++) {
      ca[j] = temp[cj[j]];
      temp[cj[j]] = 0.0;
    }
    ca += cnzi;
    cj += cnzi;
  }
  /* Free temp */
  ierr = PetscFree(temp);CHKERRQ(ierr);
  ierr = PetscLogFlops(flops);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(logkey_matmatmult_numeric,A,B,C,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMatMult_SeqAIJ_SeqAIJ"
int MatMatMult_SeqAIJ_SeqAIJ(Mat A,Mat B,Mat *C) {
  int ierr;

  PetscFunctionBegin;
  ierr = MatMatMult_SeqAIJ_SeqAIJ_Symbolic(A,B,C);CHKERRQ(ierr);
  ierr = MatMatMult_SeqAIJ_SeqAIJ_Numeric(A,B,*C);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static int logkey_matapplyptap_symbolic = 0;
static int logkey_matapplyptap_numeric  = 0;

#undef __FUNCT__
#define __FUNCT__ "MatApplyPtAP_SeqAIJ_Symbolic"
int MatApplyPtAP_SeqAIJ_Symbolic(Mat A,Mat P,Mat *C) {
  int ierr;
  FreeSpaceList  free_space=PETSC_NULL,current_space=PETSC_NULL;
  Mat_SeqAIJ     *a=(Mat_SeqAIJ*)A->data,*p=(Mat_SeqAIJ*)P->data,*c;
  int            aishift=a->indexshift,pishift=p->indexshift;
  int            *pti,*ptj,*ptfill,*ai=a->i,*aj=a->j,*ajj,*pi=p->i,*pj=p->j,*pjj;
  int            *ci,*cj,*densefill,*sparsefill,*ptadensefill,*ptasparsefill,*ptaj;
  int            an=A->N,am=A->M,pn=P->N,pm=P->M;
  int            i,j,k,ptnzi,arow,anzj,ptanzi,prow,pnzj,cnzi;
  MatScalar      *ca;

  PetscFunctionBegin;

  /* some error checking which could be moved into interface layer */
  if (aishift || pishift) SETERRQ(PETSC_ERR_SUP,"Shifted matrix indices are not supported.");
  if (pm!=an) SETERRQ2(PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, %d != %d",pm,an);
  if (am!=an) SETERRQ2(PETSC_ERR_ARG_SIZ,"Matrix 'A' must be square, %d != %d",am, an);
  
  if (!logkey_matapplyptap_symbolic) {
    ierr = PetscLogEventRegister(&logkey_matapplyptap_symbolic,"MatApplyPtAP_Symbolic",MAT_COOKIE);CHKERRQ(ierr);
  }
  ierr = PetscLogEventBegin(logkey_matapplyptap_symbolic,A,P,0,0);CHKERRQ(ierr);

  /* Create ij structure of P^T */
  /* Recall in P^T there are pn rows and pi[pm] nonzeros. */
  ierr = PetscMalloc((pn+1+pi[pm])*sizeof(int),&pti);CHKERRQ(ierr);
  ierr = PetscMemzero(pti,(pn+1+pi[pm])*sizeof(int));CHKERRQ(ierr);
  ptj = pti + pn+1;

  /* Walk through pj and count ## of non-zeros in each row of P^T. */
  for (i=0;i<pi[pm]-1;i++) {
    pti[pj[i]+1] += 1;
  }
  /* Form pti for csr format of P^T. */
  for (i=0;i<pm;i++) {
    pti[i+1] += pti[i];
  }

  /* Allocate temporary space for next insert location in each row of P^T. */
  ierr = PetscMalloc(pn*sizeof(int),&ptfill);CHKERRQ(ierr);
  ierr = PetscMemcpy(ptfill,pti,pn*sizeof(int));CHKERRQ(ierr);

  /* Walk through P row-wise and mark nonzero entries of P^T. */
  for (i=0;i<pm;i++) {
    pnzj = pi[i+1] - pi[i];
    for (j=0;j<pnzj;j++) {
      ptj[ptfill[j]] =  i;
      ptfill[j]      += 1;
    }
  }

  /* Clean-up temporary space. */
  ierr = PetscFree(ptfill);CHKERRQ(ierr);

  /* Allocate ci array, arrays for fill computation and */
  /* free space for accumulating nonzero column info */
  ierr = PetscMalloc(((pn+1)*1)*sizeof(int),&ci);CHKERRQ(ierr);
  ci[0] = 0;

  ierr = PetscMalloc((2*pn+2*an+1)*sizeof(int),&ptadensefill);CHKERRQ(ierr);
  ierr = PetscMemzero(ptadensefill,(2*pn+2*an+1)*sizeof(int));CHKERRQ(ierr);
  ptasparsefill = ptadensefill  + an;
  densefill     = ptasparsefill + an;
  sparsefill    = densefill     + pn;

  /* Set initial free space to be nnz(A) scaled by aspect ratio of P. */
  /* Reason: Take pn/pm = 1/2. */
  /*         P^T*A*P will take A(NxN) and create C(N/2xN/2). */
  /*         If C has same sparsity pattern as A, nnz(C)~1/2*nnz(A). */
  /*         Is this reasonable???? */
  ierr          = GetMoreSpace((ai[am]*pn)/pm,&free_space);
  current_space = free_space;

  /* Determine fill for each row of C: */
  for (i=0;i<pn;i++) {
    ptnzi  = pti[i+1] - pti[i];
    ptanzi = 0;
    /* Determine fill for row of PtA: */
    for (j=0;j<ptnzi;j++) {
      arow = *ptj++;
      anzj = ai[arow+1] - ai[arow];
      ajj  = aj + ai[arow];
      for (k=0;k<anzj;k++) {
        if (!ptadensefill[ajj[k]]) {
          ptadensefill[ajj[k]]    = -1;
          ptasparsefill[ptanzi++] = ajj[k];
        }
      }
    }
    /* Using fill info for row of PtA, determine fill for row of C: */
    ptaj = ptasparsefill;
    cnzi   = 0;
    for (j=0;j<ptanzi;j++) {
      prow = *ptaj++;
      pnzj = pi[prow+1] - pi[prow];
      pjj  = pj + pi[prow];
      for (k=0;k<pnzj;k++) {
        if (!densefill[pjj[k]]) {
          densefill[pjj[k]]  = -1;
          sparsefill[cnzi++] = pjj[k];
        }
      }
    }

    /* sort sparsefill */
    ierr = PetscSortInt(cnzi,sparsefill);CHKERRQ(ierr);

    /* If free space is not available, make more free space */
    /* Double the amount of total space in the list */
    if (current_space->local_remaining<cnzi) {
      ierr = GetMoreSpace(current_space->total_array_size,&current_space);CHKERRQ(ierr);
    }

    /* Copy data into free space, and zero out densefills */
    ierr = PetscMemcpy(current_space->array,sparsefill,cnzi*sizeof(int));CHKERRQ(ierr);
    current_space->array           += cnzi;
    current_space->local_used      += cnzi;
    current_space->local_remaining -= cnzi;

    for (j=0;j<ptanzi;j++) {
      ptadensefill[ptasparsefill[j]] = 0;
    }
    for (j=0;j<cnzi;j++) {
      densefill[sparsefill[j]] = 0;
    }
    /* Aside: Perhaps we should save the pta info for the numerical factorization. */
    /*        For now, we will recompute what is needed. */ 
    ci[i+1] = ci[i] + cnzi;
  }
  /* nnz is now stored in ci[ptm], column indices are in the list of free space */
  /* Allocate space for cj, initialize cj, and */
  /* destroy list of free space and other temporary array(s) */
  ierr = PetscMalloc((ci[pn]+1)*sizeof(int),&cj);CHKERRQ(ierr);
  ierr = MakeSpaceContiguous(cj,&free_space);CHKERRQ(ierr);
  ierr = PetscFree(ptadensefill);CHKERRQ(ierr);
    
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
  /* Perhaps we should attach the (i,j) info for P^T to P for future use. */
  /* For now, we won't. */
  ierr = PetscFree(pti);

  ierr = PetscLogEventEnd(logkey_matapplyptap_symbolic,A,P,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatApplyPtAP_SeqAIJ_Numeric"
int MatApplyPtAP_SeqAIJ_Numeric(Mat A,Mat P,Mat C) {
  int ierr,flops;
  Mat_SeqAIJ *a  = (Mat_SeqAIJ *) A->data;
  Mat_SeqAIJ *p  = (Mat_SeqAIJ *) P->data;
  Mat_SeqAIJ *c  = (Mat_SeqAIJ *) C->data;
  int        aishift=a->indexshift,pishift=p->indexshift,cishift=c->indexshift;
  int        *ai=a->i,*aj=a->j,*apj,*pi=p->i,*pj=p->j,*pJ=p->j,*pjj,*ci=c->i,*cj=c->j,*cjj;
  int        an=A->N,am=A->M,pn=P->N,pm=P->M,cn=C->N,cm=C->M;
  int        i,j,k,anzi,pnzi,apnzj,pnzj,cnzj,prow,crow;
  MatScalar  *aa=a->a,*apa,*pa=p->a,*pA=p->a,*paj,*ca=c->a,*caj;

  PetscFunctionBegin;

  /* This error checking should be unnecessary if the symbolic was performed */ 
  if (aishift || pishift || cishift) SETERRQ(PETSC_ERR_SUP,"Shifted matrix indices are not supported.");
  if (pn!=cm) SETERRQ2(PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, %d != %d",pn,cm);
  if (pm!=an) SETERRQ2(PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, %d != %d",pm,an);
  if (am!=an) SETERRQ2(PETSC_ERR_ARG_SIZ,"Matrix 'A' must be square, %d != %d",am, an);
  if (pn!=cn) SETERRQ2(PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, %d != %d",pn, cn);

  if (!logkey_matapplyptap_numeric) {
    ierr = PetscLogEventRegister(&logkey_matapplyptap_numeric,"MatApplyPtAP_Numeric",MAT_COOKIE);CHKERRQ(ierr);
  }
  ierr = PetscLogEventBegin(logkey_matapplyptap_numeric,A,P,C,0);CHKERRQ(ierr);
  flops = 0;

  ierr = PetscMalloc(cn*(sizeof(MatScalar)+sizeof(int)),&apa);CHKERRQ(ierr);
  ierr = PetscMemzero(apa,cn*(sizeof(MatScalar)+sizeof(int)));CHKERRQ(ierr);
  apj = (int *)(apa + cn);
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
        if (!apa[pjj[k]]) {
          apj[apnzj++]=pjj[k];
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
      int nextap=0;
      crow = *pJ++;
      cnzj = ci[crow+1] - ci[crow];
      cjj  = cj + ci[crow];
      caj  = ca + ci[crow];
      /* Perform the sparse axpy operation.  Note cjj includes apj. */
      for (k=0;k<cnzj;k++) {
        if (cjj[k]==apj[nextap]) {
          caj[k] += (*pA)*apa[apj[nextap++]];
        }
      }
      flops += 2*apnzj;
      pA++;
    }

    for (j=0;j<apnzj;j++) {
      apa[apj[j]] = 0.;
    }
  }
  ierr = PetscFree(apa);CHKERRQ(ierr);
  ierr = PetscLogFlops(flops);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(logkey_matapplyptap_numeric,A,P,C,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatApplyPtAP_SeqAIJ"
int MatApplyPtAP_SeqAIJ(Mat A,Mat P,Mat *C) {
  int ierr;

  PetscFunctionBegin;
  ierr = MatApplyPtAP_SeqAIJ_Symbolic(A,P,C);CHKERRQ(ierr);
  ierr = MatApplyPtAP_SeqAIJ_Numeric(A,P,*C);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
