/*$Id: matmatmult.c,v 1.15 2001/09/07 20:04:44 buschelm Exp $*/
/*
  Defines matrix-matrix product routines for pairs of SeqAIJ matrices
          C = A * B
          C = P^T * A * P
          C = P * A * P^T
*/

#include "src/mat/impls/aij/seq/aij.h"
#include "src/mat/utils/freespace.h"

static int logkey_matmatmult            = 0;
static int logkey_matmatmult_symbolic   = 0;
static int logkey_matmatmult_numeric    = 0;

static int logkey_matapplyptap          = 0;
static int logkey_matapplyptap_symbolic = 0;
static int logkey_matapplyptap_numeric  = 0;

static int logkey_matapplypapt          = 0;
static int logkey_matapplypapt_symbolic = 0;
static int logkey_matapplypapt_numeric  = 0;

/*
     MatMatMult_Symbolic_SeqAIJ_SeqAIJ - Forms the symbolic product of two SeqAIJ matrices
           C = A * B;

     Note: C is assumed to be uncreated.
           If this is not the case, Destroy C before calling this routine.
*/
#undef __FUNCT__  
#define __FUNCT__ "MatMatMult_Symbolic_SeqAIJ_SeqAIJ"
int MatMatMult_Symbolic_SeqAIJ_SeqAIJ(Mat A,Mat B,Mat *C)
{
  int            ierr;
  FreeSpaceList  free_space=PETSC_NULL,current_space=PETSC_NULL;
  Mat_SeqAIJ     *a=(Mat_SeqAIJ*)A->data,*b=(Mat_SeqAIJ*)B->data,*c;
  int            aishift=a->indexshift,bishift=b->indexshift;
  int            *ai=a->i,*aj=a->j,*bi=b->i,*bj=b->j,*bjj;
  int            *ci,*cj,*denserow,*sparserow;
  int            an=A->N,am=A->M,bn=B->N,bm=B->M;
  int            i,j,k,anzi,brow,bnzj,cnzi;
  MatScalar      *ca;

  PetscFunctionBegin;
  /* some error checking which could be moved into interface layer */
  if (aishift || bishift) SETERRQ(PETSC_ERR_SUP,"Shifted matrix indices are not supported.");
  if (an!=bm) SETERRQ2(PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, %d != %d",an,bm);
  
  /* Set up timers */
  if (!logkey_matmatmult_symbolic) {
    ierr = PetscLogEventRegister(&logkey_matmatmult_symbolic,"MatMatMult_Symbolic",MAT_COOKIE);CHKERRQ(ierr);
  }
  ierr = PetscLogEventBegin(logkey_matmatmult_symbolic,A,B,0,0);CHKERRQ(ierr);

  /* Set up */
  /* Allocate ci array, arrays for fill computation and */
  /* free space for accumulating nonzero column info */
  ierr = PetscMalloc(((am+1)+1)*sizeof(int),&ci);CHKERRQ(ierr);
  ci[0] = 0;

  ierr = PetscMalloc((2*bn+1)*sizeof(int),&denserow);CHKERRQ(ierr);
  ierr = PetscMemzero(denserow,(2*bn+1)*sizeof(int));CHKERRQ(ierr);
  sparserow = denserow + bn;

  /* Initial FreeSpace size is nnz(B)=bi[bm] */
  ierr          = GetMoreSpace(bi[bm],&free_space);CHKERRQ(ierr);
  current_space = free_space;

  /* Determine symbolic info for each row of the product: */
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
        if (!denserow[bjj[k]]) {
          denserow[bjj[k]]  = -1;
          sparserow[cnzi++] = bjj[k];
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

    /* Copy data into free space, and zero out denserow */
    ierr = PetscMemcpy(current_space->array,sparserow,cnzi*sizeof(int));CHKERRQ(ierr);
    current_space->array           += cnzi;
    current_space->local_used      += cnzi;
    current_space->local_remaining -= cnzi;
    for (j=0;j<cnzi;j++) {
      denserow[sparserow[j]] = 0;
    }
    ci[i+1] = ci[i] + cnzi;
  }

  /* Column indices are in the list of free space */
  /* Allocate space for cj, initialize cj, and */
  /* destroy list of free space and other temporary array(s) */
  ierr = PetscMalloc((ci[am]+1)*sizeof(int),&cj);CHKERRQ(ierr);
  ierr = MakeSpaceContiguous(&free_space,cj);CHKERRQ(ierr);
  ierr = PetscFree(denserow);CHKERRQ(ierr);
    
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
     MatMatMult_Numeric_SeqAIJ_SeqAIJ - Forms the numeric product of two SeqAIJ matrices
           C=A*B;
     Note: C must have been created by calling MatMatMult_Symbolic_SeqAIJ_SeqAIJ.
*/
#undef __FUNCT__  
#define __FUNCT__ "MatMatMult_Numeric_SeqAIJ_SeqAIJ"
int MatMatMult_Numeric_SeqAIJ_SeqAIJ(Mat A,Mat B,Mat C)
{
  int        ierr,flops=0;
  Mat_SeqAIJ *a = (Mat_SeqAIJ *)A->data;
  Mat_SeqAIJ *b = (Mat_SeqAIJ *)B->data;
  Mat_SeqAIJ *c = (Mat_SeqAIJ *)C->data;
  int        aishift=a->indexshift,bishift=b->indexshift,cishift=c->indexshift;
  int        *ai=a->i,*aj=a->j,*bi=b->i,*bj=b->j,*bjj,*ci=c->i,*cj=c->j;
  int        an=A->N,am=A->M,bn=B->N,bm=B->M,cn=C->N,cm=C->M;
  int        i,j,k,anzi,bnzi,cnzi,brow;
  MatScalar  *aa=a->a,*ba=b->a,*baj,*ca=c->a,*temp;

  PetscFunctionBegin;  

  /* This error checking should be unnecessary if the symbolic was performed */ 
  if (aishift || bishift || cishift) SETERRQ(PETSC_ERR_SUP,"Shifted matrix indices are not supported.");
  if (am!=cm) SETERRQ2(PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, %d != %d",am,cm);
  if (an!=bm) SETERRQ2(PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, %d != %d",an,bm);
  if (bn!=cn) SETERRQ2(PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, %d != %d",bn,cn);

  /* Set up timers */
  if (!logkey_matmatmult_numeric) {
    ierr = PetscLogEventRegister(&logkey_matmatmult_numeric,"MatMatMult_Numeric",MAT_COOKIE);CHKERRQ(ierr);
  }
  ierr = PetscLogEventBegin(logkey_matmatmult_numeric,A,B,C,0);CHKERRQ(ierr);

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
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
                         
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
  if (!logkey_matmatmult) {
    ierr = PetscLogEventRegister(&logkey_matmatmult,"MatMatMult",MAT_COOKIE);CHKERRQ(ierr);
  }
  ierr = PetscLogEventBegin(logkey_matmatmult,A,B,0,0);CHKERRQ(ierr);
  ierr = MatMatMult_Symbolic_SeqAIJ_SeqAIJ(A,B,C);CHKERRQ(ierr);
  ierr = MatMatMult_Numeric_SeqAIJ_SeqAIJ(A,B,*C);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(logkey_matmatmult,A,B,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*
     MatApplyPtAP_Symbolic_SeqAIJ_SeqAIJ - Forms the symbolic product of two SeqAIJ matrices
           C = P^T * A * P;

     Note: C is assumed to be uncreated.
           If this is not the case, Destroy C before calling this routine.
*/
#undef __FUNCT__
#define __FUNCT__ "MatApplyPtAP_Symbolic_SeqAIJ_SeqAIJ"
int MatApplyPtAP_Symbolic_SeqAIJ_SeqAIJ(Mat A,Mat P,Mat *C) {
  int            ierr;
  FreeSpaceList  free_space=PETSC_NULL,current_space=PETSC_NULL;
  Mat_SeqAIJ     *a=(Mat_SeqAIJ*)A->data,*p=(Mat_SeqAIJ*)P->data,*c;
  int            aishift=a->indexshift,pishift=p->indexshift;
  int            *pti,*ptj,*ptJ,*ai=a->i,*aj=a->j,*ajj,*pi=p->i,*pj=p->j,*pjj;
  int            *ci,*cj,*denserow,*sparserow,*ptadenserow,*ptasparserow,*ptaj;
  int            an=A->N,am=A->M,pn=P->N,pm=P->M;
  int            i,j,k,ptnzi,arow,anzj,ptanzi,prow,pnzj,cnzi;
  MatScalar      *ca;

  PetscFunctionBegin;

  /* some error checking which could be moved into interface layer */
  if (aishift || pishift) SETERRQ(PETSC_ERR_SUP,"Shifted matrix indices are not supported.");
  if (pm!=an) SETERRQ2(PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, %d != %d",pm,an);
  if (am!=an) SETERRQ2(PETSC_ERR_ARG_SIZ,"Matrix 'A' must be square, %d != %d",am, an);
  
  /* Set up timers */
  if (!logkey_matapplyptap_symbolic) {
    ierr = PetscLogEventRegister(&logkey_matapplyptap_symbolic,"MatApplyPtAP_Symbolic",MAT_COOKIE);CHKERRQ(ierr);
  }
  ierr = PetscLogEventBegin(logkey_matapplyptap_symbolic,A,P,0,0);CHKERRQ(ierr);

  /* Get ij structure of P^T */
  ierr = MatGetSymbolicTranspose_SeqAIJ(P,&pti,&ptj);CHKERRQ(ierr);
  ptJ=ptj;

  /* Allocate ci array, arrays for fill computation and */
  /* free space for accumulating nonzero column info */
  ierr = PetscMalloc(((pn+1)*1)*sizeof(int),&ci);CHKERRQ(ierr);
  ci[0] = 0;

  ierr = PetscMalloc((2*pn+2*an+1)*sizeof(int),&ptadenserow);CHKERRQ(ierr);
  ierr = PetscMemzero(ptadenserow,(2*pn+2*an+1)*sizeof(int));CHKERRQ(ierr);
  ptasparserow = ptadenserow  + an;
  denserow     = ptasparserow + an;
  sparserow    = denserow     + pn;

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
      for (k=0;k<pnzj;k++) {
        if (!denserow[pjj[k]]) {
          denserow[pjj[k]]  = -1;
          sparserow[cnzi++] = pjj[k];
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
    ierr = PetscMemcpy(current_space->array,sparserow,cnzi*sizeof(int));CHKERRQ(ierr);
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
    ci[i+1] = ci[i] + cnzi;
  }
  /* nnz is now stored in ci[ptm], column indices are in the list of free space */
  /* Allocate space for cj, initialize cj, and */
  /* destroy list of free space and other temporary array(s) */
  ierr = PetscMalloc((ci[pn]+1)*sizeof(int),&cj);CHKERRQ(ierr);
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

  ierr = PetscLogEventEnd(logkey_matapplyptap_symbolic,A,P,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
     MatApplyPtAP_Numeric_SeqAIJ_SeqAIJ - Forms the numeric product of two SeqAIJ matrices
           C = P^T * A * P;
     Note: C must have been created by calling MatApplyPtAP_Symbolic_SeqAIJ.
*/
#undef __FUNCT__
#define __FUNCT__ "MatApplyPtAP_Numeric_SeqAIJ_SeqAIJ"
int MatApplyPtAP_Numeric_SeqAIJ_SeqAIJ(Mat A,Mat P,Mat C) {
  int        ierr,flops=0;
  Mat_SeqAIJ *a  = (Mat_SeqAIJ *) A->data;
  Mat_SeqAIJ *p  = (Mat_SeqAIJ *) P->data;
  Mat_SeqAIJ *c  = (Mat_SeqAIJ *) C->data;
  int        aishift=a->indexshift,pishift=p->indexshift,cishift=c->indexshift;
  int        *ai=a->i,*aj=a->j,*apj,*apjdense,*pi=p->i,*pj=p->j,*pJ=p->j,*pjj;
  int        *ci=c->i,*cj=c->j,*cjj;
  int        an=A->N,am=A->M,pn=P->N,pm=P->M,cn=C->N,cm=C->M;
  int        i,j,k,anzi,pnzi,apnzj,nextap,pnzj,cnzj,prow,crow;
  MatScalar  *aa=a->a,*apa,*pa=p->a,*pA=p->a,*paj,*ca=c->a,*caj;

  PetscFunctionBegin;

  /* This error checking should be unnecessary if the symbolic was performed */ 
  if (aishift || pishift || cishift) SETERRQ(PETSC_ERR_SUP,"Shifted matrix indices are not supported.");
  if (pn!=cm) SETERRQ2(PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, %d != %d",pn,cm);
  if (pm!=an) SETERRQ2(PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, %d != %d",pm,an);
  if (am!=an) SETERRQ2(PETSC_ERR_ARG_SIZ,"Matrix 'A' must be square, %d != %d",am, an);
  if (pn!=cn) SETERRQ2(PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, %d != %d",pn, cn);

  /* Set up timers */
  if (!logkey_matapplyptap_numeric) {
    ierr = PetscLogEventRegister(&logkey_matapplyptap_numeric,"MatApplyPtAP_Numeric",MAT_COOKIE);CHKERRQ(ierr);
  }
  ierr = PetscLogEventBegin(logkey_matapplyptap_numeric,A,P,C,0);CHKERRQ(ierr);

  ierr = PetscMalloc(cn*(sizeof(MatScalar)+2*sizeof(int)),&apa);CHKERRQ(ierr);
  ierr = PetscMemzero(apa,cn*(sizeof(MatScalar)+2*sizeof(int)));CHKERRQ(ierr);
  ierr = PetscMemzero(ca,ci[cm]*sizeof(MatScalar));CHKERRQ(ierr);

  apj      = (int *)(apa + cn);
  apjdense = apj + cn;

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
      cnzj   = ci[crow+1] - ci[crow];
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
  ierr = PetscLogEventEnd(logkey_matapplyptap_numeric,A,P,C,0);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
  

#undef __FUNCT__
#define __FUNCT__ "MatApplyPtAP_SeqAIJ_SeqAIJ"
int MatApplyPtAP_SeqAIJ_SeqAIJ(Mat A,Mat P,Mat *C) {
  int ierr;

  PetscFunctionBegin;
  if (!logkey_matapplyptap) {
    ierr = PetscLogEventRegister(&logkey_matapplyptap,"MatApplyPtAP",MAT_COOKIE);CHKERRQ(ierr);
  }
  ierr = PetscLogEventBegin(logkey_matapplyptap,A,P,0,0);CHKERRQ(ierr);
 
  ierr = MatApplyPtAP_Symbolic_SeqAIJ_SeqAIJ(A,P,C);CHKERRQ(ierr);
  ierr = MatApplyPtAP_Numeric_SeqAIJ_SeqAIJ(A,P,*C);CHKERRQ(ierr);
  
  ierr = PetscLogEventEnd(logkey_matapplyptap,A,P,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
     MatApplyPAPt_Symbolic_SeqAIJ_SeqAIJ - Forms the symbolic product of two SeqAIJ matrices
           C = P * A * P^T;

     Note: C is assumed to be uncreated.
           If this is not the case, Destroy C before calling this routine.
*/
#undef __FUNCT__
#define __FUNCT__ "MatApplyPAPt_Symbolic_SeqAIJ_SeqAIJ"
int MatApplyPAPt_Symbolic_SeqAIJ_SeqAIJ(Mat A,Mat P,Mat *C) {
  /* Note: This code is virtually identical to that of MatApplyPtAP_SeqAIJ_Symbolic */
  /*        and MatMatMult_SeqAIJ_SeqAIJ_Symbolic.  Perhaps they could be merged nicely. */
  int            ierr;
  FreeSpaceList  free_space=PETSC_NULL,current_space=PETSC_NULL;
  Mat_SeqAIJ     *a=(Mat_SeqAIJ*)A->data,*p=(Mat_SeqAIJ*)P->data,*c;
  int            aishift=a->indexshift,pishift=p->indexshift;
  int            *ai=a->i,*aj=a->j,*ajj,*pi=p->i,*pj=p->j,*pti,*ptj,*ptjj;
  int            *ci,*cj,*paj,*padenserow,*pasparserow,*denserow,*sparserow;
  int            an=A->N,am=A->M,pn=P->N,pm=P->M;
  int            i,j,k,pnzi,arow,anzj,panzi,ptrow,ptnzj,cnzi;
  MatScalar      *ca;

  PetscFunctionBegin;

  /* some error checking which could be moved into interface layer */
  if (aishift || pishift) SETERRQ(PETSC_ERR_SUP,"Shifted matrix indices are not supported.");
  if (pn!=am) SETERRQ2(PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, %d != %d",pn,am);
  if (am!=an) SETERRQ2(PETSC_ERR_ARG_SIZ,"Matrix 'A' must be square, %d != %d",am, an);

  /* Set up timers */
  if (!logkey_matapplypapt_symbolic) {
    ierr = PetscLogEventRegister(&logkey_matapplypapt_symbolic,"MatApplyPAPt_Symbolic",MAT_COOKIE);CHKERRQ(ierr);
  }
  ierr = PetscLogEventBegin(logkey_matapplypapt_symbolic,A,P,0,0);CHKERRQ(ierr);

  /* Create ij structure of P^T */
  ierr = MatGetSymbolicTranspose_SeqAIJ(P,&pti,&ptj);CHKERRQ(ierr);

  /* Allocate ci array, arrays for fill computation and */
  /* free space for accumulating nonzero column info */
  ierr = PetscMalloc(((pm+1)*1)*sizeof(int),&ci);CHKERRQ(ierr);
  ci[0] = 0;

  ierr = PetscMalloc((2*an+2*pm+1)*sizeof(int),&padenserow);CHKERRQ(ierr);
  ierr = PetscMemzero(padenserow,(2*an+2*pm+1)*sizeof(int));CHKERRQ(ierr);
  pasparserow  = padenserow  + an;
  denserow     = pasparserow + an;
  sparserow    = denserow    + pm;

  /* Set initial free space to be nnz(A) scaled by aspect ratio of Pt. */
  /* This should be reasonable if sparsity of PAPt is similar to that of A. */
  ierr          = GetMoreSpace((ai[am]/pn)*pm,&free_space);
  current_space = free_space;

  /* Determine fill for each row of C: */
  for (i=0;i<pm;i++) {
    pnzi  = pi[i+1] - pi[i];
    panzi = 0;
    /* Get symbolic sparse row of PA: */
    for (j=0;j<pnzi;j++) {
      arow = *pj++;
      anzj = ai[arow+1] - ai[arow];
      ajj  = aj + ai[arow];
      for (k=0;k<anzj;k++) {
        if (!padenserow[ajj[k]]) {
          padenserow[ajj[k]]   = -1;
          pasparserow[panzi++] = ajj[k];
        }
      }
    }
    /* Using symbolic row of PA, determine symbolic row of C: */
    paj    = pasparserow;
    cnzi   = 0;
    for (j=0;j<panzi;j++) {
      ptrow = *paj++;
      ptnzj = pti[ptrow+1] - pti[ptrow];
      ptjj  = ptj + pti[ptrow];
      for (k=0;k<ptnzj;k++) {
        if (!denserow[ptjj[k]]) {
          denserow[ptjj[k]] = -1;
          sparserow[cnzi++] = ptjj[k];
        }
      }
    }

    /* sort sparse representation */
    ierr = PetscSortInt(cnzi,sparserow);CHKERRQ(ierr);

    /* If free space is not available, make more free space */
    /* Double the amount of total space in the list */
    if (current_space->local_remaining<cnzi) {
      ierr = GetMoreSpace(current_space->total_array_size,&current_space);CHKERRQ(ierr);
    }

    /* Copy data into free space, and zero out dense row */
    ierr = PetscMemcpy(current_space->array,sparserow,cnzi*sizeof(int));CHKERRQ(ierr);
    current_space->array           += cnzi;
    current_space->local_used      += cnzi;
    current_space->local_remaining -= cnzi;

    for (j=0;j<panzi;j++) {
      padenserow[pasparserow[j]] = 0;
    }
    for (j=0;j<cnzi;j++) {
      denserow[sparserow[j]] = 0;
    }
    ci[i+1] = ci[i] + cnzi;
  }
  /* column indices are in the list of free space */
  /* Allocate space for cj, initialize cj, and */
  /* destroy list of free space and other temporary array(s) */
  ierr = PetscMalloc((ci[pm]+1)*sizeof(int),&cj);CHKERRQ(ierr);
  ierr = MakeSpaceContiguous(&free_space,cj);CHKERRQ(ierr);
  ierr = PetscFree(padenserow);CHKERRQ(ierr);
    
  /* Allocate space for ca */
  ierr = PetscMalloc((ci[pm]+1)*sizeof(MatScalar),&ca);CHKERRQ(ierr);
  ierr = PetscMemzero(ca,(ci[pm]+1)*sizeof(MatScalar));CHKERRQ(ierr);
  
  /* put together the new matrix */
  ierr = MatCreateSeqAIJWithArrays(A->comm,pm,pm,ci,cj,ca,C);CHKERRQ(ierr);

  /* MatCreateSeqAIJWithArrays flags matrix so PETSc doesn't free the user's arrays. */
  /* Since these are PETSc arrays, change flags to free them as necessary. */
  c = (Mat_SeqAIJ *)((*C)->data);
  c->freedata = PETSC_TRUE;
  c->nonew    = 0;

  /* Clean up. */
  ierr = MatRestoreSymbolicTranspose_SeqAIJ(P,&pti,&ptj);CHKERRQ(ierr);

  ierr = PetscLogEventEnd(logkey_matapplypapt_symbolic,A,P,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
     MatApplyPAPt_Numeric_SeqAIJ - Forms the numeric product of two SeqAIJ matrices
           C = P * A * P^T;
     Note: C must have been created by calling MatApplyPAPt_Symbolic_SeqAIJ.
*/
#undef __FUNCT__
#define __FUNCT__ "MatApplyPAPt_Numeric_SeqAIJ_SeqAIJ"
int MatApplyPAPt_Numeric_SeqAIJ_SeqAIJ(Mat A,Mat P,Mat C) {
  int        ierr,flops=0;
  Mat_SeqAIJ *a  = (Mat_SeqAIJ *) A->data;
  Mat_SeqAIJ *p  = (Mat_SeqAIJ *) P->data;
  Mat_SeqAIJ *c  = (Mat_SeqAIJ *) C->data;
  int        aishift=a->indexshift,pishift=p->indexshift,cishift=c->indexshift;
  int        *ai=a->i,*aj=a->j,*ajj,*pi=p->i,*pj=p->j,*pjj=p->j,*paj,*pajdense,*ptj;
  int        *ci=c->i,*cj=c->j;
  int        an=A->N,am=A->M,pn=P->N,pm=P->M,cn=C->N,cm=C->M;
  int        i,j,k,k1,k2,pnzi,anzj,panzj,arow,ptcol,ptnzj,cnzi;
  MatScalar  *aa=a->a,*pa=p->a,*pta=p->a,*ptaj,*paa,*aaj,*ca=c->a,sum;

  PetscFunctionBegin;

  /* This error checking should be unnecessary if the symbolic was performed */ 
  if (aishift || pishift || cishift) SETERRQ(PETSC_ERR_SUP,"Shifted matrix indices are not supported.");
  if (pm!=cm) SETERRQ2(PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, %d != %d",pm,cm);
  if (pn!=am) SETERRQ2(PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, %d != %d",pn,am);
  if (am!=an) SETERRQ2(PETSC_ERR_ARG_SIZ,"Matrix 'A' must be square, %d != %d",am, an);
  if (pm!=cn) SETERRQ2(PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, %d != %d",pm, cn);

  /* Set up timers */
  if (!logkey_matapplypapt_numeric) {
    ierr = PetscLogEventRegister(&logkey_matapplypapt_numeric,"MatApplyPAPt_Numeric",MAT_COOKIE);CHKERRQ(ierr);
  }
  ierr = PetscLogEventBegin(logkey_matapplypapt_numeric,A,P,C,0);CHKERRQ(ierr);

  ierr = PetscMalloc(an*(sizeof(MatScalar)+2*sizeof(int)),&paa);CHKERRQ(ierr);
  ierr = PetscMemzero(paa,an*(sizeof(MatScalar)+2*sizeof(int)));CHKERRQ(ierr);
  ierr = PetscMemzero(ca,ci[cm]*sizeof(MatScalar));CHKERRQ(ierr);

  paj      = (int *)(paa + an);
  pajdense = paj + an;

  for (i=0;i<pm;i++) {
    /* Form sparse row of P*A */
    pnzi  = pi[i+1] - pi[i];
    panzj = 0;
    for (j=0;j<pnzi;j++) {
      arow = *pj++;
      anzj = ai[arow+1] - ai[arow];
      ajj  = aj + ai[arow];
      aaj  = aa + ai[arow];
      for (k=0;k<anzj;k++) {
        if (!pajdense[ajj[k]]) {
          pajdense[ajj[k]] = -1;
          paj[panzj++]     = ajj[k];
        }
        paa[ajj[k]] += (*pa)*aaj[k];
      }
      flops += 2*anzj;
      pa++;
    }

    /* Sort the j index array for quick sparse axpy. */
    ierr = PetscSortInt(panzj,paj);CHKERRQ(ierr);

    /* Compute P*A*P^T using sparse inner products. */
    /* Take advantage of pre-computed (i,j) of C for locations of non-zeros. */
    cnzi = ci[i+1] - ci[i];
    for (j=0;j<cnzi;j++) {
      /* Form sparse inner product of current row of P*A with (*cj++) col of P^T. */
      ptcol = *cj++;
      ptnzj = pi[ptcol+1] - pi[ptcol];
      ptj   = pjj + pi[ptcol];
      ptaj  = pta + pi[ptcol];
      sum   = 0.;
      k1    = 0;
      k2    = 0;
      while ((k1<panzj) && (k2<ptnzj)) {
        if (paj[k1]==ptj[k2]) {
          sum += paa[paj[k1++]]*pta[k2++];
        } else if (paj[k1] < ptj[k2]) {
          k1++;
        } else /* if (paj[k1] > ptj[k2]) */ {
          k2++;
        }
      }
      *ca++ = sum;
    }

    /* Zero the current row info for P*A */
    for (j=0;j<panzj;j++) {
      paa[paj[j]]      = 0.;
      pajdense[paj[j]] = 0;
    }
  }

  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscLogFlops(flops);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(logkey_matapplypapt_numeric,A,P,C,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
  
#undef __FUNCT__
#define __FUNCT__ "MatApplyPAPt_SeqAIJ_SeqAIJ"
int MatApplyPAPt_SeqAIJ_SeqAIJ(Mat A,Mat P,Mat *C) {
  int ierr;

  PetscFunctionBegin;
  if (!logkey_matapplypapt) {
    ierr = PetscLogEventRegister(&logkey_matapplypapt,"MatApplyPAPt",MAT_COOKIE);CHKERRQ(ierr);
  }
  ierr = PetscLogEventBegin(logkey_matapplypapt,A,P,0,0);CHKERRQ(ierr);
  ierr = MatApplyPAPt_Symbolic_SeqAIJ_SeqAIJ(A,P,C);CHKERRQ(ierr);
  ierr = MatApplyPAPt_Numeric_SeqAIJ_SeqAIJ(A,P,*C);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(logkey_matapplypapt,A,P,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
