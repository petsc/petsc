
/*
  Defines matrix-matrix product routines for pairs of SeqAIJ matrices
          C = A * B
*/

#include <../src/mat/impls/aij/seq/aij.h> /*I "petscmat.h" I*/
#include <../src/mat/utils/freespace.h>
#include <petscbt.h>
#include <petsc/private/isimpl.h>
#include <../src/mat/impls/dense/seq/dense.h>


PETSC_INTERN PetscErrorCode MatMatMult_SeqAIJ_SeqAIJ(Mat A,Mat B,MatReuse scall,PetscReal fill,Mat *C)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (scall == MAT_INITIAL_MATRIX) {
    ierr = PetscLogEventBegin(MAT_MatMultSymbolic,A,B,0,0);CHKERRQ(ierr);
    ierr = MatMatMultSymbolic_SeqAIJ_SeqAIJ(A,B,fill,C);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(MAT_MatMultSymbolic,A,B,0,0);CHKERRQ(ierr);
  }
  ierr = PetscLogEventBegin(MAT_MatMultNumeric,A,B,0,0);CHKERRQ(ierr);
  ierr = MatMatMultNumeric_SeqAIJ_SeqAIJ(A,B,*C);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_MatMultNumeric,A,B,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatMultNumeric_SeqAIJ_SeqAIJ(Mat A,Mat B,Mat C)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (C->ops->matmultnumeric) {
    ierr = (*C->ops->matmultnumeric)(A,B,C);CHKERRQ(ierr);
  } else {
    ierr = MatMatMultNumeric_SeqAIJ_SeqAIJ_Sorted(A,B,C);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatMultSymbolic_SeqAIJ_SeqAIJ(Mat A,Mat B,PetscReal fill,Mat *C)
{
  PetscErrorCode ierr;
#if !defined(PETSC_HAVE_HYPRE)
  const char     *algTypes[8] = {"sorted","scalable","scalable_fast","heap","btheap","llcondensed","combined","rowmerge"};
  PetscInt       nalg = 8;
#else
  const char     *algTypes[9] = {"sorted","scalable","scalable_fast","heap","btheap","llcondensed","combined","rowmerge","hypre"};
  PetscInt       nalg = 9;
#endif
  PetscInt       alg = 0; /* set default algorithm */

  PetscFunctionBegin;
  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)A),((PetscObject)A)->prefix,"MatMatMult","Mat");CHKERRQ(ierr);
  ierr = PetscOptionsEList("-matmatmult_via","Algorithmic approach","MatMatMult",algTypes,nalg,algTypes[0],&alg,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  switch (alg) {
  case 1:
    ierr = MatMatMultSymbolic_SeqAIJ_SeqAIJ_Scalable(A,B,fill,C);CHKERRQ(ierr);
    break;
  case 2:
    ierr = MatMatMultSymbolic_SeqAIJ_SeqAIJ_Scalable_fast(A,B,fill,C);CHKERRQ(ierr);
    break;
  case 3:
    ierr = MatMatMultSymbolic_SeqAIJ_SeqAIJ_Heap(A,B,fill,C);CHKERRQ(ierr);
    break;
  case 4:
    ierr = MatMatMultSymbolic_SeqAIJ_SeqAIJ_BTHeap(A,B,fill,C);CHKERRQ(ierr);
    break;
  case 5:
    ierr = MatMatMultSymbolic_SeqAIJ_SeqAIJ_LLCondensed(A,B,fill,C);CHKERRQ(ierr);
    break;
  case 6:
    ierr = MatMatMult_SeqAIJ_SeqAIJ_Combined(A,B,fill,C);CHKERRQ(ierr);
    break;
  case 7:
    ierr = MatMatMultSymbolic_SeqAIJ_SeqAIJ_RowMerge(A,B,fill,C);CHKERRQ(ierr);
    break;
#if defined(PETSC_HAVE_HYPRE)
  case 8:
    ierr = MatMatMultSymbolic_AIJ_AIJ_wHYPRE(A,B,fill,C);CHKERRQ(ierr);
    break;
#endif
  default:
    ierr = MatMatMultSymbolic_SeqAIJ_SeqAIJ_Sorted(A,B,fill,C);CHKERRQ(ierr);
    break;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatMultSymbolic_SeqAIJ_SeqAIJ_LLCondensed(Mat A,Mat B,PetscReal fill,Mat *C)
{
  PetscErrorCode     ierr;
  Mat_SeqAIJ         *a =(Mat_SeqAIJ*)A->data,*b=(Mat_SeqAIJ*)B->data,*c;
  PetscInt           *ai=a->i,*bi=b->i,*ci,*cj;
  PetscInt           am =A->rmap->N,bn=B->cmap->N,bm=B->rmap->N;
  PetscReal          afill;
  PetscInt           i,j,anzi,brow,bnzj,cnzi,*bj,*aj,*lnk,ndouble=0,Crmax;
  PetscTable         ta;
  PetscBT            lnkbt;
  PetscFreeSpaceList free_space=NULL,current_space=NULL;

  PetscFunctionBegin;
  /* Get ci and cj */
  /*---------------*/
  /* Allocate ci array, arrays for fill computation and */
  /* free space for accumulating nonzero column info */
  ierr  = PetscMalloc1(am+2,&ci);CHKERRQ(ierr);
  ci[0] = 0;

  /* create and initialize a linked list */
  ierr = PetscTableCreate(bn,bn,&ta);CHKERRQ(ierr);
  MatRowMergeMax_SeqAIJ(b,bm,ta);
  ierr = PetscTableGetCount(ta,&Crmax);CHKERRQ(ierr);
  ierr = PetscTableDestroy(&ta);CHKERRQ(ierr);

  ierr = PetscLLCondensedCreate(Crmax,bn,&lnk,&lnkbt);CHKERRQ(ierr);

  /* Initial FreeSpace size is fill*(nnz(A)+nnz(B)) */
  ierr = PetscFreeSpaceGet(PetscRealIntMultTruncate(fill,PetscIntSumTruncate(ai[am],bi[bm])),&free_space);CHKERRQ(ierr);

  current_space = free_space;

  /* Determine ci and cj */
  for (i=0; i<am; i++) {
    anzi = ai[i+1] - ai[i];
    aj   = a->j + ai[i];
    for (j=0; j<anzi; j++) {
      brow = aj[j];
      bnzj = bi[brow+1] - bi[brow];
      bj   = b->j + bi[brow];
      /* add non-zero cols of B into the sorted linked list lnk */
      ierr = PetscLLCondensedAddSorted(bnzj,bj,lnk,lnkbt);CHKERRQ(ierr);
    }
    cnzi = lnk[0];

    /* If free space is not available, make more free space */
    /* Double the amount of total space in the list */
    if (current_space->local_remaining<cnzi) {
      ierr = PetscFreeSpaceGet(PetscIntSumTruncate(cnzi,current_space->total_array_size),&current_space);CHKERRQ(ierr);
      ndouble++;
    }

    /* Copy data into free space, then initialize lnk */
    ierr = PetscLLCondensedClean(bn,cnzi,current_space->array,lnk,lnkbt);CHKERRQ(ierr);

    current_space->array           += cnzi;
    current_space->local_used      += cnzi;
    current_space->local_remaining -= cnzi;

    ci[i+1] = ci[i] + cnzi;
  }

  /* Column indices are in the list of free space */
  /* Allocate space for cj, initialize cj, and */
  /* destroy list of free space and other temporary array(s) */
  ierr = PetscMalloc1(ci[am]+1,&cj);CHKERRQ(ierr);
  ierr = PetscFreeSpaceContiguous(&free_space,cj);CHKERRQ(ierr);
  ierr = PetscLLCondensedDestroy(lnk,lnkbt);CHKERRQ(ierr);

  /* put together the new symbolic matrix */
  ierr = MatCreateSeqAIJWithArrays(PetscObjectComm((PetscObject)A),am,bn,ci,cj,NULL,C);CHKERRQ(ierr);
  ierr = MatSetBlockSizesFromMats(*C,A,B);CHKERRQ(ierr);
  ierr = MatSetType(*C,((PetscObject)A)->type_name);CHKERRQ(ierr);

  /* MatCreateSeqAIJWithArrays flags matrix so PETSc doesn't free the user's arrays. */
  /* These are PETSc arrays, so change flags so arrays can be deleted by PETSc */
  c                         = (Mat_SeqAIJ*)((*C)->data);
  c->free_a                 = PETSC_FALSE;
  c->free_ij                = PETSC_TRUE;
  c->nonew                  = 0;
  (*C)->ops->matmultnumeric = MatMatMultNumeric_SeqAIJ_SeqAIJ_Sorted; /* fast, needs non-scalable O(bn) array 'abdense' */

  /* set MatInfo */
  afill = (PetscReal)ci[am]/(ai[am]+bi[bm]) + 1.e-5;
  if (afill < 1.0) afill = 1.0;
  c->maxnz                     = ci[am];
  c->nz                        = ci[am];
  (*C)->info.mallocs           = ndouble;
  (*C)->info.fill_ratio_given  = fill;
  (*C)->info.fill_ratio_needed = afill;

#if defined(PETSC_USE_INFO)
  if (ci[am]) {
    ierr = PetscInfo3((*C),"Reallocs %D; Fill ratio: given %g needed %g.\n",ndouble,(double)fill,(double)afill);CHKERRQ(ierr);
    ierr = PetscInfo1((*C),"Use MatMatMult(A,B,MatReuse,%g,&C) for best performance.;\n",(double)afill);CHKERRQ(ierr);
  } else {
    ierr = PetscInfo((*C),"Empty matrix product\n");CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatMultNumeric_SeqAIJ_SeqAIJ_Sorted(Mat A,Mat B,Mat C)
{
  PetscErrorCode ierr;
  PetscLogDouble flops=0.0;
  Mat_SeqAIJ     *a   = (Mat_SeqAIJ*)A->data;
  Mat_SeqAIJ     *b   = (Mat_SeqAIJ*)B->data;
  Mat_SeqAIJ     *c   = (Mat_SeqAIJ*)C->data;
  PetscInt       *ai  =a->i,*aj=a->j,*bi=b->i,*bj=b->j,*bjj,*ci=c->i,*cj=c->j;
  PetscInt       am   =A->rmap->n,cm=C->rmap->n;
  PetscInt       i,j,k,anzi,bnzi,cnzi,brow;
  PetscScalar    *aa=a->a,*ba=b->a,*baj,*ca,valtmp;
  PetscScalar    *ab_dense;

  PetscFunctionBegin;
  if (!c->a) { /* first call of MatMatMultNumeric_SeqAIJ_SeqAIJ, allocate ca and matmult_abdense */
    ierr      = PetscMalloc1(ci[cm]+1,&ca);CHKERRQ(ierr);
    c->a      = ca;
    c->free_a = PETSC_TRUE;
  } else {
    ca        = c->a;
  }
  if (!c->matmult_abdense) {
    ierr = PetscCalloc1(B->cmap->N,&ab_dense);CHKERRQ(ierr);
    c->matmult_abdense = ab_dense;
  } else {
    ab_dense = c->matmult_abdense;
  }

  /* clean old values in C */
  ierr = PetscArrayzero(ca,ci[cm]);CHKERRQ(ierr);
  /* Traverse A row-wise. */
  /* Build the ith row in C by summing over nonzero columns in A, */
  /* the rows of B corresponding to nonzeros of A. */
  for (i=0; i<am; i++) {
    anzi = ai[i+1] - ai[i];
    for (j=0; j<anzi; j++) {
      brow = aj[j];
      bnzi = bi[brow+1] - bi[brow];
      bjj  = bj + bi[brow];
      baj  = ba + bi[brow];
      /* perform dense axpy */
      valtmp = aa[j];
      for (k=0; k<bnzi; k++) {
        ab_dense[bjj[k]] += valtmp*baj[k];
      }
      flops += 2*bnzi;
    }
    aj += anzi; aa += anzi;

    cnzi = ci[i+1] - ci[i];
    for (k=0; k<cnzi; k++) {
      ca[k]          += ab_dense[cj[k]];
      ab_dense[cj[k]] = 0.0; /* zero ab_dense */
    }
    flops += cnzi;
    cj    += cnzi; ca += cnzi;
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscLogFlops(flops);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatMultNumeric_SeqAIJ_SeqAIJ_Scalable(Mat A,Mat B,Mat C)
{
  PetscErrorCode ierr;
  PetscLogDouble flops=0.0;
  Mat_SeqAIJ     *a   = (Mat_SeqAIJ*)A->data;
  Mat_SeqAIJ     *b   = (Mat_SeqAIJ*)B->data;
  Mat_SeqAIJ     *c   = (Mat_SeqAIJ*)C->data;
  PetscInt       *ai  = a->i,*aj=a->j,*bi=b->i,*bj=b->j,*bjj,*ci=c->i,*cj=c->j;
  PetscInt       am   = A->rmap->N,cm=C->rmap->N;
  PetscInt       i,j,k,anzi,bnzi,cnzi,brow;
  PetscScalar    *aa=a->a,*ba=b->a,*baj,*ca=c->a,valtmp;
  PetscInt       nextb;

  PetscFunctionBegin;
  if (!ca) { /* first call of MatMatMultNumeric_SeqAIJ_SeqAIJ, allocate ca and matmult_abdense */
    ierr      = PetscMalloc1(ci[cm]+1,&ca);CHKERRQ(ierr);
    c->a      = ca;
    c->free_a = PETSC_TRUE;
  }

  /* clean old values in C */
  ierr = PetscArrayzero(ca,ci[cm]);CHKERRQ(ierr);
  /* Traverse A row-wise. */
  /* Build the ith row in C by summing over nonzero columns in A, */
  /* the rows of B corresponding to nonzeros of A. */
  for (i=0; i<am; i++) {
    anzi = ai[i+1] - ai[i];
    cnzi = ci[i+1] - ci[i];
    for (j=0; j<anzi; j++) {
      brow = aj[j];
      bnzi = bi[brow+1] - bi[brow];
      bjj  = bj + bi[brow];
      baj  = ba + bi[brow];
      /* perform sparse axpy */
      valtmp = aa[j];
      nextb  = 0;
      for (k=0; nextb<bnzi; k++) {
        if (cj[k] == bjj[nextb]) { /* ccol == bcol */
          ca[k] += valtmp*baj[nextb++];
        }
      }
      flops += 2*bnzi;
    }
    aj += anzi; aa += anzi;
    cj += cnzi; ca += cnzi;
  }

  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscLogFlops(flops);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatMultSymbolic_SeqAIJ_SeqAIJ_Scalable_fast(Mat A,Mat B,PetscReal fill,Mat *C)
{
  PetscErrorCode     ierr;
  Mat_SeqAIJ         *a  = (Mat_SeqAIJ*)A->data,*b=(Mat_SeqAIJ*)B->data,*c;
  PetscInt           *ai = a->i,*bi=b->i,*ci,*cj;
  PetscInt           am  = A->rmap->N,bn=B->cmap->N,bm=B->rmap->N;
  MatScalar          *ca;
  PetscReal          afill;
  PetscInt           i,j,anzi,brow,bnzj,cnzi,*bj,*aj,*lnk,ndouble=0,Crmax;
  PetscTable         ta;
  PetscFreeSpaceList free_space=NULL,current_space=NULL;

  PetscFunctionBegin;
  /* Get ci and cj - same as MatMatMultSymbolic_SeqAIJ_SeqAIJ except using PetscLLxxx_fast() */
  /*-----------------------------------------------------------------------------------------*/
  /* Allocate arrays for fill computation and free space for accumulating nonzero column */
  ierr  = PetscMalloc1(am+2,&ci);CHKERRQ(ierr);
  ci[0] = 0;

  /* create and initialize a linked list */
  ierr = PetscTableCreate(bn,bn,&ta);CHKERRQ(ierr);
  MatRowMergeMax_SeqAIJ(b,bm,ta);
  ierr = PetscTableGetCount(ta,&Crmax);CHKERRQ(ierr);
  ierr = PetscTableDestroy(&ta);CHKERRQ(ierr);

  ierr = PetscLLCondensedCreate_fast(Crmax,&lnk);CHKERRQ(ierr);

  /* Initial FreeSpace size is fill*(nnz(A)+nnz(B)) */
  ierr          = PetscFreeSpaceGet(PetscRealIntMultTruncate(fill,PetscIntSumTruncate(ai[am],bi[bm])),&free_space);CHKERRQ(ierr);
  current_space = free_space;

  /* Determine ci and cj */
  for (i=0; i<am; i++) {
    anzi = ai[i+1] - ai[i];
    aj   = a->j + ai[i];
    for (j=0; j<anzi; j++) {
      brow = aj[j];
      bnzj = bi[brow+1] - bi[brow];
      bj   = b->j + bi[brow];
      /* add non-zero cols of B into the sorted linked list lnk */
      ierr = PetscLLCondensedAddSorted_fast(bnzj,bj,lnk);CHKERRQ(ierr);
    }
    cnzi = lnk[1];

    /* If free space is not available, make more free space */
    /* Double the amount of total space in the list */
    if (current_space->local_remaining<cnzi) {
      ierr = PetscFreeSpaceGet(PetscIntSumTruncate(cnzi,current_space->total_array_size),&current_space);CHKERRQ(ierr);
      ndouble++;
    }

    /* Copy data into free space, then initialize lnk */
    ierr = PetscLLCondensedClean_fast(cnzi,current_space->array,lnk);CHKERRQ(ierr);

    current_space->array           += cnzi;
    current_space->local_used      += cnzi;
    current_space->local_remaining -= cnzi;

    ci[i+1] = ci[i] + cnzi;
  }

  /* Column indices are in the list of free space */
  /* Allocate space for cj, initialize cj, and */
  /* destroy list of free space and other temporary array(s) */
  ierr = PetscMalloc1(ci[am]+1,&cj);CHKERRQ(ierr);
  ierr = PetscFreeSpaceContiguous(&free_space,cj);CHKERRQ(ierr);
  ierr = PetscLLCondensedDestroy_fast(lnk);CHKERRQ(ierr);

  /* Allocate space for ca */
  ierr = PetscCalloc1(ci[am]+1,&ca);CHKERRQ(ierr);

  /* put together the new symbolic matrix */
  ierr = MatCreateSeqAIJWithArrays(PetscObjectComm((PetscObject)A),am,bn,ci,cj,ca,C);CHKERRQ(ierr);
  ierr = MatSetBlockSizesFromMats(*C,A,B);CHKERRQ(ierr);
  ierr = MatSetType(*C,((PetscObject)A)->type_name);CHKERRQ(ierr);

  /* MatCreateSeqAIJWithArrays flags matrix so PETSc doesn't free the user's arrays. */
  /* These are PETSc arrays, so change flags so arrays can be deleted by PETSc */
  c          = (Mat_SeqAIJ*)((*C)->data);
  c->free_a  = PETSC_TRUE;
  c->free_ij = PETSC_TRUE;
  c->nonew   = 0;

  (*C)->ops->matmultnumeric = MatMatMultNumeric_SeqAIJ_SeqAIJ_Scalable; /* slower, less memory */

  /* set MatInfo */
  afill = (PetscReal)ci[am]/(ai[am]+bi[bm]) + 1.e-5;
  if (afill < 1.0) afill = 1.0;
  c->maxnz                     = ci[am];
  c->nz                        = ci[am];
  (*C)->info.mallocs           = ndouble;
  (*C)->info.fill_ratio_given  = fill;
  (*C)->info.fill_ratio_needed = afill;

#if defined(PETSC_USE_INFO)
  if (ci[am]) {
    ierr = PetscInfo3((*C),"Reallocs %D; Fill ratio: given %g needed %g.\n",ndouble,(double)fill,(double)afill);CHKERRQ(ierr);
    ierr = PetscInfo1((*C),"Use MatMatMult(A,B,MatReuse,%g,&C) for best performance.;\n",(double)afill);CHKERRQ(ierr);
  } else {
    ierr = PetscInfo((*C),"Empty matrix product\n");CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}


PetscErrorCode MatMatMultSymbolic_SeqAIJ_SeqAIJ_Scalable(Mat A,Mat B,PetscReal fill,Mat *C)
{
  PetscErrorCode     ierr;
  Mat_SeqAIJ         *a  = (Mat_SeqAIJ*)A->data,*b=(Mat_SeqAIJ*)B->data,*c;
  PetscInt           *ai = a->i,*bi=b->i,*ci,*cj;
  PetscInt           am  = A->rmap->N,bn=B->cmap->N,bm=B->rmap->N;
  MatScalar          *ca;
  PetscReal          afill;
  PetscInt           i,j,anzi,brow,bnzj,cnzi,*bj,*aj,*lnk,ndouble=0,Crmax;
  PetscTable         ta;
  PetscFreeSpaceList free_space=NULL,current_space=NULL;

  PetscFunctionBegin;
  /* Get ci and cj - same as MatMatMultSymbolic_SeqAIJ_SeqAIJ except using PetscLLxxx_Scalalbe() */
  /*---------------------------------------------------------------------------------------------*/
  /* Allocate arrays for fill computation and free space for accumulating nonzero column */
  ierr  = PetscMalloc1(am+2,&ci);CHKERRQ(ierr);
  ci[0] = 0;

  /* create and initialize a linked list */
  ierr = PetscTableCreate(bn,bn,&ta);CHKERRQ(ierr);
  MatRowMergeMax_SeqAIJ(b,bm,ta);
  ierr = PetscTableGetCount(ta,&Crmax);CHKERRQ(ierr);
  ierr = PetscTableDestroy(&ta);CHKERRQ(ierr);
  ierr = PetscLLCondensedCreate_Scalable(Crmax,&lnk);CHKERRQ(ierr);

  /* Initial FreeSpace size is fill*(nnz(A)+nnz(B)) */
  ierr          = PetscFreeSpaceGet(PetscRealIntMultTruncate(fill,PetscIntSumTruncate(ai[am],bi[bm])),&free_space);CHKERRQ(ierr);
  current_space = free_space;

  /* Determine ci and cj */
  for (i=0; i<am; i++) {
    anzi = ai[i+1] - ai[i];
    aj   = a->j + ai[i];
    for (j=0; j<anzi; j++) {
      brow = aj[j];
      bnzj = bi[brow+1] - bi[brow];
      bj   = b->j + bi[brow];
      /* add non-zero cols of B into the sorted linked list lnk */
      ierr = PetscLLCondensedAddSorted_Scalable(bnzj,bj,lnk);CHKERRQ(ierr);
    }
    cnzi = lnk[0];

    /* If free space is not available, make more free space */
    /* Double the amount of total space in the list */
    if (current_space->local_remaining<cnzi) {
      ierr = PetscFreeSpaceGet(PetscIntSumTruncate(cnzi,current_space->total_array_size),&current_space);CHKERRQ(ierr);
      ndouble++;
    }

    /* Copy data into free space, then initialize lnk */
    ierr = PetscLLCondensedClean_Scalable(cnzi,current_space->array,lnk);CHKERRQ(ierr);

    current_space->array           += cnzi;
    current_space->local_used      += cnzi;
    current_space->local_remaining -= cnzi;

    ci[i+1] = ci[i] + cnzi;
  }

  /* Column indices are in the list of free space */
  /* Allocate space for cj, initialize cj, and */
  /* destroy list of free space and other temporary array(s) */
  ierr = PetscMalloc1(ci[am]+1,&cj);CHKERRQ(ierr);
  ierr = PetscFreeSpaceContiguous(&free_space,cj);CHKERRQ(ierr);
  ierr = PetscLLCondensedDestroy_Scalable(lnk);CHKERRQ(ierr);

  /* Allocate space for ca */
  /*-----------------------*/
  ierr = PetscCalloc1(ci[am]+1,&ca);CHKERRQ(ierr);

  /* put together the new symbolic matrix */
  ierr = MatCreateSeqAIJWithArrays(PetscObjectComm((PetscObject)A),am,bn,ci,cj,ca,C);CHKERRQ(ierr);
  ierr = MatSetBlockSizesFromMats(*C,A,B);CHKERRQ(ierr);
  ierr = MatSetType(*C,((PetscObject)A)->type_name);CHKERRQ(ierr);

  /* MatCreateSeqAIJWithArrays flags matrix so PETSc doesn't free the user's arrays. */
  /* These are PETSc arrays, so change flags so arrays can be deleted by PETSc */
  c          = (Mat_SeqAIJ*)((*C)->data);
  c->free_a  = PETSC_TRUE;
  c->free_ij = PETSC_TRUE;
  c->nonew   = 0;

  (*C)->ops->matmultnumeric = MatMatMultNumeric_SeqAIJ_SeqAIJ_Scalable; /* slower, less memory */

  /* set MatInfo */
  afill = (PetscReal)ci[am]/(ai[am]+bi[bm]) + 1.e-5;
  if (afill < 1.0) afill = 1.0;
  c->maxnz                     = ci[am];
  c->nz                        = ci[am];
  (*C)->info.mallocs           = ndouble;
  (*C)->info.fill_ratio_given  = fill;
  (*C)->info.fill_ratio_needed = afill;

#if defined(PETSC_USE_INFO)
  if (ci[am]) {
    ierr = PetscInfo3((*C),"Reallocs %D; Fill ratio: given %g needed %g.\n",ndouble,(double)fill,(double)afill);CHKERRQ(ierr);
    ierr = PetscInfo1((*C),"Use MatMatMult(A,B,MatReuse,%g,&C) for best performance.;\n",(double)afill);CHKERRQ(ierr);
  } else {
    ierr = PetscInfo((*C),"Empty matrix product\n");CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatMultSymbolic_SeqAIJ_SeqAIJ_Heap(Mat A,Mat B,PetscReal fill,Mat *C)
{
  PetscErrorCode     ierr;
  Mat_SeqAIJ         *a = (Mat_SeqAIJ*)A->data,*b=(Mat_SeqAIJ*)B->data,*c;
  const PetscInt     *ai=a->i,*bi=b->i,*aj=a->j,*bj=b->j;
  PetscInt           *ci,*cj,*bb;
  PetscInt           am=A->rmap->N,bn=B->cmap->N,bm=B->rmap->N;
  PetscReal          afill;
  PetscInt           i,j,col,ndouble = 0;
  PetscFreeSpaceList free_space=NULL,current_space=NULL;
  PetscHeap          h;

  PetscFunctionBegin;
  /* Get ci and cj - by merging sorted rows using a heap */
  /*---------------------------------------------------------------------------------------------*/
  /* Allocate arrays for fill computation and free space for accumulating nonzero column */
  ierr  = PetscMalloc1(am+2,&ci);CHKERRQ(ierr);
  ci[0] = 0;

  /* Initial FreeSpace size is fill*(nnz(A)+nnz(B)) */
  ierr          = PetscFreeSpaceGet(PetscRealIntMultTruncate(fill,PetscIntSumTruncate(ai[am],bi[bm])),&free_space);CHKERRQ(ierr);
  current_space = free_space;

  ierr = PetscHeapCreate(a->rmax,&h);CHKERRQ(ierr);
  ierr = PetscMalloc1(a->rmax,&bb);CHKERRQ(ierr);

  /* Determine ci and cj */
  for (i=0; i<am; i++) {
    const PetscInt anzi  = ai[i+1] - ai[i]; /* number of nonzeros in this row of A, this is the number of rows of B that we merge */
    const PetscInt *acol = aj + ai[i]; /* column indices of nonzero entries in this row */
    ci[i+1] = ci[i];
    /* Populate the min heap */
    for (j=0; j<anzi; j++) {
      bb[j] = bi[acol[j]];         /* bb points at the start of the row */
      if (bb[j] < bi[acol[j]+1]) { /* Add if row is nonempty */
        ierr = PetscHeapAdd(h,j,bj[bb[j]++]);CHKERRQ(ierr);
      }
    }
    /* Pick off the min element, adding it to free space */
    ierr = PetscHeapPop(h,&j,&col);CHKERRQ(ierr);
    while (j >= 0) {
      if (current_space->local_remaining < 1) { /* double the size, but don't exceed 16 MiB */
        ierr = PetscFreeSpaceGet(PetscMin(PetscIntMultTruncate(2,current_space->total_array_size),16 << 20),&current_space);CHKERRQ(ierr);
        ndouble++;
      }
      *(current_space->array++) = col;
      current_space->local_used++;
      current_space->local_remaining--;
      ci[i+1]++;

      /* stash if anything else remains in this row of B */
      if (bb[j] < bi[acol[j]+1]) {ierr = PetscHeapStash(h,j,bj[bb[j]++]);CHKERRQ(ierr);}
      while (1) {               /* pop and stash any other rows of B that also had an entry in this column */
        PetscInt j2,col2;
        ierr = PetscHeapPeek(h,&j2,&col2);CHKERRQ(ierr);
        if (col2 != col) break;
        ierr = PetscHeapPop(h,&j2,&col2);CHKERRQ(ierr);
        if (bb[j2] < bi[acol[j2]+1]) {ierr = PetscHeapStash(h,j2,bj[bb[j2]++]);CHKERRQ(ierr);}
      }
      /* Put any stashed elements back into the min heap */
      ierr = PetscHeapUnstash(h);CHKERRQ(ierr);
      ierr = PetscHeapPop(h,&j,&col);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(bb);CHKERRQ(ierr);
  ierr = PetscHeapDestroy(&h);CHKERRQ(ierr);

  /* Column indices are in the list of free space */
  /* Allocate space for cj, initialize cj, and */
  /* destroy list of free space and other temporary array(s) */
  ierr = PetscMalloc1(ci[am],&cj);CHKERRQ(ierr);
  ierr = PetscFreeSpaceContiguous(&free_space,cj);CHKERRQ(ierr);

  /* put together the new symbolic matrix */
  ierr = MatCreateSeqAIJWithArrays(PetscObjectComm((PetscObject)A),am,bn,ci,cj,NULL,C);CHKERRQ(ierr);
  ierr = MatSetBlockSizesFromMats(*C,A,B);CHKERRQ(ierr);
  ierr = MatSetType(*C,((PetscObject)A)->type_name);CHKERRQ(ierr);

  /* MatCreateSeqAIJWithArrays flags matrix so PETSc doesn't free the user's arrays. */
  /* These are PETSc arrays, so change flags so arrays can be deleted by PETSc */
  c          = (Mat_SeqAIJ*)((*C)->data);
  c->free_a  = PETSC_TRUE;
  c->free_ij = PETSC_TRUE;
  c->nonew   = 0;

  (*C)->ops->matmultnumeric = MatMatMultNumeric_SeqAIJ_SeqAIJ_Sorted;

  /* set MatInfo */
  afill = (PetscReal)ci[am]/(ai[am]+bi[bm]) + 1.e-5;
  if (afill < 1.0) afill = 1.0;
  c->maxnz                     = ci[am];
  c->nz                        = ci[am];
  (*C)->info.mallocs           = ndouble;
  (*C)->info.fill_ratio_given  = fill;
  (*C)->info.fill_ratio_needed = afill;

#if defined(PETSC_USE_INFO)
  if (ci[am]) {
    ierr = PetscInfo3((*C),"Reallocs %D; Fill ratio: given %g needed %g.\n",ndouble,(double)fill,(double)afill);CHKERRQ(ierr);
    ierr = PetscInfo1((*C),"Use MatMatMult(A,B,MatReuse,%g,&C) for best performance.;\n",(double)afill);CHKERRQ(ierr);
  } else {
    ierr = PetscInfo((*C),"Empty matrix product\n");CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatMultSymbolic_SeqAIJ_SeqAIJ_BTHeap(Mat A,Mat B,PetscReal fill,Mat *C)
{
  PetscErrorCode     ierr;
  Mat_SeqAIJ         *a  = (Mat_SeqAIJ*)A->data,*b=(Mat_SeqAIJ*)B->data,*c;
  const PetscInt     *ai = a->i,*bi=b->i,*aj=a->j,*bj=b->j;
  PetscInt           *ci,*cj,*bb;
  PetscInt           am=A->rmap->N,bn=B->cmap->N,bm=B->rmap->N;
  PetscReal          afill;
  PetscInt           i,j,col,ndouble = 0;
  PetscFreeSpaceList free_space=NULL,current_space=NULL;
  PetscHeap          h;
  PetscBT            bt;

  PetscFunctionBegin;
  /* Get ci and cj - using a heap for the sorted rows, but use BT so that each index is only added once */
  /*---------------------------------------------------------------------------------------------*/
  /* Allocate arrays for fill computation and free space for accumulating nonzero column */
  ierr  = PetscMalloc1(am+2,&ci);CHKERRQ(ierr);
  ci[0] = 0;

  /* Initial FreeSpace size is fill*(nnz(A)+nnz(B)) */
  ierr = PetscFreeSpaceGet(PetscRealIntMultTruncate(fill,PetscIntSumTruncate(ai[am],bi[bm])),&free_space);CHKERRQ(ierr);

  current_space = free_space;

  ierr = PetscHeapCreate(a->rmax,&h);CHKERRQ(ierr);
  ierr = PetscMalloc1(a->rmax,&bb);CHKERRQ(ierr);
  ierr = PetscBTCreate(bn,&bt);CHKERRQ(ierr);

  /* Determine ci and cj */
  for (i=0; i<am; i++) {
    const PetscInt anzi  = ai[i+1] - ai[i]; /* number of nonzeros in this row of A, this is the number of rows of B that we merge */
    const PetscInt *acol = aj + ai[i]; /* column indices of nonzero entries in this row */
    const PetscInt *fptr = current_space->array; /* Save beginning of the row so we can clear the BT later */
    ci[i+1] = ci[i];
    /* Populate the min heap */
    for (j=0; j<anzi; j++) {
      PetscInt brow = acol[j];
      for (bb[j] = bi[brow]; bb[j] < bi[brow+1]; bb[j]++) {
        PetscInt bcol = bj[bb[j]];
        if (!PetscBTLookupSet(bt,bcol)) { /* new entry */
          ierr = PetscHeapAdd(h,j,bcol);CHKERRQ(ierr);
          bb[j]++;
          break;
        }
      }
    }
    /* Pick off the min element, adding it to free space */
    ierr = PetscHeapPop(h,&j,&col);CHKERRQ(ierr);
    while (j >= 0) {
      if (current_space->local_remaining < 1) { /* double the size, but don't exceed 16 MiB */
        fptr = NULL;                      /* need PetscBTMemzero */
        ierr = PetscFreeSpaceGet(PetscMin(PetscIntMultTruncate(2,current_space->total_array_size),16 << 20),&current_space);CHKERRQ(ierr);
        ndouble++;
      }
      *(current_space->array++) = col;
      current_space->local_used++;
      current_space->local_remaining--;
      ci[i+1]++;

      /* stash if anything else remains in this row of B */
      for (; bb[j] < bi[acol[j]+1]; bb[j]++) {
        PetscInt bcol = bj[bb[j]];
        if (!PetscBTLookupSet(bt,bcol)) { /* new entry */
          ierr = PetscHeapAdd(h,j,bcol);CHKERRQ(ierr);
          bb[j]++;
          break;
        }
      }
      ierr = PetscHeapPop(h,&j,&col);CHKERRQ(ierr);
    }
    if (fptr) {                 /* Clear the bits for this row */
      for (; fptr<current_space->array; fptr++) {ierr = PetscBTClear(bt,*fptr);CHKERRQ(ierr);}
    } else {                    /* We reallocated so we don't remember (easily) how to clear only the bits we changed */
      ierr = PetscBTMemzero(bn,bt);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(bb);CHKERRQ(ierr);
  ierr = PetscHeapDestroy(&h);CHKERRQ(ierr);
  ierr = PetscBTDestroy(&bt);CHKERRQ(ierr);

  /* Column indices are in the list of free space */
  /* Allocate space for cj, initialize cj, and */
  /* destroy list of free space and other temporary array(s) */
  ierr = PetscMalloc1(ci[am],&cj);CHKERRQ(ierr);
  ierr = PetscFreeSpaceContiguous(&free_space,cj);CHKERRQ(ierr);

  /* put together the new symbolic matrix */
  ierr = MatCreateSeqAIJWithArrays(PetscObjectComm((PetscObject)A),am,bn,ci,cj,NULL,C);CHKERRQ(ierr);
  ierr = MatSetBlockSizesFromMats(*C,A,B);CHKERRQ(ierr);
  ierr = MatSetType(*C,((PetscObject)A)->type_name);CHKERRQ(ierr);

  /* MatCreateSeqAIJWithArrays flags matrix so PETSc doesn't free the user's arrays. */
  /* These are PETSc arrays, so change flags so arrays can be deleted by PETSc */
  c          = (Mat_SeqAIJ*)((*C)->data);
  c->free_a  = PETSC_TRUE;
  c->free_ij = PETSC_TRUE;
  c->nonew   = 0;

  (*C)->ops->matmultnumeric = MatMatMultNumeric_SeqAIJ_SeqAIJ_Sorted;

  /* set MatInfo */
  afill = (PetscReal)ci[am]/(ai[am]+bi[bm]) + 1.e-5;
  if (afill < 1.0) afill = 1.0;
  c->maxnz                     = ci[am];
  c->nz                        = ci[am];
  (*C)->info.mallocs           = ndouble;
  (*C)->info.fill_ratio_given  = fill;
  (*C)->info.fill_ratio_needed = afill;

#if defined(PETSC_USE_INFO)
  if (ci[am]) {
    ierr = PetscInfo3((*C),"Reallocs %D; Fill ratio: given %g needed %g.\n",ndouble,(double)fill,(double)afill);CHKERRQ(ierr);
    ierr = PetscInfo1((*C),"Use MatMatMult(A,B,MatReuse,%g,&C) for best performance.;\n",(double)afill);CHKERRQ(ierr);
  } else {
    ierr = PetscInfo((*C),"Empty matrix product\n");CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}


PetscErrorCode MatMatMultSymbolic_SeqAIJ_SeqAIJ_RowMerge(Mat A,Mat B,PetscReal fill,Mat *C)
{
  PetscErrorCode     ierr;
  Mat_SeqAIJ         *a=(Mat_SeqAIJ*)A->data,*b=(Mat_SeqAIJ*)B->data,*c;
  const PetscInt     *ai=a->i,*bi=b->i,*aj=a->j,*bj=b->j,*inputi,*inputj,*inputcol,*inputcol_L1;
  PetscInt           *ci,*cj,*outputj,worki_L1[9],worki_L2[9];
  PetscInt           c_maxmem,a_maxrownnz=0,a_rownnz;
  const PetscInt     workcol[8]={0,1,2,3,4,5,6,7};
  const PetscInt     am=A->rmap->N,bn=B->cmap->N,bm=B->rmap->N;
  const PetscInt     *brow_ptr[8],*brow_end[8];
  PetscInt           window[8];
  PetscInt           window_min,old_window_min,ci_nnz,outputi_nnz=0,L1_nrows,L2_nrows;
  PetscInt           i,k,ndouble=0,L1_rowsleft,rowsleft;
  PetscReal          afill;
  PetscInt           *workj_L1,*workj_L2,*workj_L3;
  PetscInt           L1_nnz,L2_nnz;

  /* Step 1: Get upper bound on memory required for allocation.
             Because of the way virtual memory works,
             only the memory pages that are actually needed will be physically allocated. */
  PetscFunctionBegin;
  ierr = PetscMalloc1(am+1,&ci);CHKERRQ(ierr);
  for (i=0; i<am; i++) {
    const PetscInt anzi  = ai[i+1] - ai[i]; /* number of nonzeros in this row of A, this is the number of rows of B that we merge */
    const PetscInt *acol = aj + ai[i]; /* column indices of nonzero entries in this row */
    a_rownnz = 0;
    for (k=0; k<anzi; ++k) {
      a_rownnz += bi[acol[k]+1] - bi[acol[k]];
      if (a_rownnz > bn) {
        a_rownnz = bn;
        break;
      }
    }
    a_maxrownnz = PetscMax(a_maxrownnz, a_rownnz);
  }
  /* temporary work areas for merging rows */
  ierr = PetscMalloc1(a_maxrownnz*8,&workj_L1);CHKERRQ(ierr);
  ierr = PetscMalloc1(a_maxrownnz*8,&workj_L2);CHKERRQ(ierr);
  ierr = PetscMalloc1(a_maxrownnz,&workj_L3);CHKERRQ(ierr);

  /* This should be enough for almost all matrices. If not, memory is reallocated later. */
  c_maxmem = 8*(ai[am]+bi[bm]);
  /* Step 2: Populate pattern for C */
  ierr  = PetscMalloc1(c_maxmem,&cj);CHKERRQ(ierr);

  ci_nnz       = 0;
  ci[0]        = 0;
  worki_L1[0]  = 0;
  worki_L2[0]  = 0;
  for (i=0; i<am; i++) {
    const PetscInt anzi  = ai[i+1] - ai[i]; /* number of nonzeros in this row of A, this is the number of rows of B that we merge */
    const PetscInt *acol = aj + ai[i];      /* column indices of nonzero entries in this row */
    rowsleft             = anzi;
    inputcol_L1          = acol;
    L2_nnz               = 0;
    L2_nrows             = 1;  /* Number of rows to be merged on Level 3. output of L3 already exists -> initial value 1   */
    worki_L2[1]          = 0;
    outputi_nnz          = 0;

    /* If the number of indices in C so far + the max number of columns in the next row > c_maxmem  -> allocate more memory */
    while (ci_nnz+a_maxrownnz > c_maxmem) {
      c_maxmem *= 2;
      ndouble++;
      ierr = PetscRealloc(sizeof(PetscInt)*c_maxmem,&cj);CHKERRQ(ierr);
    }

    while (rowsleft) {
      L1_rowsleft = PetscMin(64, rowsleft); /* In the inner loop max 64 rows of B can be merged */
      L1_nrows    = 0;
      L1_nnz      = 0;
      inputcol    = inputcol_L1;
      inputi      = bi;
      inputj      = bj;

      /* The following macro is used to specialize for small rows in A.
         This helps with compiler unrolling, improving performance substantially.
          Input:  inputj   inputi  inputcol  bn
          Output: outputj  outputi_nnz                       */
       #define MatMatMultSymbolic_RowMergeMacro(ANNZ)                        \
         window_min  = bn;                                                   \
         outputi_nnz = 0;                                                    \
         for (k=0; k<ANNZ; ++k) {                                            \
           brow_ptr[k] = inputj + inputi[inputcol[k]];                       \
           brow_end[k] = inputj + inputi[inputcol[k]+1];                     \
           window[k]   = (brow_ptr[k] != brow_end[k]) ? *brow_ptr[k] : bn;   \
           window_min  = PetscMin(window[k], window_min);                    \
         }                                                                   \
         while (window_min < bn) {                                           \
           outputj[outputi_nnz++] = window_min;                              \
           /* advance front and compute new minimum */                       \
           old_window_min = window_min;                                      \
           window_min = bn;                                                  \
           for (k=0; k<ANNZ; ++k) {                                          \
             if (window[k] == old_window_min) {                              \
               brow_ptr[k]++;                                                \
               window[k] = (brow_ptr[k] != brow_end[k]) ? *brow_ptr[k] : bn; \
             }                                                               \
             window_min = PetscMin(window[k], window_min);                   \
           }                                                                 \
         }

      /************** L E V E L  1 ***************/
      /* Merge up to 8 rows of B to L1 work array*/
      while (L1_rowsleft) {
        outputi_nnz = 0;
        if (anzi > 8)  outputj = workj_L1 + L1_nnz;     /* Level 1 rowmerge*/
        else           outputj = cj + ci_nnz;           /* Merge directly to C */

        switch (L1_rowsleft) {
        case 1:  brow_ptr[0] = inputj + inputi[inputcol[0]];
                 brow_end[0] = inputj + inputi[inputcol[0]+1];
                 for (; brow_ptr[0] != brow_end[0]; ++brow_ptr[0]) outputj[outputi_nnz++] = *brow_ptr[0]; /* copy row in b over */
                 inputcol    += L1_rowsleft;
                 rowsleft    -= L1_rowsleft;
                 L1_rowsleft  = 0;
                 break;
        case 2:  MatMatMultSymbolic_RowMergeMacro(2);
                 inputcol    += L1_rowsleft;
                 rowsleft    -= L1_rowsleft;
                 L1_rowsleft  = 0;
                 break;
        case 3: MatMatMultSymbolic_RowMergeMacro(3);
                 inputcol    += L1_rowsleft;
                 rowsleft    -= L1_rowsleft;
                 L1_rowsleft  = 0;
                 break;
        case 4:  MatMatMultSymbolic_RowMergeMacro(4);
                 inputcol    += L1_rowsleft;
                 rowsleft    -= L1_rowsleft;
                 L1_rowsleft  = 0;
                 break;
        case 5:  MatMatMultSymbolic_RowMergeMacro(5);
                 inputcol    += L1_rowsleft;
                 rowsleft    -= L1_rowsleft;
                 L1_rowsleft  = 0;
                 break;
        case 6:  MatMatMultSymbolic_RowMergeMacro(6);
                 inputcol    += L1_rowsleft;
                 rowsleft    -= L1_rowsleft;
                 L1_rowsleft  = 0;
                 break;
        case 7:  MatMatMultSymbolic_RowMergeMacro(7);
                 inputcol    += L1_rowsleft;
                 rowsleft    -= L1_rowsleft;
                 L1_rowsleft  = 0;
                 break;
        default: MatMatMultSymbolic_RowMergeMacro(8);
                 inputcol    += 8;
                 rowsleft    -= 8;
                 L1_rowsleft -= 8;
                 break;
        }
        inputcol_L1           = inputcol;
        L1_nnz               += outputi_nnz;
        worki_L1[++L1_nrows]  = L1_nnz;
      }

      /********************** L E V E L  2 ************************/
      /* Merge from L1 work array to either C or to L2 work array */
      if (anzi > 8) {
        inputi      = worki_L1;
        inputj      = workj_L1;
        inputcol    = workcol;
        outputi_nnz = 0;

        if (anzi <= 64) outputj = cj + ci_nnz;        /* Merge from L1 work array to C */
        else            outputj = workj_L2 + L2_nnz;  /* Merge from L1 work array to L2 work array */

        switch (L1_nrows) {
        case 1:  brow_ptr[0] = inputj + inputi[inputcol[0]];
                 brow_end[0] = inputj + inputi[inputcol[0]+1];
                 for (; brow_ptr[0] != brow_end[0]; ++brow_ptr[0]) outputj[outputi_nnz++] = *brow_ptr[0]; /* copy row in b over */
                 break;
        case 2:  MatMatMultSymbolic_RowMergeMacro(2); break;
        case 3:  MatMatMultSymbolic_RowMergeMacro(3); break;
        case 4:  MatMatMultSymbolic_RowMergeMacro(4); break;
        case 5:  MatMatMultSymbolic_RowMergeMacro(5); break;
        case 6:  MatMatMultSymbolic_RowMergeMacro(6); break;
        case 7:  MatMatMultSymbolic_RowMergeMacro(7); break;
        case 8:  MatMatMultSymbolic_RowMergeMacro(8); break;
        default: SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"MatMatMult logic error: Not merging 1-8 rows from L1 work array!");
        }
        L2_nnz               += outputi_nnz;
        worki_L2[++L2_nrows]  = L2_nnz;

        /************************ L E V E L  3 **********************/
        /* Merge from L2 work array to either C or to L2 work array */
        if (anzi > 64 && (L2_nrows == 8 || rowsleft == 0)) {
          inputi      = worki_L2;
          inputj      = workj_L2;
          inputcol    = workcol;
          outputi_nnz = 0;
          if (rowsleft) outputj = workj_L3;
          else          outputj = cj + ci_nnz;
          switch (L2_nrows) {
          case 1:  brow_ptr[0] = inputj + inputi[inputcol[0]];
                   brow_end[0] = inputj + inputi[inputcol[0]+1];
                   for (; brow_ptr[0] != brow_end[0]; ++brow_ptr[0]) outputj[outputi_nnz++] = *brow_ptr[0]; /* copy row in b over */
                   break;
          case 2:  MatMatMultSymbolic_RowMergeMacro(2); break;
          case 3:  MatMatMultSymbolic_RowMergeMacro(3); break;
          case 4:  MatMatMultSymbolic_RowMergeMacro(4); break;
          case 5:  MatMatMultSymbolic_RowMergeMacro(5); break;
          case 6:  MatMatMultSymbolic_RowMergeMacro(6); break;
          case 7:  MatMatMultSymbolic_RowMergeMacro(7); break;
          case 8:  MatMatMultSymbolic_RowMergeMacro(8); break;
          default: SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"MatMatMult logic error: Not merging 1-8 rows from L2 work array!");
          }
          L2_nrows    = 1;
          L2_nnz      = outputi_nnz;
          worki_L2[1] = outputi_nnz;
          /* Copy to workj_L2 */
          if (rowsleft) {
            for (k=0; k<outputi_nnz; ++k)  workj_L2[k] = outputj[k];
          }
        }
      }
    }  /* while (rowsleft) */
#undef MatMatMultSymbolic_RowMergeMacro

    /* terminate current row */
    ci_nnz += outputi_nnz;
    ci[i+1] = ci_nnz;
  }

  /* Step 3: Create the new symbolic matrix */
  ierr = MatCreateSeqAIJWithArrays(PetscObjectComm((PetscObject)A),am,bn,ci,cj,NULL,C);CHKERRQ(ierr);
  ierr = MatSetBlockSizesFromMats(*C,A,B);CHKERRQ(ierr);
  ierr = MatSetType(*C,((PetscObject)A)->type_name);CHKERRQ(ierr);

  /* MatCreateSeqAIJWithArrays flags matrix so PETSc doesn't free the user's arrays. */
  /* These are PETSc arrays, so change flags so arrays can be deleted by PETSc */
  c          = (Mat_SeqAIJ*)((*C)->data);
  c->free_a  = PETSC_TRUE;
  c->free_ij = PETSC_TRUE;
  c->nonew   = 0;

  (*C)->ops->matmultnumeric = MatMatMultNumeric_SeqAIJ_SeqAIJ_Sorted;

  /* set MatInfo */
  afill = (PetscReal)ci[am]/(ai[am]+bi[bm]) + 1.e-5;
  if (afill < 1.0) afill = 1.0;
  c->maxnz                     = ci[am];
  c->nz                        = ci[am];
  (*C)->info.mallocs           = ndouble;
  (*C)->info.fill_ratio_given  = fill;
  (*C)->info.fill_ratio_needed = afill;

#if defined(PETSC_USE_INFO)
  if (ci[am]) {
    ierr = PetscInfo3((*C),"Reallocs %D; Fill ratio: given %g needed %g.\n",ndouble,(double)fill,(double)afill);CHKERRQ(ierr);
    ierr = PetscInfo1((*C),"Use MatMatMult(A,B,MatReuse,%g,&C) for best performance.;\n",(double)afill);CHKERRQ(ierr);
  } else {
    ierr = PetscInfo((*C),"Empty matrix product\n");CHKERRQ(ierr);
  }
#endif

  /* Step 4: Free temporary work areas */
  ierr = PetscFree(workj_L1);CHKERRQ(ierr);
  ierr = PetscFree(workj_L2);CHKERRQ(ierr);
  ierr = PetscFree(workj_L3);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* concatenate unique entries and then sort */
PetscErrorCode MatMatMultSymbolic_SeqAIJ_SeqAIJ_Sorted(Mat A,Mat B,PetscReal fill,Mat *C)
{
  PetscErrorCode     ierr;
  Mat_SeqAIJ         *a  = (Mat_SeqAIJ*)A->data,*b=(Mat_SeqAIJ*)B->data,*c;
  const PetscInt     *ai = a->i,*bi=b->i,*aj=a->j,*bj=b->j;
  PetscInt           *ci,*cj;
  PetscInt           am=A->rmap->N,bn=B->cmap->N,bm=B->rmap->N;
  PetscReal          afill;
  PetscInt           i,j,ndouble = 0;
  PetscSegBuffer     seg,segrow;
  char               *seen;

  PetscFunctionBegin;
  ierr  = PetscMalloc1(am+1,&ci);CHKERRQ(ierr);
  ci[0] = 0;

  /* Initial FreeSpace size is fill*(nnz(A)+nnz(B)) */
  ierr = PetscSegBufferCreate(sizeof(PetscInt),(PetscInt)(fill*(ai[am]+bi[bm])),&seg);CHKERRQ(ierr);
  ierr = PetscSegBufferCreate(sizeof(PetscInt),100,&segrow);CHKERRQ(ierr);
  ierr = PetscCalloc1(bn,&seen);CHKERRQ(ierr);

  /* Determine ci and cj */
  for (i=0; i<am; i++) {
    const PetscInt anzi  = ai[i+1] - ai[i]; /* number of nonzeros in this row of A, this is the number of rows of B that we merge */
    const PetscInt *acol = aj + ai[i]; /* column indices of nonzero entries in this row */
    PetscInt packlen = 0,*PETSC_RESTRICT crow;
    /* Pack segrow */
    for (j=0; j<anzi; j++) {
      PetscInt brow = acol[j],bjstart = bi[brow],bjend = bi[brow+1],k;
      for (k=bjstart; k<bjend; k++) {
        PetscInt bcol = bj[k];
        if (!seen[bcol]) { /* new entry */
          PetscInt *PETSC_RESTRICT slot;
          ierr = PetscSegBufferGetInts(segrow,1,&slot);CHKERRQ(ierr);
          *slot = bcol;
          seen[bcol] = 1;
          packlen++;
        }
      }
    }
    ierr = PetscSegBufferGetInts(seg,packlen,&crow);CHKERRQ(ierr);
    ierr = PetscSegBufferExtractTo(segrow,crow);CHKERRQ(ierr);
    ierr = PetscSortInt(packlen,crow);CHKERRQ(ierr);
    ci[i+1] = ci[i] + packlen;
    for (j=0; j<packlen; j++) seen[crow[j]] = 0;
  }
  ierr = PetscSegBufferDestroy(&segrow);CHKERRQ(ierr);
  ierr = PetscFree(seen);CHKERRQ(ierr);

  /* Column indices are in the segmented buffer */
  ierr = PetscSegBufferExtractAlloc(seg,&cj);CHKERRQ(ierr);
  ierr = PetscSegBufferDestroy(&seg);CHKERRQ(ierr);

  /* put together the new symbolic matrix */
  ierr = MatCreateSeqAIJWithArrays(PetscObjectComm((PetscObject)A),am,bn,ci,cj,NULL,C);CHKERRQ(ierr);
  ierr = MatSetBlockSizesFromMats(*C,A,B);CHKERRQ(ierr);
  ierr = MatSetType(*C,((PetscObject)A)->type_name);CHKERRQ(ierr);

  /* MatCreateSeqAIJWithArrays flags matrix so PETSc doesn't free the user's arrays. */
  /* These are PETSc arrays, so change flags so arrays can be deleted by PETSc */
  c          = (Mat_SeqAIJ*)((*C)->data);
  c->free_a  = PETSC_TRUE;
  c->free_ij = PETSC_TRUE;
  c->nonew   = 0;

  (*C)->ops->matmultnumeric = MatMatMultNumeric_SeqAIJ_SeqAIJ_Sorted;

  /* set MatInfo */
  afill = (PetscReal)ci[am]/(ai[am]+bi[bm]) + 1.e-5;
  if (afill < 1.0) afill = 1.0;
  c->maxnz                     = ci[am];
  c->nz                        = ci[am];
  (*C)->info.mallocs           = ndouble;
  (*C)->info.fill_ratio_given  = fill;
  (*C)->info.fill_ratio_needed = afill;

#if defined(PETSC_USE_INFO)
  if (ci[am]) {
    ierr = PetscInfo3((*C),"Reallocs %D; Fill ratio: given %g needed %g.\n",ndouble,(double)fill,(double)afill);CHKERRQ(ierr);
    ierr = PetscInfo1((*C),"Use MatMatMult(A,B,MatReuse,%g,&C) for best performance.;\n",(double)afill);CHKERRQ(ierr);
  } else {
    ierr = PetscInfo((*C),"Empty matrix product\n");CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}

/* This routine is not used. Should be removed! */
PetscErrorCode MatMatTransposeMult_SeqAIJ_SeqAIJ(Mat A,Mat B,MatReuse scall,PetscReal fill,Mat *C)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (scall == MAT_INITIAL_MATRIX) {
    ierr = PetscLogEventBegin(MAT_MatTransposeMultSymbolic,A,B,0,0);CHKERRQ(ierr);
    ierr = MatMatTransposeMultSymbolic_SeqAIJ_SeqAIJ(A,B,fill,C);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(MAT_MatTransposeMultSymbolic,A,B,0,0);CHKERRQ(ierr);
  }
  ierr = PetscLogEventBegin(MAT_MatTransposeMultNumeric,A,B,0,0);CHKERRQ(ierr);
  ierr = MatMatTransposeMultNumeric_SeqAIJ_SeqAIJ(A,B,*C);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_MatTransposeMultNumeric,A,B,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_SeqAIJ_MatMatMultTrans(Mat A)
{
  PetscErrorCode      ierr;
  Mat_SeqAIJ          *a=(Mat_SeqAIJ*)A->data;
  Mat_MatMatTransMult *abt=a->abt;

  PetscFunctionBegin;
  ierr = (abt->destroy)(A);CHKERRQ(ierr);
  ierr = MatTransposeColoringDestroy(&abt->matcoloring);CHKERRQ(ierr);
  ierr = MatDestroy(&abt->Bt_den);CHKERRQ(ierr);
  ierr = MatDestroy(&abt->ABt_den);CHKERRQ(ierr);
  ierr = PetscFree(abt);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatTransposeMultSymbolic_SeqAIJ_SeqAIJ(Mat A,Mat B,PetscReal fill,Mat *C)
{
  PetscErrorCode      ierr;
  Mat                 Bt;
  PetscInt            *bti,*btj;
  Mat_MatMatTransMult *abt;
  Mat_SeqAIJ          *c;

  PetscFunctionBegin;
  /* create symbolic Bt */
  ierr = MatGetSymbolicTranspose_SeqAIJ(B,&bti,&btj);CHKERRQ(ierr);
  ierr = MatCreateSeqAIJWithArrays(PETSC_COMM_SELF,B->cmap->n,B->rmap->n,bti,btj,NULL,&Bt);CHKERRQ(ierr);
  ierr = MatSetBlockSizes(Bt,PetscAbs(A->cmap->bs),PetscAbs(B->cmap->bs));CHKERRQ(ierr);
  ierr = MatSetType(Bt,((PetscObject)A)->type_name);CHKERRQ(ierr);

  /* get symbolic C=A*Bt */
  ierr = MatMatMultSymbolic_SeqAIJ_SeqAIJ(A,Bt,fill,C);CHKERRQ(ierr);

  /* create a supporting struct for reuse intermidiate dense matrices with matcoloring */
  ierr   = PetscNew(&abt);CHKERRQ(ierr);
  c      = (Mat_SeqAIJ*)(*C)->data;
  c->abt = abt;

  abt->usecoloring = PETSC_FALSE;
  abt->destroy     = (*C)->ops->destroy;
  (*C)->ops->destroy     = MatDestroy_SeqAIJ_MatMatMultTrans;

  ierr = PetscOptionsGetBool(((PetscObject)A)->options,NULL,"-matmattransmult_color",&abt->usecoloring,NULL);CHKERRQ(ierr);
  if (abt->usecoloring) {
    /* Create MatTransposeColoring from symbolic C=A*B^T */
    MatTransposeColoring matcoloring;
    MatColoring          coloring;
    ISColoring           iscoloring;
    Mat                  Bt_dense,C_dense;
    Mat_SeqAIJ           *c=(Mat_SeqAIJ*)(*C)->data;
    /* inode causes memory problem, don't know why */
    if (c->inode.use) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"MAT_USE_INODES is not supported. Use '-mat_no_inode'");

    ierr = MatColoringCreate(*C,&coloring);CHKERRQ(ierr);
    ierr = MatColoringSetDistance(coloring,2);CHKERRQ(ierr);
    ierr = MatColoringSetType(coloring,MATCOLORINGSL);CHKERRQ(ierr);
    ierr = MatColoringSetFromOptions(coloring);CHKERRQ(ierr);
    ierr = MatColoringApply(coloring,&iscoloring);CHKERRQ(ierr);
    ierr = MatColoringDestroy(&coloring);CHKERRQ(ierr);
    ierr = MatTransposeColoringCreate(*C,iscoloring,&matcoloring);CHKERRQ(ierr);

    abt->matcoloring = matcoloring;

    ierr = ISColoringDestroy(&iscoloring);CHKERRQ(ierr);

    /* Create Bt_dense and C_dense = A*Bt_dense */
    ierr = MatCreate(PETSC_COMM_SELF,&Bt_dense);CHKERRQ(ierr);
    ierr = MatSetSizes(Bt_dense,A->cmap->n,matcoloring->ncolors,A->cmap->n,matcoloring->ncolors);CHKERRQ(ierr);
    ierr = MatSetType(Bt_dense,MATSEQDENSE);CHKERRQ(ierr);
    ierr = MatSeqDenseSetPreallocation(Bt_dense,NULL);CHKERRQ(ierr);

    Bt_dense->assembled = PETSC_TRUE;
    abt->Bt_den   = Bt_dense;

    ierr = MatCreate(PETSC_COMM_SELF,&C_dense);CHKERRQ(ierr);
    ierr = MatSetSizes(C_dense,A->rmap->n,matcoloring->ncolors,A->rmap->n,matcoloring->ncolors);CHKERRQ(ierr);
    ierr = MatSetType(C_dense,MATSEQDENSE);CHKERRQ(ierr);
    ierr = MatSeqDenseSetPreallocation(C_dense,NULL);CHKERRQ(ierr);

    Bt_dense->assembled = PETSC_TRUE;
    abt->ABt_den  = C_dense;

#if defined(PETSC_USE_INFO)
    {
      Mat_SeqAIJ *c = (Mat_SeqAIJ*)(*C)->data;
      ierr = PetscInfo7(*C,"Use coloring of C=A*B^T; B^T: %D %D, Bt_dense: %D,%D; Cnz %D / (cm*ncolors %D) = %g\n",B->cmap->n,B->rmap->n,Bt_dense->rmap->n,Bt_dense->cmap->n,c->nz,A->rmap->n*matcoloring->ncolors,(PetscReal)(c->nz)/(A->rmap->n*matcoloring->ncolors));CHKERRQ(ierr);
    }
#endif
  }
  /* clean up */
  ierr = MatDestroy(&Bt);CHKERRQ(ierr);
  ierr = MatRestoreSymbolicTranspose_SeqAIJ(B,&bti,&btj);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatTransposeMultNumeric_SeqAIJ_SeqAIJ(Mat A,Mat B,Mat C)
{
  PetscErrorCode      ierr;
  Mat_SeqAIJ          *a   =(Mat_SeqAIJ*)A->data,*b=(Mat_SeqAIJ*)B->data,*c=(Mat_SeqAIJ*)C->data;
  PetscInt            *ai  =a->i,*aj=a->j,*bi=b->i,*bj=b->j,anzi,bnzj,nexta,nextb,*acol,*bcol,brow;
  PetscInt            cm   =C->rmap->n,*ci=c->i,*cj=c->j,i,j,cnzi,*ccol;
  PetscLogDouble      flops=0.0;
  MatScalar           *aa  =a->a,*aval,*ba=b->a,*bval,*ca,*cval;
  Mat_MatMatTransMult *abt = c->abt;

  PetscFunctionBegin;
  /* clear old values in C */
  if (!c->a) {
    ierr      = PetscCalloc1(ci[cm]+1,&ca);CHKERRQ(ierr);
    c->a      = ca;
    c->free_a = PETSC_TRUE;
  } else {
    ca =  c->a;
    ierr = PetscArrayzero(ca,ci[cm]+1);CHKERRQ(ierr);
  }

  if (abt->usecoloring) {
    MatTransposeColoring matcoloring = abt->matcoloring;
    Mat                  Bt_dense,C_dense = abt->ABt_den;

    /* Get Bt_dense by Apply MatTransposeColoring to B */
    Bt_dense = abt->Bt_den;
    ierr = MatTransColoringApplySpToDen(matcoloring,B,Bt_dense);CHKERRQ(ierr);

    /* C_dense = A*Bt_dense */
    ierr = MatMatMultNumeric_SeqAIJ_SeqDense(A,Bt_dense,C_dense);CHKERRQ(ierr);

    /* Recover C from C_dense */
    ierr = MatTransColoringApplyDenToSp(matcoloring,C_dense,C);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  for (i=0; i<cm; i++) {
    anzi = ai[i+1] - ai[i];
    acol = aj + ai[i];
    aval = aa + ai[i];
    cnzi = ci[i+1] - ci[i];
    ccol = cj + ci[i];
    cval = ca + ci[i];
    for (j=0; j<cnzi; j++) {
      brow = ccol[j];
      bnzj = bi[brow+1] - bi[brow];
      bcol = bj + bi[brow];
      bval = ba + bi[brow];

      /* perform sparse inner-product c(i,j)=A[i,:]*B[j,:]^T */
      nexta = 0; nextb = 0;
      while (nexta<anzi && nextb<bnzj) {
        while (nexta < anzi && acol[nexta] < bcol[nextb]) nexta++;
        if (nexta == anzi) break;
        while (nextb < bnzj && acol[nexta] > bcol[nextb]) nextb++;
        if (nextb == bnzj) break;
        if (acol[nexta] == bcol[nextb]) {
          cval[j] += aval[nexta]*bval[nextb];
          nexta++; nextb++;
          flops += 2;
        }
      }
    }
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscLogFlops(flops);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_SeqAIJ_MatTransMatMult(Mat A)
{
  PetscErrorCode      ierr;
  Mat_SeqAIJ          *a = (Mat_SeqAIJ*)A->data;
  Mat_MatTransMatMult *atb = a->atb;

  PetscFunctionBegin;
  if (atb) {
    ierr = MatDestroy(&atb->At);CHKERRQ(ierr);
    ierr = (*atb->destroy)(A);CHKERRQ(ierr);
  }
  ierr = PetscFree(atb);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatTransposeMatMult_SeqAIJ_SeqAIJ(Mat A,Mat B,MatReuse scall,PetscReal fill,Mat *C)
{
  PetscErrorCode      ierr;
  const char          *algTypes[2] = {"matmatmult","outerproduct"};
  PetscInt            alg=0; /* set default algorithm */
  Mat                 At;
  Mat_MatTransMatMult *atb;
  Mat_SeqAIJ          *c;

  PetscFunctionBegin;
  if (scall == MAT_INITIAL_MATRIX) {
    ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)A),((PetscObject)A)->prefix,"MatTransposeMatMult","Mat");CHKERRQ(ierr);
    ierr = PetscOptionsEList("-mattransposematmult_via","Algorithmic approach","MatTransposeMatMult",algTypes,2,algTypes[0],&alg,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);

    switch (alg) {
    case 1:
      ierr = MatTransposeMatMultSymbolic_SeqAIJ_SeqAIJ(A,B,fill,C);CHKERRQ(ierr);
      break;
    default:
      ierr = PetscNew(&atb);CHKERRQ(ierr);
      ierr = MatTranspose_SeqAIJ(A,MAT_INITIAL_MATRIX,&At);CHKERRQ(ierr);
      ierr = MatMatMult_SeqAIJ_SeqAIJ(At,B,MAT_INITIAL_MATRIX,fill,C);CHKERRQ(ierr);

      c                  = (Mat_SeqAIJ*)(*C)->data;
      c->atb             = atb;
      atb->At            = At;
      atb->destroy       = (*C)->ops->destroy;
      (*C)->ops->destroy = MatDestroy_SeqAIJ_MatTransMatMult;

      break;
    }
  }
  if (alg) {
    ierr = (*(*C)->ops->mattransposemultnumeric)(A,B,*C);CHKERRQ(ierr);
  } else if (!alg && scall == MAT_REUSE_MATRIX) {
    c   = (Mat_SeqAIJ*)(*C)->data;
    atb = c->atb;
    At  = atb->At;
    ierr = MatTranspose_SeqAIJ(A,MAT_REUSE_MATRIX,&At);CHKERRQ(ierr);
    ierr = MatMatMult_SeqAIJ_SeqAIJ(At,B,MAT_REUSE_MATRIX,fill,C);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatTransposeMatMultSymbolic_SeqAIJ_SeqAIJ(Mat A,Mat B,PetscReal fill,Mat *C)
{
  PetscErrorCode ierr;
  Mat            At;
  PetscInt       *ati,*atj;

  PetscFunctionBegin;
  /* create symbolic At */
  ierr = MatGetSymbolicTranspose_SeqAIJ(A,&ati,&atj);CHKERRQ(ierr);
  ierr = MatCreateSeqAIJWithArrays(PETSC_COMM_SELF,A->cmap->n,A->rmap->n,ati,atj,NULL,&At);CHKERRQ(ierr);
  ierr = MatSetBlockSizes(At,PetscAbs(A->cmap->bs),PetscAbs(B->cmap->bs));CHKERRQ(ierr);
  ierr = MatSetType(At,((PetscObject)A)->type_name);CHKERRQ(ierr);

  /* get symbolic C=At*B */
  ierr = MatMatMultSymbolic_SeqAIJ_SeqAIJ(At,B,fill,C);CHKERRQ(ierr);

  /* clean up */
  ierr = MatDestroy(&At);CHKERRQ(ierr);
  ierr = MatRestoreSymbolicTranspose_SeqAIJ(A,&ati,&atj);CHKERRQ(ierr);

  (*C)->ops->mattransposemultnumeric = MatTransposeMatMultNumeric_SeqAIJ_SeqAIJ;
  PetscFunctionReturn(0);
}

PetscErrorCode MatTransposeMatMultNumeric_SeqAIJ_SeqAIJ(Mat A,Mat B,Mat C)
{
  PetscErrorCode ierr;
  Mat_SeqAIJ     *a   =(Mat_SeqAIJ*)A->data,*b=(Mat_SeqAIJ*)B->data,*c=(Mat_SeqAIJ*)C->data;
  PetscInt       am   =A->rmap->n,anzi,*ai=a->i,*aj=a->j,*bi=b->i,*bj,bnzi,nextb;
  PetscInt       cm   =C->rmap->n,*ci=c->i,*cj=c->j,crow,*cjj,i,j,k;
  PetscLogDouble flops=0.0;
  MatScalar      *aa  =a->a,*ba,*ca,*caj;

  PetscFunctionBegin;
  if (!c->a) {
    ierr = PetscCalloc1(ci[cm]+1,&ca);CHKERRQ(ierr);

    c->a      = ca;
    c->free_a = PETSC_TRUE;
  } else {
    ca   = c->a;
    ierr = PetscArrayzero(ca,ci[cm]);CHKERRQ(ierr);
  }

  /* compute A^T*B using outer product (A^T)[:,i]*B[i,:] */
  for (i=0; i<am; i++) {
    bj   = b->j + bi[i];
    ba   = b->a + bi[i];
    bnzi = bi[i+1] - bi[i];
    anzi = ai[i+1] - ai[i];
    for (j=0; j<anzi; j++) {
      nextb = 0;
      crow  = *aj++;
      cjj   = cj + ci[crow];
      caj   = ca + ci[crow];
      /* perform sparse axpy operation.  Note cjj includes bj. */
      for (k=0; nextb<bnzi; k++) {
        if (cjj[k] == *(bj+nextb)) { /* ccol == bcol */
          caj[k] += (*aa)*(*(ba+nextb));
          nextb++;
        }
      }
      flops += 2*bnzi;
      aa++;
    }
  }

  /* Assemble the final matrix and clean up */
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscLogFlops(flops);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatMatMult_SeqAIJ_SeqDense(Mat A,Mat B,MatReuse scall,PetscReal fill,Mat *C)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (scall == MAT_INITIAL_MATRIX) {
    ierr = PetscLogEventBegin(MAT_MatMultSymbolic,A,B,0,0);CHKERRQ(ierr);
    ierr = MatMatMultSymbolic_SeqAIJ_SeqDense(A,B,fill,C);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(MAT_MatMultSymbolic,A,B,0,0);CHKERRQ(ierr);
  }
  ierr = PetscLogEventBegin(MAT_MatMultNumeric,A,B,0,0);CHKERRQ(ierr);
  ierr = MatMatMultNumeric_SeqAIJ_SeqDense(A,B,*C);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_MatMultNumeric,A,B,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatMultSymbolic_SeqAIJ_SeqDense(Mat A,Mat B,PetscReal fill,Mat *C)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMatMultSymbolic_SeqDense_SeqDense(A,B,0.0,C);CHKERRQ(ierr);

  (*C)->ops->matmultnumeric = MatMatMultNumeric_SeqAIJ_SeqDense;
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatMultNumericAdd_SeqAIJ_SeqDense(Mat A,Mat B,Mat C)
{
  Mat_SeqAIJ        *a=(Mat_SeqAIJ*)A->data;
  Mat_SeqDense      *bd = (Mat_SeqDense*)B->data;
  PetscErrorCode    ierr;
  PetscScalar       *c,r1,r2,r3,r4,*c1,*c2,*c3,*c4,aatmp;
  const PetscScalar *aa,*b,*b1,*b2,*b3,*b4,*av;
  const PetscInt    *aj;
  PetscInt          cm=C->rmap->n,cn=B->cmap->n,bm=bd->lda,am=A->rmap->n;
  PetscInt          am4=4*am,bm4=4*bm,col,i,j,n,ajtmp;

  PetscFunctionBegin;
  if (!cm || !cn) PetscFunctionReturn(0);
  ierr = MatSeqAIJGetArrayRead(A,&av);CHKERRQ(ierr);
  ierr = MatDenseGetArray(C,&c);CHKERRQ(ierr);
  ierr = MatDenseGetArrayRead(B,&b);CHKERRQ(ierr);
  b1 = b; b2 = b1 + bm; b3 = b2 + bm; b4 = b3 + bm;
  c1 = c; c2 = c1 + am; c3 = c2 + am; c4 = c3 + am;
  for (col=0; col<cn-4; col += 4) {  /* over columns of C */
    for (i=0; i<am; i++) {        /* over rows of C in those columns */
      r1 = r2 = r3 = r4 = 0.0;
      n  = a->i[i+1] - a->i[i];
      aj = a->j + a->i[i];
      aa = av + a->i[i];
      for (j=0; j<n; j++) {
        aatmp = aa[j]; ajtmp = aj[j];
        r1 += aatmp*b1[ajtmp];
        r2 += aatmp*b2[ajtmp];
        r3 += aatmp*b3[ajtmp];
        r4 += aatmp*b4[ajtmp];
      }
      c1[i] += r1;
      c2[i] += r2;
      c3[i] += r3;
      c4[i] += r4;
    }
    b1 += bm4; b2 += bm4; b3 += bm4; b4 += bm4;
    c1 += am4; c2 += am4; c3 += am4; c4 += am4;
  }
  for (; col<cn; col++) {   /* over extra columns of C */
    for (i=0; i<am; i++) {  /* over rows of C in those columns */
      r1 = 0.0;
      n  = a->i[i+1] - a->i[i];
      aj = a->j + a->i[i];
      aa = av + a->i[i];
      for (j=0; j<n; j++) {
        r1 += aa[j]*b1[aj[j]];
      }
      c1[i] += r1;
    }
    b1 += bm;
    c1 += am;
  }
  ierr = PetscLogFlops(cn*(2.0*a->nz));CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(C,&c);CHKERRQ(ierr);
  ierr = MatDenseRestoreArrayRead(B,&b);CHKERRQ(ierr);
  ierr = MatSeqAIJRestoreArrayRead(A,&av);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatMultNumeric_SeqAIJ_SeqDense(Mat A,Mat B,Mat C)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (B->rmap->n != A->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Number columns in A %D not equal rows in B %D\n",A->cmap->n,B->rmap->n);
  if (A->rmap->n != C->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Number rows in C %D not equal rows in A %D\n",C->rmap->n,A->rmap->n);
  if (B->cmap->n != C->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Number columns in B %D not equal columns in C %D\n",B->cmap->n,C->cmap->n);

  ierr = MatZeroEntries(C);CHKERRQ(ierr);
  ierr = MatMatMultNumericAdd_SeqAIJ_SeqDense(A,B,C);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode  MatTransColoringApplySpToDen_SeqAIJ(MatTransposeColoring coloring,Mat B,Mat Btdense)
{
  PetscErrorCode ierr;
  Mat_SeqAIJ     *b       = (Mat_SeqAIJ*)B->data;
  Mat_SeqDense   *btdense = (Mat_SeqDense*)Btdense->data;
  PetscInt       *bi      = b->i,*bj=b->j;
  PetscInt       m        = Btdense->rmap->n,n=Btdense->cmap->n,j,k,l,col,anz,*btcol,brow,ncolumns;
  MatScalar      *btval,*btval_den,*ba=b->a;
  PetscInt       *columns=coloring->columns,*colorforcol=coloring->colorforcol,ncolors=coloring->ncolors;

  PetscFunctionBegin;
  btval_den=btdense->v;
  ierr     = PetscArrayzero(btval_den,m*n);CHKERRQ(ierr);
  for (k=0; k<ncolors; k++) {
    ncolumns = coloring->ncolumns[k];
    for (l=0; l<ncolumns; l++) { /* insert a row of B to a column of Btdense */
      col   = *(columns + colorforcol[k] + l);
      btcol = bj + bi[col];
      btval = ba + bi[col];
      anz   = bi[col+1] - bi[col];
      for (j=0; j<anz; j++) {
        brow            = btcol[j];
        btval_den[brow] = btval[j];
      }
    }
    btval_den += m;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatTransColoringApplyDenToSp_SeqAIJ(MatTransposeColoring matcoloring,Mat Cden,Mat Csp)
{
  PetscErrorCode    ierr;
  Mat_SeqAIJ        *csp = (Mat_SeqAIJ*)Csp->data;
  const PetscScalar *ca_den,*ca_den_ptr;
  PetscScalar       *ca=csp->a;
  PetscInt          k,l,m=Cden->rmap->n,ncolors=matcoloring->ncolors;
  PetscInt          brows=matcoloring->brows,*den2sp=matcoloring->den2sp;
  PetscInt          nrows,*row,*idx;
  PetscInt          *rows=matcoloring->rows,*colorforrow=matcoloring->colorforrow;

  PetscFunctionBegin;
  ierr   = MatDenseGetArrayRead(Cden,&ca_den);CHKERRQ(ierr);

  if (brows > 0) {
    PetscInt *lstart,row_end,row_start;
    lstart = matcoloring->lstart;
    ierr = PetscArrayzero(lstart,ncolors);CHKERRQ(ierr);

    row_end = brows;
    if (row_end > m) row_end = m;
    for (row_start=0; row_start<m; row_start+=brows) { /* loop over row blocks of Csp */
      ca_den_ptr = ca_den;
      for (k=0; k<ncolors; k++) { /* loop over colors (columns of Cden) */
        nrows = matcoloring->nrows[k];
        row   = rows  + colorforrow[k];
        idx   = den2sp + colorforrow[k];
        for (l=lstart[k]; l<nrows; l++) {
          if (row[l] >= row_end) {
            lstart[k] = l;
            break;
          } else {
            ca[idx[l]] = ca_den_ptr[row[l]];
          }
        }
        ca_den_ptr += m;
      }
      row_end += brows;
      if (row_end > m) row_end = m;
    }
  } else { /* non-blocked impl: loop over columns of Csp - slow if Csp is large */
    ca_den_ptr = ca_den;
    for (k=0; k<ncolors; k++) {
      nrows = matcoloring->nrows[k];
      row   = rows  + colorforrow[k];
      idx   = den2sp + colorforrow[k];
      for (l=0; l<nrows; l++) {
        ca[idx[l]] = ca_den_ptr[row[l]];
      }
      ca_den_ptr += m;
    }
  }

  ierr = MatDenseRestoreArrayRead(Cden,&ca_den);CHKERRQ(ierr);
#if defined(PETSC_USE_INFO)
  if (matcoloring->brows > 0) {
    ierr = PetscInfo1(Csp,"Loop over %D row blocks for den2sp\n",brows);CHKERRQ(ierr);
  } else {
    ierr = PetscInfo(Csp,"Loop over colors/columns of Cden, inefficient for large sparse matrix product \n");CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode MatTransposeColoringCreate_SeqAIJ(Mat mat,ISColoring iscoloring,MatTransposeColoring c)
{
  PetscErrorCode ierr;
  PetscInt       i,n,nrows,Nbs,j,k,m,ncols,col,cm;
  const PetscInt *is,*ci,*cj,*row_idx;
  PetscInt       nis = iscoloring->n,*rowhit,bs = 1;
  IS             *isa;
  Mat_SeqAIJ     *csp = (Mat_SeqAIJ*)mat->data;
  PetscInt       *colorforrow,*rows,*rows_i,*idxhit,*spidx,*den2sp,*den2sp_i;
  PetscInt       *colorforcol,*columns,*columns_i,brows;
  PetscBool      flg;

  PetscFunctionBegin;
  ierr = ISColoringGetIS(iscoloring,PETSC_USE_POINTER,PETSC_IGNORE,&isa);CHKERRQ(ierr);

  /* bs >1 is not being tested yet! */
  Nbs       = mat->cmap->N/bs;
  c->M      = mat->rmap->N/bs;  /* set total rows, columns and local rows */
  c->N      = Nbs;
  c->m      = c->M;
  c->rstart = 0;
  c->brows  = 100;

  c->ncolors = nis;
  ierr = PetscMalloc3(nis,&c->ncolumns,nis,&c->nrows,nis+1,&colorforrow);CHKERRQ(ierr);
  ierr = PetscMalloc1(csp->nz+1,&rows);CHKERRQ(ierr);
  ierr = PetscMalloc1(csp->nz+1,&den2sp);CHKERRQ(ierr);

  brows = c->brows;
  ierr = PetscOptionsGetInt(NULL,NULL,"-matden2sp_brows",&brows,&flg);CHKERRQ(ierr);
  if (flg) c->brows = brows;
  if (brows > 0) {
    ierr = PetscMalloc1(nis+1,&c->lstart);CHKERRQ(ierr);
  }

  colorforrow[0] = 0;
  rows_i         = rows;
  den2sp_i       = den2sp;

  ierr = PetscMalloc1(nis+1,&colorforcol);CHKERRQ(ierr);
  ierr = PetscMalloc1(Nbs+1,&columns);CHKERRQ(ierr);

  colorforcol[0] = 0;
  columns_i      = columns;

  /* get column-wise storage of mat */
  ierr = MatGetColumnIJ_SeqAIJ_Color(mat,0,PETSC_FALSE,PETSC_FALSE,&ncols,&ci,&cj,&spidx,NULL);CHKERRQ(ierr);

  cm   = c->m;
  ierr = PetscMalloc1(cm+1,&rowhit);CHKERRQ(ierr);
  ierr = PetscMalloc1(cm+1,&idxhit);CHKERRQ(ierr);
  for (i=0; i<nis; i++) { /* loop over color */
    ierr = ISGetLocalSize(isa[i],&n);CHKERRQ(ierr);
    ierr = ISGetIndices(isa[i],&is);CHKERRQ(ierr);

    c->ncolumns[i] = n;
    if (n) {
      ierr = PetscArraycpy(columns_i,is,n);CHKERRQ(ierr);
    }
    colorforcol[i+1] = colorforcol[i] + n;
    columns_i       += n;

    /* fast, crude version requires O(N*N) work */
    ierr = PetscArrayzero(rowhit,cm);CHKERRQ(ierr);

    for (j=0; j<n; j++) { /* loop over columns*/
      col     = is[j];
      row_idx = cj + ci[col];
      m       = ci[col+1] - ci[col];
      for (k=0; k<m; k++) { /* loop over columns marking them in rowhit */
        idxhit[*row_idx]   = spidx[ci[col] + k];
        rowhit[*row_idx++] = col + 1;
      }
    }
    /* count the number of hits */
    nrows = 0;
    for (j=0; j<cm; j++) {
      if (rowhit[j]) nrows++;
    }
    c->nrows[i]      = nrows;
    colorforrow[i+1] = colorforrow[i] + nrows;

    nrows = 0;
    for (j=0; j<cm; j++) { /* loop over rows */
      if (rowhit[j]) {
        rows_i[nrows]   = j;
        den2sp_i[nrows] = idxhit[j];
        nrows++;
      }
    }
    den2sp_i += nrows;

    ierr    = ISRestoreIndices(isa[i],&is);CHKERRQ(ierr);
    rows_i += nrows;
  }
  ierr = MatRestoreColumnIJ_SeqAIJ_Color(mat,0,PETSC_FALSE,PETSC_FALSE,&ncols,&ci,&cj,&spidx,NULL);CHKERRQ(ierr);
  ierr = PetscFree(rowhit);CHKERRQ(ierr);
  ierr = ISColoringRestoreIS(iscoloring,PETSC_USE_POINTER,&isa);CHKERRQ(ierr);
  if (csp->nz != colorforrow[nis]) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"csp->nz %d != colorforrow[nis] %d",csp->nz,colorforrow[nis]);

  c->colorforrow = colorforrow;
  c->rows        = rows;
  c->den2sp      = den2sp;
  c->colorforcol = colorforcol;
  c->columns     = columns;

  ierr = PetscFree(idxhit);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* The combine method has done the symbolic and numeric in the first phase, and so we just return here */
PetscErrorCode MatMatMultNumeric_SeqAIJ_SeqAIJ_Combined(Mat A,Mat B,Mat C)
{
  PetscFunctionBegin;
  /* Empty function */
  PetscFunctionReturn(0);
}

/* This algorithm combines the symbolic and numeric phase of matrix-matrix multiplication. */
PetscErrorCode MatMatMult_SeqAIJ_SeqAIJ_Combined(Mat A,Mat B,PetscReal fill,Mat *C)
{
  PetscErrorCode     ierr;
  PetscLogDouble     flops=0.0;
  Mat_SeqAIJ         *a=(Mat_SeqAIJ*)A->data,*b=(Mat_SeqAIJ*)B->data,*c;
  const PetscInt     *ai=a->i,*bi=b->i;
  PetscInt           *ci,*cj,*cj_i;
  PetscScalar        *ca,*ca_i;
  PetscInt           b_maxmemrow,c_maxmem,a_col;
  PetscInt           am=A->rmap->N,bn=B->cmap->N,bm=B->rmap->N;
  PetscInt           i,k,ndouble=0;
  PetscReal          afill;
  PetscScalar        *c_row_val_dense;
  PetscBool          *c_row_idx_flags;
  PetscInt           *aj_i=a->j;
  PetscScalar        *aa_i=a->a;

  PetscFunctionBegin;

  /* Step 1: Determine upper bounds on memory for C and allocate memory */
  /* This should be enough for almost all matrices. If still more memory is needed, it is reallocated later. */
  c_maxmem    = 8*(ai[am]+bi[bm]);
  b_maxmemrow = PetscMin(bi[bm],bn);
  ierr  = PetscMalloc1(am+1,&ci);CHKERRQ(ierr);
  ierr  = PetscCalloc1(bn,&c_row_val_dense);CHKERRQ(ierr);
  ierr  = PetscCalloc1(bn,&c_row_idx_flags);CHKERRQ(ierr);
  ierr  = PetscMalloc1(c_maxmem,&cj);CHKERRQ(ierr);
  ierr  = PetscMalloc1(c_maxmem,&ca);CHKERRQ(ierr);
  ca_i  = ca;
  cj_i  = cj;
  ci[0] = 0;
  for (i=0; i<am; i++) {
    /* Step 2: Initialize the dense row vector for C  */
    const PetscInt anzi = ai[i+1] - ai[i]; /* number of nonzeros in this row of A, this is the number of rows of B that we merge */
    PetscInt       cnzi = 0;
    PetscInt       *bj_i;
    PetscScalar    *ba_i;
    /* If the number of indices in C so far + the max number of columns in the next row > c_maxmem  -> allocate more memory
       Usually, there is enough memory in the first place, so this is not executed. */
    while (ci[i] + b_maxmemrow > c_maxmem) {
      c_maxmem *= 2;
      ndouble++;
      ierr = PetscRealloc(sizeof(PetscInt)*c_maxmem,&cj);CHKERRQ(ierr);
      ierr = PetscRealloc(sizeof(PetscScalar)*c_maxmem,&ca);CHKERRQ(ierr);
    }

    /* Step 3: Do the numerical calculations */
    for (a_col=0; a_col<anzi; a_col++) {          /* iterate over all non zero values in a row of A */
      PetscInt       a_col_index = aj_i[a_col];
      const PetscInt bnzi        = bi[a_col_index+1] - bi[a_col_index];
      flops += 2*bnzi;
      bj_i   = b->j + bi[a_col_index];   /* points to the current row in bj */
      ba_i   = b->a + bi[a_col_index];   /* points to the current row in ba */
      for (k=0; k<bnzi; ++k) { /* iterate over all non zeros of this row in B */
        if (c_row_idx_flags[bj_i[k]] == PETSC_FALSE) {
          cj_i[cnzi++]             = bj_i[k];
          c_row_idx_flags[bj_i[k]] = PETSC_TRUE;
        }
        c_row_val_dense[bj_i[k]] += aa_i[a_col] * ba_i[k];
      }
    }

    /* Sort array */
    ierr = PetscSortInt(cnzi,cj_i);CHKERRQ(ierr);
    /* Step 4 */
    for (k=0; k<cnzi; k++) {
      ca_i[k]                  = c_row_val_dense[cj_i[k]];
      c_row_val_dense[cj_i[k]] = 0.;
      c_row_idx_flags[cj_i[k]] = PETSC_FALSE;
    }
    /* terminate current row */
    aa_i   += anzi;
    aj_i   += anzi;
    ca_i   += cnzi;
    cj_i   += cnzi;
    ci[i+1] = ci[i] + cnzi;
    flops  += cnzi;
  }

  /* Step 5 */
  /* Create the new matrix */
  ierr = MatCreateSeqAIJWithArrays(PetscObjectComm((PetscObject)A),am,bn,ci,cj,NULL,C);CHKERRQ(ierr);
  ierr = MatSetBlockSizesFromMats(*C,A,B);CHKERRQ(ierr);
  ierr = MatSetType(*C,((PetscObject)A)->type_name);CHKERRQ(ierr);

  /* MatCreateSeqAIJWithArrays flags matrix so PETSc doesn't free the user's arrays. */
  /* These are PETSc arrays, so change flags so arrays can be deleted by PETSc */
  c          = (Mat_SeqAIJ*)((*C)->data);
  c->a       = ca;
  c->free_a  = PETSC_TRUE;
  c->free_ij = PETSC_TRUE;
  c->nonew   = 0;

  /* set MatInfo */
  afill = (PetscReal)ci[am]/(ai[am]+bi[bm]) + 1.e-5;
  if (afill < 1.0) afill = 1.0;
  c->maxnz                     = ci[am];
  c->nz                        = ci[am];
  (*C)->info.mallocs           = ndouble;
  (*C)->info.fill_ratio_given  = fill;
  (*C)->info.fill_ratio_needed = afill;
  (*C)->ops->matmultnumeric    = MatMatMultNumeric_SeqAIJ_SeqAIJ_Combined;

  ierr = PetscFree(c_row_val_dense);CHKERRQ(ierr);
  ierr = PetscFree(c_row_idx_flags);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(*C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscLogFlops(flops);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
