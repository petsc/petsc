
/*
  Defines matrix-matrix product routines for pairs of SeqAIJ matrices
          C = A * B
*/

#include <../src/mat/impls/aij/seq/aij.h> /*I "petscmat.h" I*/
#include <../src/mat/utils/freespace.h>
#include <petscbt.h>
#include <petsc/private/isimpl.h>
#include <../src/mat/impls/dense/seq/dense.h>

PetscErrorCode MatMatMultNumeric_SeqAIJ_SeqAIJ(Mat A,Mat B,Mat C)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (C->ops->matmultnumeric) {
    if (C->ops->matmultnumeric == MatMatMultNumeric_SeqAIJ_SeqAIJ) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Recursive call");
    ierr = (*C->ops->matmultnumeric)(A,B,C);CHKERRQ(ierr);
  } else {
    ierr = MatMatMultNumeric_SeqAIJ_SeqAIJ_Sorted(A,B,C);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* Modified from MatCreateSeqAIJWithArrays() */
PETSC_INTERN PetscErrorCode MatSetSeqAIJWithArrays_private(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt i[],PetscInt j[],PetscScalar a[],MatType mtype,Mat mat)
{
  PetscErrorCode ierr;
  PetscInt       ii;
  Mat_SeqAIJ     *aij;
  PetscBool      isseqaij;

  PetscFunctionBegin;
  if (m > 0 && i[0]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"i (row indices) must start with 0");
  ierr = MatSetSizes(mat,m,n,m,n);CHKERRQ(ierr);

  if (!mtype) {
    ierr = PetscObjectBaseTypeCompare((PetscObject)mat,MATSEQAIJ,&isseqaij);CHKERRQ(ierr);
    if (!isseqaij) { ierr = MatSetType(mat,MATSEQAIJ);CHKERRQ(ierr); }
  } else {
    ierr = MatSetType(mat,mtype);CHKERRQ(ierr);
  }
  ierr = MatSeqAIJSetPreallocation_SeqAIJ(mat,MAT_SKIP_ALLOCATION,0);CHKERRQ(ierr);
  aij  = (Mat_SeqAIJ*)(mat)->data;
  ierr = PetscMalloc1(m,&aij->imax);CHKERRQ(ierr);
  ierr = PetscMalloc1(m,&aij->ilen);CHKERRQ(ierr);

  aij->i            = i;
  aij->j            = j;
  aij->a            = a;
  aij->singlemalloc = PETSC_FALSE;
  aij->nonew        = -1; /*this indicates that inserting a new value in the matrix that generates a new nonzero is an error*/
  aij->free_a       = PETSC_FALSE;
  aij->free_ij      = PETSC_FALSE;

  for (ii=0; ii<m; ii++) {
    aij->ilen[ii] = aij->imax[ii] = i[ii+1] - i[ii];
  }

  PetscFunctionReturn(0);
}

PetscErrorCode MatMatMultSymbolic_SeqAIJ_SeqAIJ(Mat A,Mat B,PetscReal fill,Mat C)
{
  PetscErrorCode      ierr;
  Mat_Product         *product = C->product;
  MatProductAlgorithm alg;
  PetscBool           flg;

  PetscFunctionBegin;
  if (product) {
    alg = product->alg;
  } else {
    alg = "sorted";
  }
  /* sorted */
  ierr = PetscStrcmp(alg,"sorted",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatMatMultSymbolic_SeqAIJ_SeqAIJ_Sorted(A,B,fill,C);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  /* scalable */
  ierr = PetscStrcmp(alg,"scalable",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatMatMultSymbolic_SeqAIJ_SeqAIJ_Scalable(A,B,fill,C);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  /* scalable_fast */
  ierr = PetscStrcmp(alg,"scalable_fast",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatMatMultSymbolic_SeqAIJ_SeqAIJ_Scalable_fast(A,B,fill,C);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  /* heap */
  ierr = PetscStrcmp(alg,"heap",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatMatMultSymbolic_SeqAIJ_SeqAIJ_Heap(A,B,fill,C);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  /* btheap */
  ierr = PetscStrcmp(alg,"btheap",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatMatMultSymbolic_SeqAIJ_SeqAIJ_BTHeap(A,B,fill,C);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  /* llcondensed */
  ierr = PetscStrcmp(alg,"llcondensed",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatMatMultSymbolic_SeqAIJ_SeqAIJ_LLCondensed(A,B,fill,C);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  /* rowmerge */
  ierr = PetscStrcmp(alg,"rowmerge",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatMatMultSymbolic_SeqAIJ_SeqAIJ_RowMerge(A,B,fill,C);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

#if defined(PETSC_HAVE_HYPRE)
  ierr = PetscStrcmp(alg,"hypre",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatMatMultSymbolic_AIJ_AIJ_wHYPRE(A,B,fill,C);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
#endif

  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Mat Product Algorithm is not supported");
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatMultSymbolic_SeqAIJ_SeqAIJ_LLCondensed(Mat A,Mat B,PetscReal fill,Mat C)
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
  ierr = MatSetSeqAIJWithArrays_private(PetscObjectComm((PetscObject)A),am,bn,ci,cj,NULL,((PetscObject)A)->type_name,C);CHKERRQ(ierr);
  ierr = MatSetBlockSizesFromMats(C,A,B);CHKERRQ(ierr);

  /* MatCreateSeqAIJWithArrays flags matrix so PETSc doesn't free the user's arrays. */
  /* These are PETSc arrays, so change flags so arrays can be deleted by PETSc */
  c          = (Mat_SeqAIJ*)(C->data);
  c->free_a  = PETSC_FALSE;
  c->free_ij = PETSC_TRUE;
  c->nonew   = 0;

  /* fast, needs non-scalable O(bn) array 'abdense' */
  C->ops->matmultnumeric = MatMatMultNumeric_SeqAIJ_SeqAIJ_Sorted;

  /* set MatInfo */
  afill = (PetscReal)ci[am]/(ai[am]+bi[bm]) + 1.e-5;
  if (afill < 1.0) afill = 1.0;
  c->maxnz                  = ci[am];
  c->nz                     = ci[am];
  C->info.mallocs           = ndouble;
  C->info.fill_ratio_given  = fill;
  C->info.fill_ratio_needed = afill;

#if defined(PETSC_USE_INFO)
  if (ci[am]) {
    ierr = PetscInfo3(C,"Reallocs %D; Fill ratio: given %g needed %g.\n",ndouble,(double)fill,(double)afill);CHKERRQ(ierr);
    ierr = PetscInfo1(C,"Use MatMatMult(A,B,MatReuse,%g,&C) for best performance.;\n",(double)afill);CHKERRQ(ierr);
  } else {
    ierr = PetscInfo(C,"Empty matrix product\n");CHKERRQ(ierr);
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
  PetscContainer cab_dense;

  PetscFunctionBegin;
  if (!c->a) { /* first call of MatMatMultNumeric_SeqAIJ_SeqAIJ, allocate ca and matmult_abdense */
    ierr      = PetscMalloc1(ci[cm]+1,&ca);CHKERRQ(ierr);
    c->a      = ca;
    c->free_a = PETSC_TRUE;
  } else ca = c->a;

  /* TODO this should be done in the symbolic phase */
  /* However, this function is so heavily used (sometimes in an hidden way through multnumeric function pointers
     that is hard to eradicate) */
  ierr = PetscObjectQuery((PetscObject)C,"__PETSc__ab_dense",(PetscObject*)&cab_dense);CHKERRQ(ierr);
  if (!cab_dense) {
    ierr = PetscMalloc1(B->cmap->N,&ab_dense);CHKERRQ(ierr);
    ierr = PetscContainerCreate(PETSC_COMM_SELF,&cab_dense);CHKERRQ(ierr);
    ierr = PetscContainerSetPointer(cab_dense,ab_dense);CHKERRQ(ierr);
    ierr = PetscContainerSetUserDestroy(cab_dense,PetscContainerUserDestroyDefault);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)C,"__PETSc__ab_dense",(PetscObject)cab_dense);CHKERRQ(ierr);
    ierr = PetscObjectDereference((PetscObject)cab_dense);CHKERRQ(ierr);
  }
  ierr = PetscContainerGetPointer(cab_dense,(void**)&ab_dense);CHKERRQ(ierr);
  ierr = PetscArrayzero(ab_dense,B->cmap->N);CHKERRQ(ierr);

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

PetscErrorCode MatMatMultSymbolic_SeqAIJ_SeqAIJ_Scalable_fast(Mat A,Mat B,PetscReal fill,Mat C)
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
  ierr = MatSetSeqAIJWithArrays_private(PetscObjectComm((PetscObject)A),am,bn,ci,cj,ca,((PetscObject)A)->type_name,C);CHKERRQ(ierr);
  ierr = MatSetBlockSizesFromMats(C,A,B);CHKERRQ(ierr);

  /* MatCreateSeqAIJWithArrays flags matrix so PETSc doesn't free the user's arrays. */
  /* These are PETSc arrays, so change flags so arrays can be deleted by PETSc */
  c          = (Mat_SeqAIJ*)(C->data);
  c->free_a  = PETSC_TRUE;
  c->free_ij = PETSC_TRUE;
  c->nonew   = 0;

  /* slower, less memory */
  C->ops->matmultnumeric = MatMatMultNumeric_SeqAIJ_SeqAIJ_Scalable;

  /* set MatInfo */
  afill = (PetscReal)ci[am]/(ai[am]+bi[bm]) + 1.e-5;
  if (afill < 1.0) afill = 1.0;
  c->maxnz                     = ci[am];
  c->nz                        = ci[am];
  C->info.mallocs           = ndouble;
  C->info.fill_ratio_given  = fill;
  C->info.fill_ratio_needed = afill;

#if defined(PETSC_USE_INFO)
  if (ci[am]) {
    ierr = PetscInfo3(C,"Reallocs %D; Fill ratio: given %g needed %g.\n",ndouble,(double)fill,(double)afill);CHKERRQ(ierr);
    ierr = PetscInfo1(C,"Use MatMatMult(A,B,MatReuse,%g,&C) for best performance.;\n",(double)afill);CHKERRQ(ierr);
  } else {
    ierr = PetscInfo(C,"Empty matrix product\n");CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatMultSymbolic_SeqAIJ_SeqAIJ_Scalable(Mat A,Mat B,PetscReal fill,Mat C)
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
  ierr = MatSetSeqAIJWithArrays_private(PetscObjectComm((PetscObject)A),am,bn,ci,cj,ca,((PetscObject)A)->type_name,C);CHKERRQ(ierr);
  ierr = MatSetBlockSizesFromMats(C,A,B);CHKERRQ(ierr);

  /* MatCreateSeqAIJWithArrays flags matrix so PETSc doesn't free the user's arrays. */
  /* These are PETSc arrays, so change flags so arrays can be deleted by PETSc */
  c          = (Mat_SeqAIJ*)(C->data);
  c->free_a  = PETSC_TRUE;
  c->free_ij = PETSC_TRUE;
  c->nonew   = 0;

  /* slower, less memory */
  C->ops->matmultnumeric = MatMatMultNumeric_SeqAIJ_SeqAIJ_Scalable;

  /* set MatInfo */
  afill = (PetscReal)ci[am]/(ai[am]+bi[bm]) + 1.e-5;
  if (afill < 1.0) afill = 1.0;
  c->maxnz                     = ci[am];
  c->nz                        = ci[am];
  C->info.mallocs           = ndouble;
  C->info.fill_ratio_given  = fill;
  C->info.fill_ratio_needed = afill;

#if defined(PETSC_USE_INFO)
  if (ci[am]) {
    ierr = PetscInfo3(C,"Reallocs %D; Fill ratio: given %g needed %g.\n",ndouble,(double)fill,(double)afill);CHKERRQ(ierr);
    ierr = PetscInfo1(C,"Use MatMatMult(A,B,MatReuse,%g,&C) for best performance.;\n",(double)afill);CHKERRQ(ierr);
  } else {
    ierr = PetscInfo(C,"Empty matrix product\n");CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatMultSymbolic_SeqAIJ_SeqAIJ_Heap(Mat A,Mat B,PetscReal fill,Mat C)
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
  ierr = MatSetSeqAIJWithArrays_private(PetscObjectComm((PetscObject)A),am,bn,ci,cj,NULL,((PetscObject)A)->type_name,C);CHKERRQ(ierr);
  ierr = MatSetBlockSizesFromMats(C,A,B);CHKERRQ(ierr);

  /* MatCreateSeqAIJWithArrays flags matrix so PETSc doesn't free the user's arrays. */
  /* These are PETSc arrays, so change flags so arrays can be deleted by PETSc */
  c          = (Mat_SeqAIJ*)(C->data);
  c->free_a  = PETSC_TRUE;
  c->free_ij = PETSC_TRUE;
  c->nonew   = 0;

  C->ops->matmultnumeric = MatMatMultNumeric_SeqAIJ_SeqAIJ_Sorted;

  /* set MatInfo */
  afill = (PetscReal)ci[am]/(ai[am]+bi[bm]) + 1.e-5;
  if (afill < 1.0) afill = 1.0;
  c->maxnz                     = ci[am];
  c->nz                        = ci[am];
  C->info.mallocs           = ndouble;
  C->info.fill_ratio_given  = fill;
  C->info.fill_ratio_needed = afill;

#if defined(PETSC_USE_INFO)
  if (ci[am]) {
    ierr = PetscInfo3(C,"Reallocs %D; Fill ratio: given %g needed %g.\n",ndouble,(double)fill,(double)afill);CHKERRQ(ierr);
    ierr = PetscInfo1(C,"Use MatMatMult(A,B,MatReuse,%g,&C) for best performance.;\n",(double)afill);CHKERRQ(ierr);
  } else {
    ierr = PetscInfo(C,"Empty matrix product\n");CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatMultSymbolic_SeqAIJ_SeqAIJ_BTHeap(Mat A,Mat B,PetscReal fill,Mat C)
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
  ierr = MatSetSeqAIJWithArrays_private(PetscObjectComm((PetscObject)A),am,bn,ci,cj,NULL,((PetscObject)A)->type_name,C);CHKERRQ(ierr);
  ierr = MatSetBlockSizesFromMats(C,A,B);CHKERRQ(ierr);

  /* MatCreateSeqAIJWithArrays flags matrix so PETSc doesn't free the user's arrays. */
  /* These are PETSc arrays, so change flags so arrays can be deleted by PETSc */
  c          = (Mat_SeqAIJ*)(C->data);
  c->free_a  = PETSC_TRUE;
  c->free_ij = PETSC_TRUE;
  c->nonew   = 0;

  C->ops->matmultnumeric = MatMatMultNumeric_SeqAIJ_SeqAIJ_Sorted;

  /* set MatInfo */
  afill = (PetscReal)ci[am]/(ai[am]+bi[bm]) + 1.e-5;
  if (afill < 1.0) afill = 1.0;
  c->maxnz                     = ci[am];
  c->nz                        = ci[am];
  C->info.mallocs           = ndouble;
  C->info.fill_ratio_given  = fill;
  C->info.fill_ratio_needed = afill;

#if defined(PETSC_USE_INFO)
  if (ci[am]) {
    ierr = PetscInfo3(C,"Reallocs %D; Fill ratio: given %g needed %g.\n",ndouble,(double)fill,(double)afill);CHKERRQ(ierr);
    ierr = PetscInfo1(C,"Use MatMatMult(A,B,MatReuse,%g,&C) for best performance.;\n",(double)afill);CHKERRQ(ierr);
  } else {
    ierr = PetscInfo(C,"Empty matrix product\n");CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}


PetscErrorCode MatMatMultSymbolic_SeqAIJ_SeqAIJ_RowMerge(Mat A,Mat B,PetscReal fill,Mat C)
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
  ierr = MatSetSeqAIJWithArrays_private(PetscObjectComm((PetscObject)A),am,bn,ci,cj,NULL,((PetscObject)A)->type_name,C);CHKERRQ(ierr);
  ierr = MatSetBlockSizesFromMats(C,A,B);CHKERRQ(ierr);

  /* MatCreateSeqAIJWithArrays flags matrix so PETSc doesn't free the user's arrays. */
  /* These are PETSc arrays, so change flags so arrays can be deleted by PETSc */
  c          = (Mat_SeqAIJ*)(C->data);
  c->free_a  = PETSC_TRUE;
  c->free_ij = PETSC_TRUE;
  c->nonew   = 0;

  C->ops->matmultnumeric = MatMatMultNumeric_SeqAIJ_SeqAIJ_Sorted;

  /* set MatInfo */
  afill = (PetscReal)ci[am]/(ai[am]+bi[bm]) + 1.e-5;
  if (afill < 1.0) afill = 1.0;
  c->maxnz                     = ci[am];
  c->nz                        = ci[am];
  C->info.mallocs           = ndouble;
  C->info.fill_ratio_given  = fill;
  C->info.fill_ratio_needed = afill;

#if defined(PETSC_USE_INFO)
  if (ci[am]) {
    ierr = PetscInfo3(C,"Reallocs %D; Fill ratio: given %g needed %g.\n",ndouble,(double)fill,(double)afill);CHKERRQ(ierr);
    ierr = PetscInfo1(C,"Use MatMatMult(A,B,MatReuse,%g,&C) for best performance.;\n",(double)afill);CHKERRQ(ierr);
  } else {
    ierr = PetscInfo(C,"Empty matrix product\n");CHKERRQ(ierr);
  }
#endif

  /* Step 4: Free temporary work areas */
  ierr = PetscFree(workj_L1);CHKERRQ(ierr);
  ierr = PetscFree(workj_L2);CHKERRQ(ierr);
  ierr = PetscFree(workj_L3);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* concatenate unique entries and then sort */
PetscErrorCode MatMatMultSymbolic_SeqAIJ_SeqAIJ_Sorted(Mat A,Mat B,PetscReal fill,Mat C)
{
  PetscErrorCode ierr;
  Mat_SeqAIJ     *a  = (Mat_SeqAIJ*)A->data,*b=(Mat_SeqAIJ*)B->data,*c;
  const PetscInt *ai = a->i,*bi=b->i,*aj=a->j,*bj=b->j;
  PetscInt       *ci,*cj;
  PetscInt       am=A->rmap->N,bn=B->cmap->N,bm=B->rmap->N;
  PetscReal      afill;
  PetscInt       i,j,ndouble = 0;
  PetscSegBuffer seg,segrow;
  char           *seen;

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
  ierr = MatSetSeqAIJWithArrays_private(PetscObjectComm((PetscObject)A),am,bn,ci,cj,NULL,((PetscObject)A)->type_name,C);CHKERRQ(ierr);
  ierr = MatSetBlockSizesFromMats(C,A,B);CHKERRQ(ierr);

  /* MatCreateSeqAIJWithArrays flags matrix so PETSc doesn't free the user's arrays. */
  /* These are PETSc arrays, so change flags so arrays can be deleted by PETSc */
  c          = (Mat_SeqAIJ*)(C->data);
  c->free_a  = PETSC_TRUE;
  c->free_ij = PETSC_TRUE;
  c->nonew   = 0;

  C->ops->matmultnumeric = MatMatMultNumeric_SeqAIJ_SeqAIJ_Sorted;

  /* set MatInfo */
  afill = (PetscReal)ci[am]/(ai[am]+bi[bm]) + 1.e-5;
  if (afill < 1.0) afill = 1.0;
  c->maxnz                     = ci[am];
  c->nz                        = ci[am];
  C->info.mallocs           = ndouble;
  C->info.fill_ratio_given  = fill;
  C->info.fill_ratio_needed = afill;

#if defined(PETSC_USE_INFO)
  if (ci[am]) {
    ierr = PetscInfo3(C,"Reallocs %D; Fill ratio: given %g needed %g.\n",ndouble,(double)fill,(double)afill);CHKERRQ(ierr);
    ierr = PetscInfo1(C,"Use MatMatMult(A,B,MatReuse,%g,&C) for best performance.;\n",(double)afill);CHKERRQ(ierr);
  } else {
    ierr = PetscInfo(C,"Empty matrix product\n");CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_SeqAIJ_MatMatMultTrans(void *data)
{
  PetscErrorCode      ierr;
  Mat_MatMatTransMult *abt=(Mat_MatMatTransMult *)data;

  PetscFunctionBegin;
  ierr = MatTransposeColoringDestroy(&abt->matcoloring);CHKERRQ(ierr);
  ierr = MatDestroy(&abt->Bt_den);CHKERRQ(ierr);
  ierr = MatDestroy(&abt->ABt_den);CHKERRQ(ierr);
  ierr = PetscFree(abt);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatTransposeMultSymbolic_SeqAIJ_SeqAIJ(Mat A,Mat B,PetscReal fill,Mat C)
{
  PetscErrorCode      ierr;
  Mat                 Bt;
  PetscInt            *bti,*btj;
  Mat_MatMatTransMult *abt;
  Mat_Product         *product = C->product;
  char                *alg;

  PetscFunctionBegin;
  if (!product) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing product struct");
  if (product->data) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Extra product struct not empty");

  /* create symbolic Bt */
  ierr = MatGetSymbolicTranspose_SeqAIJ(B,&bti,&btj);CHKERRQ(ierr);
  ierr = MatCreateSeqAIJWithArrays(PETSC_COMM_SELF,B->cmap->n,B->rmap->n,bti,btj,NULL,&Bt);CHKERRQ(ierr);
  ierr = MatSetBlockSizes(Bt,PetscAbs(A->cmap->bs),PetscAbs(B->cmap->bs));CHKERRQ(ierr);
  ierr = MatSetType(Bt,((PetscObject)A)->type_name);CHKERRQ(ierr);

  /* get symbolic C=A*Bt */
  ierr = PetscStrallocpy(product->alg,&alg);CHKERRQ(ierr);
  ierr = MatProductSetAlgorithm(C,"sorted");CHKERRQ(ierr); /* set algorithm for C = A*Bt */
  ierr = MatMatMultSymbolic_SeqAIJ_SeqAIJ(A,Bt,fill,C);CHKERRQ(ierr);
  ierr = MatProductSetAlgorithm(C,alg);CHKERRQ(ierr); /* resume original algorithm for ABt product */
  ierr = PetscFree(alg);CHKERRQ(ierr);

  /* create a supporting struct for reuse intermidiate dense matrices with matcoloring */
  ierr = PetscNew(&abt);CHKERRQ(ierr);

  product->data    = abt;
  product->destroy = MatDestroy_SeqAIJ_MatMatMultTrans;

  C->ops->mattransposemultnumeric = MatMatTransposeMultNumeric_SeqAIJ_SeqAIJ;

  abt->usecoloring = PETSC_FALSE;
  ierr = PetscStrcmp(product->alg,"color",&abt->usecoloring);CHKERRQ(ierr);
  if (abt->usecoloring) {
    /* Create MatTransposeColoring from symbolic C=A*B^T */
    MatTransposeColoring matcoloring;
    MatColoring          coloring;
    ISColoring           iscoloring;
    Mat                  Bt_dense,C_dense;

    /* inode causes memory problem */
    ierr = MatSetOption(C,MAT_USE_INODES,PETSC_FALSE);CHKERRQ(ierr);

    ierr = MatColoringCreate(C,&coloring);CHKERRQ(ierr);
    ierr = MatColoringSetDistance(coloring,2);CHKERRQ(ierr);
    ierr = MatColoringSetType(coloring,MATCOLORINGSL);CHKERRQ(ierr);
    ierr = MatColoringSetFromOptions(coloring);CHKERRQ(ierr);
    ierr = MatColoringApply(coloring,&iscoloring);CHKERRQ(ierr);
    ierr = MatColoringDestroy(&coloring);CHKERRQ(ierr);
    ierr = MatTransposeColoringCreate(C,iscoloring,&matcoloring);CHKERRQ(ierr);

    abt->matcoloring = matcoloring;

    ierr = ISColoringDestroy(&iscoloring);CHKERRQ(ierr);

    /* Create Bt_dense and C_dense = A*Bt_dense */
    ierr = MatCreate(PETSC_COMM_SELF,&Bt_dense);CHKERRQ(ierr);
    ierr = MatSetSizes(Bt_dense,A->cmap->n,matcoloring->ncolors,A->cmap->n,matcoloring->ncolors);CHKERRQ(ierr);
    ierr = MatSetType(Bt_dense,MATSEQDENSE);CHKERRQ(ierr);
    ierr = MatSeqDenseSetPreallocation(Bt_dense,NULL);CHKERRQ(ierr);

    Bt_dense->assembled = PETSC_TRUE;
    abt->Bt_den         = Bt_dense;

    ierr = MatCreate(PETSC_COMM_SELF,&C_dense);CHKERRQ(ierr);
    ierr = MatSetSizes(C_dense,A->rmap->n,matcoloring->ncolors,A->rmap->n,matcoloring->ncolors);CHKERRQ(ierr);
    ierr = MatSetType(C_dense,MATSEQDENSE);CHKERRQ(ierr);
    ierr = MatSeqDenseSetPreallocation(C_dense,NULL);CHKERRQ(ierr);

    Bt_dense->assembled = PETSC_TRUE;
    abt->ABt_den  = C_dense;

#if defined(PETSC_USE_INFO)
    {
      Mat_SeqAIJ *c = (Mat_SeqAIJ*)C->data;
      ierr = PetscInfo7(C,"Use coloring of C=A*B^T; B^T: %D %D, Bt_dense: %D,%D; Cnz %D / (cm*ncolors %D) = %g\n",B->cmap->n,B->rmap->n,Bt_dense->rmap->n,Bt_dense->cmap->n,c->nz,A->rmap->n*matcoloring->ncolors,(PetscReal)(c->nz)/(A->rmap->n*matcoloring->ncolors));CHKERRQ(ierr);
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
  Mat_MatMatTransMult *abt;
  Mat_Product         *product = C->product;

  PetscFunctionBegin;
  if (!product) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing product struct");
  abt = (Mat_MatMatTransMult *)product->data;
  if (!abt) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing product struct");
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

PetscErrorCode MatDestroy_SeqAIJ_MatTransMatMult(void *data)
{
  PetscErrorCode      ierr;
  Mat_MatTransMatMult *atb = (Mat_MatTransMatMult*)data;

  PetscFunctionBegin;
  ierr = MatDestroy(&atb->At);CHKERRQ(ierr);
  if (atb->destroy) {
    ierr = (*atb->destroy)(atb->data);CHKERRQ(ierr);
  }
  ierr = PetscFree(atb);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatTransposeMatMultSymbolic_SeqAIJ_SeqAIJ(Mat A,Mat B,PetscReal fill,Mat C)
{
  PetscErrorCode      ierr;
  Mat                 At;
  PetscInt            *ati,*atj;
  Mat_Product         *product = C->product;
  MatProductAlgorithm alg;
  PetscBool           flg;

  PetscFunctionBegin;
  if (product) {
    alg = product->alg;
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"!product, not supported yet");

  /* outerproduct */
  ierr = PetscStrcmp(alg,"outerproduct",&flg);CHKERRQ(ierr);
  if (flg) {
    /* create symbolic At */
    ierr = MatGetSymbolicTranspose_SeqAIJ(A,&ati,&atj);CHKERRQ(ierr);
    ierr = MatCreateSeqAIJWithArrays(PETSC_COMM_SELF,A->cmap->n,A->rmap->n,ati,atj,NULL,&At);CHKERRQ(ierr);
    ierr = MatSetBlockSizes(At,PetscAbs(A->cmap->bs),PetscAbs(B->cmap->bs));CHKERRQ(ierr);
    ierr = MatSetType(At,((PetscObject)A)->type_name);CHKERRQ(ierr);

    /* get symbolic C=At*B */
    ierr = MatProductSetAlgorithm(C,"sorted");CHKERRQ(ierr);
    ierr = MatMatMultSymbolic_SeqAIJ_SeqAIJ(At,B,fill,C);CHKERRQ(ierr);

    /* clean up */
    ierr = MatDestroy(&At);CHKERRQ(ierr);
    ierr = MatRestoreSymbolicTranspose_SeqAIJ(A,&ati,&atj);CHKERRQ(ierr);

    C->ops->mattransposemultnumeric = MatTransposeMatMultNumeric_SeqAIJ_SeqAIJ; /* outerproduct */
    ierr = MatProductSetAlgorithm(C,"outerproduct");CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  /* matmatmult */
  ierr = PetscStrcmp(alg,"at*b",&flg);CHKERRQ(ierr);
  if (flg) {
    Mat_MatTransMatMult *atb;

    if (product->data) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Extra product struct not empty");
    ierr = PetscNew(&atb);CHKERRQ(ierr);
    ierr = MatTranspose_SeqAIJ(A,MAT_INITIAL_MATRIX,&At);CHKERRQ(ierr);
    ierr = MatProductSetAlgorithm(C,"sorted");CHKERRQ(ierr);
    ierr = MatMatMultSymbolic_SeqAIJ_SeqAIJ(At,B,fill,C);CHKERRQ(ierr);
    ierr = MatProductSetAlgorithm(C,"at*b");CHKERRQ(ierr);
    product->data    = atb;
    product->destroy = MatDestroy_SeqAIJ_MatTransMatMult;
    atb->At          = At;
    atb->updateAt    = PETSC_FALSE; /* because At is computed here */

    C->ops->mattransposemultnumeric = NULL; /* see MatProductNumeric_AtB_SeqAIJ_SeqAIJ */
    PetscFunctionReturn(0);
  }

  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Mat Product Algorithm is not supported");
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

PetscErrorCode MatMatMultSymbolic_SeqAIJ_SeqDense(Mat A,Mat B,PetscReal fill,Mat C)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMatMultSymbolic_SeqDense_SeqDense(A,B,0.0,C);CHKERRQ(ierr);

  C->ops->matmultnumeric = MatMatMultNumeric_SeqAIJ_SeqDense;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatMatMultNumericAdd_SeqAIJ_SeqDense(Mat A,Mat B,Mat C)
{
  Mat_SeqAIJ        *a=(Mat_SeqAIJ*)A->data;
  Mat_SeqDense      *bd=(Mat_SeqDense*)B->data;
  Mat_SeqDense      *cd=(Mat_SeqDense*)C->data;
  PetscErrorCode    ierr;
  PetscScalar       *c,r1,r2,r3,r4,*c1,*c2,*c3,*c4;
  const PetscScalar *aa,*b,*b1,*b2,*b3,*b4,*av;
  const PetscInt    *aj;
  PetscInt          cm=C->rmap->n,cn=B->cmap->n,bm=bd->lda,am=A->rmap->n;
  PetscInt          clda=cd->lda;
  PetscInt          am4=4*clda,bm4=4*bm,col,i,j,n;

  PetscFunctionBegin;
  if (!cm || !cn) PetscFunctionReturn(0);
  ierr = MatSeqAIJGetArrayRead(A,&av);CHKERRQ(ierr);
  ierr = MatDenseGetArray(C,&c);CHKERRQ(ierr);
  ierr = MatDenseGetArrayRead(B,&b);CHKERRQ(ierr);
  b1 = b; b2 = b1 + bm; b3 = b2 + bm; b4 = b3 + bm;
  c1 = c; c2 = c1 + clda; c3 = c2 + clda; c4 = c3 + clda;
  for (col=0; col<(cn/4)*4; col += 4) {  /* over columns of C */
    for (i=0; i<am; i++) {        /* over rows of A in those columns */
      r1 = r2 = r3 = r4 = 0.0;
      n  = a->i[i+1] - a->i[i];
      aj = a->j + a->i[i];
      aa = av + a->i[i];
      for (j=0; j<n; j++) {
        const PetscScalar aatmp = aa[j];
        const PetscInt    ajtmp = aj[j];
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
    c1 += clda;
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

/* ------------------------------------------------------- */
static PetscErrorCode MatProductSetFromOptions_SeqAIJ_SeqDense_AB(Mat C)
{
  PetscFunctionBegin;
  C->ops->matmultsymbolic = MatMatMultSymbolic_SeqAIJ_SeqDense;
  C->ops->productsymbolic = MatProductSymbolic_AB;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatTMatTMultSymbolic_SeqAIJ_SeqDense(Mat,Mat,PetscReal,Mat);

static PetscErrorCode MatProductSetFromOptions_SeqAIJ_SeqDense_AtB(Mat C)
{
  PetscFunctionBegin;
  C->ops->transposematmultsymbolic = MatTMatTMultSymbolic_SeqAIJ_SeqDense;
  C->ops->productsymbolic          = MatProductSymbolic_AtB;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductSetFromOptions_SeqAIJ_SeqDense_ABt(Mat C)
{
  PetscFunctionBegin;
  C->ops->mattransposemultsymbolic = MatTMatTMultSymbolic_SeqAIJ_SeqDense;
  C->ops->productsymbolic          = MatProductSymbolic_ABt;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatProductSetFromOptions_SeqAIJ_SeqDense(Mat C)
{
  PetscErrorCode ierr;
  Mat_Product    *product = C->product;

  PetscFunctionBegin;
  switch (product->type) {
  case MATPRODUCT_AB:
    ierr = MatProductSetFromOptions_SeqAIJ_SeqDense_AB(C);CHKERRQ(ierr);
    break;
  case MATPRODUCT_AtB:
    ierr = MatProductSetFromOptions_SeqAIJ_SeqDense_AtB(C);CHKERRQ(ierr);
    break;
  case MATPRODUCT_ABt:
    ierr = MatProductSetFromOptions_SeqAIJ_SeqDense_ABt(C);CHKERRQ(ierr);
    break;
  default:
    break;
  }
  PetscFunctionReturn(0);
}
/* ------------------------------------------------------- */
static PetscErrorCode MatProductSetFromOptions_SeqXBAIJ_SeqDense_AB(Mat C)
{
  PetscErrorCode ierr;
  Mat_Product    *product = C->product;
  Mat            A = product->A;
  PetscBool      baij;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQBAIJ,&baij);CHKERRQ(ierr);
  if (!baij) { /* A is seqsbaij */
    PetscBool sbaij;
    ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQSBAIJ,&sbaij);CHKERRQ(ierr);
    if (!sbaij) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_ARG_WRONGSTATE,"Mat must be either seqbaij or seqsbaij format");

    C->ops->matmultsymbolic = MatMatMultSymbolic_SeqSBAIJ_SeqDense;
  } else { /* A is seqbaij */
    C->ops->matmultsymbolic = MatMatMultSymbolic_SeqBAIJ_SeqDense;
  }

  C->ops->productsymbolic = MatProductSymbolic_AB;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatProductSetFromOptions_SeqXBAIJ_SeqDense(Mat C)
{
  PetscErrorCode ierr;
  Mat_Product    *product = C->product;

  PetscFunctionBegin;
  MatCheckProduct(C,1);
  if (!product->A) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing A");
  if (product->type == MATPRODUCT_AB || (product->type == MATPRODUCT_AtB && product->A->symmetric)) {
    ierr = MatProductSetFromOptions_SeqXBAIJ_SeqDense_AB(C);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------- */
static PetscErrorCode MatProductSetFromOptions_SeqDense_SeqAIJ_AB(Mat C)
{
  PetscFunctionBegin;
  C->ops->matmultsymbolic = MatMatMultSymbolic_SeqDense_SeqAIJ;
  C->ops->productsymbolic = MatProductSymbolic_AB;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatProductSetFromOptions_SeqDense_SeqAIJ(Mat C)
{
  PetscErrorCode ierr;
  Mat_Product    *product = C->product;

  PetscFunctionBegin;
  if (product->type == MATPRODUCT_AB) {
    ierr = MatProductSetFromOptions_SeqDense_SeqAIJ_AB(C);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
/* ------------------------------------------------------- */

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

/* --------------------------------------------------------------- */
static PetscErrorCode MatProductNumeric_AtB_SeqAIJ_SeqAIJ(Mat C)
{
  PetscErrorCode ierr;
  Mat_Product    *product = C->product;
  Mat            A=product->A,B=product->B;

  PetscFunctionBegin;
  if (C->ops->mattransposemultnumeric) {
    /* Alg: "outerproduct" */
    ierr = (*C->ops->mattransposemultnumeric)(A,B,C);CHKERRQ(ierr);
  } else {
    /* Alg: "matmatmult" -- C = At*B */
    Mat_MatTransMatMult *atb = (Mat_MatTransMatMult *)product->data;
    Mat                 At;

    if (!atb) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing product struct");
    At = atb->At;
    if (atb->updateAt) { /* At is computed in MatTransposeMatMultSymbolic_SeqAIJ_SeqAIJ() */
      ierr = MatTranspose_SeqAIJ(A,MAT_REUSE_MATRIX,&At);CHKERRQ(ierr);
    }
    ierr = MatMatMultNumeric_SeqAIJ_SeqAIJ(At,B,C);CHKERRQ(ierr);
    atb->updateAt = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductSymbolic_AtB_SeqAIJ_SeqAIJ(Mat C)
{
  PetscErrorCode ierr;
  Mat_Product    *product = C->product;
  Mat            A=product->A,B=product->B;
  PetscReal      fill=product->fill;

  PetscFunctionBegin;
  ierr = MatTransposeMatMultSymbolic_SeqAIJ_SeqAIJ(A,B,fill,C);CHKERRQ(ierr);

  C->ops->productnumeric = MatProductNumeric_AtB_SeqAIJ_SeqAIJ;
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------- */
static PetscErrorCode MatProductSetFromOptions_SeqAIJ_AB(Mat C)
{
  PetscErrorCode ierr;
  Mat_Product    *product = C->product;
  PetscInt       alg = 0; /* default algorithm */
  PetscBool      flg = PETSC_FALSE;
#if !defined(PETSC_HAVE_HYPRE)
  const char     *algTypes[7] = {"sorted","scalable","scalable_fast","heap","btheap","llcondensed","rowmerge"};
  PetscInt       nalg = 7;
#else
  const char     *algTypes[8] = {"sorted","scalable","scalable_fast","heap","btheap","llcondensed","rowmerge","hypre"};
  PetscInt       nalg = 8;
#endif

  PetscFunctionBegin;
  /* Set default algorithm */
  ierr = PetscStrcmp(C->product->alg,"default",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatProductSetAlgorithm(C,(MatProductAlgorithm)algTypes[alg]);CHKERRQ(ierr);
  }

  /* Get runtime option */
  if (product->api_user) {
    ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)C),((PetscObject)C)->prefix,"MatMatMult","Mat");CHKERRQ(ierr);
    ierr = PetscOptionsEList("-matmatmult_via","Algorithmic approach","MatMatMult",algTypes,nalg,algTypes[0],&alg,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
  } else {
    ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)C),((PetscObject)C)->prefix,"MatProduct_AB","Mat");CHKERRQ(ierr);
    ierr = PetscOptionsEList("-matproduct_ab_via","Algorithmic approach","MatProduct_AB",algTypes,nalg,algTypes[0],&alg,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
  }
  if (flg) {
    ierr = MatProductSetAlgorithm(C,(MatProductAlgorithm)algTypes[alg]);CHKERRQ(ierr);
  }

  C->ops->productsymbolic = MatProductSymbolic_AB;
  C->ops->matmultsymbolic = MatMatMultSymbolic_SeqAIJ_SeqAIJ;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductSetFromOptions_SeqAIJ_AtB(Mat C)
{
  PetscErrorCode ierr;
  Mat_Product    *product = C->product;
  PetscInt       alg = 0; /* default algorithm */
  PetscBool      flg = PETSC_FALSE;
  const char     *algTypes[2] = {"at*b","outerproduct"};
  PetscInt       nalg = 2;

  PetscFunctionBegin;
  /* Set default algorithm */
  ierr = PetscStrcmp(product->alg,"default",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatProductSetAlgorithm(C,(MatProductAlgorithm)algTypes[alg]);CHKERRQ(ierr);
  }

  /* Get runtime option */
  if (product->api_user) {
    ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)C),((PetscObject)C)->prefix,"MatTransposeMatMult","Mat");CHKERRQ(ierr);
    ierr = PetscOptionsEList("-mattransposematmult_via","Algorithmic approach","MatTransposeMatMult",algTypes,nalg,algTypes[alg],&alg,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
  } else {
    ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)C),((PetscObject)C)->prefix,"MatProduct_AtB","Mat");CHKERRQ(ierr);
    ierr = PetscOptionsEList("-matproduct_atb_via","Algorithmic approach","MatProduct_AtB",algTypes,nalg,algTypes[alg],&alg,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
  }
  if (flg) {
    ierr = MatProductSetAlgorithm(C,(MatProductAlgorithm)algTypes[alg]);CHKERRQ(ierr);
  }

  C->ops->productsymbolic = MatProductSymbolic_AtB_SeqAIJ_SeqAIJ;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductSetFromOptions_SeqAIJ_ABt(Mat C)
{
  PetscErrorCode ierr;
  Mat_Product    *product = C->product;
  PetscInt       alg = 0; /* default algorithm */
  PetscBool      flg = PETSC_FALSE;
  const char     *algTypes[2] = {"default","color"};
  PetscInt       nalg = 2;

  PetscFunctionBegin;
  /* Set default algorithm */
  ierr = PetscStrcmp(C->product->alg,"default",&flg);CHKERRQ(ierr);
  if (!flg) {
    alg = 1;
    ierr = MatProductSetAlgorithm(C,(MatProductAlgorithm)algTypes[alg]);CHKERRQ(ierr);
  }

  /* Get runtime option */
  if (product->api_user) {
    ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)C),((PetscObject)C)->prefix,"MatMatTransposeMult","Mat");CHKERRQ(ierr);
    ierr = PetscOptionsEList("-matmattransmult_via","Algorithmic approach","MatMatTransposeMult",algTypes,nalg,algTypes[alg],&alg,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
  } else {
    ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)C),((PetscObject)C)->prefix,"MatProduct_ABt","Mat");CHKERRQ(ierr);
    ierr = PetscOptionsEList("-matproduct_abt_via","Algorithmic approach","MatProduct_ABt",algTypes,nalg,algTypes[alg],&alg,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
  }
  if (flg) {
    ierr = MatProductSetAlgorithm(C,(MatProductAlgorithm)algTypes[alg]);CHKERRQ(ierr);
  }

  C->ops->mattransposemultsymbolic = MatMatTransposeMultSymbolic_SeqAIJ_SeqAIJ;
  C->ops->productsymbolic          = MatProductSymbolic_ABt;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductSetFromOptions_SeqAIJ_PtAP(Mat C)
{
  PetscErrorCode ierr;
  Mat_Product    *product = C->product;
  PetscBool      flg = PETSC_FALSE;
  PetscInt       alg = 0; /* default algorithm -- alg=1 should be default!!! */
#if !defined(PETSC_HAVE_HYPRE)
  const char      *algTypes[2] = {"scalable","rap"};
  PetscInt        nalg = 2;
#else
  const char      *algTypes[3] = {"scalable","rap","hypre"};
  PetscInt        nalg = 3;
#endif

  PetscFunctionBegin;
  /* Set default algorithm */
  ierr = PetscStrcmp(product->alg,"default",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatProductSetAlgorithm(C,(MatProductAlgorithm)algTypes[alg]);CHKERRQ(ierr);
  }

  /* Get runtime option */
  if (product->api_user) {
    ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)C),((PetscObject)C)->prefix,"MatPtAP","Mat");CHKERRQ(ierr);
    ierr = PetscOptionsEList("-matptap_via","Algorithmic approach","MatPtAP",algTypes,nalg,algTypes[0],&alg,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
  } else {
    ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)C),((PetscObject)C)->prefix,"MatProduct_PtAP","Mat");CHKERRQ(ierr);
    ierr = PetscOptionsEList("-matproduct_ptap_via","Algorithmic approach","MatProduct_PtAP",algTypes,nalg,algTypes[0],&alg,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
  }
  if (flg) {
    ierr = MatProductSetAlgorithm(C,(MatProductAlgorithm)algTypes[alg]);CHKERRQ(ierr);
  }

  C->ops->productsymbolic = MatProductSymbolic_PtAP_SeqAIJ_SeqAIJ;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductSetFromOptions_SeqAIJ_RARt(Mat C)
{
  PetscErrorCode ierr;
  Mat_Product    *product = C->product;
  PetscBool      flg = PETSC_FALSE;
  PetscInt       alg = 0; /* default algorithm */
  const char     *algTypes[3] = {"r*a*rt","r*art","coloring_rart"};
  PetscInt        nalg = 3;

  PetscFunctionBegin;
  /* Set default algorithm */
  ierr = PetscStrcmp(product->alg,"default",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatProductSetAlgorithm(C,(MatProductAlgorithm)algTypes[alg]);CHKERRQ(ierr);
  }

  /* Get runtime option */
  if (product->api_user) {
    ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)C),((PetscObject)C)->prefix,"MatRARt","Mat");CHKERRQ(ierr);
    ierr = PetscOptionsEList("-matrart_via","Algorithmic approach","MatRARt",algTypes,nalg,algTypes[0],&alg,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
  } else {
    ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)C),((PetscObject)C)->prefix,"MatProduct_RARt","Mat");CHKERRQ(ierr);
    ierr = PetscOptionsEList("-matproduct_rart_via","Algorithmic approach","MatProduct_RARt",algTypes,nalg,algTypes[0],&alg,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
  }
  if (flg) {
    ierr = MatProductSetAlgorithm(C,(MatProductAlgorithm)algTypes[alg]);CHKERRQ(ierr);
  }

  C->ops->productsymbolic = MatProductSymbolic_RARt_SeqAIJ_SeqAIJ;
  PetscFunctionReturn(0);
}

/* ABC = A*B*C = A*(B*C); ABC's algorithm must be chosen from AB's algorithm */
static PetscErrorCode MatProductSetFromOptions_SeqAIJ_ABC(Mat C)
{
  PetscErrorCode ierr;
  Mat_Product    *product = C->product;
  PetscInt       alg = 0; /* default algorithm */
  PetscBool      flg = PETSC_FALSE;
  const char     *algTypes[7] = {"sorted","scalable","scalable_fast","heap","btheap","llcondensed","rowmerge"};
  PetscInt       nalg = 7;

  PetscFunctionBegin;
  /* Set default algorithm */
  ierr = PetscStrcmp(product->alg,"default",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatProductSetAlgorithm(C,(MatProductAlgorithm)algTypes[alg]);CHKERRQ(ierr);
  }

  /* Get runtime option */
  if (product->api_user) {
    ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)C),((PetscObject)C)->prefix,"MatMatMatMult","Mat");CHKERRQ(ierr);
    ierr = PetscOptionsEList("-matmatmatmult_via","Algorithmic approach","MatMatMatMult",algTypes,nalg,algTypes[alg],&alg,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
  } else {
    ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)C),((PetscObject)C)->prefix,"MatProduct_ABC","Mat");CHKERRQ(ierr);
    ierr = PetscOptionsEList("-matproduct_abc_via","Algorithmic approach","MatProduct_ABC",algTypes,nalg,algTypes[alg],&alg,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
  }
  if (flg) {
    ierr = MatProductSetAlgorithm(C,(MatProductAlgorithm)algTypes[alg]);CHKERRQ(ierr);
  }

  C->ops->matmatmultsymbolic = MatMatMatMultSymbolic_SeqAIJ_SeqAIJ_SeqAIJ;
  C->ops->productsymbolic    = MatProductSymbolic_ABC;
  PetscFunctionReturn(0);
}

PetscErrorCode MatProductSetFromOptions_SeqAIJ(Mat C)
{
  PetscErrorCode ierr;
  Mat_Product    *product = C->product;

  PetscFunctionBegin;
  switch (product->type) {
  case MATPRODUCT_AB:
    ierr = MatProductSetFromOptions_SeqAIJ_AB(C);CHKERRQ(ierr);
    break;
  case MATPRODUCT_AtB:
    ierr = MatProductSetFromOptions_SeqAIJ_AtB(C);CHKERRQ(ierr);
    break;
  case MATPRODUCT_ABt:
    ierr = MatProductSetFromOptions_SeqAIJ_ABt(C);CHKERRQ(ierr);
    break;
  case MATPRODUCT_PtAP:
    ierr = MatProductSetFromOptions_SeqAIJ_PtAP(C);CHKERRQ(ierr);
    break;
  case MATPRODUCT_RARt:
    ierr = MatProductSetFromOptions_SeqAIJ_RARt(C);CHKERRQ(ierr);
    break;
  case MATPRODUCT_ABC:
    ierr = MatProductSetFromOptions_SeqAIJ_ABC(C);CHKERRQ(ierr);
    break;
  default:
    break;
  }
  PetscFunctionReturn(0);
}
