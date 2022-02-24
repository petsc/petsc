
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
  PetscFunctionBegin;
  if (C->ops->matmultnumeric) {
    PetscCheckFalse(C->ops->matmultnumeric == MatMatMultNumeric_SeqAIJ_SeqAIJ,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Recursive call");
    CHKERRQ((*C->ops->matmultnumeric)(A,B,C));
  } else {
    CHKERRQ(MatMatMultNumeric_SeqAIJ_SeqAIJ_Sorted(A,B,C));
  }
  PetscFunctionReturn(0);
}

/* Modified from MatCreateSeqAIJWithArrays() */
PETSC_INTERN PetscErrorCode MatSetSeqAIJWithArrays_private(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt i[],PetscInt j[],PetscScalar a[],MatType mtype,Mat mat)
{
  PetscInt       ii;
  Mat_SeqAIJ     *aij;
  PetscBool      isseqaij, osingle, ofree_a, ofree_ij;

  PetscFunctionBegin;
  PetscCheckFalse(m > 0 && i[0],PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"i (row indices) must start with 0");
  CHKERRQ(MatSetSizes(mat,m,n,m,n));

  if (!mtype) {
    CHKERRQ(PetscObjectBaseTypeCompare((PetscObject)mat,MATSEQAIJ,&isseqaij));
    if (!isseqaij) CHKERRQ(MatSetType(mat,MATSEQAIJ));
  } else {
    CHKERRQ(MatSetType(mat,mtype));
  }

  aij  = (Mat_SeqAIJ*)(mat)->data;
  osingle = aij->singlemalloc;
  ofree_a = aij->free_a;
  ofree_ij = aij->free_ij;
  /* changes the free flags */
  CHKERRQ(MatSeqAIJSetPreallocation_SeqAIJ(mat,MAT_SKIP_ALLOCATION,NULL));

  CHKERRQ(PetscFree(aij->ilen));
  CHKERRQ(PetscFree(aij->imax));
  CHKERRQ(PetscMalloc1(m,&aij->imax));
  CHKERRQ(PetscMalloc1(m,&aij->ilen));
  for (ii=0,aij->nonzerorowcnt=0,aij->rmax = 0; ii<m; ii++) {
    const PetscInt rnz = i[ii+1] - i[ii];
    aij->nonzerorowcnt += !!rnz;
    aij->rmax = PetscMax(aij->rmax,rnz);
    aij->ilen[ii] = aij->imax[ii] = i[ii+1] - i[ii];
  }
  aij->maxnz = i[m];
  aij->nz = i[m];

  if (osingle) {
    CHKERRQ(PetscFree3(aij->a,aij->j,aij->i));
  } else {
    if (ofree_a)  CHKERRQ(PetscFree(aij->a));
    if (ofree_ij) CHKERRQ(PetscFree(aij->j));
    if (ofree_ij) CHKERRQ(PetscFree(aij->i));
  }
  aij->i            = i;
  aij->j            = j;
  aij->a            = a;
  aij->nonew        = -1; /* this indicates that inserting a new value in the matrix that generates a new nonzero is an error */
  /* default to not retain ownership */
  aij->singlemalloc = PETSC_FALSE;
  aij->free_a       = PETSC_FALSE;
  aij->free_ij      = PETSC_FALSE;
  CHKERRQ(MatCheckCompressedRow(mat,aij->nonzerorowcnt,&aij->compressedrow,aij->i,m,0.6));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatMultSymbolic_SeqAIJ_SeqAIJ(Mat A,Mat B,PetscReal fill,Mat C)
{
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
  CHKERRQ(PetscStrcmp(alg,"sorted",&flg));
  if (flg) {
    CHKERRQ(MatMatMultSymbolic_SeqAIJ_SeqAIJ_Sorted(A,B,fill,C));
    PetscFunctionReturn(0);
  }

  /* scalable */
  CHKERRQ(PetscStrcmp(alg,"scalable",&flg));
  if (flg) {
    CHKERRQ(MatMatMultSymbolic_SeqAIJ_SeqAIJ_Scalable(A,B,fill,C));
    PetscFunctionReturn(0);
  }

  /* scalable_fast */
  CHKERRQ(PetscStrcmp(alg,"scalable_fast",&flg));
  if (flg) {
    CHKERRQ(MatMatMultSymbolic_SeqAIJ_SeqAIJ_Scalable_fast(A,B,fill,C));
    PetscFunctionReturn(0);
  }

  /* heap */
  CHKERRQ(PetscStrcmp(alg,"heap",&flg));
  if (flg) {
    CHKERRQ(MatMatMultSymbolic_SeqAIJ_SeqAIJ_Heap(A,B,fill,C));
    PetscFunctionReturn(0);
  }

  /* btheap */
  CHKERRQ(PetscStrcmp(alg,"btheap",&flg));
  if (flg) {
    CHKERRQ(MatMatMultSymbolic_SeqAIJ_SeqAIJ_BTHeap(A,B,fill,C));
    PetscFunctionReturn(0);
  }

  /* llcondensed */
  CHKERRQ(PetscStrcmp(alg,"llcondensed",&flg));
  if (flg) {
    CHKERRQ(MatMatMultSymbolic_SeqAIJ_SeqAIJ_LLCondensed(A,B,fill,C));
    PetscFunctionReturn(0);
  }

  /* rowmerge */
  CHKERRQ(PetscStrcmp(alg,"rowmerge",&flg));
  if (flg) {
    CHKERRQ(MatMatMultSymbolic_SeqAIJ_SeqAIJ_RowMerge(A,B,fill,C));
    PetscFunctionReturn(0);
  }

#if defined(PETSC_HAVE_HYPRE)
  CHKERRQ(PetscStrcmp(alg,"hypre",&flg));
  if (flg) {
    CHKERRQ(MatMatMultSymbolic_AIJ_AIJ_wHYPRE(A,B,fill,C));
    PetscFunctionReturn(0);
  }
#endif

  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Mat Product Algorithm is not supported");
}

PetscErrorCode MatMatMultSymbolic_SeqAIJ_SeqAIJ_LLCondensed(Mat A,Mat B,PetscReal fill,Mat C)
{
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
  CHKERRQ(PetscMalloc1(am+2,&ci));
  ci[0] = 0;

  /* create and initialize a linked list */
  CHKERRQ(PetscTableCreate(bn,bn,&ta));
  MatRowMergeMax_SeqAIJ(b,bm,ta);
  CHKERRQ(PetscTableGetCount(ta,&Crmax));
  CHKERRQ(PetscTableDestroy(&ta));

  CHKERRQ(PetscLLCondensedCreate(Crmax,bn,&lnk,&lnkbt));

  /* Initial FreeSpace size is fill*(nnz(A)+nnz(B)) */
  CHKERRQ(PetscFreeSpaceGet(PetscRealIntMultTruncate(fill,PetscIntSumTruncate(ai[am],bi[bm])),&free_space));

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
      CHKERRQ(PetscLLCondensedAddSorted(bnzj,bj,lnk,lnkbt));
    }
    /* add possible missing diagonal entry */
    if (C->force_diagonals) {
      CHKERRQ(PetscLLCondensedAddSorted(1,&i,lnk,lnkbt));
    }
    cnzi = lnk[0];

    /* If free space is not available, make more free space */
    /* Double the amount of total space in the list */
    if (current_space->local_remaining<cnzi) {
      CHKERRQ(PetscFreeSpaceGet(PetscIntSumTruncate(cnzi,current_space->total_array_size),&current_space));
      ndouble++;
    }

    /* Copy data into free space, then initialize lnk */
    CHKERRQ(PetscLLCondensedClean(bn,cnzi,current_space->array,lnk,lnkbt));

    current_space->array           += cnzi;
    current_space->local_used      += cnzi;
    current_space->local_remaining -= cnzi;

    ci[i+1] = ci[i] + cnzi;
  }

  /* Column indices are in the list of free space */
  /* Allocate space for cj, initialize cj, and */
  /* destroy list of free space and other temporary array(s) */
  CHKERRQ(PetscMalloc1(ci[am]+1,&cj));
  CHKERRQ(PetscFreeSpaceContiguous(&free_space,cj));
  CHKERRQ(PetscLLCondensedDestroy(lnk,lnkbt));

  /* put together the new symbolic matrix */
  CHKERRQ(MatSetSeqAIJWithArrays_private(PetscObjectComm((PetscObject)A),am,bn,ci,cj,NULL,((PetscObject)A)->type_name,C));
  CHKERRQ(MatSetBlockSizesFromMats(C,A,B));

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
  C->info.mallocs           = ndouble;
  C->info.fill_ratio_given  = fill;
  C->info.fill_ratio_needed = afill;

#if defined(PETSC_USE_INFO)
  if (ci[am]) {
    CHKERRQ(PetscInfo(C,"Reallocs %" PetscInt_FMT "; Fill ratio: given %g needed %g.\n",ndouble,(double)fill,(double)afill));
    CHKERRQ(PetscInfo(C,"Use MatMatMult(A,B,MatReuse,%g,&C) for best performance.;\n",(double)afill));
  } else {
    CHKERRQ(PetscInfo(C,"Empty matrix product\n"));
  }
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatMultNumeric_SeqAIJ_SeqAIJ_Sorted(Mat A,Mat B,Mat C)
{
  PetscLogDouble    flops=0.0;
  Mat_SeqAIJ        *a   = (Mat_SeqAIJ*)A->data;
  Mat_SeqAIJ        *b   = (Mat_SeqAIJ*)B->data;
  Mat_SeqAIJ        *c   = (Mat_SeqAIJ*)C->data;
  PetscInt          *ai  =a->i,*aj=a->j,*bi=b->i,*bj=b->j,*bjj,*ci=c->i,*cj=c->j;
  PetscInt          am   =A->rmap->n,cm=C->rmap->n;
  PetscInt          i,j,k,anzi,bnzi,cnzi,brow;
  PetscScalar       *ca,valtmp;
  PetscScalar       *ab_dense;
  PetscContainer    cab_dense;
  const PetscScalar *aa,*ba,*baj;

  PetscFunctionBegin;
  CHKERRQ(MatSeqAIJGetArrayRead(A,&aa));
  CHKERRQ(MatSeqAIJGetArrayRead(B,&ba));
  if (!c->a) { /* first call of MatMatMultNumeric_SeqAIJ_SeqAIJ, allocate ca and matmult_abdense */
    CHKERRQ(PetscMalloc1(ci[cm]+1,&ca));
    c->a      = ca;
    c->free_a = PETSC_TRUE;
  } else ca = c->a;

  /* TODO this should be done in the symbolic phase */
  /* However, this function is so heavily used (sometimes in an hidden way through multnumeric function pointers
     that is hard to eradicate) */
  CHKERRQ(PetscObjectQuery((PetscObject)C,"__PETSc__ab_dense",(PetscObject*)&cab_dense));
  if (!cab_dense) {
    CHKERRQ(PetscMalloc1(B->cmap->N,&ab_dense));
    CHKERRQ(PetscContainerCreate(PETSC_COMM_SELF,&cab_dense));
    CHKERRQ(PetscContainerSetPointer(cab_dense,ab_dense));
    CHKERRQ(PetscContainerSetUserDestroy(cab_dense,PetscContainerUserDestroyDefault));
    CHKERRQ(PetscObjectCompose((PetscObject)C,"__PETSc__ab_dense",(PetscObject)cab_dense));
    CHKERRQ(PetscObjectDereference((PetscObject)cab_dense));
  }
  CHKERRQ(PetscContainerGetPointer(cab_dense,(void**)&ab_dense));
  CHKERRQ(PetscArrayzero(ab_dense,B->cmap->N));

  /* clean old values in C */
  CHKERRQ(PetscArrayzero(ca,ci[cm]));
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
#if defined(PETSC_HAVE_DEVICE)
  if (C->offloadmask != PETSC_OFFLOAD_UNALLOCATED) C->offloadmask = PETSC_OFFLOAD_CPU;
#endif
  CHKERRQ(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(PetscLogFlops(flops));
  CHKERRQ(MatSeqAIJRestoreArrayRead(A,&aa));
  CHKERRQ(MatSeqAIJRestoreArrayRead(B,&ba));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatMultNumeric_SeqAIJ_SeqAIJ_Scalable(Mat A,Mat B,Mat C)
{
  PetscLogDouble    flops=0.0;
  Mat_SeqAIJ        *a   = (Mat_SeqAIJ*)A->data;
  Mat_SeqAIJ        *b   = (Mat_SeqAIJ*)B->data;
  Mat_SeqAIJ        *c   = (Mat_SeqAIJ*)C->data;
  PetscInt          *ai  = a->i,*aj=a->j,*bi=b->i,*bj=b->j,*bjj,*ci=c->i,*cj=c->j;
  PetscInt          am   = A->rmap->N,cm=C->rmap->N;
  PetscInt          i,j,k,anzi,bnzi,cnzi,brow;
  PetscScalar       *ca=c->a,valtmp;
  const PetscScalar *aa,*ba,*baj;
  PetscInt          nextb;

  PetscFunctionBegin;
  CHKERRQ(MatSeqAIJGetArrayRead(A,&aa));
  CHKERRQ(MatSeqAIJGetArrayRead(B,&ba));
  if (!ca) { /* first call of MatMatMultNumeric_SeqAIJ_SeqAIJ, allocate ca and matmult_abdense */
    CHKERRQ(PetscMalloc1(ci[cm]+1,&ca));
    c->a      = ca;
    c->free_a = PETSC_TRUE;
  }

  /* clean old values in C */
  CHKERRQ(PetscArrayzero(ca,ci[cm]));
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
#if defined(PETSC_HAVE_DEVICE)
  if (C->offloadmask != PETSC_OFFLOAD_UNALLOCATED) C->offloadmask = PETSC_OFFLOAD_CPU;
#endif
  CHKERRQ(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(PetscLogFlops(flops));
  CHKERRQ(MatSeqAIJRestoreArrayRead(A,&aa));
  CHKERRQ(MatSeqAIJRestoreArrayRead(B,&ba));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatMultSymbolic_SeqAIJ_SeqAIJ_Scalable_fast(Mat A,Mat B,PetscReal fill,Mat C)
{
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
  CHKERRQ(PetscMalloc1(am+2,&ci));
  ci[0] = 0;

  /* create and initialize a linked list */
  CHKERRQ(PetscTableCreate(bn,bn,&ta));
  MatRowMergeMax_SeqAIJ(b,bm,ta);
  CHKERRQ(PetscTableGetCount(ta,&Crmax));
  CHKERRQ(PetscTableDestroy(&ta));

  CHKERRQ(PetscLLCondensedCreate_fast(Crmax,&lnk));

  /* Initial FreeSpace size is fill*(nnz(A)+nnz(B)) */
  CHKERRQ(PetscFreeSpaceGet(PetscRealIntMultTruncate(fill,PetscIntSumTruncate(ai[am],bi[bm])),&free_space));
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
      CHKERRQ(PetscLLCondensedAddSorted_fast(bnzj,bj,lnk));
    }
    /* add possible missing diagonal entry */
    if (C->force_diagonals) {
      CHKERRQ(PetscLLCondensedAddSorted_fast(1,&i,lnk));
    }
    cnzi = lnk[1];

    /* If free space is not available, make more free space */
    /* Double the amount of total space in the list */
    if (current_space->local_remaining<cnzi) {
      CHKERRQ(PetscFreeSpaceGet(PetscIntSumTruncate(cnzi,current_space->total_array_size),&current_space));
      ndouble++;
    }

    /* Copy data into free space, then initialize lnk */
    CHKERRQ(PetscLLCondensedClean_fast(cnzi,current_space->array,lnk));

    current_space->array           += cnzi;
    current_space->local_used      += cnzi;
    current_space->local_remaining -= cnzi;

    ci[i+1] = ci[i] + cnzi;
  }

  /* Column indices are in the list of free space */
  /* Allocate space for cj, initialize cj, and */
  /* destroy list of free space and other temporary array(s) */
  CHKERRQ(PetscMalloc1(ci[am]+1,&cj));
  CHKERRQ(PetscFreeSpaceContiguous(&free_space,cj));
  CHKERRQ(PetscLLCondensedDestroy_fast(lnk));

  /* Allocate space for ca */
  CHKERRQ(PetscCalloc1(ci[am]+1,&ca));

  /* put together the new symbolic matrix */
  CHKERRQ(MatSetSeqAIJWithArrays_private(PetscObjectComm((PetscObject)A),am,bn,ci,cj,ca,((PetscObject)A)->type_name,C));
  CHKERRQ(MatSetBlockSizesFromMats(C,A,B));

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
  C->info.mallocs           = ndouble;
  C->info.fill_ratio_given  = fill;
  C->info.fill_ratio_needed = afill;

#if defined(PETSC_USE_INFO)
  if (ci[am]) {
    CHKERRQ(PetscInfo(C,"Reallocs %" PetscInt_FMT "; Fill ratio: given %g needed %g.\n",ndouble,(double)fill,(double)afill));
    CHKERRQ(PetscInfo(C,"Use MatMatMult(A,B,MatReuse,%g,&C) for best performance.;\n",(double)afill));
  } else {
    CHKERRQ(PetscInfo(C,"Empty matrix product\n"));
  }
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatMultSymbolic_SeqAIJ_SeqAIJ_Scalable(Mat A,Mat B,PetscReal fill,Mat C)
{
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
  CHKERRQ(PetscMalloc1(am+2,&ci));
  ci[0] = 0;

  /* create and initialize a linked list */
  CHKERRQ(PetscTableCreate(bn,bn,&ta));
  MatRowMergeMax_SeqAIJ(b,bm,ta);
  CHKERRQ(PetscTableGetCount(ta,&Crmax));
  CHKERRQ(PetscTableDestroy(&ta));
  CHKERRQ(PetscLLCondensedCreate_Scalable(Crmax,&lnk));

  /* Initial FreeSpace size is fill*(nnz(A)+nnz(B)) */
  CHKERRQ(PetscFreeSpaceGet(PetscRealIntMultTruncate(fill,PetscIntSumTruncate(ai[am],bi[bm])),&free_space));
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
      CHKERRQ(PetscLLCondensedAddSorted_Scalable(bnzj,bj,lnk));
    }
    /* add possible missing diagonal entry */
    if (C->force_diagonals) {
      CHKERRQ(PetscLLCondensedAddSorted_Scalable(1,&i,lnk));
    }

    cnzi = lnk[0];

    /* If free space is not available, make more free space */
    /* Double the amount of total space in the list */
    if (current_space->local_remaining<cnzi) {
      CHKERRQ(PetscFreeSpaceGet(PetscIntSumTruncate(cnzi,current_space->total_array_size),&current_space));
      ndouble++;
    }

    /* Copy data into free space, then initialize lnk */
    CHKERRQ(PetscLLCondensedClean_Scalable(cnzi,current_space->array,lnk));

    current_space->array           += cnzi;
    current_space->local_used      += cnzi;
    current_space->local_remaining -= cnzi;

    ci[i+1] = ci[i] + cnzi;
  }

  /* Column indices are in the list of free space */
  /* Allocate space for cj, initialize cj, and */
  /* destroy list of free space and other temporary array(s) */
  CHKERRQ(PetscMalloc1(ci[am]+1,&cj));
  CHKERRQ(PetscFreeSpaceContiguous(&free_space,cj));
  CHKERRQ(PetscLLCondensedDestroy_Scalable(lnk));

  /* Allocate space for ca */
  /*-----------------------*/
  CHKERRQ(PetscCalloc1(ci[am]+1,&ca));

  /* put together the new symbolic matrix */
  CHKERRQ(MatSetSeqAIJWithArrays_private(PetscObjectComm((PetscObject)A),am,bn,ci,cj,ca,((PetscObject)A)->type_name,C));
  CHKERRQ(MatSetBlockSizesFromMats(C,A,B));

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
  C->info.mallocs           = ndouble;
  C->info.fill_ratio_given  = fill;
  C->info.fill_ratio_needed = afill;

#if defined(PETSC_USE_INFO)
  if (ci[am]) {
    CHKERRQ(PetscInfo(C,"Reallocs %" PetscInt_FMT "; Fill ratio: given %g needed %g.\n",ndouble,(double)fill,(double)afill));
    CHKERRQ(PetscInfo(C,"Use MatMatMult(A,B,MatReuse,%g,&C) for best performance.;\n",(double)afill));
  } else {
    CHKERRQ(PetscInfo(C,"Empty matrix product\n"));
  }
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatMultSymbolic_SeqAIJ_SeqAIJ_Heap(Mat A,Mat B,PetscReal fill,Mat C)
{
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
  CHKERRQ(PetscMalloc1(am+2,&ci));
  ci[0] = 0;

  /* Initial FreeSpace size is fill*(nnz(A)+nnz(B)) */
  CHKERRQ(PetscFreeSpaceGet(PetscRealIntMultTruncate(fill,PetscIntSumTruncate(ai[am],bi[bm])),&free_space));
  current_space = free_space;

  CHKERRQ(PetscHeapCreate(a->rmax,&h));
  CHKERRQ(PetscMalloc1(a->rmax,&bb));

  /* Determine ci and cj */
  for (i=0; i<am; i++) {
    const PetscInt anzi  = ai[i+1] - ai[i]; /* number of nonzeros in this row of A, this is the number of rows of B that we merge */
    const PetscInt *acol = aj + ai[i]; /* column indices of nonzero entries in this row */
    ci[i+1] = ci[i];
    /* Populate the min heap */
    for (j=0; j<anzi; j++) {
      bb[j] = bi[acol[j]];         /* bb points at the start of the row */
      if (bb[j] < bi[acol[j]+1]) { /* Add if row is nonempty */
        CHKERRQ(PetscHeapAdd(h,j,bj[bb[j]++]));
      }
    }
    /* Pick off the min element, adding it to free space */
    CHKERRQ(PetscHeapPop(h,&j,&col));
    while (j >= 0) {
      if (current_space->local_remaining < 1) { /* double the size, but don't exceed 16 MiB */
        CHKERRQ(PetscFreeSpaceGet(PetscMin(PetscIntMultTruncate(2,current_space->total_array_size),16 << 20),&current_space));
        ndouble++;
      }
      *(current_space->array++) = col;
      current_space->local_used++;
      current_space->local_remaining--;
      ci[i+1]++;

      /* stash if anything else remains in this row of B */
      if (bb[j] < bi[acol[j]+1]) CHKERRQ(PetscHeapStash(h,j,bj[bb[j]++]));
      while (1) {               /* pop and stash any other rows of B that also had an entry in this column */
        PetscInt j2,col2;
        CHKERRQ(PetscHeapPeek(h,&j2,&col2));
        if (col2 != col) break;
        CHKERRQ(PetscHeapPop(h,&j2,&col2));
        if (bb[j2] < bi[acol[j2]+1]) CHKERRQ(PetscHeapStash(h,j2,bj[bb[j2]++]));
      }
      /* Put any stashed elements back into the min heap */
      CHKERRQ(PetscHeapUnstash(h));
      CHKERRQ(PetscHeapPop(h,&j,&col));
    }
  }
  CHKERRQ(PetscFree(bb));
  CHKERRQ(PetscHeapDestroy(&h));

  /* Column indices are in the list of free space */
  /* Allocate space for cj, initialize cj, and */
  /* destroy list of free space and other temporary array(s) */
  CHKERRQ(PetscMalloc1(ci[am],&cj));
  CHKERRQ(PetscFreeSpaceContiguous(&free_space,cj));

  /* put together the new symbolic matrix */
  CHKERRQ(MatSetSeqAIJWithArrays_private(PetscObjectComm((PetscObject)A),am,bn,ci,cj,NULL,((PetscObject)A)->type_name,C));
  CHKERRQ(MatSetBlockSizesFromMats(C,A,B));

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
  C->info.mallocs           = ndouble;
  C->info.fill_ratio_given  = fill;
  C->info.fill_ratio_needed = afill;

#if defined(PETSC_USE_INFO)
  if (ci[am]) {
    CHKERRQ(PetscInfo(C,"Reallocs %" PetscInt_FMT "; Fill ratio: given %g needed %g.\n",ndouble,(double)fill,(double)afill));
    CHKERRQ(PetscInfo(C,"Use MatMatMult(A,B,MatReuse,%g,&C) for best performance.;\n",(double)afill));
  } else {
    CHKERRQ(PetscInfo(C,"Empty matrix product\n"));
  }
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatMultSymbolic_SeqAIJ_SeqAIJ_BTHeap(Mat A,Mat B,PetscReal fill,Mat C)
{
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
  CHKERRQ(PetscMalloc1(am+2,&ci));
  ci[0] = 0;

  /* Initial FreeSpace size is fill*(nnz(A)+nnz(B)) */
  CHKERRQ(PetscFreeSpaceGet(PetscRealIntMultTruncate(fill,PetscIntSumTruncate(ai[am],bi[bm])),&free_space));

  current_space = free_space;

  CHKERRQ(PetscHeapCreate(a->rmax,&h));
  CHKERRQ(PetscMalloc1(a->rmax,&bb));
  CHKERRQ(PetscBTCreate(bn,&bt));

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
          CHKERRQ(PetscHeapAdd(h,j,bcol));
          bb[j]++;
          break;
        }
      }
    }
    /* Pick off the min element, adding it to free space */
    CHKERRQ(PetscHeapPop(h,&j,&col));
    while (j >= 0) {
      if (current_space->local_remaining < 1) { /* double the size, but don't exceed 16 MiB */
        fptr = NULL;                      /* need PetscBTMemzero */
        CHKERRQ(PetscFreeSpaceGet(PetscMin(PetscIntMultTruncate(2,current_space->total_array_size),16 << 20),&current_space));
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
          CHKERRQ(PetscHeapAdd(h,j,bcol));
          bb[j]++;
          break;
        }
      }
      CHKERRQ(PetscHeapPop(h,&j,&col));
    }
    if (fptr) {                 /* Clear the bits for this row */
      for (; fptr<current_space->array; fptr++) CHKERRQ(PetscBTClear(bt,*fptr));
    } else {                    /* We reallocated so we don't remember (easily) how to clear only the bits we changed */
      CHKERRQ(PetscBTMemzero(bn,bt));
    }
  }
  CHKERRQ(PetscFree(bb));
  CHKERRQ(PetscHeapDestroy(&h));
  CHKERRQ(PetscBTDestroy(&bt));

  /* Column indices are in the list of free space */
  /* Allocate space for cj, initialize cj, and */
  /* destroy list of free space and other temporary array(s) */
  CHKERRQ(PetscMalloc1(ci[am],&cj));
  CHKERRQ(PetscFreeSpaceContiguous(&free_space,cj));

  /* put together the new symbolic matrix */
  CHKERRQ(MatSetSeqAIJWithArrays_private(PetscObjectComm((PetscObject)A),am,bn,ci,cj,NULL,((PetscObject)A)->type_name,C));
  CHKERRQ(MatSetBlockSizesFromMats(C,A,B));

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
  C->info.mallocs           = ndouble;
  C->info.fill_ratio_given  = fill;
  C->info.fill_ratio_needed = afill;

#if defined(PETSC_USE_INFO)
  if (ci[am]) {
    CHKERRQ(PetscInfo(C,"Reallocs %" PetscInt_FMT "; Fill ratio: given %g needed %g.\n",ndouble,(double)fill,(double)afill));
    CHKERRQ(PetscInfo(C,"Use MatMatMult(A,B,MatReuse,%g,&C) for best performance.;\n",(double)afill));
  } else {
    CHKERRQ(PetscInfo(C,"Empty matrix product\n"));
  }
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatMultSymbolic_SeqAIJ_SeqAIJ_RowMerge(Mat A,Mat B,PetscReal fill,Mat C)
{
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
  CHKERRQ(PetscMalloc1(am+1,&ci));
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
  CHKERRQ(PetscMalloc1(a_maxrownnz*8,&workj_L1));
  CHKERRQ(PetscMalloc1(a_maxrownnz*8,&workj_L2));
  CHKERRQ(PetscMalloc1(a_maxrownnz,&workj_L3));

  /* This should be enough for almost all matrices. If not, memory is reallocated later. */
  c_maxmem = 8*(ai[am]+bi[bm]);
  /* Step 2: Populate pattern for C */
  CHKERRQ(PetscMalloc1(c_maxmem,&cj));

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
      CHKERRQ(PetscRealloc(sizeof(PetscInt)*c_maxmem,&cj));
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
  CHKERRQ(MatSetSeqAIJWithArrays_private(PetscObjectComm((PetscObject)A),am,bn,ci,cj,NULL,((PetscObject)A)->type_name,C));
  CHKERRQ(MatSetBlockSizesFromMats(C,A,B));

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
  C->info.mallocs           = ndouble;
  C->info.fill_ratio_given  = fill;
  C->info.fill_ratio_needed = afill;

#if defined(PETSC_USE_INFO)
  if (ci[am]) {
    CHKERRQ(PetscInfo(C,"Reallocs %" PetscInt_FMT "; Fill ratio: given %g needed %g.\n",ndouble,(double)fill,(double)afill));
    CHKERRQ(PetscInfo(C,"Use MatMatMult(A,B,MatReuse,%g,&C) for best performance.;\n",(double)afill));
  } else {
    CHKERRQ(PetscInfo(C,"Empty matrix product\n"));
  }
#endif

  /* Step 4: Free temporary work areas */
  CHKERRQ(PetscFree(workj_L1));
  CHKERRQ(PetscFree(workj_L2));
  CHKERRQ(PetscFree(workj_L3));
  PetscFunctionReturn(0);
}

/* concatenate unique entries and then sort */
PetscErrorCode MatMatMultSymbolic_SeqAIJ_SeqAIJ_Sorted(Mat A,Mat B,PetscReal fill,Mat C)
{
  Mat_SeqAIJ     *a  = (Mat_SeqAIJ*)A->data,*b=(Mat_SeqAIJ*)B->data,*c;
  const PetscInt *ai = a->i,*bi=b->i,*aj=a->j,*bj=b->j;
  PetscInt       *ci,*cj,bcol;
  PetscInt       am=A->rmap->N,bn=B->cmap->N,bm=B->rmap->N;
  PetscReal      afill;
  PetscInt       i,j,ndouble = 0;
  PetscSegBuffer seg,segrow;
  char           *seen;

  PetscFunctionBegin;
  CHKERRQ(PetscMalloc1(am+1,&ci));
  ci[0] = 0;

  /* Initial FreeSpace size is fill*(nnz(A)+nnz(B)) */
  CHKERRQ(PetscSegBufferCreate(sizeof(PetscInt),(PetscInt)(fill*(ai[am]+bi[bm])),&seg));
  CHKERRQ(PetscSegBufferCreate(sizeof(PetscInt),100,&segrow));
  CHKERRQ(PetscCalloc1(bn,&seen));

  /* Determine ci and cj */
  for (i=0; i<am; i++) {
    const PetscInt anzi  = ai[i+1] - ai[i]; /* number of nonzeros in this row of A, this is the number of rows of B that we merge */
    const PetscInt *acol = aj + ai[i]; /* column indices of nonzero entries in this row */
    PetscInt packlen = 0,*PETSC_RESTRICT crow;

    /* Pack segrow */
    for (j=0; j<anzi; j++) {
      PetscInt brow = acol[j],bjstart = bi[brow],bjend = bi[brow+1],k;
      for (k=bjstart; k<bjend; k++) {
        bcol = bj[k];
        if (!seen[bcol]) { /* new entry */
          PetscInt *PETSC_RESTRICT slot;
          CHKERRQ(PetscSegBufferGetInts(segrow,1,&slot));
          *slot = bcol;
          seen[bcol] = 1;
          packlen++;
        }
      }
    }

    /* Check i-th diagonal entry */
    if (C->force_diagonals && !seen[i]) {
      PetscInt *PETSC_RESTRICT slot;
      CHKERRQ(PetscSegBufferGetInts(segrow,1,&slot));
      *slot   = i;
      seen[i] = 1;
      packlen++;
    }

    CHKERRQ(PetscSegBufferGetInts(seg,packlen,&crow));
    CHKERRQ(PetscSegBufferExtractTo(segrow,crow));
    CHKERRQ(PetscSortInt(packlen,crow));
    ci[i+1] = ci[i] + packlen;
    for (j=0; j<packlen; j++) seen[crow[j]] = 0;
  }
  CHKERRQ(PetscSegBufferDestroy(&segrow));
  CHKERRQ(PetscFree(seen));

  /* Column indices are in the segmented buffer */
  CHKERRQ(PetscSegBufferExtractAlloc(seg,&cj));
  CHKERRQ(PetscSegBufferDestroy(&seg));

  /* put together the new symbolic matrix */
  CHKERRQ(MatSetSeqAIJWithArrays_private(PetscObjectComm((PetscObject)A),am,bn,ci,cj,NULL,((PetscObject)A)->type_name,C));
  CHKERRQ(MatSetBlockSizesFromMats(C,A,B));

  /* MatCreateSeqAIJWithArrays flags matrix so PETSc doesn't free the user's arrays. */
  /* These are PETSc arrays, so change flags so arrays can be deleted by PETSc */
  c          = (Mat_SeqAIJ*)(C->data);
  c->free_a  = PETSC_TRUE;
  c->free_ij = PETSC_TRUE;
  c->nonew   = 0;

  C->ops->matmultnumeric = MatMatMultNumeric_SeqAIJ_SeqAIJ_Sorted;

  /* set MatInfo */
  afill = (PetscReal)ci[am]/PetscMax(ai[am]+bi[bm],1) + 1.e-5;
  if (afill < 1.0) afill = 1.0;
  C->info.mallocs           = ndouble;
  C->info.fill_ratio_given  = fill;
  C->info.fill_ratio_needed = afill;

#if defined(PETSC_USE_INFO)
  if (ci[am]) {
    CHKERRQ(PetscInfo(C,"Reallocs %" PetscInt_FMT "; Fill ratio: given %g needed %g.\n",ndouble,(double)fill,(double)afill));
    CHKERRQ(PetscInfo(C,"Use MatMatMult(A,B,MatReuse,%g,&C) for best performance.;\n",(double)afill));
  } else {
    CHKERRQ(PetscInfo(C,"Empty matrix product\n"));
  }
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_SeqAIJ_MatMatMultTrans(void *data)
{
  Mat_MatMatTransMult *abt=(Mat_MatMatTransMult *)data;

  PetscFunctionBegin;
  CHKERRQ(MatTransposeColoringDestroy(&abt->matcoloring));
  CHKERRQ(MatDestroy(&abt->Bt_den));
  CHKERRQ(MatDestroy(&abt->ABt_den));
  CHKERRQ(PetscFree(abt));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatTransposeMultSymbolic_SeqAIJ_SeqAIJ(Mat A,Mat B,PetscReal fill,Mat C)
{
  Mat                 Bt;
  PetscInt            *bti,*btj;
  Mat_MatMatTransMult *abt;
  Mat_Product         *product = C->product;
  char                *alg;

  PetscFunctionBegin;
  PetscCheckFalse(!product,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing product struct");
  PetscCheckFalse(product->data,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Extra product struct not empty");

  /* create symbolic Bt */
  CHKERRQ(MatGetSymbolicTranspose_SeqAIJ(B,&bti,&btj));
  CHKERRQ(MatCreateSeqAIJWithArrays(PETSC_COMM_SELF,B->cmap->n,B->rmap->n,bti,btj,NULL,&Bt));
  CHKERRQ(MatSetBlockSizes(Bt,PetscAbs(A->cmap->bs),PetscAbs(B->cmap->bs)));
  CHKERRQ(MatSetType(Bt,((PetscObject)A)->type_name));

  /* get symbolic C=A*Bt */
  CHKERRQ(PetscStrallocpy(product->alg,&alg));
  CHKERRQ(MatProductSetAlgorithm(C,"sorted")); /* set algorithm for C = A*Bt */
  CHKERRQ(MatMatMultSymbolic_SeqAIJ_SeqAIJ(A,Bt,fill,C));
  CHKERRQ(MatProductSetAlgorithm(C,alg)); /* resume original algorithm for ABt product */
  CHKERRQ(PetscFree(alg));

  /* create a supporting struct for reuse intermediate dense matrices with matcoloring */
  CHKERRQ(PetscNew(&abt));

  product->data    = abt;
  product->destroy = MatDestroy_SeqAIJ_MatMatMultTrans;

  C->ops->mattransposemultnumeric = MatMatTransposeMultNumeric_SeqAIJ_SeqAIJ;

  abt->usecoloring = PETSC_FALSE;
  CHKERRQ(PetscStrcmp(product->alg,"color",&abt->usecoloring));
  if (abt->usecoloring) {
    /* Create MatTransposeColoring from symbolic C=A*B^T */
    MatTransposeColoring matcoloring;
    MatColoring          coloring;
    ISColoring           iscoloring;
    Mat                  Bt_dense,C_dense;

    /* inode causes memory problem */
    CHKERRQ(MatSetOption(C,MAT_USE_INODES,PETSC_FALSE));

    CHKERRQ(MatColoringCreate(C,&coloring));
    CHKERRQ(MatColoringSetDistance(coloring,2));
    CHKERRQ(MatColoringSetType(coloring,MATCOLORINGSL));
    CHKERRQ(MatColoringSetFromOptions(coloring));
    CHKERRQ(MatColoringApply(coloring,&iscoloring));
    CHKERRQ(MatColoringDestroy(&coloring));
    CHKERRQ(MatTransposeColoringCreate(C,iscoloring,&matcoloring));

    abt->matcoloring = matcoloring;

    CHKERRQ(ISColoringDestroy(&iscoloring));

    /* Create Bt_dense and C_dense = A*Bt_dense */
    CHKERRQ(MatCreate(PETSC_COMM_SELF,&Bt_dense));
    CHKERRQ(MatSetSizes(Bt_dense,A->cmap->n,matcoloring->ncolors,A->cmap->n,matcoloring->ncolors));
    CHKERRQ(MatSetType(Bt_dense,MATSEQDENSE));
    CHKERRQ(MatSeqDenseSetPreallocation(Bt_dense,NULL));

    Bt_dense->assembled = PETSC_TRUE;
    abt->Bt_den         = Bt_dense;

    CHKERRQ(MatCreate(PETSC_COMM_SELF,&C_dense));
    CHKERRQ(MatSetSizes(C_dense,A->rmap->n,matcoloring->ncolors,A->rmap->n,matcoloring->ncolors));
    CHKERRQ(MatSetType(C_dense,MATSEQDENSE));
    CHKERRQ(MatSeqDenseSetPreallocation(C_dense,NULL));

    Bt_dense->assembled = PETSC_TRUE;
    abt->ABt_den  = C_dense;

#if defined(PETSC_USE_INFO)
    {
      Mat_SeqAIJ *c = (Mat_SeqAIJ*)C->data;
      CHKERRQ(PetscInfo(C,"Use coloring of C=A*B^T; B^T: %" PetscInt_FMT " %" PetscInt_FMT ", Bt_dense: %" PetscInt_FMT ",%" PetscInt_FMT "; Cnz %" PetscInt_FMT " / (cm*ncolors %" PetscInt_FMT ") = %g\n",B->cmap->n,B->rmap->n,Bt_dense->rmap->n,Bt_dense->cmap->n,c->nz,A->rmap->n*matcoloring->ncolors,(double)(((PetscReal)(c->nz))/((PetscReal)(A->rmap->n*matcoloring->ncolors)))));
    }
#endif
  }
  /* clean up */
  CHKERRQ(MatDestroy(&Bt));
  CHKERRQ(MatRestoreSymbolicTranspose_SeqAIJ(B,&bti,&btj));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatTransposeMultNumeric_SeqAIJ_SeqAIJ(Mat A,Mat B,Mat C)
{
  Mat_SeqAIJ          *a   =(Mat_SeqAIJ*)A->data,*b=(Mat_SeqAIJ*)B->data,*c=(Mat_SeqAIJ*)C->data;
  PetscInt            *ai  =a->i,*aj=a->j,*bi=b->i,*bj=b->j,anzi,bnzj,nexta,nextb,*acol,*bcol,brow;
  PetscInt            cm   =C->rmap->n,*ci=c->i,*cj=c->j,i,j,cnzi,*ccol;
  PetscLogDouble      flops=0.0;
  MatScalar           *aa  =a->a,*aval,*ba=b->a,*bval,*ca,*cval;
  Mat_MatMatTransMult *abt;
  Mat_Product         *product = C->product;

  PetscFunctionBegin;
  PetscCheckFalse(!product,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing product struct");
  abt = (Mat_MatMatTransMult *)product->data;
  PetscCheckFalse(!abt,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing product struct");
  /* clear old values in C */
  if (!c->a) {
    CHKERRQ(PetscCalloc1(ci[cm]+1,&ca));
    c->a      = ca;
    c->free_a = PETSC_TRUE;
  } else {
    ca =  c->a;
    CHKERRQ(PetscArrayzero(ca,ci[cm]+1));
  }

  if (abt->usecoloring) {
    MatTransposeColoring matcoloring = abt->matcoloring;
    Mat                  Bt_dense,C_dense = abt->ABt_den;

    /* Get Bt_dense by Apply MatTransposeColoring to B */
    Bt_dense = abt->Bt_den;
    CHKERRQ(MatTransColoringApplySpToDen(matcoloring,B,Bt_dense));

    /* C_dense = A*Bt_dense */
    CHKERRQ(MatMatMultNumeric_SeqAIJ_SeqDense(A,Bt_dense,C_dense));

    /* Recover C from C_dense */
    CHKERRQ(MatTransColoringApplyDenToSp(matcoloring,C_dense,C));
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
  CHKERRQ(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(PetscLogFlops(flops));
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_SeqAIJ_MatTransMatMult(void *data)
{
  Mat_MatTransMatMult *atb = (Mat_MatTransMatMult*)data;

  PetscFunctionBegin;
  CHKERRQ(MatDestroy(&atb->At));
  if (atb->destroy) {
    CHKERRQ((*atb->destroy)(atb->data));
  }
  CHKERRQ(PetscFree(atb));
  PetscFunctionReturn(0);
}

PetscErrorCode MatTransposeMatMultSymbolic_SeqAIJ_SeqAIJ(Mat A,Mat B,PetscReal fill,Mat C)
{
  Mat            At = NULL;
  PetscInt       *ati,*atj;
  Mat_Product    *product = C->product;
  PetscBool      flg,def,square;

  PetscFunctionBegin;
  MatCheckProduct(C,4);
  square = (PetscBool)(A == B && A->symmetric && A->symmetric_set);
  /* outerproduct */
  CHKERRQ(PetscStrcmp(product->alg,"outerproduct",&flg));
  if (flg) {
    /* create symbolic At */
    if (!square) {
      CHKERRQ(MatGetSymbolicTranspose_SeqAIJ(A,&ati,&atj));
      CHKERRQ(MatCreateSeqAIJWithArrays(PETSC_COMM_SELF,A->cmap->n,A->rmap->n,ati,atj,NULL,&At));
      CHKERRQ(MatSetBlockSizes(At,PetscAbs(A->cmap->bs),PetscAbs(B->cmap->bs)));
      CHKERRQ(MatSetType(At,((PetscObject)A)->type_name));
    }
    /* get symbolic C=At*B */
    CHKERRQ(MatProductSetAlgorithm(C,"sorted"));
    CHKERRQ(MatMatMultSymbolic_SeqAIJ_SeqAIJ(square ? A : At,B,fill,C));

    /* clean up */
    if (!square) {
      CHKERRQ(MatDestroy(&At));
      CHKERRQ(MatRestoreSymbolicTranspose_SeqAIJ(A,&ati,&atj));
    }

    C->ops->mattransposemultnumeric = MatTransposeMatMultNumeric_SeqAIJ_SeqAIJ; /* outerproduct */
    CHKERRQ(MatProductSetAlgorithm(C,"outerproduct"));
    PetscFunctionReturn(0);
  }

  /* matmatmult */
  CHKERRQ(PetscStrcmp(product->alg,"default",&def));
  CHKERRQ(PetscStrcmp(product->alg,"at*b",&flg));
  if (flg || def) {
    Mat_MatTransMatMult *atb;

    PetscCheckFalse(product->data,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Extra product struct not empty");
    CHKERRQ(PetscNew(&atb));
    if (!square) {
      CHKERRQ(MatTranspose_SeqAIJ(A,MAT_INITIAL_MATRIX,&At));
    }
    CHKERRQ(MatProductSetAlgorithm(C,"sorted"));
    CHKERRQ(MatMatMultSymbolic_SeqAIJ_SeqAIJ(square ? A : At,B,fill,C));
    CHKERRQ(MatProductSetAlgorithm(C,"at*b"));
    product->data    = atb;
    product->destroy = MatDestroy_SeqAIJ_MatTransMatMult;
    atb->At          = At;
    atb->updateAt    = PETSC_FALSE; /* because At is computed here */

    C->ops->mattransposemultnumeric = NULL; /* see MatProductNumeric_AtB_SeqAIJ_SeqAIJ */
    PetscFunctionReturn(0);
  }

  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Mat Product Algorithm is not supported");
}

PetscErrorCode MatTransposeMatMultNumeric_SeqAIJ_SeqAIJ(Mat A,Mat B,Mat C)
{
  Mat_SeqAIJ     *a=(Mat_SeqAIJ*)A->data,*b=(Mat_SeqAIJ*)B->data,*c=(Mat_SeqAIJ*)C->data;
  PetscInt       am=A->rmap->n,anzi,*ai=a->i,*aj=a->j,*bi=b->i,*bj,bnzi,nextb;
  PetscInt       cm=C->rmap->n,*ci=c->i,*cj=c->j,crow,*cjj,i,j,k;
  PetscLogDouble flops=0.0;
  MatScalar      *aa=a->a,*ba,*ca,*caj;

  PetscFunctionBegin;
  if (!c->a) {
    CHKERRQ(PetscCalloc1(ci[cm]+1,&ca));

    c->a      = ca;
    c->free_a = PETSC_TRUE;
  } else {
    ca   = c->a;
    CHKERRQ(PetscArrayzero(ca,ci[cm]));
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
  CHKERRQ(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(PetscLogFlops(flops));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatMultSymbolic_SeqAIJ_SeqDense(Mat A,Mat B,PetscReal fill,Mat C)
{
  PetscFunctionBegin;
  CHKERRQ(MatMatMultSymbolic_SeqDense_SeqDense(A,B,0.0,C));
  C->ops->matmultnumeric = MatMatMultNumeric_SeqAIJ_SeqDense;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatMatMultNumericAdd_SeqAIJ_SeqDense(Mat A,Mat B,Mat C,const PetscBool add)
{
  Mat_SeqAIJ        *a=(Mat_SeqAIJ*)A->data;
  PetscScalar       *c,r1,r2,r3,r4,*c1,*c2,*c3,*c4;
  const PetscScalar *aa,*b,*b1,*b2,*b3,*b4,*av;
  const PetscInt    *aj;
  PetscInt          cm=C->rmap->n,cn=B->cmap->n,bm,am=A->rmap->n;
  PetscInt          clda;
  PetscInt          am4,bm4,col,i,j,n;

  PetscFunctionBegin;
  if (!cm || !cn) PetscFunctionReturn(0);
  CHKERRQ(MatSeqAIJGetArrayRead(A,&av));
  if (add) {
    CHKERRQ(MatDenseGetArray(C,&c));
  } else {
    CHKERRQ(MatDenseGetArrayWrite(C,&c));
  }
  CHKERRQ(MatDenseGetArrayRead(B,&b));
  CHKERRQ(MatDenseGetLDA(B,&bm));
  CHKERRQ(MatDenseGetLDA(C,&clda));
  am4 = 4*clda;
  bm4 = 4*bm;
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
      if (add) {
        c1[i] += r1;
        c2[i] += r2;
        c3[i] += r3;
        c4[i] += r4;
      } else {
        c1[i] = r1;
        c2[i] = r2;
        c3[i] = r3;
        c4[i] = r4;
      }
    }
    b1 += bm4; b2 += bm4; b3 += bm4; b4 += bm4;
    c1 += am4; c2 += am4; c3 += am4; c4 += am4;
  }
  /* process remaining columns */
  if (col != cn) {
    PetscInt rc = cn-col;

    if (rc == 1) {
      for (i=0; i<am; i++) {
        r1 = 0.0;
        n  = a->i[i+1] - a->i[i];
        aj = a->j + a->i[i];
        aa = av + a->i[i];
        for (j=0; j<n; j++) r1 += aa[j]*b1[aj[j]];
        if (add) c1[i] += r1;
        else c1[i] = r1;
      }
    } else if (rc == 2) {
      for (i=0; i<am; i++) {
        r1 = r2 = 0.0;
        n  = a->i[i+1] - a->i[i];
        aj = a->j + a->i[i];
        aa = av + a->i[i];
        for (j=0; j<n; j++) {
          const PetscScalar aatmp = aa[j];
          const PetscInt    ajtmp = aj[j];
          r1 += aatmp*b1[ajtmp];
          r2 += aatmp*b2[ajtmp];
        }
        if (add) {
          c1[i] += r1;
          c2[i] += r2;
        } else {
          c1[i] = r1;
          c2[i] = r2;
        }
      }
    } else {
      for (i=0; i<am; i++) {
        r1 = r2 = r3 = 0.0;
        n  = a->i[i+1] - a->i[i];
        aj = a->j + a->i[i];
        aa = av + a->i[i];
        for (j=0; j<n; j++) {
          const PetscScalar aatmp = aa[j];
          const PetscInt    ajtmp = aj[j];
          r1 += aatmp*b1[ajtmp];
          r2 += aatmp*b2[ajtmp];
          r3 += aatmp*b3[ajtmp];
        }
        if (add) {
          c1[i] += r1;
          c2[i] += r2;
          c3[i] += r3;
        } else {
          c1[i] = r1;
          c2[i] = r2;
          c3[i] = r3;
        }
      }
    }
  }
  CHKERRQ(PetscLogFlops(cn*(2.0*a->nz)));
  if (add) {
    CHKERRQ(MatDenseRestoreArray(C,&c));
  } else {
    CHKERRQ(MatDenseRestoreArrayWrite(C,&c));
  }
  CHKERRQ(MatDenseRestoreArrayRead(B,&b));
  CHKERRQ(MatSeqAIJRestoreArrayRead(A,&av));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatMultNumeric_SeqAIJ_SeqDense(Mat A,Mat B,Mat C)
{
  PetscFunctionBegin;
  PetscCheckFalse(B->rmap->n != A->cmap->n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Number columns in A %" PetscInt_FMT " not equal rows in B %" PetscInt_FMT,A->cmap->n,B->rmap->n);
  PetscCheckFalse(A->rmap->n != C->rmap->n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Number rows in C %" PetscInt_FMT " not equal rows in A %" PetscInt_FMT,C->rmap->n,A->rmap->n);
  PetscCheckFalse(B->cmap->n != C->cmap->n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Number columns in B %" PetscInt_FMT " not equal columns in C %" PetscInt_FMT,B->cmap->n,C->cmap->n);

  CHKERRQ(MatMatMultNumericAdd_SeqAIJ_SeqDense(A,B,C,PETSC_FALSE));
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
  Mat_Product    *product = C->product;

  PetscFunctionBegin;
  switch (product->type) {
  case MATPRODUCT_AB:
    CHKERRQ(MatProductSetFromOptions_SeqAIJ_SeqDense_AB(C));
    break;
  case MATPRODUCT_AtB:
    CHKERRQ(MatProductSetFromOptions_SeqAIJ_SeqDense_AtB(C));
    break;
  case MATPRODUCT_ABt:
    CHKERRQ(MatProductSetFromOptions_SeqAIJ_SeqDense_ABt(C));
    break;
  default:
    break;
  }
  PetscFunctionReturn(0);
}
/* ------------------------------------------------------- */
static PetscErrorCode MatProductSetFromOptions_SeqXBAIJ_SeqDense_AB(Mat C)
{
  Mat_Product    *product = C->product;
  Mat            A = product->A;
  PetscBool      baij;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)A,MATSEQBAIJ,&baij));
  if (!baij) { /* A is seqsbaij */
    PetscBool sbaij;
    CHKERRQ(PetscObjectTypeCompare((PetscObject)A,MATSEQSBAIJ,&sbaij));
    PetscCheckFalse(!sbaij,PetscObjectComm((PetscObject)C),PETSC_ERR_ARG_WRONGSTATE,"Mat must be either seqbaij or seqsbaij format");

    C->ops->matmultsymbolic = MatMatMultSymbolic_SeqSBAIJ_SeqDense;
  } else { /* A is seqbaij */
    C->ops->matmultsymbolic = MatMatMultSymbolic_SeqBAIJ_SeqDense;
  }

  C->ops->productsymbolic = MatProductSymbolic_AB;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatProductSetFromOptions_SeqXBAIJ_SeqDense(Mat C)
{
  Mat_Product    *product = C->product;

  PetscFunctionBegin;
  MatCheckProduct(C,1);
  PetscCheckFalse(!product->A,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing A");
  if (product->type == MATPRODUCT_AB || (product->type == MATPRODUCT_AtB && product->A->symmetric)) {
    CHKERRQ(MatProductSetFromOptions_SeqXBAIJ_SeqDense_AB(C));
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
  Mat_Product    *product = C->product;

  PetscFunctionBegin;
  if (product->type == MATPRODUCT_AB) {
    CHKERRQ(MatProductSetFromOptions_SeqDense_SeqAIJ_AB(C));
  }
  PetscFunctionReturn(0);
}
/* ------------------------------------------------------- */

PetscErrorCode  MatTransColoringApplySpToDen_SeqAIJ(MatTransposeColoring coloring,Mat B,Mat Btdense)
{
  Mat_SeqAIJ     *b       = (Mat_SeqAIJ*)B->data;
  Mat_SeqDense   *btdense = (Mat_SeqDense*)Btdense->data;
  PetscInt       *bi      = b->i,*bj=b->j;
  PetscInt       m        = Btdense->rmap->n,n=Btdense->cmap->n,j,k,l,col,anz,*btcol,brow,ncolumns;
  MatScalar      *btval,*btval_den,*ba=b->a;
  PetscInt       *columns=coloring->columns,*colorforcol=coloring->colorforcol,ncolors=coloring->ncolors;

  PetscFunctionBegin;
  btval_den=btdense->v;
  CHKERRQ(PetscArrayzero(btval_den,m*n));
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
  Mat_SeqAIJ        *csp = (Mat_SeqAIJ*)Csp->data;
  const PetscScalar *ca_den,*ca_den_ptr;
  PetscScalar       *ca=csp->a;
  PetscInt          k,l,m=Cden->rmap->n,ncolors=matcoloring->ncolors;
  PetscInt          brows=matcoloring->brows,*den2sp=matcoloring->den2sp;
  PetscInt          nrows,*row,*idx;
  PetscInt          *rows=matcoloring->rows,*colorforrow=matcoloring->colorforrow;

  PetscFunctionBegin;
  CHKERRQ(MatDenseGetArrayRead(Cden,&ca_den));

  if (brows > 0) {
    PetscInt *lstart,row_end,row_start;
    lstart = matcoloring->lstart;
    CHKERRQ(PetscArrayzero(lstart,ncolors));

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

  CHKERRQ(MatDenseRestoreArrayRead(Cden,&ca_den));
#if defined(PETSC_USE_INFO)
  if (matcoloring->brows > 0) {
    CHKERRQ(PetscInfo(Csp,"Loop over %" PetscInt_FMT " row blocks for den2sp\n",brows));
  } else {
    CHKERRQ(PetscInfo(Csp,"Loop over colors/columns of Cden, inefficient for large sparse matrix product \n"));
  }
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode MatTransposeColoringCreate_SeqAIJ(Mat mat,ISColoring iscoloring,MatTransposeColoring c)
{
  PetscInt       i,n,nrows,Nbs,j,k,m,ncols,col,cm;
  const PetscInt *is,*ci,*cj,*row_idx;
  PetscInt       nis = iscoloring->n,*rowhit,bs = 1;
  IS             *isa;
  Mat_SeqAIJ     *csp = (Mat_SeqAIJ*)mat->data;
  PetscInt       *colorforrow,*rows,*rows_i,*idxhit,*spidx,*den2sp,*den2sp_i;
  PetscInt       *colorforcol,*columns,*columns_i,brows;
  PetscBool      flg;

  PetscFunctionBegin;
  CHKERRQ(ISColoringGetIS(iscoloring,PETSC_USE_POINTER,PETSC_IGNORE,&isa));

  /* bs >1 is not being tested yet! */
  Nbs       = mat->cmap->N/bs;
  c->M      = mat->rmap->N/bs;  /* set total rows, columns and local rows */
  c->N      = Nbs;
  c->m      = c->M;
  c->rstart = 0;
  c->brows  = 100;

  c->ncolors = nis;
  CHKERRQ(PetscMalloc3(nis,&c->ncolumns,nis,&c->nrows,nis+1,&colorforrow));
  CHKERRQ(PetscMalloc1(csp->nz+1,&rows));
  CHKERRQ(PetscMalloc1(csp->nz+1,&den2sp));

  brows = c->brows;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-matden2sp_brows",&brows,&flg));
  if (flg) c->brows = brows;
  if (brows > 0) {
    CHKERRQ(PetscMalloc1(nis+1,&c->lstart));
  }

  colorforrow[0] = 0;
  rows_i         = rows;
  den2sp_i       = den2sp;

  CHKERRQ(PetscMalloc1(nis+1,&colorforcol));
  CHKERRQ(PetscMalloc1(Nbs+1,&columns));

  colorforcol[0] = 0;
  columns_i      = columns;

  /* get column-wise storage of mat */
  CHKERRQ(MatGetColumnIJ_SeqAIJ_Color(mat,0,PETSC_FALSE,PETSC_FALSE,&ncols,&ci,&cj,&spidx,NULL));

  cm   = c->m;
  CHKERRQ(PetscMalloc1(cm+1,&rowhit));
  CHKERRQ(PetscMalloc1(cm+1,&idxhit));
  for (i=0; i<nis; i++) { /* loop over color */
    CHKERRQ(ISGetLocalSize(isa[i],&n));
    CHKERRQ(ISGetIndices(isa[i],&is));

    c->ncolumns[i] = n;
    if (n) {
      CHKERRQ(PetscArraycpy(columns_i,is,n));
    }
    colorforcol[i+1] = colorforcol[i] + n;
    columns_i       += n;

    /* fast, crude version requires O(N*N) work */
    CHKERRQ(PetscArrayzero(rowhit,cm));

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

    CHKERRQ(ISRestoreIndices(isa[i],&is));
    rows_i += nrows;
  }
  CHKERRQ(MatRestoreColumnIJ_SeqAIJ_Color(mat,0,PETSC_FALSE,PETSC_FALSE,&ncols,&ci,&cj,&spidx,NULL));
  CHKERRQ(PetscFree(rowhit));
  CHKERRQ(ISColoringRestoreIS(iscoloring,PETSC_USE_POINTER,&isa));
  PetscCheck(csp->nz == colorforrow[nis],PETSC_COMM_SELF,PETSC_ERR_PLIB,"csp->nz %" PetscInt_FMT " != colorforrow[nis] %" PetscInt_FMT,csp->nz,colorforrow[nis]);

  c->colorforrow = colorforrow;
  c->rows        = rows;
  c->den2sp      = den2sp;
  c->colorforcol = colorforcol;
  c->columns     = columns;

  CHKERRQ(PetscFree(idxhit));
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------- */
static PetscErrorCode MatProductNumeric_AtB_SeqAIJ_SeqAIJ(Mat C)
{
  Mat_Product    *product = C->product;
  Mat            A=product->A,B=product->B;

  PetscFunctionBegin;
  if (C->ops->mattransposemultnumeric) {
    /* Alg: "outerproduct" */
    CHKERRQ((*C->ops->mattransposemultnumeric)(A,B,C));
  } else {
    /* Alg: "matmatmult" -- C = At*B */
    Mat_MatTransMatMult *atb = (Mat_MatTransMatMult *)product->data;
    Mat                 At;

    PetscCheckFalse(!atb,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing product struct");
    At = atb->At;
    if (atb->updateAt && At) { /* At is computed in MatTransposeMatMultSymbolic_SeqAIJ_SeqAIJ() */
      CHKERRQ(MatTranspose_SeqAIJ(A,MAT_REUSE_MATRIX,&At));
    }
    CHKERRQ(MatMatMultNumeric_SeqAIJ_SeqAIJ(At ? At : A,B,C));
    atb->updateAt = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductSymbolic_AtB_SeqAIJ_SeqAIJ(Mat C)
{
  Mat_Product    *product = C->product;
  Mat            A=product->A,B=product->B;
  PetscReal      fill=product->fill;

  PetscFunctionBegin;
  CHKERRQ(MatTransposeMatMultSymbolic_SeqAIJ_SeqAIJ(A,B,fill,C));

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
  CHKERRQ(PetscStrcmp(C->product->alg,"default",&flg));
  if (flg) {
    CHKERRQ(MatProductSetAlgorithm(C,(MatProductAlgorithm)algTypes[alg]));
  }

  /* Get runtime option */
  if (product->api_user) {
    ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)C),((PetscObject)C)->prefix,"MatMatMult","Mat");CHKERRQ(ierr);
    CHKERRQ(PetscOptionsEList("-matmatmult_via","Algorithmic approach","MatMatMult",algTypes,nalg,algTypes[0],&alg,&flg));
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
  } else {
    ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)C),((PetscObject)C)->prefix,"MatProduct_AB","Mat");CHKERRQ(ierr);
    CHKERRQ(PetscOptionsEList("-mat_product_algorithm","Algorithmic approach","MatProduct_AB",algTypes,nalg,algTypes[0],&alg,&flg));
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
  }
  if (flg) {
    CHKERRQ(MatProductSetAlgorithm(C,(MatProductAlgorithm)algTypes[alg]));
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
  const char     *algTypes[3] = {"default","at*b","outerproduct"};
  PetscInt       nalg = 3;

  PetscFunctionBegin;
  /* Get runtime option */
  if (product->api_user) {
    ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)C),((PetscObject)C)->prefix,"MatTransposeMatMult","Mat");CHKERRQ(ierr);
    CHKERRQ(PetscOptionsEList("-mattransposematmult_via","Algorithmic approach","MatTransposeMatMult",algTypes,nalg,algTypes[alg],&alg,&flg));
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
  } else {
    ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)C),((PetscObject)C)->prefix,"MatProduct_AtB","Mat");CHKERRQ(ierr);
    CHKERRQ(PetscOptionsEList("-mat_product_algorithm","Algorithmic approach","MatProduct_AtB",algTypes,nalg,algTypes[alg],&alg,&flg));
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
  }
  if (flg) {
    CHKERRQ(MatProductSetAlgorithm(C,(MatProductAlgorithm)algTypes[alg]));
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
  CHKERRQ(PetscStrcmp(C->product->alg,"default",&flg));
  if (!flg) {
    alg = 1;
    CHKERRQ(MatProductSetAlgorithm(C,(MatProductAlgorithm)algTypes[alg]));
  }

  /* Get runtime option */
  if (product->api_user) {
    ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)C),((PetscObject)C)->prefix,"MatMatTransposeMult","Mat");CHKERRQ(ierr);
    CHKERRQ(PetscOptionsEList("-matmattransmult_via","Algorithmic approach","MatMatTransposeMult",algTypes,nalg,algTypes[alg],&alg,&flg));
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
  } else {
    ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)C),((PetscObject)C)->prefix,"MatProduct_ABt","Mat");CHKERRQ(ierr);
    CHKERRQ(PetscOptionsEList("-mat_product_algorithm","Algorithmic approach","MatProduct_ABt",algTypes,nalg,algTypes[alg],&alg,&flg));
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
  }
  if (flg) {
    CHKERRQ(MatProductSetAlgorithm(C,(MatProductAlgorithm)algTypes[alg]));
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
  const char     *algTypes[2] = {"scalable","rap"};
  PetscInt       nalg = 2;
#else
  const char     *algTypes[3] = {"scalable","rap","hypre"};
  PetscInt       nalg = 3;
#endif

  PetscFunctionBegin;
  /* Set default algorithm */
  CHKERRQ(PetscStrcmp(product->alg,"default",&flg));
  if (flg) {
    CHKERRQ(MatProductSetAlgorithm(C,(MatProductAlgorithm)algTypes[alg]));
  }

  /* Get runtime option */
  if (product->api_user) {
    ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)C),((PetscObject)C)->prefix,"MatPtAP","Mat");CHKERRQ(ierr);
    CHKERRQ(PetscOptionsEList("-matptap_via","Algorithmic approach","MatPtAP",algTypes,nalg,algTypes[0],&alg,&flg));
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
  } else {
    ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)C),((PetscObject)C)->prefix,"MatProduct_PtAP","Mat");CHKERRQ(ierr);
    CHKERRQ(PetscOptionsEList("-mat_product_algorithm","Algorithmic approach","MatProduct_PtAP",algTypes,nalg,algTypes[0],&alg,&flg));
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
  }
  if (flg) {
    CHKERRQ(MatProductSetAlgorithm(C,(MatProductAlgorithm)algTypes[alg]));
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
  CHKERRQ(PetscStrcmp(product->alg,"default",&flg));
  if (flg) {
    CHKERRQ(MatProductSetAlgorithm(C,(MatProductAlgorithm)algTypes[alg]));
  }

  /* Get runtime option */
  if (product->api_user) {
    ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)C),((PetscObject)C)->prefix,"MatRARt","Mat");CHKERRQ(ierr);
    CHKERRQ(PetscOptionsEList("-matrart_via","Algorithmic approach","MatRARt",algTypes,nalg,algTypes[0],&alg,&flg));
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
  } else {
    ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)C),((PetscObject)C)->prefix,"MatProduct_RARt","Mat");CHKERRQ(ierr);
    CHKERRQ(PetscOptionsEList("-mat_product_algorithm","Algorithmic approach","MatProduct_RARt",algTypes,nalg,algTypes[0],&alg,&flg));
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
  }
  if (flg) {
    CHKERRQ(MatProductSetAlgorithm(C,(MatProductAlgorithm)algTypes[alg]));
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
  CHKERRQ(PetscStrcmp(product->alg,"default",&flg));
  if (flg) {
    CHKERRQ(MatProductSetAlgorithm(C,(MatProductAlgorithm)algTypes[alg]));
  }

  /* Get runtime option */
  if (product->api_user) {
    ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)C),((PetscObject)C)->prefix,"MatMatMatMult","Mat");CHKERRQ(ierr);
    CHKERRQ(PetscOptionsEList("-matmatmatmult_via","Algorithmic approach","MatMatMatMult",algTypes,nalg,algTypes[alg],&alg,&flg));
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
  } else {
    ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)C),((PetscObject)C)->prefix,"MatProduct_ABC","Mat");CHKERRQ(ierr);
    CHKERRQ(PetscOptionsEList("-mat_product_algorithm","Algorithmic approach","MatProduct_ABC",algTypes,nalg,algTypes[alg],&alg,&flg));
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
  }
  if (flg) {
    CHKERRQ(MatProductSetAlgorithm(C,(MatProductAlgorithm)algTypes[alg]));
  }

  C->ops->matmatmultsymbolic = MatMatMatMultSymbolic_SeqAIJ_SeqAIJ_SeqAIJ;
  C->ops->productsymbolic    = MatProductSymbolic_ABC;
  PetscFunctionReturn(0);
}

PetscErrorCode MatProductSetFromOptions_SeqAIJ(Mat C)
{
  Mat_Product    *product = C->product;

  PetscFunctionBegin;
  switch (product->type) {
  case MATPRODUCT_AB:
    CHKERRQ(MatProductSetFromOptions_SeqAIJ_AB(C));
    break;
  case MATPRODUCT_AtB:
    CHKERRQ(MatProductSetFromOptions_SeqAIJ_AtB(C));
    break;
  case MATPRODUCT_ABt:
    CHKERRQ(MatProductSetFromOptions_SeqAIJ_ABt(C));
    break;
  case MATPRODUCT_PtAP:
    CHKERRQ(MatProductSetFromOptions_SeqAIJ_PtAP(C));
    break;
  case MATPRODUCT_RARt:
    CHKERRQ(MatProductSetFromOptions_SeqAIJ_RARt(C));
    break;
  case MATPRODUCT_ABC:
    CHKERRQ(MatProductSetFromOptions_SeqAIJ_ABC(C));
    break;
  default:
    break;
  }
  PetscFunctionReturn(0);
}
