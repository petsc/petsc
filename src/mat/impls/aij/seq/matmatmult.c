/*
  Defines matrix-matrix product routines for pairs of AIJ matrices
          C = A * B
*/

#include "src/mat/impls/aij/seq/aij.h" /*I "petscmat.h" I*/
#include "src/mat/utils/freespace.h"
#include "src/mat/impls/aij/mpi/mpiaij.h"

typedef struct { /* used by MatMatMult_MPIAIJ_MPIAIJ for reusing symbolic mat product */
  IS     isrowa,isrowb,iscolb;
  Mat    *aseq,*bseq,C_seq;
} Mat_MatMatMultMPI;

static int logkey_matmatmult_symbolic = 0;
static int logkey_matmatmult_numeric  = 0;

#undef __FUNCT__
#define __FUNCT__ "MatMatMult"
/*@
   MatMatMult - Performs Matrix-Matrix Multiplication C=A*B.

   Collective on Mat

   Input Parameters:
+  A - the left matrix
.  B - the right matrix
.  scall - either MAT_INITIAL_MATRIX or MAT_REUSE_MATRIX
-  fill - expected fill as ratio of nnz(C)/(nnz(A) + nnz(B))

   Output Parameters:
.  C - the product matrix

   Notes:
   C will be created and must be destroyed by the user with MatDestroy().

   This routine is currently only implemented for pairs of SeqAIJ matrices and classes
   which inherit from SeqAIJ.  C will be of type MATSEQAIJ.

   Level: intermediate

.seealso: MatMatMultSymbolic(),MatMatMultNumeric()
@*/
PetscErrorCode MatMatMult(Mat A,Mat B,MatReuse scall,PetscReal fill,Mat *C) 
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_COOKIE,1);
  PetscValidType(A,1);
  MatPreallocated(A);
  if (!A->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (A->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  PetscValidHeaderSpecific(B,MAT_COOKIE,2);
  PetscValidType(B,2);
  MatPreallocated(B);
  if (!B->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (B->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  PetscValidPointer(C,3);
  if (B->M!=A->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, %d != %d",B->M,A->N);

  if (fill <=0.0) SETERRQ1(PETSC_ERR_ARG_SIZ,"fill=%g must be > 0.0",fill);

  ierr = PetscLogEventBegin(MAT_MatMult,A,B,0,0);CHKERRQ(ierr); 
  ierr = (*A->ops->matmult)(A,B,scall,fill,C);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_MatMult,A,B,0,0);CHKERRQ(ierr); 
  
  PetscFunctionReturn(0);
} 

#undef __FUNCT__
#define __FUNCT__ "MatMatMult_MPIAIJ_MPIAIJ"
PetscErrorCode MatMatMult_MPIAIJ_MPIAIJ(Mat A,Mat B,MatReuse scall,PetscReal fill, Mat *C) 
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (scall == MAT_INITIAL_MATRIX){ 
    ierr = MatMatMultSymbolic_MPIAIJ_MPIAIJ(A,B,fill,C);CHKERRQ(ierr);/* numeric product is computed as well */
  } else if (scall == MAT_REUSE_MATRIX){
    ierr = MatMatMultNumeric_MPIAIJ_MPIAIJ(A,B,*C);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_ERR_ARG_WRONG,"Invalid MatReuse %d",scall);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMatMult_SeqAIJ_SeqAIJ"
PetscErrorCode MatMatMult_SeqAIJ_SeqAIJ(Mat A,Mat B,MatReuse scall,PetscReal fill,Mat *C) {
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (scall == MAT_INITIAL_MATRIX){
    ierr = MatMatMultSymbolic_SeqAIJ_SeqAIJ(A,B,fill,C);CHKERRQ(ierr);
  }
  ierr = MatMatMultNumeric_SeqAIJ_SeqAIJ(A,B,*C);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMatMultSymbolic"
/*@
   MatMatMultSymbolic - Performs construction, preallocation, and computes the ij structure
   of the matrix-matrix product C=A*B.  Call this routine before calling MatMatMultNumeric().

   Collective on Mat

   Input Parameters:
+  A - the left matrix
.  B - the right matrix
-  fill - expected fill as ratio of nnz(C)/(nnz(A) + nnz(B))

   Output Parameters:
.  C - the matrix containing the ij structure of product matrix

   Notes:
   C will be created as a MATSEQAIJ matrix and must be destroyed by the user with MatDestroy().

   This routine is currently only implemented for SeqAIJ matrices and classes which inherit from SeqAIJ.

   Level: intermediate

.seealso: MatMatMult(),MatMatMultNumeric()
@*/
PetscErrorCode MatMatMultSymbolic(Mat A,Mat B,PetscReal fill,Mat *C) {
  /* Perhaps this "interface" routine should be moved into the interface directory.*/
  /* To facilitate implementations with varying types, QueryFunction is used.*/
  /* It is assumed that implementations will be composed as "MatMatMultSymbolic_<type of A><type of B>". */
  PetscErrorCode ierr;
  char symfunct[80];
  int  (*symbolic)(Mat,Mat,PetscReal,Mat *);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_COOKIE,1);
  PetscValidType(A,1);
  MatPreallocated(A);
  if (!A->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (A->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 

  PetscValidHeaderSpecific(B,MAT_COOKIE,2);
  PetscValidType(B,2);
  MatPreallocated(B);
  if (!B->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (B->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 
  PetscValidPointer(C,3);

  if (B->M!=A->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, %d != %d",B->M,A->N);
  if (fill <=0.0) SETERRQ1(PETSC_ERR_ARG_SIZ,"fill=%g must be > 0.0",fill);

  /* Currently only _seqaijseqaij is implemented, so just query for it in A and B. */
  /* When other implementations exist, attack the multiple dispatch problem. */
  ierr = PetscStrcpy(symfunct,"MatMatMultSymbolic_seqaijseqaij");CHKERRQ(ierr);
  ierr = PetscObjectQueryFunction((PetscObject)B,symfunct,(PetscVoidFunction)&symbolic);CHKERRQ(ierr);
  if (!symbolic) SETERRQ1(PETSC_ERR_SUP,"C=A*B not implemented for B of type %s",B->type_name);
  ierr = PetscObjectQueryFunction((PetscObject)A,symfunct,(PetscVoidFunction)&symbolic);CHKERRQ(ierr);
  if (!symbolic) SETERRQ1(PETSC_ERR_SUP,"C=A*B not implemented for A of type %s",A->type_name);
  ierr = (*symbolic)(A,B,fill,C);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

EXTERN PetscErrorCode MatDestroy_MPIAIJ(Mat);
#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_MPIAIJ_MatMatMult"
PetscErrorCode MatDestroy_MPIAIJ_MatMatMult(Mat A)
{
  PetscErrorCode ierr;
  Mat_MatMatMultMPI *mult=(Mat_MatMatMultMPI*)A->spptr; 

  PetscFunctionBegin;
  ierr = ISDestroy(mult->isrowb);CHKERRQ(ierr);
  ierr = ISDestroy(mult->iscolb);CHKERRQ(ierr);
  ierr = ISDestroy(mult->isrowa);CHKERRQ(ierr);
  ierr = MatDestroyMatrices(1,&mult->aseq);CHKERRQ(ierr); 
  ierr = MatDestroyMatrices(1,&mult->bseq);CHKERRQ(ierr); 
  ierr = MatDestroy(mult->C_seq);CHKERRQ(ierr); 
  ierr = PetscFree(mult);CHKERRQ(ierr); 

  ierr = MatDestroy_MPIAIJ(A);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMatMultSymbolic_MPIAIJ_MPIAIJ"
PetscErrorCode MatMatMultSymbolic_MPIAIJ_MPIAIJ(Mat A,Mat B,PetscReal fill,Mat *C)
{
  Mat_MPIAIJ        *a = (Mat_MPIAIJ*)A->data;
  PetscErrorCode ierr;
  int               *idx,i,start,end,ncols,imark,nzA,nzB,*cmap;
  Mat_MatMatMultMPI *mult;
 
  PetscFunctionBegin;
  ierr = PetscNew(Mat_MatMatMultMPI,&mult);CHKERRQ(ierr);

  /* create a seq matrix B_seq = submatrix of B by taking rows of B that equal to nonzero col of A */
  start = a->cstart;
  cmap  = a->garray;
  nzA   = a->A->n; 
  nzB   = a->B->n;
  ierr  = PetscMalloc((nzA+nzB)*sizeof(int), &idx);CHKERRQ(ierr);
  ncols = 0;
  for (i=0; i<nzB; i++) {
    if (cmap[i] < start) idx[ncols++] = cmap[i];
    else break;
  }
  imark = i;
  for (i=0; i<nzA; i++) idx[ncols++] = start + i;
  for (i=imark; i<nzB; i++) idx[ncols++] = cmap[i];
  ierr = ISCreateGeneral(PETSC_COMM_SELF,ncols,idx,&mult->isrowb);CHKERRQ(ierr);
  ierr = PetscFree(idx);CHKERRQ(ierr); 
  ierr = ISCreateStride(PETSC_COMM_SELF,B->N,0,1,&mult->iscolb);CHKERRQ(ierr);
  ierr = MatGetSubMatrices(B,1,&mult->isrowb,&mult->iscolb,MAT_INITIAL_MATRIX,&mult->bseq);CHKERRQ(ierr)
 
  /*  create a seq matrix A_seq = submatrix of A by taking all local rows of A */
  start = a->rstart; end = a->rend;
  ierr = ISCreateStride(PETSC_COMM_SELF,end-start,start,1,&mult->isrowa);CHKERRQ(ierr); 
  ierr = MatGetSubMatrices(A,1,&mult->isrowa,&mult->isrowb,MAT_INITIAL_MATRIX,&mult->aseq);CHKERRQ(ierr); 

  /* compute C_seq = A_seq * B_seq */
  ierr = MatMatMult_SeqAIJ_SeqAIJ(mult->aseq[0],mult->bseq[0],MAT_INITIAL_MATRIX,fill,&mult->C_seq);CHKERRQ(ierr);

  /* create mpi matrix C by concatinating C_seq */
  ierr = PetscObjectReference((PetscObject)mult->C_seq);CHKERRQ(ierr); /* prevent C_seq being destroyed by MatMerge() */
  ierr = MatMerge(A->comm,mult->C_seq,MAT_INITIAL_MATRIX,C);CHKERRQ(ierr); 
 
  /* attach the supporting struct to C for reuse of symbolic C */
  (*C)->spptr         = (void*)mult;
  (*C)->ops->destroy  = MatDestroy_MPIAIJ_MatMatMult; 
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMatMultSymbolic_SeqAIJ_SeqAIJ"
PetscErrorCode MatMatMultSymbolic_SeqAIJ_SeqAIJ(Mat A,Mat B,PetscReal fill,Mat *C)
{
  PetscErrorCode ierr;
  FreeSpaceList  free_space=PETSC_NULL,current_space=PETSC_NULL;
  Mat_SeqAIJ     *a=(Mat_SeqAIJ*)A->data,*b=(Mat_SeqAIJ*)B->data,*c;
  int            *ai=a->i,*aj=a->j,*bi=b->i,*bj=b->j,*bjj;
  int            *ci,*cj,*lnk;
  int            am=A->M,bn=B->N,bm=B->M;
  int            i,j,anzi,brow,bnzj,cnzi,nlnk,lnk_init=-1,nspacedouble=0;
  MatScalar      *ca;

  PetscFunctionBegin;
  /* Start timers */
  ierr = PetscLogEventBegin(logkey_matmatmult_symbolic,A,B,0,0);CHKERRQ(ierr);
  /* Set up */
  /* Allocate ci array, arrays for fill computation and */
  /* free space for accumulating nonzero column info */
  ierr = PetscMalloc(((am+1)+1)*sizeof(int),&ci);CHKERRQ(ierr);
  ci[0] = 0;
  
  ierr = PetscMalloc((bn+1)*sizeof(int),&lnk);CHKERRQ(ierr);
  nlnk = bn+1;
  ierr = PetscLLInitialize(lnk_init,nlnk,lnk);CHKERRQ(ierr);

  /* Initial FreeSpace size is fill*(nnz(A)+nnz(B)) */
  ierr = GetMoreSpace((int)(fill*(ai[am]+bi[bm])),&free_space);CHKERRQ(ierr);
  current_space = free_space;

  /* Determine symbolic info for each row of the product: */
  for (i=0;i<am;i++) {
    anzi = ai[i+1] - ai[i];
    cnzi = 0;
    lnk[bn] = bn;
    j       = anzi;
    aj      = a->j + ai[i];
    while (j){/* assume cols are almost in increasing order, starting from its end saves computation */
      j--;
      brow = *(aj + j);
      bnzj = bi[brow+1] - bi[brow];
      bjj  = bj + bi[brow];
      /* add non-zero cols of B into the sorted linked list lnk */
      ierr = PetscLLAdd(bnzj,bjj,bn,lnk_init,nlnk,lnk);CHKERRQ(ierr);
      cnzi += nlnk;
    }

    /* If free space is not available, make more free space */
    /* Double the amount of total space in the list */
    if (current_space->local_remaining<cnzi) {
      /* printf("MatMatMultSymbolic_SeqAIJ_SeqAIJ()...%d -th row, double space ...\n",i);*/
      ierr = GetMoreSpace(current_space->total_array_size,&current_space);CHKERRQ(ierr);
      nspacedouble++;
    }

    /* Copy data into free space, then initialize lnk */
    ierr = PetscLLClear(bn,lnk_init,cnzi,lnk,current_space->array);CHKERRQ(ierr);
    current_space->array += cnzi;

    current_space->local_used      += cnzi;
    current_space->local_remaining -= cnzi;

    ci[i+1] = ci[i] + cnzi;
  }

  /* Column indices are in the list of free space */
  /* Allocate space for cj, initialize cj, and */
  /* destroy list of free space and other temporary array(s) */
  ierr = PetscMalloc((ci[am]+1)*sizeof(int),&cj);CHKERRQ(ierr);
  ierr = MakeSpaceContiguous(&free_space,cj);CHKERRQ(ierr);
  ierr = PetscFree(lnk);CHKERRQ(ierr);
    
  /* Allocate space for ca */
  ierr = PetscMalloc((ci[am]+1)*sizeof(MatScalar),&ca);CHKERRQ(ierr);
  ierr = PetscMemzero(ca,(ci[am]+1)*sizeof(MatScalar));CHKERRQ(ierr);
  
  /* put together the new symbolic matrix */
  ierr = MatCreateSeqAIJWithArrays(A->comm,am,bn,ci,cj,ca,C);CHKERRQ(ierr);

  /* MatCreateSeqAIJWithArrays flags matrix so PETSc doesn't free the user's arrays. */
  /* These are PETSc arrays, so change flags so arrays can be deleted by PETSc */
  c = (Mat_SeqAIJ *)((*C)->data);
  c->freedata = PETSC_TRUE;
  c->nonew    = 0;

  ierr = PetscLogEventEnd(logkey_matmatmult_symbolic,A,B,0,0);CHKERRQ(ierr);
  PetscLogInfo((PetscObject)(*C),"Number of calls to GetMoreSpace(): %d\n",nspacedouble);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMatMultNumeric"
/*@
   MatMatMultNumeric - Performs the numeric matrix-matrix product.
   Call this routine after first calling MatMatMultSymbolic().

   Collective on Mat

   Input Parameters:
+  A - the left matrix
-  B - the right matrix

   Output Parameters:
.  C - the product matrix, whose ij structure was defined from MatMatMultSymbolic().

   Notes:
   C must have been created with MatMatMultSymbolic.

   This routine is currently only implemented for SeqAIJ type matrices.

   Level: intermediate

.seealso: MatMatMult(),MatMatMultSymbolic()
@*/
PetscErrorCode MatMatMultNumeric(Mat A,Mat B,Mat C){
  /* Perhaps this "interface" routine should be moved into the interface directory.*/
  /* To facilitate implementations with varying types, QueryFunction is used.*/
  /* It is assumed that implementations will be composed as "MatMatMultNumeric_<type of A><type of B>". */
  PetscErrorCode ierr;
  char numfunct[80];
  int (*numeric)(Mat,Mat,Mat);

  PetscFunctionBegin;

  PetscValidHeaderSpecific(A,MAT_COOKIE,1);
  PetscValidType(A,1);
  MatPreallocated(A);
  if (!A->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (A->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 

  PetscValidHeaderSpecific(B,MAT_COOKIE,2);
  PetscValidType(B,2);
  MatPreallocated(B);
  if (!B->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (B->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 

  PetscValidHeaderSpecific(C,MAT_COOKIE,3);
  PetscValidType(C,3);
  MatPreallocated(C);
  if (!C->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (C->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 

  if (B->N!=C->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, %d != %d",B->N,C->N);
  if (B->M!=A->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, %d != %d",B->M,A->N);
  if (A->M!=C->M) SETERRQ2(PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, %d != %d",A->M,C->M);

  /* Currently only _seqaijseqaij is implemented, so just query for it in A and B. */
  /* When other implementations exist, attack the multiple dispatch problem. */
  ierr = PetscStrcpy(numfunct,"MatMatMultNumeric_seqaijseqaij");CHKERRQ(ierr);
  ierr = PetscObjectQueryFunction((PetscObject)A,numfunct,(PetscVoidFunction)&numeric);CHKERRQ(ierr);
  if (!numeric) SETERRQ1(PETSC_ERR_SUP,"C=A*B not implemented for A of type %s",A->type_name);
  ierr = PetscObjectQueryFunction((PetscObject)B,numfunct,(PetscVoidFunction)&numeric);CHKERRQ(ierr);
  if (!numeric) SETERRQ1(PETSC_ERR_SUP,"C=A*B not implemented for B of type %s",B->type_name);

  ierr = (*numeric)(A,B,C);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* This routine is called ONLY in the case of reusing previously computed symbolic C */
#undef __FUNCT__  
#define __FUNCT__ "MatMatMultNumeric_MPIAIJ_MPIAIJ"
PetscErrorCode MatMatMultNumeric_MPIAIJ_MPIAIJ(Mat A,Mat B,Mat C)
{
  PetscErrorCode ierr;
  Mat_MatMatMultMPI *mult=(Mat_MatMatMultMPI*)C->spptr;

  PetscFunctionBegin;
  ierr = MatGetSubMatrices(B,1,&mult->isrowb,&mult->iscolb,MAT_REUSE_MATRIX,&mult->bseq);CHKERRQ(ierr)
  ierr = MatGetSubMatrices(A,1,&mult->isrowa,&mult->isrowb,MAT_REUSE_MATRIX,&mult->aseq);CHKERRQ(ierr);
  ierr = MatMatMult_SeqAIJ_SeqAIJ(mult->aseq[0],mult->bseq[0],MAT_REUSE_MATRIX,0.0,&mult->C_seq);CHKERRQ(ierr);

  ierr = PetscObjectReference((PetscObject)mult->C_seq);CHKERRQ(ierr); 
  ierr = MatMerge(A->comm,mult->C_seq,MAT_REUSE_MATRIX,&C);CHKERRQ(ierr); 

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMatMultNumeric_SeqAIJ_SeqAIJ"
PetscErrorCode MatMatMultNumeric_SeqAIJ_SeqAIJ(Mat A,Mat B,Mat C)
{
  PetscErrorCode ierr;
  int        flops=0;
  Mat_SeqAIJ *a = (Mat_SeqAIJ *)A->data;
  Mat_SeqAIJ *b = (Mat_SeqAIJ *)B->data;
  Mat_SeqAIJ *c = (Mat_SeqAIJ *)C->data;
  int        *ai=a->i,*aj=a->j,*bi=b->i,*bj=b->j,*bjj,*ci=c->i,*cj=c->j;
  int        am=A->M,cn=C->N;
  int        i,j,k,anzi,bnzi,cnzi,brow;
  MatScalar  *aa=a->a,*ba=b->a,*baj,*ca=c->a,*temp;

  PetscFunctionBegin;  

  /* Start timers */
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
#define __FUNCT__ "RegisterMatMatMultRoutines_Private"
PetscErrorCode RegisterMatMatMultRoutines_Private(Mat A) 
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!logkey_matmatmult_symbolic) {
    ierr = PetscLogEventRegister(&logkey_matmatmult_symbolic,"MatMatMultSymbolic",MAT_COOKIE);CHKERRQ(ierr);
  }
  if (!logkey_matmatmult_numeric) {
    ierr = PetscLogEventRegister(&logkey_matmatmult_numeric,"MatMatMultNumeric",MAT_COOKIE);CHKERRQ(ierr);
  }
  
  PetscFunctionReturn(0);
}
