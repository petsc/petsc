/*
  Defines matrix-matrix product routines for pairs of SeqAIJ matrices
          C = A * B
*/

#include "src/mat/impls/aij/seq/aij.h" /*I "petscmat.h" I*/
#include "src/mat/utils/freespace.h"
#include "src/mat/impls/aij/mpi/mpiaij.h"

/*
  Add a index set into a sorted linked list
  input:
    nidx      - number of input indices
    indices   - interger array
    idx_head  - the header of the list
    idx_unset - the value indicating the entry in the list is not set yet
    lnk       - linked list(an integer array) that is created
  output:
    nlnk      - number of newly added indices
    lnk       - the sorted(increasing order) linked list containing new and non-redundate entries from indices
 */
#undef __FUNCT__
#define __FUNCT__ "LnklistAdd"
int LnklistAdd(int nidx,int *indices,int idx_head,int idx_unset,int *nlnk,int *lnk)
{
  int i,idx,lidx,entry,n;

  PetscFunctionBegin;
  n    = 0;
  lidx = idx_head;
  i    = nidx;
  while (i){
    /* assume indices are almost in increasing order, starting from its end saves computation */
    entry = indices[--i];
    if (lnk[entry] == idx_unset) { /* new entry */   
      do { 
        idx = lidx;
        lidx  = lnk[idx];
      } while (entry > lidx);           
      lnk[idx] = entry;
      lnk[entry] = lidx;
      n++;
    }
  }
  *nlnk = n;
  PetscFunctionReturn(0);
}


static int logkey_matmatmult          = 0;
static int logkey_matmatmult_symbolic = 0;
static int logkey_matmatmult_numeric  = 0;

#undef __FUNCT__
#define __FUNCT__ "MatMatMult"
/*@
   MatMatMult - Performs Matrix-Matrix Multiplication C=A*B.

   Collective on Mat

   Input Parameters:
+  A - the left matrix
-  B - the right matrix

   Output Parameters:
.  C - the product matrix

   Notes:
   C will be created and must be destroyed by the user with MatDestroy().

   This routine is currently only implemented for pairs of SeqAIJ matrices and classes
   which inherit from SeqAIJ.  C will be of type MATSEQAIJ.

   Level: intermediate

.seealso: MatMatMultSymbolic(),MatMatMultNumeric()
@*/
int MatMatMult(Mat A,Mat B, Mat *C) 
{
  int  ierr;

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

  ierr = PetscLogEventBegin(MAT_MatMult,A,B,0,0);CHKERRQ(ierr); 
  ierr = (*A->ops->matmult)(A,B,C);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_MatMult,A,B,0,0);CHKERRQ(ierr); 
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMatMult_MPIAIJ_MPIAIJ"
int MatMatMult_MPIAIJ_MPIAIJ(Mat A,Mat B, Mat *C) 
{
  Mat           *aseq,*bseq,A_seq=PETSC_NULL,B_seq=PETSC_NULL,*cseq,C_seq,C_mpi;
  Mat_MPIAIJ    *a = (Mat_MPIAIJ*)A->data;
  Mat_SeqAIJ    *c;
  MatScalar     *ca;
  int           ierr,*idx,i,start,end,ncols,imark,nzA,nzB,*cmap,*ci,*cj,grow,*d_nnz,*o_nnz;
  IS            isrow,iscol;
 
  PetscFunctionBegin;
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
  ierr = ISCreateGeneral(PETSC_COMM_SELF,ncols,idx,&isrow);CHKERRQ(ierr); /* isrow of B = iscol of A! */
  ierr = PetscFree(idx);CHKERRQ(ierr); 
  ierr = ISCreateStride(PETSC_COMM_SELF,B->N,0,1,&iscol);CHKERRQ(ierr);
  ierr = MatGetSubMatrices(B,1,&isrow,&iscol,MAT_INITIAL_MATRIX,&bseq);CHKERRQ(ierr);
  ierr = ISDestroy(iscol);CHKERRQ(ierr);
  B_seq = bseq[0];
 
  /*  create a seq matrix A_seq = submatrix of A by taking all local rows of A */
  start = a->rstart; end = a->rend;
  ierr = ISCreateStride(PETSC_COMM_SELF,end-start,start,1,&iscol);CHKERRQ(ierr); /* isrow of A = iscol */
  ierr = MatGetSubMatrices(A,1,&iscol,&isrow,MAT_INITIAL_MATRIX,&aseq);CHKERRQ(ierr);
  ierr = ISDestroy(isrow);CHKERRQ(ierr);
  ierr = ISDestroy(iscol);CHKERRQ(ierr);
  A_seq = aseq[0];

  /* compute C_seq = A_seq * B_seq */
  ierr = MatMatMult_SeqAIJ_SeqAIJ(A_seq, B_seq, &C_seq);CHKERRQ(ierr);
  /*
  int rank;
  ierr = MPI_Comm_rank(A->comm,&rank);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF," [%d] A_seq: %d, %d; B_seq: %d, %d; C_seq: %d, %d;\n",rank,A_seq->m,A_seq->n,B_seq->m,B_seq->n,C_seq->m,C_seq->n); 
  */
  ierr = MatDestroyMatrices(1,&aseq);CHKERRQ(ierr); 
  ierr = MatDestroyMatrices(1,&bseq);CHKERRQ(ierr); 

  /* create a mpi matrix C_mpi that has C_seq as its local entries */
  ierr = MatCreate(A->comm,C_seq->m,PETSC_DECIDE,PETSC_DECIDE,C_seq->N,&C_mpi);CHKERRQ(ierr); 
  ierr = MatSetType(C_mpi,MATMPIAIJ);CHKERRQ(ierr);

  c  = (Mat_SeqAIJ*)C_seq->data;
  ci = c->i; cj = c->j; ca = c->a;
  ierr = PetscMalloc((2*C_seq->m+1)*sizeof(int),&d_nnz);CHKERRQ(ierr);
  o_nnz = d_nnz + C_seq->m;
  nzA   = end-start;    /* max nonezero cols of the local diagonal part of C_mpi */
  nzB   = C_seq->n-nzA; /* max nonezero cols of the local off-diagonal part of C_mpi */
  for (i=0; i< C_seq->m; i++){ 
    ncols = ci[i+1] - ci[i];
    d_nnz[i] = PetscMin(ncols,nzA);
    o_nnz[i] = PetscMin(ncols,nzB);
  }
  ierr = MatMPIAIJSetPreallocation(C_mpi,PETSC_DECIDE,d_nnz,PETSC_DECIDE,o_nnz);CHKERRQ(ierr); 
  ierr = PetscFree(d_nnz);CHKERRQ(ierr);

  /* set row values of C_mpi */
  for (i=0; i< C_seq->m; i++){
    grow  = start + i;
    ncols = ci[i+1] - ci[i];
    ierr = MatSetValues(C_mpi,1,&grow,ncols,cj+ci[i],ca+ci[i],INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(C_mpi,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C_mpi,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatDestroy(C_seq);CHKERRQ(ierr);

  *C = C_mpi;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMatMult_SeqAIJ_SeqAIJ"
int MatMatMult_SeqAIJ_SeqAIJ(Mat A,Mat B, Mat *C) {
  int ierr;
  char symfunct[80],numfunct[80];
  int (*symbolic)(Mat,Mat,Mat*),(*numeric)(Mat,Mat,Mat);

  PetscFunctionBegin;
  /* Currently only _seqaijseqaij is implemented, so just query for it in A and B. */
  /* When other implementations exist, attack the multiple dispatch problem. */
  ierr = PetscStrcpy(symfunct,"MatMatMultSymbolic_seqaijseqaij");CHKERRQ(ierr);
  ierr = PetscObjectQueryFunction((PetscObject)B,symfunct,(PetscVoidFunction)&symbolic);CHKERRQ(ierr);
  if (!symbolic) SETERRQ1(PETSC_ERR_SUP,"C=A*B not implemented for B of type %s",B->type_name);
  ierr = PetscObjectQueryFunction((PetscObject)A,symfunct,(PetscVoidFunction)&symbolic);CHKERRQ(ierr);
  if (!symbolic) SETERRQ1(PETSC_ERR_SUP,"C=A*B not implemented for A of type %s",A->type_name);

  ierr = PetscStrcpy(numfunct,"MatMatMultNumeric_seqaijseqaij");CHKERRQ(ierr);
  ierr = PetscObjectQueryFunction((PetscObject)A,numfunct,(PetscVoidFunction)&numeric);CHKERRQ(ierr);
  if (!numeric) SETERRQ1(PETSC_ERR_SUP,"C=A*B not implemented for A of type %s",A->type_name);
  ierr = PetscObjectQueryFunction((PetscObject)B,numfunct,(PetscVoidFunction)&numeric);CHKERRQ(ierr);
  if (!numeric) SETERRQ1(PETSC_ERR_SUP,"C=A*B not implemented for B of type %s",B->type_name);

  ierr = PetscLogEventBegin(logkey_matmatmult,A,B,0,0);CHKERRQ(ierr);
  ierr = (*symbolic)(A,B,C);CHKERRQ(ierr);
  ierr = (*numeric)(A,B,*C);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(logkey_matmatmult,A,B,0,0);CHKERRQ(ierr);
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
-  B - the right matrix

   Output Parameters:
.  C - the matrix containing the ij structure of product matrix

   Notes:
   C will be created as a MATSEQAIJ matrix and must be destroyed by the user with MatDestroy().

   This routine is currently only implemented for SeqAIJ matrices and classes which inherit from SeqAIJ.

   Level: intermediate

.seealso: MatMatMult(),MatMatMultNumeric()
@*/
int MatMatMultSymbolic(Mat A,Mat B,Mat *C) {
  /* Perhaps this "interface" routine should be moved into the interface directory.*/
  /* To facilitate implementations with varying types, QueryFunction is used.*/
  /* It is assumed that implementations will be composed as "MatMatMultSymbolic_<type of A><type of B>". */
  int  ierr;
  char symfunct[80];
  int  (*symbolic)(Mat,Mat,Mat *);

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

  /* Currently only _seqaijseqaij is implemented, so just query for it in A and B. */
  /* When other implementations exist, attack the multiple dispatch problem. */
  ierr = PetscStrcpy(symfunct,"MatMatMultSymbolic_seqaijseqaij");CHKERRQ(ierr);
  ierr = PetscObjectQueryFunction((PetscObject)B,symfunct,(PetscVoidFunction)&symbolic);CHKERRQ(ierr);
  if (!symbolic) SETERRQ1(PETSC_ERR_SUP,"C=A*B not implemented for B of type %s",B->type_name);
  ierr = PetscObjectQueryFunction((PetscObject)A,symfunct,(PetscVoidFunction)&symbolic);CHKERRQ(ierr);
  if (!symbolic) SETERRQ1(PETSC_ERR_SUP,"C=A*B not implemented for A of type %s",A->type_name);
  ierr = (*symbolic)(A,B,C);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMatMult_Symbolic_SeqAIJ_SeqAIJ"
int MatMatMult_Symbolic_SeqAIJ_SeqAIJ(Mat A,Mat B,Mat *C)
{
  int            ierr;
  FreeSpaceList  free_space=PETSC_NULL,current_space=PETSC_NULL;
  Mat_SeqAIJ     *a=(Mat_SeqAIJ*)A->data,*b=(Mat_SeqAIJ*)B->data,*c;
  int            *ai=a->i,*aj=a->j,*bi=b->i,*bj=b->j,*bjj;
  int            *ci,*cj,*lnk,idx0,idx;
  int            am=A->M,bn=B->N,bm=B->M;
  int            i,j,anzi,brow,bnzj,cnzi,nlnk;
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
  for (i=0; i<bn; i++) lnk[i] = -1;

  /* Initial FreeSpace size is nnz(B)=4*bi[bm] */
  ierr = GetMoreSpace(4*bi[bm],&free_space);CHKERRQ(ierr);
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
      ierr = LnklistAdd(bnzj,bjj,bn,-1,&nlnk,lnk);CHKERRQ(ierr);
      cnzi += nlnk;
    }

    /* If free space is not available, make more free space */
    /* Double the amount of total space in the list */
    if (current_space->local_remaining<cnzi) {
      printf("...%d -th row, double space ...\n",i);
      ierr = GetMoreSpace(current_space->total_array_size,&current_space);CHKERRQ(ierr);
    }

    /* Copy data into free space, and zero out denserow and lnk */
    idx = bn;
    for (j=0; j<cnzi; j++){
      idx0 = idx;
      idx  = lnk[idx0];     
      *current_space->array++ = idx; 
      lnk[idx0] = -1;
    }
    lnk[idx] = -1;

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
int MatMatMultNumeric(Mat A,Mat B,Mat C){
  /* Perhaps this "interface" routine should be moved into the interface directory.*/
  /* To facilitate implementations with varying types, QueryFunction is used.*/
  /* It is assumed that implementations will be composed as "MatMatMultNumeric_<type of A><type of B>". */
  int ierr;
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

#undef __FUNCT__  
#define __FUNCT__ "MatMatMult_Numeric_SeqAIJ_SeqAIJ"
int MatMatMult_Numeric_SeqAIJ_SeqAIJ(Mat A,Mat B,Mat C)
{
  int        ierr,flops=0;
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
int RegisterMatMatMultRoutines_Private(Mat A) {
  int ierr;

  PetscFunctionBegin;
#ifdef OLD
  if (!logkey_matmatmult) {
    ierr = PetscLogEventRegister(&logkey_matmatmult,"MatMatMult",MAT_COOKIE);CHKERRQ(ierr);
  }
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)A,"MatMatMult_seqaijseqaij",
                                           "MatMatMult_SeqAIJ_SeqAIJ",
                                           MatMatMult_SeqAIJ_SeqAIJ);CHKERRQ(ierr);
#endif
  if (!logkey_matmatmult_symbolic) {
    ierr = PetscLogEventRegister(&logkey_matmatmult_symbolic,"MatMatMult_Symbolic",MAT_COOKIE);CHKERRQ(ierr);
  }
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)A,"MatMatMultSymbolic_seqaijseqaij",
                                           "MatMatMult_Symbolic_SeqAIJ_SeqAIJ",
                                           MatMatMult_Symbolic_SeqAIJ_SeqAIJ);CHKERRQ(ierr);
  if (!logkey_matmatmult_numeric) {
    ierr = PetscLogEventRegister(&logkey_matmatmult_numeric,"MatMatMult_Numeric",MAT_COOKIE);CHKERRQ(ierr);
  }
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)A,"MatMatMultNumeric_seqaijseqaij",
                                           "MatMatMult_Numeric_SeqAIJ_SeqAIJ",
                                           MatMatMult_Numeric_SeqAIJ_SeqAIJ);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
