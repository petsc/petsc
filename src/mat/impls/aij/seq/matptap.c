/*
  Defines projective product routines where A is a SeqAIJ matrix
          C = P^T * A * P
*/

#include "src/mat/impls/aij/seq/aij.h"   /*I "petscmat.h" I*/
#include "src/mat/utils/freespace.h"

EXTERN int MatSeqAIJPtAP(Mat,Mat,Mat*);
EXTERN int MatSeqAIJPtAPSymbolic(Mat,Mat,Mat*);
EXTERN int MatSeqAIJPtAPNumeric(Mat,Mat,Mat);
EXTERN int RegisterMatMatMultRoutines_Private(Mat);

static int MATSeqAIJ_PtAP         = 0;
static int MATSeqAIJ_PtAPSymbolic = 0;
static int MATSeqAIJ_PtAPNumeric  = 0;

#undef __FUNCT__
#define __FUNCT__ "MatSeqAIJPtAP"
/*@
   MatSeqAIJPtAP - Creates the matrix projection C = P^T * A * P

   Collective on Mat

   Input Parameters:
+  A - the matrix
-  P - the projection matrix

   Output Parameters:
.  C - the product matrix

   Notes:
   C will be created and must be destroyed by the user with MatDestroy().

   This routine is currently only implemented for pairs of SeqAIJ matrices and classes
   which inherit from SeqAIJ.  C will be of type MATSEQAIJ.

   Level: intermediate

.seealso: MatSeqAIJPtAPSymbolic(),MatSeqAIJPtAPNumeric(),MatMatMult()
@*/
int MatSeqAIJPtAP(Mat A,Mat P,Mat *C) {
  int ierr;
  char funct[80];
  int (*f)(Mat,Mat,Mat);

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(MATSeqAIJ_PtAP,A,P,0,0);CHKERRQ(ierr);

  ierr = MatSeqAIJPtAPSymbolic(A,P,C);CHKERRQ(ierr);

  /* Avoid additional error checking included in */
/*   ierr = MatSeqAIJApplyPtAPNumeric(A,P,*C);CHKERRQ(ierr); */

  /* Currently only _seqaij_seqaij is implemented, so just query for it in A and P. */
  /* When other implementations exist, attack the multiple dispatch problem. */
  ierr = PetscStrcpy(funct,"MatApplyPtAPNumeric_seqaij_seqaij");CHKERRQ(ierr);
  ierr = PetscObjectQueryFunction((PetscObject)P,funct,(PetscVoidFunction)&f);CHKERRQ(ierr);
  if (!f) SETERRQ1(1,"MatSeqAIJPtAPNumeric is not supported for P of type %s",P->type_name);
  ierr = PetscObjectQueryFunction((PetscObject)A,funct,(PetscVoidFunction)&f);CHKERRQ(ierr);
  if (!f) SETERRQ1(1,"MatSeqAIJPtAPNumeric is not supported for A of type %s",A->type_name);

  ierr = (*f)(A,P,*C);CHKERRQ(ierr);
    
  ierr = PetscLogEventEnd(MATSeqAIJ_PtAP,A,P,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSeqAIJPtAPSymbolic"
/*@
   MatSeqAIJPtAPSymbolic - Creates the (i,j) structure of the matrix projection C = P^T * A * P

   Collective on Mat

   Input Parameters:
+  A - the matrix
-  P - the projection matrix

   Output Parameters:
.  C - the (i,j) structure of the product matrix

   Notes:
   C will be created and must be destroyed by the user with MatDestroy().

   This routine is currently only implemented for pairs of SeqAIJ matrices and classes
   which inherit from SeqAIJ.  C will be of type MATSEQAIJ.  The product is computed using
   this (i,j) structure by calling MatSeqAIJPtAPNumeric().

   Level: intermediate

.seealso: MatSeqAIJPtAP(),MatSeqAIJPtAPNumeric(),MatMatMultSymbolic()
@*/
int MatSeqAIJPtAPSymbolic(Mat A,Mat P,Mat *C) {
  int ierr;
  char funct[80];
  int (*f)(Mat,Mat,Mat*);

  PetscFunctionBegin;

  PetscValidHeaderSpecific(A,MAT_COOKIE,1);
  PetscValidType(A,1);
  MatPreallocated(A);
  if (!A->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (A->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 

  PetscValidHeaderSpecific(P,MAT_COOKIE,2);
  PetscValidType(P,2);
  MatPreallocated(P);
  if (!P->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (P->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 

  PetscValidPointer(C,3);

  if (P->M!=A->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, %d != %d",P->M,A->N);
  if (A->M!=A->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Matrix 'A' must be square, %d != %d",A->M,A->N);

  /* Currently only _seqaij_seqaij is implemented, so just query for it. */
  /* When other implementations exist, attack the multiple dispatch problem. */
  ierr = PetscStrcpy(funct,"MatApplyPtAPSymbolic_seqaij_seqaij");CHKERRQ(ierr);
  ierr = PetscObjectQueryFunction((PetscObject)P,funct,(PetscVoidFunction)&f);CHKERRQ(ierr);
  if (!f) SETERRQ1(1,"MatSeqAIJPtAPSymbolic is not supported for P of type %s",P->type_name);
  ierr = PetscObjectQueryFunction((PetscObject)A,funct,(PetscVoidFunction)&f);CHKERRQ(ierr);
  if (!f) SETERRQ1(1,"MatSeqAIJPtAPSymbolic is not supported for A of type %s",A->type_name);

  ierr = (*f)(A,P,C);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatApplyPtAPSymbolic_SeqAIJ_SeqAIJ"
int MatApplyPtAPSymbolic_SeqAIJ_SeqAIJ(Mat A,Mat P,Mat *C) {
  int            ierr;
  FreeSpaceList  free_space=PETSC_NULL,current_space=PETSC_NULL;
  Mat_SeqAIJ     *a=(Mat_SeqAIJ*)A->data,*p=(Mat_SeqAIJ*)P->data,*c;
  int            *pti,*ptj,*ptJ,*ai=a->i,*aj=a->j,*ajj,*pi=p->i,*pj=p->j,*pjj;
  int            *ci,*cj,*denserow,*sparserow,*ptadenserow,*ptasparserow,*ptaj;
  int            an=A->N,am=A->M,pn=P->N,pm=P->M;
  int            i,j,k,ptnzi,arow,anzj,ptanzi,prow,pnzj,cnzi;
  MatScalar      *ca;

  PetscFunctionBegin;

  /* Start timer */
  ierr = PetscLogEventBegin(MATSeqAIJ_PtAPSymbolic,A,P,0,0);CHKERRQ(ierr);

  /* Get ij structure of P^T */
  ierr = MatGetSymbolicTranspose_SeqAIJ(P,&pti,&ptj);CHKERRQ(ierr);
  ptJ=ptj;

  /* Allocate ci array, arrays for fill computation and */
  /* free space for accumulating nonzero column info */
  ierr = PetscMalloc((pn+1)*sizeof(int),&ci);CHKERRQ(ierr);
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

  ierr = PetscLogEventEnd(MATSeqAIJ_PtAPSymbolic,A,P,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#include "src/mat/impls/maij/maij.h"
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatApplyPtAPSymbolic_SeqAIJ_SeqMAIJ"
int MatApplyPtAPSymbolic_SeqAIJ_SeqMAIJ(Mat A,Mat PP,Mat *C) {
  /* This routine requires testing -- I don't think it works. */
  int            ierr;
  FreeSpaceList  free_space=PETSC_NULL,current_space=PETSC_NULL;
  Mat_SeqMAIJ    *pp=(Mat_SeqMAIJ*)PP->data;
  Mat            P=pp->AIJ;
  Mat_SeqAIJ     *a=(Mat_SeqAIJ*)A->data,*p=(Mat_SeqAIJ*)P->data,*c;
  int            *pti,*ptj,*ptJ,*ai=a->i,*aj=a->j,*ajj,*pi=p->i,*pj=p->j,*pjj;
  int            *ci,*cj,*denserow,*sparserow,*ptadenserow,*ptasparserow,*ptaj;
  int            an=A->N,am=A->M,pn=P->N,pm=P->M,ppdof=pp->dof;
  int            i,j,k,dof,pdof,ptnzi,arow,anzj,ptanzi,prow,pnzj,cnzi;
  MatScalar      *ca;

  PetscFunctionBegin;  
  /* Start timer */
  ierr = PetscLogEventBegin(MATSeqAIJ_PtAPSymbolic,A,PP,0,0);CHKERRQ(ierr);

  /* Get ij structure of P^T */
  ierr = MatGetSymbolicTranspose_SeqAIJ(P,&pti,&ptj);CHKERRQ(ierr);

  /* Allocate ci array, arrays for fill computation and */
  /* free space for accumulating nonzero column info */
  ierr = PetscMalloc((pn+1)*sizeof(int),&ci);CHKERRQ(ierr);
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
  for (i=0;i<pn/ppdof;i++) {
    ptnzi  = pti[i+1] - pti[i];
    ptanzi = 0;
    ptJ    = ptj + pti[i];
    for (dof=0;dof<ppdof;dof++) {
    /* Determine symbolic row of PtA: */
      for (j=0;j<ptnzi;j++) {
        arow = ptJ[j] + dof;
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
        pdof = *ptaj%dof;
        prow = (*ptaj++)/dof;
        pnzj = pi[prow+1] - pi[prow];
        pjj  = pj + pi[prow];
        for (k=0;k<pnzj;k++) {
          if (!denserow[pjj[k]+pdof]) {
            denserow[pjj[k]+pdof] = -1;
            sparserow[cnzi++]     = pjj[k]+pdof;
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
      ci[i+1+dof] = ci[i+dof] + cnzi;
    }
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

  ierr = PetscLogEventEnd(MATSeqAIJ_PtAPSymbolic,A,PP,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "MatSeqAIJPtAPNumeric"
/*@
   MatSeqAIJPtAPNumeric - Computes the matrix projection C = P^T * A * P

   Collective on Mat

   Input Parameters:
+  A - the matrix
-  P - the projection matrix

   Output Parameters:
.  C - the product matrix

   Notes:
   C must have been created by calling MatSeqAIJPtAPSymbolic and must be destroyed by
   the user using MatDeatroy().

   This routine is currently only implemented for pairs of SeqAIJ matrices and classes
   which inherit from SeqAIJ.  C will be of type MATSEQAIJ.

   Level: intermediate

.seealso: MatSeqAIJPtAP(),MatSeqAIJPtAPSymbolic(),MatMatMultNumeric()
@*/
int MatSeqAIJPtAPNumeric(Mat A,Mat P,Mat C) {
  int ierr;
  char funct[80];
  int (*f)(Mat,Mat,Mat);

  PetscFunctionBegin;

  PetscValidHeaderSpecific(A,MAT_COOKIE,1);
  PetscValidType(A,1);
  MatPreallocated(A);
  if (!A->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (A->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 

  PetscValidHeaderSpecific(P,MAT_COOKIE,2);
  PetscValidType(P,2);
  MatPreallocated(P);
  if (!P->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (P->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 

  PetscValidHeaderSpecific(C,MAT_COOKIE,3);
  PetscValidType(C,3);
  MatPreallocated(C);
  if (!C->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (C->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 

  if (P->N!=C->M) SETERRQ2(PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, %d != %d",P->N,C->M);
  if (P->M!=A->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, %d != %d",P->M,A->N);
  if (A->M!=A->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Matrix 'A' must be square, %d != %d",A->M,A->N);
  if (P->N!=C->N) SETERRQ2(PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, %d != %d",P->N,C->N);

  /* Currently only _seqaij_seqaij is implemented, so just query for it. */
  /* When other implementations exist, attack the multiple dispatch problem. */
  ierr = PetscStrcpy(funct,"MatApplyPtAPNumeric_seqaij_seqaij");CHKERRQ(ierr);
  ierr = PetscObjectQueryFunction((PetscObject)P,funct,(PetscVoidFunction)&f);CHKERRQ(ierr);
  if (!f) SETERRQ1(1,"MatSeqAIJPtAPNumeric is not supported for P of type %s",P->type_name);
  ierr = PetscObjectQueryFunction((PetscObject)A,funct,(PetscVoidFunction)&f);CHKERRQ(ierr);
  if (!f) SETERRQ1(1,"MatSeqAIJPtAPNumeric is not supported for A of type %s",A->type_name);

  ierr = (*f)(A,P,C);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatApplyPtAPNumeric_SeqAIJ_SeqAIJ"
int MatApplyPtAPNumeric_SeqAIJ_SeqAIJ(Mat A,Mat P,Mat C) {
  int        ierr,flops=0;
  Mat_SeqAIJ *a  = (Mat_SeqAIJ *) A->data;
  Mat_SeqAIJ *p  = (Mat_SeqAIJ *) P->data;
  Mat_SeqAIJ *c  = (Mat_SeqAIJ *) C->data;
  int        *ai=a->i,*aj=a->j,*apj,*apjdense,*pi=p->i,*pj=p->j,*pJ=p->j,*pjj;
  int        *ci=c->i,*cj=c->j,*cjj;
  int        am=A->M,cn=C->N,cm=C->M;
  int        i,j,k,anzi,pnzi,apnzj,nextap,pnzj,prow,crow;
  MatScalar  *aa=a->a,*apa,*pa=p->a,*pA=p->a,*paj,*ca=c->a,*caj;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(MATSeqAIJ_PtAPNumeric,A,P,C,0);CHKERRQ(ierr);

  /* Allocate temporary array for storage of one row of A*P */
  ierr = PetscMalloc(cn*(sizeof(MatScalar)+2*sizeof(int)),&apa);CHKERRQ(ierr);
  ierr = PetscMemzero(apa,cn*(sizeof(MatScalar)+2*sizeof(int)));CHKERRQ(ierr);

  apj      = (int *)(apa + cn);
  apjdense = apj + cn;

  /* Clear old values in C */
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
  ierr = PetscLogEventEnd(MATSeqAIJ_PtAPNumeric,A,P,C,0);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "RegisterApplyPtAPRoutines_Private"
int RegisterApplyPtAPRoutines_Private(Mat A) {
  int ierr;

  PetscFunctionBegin;

  if (!MATSeqAIJ_PtAP) {
    ierr = PetscLogEventRegister(&MATSeqAIJ_PtAP,"MatSeqAIJApplyPtAP",MAT_COOKIE);CHKERRQ(ierr);
  }

  if (!MATSeqAIJ_PtAPSymbolic) {
    ierr = PetscLogEventRegister(&MATSeqAIJ_PtAPSymbolic,"MatSeqAIJApplyPtAPSymbolic",MAT_COOKIE);CHKERRQ(ierr);
  }
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)A,"MatApplyPtAPSymbolic_seqaij_seqaij",
                                           "MatApplyPtAPSymbolic_SeqAIJ_SeqAIJ",
                                           MatApplyPtAPSymbolic_SeqAIJ_SeqAIJ);CHKERRQ(ierr);

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)A,"MatApplyPtAPSymbolic_seqaij_aij",
                                           "MatApplyPtAPSymbolic_SeqAIJ_SeqAIJ",
                                           MatApplyPtAPSymbolic_SeqAIJ_SeqAIJ);CHKERRQ(ierr);

  if (!MATSeqAIJ_PtAPNumeric) {
    ierr = PetscLogEventRegister(&MATSeqAIJ_PtAPNumeric,"MatSeqAIJApplyPtAPNumeric",MAT_COOKIE);CHKERRQ(ierr);
  }
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)A,"MatApplyPtAPNumeric_seqaij_aij",
                                           "MatApplyPtAPNumeric_SeqAIJ_SeqAIJ",
                                           MatApplyPtAPNumeric_SeqAIJ_SeqAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)A,"MatApplyPtAPNumeric_seqaij_seqaij",
                                           "MatApplyPtAPNumeric_SeqAIJ_SeqAIJ",
                                           MatApplyPtAPNumeric_SeqAIJ_SeqAIJ);CHKERRQ(ierr);
  ierr = RegisterMatMatMultRoutines_Private(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
