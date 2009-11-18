#define PETSCMAT_DLL

/*
  Defines matrix-matrix product routines for pairs of SeqAIJ matrices
          C = P * A * P^T
*/

#include "../src/mat/impls/aij/seq/aij.h"
#include "../src/mat/utils/freespace.h"


/*
     MatApplyPAPt_Symbolic_SeqAIJ_SeqAIJ - Forms the symbolic product of two SeqAIJ matrices
           C = P * A * P^T;

     Note: C is assumed to be uncreated.
           If this is not the case, Destroy C before calling this routine.
*/
#undef __FUNCT__
#define __FUNCT__ "MatApplyPAPt_Symbolic_SeqAIJ_SeqAIJ"
PetscErrorCode MatApplyPAPt_Symbolic_SeqAIJ_SeqAIJ(Mat A,Mat P,Mat *C) 
{
  /* Note: This code is virtually identical to that of MatApplyPtAP_SeqAIJ_Symbolic */
  /*        and MatMatMult_SeqAIJ_SeqAIJ_Symbolic.  Perhaps they could be merged nicely. */
  PetscErrorCode     ierr;
  PetscFreeSpaceList free_space=PETSC_NULL,current_space=PETSC_NULL;
  Mat_SeqAIJ         *a=(Mat_SeqAIJ*)A->data,*p=(Mat_SeqAIJ*)P->data,*c;
  PetscInt           *ai=a->i,*aj=a->j,*ajj,*pi=p->i,*pj=p->j,*pti,*ptj,*ptjj;
  PetscInt           *ci,*cj,*paj,*padenserow,*pasparserow,*denserow,*sparserow;
  PetscInt           an=A->cmap->N,am=A->rmap->N,pn=P->cmap->N,pm=P->rmap->N;
  PetscInt           i,j,k,pnzi,arow,anzj,panzi,ptrow,ptnzj,cnzi;
  MatScalar          *ca;

  PetscFunctionBegin;
  /* some error checking which could be moved into interface layer */
  if (pn!=am) SETERRQ2(PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, %D != %D",pn,am);
  if (am!=an) SETERRQ2(PETSC_ERR_ARG_SIZ,"Matrix 'A' must be square, %D != %D",am, an);

  /* Set up timers */
  ierr = PetscLogEventBegin(MAT_Applypapt_symbolic,A,P,0,0);CHKERRQ(ierr);

  /* Create ij structure of P^T */
  ierr = MatGetSymbolicTranspose_SeqAIJ(P,&pti,&ptj);CHKERRQ(ierr);

  /* Allocate ci array, arrays for fill computation and */
  /* free space for accumulating nonzero column info */
  ierr = PetscMalloc(((pm+1)*1)*sizeof(PetscInt),&ci);CHKERRQ(ierr);
  ci[0] = 0;

  ierr = PetscMalloc4(an,PetscInt,&padenserow,an,PetscInt,&pasparserow,pm,PetscInt,&denserow,pm,PetscInt,&sparserow);CHKERRQ(ierr);
  ierr = PetscMemzero(padenserow,an*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemzero(pasparserow,an*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemzero(denserow,pm*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemzero(sparserow,pm*sizeof(PetscInt));CHKERRQ(ierr);

  /* Set initial free space to be nnz(A) scaled by aspect ratio of Pt. */
  /* This should be reasonable if sparsity of PAPt is similar to that of A. */
  ierr          = PetscFreeSpaceGet((ai[am]/pn)*pm,&free_space);
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
      ierr = PetscFreeSpaceGet(cnzi+current_space->total_array_size,&current_space);CHKERRQ(ierr);
    }

    /* Copy data into free space, and zero out dense row */
    ierr = PetscMemcpy(current_space->array,sparserow,cnzi*sizeof(PetscInt));CHKERRQ(ierr);
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
  ierr = PetscMalloc((ci[pm]+1)*sizeof(PetscInt),&cj);CHKERRQ(ierr);
  ierr = PetscFreeSpaceContiguous(&free_space,cj);CHKERRQ(ierr);
  ierr = PetscFree4(padenserow,pasparserow,denserow,sparserow);CHKERRQ(ierr);
    
  /* Allocate space for ca */
  ierr = PetscMalloc((ci[pm]+1)*sizeof(MatScalar),&ca);CHKERRQ(ierr);
  ierr = PetscMemzero(ca,(ci[pm]+1)*sizeof(MatScalar));CHKERRQ(ierr);
  
  /* put together the new matrix */
  ierr = MatCreateSeqAIJWithArrays(((PetscObject)A)->comm,pm,pm,ci,cj,ca,C);CHKERRQ(ierr);

  /* MatCreateSeqAIJWithArrays flags matrix so PETSc doesn't free the user's arrays. */
  /* Since these are PETSc arrays, change flags to free them as necessary. */
  c = (Mat_SeqAIJ *)((*C)->data);
  c->free_a  = PETSC_TRUE;
  c->free_ij = PETSC_TRUE;
  c->nonew   = 0;

  /* Clean up. */
  ierr = MatRestoreSymbolicTranspose_SeqAIJ(P,&pti,&ptj);CHKERRQ(ierr);

  ierr = PetscLogEventEnd(MAT_Applypapt_symbolic,A,P,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
     MatApplyPAPt_Numeric_SeqAIJ - Forms the numeric product of two SeqAIJ matrices
           C = P * A * P^T;
     Note: C must have been created by calling MatApplyPAPt_Symbolic_SeqAIJ.
*/
#undef __FUNCT__
#define __FUNCT__ "MatApplyPAPt_Numeric_SeqAIJ_SeqAIJ"
PetscErrorCode MatApplyPAPt_Numeric_SeqAIJ_SeqAIJ(Mat A,Mat P,Mat C)
{
  PetscErrorCode ierr;
  PetscInt       flops=0;
  Mat_SeqAIJ     *a  = (Mat_SeqAIJ *) A->data;
  Mat_SeqAIJ     *p  = (Mat_SeqAIJ *) P->data;
  Mat_SeqAIJ     *c  = (Mat_SeqAIJ *) C->data;
  PetscInt       *ai=a->i,*aj=a->j,*ajj,*pi=p->i,*pj=p->j,*pjj=p->j,*paj,*pajdense,*ptj;
  PetscInt       *ci=c->i,*cj=c->j;
  PetscInt       an=A->cmap->N,am=A->rmap->N,pn=P->cmap->N,pm=P->rmap->N,cn=C->cmap->N,cm=C->rmap->N;
  PetscInt       i,j,k,k1,k2,pnzi,anzj,panzj,arow,ptcol,ptnzj,cnzi;
  MatScalar      *aa=a->a,*pa=p->a,*pta=p->a,*ptaj,*paa,*aaj,*ca=c->a,sum;

  PetscFunctionBegin;
  /* This error checking should be unnecessary if the symbolic was performed */ 
  if (pm!=cm) SETERRQ2(PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, %D != %D",pm,cm);
  if (pn!=am) SETERRQ2(PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, %D != %D",pn,am);
  if (am!=an) SETERRQ2(PETSC_ERR_ARG_SIZ,"Matrix 'A' must be square, %D != %D",am, an);
  if (pm!=cn) SETERRQ2(PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, %D != %D",pm, cn);

  /* Set up timers */
  ierr = PetscLogEventBegin(MAT_Applypapt_numeric,A,P,C,0);CHKERRQ(ierr);
  ierr = PetscMemzero(ca,ci[cm]*sizeof(MatScalar));CHKERRQ(ierr);

  ierr = PetscMalloc3(an,MatScalar,&paa,an,PetscInt,&paj,an,PetscInt,&pajdense);CHKERRQ(ierr);
  ierr = PetscMemzero(paa,an*(sizeof(MatScalar)+2*sizeof(PetscInt)));CHKERRQ(ierr);
  
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
          sum += paa[paj[k1++]]*ptaj[k2++];
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
  ierr = PetscFree3(paa,paj,pajdense);CHKERRQ(ierr);
  ierr = PetscLogFlops(flops);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_Applypapt_numeric,A,P,C,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
  
#undef __FUNCT__
#define __FUNCT__ "MatApplyPAPt_SeqAIJ_SeqAIJ"
PetscErrorCode MatApplyPAPt_SeqAIJ_SeqAIJ(Mat A,Mat P,Mat *C) 
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(MAT_Applypapt,A,P,0,0);CHKERRQ(ierr);
  ierr = MatApplyPAPt_Symbolic_SeqAIJ_SeqAIJ(A,P,C);CHKERRQ(ierr);
  ierr = MatApplyPAPt_Numeric_SeqAIJ_SeqAIJ(A,P,*C);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_Applypapt,A,P,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
