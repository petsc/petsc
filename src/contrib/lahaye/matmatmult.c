/*$Id: matmatmult.c,v 1.15 2001/09/07 20:04:44 buschelm Exp $*/
/*
  Defines a matrix-matrix product for 2 SeqAIJ matrices
          C = A * B
*/

#include "src/mat/impls/aij/seq/aij.h"

typedef struct _p_Space *FreeSpace;
typedef struct _p_Space {
  FreeSpace morespace;
  int       *head;
  int       *space;
  int       used;
  int       remaining;
} _p_FreeSpace;  

static int logkey_symbolic=0;
static int logkey_numeric=0;

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
  FreeSpace   free_space,current_space;
  Mat_SeqAIJ  *a=(Mat_SeqAIJ*)A->data,*b=(Mat_SeqAIJ*)B->data,*c;
  int         aishift=a->indexshift,bishift=b->indexshift;
  int         *ai=a->i,*aj=a->j,*bi=b->i,*bj=b->j;
  int         *ci,*bjj,*cj,*cj2,*densefill,*sparsefill;
  int         an=A->N,am=A->M,bn=B->N,bm=B->M;
  int         ierr,i,j,k,anzi,brow,bnzj,cnzi,free_space_size=bi[bm];
  MatScalar   *ca;

  PetscFunctionBegin;
  /* some error checking which could be moved into interface layer */
  if (aishift || bishift) SETERRQ(PETSC_ERR_SUP,"Shifted matrix indices are not supported.");
  if (an!=bm) SETERRQ2(PETSC_ERR_ARG_SIZ,"Matrix dimensions are incompatible, %d != %d",an,bm);
  
  if (!logkey_symbolic) {
    ierr = PetscLogEventRegister(&logkey_symbolic,"MatMatMult_Symbolic",PETSC_NULL,MAT_COOKIE);CHKERRQ(ierr);
  }
  ierr = PetscLogEventBegin(logkey_symbolic,A,B,0,0);CHKERRQ(ierr);
  /* Set up */
  /* Allocate ci array, arrays for fill computation and */
  /* free space for accumulating nonzero column info */
  ierr = PetscMalloc(((am+1)+1)*sizeof(int),&ci);CHKERRQ(ierr);CHKMEMQ;
  ci[0] = 0;CHKMEMQ;

  ierr = PetscMalloc((2*bn+1)*sizeof(int),&densefill);CHKERRQ(ierr);CHKMEMQ;
  ierr = PetscMemzero(densefill,(2*bn+1)*sizeof(int));CHKERRQ(ierr);CHKMEMQ;
  sparsefill = densefill + bn;

  ierr = PetscMalloc(sizeof(_p_FreeSpace),&free_space);CHKMEMQ;
  ierr = PetscMalloc((free_space_size+1)*sizeof(int),&(free_space->head));CHKERRQ(ierr);CHKMEMQ;
  free_space->space     = free_space->head;
  free_space->remaining = free_space_size;
  free_space->used      = 0;
  free_space->morespace = NULL;
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
          densefill[bjj[k]]  = -1;CHKMEMQ;
          sparsefill[cnzi++] = bjj[k];CHKMEMQ;
        }
      }
    }

    /* sort sparsefill */
    ierr = PetscSortInt(cnzi,sparsefill);CHKERRQ(ierr);CHKMEMQ;

    /* If free space is not available, make more free space */
    /* For Lahaye's code, the estimated nnz in the product equals b->nz, so just increment */
    /* using the same estimate.  For other codes, this might not be such a good estimator. */
    if (current_space->remaining<cnzi) {
      ierr = PetscMalloc(sizeof(_p_FreeSpace),&(current_space->morespace));CHKERRQ(ierr);CHKMEMQ;
      current_space = current_space->morespace;
      ierr = PetscMalloc((free_space_size+1)*sizeof(int),&(current_space->head));CHKERRQ(ierr);CHKMEMQ;
      current_space->space     = current_space->head;
      current_space->remaining = free_space_size;
      current_space->used      = 0;
      current_space->morespace = NULL;
    }

    /* Copy data into free space, and zero out densefill */
    ierr = PetscMemcpy(current_space->space,sparsefill,cnzi*sizeof(int));CHKERRQ(ierr);CHKMEMQ;
    current_space->space     += cnzi;
    current_space->used      += cnzi;
    current_space->remaining -= cnzi;
    for (j=0;j<cnzi;j++) {
      densefill[sparsefill[j]] = 0;CHKMEMQ;
    }
    ci[i+1] = ci[i] + cnzi;CHKMEMQ;
  }

  /* nnz is now stored in ci[an], column indices are in the list of free space */
  /* Allocate space for cj, initialize cj, and */
  /* destroy list of free space and other temporary array(s) */
  ierr = PetscMalloc((ci[an]+1)*sizeof(int),&cj);CHKERRQ(ierr);CHKMEMQ;
  cj2 = cj;
  while (free_space != NULL) {
    current_space = free_space->morespace;
    ierr = PetscMemcpy(cj2,free_space->head,(free_space->used)*sizeof(int));CHKERRQ(ierr);CHKMEMQ;
    cj2 += free_space->used;
    ierr = PetscFree(free_space->head);CHKERRQ(ierr);
    ierr = PetscFree(free_space);CHKERRQ(ierr);
    free_space = current_space;
  }
  ierr = PetscFree(densefill);CHKERRQ(ierr);
    
  /* Allocate space for ca */
  ierr = PetscMalloc((ci[an]+1)*sizeof(MatScalar),&ca);CHKERRQ(ierr);CHKMEMQ;
  
  /* put together the new matrix */
  ierr = MatCreateSeqAIJWithArrays(A->comm,am,bn,ci,cj,ca,C);CHKERRQ(ierr);

  /* MatCreateSeqAIJWithArrays flags matrix so PETSc doesn't free the user's arrays. */
  /* These are PETSc arrays, so change flag to free them */
  c = (Mat_SeqAIJ *)((*C)->data);
  c->freedata = PETSC_TRUE;

  ierr = PetscLogEventEnd(logkey_symbolic,A,B,0,0);CHKERRQ(ierr);
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

  if (!logkey_numeric) {
    ierr = PetscLogEventRegister(&logkey_numeric,"MatMatMult_Numeric",PETSC_NULL,MAT_COOKIE);CHKERRQ(ierr);
  }
  ierr = PetscLogEventBegin(logkey_numeric,A,B,C,0);CHKERRQ(ierr);
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
    /* Store row back into C, don't forget to re-zero temp */
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
  ierr = PetscLogEventEnd(logkey_numeric,A,B,C,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
