
/*
  Defines symbolic transpose routines for SeqAIJ matrices.

  Currently Get/Restore only allocates/frees memory for holding the
  (i,j) info for the transpose.  Someday, this info could be
  maintained so successive calls to Get will not recompute the info.

  Also defined is a "faster" implementation of MatTranspose for SeqAIJ
  matrices which avoids calls to MatSetValues.  This routine has not
  been adopted as the standard yet as it is somewhat untested.

*/

#include <../src/mat/impls/aij/seq/aij.h>


#undef __FUNCT__
#define __FUNCT__ "MatGetSymbolicTranspose_SeqAIJ"
PetscErrorCode MatGetSymbolicTranspose_SeqAIJ(Mat A,PetscInt *Ati[],PetscInt *Atj[]) 
{
  PetscErrorCode ierr;
  PetscInt       i,j,anzj;
  Mat_SeqAIJ     *a=(Mat_SeqAIJ *)A->data;
  PetscInt       an=A->cmap->N,am=A->rmap->N;
  PetscInt       *ati,*atj,*atfill,*ai=a->i,*aj=a->j;

  PetscFunctionBegin;
  ierr = PetscInfo(A,"Getting Symbolic Transpose.\n");CHKERRQ(ierr);

  /* Set up timers */
  ierr = PetscLogEventBegin(MAT_Getsymtranspose,A,0,0,0);CHKERRQ(ierr);

  /* Allocate space for symbolic transpose info and work array */
  ierr = PetscMalloc((an+1)*sizeof(PetscInt),&ati);CHKERRQ(ierr);
  ierr = PetscMalloc(ai[am]*sizeof(PetscInt),&atj);CHKERRQ(ierr);
  ierr = PetscMalloc(an*sizeof(PetscInt),&atfill);CHKERRQ(ierr);
  ierr = PetscMemzero(ati,(an+1)*sizeof(PetscInt));CHKERRQ(ierr);

  /* Walk through aj and count ## of non-zeros in each row of A^T. */
  /* Note: offset by 1 for fast conversion into csr format. */
  for (i=0;i<ai[am];i++) {
    ati[aj[i]+1] += 1;
  }
  /* Form ati for csr format of A^T. */
  for (i=0;i<an;i++) {
    ati[i+1] += ati[i];
  }

  /* Copy ati into atfill so we have locations of the next free space in atj */
  ierr = PetscMemcpy(atfill,ati,an*sizeof(PetscInt));CHKERRQ(ierr);

  /* Walk through A row-wise and mark nonzero entries of A^T. */
  for (i=0;i<am;i++) {
    anzj = ai[i+1] - ai[i];
    for (j=0;j<anzj;j++) {
      atj[atfill[*aj]] = i;
      atfill[*aj++]   += 1;
    }
  }

  /* Clean up temporary space and complete requests. */
  ierr = PetscFree(atfill);CHKERRQ(ierr);
  *Ati = ati;
  *Atj = atj;

  ierr = PetscLogEventEnd(MAT_Getsymtranspose,A,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*
  MatGetSymbolicTransposeReduced_SeqAIJ() - Get symbolic matrix structure of submatrix A[rstart:rend,:],
     modified from MatGetSymbolicTranspose_SeqAIJ()
*/
#undef __FUNCT__
#define __FUNCT__ "MatGetSymbolicTransposeReduced_SeqAIJ"
PetscErrorCode MatGetSymbolicTransposeReduced_SeqAIJ(Mat A,PetscInt rstart,PetscInt rend,PetscInt *Ati[],PetscInt *Atj[]) 
{
  PetscErrorCode ierr;
  PetscInt       i,j,anzj;
  Mat_SeqAIJ     *a=(Mat_SeqAIJ *)A->data;
  PetscInt       an=A->cmap->N;
  PetscInt       *ati,*atj,*atfill,*ai=a->i,*aj=a->j;

  PetscFunctionBegin;
  ierr = PetscInfo(A,"Getting Symbolic Transpose\n");CHKERRQ(ierr);
  ierr = PetscLogEventBegin(MAT_Getsymtransreduced,A,0,0,0);CHKERRQ(ierr);

  /* Allocate space for symbolic transpose info and work array */
  ierr = PetscMalloc((an+1)*sizeof(PetscInt),&ati);CHKERRQ(ierr);
  anzj = ai[rend] - ai[rstart];
  ierr = PetscMalloc((anzj+1)*sizeof(PetscInt),&atj);CHKERRQ(ierr);
  ierr = PetscMalloc((an+1)*sizeof(PetscInt),&atfill);CHKERRQ(ierr);
  ierr = PetscMemzero(ati,(an+1)*sizeof(PetscInt));CHKERRQ(ierr);

  /* Walk through aj and count ## of non-zeros in each row of A^T. */
  /* Note: offset by 1 for fast conversion into csr format. */
  for (i=ai[rstart]; i<ai[rend]; i++) {
    ati[aj[i]+1] += 1;
  }
  /* Form ati for csr format of A^T. */
  for (i=0;i<an;i++) {
    ati[i+1] += ati[i];
  }

  /* Copy ati into atfill so we have locations of the next free space in atj */
  ierr = PetscMemcpy(atfill,ati,an*sizeof(PetscInt));CHKERRQ(ierr);

  /* Walk through A row-wise and mark nonzero entries of A^T. */
  aj = aj + ai[rstart];
  for (i=rstart; i<rend; i++) {
    anzj = ai[i+1] - ai[i];
    for (j=0;j<anzj;j++) {
      atj[atfill[*aj]] = i-rstart;
      atfill[*aj++]   += 1;
    }
  }

  /* Clean up temporary space and complete requests. */
  ierr = PetscFree(atfill);CHKERRQ(ierr);
  *Ati = ati;
  *Atj = atj;

  ierr = PetscLogEventEnd(MAT_Getsymtransreduced,A,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatTranspose_SeqAIJ_FAST"
PetscErrorCode MatTranspose_SeqAIJ_FAST(Mat A,MatReuse reuse,Mat *B) 
{
  PetscErrorCode ierr;
  PetscInt       i,j,anzj;
  Mat            At;
  Mat_SeqAIJ     *a=(Mat_SeqAIJ *)A->data,*at;
  PetscInt       an=A->cmap->N,am=A->rmap->N;
  PetscInt       *ati,*atj,*atfill,*ai=a->i,*aj=a->j;
  MatScalar      *ata,*aa=a->a;

  PetscFunctionBegin;

  ierr = PetscLogEventBegin(MAT_Transpose_SeqAIJ,A,0,0,0);CHKERRQ(ierr);

  if (reuse == MAT_INITIAL_MATRIX || *B == A) {
    /* Allocate space for symbolic transpose info and work array */
    ierr = PetscMalloc((an+1)*sizeof(PetscInt),&ati);CHKERRQ(ierr);
    ierr = PetscMalloc(ai[am]*sizeof(PetscInt),&atj);CHKERRQ(ierr);
    ierr = PetscMalloc(ai[am]*sizeof(MatScalar),&ata);CHKERRQ(ierr);
    ierr = PetscMemzero(ati,(an+1)*sizeof(PetscInt));CHKERRQ(ierr);
    /* Walk through aj and count ## of non-zeros in each row of A^T. */
    /* Note: offset by 1 for fast conversion into csr format. */
    for (i=0;i<ai[am];i++) {
      ati[aj[i]+1] += 1;
    }
    /* Form ati for csr format of A^T. */
    for (i=0;i<an;i++) {
      ati[i+1] += ati[i];
    }
  } else {
    Mat_SeqAIJ *sub_B = (Mat_SeqAIJ*) (*B)->data;
    ati = sub_B->i;
    atj = sub_B->j;
    ata = sub_B->a;
    At = *B;
  }

  /* Copy ati into atfill so we have locations of the next free space in atj */
  ierr = PetscMalloc(an*sizeof(PetscInt),&atfill);CHKERRQ(ierr);
  ierr = PetscMemcpy(atfill,ati,an*sizeof(PetscInt));CHKERRQ(ierr);

  /* Walk through A row-wise and mark nonzero entries of A^T. */
  for (i=0;i<am;i++) {
    anzj = ai[i+1] - ai[i];
    for (j=0;j<anzj;j++) {
      atj[atfill[*aj]] = i;
      ata[atfill[*aj]] = *aa++;
      atfill[*aj++]   += 1;
    }
  }

  /* Clean up temporary space and complete requests. */
  ierr = PetscFree(atfill);CHKERRQ(ierr);
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatCreateSeqAIJWithArrays(((PetscObject)A)->comm,an,am,ati,atj,ata,&At);CHKERRQ(ierr);
    at   = (Mat_SeqAIJ *)(At->data);
    at->free_a  = PETSC_TRUE;
    at->free_ij  = PETSC_TRUE;
    at->nonew   = 0;
  }

  if (reuse == MAT_INITIAL_MATRIX || *B != A) {
    *B = At;
  } else {
    ierr = MatHeaderMerge(A,At);
  }
  ierr = PetscLogEventEnd(MAT_Transpose_SeqAIJ,A,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatRestoreSymbolicTranspose_SeqAIJ"
PetscErrorCode MatRestoreSymbolicTranspose_SeqAIJ(Mat A,PetscInt *ati[],PetscInt *atj[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInfo(A,"Restoring Symbolic Transpose.\n");CHKERRQ(ierr);
  ierr = PetscFree(*ati);CHKERRQ(ierr);
  ierr = PetscFree(*atj);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

