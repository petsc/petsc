
/*
  Defines symbolic transpose routines for SeqAIJ matrices.

  Currently Get/Restore only allocates/frees memory for holding the
  (i,j) info for the transpose.  Someday, this info could be
  maintained so successive calls to Get will not recompute the info.

  Also defined is a faster implementation of MatTranspose for SeqAIJ
  matrices which avoids calls to MatSetValues. This routine is the new
  standard since it is much faster than MatTranspose_AIJ.

*/

#include <../src/mat/impls/aij/seq/aij.h>

PetscErrorCode MatGetSymbolicTranspose_SeqAIJ(Mat A,PetscInt *Ati[],PetscInt *Atj[])
{
  PetscInt       i,j,anzj;
  Mat_SeqAIJ     *a=(Mat_SeqAIJ*)A->data;
  PetscInt       an=A->cmap->N,am=A->rmap->N;
  PetscInt       *ati,*atj,*atfill,*ai=a->i,*aj=a->j;

  PetscFunctionBegin;
  PetscCall(PetscInfo(A,"Getting Symbolic Transpose.\n"));

  /* Set up timers */
  PetscCall(PetscLogEventBegin(MAT_Getsymtranspose,A,0,0,0));

  /* Allocate space for symbolic transpose info and work array */
  PetscCall(PetscCalloc1(an+1,&ati));
  PetscCall(PetscMalloc1(ai[am],&atj));
  PetscCall(PetscMalloc1(an,&atfill));

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
  PetscCall(PetscArraycpy(atfill,ati,an));

  /* Walk through A row-wise and mark nonzero entries of A^T. */
  for (i=0; i<am; i++) {
    anzj = ai[i+1] - ai[i];
    for (j=0; j<anzj; j++) {
      atj[atfill[*aj]] = i;
      atfill[*aj++]   += 1;
    }
  }

  /* Clean up temporary space and complete requests. */
  PetscCall(PetscFree(atfill));
  *Ati = ati;
  *Atj = atj;

  PetscCall(PetscLogEventEnd(MAT_Getsymtranspose,A,0,0,0));
  PetscFunctionReturn(0);
}
/*
  MatGetSymbolicTransposeReduced_SeqAIJ() - Get symbolic matrix structure of submatrix A[rstart:rend,:],
     modified from MatGetSymbolicTranspose_SeqAIJ()
*/
PetscErrorCode MatGetSymbolicTransposeReduced_SeqAIJ(Mat A,PetscInt rstart,PetscInt rend,PetscInt *Ati[],PetscInt *Atj[])
{
  PetscInt       i,j,anzj;
  Mat_SeqAIJ     *a=(Mat_SeqAIJ*)A->data;
  PetscInt       an=A->cmap->N;
  PetscInt       *ati,*atj,*atfill,*ai=a->i,*aj=a->j;

  PetscFunctionBegin;
  PetscCall(PetscInfo(A,"Getting Symbolic Transpose\n"));
  PetscCall(PetscLogEventBegin(MAT_Getsymtransreduced,A,0,0,0));

  /* Allocate space for symbolic transpose info and work array */
  PetscCall(PetscCalloc1(an+1,&ati));
  anzj = ai[rend] - ai[rstart];
  PetscCall(PetscMalloc1(anzj+1,&atj));
  PetscCall(PetscMalloc1(an+1,&atfill));

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
  PetscCall(PetscArraycpy(atfill,ati,an));

  /* Walk through A row-wise and mark nonzero entries of A^T. */
  aj = aj + ai[rstart];
  for (i=rstart; i<rend; i++) {
    anzj = ai[i+1] - ai[i];
    for (j=0; j<anzj; j++) {
      atj[atfill[*aj]] = i-rstart;
      atfill[*aj++]   += 1;
    }
  }

  /* Clean up temporary space and complete requests. */
  PetscCall(PetscFree(atfill));
  *Ati = ati;
  *Atj = atj;

  PetscCall(PetscLogEventEnd(MAT_Getsymtransreduced,A,0,0,0));
  PetscFunctionReturn(0);
}

PetscErrorCode MatTranspose_SeqAIJ(Mat A,MatReuse reuse,Mat *B)
{
  PetscInt        i,j,anzj;
  Mat             At;
  Mat_SeqAIJ      *a=(Mat_SeqAIJ*)A->data,*at;
  PetscInt        an=A->cmap->N,am=A->rmap->N;
  PetscInt        *ati,*atj,*atfill,*ai=a->i,*aj=a->j;
  MatScalar       *ata;
  const MatScalar *aa,*av;

  PetscFunctionBegin;
  PetscCall(MatSeqAIJGetArrayRead(A,&av));
  aa   = av;
  if (reuse == MAT_INITIAL_MATRIX || reuse == MAT_INPLACE_MATRIX) {
    /* Allocate space for symbolic transpose info and work array */
    PetscCall(PetscCalloc1(an+1,&ati));
    PetscCall(PetscMalloc1(ai[am],&atj));
    PetscCall(PetscMalloc1(ai[am],&ata));
    /* Walk through aj and count ## of non-zeros in each row of A^T. */
    /* Note: offset by 1 for fast conversion into csr format. */
    for (i=0;i<ai[am];i++) {
      ati[aj[i]+1] += 1; /* count ## of non-zeros for row aj[i] of A^T */
    }
    /* Form ati for csr format of A^T. */
    for (i=0;i<an;i++) {
      ati[i+1] += ati[i];
    }
  } else { /* This segment is called by MatTranspose_MPIAIJ(...,MAT_INITIAL_MATRIX,..) directly! */
    Mat_SeqAIJ *sub_B = (Mat_SeqAIJ*) (*B)->data;
    ati = sub_B->i;
    atj = sub_B->j;
    ata = sub_B->a;
    At  = *B;
  }

  /* Copy ati into atfill so we have locations of the next free space in atj */
  PetscCall(PetscMalloc1(an,&atfill));
  PetscCall(PetscArraycpy(atfill,ati,an));

  /* Walk through A row-wise and mark nonzero entries of A^T. */
  for (i=0;i<am;i++) {
    anzj = ai[i+1] - ai[i];
    for (j=0;j<anzj;j++) {
      atj[atfill[*aj]] = i;
      ata[atfill[*aj]] = *aa++;
      atfill[*aj++]   += 1;
    }
  }
  PetscCall(MatSeqAIJRestoreArrayRead(A,&av));

  /* Clean up temporary space and complete requests. */
  PetscCall(PetscFree(atfill));
  if (reuse == MAT_INITIAL_MATRIX || reuse == MAT_INPLACE_MATRIX) {
    PetscCall(MatCreateSeqAIJWithArrays(PetscObjectComm((PetscObject)A),an,am,ati,atj,ata,&At));
    PetscCall(MatSetBlockSizes(At,PetscAbs(A->cmap->bs),PetscAbs(A->rmap->bs)));

    at          = (Mat_SeqAIJ*)(At->data);
    at->free_a  = PETSC_TRUE;
    at->free_ij = PETSC_TRUE;
    at->nonew   = 0;
    at->maxnz   = ati[an];

    PetscCall(MatSetType(At,((PetscObject)A)->type_name));
  }

  if (reuse == MAT_INITIAL_MATRIX || reuse == MAT_REUSE_MATRIX) {
    *B = At;
  } else {
    PetscCall(MatHeaderMerge(A,&At));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatRestoreSymbolicTranspose_SeqAIJ(Mat A,PetscInt *ati[],PetscInt *atj[])
{
  PetscFunctionBegin;
  PetscCall(PetscInfo(A,"Restoring Symbolic Transpose.\n"));
  PetscCall(PetscFree(*ati));
  PetscCall(PetscFree(*atj));
  PetscFunctionReturn(0);
}
