/*
  Defines symbolic transpose routines for SeqAIJ matrices.

  Currently Get/Restore only allocates/frees memory for holding the
  (i,j) info for the transpose.  Someday, this info could be
  maintained so successive calls to Get will not recompute the info.

  Also defined is a "faster" implementation of MatTranspose for SeqAIJ
  matrices which avoids calls to MatSetValues.  This routine has not
  been adopted as the standard yet as it is somewhat untested.

*/

#include "src/mat/impls/aij/seq/aij.h"

static int logkey_matgetsymtranspose    = 0;
static int logkey_mattranspose          = 0;


#undef __FUNCT__
#define __FUNCT__ "MatGetSymbolicTranspose_SeqIJ"
int MatGetSymbolicTranspose_SeqAIJ(Mat A,int *Ati[],int *Atj[]) {
  int        ierr,i,j,anzj;
  Mat_SeqAIJ *a=(Mat_SeqAIJ *)A->data;
  int        aishift = a->indexshift,an=A->N,am=A->M;
  int        *ati,*atj,*atfill,*ai=a->i,*aj=a->j;

  PetscFunctionBegin;

  ierr = PetscLogInfo(A,"Getting Symbolic Transpose.\n");CHKERRQ(ierr);
  if (aishift) SETERRQ(PETSC_ERR_SUP,"Shifted matrix indices are not supported.");

  /* Set up timers */
  if (!logkey_matgetsymtranspose) {
    ierr = PetscLogEventRegister(&logkey_matgetsymtranspose,"MatGetSymbolicTranspose",MAT_COOKIE);CHKERRQ(ierr);
  }
  ierr = PetscLogEventBegin(logkey_matgetsymtranspose,A,0,0,0);CHKERRQ(ierr);

  /* Allocate space for symbolic transpose info and work array */
  ierr = PetscMalloc((an+1)*sizeof(int),&ati);CHKERRQ(ierr);
  ierr = PetscMalloc(ai[am]*sizeof(int),&atj);CHKERRQ(ierr);
  ierr = PetscMalloc(an*sizeof(int),&atfill);CHKERRQ(ierr);
  ierr = PetscMemzero(ati,(an+1)*sizeof(int));CHKERRQ(ierr);

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
  ierr = PetscMemcpy(atfill,ati,an*sizeof(int));CHKERRQ(ierr);

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

  ierr = PetscLogEventEnd(logkey_matgetsymtranspose,A,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatTranspose_SeqIJ_FAST"
int MatTranspose_SeqAIJ_FAST(Mat A,Mat *B) {
  int        ierr,i,j,anzj;
  Mat        At;
  Mat_SeqAIJ *a=(Mat_SeqAIJ *)A->data,*at;
  int        aishift = a->indexshift,an=A->N,am=A->M;
  int        *ati,*atj,*atfill,*ai=a->i,*aj=a->j;
  MatScalar  *ata,*aa=a->a;
  PetscFunctionBegin;

  if (aishift) SETERRQ(PETSC_ERR_SUP,"Shifted matrix indices are not supported.");

  /* Set up timers */
  if (!logkey_mattranspose) {
    ierr = PetscLogEventRegister(&logkey_mattranspose,"MatTranspose_SeqAIJ_FAST",MAT_COOKIE);CHKERRQ(ierr);
  }
  ierr = PetscLogEventBegin(logkey_mattranspose,A,0,0,0);CHKERRQ(ierr);

  /* Allocate space for symbolic transpose info and work array */
  ierr = PetscMalloc((an+1)*sizeof(int),&ati);CHKERRQ(ierr);
  ierr = PetscMalloc(ai[am]*sizeof(int),&atj);CHKERRQ(ierr);
  ierr = PetscMalloc(ai[am]*sizeof(MatScalar),&ata);CHKERRQ(ierr);
  ierr = PetscMalloc(an*sizeof(int),&atfill);CHKERRQ(ierr);
  ierr = PetscMemzero(ati,(an+1)*sizeof(int));CHKERRQ(ierr);
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
  ierr = PetscMemcpy(atfill,ati,an*sizeof(int));CHKERRQ(ierr);

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
  ierr = MatCreateSeqAIJWithArrays(A->comm,an,am,ati,atj,ata,&At);CHKERRQ(ierr);
  at   = (Mat_SeqAIJ *)(At->data);
  at->freedata = PETSC_TRUE;
  at->nonew    = 0;
  if (B) {
    *B = At;
  } else {
    ierr = MatHeaderCopy(A,At);
  }
  ierr = PetscLogEventEnd(logkey_mattranspose,A,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatRestoreSymbolicTranspose_SeqAIJ"
int MatRestoreSymbolicTranspose_SeqAIJ(Mat A,int *ati[],int *atj[]) {
  int ierr;

  PetscFunctionBegin;
  ierr = PetscLogInfo(A,"Restoring Symbolic Transpose.\n");CHKERRQ(ierr);
  ierr = PetscFree(*ati);CHKERRQ(ierr);
  ati  = PETSC_NULL;
  ierr = PetscFree(*atj);CHKERRQ(ierr);
  atj  = PETSC_NULL;
  PetscFunctionReturn(0);
}

