#include <../src/mat/impls/ell/seq/ell.h>
#include <../src/mat/impls/aij/seq/aij.h>
#include <petsc/private/isimpl.h>

/*
 MatGetColumnIJ_SeqELL_Color() and MatRestoreColumnIJ_SeqELL_Color() are customized from
 MatGetColumnIJ_SeqELL() and MatRestoreColumnIJ_SeqELL() by adding an output
 spidx[], index of a->a, to be used in MatTransposeColoringCreate_SeqELL() and MatFDColoringCreate_SeqELL()
*/
PetscErrorCode MatGetColumnIJ_SeqELL_Color(Mat A,PetscInt oshift,PetscBool symmetric,PetscBool inodecompressed,PetscInt *nn,const PetscInt *ia[],const PetscInt *ja[],PetscInt *spidx[],PetscBool  *done)
{
  Mat_SeqELL     *a = (Mat_SeqELL*)A->data;
  PetscInt       i,j,*collengths,*cia,*cja,n = A->cmap->n,totalslices;
  PetscInt       row,col;
  PetscInt       *cspidx;
  PetscBool      bflag;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *nn = n;
  if (!ia) PetscFunctionReturn(0);

  ierr    = PetscCalloc1(n+1,&collengths);CHKERRQ(ierr);
  ierr    = PetscMalloc1(n+1,&cia);CHKERRQ(ierr);
  ierr    = PetscMalloc1(a->nz+1,&cja);CHKERRQ(ierr);
  ierr    = PetscMalloc1(a->nz+1,&cspidx);CHKERRQ(ierr);

  totalslices = A->rmap->n/8+((A->rmap->n & 0x07)?1:0); /* floor(n/8) */
  for (i=0; i<totalslices; i++) { /* loop over slices */
    for (j=a->sliidx[i],row=0; j<a->sliidx[i+1]; j++,row=((row+1)&0x07)) {
      bflag = (PetscBool)(a->bt[j>>3] & (char)(1<<row));
      if (bflag) collengths[a->colidx[j]]++;
    }
  }

  cia[0] = oshift;
  for (i=0; i<n; i++) {
    cia[i+1] = cia[i] + collengths[i];
  }
  ierr = PetscMemzero(collengths,n*sizeof(PetscInt));CHKERRQ(ierr);

  for (i=0; i<totalslices; i++) { /* loop over slices */
    for (j=a->sliidx[i],row=0; j<a->sliidx[i+1]; j++,row=((row+1)&0x07)) {
      bflag = (PetscBool)(a->bt[j>>3] & (char)(1<<row));
      if (bflag) {
        col = a->colidx[j];
        cspidx[cia[col]+collengths[col]-oshift] = j; /* index of a->colidx */
        cja[cia[col]+collengths[col]-oshift]  = 8*i+row +oshift; /* row index */
        collengths[col]++;
      }
    }
  }

  ierr   = PetscFree(collengths);CHKERRQ(ierr);
  *ia    = cia; *ja = cja;
  *spidx = cspidx;
  PetscFunctionReturn(0);
}

PetscErrorCode MatRestoreColumnIJ_SeqELL_Color(Mat A,PetscInt oshift,PetscBool symmetric,PetscBool inodecompressed,PetscInt *n,const PetscInt *ia[],const PetscInt *ja[],PetscInt *spidx[],PetscBool  *done)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;

  if (!ia) PetscFunctionReturn(0);
  ierr = PetscFree(*ia);CHKERRQ(ierr);
  ierr = PetscFree(*ja);CHKERRQ(ierr);
  ierr = PetscFree(*spidx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
