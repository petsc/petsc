#include <../src/mat/impls/sell/seq/sell.h>
#include <../src/mat/impls/aij/seq/aij.h>
#include <petsc/private/isimpl.h>

/*
 MatGetColumnIJ_SeqSELL_Color() and MatRestoreColumnIJ_SeqSELL_Color() are customized from
 MatGetColumnIJ_SeqSELL() and MatRestoreColumnIJ_SeqSELL() by adding an output
 spidx[], index of a->a, to be used in MatTransposeColoringCreate_SeqSELL() and MatFDColoringCreate_SeqSELL()
*/
PetscErrorCode MatGetColumnIJ_SeqSELL_Color(Mat A,PetscInt oshift,PetscBool symmetric,PetscBool inodecompressed,PetscInt *nn,const PetscInt *ia[],const PetscInt *ja[],PetscInt *spidx[],PetscBool *done)
{
  Mat_SeqSELL    *a = (Mat_SeqSELL*)A->data;
  PetscInt       i,j,*collengths,*cia,*cja,n = A->cmap->n,totalslices;
  PetscInt       row,col;
  PetscInt       *cspidx;
  PetscBool      isnonzero;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *nn = n;
  if (!ia) PetscFunctionReturn(0);

  ierr = PetscCalloc1(n+1,&collengths);CHKERRQ(ierr);
  ierr = PetscMalloc1(n+1,&cia);CHKERRQ(ierr);
  ierr = PetscMalloc1(a->nz+1,&cja);CHKERRQ(ierr);
  ierr = PetscMalloc1(a->nz+1,&cspidx);CHKERRQ(ierr);

  totalslices = A->rmap->n/8+((A->rmap->n & 0x07)?1:0); /* floor(n/8) */
  for (i=0; i<totalslices; i++) { /* loop over slices */
    for (j=a->sliidx[i],row=0; j<a->sliidx[i+1]; j++,row=((row+1)&0x07)) {
      isnonzero = (PetscBool)((j-a->sliidx[i])/8 < a->rlen[8*i+row]);
      if (isnonzero) collengths[a->colidx[j]]++;
    }
  }

  cia[0] = oshift;
  for (i=0; i<n; i++) {
    cia[i+1] = cia[i] + collengths[i];
  }
  ierr = PetscArrayzero(collengths,n);CHKERRQ(ierr);

  for (i=0; i<totalslices; i++) { /* loop over slices */
    for (j=a->sliidx[i],row=0; j<a->sliidx[i+1]; j++,row=((row+1)&0x07)) {
      isnonzero = (PetscBool)((j-a->sliidx[i])/8 < a->rlen[8*i+row]);
      if (isnonzero) {
        col = a->colidx[j];
        cspidx[cia[col]+collengths[col]-oshift] = j; /* index of a->colidx */
        cja[cia[col]+collengths[col]-oshift] = 8*i+row +oshift; /* row index */
        collengths[col]++;
      }
    }
  }

  ierr   = PetscFree(collengths);CHKERRQ(ierr);
  *ia    = cia; *ja = cja;
  *spidx = cspidx;
  PetscFunctionReturn(0);
}

PetscErrorCode MatRestoreColumnIJ_SeqSELL_Color(Mat A,PetscInt oshift,PetscBool symmetric,PetscBool inodecompressed,PetscInt *n,const PetscInt *ia[],const PetscInt *ja[],PetscInt *spidx[],PetscBool *done)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;

  if (!ia) PetscFunctionReturn(0);
  ierr = PetscFree(*ia);CHKERRQ(ierr);
  ierr = PetscFree(*ja);CHKERRQ(ierr);
  ierr = PetscFree(*spidx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
