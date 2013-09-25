
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/baij/seq/baij.h>
                              
/*
    This routine is shared by SeqAIJ and SeqBAIJ matrices, 
    since it operators only on the nonzero structure of the elements or blocks.
*/
#undef __FUNCT__
#define __FUNCT__ "MatFDColoringCreate_SeqXAIJ"
PetscErrorCode MatFDColoringCreate_SeqXAIJ(Mat mat,ISColoring iscoloring,MatFDColoring c)
{
  PetscErrorCode ierr;
  PetscInt       i,n,nrows,N,j,k,m,ncols,col,nis=iscoloring->n,*rowhit,bs,bs2,*spidx,nz;
  const PetscInt *is,*row,*ci,*cj;
  IS             *isa;
  PetscBool      isBAIJ;     
  PetscScalar    *A_val,**valaddrhit;
  MatEntry       *Jentry;

  PetscFunctionBegin;
  ierr = ISColoringGetIS(iscoloring,PETSC_IGNORE,&isa);CHKERRQ(ierr);

  /* this is ugly way to get blocksize but cannot call MatGetBlockSize() because AIJ can have bs > 1 */
  ierr = MatGetBlockSize(mat,&bs);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)mat,MATSEQBAIJ,&isBAIJ);CHKERRQ(ierr);
  if (isBAIJ) { 
    Mat_SeqBAIJ *spA = (Mat_SeqBAIJ*)mat->data;
    A_val = spA->a;
    nz    = spA->nz;
  } else {
    Mat_SeqAIJ  *spA = (Mat_SeqAIJ*)mat->data;
    A_val = spA->a;
    nz    = spA->nz;
    bs    = 1; /* only bs=1 is supported for SeqAIJ matrix */
  } 

  N         = mat->cmap->N/bs;
  c->M      = mat->rmap->N/bs;   /* set total rows, columns and local rows */
  c->N      = mat->cmap->N/bs;
  c->m      = mat->rmap->N/bs;
  c->rstart = 0;

  c->ncolors = nis;
  ierr       = PetscMalloc(nis*sizeof(PetscInt),&c->ncolumns);CHKERRQ(ierr);
  ierr       = PetscMalloc(nis*sizeof(PetscInt*),&c->columns);CHKERRQ(ierr);
  ierr       = PetscMalloc(nis*sizeof(PetscInt),&c->nrows);CHKERRQ(ierr);
  ierr       = PetscLogObjectMemory((PetscObject)c,3*nis*sizeof(PetscInt));CHKERRQ(ierr);

  ierr       = PetscMalloc(nz*sizeof(MatEntry),&Jentry);CHKERRQ(ierr);
  ierr       = PetscLogObjectMemory((PetscObject)c,nz*sizeof(MatEntry));CHKERRQ(ierr);
  c->matentry = Jentry;

  if (isBAIJ) {
    ierr = MatGetColumnIJ_SeqBAIJ_Color(mat,0,PETSC_FALSE,PETSC_FALSE,&ncols,&ci,&cj,&spidx,NULL);CHKERRQ(ierr);
  } else {
    ierr = MatGetColumnIJ_SeqAIJ_Color(mat,0,PETSC_FALSE,PETSC_FALSE,&ncols,&ci,&cj,&spidx,NULL);CHKERRQ(ierr);
  }
 
  ierr = PetscMalloc2(c->m,PetscInt,&rowhit,c->m,PetscScalar*,&valaddrhit);CHKERRQ(ierr);
  ierr = PetscMemzero(rowhit,c->m*sizeof(PetscInt));CHKERRQ(ierr);

  nz = 0;
  for (i=0; i<nis; i++) { /* loop over colors */
    ierr = ISGetLocalSize(isa[i],&n);CHKERRQ(ierr);
    ierr = ISGetIndices(isa[i],&is);CHKERRQ(ierr);

    c->ncolumns[i] = n;
    if (n) {
      ierr = PetscMalloc(n*sizeof(PetscInt),&c->columns[i]);CHKERRQ(ierr);
      ierr = PetscLogObjectMemory((PetscObject)c,n*sizeof(PetscInt));CHKERRQ(ierr);
      ierr = PetscMemcpy(c->columns[i],is,n*sizeof(PetscInt));CHKERRQ(ierr);
    } else {
      c->columns[i] = 0;
    }

    /* fast, crude version requires O(N*N) work */
    bs2   = bs*bs;
    nrows = 0;
    for (j=0; j<n; j++) {  /* loop over columns */
      col    = is[j];
      row    = cj + ci[col];
      m      = ci[col+1] - ci[col];  
      nrows += m;
      for (k=0; k<m; k++) {  /* loop over columns marking them in rowhit */
        rowhit[*row]       = col + 1;
        valaddrhit[*row++] = &A_val[bs2*spidx[ci[col] + k]]; 
      }
    }
    c->nrows[i] = nrows; /* total num of rows for this color */
   
    for (j=0; j<N; j++) { /* loop over rows */
      if (rowhit[j]) {
        Jentry[nz].row     = j;              /* local row index */
        Jentry[nz].col     = rowhit[j] - 1;  /* local column index */
        Jentry[nz].valaddr = valaddrhit[j];  /* address of mat value for this entry */ 
        nz++;
        rowhit[j] = 0.0;                     /* zero rowhit for reuse */
      }
    } 
    ierr = ISRestoreIndices(isa[i],&is);CHKERRQ(ierr);
  }

  if (isBAIJ) {
    ierr = MatRestoreColumnIJ_SeqBAIJ_Color(mat,0,PETSC_FALSE,PETSC_FALSE,&ncols,&ci,&cj,&spidx,NULL);CHKERRQ(ierr);
    ierr = PetscMalloc(bs*mat->rmap->n*sizeof(PetscScalar),&c->dy);CHKERRQ(ierr);
  } else {
    ierr = MatRestoreColumnIJ_SeqAIJ_Color(mat,0,PETSC_FALSE,PETSC_FALSE,&ncols,&ci,&cj,&spidx,NULL);CHKERRQ(ierr);
  }
  ierr = PetscFree2(rowhit,valaddrhit);CHKERRQ(ierr);
  ierr = ISColoringRestoreIS(iscoloring,&isa);CHKERRQ(ierr);

  c->ctype = IS_COLORING_GHOSTED;
  ierr = VecCreateGhost(PetscObjectComm((PetscObject)mat),mat->rmap->n,PETSC_DETERMINE,0,NULL,&c->vscale);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}
