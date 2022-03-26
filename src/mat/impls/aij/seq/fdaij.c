#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/baij/seq/baij.h>
#include <../src/mat/impls/sell/seq/sell.h>
#include <petsc/private/isimpl.h>

/*
    This routine is shared by SeqAIJ and SeqBAIJ matrices,
    since it operators only on the nonzero structure of the elements or blocks.
*/
PetscErrorCode MatFDColoringCreate_SeqXAIJ(Mat mat,ISColoring iscoloring,MatFDColoring c)
{
  PetscInt       bs,nis=iscoloring->n,m=mat->rmap->n;
  PetscBool      isBAIJ,isSELL;

  PetscFunctionBegin;
  /* set default brows and bcols for speedup inserting the dense matrix into sparse Jacobian */
  PetscCall(MatGetBlockSize(mat,&bs));
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)mat,MATSEQBAIJ,&isBAIJ));
  PetscCall(PetscObjectTypeCompare((PetscObject)mat,MATSEQSELL,&isSELL));
  if (isBAIJ) {
    c->brows = m;
    c->bcols = 1;
  } else { /* seqaij matrix */
    /* bcols is chosen s.t. dy-array takes 50% of memory space as mat */
    PetscReal  mem;
    PetscInt   nz,brows,bcols;
    if (isSELL) {
      Mat_SeqSELL *spA = (Mat_SeqSELL*)mat->data;
      nz = spA->nz;
    } else {
      Mat_SeqAIJ *spA = (Mat_SeqAIJ*)mat->data;
      nz = spA->nz;
    }

    bs    = 1; /* only bs=1 is supported for SeqAIJ matrix */
    mem   = nz*(sizeof(PetscScalar) + sizeof(PetscInt)) + 3*m*sizeof(PetscInt);
    bcols = (PetscInt)(0.5*mem /(m*sizeof(PetscScalar)));
    brows = 1000/bcols;
    if (bcols > nis) bcols = nis;
    if (brows == 0 || brows > m) brows = m;
    c->brows = brows;
    c->bcols = bcols;
  }

  c->M       = mat->rmap->N/bs;   /* set total rows, columns and local rows */
  c->N       = mat->cmap->N/bs;
  c->m       = mat->rmap->N/bs;
  c->rstart  = 0;
  c->ncolors = nis;
  c->ctype   = iscoloring->ctype;
  PetscFunctionReturn(0);
}

/*
 Reorder Jentry such that blocked brows*bols of entries from dense matrix are inserted into Jacobian for improved cache performance
   Input Parameters:
+  mat - the matrix containing the nonzero structure of the Jacobian
.  color - the coloring context
-  nz - number of local non-zeros in mat
*/
PetscErrorCode MatFDColoringSetUpBlocked_AIJ_Private(Mat mat,MatFDColoring c,PetscInt nz)
{
  PetscInt       i,j,nrows,nbcols,brows=c->brows,bcols=c->bcols,mbs=c->m,nis=c->ncolors;
  PetscInt       *color_start,*row_start,*nrows_new,nz_new,row_end;

  PetscFunctionBegin;
  if (brows < 1 || brows > mbs) brows = mbs;
  PetscCall(PetscMalloc2(bcols+1,&color_start,bcols,&row_start));
  PetscCall(PetscCalloc1(nis,&nrows_new));
  PetscCall(PetscMalloc1(bcols*mat->rmap->n,&c->dy));
  PetscCall(PetscLogObjectMemory((PetscObject)c,bcols*mat->rmap->n*sizeof(PetscScalar)));

  nz_new = 0;
  nbcols = 0;
  color_start[bcols] = 0;

  if (c->htype[0] == 'd') { /* ----  c->htype == 'ds', use MatEntry --------*/
    MatEntry *Jentry_new,*Jentry=c->matentry;

    PetscCall(PetscMalloc1(nz,&Jentry_new));
    for (i=0; i<nis; i+=bcols) { /* loop over colors */
      if (i + bcols > nis) {
        color_start[nis - i] = color_start[bcols];
        bcols                = nis - i;
      }

      color_start[0] = color_start[bcols];
      for (j=0; j<bcols; j++) {
        color_start[j+1] = c->nrows[i+j] + color_start[j];
        row_start[j]     = 0;
      }

      row_end = brows;
      if (row_end > mbs) row_end = mbs;

      while (row_end <= mbs) {   /* loop over block rows */
        for (j=0; j<bcols; j++) {       /* loop over block columns */
          nrows = c->nrows[i+j];
          nz    = color_start[j];
          while (row_start[j] < nrows) {
            if (Jentry[nz].row >= row_end) {
              color_start[j] = nz;
              break;
            } else { /* copy Jentry[nz] to Jentry_new[nz_new] */
              Jentry_new[nz_new].row     = Jentry[nz].row + j*mbs; /* index in dy-array */
              Jentry_new[nz_new].col     = Jentry[nz].col;
              Jentry_new[nz_new].valaddr = Jentry[nz].valaddr;
              nz_new++; nz++; row_start[j]++;
            }
          }
        }
        if (row_end == mbs) break;
        row_end += brows;
        if (row_end > mbs) row_end = mbs;
      }
      nrows_new[nbcols++] = nz_new;
    }
    PetscCall(PetscFree(Jentry));
    c->matentry = Jentry_new;
  } else { /* ---------  c->htype == 'wp', use MatEntry2 ------------------*/
    MatEntry2 *Jentry2_new,*Jentry2=c->matentry2;

    PetscCall(PetscMalloc1(nz,&Jentry2_new));
    for (i=0; i<nis; i+=bcols) { /* loop over colors */
      if (i + bcols > nis) {
        color_start[nis - i] = color_start[bcols];
        bcols                = nis - i;
      }

      color_start[0] = color_start[bcols];
      for (j=0; j<bcols; j++) {
        color_start[j+1] = c->nrows[i+j] + color_start[j];
        row_start[j]     = 0;
      }

      row_end = brows;
      if (row_end > mbs) row_end = mbs;

      while (row_end <= mbs) {   /* loop over block rows */
        for (j=0; j<bcols; j++) {       /* loop over block columns */
          nrows = c->nrows[i+j];
          nz    = color_start[j];
          while (row_start[j] < nrows) {
            if (Jentry2[nz].row >= row_end) {
              color_start[j] = nz;
              break;
            } else { /* copy Jentry2[nz] to Jentry2_new[nz_new] */
              Jentry2_new[nz_new].row     = Jentry2[nz].row + j*mbs; /* index in dy-array */
              Jentry2_new[nz_new].valaddr = Jentry2[nz].valaddr;
              nz_new++; nz++; row_start[j]++;
            }
          }
        }
        if (row_end == mbs) break;
        row_end += brows;
        if (row_end > mbs) row_end = mbs;
      }
      nrows_new[nbcols++] = nz_new;
    }
    PetscCall(PetscFree(Jentry2));
    c->matentry2 = Jentry2_new;
  } /* ---------------------------------------------*/

  PetscCall(PetscFree2(color_start,row_start));

  for (i=nbcols-1; i>0; i--) nrows_new[i] -= nrows_new[i-1];
  PetscCall(PetscFree(c->nrows));
  c->nrows = nrows_new;
  PetscFunctionReturn(0);
}

PetscErrorCode MatFDColoringSetUp_SeqXAIJ(Mat mat,ISColoring iscoloring,MatFDColoring c)
{
  PetscInt          i,n,nrows,mbs=c->m,j,k,m,ncols,col,nis=iscoloring->n,*rowhit,bs,bs2,*spidx,nz,tmp;
  const PetscInt    *is,*row,*ci,*cj;
  PetscBool         isBAIJ,isSELL;
  const PetscScalar *A_val;
  PetscScalar       **valaddrhit;
  MatEntry          *Jentry;
  MatEntry2         *Jentry2;

  PetscFunctionBegin;
  PetscCall(ISColoringGetIS(iscoloring,PETSC_OWN_POINTER,PETSC_IGNORE,&c->isa));

  PetscCall(MatGetBlockSize(mat,&bs));
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)mat,MATSEQBAIJ,&isBAIJ));
  PetscCall(PetscObjectTypeCompare((PetscObject)mat,MATSEQSELL,&isSELL));
  if (isBAIJ) {
    Mat_SeqBAIJ *spA = (Mat_SeqBAIJ*)mat->data;

    A_val = spA->a;
    nz    = spA->nz;
  } else if (isSELL) {
    Mat_SeqSELL *spA = (Mat_SeqSELL*)mat->data;

    A_val = spA->val;
    nz    = spA->nz;
    bs    = 1; /* only bs=1 is supported for SeqSELL matrix */
  } else {
    Mat_SeqAIJ *spA = (Mat_SeqAIJ*)mat->data;

    A_val = spA->a;
    nz    = spA->nz;
    bs    = 1; /* only bs=1 is supported for SeqAIJ matrix */
  }

  PetscCall(PetscMalloc2(nis,&c->ncolumns,nis,&c->columns));
  PetscCall(PetscMalloc1(nis,&c->nrows)); /* nrows is freeed separately from ncolumns and columns */
  PetscCall(PetscLogObjectMemory((PetscObject)c,3*nis*sizeof(PetscInt)));

  if (c->htype[0] == 'd') {
    PetscCall(PetscMalloc1(nz,&Jentry));
    PetscCall(PetscLogObjectMemory((PetscObject)c,nz*sizeof(MatEntry)));
    c->matentry = Jentry;
  } else if (c->htype[0] == 'w') {
    PetscCall(PetscMalloc1(nz,&Jentry2));
    PetscCall(PetscLogObjectMemory((PetscObject)c,nz*sizeof(MatEntry2)));
    c->matentry2 = Jentry2;
  } else SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"htype is not supported");

  if (isBAIJ) {
    PetscCall(MatGetColumnIJ_SeqBAIJ_Color(mat,0,PETSC_FALSE,PETSC_FALSE,&ncols,&ci,&cj,&spidx,NULL));
  } else if (isSELL) {
    PetscCall(MatGetColumnIJ_SeqSELL_Color(mat,0,PETSC_FALSE,PETSC_FALSE,&ncols,&ci,&cj,&spidx,NULL));
  } else {
    PetscCall(MatGetColumnIJ_SeqAIJ_Color(mat,0,PETSC_FALSE,PETSC_FALSE,&ncols,&ci,&cj,&spidx,NULL));
  }

  PetscCall(PetscCalloc1(c->m,&rowhit));
  PetscCall(PetscMalloc1(c->m,&valaddrhit));

  nz = 0;
  for (i=0; i<nis; i++) { /* loop over colors */
    PetscCall(ISGetLocalSize(c->isa[i],&n));
    PetscCall(ISGetIndices(c->isa[i],&is));

    c->ncolumns[i] = n;
    c->columns[i]  = (PetscInt*)is;
    /* note: we know that c->isa is going to be around as long at the c->columns values */
    PetscCall(ISRestoreIndices(c->isa[i],&is));

    /* fast, crude version requires O(N*N) work */
    bs2   = bs*bs;
    nrows = 0;
    for (j=0; j<n; j++) {  /* loop over columns */
      col    = is[j];
      tmp    = ci[col];
      row    = cj + tmp;
      m      = ci[col+1] - tmp;
      nrows += m;
      for (k=0; k<m; k++) {  /* loop over columns marking them in rowhit */
        rowhit[*row]       = col + 1;
        valaddrhit[*row++] = (PetscScalar*)&A_val[bs2*spidx[tmp + k]];
      }
    }
    c->nrows[i] = nrows; /* total num of rows for this color */

    if (c->htype[0] == 'd') {
      for (j=0; j<mbs; j++) { /* loop over rows */
        if (rowhit[j]) {
          Jentry[nz].row     = j;              /* local row index */
          Jentry[nz].col     = rowhit[j] - 1;  /* local column index */
          Jentry[nz].valaddr = valaddrhit[j];  /* address of mat value for this entry */
          nz++;
          rowhit[j] = 0.0;                     /* zero rowhit for reuse */
        }
      }
    }  else { /* c->htype == 'wp' */
      for (j=0; j<mbs; j++) { /* loop over rows */
        if (rowhit[j]) {
          Jentry2[nz].row     = j;              /* local row index */
          Jentry2[nz].valaddr = valaddrhit[j];  /* address of mat value for this entry */
          nz++;
          rowhit[j] = 0.0;                     /* zero rowhit for reuse */
        }
      }
    }
  }

  if (c->bcols > 1) {  /* reorder Jentry for faster MatFDColoringApply() */
    PetscCall(MatFDColoringSetUpBlocked_AIJ_Private(mat,c,nz));
  }

  if (isBAIJ) {
    PetscCall(MatRestoreColumnIJ_SeqBAIJ_Color(mat,0,PETSC_FALSE,PETSC_FALSE,&ncols,&ci,&cj,&spidx,NULL));
    PetscCall(PetscMalloc1(bs*mat->rmap->n,&c->dy));
    PetscCall(PetscLogObjectMemory((PetscObject)c,bs*mat->rmap->n*sizeof(PetscScalar)));
  } else if (isSELL) {
    PetscCall(MatRestoreColumnIJ_SeqSELL_Color(mat,0,PETSC_FALSE,PETSC_FALSE,&ncols,&ci,&cj,&spidx,NULL));
  } else {
    PetscCall(MatRestoreColumnIJ_SeqAIJ_Color(mat,0,PETSC_FALSE,PETSC_FALSE,&ncols,&ci,&cj,&spidx,NULL));
  }
  PetscCall(PetscFree(rowhit));
  PetscCall(PetscFree(valaddrhit));
  PetscCall(ISColoringRestoreIS(iscoloring,PETSC_OWN_POINTER,&c->isa));

  PetscCall(VecCreateGhost(PetscObjectComm((PetscObject)mat),mat->rmap->n,PETSC_DETERMINE,0,NULL,&c->vscale));
  PetscCall(PetscInfo(c,"ncolors %" PetscInt_FMT ", brows %" PetscInt_FMT " and bcols %" PetscInt_FMT " are used.\n",c->ncolors,c->brows,c->bcols));
  PetscFunctionReturn(0);
}
