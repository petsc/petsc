
/*
   This provides a matrix that consists of Mats
*/

#include <petsc/private/matimpl.h>              /*I "petscmat.h" I*/
#include <../src/mat/impls/baij/seq/baij.h>    /* use the common AIJ data-structure */

typedef struct {
  SEQAIJHEADER(Mat);
  SEQBAIJHEADER;
  Mat *diags;

  Vec left,right,middle,workb;                 /* dummy vectors to perform local parts of product */
} Mat_BlockMat;

static PetscErrorCode MatSOR_BlockMat_Symmetric(Mat A,Vec bb,PetscReal omega,MatSORType flag,PetscReal fshift,PetscInt its,PetscInt lits,Vec xx)
{
  Mat_BlockMat      *a = (Mat_BlockMat*)A->data;
  PetscScalar       *x;
  const Mat         *v;
  const PetscScalar *b;
  PetscInt          n = A->cmap->n,i,mbs = n/A->rmap->bs,j,bs = A->rmap->bs;
  const PetscInt    *idx;
  IS                row,col;
  MatFactorInfo     info;
  Vec               left = a->left,right = a->right, middle = a->middle;
  Mat               *diag;

  PetscFunctionBegin;
  its = its*lits;
  PetscCheck(its > 0,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Relaxation requires global its %" PetscInt_FMT " and local its %" PetscInt_FMT " both positive",its,lits);
  PetscCheckFalse(flag & SOR_EISENSTAT,PETSC_COMM_SELF,PETSC_ERR_SUP,"No support yet for Eisenstat");
  PetscCheck(omega == 1.0,PETSC_COMM_SELF,PETSC_ERR_SUP,"No support yet for omega not equal to 1.0");
  PetscCheck(!fshift,PETSC_COMM_SELF,PETSC_ERR_SUP,"No support yet for fshift");
  if ((flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP) && !(flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP)) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot do backward sweep without forward sweep");
  }

  if (!a->diags) {
    PetscCall(PetscMalloc1(mbs,&a->diags));
    PetscCall(MatFactorInfoInitialize(&info));
    for (i=0; i<mbs; i++) {
      PetscCall(MatGetOrdering(a->a[a->diag[i]], MATORDERINGND,&row,&col));
      PetscCall(MatCholeskyFactorSymbolic(a->diags[i],a->a[a->diag[i]],row,&info));
      PetscCall(MatCholeskyFactorNumeric(a->diags[i],a->a[a->diag[i]],&info));
      PetscCall(ISDestroy(&row));
      PetscCall(ISDestroy(&col));
    }
    PetscCall(VecDuplicate(bb,&a->workb));
  }
  diag = a->diags;

  PetscCall(VecSet(xx,0.0));
  PetscCall(VecGetArray(xx,&x));
  /* copy right hand side because it must be modified during iteration */
  PetscCall(VecCopy(bb,a->workb));
  PetscCall(VecGetArrayRead(a->workb,&b));

  /* need to add code for when initial guess is zero, see MatSOR_SeqAIJ */
  while (its--) {
    if (flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP) {

      for (i=0; i<mbs; i++) {
        n   = a->i[i+1] - a->i[i] - 1;
        idx = a->j + a->i[i] + 1;
        v   = a->a + a->i[i] + 1;

        PetscCall(VecSet(left,0.0));
        for (j=0; j<n; j++) {
          PetscCall(VecPlaceArray(right,x + idx[j]*bs));
          PetscCall(MatMultAdd(v[j],right,left,left));
          PetscCall(VecResetArray(right));
        }
        PetscCall(VecPlaceArray(right,b + i*bs));
        PetscCall(VecAYPX(left,-1.0,right));
        PetscCall(VecResetArray(right));

        PetscCall(VecPlaceArray(right,x + i*bs));
        PetscCall(MatSolve(diag[i],left,right));

        /* now adjust right hand side, see MatSOR_SeqSBAIJ */
        for (j=0; j<n; j++) {
          PetscCall(MatMultTranspose(v[j],right,left));
          PetscCall(VecPlaceArray(middle,b + idx[j]*bs));
          PetscCall(VecAXPY(middle,-1.0,left));
          PetscCall(VecResetArray(middle));
        }
        PetscCall(VecResetArray(right));

      }
    }
    if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP) {

      for (i=mbs-1; i>=0; i--) {
        n   = a->i[i+1] - a->i[i] - 1;
        idx = a->j + a->i[i] + 1;
        v   = a->a + a->i[i] + 1;

        PetscCall(VecSet(left,0.0));
        for (j=0; j<n; j++) {
          PetscCall(VecPlaceArray(right,x + idx[j]*bs));
          PetscCall(MatMultAdd(v[j],right,left,left));
          PetscCall(VecResetArray(right));
        }
        PetscCall(VecPlaceArray(right,b + i*bs));
        PetscCall(VecAYPX(left,-1.0,right));
        PetscCall(VecResetArray(right));

        PetscCall(VecPlaceArray(right,x + i*bs));
        PetscCall(MatSolve(diag[i],left,right));
        PetscCall(VecResetArray(right));

      }
    }
  }
  PetscCall(VecRestoreArray(xx,&x));
  PetscCall(VecRestoreArrayRead(a->workb,&b));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSOR_BlockMat(Mat A,Vec bb,PetscReal omega,MatSORType flag,PetscReal fshift,PetscInt its,PetscInt lits,Vec xx)
{
  Mat_BlockMat      *a = (Mat_BlockMat*)A->data;
  PetscScalar       *x;
  const Mat         *v;
  const PetscScalar *b;
  PetscInt          n = A->cmap->n,i,mbs = n/A->rmap->bs,j,bs = A->rmap->bs;
  const PetscInt    *idx;
  IS                row,col;
  MatFactorInfo     info;
  Vec               left = a->left,right = a->right;
  Mat               *diag;

  PetscFunctionBegin;
  its = its*lits;
  PetscCheck(its > 0,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Relaxation requires global its %" PetscInt_FMT " and local its %" PetscInt_FMT " both positive",its,lits);
  PetscCheckFalse(flag & SOR_EISENSTAT,PETSC_COMM_SELF,PETSC_ERR_SUP,"No support yet for Eisenstat");
  PetscCheck(omega == 1.0,PETSC_COMM_SELF,PETSC_ERR_SUP,"No support yet for omega not equal to 1.0");
  PetscCheck(!fshift,PETSC_COMM_SELF,PETSC_ERR_SUP,"No support yet for fshift");

  if (!a->diags) {
    PetscCall(PetscMalloc1(mbs,&a->diags));
    PetscCall(MatFactorInfoInitialize(&info));
    for (i=0; i<mbs; i++) {
      PetscCall(MatGetOrdering(a->a[a->diag[i]], MATORDERINGND,&row,&col));
      PetscCall(MatLUFactorSymbolic(a->diags[i],a->a[a->diag[i]],row,col,&info));
      PetscCall(MatLUFactorNumeric(a->diags[i],a->a[a->diag[i]],&info));
      PetscCall(ISDestroy(&row));
      PetscCall(ISDestroy(&col));
    }
  }
  diag = a->diags;

  PetscCall(VecSet(xx,0.0));
  PetscCall(VecGetArray(xx,&x));
  PetscCall(VecGetArrayRead(bb,&b));

  /* need to add code for when initial guess is zero, see MatSOR_SeqAIJ */
  while (its--) {
    if (flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP) {

      for (i=0; i<mbs; i++) {
        n   = a->i[i+1] - a->i[i];
        idx = a->j + a->i[i];
        v   = a->a + a->i[i];

        PetscCall(VecSet(left,0.0));
        for (j=0; j<n; j++) {
          if (idx[j] != i) {
            PetscCall(VecPlaceArray(right,x + idx[j]*bs));
            PetscCall(MatMultAdd(v[j],right,left,left));
            PetscCall(VecResetArray(right));
          }
        }
        PetscCall(VecPlaceArray(right,b + i*bs));
        PetscCall(VecAYPX(left,-1.0,right));
        PetscCall(VecResetArray(right));

        PetscCall(VecPlaceArray(right,x + i*bs));
        PetscCall(MatSolve(diag[i],left,right));
        PetscCall(VecResetArray(right));
      }
    }
    if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP) {

      for (i=mbs-1; i>=0; i--) {
        n   = a->i[i+1] - a->i[i];
        idx = a->j + a->i[i];
        v   = a->a + a->i[i];

        PetscCall(VecSet(left,0.0));
        for (j=0; j<n; j++) {
          if (idx[j] != i) {
            PetscCall(VecPlaceArray(right,x + idx[j]*bs));
            PetscCall(MatMultAdd(v[j],right,left,left));
            PetscCall(VecResetArray(right));
          }
        }
        PetscCall(VecPlaceArray(right,b + i*bs));
        PetscCall(VecAYPX(left,-1.0,right));
        PetscCall(VecResetArray(right));

        PetscCall(VecPlaceArray(right,x + i*bs));
        PetscCall(MatSolve(diag[i],left,right));
        PetscCall(VecResetArray(right));

      }
    }
  }
  PetscCall(VecRestoreArray(xx,&x));
  PetscCall(VecRestoreArrayRead(bb,&b));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetValues_BlockMat(Mat A,PetscInt m,const PetscInt im[],PetscInt n,const PetscInt in[],const PetscScalar v[],InsertMode is)
{
  Mat_BlockMat   *a = (Mat_BlockMat*)A->data;
  PetscInt       *rp,k,low,high,t,ii,row,nrow,i,col,l,rmax,N,lastcol = -1;
  PetscInt       *imax=a->imax,*ai=a->i,*ailen=a->ilen;
  PetscInt       *aj  =a->j,nonew=a->nonew,bs=A->rmap->bs,brow,bcol;
  PetscInt       ridx,cidx;
  PetscBool      roworiented=a->roworiented;
  MatScalar      value;
  Mat            *ap,*aa = a->a;

  PetscFunctionBegin;
  for (k=0; k<m; k++) { /* loop over added rows */
    row  = im[k];
    brow = row/bs;
    if (row < 0) continue;
    PetscCheck(row < A->rmap->N,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %" PetscInt_FMT " max %" PetscInt_FMT,row,A->rmap->N-1);
    rp   = aj + ai[brow];
    ap   = aa + ai[brow];
    rmax = imax[brow];
    nrow = ailen[brow];
    low  = 0;
    high = nrow;
    for (l=0; l<n; l++) { /* loop over added columns */
      if (in[l] < 0) continue;
      PetscCheck(in[l] < A->cmap->n,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Column too large: col %" PetscInt_FMT " max %" PetscInt_FMT,in[l],A->cmap->n-1);
      col = in[l]; bcol = col/bs;
      if (A->symmetric && brow > bcol) continue;
      ridx = row % bs; cidx = col % bs;
      if (roworiented) value = v[l + k*n];
      else value = v[k + l*m];

      if (col <= lastcol) low = 0;
      else high = nrow;
      lastcol = col;
      while (high-low > 7) {
        t = (low+high)/2;
        if (rp[t] > bcol) high = t;
        else              low  = t;
      }
      for (i=low; i<high; i++) {
        if (rp[i] > bcol) break;
        if (rp[i] == bcol) goto noinsert1;
      }
      if (nonew == 1) goto noinsert1;
      PetscCheck(nonew != -1,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new nonzero (%" PetscInt_FMT ", %" PetscInt_FMT ") in the matrix", row, col);
      MatSeqXAIJReallocateAIJ(A,a->mbs,1,nrow,brow,bcol,rmax,aa,ai,aj,rp,ap,imax,nonew,Mat);
      N = nrow++ - 1; high++;
      /* shift up all the later entries in this row */
      for (ii=N; ii>=i; ii--) {
        rp[ii+1] = rp[ii];
        ap[ii+1] = ap[ii];
      }
      if (N>=i) ap[i] = NULL;
      rp[i] = bcol;
      a->nz++;
      A->nonzerostate++;
noinsert1:;
      if (!*(ap+i)) {
        PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF,bs,bs,0,NULL,ap+i));
      }
      PetscCall(MatSetValues(ap[i],1,&ridx,1,&cidx,&value,is));
      low  = i;
    }
    ailen[brow] = nrow;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatLoad_BlockMat(Mat newmat, PetscViewer viewer)
{
  Mat               tmpA;
  PetscInt          i,j,m,n,bs = 1,ncols,*lens,currentcol,mbs,**ii,*ilens,nextcol,*llens,cnt = 0;
  const PetscInt    *cols;
  const PetscScalar *values;
  PetscBool         flg = PETSC_FALSE,notdone;
  Mat_SeqAIJ        *a;
  Mat_BlockMat      *amat;

  PetscFunctionBegin;
  /* force binary viewer to load .info file if it has not yet done so */
  PetscCall(PetscViewerSetUp(viewer));
  PetscCall(MatCreate(PETSC_COMM_SELF,&tmpA));
  PetscCall(MatSetType(tmpA,MATSEQAIJ));
  PetscCall(MatLoad_SeqAIJ(tmpA,viewer));

  PetscCall(MatGetLocalSize(tmpA,&m,&n));
  PetscOptionsBegin(PETSC_COMM_SELF,NULL,"Options for loading BlockMat matrix 1","Mat");
  PetscCall(PetscOptionsInt("-matload_block_size","Set the blocksize used to store the matrix","MatLoad",bs,&bs,NULL));
  PetscCall(PetscOptionsBool("-matload_symmetric","Store the matrix as symmetric","MatLoad",flg,&flg,NULL));
  PetscOptionsEnd();

  /* Determine number of nonzero blocks for each block row */
  a    = (Mat_SeqAIJ*) tmpA->data;
  mbs  = m/bs;
  PetscCall(PetscMalloc3(mbs,&lens,bs,&ii,bs,&ilens));
  PetscCall(PetscArrayzero(lens,mbs));

  for (i=0; i<mbs; i++) {
    for (j=0; j<bs; j++) {
      ii[j]    = a->j + a->i[i*bs + j];
      ilens[j] = a->i[i*bs + j + 1] - a->i[i*bs + j];
    }

    currentcol = -1;
    while (PETSC_TRUE) {
      notdone = PETSC_FALSE;
      nextcol = 1000000000;
      for (j=0; j<bs; j++) {
        while ((ilens[j] > 0 && ii[j][0]/bs <= currentcol)) {
          ii[j]++;
          ilens[j]--;
        }
        if (ilens[j] > 0) {
          notdone = PETSC_TRUE;
          nextcol = PetscMin(nextcol,ii[j][0]/bs);
        }
      }
      if (!notdone) break;
      if (!flg || (nextcol >= i)) lens[i]++;
      currentcol = nextcol;
    }
  }

  if (newmat->rmap->n < 0 && newmat->rmap->N < 0 && newmat->cmap->n < 0 && newmat->cmap->N < 0) {
    PetscCall(MatSetSizes(newmat,m,n,PETSC_DETERMINE,PETSC_DETERMINE));
  }
  PetscCall(MatBlockMatSetPreallocation(newmat,bs,0,lens));
  if (flg) {
    PetscCall(MatSetOption(newmat,MAT_SYMMETRIC,PETSC_TRUE));
  }
  amat = (Mat_BlockMat*)(newmat)->data;

  /* preallocate the submatrices */
  PetscCall(PetscMalloc1(bs,&llens));
  for (i=0; i<mbs; i++) { /* loops for block rows */
    for (j=0; j<bs; j++) {
      ii[j]    = a->j + a->i[i*bs + j];
      ilens[j] = a->i[i*bs + j + 1] - a->i[i*bs + j];
    }

    currentcol = 1000000000;
    for (j=0; j<bs; j++) { /* loop over rows in block finding first nonzero block */
      if (ilens[j] > 0) {
        currentcol = PetscMin(currentcol,ii[j][0]/bs);
      }
    }

    while (PETSC_TRUE) {  /* loops over blocks in block row */
      notdone = PETSC_FALSE;
      nextcol = 1000000000;
      PetscCall(PetscArrayzero(llens,bs));
      for (j=0; j<bs; j++) { /* loop over rows in block */
        while ((ilens[j] > 0 && ii[j][0]/bs <= currentcol)) { /* loop over columns in row */
          ii[j]++;
          ilens[j]--;
          llens[j]++;
        }
        if (ilens[j] > 0) {
          notdone = PETSC_TRUE;
          nextcol = PetscMin(nextcol,ii[j][0]/bs);
        }
      }
      PetscCheck(cnt < amat->maxnz,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Number of blocks found greater than expected %" PetscInt_FMT,cnt);
      if (!flg || currentcol >= i) {
        amat->j[cnt] = currentcol;
        PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF,bs,bs,0,llens,amat->a+cnt++));
      }

      if (!notdone) break;
      currentcol = nextcol;
    }
    amat->ilen[i] = lens[i];
  }

  PetscCall(PetscFree3(lens,ii,ilens));
  PetscCall(PetscFree(llens));

  /* copy over the matrix, one row at a time */
  for (i=0; i<m; i++) {
    PetscCall(MatGetRow(tmpA,i,&ncols,&cols,&values));
    PetscCall(MatSetValues(newmat,1,&i,ncols,cols,values,INSERT_VALUES));
    PetscCall(MatRestoreRow(tmpA,i,&ncols,&cols,&values));
  }
  PetscCall(MatAssemblyBegin(newmat,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(newmat,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatView_BlockMat(Mat A,PetscViewer viewer)
{
  Mat_BlockMat      *a = (Mat_BlockMat*)A->data;
  const char        *name;
  PetscViewerFormat format;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetName((PetscObject)A,&name));
  PetscCall(PetscViewerGetFormat(viewer,&format));
  if (format == PETSC_VIEWER_ASCII_FACTOR_INFO || format == PETSC_VIEWER_ASCII_INFO) {
    PetscCall(PetscViewerASCIIPrintf(viewer,"Nonzero block matrices = %" PetscInt_FMT " \n",a->nz));
    if (A->symmetric) {
      PetscCall(PetscViewerASCIIPrintf(viewer,"Only upper triangular part of symmetric matrix is stored\n"));
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_BlockMat(Mat mat)
{
  Mat_BlockMat   *bmat = (Mat_BlockMat*)mat->data;
  PetscInt       i;

  PetscFunctionBegin;
  PetscCall(VecDestroy(&bmat->right));
  PetscCall(VecDestroy(&bmat->left));
  PetscCall(VecDestroy(&bmat->middle));
  PetscCall(VecDestroy(&bmat->workb));
  if (bmat->diags) {
    for (i=0; i<mat->rmap->n/mat->rmap->bs; i++) {
      PetscCall(MatDestroy(&bmat->diags[i]));
    }
  }
  if (bmat->a) {
    for (i=0; i<bmat->nz; i++) {
      PetscCall(MatDestroy(&bmat->a[i]));
    }
  }
  PetscCall(MatSeqXAIJFreeAIJ(mat,(PetscScalar**)&bmat->a,&bmat->j,&bmat->i));
  PetscCall(PetscFree(mat->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMult_BlockMat(Mat A,Vec x,Vec y)
{
  Mat_BlockMat   *bmat = (Mat_BlockMat*)A->data;
  PetscScalar    *xx,*yy;
  PetscInt       *aj,i,*ii,jrow,m = A->rmap->n/A->rmap->bs,bs = A->rmap->bs,n,j;
  Mat            *aa;

  PetscFunctionBegin;
  /*
     Standard CSR multiply except each entry is a Mat
  */
  PetscCall(VecGetArray(x,&xx));

  PetscCall(VecSet(y,0.0));
  PetscCall(VecGetArray(y,&yy));
  aj   = bmat->j;
  aa   = bmat->a;
  ii   = bmat->i;
  for (i=0; i<m; i++) {
    jrow = ii[i];
    PetscCall(VecPlaceArray(bmat->left,yy + bs*i));
    n    = ii[i+1] - jrow;
    for (j=0; j<n; j++) {
      PetscCall(VecPlaceArray(bmat->right,xx + bs*aj[jrow]));
      PetscCall(MatMultAdd(aa[jrow],bmat->right,bmat->left,bmat->left));
      PetscCall(VecResetArray(bmat->right));
      jrow++;
    }
    PetscCall(VecResetArray(bmat->left));
  }
  PetscCall(VecRestoreArray(x,&xx));
  PetscCall(VecRestoreArray(y,&yy));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMult_BlockMat_Symmetric(Mat A,Vec x,Vec y)
{
  Mat_BlockMat   *bmat = (Mat_BlockMat*)A->data;
  PetscScalar    *xx,*yy;
  PetscInt       *aj,i,*ii,jrow,m = A->rmap->n/A->rmap->bs,bs = A->rmap->bs,n,j;
  Mat            *aa;

  PetscFunctionBegin;
  /*
     Standard CSR multiply except each entry is a Mat
  */
  PetscCall(VecGetArray(x,&xx));

  PetscCall(VecSet(y,0.0));
  PetscCall(VecGetArray(y,&yy));
  aj   = bmat->j;
  aa   = bmat->a;
  ii   = bmat->i;
  for (i=0; i<m; i++) {
    jrow = ii[i];
    n    = ii[i+1] - jrow;
    PetscCall(VecPlaceArray(bmat->left,yy + bs*i));
    PetscCall(VecPlaceArray(bmat->middle,xx + bs*i));
    /* if we ALWAYS required a diagonal entry then could remove this if test */
    if (aj[jrow] == i) {
      PetscCall(VecPlaceArray(bmat->right,xx + bs*aj[jrow]));
      PetscCall(MatMultAdd(aa[jrow],bmat->right,bmat->left,bmat->left));
      PetscCall(VecResetArray(bmat->right));
      jrow++;
      n--;
    }
    for (j=0; j<n; j++) {
      PetscCall(VecPlaceArray(bmat->right,xx + bs*aj[jrow]));            /* upper triangular part */
      PetscCall(MatMultAdd(aa[jrow],bmat->right,bmat->left,bmat->left));
      PetscCall(VecResetArray(bmat->right));

      PetscCall(VecPlaceArray(bmat->right,yy + bs*aj[jrow]));            /* lower triangular part */
      PetscCall(MatMultTransposeAdd(aa[jrow],bmat->middle,bmat->right,bmat->right));
      PetscCall(VecResetArray(bmat->right));
      jrow++;
    }
    PetscCall(VecResetArray(bmat->left));
    PetscCall(VecResetArray(bmat->middle));
  }
  PetscCall(VecRestoreArray(x,&xx));
  PetscCall(VecRestoreArray(y,&yy));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultAdd_BlockMat(Mat A,Vec x,Vec y,Vec z)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTranspose_BlockMat(Mat A,Vec x,Vec y)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTransposeAdd_BlockMat(Mat A,Vec x,Vec y,Vec z)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

/*
     Adds diagonal pointers to sparse matrix structure.
*/
static PetscErrorCode MatMarkDiagonal_BlockMat(Mat A)
{
  Mat_BlockMat   *a = (Mat_BlockMat*)A->data;
  PetscInt       i,j,mbs = A->rmap->n/A->rmap->bs;

  PetscFunctionBegin;
  if (!a->diag) {
    PetscCall(PetscMalloc1(mbs,&a->diag));
  }
  for (i=0; i<mbs; i++) {
    a->diag[i] = a->i[i+1];
    for (j=a->i[i]; j<a->i[i+1]; j++) {
      if (a->j[j] == i) {
        a->diag[i] = j;
        break;
      }
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCreateSubMatrix_BlockMat(Mat A,IS isrow,IS iscol,MatReuse scall,Mat *B)
{
  Mat_BlockMat   *a = (Mat_BlockMat*)A->data;
  Mat_SeqAIJ     *c;
  PetscInt       i,k,first,step,lensi,nrows,ncols;
  PetscInt       *j_new,*i_new,*aj = a->j,*ailen = a->ilen;
  PetscScalar    *a_new;
  Mat            C,*aa = a->a;
  PetscBool      stride,equal;

  PetscFunctionBegin;
  PetscCall(ISEqual(isrow,iscol,&equal));
  PetscCheck(equal,PETSC_COMM_SELF,PETSC_ERR_SUP,"Only for identical column and row indices");
  PetscCall(PetscObjectTypeCompare((PetscObject)iscol,ISSTRIDE,&stride));
  PetscCheck(stride,PETSC_COMM_SELF,PETSC_ERR_SUP,"Only for stride indices");
  PetscCall(ISStrideGetInfo(iscol,&first,&step));
  PetscCheck(step == A->rmap->bs,PETSC_COMM_SELF,PETSC_ERR_SUP,"Can only select one entry from each block");

  PetscCall(ISGetLocalSize(isrow,&nrows));
  ncols = nrows;

  /* create submatrix */
  if (scall == MAT_REUSE_MATRIX) {
    PetscInt n_cols,n_rows;
    C    = *B;
    PetscCall(MatGetSize(C,&n_rows,&n_cols));
    PetscCheckFalse(n_rows != nrows || n_cols != ncols,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Reused submatrix wrong size");
    PetscCall(MatZeroEntries(C));
  } else {
    PetscCall(MatCreate(PetscObjectComm((PetscObject)A),&C));
    PetscCall(MatSetSizes(C,nrows,ncols,PETSC_DETERMINE,PETSC_DETERMINE));
    if (A->symmetric) {
      PetscCall(MatSetType(C,MATSEQSBAIJ));
    } else {
      PetscCall(MatSetType(C,MATSEQAIJ));
    }
    PetscCall(MatSeqAIJSetPreallocation(C,0,ailen));
    PetscCall(MatSeqSBAIJSetPreallocation(C,1,0,ailen));
  }
  c = (Mat_SeqAIJ*)C->data;

  /* loop over rows inserting into submatrix */
  a_new = c->a;
  j_new = c->j;
  i_new = c->i;

  for (i=0; i<nrows; i++) {
    lensi = ailen[i];
    for (k=0; k<lensi; k++) {
      *j_new++ = *aj++;
      PetscCall(MatGetValue(*aa++,first,first,a_new++));
    }
    i_new[i+1] = i_new[i] + lensi;
    c->ilen[i] = lensi;
  }

  PetscCall(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));
  *B   = C;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatAssemblyEnd_BlockMat(Mat A,MatAssemblyType mode)
{
  Mat_BlockMat   *a = (Mat_BlockMat*)A->data;
  PetscInt       fshift = 0,i,j,*ai = a->i,*aj = a->j,*imax = a->imax;
  PetscInt       m      = a->mbs,*ip,N,*ailen = a->ilen,rmax = 0;
  Mat            *aa    = a->a,*ap;

  PetscFunctionBegin;
  if (mode == MAT_FLUSH_ASSEMBLY) PetscFunctionReturn(0);

  if (m) rmax = ailen[0]; /* determine row with most nonzeros */
  for (i=1; i<m; i++) {
    /* move each row back by the amount of empty slots (fshift) before it*/
    fshift += imax[i-1] - ailen[i-1];
    rmax    = PetscMax(rmax,ailen[i]);
    if (fshift) {
      ip = aj + ai[i];
      ap = aa + ai[i];
      N  = ailen[i];
      for (j=0; j<N; j++) {
        ip[j-fshift] = ip[j];
        ap[j-fshift] = ap[j];
      }
    }
    ai[i] = ai[i-1] + ailen[i-1];
  }
  if (m) {
    fshift += imax[m-1] - ailen[m-1];
    ai[m]   = ai[m-1] + ailen[m-1];
  }
  /* reset ilen and imax for each row */
  for (i=0; i<m; i++) {
    ailen[i] = imax[i] = ai[i+1] - ai[i];
  }
  a->nz = ai[m];
  for (i=0; i<a->nz; i++) {
    PetscAssert(aa[i],PETSC_COMM_SELF,PETSC_ERR_PLIB,"Null matrix at location %" PetscInt_FMT " column %" PetscInt_FMT " nz %" PetscInt_FMT,i,aj[i],a->nz);
    PetscCall(MatAssemblyBegin(aa[i],MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(aa[i],MAT_FINAL_ASSEMBLY));
  }
  PetscCall(PetscInfo(A,"Matrix size: %" PetscInt_FMT " X %" PetscInt_FMT "; storage space: %" PetscInt_FMT " unneeded,%" PetscInt_FMT " used\n",m,A->cmap->n/A->cmap->bs,fshift,a->nz));
  PetscCall(PetscInfo(A,"Number of mallocs during MatSetValues() is %" PetscInt_FMT "\n",a->reallocs));
  PetscCall(PetscInfo(A,"Maximum nonzeros in any row is %" PetscInt_FMT "\n",rmax));

  A->info.mallocs    += a->reallocs;
  a->reallocs         = 0;
  A->info.nz_unneeded = (double)fshift;
  a->rmax             = rmax;
  PetscCall(MatMarkDiagonal_BlockMat(A));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetOption_BlockMat(Mat A,MatOption opt,PetscBool flg)
{
  PetscFunctionBegin;
  if (opt == MAT_SYMMETRIC && flg) {
    A->ops->sor  = MatSOR_BlockMat_Symmetric;
    A->ops->mult = MatMult_BlockMat_Symmetric;
  } else {
    PetscCall(PetscInfo(A,"Unused matrix option %s\n",MatOptions[opt]));
  }
  PetscFunctionReturn(0);
}

static struct _MatOps MatOps_Values = {MatSetValues_BlockMat,
                                       NULL,
                                       NULL,
                                       MatMult_BlockMat,
                               /*  4*/ MatMultAdd_BlockMat,
                                       MatMultTranspose_BlockMat,
                                       MatMultTransposeAdd_BlockMat,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 10*/ NULL,
                                       NULL,
                                       NULL,
                                       MatSOR_BlockMat,
                                       NULL,
                               /* 15*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 20*/ NULL,
                                       MatAssemblyEnd_BlockMat,
                                       MatSetOption_BlockMat,
                                       NULL,
                               /* 24*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 29*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 34*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 39*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 44*/ NULL,
                                       NULL,
                                       MatShift_Basic,
                                       NULL,
                                       NULL,
                               /* 49*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 54*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 59*/ MatCreateSubMatrix_BlockMat,
                                       MatDestroy_BlockMat,
                                       MatView_BlockMat,
                                       NULL,
                                       NULL,
                               /* 64*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 69*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 74*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 79*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       MatLoad_BlockMat,
                               /* 84*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 89*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 94*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 99*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /*104*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /*109*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /*114*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /*119*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /*124*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /*129*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /*134*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /*139*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /*144*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL
};

/*@C
   MatBlockMatSetPreallocation - For good matrix assembly performance
   the user should preallocate the matrix storage by setting the parameter nz
   (or the array nnz).  By setting these parameters accurately, performance
   during matrix assembly can be increased by more than a factor of 50.

   Collective

   Input Parameters:
+  B - The matrix
.  bs - size of each block in matrix
.  nz - number of nonzeros per block row (same for all rows)
-  nnz - array containing the number of nonzeros in the various block rows
         (possibly different for each row) or NULL

   Notes:
     If nnz is given then nz is ignored

   Specify the preallocated storage with either nz or nnz (not both).
   Set nz=PETSC_DEFAULT and nnz=NULL for PETSc to control dynamic memory
   allocation.  For large problems you MUST preallocate memory or you
   will get TERRIBLE performance, see the users' manual chapter on matrices.

   Level: intermediate

.seealso: `MatCreate()`, `MatCreateBlockMat()`, `MatSetValues()`

@*/
PetscErrorCode  MatBlockMatSetPreallocation(Mat B,PetscInt bs,PetscInt nz,const PetscInt nnz[])
{
  PetscFunctionBegin;
  PetscTryMethod(B,"MatBlockMatSetPreallocation_C",(Mat,PetscInt,PetscInt,const PetscInt[]),(B,bs,nz,nnz));
  PetscFunctionReturn(0);
}

static PetscErrorCode  MatBlockMatSetPreallocation_BlockMat(Mat A,PetscInt bs,PetscInt nz,PetscInt *nnz)
{
  Mat_BlockMat   *bmat = (Mat_BlockMat*)A->data;
  PetscInt       i;

  PetscFunctionBegin;
  PetscCall(PetscLayoutSetBlockSize(A->rmap,bs));
  PetscCall(PetscLayoutSetBlockSize(A->cmap,bs));
  PetscCall(PetscLayoutSetUp(A->rmap));
  PetscCall(PetscLayoutSetUp(A->cmap));
  PetscCall(PetscLayoutGetBlockSize(A->rmap,&bs));

  if (nz == PETSC_DEFAULT || nz == PETSC_DECIDE) nz = 5;
  PetscCheck(nz >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"nz cannot be less than 0: value %" PetscInt_FMT,nz);
  if (nnz) {
    for (i=0; i<A->rmap->n/bs; i++) {
      PetscCheck(nnz[i] >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"nnz cannot be less than 0: local row %" PetscInt_FMT " value %" PetscInt_FMT,i,nnz[i]);
      PetscCheck(nnz[i] <= A->cmap->n/bs,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"nnz cannot be greater than row length: local row %" PetscInt_FMT " value %" PetscInt_FMT " rowlength %" PetscInt_FMT,i,nnz[i],A->cmap->n/bs);
    }
  }
  bmat->mbs = A->rmap->n/bs;

  PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF,1,bs,NULL,&bmat->right));
  PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF,1,bs,NULL,&bmat->middle));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF,bs,&bmat->left));

  if (!bmat->imax) {
    PetscCall(PetscMalloc2(A->rmap->n,&bmat->imax,A->rmap->n,&bmat->ilen));
    PetscCall(PetscLogObjectMemory((PetscObject)A,2*A->rmap->n*sizeof(PetscInt)));
  }
  if (PetscLikely(nnz)) {
    nz = 0;
    for (i=0; i<A->rmap->n/A->rmap->bs; i++) {
      bmat->imax[i] = nnz[i];
      nz           += nnz[i];
    }
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Currently requires block row by row preallocation");

  /* bmat->ilen will count nonzeros in each row so far. */
  PetscCall(PetscArrayzero(bmat->ilen,bmat->mbs));

  /* allocate the matrix space */
  PetscCall(MatSeqXAIJFreeAIJ(A,(PetscScalar**)&bmat->a,&bmat->j,&bmat->i));
  PetscCall(PetscMalloc3(nz,&bmat->a,nz,&bmat->j,A->rmap->n+1,&bmat->i));
  PetscCall(PetscLogObjectMemory((PetscObject)A,(A->rmap->n+1)*sizeof(PetscInt)+nz*(sizeof(PetscScalar)+sizeof(PetscInt))));
  bmat->i[0] = 0;
  for (i=1; i<bmat->mbs+1; i++) {
    bmat->i[i] = bmat->i[i-1] + bmat->imax[i-1];
  }
  bmat->singlemalloc = PETSC_TRUE;
  bmat->free_a       = PETSC_TRUE;
  bmat->free_ij      = PETSC_TRUE;

  bmat->nz            = 0;
  bmat->maxnz         = nz;
  A->info.nz_unneeded = (double)bmat->maxnz;
  PetscCall(MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE));
  PetscFunctionReturn(0);
}

/*MC
   MATBLOCKMAT - A matrix that is defined by a set of Mat's that represents a sparse block matrix
                 consisting of (usually) sparse blocks.

  Level: advanced

.seealso: `MatCreateBlockMat()`

M*/

PETSC_EXTERN PetscErrorCode MatCreate_BlockMat(Mat A)
{
  Mat_BlockMat   *b;

  PetscFunctionBegin;
  PetscCall(PetscNewLog(A,&b));
  A->data = (void*)b;
  PetscCall(PetscMemcpy(A->ops,&MatOps_Values,sizeof(struct _MatOps)));

  A->assembled    = PETSC_TRUE;
  A->preallocated = PETSC_FALSE;
  PetscCall(PetscObjectChangeTypeName((PetscObject)A,MATBLOCKMAT));

  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatBlockMatSetPreallocation_C",MatBlockMatSetPreallocation_BlockMat));
  PetscFunctionReturn(0);
}

/*@C
   MatCreateBlockMat - Creates a new matrix in which each block contains a uniform-size sequential Mat object

  Collective

   Input Parameters:
+  comm - MPI communicator
.  m - number of rows
.  n  - number of columns
.  bs - size of each submatrix
.  nz  - expected maximum number of nonzero blocks in row (use PETSC_DEFAULT if not known)
-  nnz - expected number of nonzers per block row if known (use NULL otherwise)

   Output Parameter:
.  A - the matrix

   Level: intermediate

   Notes:
    Matrices of this type are nominally-sparse matrices in which each "entry" is a Mat object.  Each Mat must
   have the same size and be sequential.  The local and global sizes must be compatible with this decomposition.

   For matrices containing parallel submatrices and variable block sizes, see MATNEST.

.seealso: `MATBLOCKMAT`, `MatCreateNest()`
@*/
PetscErrorCode  MatCreateBlockMat(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt bs,PetscInt nz,PetscInt *nnz, Mat *A)
{
  PetscFunctionBegin;
  PetscCall(MatCreate(comm,A));
  PetscCall(MatSetSizes(*A,m,n,PETSC_DETERMINE,PETSC_DETERMINE));
  PetscCall(MatSetType(*A,MATBLOCKMAT));
  PetscCall(MatBlockMatSetPreallocation(*A,bs,nz,nnz));
  PetscFunctionReturn(0);
}
