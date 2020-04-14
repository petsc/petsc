
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
  PetscErrorCode    ierr;
  PetscInt          n = A->cmap->n,i,mbs = n/A->rmap->bs,j,bs = A->rmap->bs;
  const PetscInt    *idx;
  IS                row,col;
  MatFactorInfo     info;
  Vec               left = a->left,right = a->right, middle = a->middle;
  Mat               *diag;

  PetscFunctionBegin;
  its = its*lits;
  if (its <= 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Relaxation requires global its %D and local its %D both positive",its,lits);
  if (flag & SOR_EISENSTAT) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support yet for Eisenstat");
  if (omega != 1.0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support yet for omega not equal to 1.0");
  if (fshift) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support yet for fshift");
  if ((flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP) && !(flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP)) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot do backward sweep without forward sweep");
  }

  if (!a->diags) {
    ierr = PetscMalloc1(mbs,&a->diags);CHKERRQ(ierr);
    ierr = MatFactorInfoInitialize(&info);CHKERRQ(ierr);
    for (i=0; i<mbs; i++) {
      ierr = MatGetOrdering(a->a[a->diag[i]], MATORDERINGND,&row,&col);CHKERRQ(ierr);
      ierr = MatCholeskyFactorSymbolic(a->diags[i],a->a[a->diag[i]],row,&info);CHKERRQ(ierr);
      ierr = MatCholeskyFactorNumeric(a->diags[i],a->a[a->diag[i]],&info);CHKERRQ(ierr);
      ierr = ISDestroy(&row);CHKERRQ(ierr);
      ierr = ISDestroy(&col);CHKERRQ(ierr);
    }
    ierr = VecDuplicate(bb,&a->workb);CHKERRQ(ierr);
  }
  diag = a->diags;

  ierr = VecSet(xx,0.0);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  /* copy right hand side because it must be modified during iteration */
  ierr = VecCopy(bb,a->workb);CHKERRQ(ierr);
  ierr = VecGetArrayRead(a->workb,&b);CHKERRQ(ierr);

  /* need to add code for when initial guess is zero, see MatSOR_SeqAIJ */
  while (its--) {
    if (flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP) {

      for (i=0; i<mbs; i++) {
        n   = a->i[i+1] - a->i[i] - 1;
        idx = a->j + a->i[i] + 1;
        v   = a->a + a->i[i] + 1;

        ierr = VecSet(left,0.0);CHKERRQ(ierr);
        for (j=0; j<n; j++) {
          ierr = VecPlaceArray(right,x + idx[j]*bs);CHKERRQ(ierr);
          ierr = MatMultAdd(v[j],right,left,left);CHKERRQ(ierr);
          ierr = VecResetArray(right);CHKERRQ(ierr);
        }
        ierr = VecPlaceArray(right,b + i*bs);CHKERRQ(ierr);
        ierr = VecAYPX(left,-1.0,right);CHKERRQ(ierr);
        ierr = VecResetArray(right);CHKERRQ(ierr);

        ierr = VecPlaceArray(right,x + i*bs);CHKERRQ(ierr);
        ierr = MatSolve(diag[i],left,right);CHKERRQ(ierr);

        /* now adjust right hand side, see MatSOR_SeqSBAIJ */
        for (j=0; j<n; j++) {
          ierr = MatMultTranspose(v[j],right,left);CHKERRQ(ierr);
          ierr = VecPlaceArray(middle,b + idx[j]*bs);CHKERRQ(ierr);
          ierr = VecAXPY(middle,-1.0,left);CHKERRQ(ierr);
          ierr = VecResetArray(middle);CHKERRQ(ierr);
        }
        ierr = VecResetArray(right);CHKERRQ(ierr);

      }
    }
    if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP) {

      for (i=mbs-1; i>=0; i--) {
        n   = a->i[i+1] - a->i[i] - 1;
        idx = a->j + a->i[i] + 1;
        v   = a->a + a->i[i] + 1;

        ierr = VecSet(left,0.0);CHKERRQ(ierr);
        for (j=0; j<n; j++) {
          ierr = VecPlaceArray(right,x + idx[j]*bs);CHKERRQ(ierr);
          ierr = MatMultAdd(v[j],right,left,left);CHKERRQ(ierr);
          ierr = VecResetArray(right);CHKERRQ(ierr);
        }
        ierr = VecPlaceArray(right,b + i*bs);CHKERRQ(ierr);
        ierr = VecAYPX(left,-1.0,right);CHKERRQ(ierr);
        ierr = VecResetArray(right);CHKERRQ(ierr);

        ierr = VecPlaceArray(right,x + i*bs);CHKERRQ(ierr);
        ierr = MatSolve(diag[i],left,right);CHKERRQ(ierr);
        ierr = VecResetArray(right);CHKERRQ(ierr);

      }
    }
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(a->workb,&b);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSOR_BlockMat(Mat A,Vec bb,PetscReal omega,MatSORType flag,PetscReal fshift,PetscInt its,PetscInt lits,Vec xx)
{
  Mat_BlockMat      *a = (Mat_BlockMat*)A->data;
  PetscScalar       *x;
  const Mat         *v;
  const PetscScalar *b;
  PetscErrorCode    ierr;
  PetscInt          n = A->cmap->n,i,mbs = n/A->rmap->bs,j,bs = A->rmap->bs;
  const PetscInt    *idx;
  IS                row,col;
  MatFactorInfo     info;
  Vec               left = a->left,right = a->right;
  Mat               *diag;

  PetscFunctionBegin;
  its = its*lits;
  if (its <= 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Relaxation requires global its %D and local its %D both positive",its,lits);
  if (flag & SOR_EISENSTAT) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support yet for Eisenstat");
  if (omega != 1.0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support yet for omega not equal to 1.0");
  if (fshift) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support yet for fshift");

  if (!a->diags) {
    ierr = PetscMalloc1(mbs,&a->diags);CHKERRQ(ierr);
    ierr = MatFactorInfoInitialize(&info);CHKERRQ(ierr);
    for (i=0; i<mbs; i++) {
      ierr = MatGetOrdering(a->a[a->diag[i]], MATORDERINGND,&row,&col);CHKERRQ(ierr);
      ierr = MatLUFactorSymbolic(a->diags[i],a->a[a->diag[i]],row,col,&info);CHKERRQ(ierr);
      ierr = MatLUFactorNumeric(a->diags[i],a->a[a->diag[i]],&info);CHKERRQ(ierr);
      ierr = ISDestroy(&row);CHKERRQ(ierr);
      ierr = ISDestroy(&col);CHKERRQ(ierr);
    }
  }
  diag = a->diags;

  ierr = VecSet(xx,0.0);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);

  /* need to add code for when initial guess is zero, see MatSOR_SeqAIJ */
  while (its--) {
    if (flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP) {

      for (i=0; i<mbs; i++) {
        n   = a->i[i+1] - a->i[i];
        idx = a->j + a->i[i];
        v   = a->a + a->i[i];

        ierr = VecSet(left,0.0);CHKERRQ(ierr);
        for (j=0; j<n; j++) {
          if (idx[j] != i) {
            ierr = VecPlaceArray(right,x + idx[j]*bs);CHKERRQ(ierr);
            ierr = MatMultAdd(v[j],right,left,left);CHKERRQ(ierr);
            ierr = VecResetArray(right);CHKERRQ(ierr);
          }
        }
        ierr = VecPlaceArray(right,b + i*bs);CHKERRQ(ierr);
        ierr = VecAYPX(left,-1.0,right);CHKERRQ(ierr);
        ierr = VecResetArray(right);CHKERRQ(ierr);

        ierr = VecPlaceArray(right,x + i*bs);CHKERRQ(ierr);
        ierr = MatSolve(diag[i],left,right);CHKERRQ(ierr);
        ierr = VecResetArray(right);CHKERRQ(ierr);
      }
    }
    if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP) {

      for (i=mbs-1; i>=0; i--) {
        n   = a->i[i+1] - a->i[i];
        idx = a->j + a->i[i];
        v   = a->a + a->i[i];

        ierr = VecSet(left,0.0);CHKERRQ(ierr);
        for (j=0; j<n; j++) {
          if (idx[j] != i) {
            ierr = VecPlaceArray(right,x + idx[j]*bs);CHKERRQ(ierr);
            ierr = MatMultAdd(v[j],right,left,left);CHKERRQ(ierr);
            ierr = VecResetArray(right);CHKERRQ(ierr);
          }
        }
        ierr = VecPlaceArray(right,b + i*bs);CHKERRQ(ierr);
        ierr = VecAYPX(left,-1.0,right);CHKERRQ(ierr);
        ierr = VecResetArray(right);CHKERRQ(ierr);

        ierr = VecPlaceArray(right,x + i*bs);CHKERRQ(ierr);
        ierr = MatSolve(diag[i],left,right);CHKERRQ(ierr);
        ierr = VecResetArray(right);CHKERRQ(ierr);

      }
    }
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetValues_BlockMat(Mat A,PetscInt m,const PetscInt im[],PetscInt n,const PetscInt in[],const PetscScalar v[],InsertMode is)
{
  Mat_BlockMat   *a = (Mat_BlockMat*)A->data;
  PetscInt       *rp,k,low,high,t,ii,row,nrow,i,col,l,rmax,N,lastcol = -1;
  PetscInt       *imax=a->imax,*ai=a->i,*ailen=a->ilen;
  PetscInt       *aj  =a->j,nonew=a->nonew,bs=A->rmap->bs,brow,bcol;
  PetscErrorCode ierr;
  PetscInt       ridx,cidx;
  PetscBool      roworiented=a->roworiented;
  MatScalar      value;
  Mat            *ap,*aa = a->a;

  PetscFunctionBegin;
  for (k=0; k<m; k++) { /* loop over added rows */
    row  = im[k];
    brow = row/bs;
    if (row < 0) continue;
    if (PetscDefined(USE_DEBUG) && row >= A->rmap->N) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %D max %D",row,A->rmap->N-1);
    rp   = aj + ai[brow];
    ap   = aa + ai[brow];
    rmax = imax[brow];
    nrow = ailen[brow];
    low  = 0;
    high = nrow;
    for (l=0; l<n; l++) { /* loop over added columns */
      if (in[l] < 0) continue;
      if (PetscDefined(USE_DEBUG) && in[l] >= A->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Column too large: col %D max %D",in[l],A->cmap->n-1);
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
      if (nonew == -1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new nonzero (%D, %D) in the matrix", row, col);
      MatSeqXAIJReallocateAIJ(A,a->mbs,1,nrow,brow,bcol,rmax,aa,ai,aj,rp,ap,imax,nonew,Mat);
      N = nrow++ - 1; high++;
      /* shift up all the later entries in this row */
      for (ii=N; ii>=i; ii--) {
        rp[ii+1] = rp[ii];
        ap[ii+1] = ap[ii];
      }
      if (N>=i) ap[i] = 0;
      rp[i] = bcol;
      a->nz++;
      A->nonzerostate++;
noinsert1:;
      if (!*(ap+i)) {
        ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,bs,bs,0,0,ap+i);CHKERRQ(ierr);
      }
      ierr = MatSetValues(ap[i],1,&ridx,1,&cidx,&value,is);CHKERRQ(ierr);
      low  = i;
    }
    ailen[brow] = nrow;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatLoad_BlockMat(Mat newmat, PetscViewer viewer)
{
  PetscErrorCode    ierr;
  Mat               tmpA;
  PetscInt          i,j,m,n,bs = 1,ncols,*lens,currentcol,mbs,**ii,*ilens,nextcol,*llens,cnt = 0;
  const PetscInt    *cols;
  const PetscScalar *values;
  PetscBool         flg = PETSC_FALSE,notdone;
  Mat_SeqAIJ        *a;
  Mat_BlockMat      *amat;

  PetscFunctionBegin;
  /* force binary viewer to load .info file if it has not yet done so */
  ierr = PetscViewerSetUp(viewer);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_SELF,&tmpA);CHKERRQ(ierr);
  ierr = MatSetType(tmpA,MATSEQAIJ);CHKERRQ(ierr);
  ierr = MatLoad_SeqAIJ(tmpA,viewer);CHKERRQ(ierr);

  ierr = MatGetLocalSize(tmpA,&m,&n);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(PETSC_COMM_SELF,NULL,"Options for loading BlockMat matrix 1","Mat");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-matload_block_size","Set the blocksize used to store the matrix","MatLoad",bs,&bs,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-matload_symmetric","Store the matrix as symmetric","MatLoad",flg,&flg,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* Determine number of nonzero blocks for each block row */
  a    = (Mat_SeqAIJ*) tmpA->data;
  mbs  = m/bs;
  ierr = PetscMalloc3(mbs,&lens,bs,&ii,bs,&ilens);CHKERRQ(ierr);
  ierr = PetscArrayzero(lens,mbs);CHKERRQ(ierr);

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
    ierr = MatSetSizes(newmat,m,n,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  }
  ierr = MatBlockMatSetPreallocation(newmat,bs,0,lens);CHKERRQ(ierr);
  if (flg) {
    ierr = MatSetOption(newmat,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
  }
  amat = (Mat_BlockMat*)(newmat)->data;

  /* preallocate the submatrices */
  ierr = PetscMalloc1(bs,&llens);CHKERRQ(ierr);
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
      ierr    = PetscArrayzero(llens,bs);CHKERRQ(ierr);
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
      if (cnt >= amat->maxnz) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Number of blocks found greater than expected %D",cnt);
      if (!flg || currentcol >= i) {
        amat->j[cnt] = currentcol;
        ierr         = MatCreateSeqAIJ(PETSC_COMM_SELF,bs,bs,0,llens,amat->a+cnt++);CHKERRQ(ierr);
      }

      if (!notdone) break;
      currentcol = nextcol;
    }
    amat->ilen[i] = lens[i];
  }

  ierr = PetscFree3(lens,ii,ilens);CHKERRQ(ierr);
  ierr = PetscFree(llens);CHKERRQ(ierr);

  /* copy over the matrix, one row at a time */
  for (i=0; i<m; i++) {
    ierr = MatGetRow(tmpA,i,&ncols,&cols,&values);CHKERRQ(ierr);
    ierr = MatSetValues(newmat,1,&i,ncols,cols,values,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatRestoreRow(tmpA,i,&ncols,&cols,&values);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(newmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(newmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatView_BlockMat(Mat A,PetscViewer viewer)
{
  Mat_BlockMat      *a = (Mat_BlockMat*)A->data;
  PetscErrorCode    ierr;
  const char        *name;
  PetscViewerFormat format;

  PetscFunctionBegin;
  ierr = PetscObjectGetName((PetscObject)A,&name);CHKERRQ(ierr);
  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_FACTOR_INFO || format == PETSC_VIEWER_ASCII_INFO) {
    ierr = PetscViewerASCIIPrintf(viewer,"Nonzero block matrices = %D \n",a->nz);CHKERRQ(ierr);
    if (A->symmetric) {
      ierr = PetscViewerASCIIPrintf(viewer,"Only upper triangular part of symmetric matrix is stored\n");CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_BlockMat(Mat mat)
{
  PetscErrorCode ierr;
  Mat_BlockMat   *bmat = (Mat_BlockMat*)mat->data;
  PetscInt       i;

  PetscFunctionBegin;
  ierr = VecDestroy(&bmat->right);CHKERRQ(ierr);
  ierr = VecDestroy(&bmat->left);CHKERRQ(ierr);
  ierr = VecDestroy(&bmat->middle);CHKERRQ(ierr);
  ierr = VecDestroy(&bmat->workb);CHKERRQ(ierr);
  if (bmat->diags) {
    for (i=0; i<mat->rmap->n/mat->rmap->bs; i++) {
      ierr = MatDestroy(&bmat->diags[i]);CHKERRQ(ierr);
    }
  }
  if (bmat->a) {
    for (i=0; i<bmat->nz; i++) {
      ierr = MatDestroy(&bmat->a[i]);CHKERRQ(ierr);
    }
  }
  ierr = MatSeqXAIJFreeAIJ(mat,(PetscScalar**)&bmat->a,&bmat->j,&bmat->i);CHKERRQ(ierr);
  ierr = PetscFree(mat->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMult_BlockMat(Mat A,Vec x,Vec y)
{
  Mat_BlockMat   *bmat = (Mat_BlockMat*)A->data;
  PetscErrorCode ierr;
  PetscScalar    *xx,*yy;
  PetscInt       *aj,i,*ii,jrow,m = A->rmap->n/A->rmap->bs,bs = A->rmap->bs,n,j;
  Mat            *aa;

  PetscFunctionBegin;
  /*
     Standard CSR multiply except each entry is a Mat
  */
  ierr = VecGetArray(x,&xx);CHKERRQ(ierr);

  ierr = VecSet(y,0.0);CHKERRQ(ierr);
  ierr = VecGetArray(y,&yy);CHKERRQ(ierr);
  aj   = bmat->j;
  aa   = bmat->a;
  ii   = bmat->i;
  for (i=0; i<m; i++) {
    jrow = ii[i];
    ierr = VecPlaceArray(bmat->left,yy + bs*i);CHKERRQ(ierr);
    n    = ii[i+1] - jrow;
    for (j=0; j<n; j++) {
      ierr = VecPlaceArray(bmat->right,xx + bs*aj[jrow]);CHKERRQ(ierr);
      ierr = MatMultAdd(aa[jrow],bmat->right,bmat->left,bmat->left);CHKERRQ(ierr);
      ierr = VecResetArray(bmat->right);CHKERRQ(ierr);
      jrow++;
    }
    ierr = VecResetArray(bmat->left);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&yy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMult_BlockMat_Symmetric(Mat A,Vec x,Vec y)
{
  Mat_BlockMat   *bmat = (Mat_BlockMat*)A->data;
  PetscErrorCode ierr;
  PetscScalar    *xx,*yy;
  PetscInt       *aj,i,*ii,jrow,m = A->rmap->n/A->rmap->bs,bs = A->rmap->bs,n,j;
  Mat            *aa;

  PetscFunctionBegin;
  /*
     Standard CSR multiply except each entry is a Mat
  */
  ierr = VecGetArray(x,&xx);CHKERRQ(ierr);

  ierr = VecSet(y,0.0);CHKERRQ(ierr);
  ierr = VecGetArray(y,&yy);CHKERRQ(ierr);
  aj   = bmat->j;
  aa   = bmat->a;
  ii   = bmat->i;
  for (i=0; i<m; i++) {
    jrow = ii[i];
    n    = ii[i+1] - jrow;
    ierr = VecPlaceArray(bmat->left,yy + bs*i);CHKERRQ(ierr);
    ierr = VecPlaceArray(bmat->middle,xx + bs*i);CHKERRQ(ierr);
    /* if we ALWAYS required a diagonal entry then could remove this if test */
    if (aj[jrow] == i) {
      ierr = VecPlaceArray(bmat->right,xx + bs*aj[jrow]);CHKERRQ(ierr);
      ierr = MatMultAdd(aa[jrow],bmat->right,bmat->left,bmat->left);CHKERRQ(ierr);
      ierr = VecResetArray(bmat->right);CHKERRQ(ierr);
      jrow++;
      n--;
    }
    for (j=0; j<n; j++) {
      ierr = VecPlaceArray(bmat->right,xx + bs*aj[jrow]);CHKERRQ(ierr);            /* upper triangular part */
      ierr = MatMultAdd(aa[jrow],bmat->right,bmat->left,bmat->left);CHKERRQ(ierr);
      ierr = VecResetArray(bmat->right);CHKERRQ(ierr);

      ierr = VecPlaceArray(bmat->right,yy + bs*aj[jrow]);CHKERRQ(ierr);            /* lower triangular part */
      ierr = MatMultTransposeAdd(aa[jrow],bmat->middle,bmat->right,bmat->right);CHKERRQ(ierr);
      ierr = VecResetArray(bmat->right);CHKERRQ(ierr);
      jrow++;
    }
    ierr = VecResetArray(bmat->left);CHKERRQ(ierr);
    ierr = VecResetArray(bmat->middle);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&yy);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  PetscInt       i,j,mbs = A->rmap->n/A->rmap->bs;

  PetscFunctionBegin;
  if (!a->diag) {
    ierr = PetscMalloc1(mbs,&a->diag);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  PetscInt       i,k,first,step,lensi,nrows,ncols;
  PetscInt       *j_new,*i_new,*aj = a->j,*ailen = a->ilen;
  PetscScalar    *a_new;
  Mat            C,*aa = a->a;
  PetscBool      stride,equal;

  PetscFunctionBegin;
  ierr = ISEqual(isrow,iscol,&equal);CHKERRQ(ierr);
  if (!equal) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only for idential column and row indices");
  ierr = PetscObjectTypeCompare((PetscObject)iscol,ISSTRIDE,&stride);CHKERRQ(ierr);
  if (!stride) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only for stride indices");
  ierr = ISStrideGetInfo(iscol,&first,&step);CHKERRQ(ierr);
  if (step != A->rmap->bs) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Can only select one entry from each block");

  ierr  = ISGetLocalSize(isrow,&nrows);CHKERRQ(ierr);
  ncols = nrows;

  /* create submatrix */
  if (scall == MAT_REUSE_MATRIX) {
    PetscInt n_cols,n_rows;
    C    = *B;
    ierr = MatGetSize(C,&n_rows,&n_cols);CHKERRQ(ierr);
    if (n_rows != nrows || n_cols != ncols) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Reused submatrix wrong size");
    ierr = MatZeroEntries(C);CHKERRQ(ierr);
  } else {
    ierr = MatCreate(PetscObjectComm((PetscObject)A),&C);CHKERRQ(ierr);
    ierr = MatSetSizes(C,nrows,ncols,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
    if (A->symmetric) {
      ierr = MatSetType(C,MATSEQSBAIJ);CHKERRQ(ierr);
    } else {
      ierr = MatSetType(C,MATSEQAIJ);CHKERRQ(ierr);
    }
    ierr = MatSeqAIJSetPreallocation(C,0,ailen);CHKERRQ(ierr);
    ierr = MatSeqSBAIJSetPreallocation(C,1,0,ailen);CHKERRQ(ierr);
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
      ierr     = MatGetValue(*aa++,first,first,a_new++);CHKERRQ(ierr);
    }
    i_new[i+1] = i_new[i] + lensi;
    c->ilen[i] = lensi;
  }

  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  *B   = C;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatAssemblyEnd_BlockMat(Mat A,MatAssemblyType mode)
{
  Mat_BlockMat   *a = (Mat_BlockMat*)A->data;
  PetscErrorCode ierr;
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
    if (PetscDefined(USE_DEBUG) && !aa[i]) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Null matrix at location %D column %D nz %D",i,aj[i],a->nz);
    ierr = MatAssemblyBegin(aa[i],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(aa[i],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  ierr = PetscInfo4(A,"Matrix size: %D X %D; storage space: %D unneeded,%D used\n",m,A->cmap->n/A->cmap->bs,fshift,a->nz);CHKERRQ(ierr);
  ierr = PetscInfo1(A,"Number of mallocs during MatSetValues() is %D\n",a->reallocs);CHKERRQ(ierr);
  ierr = PetscInfo1(A,"Maximum nonzeros in any row is %D\n",rmax);CHKERRQ(ierr);

  A->info.mallocs    += a->reallocs;
  a->reallocs         = 0;
  A->info.nz_unneeded = (double)fshift;
  a->rmax             = rmax;
  ierr                = MatMarkDiagonal_BlockMat(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetOption_BlockMat(Mat A,MatOption opt,PetscBool flg)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (opt == MAT_SYMMETRIC && flg) {
    A->ops->sor  = MatSOR_BlockMat_Symmetric;
    A->ops->mult = MatMult_BlockMat_Symmetric;
  } else {
    ierr = PetscInfo1(A,"Unused matrix option %s\n",MatOptions[opt]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


static struct _MatOps MatOps_Values = {MatSetValues_BlockMat,
                                       0,
                                       0,
                                       MatMult_BlockMat,
                               /*  4*/ MatMultAdd_BlockMat,
                                       MatMultTranspose_BlockMat,
                                       MatMultTransposeAdd_BlockMat,
                                       0,
                                       0,
                                       0,
                               /* 10*/ 0,
                                       0,
                                       0,
                                       MatSOR_BlockMat,
                                       0,
                               /* 15*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /* 20*/ 0,
                                       MatAssemblyEnd_BlockMat,
                                       MatSetOption_BlockMat,
                                       0,
                               /* 24*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /* 29*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /* 34*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /* 39*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /* 44*/ 0,
                                       0,
                                       MatShift_Basic,
                                       0,
                                       0,
                               /* 49*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /* 54*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /* 59*/ MatCreateSubMatrix_BlockMat,
                                       MatDestroy_BlockMat,
                                       MatView_BlockMat,
                                       0,
                                       0,
                               /* 64*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /* 69*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /* 74*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /* 79*/ 0,
                                       0,
                                       0,
                                       0,
                                       MatLoad_BlockMat,
                               /* 84*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /* 89*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /* 94*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /* 99*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /*104*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /*109*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /*114*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /*119*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /*124*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /*129*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /*134*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /*139*/ 0,
                                       0,
                                       0
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

.seealso: MatCreate(), MatCreateBlockMat(), MatSetValues()

@*/
PetscErrorCode  MatBlockMatSetPreallocation(Mat B,PetscInt bs,PetscInt nz,const PetscInt nnz[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTryMethod(B,"MatBlockMatSetPreallocation_C",(Mat,PetscInt,PetscInt,const PetscInt[]),(B,bs,nz,nnz));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode  MatBlockMatSetPreallocation_BlockMat(Mat A,PetscInt bs,PetscInt nz,PetscInt *nnz)
{
  Mat_BlockMat   *bmat = (Mat_BlockMat*)A->data;
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  ierr = PetscLayoutSetBlockSize(A->rmap,bs);CHKERRQ(ierr);
  ierr = PetscLayoutSetBlockSize(A->cmap,bs);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(A->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(A->cmap);CHKERRQ(ierr);
  ierr = PetscLayoutGetBlockSize(A->rmap,&bs);CHKERRQ(ierr);

  if (nz == PETSC_DEFAULT || nz == PETSC_DECIDE) nz = 5;
  if (nz < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"nz cannot be less than 0: value %d",nz);
  if (nnz) {
    for (i=0; i<A->rmap->n/bs; i++) {
      if (nnz[i] < 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"nnz cannot be less than 0: local row %d value %d",i,nnz[i]);
      if (nnz[i] > A->cmap->n/bs) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"nnz cannot be greater than row length: local row %d value %d rowlength %d",i,nnz[i],A->cmap->n/bs);
    }
  }
  bmat->mbs = A->rmap->n/bs;

  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,bs,NULL,&bmat->right);CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,bs,NULL,&bmat->middle);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,bs,&bmat->left);CHKERRQ(ierr);

  if (!bmat->imax) {
    ierr = PetscMalloc2(A->rmap->n,&bmat->imax,A->rmap->n,&bmat->ilen);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)A,2*A->rmap->n*sizeof(PetscInt));CHKERRQ(ierr);
  }
  if (nnz) {
    nz = 0;
    for (i=0; i<A->rmap->n/A->rmap->bs; i++) {
      bmat->imax[i] = nnz[i];
      nz           += nnz[i];
    }
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Currently requires block row by row preallocation");

  /* bmat->ilen will count nonzeros in each row so far. */
  for (i=0; i<bmat->mbs; i++) bmat->ilen[i] = 0;

  /* allocate the matrix space */
  ierr       = MatSeqXAIJFreeAIJ(A,(PetscScalar**)&bmat->a,&bmat->j,&bmat->i);CHKERRQ(ierr);
  ierr       = PetscMalloc3(nz,&bmat->a,nz,&bmat->j,A->rmap->n+1,&bmat->i);CHKERRQ(ierr);
  ierr       = PetscLogObjectMemory((PetscObject)A,(A->rmap->n+1)*sizeof(PetscInt)+nz*(sizeof(PetscScalar)+sizeof(PetscInt)));CHKERRQ(ierr);
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
  ierr                = MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
   MATBLOCKMAT - A matrix that is defined by a set of Mat's that represents a sparse block matrix
                 consisting of (usually) sparse blocks.

  Level: advanced

.seealso: MatCreateBlockMat()

M*/

PETSC_EXTERN PetscErrorCode MatCreate_BlockMat(Mat A)
{
  Mat_BlockMat   *b;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr    = PetscNewLog(A,&b);CHKERRQ(ierr);
  A->data = (void*)b;
  ierr    = PetscMemcpy(A->ops,&MatOps_Values,sizeof(struct _MatOps));CHKERRQ(ierr);

  A->assembled    = PETSC_TRUE;
  A->preallocated = PETSC_FALSE;
  ierr            = PetscObjectChangeTypeName((PetscObject)A,MATBLOCKMAT);CHKERRQ(ierr);

  ierr = PetscObjectComposeFunction((PetscObject)A,"MatBlockMatSetPreallocation_C",MatBlockMatSetPreallocation_BlockMat);CHKERRQ(ierr);
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

.seealso: MATBLOCKMAT, MatCreateNest()
@*/
PetscErrorCode  MatCreateBlockMat(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt bs,PetscInt nz,PetscInt *nnz, Mat *A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetType(*A,MATBLOCKMAT);CHKERRQ(ierr);
  ierr = MatBlockMatSetPreallocation(*A,bs,nz,nnz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
