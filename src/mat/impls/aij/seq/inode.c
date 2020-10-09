
/*
  This file provides high performance routines for the Inode format (compressed sparse row)
  by taking advantage of rows with identical nonzero structure (I-nodes).
*/
#include <../src/mat/impls/aij/seq/aij.h>

static PetscErrorCode MatCreateColInode_Private(Mat A,PetscInt *size,PetscInt **ns)
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       i,count,m,n,min_mn,*ns_row,*ns_col;

  PetscFunctionBegin;
  n      = A->cmap->n;
  m      = A->rmap->n;
  ns_row = a->inode.size;

  min_mn = (m < n) ? m : n;
  if (!ns) {
    for (count=0,i=0; count<min_mn; count+=ns_row[i],i++) ;
    for (; count+1 < n; count++,i++) ;
    if (count < n)  {
      i++;
    }
    *size = i;
    PetscFunctionReturn(0);
  }
  ierr = PetscMalloc1(n+1,&ns_col);CHKERRQ(ierr);

  /* Use the same row structure wherever feasible. */
  for (count=0,i=0; count<min_mn; count+=ns_row[i],i++) {
    ns_col[i] = ns_row[i];
  }

  /* if m < n; pad up the remainder with inode_limit */
  for (; count+1 < n; count++,i++) {
    ns_col[i] = 1;
  }
  /* The last node is the odd ball. padd it up with the remaining rows; */
  if (count < n) {
    ns_col[i] = n - count;
    i++;
  } else if (count > n) {
    /* Adjust for the over estimation */
    ns_col[i-1] += n - count;
  }
  *size = i;
  *ns   = ns_col;
  PetscFunctionReturn(0);
}


/*
      This builds symmetric version of nonzero structure,
*/
static PetscErrorCode MatGetRowIJ_SeqAIJ_Inode_Symmetric(Mat A,const PetscInt *iia[],const PetscInt *jja[],PetscInt ishift,PetscInt oshift)
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       *work,*ia,*ja,nz,nslim_row,nslim_col,m,row,col,n;
  PetscInt       *tns,*tvc,*ns_row = a->inode.size,*ns_col,nsz,i1,i2;
  const PetscInt *j,*jmax,*ai= a->i,*aj = a->j;

  PetscFunctionBegin;
  nslim_row = a->inode.node_count;
  m         = A->rmap->n;
  n         = A->cmap->n;
  if (m != n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"MatGetRowIJ_SeqAIJ_Inode_Symmetric: Matrix should be square");

  /* Use the row_inode as column_inode */
  nslim_col = nslim_row;
  ns_col    = ns_row;

  /* allocate space for reformated inode structure */
  ierr = PetscMalloc2(nslim_col+1,&tns,n+1,&tvc);CHKERRQ(ierr);
  for (i1=0,tns[0]=0; i1<nslim_col; ++i1) tns[i1+1] = tns[i1]+ ns_row[i1];

  for (i1=0,col=0; i1<nslim_col; ++i1) {
    nsz = ns_col[i1];
    for (i2=0; i2<nsz; ++i2,++col) tvc[col] = i1;
  }
  /* allocate space for row pointers */
  ierr = PetscCalloc1(nslim_row+1,&ia);CHKERRQ(ierr);
  *iia = ia;
  ierr = PetscMalloc1(nslim_row+1,&work);CHKERRQ(ierr);

  /* determine the number of columns in each row */
  ia[0] = oshift;
  for (i1=0,row=0; i1<nslim_row; row+=ns_row[i1],i1++) {

    j    = aj + ai[row] + ishift;
    jmax = aj + ai[row+1] + ishift;
    if (j==jmax) continue; /* empty row */
    col  = *j++ + ishift;
    i2   = tvc[col];
    while (i2<i1 && j<jmax) { /* 1.[-xx-d-xx--] 2.[-xx-------],off-diagonal elemets */
      ia[i1+1]++;
      ia[i2+1]++;
      i2++;                     /* Start col of next node */
      while ((j<jmax) && ((col=*j+ishift)<tns[i2])) ++j;
      i2 = tvc[col];
    }
    if (i2 == i1) ia[i2+1]++;    /* now the diagonal element */
  }

  /* shift ia[i] to point to next row */
  for (i1=1; i1<nslim_row+1; i1++) {
    row        = ia[i1-1];
    ia[i1]    += row;
    work[i1-1] = row - oshift;
  }

  /* allocate space for column pointers */
  nz   = ia[nslim_row] + (!ishift);
  ierr = PetscMalloc1(nz,&ja);CHKERRQ(ierr);
  *jja = ja;

  /* loop over lower triangular part putting into ja */
  for (i1=0,row=0; i1<nslim_row; row += ns_row[i1],i1++) {
    j    = aj + ai[row] + ishift;
    jmax = aj + ai[row+1] + ishift;
    if (j==jmax) continue; /* empty row */
    col  = *j++ + ishift;
    i2   = tvc[col];
    while (i2<i1 && j<jmax) {
      ja[work[i2]++] = i1 + oshift;
      ja[work[i1]++] = i2 + oshift;
      ++i2;
      while ((j<jmax) && ((col=*j+ishift)< tns[i2])) ++j; /* Skip rest col indices in this node */
      i2 = tvc[col];
    }
    if (i2 == i1) ja[work[i1]++] = i2 + oshift;

  }
  ierr = PetscFree(work);CHKERRQ(ierr);
  ierr = PetscFree2(tns,tvc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
      This builds nonsymmetric version of nonzero structure,
*/
static PetscErrorCode MatGetRowIJ_SeqAIJ_Inode_Nonsymmetric(Mat A,const PetscInt *iia[],const PetscInt *jja[],PetscInt ishift,PetscInt oshift)
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       *work,*ia,*ja,nz,nslim_row,n,row,col,*ns_col,nslim_col;
  PetscInt       *tns,*tvc,nsz,i1,i2;
  const PetscInt *j,*ai= a->i,*aj = a->j,*ns_row = a->inode.size;

  PetscFunctionBegin;
  nslim_row = a->inode.node_count;
  n         = A->cmap->n;

  /* Create The column_inode for this matrix */
  ierr = MatCreateColInode_Private(A,&nslim_col,&ns_col);CHKERRQ(ierr);

  /* allocate space for reformated column_inode structure */
  ierr = PetscMalloc2(nslim_col +1,&tns,n + 1,&tvc);CHKERRQ(ierr);
  for (i1=0,tns[0]=0; i1<nslim_col; ++i1) tns[i1+1] = tns[i1] + ns_col[i1];

  for (i1=0,col=0; i1<nslim_col; ++i1) {
    nsz = ns_col[i1];
    for (i2=0; i2<nsz; ++i2,++col) tvc[col] = i1;
  }
  /* allocate space for row pointers */
  ierr = PetscCalloc1(nslim_row+1,&ia);CHKERRQ(ierr);
  *iia = ia;
  ierr = PetscMalloc1(nslim_row+1,&work);CHKERRQ(ierr);

  /* determine the number of columns in each row */
  ia[0] = oshift;
  for (i1=0,row=0; i1<nslim_row; row+=ns_row[i1],i1++) {
    j   = aj + ai[row] + ishift;
    nz  = ai[row+1] - ai[row];
    if (!nz) continue; /* empty row */
    col = *j++ + ishift;
    i2  = tvc[col];
    while (nz-- > 0) {           /* off-diagonal elemets */
      ia[i1+1]++;
      i2++;                     /* Start col of next node */
      while (nz > 0 && ((col = *j++ + ishift) < tns[i2])) nz--;
      if (nz > 0) i2 = tvc[col];
    }
  }

  /* shift ia[i] to point to next row */
  for (i1=1; i1<nslim_row+1; i1++) {
    row        = ia[i1-1];
    ia[i1]    += row;
    work[i1-1] = row - oshift;
  }

  /* allocate space for column pointers */
  nz   = ia[nslim_row] + (!ishift);
  ierr = PetscMalloc1(nz,&ja);CHKERRQ(ierr);
  *jja = ja;

  /* loop over matrix putting into ja */
  for (i1=0,row=0; i1<nslim_row; row+=ns_row[i1],i1++) {
    j   = aj + ai[row] + ishift;
    nz  = ai[row+1] - ai[row];
    if (!nz) continue; /* empty row */
    col = *j++ + ishift;
    i2  = tvc[col];
    while (nz-- > 0) {
      ja[work[i1]++] = i2 + oshift;
      ++i2;
      while (nz > 0 && ((col = *j++ + ishift) < tns[i2])) nz--;
      if (nz > 0) i2 = tvc[col];
    }
  }
  ierr = PetscFree(ns_col);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
  ierr = PetscFree2(tns,tvc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetRowIJ_SeqAIJ_Inode(Mat A,PetscInt oshift,PetscBool symmetric,PetscBool blockcompressed,PetscInt *n,const PetscInt *ia[],const PetscInt *ja[],PetscBool  *done)
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *n = a->inode.node_count;
  if (!ia) PetscFunctionReturn(0);
  if (!blockcompressed) {
    ierr = MatGetRowIJ_SeqAIJ(A,oshift,symmetric,blockcompressed,n,ia,ja,done);CHKERRQ(ierr);
  } else if (symmetric) {
    ierr = MatGetRowIJ_SeqAIJ_Inode_Symmetric(A,ia,ja,0,oshift);CHKERRQ(ierr);
  } else {
    ierr = MatGetRowIJ_SeqAIJ_Inode_Nonsymmetric(A,ia,ja,0,oshift);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatRestoreRowIJ_SeqAIJ_Inode(Mat A,PetscInt oshift,PetscBool symmetric,PetscBool blockcompressed,PetscInt *n,const PetscInt *ia[],const PetscInt *ja[],PetscBool  *done)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!ia) PetscFunctionReturn(0);

  if (!blockcompressed) {
    ierr = MatRestoreRowIJ_SeqAIJ(A,oshift,symmetric,blockcompressed,n,ia,ja,done);CHKERRQ(ierr);
  } else {
    ierr = PetscFree(*ia);CHKERRQ(ierr);
    ierr = PetscFree(*ja);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------- */

static PetscErrorCode MatGetColumnIJ_SeqAIJ_Inode_Nonsymmetric(Mat A,const PetscInt *iia[],const PetscInt *jja[],PetscInt ishift,PetscInt oshift)
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       *work,*ia,*ja,*j,nz,nslim_row, n,row,col,*ns_col,nslim_col;
  PetscInt       *tns,*tvc,*ns_row = a->inode.size,nsz,i1,i2,*ai= a->i,*aj = a->j;

  PetscFunctionBegin;
  nslim_row = a->inode.node_count;
  n         = A->cmap->n;

  /* Create The column_inode for this matrix */
  ierr = MatCreateColInode_Private(A,&nslim_col,&ns_col);CHKERRQ(ierr);

  /* allocate space for reformated column_inode structure */
  ierr = PetscMalloc2(nslim_col + 1,&tns,n + 1,&tvc);CHKERRQ(ierr);
  for (i1=0,tns[0]=0; i1<nslim_col; ++i1) tns[i1+1] = tns[i1] + ns_col[i1];

  for (i1=0,col=0; i1<nslim_col; ++i1) {
    nsz = ns_col[i1];
    for (i2=0; i2<nsz; ++i2,++col) tvc[col] = i1;
  }
  /* allocate space for column pointers */
  ierr = PetscCalloc1(nslim_col+1,&ia);CHKERRQ(ierr);
  *iia = ia;
  ierr = PetscMalloc1(nslim_col+1,&work);CHKERRQ(ierr);

  /* determine the number of columns in each row */
  ia[0] = oshift;
  for (i1=0,row=0; i1<nslim_row; row+=ns_row[i1],i1++) {
    j   = aj + ai[row] + ishift;
    col = *j++ + ishift;
    i2  = tvc[col];
    nz  = ai[row+1] - ai[row];
    while (nz-- > 0) {           /* off-diagonal elemets */
      /* ia[i1+1]++; */
      ia[i2+1]++;
      i2++;
      while (nz > 0 && ((col = *j++ + ishift) < tns[i2])) nz--;
      if (nz > 0) i2 = tvc[col];
    }
  }

  /* shift ia[i] to point to next col */
  for (i1=1; i1<nslim_col+1; i1++) {
    col        = ia[i1-1];
    ia[i1]    += col;
    work[i1-1] = col - oshift;
  }

  /* allocate space for column pointers */
  nz   = ia[nslim_col] + (!ishift);
  ierr = PetscMalloc1(nz,&ja);CHKERRQ(ierr);
  *jja = ja;

  /* loop over matrix putting into ja */
  for (i1=0,row=0; i1<nslim_row; row+=ns_row[i1],i1++) {
    j   = aj + ai[row] + ishift;
    col = *j++ + ishift;
    i2  = tvc[col];
    nz  = ai[row+1] - ai[row];
    while (nz-- > 0) {
      /* ja[work[i1]++] = i2 + oshift; */
      ja[work[i2]++] = i1 + oshift;
      i2++;
      while (nz > 0 && ((col = *j++ + ishift) < tns[i2])) nz--;
      if (nz > 0) i2 = tvc[col];
    }
  }
  ierr = PetscFree(ns_col);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
  ierr = PetscFree2(tns,tvc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetColumnIJ_SeqAIJ_Inode(Mat A,PetscInt oshift,PetscBool symmetric,PetscBool blockcompressed,PetscInt *n,const PetscInt *ia[],const PetscInt *ja[],PetscBool  *done)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreateColInode_Private(A,n,NULL);CHKERRQ(ierr);
  if (!ia) PetscFunctionReturn(0);

  if (!blockcompressed) {
    ierr = MatGetColumnIJ_SeqAIJ(A,oshift,symmetric,blockcompressed,n,ia,ja,done);CHKERRQ(ierr);
  } else if (symmetric) {
    /* Since the indices are symmetric it does'nt matter */
    ierr = MatGetRowIJ_SeqAIJ_Inode_Symmetric(A,ia,ja,0,oshift);CHKERRQ(ierr);
  } else {
    ierr = MatGetColumnIJ_SeqAIJ_Inode_Nonsymmetric(A,ia,ja,0,oshift);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatRestoreColumnIJ_SeqAIJ_Inode(Mat A,PetscInt oshift,PetscBool symmetric,PetscBool blockcompressed,PetscInt *n,const PetscInt *ia[],const PetscInt *ja[],PetscBool  *done)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!ia) PetscFunctionReturn(0);
  if (!blockcompressed) {
    ierr = MatRestoreColumnIJ_SeqAIJ(A,oshift,symmetric,blockcompressed,n,ia,ja,done);CHKERRQ(ierr);
  } else {
    ierr = PetscFree(*ia);CHKERRQ(ierr);
    ierr = PetscFree(*ja);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------- */

static PetscErrorCode MatMult_SeqAIJ_Inode(Mat A,Vec xx,Vec yy)
{
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)A->data;
  PetscScalar       sum1,sum2,sum3,sum4,sum5,tmp0,tmp1;
  PetscScalar       *y;
  const PetscScalar *x;
  const MatScalar   *v1,*v2,*v3,*v4,*v5;
  PetscErrorCode    ierr;
  PetscInt          i1,i2,n,i,row,node_max,nsz,sz,nonzerorow=0;
  const PetscInt    *idx,*ns,*ii;

#if defined(PETSC_HAVE_PRAGMA_DISJOINT)
#pragma disjoint(*x,*y,*v1,*v2,*v3,*v4,*v5)
#endif

  PetscFunctionBegin;
  if (!a->inode.size) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Missing Inode Structure");
  node_max = a->inode.node_count;
  ns       = a->inode.size;     /* Node Size array */
  ierr     = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
  ierr     = VecGetArray(yy,&y);CHKERRQ(ierr);
  idx      = a->j;
  v1       = a->a;
  ii       = a->i;

  for (i = 0,row = 0; i< node_max; ++i) {
    nsz         = ns[i];
    n           = ii[1] - ii[0];
    nonzerorow += (n>0)*nsz;
    ii         += nsz;
    PetscPrefetchBlock(idx+nsz*n,n,0,PETSC_PREFETCH_HINT_NTA);    /* Prefetch the indices for the block row after the current one */
    PetscPrefetchBlock(v1+nsz*n,nsz*n,0,PETSC_PREFETCH_HINT_NTA); /* Prefetch the values for the block row after the current one  */
    sz = n;                     /* No of non zeros in this row */
                                /* Switch on the size of Node */
    switch (nsz) {               /* Each loop in 'case' is unrolled */
    case 1:
      sum1 = 0.;

      for (n = 0; n< sz-1; n+=2) {
        i1    = idx[0];         /* The instructions are ordered to */
        i2    = idx[1];         /* make the compiler's job easy */
        idx  += 2;
        tmp0  = x[i1];
        tmp1  = x[i2];
        sum1 += v1[0] * tmp0 + v1[1] * tmp1; v1 += 2;
      }

      if (n == sz-1) {          /* Take care of the last nonzero  */
        tmp0  = x[*idx++];
        sum1 += *v1++ *tmp0;
      }
      y[row++]=sum1;
      break;
    case 2:
      sum1 = 0.;
      sum2 = 0.;
      v2   = v1 + n;

      for (n = 0; n< sz-1; n+=2) {
        i1    = idx[0];
        i2    = idx[1];
        idx  += 2;
        tmp0  = x[i1];
        tmp1  = x[i2];
        sum1 += v1[0] * tmp0 + v1[1] * tmp1; v1 += 2;
        sum2 += v2[0] * tmp0 + v2[1] * tmp1; v2 += 2;
      }
      if (n == sz-1) {
        tmp0  = x[*idx++];
        sum1 += *v1++ * tmp0;
        sum2 += *v2++ * tmp0;
      }
      y[row++]=sum1;
      y[row++]=sum2;
      v1      =v2;              /* Since the next block to be processed starts there*/
      idx    +=sz;
      break;
    case 3:
      sum1 = 0.;
      sum2 = 0.;
      sum3 = 0.;
      v2   = v1 + n;
      v3   = v2 + n;

      for (n = 0; n< sz-1; n+=2) {
        i1    = idx[0];
        i2    = idx[1];
        idx  += 2;
        tmp0  = x[i1];
        tmp1  = x[i2];
        sum1 += v1[0] * tmp0 + v1[1] * tmp1; v1 += 2;
        sum2 += v2[0] * tmp0 + v2[1] * tmp1; v2 += 2;
        sum3 += v3[0] * tmp0 + v3[1] * tmp1; v3 += 2;
      }
      if (n == sz-1) {
        tmp0  = x[*idx++];
        sum1 += *v1++ * tmp0;
        sum2 += *v2++ * tmp0;
        sum3 += *v3++ * tmp0;
      }
      y[row++]=sum1;
      y[row++]=sum2;
      y[row++]=sum3;
      v1      =v3;              /* Since the next block to be processed starts there*/
      idx    +=2*sz;
      break;
    case 4:
      sum1 = 0.;
      sum2 = 0.;
      sum3 = 0.;
      sum4 = 0.;
      v2   = v1 + n;
      v3   = v2 + n;
      v4   = v3 + n;

      for (n = 0; n< sz-1; n+=2) {
        i1    = idx[0];
        i2    = idx[1];
        idx  += 2;
        tmp0  = x[i1];
        tmp1  = x[i2];
        sum1 += v1[0] * tmp0 + v1[1] *tmp1; v1 += 2;
        sum2 += v2[0] * tmp0 + v2[1] *tmp1; v2 += 2;
        sum3 += v3[0] * tmp0 + v3[1] *tmp1; v3 += 2;
        sum4 += v4[0] * tmp0 + v4[1] *tmp1; v4 += 2;
      }
      if (n == sz-1) {
        tmp0  = x[*idx++];
        sum1 += *v1++ * tmp0;
        sum2 += *v2++ * tmp0;
        sum3 += *v3++ * tmp0;
        sum4 += *v4++ * tmp0;
      }
      y[row++]=sum1;
      y[row++]=sum2;
      y[row++]=sum3;
      y[row++]=sum4;
      v1      =v4;              /* Since the next block to be processed starts there*/
      idx    +=3*sz;
      break;
    case 5:
      sum1 = 0.;
      sum2 = 0.;
      sum3 = 0.;
      sum4 = 0.;
      sum5 = 0.;
      v2   = v1 + n;
      v3   = v2 + n;
      v4   = v3 + n;
      v5   = v4 + n;

      for (n = 0; n<sz-1; n+=2) {
        i1    = idx[0];
        i2    = idx[1];
        idx  += 2;
        tmp0  = x[i1];
        tmp1  = x[i2];
        sum1 += v1[0] * tmp0 + v1[1] *tmp1; v1 += 2;
        sum2 += v2[0] * tmp0 + v2[1] *tmp1; v2 += 2;
        sum3 += v3[0] * tmp0 + v3[1] *tmp1; v3 += 2;
        sum4 += v4[0] * tmp0 + v4[1] *tmp1; v4 += 2;
        sum5 += v5[0] * tmp0 + v5[1] *tmp1; v5 += 2;
      }
      if (n == sz-1) {
        tmp0  = x[*idx++];
        sum1 += *v1++ * tmp0;
        sum2 += *v2++ * tmp0;
        sum3 += *v3++ * tmp0;
        sum4 += *v4++ * tmp0;
        sum5 += *v5++ * tmp0;
      }
      y[row++]=sum1;
      y[row++]=sum2;
      y[row++]=sum3;
      y[row++]=sum4;
      y[row++]=sum5;
      v1      =v5;       /* Since the next block to be processed starts there */
      idx    +=4*sz;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Node size not yet supported");
    }
  }
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*a->nz - nonzerorow);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* ----------------------------------------------------------- */
/* Almost same code as the MatMult_SeqAIJ_Inode() */
static PetscErrorCode MatMultAdd_SeqAIJ_Inode(Mat A,Vec xx,Vec zz,Vec yy)
{
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)A->data;
  PetscScalar       sum1,sum2,sum3,sum4,sum5,tmp0,tmp1;
  const MatScalar   *v1,*v2,*v3,*v4,*v5;
  const PetscScalar *x;
  PetscScalar       *y,*z,*zt;
  PetscErrorCode    ierr;
  PetscInt          i1,i2,n,i,row,node_max,nsz,sz;
  const PetscInt    *idx,*ns,*ii;

  PetscFunctionBegin;
  if (!a->inode.size) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Missing Inode Structure");
  node_max = a->inode.node_count;
  ns       = a->inode.size;     /* Node Size array */

  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArrayPair(zz,yy,&z,&y);CHKERRQ(ierr);
  zt = z;

  idx = a->j;
  v1  = a->a;
  ii  = a->i;

  for (i = 0,row = 0; i< node_max; ++i) {
    nsz = ns[i];
    n   = ii[1] - ii[0];
    ii += nsz;
    sz  = n;                    /* No of non zeros in this row */
                                /* Switch on the size of Node */
    switch (nsz) {               /* Each loop in 'case' is unrolled */
    case 1:
      sum1 = *zt++;

      for (n = 0; n< sz-1; n+=2) {
        i1    = idx[0];         /* The instructions are ordered to */
        i2    = idx[1];         /* make the compiler's job easy */
        idx  += 2;
        tmp0  = x[i1];
        tmp1  = x[i2];
        sum1 += v1[0] * tmp0 + v1[1] * tmp1; v1 += 2;
      }

      if (n   == sz-1) {          /* Take care of the last nonzero  */
        tmp0  = x[*idx++];
        sum1 += *v1++ * tmp0;
      }
      y[row++]=sum1;
      break;
    case 2:
      sum1 = *zt++;
      sum2 = *zt++;
      v2   = v1 + n;

      for (n = 0; n< sz-1; n+=2) {
        i1    = idx[0];
        i2    = idx[1];
        idx  += 2;
        tmp0  = x[i1];
        tmp1  = x[i2];
        sum1 += v1[0] * tmp0 + v1[1] * tmp1; v1 += 2;
        sum2 += v2[0] * tmp0 + v2[1] * tmp1; v2 += 2;
      }
      if (n   == sz-1) {
        tmp0  = x[*idx++];
        sum1 += *v1++ * tmp0;
        sum2 += *v2++ * tmp0;
      }
      y[row++]=sum1;
      y[row++]=sum2;
      v1      =v2;              /* Since the next block to be processed starts there*/
      idx    +=sz;
      break;
    case 3:
      sum1 = *zt++;
      sum2 = *zt++;
      sum3 = *zt++;
      v2   = v1 + n;
      v3   = v2 + n;

      for (n = 0; n< sz-1; n+=2) {
        i1    = idx[0];
        i2    = idx[1];
        idx  += 2;
        tmp0  = x[i1];
        tmp1  = x[i2];
        sum1 += v1[0] * tmp0 + v1[1] * tmp1; v1 += 2;
        sum2 += v2[0] * tmp0 + v2[1] * tmp1; v2 += 2;
        sum3 += v3[0] * tmp0 + v3[1] * tmp1; v3 += 2;
      }
      if (n == sz-1) {
        tmp0  = x[*idx++];
        sum1 += *v1++ * tmp0;
        sum2 += *v2++ * tmp0;
        sum3 += *v3++ * tmp0;
      }
      y[row++]=sum1;
      y[row++]=sum2;
      y[row++]=sum3;
      v1      =v3;              /* Since the next block to be processed starts there*/
      idx    +=2*sz;
      break;
    case 4:
      sum1 = *zt++;
      sum2 = *zt++;
      sum3 = *zt++;
      sum4 = *zt++;
      v2   = v1 + n;
      v3   = v2 + n;
      v4   = v3 + n;

      for (n = 0; n< sz-1; n+=2) {
        i1    = idx[0];
        i2    = idx[1];
        idx  += 2;
        tmp0  = x[i1];
        tmp1  = x[i2];
        sum1 += v1[0] * tmp0 + v1[1] *tmp1; v1 += 2;
        sum2 += v2[0] * tmp0 + v2[1] *tmp1; v2 += 2;
        sum3 += v3[0] * tmp0 + v3[1] *tmp1; v3 += 2;
        sum4 += v4[0] * tmp0 + v4[1] *tmp1; v4 += 2;
      }
      if (n == sz-1) {
        tmp0  = x[*idx++];
        sum1 += *v1++ * tmp0;
        sum2 += *v2++ * tmp0;
        sum3 += *v3++ * tmp0;
        sum4 += *v4++ * tmp0;
      }
      y[row++]=sum1;
      y[row++]=sum2;
      y[row++]=sum3;
      y[row++]=sum4;
      v1      =v4;              /* Since the next block to be processed starts there*/
      idx    +=3*sz;
      break;
    case 5:
      sum1 = *zt++;
      sum2 = *zt++;
      sum3 = *zt++;
      sum4 = *zt++;
      sum5 = *zt++;
      v2   = v1 + n;
      v3   = v2 + n;
      v4   = v3 + n;
      v5   = v4 + n;

      for (n = 0; n<sz-1; n+=2) {
        i1    = idx[0];
        i2    = idx[1];
        idx  += 2;
        tmp0  = x[i1];
        tmp1  = x[i2];
        sum1 += v1[0] * tmp0 + v1[1] *tmp1; v1 += 2;
        sum2 += v2[0] * tmp0 + v2[1] *tmp1; v2 += 2;
        sum3 += v3[0] * tmp0 + v3[1] *tmp1; v3 += 2;
        sum4 += v4[0] * tmp0 + v4[1] *tmp1; v4 += 2;
        sum5 += v5[0] * tmp0 + v5[1] *tmp1; v5 += 2;
      }
      if (n   == sz-1) {
        tmp0  = x[*idx++];
        sum1 += *v1++ * tmp0;
        sum2 += *v2++ * tmp0;
        sum3 += *v3++ * tmp0;
        sum4 += *v4++ * tmp0;
        sum5 += *v5++ * tmp0;
      }
      y[row++]=sum1;
      y[row++]=sum2;
      y[row++]=sum3;
      y[row++]=sum4;
      y[row++]=sum5;
      v1      =v5;       /* Since the next block to be processed starts there */
      idx    +=4*sz;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Node size not yet supported");
    }
  }
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayPair(zz,yy,&z,&y);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*a->nz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------- */
PetscErrorCode MatSolve_SeqAIJ_Inode_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqAIJ        *a    = (Mat_SeqAIJ*)A->data;
  IS                iscol = a->col,isrow = a->row;
  PetscErrorCode    ierr;
  const PetscInt    *r,*c,*rout,*cout;
  PetscInt          i,j,n = A->rmap->n,nz;
  PetscInt          node_max,*ns,row,nsz,aii,i0,i1;
  const PetscInt    *ai = a->i,*a_j = a->j,*vi,*ad,*aj;
  PetscScalar       *x,*tmp,*tmps,tmp0,tmp1;
  PetscScalar       sum1,sum2,sum3,sum4,sum5;
  const MatScalar   *v1,*v2,*v3,*v4,*v5,*a_a = a->a,*aa;
  const PetscScalar *b;

  PetscFunctionBegin;
  if (!a->inode.size) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Missing Inode Structure");
  node_max = a->inode.node_count;
  ns       = a->inode.size;     /* Node Size array */

  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArrayWrite(xx,&x);CHKERRQ(ierr);
  tmp  = a->solve_work;

  ierr = ISGetIndices(isrow,&rout);CHKERRQ(ierr); r = rout;
  ierr = ISGetIndices(iscol,&cout);CHKERRQ(ierr); c = cout + (n-1);

  /* forward solve the lower triangular */
  tmps = tmp;
  aa   = a_a;
  aj   = a_j;
  ad   = a->diag;

  for (i = 0,row = 0; i< node_max; ++i) {
    nsz = ns[i];
    aii = ai[row];
    v1  = aa + aii;
    vi  = aj + aii;
    nz  = ad[row]- aii;
    if (i < node_max-1) {
      /* Prefetch the block after the current one, the prefetch itself can't cause a memory error,
      * but our indexing to determine it's size could. */
      PetscPrefetchBlock(aj+ai[row+nsz],ad[row+nsz]-ai[row+nsz],0,PETSC_PREFETCH_HINT_NTA); /* indices */
      /* In my tests, it seems to be better to fetch entire rows instead of just the lower-triangular part */
      PetscPrefetchBlock(aa+ai[row+nsz],ad[row+nsz+ns[i+1]-1]-ai[row+nsz],0,PETSC_PREFETCH_HINT_NTA);
      /* for (j=0; j<ns[i+1]; j++) PetscPrefetchBlock(aa+ai[row+nsz+j],ad[row+nsz+j]-ai[row+nsz+j],0,0); */
    }

    switch (nsz) {               /* Each loop in 'case' is unrolled */
    case 1:
      sum1 = b[*r++];
      for (j=0; j<nz-1; j+=2) {
        i0    = vi[0];
        i1    = vi[1];
        vi   +=2;
        tmp0  = tmps[i0];
        tmp1  = tmps[i1];
        sum1 -= v1[0] * tmp0 + v1[1] * tmp1; v1 += 2;
      }
      if (j == nz-1) {
        tmp0  = tmps[*vi++];
        sum1 -= *v1++ *tmp0;
      }
      tmp[row++]=sum1;
      break;
    case 2:
      sum1 = b[*r++];
      sum2 = b[*r++];
      v2   = aa + ai[row+1];

      for (j=0; j<nz-1; j+=2) {
        i0    = vi[0];
        i1    = vi[1];
        vi   +=2;
        tmp0  = tmps[i0];
        tmp1  = tmps[i1];
        sum1 -= v1[0] * tmp0 + v1[1] * tmp1; v1 += 2;
        sum2 -= v2[0] * tmp0 + v2[1] * tmp1; v2 += 2;
      }
      if (j == nz-1) {
        tmp0  = tmps[*vi++];
        sum1 -= *v1++ *tmp0;
        sum2 -= *v2++ *tmp0;
      }
      sum2     -= *v2++ *sum1;
      tmp[row++]=sum1;
      tmp[row++]=sum2;
      break;
    case 3:
      sum1 = b[*r++];
      sum2 = b[*r++];
      sum3 = b[*r++];
      v2   = aa + ai[row+1];
      v3   = aa + ai[row+2];

      for (j=0; j<nz-1; j+=2) {
        i0    = vi[0];
        i1    = vi[1];
        vi   +=2;
        tmp0  = tmps[i0];
        tmp1  = tmps[i1];
        sum1 -= v1[0] * tmp0 + v1[1] * tmp1; v1 += 2;
        sum2 -= v2[0] * tmp0 + v2[1] * tmp1; v2 += 2;
        sum3 -= v3[0] * tmp0 + v3[1] * tmp1; v3 += 2;
      }
      if (j == nz-1) {
        tmp0  = tmps[*vi++];
        sum1 -= *v1++ *tmp0;
        sum2 -= *v2++ *tmp0;
        sum3 -= *v3++ *tmp0;
      }
      sum2 -= *v2++ * sum1;
      sum3 -= *v3++ * sum1;
      sum3 -= *v3++ * sum2;

      tmp[row++]=sum1;
      tmp[row++]=sum2;
      tmp[row++]=sum3;
      break;

    case 4:
      sum1 = b[*r++];
      sum2 = b[*r++];
      sum3 = b[*r++];
      sum4 = b[*r++];
      v2   = aa + ai[row+1];
      v3   = aa + ai[row+2];
      v4   = aa + ai[row+3];

      for (j=0; j<nz-1; j+=2) {
        i0    = vi[0];
        i1    = vi[1];
        vi   +=2;
        tmp0  = tmps[i0];
        tmp1  = tmps[i1];
        sum1 -= v1[0] * tmp0 + v1[1] * tmp1; v1 += 2;
        sum2 -= v2[0] * tmp0 + v2[1] * tmp1; v2 += 2;
        sum3 -= v3[0] * tmp0 + v3[1] * tmp1; v3 += 2;
        sum4 -= v4[0] * tmp0 + v4[1] * tmp1; v4 += 2;
      }
      if (j == nz-1) {
        tmp0  = tmps[*vi++];
        sum1 -= *v1++ *tmp0;
        sum2 -= *v2++ *tmp0;
        sum3 -= *v3++ *tmp0;
        sum4 -= *v4++ *tmp0;
      }
      sum2 -= *v2++ * sum1;
      sum3 -= *v3++ * sum1;
      sum4 -= *v4++ * sum1;
      sum3 -= *v3++ * sum2;
      sum4 -= *v4++ * sum2;
      sum4 -= *v4++ * sum3;

      tmp[row++]=sum1;
      tmp[row++]=sum2;
      tmp[row++]=sum3;
      tmp[row++]=sum4;
      break;
    case 5:
      sum1 = b[*r++];
      sum2 = b[*r++];
      sum3 = b[*r++];
      sum4 = b[*r++];
      sum5 = b[*r++];
      v2   = aa + ai[row+1];
      v3   = aa + ai[row+2];
      v4   = aa + ai[row+3];
      v5   = aa + ai[row+4];

      for (j=0; j<nz-1; j+=2) {
        i0    = vi[0];
        i1    = vi[1];
        vi   +=2;
        tmp0  = tmps[i0];
        tmp1  = tmps[i1];
        sum1 -= v1[0] * tmp0 + v1[1] * tmp1; v1 += 2;
        sum2 -= v2[0] * tmp0 + v2[1] * tmp1; v2 += 2;
        sum3 -= v3[0] * tmp0 + v3[1] * tmp1; v3 += 2;
        sum4 -= v4[0] * tmp0 + v4[1] * tmp1; v4 += 2;
        sum5 -= v5[0] * tmp0 + v5[1] * tmp1; v5 += 2;
      }
      if (j == nz-1) {
        tmp0  = tmps[*vi++];
        sum1 -= *v1++ *tmp0;
        sum2 -= *v2++ *tmp0;
        sum3 -= *v3++ *tmp0;
        sum4 -= *v4++ *tmp0;
        sum5 -= *v5++ *tmp0;
      }

      sum2 -= *v2++ * sum1;
      sum3 -= *v3++ * sum1;
      sum4 -= *v4++ * sum1;
      sum5 -= *v5++ * sum1;
      sum3 -= *v3++ * sum2;
      sum4 -= *v4++ * sum2;
      sum5 -= *v5++ * sum2;
      sum4 -= *v4++ * sum3;
      sum5 -= *v5++ * sum3;
      sum5 -= *v5++ * sum4;

      tmp[row++]=sum1;
      tmp[row++]=sum2;
      tmp[row++]=sum3;
      tmp[row++]=sum4;
      tmp[row++]=sum5;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Node size not yet supported \n");
    }
  }
  /* backward solve the upper triangular */
  for (i=node_max -1,row = n-1; i>=0; i--) {
    nsz = ns[i];
    aii = ai[row+1] -1;
    v1  = aa + aii;
    vi  = aj + aii;
    nz  = aii- ad[row];
    switch (nsz) {               /* Each loop in 'case' is unrolled */
    case 1:
      sum1 = tmp[row];

      for (j=nz; j>1; j-=2) {
        vi   -=2;
        i0    = vi[2];
        i1    = vi[1];
        tmp0  = tmps[i0];
        tmp1  = tmps[i1];
        v1   -= 2;
        sum1 -= v1[2] * tmp0 + v1[1] * tmp1;
      }
      if (j==1) {
        tmp0  = tmps[*vi--];
        sum1 -= *v1-- * tmp0;
      }
      x[*c--] = tmp[row] = sum1*a_a[ad[row]]; row--;
      break;
    case 2:
      sum1 = tmp[row];
      sum2 = tmp[row -1];
      v2   = aa + ai[row]-1;
      for (j=nz; j>1; j-=2) {
        vi   -=2;
        i0    = vi[2];
        i1    = vi[1];
        tmp0  = tmps[i0];
        tmp1  = tmps[i1];
        v1   -= 2;
        v2   -= 2;
        sum1 -= v1[2] * tmp0 + v1[1] * tmp1;
        sum2 -= v2[2] * tmp0 + v2[1] * tmp1;
      }
      if (j==1) {
        tmp0  = tmps[*vi--];
        sum1 -= *v1-- * tmp0;
        sum2 -= *v2-- * tmp0;
      }

      tmp0    = x[*c--] = tmp[row] = sum1*a_a[ad[row]]; row--;
      sum2   -= *v2-- * tmp0;
      x[*c--] = tmp[row] = sum2*a_a[ad[row]]; row--;
      break;
    case 3:
      sum1 = tmp[row];
      sum2 = tmp[row -1];
      sum3 = tmp[row -2];
      v2   = aa + ai[row]-1;
      v3   = aa + ai[row -1]-1;
      for (j=nz; j>1; j-=2) {
        vi   -=2;
        i0    = vi[2];
        i1    = vi[1];
        tmp0  = tmps[i0];
        tmp1  = tmps[i1];
        v1   -= 2;
        v2   -= 2;
        v3   -= 2;
        sum1 -= v1[2] * tmp0 + v1[1] * tmp1;
        sum2 -= v2[2] * tmp0 + v2[1] * tmp1;
        sum3 -= v3[2] * tmp0 + v3[1] * tmp1;
      }
      if (j==1) {
        tmp0  = tmps[*vi--];
        sum1 -= *v1-- * tmp0;
        sum2 -= *v2-- * tmp0;
        sum3 -= *v3-- * tmp0;
      }
      tmp0    = x[*c--] = tmp[row] = sum1*a_a[ad[row]]; row--;
      sum2   -= *v2-- * tmp0;
      sum3   -= *v3-- * tmp0;
      tmp0    = x[*c--] = tmp[row] = sum2*a_a[ad[row]]; row--;
      sum3   -= *v3-- * tmp0;
      x[*c--] = tmp[row] = sum3*a_a[ad[row]]; row--;

      break;
    case 4:
      sum1 = tmp[row];
      sum2 = tmp[row -1];
      sum3 = tmp[row -2];
      sum4 = tmp[row -3];
      v2   = aa + ai[row]-1;
      v3   = aa + ai[row -1]-1;
      v4   = aa + ai[row -2]-1;

      for (j=nz; j>1; j-=2) {
        vi   -=2;
        i0    = vi[2];
        i1    = vi[1];
        tmp0  = tmps[i0];
        tmp1  = tmps[i1];
        v1   -= 2;
        v2   -= 2;
        v3   -= 2;
        v4   -= 2;
        sum1 -= v1[2] * tmp0 + v1[1] * tmp1;
        sum2 -= v2[2] * tmp0 + v2[1] * tmp1;
        sum3 -= v3[2] * tmp0 + v3[1] * tmp1;
        sum4 -= v4[2] * tmp0 + v4[1] * tmp1;
      }
      if (j==1) {
        tmp0  = tmps[*vi--];
        sum1 -= *v1-- * tmp0;
        sum2 -= *v2-- * tmp0;
        sum3 -= *v3-- * tmp0;
        sum4 -= *v4-- * tmp0;
      }

      tmp0    = x[*c--] = tmp[row] = sum1*a_a[ad[row]]; row--;
      sum2   -= *v2-- * tmp0;
      sum3   -= *v3-- * tmp0;
      sum4   -= *v4-- * tmp0;
      tmp0    = x[*c--] = tmp[row] = sum2*a_a[ad[row]]; row--;
      sum3   -= *v3-- * tmp0;
      sum4   -= *v4-- * tmp0;
      tmp0    = x[*c--] = tmp[row] = sum3*a_a[ad[row]]; row--;
      sum4   -= *v4-- * tmp0;
      x[*c--] = tmp[row] = sum4*a_a[ad[row]]; row--;
      break;
    case 5:
      sum1 = tmp[row];
      sum2 = tmp[row -1];
      sum3 = tmp[row -2];
      sum4 = tmp[row -3];
      sum5 = tmp[row -4];
      v2   = aa + ai[row]-1;
      v3   = aa + ai[row -1]-1;
      v4   = aa + ai[row -2]-1;
      v5   = aa + ai[row -3]-1;
      for (j=nz; j>1; j-=2) {
        vi   -= 2;
        i0    = vi[2];
        i1    = vi[1];
        tmp0  = tmps[i0];
        tmp1  = tmps[i1];
        v1   -= 2;
        v2   -= 2;
        v3   -= 2;
        v4   -= 2;
        v5   -= 2;
        sum1 -= v1[2] * tmp0 + v1[1] * tmp1;
        sum2 -= v2[2] * tmp0 + v2[1] * tmp1;
        sum3 -= v3[2] * tmp0 + v3[1] * tmp1;
        sum4 -= v4[2] * tmp0 + v4[1] * tmp1;
        sum5 -= v5[2] * tmp0 + v5[1] * tmp1;
      }
      if (j==1) {
        tmp0  = tmps[*vi--];
        sum1 -= *v1-- * tmp0;
        sum2 -= *v2-- * tmp0;
        sum3 -= *v3-- * tmp0;
        sum4 -= *v4-- * tmp0;
        sum5 -= *v5-- * tmp0;
      }

      tmp0    = x[*c--] = tmp[row] = sum1*a_a[ad[row]]; row--;
      sum2   -= *v2-- * tmp0;
      sum3   -= *v3-- * tmp0;
      sum4   -= *v4-- * tmp0;
      sum5   -= *v5-- * tmp0;
      tmp0    = x[*c--] = tmp[row] = sum2*a_a[ad[row]]; row--;
      sum3   -= *v3-- * tmp0;
      sum4   -= *v4-- * tmp0;
      sum5   -= *v5-- * tmp0;
      tmp0    = x[*c--] = tmp[row] = sum3*a_a[ad[row]]; row--;
      sum4   -= *v4-- * tmp0;
      sum5   -= *v5-- * tmp0;
      tmp0    = x[*c--] = tmp[row] = sum4*a_a[ad[row]]; row--;
      sum5   -= *v5-- * tmp0;
      x[*c--] = tmp[row] = sum5*a_a[ad[row]]; row--;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Node size not yet supported \n");
    }
  }
  ierr = ISRestoreIndices(isrow,&rout);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscol,&cout);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArrayWrite(xx,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*a->nz - A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatLUFactorNumeric_SeqAIJ_Inode(Mat B,Mat A,const MatFactorInfo *info)
{
  Mat             C     =B;
  Mat_SeqAIJ      *a    =(Mat_SeqAIJ*)A->data,*b=(Mat_SeqAIJ*)C->data;
  IS              isrow = b->row,isicol = b->icol;
  PetscErrorCode  ierr;
  const PetscInt  *r,*ic,*ics;
  const PetscInt  n=A->rmap->n,*ai=a->i,*aj=a->j,*bi=b->i,*bj=b->j,*bdiag=b->diag;
  PetscInt        i,j,k,nz,nzL,row,*pj;
  const PetscInt  *ajtmp,*bjtmp;
  MatScalar       *pc,*pc1,*pc2,*pc3,*pc4,mul1,mul2,mul3,mul4,*pv,*rtmp1,*rtmp2,*rtmp3,*rtmp4;
  const MatScalar *aa=a->a,*v,*v1,*v2,*v3,*v4;
  FactorShiftCtx  sctx;
  const PetscInt  *ddiag;
  PetscReal       rs;
  MatScalar       d;
  PetscInt        inod,nodesz,node_max,col;
  const PetscInt  *ns;
  PetscInt        *tmp_vec1,*tmp_vec2,*nsmap;

  PetscFunctionBegin;
  /* MatPivotSetUp(): initialize shift context sctx */
  ierr = PetscMemzero(&sctx,sizeof(FactorShiftCtx));CHKERRQ(ierr);

  if (info->shifttype == (PetscReal)MAT_SHIFT_POSITIVE_DEFINITE) { /* set sctx.shift_top=max{rs} */
    ddiag          = a->diag;
    sctx.shift_top = info->zeropivot;
    for (i=0; i<n; i++) {
      /* calculate sum(|aij|)-RealPart(aii), amt of shift needed for this row */
      d  = (aa)[ddiag[i]];
      rs = -PetscAbsScalar(d) - PetscRealPart(d);
      v  = aa+ai[i];
      nz = ai[i+1] - ai[i];
      for (j=0; j<nz; j++) rs += PetscAbsScalar(v[j]);
      if (rs>sctx.shift_top) sctx.shift_top = rs;
    }
    sctx.shift_top *= 1.1;
    sctx.nshift_max = 5;
    sctx.shift_lo   = 0.;
    sctx.shift_hi   = 1.;
  }

  ierr = ISGetIndices(isrow,&r);CHKERRQ(ierr);
  ierr = ISGetIndices(isicol,&ic);CHKERRQ(ierr);

  ierr  = PetscCalloc4(n,&rtmp1,n,&rtmp2,n,&rtmp3,n,&rtmp4);CHKERRQ(ierr);
  ics   = ic;

  node_max = a->inode.node_count;
  ns       = a->inode.size;
  if (!ns) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Matrix without inode information");

  /* If max inode size > 4, split it into two inodes.*/
  /* also map the inode sizes according to the ordering */
  ierr = PetscMalloc1(n+1,&tmp_vec1);CHKERRQ(ierr);
  for (i=0,j=0; i<node_max; ++i,++j) {
    if (ns[i] > 4) {
      tmp_vec1[j] = 4;
      ++j;
      tmp_vec1[j] = ns[i] - tmp_vec1[j-1];
    } else {
      tmp_vec1[j] = ns[i];
    }
  }
  /* Use the correct node_max */
  node_max = j;

  /* Now reorder the inode info based on mat re-ordering info */
  /* First create a row -> inode_size_array_index map */
  ierr = PetscMalloc1(n+1,&nsmap);CHKERRQ(ierr);
  ierr = PetscMalloc1(node_max+1,&tmp_vec2);CHKERRQ(ierr);
  for (i=0,row=0; i<node_max; i++) {
    nodesz = tmp_vec1[i];
    for (j=0; j<nodesz; j++,row++) {
      nsmap[row] = i;
    }
  }
  /* Using nsmap, create a reordered ns structure */
  for (i=0,j=0; i< node_max; i++) {
    nodesz      = tmp_vec1[nsmap[r[j]]];     /* here the reordered row_no is in r[] */
    tmp_vec2[i] = nodesz;
    j          += nodesz;
  }
  ierr = PetscFree(nsmap);CHKERRQ(ierr);
  ierr = PetscFree(tmp_vec1);CHKERRQ(ierr);

  /* Now use the correct ns */
  ns = tmp_vec2;

  do {
    sctx.newshift = PETSC_FALSE;
    /* Now loop over each block-row, and do the factorization */
    for (inod=0,i=0; inod<node_max; inod++) { /* i: row index; inod: inode index */
      nodesz = ns[inod];

      switch (nodesz) {
      case 1:
        /*----------*/
        /* zero rtmp1 */
        /* L part */
        nz    = bi[i+1] - bi[i];
        bjtmp = bj + bi[i];
        for (j=0; j<nz; j++) rtmp1[bjtmp[j]] = 0.0;

        /* U part */
        nz    = bdiag[i]-bdiag[i+1];
        bjtmp = bj + bdiag[i+1]+1;
        for (j=0; j<nz; j++) rtmp1[bjtmp[j]] = 0.0;

        /* load in initial (unfactored row) */
        nz    = ai[r[i]+1] - ai[r[i]];
        ajtmp = aj + ai[r[i]];
        v     = aa + ai[r[i]];
        for (j=0; j<nz; j++) rtmp1[ics[ajtmp[j]]] = v[j];

        /* ZeropivotApply() */
        rtmp1[i] += sctx.shift_amount;  /* shift the diagonal of the matrix */

        /* elimination */
        bjtmp = bj + bi[i];
        row   = *bjtmp++;
        nzL   = bi[i+1] - bi[i];
        for (k=0; k < nzL; k++) {
          pc = rtmp1 + row;
          if (*pc != 0.0) {
            pv   = b->a + bdiag[row];
            mul1 = *pc * (*pv);
            *pc  = mul1;
            pj   = b->j + bdiag[row+1]+1; /* beginning of U(row,:) */
            pv   = b->a + bdiag[row+1]+1;
            nz   = bdiag[row]-bdiag[row+1]-1; /* num of entries in U(row,:) excluding diag */
            for (j=0; j<nz; j++) rtmp1[pj[j]] -= mul1 * pv[j];
            ierr = PetscLogFlops(1+2.0*nz);CHKERRQ(ierr);
          }
          row = *bjtmp++;
        }

        /* finished row so stick it into b->a */
        rs = 0.0;
        /* L part */
        pv = b->a + bi[i];
        pj = b->j + bi[i];
        nz = bi[i+1] - bi[i];
        for (j=0; j<nz; j++) {
          pv[j] = rtmp1[pj[j]]; rs += PetscAbsScalar(pv[j]);
        }

        /* U part */
        pv = b->a + bdiag[i+1]+1;
        pj = b->j + bdiag[i+1]+1;
        nz = bdiag[i] - bdiag[i+1]-1;
        for (j=0; j<nz; j++) {
          pv[j] = rtmp1[pj[j]]; rs += PetscAbsScalar(pv[j]);
        }

        /* Check zero pivot */
        sctx.rs = rs;
        sctx.pv = rtmp1[i];
        ierr    = MatPivotCheck(B,A,info,&sctx,i);CHKERRQ(ierr);
        if (sctx.newshift) break;

        /* Mark diagonal and invert diagonal for simplier triangular solves */
        pv  = b->a + bdiag[i];
        *pv = 1.0/sctx.pv; /* sctx.pv = rtmp1[i]+shiftamount if shifttype==MAT_SHIFT_INBLOCKS */
        break;

      case 2:
        /*----------*/
        /* zero rtmp1 and rtmp2 */
        /* L part */
        nz    = bi[i+1] - bi[i];
        bjtmp = bj + bi[i];
        for  (j=0; j<nz; j++) {
          col        = bjtmp[j];
          rtmp1[col] = 0.0; rtmp2[col] = 0.0;
        }

        /* U part */
        nz    = bdiag[i]-bdiag[i+1];
        bjtmp = bj + bdiag[i+1]+1;
        for  (j=0; j<nz; j++) {
          col        = bjtmp[j];
          rtmp1[col] = 0.0; rtmp2[col] = 0.0;
        }

        /* load in initial (unfactored row) */
        nz    = ai[r[i]+1] - ai[r[i]];
        ajtmp = aj + ai[r[i]];
        v1    = aa + ai[r[i]]; v2 = aa + ai[r[i]+1];
        for (j=0; j<nz; j++) {
          col        = ics[ajtmp[j]];
          rtmp1[col] = v1[j]; rtmp2[col] = v2[j];
        }
        /* ZeropivotApply(): shift the diagonal of the matrix  */
        rtmp1[i] += sctx.shift_amount; rtmp2[i+1] += sctx.shift_amount;

        /* elimination */
        bjtmp = bj + bi[i];
        row   = *bjtmp++; /* pivot row */
        nzL   = bi[i+1] - bi[i];
        for (k=0; k < nzL; k++) {
          pc1 = rtmp1 + row;
          pc2 = rtmp2 + row;
          if (*pc1 != 0.0 || *pc2 != 0.0) {
            pv   = b->a + bdiag[row];
            mul1 = *pc1*(*pv); mul2 = *pc2*(*pv);
            *pc1 = mul1;       *pc2 = mul2;

            pj = b->j + bdiag[row+1]+1;     /* beginning of U(row,:) */
            pv = b->a + bdiag[row+1]+1;
            nz = bdiag[row]-bdiag[row+1]-1; /* num of entries in U(row,:) excluding diag */
            for (j=0; j<nz; j++) {
              col         = pj[j];
              rtmp1[col] -= mul1 * pv[j];
              rtmp2[col] -= mul2 * pv[j];
            }
            ierr = PetscLogFlops(2+4.0*nz);CHKERRQ(ierr);
          }
          row = *bjtmp++;
        }

        /* finished row i; check zero pivot, then stick row i into b->a */
        rs = 0.0;
        /* L part */
        pc1 = b->a + bi[i];
        pj  = b->j + bi[i];
        nz  = bi[i+1] - bi[i];
        for (j=0; j<nz; j++) {
          col    = pj[j];
          pc1[j] = rtmp1[col]; rs += PetscAbsScalar(pc1[j]);
        }
        /* U part */
        pc1 = b->a + bdiag[i+1]+1;
        pj  = b->j + bdiag[i+1]+1;
        nz  = bdiag[i] - bdiag[i+1] - 1; /* exclude diagonal */
        for (j=0; j<nz; j++) {
          col    = pj[j];
          pc1[j] = rtmp1[col]; rs += PetscAbsScalar(pc1[j]);
        }

        sctx.rs = rs;
        sctx.pv = rtmp1[i];
        ierr    = MatPivotCheck(B,A,info,&sctx,i);CHKERRQ(ierr);
        if (sctx.newshift) break;
        pc1  = b->a + bdiag[i]; /* Mark diagonal */
        *pc1 = 1.0/sctx.pv;

        /* Now take care of diagonal 2x2 block. */
        pc2 = rtmp2 + i;
        if (*pc2 != 0.0) {
          mul1 = (*pc2)*(*pc1); /* *pc1=diag[i] is inverted! */
          *pc2 = mul1;          /* insert L entry */
          pj   = b->j + bdiag[i+1]+1;   /* beginning of U(i,:) */
          nz   = bdiag[i]-bdiag[i+1]-1; /* num of entries in U(i,:) excluding diag */
          for (j=0; j<nz; j++) {
            col = pj[j]; rtmp2[col] -= mul1 * rtmp1[col];
          }
          ierr = PetscLogFlops(1+2.0*nz);CHKERRQ(ierr);
        }

        /* finished row i+1; check zero pivot, then stick row i+1 into b->a */
        rs = 0.0;
        /* L part */
        pc2 = b->a + bi[i+1];
        pj  = b->j + bi[i+1];
        nz  = bi[i+2] - bi[i+1];
        for (j=0; j<nz; j++) {
          col    = pj[j];
          pc2[j] = rtmp2[col]; rs += PetscAbsScalar(pc2[j]);
        }
        /* U part */
        pc2 = b->a + bdiag[i+2]+1;
        pj  = b->j + bdiag[i+2]+1;
        nz  = bdiag[i+1] - bdiag[i+2] - 1; /* exclude diagonal */
        for (j=0; j<nz; j++) {
          col    = pj[j];
          pc2[j] = rtmp2[col]; rs += PetscAbsScalar(pc2[j]);
        }

        sctx.rs = rs;
        sctx.pv = rtmp2[i+1];
        ierr    = MatPivotCheck(B,A,info,&sctx,i+1);CHKERRQ(ierr);
        if (sctx.newshift) break;
        pc2  = b->a + bdiag[i+1];
        *pc2 = 1.0/sctx.pv;
        break;

      case 3:
        /*----------*/
        /* zero rtmp */
        /* L part */
        nz    = bi[i+1] - bi[i];
        bjtmp = bj + bi[i];
        for  (j=0; j<nz; j++) {
          col        = bjtmp[j];
          rtmp1[col] = 0.0; rtmp2[col] = 0.0; rtmp3[col] = 0.0;
        }

        /* U part */
        nz    = bdiag[i]-bdiag[i+1];
        bjtmp = bj + bdiag[i+1]+1;
        for  (j=0; j<nz; j++) {
          col        = bjtmp[j];
          rtmp1[col] = 0.0; rtmp2[col] = 0.0; rtmp3[col] = 0.0;
        }

        /* load in initial (unfactored row) */
        nz    = ai[r[i]+1] - ai[r[i]];
        ajtmp = aj + ai[r[i]];
        v1    = aa + ai[r[i]]; v2 = aa + ai[r[i]+1]; v3 = aa + ai[r[i]+2];
        for (j=0; j<nz; j++) {
          col        = ics[ajtmp[j]];
          rtmp1[col] = v1[j]; rtmp2[col] = v2[j]; rtmp3[col] = v3[j];
        }
        /* ZeropivotApply(): shift the diagonal of the matrix  */
        rtmp1[i] += sctx.shift_amount; rtmp2[i+1] += sctx.shift_amount; rtmp3[i+2] += sctx.shift_amount;

        /* elimination */
        bjtmp = bj + bi[i];
        row   = *bjtmp++; /* pivot row */
        nzL   = bi[i+1] - bi[i];
        for (k=0; k < nzL; k++) {
          pc1 = rtmp1 + row;
          pc2 = rtmp2 + row;
          pc3 = rtmp3 + row;
          if (*pc1 != 0.0 || *pc2 != 0.0 || *pc3 != 0.0) {
            pv   = b->a + bdiag[row];
            mul1 = *pc1*(*pv); mul2 = *pc2*(*pv); mul3 = *pc3*(*pv);
            *pc1 = mul1; *pc2 = mul2; *pc3 = mul3;

            pj = b->j + bdiag[row+1]+1;     /* beginning of U(row,:) */
            pv = b->a + bdiag[row+1]+1;
            nz = bdiag[row]-bdiag[row+1]-1; /* num of entries in U(row,:) excluding diag */
            for (j=0; j<nz; j++) {
              col         = pj[j];
              rtmp1[col] -= mul1 * pv[j];
              rtmp2[col] -= mul2 * pv[j];
              rtmp3[col] -= mul3 * pv[j];
            }
            ierr = PetscLogFlops(3+6.0*nz);CHKERRQ(ierr);
          }
          row = *bjtmp++;
        }

        /* finished row i; check zero pivot, then stick row i into b->a */
        rs = 0.0;
        /* L part */
        pc1 = b->a + bi[i];
        pj  = b->j + bi[i];
        nz  = bi[i+1] - bi[i];
        for (j=0; j<nz; j++) {
          col    = pj[j];
          pc1[j] = rtmp1[col]; rs += PetscAbsScalar(pc1[j]);
        }
        /* U part */
        pc1 = b->a + bdiag[i+1]+1;
        pj  = b->j + bdiag[i+1]+1;
        nz  = bdiag[i] - bdiag[i+1] - 1; /* exclude diagonal */
        for (j=0; j<nz; j++) {
          col    = pj[j];
          pc1[j] = rtmp1[col]; rs += PetscAbsScalar(pc1[j]);
        }

        sctx.rs = rs;
        sctx.pv = rtmp1[i];
        ierr    = MatPivotCheck(B,A,info,&sctx,i);CHKERRQ(ierr);
        if (sctx.newshift) break;
        pc1  = b->a + bdiag[i]; /* Mark diag[i] */
        *pc1 = 1.0/sctx.pv;

        /* Now take care of 1st column of diagonal 3x3 block. */
        pc2 = rtmp2 + i;
        pc3 = rtmp3 + i;
        if (*pc2 != 0.0 || *pc3 != 0.0) {
          mul2 = (*pc2)*(*pc1); *pc2 = mul2;
          mul3 = (*pc3)*(*pc1); *pc3 = mul3;
          pj   = b->j + bdiag[i+1]+1;   /* beginning of U(i,:) */
          nz   = bdiag[i]-bdiag[i+1]-1; /* num of entries in U(i,:) excluding diag */
          for (j=0; j<nz; j++) {
            col         = pj[j];
            rtmp2[col] -= mul2 * rtmp1[col];
            rtmp3[col] -= mul3 * rtmp1[col];
          }
          ierr = PetscLogFlops(2+4.0*nz);CHKERRQ(ierr);
        }

        /* finished row i+1; check zero pivot, then stick row i+1 into b->a */
        rs = 0.0;
        /* L part */
        pc2 = b->a + bi[i+1];
        pj  = b->j + bi[i+1];
        nz  = bi[i+2] - bi[i+1];
        for (j=0; j<nz; j++) {
          col    = pj[j];
          pc2[j] = rtmp2[col]; rs += PetscAbsScalar(pc2[j]);
        }
        /* U part */
        pc2 = b->a + bdiag[i+2]+1;
        pj  = b->j + bdiag[i+2]+1;
        nz  = bdiag[i+1] - bdiag[i+2] - 1; /* exclude diagonal */
        for (j=0; j<nz; j++) {
          col    = pj[j];
          pc2[j] = rtmp2[col]; rs += PetscAbsScalar(pc2[j]);
        }

        sctx.rs = rs;
        sctx.pv = rtmp2[i+1];
        ierr    = MatPivotCheck(B,A,info,&sctx,i+1);CHKERRQ(ierr);
        if (sctx.newshift) break;
        pc2  = b->a + bdiag[i+1];
        *pc2 = 1.0/sctx.pv; /* Mark diag[i+1] */

        /* Now take care of 2nd column of diagonal 3x3 block. */
        pc3 = rtmp3 + i+1;
        if (*pc3 != 0.0) {
          mul3 = (*pc3)*(*pc2); *pc3 = mul3;
          pj   = b->j + bdiag[i+2]+1;     /* beginning of U(i+1,:) */
          nz   = bdiag[i+1]-bdiag[i+2]-1; /* num of entries in U(i+1,:) excluding diag */
          for (j=0; j<nz; j++) {
            col         = pj[j];
            rtmp3[col] -= mul3 * rtmp2[col];
          }
          ierr = PetscLogFlops(1+2.0*nz);CHKERRQ(ierr);
        }

        /* finished i+2; check zero pivot, then stick row i+2 into b->a */
        rs = 0.0;
        /* L part */
        pc3 = b->a + bi[i+2];
        pj  = b->j + bi[i+2];
        nz  = bi[i+3] - bi[i+2];
        for (j=0; j<nz; j++) {
          col    = pj[j];
          pc3[j] = rtmp3[col]; rs += PetscAbsScalar(pc3[j]);
        }
        /* U part */
        pc3 = b->a + bdiag[i+3]+1;
        pj  = b->j + bdiag[i+3]+1;
        nz  = bdiag[i+2] - bdiag[i+3] - 1; /* exclude diagonal */
        for (j=0; j<nz; j++) {
          col    = pj[j];
          pc3[j] = rtmp3[col]; rs += PetscAbsScalar(pc3[j]);
        }

        sctx.rs = rs;
        sctx.pv = rtmp3[i+2];
        ierr    = MatPivotCheck(B,A,info,&sctx,i+2);CHKERRQ(ierr);
        if (sctx.newshift) break;
        pc3  = b->a + bdiag[i+2];
        *pc3 = 1.0/sctx.pv; /* Mark diag[i+2] */
        break;
      case 4:
        /*----------*/
        /* zero rtmp */
        /* L part */
        nz    = bi[i+1] - bi[i];
        bjtmp = bj + bi[i];
        for  (j=0; j<nz; j++) {
          col        = bjtmp[j];
          rtmp1[col] = 0.0; rtmp2[col] = 0.0; rtmp3[col] = 0.0;rtmp4[col] = 0.0;
        }

        /* U part */
        nz    = bdiag[i]-bdiag[i+1];
        bjtmp = bj + bdiag[i+1]+1;
        for  (j=0; j<nz; j++) {
          col        = bjtmp[j];
          rtmp1[col] = 0.0; rtmp2[col] = 0.0; rtmp3[col] = 0.0; rtmp4[col] = 0.0;
        }

        /* load in initial (unfactored row) */
        nz    = ai[r[i]+1] - ai[r[i]];
        ajtmp = aj + ai[r[i]];
        v1    = aa + ai[r[i]]; v2 = aa + ai[r[i]+1]; v3 = aa + ai[r[i]+2]; v4 = aa + ai[r[i]+3];
        for (j=0; j<nz; j++) {
          col        = ics[ajtmp[j]];
          rtmp1[col] = v1[j]; rtmp2[col] = v2[j]; rtmp3[col] = v3[j]; rtmp4[col] = v4[j];
        }
        /* ZeropivotApply(): shift the diagonal of the matrix  */
        rtmp1[i] += sctx.shift_amount; rtmp2[i+1] += sctx.shift_amount; rtmp3[i+2] += sctx.shift_amount; rtmp4[i+3] += sctx.shift_amount;

        /* elimination */
        bjtmp = bj + bi[i];
        row   = *bjtmp++; /* pivot row */
        nzL   = bi[i+1] - bi[i];
        for (k=0; k < nzL; k++) {
          pc1 = rtmp1 + row;
          pc2 = rtmp2 + row;
          pc3 = rtmp3 + row;
          pc4 = rtmp4 + row;
          if (*pc1 != 0.0 || *pc2 != 0.0 || *pc3 != 0.0 || *pc4 != 0.0) {
            pv   = b->a + bdiag[row];
            mul1 = *pc1*(*pv); mul2 = *pc2*(*pv); mul3 = *pc3*(*pv); mul4 = *pc4*(*pv);
            *pc1 = mul1; *pc2 = mul2; *pc3 = mul3; *pc4 = mul4;

            pj = b->j + bdiag[row+1]+1; /* beginning of U(row,:) */
            pv = b->a + bdiag[row+1]+1;
            nz = bdiag[row]-bdiag[row+1]-1; /* num of entries in U(row,:) excluding diag */
            for (j=0; j<nz; j++) {
              col         = pj[j];
              rtmp1[col] -= mul1 * pv[j];
              rtmp2[col] -= mul2 * pv[j];
              rtmp3[col] -= mul3 * pv[j];
              rtmp4[col] -= mul4 * pv[j];
            }
            ierr = PetscLogFlops(4+8.0*nz);CHKERRQ(ierr);
          }
          row = *bjtmp++;
        }

        /* finished row i; check zero pivot, then stick row i into b->a */
        rs = 0.0;
        /* L part */
        pc1 = b->a + bi[i];
        pj  = b->j + bi[i];
        nz  = bi[i+1] - bi[i];
        for (j=0; j<nz; j++) {
          col    = pj[j];
          pc1[j] = rtmp1[col]; rs += PetscAbsScalar(pc1[j]);
        }
        /* U part */
        pc1 = b->a + bdiag[i+1]+1;
        pj  = b->j + bdiag[i+1]+1;
        nz  = bdiag[i] - bdiag[i+1] - 1; /* exclude diagonal */
        for (j=0; j<nz; j++) {
          col    = pj[j];
          pc1[j] = rtmp1[col]; rs += PetscAbsScalar(pc1[j]);
        }

        sctx.rs = rs;
        sctx.pv = rtmp1[i];
        ierr    = MatPivotCheck(B,A,info,&sctx,i);CHKERRQ(ierr);
        if (sctx.newshift) break;
        pc1  = b->a + bdiag[i]; /* Mark diag[i] */
        *pc1 = 1.0/sctx.pv;

        /* Now take care of 1st column of diagonal 4x4 block. */
        pc2 = rtmp2 + i;
        pc3 = rtmp3 + i;
        pc4 = rtmp4 + i;
        if (*pc2 != 0.0 || *pc3 != 0.0 || *pc4 != 0.0) {
          mul2 = (*pc2)*(*pc1); *pc2 = mul2;
          mul3 = (*pc3)*(*pc1); *pc3 = mul3;
          mul4 = (*pc4)*(*pc1); *pc4 = mul4;
          pj   = b->j + bdiag[i+1]+1;   /* beginning of U(i,:) */
          nz   = bdiag[i]-bdiag[i+1]-1; /* num of entries in U(i,:) excluding diag */
          for (j=0; j<nz; j++) {
            col         = pj[j];
            rtmp2[col] -= mul2 * rtmp1[col];
            rtmp3[col] -= mul3 * rtmp1[col];
            rtmp4[col] -= mul4 * rtmp1[col];
          }
          ierr = PetscLogFlops(3+6.0*nz);CHKERRQ(ierr);
        }

        /* finished row i+1; check zero pivot, then stick row i+1 into b->a */
        rs = 0.0;
        /* L part */
        pc2 = b->a + bi[i+1];
        pj  = b->j + bi[i+1];
        nz  = bi[i+2] - bi[i+1];
        for (j=0; j<nz; j++) {
          col    = pj[j];
          pc2[j] = rtmp2[col]; rs += PetscAbsScalar(pc2[j]);
        }
        /* U part */
        pc2 = b->a + bdiag[i+2]+1;
        pj  = b->j + bdiag[i+2]+1;
        nz  = bdiag[i+1] - bdiag[i+2] - 1; /* exclude diagonal */
        for (j=0; j<nz; j++) {
          col    = pj[j];
          pc2[j] = rtmp2[col]; rs += PetscAbsScalar(pc2[j]);
        }

        sctx.rs = rs;
        sctx.pv = rtmp2[i+1];
        ierr    = MatPivotCheck(B,A,info,&sctx,i+1);CHKERRQ(ierr);
        if (sctx.newshift) break;
        pc2  = b->a + bdiag[i+1];
        *pc2 = 1.0/sctx.pv; /* Mark diag[i+1] */

        /* Now take care of 2nd column of diagonal 4x4 block. */
        pc3 = rtmp3 + i+1;
        pc4 = rtmp4 + i+1;
        if (*pc3 != 0.0 || *pc4 != 0.0) {
          mul3 = (*pc3)*(*pc2); *pc3 = mul3;
          mul4 = (*pc4)*(*pc2); *pc4 = mul4;
          pj   = b->j + bdiag[i+2]+1;     /* beginning of U(i+1,:) */
          nz   = bdiag[i+1]-bdiag[i+2]-1; /* num of entries in U(i+1,:) excluding diag */
          for (j=0; j<nz; j++) {
            col         = pj[j];
            rtmp3[col] -= mul3 * rtmp2[col];
            rtmp4[col] -= mul4 * rtmp2[col];
          }
          ierr = PetscLogFlops(4.0*nz);CHKERRQ(ierr);
        }

        /* finished i+2; check zero pivot, then stick row i+2 into b->a */
        rs = 0.0;
        /* L part */
        pc3 = b->a + bi[i+2];
        pj  = b->j + bi[i+2];
        nz  = bi[i+3] - bi[i+2];
        for (j=0; j<nz; j++) {
          col    = pj[j];
          pc3[j] = rtmp3[col]; rs += PetscAbsScalar(pc3[j]);
        }
        /* U part */
        pc3 = b->a + bdiag[i+3]+1;
        pj  = b->j + bdiag[i+3]+1;
        nz  = bdiag[i+2] - bdiag[i+3] - 1; /* exclude diagonal */
        for (j=0; j<nz; j++) {
          col    = pj[j];
          pc3[j] = rtmp3[col]; rs += PetscAbsScalar(pc3[j]);
        }

        sctx.rs = rs;
        sctx.pv = rtmp3[i+2];
        ierr    = MatPivotCheck(B,A,info,&sctx,i+2);CHKERRQ(ierr);
        if (sctx.newshift) break;
        pc3  = b->a + bdiag[i+2];
        *pc3 = 1.0/sctx.pv; /* Mark diag[i+2] */

        /* Now take care of 3rd column of diagonal 4x4 block. */
        pc4 = rtmp4 + i+2;
        if (*pc4 != 0.0) {
          mul4 = (*pc4)*(*pc3); *pc4 = mul4;
          pj   = b->j + bdiag[i+3]+1;     /* beginning of U(i+2,:) */
          nz   = bdiag[i+2]-bdiag[i+3]-1; /* num of entries in U(i+2,:) excluding diag */
          for (j=0; j<nz; j++) {
            col         = pj[j];
            rtmp4[col] -= mul4 * rtmp3[col];
          }
          ierr = PetscLogFlops(1+2.0*nz);CHKERRQ(ierr);
        }

        /* finished i+3; check zero pivot, then stick row i+3 into b->a */
        rs = 0.0;
        /* L part */
        pc4 = b->a + bi[i+3];
        pj  = b->j + bi[i+3];
        nz  = bi[i+4] - bi[i+3];
        for (j=0; j<nz; j++) {
          col    = pj[j];
          pc4[j] = rtmp4[col]; rs += PetscAbsScalar(pc4[j]);
        }
        /* U part */
        pc4 = b->a + bdiag[i+4]+1;
        pj  = b->j + bdiag[i+4]+1;
        nz  = bdiag[i+3] - bdiag[i+4] - 1; /* exclude diagonal */
        for (j=0; j<nz; j++) {
          col    = pj[j];
          pc4[j] = rtmp4[col]; rs += PetscAbsScalar(pc4[j]);
        }

        sctx.rs = rs;
        sctx.pv = rtmp4[i+3];
        ierr    = MatPivotCheck(B,A,info,&sctx,i+3);CHKERRQ(ierr);
        if (sctx.newshift) break;
        pc4  = b->a + bdiag[i+3];
        *pc4 = 1.0/sctx.pv; /* Mark diag[i+3] */
        break;

      default:
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Node size not yet supported \n");
      }
      if (sctx.newshift) break; /* break for (inod=0,i=0; inod<node_max; inod++) */
      i += nodesz;                 /* Update the row */
    }

    /* MatPivotRefine() */
    if (info->shifttype == (PetscReal) MAT_SHIFT_POSITIVE_DEFINITE && !sctx.newshift && sctx.shift_fraction>0 && sctx.nshift<sctx.nshift_max) {
      /*
       * if no shift in this attempt & shifting & started shifting & can refine,
       * then try lower shift
       */
      sctx.shift_hi       = sctx.shift_fraction;
      sctx.shift_fraction = (sctx.shift_hi+sctx.shift_lo)/2.;
      sctx.shift_amount   = sctx.shift_fraction * sctx.shift_top;
      sctx.newshift       = PETSC_TRUE;
      sctx.nshift++;
    }
  } while (sctx.newshift);

  ierr = PetscFree4(rtmp1,rtmp2,rtmp3,rtmp4);CHKERRQ(ierr);
  ierr = PetscFree(tmp_vec2);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isicol,&ic);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isrow,&r);CHKERRQ(ierr);

  if (b->inode.size) {
    C->ops->solve           = MatSolve_SeqAIJ_Inode;
  } else {
    C->ops->solve           = MatSolve_SeqAIJ;
  }
  C->ops->solveadd          = MatSolveAdd_SeqAIJ;
  C->ops->solvetranspose    = MatSolveTranspose_SeqAIJ;
  C->ops->solvetransposeadd = MatSolveTransposeAdd_SeqAIJ;
  C->ops->matsolve          = MatMatSolve_SeqAIJ;
  C->assembled              = PETSC_TRUE;
  C->preallocated           = PETSC_TRUE;

  ierr = PetscLogFlops(C->cmap->n);CHKERRQ(ierr);

  /* MatShiftView(A,info,&sctx) */
  if (sctx.nshift) {
    if (info->shifttype == (PetscReal) MAT_SHIFT_POSITIVE_DEFINITE) {
      ierr = PetscInfo4(A,"number of shift_pd tries %D, shift_amount %g, diagonal shifted up by %e fraction top_value %e\n",sctx.nshift,(double)sctx.shift_amount,(double)sctx.shift_fraction,(double)sctx.shift_top);CHKERRQ(ierr);
    } else if (info->shifttype == (PetscReal)MAT_SHIFT_NONZERO) {
      ierr = PetscInfo2(A,"number of shift_nz tries %D, shift_amount %g\n",sctx.nshift,(double)sctx.shift_amount);CHKERRQ(ierr);
    } else if (info->shifttype == (PetscReal)MAT_SHIFT_INBLOCKS) {
      ierr = PetscInfo2(A,"number of shift_inblocks applied %D, each shift_amount %g\n",sctx.nshift,(double)info->shiftamount);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatLUFactorNumeric_SeqAIJ_Inode_inplace(Mat B,Mat A,const MatFactorInfo *info)
{
  Mat             C     = B;
  Mat_SeqAIJ      *a    = (Mat_SeqAIJ*)A->data,*b = (Mat_SeqAIJ*)C->data;
  IS              iscol = b->col,isrow = b->row,isicol = b->icol;
  PetscErrorCode  ierr;
  const PetscInt  *r,*ic,*c,*ics;
  PetscInt        n   = A->rmap->n,*bi = b->i;
  PetscInt        *bj = b->j,*nbj=b->j +1,*ajtmp,*bjtmp,nz,nz_tmp,row,prow;
  PetscInt        i,j,idx,*bd = b->diag,node_max,nodesz;
  PetscInt        *ai = a->i,*aj = a->j;
  PetscInt        *ns,*tmp_vec1,*tmp_vec2,*nsmap,*pj;
  PetscScalar     mul1,mul2,mul3,tmp;
  MatScalar       *pc1,*pc2,*pc3,*ba = b->a,*pv,*rtmp11,*rtmp22,*rtmp33;
  const MatScalar *v1,*v2,*v3,*aa = a->a,*rtmp1;
  PetscReal       rs=0.0;
  FactorShiftCtx  sctx;

  PetscFunctionBegin;
  sctx.shift_top      = 0;
  sctx.nshift_max     = 0;
  sctx.shift_lo       = 0;
  sctx.shift_hi       = 0;
  sctx.shift_fraction = 0;

  /* if both shift schemes are chosen by user, only use info->shiftpd */
  if (info->shifttype==(PetscReal)MAT_SHIFT_POSITIVE_DEFINITE) { /* set sctx.shift_top=max{rs} */
    sctx.shift_top = 0;
    for (i=0; i<n; i++) {
      /* calculate rs = sum(|aij|)-RealPart(aii), amt of shift needed for this row */
      rs    = 0.0;
      ajtmp = aj + ai[i];
      rtmp1 = aa + ai[i];
      nz    = ai[i+1] - ai[i];
      for (j=0; j<nz; j++) {
        if (*ajtmp != i) {
          rs += PetscAbsScalar(*rtmp1++);
        } else {
          rs -= PetscRealPart(*rtmp1++);
        }
        ajtmp++;
      }
      if (rs>sctx.shift_top) sctx.shift_top = rs;
    }
    if (sctx.shift_top == 0.0) sctx.shift_top += 1.e-12;
    sctx.shift_top *= 1.1;
    sctx.nshift_max = 5;
    sctx.shift_lo   = 0.;
    sctx.shift_hi   = 1.;
  }
  sctx.shift_amount = 0;
  sctx.nshift       = 0;

  ierr   = ISGetIndices(isrow,&r);CHKERRQ(ierr);
  ierr   = ISGetIndices(iscol,&c);CHKERRQ(ierr);
  ierr   = ISGetIndices(isicol,&ic);CHKERRQ(ierr);
  ierr   = PetscCalloc3(n,&rtmp11,n,&rtmp22,n,&rtmp33);CHKERRQ(ierr);
  ics    = ic;

  node_max = a->inode.node_count;
  ns       = a->inode.size;
  if (!ns) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Matrix without inode information");

  /* If max inode size > 3, split it into two inodes.*/
  /* also map the inode sizes according to the ordering */
  ierr = PetscMalloc1(n+1,&tmp_vec1);CHKERRQ(ierr);
  for (i=0,j=0; i<node_max; ++i,++j) {
    if (ns[i]>3) {
      tmp_vec1[j] = ns[i]/2; /* Assuming ns[i] < =5  */
      ++j;
      tmp_vec1[j] = ns[i] - tmp_vec1[j-1];
    } else {
      tmp_vec1[j] = ns[i];
    }
  }
  /* Use the correct node_max */
  node_max = j;

  /* Now reorder the inode info based on mat re-ordering info */
  /* First create a row -> inode_size_array_index map */
  ierr = PetscMalloc2(n+1,&nsmap,node_max+1,&tmp_vec2);CHKERRQ(ierr);
  for (i=0,row=0; i<node_max; i++) {
    nodesz = tmp_vec1[i];
    for (j=0; j<nodesz; j++,row++) {
      nsmap[row] = i;
    }
  }
  /* Using nsmap, create a reordered ns structure */
  for (i=0,j=0; i< node_max; i++) {
    nodesz      = tmp_vec1[nsmap[r[j]]];     /* here the reordered row_no is in r[] */
    tmp_vec2[i] = nodesz;
    j          += nodesz;
  }
  ierr = PetscFree2(nsmap,tmp_vec1);CHKERRQ(ierr);
  /* Now use the correct ns */
  ns = tmp_vec2;

  do {
    sctx.newshift = PETSC_FALSE;
    /* Now loop over each block-row, and do the factorization */
    for (i=0,row=0; i<node_max; i++) {
      nodesz = ns[i];
      nz     = bi[row+1] - bi[row];
      bjtmp  = bj + bi[row];

      switch (nodesz) {
      case 1:
        for  (j=0; j<nz; j++) {
          idx         = bjtmp[j];
          rtmp11[idx] = 0.0;
        }

        /* load in initial (unfactored row) */
        idx    = r[row];
        nz_tmp = ai[idx+1] - ai[idx];
        ajtmp  = aj + ai[idx];
        v1     = aa + ai[idx];

        for (j=0; j<nz_tmp; j++) {
          idx         = ics[ajtmp[j]];
          rtmp11[idx] = v1[j];
        }
        rtmp11[ics[r[row]]] += sctx.shift_amount;

        prow = *bjtmp++;
        while (prow < row) {
          pc1 = rtmp11 + prow;
          if (*pc1 != 0.0) {
            pv     = ba + bd[prow];
            pj     = nbj + bd[prow];
            mul1   = *pc1 * *pv++;
            *pc1   = mul1;
            nz_tmp = bi[prow+1] - bd[prow] - 1;
            ierr   = PetscLogFlops(1+2.0*nz_tmp);CHKERRQ(ierr);
            for (j=0; j<nz_tmp; j++) {
              tmp          = pv[j];
              idx          = pj[j];
              rtmp11[idx] -= mul1 * tmp;
            }
          }
          prow = *bjtmp++;
        }
        pj  = bj + bi[row];
        pc1 = ba + bi[row];

        sctx.pv     = rtmp11[row];
        rtmp11[row] = 1.0/rtmp11[row]; /* invert diag */
        rs          = 0.0;
        for (j=0; j<nz; j++) {
          idx    = pj[j];
          pc1[j] = rtmp11[idx]; /* rtmp11 -> ba */
          if (idx != row) rs += PetscAbsScalar(pc1[j]);
        }
        sctx.rs = rs;
        ierr    = MatPivotCheck(B,A,info,&sctx,row);CHKERRQ(ierr);
        if (sctx.newshift) goto endofwhile;
        break;

      case 2:
        for (j=0; j<nz; j++) {
          idx         = bjtmp[j];
          rtmp11[idx] = 0.0;
          rtmp22[idx] = 0.0;
        }

        /* load in initial (unfactored row) */
        idx    = r[row];
        nz_tmp = ai[idx+1] - ai[idx];
        ajtmp  = aj + ai[idx];
        v1     = aa + ai[idx];
        v2     = aa + ai[idx+1];
        for (j=0; j<nz_tmp; j++) {
          idx         = ics[ajtmp[j]];
          rtmp11[idx] = v1[j];
          rtmp22[idx] = v2[j];
        }
        rtmp11[ics[r[row]]]   += sctx.shift_amount;
        rtmp22[ics[r[row+1]]] += sctx.shift_amount;

        prow = *bjtmp++;
        while (prow < row) {
          pc1 = rtmp11 + prow;
          pc2 = rtmp22 + prow;
          if (*pc1 != 0.0 || *pc2 != 0.0) {
            pv   = ba + bd[prow];
            pj   = nbj + bd[prow];
            mul1 = *pc1 * *pv;
            mul2 = *pc2 * *pv;
            ++pv;
            *pc1 = mul1;
            *pc2 = mul2;

            nz_tmp = bi[prow+1] - bd[prow] - 1;
            for (j=0; j<nz_tmp; j++) {
              tmp          = pv[j];
              idx          = pj[j];
              rtmp11[idx] -= mul1 * tmp;
              rtmp22[idx] -= mul2 * tmp;
            }
            ierr = PetscLogFlops(2+4.0*nz_tmp);CHKERRQ(ierr);
          }
          prow = *bjtmp++;
        }

        /* Now take care of diagonal 2x2 block. Note: prow = row here */
        pc1 = rtmp11 + prow;
        pc2 = rtmp22 + prow;

        sctx.pv = *pc1;
        pj      = bj + bi[prow];
        rs      = 0.0;
        for (j=0; j<nz; j++) {
          idx = pj[j];
          if (idx != prow) rs += PetscAbsScalar(rtmp11[idx]);
        }
        sctx.rs = rs;
        ierr    = MatPivotCheck(B,A,info,&sctx,row);CHKERRQ(ierr);
        if (sctx.newshift) goto endofwhile;

        if (*pc2 != 0.0) {
          pj     = nbj + bd[prow];
          mul2   = (*pc2)/(*pc1); /* since diag is not yet inverted.*/
          *pc2   = mul2;
          nz_tmp = bi[prow+1] - bd[prow] - 1;
          for (j=0; j<nz_tmp; j++) {
            idx          = pj[j];
            tmp          = rtmp11[idx];
            rtmp22[idx] -= mul2 * tmp;
          }
          ierr = PetscLogFlops(1+2.0*nz_tmp);CHKERRQ(ierr);
        }

        pj  = bj + bi[row];
        pc1 = ba + bi[row];
        pc2 = ba + bi[row+1];

        sctx.pv       = rtmp22[row+1];
        rs            = 0.0;
        rtmp11[row]   = 1.0/rtmp11[row];
        rtmp22[row+1] = 1.0/rtmp22[row+1];
        /* copy row entries from dense representation to sparse */
        for (j=0; j<nz; j++) {
          idx    = pj[j];
          pc1[j] = rtmp11[idx];
          pc2[j] = rtmp22[idx];
          if (idx != row+1) rs += PetscAbsScalar(pc2[j]);
        }
        sctx.rs = rs;
        ierr    = MatPivotCheck(B,A,info,&sctx,row+1);CHKERRQ(ierr);
        if (sctx.newshift) goto endofwhile;
        break;

      case 3:
        for  (j=0; j<nz; j++) {
          idx         = bjtmp[j];
          rtmp11[idx] = 0.0;
          rtmp22[idx] = 0.0;
          rtmp33[idx] = 0.0;
        }
        /* copy the nonzeros for the 3 rows from sparse representation to dense in rtmp*[] */
        idx    = r[row];
        nz_tmp = ai[idx+1] - ai[idx];
        ajtmp  = aj + ai[idx];
        v1     = aa + ai[idx];
        v2     = aa + ai[idx+1];
        v3     = aa + ai[idx+2];
        for (j=0; j<nz_tmp; j++) {
          idx         = ics[ajtmp[j]];
          rtmp11[idx] = v1[j];
          rtmp22[idx] = v2[j];
          rtmp33[idx] = v3[j];
        }
        rtmp11[ics[r[row]]]   += sctx.shift_amount;
        rtmp22[ics[r[row+1]]] += sctx.shift_amount;
        rtmp33[ics[r[row+2]]] += sctx.shift_amount;

        /* loop over all pivot row blocks above this row block */
        prow = *bjtmp++;
        while (prow < row) {
          pc1 = rtmp11 + prow;
          pc2 = rtmp22 + prow;
          pc3 = rtmp33 + prow;
          if (*pc1 != 0.0 || *pc2 != 0.0 || *pc3 !=0.0) {
            pv   = ba  + bd[prow];
            pj   = nbj + bd[prow];
            mul1 = *pc1 * *pv;
            mul2 = *pc2 * *pv;
            mul3 = *pc3 * *pv;
            ++pv;
            *pc1 = mul1;
            *pc2 = mul2;
            *pc3 = mul3;

            nz_tmp = bi[prow+1] - bd[prow] - 1;
            /* update this row based on pivot row */
            for (j=0; j<nz_tmp; j++) {
              tmp          = pv[j];
              idx          = pj[j];
              rtmp11[idx] -= mul1 * tmp;
              rtmp22[idx] -= mul2 * tmp;
              rtmp33[idx] -= mul3 * tmp;
            }
            ierr = PetscLogFlops(3+6.0*nz_tmp);CHKERRQ(ierr);
          }
          prow = *bjtmp++;
        }

        /* Now take care of diagonal 3x3 block in this set of rows */
        /* note: prow = row here */
        pc1 = rtmp11 + prow;
        pc2 = rtmp22 + prow;
        pc3 = rtmp33 + prow;

        sctx.pv = *pc1;
        pj      = bj + bi[prow];
        rs      = 0.0;
        for (j=0; j<nz; j++) {
          idx = pj[j];
          if (idx != row) rs += PetscAbsScalar(rtmp11[idx]);
        }
        sctx.rs = rs;
        ierr    = MatPivotCheck(B,A,info,&sctx,row);CHKERRQ(ierr);
        if (sctx.newshift) goto endofwhile;

        if (*pc2 != 0.0 || *pc3 != 0.0) {
          mul2   = (*pc2)/(*pc1);
          mul3   = (*pc3)/(*pc1);
          *pc2   = mul2;
          *pc3   = mul3;
          nz_tmp = bi[prow+1] - bd[prow] - 1;
          pj     = nbj + bd[prow];
          for (j=0; j<nz_tmp; j++) {
            idx          = pj[j];
            tmp          = rtmp11[idx];
            rtmp22[idx] -= mul2 * tmp;
            rtmp33[idx] -= mul3 * tmp;
          }
          ierr = PetscLogFlops(2+4.0*nz_tmp);CHKERRQ(ierr);
        }
        ++prow;

        pc2     = rtmp22 + prow;
        pc3     = rtmp33 + prow;
        sctx.pv = *pc2;
        pj      = bj + bi[prow];
        rs      = 0.0;
        for (j=0; j<nz; j++) {
          idx = pj[j];
          if (idx != prow) rs += PetscAbsScalar(rtmp22[idx]);
        }
        sctx.rs = rs;
        ierr    = MatPivotCheck(B,A,info,&sctx,row+1);CHKERRQ(ierr);
        if (sctx.newshift) goto endofwhile;

        if (*pc3 != 0.0) {
          mul3   = (*pc3)/(*pc2);
          *pc3   = mul3;
          pj     = nbj + bd[prow];
          nz_tmp = bi[prow+1] - bd[prow] - 1;
          for (j=0; j<nz_tmp; j++) {
            idx          = pj[j];
            tmp          = rtmp22[idx];
            rtmp33[idx] -= mul3 * tmp;
          }
          ierr = PetscLogFlops(1+2.0*nz_tmp);CHKERRQ(ierr);
        }

        pj  = bj + bi[row];
        pc1 = ba + bi[row];
        pc2 = ba + bi[row+1];
        pc3 = ba + bi[row+2];

        sctx.pv       = rtmp33[row+2];
        rs            = 0.0;
        rtmp11[row]   = 1.0/rtmp11[row];
        rtmp22[row+1] = 1.0/rtmp22[row+1];
        rtmp33[row+2] = 1.0/rtmp33[row+2];
        /* copy row entries from dense representation to sparse */
        for (j=0; j<nz; j++) {
          idx    = pj[j];
          pc1[j] = rtmp11[idx];
          pc2[j] = rtmp22[idx];
          pc3[j] = rtmp33[idx];
          if (idx != row+2) rs += PetscAbsScalar(pc3[j]);
        }

        sctx.rs = rs;
        ierr    = MatPivotCheck(B,A,info,&sctx,row+2);CHKERRQ(ierr);
        if (sctx.newshift) goto endofwhile;
        break;

      default:
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Node size not yet supported \n");
      }
      row += nodesz;                 /* Update the row */
    }
endofwhile:;
  } while (sctx.newshift);
  ierr = PetscFree3(rtmp11,rtmp22,rtmp33);CHKERRQ(ierr);
  ierr = PetscFree(tmp_vec2);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isicol,&ic);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isrow,&r);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscol,&c);CHKERRQ(ierr);

  (B)->ops->solve = MatSolve_SeqAIJ_inplace;
  /* do not set solve add, since MatSolve_Inode + Add is faster */
  C->ops->solvetranspose    = MatSolveTranspose_SeqAIJ_inplace;
  C->ops->solvetransposeadd = MatSolveTransposeAdd_SeqAIJ_inplace;
  C->assembled              = PETSC_TRUE;
  C->preallocated           = PETSC_TRUE;
  if (sctx.nshift) {
    if (info->shifttype == (PetscReal)MAT_SHIFT_POSITIVE_DEFINITE) {
      ierr = PetscInfo4(A,"number of shift_pd tries %D, shift_amount %g, diagonal shifted up by %e fraction top_value %e\n",sctx.nshift,(double)sctx.shift_amount,(double)sctx.shift_fraction,(double)sctx.shift_top);CHKERRQ(ierr);
    } else if (info->shifttype == (PetscReal)MAT_SHIFT_NONZERO) {
      ierr = PetscInfo2(A,"number of shift_nz tries %D, shift_amount %g\n",sctx.nshift,(double)sctx.shift_amount);CHKERRQ(ierr);
    }
  }
  ierr = PetscLogFlops(C->cmap->n);CHKERRQ(ierr);
  ierr = MatSeqAIJCheckInode(C);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/* ----------------------------------------------------------- */
PetscErrorCode MatSolve_SeqAIJ_Inode(Mat A,Vec bb,Vec xx)
{
  Mat_SeqAIJ        *a    = (Mat_SeqAIJ*)A->data;
  IS                iscol = a->col,isrow = a->row;
  PetscErrorCode    ierr;
  const PetscInt    *r,*c,*rout,*cout;
  PetscInt          i,j,n = A->rmap->n;
  PetscInt          node_max,row,nsz,aii,i0,i1,nz;
  const PetscInt    *ai = a->i,*a_j = a->j,*ns,*vi,*ad,*aj;
  PetscScalar       *x,*tmp,*tmps,tmp0,tmp1;
  PetscScalar       sum1,sum2,sum3,sum4,sum5;
  const MatScalar   *v1,*v2,*v3,*v4,*v5,*a_a = a->a,*aa;
  const PetscScalar *b;

  PetscFunctionBegin;
  if (!a->inode.size) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Missing Inode Structure");
  node_max = a->inode.node_count;
  ns       = a->inode.size;     /* Node Size array */

  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArrayWrite(xx,&x);CHKERRQ(ierr);
  tmp  = a->solve_work;

  ierr = ISGetIndices(isrow,&rout);CHKERRQ(ierr); r = rout;
  ierr = ISGetIndices(iscol,&cout);CHKERRQ(ierr); c = cout;

  /* forward solve the lower triangular */
  tmps = tmp;
  aa   = a_a;
  aj   = a_j;
  ad   = a->diag;

  for (i = 0,row = 0; i< node_max; ++i) {
    nsz = ns[i];
    aii = ai[row];
    v1  = aa + aii;
    vi  = aj + aii;
    nz  = ai[row+1]- ai[row];

    if (i < node_max-1) {
      /* Prefetch the indices for the next block */
      PetscPrefetchBlock(aj+ai[row+nsz],ai[row+nsz+1]-ai[row+nsz],0,PETSC_PREFETCH_HINT_NTA); /* indices */
      /* Prefetch the data for the next block */
      PetscPrefetchBlock(aa+ai[row+nsz],ai[row+nsz+ns[i+1]]-ai[row+nsz],0,PETSC_PREFETCH_HINT_NTA);
    }

    switch (nsz) {               /* Each loop in 'case' is unrolled */
    case 1:
      sum1 = b[r[row]];
      for (j=0; j<nz-1; j+=2) {
        i0    = vi[j];
        i1    = vi[j+1];
        tmp0  = tmps[i0];
        tmp1  = tmps[i1];
        sum1 -= v1[j]*tmp0 + v1[j+1]*tmp1;
      }
      if (j == nz-1) {
        tmp0  = tmps[vi[j]];
        sum1 -= v1[j]*tmp0;
      }
      tmp[row++]=sum1;
      break;
    case 2:
      sum1 = b[r[row]];
      sum2 = b[r[row+1]];
      v2   = aa + ai[row+1];

      for (j=0; j<nz-1; j+=2) {
        i0    = vi[j];
        i1    = vi[j+1];
        tmp0  = tmps[i0];
        tmp1  = tmps[i1];
        sum1 -= v1[j] * tmp0 + v1[j+1] * tmp1;
        sum2 -= v2[j] * tmp0 + v2[j+1] * tmp1;
      }
      if (j == nz-1) {
        tmp0  = tmps[vi[j]];
        sum1 -= v1[j] *tmp0;
        sum2 -= v2[j] *tmp0;
      }
      sum2     -= v2[nz] * sum1;
      tmp[row++]=sum1;
      tmp[row++]=sum2;
      break;
    case 3:
      sum1 = b[r[row]];
      sum2 = b[r[row+1]];
      sum3 = b[r[row+2]];
      v2   = aa + ai[row+1];
      v3   = aa + ai[row+2];

      for (j=0; j<nz-1; j+=2) {
        i0    = vi[j];
        i1    = vi[j+1];
        tmp0  = tmps[i0];
        tmp1  = tmps[i1];
        sum1 -= v1[j] * tmp0 + v1[j+1] * tmp1;
        sum2 -= v2[j] * tmp0 + v2[j+1] * tmp1;
        sum3 -= v3[j] * tmp0 + v3[j+1] * tmp1;
      }
      if (j == nz-1) {
        tmp0  = tmps[vi[j]];
        sum1 -= v1[j] *tmp0;
        sum2 -= v2[j] *tmp0;
        sum3 -= v3[j] *tmp0;
      }
      sum2     -= v2[nz] * sum1;
      sum3     -= v3[nz] * sum1;
      sum3     -= v3[nz+1] * sum2;
      tmp[row++]=sum1;
      tmp[row++]=sum2;
      tmp[row++]=sum3;
      break;

    case 4:
      sum1 = b[r[row]];
      sum2 = b[r[row+1]];
      sum3 = b[r[row+2]];
      sum4 = b[r[row+3]];
      v2   = aa + ai[row+1];
      v3   = aa + ai[row+2];
      v4   = aa + ai[row+3];

      for (j=0; j<nz-1; j+=2) {
        i0    = vi[j];
        i1    = vi[j+1];
        tmp0  = tmps[i0];
        tmp1  = tmps[i1];
        sum1 -= v1[j] * tmp0 + v1[j+1] * tmp1;
        sum2 -= v2[j] * tmp0 + v2[j+1] * tmp1;
        sum3 -= v3[j] * tmp0 + v3[j+1] * tmp1;
        sum4 -= v4[j] * tmp0 + v4[j+1] * tmp1;
      }
      if (j == nz-1) {
        tmp0  = tmps[vi[j]];
        sum1 -= v1[j] *tmp0;
        sum2 -= v2[j] *tmp0;
        sum3 -= v3[j] *tmp0;
        sum4 -= v4[j] *tmp0;
      }
      sum2 -= v2[nz] * sum1;
      sum3 -= v3[nz] * sum1;
      sum4 -= v4[nz] * sum1;
      sum3 -= v3[nz+1] * sum2;
      sum4 -= v4[nz+1] * sum2;
      sum4 -= v4[nz+2] * sum3;

      tmp[row++]=sum1;
      tmp[row++]=sum2;
      tmp[row++]=sum3;
      tmp[row++]=sum4;
      break;
    case 5:
      sum1 = b[r[row]];
      sum2 = b[r[row+1]];
      sum3 = b[r[row+2]];
      sum4 = b[r[row+3]];
      sum5 = b[r[row+4]];
      v2   = aa + ai[row+1];
      v3   = aa + ai[row+2];
      v4   = aa + ai[row+3];
      v5   = aa + ai[row+4];

      for (j=0; j<nz-1; j+=2) {
        i0    = vi[j];
        i1    = vi[j+1];
        tmp0  = tmps[i0];
        tmp1  = tmps[i1];
        sum1 -= v1[j] * tmp0 + v1[j+1] * tmp1;
        sum2 -= v2[j] * tmp0 + v2[j+1] * tmp1;
        sum3 -= v3[j] * tmp0 + v3[j+1] * tmp1;
        sum4 -= v4[j] * tmp0 + v4[j+1] * tmp1;
        sum5 -= v5[j] * tmp0 + v5[j+1] * tmp1;
      }
      if (j == nz-1) {
        tmp0  = tmps[vi[j]];
        sum1 -= v1[j] *tmp0;
        sum2 -= v2[j] *tmp0;
        sum3 -= v3[j] *tmp0;
        sum4 -= v4[j] *tmp0;
        sum5 -= v5[j] *tmp0;
      }

      sum2 -= v2[nz] * sum1;
      sum3 -= v3[nz] * sum1;
      sum4 -= v4[nz] * sum1;
      sum5 -= v5[nz] * sum1;
      sum3 -= v3[nz+1] * sum2;
      sum4 -= v4[nz+1] * sum2;
      sum5 -= v5[nz+1] * sum2;
      sum4 -= v4[nz+2] * sum3;
      sum5 -= v5[nz+2] * sum3;
      sum5 -= v5[nz+3] * sum4;

      tmp[row++]=sum1;
      tmp[row++]=sum2;
      tmp[row++]=sum3;
      tmp[row++]=sum4;
      tmp[row++]=sum5;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Node size not yet supported \n");
    }
  }
  /* backward solve the upper triangular */
  for (i=node_max -1,row = n-1; i>=0; i--) {
    nsz = ns[i];
    aii = ad[row+1] + 1;
    v1  = aa + aii;
    vi  = aj + aii;
    nz  = ad[row]- ad[row+1] - 1;

    if (i > 0) {
      /* Prefetch the indices for the next block */
      PetscPrefetchBlock(aj+ad[row-nsz+1]+1,ad[row-nsz]-ad[row-nsz+1],0,PETSC_PREFETCH_HINT_NTA);
      /* Prefetch the data for the next block */
      PetscPrefetchBlock(aa+ad[row-nsz+1]+1,ad[row-nsz-ns[i-1]+1]-ad[row-nsz+1],0,PETSC_PREFETCH_HINT_NTA);
    }

    switch (nsz) {               /* Each loop in 'case' is unrolled */
    case 1:
      sum1 = tmp[row];

      for (j=0; j<nz-1; j+=2) {
        i0    = vi[j];
        i1    = vi[j+1];
        tmp0  = tmps[i0];
        tmp1  = tmps[i1];
        sum1 -= v1[j] * tmp0 + v1[j+1] * tmp1;
      }
      if (j == nz-1) {
        tmp0  = tmps[vi[j]];
        sum1 -= v1[j]*tmp0;
      }
      x[c[row]] = tmp[row] = sum1*v1[nz]; row--;
      break;
    case 2:
      sum1 = tmp[row];
      sum2 = tmp[row-1];
      v2   = aa + ad[row] + 1;
      for (j=0; j<nz-1; j+=2) {
        i0    = vi[j];
        i1    = vi[j+1];
        tmp0  = tmps[i0];
        tmp1  = tmps[i1];
        sum1 -= v1[j] * tmp0 + v1[j+1] * tmp1;
        sum2 -= v2[j+1] * tmp0 + v2[j+2] * tmp1;
      }
      if (j == nz-1) {
        tmp0  = tmps[vi[j]];
        sum1 -= v1[j]* tmp0;
        sum2 -= v2[j+1]* tmp0;
      }

      tmp0      = x[c[row]] = tmp[row] = sum1*v1[nz]; row--;
      sum2     -= v2[0] * tmp0;
      x[c[row]] = tmp[row] = sum2*v2[nz+1]; row--;
      break;
    case 3:
      sum1 = tmp[row];
      sum2 = tmp[row -1];
      sum3 = tmp[row -2];
      v2   = aa + ad[row] + 1;
      v3   = aa + ad[row -1] + 1;
      for (j=0; j<nz-1; j+=2) {
        i0    = vi[j];
        i1    = vi[j+1];
        tmp0  = tmps[i0];
        tmp1  = tmps[i1];
        sum1 -= v1[j] * tmp0 + v1[j+1] * tmp1;
        sum2 -= v2[j+1] * tmp0 + v2[j+2] * tmp1;
        sum3 -= v3[j+2] * tmp0 + v3[j+3] * tmp1;
      }
      if (j== nz-1) {
        tmp0  = tmps[vi[j]];
        sum1 -= v1[j] * tmp0;
        sum2 -= v2[j+1] * tmp0;
        sum3 -= v3[j+2] * tmp0;
      }
      tmp0      = x[c[row]] = tmp[row] = sum1*v1[nz]; row--;
      sum2     -= v2[0]* tmp0;
      sum3     -= v3[1] * tmp0;
      tmp0      = x[c[row]] = tmp[row] = sum2*v2[nz+1]; row--;
      sum3     -= v3[0]* tmp0;
      x[c[row]] = tmp[row] = sum3*v3[nz+2]; row--;

      break;
    case 4:
      sum1 = tmp[row];
      sum2 = tmp[row -1];
      sum3 = tmp[row -2];
      sum4 = tmp[row -3];
      v2   = aa + ad[row]+1;
      v3   = aa + ad[row -1]+1;
      v4   = aa + ad[row -2]+1;

      for (j=0; j<nz-1; j+=2) {
        i0    = vi[j];
        i1    = vi[j+1];
        tmp0  = tmps[i0];
        tmp1  = tmps[i1];
        sum1 -= v1[j] * tmp0   + v1[j+1] * tmp1;
        sum2 -= v2[j+1] * tmp0 + v2[j+2] * tmp1;
        sum3 -= v3[j+2] * tmp0 + v3[j+3] * tmp1;
        sum4 -= v4[j+3] * tmp0 + v4[j+4] * tmp1;
      }
      if (j== nz-1) {
        tmp0  = tmps[vi[j]];
        sum1 -= v1[j] * tmp0;
        sum2 -= v2[j+1] * tmp0;
        sum3 -= v3[j+2] * tmp0;
        sum4 -= v4[j+3] * tmp0;
      }

      tmp0      = x[c[row]] = tmp[row] = sum1*v1[nz]; row--;
      sum2     -= v2[0] * tmp0;
      sum3     -= v3[1] * tmp0;
      sum4     -= v4[2] * tmp0;
      tmp0      = x[c[row]] = tmp[row] = sum2*v2[nz+1]; row--;
      sum3     -= v3[0] * tmp0;
      sum4     -= v4[1] * tmp0;
      tmp0      = x[c[row]] = tmp[row] = sum3*v3[nz+2]; row--;
      sum4     -= v4[0] * tmp0;
      x[c[row]] = tmp[row] = sum4*v4[nz+3]; row--;
      break;
    case 5:
      sum1 = tmp[row];
      sum2 = tmp[row -1];
      sum3 = tmp[row -2];
      sum4 = tmp[row -3];
      sum5 = tmp[row -4];
      v2   = aa + ad[row]+1;
      v3   = aa + ad[row -1]+1;
      v4   = aa + ad[row -2]+1;
      v5   = aa + ad[row -3]+1;
      for (j=0; j<nz-1; j+=2) {
        i0    = vi[j];
        i1    = vi[j+1];
        tmp0  = tmps[i0];
        tmp1  = tmps[i1];
        sum1 -= v1[j] * tmp0 + v1[j+1] * tmp1;
        sum2 -= v2[j+1] * tmp0 + v2[j+2] * tmp1;
        sum3 -= v3[j+2] * tmp0 + v3[j+3] * tmp1;
        sum4 -= v4[j+3] * tmp0 + v4[j+4] * tmp1;
        sum5 -= v5[j+4] * tmp0 + v5[j+5] * tmp1;
      }
      if (j==nz-1) {
        tmp0  = tmps[vi[j]];
        sum1 -= v1[j] * tmp0;
        sum2 -= v2[j+1] * tmp0;
        sum3 -= v3[j+2] * tmp0;
        sum4 -= v4[j+3] * tmp0;
        sum5 -= v5[j+4] * tmp0;
      }

      tmp0      = x[c[row]] = tmp[row] = sum1*v1[nz]; row--;
      sum2     -= v2[0] * tmp0;
      sum3     -= v3[1] * tmp0;
      sum4     -= v4[2] * tmp0;
      sum5     -= v5[3] * tmp0;
      tmp0      = x[c[row]] = tmp[row] = sum2*v2[nz+1]; row--;
      sum3     -= v3[0] * tmp0;
      sum4     -= v4[1] * tmp0;
      sum5     -= v5[2] * tmp0;
      tmp0      = x[c[row]] = tmp[row] = sum3*v3[nz+2]; row--;
      sum4     -= v4[0] * tmp0;
      sum5     -= v5[1] * tmp0;
      tmp0      = x[c[row]] = tmp[row] = sum4*v4[nz+3]; row--;
      sum5     -= v5[0] * tmp0;
      x[c[row]] = tmp[row] = sum5*v5[nz+4]; row--;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Node size not yet supported \n");
    }
  }
  ierr = ISRestoreIndices(isrow,&rout);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscol,&cout);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArrayWrite(xx,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*a->nz - A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*
     Makes a longer coloring[] array and calls the usual code with that
*/
PetscErrorCode MatColoringPatch_SeqAIJ_Inode(Mat mat,PetscInt ncolors,PetscInt nin,ISColoringValue coloring[],ISColoring *iscoloring)
{
  Mat_SeqAIJ      *a = (Mat_SeqAIJ*)mat->data;
  PetscErrorCode  ierr;
  PetscInt        n = mat->cmap->n,m = a->inode.node_count,j,*ns = a->inode.size,row;
  PetscInt        *colorused,i;
  ISColoringValue *newcolor;

  PetscFunctionBegin;
  ierr = PetscMalloc1(n+1,&newcolor);CHKERRQ(ierr);
  /* loop over inodes, marking a color for each column*/
  row = 0;
  for (i=0; i<m; i++) {
    for (j=0; j<ns[i]; j++) {
      newcolor[row++] = coloring[i] + j*ncolors;
    }
  }

  /* eliminate unneeded colors */
  ierr = PetscCalloc1(5*ncolors,&colorused);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    colorused[newcolor[i]] = 1;
  }

  for (i=1; i<5*ncolors; i++) {
    colorused[i] += colorused[i-1];
  }
  ncolors = colorused[5*ncolors-1];
  for (i=0; i<n; i++) {
    newcolor[i] = colorused[newcolor[i]]-1;
  }
  ierr = PetscFree(colorused);CHKERRQ(ierr);
  ierr = ISColoringCreate(PetscObjectComm((PetscObject)mat),ncolors,n,newcolor,PETSC_OWN_POINTER,iscoloring);CHKERRQ(ierr);
  ierr = PetscFree(coloring);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include <petsc/private/kernels/blockinvert.h>

PetscErrorCode MatSOR_SeqAIJ_Inode(Mat A,Vec bb,PetscReal omega,MatSORType flag,PetscReal fshift,PetscInt its,PetscInt lits,Vec xx)
{
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)A->data;
  PetscScalar       sum1 = 0.0,sum2 = 0.0,sum3 = 0.0,sum4 = 0.0,sum5 = 0.0,tmp0,tmp1,tmp2,tmp3;
  MatScalar         *ibdiag,*bdiag,work[25],*t;
  PetscScalar       *x,tmp4,tmp5,x1,x2,x3,x4,x5;
  const MatScalar   *v = a->a,*v1 = NULL,*v2 = NULL,*v3 = NULL,*v4 = NULL,*v5 = NULL;
  const PetscScalar *xb, *b;
  PetscReal         zeropivot = 100.*PETSC_MACHINE_EPSILON, shift = 0.0;
  PetscErrorCode    ierr;
  PetscInt          n,m = a->inode.node_count,cnt = 0,i,j,row,i1,i2;
  PetscInt          sz,k,ipvt[5];
  PetscBool         allowzeropivot,zeropivotdetected;
  const PetscInt    *sizes = a->inode.size,*idx,*diag = a->diag,*ii = a->i;

  PetscFunctionBegin;
  allowzeropivot = PetscNot(A->erroriffailure);
  if (omega != 1.0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for omega != 1.0; use -mat_no_inode");
  if (fshift != 0.0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support for fshift != 0.0; use -mat_no_inode");

  if (!a->inode.ibdiagvalid) {
    if (!a->inode.ibdiag) {
      /* calculate space needed for diagonal blocks */
      for (i=0; i<m; i++) {
        cnt += sizes[i]*sizes[i];
      }
      a->inode.bdiagsize = cnt;

      ierr = PetscMalloc3(cnt,&a->inode.ibdiag,cnt,&a->inode.bdiag,A->rmap->n,&a->inode.ssor_work);CHKERRQ(ierr);
    }

    /* copy over the diagonal blocks and invert them */
    ibdiag = a->inode.ibdiag;
    bdiag  = a->inode.bdiag;
    cnt    = 0;
    for (i=0, row = 0; i<m; i++) {
      for (j=0; j<sizes[i]; j++) {
        for (k=0; k<sizes[i]; k++) {
          bdiag[cnt+k*sizes[i]+j] = v[diag[row+j] - j + k];
        }
      }
      ierr = PetscArraycpy(ibdiag+cnt,bdiag+cnt,sizes[i]*sizes[i]);CHKERRQ(ierr);

      switch (sizes[i]) {
      case 1:
        /* Create matrix data structure */
        if (PetscAbsScalar(ibdiag[cnt]) < zeropivot) {
          if (allowzeropivot) {
            A->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
            A->factorerror_zeropivot_value = PetscAbsScalar(ibdiag[cnt]);
            A->factorerror_zeropivot_row   = row;
            ierr = PetscInfo1(A,"Zero pivot, row %D\n",row);CHKERRQ(ierr);
          } else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_MAT_LU_ZRPVT,"Zero pivot on row %D",row);
        }
        ibdiag[cnt] = 1.0/ibdiag[cnt];
        break;
      case 2:
        ierr = PetscKernel_A_gets_inverse_A_2(ibdiag+cnt,shift,allowzeropivot,&zeropivotdetected);CHKERRQ(ierr);
        if (zeropivotdetected) A->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
        break;
      case 3:
        ierr = PetscKernel_A_gets_inverse_A_3(ibdiag+cnt,shift,allowzeropivot,&zeropivotdetected);CHKERRQ(ierr);
        if (zeropivotdetected) A->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
        break;
      case 4:
        ierr = PetscKernel_A_gets_inverse_A_4(ibdiag+cnt,shift,allowzeropivot,&zeropivotdetected);CHKERRQ(ierr);
        if (zeropivotdetected) A->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
        break;
      case 5:
        ierr = PetscKernel_A_gets_inverse_A_5(ibdiag+cnt,ipvt,work,shift,allowzeropivot,&zeropivotdetected);CHKERRQ(ierr);
        if (zeropivotdetected) A->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
        break;
      default:
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Inode size %D not supported",sizes[i]);
      }
      cnt += sizes[i]*sizes[i];
      row += sizes[i];
    }
    a->inode.ibdiagvalid = PETSC_TRUE;
  }
  ibdiag = a->inode.ibdiag;
  bdiag  = a->inode.bdiag;
  t      = a->inode.ssor_work;

  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
  /* We count flops by assuming the upper triangular and lower triangular parts have the same number of nonzeros */
  if (flag & SOR_ZERO_INITIAL_GUESS) {
    if (flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP) {

      for (i=0, row=0; i<m; i++) {
        sz  = diag[row] - ii[row];
        v1  = a->a + ii[row];
        idx = a->j + ii[row];

        /* see comments for MatMult_SeqAIJ_Inode() for how this is coded */
        switch (sizes[i]) {
        case 1:

          sum1 = b[row];
          for (n = 0; n<sz-1; n+=2) {
            i1    = idx[0];
            i2    = idx[1];
            idx  += 2;
            tmp0  = x[i1];
            tmp1  = x[i2];
            sum1 -= v1[0] * tmp0 + v1[1] * tmp1; v1 += 2;
          }

          if (n == sz-1) {
            tmp0  = x[*idx];
            sum1 -= *v1 * tmp0;
          }
          t[row]   = sum1;
          x[row++] = sum1*(*ibdiag++);
          break;
        case 2:
          v2   = a->a + ii[row+1];
          sum1 = b[row];
          sum2 = b[row+1];
          for (n = 0; n<sz-1; n+=2) {
            i1    = idx[0];
            i2    = idx[1];
            idx  += 2;
            tmp0  = x[i1];
            tmp1  = x[i2];
            sum1 -= v1[0] * tmp0 + v1[1] * tmp1; v1 += 2;
            sum2 -= v2[0] * tmp0 + v2[1] * tmp1; v2 += 2;
          }

          if (n == sz-1) {
            tmp0  = x[*idx];
            sum1 -= v1[0] * tmp0;
            sum2 -= v2[0] * tmp0;
          }
          t[row]   = sum1;
          t[row+1] = sum2;
          x[row++] = sum1*ibdiag[0] + sum2*ibdiag[2];
          x[row++] = sum1*ibdiag[1] + sum2*ibdiag[3];
          ibdiag  += 4;
          break;
        case 3:
          v2   = a->a + ii[row+1];
          v3   = a->a + ii[row+2];
          sum1 = b[row];
          sum2 = b[row+1];
          sum3 = b[row+2];
          for (n = 0; n<sz-1; n+=2) {
            i1    = idx[0];
            i2    = idx[1];
            idx  += 2;
            tmp0  = x[i1];
            tmp1  = x[i2];
            sum1 -= v1[0] * tmp0 + v1[1] * tmp1; v1 += 2;
            sum2 -= v2[0] * tmp0 + v2[1] * tmp1; v2 += 2;
            sum3 -= v3[0] * tmp0 + v3[1] * tmp1; v3 += 2;
          }

          if (n == sz-1) {
            tmp0  = x[*idx];
            sum1 -= v1[0] * tmp0;
            sum2 -= v2[0] * tmp0;
            sum3 -= v3[0] * tmp0;
          }
          t[row]   = sum1;
          t[row+1] = sum2;
          t[row+2] = sum3;
          x[row++] = sum1*ibdiag[0] + sum2*ibdiag[3] + sum3*ibdiag[6];
          x[row++] = sum1*ibdiag[1] + sum2*ibdiag[4] + sum3*ibdiag[7];
          x[row++] = sum1*ibdiag[2] + sum2*ibdiag[5] + sum3*ibdiag[8];
          ibdiag  += 9;
          break;
        case 4:
          v2   = a->a + ii[row+1];
          v3   = a->a + ii[row+2];
          v4   = a->a + ii[row+3];
          sum1 = b[row];
          sum2 = b[row+1];
          sum3 = b[row+2];
          sum4 = b[row+3];
          for (n = 0; n<sz-1; n+=2) {
            i1    = idx[0];
            i2    = idx[1];
            idx  += 2;
            tmp0  = x[i1];
            tmp1  = x[i2];
            sum1 -= v1[0] * tmp0 + v1[1] * tmp1; v1 += 2;
            sum2 -= v2[0] * tmp0 + v2[1] * tmp1; v2 += 2;
            sum3 -= v3[0] * tmp0 + v3[1] * tmp1; v3 += 2;
            sum4 -= v4[0] * tmp0 + v4[1] * tmp1; v4 += 2;
          }

          if (n == sz-1) {
            tmp0  = x[*idx];
            sum1 -= v1[0] * tmp0;
            sum2 -= v2[0] * tmp0;
            sum3 -= v3[0] * tmp0;
            sum4 -= v4[0] * tmp0;
          }
          t[row]   = sum1;
          t[row+1] = sum2;
          t[row+2] = sum3;
          t[row+3] = sum4;
          x[row++] = sum1*ibdiag[0] + sum2*ibdiag[4] + sum3*ibdiag[8] + sum4*ibdiag[12];
          x[row++] = sum1*ibdiag[1] + sum2*ibdiag[5] + sum3*ibdiag[9] + sum4*ibdiag[13];
          x[row++] = sum1*ibdiag[2] + sum2*ibdiag[6] + sum3*ibdiag[10] + sum4*ibdiag[14];
          x[row++] = sum1*ibdiag[3] + sum2*ibdiag[7] + sum3*ibdiag[11] + sum4*ibdiag[15];
          ibdiag  += 16;
          break;
        case 5:
          v2   = a->a + ii[row+1];
          v3   = a->a + ii[row+2];
          v4   = a->a + ii[row+3];
          v5   = a->a + ii[row+4];
          sum1 = b[row];
          sum2 = b[row+1];
          sum3 = b[row+2];
          sum4 = b[row+3];
          sum5 = b[row+4];
          for (n = 0; n<sz-1; n+=2) {
            i1    = idx[0];
            i2    = idx[1];
            idx  += 2;
            tmp0  = x[i1];
            tmp1  = x[i2];
            sum1 -= v1[0] * tmp0 + v1[1] * tmp1; v1 += 2;
            sum2 -= v2[0] * tmp0 + v2[1] * tmp1; v2 += 2;
            sum3 -= v3[0] * tmp0 + v3[1] * tmp1; v3 += 2;
            sum4 -= v4[0] * tmp0 + v4[1] * tmp1; v4 += 2;
            sum5 -= v5[0] * tmp0 + v5[1] * tmp1; v5 += 2;
          }

          if (n == sz-1) {
            tmp0  = x[*idx];
            sum1 -= v1[0] * tmp0;
            sum2 -= v2[0] * tmp0;
            sum3 -= v3[0] * tmp0;
            sum4 -= v4[0] * tmp0;
            sum5 -= v5[0] * tmp0;
          }
          t[row]   = sum1;
          t[row+1] = sum2;
          t[row+2] = sum3;
          t[row+3] = sum4;
          t[row+4] = sum5;
          x[row++] = sum1*ibdiag[0] + sum2*ibdiag[5] + sum3*ibdiag[10] + sum4*ibdiag[15] + sum5*ibdiag[20];
          x[row++] = sum1*ibdiag[1] + sum2*ibdiag[6] + sum3*ibdiag[11] + sum4*ibdiag[16] + sum5*ibdiag[21];
          x[row++] = sum1*ibdiag[2] + sum2*ibdiag[7] + sum3*ibdiag[12] + sum4*ibdiag[17] + sum5*ibdiag[22];
          x[row++] = sum1*ibdiag[3] + sum2*ibdiag[8] + sum3*ibdiag[13] + sum4*ibdiag[18] + sum5*ibdiag[23];
          x[row++] = sum1*ibdiag[4] + sum2*ibdiag[9] + sum3*ibdiag[14] + sum4*ibdiag[19] + sum5*ibdiag[24];
          ibdiag  += 25;
          break;
        default:
          SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Inode size %D not supported",sizes[i]);
        }
      }

      xb   = t;
      ierr = PetscLogFlops(a->nz);CHKERRQ(ierr);
    } else xb = b;
    if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP) {

      ibdiag = a->inode.ibdiag+a->inode.bdiagsize;
      for (i=m-1, row=A->rmap->n-1; i>=0; i--) {
        ibdiag -= sizes[i]*sizes[i];
        sz      = ii[row+1] - diag[row] - 1;
        v1      = a->a + diag[row] + 1;
        idx     = a->j + diag[row] + 1;

        /* see comments for MatMult_SeqAIJ_Inode() for how this is coded */
        switch (sizes[i]) {
        case 1:

          sum1 = xb[row];
          for (n = 0; n<sz-1; n+=2) {
            i1    = idx[0];
            i2    = idx[1];
            idx  += 2;
            tmp0  = x[i1];
            tmp1  = x[i2];
            sum1 -= v1[0] * tmp0 + v1[1] * tmp1; v1 += 2;
          }

          if (n == sz-1) {
            tmp0  = x[*idx];
            sum1 -= *v1*tmp0;
          }
          x[row--] = sum1*(*ibdiag);
          break;

        case 2:

          sum1 = xb[row];
          sum2 = xb[row-1];
          /* note that sum1 is associated with the second of the two rows */
          v2 = a->a + diag[row-1] + 2;
          for (n = 0; n<sz-1; n+=2) {
            i1    = idx[0];
            i2    = idx[1];
            idx  += 2;
            tmp0  = x[i1];
            tmp1  = x[i2];
            sum1 -= v1[0] * tmp0 + v1[1] * tmp1; v1 += 2;
            sum2 -= v2[0] * tmp0 + v2[1] * tmp1; v2 += 2;
          }

          if (n == sz-1) {
            tmp0  = x[*idx];
            sum1 -= *v1*tmp0;
            sum2 -= *v2*tmp0;
          }
          x[row--] = sum2*ibdiag[1] + sum1*ibdiag[3];
          x[row--] = sum2*ibdiag[0] + sum1*ibdiag[2];
          break;
        case 3:

          sum1 = xb[row];
          sum2 = xb[row-1];
          sum3 = xb[row-2];
          v2   = a->a + diag[row-1] + 2;
          v3   = a->a + diag[row-2] + 3;
          for (n = 0; n<sz-1; n+=2) {
            i1    = idx[0];
            i2    = idx[1];
            idx  += 2;
            tmp0  = x[i1];
            tmp1  = x[i2];
            sum1 -= v1[0] * tmp0 + v1[1] * tmp1; v1 += 2;
            sum2 -= v2[0] * tmp0 + v2[1] * tmp1; v2 += 2;
            sum3 -= v3[0] * tmp0 + v3[1] * tmp1; v3 += 2;
          }

          if (n == sz-1) {
            tmp0  = x[*idx];
            sum1 -= *v1*tmp0;
            sum2 -= *v2*tmp0;
            sum3 -= *v3*tmp0;
          }
          x[row--] = sum3*ibdiag[2] + sum2*ibdiag[5] + sum1*ibdiag[8];
          x[row--] = sum3*ibdiag[1] + sum2*ibdiag[4] + sum1*ibdiag[7];
          x[row--] = sum3*ibdiag[0] + sum2*ibdiag[3] + sum1*ibdiag[6];
          break;
        case 4:

          sum1 = xb[row];
          sum2 = xb[row-1];
          sum3 = xb[row-2];
          sum4 = xb[row-3];
          v2   = a->a + diag[row-1] + 2;
          v3   = a->a + diag[row-2] + 3;
          v4   = a->a + diag[row-3] + 4;
          for (n = 0; n<sz-1; n+=2) {
            i1    = idx[0];
            i2    = idx[1];
            idx  += 2;
            tmp0  = x[i1];
            tmp1  = x[i2];
            sum1 -= v1[0] * tmp0 + v1[1] * tmp1; v1 += 2;
            sum2 -= v2[0] * tmp0 + v2[1] * tmp1; v2 += 2;
            sum3 -= v3[0] * tmp0 + v3[1] * tmp1; v3 += 2;
            sum4 -= v4[0] * tmp0 + v4[1] * tmp1; v4 += 2;
          }

          if (n == sz-1) {
            tmp0  = x[*idx];
            sum1 -= *v1*tmp0;
            sum2 -= *v2*tmp0;
            sum3 -= *v3*tmp0;
            sum4 -= *v4*tmp0;
          }
          x[row--] = sum4*ibdiag[3] + sum3*ibdiag[7] + sum2*ibdiag[11] + sum1*ibdiag[15];
          x[row--] = sum4*ibdiag[2] + sum3*ibdiag[6] + sum2*ibdiag[10] + sum1*ibdiag[14];
          x[row--] = sum4*ibdiag[1] + sum3*ibdiag[5] + sum2*ibdiag[9] + sum1*ibdiag[13];
          x[row--] = sum4*ibdiag[0] + sum3*ibdiag[4] + sum2*ibdiag[8] + sum1*ibdiag[12];
          break;
        case 5:

          sum1 = xb[row];
          sum2 = xb[row-1];
          sum3 = xb[row-2];
          sum4 = xb[row-3];
          sum5 = xb[row-4];
          v2   = a->a + diag[row-1] + 2;
          v3   = a->a + diag[row-2] + 3;
          v4   = a->a + diag[row-3] + 4;
          v5   = a->a + diag[row-4] + 5;
          for (n = 0; n<sz-1; n+=2) {
            i1    = idx[0];
            i2    = idx[1];
            idx  += 2;
            tmp0  = x[i1];
            tmp1  = x[i2];
            sum1 -= v1[0] * tmp0 + v1[1] * tmp1; v1 += 2;
            sum2 -= v2[0] * tmp0 + v2[1] * tmp1; v2 += 2;
            sum3 -= v3[0] * tmp0 + v3[1] * tmp1; v3 += 2;
            sum4 -= v4[0] * tmp0 + v4[1] * tmp1; v4 += 2;
            sum5 -= v5[0] * tmp0 + v5[1] * tmp1; v5 += 2;
          }

          if (n == sz-1) {
            tmp0  = x[*idx];
            sum1 -= *v1*tmp0;
            sum2 -= *v2*tmp0;
            sum3 -= *v3*tmp0;
            sum4 -= *v4*tmp0;
            sum5 -= *v5*tmp0;
          }
          x[row--] = sum5*ibdiag[4] + sum4*ibdiag[9] + sum3*ibdiag[14] + sum2*ibdiag[19] + sum1*ibdiag[24];
          x[row--] = sum5*ibdiag[3] + sum4*ibdiag[8] + sum3*ibdiag[13] + sum2*ibdiag[18] + sum1*ibdiag[23];
          x[row--] = sum5*ibdiag[2] + sum4*ibdiag[7] + sum3*ibdiag[12] + sum2*ibdiag[17] + sum1*ibdiag[22];
          x[row--] = sum5*ibdiag[1] + sum4*ibdiag[6] + sum3*ibdiag[11] + sum2*ibdiag[16] + sum1*ibdiag[21];
          x[row--] = sum5*ibdiag[0] + sum4*ibdiag[5] + sum3*ibdiag[10] + sum2*ibdiag[15] + sum1*ibdiag[20];
          break;
        default:
          SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Inode size %D not supported",sizes[i]);
        }
      }

      ierr = PetscLogFlops(a->nz);CHKERRQ(ierr);
    }
    its--;
  }
  while (its--) {

    if (flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP) {
      for (i=0, row=0, ibdiag = a->inode.ibdiag;
           i<m;
           row += sizes[i], ibdiag += sizes[i]*sizes[i], i++) {

        sz  = diag[row] - ii[row];
        v1  = a->a + ii[row];
        idx = a->j + ii[row];
        /* see comments for MatMult_SeqAIJ_Inode() for how this is coded */
        switch (sizes[i]) {
        case 1:
          sum1 = b[row];
          for (n = 0; n<sz-1; n+=2) {
            i1    = idx[0];
            i2    = idx[1];
            idx  += 2;
            tmp0  = x[i1];
            tmp1  = x[i2];
            sum1 -= v1[0] * tmp0 + v1[1] * tmp1; v1 += 2;
          }
          if (n == sz-1) {
            tmp0  = x[*idx++];
            sum1 -= *v1 * tmp0;
            v1++;
          }
          t[row]   = sum1;
          sz      = ii[row+1] - diag[row] - 1;
          idx     = a->j + diag[row] + 1;
          v1 += 1;
          for (n = 0; n<sz-1; n+=2) {
            i1    = idx[0];
            i2    = idx[1];
            idx  += 2;
            tmp0  = x[i1];
            tmp1  = x[i2];
            sum1 -= v1[0] * tmp0 + v1[1] * tmp1; v1 += 2;
          }
          if (n == sz-1) {
            tmp0  = x[*idx++];
            sum1 -= *v1 * tmp0;
          }
          /* in MatSOR_SeqAIJ this line would be
           *
           * x[row] = (1-omega)*x[row]+(sum1+(*bdiag++)*x[row])*(*ibdiag++);
           *
           * but omega == 1, so this becomes
           *
           * x[row] = sum1*(*ibdiag++);
           *
           */
          x[row] = sum1*(*ibdiag);
          break;
        case 2:
          v2   = a->a + ii[row+1];
          sum1 = b[row];
          sum2 = b[row+1];
          for (n = 0; n<sz-1; n+=2) {
            i1    = idx[0];
            i2    = idx[1];
            idx  += 2;
            tmp0  = x[i1];
            tmp1  = x[i2];
            sum1 -= v1[0] * tmp0 + v1[1] * tmp1; v1 += 2;
            sum2 -= v2[0] * tmp0 + v2[1] * tmp1; v2 += 2;
          }
          if (n == sz-1) {
            tmp0  = x[*idx++];
            sum1 -= v1[0] * tmp0;
            sum2 -= v2[0] * tmp0;
            v1++; v2++;
          }
          t[row]   = sum1;
          t[row+1] = sum2;
          sz      = ii[row+1] - diag[row] - 2;
          idx     = a->j + diag[row] + 2;
          v1 += 2;
          v2 += 2;
          for (n = 0; n<sz-1; n+=2) {
            i1    = idx[0];
            i2    = idx[1];
            idx  += 2;
            tmp0  = x[i1];
            tmp1  = x[i2];
            sum1 -= v1[0] * tmp0 + v1[1] * tmp1; v1 += 2;
            sum2 -= v2[0] * tmp0 + v2[1] * tmp1; v2 += 2;
          }
          if (n == sz-1) {
            tmp0  = x[*idx];
            sum1 -= v1[0] * tmp0;
            sum2 -= v2[0] * tmp0;
          }
          x[row] = sum1*ibdiag[0] + sum2*ibdiag[2];
          x[row+1] = sum1*ibdiag[1] + sum2*ibdiag[3];
          break;
        case 3:
          v2   = a->a + ii[row+1];
          v3   = a->a + ii[row+2];
          sum1 = b[row];
          sum2 = b[row+1];
          sum3 = b[row+2];
          for (n = 0; n<sz-1; n+=2) {
            i1    = idx[0];
            i2    = idx[1];
            idx  += 2;
            tmp0  = x[i1];
            tmp1  = x[i2];
            sum1 -= v1[0] * tmp0 + v1[1] * tmp1; v1 += 2;
            sum2 -= v2[0] * tmp0 + v2[1] * tmp1; v2 += 2;
            sum3 -= v3[0] * tmp0 + v3[1] * tmp1; v3 += 2;
          }
          if (n == sz-1) {
            tmp0  = x[*idx++];
            sum1 -= v1[0] * tmp0;
            sum2 -= v2[0] * tmp0;
            sum3 -= v3[0] * tmp0;
            v1++; v2++; v3++;
          }
          t[row]   = sum1;
          t[row+1] = sum2;
          t[row+2] = sum3;
          sz      = ii[row+1] - diag[row] - 3;
          idx     = a->j + diag[row] + 3;
          v1 += 3;
          v2 += 3;
          v3 += 3;
          for (n = 0; n<sz-1; n+=2) {
            i1    = idx[0];
            i2    = idx[1];
            idx  += 2;
            tmp0  = x[i1];
            tmp1  = x[i2];
            sum1 -= v1[0] * tmp0 + v1[1] * tmp1; v1 += 2;
            sum2 -= v2[0] * tmp0 + v2[1] * tmp1; v2 += 2;
            sum3 -= v3[0] * tmp0 + v3[1] * tmp1; v3 += 2;
          }
          if (n == sz-1) {
            tmp0  = x[*idx];
            sum1 -= v1[0] * tmp0;
            sum2 -= v2[0] * tmp0;
            sum3 -= v3[0] * tmp0;
          }
          x[row] = sum1*ibdiag[0] + sum2*ibdiag[3] + sum3*ibdiag[6];
          x[row+1] = sum1*ibdiag[1] + sum2*ibdiag[4] + sum3*ibdiag[7];
          x[row+2] = sum1*ibdiag[2] + sum2*ibdiag[5] + sum3*ibdiag[8];
          break;
        case 4:
          v2   = a->a + ii[row+1];
          v3   = a->a + ii[row+2];
          v4   = a->a + ii[row+3];
          sum1 = b[row];
          sum2 = b[row+1];
          sum3 = b[row+2];
          sum4 = b[row+3];
          for (n = 0; n<sz-1; n+=2) {
            i1    = idx[0];
            i2    = idx[1];
            idx  += 2;
            tmp0  = x[i1];
            tmp1  = x[i2];
            sum1 -= v1[0] * tmp0 + v1[1] * tmp1; v1 += 2;
            sum2 -= v2[0] * tmp0 + v2[1] * tmp1; v2 += 2;
            sum3 -= v3[0] * tmp0 + v3[1] * tmp1; v3 += 2;
            sum4 -= v4[0] * tmp0 + v4[1] * tmp1; v4 += 2;
          }
          if (n == sz-1) {
            tmp0  = x[*idx++];
            sum1 -= v1[0] * tmp0;
            sum2 -= v2[0] * tmp0;
            sum3 -= v3[0] * tmp0;
            sum4 -= v4[0] * tmp0;
            v1++; v2++; v3++; v4++;
          }
          t[row]   = sum1;
          t[row+1] = sum2;
          t[row+2] = sum3;
          t[row+3] = sum4;
          sz      = ii[row+1] - diag[row] - 4;
          idx     = a->j + diag[row] + 4;
          v1 += 4;
          v2 += 4;
          v3 += 4;
          v4 += 4;
          for (n = 0; n<sz-1; n+=2) {
            i1    = idx[0];
            i2    = idx[1];
            idx  += 2;
            tmp0  = x[i1];
            tmp1  = x[i2];
            sum1 -= v1[0] * tmp0 + v1[1] * tmp1; v1 += 2;
            sum2 -= v2[0] * tmp0 + v2[1] * tmp1; v2 += 2;
            sum3 -= v3[0] * tmp0 + v3[1] * tmp1; v3 += 2;
            sum4 -= v4[0] * tmp0 + v4[1] * tmp1; v4 += 2;
          }
          if (n == sz-1) {
            tmp0  = x[*idx];
            sum1 -= v1[0] * tmp0;
            sum2 -= v2[0] * tmp0;
            sum3 -= v3[0] * tmp0;
            sum4 -= v4[0] * tmp0;
          }
          x[row] =   sum1*ibdiag[0] + sum2*ibdiag[4] + sum3*ibdiag[8] + sum4*ibdiag[12];
          x[row+1] = sum1*ibdiag[1] + sum2*ibdiag[5] + sum3*ibdiag[9] + sum4*ibdiag[13];
          x[row+2] = sum1*ibdiag[2] + sum2*ibdiag[6] + sum3*ibdiag[10] + sum4*ibdiag[14];
          x[row+3] = sum1*ibdiag[3] + sum2*ibdiag[7] + sum3*ibdiag[11] + sum4*ibdiag[15];
          break;
        case 5:
          v2   = a->a + ii[row+1];
          v3   = a->a + ii[row+2];
          v4   = a->a + ii[row+3];
          v5   = a->a + ii[row+4];
          sum1 = b[row];
          sum2 = b[row+1];
          sum3 = b[row+2];
          sum4 = b[row+3];
          sum5 = b[row+4];
          for (n = 0; n<sz-1; n+=2) {
            i1    = idx[0];
            i2    = idx[1];
            idx  += 2;
            tmp0  = x[i1];
            tmp1  = x[i2];
            sum1 -= v1[0] * tmp0 + v1[1] * tmp1; v1 += 2;
            sum2 -= v2[0] * tmp0 + v2[1] * tmp1; v2 += 2;
            sum3 -= v3[0] * tmp0 + v3[1] * tmp1; v3 += 2;
            sum4 -= v4[0] * tmp0 + v4[1] * tmp1; v4 += 2;
            sum5 -= v5[0] * tmp0 + v5[1] * tmp1; v5 += 2;
          }
          if (n == sz-1) {
            tmp0  = x[*idx++];
            sum1 -= v1[0] * tmp0;
            sum2 -= v2[0] * tmp0;
            sum3 -= v3[0] * tmp0;
            sum4 -= v4[0] * tmp0;
            sum5 -= v5[0] * tmp0;
            v1++; v2++; v3++; v4++; v5++;
          }
          t[row]   = sum1;
          t[row+1] = sum2;
          t[row+2] = sum3;
          t[row+3] = sum4;
          t[row+4] = sum5;
          sz      = ii[row+1] - diag[row] - 5;
          idx     = a->j + diag[row] + 5;
          v1 += 5;
          v2 += 5;
          v3 += 5;
          v4 += 5;
          v5 += 5;
          for (n = 0; n<sz-1; n+=2) {
            i1    = idx[0];
            i2    = idx[1];
            idx  += 2;
            tmp0  = x[i1];
            tmp1  = x[i2];
            sum1 -= v1[0] * tmp0 + v1[1] * tmp1; v1 += 2;
            sum2 -= v2[0] * tmp0 + v2[1] * tmp1; v2 += 2;
            sum3 -= v3[0] * tmp0 + v3[1] * tmp1; v3 += 2;
            sum4 -= v4[0] * tmp0 + v4[1] * tmp1; v4 += 2;
            sum5 -= v5[0] * tmp0 + v5[1] * tmp1; v5 += 2;
          }
          if (n == sz-1) {
            tmp0  = x[*idx];
            sum1 -= v1[0] * tmp0;
            sum2 -= v2[0] * tmp0;
            sum3 -= v3[0] * tmp0;
            sum4 -= v4[0] * tmp0;
            sum5 -= v5[0] * tmp0;
          }
          x[row]   = sum1*ibdiag[0] + sum2*ibdiag[5] + sum3*ibdiag[10] + sum4*ibdiag[15] + sum5*ibdiag[20];
          x[row+1] = sum1*ibdiag[1] + sum2*ibdiag[6] + sum3*ibdiag[11] + sum4*ibdiag[16] + sum5*ibdiag[21];
          x[row+2] = sum1*ibdiag[2] + sum2*ibdiag[7] + sum3*ibdiag[12] + sum4*ibdiag[17] + sum5*ibdiag[22];
          x[row+3] = sum1*ibdiag[3] + sum2*ibdiag[8] + sum3*ibdiag[13] + sum4*ibdiag[18] + sum5*ibdiag[23];
          x[row+4] = sum1*ibdiag[4] + sum2*ibdiag[9] + sum3*ibdiag[14] + sum4*ibdiag[19] + sum5*ibdiag[24];
          break;
        default:
          SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Inode size %D not supported",sizes[i]);
        }
      }
      xb   = t;
      ierr = PetscLogFlops(2.0*a->nz);CHKERRQ(ierr);  /* undercounts diag inverse */
    } else xb = b;

    if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP) {

      ibdiag = a->inode.ibdiag+a->inode.bdiagsize;
      for (i=m-1, row=A->rmap->n-1; i>=0; i--) {
        ibdiag -= sizes[i]*sizes[i];

        /* set RHS */
        if (xb == b) {
          /* whole (old way) */
          sz      = ii[row+1] - ii[row];
          idx     = a->j + ii[row];
          switch (sizes[i]) {
          case 5:
            v5      = a->a + ii[row-4];
          case 4: /* fall through */
            v4      = a->a + ii[row-3];
          case 3:
            v3      = a->a + ii[row-2];
          case 2:
            v2      = a->a + ii[row-1];
          case 1:
            v1      = a->a + ii[row];
            break;
          default:
            SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Inode size %D not supported",sizes[i]);
          }
        } else {
          /* upper, no diag */
          sz      = ii[row+1] - diag[row] - 1;
          idx     = a->j + diag[row] + 1;
          switch (sizes[i]) {
          case 5:
            v5      = a->a + diag[row-4] + 5;
          case 4: /* fall through */
            v4      = a->a + diag[row-3] + 4;
          case 3:
            v3      = a->a + diag[row-2] + 3;
          case 2:
            v2      = a->a + diag[row-1] + 2;
          case 1:
            v1      = a->a + diag[row] + 1;
          }
        }
        /* set sum */
        switch (sizes[i]) {
        case 5:
          sum5 = xb[row-4];
        case 4: /* fall through */
          sum4 = xb[row-3];
        case 3:
          sum3 = xb[row-2];
        case 2:
          sum2 = xb[row-1];
        case 1:
          /* note that sum1 is associated with the last row */
          sum1 = xb[row];
        }
        /* do sums */
        for (n = 0; n<sz-1; n+=2) {
            i1    = idx[0];
            i2    = idx[1];
            idx  += 2;
            tmp0  = x[i1];
            tmp1  = x[i2];
            switch (sizes[i]) {
            case 5:
              sum5 -= v5[0] * tmp0 + v5[1] * tmp1; v5 += 2;
            case 4: /* fall through */
              sum4 -= v4[0] * tmp0 + v4[1] * tmp1; v4 += 2;
            case 3:
              sum3 -= v3[0] * tmp0 + v3[1] * tmp1; v3 += 2;
            case 2:
              sum2 -= v2[0] * tmp0 + v2[1] * tmp1; v2 += 2;
            case 1:
              sum1 -= v1[0] * tmp0 + v1[1] * tmp1; v1 += 2;
            }
        }
        /* ragged edge */
        if (n == sz-1) {
          tmp0  = x[*idx];
          switch (sizes[i]) {
          case 5:
            sum5 -= *v5*tmp0;
          case 4: /* fall through */
            sum4 -= *v4*tmp0;
          case 3:
            sum3 -= *v3*tmp0;
          case 2:
            sum2 -= *v2*tmp0;
          case 1:
            sum1 -= *v1*tmp0;
          }
        }
        /* update */
        if (xb == b) {
          /* whole (old way) w/ diag */
          switch (sizes[i]) {
          case 5:
            x[row--] += sum5*ibdiag[4] + sum4*ibdiag[9] + sum3*ibdiag[14] + sum2*ibdiag[19] + sum1*ibdiag[24];
            x[row--] += sum5*ibdiag[3] + sum4*ibdiag[8] + sum3*ibdiag[13] + sum2*ibdiag[18] + sum1*ibdiag[23];
            x[row--] += sum5*ibdiag[2] + sum4*ibdiag[7] + sum3*ibdiag[12] + sum2*ibdiag[17] + sum1*ibdiag[22];
            x[row--] += sum5*ibdiag[1] + sum4*ibdiag[6] + sum3*ibdiag[11] + sum2*ibdiag[16] + sum1*ibdiag[21];
            x[row--] += sum5*ibdiag[0] + sum4*ibdiag[5] + sum3*ibdiag[10] + sum2*ibdiag[15] + sum1*ibdiag[20];
            break;
          case 4:
            x[row--] += sum4*ibdiag[3] + sum3*ibdiag[7] + sum2*ibdiag[11] + sum1*ibdiag[15];
            x[row--] += sum4*ibdiag[2] + sum3*ibdiag[6] + sum2*ibdiag[10] + sum1*ibdiag[14];
            x[row--] += sum4*ibdiag[1] + sum3*ibdiag[5] + sum2*ibdiag[9] + sum1*ibdiag[13];
            x[row--] += sum4*ibdiag[0] + sum3*ibdiag[4] + sum2*ibdiag[8] + sum1*ibdiag[12];
            break;
          case 3:
            x[row--] += sum3*ibdiag[2] + sum2*ibdiag[5] + sum1*ibdiag[8];
            x[row--] += sum3*ibdiag[1] + sum2*ibdiag[4] + sum1*ibdiag[7];
            x[row--] += sum3*ibdiag[0] + sum2*ibdiag[3] + sum1*ibdiag[6];
            break;
          case 2:
            x[row--] += sum2*ibdiag[1] + sum1*ibdiag[3];
            x[row--] += sum2*ibdiag[0] + sum1*ibdiag[2];
            break;
          case 1:
            x[row--] += sum1*(*ibdiag);
            break;
          }
        } else {
          /* no diag so set =  */
          switch (sizes[i]) {
          case 5:
            x[row--] = sum5*ibdiag[4] + sum4*ibdiag[9] + sum3*ibdiag[14] + sum2*ibdiag[19] + sum1*ibdiag[24];
            x[row--] = sum5*ibdiag[3] + sum4*ibdiag[8] + sum3*ibdiag[13] + sum2*ibdiag[18] + sum1*ibdiag[23];
            x[row--] = sum5*ibdiag[2] + sum4*ibdiag[7] + sum3*ibdiag[12] + sum2*ibdiag[17] + sum1*ibdiag[22];
            x[row--] = sum5*ibdiag[1] + sum4*ibdiag[6] + sum3*ibdiag[11] + sum2*ibdiag[16] + sum1*ibdiag[21];
            x[row--] = sum5*ibdiag[0] + sum4*ibdiag[5] + sum3*ibdiag[10] + sum2*ibdiag[15] + sum1*ibdiag[20];
            break;
          case 4:
            x[row--] = sum4*ibdiag[3] + sum3*ibdiag[7] + sum2*ibdiag[11] + sum1*ibdiag[15];
            x[row--] = sum4*ibdiag[2] + sum3*ibdiag[6] + sum2*ibdiag[10] + sum1*ibdiag[14];
            x[row--] = sum4*ibdiag[1] + sum3*ibdiag[5] + sum2*ibdiag[9] + sum1*ibdiag[13];
            x[row--] = sum4*ibdiag[0] + sum3*ibdiag[4] + sum2*ibdiag[8] + sum1*ibdiag[12];
            break;
          case 3:
            x[row--] = sum3*ibdiag[2] + sum2*ibdiag[5] + sum1*ibdiag[8];
            x[row--] = sum3*ibdiag[1] + sum2*ibdiag[4] + sum1*ibdiag[7];
            x[row--] = sum3*ibdiag[0] + sum2*ibdiag[3] + sum1*ibdiag[6];
            break;
          case 2:
            x[row--] = sum2*ibdiag[1] + sum1*ibdiag[3];
            x[row--] = sum2*ibdiag[0] + sum1*ibdiag[2];
            break;
          case 1:
            x[row--] = sum1*(*ibdiag);
            break;
          }
        }
      }
      if (xb == b) {
        ierr = PetscLogFlops(2.0*a->nz);CHKERRQ(ierr);
      } else {
        ierr = PetscLogFlops(a->nz);CHKERRQ(ierr); /* assumes 1/2 in upper, undercounts diag inverse */
      }
    }
  }
  if (flag & SOR_EISENSTAT) {
    /*
          Apply  (U + D)^-1  where D is now the block diagonal
    */
    ibdiag = a->inode.ibdiag+a->inode.bdiagsize;
    for (i=m-1, row=A->rmap->n-1; i>=0; i--) {
      ibdiag -= sizes[i]*sizes[i];
      sz      = ii[row+1] - diag[row] - 1;
      v1      = a->a + diag[row] + 1;
      idx     = a->j + diag[row] + 1;
      /* see comments for MatMult_SeqAIJ_Inode() for how this is coded */
      switch (sizes[i]) {
      case 1:

        sum1 = b[row];
        for (n = 0; n<sz-1; n+=2) {
          i1    = idx[0];
          i2    = idx[1];
          idx  += 2;
          tmp0  = x[i1];
          tmp1  = x[i2];
          sum1 -= v1[0] * tmp0 + v1[1] * tmp1; v1 += 2;
        }

        if (n == sz-1) {
          tmp0  = x[*idx];
          sum1 -= *v1*tmp0;
        }
        x[row] = sum1*(*ibdiag);row--;
        break;

      case 2:

        sum1 = b[row];
        sum2 = b[row-1];
        /* note that sum1 is associated with the second of the two rows */
        v2 = a->a + diag[row-1] + 2;
        for (n = 0; n<sz-1; n+=2) {
          i1    = idx[0];
          i2    = idx[1];
          idx  += 2;
          tmp0  = x[i1];
          tmp1  = x[i2];
          sum1 -= v1[0] * tmp0 + v1[1] * tmp1; v1 += 2;
          sum2 -= v2[0] * tmp0 + v2[1] * tmp1; v2 += 2;
        }

        if (n == sz-1) {
          tmp0  = x[*idx];
          sum1 -= *v1*tmp0;
          sum2 -= *v2*tmp0;
        }
        x[row]   = sum2*ibdiag[1] + sum1*ibdiag[3];
        x[row-1] = sum2*ibdiag[0] + sum1*ibdiag[2];
        row     -= 2;
        break;
      case 3:

        sum1 = b[row];
        sum2 = b[row-1];
        sum3 = b[row-2];
        v2   = a->a + diag[row-1] + 2;
        v3   = a->a + diag[row-2] + 3;
        for (n = 0; n<sz-1; n+=2) {
          i1    = idx[0];
          i2    = idx[1];
          idx  += 2;
          tmp0  = x[i1];
          tmp1  = x[i2];
          sum1 -= v1[0] * tmp0 + v1[1] * tmp1; v1 += 2;
          sum2 -= v2[0] * tmp0 + v2[1] * tmp1; v2 += 2;
          sum3 -= v3[0] * tmp0 + v3[1] * tmp1; v3 += 2;
        }

        if (n == sz-1) {
          tmp0  = x[*idx];
          sum1 -= *v1*tmp0;
          sum2 -= *v2*tmp0;
          sum3 -= *v3*tmp0;
        }
        x[row]   = sum3*ibdiag[2] + sum2*ibdiag[5] + sum1*ibdiag[8];
        x[row-1] = sum3*ibdiag[1] + sum2*ibdiag[4] + sum1*ibdiag[7];
        x[row-2] = sum3*ibdiag[0] + sum2*ibdiag[3] + sum1*ibdiag[6];
        row     -= 3;
        break;
      case 4:

        sum1 = b[row];
        sum2 = b[row-1];
        sum3 = b[row-2];
        sum4 = b[row-3];
        v2   = a->a + diag[row-1] + 2;
        v3   = a->a + diag[row-2] + 3;
        v4   = a->a + diag[row-3] + 4;
        for (n = 0; n<sz-1; n+=2) {
          i1    = idx[0];
          i2    = idx[1];
          idx  += 2;
          tmp0  = x[i1];
          tmp1  = x[i2];
          sum1 -= v1[0] * tmp0 + v1[1] * tmp1; v1 += 2;
          sum2 -= v2[0] * tmp0 + v2[1] * tmp1; v2 += 2;
          sum3 -= v3[0] * tmp0 + v3[1] * tmp1; v3 += 2;
          sum4 -= v4[0] * tmp0 + v4[1] * tmp1; v4 += 2;
        }

        if (n == sz-1) {
          tmp0  = x[*idx];
          sum1 -= *v1*tmp0;
          sum2 -= *v2*tmp0;
          sum3 -= *v3*tmp0;
          sum4 -= *v4*tmp0;
        }
        x[row]   = sum4*ibdiag[3] + sum3*ibdiag[7] + sum2*ibdiag[11] + sum1*ibdiag[15];
        x[row-1] = sum4*ibdiag[2] + sum3*ibdiag[6] + sum2*ibdiag[10] + sum1*ibdiag[14];
        x[row-2] = sum4*ibdiag[1] + sum3*ibdiag[5] + sum2*ibdiag[9] + sum1*ibdiag[13];
        x[row-3] = sum4*ibdiag[0] + sum3*ibdiag[4] + sum2*ibdiag[8] + sum1*ibdiag[12];
        row     -= 4;
        break;
      case 5:

        sum1 = b[row];
        sum2 = b[row-1];
        sum3 = b[row-2];
        sum4 = b[row-3];
        sum5 = b[row-4];
        v2   = a->a + diag[row-1] + 2;
        v3   = a->a + diag[row-2] + 3;
        v4   = a->a + diag[row-3] + 4;
        v5   = a->a + diag[row-4] + 5;
        for (n = 0; n<sz-1; n+=2) {
          i1    = idx[0];
          i2    = idx[1];
          idx  += 2;
          tmp0  = x[i1];
          tmp1  = x[i2];
          sum1 -= v1[0] * tmp0 + v1[1] * tmp1; v1 += 2;
          sum2 -= v2[0] * tmp0 + v2[1] * tmp1; v2 += 2;
          sum3 -= v3[0] * tmp0 + v3[1] * tmp1; v3 += 2;
          sum4 -= v4[0] * tmp0 + v4[1] * tmp1; v4 += 2;
          sum5 -= v5[0] * tmp0 + v5[1] * tmp1; v5 += 2;
        }

        if (n == sz-1) {
          tmp0  = x[*idx];
          sum1 -= *v1*tmp0;
          sum2 -= *v2*tmp0;
          sum3 -= *v3*tmp0;
          sum4 -= *v4*tmp0;
          sum5 -= *v5*tmp0;
        }
        x[row]   = sum5*ibdiag[4] + sum4*ibdiag[9] + sum3*ibdiag[14] + sum2*ibdiag[19] + sum1*ibdiag[24];
        x[row-1] = sum5*ibdiag[3] + sum4*ibdiag[8] + sum3*ibdiag[13] + sum2*ibdiag[18] + sum1*ibdiag[23];
        x[row-2] = sum5*ibdiag[2] + sum4*ibdiag[7] + sum3*ibdiag[12] + sum2*ibdiag[17] + sum1*ibdiag[22];
        x[row-3] = sum5*ibdiag[1] + sum4*ibdiag[6] + sum3*ibdiag[11] + sum2*ibdiag[16] + sum1*ibdiag[21];
        x[row-4] = sum5*ibdiag[0] + sum4*ibdiag[5] + sum3*ibdiag[10] + sum2*ibdiag[15] + sum1*ibdiag[20];
        row     -= 5;
        break;
      default:
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Inode size %D not supported",sizes[i]);
      }
    }
    ierr = PetscLogFlops(a->nz);CHKERRQ(ierr);

    /*
           t = b - D x    where D is the block diagonal
    */
    cnt = 0;
    for (i=0, row=0; i<m; i++) {
      switch (sizes[i]) {
      case 1:
        t[row] = b[row] - bdiag[cnt++]*x[row]; row++;
        break;
      case 2:
        x1       = x[row]; x2 = x[row+1];
        tmp1     = x1*bdiag[cnt] + x2*bdiag[cnt+2];
        tmp2     = x1*bdiag[cnt+1] + x2*bdiag[cnt+3];
        t[row]   = b[row] - tmp1;
        t[row+1] = b[row+1] - tmp2; row += 2;
        cnt     += 4;
        break;
      case 3:
        x1       = x[row]; x2 = x[row+1]; x3 = x[row+2];
        tmp1     = x1*bdiag[cnt] + x2*bdiag[cnt+3] + x3*bdiag[cnt+6];
        tmp2     = x1*bdiag[cnt+1] + x2*bdiag[cnt+4] + x3*bdiag[cnt+7];
        tmp3     = x1*bdiag[cnt+2] + x2*bdiag[cnt+5] + x3*bdiag[cnt+8];
        t[row]   = b[row] - tmp1;
        t[row+1] = b[row+1] - tmp2;
        t[row+2] = b[row+2] - tmp3; row += 3;
        cnt     += 9;
        break;
      case 4:
        x1       = x[row]; x2 = x[row+1]; x3 = x[row+2]; x4 = x[row+3];
        tmp1     = x1*bdiag[cnt] + x2*bdiag[cnt+4] + x3*bdiag[cnt+8] + x4*bdiag[cnt+12];
        tmp2     = x1*bdiag[cnt+1] + x2*bdiag[cnt+5] + x3*bdiag[cnt+9] + x4*bdiag[cnt+13];
        tmp3     = x1*bdiag[cnt+2] + x2*bdiag[cnt+6] + x3*bdiag[cnt+10] + x4*bdiag[cnt+14];
        tmp4     = x1*bdiag[cnt+3] + x2*bdiag[cnt+7] + x3*bdiag[cnt+11] + x4*bdiag[cnt+15];
        t[row]   = b[row] - tmp1;
        t[row+1] = b[row+1] - tmp2;
        t[row+2] = b[row+2] - tmp3;
        t[row+3] = b[row+3] - tmp4; row += 4;
        cnt     += 16;
        break;
      case 5:
        x1       = x[row]; x2 = x[row+1]; x3 = x[row+2]; x4 = x[row+3]; x5 = x[row+4];
        tmp1     = x1*bdiag[cnt] + x2*bdiag[cnt+5] + x3*bdiag[cnt+10] + x4*bdiag[cnt+15] + x5*bdiag[cnt+20];
        tmp2     = x1*bdiag[cnt+1] + x2*bdiag[cnt+6] + x3*bdiag[cnt+11] + x4*bdiag[cnt+16] + x5*bdiag[cnt+21];
        tmp3     = x1*bdiag[cnt+2] + x2*bdiag[cnt+7] + x3*bdiag[cnt+12] + x4*bdiag[cnt+17] + x5*bdiag[cnt+22];
        tmp4     = x1*bdiag[cnt+3] + x2*bdiag[cnt+8] + x3*bdiag[cnt+13] + x4*bdiag[cnt+18] + x5*bdiag[cnt+23];
        tmp5     = x1*bdiag[cnt+4] + x2*bdiag[cnt+9] + x3*bdiag[cnt+14] + x4*bdiag[cnt+19] + x5*bdiag[cnt+24];
        t[row]   = b[row] - tmp1;
        t[row+1] = b[row+1] - tmp2;
        t[row+2] = b[row+2] - tmp3;
        t[row+3] = b[row+3] - tmp4;
        t[row+4] = b[row+4] - tmp5;row += 5;
        cnt     += 25;
        break;
      default:
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Inode size %D not supported",sizes[i]);
      }
    }
    ierr = PetscLogFlops(m);CHKERRQ(ierr);



    /*
          Apply (L + D)^-1 where D is the block diagonal
    */
    for (i=0, row=0; i<m; i++) {
      sz  = diag[row] - ii[row];
      v1  = a->a + ii[row];
      idx = a->j + ii[row];
      /* see comments for MatMult_SeqAIJ_Inode() for how this is coded */
      switch (sizes[i]) {
      case 1:

        sum1 = t[row];
        for (n = 0; n<sz-1; n+=2) {
          i1    = idx[0];
          i2    = idx[1];
          idx  += 2;
          tmp0  = t[i1];
          tmp1  = t[i2];
          sum1 -= v1[0] * tmp0 + v1[1] * tmp1; v1 += 2;
        }

        if (n == sz-1) {
          tmp0  = t[*idx];
          sum1 -= *v1 * tmp0;
        }
        x[row] += t[row] = sum1*(*ibdiag++); row++;
        break;
      case 2:
        v2   = a->a + ii[row+1];
        sum1 = t[row];
        sum2 = t[row+1];
        for (n = 0; n<sz-1; n+=2) {
          i1    = idx[0];
          i2    = idx[1];
          idx  += 2;
          tmp0  = t[i1];
          tmp1  = t[i2];
          sum1 -= v1[0] * tmp0 + v1[1] * tmp1; v1 += 2;
          sum2 -= v2[0] * tmp0 + v2[1] * tmp1; v2 += 2;
        }

        if (n == sz-1) {
          tmp0  = t[*idx];
          sum1 -= v1[0] * tmp0;
          sum2 -= v2[0] * tmp0;
        }
        x[row]   += t[row] = sum1*ibdiag[0] + sum2*ibdiag[2];
        x[row+1] += t[row+1] = sum1*ibdiag[1] + sum2*ibdiag[3];
        ibdiag   += 4; row += 2;
        break;
      case 3:
        v2   = a->a + ii[row+1];
        v3   = a->a + ii[row+2];
        sum1 = t[row];
        sum2 = t[row+1];
        sum3 = t[row+2];
        for (n = 0; n<sz-1; n+=2) {
          i1    = idx[0];
          i2    = idx[1];
          idx  += 2;
          tmp0  = t[i1];
          tmp1  = t[i2];
          sum1 -= v1[0] * tmp0 + v1[1] * tmp1; v1 += 2;
          sum2 -= v2[0] * tmp0 + v2[1] * tmp1; v2 += 2;
          sum3 -= v3[0] * tmp0 + v3[1] * tmp1; v3 += 2;
        }

        if (n == sz-1) {
          tmp0  = t[*idx];
          sum1 -= v1[0] * tmp0;
          sum2 -= v2[0] * tmp0;
          sum3 -= v3[0] * tmp0;
        }
        x[row]   += t[row] = sum1*ibdiag[0] + sum2*ibdiag[3] + sum3*ibdiag[6];
        x[row+1] += t[row+1] = sum1*ibdiag[1] + sum2*ibdiag[4] + sum3*ibdiag[7];
        x[row+2] += t[row+2] = sum1*ibdiag[2] + sum2*ibdiag[5] + sum3*ibdiag[8];
        ibdiag   += 9; row += 3;
        break;
      case 4:
        v2   = a->a + ii[row+1];
        v3   = a->a + ii[row+2];
        v4   = a->a + ii[row+3];
        sum1 = t[row];
        sum2 = t[row+1];
        sum3 = t[row+2];
        sum4 = t[row+3];
        for (n = 0; n<sz-1; n+=2) {
          i1    = idx[0];
          i2    = idx[1];
          idx  += 2;
          tmp0  = t[i1];
          tmp1  = t[i2];
          sum1 -= v1[0] * tmp0 + v1[1] * tmp1; v1 += 2;
          sum2 -= v2[0] * tmp0 + v2[1] * tmp1; v2 += 2;
          sum3 -= v3[0] * tmp0 + v3[1] * tmp1; v3 += 2;
          sum4 -= v4[0] * tmp0 + v4[1] * tmp1; v4 += 2;
        }

        if (n == sz-1) {
          tmp0  = t[*idx];
          sum1 -= v1[0] * tmp0;
          sum2 -= v2[0] * tmp0;
          sum3 -= v3[0] * tmp0;
          sum4 -= v4[0] * tmp0;
        }
        x[row]   += t[row] = sum1*ibdiag[0] + sum2*ibdiag[4] + sum3*ibdiag[8] + sum4*ibdiag[12];
        x[row+1] += t[row+1] = sum1*ibdiag[1] + sum2*ibdiag[5] + sum3*ibdiag[9] + sum4*ibdiag[13];
        x[row+2] += t[row+2] = sum1*ibdiag[2] + sum2*ibdiag[6] + sum3*ibdiag[10] + sum4*ibdiag[14];
        x[row+3] += t[row+3] = sum1*ibdiag[3] + sum2*ibdiag[7] + sum3*ibdiag[11] + sum4*ibdiag[15];
        ibdiag   += 16; row += 4;
        break;
      case 5:
        v2   = a->a + ii[row+1];
        v3   = a->a + ii[row+2];
        v4   = a->a + ii[row+3];
        v5   = a->a + ii[row+4];
        sum1 = t[row];
        sum2 = t[row+1];
        sum3 = t[row+2];
        sum4 = t[row+3];
        sum5 = t[row+4];
        for (n = 0; n<sz-1; n+=2) {
          i1    = idx[0];
          i2    = idx[1];
          idx  += 2;
          tmp0  = t[i1];
          tmp1  = t[i2];
          sum1 -= v1[0] * tmp0 + v1[1] * tmp1; v1 += 2;
          sum2 -= v2[0] * tmp0 + v2[1] * tmp1; v2 += 2;
          sum3 -= v3[0] * tmp0 + v3[1] * tmp1; v3 += 2;
          sum4 -= v4[0] * tmp0 + v4[1] * tmp1; v4 += 2;
          sum5 -= v5[0] * tmp0 + v5[1] * tmp1; v5 += 2;
        }

        if (n == sz-1) {
          tmp0  = t[*idx];
          sum1 -= v1[0] * tmp0;
          sum2 -= v2[0] * tmp0;
          sum3 -= v3[0] * tmp0;
          sum4 -= v4[0] * tmp0;
          sum5 -= v5[0] * tmp0;
        }
        x[row]   += t[row] = sum1*ibdiag[0] + sum2*ibdiag[5] + sum3*ibdiag[10] + sum4*ibdiag[15] + sum5*ibdiag[20];
        x[row+1] += t[row+1] = sum1*ibdiag[1] + sum2*ibdiag[6] + sum3*ibdiag[11] + sum4*ibdiag[16] + sum5*ibdiag[21];
        x[row+2] += t[row+2] = sum1*ibdiag[2] + sum2*ibdiag[7] + sum3*ibdiag[12] + sum4*ibdiag[17] + sum5*ibdiag[22];
        x[row+3] += t[row+3] = sum1*ibdiag[3] + sum2*ibdiag[8] + sum3*ibdiag[13] + sum4*ibdiag[18] + sum5*ibdiag[23];
        x[row+4] += t[row+4] = sum1*ibdiag[4] + sum2*ibdiag[9] + sum3*ibdiag[14] + sum4*ibdiag[19] + sum5*ibdiag[24];
        ibdiag   += 25; row += 5;
        break;
      default:
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Inode size %D not supported",sizes[i]);
      }
    }
    ierr = PetscLogFlops(a->nz);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultDiagonalBlock_SeqAIJ_Inode(Mat A,Vec bb,Vec xx)
{
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)A->data;
  PetscScalar       *x,tmp1,tmp2,tmp3,tmp4,tmp5,x1,x2,x3,x4,x5;
  const MatScalar   *bdiag = a->inode.bdiag;
  const PetscScalar *b;
  PetscErrorCode    ierr;
  PetscInt          m      = a->inode.node_count,cnt = 0,i,row;
  const PetscInt    *sizes = a->inode.size;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
  cnt  = 0;
  for (i=0, row=0; i<m; i++) {
    switch (sizes[i]) {
    case 1:
      x[row] = b[row]*bdiag[cnt++];row++;
      break;
    case 2:
      x1       = b[row]; x2 = b[row+1];
      tmp1     = x1*bdiag[cnt] + x2*bdiag[cnt+2];
      tmp2     = x1*bdiag[cnt+1] + x2*bdiag[cnt+3];
      x[row++] = tmp1;
      x[row++] = tmp2;
      cnt     += 4;
      break;
    case 3:
      x1       = b[row]; x2 = b[row+1]; x3 = b[row+2];
      tmp1     = x1*bdiag[cnt] + x2*bdiag[cnt+3] + x3*bdiag[cnt+6];
      tmp2     = x1*bdiag[cnt+1] + x2*bdiag[cnt+4] + x3*bdiag[cnt+7];
      tmp3     = x1*bdiag[cnt+2] + x2*bdiag[cnt+5] + x3*bdiag[cnt+8];
      x[row++] = tmp1;
      x[row++] = tmp2;
      x[row++] = tmp3;
      cnt     += 9;
      break;
    case 4:
      x1       = b[row]; x2 = b[row+1]; x3 = b[row+2]; x4 = b[row+3];
      tmp1     = x1*bdiag[cnt] + x2*bdiag[cnt+4] + x3*bdiag[cnt+8] + x4*bdiag[cnt+12];
      tmp2     = x1*bdiag[cnt+1] + x2*bdiag[cnt+5] + x3*bdiag[cnt+9] + x4*bdiag[cnt+13];
      tmp3     = x1*bdiag[cnt+2] + x2*bdiag[cnt+6] + x3*bdiag[cnt+10] + x4*bdiag[cnt+14];
      tmp4     = x1*bdiag[cnt+3] + x2*bdiag[cnt+7] + x3*bdiag[cnt+11] + x4*bdiag[cnt+15];
      x[row++] = tmp1;
      x[row++] = tmp2;
      x[row++] = tmp3;
      x[row++] = tmp4;
      cnt     += 16;
      break;
    case 5:
      x1       = b[row]; x2 = b[row+1]; x3 = b[row+2]; x4 = b[row+3]; x5 = b[row+4];
      tmp1     = x1*bdiag[cnt] + x2*bdiag[cnt+5] + x3*bdiag[cnt+10] + x4*bdiag[cnt+15] + x5*bdiag[cnt+20];
      tmp2     = x1*bdiag[cnt+1] + x2*bdiag[cnt+6] + x3*bdiag[cnt+11] + x4*bdiag[cnt+16] + x5*bdiag[cnt+21];
      tmp3     = x1*bdiag[cnt+2] + x2*bdiag[cnt+7] + x3*bdiag[cnt+12] + x4*bdiag[cnt+17] + x5*bdiag[cnt+22];
      tmp4     = x1*bdiag[cnt+3] + x2*bdiag[cnt+8] + x3*bdiag[cnt+13] + x4*bdiag[cnt+18] + x5*bdiag[cnt+23];
      tmp5     = x1*bdiag[cnt+4] + x2*bdiag[cnt+9] + x3*bdiag[cnt+14] + x4*bdiag[cnt+19] + x5*bdiag[cnt+24];
      x[row++] = tmp1;
      x[row++] = tmp2;
      x[row++] = tmp3;
      x[row++] = tmp4;
      x[row++] = tmp5;
      cnt     += 25;
      break;
    default:
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Inode size %D not supported",sizes[i]);
    }
  }
  ierr = PetscLogFlops(2.0*cnt);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
    samestructure indicates that the matrix has not changed its nonzero structure so we
    do not need to recompute the inodes
*/
PetscErrorCode MatSeqAIJCheckInode(Mat A)
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       i,j,m,nzx,nzy,*ns,node_count,blk_size;
  PetscBool      flag;
  const PetscInt *idx,*idy,*ii;

  PetscFunctionBegin;
  if (!a->inode.use) PetscFunctionReturn(0);
  if (a->inode.checked && A->nonzerostate == a->inode.mat_nonzerostate) PetscFunctionReturn(0);

  m = A->rmap->n;
  if (a->inode.size) ns = a->inode.size;
  else {
    ierr = PetscMalloc1(m+1,&ns);CHKERRQ(ierr);
  }

  i          = 0;
  node_count = 0;
  idx        = a->j;
  ii         = a->i;
  while (i < m) {                /* For each row */
    nzx = ii[i+1] - ii[i];       /* Number of nonzeros */
    /* Limits the number of elements in a node to 'a->inode.limit' */
    for (j=i+1,idy=idx,blk_size=1; j<m && blk_size <a->inode.limit; ++j,++blk_size) {
      nzy = ii[j+1] - ii[j];     /* Same number of nonzeros */
      if (nzy != nzx) break;
      idy += nzx;              /* Same nonzero pattern */
      ierr = PetscArraycmp(idx,idy,nzx,&flag);CHKERRQ(ierr);
      if (!flag) break;
    }
    ns[node_count++] = blk_size;
    idx             += blk_size*nzx;
    i                = j;
  }

  {
    PetscBool is_cudatype;
    ierr = PetscObjectTypeCompareAny((PetscObject)A,&is_cudatype, MATAIJCUSPARSE, MATSEQAIJCUSPARSE, MATMPIAIJCUSPARSE, MATAIJVIENNACL, MATSEQAIJVIENNACL, MATMPIAIJVIENNACL,"");CHKERRQ(ierr);
    if (is_cudatype) {
      ierr = PetscInfo(A,"Not using Inode routines on GPU matrix\n");CHKERRQ(ierr);
      ierr = PetscFree(ns);CHKERRQ(ierr);
      a->inode.node_count       = 0;
      a->inode.size             = NULL;
      a->inode.use              = PETSC_FALSE;
      a->inode.checked          = PETSC_TRUE;
      a->inode.mat_nonzerostate = A->nonzerostate;
      PetscFunctionReturn(0);
    }
  }

  /* If not enough inodes found,, do not use inode version of the routines */
  if (!m || node_count > .8*m) {
    ierr = PetscFree(ns);CHKERRQ(ierr);

    a->inode.node_count       = 0;
    a->inode.size             = NULL;
    a->inode.use              = PETSC_FALSE;
    A->ops->mult              = MatMult_SeqAIJ;
    A->ops->sor               = MatSOR_SeqAIJ;
    A->ops->multadd           = MatMultAdd_SeqAIJ;
    A->ops->getrowij          = MatGetRowIJ_SeqAIJ;
    A->ops->restorerowij      = MatRestoreRowIJ_SeqAIJ;
    A->ops->getcolumnij       = MatGetColumnIJ_SeqAIJ;
    A->ops->restorecolumnij   = MatRestoreColumnIJ_SeqAIJ;
    A->ops->coloringpatch     = NULL;
    A->ops->multdiagonalblock = NULL;

    ierr = PetscInfo2(A,"Found %D nodes out of %D rows. Not using Inode routines\n",node_count,m);CHKERRQ(ierr);
  } else {
    if (!A->factortype) {
      A->ops->mult              = MatMult_SeqAIJ_Inode;
      A->ops->sor               = MatSOR_SeqAIJ_Inode;
      A->ops->multadd           = MatMultAdd_SeqAIJ_Inode;
      A->ops->multdiagonalblock = MatMultDiagonalBlock_SeqAIJ_Inode;
      if (A->rmap->n == A->cmap->n) {
        A->ops->getrowij          = MatGetRowIJ_SeqAIJ_Inode;
        A->ops->restorerowij      = MatRestoreRowIJ_SeqAIJ_Inode;
        A->ops->getcolumnij       = MatGetColumnIJ_SeqAIJ_Inode;
        A->ops->restorecolumnij   = MatRestoreColumnIJ_SeqAIJ_Inode;
        A->ops->coloringpatch     = MatColoringPatch_SeqAIJ_Inode;
      }
    } else {
      A->ops->solve = MatSolve_SeqAIJ_Inode_inplace;
    }
    a->inode.node_count = node_count;
    a->inode.size       = ns;
    ierr = PetscInfo3(A,"Found %D nodes of %D. Limit used: %D. Using Inode routines\n",node_count,m,a->inode.limit);CHKERRQ(ierr);
  }
  a->inode.checked          = PETSC_TRUE;
  a->inode.mat_nonzerostate = A->nonzerostate;
  PetscFunctionReturn(0);
}

PetscErrorCode MatDuplicate_SeqAIJ_Inode(Mat A,MatDuplicateOption cpvalues,Mat *C)
{
  Mat            B =*C;
  Mat_SeqAIJ     *c=(Mat_SeqAIJ*)B->data,*a=(Mat_SeqAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       m=A->rmap->n;

  PetscFunctionBegin;
  c->inode.use       = a->inode.use;
  c->inode.limit     = a->inode.limit;
  c->inode.max_limit = a->inode.max_limit;
  if (a->inode.size) {
    ierr                = PetscMalloc1(m+1,&c->inode.size);CHKERRQ(ierr);
    c->inode.node_count = a->inode.node_count;
    ierr                = PetscArraycpy(c->inode.size,a->inode.size,m+1);CHKERRQ(ierr);
    /* note the table of functions below should match that in MatSeqAIJCheckInode() */
    if (!B->factortype) {
      B->ops->mult              = MatMult_SeqAIJ_Inode;
      B->ops->sor               = MatSOR_SeqAIJ_Inode;
      B->ops->multadd           = MatMultAdd_SeqAIJ_Inode;
      B->ops->getrowij          = MatGetRowIJ_SeqAIJ_Inode;
      B->ops->restorerowij      = MatRestoreRowIJ_SeqAIJ_Inode;
      B->ops->getcolumnij       = MatGetColumnIJ_SeqAIJ_Inode;
      B->ops->restorecolumnij   = MatRestoreColumnIJ_SeqAIJ_Inode;
      B->ops->coloringpatch     = MatColoringPatch_SeqAIJ_Inode;
      B->ops->multdiagonalblock = MatMultDiagonalBlock_SeqAIJ_Inode;
    } else {
      B->ops->solve = MatSolve_SeqAIJ_Inode_inplace;
    }
  } else {
    c->inode.size       = NULL;
    c->inode.node_count = 0;
  }
  c->inode.ibdiagvalid = PETSC_FALSE;
  c->inode.ibdiag      = NULL;
  c->inode.bdiag       = NULL;
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode MatGetRow_FactoredLU(PetscInt *cols,PetscInt nzl,PetscInt nzu,PetscInt nz,const PetscInt *ai,const PetscInt *aj,const PetscInt *adiag,PetscInt row)
{
  PetscInt       k;
  const PetscInt *vi;

  PetscFunctionBegin;
  vi = aj + ai[row];
  for (k=0; k<nzl; k++) cols[k] = vi[k];
  vi        = aj + adiag[row];
  cols[nzl] = vi[0];
  vi        = aj + adiag[row+1]+1;
  for (k=0; k<nzu; k++) cols[nzl+1+k] = vi[k];
  PetscFunctionReturn(0);
}
/*
   MatSeqAIJCheckInode_FactorLU - Check Inode for factored seqaij matrix.
   Modified from MatSeqAIJCheckInode().

   Input Parameters:
.  Mat A - ILU or LU matrix factor

*/
PetscErrorCode MatSeqAIJCheckInode_FactorLU(Mat A)
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       i,j,m,nzl1,nzu1,nzl2,nzu2,nzx,nzy,node_count,blk_size;
  PetscInt       *cols1,*cols2,*ns;
  const PetscInt *ai = a->i,*aj = a->j, *adiag = a->diag;
  PetscBool      flag;

  PetscFunctionBegin;
  if (!a->inode.use)    PetscFunctionReturn(0);
  if (a->inode.checked) PetscFunctionReturn(0);

  m = A->rmap->n;
  if (a->inode.size) ns = a->inode.size;
  else {
    ierr = PetscMalloc1(m+1,&ns);CHKERRQ(ierr);
  }

  i          = 0;
  node_count = 0;
  ierr = PetscMalloc2(m,&cols1,m,&cols2);CHKERRQ(ierr);
  while (i < m) {                /* For each row */
    nzl1 = ai[i+1] - ai[i];       /* Number of nonzeros in L */
    nzu1 = adiag[i] - adiag[i+1] - 1; /* Number of nonzeros in U excluding diagonal*/
    nzx  = nzl1 + nzu1 + 1;
    MatGetRow_FactoredLU(cols1,nzl1,nzu1,nzx,ai,aj,adiag,i);

    /* Limits the number of elements in a node to 'a->inode.limit' */
    for (j=i+1,blk_size=1; j<m && blk_size <a->inode.limit; ++j,++blk_size) {
      nzl2 = ai[j+1] - ai[j];
      nzu2 = adiag[j] - adiag[j+1] - 1;
      nzy  = nzl2 + nzu2 + 1;
      if (nzy != nzx) break;
      ierr = MatGetRow_FactoredLU(cols2,nzl2,nzu2,nzy,ai,aj,adiag,j);CHKERRQ(ierr);
      ierr = PetscArraycmp(cols1,cols2,nzx,&flag);CHKERRQ(ierr);
      if (!flag) break;
    }
    ns[node_count++] = blk_size;
    i                = j;
  }
  ierr             = PetscFree2(cols1,cols2);CHKERRQ(ierr);
  /* If not enough inodes found,, do not use inode version of the routines */
  if (!m || node_count > .8*m) {
    ierr = PetscFree(ns);CHKERRQ(ierr);

    a->inode.node_count = 0;
    a->inode.size       = NULL;
    a->inode.use        = PETSC_FALSE;

    ierr = PetscInfo2(A,"Found %D nodes out of %D rows. Not using Inode routines\n",node_count,m);CHKERRQ(ierr);
  } else {
    A->ops->mult              = NULL;
    A->ops->sor               = NULL;
    A->ops->multadd           = NULL;
    A->ops->getrowij          = NULL;
    A->ops->restorerowij      = NULL;
    A->ops->getcolumnij       = NULL;
    A->ops->restorecolumnij   = NULL;
    A->ops->coloringpatch     = NULL;
    A->ops->multdiagonalblock = NULL;
    a->inode.node_count       = node_count;
    a->inode.size             = ns;

    ierr = PetscInfo3(A,"Found %D nodes of %D. Limit used: %D. Using Inode routines\n",node_count,m,a->inode.limit);CHKERRQ(ierr);
  }
  a->inode.checked = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode MatSeqAIJInvalidateDiagonal_Inode(Mat A)
{
  Mat_SeqAIJ *a=(Mat_SeqAIJ*)A->data;

  PetscFunctionBegin;
  a->inode.ibdiagvalid = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*
     This is really ugly. if inodes are used this replaces the
  permutations with ones that correspond to rows/cols of the matrix
  rather then inode blocks
*/
PetscErrorCode  MatInodeAdjustForInodes(Mat A,IS *rperm,IS *cperm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTryMethod(A,"MatInodeAdjustForInodes_C",(Mat,IS*,IS*),(A,rperm,cperm));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode  MatInodeAdjustForInodes_SeqAIJ_Inode(Mat A,IS *rperm,IS *cperm)
{
  Mat_SeqAIJ     *a=(Mat_SeqAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       m = A->rmap->n,n = A->cmap->n,i,j,nslim_row = a->inode.node_count;
  const PetscInt *ridx,*cidx;
  PetscInt       row,col,*permr,*permc,*ns_row =  a->inode.size,*tns,start_val,end_val,indx;
  PetscInt       nslim_col,*ns_col;
  IS             ris = *rperm,cis = *cperm;

  PetscFunctionBegin;
  if (!a->inode.size) PetscFunctionReturn(0); /* no inodes so return */
  if (a->inode.node_count == m) PetscFunctionReturn(0); /* all inodes are of size 1 */

  ierr = MatCreateColInode_Private(A,&nslim_col,&ns_col);CHKERRQ(ierr);
  ierr = PetscMalloc1(((nslim_row>nslim_col) ? nslim_row : nslim_col)+1,&tns);CHKERRQ(ierr);
  ierr = PetscMalloc2(m,&permr,n,&permc);CHKERRQ(ierr);

  ierr = ISGetIndices(ris,&ridx);CHKERRQ(ierr);
  ierr = ISGetIndices(cis,&cidx);CHKERRQ(ierr);

  /* Form the inode structure for the rows of permuted matric using inv perm*/
  for (i=0,tns[0]=0; i<nslim_row; ++i) tns[i+1] = tns[i] + ns_row[i];

  /* Construct the permutations for rows*/
  for (i=0,row = 0; i<nslim_row; ++i) {
    indx      = ridx[i];
    start_val = tns[indx];
    end_val   = tns[indx + 1];
    for (j=start_val; j<end_val; ++j,++row) permr[row]= j;
  }

  /* Form the inode structure for the columns of permuted matrix using inv perm*/
  for (i=0,tns[0]=0; i<nslim_col; ++i) tns[i+1] = tns[i] + ns_col[i];

  /* Construct permutations for columns */
  for (i=0,col=0; i<nslim_col; ++i) {
    indx      = cidx[i];
    start_val = tns[indx];
    end_val   = tns[indx + 1];
    for (j = start_val; j<end_val; ++j,++col) permc[col]= j;
  }

  ierr = ISCreateGeneral(PETSC_COMM_SELF,n,permr,PETSC_COPY_VALUES,rperm);CHKERRQ(ierr);
  ierr = ISSetPermutation(*rperm);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF,n,permc,PETSC_COPY_VALUES,cperm);CHKERRQ(ierr);
  ierr = ISSetPermutation(*cperm);CHKERRQ(ierr);

  ierr = ISRestoreIndices(ris,&ridx);CHKERRQ(ierr);
  ierr = ISRestoreIndices(cis,&cidx);CHKERRQ(ierr);

  ierr = PetscFree(ns_col);CHKERRQ(ierr);
  ierr = PetscFree2(permr,permc);CHKERRQ(ierr);
  ierr = ISDestroy(&cis);CHKERRQ(ierr);
  ierr = ISDestroy(&ris);CHKERRQ(ierr);
  ierr = PetscFree(tns);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   MatInodeGetInodeSizes - Returns the inode information of the Inode matrix.

   Not Collective

   Input Parameter:
.  A - the Inode matrix or matrix derived from the Inode class -- e.g., SeqAIJ

   Output Parameter:
+  node_count - no of inodes present in the matrix.
.  sizes      - an array of size node_count,with sizes of each inode.
-  limit      - the max size used to generate the inodes.

   Level: advanced

   Notes:
    This routine returns some internal storage information
   of the matrix, it is intended to be used by advanced users.
   It should be called after the matrix is assembled.
   The contents of the sizes[] array should not be changed.
   NULL may be passed for information not requested.

.seealso: MatGetInfo()
@*/
PetscErrorCode  MatInodeGetInodeSizes(Mat A,PetscInt *node_count,PetscInt *sizes[],PetscInt *limit)
{
  PetscErrorCode ierr,(*f)(Mat,PetscInt*,PetscInt*[],PetscInt*);

  PetscFunctionBegin;
  if (!A->assembled) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  ierr = PetscObjectQueryFunction((PetscObject)A,"MatInodeGetInodeSizes_C",&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(A,node_count,sizes,limit);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode  MatInodeGetInodeSizes_SeqAIJ_Inode(Mat A,PetscInt *node_count,PetscInt *sizes[],PetscInt *limit)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ*)A->data;

  PetscFunctionBegin;
  if (node_count) *node_count = a->inode.node_count;
  if (sizes)      *sizes      = a->inode.size;
  if (limit)      *limit      = a->inode.limit;
  PetscFunctionReturn(0);
}
