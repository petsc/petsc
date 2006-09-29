#define PETSCMAT_DLL

/*
   This provides a matrix that consists of Mats
*/

#include "src/mat/matimpl.h"              /*I "petscmat.h" I*/
#include "src/mat/impls/baij/seq/baij.h"    /* use the common AIJ data-structure */

#define CHUNKSIZE   15

typedef struct {
  SEQAIJHEADER(Mat);
  SEQBAIJHEADER;

  Vec  left,right;   /* dummy vectors to perform local parts of product */
} Mat_BlockMat;      

#undef __FUNCT__  
#define __FUNCT__ "MatSetValues_BlockMat"
PetscErrorCode MatSetValues_BlockMat(Mat A,PetscInt m,const PetscInt im[],PetscInt n,const PetscInt in[],const PetscScalar v[],InsertMode is)
{
  Mat_BlockMat   *a = (Mat_BlockMat*)A->data;
  PetscInt       *rp,k,low,high,t,ii,row,nrow,i,col,l,rmax,N,lastcol = -1;
  PetscInt       *imax=a->imax,*ai=a->i,*ailen=a->ilen;
  PetscInt       *aj=a->j,nonew=a->nonew,bs=A->rmap.bs,brow,bcol;
  PetscErrorCode ierr;
  PetscInt       ridx,cidx;
  PetscTruth     roworiented=a->roworiented;
  MatScalar      value;
  Mat            *ap,*aa = a->a;

  PetscFunctionBegin;
  for (k=0; k<m; k++) { /* loop over added rows */
    row  = im[k];
    brow = row/bs;  
    if (row < 0) continue;
#if defined(PETSC_USE_DEBUG)  
    if (row >= A->rmap.N) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %D max %D",row,A->rmap.N-1);
#endif
    rp   = aj + ai[brow]; 
    ap   = aa + ai[brow];
    rmax = imax[brow]; 
    nrow = ailen[brow]; 
    low  = 0;
    high = nrow;
    for (l=0; l<n; l++) { /* loop over added columns */
      if (in[l] < 0) continue;
#if defined(PETSC_USE_DEBUG)  
      if (in[l] >= A->cmap.n) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Column too large: col %D max %D",in[l],A->cmap.n-1);
#endif
      col = in[l]; bcol = col/bs;
      ridx = row % bs; cidx = col % bs;
      if (roworiented) {
        value = v[l + k*n]; 
      } else {
        value = v[k + l*m];
      }
      if (col <= lastcol) low = 0; else high = nrow;
      lastcol = col;
      while (high-low > 7) {
        t = (low+high)/2;
        if (rp[t] > bcol) high = t;
        else              low  = t;
      }
      for (i=low; i<high; i++) {
        if (rp[i] > bcol) break;
        if (rp[i] == bcol) {
          goto noinsert1;
        }
      } 
      if (nonew == 1) goto noinsert1;
      if (nonew == -1) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new nonzero (%D, %D) in the matrix", row, col);
      MatSeqXAIJReallocateAIJ(A,a->mbs,1,nrow,brow,bcol,rmax,aa,ai,aj,rp,ap,imax,nonew,Mat);
      N = nrow++ - 1; high++;
      /* shift up all the later entries in this row */
      for (ii=N; ii>=i; ii--) {
        rp[ii+1] = rp[ii];
        ap[ii+1] = ap[ii];
      }
      if (N>=i) ap[i] = 0;
      rp[i]           = bcol; 
      a->nz++;
      noinsert1:;
      if (!*(ap+i)) {
        ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,bs,bs,0,0,ap+i);CHKERRQ(ierr);
      }
      ierr = MatSetValues(ap[i],1,&ridx,1,&cidx,&value,is);CHKERRQ(ierr);
      low = i;
    }
    ailen[brow] = nrow;
  }
  A->same_nonzero = PETSC_FALSE;
  PetscFunctionReturn(0);
} 

#undef __FUNCT__  
#define __FUNCT__ "MatLoad_BlockMat"
PetscErrorCode MatLoad_BlockMat(PetscViewer viewer, MatType type,Mat *A)
{
  PetscErrorCode    ierr;
  Mat               tmpA;
  PetscInt          i,m,n,bs = 1,ncols;
  const PetscInt    *cols;
  const PetscScalar *values;

  PetscFunctionBegin;
  ierr = MatLoad_SeqAIJ(viewer,MATSEQAIJ,&tmpA);CHKERRQ(ierr);

  ierr = MatGetLocalSize(tmpA,&m,&n);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(PETSC_COMM_SELF,PETSC_NULL,"Options for loading BlockMat matrix","Mat");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-matload_block_size","Set the blocksize used to store the matrix","MatLoad",bs,&bs,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  ierr = MatCreateBlockMat(PETSC_COMM_SELF,m,n,bs,A);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    ierr = MatGetRow(tmpA,i,&ncols,&cols,&values);CHKERRQ(ierr);
    ierr = MatSetValues(*A,1,&i,ncols,cols,values,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_BlockMat"
PetscErrorCode MatDestroy_BlockMat(Mat mat)
{
  PetscErrorCode ierr;
  Mat_BlockMat   *bmat = (Mat_BlockMat*)mat->data;

  PetscFunctionBegin;
  if (bmat->right) {
    ierr = VecDestroy(bmat->right);CHKERRQ(ierr);
  }
  if (bmat->left) {
    ierr = VecDestroy(bmat->left);CHKERRQ(ierr);
  }
  ierr = PetscFree(bmat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMult_BlockMat"
PetscErrorCode MatMult_BlockMat(Mat A,Vec x,Vec y)
{
  Mat_BlockMat   *bmat = (Mat_BlockMat*)A->data;  
  PetscErrorCode ierr;
  PetscScalar    *xx,*yy;
  PetscInt       *aj,i,*ii,jrow,m = A->rmap.n/A->rmap.bs,bs = A->rmap.bs,n,j;
  Mat            *aa;

  PetscFunctionBegin;
  /*
     Standard CSR multiply except each entry is a Mat
  */
  ierr = VecGetArray(x,&xx);CHKERRQ(ierr);
  ierr = VecSet(y,0.0);CHKERRQ(ierr);
  ierr = VecGetArray(y,&yy);CHKERRQ(ierr);
  aj  = bmat->j;
  aa  = bmat->a;
  ii  = bmat->i;
  for (i=0; i<m; i++) {
    jrow = ii[i];
    ierr = VecPlaceArray(bmat->left,yy + bs*jrow);CHKERRQ(ierr);
    n    = ii[i+1] - jrow;
    ierr = VecSet(bmat->left,0.0);CHKERRQ(ierr);
    for (j=0; j<n; j++) {
      ierr = VecPlaceArray(bmat->right,xx + bs*aj[jrow]);CHKERRQ(ierr);
      ierr = MatMultAdd(aa[jrow],bmat->right,bmat->left,bmat->left);
      ierr = VecResetArray(bmat->right);CHKERRQ(ierr);
      jrow++;
    }
    ierr = VecResetArray(bmat->left);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&yy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_BlockMat"
PetscErrorCode MatMultAdd_BlockMat(Mat A,Vec x,Vec y,Vec z)
{
  Mat_BlockMat   *bmat = (Mat_BlockMat*)A->data;  
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTranspose_BlockMat"
PetscErrorCode MatMultTranspose_BlockMat(Mat A,Vec x,Vec y)
{
  Mat_BlockMat   *bmat = (Mat_BlockMat*)A->data;  
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTransposeAdd_BlockMat"
PetscErrorCode MatMultTransposeAdd_BlockMat(Mat A,Vec x,Vec y,Vec z)
{
  Mat_BlockMat   *bmat = (Mat_BlockMat*)A->data;  
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetBlockSize_BlockMat"
PetscErrorCode MatSetBlockSize_BlockMat(Mat A,PetscInt bs)
{
  Mat_BlockMat   *bmat = (Mat_BlockMat*)A->data;  
  PetscErrorCode ierr;
  PetscInt       nz = 10,i,m;

  PetscFunctionBegin;
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,bs,PETSC_NULL,&bmat->right);CHKERRQ(ierr);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,bs,PETSC_NULL,&bmat->left);CHKERRQ(ierr);

  A->rmap.bs = A->cmap.bs = bs;

  ierr = PetscMalloc2(A->rmap.n,PetscInt,&bmat->imax,A->rmap.n,PetscInt,&bmat->ilen);CHKERRQ(ierr);
  for (i=0; i<A->rmap.n; i++) bmat->imax[i] = nz;
  nz = nz*A->rmap.n;


  /* bmat->ilen will count nonzeros in each row so far. */
  for (i=0; i<A->rmap.n; i++) { bmat->ilen[i] = 0;}

  /* allocate the matrix space */
  ierr = PetscMalloc3(nz,Mat,&bmat->a,nz,PetscInt,&bmat->j,A->rmap.n+1,PetscInt,&bmat->i);CHKERRQ(ierr);
  bmat->i[0] = 0;
  for (i=1; i<A->rmap.n+1; i++) {
    bmat->i[i] = bmat->i[i-1] + bmat->imax[i-1];
  }
  bmat->singlemalloc = PETSC_TRUE;
  bmat->free_a       = PETSC_TRUE;
  bmat->free_ij      = PETSC_TRUE;

  bmat->nz                = 0;
  bmat->maxnz             = nz;
  A->info.nz_unneeded  = (double)bmat->maxnz;

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatAssemblyEnd_BlockMat"
PetscErrorCode MatAssemblyEnd_BlockMat(Mat A,MatAssemblyType mode)
{
  Mat_BlockMat   *a = (Mat_BlockMat*)A->data;
  PetscErrorCode ierr;
  PetscInt       fshift = 0,i,j,*ai = a->i,*aj = a->j,*imax = a->imax;
  PetscInt       m = A->rmap.n/A->rmap.bs,*ip,N,*ailen = a->ilen,rmax = 0;
  Mat            *aa = a->a,*ap;
  PetscReal      ratio=0.6;

  PetscFunctionBegin;  
  if (mode == MAT_FLUSH_ASSEMBLY) PetscFunctionReturn(0);

  if (m) rmax = ailen[0]; /* determine row with most nonzeros */
  for (i=1; i<m; i++) {
    /* move each row back by the amount of empty slots (fshift) before it*/
    fshift += imax[i-1] - ailen[i-1];
    rmax   = PetscMax(rmax,ailen[i]);
    if (fshift) {
      ip = aj + ai[i] ; 
      ap = aa + ai[i] ;
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
    ai[m]  = ai[m-1] + ailen[m-1];
  }
  /* reset ilen and imax for each row */
  for (i=0; i<m; i++) {
    ailen[i] = imax[i] = ai[i+1] - ai[i];
  }
  a->nz = ai[m]; 

  ierr = PetscInfo4(A,"Matrix size: %D X %D; storage space: %D unneeded,%D used\n",m,A->cmap.n,fshift,a->nz);CHKERRQ(ierr);
  ierr = PetscInfo1(A,"Number of mallocs during MatSetValues() is %D\n",a->reallocs);CHKERRQ(ierr);
  ierr = PetscInfo1(A,"Maximum nonzeros in any row is %D\n",rmax);CHKERRQ(ierr);
  a->reallocs          = 0;
  A->info.nz_unneeded  = (double)fshift;
  a->rmax              = rmax;

  A->same_nonzero = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static struct _MatOps MatOps_Values = {MatSetValues_BlockMat,
       0,
       0,
       MatMult_BlockMat,
/* 4*/ MatMultAdd_BlockMat,
       MatMultTranspose_BlockMat,
       MatMultTransposeAdd_BlockMat,
       0,
       0,
       0,
/*10*/ 0,
       0,
       0,
       0,
       0,
/*15*/ 0,
       0,
       0,
       0,
       0,
/*20*/ 0,
       MatAssemblyEnd_BlockMat,
       0,
       0,
       0,
/*25*/ 0,
       0,
       0,
       0,
       0,
/*30*/ 0,
       0,
       0,
       0,
       0,
/*35*/ 0,
       0,
       0,
       0,
       0,
/*40*/ 0,
       0,
       0,
       0,
       0,
/*45*/ 0,
       0,
       0,
       0,
       0,
/*50*/ MatSetBlockSize_BlockMat,
       0,
       0,
       0,
       0,
/*55*/ 0,
       0,
       0,
       0,
       0,
/*60*/ 0,
       MatDestroy_BlockMat,
       0,
       0,
       0,
/*65*/ 0,
       0,
       0,
       0,
       0,
/*70*/ 0,
       0,
       0,
       0,
       0,
/*75*/ 0,
       0,
       0,
       0,
       0,
/*80*/ 0,
       0,
       0,
       0,
       MatLoad_BlockMat,
/*85*/ 0,
       0,
       0,
       0,
       0,
/*90*/ 0,
       0,
       0,
       0,
       0,
/*95*/ 0,
       0,
       0,
       0};

/*MC
   MATBLOCKMAT - A matrix that is defined by a set of Mat's that represents a sparse block matrix
                 consisting of (usually) sparse blocks.

  Level: advanced

.seealso: MatCreateBlockMat()

M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatCreate_BlockMat"
PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_BlockMat(Mat A)
{
  Mat_BlockMat    *b;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMemcpy(A->ops,&MatOps_Values,sizeof(struct _MatOps));CHKERRQ(ierr);
  ierr = PetscNew(Mat_BlockMat,&b);CHKERRQ(ierr);

  A->data = (void*)b;

  ierr = PetscMapInitialize(A->comm,&A->rmap);CHKERRQ(ierr);
  ierr = PetscMapInitialize(A->comm,&A->cmap);CHKERRQ(ierr);

  A->assembled     = PETSC_TRUE;
  A->preallocated  = PETSC_FALSE;
  ierr = PetscObjectChangeTypeName((PetscObject)A,MATBLOCKMAT);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatCreateBlockMat"
/*@C
   MatCreateBlockMat - Creates a new matrix based sparse Mat storage

  Collective on MPI_Comm

   Input Parameters:
+  comm - MPI communicator
.  m - number of rows
.  n  - number of columns
-  bs - size of each submatrix


   Output Parameter:
.  A - the matrix

   Level: intermediate

   PETSc requires that matrices and vectors being used for certain
   operations are partitioned accordingly.  For example, when
   creating a bmat matrix, A, that supports parallel matrix-vector
   products using MatMult(A,x,y) the user should set the number
   of local matrix rows to be the number of local elements of the
   corresponding result vector, y. Note that this is information is
   required for use of the matrix interface routines, even though
   the bmat matrix may not actually be physically partitioned.
   For example,

.keywords: matrix, bmat, create

.seealso: MATBLOCKMAT
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatCreateBlockMat(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt bs,Mat *A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetType(*A,MATBLOCKMAT);CHKERRQ(ierr);
  ierr = MatSetBlockSize(*A,bs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



