
/*
     Defines the basic matrix operations for sequential dense.
*/

#include <../src/mat/impls/dense/seq/dense.h> /*I "petscmat.h" I*/
#include <petscblaslapack.h>

#include <../src/mat/impls/aij/seq/aij.h>

PetscErrorCode MatSeqDenseSymmetrize_Private(Mat A, PetscBool hermitian)
{
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
  PetscInt       j, k, n = A->rmap->n;
  PetscScalar    *v;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (A->rmap->n != A->cmap->n) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Cannot symmetrize a rectangular matrix");
  ierr = MatDenseGetArray(A,&v);CHKERRQ(ierr);
  if (!hermitian) {
    for (k=0;k<n;k++) {
      for (j=k;j<n;j++) {
        v[j*mat->lda + k] = v[k*mat->lda + j];
      }
    }
  } else {
    for (k=0;k<n;k++) {
      for (j=k;j<n;j++) {
        v[j*mat->lda + k] = PetscConj(v[k*mat->lda + j]);
      }
    }
  }
  ierr = MatDenseRestoreArray(A,&v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatSeqDenseInvertFactors_Private(Mat A)
{
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;
  PetscBLASInt   info,n;

  PetscFunctionBegin;
  if (!A->rmap->n || !A->cmap->n) PetscFunctionReturn(0);
  ierr = PetscBLASIntCast(A->cmap->n,&n);CHKERRQ(ierr);
  if (A->factortype == MAT_FACTOR_LU) {
    if (!mat->pivots) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Pivots not present");
    if (!mat->fwork) {
      mat->lfwork = n;
      ierr = PetscMalloc1(mat->lfwork,&mat->fwork);CHKERRQ(ierr);
      ierr = PetscLogObjectMemory((PetscObject)A,mat->lfwork*sizeof(PetscBLASInt));CHKERRQ(ierr);
    }
    ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
    PetscStackCallBLAS("LAPACKgetri",LAPACKgetri_(&n,mat->v,&mat->lda,mat->pivots,mat->fwork,&mat->lfwork,&info));
    ierr = PetscFPTrapPop();CHKERRQ(ierr);
    ierr = PetscLogFlops((1.0*A->cmap->n*A->cmap->n*A->cmap->n)/3.0);CHKERRQ(ierr);
  } else if (A->factortype == MAT_FACTOR_CHOLESKY) {
    if (A->spd) {
      ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
      PetscStackCallBLAS("LAPACKpotri",LAPACKpotri_("L",&n,mat->v,&mat->lda,&info));
      ierr = PetscFPTrapPop();CHKERRQ(ierr);
      ierr = MatSeqDenseSymmetrize_Private(A,PETSC_TRUE);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
    } else if (A->hermitian) {
      if (!mat->pivots) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Pivots not present");
      if (!mat->fwork) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Fwork not present");
      ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
      PetscStackCallBLAS("LAPACKhetri",LAPACKhetri_("L",&n,mat->v,&mat->lda,mat->pivots,mat->fwork,&info));
      ierr = PetscFPTrapPop();CHKERRQ(ierr);
      ierr = MatSeqDenseSymmetrize_Private(A,PETSC_TRUE);CHKERRQ(ierr);
#endif
    } else { /* symmetric case */
      if (!mat->pivots) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Pivots not present");
      if (!mat->fwork) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Fwork not present");
      ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
      PetscStackCallBLAS("LAPACKsytri",LAPACKsytri_("L",&n,mat->v,&mat->lda,mat->pivots,mat->fwork,&info));
      ierr = PetscFPTrapPop();CHKERRQ(ierr);
      ierr = MatSeqDenseSymmetrize_Private(A,PETSC_FALSE);CHKERRQ(ierr);
    }
    if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_MAT_CH_ZRPVT,"Bad Inversion: zero pivot in row %D",(PetscInt)info-1);
    ierr = PetscLogFlops((1.0*A->cmap->n*A->cmap->n*A->cmap->n)/3.0);CHKERRQ(ierr);
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Matrix must be factored to solve");

  A->ops->solve             = NULL;
  A->ops->matsolve          = NULL;
  A->ops->solvetranspose    = NULL;
  A->ops->matsolvetranspose = NULL;
  A->ops->solveadd          = NULL;
  A->ops->solvetransposeadd = NULL;
  A->factortype             = MAT_FACTOR_NONE;
  ierr                      = PetscFree(A->solvertype);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatZeroRowsColumns_SeqDense(Mat A,PetscInt N,const PetscInt rows[],PetscScalar diag,Vec x,Vec b)
{
  PetscErrorCode    ierr;
  Mat_SeqDense      *l = (Mat_SeqDense*)A->data;
  PetscInt          m  = l->lda, n = A->cmap->n,r = A->rmap->n, i,j;
  PetscScalar       *slot,*bb,*v;
  const PetscScalar *xx;

  PetscFunctionBegin;
  if (PetscDefined(USE_DEBUG)) {
    for (i=0; i<N; i++) {
      if (rows[i] < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative row requested to be zeroed");
      if (rows[i] >= A->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row %D requested to be zeroed greater than or equal number of rows %D",rows[i],A->rmap->n);
      if (rows[i] >= A->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Col %D requested to be zeroed greater than or equal number of cols %D",rows[i],A->cmap->n);
    }
  }
  if (!N) PetscFunctionReturn(0);

  /* fix right hand side if needed */
  if (x && b) {
    Vec xt;

    if (A->rmap->n != A->cmap->n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only coded for square matrices");
    ierr = VecDuplicate(x,&xt);CHKERRQ(ierr);
    ierr = VecCopy(x,xt);CHKERRQ(ierr);
    ierr = VecScale(xt,-1.0);CHKERRQ(ierr);
    ierr = MatMultAdd(A,xt,b,b);CHKERRQ(ierr);
    ierr = VecDestroy(&xt);CHKERRQ(ierr);
    ierr = VecGetArrayRead(x,&xx);CHKERRQ(ierr);
    ierr = VecGetArray(b,&bb);CHKERRQ(ierr);
    for (i=0; i<N; i++) bb[rows[i]] = diag*xx[rows[i]];
    ierr = VecRestoreArrayRead(x,&xx);CHKERRQ(ierr);
    ierr = VecRestoreArray(b,&bb);CHKERRQ(ierr);
  }

  ierr = MatDenseGetArray(A,&v);CHKERRQ(ierr);
  for (i=0; i<N; i++) {
    slot = v + rows[i]*m;
    ierr = PetscArrayzero(slot,r);CHKERRQ(ierr);
  }
  for (i=0; i<N; i++) {
    slot = v + rows[i];
    for (j=0; j<n; j++) { *slot = 0.0; slot += m;}
  }
  if (diag != 0.0) {
    if (A->rmap->n != A->cmap->n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only coded for square matrices");
    for (i=0; i<N; i++) {
      slot  = v + (m+1)*rows[i];
      *slot = diag;
    }
  }
  ierr = MatDenseRestoreArray(A,&v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatPtAPNumeric_SeqDense_SeqDense(Mat A,Mat P,Mat C)
{
  Mat_SeqDense   *c = (Mat_SeqDense*)(C->data);
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (c->ptapwork) {
    ierr = (*C->ops->matmultnumeric)(A,P,c->ptapwork);CHKERRQ(ierr);
    ierr = (*C->ops->transposematmultnumeric)(P,c->ptapwork,C);CHKERRQ(ierr);
  } else SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_SUP,"Must call MatPtAPSymbolic_SeqDense_SeqDense() first");
  PetscFunctionReturn(0);
}

PetscErrorCode MatPtAPSymbolic_SeqDense_SeqDense(Mat A,Mat P,PetscReal fill,Mat C)
{
  Mat_SeqDense   *c;
  PetscBool      cisdense;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSetSizes(C,P->cmap->n,P->cmap->n,P->cmap->N,P->cmap->N);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompareAny((PetscObject)C,&cisdense,MATSEQDENSE,MATSEQDENSECUDA,"");CHKERRQ(ierr);
  if (!cisdense) {
    PetscBool flg;

    ierr = PetscObjectTypeCompare((PetscObject)P,((PetscObject)A)->type_name,&flg);CHKERRQ(ierr);
    ierr = MatSetType(C,flg ? ((PetscObject)A)->type_name : MATDENSE);CHKERRQ(ierr);
  }
  ierr = MatSetUp(C);CHKERRQ(ierr);
  c    = (Mat_SeqDense*)C->data;
  ierr = MatCreate(PetscObjectComm((PetscObject)A),&c->ptapwork);CHKERRQ(ierr);
  ierr = MatSetSizes(c->ptapwork,A->rmap->n,P->cmap->n,A->rmap->N,P->cmap->N);CHKERRQ(ierr);
  ierr = MatSetType(c->ptapwork,((PetscObject)C)->type_name);CHKERRQ(ierr);
  ierr = MatSetUp(c->ptapwork);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatConvert_SeqAIJ_SeqDense(Mat A,MatType newtype,MatReuse reuse,Mat *newmat)
{
  Mat             B = NULL;
  Mat_SeqAIJ      *a = (Mat_SeqAIJ*)A->data;
  Mat_SeqDense    *b;
  PetscErrorCode  ierr;
  PetscInt        *ai=a->i,*aj=a->j,m=A->rmap->N,n=A->cmap->N,i;
  const MatScalar *av;
  PetscBool       isseqdense;

  PetscFunctionBegin;
  if (reuse == MAT_REUSE_MATRIX) {
    ierr = PetscObjectTypeCompare((PetscObject)*newmat,MATSEQDENSE,&isseqdense);CHKERRQ(ierr);
    if (!isseqdense) SETERRQ1(PetscObjectComm((PetscObject)*newmat),PETSC_ERR_USER,"Cannot reuse matrix of type %s",((PetscObject)(*newmat))->type_name);
  }
  if (reuse != MAT_REUSE_MATRIX) {
    ierr = MatCreate(PetscObjectComm((PetscObject)A),&B);CHKERRQ(ierr);
    ierr = MatSetSizes(B,m,n,m,n);CHKERRQ(ierr);
    ierr = MatSetType(B,MATSEQDENSE);CHKERRQ(ierr);
    ierr = MatSeqDenseSetPreallocation(B,NULL);CHKERRQ(ierr);
    b    = (Mat_SeqDense*)(B->data);
  } else {
    b    = (Mat_SeqDense*)((*newmat)->data);
    ierr = PetscArrayzero(b->v,m*n);CHKERRQ(ierr);
  }
  ierr = MatSeqAIJGetArrayRead(A,&av);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    PetscInt j;
    for (j=0;j<ai[1]-ai[0];j++) {
      b->v[*aj*m+i] = *av;
      aj++;
      av++;
    }
    ai++;
  }
  ierr = MatSeqAIJRestoreArrayRead(A,&av);CHKERRQ(ierr);

  if (reuse == MAT_INPLACE_MATRIX) {
    ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatHeaderReplace(A,&B);CHKERRQ(ierr);
  } else {
    if (B) *newmat = B;
    ierr = MatAssemblyBegin(*newmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*newmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatConvert_SeqDense_SeqAIJ(Mat A, MatType newtype,MatReuse reuse,Mat *newmat)
{
  Mat            B = NULL;
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;
  PetscInt       i, j;
  PetscInt       *rows, *nnz;
  MatScalar      *aa = a->v, *vals;

  PetscFunctionBegin;
  ierr = PetscCalloc3(A->rmap->n,&rows,A->rmap->n,&nnz,A->rmap->n,&vals);CHKERRQ(ierr);
  if (reuse != MAT_REUSE_MATRIX) {
    ierr = MatCreate(PetscObjectComm((PetscObject)A),&B);CHKERRQ(ierr);
    ierr = MatSetSizes(B,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N);CHKERRQ(ierr);
    ierr = MatSetType(B,MATSEQAIJ);CHKERRQ(ierr);
    for (j=0; j<A->cmap->n; j++) {
      for (i=0; i<A->rmap->n; i++) if (aa[i] != 0.0 || (i == j && A->cmap->n == A->rmap->n)) ++nnz[i];
      aa += a->lda;
    }
    ierr = MatSeqAIJSetPreallocation(B,PETSC_DETERMINE,nnz);CHKERRQ(ierr);
  } else B = *newmat;
  aa = a->v;
  for (j=0; j<A->cmap->n; j++) {
    PetscInt numRows = 0;
    for (i=0; i<A->rmap->n; i++) if (aa[i] != 0.0 || (i == j && A->cmap->n == A->rmap->n)) {rows[numRows] = i; vals[numRows++] = aa[i];}
    ierr = MatSetValues(B,numRows,rows,1,&j,vals,INSERT_VALUES);CHKERRQ(ierr);
    aa  += a->lda;
  }
  ierr = PetscFree3(rows,nnz,vals);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  if (reuse == MAT_INPLACE_MATRIX) {
    ierr = MatHeaderReplace(A,&B);CHKERRQ(ierr);
  } else if (reuse != MAT_REUSE_MATRIX) *newmat = B;
  PetscFunctionReturn(0);
}

PetscErrorCode MatAXPY_SeqDense(Mat Y,PetscScalar alpha,Mat X,MatStructure str)
{
  Mat_SeqDense      *x = (Mat_SeqDense*)X->data,*y = (Mat_SeqDense*)Y->data;
  const PetscScalar *xv;
  PetscScalar       *yv;
  PetscBLASInt      N,m,ldax = 0,lday = 0,one = 1;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatDenseGetArrayRead(X,&xv);CHKERRQ(ierr);
  ierr = MatDenseGetArray(Y,&yv);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(X->rmap->n*X->cmap->n,&N);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(X->rmap->n,&m);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(x->lda,&ldax);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(y->lda,&lday);CHKERRQ(ierr);
  if (ldax>m || lday>m) {
    PetscInt j;

    for (j=0; j<X->cmap->n; j++) {
      PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&m,&alpha,xv+j*ldax,&one,yv+j*lday,&one));
    }
  } else {
    PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&N,&alpha,xv,&one,yv,&one));
  }
  ierr = MatDenseRestoreArrayRead(X,&xv);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(Y,&yv);CHKERRQ(ierr);
  ierr = PetscLogFlops(PetscMax(2.0*N-1,0));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetInfo_SeqDense(Mat A,MatInfoType flag,MatInfo *info)
{
  PetscLogDouble N = A->rmap->n*A->cmap->n;

  PetscFunctionBegin;
  info->block_size        = 1.0;
  info->nz_allocated      = N;
  info->nz_used           = N;
  info->nz_unneeded       = 0;
  info->assemblies        = A->num_ass;
  info->mallocs           = 0;
  info->memory            = ((PetscObject)A)->mem;
  info->fill_ratio_given  = 0;
  info->fill_ratio_needed = 0;
  info->factor_mallocs    = 0;
  PetscFunctionReturn(0);
}

PetscErrorCode MatScale_SeqDense(Mat A,PetscScalar alpha)
{
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;
  PetscScalar    *v;
  PetscErrorCode ierr;
  PetscBLASInt   one = 1,j,nz,lda = 0;

  PetscFunctionBegin;
  ierr = MatDenseGetArray(A,&v);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(a->lda,&lda);CHKERRQ(ierr);
  if (lda>A->rmap->n) {
    ierr = PetscBLASIntCast(A->rmap->n,&nz);CHKERRQ(ierr);
    for (j=0; j<A->cmap->n; j++) {
      PetscStackCallBLAS("BLASscal",BLASscal_(&nz,&alpha,v+j*lda,&one));
    }
  } else {
    ierr = PetscBLASIntCast(A->rmap->n*A->cmap->n,&nz);CHKERRQ(ierr);
    PetscStackCallBLAS("BLASscal",BLASscal_(&nz,&alpha,v,&one));
  }
  ierr = PetscLogFlops(nz);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(A,&v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatIsHermitian_SeqDense(Mat A,PetscReal rtol,PetscBool  *fl)
{
  Mat_SeqDense      *a = (Mat_SeqDense*)A->data;
  PetscInt          i,j,m = A->rmap->n,N = a->lda;
  const PetscScalar *v;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  *fl = PETSC_FALSE;
  if (A->rmap->n != A->cmap->n) PetscFunctionReturn(0);
  ierr = MatDenseGetArrayRead(A,&v);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    for (j=i; j<m; j++) {
      if (PetscAbsScalar(v[i+j*N] - PetscConj(v[j+i*N])) > rtol) {
        goto restore;
      }
    }
  }
  *fl  = PETSC_TRUE;
restore:
  ierr = MatDenseRestoreArrayRead(A,&v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatIsSymmetric_SeqDense(Mat A,PetscReal rtol,PetscBool  *fl)
{
  Mat_SeqDense      *a = (Mat_SeqDense*)A->data;
  PetscInt          i,j,m = A->rmap->n,N = a->lda;
  const PetscScalar *v;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  *fl = PETSC_FALSE;
  if (A->rmap->n != A->cmap->n) PetscFunctionReturn(0);
  ierr = MatDenseGetArrayRead(A,&v);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    for (j=i; j<m; j++) {
      if (PetscAbsScalar(v[i+j*N] - v[j+i*N]) > rtol) {
        goto restore;
      }
    }
  }
  *fl  = PETSC_TRUE;
restore:
  ierr = MatDenseRestoreArrayRead(A,&v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatDuplicateNoCreate_SeqDense(Mat newi,Mat A,MatDuplicateOption cpvalues)
{
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;
  PetscInt       lda = (PetscInt)mat->lda,j,m,nlda = lda;

  PetscFunctionBegin;
  ierr = PetscLayoutReference(A->rmap,&newi->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutReference(A->cmap,&newi->cmap);CHKERRQ(ierr);
  if (cpvalues == MAT_SHARE_NONZERO_PATTERN) { /* propagate LDA */
    ierr = MatDenseSetLDA(newi,lda);CHKERRQ(ierr);
  }
  ierr = MatSeqDenseSetPreallocation(newi,NULL);CHKERRQ(ierr);
  if (cpvalues == MAT_COPY_VALUES) {
    const PetscScalar *av;
    PetscScalar       *v;

    ierr = MatDenseGetArrayRead(A,&av);CHKERRQ(ierr);
    ierr = MatDenseGetArray(newi,&v);CHKERRQ(ierr);
    ierr = MatDenseGetLDA(newi,&nlda);CHKERRQ(ierr);
    m    = A->rmap->n;
    if (lda>m || nlda>m) {
      for (j=0; j<A->cmap->n; j++) {
        ierr = PetscArraycpy(v+j*nlda,av+j*lda,m);CHKERRQ(ierr);
      }
    } else {
      ierr = PetscArraycpy(v,av,A->rmap->n*A->cmap->n);CHKERRQ(ierr);
    }
    ierr = MatDenseRestoreArray(newi,&v);CHKERRQ(ierr);
    ierr = MatDenseRestoreArrayRead(A,&av);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatDuplicate_SeqDense(Mat A,MatDuplicateOption cpvalues,Mat *newmat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(PetscObjectComm((PetscObject)A),newmat);CHKERRQ(ierr);
  ierr = MatSetSizes(*newmat,A->rmap->n,A->cmap->n,A->rmap->n,A->cmap->n);CHKERRQ(ierr);
  ierr = MatSetType(*newmat,((PetscObject)A)->type_name);CHKERRQ(ierr);
  ierr = MatDuplicateNoCreate_SeqDense(*newmat,A,cpvalues);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolve_SeqDense_Internal_LU(Mat A, PetscScalar *x, PetscBLASInt ldx, PetscBLASInt m, PetscBLASInt nrhs, PetscBLASInt k, PetscBool T)
{
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
  PetscBLASInt    info;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
  PetscStackCallBLAS("LAPACKgetrs",LAPACKgetrs_(T ? "T" : "N",&m,&nrhs,mat->v,&mat->lda,mat->pivots,x,&m,&info));
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"GETRS - Bad solve");
  ierr = PetscLogFlops(nrhs*(2.0*m*m - m));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatConjugate_SeqDense(Mat);

static PetscErrorCode MatSolve_SeqDense_Internal_Cholesky(Mat A, PetscScalar *x, PetscBLASInt ldx, PetscBLASInt m, PetscBLASInt nrhs, PetscBLASInt k, PetscBool T)
{
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
  PetscBLASInt    info;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (A->spd) {
    if (PetscDefined(USE_COMPLEX) && T) {ierr = MatConjugate_SeqDense(A);CHKERRQ(ierr);}
    ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
    PetscStackCallBLAS("LAPACKpotrs",LAPACKpotrs_("L",&m,&nrhs,mat->v,&mat->lda,x,&m,&info));
    ierr = PetscFPTrapPop();CHKERRQ(ierr);
    if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"POTRS Bad solve");
    if (PetscDefined(USE_COMPLEX) && T) {ierr = MatConjugate_SeqDense(A);CHKERRQ(ierr);}
#if defined(PETSC_USE_COMPLEX)
  } else if (A->hermitian) {
    if (T) {ierr = MatConjugate_SeqDense(A);CHKERRQ(ierr);}
    ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
    PetscStackCallBLAS("LAPACKhetrs",LAPACKhetrs_("L",&m,&nrhs,mat->v,&mat->lda,mat->pivots,x,&m,&info));
    ierr = PetscFPTrapPop();CHKERRQ(ierr);
    if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"HETRS Bad solve");
    if (T) {ierr = MatConjugate_SeqDense(A);CHKERRQ(ierr);}
#endif
  } else { /* symmetric case */
    ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
    PetscStackCallBLAS("LAPACKsytrs",LAPACKsytrs_("L",&m,&nrhs,mat->v,&mat->lda,mat->pivots,x,&m,&info));
    ierr = PetscFPTrapPop();CHKERRQ(ierr);
    if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"SYTRS Bad solve");
  }
  ierr = PetscLogFlops(nrhs*(2.0*m*m - m));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolve_SeqDense_Internal_QR(Mat A, PetscScalar *x, PetscBLASInt ldx, PetscBLASInt m, PetscBLASInt nrhs, PetscBLASInt k)
{
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
  PetscBLASInt    info;
  char            trans;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (PetscDefined(USE_COMPLEX)) {
    trans = 'C';
  } else {
    trans = 'T';
  }
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
  PetscStackCallBLAS("LAPACKormqr",LAPACKormqr_("L", &trans, &m,&nrhs,&mat->rank,mat->v,&mat->lda,mat->tau,x,&ldx,mat->fwork,&mat->lfwork,&info));
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"ORMQR - Bad orthogonal transform");
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
  PetscStackCallBLAS("LAPACKtrtrs",LAPACKtrtrs_("U", "N", "N", &mat->rank,&nrhs,mat->v,&mat->lda,x,&ldx,&info));
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"TRTRS - Bad triangular solve");
  for (PetscInt j = 0; j < nrhs; j++) {
    for (PetscInt i = mat->rank; i < k; i++) {
      x[j*ldx + i] = 0.;
    }
  }
  ierr = PetscLogFlops(nrhs*(4.0*m*mat->rank - PetscSqr(mat->rank)));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolveTranspose_SeqDense_Internal_QR(Mat A, PetscScalar *x, PetscBLASInt ldx, PetscBLASInt m, PetscBLASInt nrhs, PetscBLASInt k)
{
  Mat_SeqDense      *mat = (Mat_SeqDense*)A->data;
  PetscBLASInt      info;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (A->rmap->n == A->cmap->n && mat->rank == A->rmap->n) {
    ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
    PetscStackCallBLAS("LAPACKtrtrs",LAPACKtrtrs_("U", "T", "N", &m,&nrhs,mat->v,&mat->lda,x,&ldx,&info));
    ierr = PetscFPTrapPop();CHKERRQ(ierr);
    if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"TRTRS - Bad triangular solve");
    if (PetscDefined(USE_COMPLEX)) {ierr = MatConjugate_SeqDense(A);CHKERRQ(ierr);}
    ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
    PetscStackCallBLAS("LAPACKormqr",LAPACKormqr_("L", "N", &m,&nrhs,&mat->rank,mat->v,&mat->lda,mat->tau,x,&ldx,mat->fwork,&mat->lfwork,&info));
    ierr = PetscFPTrapPop();CHKERRQ(ierr);
    if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"ORMQR - Bad orthogonal transform");
    if (PetscDefined(USE_COMPLEX)) {ierr = MatConjugate_SeqDense(A);CHKERRQ(ierr);}
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"QR factored matrix cannot be used for transpose solve");
  ierr = PetscLogFlops(nrhs*(4.0*m*mat->rank - PetscSqr(mat->rank)));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolve_SeqDense_SetUp(Mat A, Vec xx, Vec yy, PetscScalar **_y, PetscBLASInt *_m, PetscBLASInt *_k)
{
  Mat_SeqDense      *mat = (Mat_SeqDense *) A->data;
  PetscScalar       *y;
  PetscBLASInt      m=0, k=0;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(A->rmap->n,&m);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(A->cmap->n,&k);CHKERRQ(ierr);
  if (k < m) {
    ierr = VecCopy(xx, mat->qrrhs);CHKERRQ(ierr);
    ierr = VecGetArray(mat->qrrhs,&y);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(xx, yy);CHKERRQ(ierr);
    ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  }
  *_y = y;
  *_k = k;
  *_m = m;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolve_SeqDense_TearDown(Mat A, Vec xx, Vec yy, PetscScalar **_y, PetscBLASInt *_m, PetscBLASInt *_k)
{
  Mat_SeqDense   *mat = (Mat_SeqDense *) A->data;
  PetscScalar    *y = NULL;
  PetscBLASInt   m, k;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  y   = *_y;
  *_y = NULL;
  k   = *_k;
  m   = *_m;
  if (k < m) {
    PetscScalar *yv;
    ierr = VecGetArray(yy,&yv);CHKERRQ(ierr);
    ierr = PetscArraycpy(yv, y, k);CHKERRQ(ierr);
    ierr = VecRestoreArray(yy,&yv);CHKERRQ(ierr);
    ierr = VecRestoreArray(mat->qrrhs, &y);CHKERRQ(ierr);
  } else {
    ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolve_SeqDense_LU(Mat A, Vec xx, Vec yy)
{
  PetscScalar    *y = NULL;
  PetscBLASInt   m = 0, k = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSolve_SeqDense_SetUp(A, xx, yy, &y, &m, &k);CHKERRQ(ierr);
  ierr = MatSolve_SeqDense_Internal_LU(A, y, m, m, 1, k, PETSC_FALSE);CHKERRQ(ierr);
  ierr = MatSolve_SeqDense_TearDown(A, xx, yy, &y, &m, &k);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolveTranspose_SeqDense_LU(Mat A, Vec xx, Vec yy)
{
  PetscScalar    *y = NULL;
  PetscBLASInt   m = 0, k = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSolve_SeqDense_SetUp(A, xx, yy, &y, &m, &k);CHKERRQ(ierr);
  ierr = MatSolve_SeqDense_Internal_LU(A, y, m, m, 1, k, PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatSolve_SeqDense_TearDown(A, xx, yy, &y, &m, &k);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolve_SeqDense_Cholesky(Mat A, Vec xx, Vec yy)
{
  PetscScalar    *y = NULL;
  PetscBLASInt   m = 0, k = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSolve_SeqDense_SetUp(A, xx, yy, &y, &m, &k);CHKERRQ(ierr);
  ierr = MatSolve_SeqDense_Internal_Cholesky(A, y, m, m, 1, k, PETSC_FALSE);CHKERRQ(ierr);
  ierr = MatSolve_SeqDense_TearDown(A, xx, yy, &y, &m, &k);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolveTranspose_SeqDense_Cholesky(Mat A, Vec xx, Vec yy)
{
  PetscScalar    *y = NULL;
  PetscBLASInt   m = 0, k = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSolve_SeqDense_SetUp(A, xx, yy, &y, &m, &k);CHKERRQ(ierr);
  ierr = MatSolve_SeqDense_Internal_Cholesky(A, y, m, m, 1, k, PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatSolve_SeqDense_TearDown(A, xx, yy, &y, &m, &k);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolve_SeqDense_QR(Mat A, Vec xx, Vec yy)
{
  PetscScalar    *y = NULL;
  PetscBLASInt   m = 0, k = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSolve_SeqDense_SetUp(A, xx, yy, &y, &m, &k);CHKERRQ(ierr);
  ierr = MatSolve_SeqDense_Internal_QR(A, y, PetscMax(m,k), m, 1, k);CHKERRQ(ierr);
  ierr = MatSolve_SeqDense_TearDown(A, xx, yy, &y, &m, &k);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolveTranspose_SeqDense_QR(Mat A, Vec xx, Vec yy)
{
  PetscScalar    *y = NULL;
  PetscBLASInt   m = 0, k = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSolve_SeqDense_SetUp(A, xx, yy, &y, &m, &k);CHKERRQ(ierr);
  ierr = MatSolveTranspose_SeqDense_Internal_QR(A, y, PetscMax(m,k), m, 1, k);CHKERRQ(ierr);
  ierr = MatSolve_SeqDense_TearDown(A, xx, yy, &y, &m, &k);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatSolve_SeqDense_SetUp(Mat A, Mat B, Mat X, PetscScalar **_y, PetscBLASInt *_ldy, PetscBLASInt *_m, PetscBLASInt *_nrhs, PetscBLASInt *_k)
{
  PetscErrorCode    ierr;
  const PetscScalar *b;
  PetscScalar       *y;
  PetscInt          n, _ldb, _ldx;
  PetscBLASInt      nrhs=0,m=0,k=0,ldb=0,ldx=0,ldy=0;

  PetscFunctionBegin;
  *_ldy=0; *_m=0; *_nrhs=0; *_k=0; *_y = NULL;
  ierr = PetscBLASIntCast(A->rmap->n,&m);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(A->cmap->n,&k);CHKERRQ(ierr);
  ierr = MatGetSize(B,NULL,&n);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(n,&nrhs);CHKERRQ(ierr);
  ierr = MatDenseGetLDA(B,&_ldb);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(_ldb, &ldb);CHKERRQ(ierr);
  ierr = MatDenseGetLDA(X,&_ldx);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(_ldx, &ldx);CHKERRQ(ierr);
  if (ldx < m) {
    ierr = MatDenseGetArrayRead(B,&b);CHKERRQ(ierr);
    ierr = PetscMalloc1(nrhs * m, &y);CHKERRQ(ierr);
    if (ldb == m) {
      ierr = PetscArraycpy(y,b,ldb*nrhs);CHKERRQ(ierr);
    } else {
      for (PetscInt j = 0; j < nrhs; j++) {
        ierr = PetscArraycpy(&y[j*m],&b[j*ldb],m);CHKERRQ(ierr);
      }
    }
    ldy = m;
    ierr = MatDenseRestoreArrayRead(B,&b);CHKERRQ(ierr);
  } else {
    if (ldb == ldx) {
      ierr = MatCopy(B, X, SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatDenseGetArray(X,&y);CHKERRQ(ierr);
    } else {
      ierr = MatDenseGetArray(X,&y);CHKERRQ(ierr);
      ierr = MatDenseGetArrayRead(B,&b);CHKERRQ(ierr);
      for (PetscInt j = 0; j < nrhs; j++) {
        ierr = PetscArraycpy(&y[j*ldx],&b[j*ldb],m);CHKERRQ(ierr);
      }
      ierr = MatDenseRestoreArrayRead(B,&b);CHKERRQ(ierr);
    }
    ldy = ldx;
  }
  *_y    = y;
  *_ldy = ldy;
  *_k    = k;
  *_m    = m;
  *_nrhs = nrhs;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatSolve_SeqDense_TearDown(Mat A, Mat B, Mat X, PetscScalar **_y, PetscBLASInt *_ldy, PetscBLASInt *_m, PetscBLASInt *_nrhs, PetscBLASInt *_k)
{
  PetscScalar       *y;
  PetscInt          _ldx;
  PetscBLASInt      k,ldy,nrhs,ldx=0;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  y    = *_y;
  *_y  = NULL;
  k    = *_k;
  ldy = *_ldy;
  nrhs = *_nrhs;
  ierr = MatDenseGetLDA(X,&_ldx);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(_ldx, &ldx);CHKERRQ(ierr);
  if (ldx != ldy) {
    PetscScalar *xv;
    ierr = MatDenseGetArray(X,&xv);CHKERRQ(ierr);
    for (PetscInt j = 0; j < nrhs; j++) {
      ierr = PetscArraycpy(&xv[j*ldx],&y[j*ldy],k);CHKERRQ(ierr);
    }
    ierr = MatDenseRestoreArray(X,&xv);CHKERRQ(ierr);
    ierr = PetscFree(y);CHKERRQ(ierr);
  } else {
    ierr = MatDenseRestoreArray(X,&y);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatSolve_SeqDense_LU(Mat A, Mat B, Mat X)
{
  PetscScalar    *y;
  PetscBLASInt   m, k, ldy, nrhs;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMatSolve_SeqDense_SetUp(A, B, X, &y, &ldy, &m, &nrhs, &k);CHKERRQ(ierr);
  ierr = MatSolve_SeqDense_Internal_LU(A, y, ldy, m, nrhs, k, PETSC_FALSE);CHKERRQ(ierr);
  ierr = MatMatSolve_SeqDense_TearDown(A, B, X, &y, &ldy, &m, &nrhs, &k);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatSolveTranspose_SeqDense_LU(Mat A, Mat B, Mat X)
{
  PetscScalar    *y;
  PetscBLASInt   m, k, ldy, nrhs;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMatSolve_SeqDense_SetUp(A, B, X, &y, &ldy, &m, &nrhs, &k);CHKERRQ(ierr);
  ierr = MatSolve_SeqDense_Internal_LU(A, y, ldy, m, nrhs, k, PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatMatSolve_SeqDense_TearDown(A, B, X, &y, &ldy, &m, &nrhs, &k);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatSolve_SeqDense_Cholesky(Mat A, Mat B, Mat X)
{
  PetscScalar    *y;
  PetscBLASInt   m, k, ldy, nrhs;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMatSolve_SeqDense_SetUp(A, B, X, &y, &ldy, &m, &nrhs, &k);CHKERRQ(ierr);
  ierr = MatSolve_SeqDense_Internal_Cholesky(A, y, ldy, m, nrhs, k, PETSC_FALSE);CHKERRQ(ierr);
  ierr = MatMatSolve_SeqDense_TearDown(A, B, X, &y, &ldy, &m, &nrhs, &k);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatSolveTranspose_SeqDense_Cholesky(Mat A, Mat B, Mat X)
{
  PetscScalar    *y;
  PetscBLASInt   m, k, ldy, nrhs;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMatSolve_SeqDense_SetUp(A, B, X, &y, &ldy, &m, &nrhs, &k);CHKERRQ(ierr);
  ierr = MatSolve_SeqDense_Internal_Cholesky(A, y, ldy, m, nrhs, k, PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatMatSolve_SeqDense_TearDown(A, B, X, &y, &ldy, &m, &nrhs, &k);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatSolve_SeqDense_QR(Mat A, Mat B, Mat X)
{
  PetscScalar    *y;
  PetscBLASInt   m, k, ldy, nrhs;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMatSolve_SeqDense_SetUp(A, B, X, &y, &ldy, &m, &nrhs, &k);CHKERRQ(ierr);
  ierr = MatSolve_SeqDense_Internal_QR(A, y, ldy, m, nrhs, k);CHKERRQ(ierr);
  ierr = MatMatSolve_SeqDense_TearDown(A, B, X, &y, &ldy, &m, &nrhs, &k);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatSolveTranspose_SeqDense_QR(Mat A, Mat B, Mat X)
{
  PetscScalar    *y;
  PetscBLASInt   m, k, ldy, nrhs;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMatSolve_SeqDense_SetUp(A, B, X, &y, &ldy, &m, &nrhs, &k);CHKERRQ(ierr);
  ierr = MatSolveTranspose_SeqDense_Internal_QR(A, y, ldy, m, nrhs, k);CHKERRQ(ierr);
  ierr = MatMatSolve_SeqDense_TearDown(A, B, X, &y, &ldy, &m, &nrhs, &k);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatConjugate_SeqDense(Mat);

/* ---------------------------------------------------------------*/
/* COMMENT: I have chosen to hide row permutation in the pivots,
   rather than put it in the Mat->row slot.*/
PetscErrorCode MatLUFactor_SeqDense(Mat A,IS row,IS col,const MatFactorInfo *minfo)
{
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;
  PetscBLASInt   n,m,info;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(A->cmap->n,&n);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(A->rmap->n,&m);CHKERRQ(ierr);
  if (!mat->pivots) {
    ierr = PetscMalloc1(A->rmap->n,&mat->pivots);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)A,A->rmap->n*sizeof(PetscBLASInt));CHKERRQ(ierr);
  }
  if (!A->rmap->n || !A->cmap->n) PetscFunctionReturn(0);
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
  PetscStackCallBLAS("LAPACKgetrf",LAPACKgetrf_(&m,&n,mat->v,&mat->lda,mat->pivots,&info));
  ierr = PetscFPTrapPop();CHKERRQ(ierr);

  if (info<0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Bad argument to LU factorization");
  if (info>0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MAT_LU_ZRPVT,"Bad LU factorization");

  A->ops->solve             = MatSolve_SeqDense_LU;
  A->ops->matsolve          = MatMatSolve_SeqDense_LU;
  A->ops->solvetranspose    = MatSolveTranspose_SeqDense_LU;
  A->ops->matsolvetranspose = MatMatSolveTranspose_SeqDense_LU;
  A->factortype             = MAT_FACTOR_LU;

  ierr = PetscFree(A->solvertype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(MATSOLVERPETSC,&A->solvertype);CHKERRQ(ierr);

  ierr = PetscLogFlops((2.0*A->cmap->n*A->cmap->n*A->cmap->n)/3);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatLUFactorNumeric_SeqDense(Mat fact,Mat A,const MatFactorInfo *info_dummy)
{
  MatFactorInfo  info;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDuplicateNoCreate_SeqDense(fact,A,MAT_COPY_VALUES);CHKERRQ(ierr);
  ierr = (*fact->ops->lufactor)(fact,NULL,NULL,&info);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatLUFactorSymbolic_SeqDense(Mat fact,Mat A,IS row,IS col,const MatFactorInfo *info)
{
  PetscFunctionBegin;
  fact->preallocated           = PETSC_TRUE;
  fact->assembled              = PETSC_TRUE;
  fact->ops->lufactornumeric   = MatLUFactorNumeric_SeqDense;
  PetscFunctionReturn(0);
}

/* Cholesky as L*L^T or L*D*L^T and the symmetric/hermitian complex variants */
PetscErrorCode MatCholeskyFactor_SeqDense(Mat A,IS perm,const MatFactorInfo *factinfo)
{
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;
  PetscBLASInt   info,n;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(A->cmap->n,&n);CHKERRQ(ierr);
  if (!A->rmap->n || !A->cmap->n) PetscFunctionReturn(0);
  if (A->spd) {
    ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
    PetscStackCallBLAS("LAPACKpotrf",LAPACKpotrf_("L",&n,mat->v,&mat->lda,&info));
    ierr = PetscFPTrapPop();CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  } else if (A->hermitian) {
    if (!mat->pivots) {
      ierr = PetscMalloc1(A->rmap->n,&mat->pivots);CHKERRQ(ierr);
      ierr = PetscLogObjectMemory((PetscObject)A,A->rmap->n*sizeof(PetscBLASInt));CHKERRQ(ierr);
    }
    if (!mat->fwork) {
      PetscScalar dummy;

      mat->lfwork = -1;
      ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
      PetscStackCallBLAS("LAPACKhetrf",LAPACKhetrf_("L",&n,mat->v,&mat->lda,mat->pivots,&dummy,&mat->lfwork,&info));
      ierr = PetscFPTrapPop();CHKERRQ(ierr);
      mat->lfwork = (PetscInt)PetscRealPart(dummy);
      ierr = PetscMalloc1(mat->lfwork,&mat->fwork);CHKERRQ(ierr);
      ierr = PetscLogObjectMemory((PetscObject)A,mat->lfwork*sizeof(PetscBLASInt));CHKERRQ(ierr);
    }
    ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
    PetscStackCallBLAS("LAPACKhetrf",LAPACKhetrf_("L",&n,mat->v,&mat->lda,mat->pivots,mat->fwork,&mat->lfwork,&info));
    ierr = PetscFPTrapPop();CHKERRQ(ierr);
#endif
  } else { /* symmetric case */
    if (!mat->pivots) {
      ierr = PetscMalloc1(A->rmap->n,&mat->pivots);CHKERRQ(ierr);
      ierr = PetscLogObjectMemory((PetscObject)A,A->rmap->n*sizeof(PetscBLASInt));CHKERRQ(ierr);
    }
    if (!mat->fwork) {
      PetscScalar dummy;

      mat->lfwork = -1;
      ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
      PetscStackCallBLAS("LAPACKsytrf",LAPACKsytrf_("L",&n,mat->v,&mat->lda,mat->pivots,&dummy,&mat->lfwork,&info));
      ierr = PetscFPTrapPop();CHKERRQ(ierr);
      mat->lfwork = (PetscInt)PetscRealPart(dummy);
      ierr = PetscMalloc1(mat->lfwork,&mat->fwork);CHKERRQ(ierr);
      ierr = PetscLogObjectMemory((PetscObject)A,mat->lfwork*sizeof(PetscBLASInt));CHKERRQ(ierr);
    }
    ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
    PetscStackCallBLAS("LAPACKsytrf",LAPACKsytrf_("L",&n,mat->v,&mat->lda,mat->pivots,mat->fwork,&mat->lfwork,&info));
    ierr = PetscFPTrapPop();CHKERRQ(ierr);
  }
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_MAT_CH_ZRPVT,"Bad factorization: zero pivot in row %D",(PetscInt)info-1);

  A->ops->solve             = MatSolve_SeqDense_Cholesky;
  A->ops->matsolve          = MatMatSolve_SeqDense_Cholesky;
  A->ops->solvetranspose    = MatSolveTranspose_SeqDense_Cholesky;
  A->ops->matsolvetranspose = MatMatSolveTranspose_SeqDense_Cholesky;
  A->factortype             = MAT_FACTOR_CHOLESKY;

  ierr = PetscFree(A->solvertype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(MATSOLVERPETSC,&A->solvertype);CHKERRQ(ierr);

  ierr = PetscLogFlops((1.0*A->cmap->n*A->cmap->n*A->cmap->n)/3.0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCholeskyFactorNumeric_SeqDense(Mat fact,Mat A,const MatFactorInfo *info_dummy)
{
  PetscErrorCode ierr;
  MatFactorInfo  info;

  PetscFunctionBegin;
  info.fill = 1.0;

  ierr = MatDuplicateNoCreate_SeqDense(fact,A,MAT_COPY_VALUES);CHKERRQ(ierr);
  ierr = (*fact->ops->choleskyfactor)(fact,NULL,&info);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatCholeskyFactorSymbolic_SeqDense(Mat fact,Mat A,IS row,const MatFactorInfo *info)
{
  PetscFunctionBegin;
  fact->assembled                  = PETSC_TRUE;
  fact->preallocated               = PETSC_TRUE;
  fact->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqDense;
  PetscFunctionReturn(0);
}

PetscErrorCode MatQRFactor_SeqDense(Mat A,IS col,const MatFactorInfo *minfo)
{
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;
  PetscBLASInt   n,m,info, min, max;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(A->cmap->n,&n);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(A->rmap->n,&m);CHKERRQ(ierr);
  max = PetscMax(m, n);
  min = PetscMin(m, n);
  if (!mat->tau) {
    ierr = PetscMalloc1(min,&mat->tau);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)A,min*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  if (!mat->pivots) {
    ierr = PetscMalloc1(n,&mat->pivots);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)A,n*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  if (!mat->qrrhs) {
    ierr = MatCreateVecs(A, NULL, &(mat->qrrhs));CHKERRQ(ierr);
  }
  if (!A->rmap->n || !A->cmap->n) PetscFunctionReturn(0);
  if (!mat->fwork) {
    PetscScalar dummy;

    mat->lfwork = -1;
    ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
    PetscStackCallBLAS("LAPACKgeqrf",LAPACKgeqrf_(&m,&n,mat->v,&mat->lda,mat->tau,&dummy,&mat->lfwork,&info));
    ierr = PetscFPTrapPop();CHKERRQ(ierr);
    mat->lfwork = (PetscInt)PetscRealPart(dummy);
    ierr = PetscMalloc1(mat->lfwork,&mat->fwork);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)A,mat->lfwork*sizeof(PetscBLASInt));CHKERRQ(ierr);
  }
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
  PetscStackCallBLAS("LAPACKgeqrf",LAPACKgeqrf_(&m,&n,mat->v,&mat->lda,mat->tau,mat->fwork,&mat->lfwork,&info));
  ierr = PetscFPTrapPop();CHKERRQ(ierr);
  if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Bad argument to QR factorization");
  // TODO: try to estimate rank or test for and use geqp3 for rank revealing QR.  For now just say rank is min of m and n
  mat->rank = min;

  A->ops->solve             = MatSolve_SeqDense_QR;
  A->ops->matsolve          = MatMatSolve_SeqDense_QR;
  A->factortype             = MAT_FACTOR_QR;
  if (m == n) {
    A->ops->solvetranspose    = MatSolveTranspose_SeqDense_QR;
    A->ops->matsolvetranspose = MatMatSolveTranspose_SeqDense_QR;
  }

  ierr = PetscFree(A->solvertype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(MATSOLVERPETSC,&A->solvertype);CHKERRQ(ierr);

  ierr = PetscLogFlops(2.0*min*min*(max-min/3.0));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatQRFactorNumeric_SeqDense(Mat fact,Mat A,const MatFactorInfo *info_dummy)
{
  PetscErrorCode ierr;
  MatFactorInfo  info;

  PetscFunctionBegin;
  info.fill = 1.0;

  ierr = MatDuplicateNoCreate_SeqDense(fact,A,MAT_COPY_VALUES);CHKERRQ(ierr);
  ierr = PetscUseMethod(fact,"MatQRFactor_C",(Mat,IS,const MatFactorInfo *),(fact,NULL,&info));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatQRFactorSymbolic_SeqDense(Mat fact,Mat A,IS row,const MatFactorInfo *info)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  fact->assembled                  = PETSC_TRUE;
  fact->preallocated               = PETSC_TRUE;
  ierr = PetscObjectComposeFunction((PetscObject)fact,"MatQRFactorNumeric_C",MatQRFactorNumeric_SeqDense);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* uses LAPACK */
PETSC_INTERN PetscErrorCode MatGetFactor_seqdense_petsc(Mat A,MatFactorType ftype,Mat *fact)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(PetscObjectComm((PetscObject)A),fact);CHKERRQ(ierr);
  ierr = MatSetSizes(*fact,A->rmap->n,A->cmap->n,A->rmap->n,A->cmap->n);CHKERRQ(ierr);
  ierr = MatSetType(*fact,MATDENSE);CHKERRQ(ierr);
  (*fact)->trivialsymbolic = PETSC_TRUE;
  if (ftype == MAT_FACTOR_LU || ftype == MAT_FACTOR_ILU) {
    (*fact)->ops->lufactorsymbolic = MatLUFactorSymbolic_SeqDense;
    (*fact)->ops->ilufactorsymbolic = MatLUFactorSymbolic_SeqDense;
  } else if (ftype == MAT_FACTOR_CHOLESKY || ftype == MAT_FACTOR_ICC) {
    (*fact)->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_SeqDense;
  } else if (ftype == MAT_FACTOR_QR) {
    ierr = PetscObjectComposeFunction((PetscObject)(*fact),"MatQRFactorSymbolic_C",MatQRFactorSymbolic_SeqDense);CHKERRQ(ierr);
  }
  (*fact)->factortype = ftype;

  ierr = PetscFree((*fact)->solvertype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(MATSOLVERPETSC,&(*fact)->solvertype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(MATORDERINGEXTERNAL,(char**)&(*fact)->preferredordering[MAT_FACTOR_LU]);CHKERRQ(ierr);
  ierr = PetscStrallocpy(MATORDERINGEXTERNAL,(char**)&(*fact)->preferredordering[MAT_FACTOR_ILU]);CHKERRQ(ierr);
  ierr = PetscStrallocpy(MATORDERINGEXTERNAL,(char**)&(*fact)->preferredordering[MAT_FACTOR_CHOLESKY]);CHKERRQ(ierr);
  ierr = PetscStrallocpy(MATORDERINGEXTERNAL,(char**)&(*fact)->preferredordering[MAT_FACTOR_ICC]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------*/
static PetscErrorCode MatSOR_SeqDense(Mat A,Vec bb,PetscReal omega,MatSORType flag,PetscReal shift,PetscInt its,PetscInt lits,Vec xx)
{
  Mat_SeqDense      *mat = (Mat_SeqDense*)A->data;
  PetscScalar       *x,*v = mat->v,zero = 0.0,xt;
  const PetscScalar *b;
  PetscErrorCode    ierr;
  PetscInt          m = A->rmap->n,i;
  PetscBLASInt      o = 1,bm = 0;

  PetscFunctionBegin;
#if defined(PETSC_HAVE_CUDA)
  if (A->offloadmask == PETSC_OFFLOAD_GPU) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not implemented");
#endif
  if (shift == -1) shift = 0.0; /* negative shift indicates do not error on zero diagonal; this code never zeros on zero diagonal */
  ierr = PetscBLASIntCast(m,&bm);CHKERRQ(ierr);
  if (flag & SOR_ZERO_INITIAL_GUESS) {
    /* this is a hack fix, should have another version without the second BLASdotu */
    ierr = VecSet(xx,zero);CHKERRQ(ierr);
  }
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
  its  = its*lits;
  if (its <= 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Relaxation requires global its %D and local its %D both positive",its,lits);
  while (its--) {
    if (flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP) {
      for (i=0; i<m; i++) {
        PetscStackCallBLAS("BLASdotu",xt   = b[i] - BLASdotu_(&bm,v+i,&bm,x,&o));
        x[i] = (1. - omega)*x[i] + omega*(xt+v[i + i*m]*x[i])/(v[i + i*m]+shift);
      }
    }
    if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP) {
      for (i=m-1; i>=0; i--) {
        PetscStackCallBLAS("BLASdotu",xt   = b[i] - BLASdotu_(&bm,v+i,&bm,x,&o));
        x[i] = (1. - omega)*x[i] + omega*(xt+v[i + i*m]*x[i])/(v[i + i*m]+shift);
      }
    }
  }
  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -----------------------------------------------------------------*/
PetscErrorCode MatMultTranspose_SeqDense(Mat A,Vec xx,Vec yy)
{
  Mat_SeqDense      *mat = (Mat_SeqDense*)A->data;
  const PetscScalar *v   = mat->v,*x;
  PetscScalar       *y;
  PetscErrorCode    ierr;
  PetscBLASInt      m, n,_One=1;
  PetscScalar       _DOne=1.0,_DZero=0.0;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(A->rmap->n,&m);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(A->cmap->n,&n);CHKERRQ(ierr);
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArrayWrite(yy,&y);CHKERRQ(ierr);
  if (!A->rmap->n || !A->cmap->n) {
    PetscBLASInt i;
    for (i=0; i<n; i++) y[i] = 0.0;
  } else {
    PetscStackCallBLAS("BLASgemv",BLASgemv_("T",&m,&n,&_DOne,v,&mat->lda,x,&_One,&_DZero,y,&_One));
    ierr = PetscLogFlops(2.0*A->rmap->n*A->cmap->n - A->cmap->n);CHKERRQ(ierr);
  }
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayWrite(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMult_SeqDense(Mat A,Vec xx,Vec yy)
{
  Mat_SeqDense      *mat = (Mat_SeqDense*)A->data;
  PetscScalar       *y,_DOne=1.0,_DZero=0.0;
  PetscErrorCode    ierr;
  PetscBLASInt      m, n, _One=1;
  const PetscScalar *v = mat->v,*x;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(A->rmap->n,&m);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(A->cmap->n,&n);CHKERRQ(ierr);
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArrayWrite(yy,&y);CHKERRQ(ierr);
  if (!A->rmap->n || !A->cmap->n) {
    PetscBLASInt i;
    for (i=0; i<m; i++) y[i] = 0.0;
  } else {
    PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&m,&n,&_DOne,v,&(mat->lda),x,&_One,&_DZero,y,&_One));
    ierr = PetscLogFlops(2.0*A->rmap->n*A->cmap->n - A->rmap->n);CHKERRQ(ierr);
  }
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayWrite(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultAdd_SeqDense(Mat A,Vec xx,Vec zz,Vec yy)
{
  Mat_SeqDense      *mat = (Mat_SeqDense*)A->data;
  const PetscScalar *v = mat->v,*x;
  PetscScalar       *y,_DOne=1.0;
  PetscErrorCode    ierr;
  PetscBLASInt      m, n, _One=1;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(A->rmap->n,&m);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(A->cmap->n,&n);CHKERRQ(ierr);
  ierr = VecCopy(zz,yy);CHKERRQ(ierr);
  if (!A->rmap->n || !A->cmap->n) PetscFunctionReturn(0);
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&m,&n,&_DOne,v,&(mat->lda),x,&_One,&_DOne,y,&_One));
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*A->rmap->n*A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultTransposeAdd_SeqDense(Mat A,Vec xx,Vec zz,Vec yy)
{
  Mat_SeqDense      *mat = (Mat_SeqDense*)A->data;
  const PetscScalar *v = mat->v,*x;
  PetscScalar       *y;
  PetscErrorCode    ierr;
  PetscBLASInt      m, n, _One=1;
  PetscScalar       _DOne=1.0;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(A->rmap->n,&m);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(A->cmap->n,&n);CHKERRQ(ierr);
  ierr = VecCopy(zz,yy);CHKERRQ(ierr);
  if (!A->rmap->n || !A->cmap->n) PetscFunctionReturn(0);
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  PetscStackCallBLAS("BLASgemv",BLASgemv_("T",&m,&n,&_DOne,v,&(mat->lda),x,&_One,&_DOne,y,&_One));
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*A->rmap->n*A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -----------------------------------------------------------------*/
static PetscErrorCode MatGetRow_SeqDense(Mat A,PetscInt row,PetscInt *ncols,PetscInt **cols,PetscScalar **vals)
{
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  *ncols = A->cmap->n;
  if (cols) {
    ierr = PetscMalloc1(A->cmap->n,cols);CHKERRQ(ierr);
    for (i=0; i<A->cmap->n; i++) (*cols)[i] = i;
  }
  if (vals) {
    const PetscScalar *v;

    ierr = MatDenseGetArrayRead(A,&v);CHKERRQ(ierr);
    ierr = PetscMalloc1(A->cmap->n,vals);CHKERRQ(ierr);
    v   += row;
    for (i=0; i<A->cmap->n; i++) {(*vals)[i] = *v; v += mat->lda;}
    ierr = MatDenseRestoreArrayRead(A,&v);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatRestoreRow_SeqDense(Mat A,PetscInt row,PetscInt *ncols,PetscInt **cols,PetscScalar **vals)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ncols) *ncols = 0;
  if (cols) {ierr = PetscFree(*cols);CHKERRQ(ierr);}
  if (vals) {ierr = PetscFree(*vals);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}
/* ----------------------------------------------------------------*/
static PetscErrorCode MatSetValues_SeqDense(Mat A,PetscInt m,const PetscInt indexm[],PetscInt n,const PetscInt indexn[],const PetscScalar v[],InsertMode addv)
{
  Mat_SeqDense     *mat = (Mat_SeqDense*)A->data;
  PetscScalar      *av;
  PetscInt         i,j,idx=0;
#if defined(PETSC_HAVE_CUDA)
  PetscOffloadMask oldf;
#endif
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = MatDenseGetArray(A,&av);CHKERRQ(ierr);
  if (!mat->roworiented) {
    if (addv == INSERT_VALUES) {
      for (j=0; j<n; j++) {
        if (indexn[j] < 0) {idx += m; continue;}
        if (PetscUnlikelyDebug(indexn[j] >= A->cmap->n)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Column too large: col %D max %D",indexn[j],A->cmap->n-1);
        for (i=0; i<m; i++) {
          if (indexm[i] < 0) {idx++; continue;}
          if (PetscUnlikelyDebug(indexm[i] >= A->rmap->n)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %D max %D",indexm[i],A->rmap->n-1);
          av[indexn[j]*mat->lda + indexm[i]] = v[idx++];
        }
      }
    } else {
      for (j=0; j<n; j++) {
        if (indexn[j] < 0) {idx += m; continue;}
        if (PetscUnlikelyDebug(indexn[j] >= A->cmap->n)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Column too large: col %D max %D",indexn[j],A->cmap->n-1);
        for (i=0; i<m; i++) {
          if (indexm[i] < 0) {idx++; continue;}
          if (PetscUnlikelyDebug(indexm[i] >= A->rmap->n)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %D max %D",indexm[i],A->rmap->n-1);
          av[indexn[j]*mat->lda + indexm[i]] += v[idx++];
        }
      }
    }
  } else {
    if (addv == INSERT_VALUES) {
      for (i=0; i<m; i++) {
        if (indexm[i] < 0) { idx += n; continue;}
        if (PetscUnlikelyDebug(indexm[i] >= A->rmap->n)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %D max %D",indexm[i],A->rmap->n-1);
        for (j=0; j<n; j++) {
          if (indexn[j] < 0) { idx++; continue;}
          if (PetscUnlikelyDebug(indexn[j] >= A->cmap->n)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Column too large: col %D max %D",indexn[j],A->cmap->n-1);
          av[indexn[j]*mat->lda + indexm[i]] = v[idx++];
        }
      }
    } else {
      for (i=0; i<m; i++) {
        if (indexm[i] < 0) { idx += n; continue;}
        if (PetscUnlikelyDebug(indexm[i] >= A->rmap->n)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %D max %D",indexm[i],A->rmap->n-1);
        for (j=0; j<n; j++) {
          if (indexn[j] < 0) { idx++; continue;}
          if (PetscUnlikelyDebug(indexn[j] >= A->cmap->n)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Column too large: col %D max %D",indexn[j],A->cmap->n-1);
          av[indexn[j]*mat->lda + indexm[i]] += v[idx++];
        }
      }
    }
  }
  /* hack to prevent unneeded copy to the GPU while returning the array */
#if defined(PETSC_HAVE_CUDA)
  oldf = A->offloadmask;
  A->offloadmask = PETSC_OFFLOAD_GPU;
#endif
  ierr = MatDenseRestoreArray(A,&av);CHKERRQ(ierr);
#if defined(PETSC_HAVE_CUDA)
  A->offloadmask = (oldf == PETSC_OFFLOAD_UNALLOCATED ? PETSC_OFFLOAD_UNALLOCATED : PETSC_OFFLOAD_CPU);
#endif
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetValues_SeqDense(Mat A,PetscInt m,const PetscInt indexm[],PetscInt n,const PetscInt indexn[],PetscScalar v[])
{
  Mat_SeqDense      *mat = (Mat_SeqDense*)A->data;
  const PetscScalar *vv;
  PetscInt          i,j;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatDenseGetArrayRead(A,&vv);CHKERRQ(ierr);
  /* row-oriented output */
  for (i=0; i<m; i++) {
    if (indexm[i] < 0) {v += n;continue;}
    if (indexm[i] >= A->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row %D requested larger than number rows %D",indexm[i],A->rmap->n);
    for (j=0; j<n; j++) {
      if (indexn[j] < 0) {v++; continue;}
      if (indexn[j] >= A->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Column %D requested larger than number columns %D",indexn[j],A->cmap->n);
      *v++ = vv[indexn[j]*mat->lda + indexm[i]];
    }
  }
  ierr = MatDenseRestoreArrayRead(A,&vv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -----------------------------------------------------------------*/

PetscErrorCode MatView_Dense_Binary(Mat mat,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscBool         skipHeader;
  PetscViewerFormat format;
  PetscInt          header[4],M,N,m,lda,i,j,k;
  const PetscScalar *v;
  PetscScalar       *vwork;

  PetscFunctionBegin;
  ierr = PetscViewerSetUp(viewer);CHKERRQ(ierr);
  ierr = PetscViewerBinaryGetSkipHeader(viewer,&skipHeader);CHKERRQ(ierr);
  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  if (skipHeader) format = PETSC_VIEWER_NATIVE;

  ierr = MatGetSize(mat,&M,&N);CHKERRQ(ierr);

  /* write matrix header */
  header[0] = MAT_FILE_CLASSID; header[1] = M; header[2] = N;
  header[3] = (format == PETSC_VIEWER_NATIVE) ? MATRIX_BINARY_FORMAT_DENSE : M*N;
  if (!skipHeader) {ierr = PetscViewerBinaryWrite(viewer,header,4,PETSC_INT);CHKERRQ(ierr);}

  ierr = MatGetLocalSize(mat,&m,NULL);CHKERRQ(ierr);
  if (format != PETSC_VIEWER_NATIVE) {
    PetscInt nnz = m*N, *iwork;
    /* store row lengths for each row */
    ierr = PetscMalloc1(nnz,&iwork);CHKERRQ(ierr);
    for (i=0; i<m; i++) iwork[i] = N;
    ierr = PetscViewerBinaryWriteAll(viewer,iwork,m,PETSC_DETERMINE,PETSC_DETERMINE,PETSC_INT);CHKERRQ(ierr);
    /* store column indices (zero start index) */
    for (k=0, i=0; i<m; i++)
      for (j=0; j<N; j++, k++)
        iwork[k] = j;
    ierr = PetscViewerBinaryWriteAll(viewer,iwork,nnz,PETSC_DETERMINE,PETSC_DETERMINE,PETSC_INT);CHKERRQ(ierr);
    ierr = PetscFree(iwork);CHKERRQ(ierr);
  }
  /* store matrix values as a dense matrix in row major order */
  ierr = PetscMalloc1(m*N,&vwork);CHKERRQ(ierr);
  ierr = MatDenseGetArrayRead(mat,&v);CHKERRQ(ierr);
  ierr = MatDenseGetLDA(mat,&lda);CHKERRQ(ierr);
  for (k=0, i=0; i<m; i++)
    for (j=0; j<N; j++, k++)
      vwork[k] = v[i+lda*j];
  ierr = MatDenseRestoreArrayRead(mat,&v);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWriteAll(viewer,vwork,m*N,PETSC_DETERMINE,PETSC_DETERMINE,PETSC_SCALAR);CHKERRQ(ierr);
  ierr = PetscFree(vwork);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatLoad_Dense_Binary(Mat mat,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      skipHeader;
  PetscInt       header[4],M,N,m,nz,lda,i,j,k;
  PetscInt       rows,cols;
  PetscScalar    *v,*vwork;

  PetscFunctionBegin;
  ierr = PetscViewerSetUp(viewer);CHKERRQ(ierr);
  ierr = PetscViewerBinaryGetSkipHeader(viewer,&skipHeader);CHKERRQ(ierr);

  if (!skipHeader) {
    ierr = PetscViewerBinaryRead(viewer,header,4,NULL,PETSC_INT);CHKERRQ(ierr);
    if (header[0] != MAT_FILE_CLASSID) SETERRQ(PetscObjectComm((PetscObject)viewer),PETSC_ERR_FILE_UNEXPECTED,"Not a matrix object in file");
    M = header[1]; N = header[2];
    if (M < 0) SETERRQ1(PetscObjectComm((PetscObject)viewer),PETSC_ERR_FILE_UNEXPECTED,"Matrix row size (%D) in file is negative",M);
    if (N < 0) SETERRQ1(PetscObjectComm((PetscObject)viewer),PETSC_ERR_FILE_UNEXPECTED,"Matrix column size (%D) in file is negative",N);
    nz = header[3];
    if (nz != MATRIX_BINARY_FORMAT_DENSE && nz < 0) SETERRQ1(PetscObjectComm((PetscObject)viewer),PETSC_ERR_FILE_UNEXPECTED,"Unknown matrix format %D in file",nz);
  } else {
    ierr = MatGetSize(mat,&M,&N);CHKERRQ(ierr);
    if (M < 0 || N < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Matrix binary file header was skipped, thus the user must specify the global sizes of input matrix");
    nz = MATRIX_BINARY_FORMAT_DENSE;
  }

  /* setup global sizes if not set */
  if (mat->rmap->N < 0) mat->rmap->N = M;
  if (mat->cmap->N < 0) mat->cmap->N = N;
  ierr = MatSetUp(mat);CHKERRQ(ierr);
  /* check if global sizes are correct */
  ierr = MatGetSize(mat,&rows,&cols);CHKERRQ(ierr);
  if (M != rows || N != cols) SETERRQ4(PetscObjectComm((PetscObject)viewer),PETSC_ERR_FILE_UNEXPECTED, "Matrix in file of different sizes (%d, %d) than the input matrix (%d, %d)",M,N,rows,cols);

  ierr = MatGetSize(mat,NULL,&N);CHKERRQ(ierr);
  ierr = MatGetLocalSize(mat,&m,NULL);CHKERRQ(ierr);
  ierr = MatDenseGetArray(mat,&v);CHKERRQ(ierr);
  ierr = MatDenseGetLDA(mat,&lda);CHKERRQ(ierr);
  if (nz == MATRIX_BINARY_FORMAT_DENSE) {  /* matrix in file is dense format */
    PetscInt nnz = m*N;
    /* read in matrix values */
    ierr = PetscMalloc1(nnz,&vwork);CHKERRQ(ierr);
    ierr = PetscViewerBinaryReadAll(viewer,vwork,nnz,PETSC_DETERMINE,PETSC_DETERMINE,PETSC_SCALAR);CHKERRQ(ierr);
    /* store values in column major order */
    for (j=0; j<N; j++)
      for (i=0; i<m; i++)
        v[i+lda*j] = vwork[i*N+j];
    ierr = PetscFree(vwork);CHKERRQ(ierr);
  } else { /* matrix in file is sparse format */
    PetscInt nnz = 0, *rlens, *icols;
    /* read in row lengths */
    ierr = PetscMalloc1(m,&rlens);CHKERRQ(ierr);
    ierr = PetscViewerBinaryReadAll(viewer,rlens,m,PETSC_DETERMINE,PETSC_DETERMINE,PETSC_INT);CHKERRQ(ierr);
    for (i=0; i<m; i++) nnz += rlens[i];
    /* read in column indices and values */
    ierr = PetscMalloc2(nnz,&icols,nnz,&vwork);CHKERRQ(ierr);
    ierr = PetscViewerBinaryReadAll(viewer,icols,nnz,PETSC_DETERMINE,PETSC_DETERMINE,PETSC_INT);CHKERRQ(ierr);
    ierr = PetscViewerBinaryReadAll(viewer,vwork,nnz,PETSC_DETERMINE,PETSC_DETERMINE,PETSC_SCALAR);CHKERRQ(ierr);
    /* store values in column major order */
    for (k=0, i=0; i<m; i++)
      for (j=0; j<rlens[i]; j++, k++)
        v[i+lda*icols[k]] = vwork[k];
    ierr = PetscFree(rlens);CHKERRQ(ierr);
    ierr = PetscFree2(icols,vwork);CHKERRQ(ierr);
  }
  ierr = MatDenseRestoreArray(mat,&v);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatLoad_SeqDense(Mat newMat, PetscViewer viewer)
{
  PetscBool      isbinary, ishdf5;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(newMat,MAT_CLASSID,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  /* force binary viewer to load .info file if it has not yet done so */
  ierr = PetscViewerSetUp(viewer);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERHDF5,  &ishdf5);CHKERRQ(ierr);
  if (isbinary) {
    ierr = MatLoad_Dense_Binary(newMat,viewer);CHKERRQ(ierr);
  } else if (ishdf5) {
#if defined(PETSC_HAVE_HDF5)
    ierr = MatLoad_Dense_HDF5(newMat,viewer);CHKERRQ(ierr);
#else
    SETERRQ(PetscObjectComm((PetscObject)newMat),PETSC_ERR_SUP,"HDF5 not supported in this build.\nPlease reconfigure using --download-hdf5");
#endif
  } else {
    SETERRQ2(PetscObjectComm((PetscObject)newMat),PETSC_ERR_SUP,"Viewer type %s not yet supported for reading %s matrices",((PetscObject)viewer)->type_name,((PetscObject)newMat)->type_name);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatView_SeqDense_ASCII(Mat A,PetscViewer viewer)
{
  Mat_SeqDense      *a = (Mat_SeqDense*)A->data;
  PetscErrorCode    ierr;
  PetscInt          i,j;
  const char        *name;
  PetscScalar       *v,*av;
  PetscViewerFormat format;
#if defined(PETSC_USE_COMPLEX)
  PetscBool         allreal = PETSC_TRUE;
#endif

  PetscFunctionBegin;
  ierr = MatDenseGetArrayRead(A,(const PetscScalar**)&av);CHKERRQ(ierr);
  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
    PetscFunctionReturn(0);  /* do nothing for now */
  } else if (format == PETSC_VIEWER_ASCII_COMMON) {
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
    for (i=0; i<A->rmap->n; i++) {
      v    = av + i;
      ierr = PetscViewerASCIIPrintf(viewer,"row %D:",i);CHKERRQ(ierr);
      for (j=0; j<A->cmap->n; j++) {
#if defined(PETSC_USE_COMPLEX)
        if (PetscRealPart(*v) != 0.0 && PetscImaginaryPart(*v) != 0.0) {
          ierr = PetscViewerASCIIPrintf(viewer," (%D, %g + %g i) ",j,(double)PetscRealPart(*v),(double)PetscImaginaryPart(*v));CHKERRQ(ierr);
        } else if (PetscRealPart(*v)) {
          ierr = PetscViewerASCIIPrintf(viewer," (%D, %g) ",j,(double)PetscRealPart(*v));CHKERRQ(ierr);
        }
#else
        if (*v) {
          ierr = PetscViewerASCIIPrintf(viewer," (%D, %g) ",j,(double)*v);CHKERRQ(ierr);
        }
#endif
        v += a->lda;
      }
      ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
    /* determine if matrix has all real values */
    v = av;
    for (i=0; i<A->rmap->n*A->cmap->n; i++) {
      if (PetscImaginaryPart(v[i])) { allreal = PETSC_FALSE; break;}
    }
#endif
    if (format == PETSC_VIEWER_ASCII_MATLAB) {
      ierr = PetscObjectGetName((PetscObject)A,&name);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"%% Size = %D %D \n",A->rmap->n,A->cmap->n);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"%s = zeros(%D,%D);\n",name,A->rmap->n,A->cmap->n);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"%s = [\n",name);CHKERRQ(ierr);
    }

    for (i=0; i<A->rmap->n; i++) {
      v = av + i;
      for (j=0; j<A->cmap->n; j++) {
#if defined(PETSC_USE_COMPLEX)
        if (allreal) {
          ierr = PetscViewerASCIIPrintf(viewer,"%18.16e ",(double)PetscRealPart(*v));CHKERRQ(ierr);
        } else {
          ierr = PetscViewerASCIIPrintf(viewer,"%18.16e + %18.16ei ",(double)PetscRealPart(*v),(double)PetscImaginaryPart(*v));CHKERRQ(ierr);
        }
#else
        ierr = PetscViewerASCIIPrintf(viewer,"%18.16e ",(double)*v);CHKERRQ(ierr);
#endif
        v += a->lda;
      }
      ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
    }
    if (format == PETSC_VIEWER_ASCII_MATLAB) {
      ierr = PetscViewerASCIIPrintf(viewer,"];\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
  }
  ierr = MatDenseRestoreArrayRead(A,(const PetscScalar**)&av);CHKERRQ(ierr);
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include <petscdraw.h>
static PetscErrorCode MatView_SeqDense_Draw_Zoom(PetscDraw draw,void *Aa)
{
  Mat               A  = (Mat) Aa;
  PetscErrorCode    ierr;
  PetscInt          m  = A->rmap->n,n = A->cmap->n,i,j;
  int               color = PETSC_DRAW_WHITE;
  const PetscScalar *v;
  PetscViewer       viewer;
  PetscReal         xl,yl,xr,yr,x_l,x_r,y_l,y_r;
  PetscViewerFormat format;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)A,"Zoomviewer",(PetscObject*)&viewer);CHKERRQ(ierr);
  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  ierr = PetscDrawGetCoordinates(draw,&xl,&yl,&xr,&yr);CHKERRQ(ierr);

  /* Loop over matrix elements drawing boxes */
  ierr = MatDenseGetArrayRead(A,&v);CHKERRQ(ierr);
  if (format != PETSC_VIEWER_DRAW_CONTOUR) {
    ierr = PetscDrawCollectiveBegin(draw);CHKERRQ(ierr);
    /* Blue for negative and Red for positive */
    for (j = 0; j < n; j++) {
      x_l = j; x_r = x_l + 1.0;
      for (i = 0; i < m; i++) {
        y_l = m - i - 1.0;
        y_r = y_l + 1.0;
        if (PetscRealPart(v[j*m+i]) >  0.) color = PETSC_DRAW_RED;
        else if (PetscRealPart(v[j*m+i]) <  0.) color = PETSC_DRAW_BLUE;
        else continue;
        ierr = PetscDrawRectangle(draw,x_l,y_l,x_r,y_r,color,color,color,color);CHKERRQ(ierr);
      }
    }
    ierr = PetscDrawCollectiveEnd(draw);CHKERRQ(ierr);
  } else {
    /* use contour shading to indicate magnitude of values */
    /* first determine max of all nonzero values */
    PetscReal minv = 0.0, maxv = 0.0;
    PetscDraw popup;

    for (i=0; i < m*n; i++) {
      if (PetscAbsScalar(v[i]) > maxv) maxv = PetscAbsScalar(v[i]);
    }
    if (minv >= maxv) maxv = minv + PETSC_SMALL;
    ierr = PetscDrawGetPopup(draw,&popup);CHKERRQ(ierr);
    ierr = PetscDrawScalePopup(popup,minv,maxv);CHKERRQ(ierr);

    ierr = PetscDrawCollectiveBegin(draw);CHKERRQ(ierr);
    for (j=0; j<n; j++) {
      x_l = j;
      x_r = x_l + 1.0;
      for (i=0; i<m; i++) {
        y_l   = m - i - 1.0;
        y_r   = y_l + 1.0;
        color = PetscDrawRealToColor(PetscAbsScalar(v[j*m+i]),minv,maxv);
        ierr  = PetscDrawRectangle(draw,x_l,y_l,x_r,y_r,color,color,color,color);CHKERRQ(ierr);
      }
    }
    ierr = PetscDrawCollectiveEnd(draw);CHKERRQ(ierr);
  }
  ierr = MatDenseRestoreArrayRead(A,&v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatView_SeqDense_Draw(Mat A,PetscViewer viewer)
{
  PetscDraw      draw;
  PetscBool      isnull;
  PetscReal      xr,yr,xl,yl,h,w;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
  ierr = PetscDrawIsNull(draw,&isnull);CHKERRQ(ierr);
  if (isnull) PetscFunctionReturn(0);

  xr   = A->cmap->n; yr = A->rmap->n; h = yr/10.0; w = xr/10.0;
  xr  += w;          yr += h;        xl = -w;     yl = -h;
  ierr = PetscDrawSetCoordinates(draw,xl,yl,xr,yr);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)A,"Zoomviewer",(PetscObject)viewer);CHKERRQ(ierr);
  ierr = PetscDrawZoom(draw,MatView_SeqDense_Draw_Zoom,A);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)A,"Zoomviewer",NULL);CHKERRQ(ierr);
  ierr = PetscDrawSave(draw);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatView_SeqDense(Mat A,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      iascii,isbinary,isdraw;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&isdraw);CHKERRQ(ierr);

  if (iascii) {
    ierr = MatView_SeqDense_ASCII(A,viewer);CHKERRQ(ierr);
  } else if (isbinary) {
    ierr = MatView_Dense_Binary(A,viewer);CHKERRQ(ierr);
  } else if (isdraw) {
    ierr = MatView_SeqDense_Draw(A,viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDensePlaceArray_SeqDense(Mat A,const PetscScalar *array)
{
  Mat_SeqDense *a = (Mat_SeqDense*)A->data;

  PetscFunctionBegin;
  if (a->vecinuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  if (a->matinuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  if (a->unplacedarray) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreArray() first");
  a->unplacedarray       = a->v;
  a->unplaced_user_alloc = a->user_alloc;
  a->v                   = (PetscScalar*) array;
  a->user_alloc          = PETSC_TRUE;
#if defined(PETSC_HAVE_CUDA)
  A->offloadmask = PETSC_OFFLOAD_CPU;
#endif
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseResetArray_SeqDense(Mat A)
{
  Mat_SeqDense *a = (Mat_SeqDense*)A->data;

  PetscFunctionBegin;
  if (a->vecinuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  if (a->matinuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  a->v             = a->unplacedarray;
  a->user_alloc    = a->unplaced_user_alloc;
  a->unplacedarray = NULL;
#if defined(PETSC_HAVE_CUDA)
  A->offloadmask = PETSC_OFFLOAD_CPU;
#endif
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseReplaceArray_SeqDense(Mat A,const PetscScalar *array)
{
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (a->vecinuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  if (a->matinuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  if (!a->user_alloc) { ierr = PetscFree(a->v);CHKERRQ(ierr); }
  a->v           = (PetscScalar*) array;
  a->user_alloc  = PETSC_FALSE;
#if defined(PETSC_HAVE_CUDA)
  A->offloadmask = PETSC_OFFLOAD_CPU;
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_SeqDense(Mat mat)
{
  Mat_SeqDense   *l = (Mat_SeqDense*)mat->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if defined(PETSC_USE_LOG)
  PetscLogObjectState((PetscObject)mat,"Rows %D Cols %D",mat->rmap->n,mat->cmap->n);
#endif
  ierr = VecDestroy(&(l->qrrhs));CHKERRQ(ierr);
  ierr = PetscFree(l->tau);CHKERRQ(ierr);
  ierr = PetscFree(l->pivots);CHKERRQ(ierr);
  ierr = PetscFree(l->fwork);CHKERRQ(ierr);
  ierr = MatDestroy(&l->ptapwork);CHKERRQ(ierr);
  if (!l->user_alloc) {ierr = PetscFree(l->v);CHKERRQ(ierr);}
  if (!l->unplaced_user_alloc) {ierr = PetscFree(l->unplacedarray);CHKERRQ(ierr);}
  if (l->vecinuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  if (l->matinuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  ierr = VecDestroy(&l->cvec);CHKERRQ(ierr);
  ierr = MatDestroy(&l->cmat);CHKERRQ(ierr);
  ierr = PetscFree(mat->data);CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)mat,NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatQRFactor_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetLDA_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseSetLDA_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetArray_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseRestoreArray_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDensePlaceArray_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseResetArray_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseReplaceArray_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetArrayRead_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseRestoreArrayRead_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetArrayWrite_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseRestoreArrayWrite_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatConvert_seqdense_seqaij_C",NULL);CHKERRQ(ierr);
#if defined(PETSC_HAVE_ELEMENTAL)
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatConvert_seqdense_elemental_C",NULL);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_SCALAPACK)
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatConvert_seqdense_scalapack_C",NULL);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_CUDA)
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatConvert_seqdense_seqdensecuda_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatProductSetFromOptions_seqdensecuda_seqdensecuda_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatProductSetFromOptions_seqdensecuda_seqdense_C",NULL);CHKERRQ(ierr);
#endif
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatSeqDenseSetPreallocation_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatProductSetFromOptions_seqaij_seqdense_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatProductSetFromOptions_seqdense_seqdense_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatProductSetFromOptions_seqbaij_seqdense_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatProductSetFromOptions_seqsbaij_seqdense_C",NULL);CHKERRQ(ierr);

  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetColumn_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseRestoreColumn_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetColumnVec_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseRestoreColumnVec_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetColumnVecRead_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseRestoreColumnVecRead_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetColumnVecWrite_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseRestoreColumnVecWrite_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetSubMatrix_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseRestoreSubMatrix_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatTranspose_SeqDense(Mat A,MatReuse reuse,Mat *matout)
{
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;
  PetscInt       k,j,m = A->rmap->n, M = mat->lda, n = A->cmap->n;
  PetscScalar    *v,tmp;

  PetscFunctionBegin;
  if (reuse == MAT_INPLACE_MATRIX) {
    if (m == n) { /* in place transpose */
      ierr = MatDenseGetArray(A,&v);CHKERRQ(ierr);
      for (j=0; j<m; j++) {
        for (k=0; k<j; k++) {
          tmp        = v[j + k*M];
          v[j + k*M] = v[k + j*M];
          v[k + j*M] = tmp;
        }
      }
      ierr = MatDenseRestoreArray(A,&v);CHKERRQ(ierr);
    } else { /* reuse memory, temporary allocates new memory */
      PetscScalar *v2;
      PetscLayout tmplayout;

      ierr = PetscMalloc1((size_t)m*n,&v2);CHKERRQ(ierr);
      ierr = MatDenseGetArray(A,&v);CHKERRQ(ierr);
      for (j=0; j<n; j++) {
        for (k=0; k<m; k++) v2[j + (size_t)k*n] = v[k + (size_t)j*M];
      }
      ierr = PetscArraycpy(v,v2,(size_t)m*n);CHKERRQ(ierr);
      ierr = PetscFree(v2);CHKERRQ(ierr);
      ierr = MatDenseRestoreArray(A,&v);CHKERRQ(ierr);
      /* cleanup size dependent quantities */
      ierr = VecDestroy(&mat->cvec);CHKERRQ(ierr);
      ierr = MatDestroy(&mat->cmat);CHKERRQ(ierr);
      ierr = PetscFree(mat->pivots);CHKERRQ(ierr);
      ierr = PetscFree(mat->fwork);CHKERRQ(ierr);
      ierr = MatDestroy(&mat->ptapwork);CHKERRQ(ierr);
      /* swap row/col layouts */
      mat->lda  = n;
      tmplayout = A->rmap;
      A->rmap   = A->cmap;
      A->cmap   = tmplayout;
    }
  } else { /* out-of-place transpose */
    Mat          tmat;
    Mat_SeqDense *tmatd;
    PetscScalar  *v2;
    PetscInt     M2;

    if (reuse == MAT_INITIAL_MATRIX) {
      ierr = MatCreate(PetscObjectComm((PetscObject)A),&tmat);CHKERRQ(ierr);
      ierr = MatSetSizes(tmat,A->cmap->n,A->rmap->n,A->cmap->n,A->rmap->n);CHKERRQ(ierr);
      ierr = MatSetType(tmat,((PetscObject)A)->type_name);CHKERRQ(ierr);
      ierr = MatSeqDenseSetPreallocation(tmat,NULL);CHKERRQ(ierr);
    } else tmat = *matout;

    ierr  = MatDenseGetArrayRead(A,(const PetscScalar**)&v);CHKERRQ(ierr);
    ierr  = MatDenseGetArray(tmat,&v2);CHKERRQ(ierr);
    tmatd = (Mat_SeqDense*)tmat->data;
    M2    = tmatd->lda;
    for (j=0; j<n; j++) {
      for (k=0; k<m; k++) v2[j + k*M2] = v[k + j*M];
    }
    ierr = MatDenseRestoreArray(tmat,&v2);CHKERRQ(ierr);
    ierr = MatDenseRestoreArrayRead(A,(const PetscScalar**)&v);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(tmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(tmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    *matout = tmat;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatEqual_SeqDense(Mat A1,Mat A2,PetscBool  *flg)
{
  Mat_SeqDense      *mat1 = (Mat_SeqDense*)A1->data;
  Mat_SeqDense      *mat2 = (Mat_SeqDense*)A2->data;
  PetscInt          i;
  const PetscScalar *v1,*v2;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (A1->rmap->n != A2->rmap->n) {*flg = PETSC_FALSE; PetscFunctionReturn(0);}
  if (A1->cmap->n != A2->cmap->n) {*flg = PETSC_FALSE; PetscFunctionReturn(0);}
  ierr = MatDenseGetArrayRead(A1,&v1);CHKERRQ(ierr);
  ierr = MatDenseGetArrayRead(A2,&v2);CHKERRQ(ierr);
  for (i=0; i<A1->cmap->n; i++) {
    ierr = PetscArraycmp(v1,v2,A1->rmap->n,flg);CHKERRQ(ierr);
    if (*flg == PETSC_FALSE) PetscFunctionReturn(0);
    v1 += mat1->lda;
    v2 += mat2->lda;
  }
  ierr = MatDenseRestoreArrayRead(A1,&v1);CHKERRQ(ierr);
  ierr = MatDenseRestoreArrayRead(A2,&v2);CHKERRQ(ierr);
  *flg = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetDiagonal_SeqDense(Mat A,Vec v)
{
  Mat_SeqDense      *mat = (Mat_SeqDense*)A->data;
  PetscInt          i,n,len;
  PetscScalar       *x;
  const PetscScalar *vv;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = VecGetSize(v,&n);CHKERRQ(ierr);
  ierr = VecGetArray(v,&x);CHKERRQ(ierr);
  len  = PetscMin(A->rmap->n,A->cmap->n);
  ierr = MatDenseGetArrayRead(A,&vv);CHKERRQ(ierr);
  if (n != A->rmap->n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Nonconforming mat and vec");
  for (i=0; i<len; i++) {
    x[i] = vv[i*mat->lda + i];
  }
  ierr = MatDenseRestoreArrayRead(A,&vv);CHKERRQ(ierr);
  ierr = VecRestoreArray(v,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDiagonalScale_SeqDense(Mat A,Vec ll,Vec rr)
{
  Mat_SeqDense      *mat = (Mat_SeqDense*)A->data;
  const PetscScalar *l,*r;
  PetscScalar       x,*v,*vv;
  PetscErrorCode    ierr;
  PetscInt          i,j,m = A->rmap->n,n = A->cmap->n;

  PetscFunctionBegin;
  ierr = MatDenseGetArray(A,&vv);CHKERRQ(ierr);
  if (ll) {
    ierr = VecGetSize(ll,&m);CHKERRQ(ierr);
    ierr = VecGetArrayRead(ll,&l);CHKERRQ(ierr);
    if (m != A->rmap->n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Left scaling vec wrong size");
    for (i=0; i<m; i++) {
      x = l[i];
      v = vv + i;
      for (j=0; j<n; j++) { (*v) *= x; v+= mat->lda;}
    }
    ierr = VecRestoreArrayRead(ll,&l);CHKERRQ(ierr);
    ierr = PetscLogFlops(1.0*n*m);CHKERRQ(ierr);
  }
  if (rr) {
    ierr = VecGetSize(rr,&n);CHKERRQ(ierr);
    ierr = VecGetArrayRead(rr,&r);CHKERRQ(ierr);
    if (n != A->cmap->n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Right scaling vec wrong size");
    for (i=0; i<n; i++) {
      x = r[i];
      v = vv + i*mat->lda;
      for (j=0; j<m; j++) (*v++) *= x;
    }
    ierr = VecRestoreArrayRead(rr,&r);CHKERRQ(ierr);
    ierr = PetscLogFlops(1.0*n*m);CHKERRQ(ierr);
  }
  ierr = MatDenseRestoreArray(A,&vv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatNorm_SeqDense(Mat A,NormType type,PetscReal *nrm)
{
  Mat_SeqDense      *mat = (Mat_SeqDense*)A->data;
  PetscScalar       *v,*vv;
  PetscReal         sum  = 0.0;
  PetscInt          lda  =mat->lda,m=A->rmap->n,i,j;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatDenseGetArrayRead(A,(const PetscScalar**)&vv);CHKERRQ(ierr);
  v    = vv;
  if (type == NORM_FROBENIUS) {
    if (lda>m) {
      for (j=0; j<A->cmap->n; j++) {
        v = vv+j*lda;
        for (i=0; i<m; i++) {
          sum += PetscRealPart(PetscConj(*v)*(*v)); v++;
        }
      }
    } else {
#if defined(PETSC_USE_REAL___FP16)
      PetscBLASInt one = 1,cnt = A->cmap->n*A->rmap->n;
      PetscStackCallBLAS("BLASnrm2",*nrm = BLASnrm2_(&cnt,v,&one));
    }
#else
      for (i=0; i<A->cmap->n*A->rmap->n; i++) {
        sum += PetscRealPart(PetscConj(*v)*(*v)); v++;
      }
    }
    *nrm = PetscSqrtReal(sum);
#endif
    ierr = PetscLogFlops(2.0*A->cmap->n*A->rmap->n);CHKERRQ(ierr);
  } else if (type == NORM_1) {
    *nrm = 0.0;
    for (j=0; j<A->cmap->n; j++) {
      v   = vv + j*mat->lda;
      sum = 0.0;
      for (i=0; i<A->rmap->n; i++) {
        sum += PetscAbsScalar(*v);  v++;
      }
      if (sum > *nrm) *nrm = sum;
    }
    ierr = PetscLogFlops(1.0*A->cmap->n*A->rmap->n);CHKERRQ(ierr);
  } else if (type == NORM_INFINITY) {
    *nrm = 0.0;
    for (j=0; j<A->rmap->n; j++) {
      v   = vv + j;
      sum = 0.0;
      for (i=0; i<A->cmap->n; i++) {
        sum += PetscAbsScalar(*v); v += mat->lda;
      }
      if (sum > *nrm) *nrm = sum;
    }
    ierr = PetscLogFlops(1.0*A->cmap->n*A->rmap->n);CHKERRQ(ierr);
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No two norm");
  ierr = MatDenseRestoreArrayRead(A,(const PetscScalar**)&vv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetOption_SeqDense(Mat A,MatOption op,PetscBool flg)
{
  Mat_SeqDense   *aij = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch (op) {
  case MAT_ROW_ORIENTED:
    aij->roworiented = flg;
    break;
  case MAT_NEW_NONZERO_LOCATIONS:
  case MAT_NEW_NONZERO_LOCATION_ERR:
  case MAT_NEW_NONZERO_ALLOCATION_ERR:
  case MAT_FORCE_DIAGONAL_ENTRIES:
  case MAT_KEEP_NONZERO_PATTERN:
  case MAT_IGNORE_OFF_PROC_ENTRIES:
  case MAT_USE_HASH_TABLE:
  case MAT_IGNORE_ZERO_ENTRIES:
  case MAT_IGNORE_LOWER_TRIANGULAR:
  case MAT_SORTED_FULL:
    ierr = PetscInfo1(A,"Option %s ignored\n",MatOptions[op]);CHKERRQ(ierr);
    break;
  case MAT_SPD:
  case MAT_SYMMETRIC:
  case MAT_STRUCTURALLY_SYMMETRIC:
  case MAT_HERMITIAN:
  case MAT_SYMMETRY_ETERNAL:
    /* These options are handled directly by MatSetOption() */
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"unknown option %s",MatOptions[op]);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatZeroEntries_SeqDense(Mat A)
{
  Mat_SeqDense   *l = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;
  PetscInt       lda=l->lda,m=A->rmap->n,n=A->cmap->n,j;
  PetscScalar    *v;

  PetscFunctionBegin;
  ierr = MatDenseGetArrayWrite(A,&v);CHKERRQ(ierr);
  if (lda>m) {
    for (j=0; j<n; j++) {
      ierr = PetscArrayzero(v+j*lda,m);CHKERRQ(ierr);
    }
  } else {
    ierr = PetscArrayzero(v,PetscInt64Mult(m,n));CHKERRQ(ierr);
  }
  ierr = MatDenseRestoreArrayWrite(A,&v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatZeroRows_SeqDense(Mat A,PetscInt N,const PetscInt rows[],PetscScalar diag,Vec x,Vec b)
{
  PetscErrorCode    ierr;
  Mat_SeqDense      *l = (Mat_SeqDense*)A->data;
  PetscInt          m  = l->lda, n = A->cmap->n, i,j;
  PetscScalar       *slot,*bb,*v;
  const PetscScalar *xx;

  PetscFunctionBegin;
  if (PetscDefined(USE_DEBUG)) {
    for (i=0; i<N; i++) {
      if (rows[i] < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative row requested to be zeroed");
      if (rows[i] >= A->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row %D requested to be zeroed greater than or equal number of rows %D",rows[i],A->rmap->n);
    }
  }
  if (!N) PetscFunctionReturn(0);

  /* fix right hand side if needed */
  if (x && b) {
    ierr = VecGetArrayRead(x,&xx);CHKERRQ(ierr);
    ierr = VecGetArray(b,&bb);CHKERRQ(ierr);
    for (i=0; i<N; i++) bb[rows[i]] = diag*xx[rows[i]];
    ierr = VecRestoreArrayRead(x,&xx);CHKERRQ(ierr);
    ierr = VecRestoreArray(b,&bb);CHKERRQ(ierr);
  }

  ierr = MatDenseGetArray(A,&v);CHKERRQ(ierr);
  for (i=0; i<N; i++) {
    slot = v + rows[i];
    for (j=0; j<n; j++) { *slot = 0.0; slot += m;}
  }
  if (diag != 0.0) {
    if (A->rmap->n != A->cmap->n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only coded for square matrices");
    for (i=0; i<N; i++) {
      slot  = v + (m+1)*rows[i];
      *slot = diag;
    }
  }
  ierr = MatDenseRestoreArray(A,&v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseGetLDA_SeqDense(Mat A,PetscInt *lda)
{
  Mat_SeqDense *mat = (Mat_SeqDense*)A->data;

  PetscFunctionBegin;
  *lda = mat->lda;
  PetscFunctionReturn(0);
}

PetscErrorCode MatDenseGetArray_SeqDense(Mat A,PetscScalar **array)
{
  Mat_SeqDense *mat = (Mat_SeqDense*)A->data;

  PetscFunctionBegin;
  if (mat->matinuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  *array = mat->v;
  PetscFunctionReturn(0);
}

PetscErrorCode MatDenseRestoreArray_SeqDense(Mat A,PetscScalar **array)
{
  PetscFunctionBegin;
  *array = NULL;
  PetscFunctionReturn(0);
}

/*@
   MatDenseGetLDA - gets the leading dimension of the array returned from MatDenseGetArray()

   Not collective

   Input Parameter:
.  mat - a MATSEQDENSE or MATMPIDENSE matrix

   Output Parameter:
.   lda - the leading dimension

   Level: intermediate

.seealso: MatDenseGetArray(), MatDenseRestoreArray(), MatDenseGetArrayRead(), MatDenseRestoreArrayRead(), MatDenseSetLDA()
@*/
PetscErrorCode  MatDenseGetLDA(Mat A,PetscInt *lda)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(lda,2);
  ierr = PetscUseMethod(A,"MatDenseGetLDA_C",(Mat,PetscInt*),(A,lda));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   MatDenseSetLDA - Sets the leading dimension of the array used by the dense matrix

   Not collective

   Input Parameters:
+  mat - a MATSEQDENSE or MATMPIDENSE matrix
-  lda - the leading dimension

   Level: intermediate

.seealso: MatDenseGetArray(), MatDenseRestoreArray(), MatDenseGetArrayRead(), MatDenseRestoreArrayRead(), MatDenseGetLDA()
@*/
PetscErrorCode  MatDenseSetLDA(Mat A,PetscInt lda)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  ierr = PetscTryMethod(A,"MatDenseSetLDA_C",(Mat,PetscInt),(A,lda));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   MatDenseGetArray - gives read-write access to the array where the data for a dense matrix is stored

   Logically Collective on Mat

   Input Parameter:
.  mat - a dense matrix

   Output Parameter:
.   array - pointer to the data

   Level: intermediate

.seealso: MatDenseRestoreArray(), MatDenseGetArrayRead(), MatDenseRestoreArrayRead(), MatDenseGetArrayWrite(), MatDenseRestoreArrayWrite()
@*/
PetscErrorCode  MatDenseGetArray(Mat A,PetscScalar **array)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(array,2);
  ierr = PetscUseMethod(A,"MatDenseGetArray_C",(Mat,PetscScalar**),(A,array));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   MatDenseRestoreArray - returns access to the array where the data for a dense matrix is stored obtained by MatDenseGetArray()

   Logically Collective on Mat

   Input Parameters:
+  mat - a dense matrix
-  array - pointer to the data

   Level: intermediate

.seealso: MatDenseGetArray(), MatDenseGetArrayRead(), MatDenseRestoreArrayRead(), MatDenseGetArrayWrite(), MatDenseRestoreArrayWrite()
@*/
PetscErrorCode  MatDenseRestoreArray(Mat A,PetscScalar **array)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(array,2);
  ierr = PetscUseMethod(A,"MatDenseRestoreArray_C",(Mat,PetscScalar**),(A,array));CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)A);CHKERRQ(ierr);
#if defined(PETSC_HAVE_CUDA)
  A->offloadmask = PETSC_OFFLOAD_CPU;
#endif
  PetscFunctionReturn(0);
}

/*@C
   MatDenseGetArrayRead - gives read-only access to the array where the data for a dense matrix is stored

   Not Collective

   Input Parameter:
.  mat - a dense matrix

   Output Parameter:
.   array - pointer to the data

   Level: intermediate

.seealso: MatDenseRestoreArrayRead(), MatDenseGetArray(), MatDenseRestoreArray(), MatDenseGetArrayWrite(), MatDenseRestoreArrayWrite()
@*/
PetscErrorCode  MatDenseGetArrayRead(Mat A,const PetscScalar **array)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(array,2);
  ierr = PetscUseMethod(A,"MatDenseGetArrayRead_C",(Mat,const PetscScalar**),(A,array));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   MatDenseRestoreArrayRead - returns access to the array where the data for a dense matrix is stored obtained by MatDenseGetArrayRead()

   Not Collective

   Input Parameters:
+  mat - a dense matrix
-  array - pointer to the data

   Level: intermediate

.seealso: MatDenseGetArrayRead(), MatDenseGetArray(), MatDenseRestoreArray(), MatDenseGetArrayWrite(), MatDenseRestoreArrayWrite()
@*/
PetscErrorCode  MatDenseRestoreArrayRead(Mat A,const PetscScalar **array)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(array,2);
  ierr = PetscUseMethod(A,"MatDenseRestoreArrayRead_C",(Mat,const PetscScalar**),(A,array));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   MatDenseGetArrayWrite - gives write-only access to the array where the data for a dense matrix is stored

   Not Collective

   Input Parameter:
.  mat - a dense matrix

   Output Parameter:
.   array - pointer to the data

   Level: intermediate

.seealso: MatDenseRestoreArrayWrite(), MatDenseGetArray(), MatDenseRestoreArray(), MatDenseGetArrayRead(), MatDenseRestoreArrayRead()
@*/
PetscErrorCode  MatDenseGetArrayWrite(Mat A,PetscScalar **array)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(array,2);
  ierr = PetscUseMethod(A,"MatDenseGetArrayWrite_C",(Mat,PetscScalar**),(A,array));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   MatDenseRestoreArrayWrite - returns access to the array where the data for a dense matrix is stored obtained by MatDenseGetArrayWrite()

   Not Collective

   Input Parameters:
+  mat - a dense matrix
-  array - pointer to the data

   Level: intermediate

.seealso: MatDenseGetArrayWrite(), MatDenseGetArray(), MatDenseRestoreArray(), MatDenseGetArrayRead(), MatDenseRestoreArrayRead()
@*/
PetscErrorCode  MatDenseRestoreArrayWrite(Mat A,PetscScalar **array)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(array,2);
  ierr = PetscUseMethod(A,"MatDenseRestoreArrayWrite_C",(Mat,PetscScalar**),(A,array));CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)A);CHKERRQ(ierr);
#if defined(PETSC_HAVE_CUDA)
  A->offloadmask = PETSC_OFFLOAD_CPU;
#endif
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCreateSubMatrix_SeqDense(Mat A,IS isrow,IS iscol,MatReuse scall,Mat *B)
{
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;
  PetscInt       i,j,nrows,ncols,ldb;
  const PetscInt *irow,*icol;
  PetscScalar    *av,*bv,*v = mat->v;
  Mat            newmat;

  PetscFunctionBegin;
  ierr = ISGetIndices(isrow,&irow);CHKERRQ(ierr);
  ierr = ISGetIndices(iscol,&icol);CHKERRQ(ierr);
  ierr = ISGetLocalSize(isrow,&nrows);CHKERRQ(ierr);
  ierr = ISGetLocalSize(iscol,&ncols);CHKERRQ(ierr);

  /* Check submatrixcall */
  if (scall == MAT_REUSE_MATRIX) {
    PetscInt n_cols,n_rows;
    ierr = MatGetSize(*B,&n_rows,&n_cols);CHKERRQ(ierr);
    if (n_rows != nrows || n_cols != ncols) {
      /* resize the result matrix to match number of requested rows/columns */
      ierr = MatSetSizes(*B,nrows,ncols,nrows,ncols);CHKERRQ(ierr);
    }
    newmat = *B;
  } else {
    /* Create and fill new matrix */
    ierr = MatCreate(PetscObjectComm((PetscObject)A),&newmat);CHKERRQ(ierr);
    ierr = MatSetSizes(newmat,nrows,ncols,nrows,ncols);CHKERRQ(ierr);
    ierr = MatSetType(newmat,((PetscObject)A)->type_name);CHKERRQ(ierr);
    ierr = MatSeqDenseSetPreallocation(newmat,NULL);CHKERRQ(ierr);
  }

  /* Now extract the data pointers and do the copy,column at a time */
  ierr = MatDenseGetArray(newmat,&bv);CHKERRQ(ierr);
  ierr = MatDenseGetLDA(newmat,&ldb);CHKERRQ(ierr);
  for (i=0; i<ncols; i++) {
    av = v + mat->lda*icol[i];
    for (j=0; j<nrows; j++) bv[j] = av[irow[j]];
    bv += ldb;
  }
  ierr = MatDenseRestoreArray(newmat,&bv);CHKERRQ(ierr);

  /* Assemble the matrices so that the correct flags are set */
  ierr = MatAssemblyBegin(newmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(newmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Free work space */
  ierr = ISRestoreIndices(isrow,&irow);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscol,&icol);CHKERRQ(ierr);
  *B   = newmat;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCreateSubMatrices_SeqDense(Mat A,PetscInt n,const IS irow[],const IS icol[],MatReuse scall,Mat *B[])
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  if (scall == MAT_INITIAL_MATRIX) {
    ierr = PetscCalloc1(n,B);CHKERRQ(ierr);
  }

  for (i=0; i<n; i++) {
    ierr = MatCreateSubMatrix_SeqDense(A,irow[i],icol[i],scall,&(*B)[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatAssemblyBegin_SeqDense(Mat mat,MatAssemblyType mode)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatAssemblyEnd_SeqDense(Mat mat,MatAssemblyType mode)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PetscErrorCode MatCopy_SeqDense(Mat A,Mat B,MatStructure str)
{
  Mat_SeqDense      *a = (Mat_SeqDense*)A->data,*b = (Mat_SeqDense*)B->data;
  PetscErrorCode    ierr;
  const PetscScalar *va;
  PetscScalar       *vb;
  PetscInt          lda1=a->lda,lda2=b->lda, m=A->rmap->n,n=A->cmap->n, j;

  PetscFunctionBegin;
  /* If the two matrices don't have the same copy implementation, they aren't compatible for fast copy. */
  if (A->ops->copy != B->ops->copy) {
    ierr = MatCopy_Basic(A,B,str);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  if (m != B->rmap->n || n != B->cmap->n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"size(B) != size(A)");
  ierr = MatDenseGetArrayRead(A,&va);CHKERRQ(ierr);
  ierr = MatDenseGetArray(B,&vb);CHKERRQ(ierr);
  if (lda1>m || lda2>m) {
    for (j=0; j<n; j++) {
      ierr = PetscArraycpy(vb+j*lda2,va+j*lda1,m);CHKERRQ(ierr);
    }
  } else {
    ierr = PetscArraycpy(vb,va,A->rmap->n*A->cmap->n);CHKERRQ(ierr);
  }
  ierr = MatDenseRestoreArray(B,&vb);CHKERRQ(ierr);
  ierr = MatDenseRestoreArrayRead(A,&va);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetUp_SeqDense(Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLayoutSetUp(A->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(A->cmap);CHKERRQ(ierr);
  if (!A->preallocated) {
    ierr = MatSeqDenseSetPreallocation(A,NULL);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatConjugate_SeqDense(Mat A)
{
  Mat_SeqDense   *mat = (Mat_SeqDense *) A->data;
  PetscInt       i,nz = A->rmap->n*A->cmap->n;
  PetscInt       min = PetscMin(A->rmap->n,A->cmap->n);
  PetscScalar    *aa;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDenseGetArray(A,&aa);CHKERRQ(ierr);
  for (i=0; i<nz; i++) aa[i] = PetscConj(aa[i]);
  ierr = MatDenseRestoreArray(A,&aa);CHKERRQ(ierr);
  if (mat->tau) for (i = 0; i < min; i++) mat->tau[i] = PetscConj(mat->tau[i]);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatRealPart_SeqDense(Mat A)
{
  PetscInt       i,nz = A->rmap->n*A->cmap->n;
  PetscScalar    *aa;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDenseGetArray(A,&aa);CHKERRQ(ierr);
  for (i=0; i<nz; i++) aa[i] = PetscRealPart(aa[i]);
  ierr = MatDenseRestoreArray(A,&aa);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatImaginaryPart_SeqDense(Mat A)
{
  PetscInt       i,nz = A->rmap->n*A->cmap->n;
  PetscScalar    *aa;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDenseGetArray(A,&aa);CHKERRQ(ierr);
  for (i=0; i<nz; i++) aa[i] = PetscImaginaryPart(aa[i]);
  ierr = MatDenseRestoreArray(A,&aa);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------*/
PetscErrorCode MatMatMultSymbolic_SeqDense_SeqDense(Mat A,Mat B,PetscReal fill,Mat C)
{
  PetscErrorCode ierr;
  PetscInt       m=A->rmap->n,n=B->cmap->n;
  PetscBool      cisdense;

  PetscFunctionBegin;
  ierr = MatSetSizes(C,m,n,m,n);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompareAny((PetscObject)C,&cisdense,MATSEQDENSE,MATSEQDENSECUDA,"");CHKERRQ(ierr);
  if (!cisdense) {
    PetscBool flg;

    ierr = PetscObjectTypeCompare((PetscObject)B,((PetscObject)A)->type_name,&flg);CHKERRQ(ierr);
    ierr = MatSetType(C,flg ? ((PetscObject)A)->type_name : MATDENSE);CHKERRQ(ierr);
  }
  ierr = MatSetUp(C);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatMultNumeric_SeqDense_SeqDense(Mat A,Mat B,Mat C)
{
  Mat_SeqDense       *a=(Mat_SeqDense*)A->data,*b=(Mat_SeqDense*)B->data,*c=(Mat_SeqDense*)C->data;
  PetscBLASInt       m,n,k;
  const PetscScalar *av,*bv;
  PetscScalar       *cv;
  PetscScalar       _DOne=1.0,_DZero=0.0;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(C->rmap->n,&m);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(C->cmap->n,&n);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(A->cmap->n,&k);CHKERRQ(ierr);
  if (!m || !n || !k) PetscFunctionReturn(0);
  ierr = MatDenseGetArrayRead(A,&av);CHKERRQ(ierr);
  ierr = MatDenseGetArrayRead(B,&bv);CHKERRQ(ierr);
  ierr = MatDenseGetArrayWrite(C,&cv);CHKERRQ(ierr);
  PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&m,&n,&k,&_DOne,av,&a->lda,bv,&b->lda,&_DZero,cv,&c->lda));
  ierr = PetscLogFlops(1.0*m*n*k + 1.0*m*n*(k-1));CHKERRQ(ierr);
  ierr = MatDenseRestoreArrayRead(A,&av);CHKERRQ(ierr);
  ierr = MatDenseRestoreArrayRead(B,&bv);CHKERRQ(ierr);
  ierr = MatDenseRestoreArrayWrite(C,&cv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatTransposeMultSymbolic_SeqDense_SeqDense(Mat A,Mat B,PetscReal fill,Mat C)
{
  PetscErrorCode ierr;
  PetscInt       m=A->rmap->n,n=B->rmap->n;
  PetscBool      cisdense;

  PetscFunctionBegin;
  ierr = MatSetSizes(C,m,n,m,n);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompareAny((PetscObject)C,&cisdense,MATSEQDENSE,MATSEQDENSECUDA,"");CHKERRQ(ierr);
  if (!cisdense) {
    PetscBool flg;

    ierr = PetscObjectTypeCompare((PetscObject)B,((PetscObject)A)->type_name,&flg);CHKERRQ(ierr);
    ierr = MatSetType(C,flg ? ((PetscObject)A)->type_name : MATDENSE);CHKERRQ(ierr);
  }
  ierr = MatSetUp(C);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatTransposeMultNumeric_SeqDense_SeqDense(Mat A,Mat B,Mat C)
{
  Mat_SeqDense      *a = (Mat_SeqDense*)A->data;
  Mat_SeqDense      *b = (Mat_SeqDense*)B->data;
  Mat_SeqDense      *c = (Mat_SeqDense*)C->data;
  const PetscScalar *av,*bv;
  PetscScalar       *cv;
  PetscBLASInt      m,n,k;
  PetscScalar       _DOne=1.0,_DZero=0.0;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(C->rmap->n,&m);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(C->cmap->n,&n);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(A->cmap->n,&k);CHKERRQ(ierr);
  if (!m || !n || !k) PetscFunctionReturn(0);
  ierr = MatDenseGetArrayRead(A,&av);CHKERRQ(ierr);
  ierr = MatDenseGetArrayRead(B,&bv);CHKERRQ(ierr);
  ierr = MatDenseGetArrayWrite(C,&cv);CHKERRQ(ierr);
  PetscStackCallBLAS("BLASgemm",BLASgemm_("N","T",&m,&n,&k,&_DOne,av,&a->lda,bv,&b->lda,&_DZero,cv,&c->lda));
  ierr = MatDenseRestoreArrayRead(A,&av);CHKERRQ(ierr);
  ierr = MatDenseRestoreArrayRead(B,&bv);CHKERRQ(ierr);
  ierr = MatDenseRestoreArrayWrite(C,&cv);CHKERRQ(ierr);
  ierr = PetscLogFlops(1.0*m*n*k + 1.0*m*n*(k-1));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatTransposeMatMultSymbolic_SeqDense_SeqDense(Mat A,Mat B,PetscReal fill,Mat C)
{
  PetscErrorCode ierr;
  PetscInt       m=A->cmap->n,n=B->cmap->n;
  PetscBool      cisdense;

  PetscFunctionBegin;
  ierr = MatSetSizes(C,m,n,m,n);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompareAny((PetscObject)C,&cisdense,MATSEQDENSE,MATSEQDENSECUDA,"");CHKERRQ(ierr);
  if (!cisdense) {
    PetscBool flg;

    ierr = PetscObjectTypeCompare((PetscObject)B,((PetscObject)A)->type_name,&flg);CHKERRQ(ierr);
    ierr = MatSetType(C,flg ? ((PetscObject)A)->type_name : MATDENSE);CHKERRQ(ierr);
  }
  ierr = MatSetUp(C);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatTransposeMatMultNumeric_SeqDense_SeqDense(Mat A,Mat B,Mat C)
{
  Mat_SeqDense      *a = (Mat_SeqDense*)A->data;
  Mat_SeqDense      *b = (Mat_SeqDense*)B->data;
  Mat_SeqDense      *c = (Mat_SeqDense*)C->data;
  const PetscScalar *av,*bv;
  PetscScalar       *cv;
  PetscBLASInt      m,n,k;
  PetscScalar       _DOne=1.0,_DZero=0.0;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(C->rmap->n,&m);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(C->cmap->n,&n);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(A->rmap->n,&k);CHKERRQ(ierr);
  if (!m || !n || !k) PetscFunctionReturn(0);
  ierr = MatDenseGetArrayRead(A,&av);CHKERRQ(ierr);
  ierr = MatDenseGetArrayRead(B,&bv);CHKERRQ(ierr);
  ierr = MatDenseGetArrayWrite(C,&cv);CHKERRQ(ierr);
  PetscStackCallBLAS("BLASgemm",BLASgemm_("T","N",&m,&n,&k,&_DOne,av,&a->lda,bv,&b->lda,&_DZero,cv,&c->lda));
  ierr = MatDenseRestoreArrayRead(A,&av);CHKERRQ(ierr);
  ierr = MatDenseRestoreArrayRead(B,&bv);CHKERRQ(ierr);
  ierr = MatDenseRestoreArrayWrite(C,&cv);CHKERRQ(ierr);
  ierr = PetscLogFlops(1.0*m*n*k + 1.0*m*n*(k-1));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ----------------------------------------------- */
static PetscErrorCode MatProductSetFromOptions_SeqDense_AB(Mat C)
{
  PetscFunctionBegin;
  C->ops->matmultsymbolic = MatMatMultSymbolic_SeqDense_SeqDense;
  C->ops->productsymbolic = MatProductSymbolic_AB;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductSetFromOptions_SeqDense_AtB(Mat C)
{
  PetscFunctionBegin;
  C->ops->transposematmultsymbolic = MatTransposeMatMultSymbolic_SeqDense_SeqDense;
  C->ops->productsymbolic          = MatProductSymbolic_AtB;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductSetFromOptions_SeqDense_ABt(Mat C)
{
  PetscFunctionBegin;
  C->ops->mattransposemultsymbolic = MatMatTransposeMultSymbolic_SeqDense_SeqDense;
  C->ops->productsymbolic          = MatProductSymbolic_ABt;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatProductSetFromOptions_SeqDense(Mat C)
{
  PetscErrorCode ierr;
  Mat_Product    *product = C->product;

  PetscFunctionBegin;
  switch (product->type) {
  case MATPRODUCT_AB:
    ierr = MatProductSetFromOptions_SeqDense_AB(C);CHKERRQ(ierr);
    break;
  case MATPRODUCT_AtB:
    ierr = MatProductSetFromOptions_SeqDense_AtB(C);CHKERRQ(ierr);
    break;
  case MATPRODUCT_ABt:
    ierr = MatProductSetFromOptions_SeqDense_ABt(C);CHKERRQ(ierr);
    break;
  default:
    break;
  }
  PetscFunctionReturn(0);
}
/* ----------------------------------------------- */

static PetscErrorCode MatGetRowMax_SeqDense(Mat A,Vec v,PetscInt idx[])
{
  Mat_SeqDense       *a = (Mat_SeqDense*)A->data;
  PetscErrorCode     ierr;
  PetscInt           i,j,m = A->rmap->n,n = A->cmap->n,p;
  PetscScalar        *x;
  const PetscScalar *aa;

  PetscFunctionBegin;
  if (A->factortype) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  ierr = VecGetArray(v,&x);CHKERRQ(ierr);
  ierr = VecGetLocalSize(v,&p);CHKERRQ(ierr);
  ierr = MatDenseGetArrayRead(A,&aa);CHKERRQ(ierr);
  if (p != A->rmap->n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Nonconforming matrix and vector");
  for (i=0; i<m; i++) {
    x[i] = aa[i]; if (idx) idx[i] = 0;
    for (j=1; j<n; j++) {
      if (PetscRealPart(x[i]) < PetscRealPart(aa[i+a->lda*j])) {x[i] = aa[i + a->lda*j]; if (idx) idx[i] = j;}
    }
  }
  ierr = MatDenseRestoreArrayRead(A,&aa);CHKERRQ(ierr);
  ierr = VecRestoreArray(v,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetRowMaxAbs_SeqDense(Mat A,Vec v,PetscInt idx[])
{
  Mat_SeqDense      *a = (Mat_SeqDense*)A->data;
  PetscErrorCode    ierr;
  PetscInt          i,j,m = A->rmap->n,n = A->cmap->n,p;
  PetscScalar       *x;
  PetscReal         atmp;
  const PetscScalar *aa;

  PetscFunctionBegin;
  if (A->factortype) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  ierr = VecGetArray(v,&x);CHKERRQ(ierr);
  ierr = VecGetLocalSize(v,&p);CHKERRQ(ierr);
  ierr = MatDenseGetArrayRead(A,&aa);CHKERRQ(ierr);
  if (p != A->rmap->n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Nonconforming matrix and vector");
  for (i=0; i<m; i++) {
    x[i] = PetscAbsScalar(aa[i]);
    for (j=1; j<n; j++) {
      atmp = PetscAbsScalar(aa[i+a->lda*j]);
      if (PetscAbsScalar(x[i]) < atmp) {x[i] = atmp; if (idx) idx[i] = j;}
    }
  }
  ierr = MatDenseRestoreArrayRead(A,&aa);CHKERRQ(ierr);
  ierr = VecRestoreArray(v,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetRowMin_SeqDense(Mat A,Vec v,PetscInt idx[])
{
  Mat_SeqDense      *a = (Mat_SeqDense*)A->data;
  PetscErrorCode    ierr;
  PetscInt          i,j,m = A->rmap->n,n = A->cmap->n,p;
  PetscScalar       *x;
  const PetscScalar *aa;

  PetscFunctionBegin;
  if (A->factortype) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  ierr = MatDenseGetArrayRead(A,&aa);CHKERRQ(ierr);
  ierr = VecGetArray(v,&x);CHKERRQ(ierr);
  ierr = VecGetLocalSize(v,&p);CHKERRQ(ierr);
  if (p != A->rmap->n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Nonconforming matrix and vector");
  for (i=0; i<m; i++) {
    x[i] = aa[i]; if (idx) idx[i] = 0;
    for (j=1; j<n; j++) {
      if (PetscRealPart(x[i]) > PetscRealPart(aa[i+a->lda*j])) {x[i] = aa[i + a->lda*j]; if (idx) idx[i] = j;}
    }
  }
  ierr = VecRestoreArray(v,&x);CHKERRQ(ierr);
  ierr = MatDenseRestoreArrayRead(A,&aa);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetColumnVector_SeqDense(Mat A,Vec v,PetscInt col)
{
  Mat_SeqDense      *a = (Mat_SeqDense*)A->data;
  PetscErrorCode    ierr;
  PetscScalar       *x;
  const PetscScalar *aa;

  PetscFunctionBegin;
  if (A->factortype) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  ierr = MatDenseGetArrayRead(A,&aa);CHKERRQ(ierr);
  ierr = VecGetArray(v,&x);CHKERRQ(ierr);
  ierr = PetscArraycpy(x,aa+col*a->lda,A->rmap->n);CHKERRQ(ierr);
  ierr = VecRestoreArray(v,&x);CHKERRQ(ierr);
  ierr = MatDenseRestoreArrayRead(A,&aa);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatGetColumnReductions_SeqDense(Mat A,PetscInt type,PetscReal *reductions)
{
  PetscErrorCode    ierr;
  PetscInt          i,j,m,n;
  const PetscScalar *a;

  PetscFunctionBegin;
  ierr = MatGetSize(A,&m,&n);CHKERRQ(ierr);
  ierr = PetscArrayzero(reductions,n);CHKERRQ(ierr);
  ierr = MatDenseGetArrayRead(A,&a);CHKERRQ(ierr);
  if (type == NORM_2) {
    for (i=0; i<n; i++) {
      for (j=0; j<m; j++) {
        reductions[i] += PetscAbsScalar(a[j]*a[j]);
      }
      a += m;
    }
  } else if (type == NORM_1) {
    for (i=0; i<n; i++) {
      for (j=0; j<m; j++) {
        reductions[i] += PetscAbsScalar(a[j]);
      }
      a += m;
    }
  } else if (type == NORM_INFINITY) {
    for (i=0; i<n; i++) {
      for (j=0; j<m; j++) {
        reductions[i] = PetscMax(PetscAbsScalar(a[j]),reductions[i]);
      }
      a += m;
    }
  } else if (type == REDUCTION_SUM_REALPART || type == REDUCTION_MEAN_REALPART) {
    for (i=0; i<n; i++) {
      for (j=0; j<m; j++) {
        reductions[i] += PetscRealPart(a[j]);
      }
      a += m;
    }
  } else if (type == REDUCTION_SUM_IMAGINARYPART || type == REDUCTION_MEAN_IMAGINARYPART) {
    for (i=0; i<n; i++) {
      for (j=0; j<m; j++) {
        reductions[i] += PetscImaginaryPart(a[j]);
      }
      a += m;
    }
  } else SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Unknown reduction type");
  ierr = MatDenseRestoreArrayRead(A,&a);CHKERRQ(ierr);
  if (type == NORM_2) {
    for (i=0; i<n; i++) reductions[i] = PetscSqrtReal(reductions[i]);
  } else if (type == REDUCTION_MEAN_REALPART || type == REDUCTION_MEAN_IMAGINARYPART) {
    for (i=0; i<n; i++) reductions[i] /= m;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode  MatSetRandom_SeqDense(Mat x,PetscRandom rctx)
{
  PetscErrorCode ierr;
  PetscScalar    *a;
  PetscInt       lda,m,n,i,j;

  PetscFunctionBegin;
  ierr = MatGetSize(x,&m,&n);CHKERRQ(ierr);
  ierr = MatDenseGetLDA(x,&lda);CHKERRQ(ierr);
  ierr = MatDenseGetArray(x,&a);CHKERRQ(ierr);
  for (j=0; j<n; j++) {
    for (i=0; i<m; i++) {
      ierr = PetscRandomGetValue(rctx,a+j*lda+i);CHKERRQ(ierr);
    }
  }
  ierr = MatDenseRestoreArray(x,&a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMissingDiagonal_SeqDense(Mat A,PetscBool  *missing,PetscInt *d)
{
  PetscFunctionBegin;
  *missing = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/* vals is not const */
static PetscErrorCode MatDenseGetColumn_SeqDense(Mat A,PetscInt col,PetscScalar **vals)
{
  PetscErrorCode ierr;
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;
  PetscScalar    *v;

  PetscFunctionBegin;
  if (A->factortype) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  ierr  = MatDenseGetArray(A,&v);CHKERRQ(ierr);
  *vals = v+col*a->lda;
  ierr  = MatDenseRestoreArray(A,&v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseRestoreColumn_SeqDense(Mat A,PetscScalar **vals)
{
  PetscFunctionBegin;
  *vals = NULL; /* user cannot accidentally use the array later */
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------*/
static struct _MatOps MatOps_Values = { MatSetValues_SeqDense,
                                        MatGetRow_SeqDense,
                                        MatRestoreRow_SeqDense,
                                        MatMult_SeqDense,
                                /*  4*/ MatMultAdd_SeqDense,
                                        MatMultTranspose_SeqDense,
                                        MatMultTransposeAdd_SeqDense,
                                        NULL,
                                        NULL,
                                        NULL,
                                /* 10*/ NULL,
                                        MatLUFactor_SeqDense,
                                        MatCholeskyFactor_SeqDense,
                                        MatSOR_SeqDense,
                                        MatTranspose_SeqDense,
                                /* 15*/ MatGetInfo_SeqDense,
                                        MatEqual_SeqDense,
                                        MatGetDiagonal_SeqDense,
                                        MatDiagonalScale_SeqDense,
                                        MatNorm_SeqDense,
                                /* 20*/ MatAssemblyBegin_SeqDense,
                                        MatAssemblyEnd_SeqDense,
                                        MatSetOption_SeqDense,
                                        MatZeroEntries_SeqDense,
                                /* 24*/ MatZeroRows_SeqDense,
                                        NULL,
                                        NULL,
                                        NULL,
                                        NULL,
                                /* 29*/ MatSetUp_SeqDense,
                                        NULL,
                                        NULL,
                                        NULL,
                                        NULL,
                                /* 34*/ MatDuplicate_SeqDense,
                                        NULL,
                                        NULL,
                                        NULL,
                                        NULL,
                                /* 39*/ MatAXPY_SeqDense,
                                        MatCreateSubMatrices_SeqDense,
                                        NULL,
                                        MatGetValues_SeqDense,
                                        MatCopy_SeqDense,
                                /* 44*/ MatGetRowMax_SeqDense,
                                        MatScale_SeqDense,
                                        MatShift_Basic,
                                        NULL,
                                        MatZeroRowsColumns_SeqDense,
                                /* 49*/ MatSetRandom_SeqDense,
                                        NULL,
                                        NULL,
                                        NULL,
                                        NULL,
                                /* 54*/ NULL,
                                        NULL,
                                        NULL,
                                        NULL,
                                        NULL,
                                /* 59*/ MatCreateSubMatrix_SeqDense,
                                        MatDestroy_SeqDense,
                                        MatView_SeqDense,
                                        NULL,
                                        NULL,
                                /* 64*/ NULL,
                                        NULL,
                                        NULL,
                                        NULL,
                                        NULL,
                                /* 69*/ MatGetRowMaxAbs_SeqDense,
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
                                /* 83*/ MatLoad_SeqDense,
                                        MatIsSymmetric_SeqDense,
                                        MatIsHermitian_SeqDense,
                                        NULL,
                                        NULL,
                                        NULL,
                                /* 89*/ NULL,
                                        NULL,
                                        MatMatMultNumeric_SeqDense_SeqDense,
                                        NULL,
                                        NULL,
                                /* 94*/ NULL,
                                        NULL,
                                        NULL,
                                        MatMatTransposeMultNumeric_SeqDense_SeqDense,
                                        NULL,
                                /* 99*/ MatProductSetFromOptions_SeqDense,
                                        NULL,
                                        NULL,
                                        MatConjugate_SeqDense,
                                        NULL,
                                /*104*/ NULL,
                                        MatRealPart_SeqDense,
                                        MatImaginaryPart_SeqDense,
                                        NULL,
                                        NULL,
                                /*109*/ NULL,
                                        NULL,
                                        MatGetRowMin_SeqDense,
                                        MatGetColumnVector_SeqDense,
                                        MatMissingDiagonal_SeqDense,
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
                                        MatGetColumnReductions_SeqDense,
                                        NULL,
                                        NULL,
                                        NULL,
                                /*129*/ NULL,
                                        NULL,
                                        NULL,
                                        MatTransposeMatMultNumeric_SeqDense_SeqDense,
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
                                        MatCreateMPIMatConcatenateSeqMat_SeqDense,
                                /*145*/ NULL,
                                        NULL,
                                        NULL
};

/*@C
   MatCreateSeqDense - Creates a sequential dense matrix that
   is stored in column major order (the usual Fortran 77 manner). Many
   of the matrix operations use the BLAS and LAPACK routines.

   Collective

   Input Parameters:
+  comm - MPI communicator, set to PETSC_COMM_SELF
.  m - number of rows
.  n - number of columns
-  data - optional location of matrix data in column major order.  Set data=NULL for PETSc
   to control all matrix memory allocation.

   Output Parameter:
.  A - the matrix

   Notes:
   The data input variable is intended primarily for Fortran programmers
   who wish to allocate their own matrix memory space.  Most users should
   set data=NULL.

   Level: intermediate

.seealso: MatCreate(), MatCreateDense(), MatSetValues()
@*/
PetscErrorCode  MatCreateSeqDense(MPI_Comm comm,PetscInt m,PetscInt n,PetscScalar *data,Mat *A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,m,n);CHKERRQ(ierr);
  ierr = MatSetType(*A,MATSEQDENSE);CHKERRQ(ierr);
  ierr = MatSeqDenseSetPreallocation(*A,data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   MatSeqDenseSetPreallocation - Sets the array used for storing the matrix elements

   Collective

   Input Parameters:
+  B - the matrix
-  data - the array (or NULL)

   Notes:
   The data input variable is intended primarily for Fortran programmers
   who wish to allocate their own matrix memory space.  Most users should
   need not call this routine.

   Level: intermediate

.seealso: MatCreate(), MatCreateDense(), MatSetValues(), MatDenseSetLDA()

@*/
PetscErrorCode  MatSeqDenseSetPreallocation(Mat B,PetscScalar data[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B,MAT_CLASSID,1);
  ierr = PetscTryMethod(B,"MatSeqDenseSetPreallocation_C",(Mat,PetscScalar[]),(B,data));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode  MatSeqDenseSetPreallocation_SeqDense(Mat B,PetscScalar *data)
{
  Mat_SeqDense   *b = (Mat_SeqDense*)B->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (b->matinuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  B->preallocated = PETSC_TRUE;

  ierr = PetscLayoutSetUp(B->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(B->cmap);CHKERRQ(ierr);

  if (b->lda <= 0) b->lda = B->rmap->n;

  if (!data) { /* petsc-allocated storage */
    if (!b->user_alloc) { ierr = PetscFree(b->v);CHKERRQ(ierr); }
    ierr = PetscCalloc1((size_t)b->lda*B->cmap->n,&b->v);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)B,b->lda*B->cmap->n*sizeof(PetscScalar));CHKERRQ(ierr);

    b->user_alloc = PETSC_FALSE;
  } else { /* user-allocated storage */
    if (!b->user_alloc) { ierr = PetscFree(b->v);CHKERRQ(ierr); }
    b->v          = data;
    b->user_alloc = PETSC_TRUE;
  }
  B->assembled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_ELEMENTAL)
PETSC_INTERN PetscErrorCode MatConvert_SeqDense_Elemental(Mat A, MatType newtype,MatReuse reuse,Mat *newmat)
{
  Mat               mat_elemental;
  PetscErrorCode    ierr;
  const PetscScalar *array;
  PetscScalar       *v_colwise;
  PetscInt          M=A->rmap->N,N=A->cmap->N,i,j,k,*rows,*cols;

  PetscFunctionBegin;
  ierr = PetscMalloc3(M*N,&v_colwise,M,&rows,N,&cols);CHKERRQ(ierr);
  ierr = MatDenseGetArrayRead(A,&array);CHKERRQ(ierr);
  /* convert column-wise array into row-wise v_colwise, see MatSetValues_Elemental() */
  k = 0;
  for (j=0; j<N; j++) {
    cols[j] = j;
    for (i=0; i<M; i++) {
      v_colwise[j*M+i] = array[k++];
    }
  }
  for (i=0; i<M; i++) {
    rows[i] = i;
  }
  ierr = MatDenseRestoreArrayRead(A,&array);CHKERRQ(ierr);

  ierr = MatCreate(PetscObjectComm((PetscObject)A), &mat_elemental);CHKERRQ(ierr);
  ierr = MatSetSizes(mat_elemental,PETSC_DECIDE,PETSC_DECIDE,M,N);CHKERRQ(ierr);
  ierr = MatSetType(mat_elemental,MATELEMENTAL);CHKERRQ(ierr);
  ierr = MatSetUp(mat_elemental);CHKERRQ(ierr);

  /* PETSc-Elemental interaface uses axpy for setting off-processor entries, only ADD_VALUES is allowed */
  ierr = MatSetValues(mat_elemental,M,rows,N,cols,v_colwise,ADD_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(mat_elemental, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat_elemental, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscFree3(v_colwise,rows,cols);CHKERRQ(ierr);

  if (reuse == MAT_INPLACE_MATRIX) {
    ierr = MatHeaderReplace(A,&mat_elemental);CHKERRQ(ierr);
  } else {
    *newmat = mat_elemental;
  }
  PetscFunctionReturn(0);
}
#endif

PetscErrorCode  MatDenseSetLDA_SeqDense(Mat B,PetscInt lda)
{
  Mat_SeqDense *b = (Mat_SeqDense*)B->data;
  PetscBool    data;

  PetscFunctionBegin;
  data = (PetscBool)((B->rmap->n > 0 && B->cmap->n > 0) ? (b->v ? PETSC_TRUE : PETSC_FALSE) : PETSC_FALSE);
  if (!b->user_alloc && data && b->lda!=lda) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"LDA cannot be changed after allocation of internal storage");
  if (lda < B->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"LDA %D must be at least matrix dimension %D",lda,B->rmap->n);
  b->lda = lda;
  PetscFunctionReturn(0);
}

PetscErrorCode MatCreateMPIMatConcatenateSeqMat_SeqDense(MPI_Comm comm,Mat inmat,PetscInt n,MatReuse scall,Mat *outmat)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  if (size == 1) {
    if (scall == MAT_INITIAL_MATRIX) {
      ierr = MatDuplicate(inmat,MAT_COPY_VALUES,outmat);CHKERRQ(ierr);
    } else {
      ierr = MatCopy(inmat,*outmat,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    }
  } else {
    ierr = MatCreateMPIMatConcatenateSeqMat_MPIDense(comm,inmat,n,scall,outmat);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatDenseGetColumnVec_SeqDense(Mat A,PetscInt col,Vec *v)
{
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (a->vecinuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  if (a->matinuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  if (!a->cvec) {
    ierr = VecCreateSeqWithArray(PetscObjectComm((PetscObject)A),A->rmap->bs,A->rmap->n,NULL,&a->cvec);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)A,(PetscObject)a->cvec);CHKERRQ(ierr);
  }
  a->vecinuse = col + 1;
  ierr = MatDenseGetArray(A,(PetscScalar**)&a->ptrinuse);CHKERRQ(ierr);
  ierr = VecPlaceArray(a->cvec,a->ptrinuse + (size_t)col * (size_t)a->lda);CHKERRQ(ierr);
  *v   = a->cvec;
  PetscFunctionReturn(0);
}

PetscErrorCode MatDenseRestoreColumnVec_SeqDense(Mat A,PetscInt col,Vec *v)
{
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!a->vecinuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseGetColumnVec() first");
  if (!a->cvec) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing internal column vector");
  a->vecinuse = 0;
  ierr = MatDenseRestoreArray(A,(PetscScalar**)&a->ptrinuse);CHKERRQ(ierr);
  ierr = VecResetArray(a->cvec);CHKERRQ(ierr);
  *v   = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode MatDenseGetColumnVecRead_SeqDense(Mat A,PetscInt col,Vec *v)
{
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (a->vecinuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  if (a->matinuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  if (!a->cvec) {
    ierr = VecCreateSeqWithArray(PetscObjectComm((PetscObject)A),A->rmap->bs,A->rmap->n,NULL,&a->cvec);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)A,(PetscObject)a->cvec);CHKERRQ(ierr);
  }
  a->vecinuse = col + 1;
  ierr = MatDenseGetArrayRead(A,&a->ptrinuse);CHKERRQ(ierr);
  ierr = VecPlaceArray(a->cvec,a->ptrinuse + (size_t)col * (size_t)a->lda);CHKERRQ(ierr);
  ierr = VecLockReadPush(a->cvec);CHKERRQ(ierr);
  *v   = a->cvec;
  PetscFunctionReturn(0);
}

PetscErrorCode MatDenseRestoreColumnVecRead_SeqDense(Mat A,PetscInt col,Vec *v)
{
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!a->vecinuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseGetColumnVec() first");
  if (!a->cvec) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing internal column vector");
  a->vecinuse = 0;
  ierr = MatDenseRestoreArrayRead(A,&a->ptrinuse);CHKERRQ(ierr);
  ierr = VecLockReadPop(a->cvec);CHKERRQ(ierr);
  ierr = VecResetArray(a->cvec);CHKERRQ(ierr);
  *v   = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode MatDenseGetColumnVecWrite_SeqDense(Mat A,PetscInt col,Vec *v)
{
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (a->vecinuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  if (a->matinuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  if (!a->cvec) {
    ierr = VecCreateSeqWithArray(PetscObjectComm((PetscObject)A),A->rmap->bs,A->rmap->n,NULL,&a->cvec);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)A,(PetscObject)a->cvec);CHKERRQ(ierr);
  }
  a->vecinuse = col + 1;
  ierr = MatDenseGetArrayWrite(A,(PetscScalar**)&a->ptrinuse);CHKERRQ(ierr);
  ierr = VecPlaceArray(a->cvec,a->ptrinuse + (size_t)col * (size_t)a->lda);CHKERRQ(ierr);
  *v   = a->cvec;
  PetscFunctionReturn(0);
}

PetscErrorCode MatDenseRestoreColumnVecWrite_SeqDense(Mat A,PetscInt col,Vec *v)
{
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!a->vecinuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseGetColumnVec() first");
  if (!a->cvec) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing internal column vector");
  a->vecinuse = 0;
  ierr = MatDenseRestoreArrayWrite(A,(PetscScalar**)&a->ptrinuse);CHKERRQ(ierr);
  ierr = VecResetArray(a->cvec);CHKERRQ(ierr);
  *v   = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode MatDenseGetSubMatrix_SeqDense(Mat A,PetscInt cbegin,PetscInt cend,Mat *v)
{
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (a->vecinuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  if (a->matinuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  if (a->cmat && cend-cbegin != a->cmat->cmap->N) {
    ierr = MatDestroy(&a->cmat);CHKERRQ(ierr);
  }
  if (!a->cmat) {
    ierr = MatCreateDense(PetscObjectComm((PetscObject)A),A->rmap->n,PETSC_DECIDE,A->rmap->N,cend-cbegin,(PetscScalar*)a->v + (size_t)cbegin * (size_t)a->lda,&a->cmat);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)A,(PetscObject)a->cmat);CHKERRQ(ierr);
  } else {
    ierr = MatDensePlaceArray(a->cmat,a->v + (size_t)cbegin * (size_t)a->lda);CHKERRQ(ierr);
  }
  ierr = MatDenseSetLDA(a->cmat,a->lda);CHKERRQ(ierr);
  a->matinuse = cbegin + 1;
  *v = a->cmat;
  PetscFunctionReturn(0);
}

PetscErrorCode MatDenseRestoreSubMatrix_SeqDense(Mat A,Mat *v)
{
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!a->matinuse) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseGetSubMatrix() first");
  if (!a->cmat) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing internal column matrix");
  if (*v != a->cmat) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Not the matrix obtained from MatDenseGetSubMatrix()");
  a->matinuse = 0;
  ierr = MatDenseResetArray(a->cmat);CHKERRQ(ierr);
  *v   = NULL;
  PetscFunctionReturn(0);
}

/*MC
   MATSEQDENSE - MATSEQDENSE = "seqdense" - A matrix type to be used for sequential dense matrices.

   Options Database Keys:
. -mat_type seqdense - sets the matrix type to "seqdense" during a call to MatSetFromOptions()

  Level: beginner

.seealso: MatCreateSeqDense()

M*/
PetscErrorCode MatCreate_SeqDense(Mat B)
{
  Mat_SeqDense   *b;
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)B),&size);CHKERRMPI(ierr);
  if (size > 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Comm must be of size 1");

  ierr    = PetscNewLog(B,&b);CHKERRQ(ierr);
  ierr    = PetscMemcpy(B->ops,&MatOps_Values,sizeof(struct _MatOps));CHKERRQ(ierr);
  B->data = (void*)b;

  b->roworiented = PETSC_TRUE;

  ierr = PetscObjectComposeFunction((PetscObject)B,"MatQRFactor_C",MatQRFactor_SeqDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseGetLDA_C",MatDenseGetLDA_SeqDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseSetLDA_C",MatDenseSetLDA_SeqDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseGetArray_C",MatDenseGetArray_SeqDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseRestoreArray_C",MatDenseRestoreArray_SeqDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDensePlaceArray_C",MatDensePlaceArray_SeqDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseResetArray_C",MatDenseResetArray_SeqDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseReplaceArray_C",MatDenseReplaceArray_SeqDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseGetArrayRead_C",MatDenseGetArray_SeqDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseRestoreArrayRead_C",MatDenseRestoreArray_SeqDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseGetArrayWrite_C",MatDenseGetArray_SeqDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseRestoreArrayWrite_C",MatDenseRestoreArray_SeqDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_seqdense_seqaij_C",MatConvert_SeqDense_SeqAIJ);CHKERRQ(ierr);
#if defined(PETSC_HAVE_ELEMENTAL)
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_seqdense_elemental_C",MatConvert_SeqDense_Elemental);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_SCALAPACK)
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_seqdense_scalapack_C",MatConvert_Dense_ScaLAPACK);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_CUDA)
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_seqdense_seqdensecuda_C",MatConvert_SeqDense_SeqDenseCUDA);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatProductSetFromOptions_seqdensecuda_seqdensecuda_C",MatProductSetFromOptions_SeqDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatProductSetFromOptions_seqdensecuda_seqdense_C",MatProductSetFromOptions_SeqDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatProductSetFromOptions_seqdense_seqdensecuda_C",MatProductSetFromOptions_SeqDense);CHKERRQ(ierr);
#endif
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatSeqDenseSetPreallocation_C",MatSeqDenseSetPreallocation_SeqDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatProductSetFromOptions_seqaij_seqdense_C",MatProductSetFromOptions_SeqAIJ_SeqDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatProductSetFromOptions_seqdense_seqdense_C",MatProductSetFromOptions_SeqDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatProductSetFromOptions_seqbaij_seqdense_C",MatProductSetFromOptions_SeqXBAIJ_SeqDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatProductSetFromOptions_seqsbaij_seqdense_C",MatProductSetFromOptions_SeqXBAIJ_SeqDense);CHKERRQ(ierr);

  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseGetColumn_C",MatDenseGetColumn_SeqDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseRestoreColumn_C",MatDenseRestoreColumn_SeqDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseGetColumnVec_C",MatDenseGetColumnVec_SeqDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseRestoreColumnVec_C",MatDenseRestoreColumnVec_SeqDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseGetColumnVecRead_C",MatDenseGetColumnVecRead_SeqDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseRestoreColumnVecRead_C",MatDenseRestoreColumnVecRead_SeqDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseGetColumnVecWrite_C",MatDenseGetColumnVecWrite_SeqDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseRestoreColumnVecWrite_C",MatDenseRestoreColumnVecWrite_SeqDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseGetSubMatrix_C",MatDenseGetSubMatrix_SeqDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseRestoreSubMatrix_C",MatDenseRestoreSubMatrix_SeqDense);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATSEQDENSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   MatDenseGetColumn - gives access to a column of a dense matrix. This is only the local part of the column. You MUST call MatDenseRestoreColumn() to avoid memory bleeding.

   Not Collective

   Input Parameters:
+  mat - a MATSEQDENSE or MATMPIDENSE matrix
-  col - column index

   Output Parameter:
.  vals - pointer to the data

   Level: intermediate

.seealso: MatDenseRestoreColumn()
@*/
PetscErrorCode MatDenseGetColumn(Mat A,PetscInt col,PetscScalar **vals)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidLogicalCollectiveInt(A,col,2);
  PetscValidPointer(vals,3);
  ierr = PetscUseMethod(A,"MatDenseGetColumn_C",(Mat,PetscInt,PetscScalar**),(A,col,vals));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   MatDenseRestoreColumn - returns access to a column of a dense matrix which is returned by MatDenseGetColumn().

   Not Collective

   Input Parameter:
.  mat - a MATSEQDENSE or MATMPIDENSE matrix

   Output Parameter:
.  vals - pointer to the data

   Level: intermediate

.seealso: MatDenseGetColumn()
@*/
PetscErrorCode MatDenseRestoreColumn(Mat A,PetscScalar **vals)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(vals,2);
  ierr = PetscUseMethod(A,"MatDenseRestoreColumn_C",(Mat,PetscScalar**),(A,vals));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   MatDenseGetColumnVec - Gives read-write access to a column of a dense matrix, represented as a Vec.

   Collective

   Input Parameters:
+  mat - the Mat object
-  col - the column index

   Output Parameter:
.  v - the vector

   Notes:
     The vector is owned by PETSc. Users need to call MatDenseRestoreColumnVec() when the vector is no longer needed.
     Use MatDenseGetColumnVecRead() to obtain read-only access or MatDenseGetColumnVecWrite() for write-only access.

   Level: intermediate

.seealso: MATDENSE, MATDENSECUDA, MatDenseGetColumnVecRead(), MatDenseGetColumnVecWrite(), MatDenseRestoreColumnVec(), MatDenseRestoreColumnVecRead(), MatDenseRestoreColumnVecWrite()
@*/
PetscErrorCode MatDenseGetColumnVec(Mat A,PetscInt col,Vec *v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  PetscValidLogicalCollectiveInt(A,col,2);
  PetscValidPointer(v,3);
  if (!A->preallocated) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Matrix not preallocated");
  if (col < 0 || col > A->cmap->N) SETERRQ2(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Invalid col %D, should be in [0,%D)",col,A->cmap->N);
  ierr = PetscUseMethod(A,"MatDenseGetColumnVec_C",(Mat,PetscInt,Vec*),(A,col,v));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   MatDenseRestoreColumnVec - Returns access to a column of a dense matrix obtained from MatDenseGetColumnVec().

   Collective

   Input Parameters:
+  mat - the Mat object
.  col - the column index
-  v - the Vec object

   Level: intermediate

.seealso: MATDENSE, MATDENSECUDA, MatDenseGetColumnVec(), MatDenseGetColumnVecRead(), MatDenseGetColumnVecWrite(), MatDenseRestoreColumnVecRead(), MatDenseRestoreColumnVecWrite()
@*/
PetscErrorCode MatDenseRestoreColumnVec(Mat A,PetscInt col,Vec *v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  PetscValidLogicalCollectiveInt(A,col,2);
  PetscValidPointer(v,3);
  if (!A->preallocated) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Matrix not preallocated");
  if (col < 0 || col > A->cmap->N) SETERRQ2(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Invalid col %D, should be in [0,%D)",col,A->cmap->N);
  ierr = PetscUseMethod(A,"MatDenseRestoreColumnVec_C",(Mat,PetscInt,Vec*),(A,col,v));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   MatDenseGetColumnVecRead - Gives read-only access to a column of a dense matrix, represented as a Vec.

   Collective

   Input Parameters:
+  mat - the Mat object
-  col - the column index

   Output Parameter:
.  v - the vector

   Notes:
     The vector is owned by PETSc and users cannot modify it.
     Users need to call MatDenseRestoreColumnVecRead() when the vector is no longer needed.
     Use MatDenseGetColumnVec() to obtain read-write access or MatDenseGetColumnVecWrite() for write-only access.

   Level: intermediate

.seealso: MATDENSE, MATDENSECUDA, MatDenseGetColumnVec(), MatDenseGetColumnVecWrite(), MatDenseRestoreColumnVec(), MatDenseRestoreColumnVecRead(), MatDenseRestoreColumnVecWrite()
@*/
PetscErrorCode MatDenseGetColumnVecRead(Mat A,PetscInt col,Vec *v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  PetscValidLogicalCollectiveInt(A,col,2);
  PetscValidPointer(v,3);
  if (!A->preallocated) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Matrix not preallocated");
  if (col < 0 || col > A->cmap->N) SETERRQ2(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Invalid col %D, should be in [0,%D)",col,A->cmap->N);
  ierr = PetscUseMethod(A,"MatDenseGetColumnVecRead_C",(Mat,PetscInt,Vec*),(A,col,v));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   MatDenseRestoreColumnVecRead - Returns access to a column of a dense matrix obtained from MatDenseGetColumnVecRead().

   Collective

   Input Parameters:
+  mat - the Mat object
.  col - the column index
-  v - the Vec object

   Level: intermediate

.seealso: MATDENSE, MATDENSECUDA, MatDenseGetColumnVec(), MatDenseGetColumnVecRead(), MatDenseGetColumnVecWrite(), MatDenseRestoreColumnVec(), MatDenseRestoreColumnVecWrite()
@*/
PetscErrorCode MatDenseRestoreColumnVecRead(Mat A,PetscInt col,Vec *v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  PetscValidLogicalCollectiveInt(A,col,2);
  PetscValidPointer(v,3);
  if (!A->preallocated) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Matrix not preallocated");
  if (col < 0 || col > A->cmap->N) SETERRQ2(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Invalid col %D, should be in [0,%D)",col,A->cmap->N);
  ierr = PetscUseMethod(A,"MatDenseRestoreColumnVecRead_C",(Mat,PetscInt,Vec*),(A,col,v));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   MatDenseGetColumnVecWrite - Gives write-only access to a column of a dense matrix, represented as a Vec.

   Collective

   Input Parameters:
+  mat - the Mat object
-  col - the column index

   Output Parameter:
.  v - the vector

   Notes:
     The vector is owned by PETSc. Users need to call MatDenseRestoreColumnVecWrite() when the vector is no longer needed.
     Use MatDenseGetColumnVec() to obtain read-write access or MatDenseGetColumnVecRead() for read-only access.

   Level: intermediate

.seealso: MATDENSE, MATDENSECUDA, MatDenseGetColumnVec(), MatDenseGetColumnVecRead(), MatDenseRestoreColumnVec(), MatDenseRestoreColumnVecRead(), MatDenseRestoreColumnVecWrite()
@*/
PetscErrorCode MatDenseGetColumnVecWrite(Mat A,PetscInt col,Vec *v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  PetscValidLogicalCollectiveInt(A,col,2);
  PetscValidPointer(v,3);
  if (!A->preallocated) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Matrix not preallocated");
  if (col < 0 || col > A->cmap->N) SETERRQ2(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Invalid col %D, should be in [0,%D)",col,A->cmap->N);
  ierr = PetscUseMethod(A,"MatDenseGetColumnVecWrite_C",(Mat,PetscInt,Vec*),(A,col,v));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   MatDenseRestoreColumnVecWrite - Returns access to a column of a dense matrix obtained from MatDenseGetColumnVecWrite().

   Collective

   Input Parameters:
+  mat - the Mat object
.  col - the column index
-  v - the Vec object

   Level: intermediate

.seealso: MATDENSE, MATDENSECUDA, MatDenseGetColumnVec(), MatDenseGetColumnVecRead(), MatDenseGetColumnVecWrite(), MatDenseRestoreColumnVec(), MatDenseRestoreColumnVecRead()
@*/
PetscErrorCode MatDenseRestoreColumnVecWrite(Mat A,PetscInt col,Vec *v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  PetscValidLogicalCollectiveInt(A,col,2);
  PetscValidPointer(v,3);
  if (!A->preallocated) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Matrix not preallocated");
  if (col < 0 || col > A->cmap->N) SETERRQ2(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Invalid col %D, should be in [0,%D)",col,A->cmap->N);
  ierr = PetscUseMethod(A,"MatDenseRestoreColumnVecWrite_C",(Mat,PetscInt,Vec*),(A,col,v));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   MatDenseGetSubMatrix - Gives access to a block of columns of a dense matrix, represented as a Mat.

   Collective

   Input Parameters:
+  mat - the Mat object
.  cbegin - the first index in the block
-  cend - the last index in the block

   Output Parameter:
.  v - the matrix

   Notes:
     The matrix is owned by PETSc. Users need to call MatDenseRestoreSubMatrix() when the matrix is no longer needed.

   Level: intermediate

.seealso: MATDENSE, MATDENSECUDA, MatDenseGetColumnVec(), MatDenseRestoreColumnVec(), MatDenseRestoreSubMatrix()
@*/
PetscErrorCode MatDenseGetSubMatrix(Mat A,PetscInt cbegin,PetscInt cend,Mat *v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  PetscValidLogicalCollectiveInt(A,cbegin,2);
  PetscValidLogicalCollectiveInt(A,cend,3);
  PetscValidPointer(v,4);
  if (!A->preallocated) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Matrix not preallocated");
  if (cbegin < 0 || cbegin > A->cmap->N) SETERRQ2(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Invalid cbegin %D, should be in [0,%D)",cbegin,A->cmap->N);
  if (cend < cbegin || cend > A->cmap->N) SETERRQ3(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Invalid cend %D, should be in [%D,%D)",cend,cbegin,A->cmap->N);
  ierr = PetscUseMethod(A,"MatDenseGetSubMatrix_C",(Mat,PetscInt,PetscInt,Mat*),(A,cbegin,cend,v));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   MatDenseRestoreSubMatrix - Returns access to a block of columns of a dense matrix obtained from MatDenseGetSubMatrix().

   Collective

   Input Parameters:
+  mat - the Mat object
-  v - the Mat object

   Level: intermediate

.seealso: MATDENSE, MATDENSECUDA, MatDenseGetColumnVec(), MatDenseRestoreColumnVec(), MatDenseGetSubMatrix()
@*/
PetscErrorCode MatDenseRestoreSubMatrix(Mat A,Mat *v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  PetscValidPointer(v,2);
  ierr = PetscUseMethod(A,"MatDenseRestoreSubMatrix_C",(Mat,Mat*),(A,v));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
