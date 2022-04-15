
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

  PetscFunctionBegin;
  PetscCheck(A->rmap->n == A->cmap->n,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Cannot symmetrize a rectangular matrix");
  PetscCall(MatDenseGetArray(A,&v));
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
  PetscCall(MatDenseRestoreArray(A,&v));
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatSeqDenseInvertFactors_Private(Mat A)
{
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
  PetscBLASInt   info,n;

  PetscFunctionBegin;
  if (!A->rmap->n || !A->cmap->n) PetscFunctionReturn(0);
  PetscCall(PetscBLASIntCast(A->cmap->n,&n));
  if (A->factortype == MAT_FACTOR_LU) {
    PetscCheck(mat->pivots,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Pivots not present");
    if (!mat->fwork) {
      mat->lfwork = n;
      PetscCall(PetscMalloc1(mat->lfwork,&mat->fwork));
      PetscCall(PetscLogObjectMemory((PetscObject)A,mat->lfwork*sizeof(PetscBLASInt)));
    }
    PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
    PetscStackCallBLAS("LAPACKgetri",LAPACKgetri_(&n,mat->v,&mat->lda,mat->pivots,mat->fwork,&mat->lfwork,&info));
    PetscCall(PetscFPTrapPop());
    PetscCall(PetscLogFlops((1.0*A->cmap->n*A->cmap->n*A->cmap->n)/3.0));
  } else if (A->factortype == MAT_FACTOR_CHOLESKY) {
    if (A->spd) {
      PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
      PetscStackCallBLAS("LAPACKpotri",LAPACKpotri_("L",&n,mat->v,&mat->lda,&info));
      PetscCall(PetscFPTrapPop());
      PetscCall(MatSeqDenseSymmetrize_Private(A,PETSC_TRUE));
#if defined(PETSC_USE_COMPLEX)
    } else if (A->hermitian) {
      PetscCheck(mat->pivots,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Pivots not present");
      PetscCheck(mat->fwork,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Fwork not present");
      PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
      PetscStackCallBLAS("LAPACKhetri",LAPACKhetri_("L",&n,mat->v,&mat->lda,mat->pivots,mat->fwork,&info));
      PetscCall(PetscFPTrapPop());
      PetscCall(MatSeqDenseSymmetrize_Private(A,PETSC_TRUE));
#endif
    } else { /* symmetric case */
      PetscCheck(mat->pivots,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Pivots not present");
      PetscCheck(mat->fwork,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Fwork not present");
      PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
      PetscStackCallBLAS("LAPACKsytri",LAPACKsytri_("L",&n,mat->v,&mat->lda,mat->pivots,mat->fwork,&info));
      PetscCall(PetscFPTrapPop());
      PetscCall(MatSeqDenseSymmetrize_Private(A,PETSC_FALSE));
    }
    PetscCheck(!info,PETSC_COMM_SELF,PETSC_ERR_MAT_CH_ZRPVT,"Bad Inversion: zero pivot in row %" PetscInt_FMT,(PetscInt)info-1);
    PetscCall(PetscLogFlops((1.0*A->cmap->n*A->cmap->n*A->cmap->n)/3.0));
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Matrix must be factored to solve");

  A->ops->solve             = NULL;
  A->ops->matsolve          = NULL;
  A->ops->solvetranspose    = NULL;
  A->ops->matsolvetranspose = NULL;
  A->ops->solveadd          = NULL;
  A->ops->solvetransposeadd = NULL;
  A->factortype             = MAT_FACTOR_NONE;
  PetscCall(PetscFree(A->solvertype));
  PetscFunctionReturn(0);
}

PetscErrorCode MatZeroRowsColumns_SeqDense(Mat A,PetscInt N,const PetscInt rows[],PetscScalar diag,Vec x,Vec b)
{
  Mat_SeqDense      *l = (Mat_SeqDense*)A->data;
  PetscInt          m  = l->lda, n = A->cmap->n,r = A->rmap->n, i,j;
  PetscScalar       *slot,*bb,*v;
  const PetscScalar *xx;

  PetscFunctionBegin;
  if (PetscDefined(USE_DEBUG)) {
    for (i=0; i<N; i++) {
      PetscCheck(rows[i] >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative row requested to be zeroed");
      PetscCheck(rows[i] < A->rmap->n,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row %" PetscInt_FMT " requested to be zeroed greater than or equal number of rows %" PetscInt_FMT,rows[i],A->rmap->n);
      PetscCheck(rows[i] < A->cmap->n,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Col %" PetscInt_FMT " requested to be zeroed greater than or equal number of cols %" PetscInt_FMT,rows[i],A->cmap->n);
    }
  }
  if (!N) PetscFunctionReturn(0);

  /* fix right hand side if needed */
  if (x && b) {
    Vec xt;

    PetscCheck(A->rmap->n == A->cmap->n,PETSC_COMM_SELF,PETSC_ERR_SUP,"Only coded for square matrices");
    PetscCall(VecDuplicate(x,&xt));
    PetscCall(VecCopy(x,xt));
    PetscCall(VecScale(xt,-1.0));
    PetscCall(MatMultAdd(A,xt,b,b));
    PetscCall(VecDestroy(&xt));
    PetscCall(VecGetArrayRead(x,&xx));
    PetscCall(VecGetArray(b,&bb));
    for (i=0; i<N; i++) bb[rows[i]] = diag*xx[rows[i]];
    PetscCall(VecRestoreArrayRead(x,&xx));
    PetscCall(VecRestoreArray(b,&bb));
  }

  PetscCall(MatDenseGetArray(A,&v));
  for (i=0; i<N; i++) {
    slot = v + rows[i]*m;
    PetscCall(PetscArrayzero(slot,r));
  }
  for (i=0; i<N; i++) {
    slot = v + rows[i];
    for (j=0; j<n; j++) { *slot = 0.0; slot += m;}
  }
  if (diag != 0.0) {
    PetscCheck(A->rmap->n == A->cmap->n,PETSC_COMM_SELF,PETSC_ERR_SUP,"Only coded for square matrices");
    for (i=0; i<N; i++) {
      slot  = v + (m+1)*rows[i];
      *slot = diag;
    }
  }
  PetscCall(MatDenseRestoreArray(A,&v));
  PetscFunctionReturn(0);
}

PetscErrorCode MatPtAPNumeric_SeqDense_SeqDense(Mat A,Mat P,Mat C)
{
  Mat_SeqDense   *c = (Mat_SeqDense*)(C->data);

  PetscFunctionBegin;
  if (c->ptapwork) {
    PetscCall((*C->ops->matmultnumeric)(A,P,c->ptapwork));
    PetscCall((*C->ops->transposematmultnumeric)(P,c->ptapwork,C));
  } else SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_SUP,"Must call MatPtAPSymbolic_SeqDense_SeqDense() first");
  PetscFunctionReturn(0);
}

PetscErrorCode MatPtAPSymbolic_SeqDense_SeqDense(Mat A,Mat P,PetscReal fill,Mat C)
{
  Mat_SeqDense   *c;
  PetscBool      cisdense;

  PetscFunctionBegin;
  PetscCall(MatSetSizes(C,P->cmap->n,P->cmap->n,P->cmap->N,P->cmap->N));
  PetscCall(PetscObjectTypeCompareAny((PetscObject)C,&cisdense,MATSEQDENSE,MATSEQDENSECUDA,""));
  if (!cisdense) {
    PetscBool flg;

    PetscCall(PetscObjectTypeCompare((PetscObject)P,((PetscObject)A)->type_name,&flg));
    PetscCall(MatSetType(C,flg ? ((PetscObject)A)->type_name : MATDENSE));
  }
  PetscCall(MatSetUp(C));
  c    = (Mat_SeqDense*)C->data;
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A),&c->ptapwork));
  PetscCall(MatSetSizes(c->ptapwork,A->rmap->n,P->cmap->n,A->rmap->N,P->cmap->N));
  PetscCall(MatSetType(c->ptapwork,((PetscObject)C)->type_name));
  PetscCall(MatSetUp(c->ptapwork));
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatConvert_SeqAIJ_SeqDense(Mat A,MatType newtype,MatReuse reuse,Mat *newmat)
{
  Mat             B = NULL;
  Mat_SeqAIJ      *a = (Mat_SeqAIJ*)A->data;
  Mat_SeqDense    *b;
  PetscInt        *ai=a->i,*aj=a->j,m=A->rmap->N,n=A->cmap->N,i;
  const MatScalar *av;
  PetscBool       isseqdense;

  PetscFunctionBegin;
  if (reuse == MAT_REUSE_MATRIX) {
    PetscCall(PetscObjectTypeCompare((PetscObject)*newmat,MATSEQDENSE,&isseqdense));
    PetscCheck(isseqdense,PetscObjectComm((PetscObject)*newmat),PETSC_ERR_USER,"Cannot reuse matrix of type %s",((PetscObject)(*newmat))->type_name);
  }
  if (reuse != MAT_REUSE_MATRIX) {
    PetscCall(MatCreate(PetscObjectComm((PetscObject)A),&B));
    PetscCall(MatSetSizes(B,m,n,m,n));
    PetscCall(MatSetType(B,MATSEQDENSE));
    PetscCall(MatSeqDenseSetPreallocation(B,NULL));
    b    = (Mat_SeqDense*)(B->data);
  } else {
    b    = (Mat_SeqDense*)((*newmat)->data);
    PetscCall(PetscArrayzero(b->v,m*n));
  }
  PetscCall(MatSeqAIJGetArrayRead(A,&av));
  for (i=0; i<m; i++) {
    PetscInt j;
    for (j=0;j<ai[1]-ai[0];j++) {
      b->v[*aj*m+i] = *av;
      aj++;
      av++;
    }
    ai++;
  }
  PetscCall(MatSeqAIJRestoreArrayRead(A,&av));

  if (reuse == MAT_INPLACE_MATRIX) {
    PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
    PetscCall(MatHeaderReplace(A,&B));
  } else {
    if (B) *newmat = B;
    PetscCall(MatAssemblyBegin(*newmat,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(*newmat,MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatConvert_SeqDense_SeqAIJ(Mat A, MatType newtype,MatReuse reuse,Mat *newmat)
{
  Mat            B = NULL;
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;
  PetscInt       i, j;
  PetscInt       *rows, *nnz;
  MatScalar      *aa = a->v, *vals;

  PetscFunctionBegin;
  PetscCall(PetscCalloc3(A->rmap->n,&rows,A->rmap->n,&nnz,A->rmap->n,&vals));
  if (reuse != MAT_REUSE_MATRIX) {
    PetscCall(MatCreate(PetscObjectComm((PetscObject)A),&B));
    PetscCall(MatSetSizes(B,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N));
    PetscCall(MatSetType(B,MATSEQAIJ));
    for (j=0; j<A->cmap->n; j++) {
      for (i=0; i<A->rmap->n; i++) if (aa[i] != 0.0 || (i == j && A->cmap->n == A->rmap->n)) ++nnz[i];
      aa += a->lda;
    }
    PetscCall(MatSeqAIJSetPreallocation(B,PETSC_DETERMINE,nnz));
  } else B = *newmat;
  aa = a->v;
  for (j=0; j<A->cmap->n; j++) {
    PetscInt numRows = 0;
    for (i=0; i<A->rmap->n; i++) if (aa[i] != 0.0 || (i == j && A->cmap->n == A->rmap->n)) {rows[numRows] = i; vals[numRows++] = aa[i];}
    PetscCall(MatSetValues(B,numRows,rows,1,&j,vals,INSERT_VALUES));
    aa  += a->lda;
  }
  PetscCall(PetscFree3(rows,nnz,vals));
  PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));

  if (reuse == MAT_INPLACE_MATRIX) {
    PetscCall(MatHeaderReplace(A,&B));
  } else if (reuse != MAT_REUSE_MATRIX) *newmat = B;
  PetscFunctionReturn(0);
}

PetscErrorCode MatAXPY_SeqDense(Mat Y,PetscScalar alpha,Mat X,MatStructure str)
{
  Mat_SeqDense      *x = (Mat_SeqDense*)X->data,*y = (Mat_SeqDense*)Y->data;
  const PetscScalar *xv;
  PetscScalar       *yv;
  PetscBLASInt      N,m,ldax = 0,lday = 0,one = 1;

  PetscFunctionBegin;
  PetscCall(MatDenseGetArrayRead(X,&xv));
  PetscCall(MatDenseGetArray(Y,&yv));
  PetscCall(PetscBLASIntCast(X->rmap->n*X->cmap->n,&N));
  PetscCall(PetscBLASIntCast(X->rmap->n,&m));
  PetscCall(PetscBLASIntCast(x->lda,&ldax));
  PetscCall(PetscBLASIntCast(y->lda,&lday));
  if (ldax>m || lday>m) {
    PetscInt j;

    for (j=0; j<X->cmap->n; j++) {
      PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&m,&alpha,xv+j*ldax,&one,yv+j*lday,&one));
    }
  } else {
    PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&N,&alpha,xv,&one,yv,&one));
  }
  PetscCall(MatDenseRestoreArrayRead(X,&xv));
  PetscCall(MatDenseRestoreArray(Y,&yv));
  PetscCall(PetscLogFlops(PetscMax(2.0*N-1,0)));
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
  PetscBLASInt   one = 1,j,nz,lda = 0;

  PetscFunctionBegin;
  PetscCall(MatDenseGetArray(A,&v));
  PetscCall(PetscBLASIntCast(a->lda,&lda));
  if (lda>A->rmap->n) {
    PetscCall(PetscBLASIntCast(A->rmap->n,&nz));
    for (j=0; j<A->cmap->n; j++) {
      PetscStackCallBLAS("BLASscal",BLASscal_(&nz,&alpha,v+j*lda,&one));
    }
  } else {
    PetscCall(PetscBLASIntCast(A->rmap->n*A->cmap->n,&nz));
    PetscStackCallBLAS("BLASscal",BLASscal_(&nz,&alpha,v,&one));
  }
  PetscCall(PetscLogFlops(A->rmap->n*A->cmap->n));
  PetscCall(MatDenseRestoreArray(A,&v));
  PetscFunctionReturn(0);
}

PetscErrorCode MatShift_SeqDense(Mat A,PetscScalar alpha)
{
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;
  PetscScalar    *v;
  PetscInt       j,k;

  PetscFunctionBegin;
  PetscCall(MatDenseGetArray(A,&v));
  k = PetscMin(A->rmap->n,A->cmap->n);
  for (j=0; j<k; j++) v[j+j*a->lda] += alpha;
  PetscCall(PetscLogFlops(k));
  PetscCall(MatDenseRestoreArray(A,&v));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatIsHermitian_SeqDense(Mat A,PetscReal rtol,PetscBool  *fl)
{
  Mat_SeqDense      *a = (Mat_SeqDense*)A->data;
  PetscInt          i,j,m = A->rmap->n,N = a->lda;
  const PetscScalar *v;

  PetscFunctionBegin;
  *fl = PETSC_FALSE;
  if (A->rmap->n != A->cmap->n) PetscFunctionReturn(0);
  PetscCall(MatDenseGetArrayRead(A,&v));
  for (i=0; i<m; i++) {
    for (j=i; j<m; j++) {
      if (PetscAbsScalar(v[i+j*N] - PetscConj(v[j+i*N])) > rtol) {
        goto restore;
      }
    }
  }
  *fl  = PETSC_TRUE;
restore:
  PetscCall(MatDenseRestoreArrayRead(A,&v));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatIsSymmetric_SeqDense(Mat A,PetscReal rtol,PetscBool  *fl)
{
  Mat_SeqDense      *a = (Mat_SeqDense*)A->data;
  PetscInt          i,j,m = A->rmap->n,N = a->lda;
  const PetscScalar *v;

  PetscFunctionBegin;
  *fl = PETSC_FALSE;
  if (A->rmap->n != A->cmap->n) PetscFunctionReturn(0);
  PetscCall(MatDenseGetArrayRead(A,&v));
  for (i=0; i<m; i++) {
    for (j=i; j<m; j++) {
      if (PetscAbsScalar(v[i+j*N] - v[j+i*N]) > rtol) {
        goto restore;
      }
    }
  }
  *fl  = PETSC_TRUE;
restore:
  PetscCall(MatDenseRestoreArrayRead(A,&v));
  PetscFunctionReturn(0);
}

PetscErrorCode MatDuplicateNoCreate_SeqDense(Mat newi,Mat A,MatDuplicateOption cpvalues)
{
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
  PetscInt       lda = (PetscInt)mat->lda,j,m,nlda = lda;
  PetscBool      isdensecpu;

  PetscFunctionBegin;
  PetscCall(PetscLayoutReference(A->rmap,&newi->rmap));
  PetscCall(PetscLayoutReference(A->cmap,&newi->cmap));
  if (cpvalues == MAT_SHARE_NONZERO_PATTERN) { /* propagate LDA */
    PetscCall(MatDenseSetLDA(newi,lda));
  }
  PetscCall(PetscObjectTypeCompare((PetscObject)newi,MATSEQDENSE,&isdensecpu));
  if (isdensecpu) PetscCall(MatSeqDenseSetPreallocation(newi,NULL));
  if (cpvalues == MAT_COPY_VALUES) {
    const PetscScalar *av;
    PetscScalar       *v;

    PetscCall(MatDenseGetArrayRead(A,&av));
    PetscCall(MatDenseGetArrayWrite(newi,&v));
    PetscCall(MatDenseGetLDA(newi,&nlda));
    m    = A->rmap->n;
    if (lda>m || nlda>m) {
      for (j=0; j<A->cmap->n; j++) {
        PetscCall(PetscArraycpy(v+j*nlda,av+j*lda,m));
      }
    } else {
      PetscCall(PetscArraycpy(v,av,A->rmap->n*A->cmap->n));
    }
    PetscCall(MatDenseRestoreArrayWrite(newi,&v));
    PetscCall(MatDenseRestoreArrayRead(A,&av));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatDuplicate_SeqDense(Mat A,MatDuplicateOption cpvalues,Mat *newmat)
{
  PetscFunctionBegin;
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A),newmat));
  PetscCall(MatSetSizes(*newmat,A->rmap->n,A->cmap->n,A->rmap->n,A->cmap->n));
  PetscCall(MatSetType(*newmat,((PetscObject)A)->type_name));
  PetscCall(MatDuplicateNoCreate_SeqDense(*newmat,A,cpvalues));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolve_SeqDense_Internal_LU(Mat A, PetscScalar *x, PetscBLASInt ldx, PetscBLASInt m, PetscBLASInt nrhs, PetscBLASInt k, PetscBool T)
{
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
  PetscBLASInt    info;

  PetscFunctionBegin;
  PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
  PetscStackCallBLAS("LAPACKgetrs",LAPACKgetrs_(T ? "T" : "N",&m,&nrhs,mat->v,&mat->lda,mat->pivots,x,&m,&info));
  PetscCall(PetscFPTrapPop());
  PetscCheck(!info,PETSC_COMM_SELF,PETSC_ERR_LIB,"GETRS - Bad solve %d",(int)info);
  PetscCall(PetscLogFlops(nrhs*(2.0*m*m - m)));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatConjugate_SeqDense(Mat);

static PetscErrorCode MatSolve_SeqDense_Internal_Cholesky(Mat A, PetscScalar *x, PetscBLASInt ldx, PetscBLASInt m, PetscBLASInt nrhs, PetscBLASInt k, PetscBool T)
{
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
  PetscBLASInt    info;

  PetscFunctionBegin;
  if (A->spd) {
    if (PetscDefined(USE_COMPLEX) && T) PetscCall(MatConjugate_SeqDense(A));
    PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
    PetscStackCallBLAS("LAPACKpotrs",LAPACKpotrs_("L",&m,&nrhs,mat->v,&mat->lda,x,&m,&info));
    PetscCall(PetscFPTrapPop());
    PetscCheck(!info,PETSC_COMM_SELF,PETSC_ERR_LIB,"POTRS Bad solve %d",(int)info);
    if (PetscDefined(USE_COMPLEX) && T) PetscCall(MatConjugate_SeqDense(A));
#if defined(PETSC_USE_COMPLEX)
  } else if (A->hermitian) {
    if (T) PetscCall(MatConjugate_SeqDense(A));
    PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
    PetscStackCallBLAS("LAPACKhetrs",LAPACKhetrs_("L",&m,&nrhs,mat->v,&mat->lda,mat->pivots,x,&m,&info));
    PetscCall(PetscFPTrapPop());
    PetscCheck(!info,PETSC_COMM_SELF,PETSC_ERR_LIB,"HETRS Bad solve %d",(int)info);
    if (T) PetscCall(MatConjugate_SeqDense(A));
#endif
  } else { /* symmetric case */
    PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
    PetscStackCallBLAS("LAPACKsytrs",LAPACKsytrs_("L",&m,&nrhs,mat->v,&mat->lda,mat->pivots,x,&m,&info));
    PetscCall(PetscFPTrapPop());
    PetscCheck(!info,PETSC_COMM_SELF,PETSC_ERR_LIB,"SYTRS Bad solve %d",(int)info);
  }
  PetscCall(PetscLogFlops(nrhs*(2.0*m*m - m)));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolve_SeqDense_Internal_QR(Mat A, PetscScalar *x, PetscBLASInt ldx, PetscBLASInt m, PetscBLASInt nrhs, PetscBLASInt k)
{
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
  PetscBLASInt    info;
  char            trans;

  PetscFunctionBegin;
  if (PetscDefined(USE_COMPLEX)) {
    trans = 'C';
  } else {
    trans = 'T';
  }
  PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
  { /* lwork depends on the number of right-hand sides */
    PetscBLASInt nlfwork, lfwork = -1;
    PetscScalar  fwork;

    PetscStackCallBLAS("LAPACKormqr",LAPACKormqr_("L", &trans, &m,&nrhs,&mat->rank,mat->v,&mat->lda,mat->tau,x,&ldx,&fwork,&lfwork,&info));
    nlfwork = (PetscBLASInt)PetscRealPart(fwork);
    if (nlfwork > mat->lfwork) {
      mat->lfwork = nlfwork;
      PetscCall(PetscFree(mat->fwork));
      PetscCall(PetscMalloc1(mat->lfwork,&mat->fwork));
    }
  }
  PetscStackCallBLAS("LAPACKormqr",LAPACKormqr_("L", &trans, &m,&nrhs,&mat->rank,mat->v,&mat->lda,mat->tau,x,&ldx,mat->fwork,&mat->lfwork,&info));
  PetscCall(PetscFPTrapPop());
  PetscCheck(!info,PETSC_COMM_SELF,PETSC_ERR_LIB,"ORMQR - Bad orthogonal transform %d",(int)info);
  PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
  PetscStackCallBLAS("LAPACKtrtrs",LAPACKtrtrs_("U", "N", "N", &mat->rank,&nrhs,mat->v,&mat->lda,x,&ldx,&info));
  PetscCall(PetscFPTrapPop());
  PetscCheck(!info,PETSC_COMM_SELF,PETSC_ERR_LIB,"TRTRS - Bad triangular solve %d",(int)info);
  for (PetscInt j = 0; j < nrhs; j++) {
    for (PetscInt i = mat->rank; i < k; i++) {
      x[j*ldx + i] = 0.;
    }
  }
  PetscCall(PetscLogFlops(nrhs*(4.0*m*mat->rank - PetscSqr(mat->rank))));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolveTranspose_SeqDense_Internal_QR(Mat A, PetscScalar *x, PetscBLASInt ldx, PetscBLASInt m, PetscBLASInt nrhs, PetscBLASInt k)
{
  Mat_SeqDense      *mat = (Mat_SeqDense*)A->data;
  PetscBLASInt      info;

  PetscFunctionBegin;
  if (A->rmap->n == A->cmap->n && mat->rank == A->rmap->n) {
    PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
    PetscStackCallBLAS("LAPACKtrtrs",LAPACKtrtrs_("U", "T", "N", &m,&nrhs,mat->v,&mat->lda,x,&ldx,&info));
    PetscCall(PetscFPTrapPop());
    PetscCheck(!info,PETSC_COMM_SELF,PETSC_ERR_LIB,"TRTRS - Bad triangular solve %d",(int)info);
    if (PetscDefined(USE_COMPLEX)) PetscCall(MatConjugate_SeqDense(A));
    { /* lwork depends on the number of right-hand sides */
      PetscBLASInt nlfwork, lfwork = -1;
      PetscScalar  fwork;

      PetscStackCallBLAS("LAPACKormqr",LAPACKormqr_("L", "N", &m,&nrhs,&mat->rank,mat->v,&mat->lda,mat->tau,x,&ldx,&fwork,&lfwork,&info));
      nlfwork = (PetscBLASInt)PetscRealPart(fwork);
      if (nlfwork > mat->lfwork) {
        mat->lfwork = nlfwork;
        PetscCall(PetscFree(mat->fwork));
        PetscCall(PetscMalloc1(mat->lfwork,&mat->fwork));
      }
    }
    PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
    PetscStackCallBLAS("LAPACKormqr",LAPACKormqr_("L", "N", &m,&nrhs,&mat->rank,mat->v,&mat->lda,mat->tau,x,&ldx,mat->fwork,&mat->lfwork,&info));
    PetscCall(PetscFPTrapPop());
    PetscCheck(!info,PETSC_COMM_SELF,PETSC_ERR_LIB,"ORMQR - Bad orthogonal transform %d",(int)info);
    if (PetscDefined(USE_COMPLEX)) PetscCall(MatConjugate_SeqDense(A));
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"QR factored matrix cannot be used for transpose solve");
  PetscCall(PetscLogFlops(nrhs*(4.0*m*mat->rank - PetscSqr(mat->rank))));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolve_SeqDense_SetUp(Mat A, Vec xx, Vec yy, PetscScalar **_y, PetscBLASInt *_m, PetscBLASInt *_k)
{
  Mat_SeqDense      *mat = (Mat_SeqDense *) A->data;
  PetscScalar       *y;
  PetscBLASInt      m=0, k=0;

  PetscFunctionBegin;
  PetscCall(PetscBLASIntCast(A->rmap->n,&m));
  PetscCall(PetscBLASIntCast(A->cmap->n,&k));
  if (k < m) {
    PetscCall(VecCopy(xx, mat->qrrhs));
    PetscCall(VecGetArray(mat->qrrhs,&y));
  } else {
    PetscCall(VecCopy(xx, yy));
    PetscCall(VecGetArray(yy,&y));
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

  PetscFunctionBegin;
  y   = *_y;
  *_y = NULL;
  k   = *_k;
  m   = *_m;
  if (k < m) {
    PetscScalar *yv;
    PetscCall(VecGetArray(yy,&yv));
    PetscCall(PetscArraycpy(yv, y, k));
    PetscCall(VecRestoreArray(yy,&yv));
    PetscCall(VecRestoreArray(mat->qrrhs, &y));
  } else {
    PetscCall(VecRestoreArray(yy,&y));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolve_SeqDense_LU(Mat A, Vec xx, Vec yy)
{
  PetscScalar    *y = NULL;
  PetscBLASInt   m = 0, k = 0;

  PetscFunctionBegin;
  PetscCall(MatSolve_SeqDense_SetUp(A, xx, yy, &y, &m, &k));
  PetscCall(MatSolve_SeqDense_Internal_LU(A, y, m, m, 1, k, PETSC_FALSE));
  PetscCall(MatSolve_SeqDense_TearDown(A, xx, yy, &y, &m, &k));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolveTranspose_SeqDense_LU(Mat A, Vec xx, Vec yy)
{
  PetscScalar    *y = NULL;
  PetscBLASInt   m = 0, k = 0;

  PetscFunctionBegin;
  PetscCall(MatSolve_SeqDense_SetUp(A, xx, yy, &y, &m, &k));
  PetscCall(MatSolve_SeqDense_Internal_LU(A, y, m, m, 1, k, PETSC_TRUE));
  PetscCall(MatSolve_SeqDense_TearDown(A, xx, yy, &y, &m, &k));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolve_SeqDense_Cholesky(Mat A, Vec xx, Vec yy)
{
  PetscScalar    *y = NULL;
  PetscBLASInt   m = 0, k = 0;

  PetscFunctionBegin;
  PetscCall(MatSolve_SeqDense_SetUp(A, xx, yy, &y, &m, &k));
  PetscCall(MatSolve_SeqDense_Internal_Cholesky(A, y, m, m, 1, k, PETSC_FALSE));
  PetscCall(MatSolve_SeqDense_TearDown(A, xx, yy, &y, &m, &k));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolveTranspose_SeqDense_Cholesky(Mat A, Vec xx, Vec yy)
{
  PetscScalar    *y = NULL;
  PetscBLASInt   m = 0, k = 0;

  PetscFunctionBegin;
  PetscCall(MatSolve_SeqDense_SetUp(A, xx, yy, &y, &m, &k));
  PetscCall(MatSolve_SeqDense_Internal_Cholesky(A, y, m, m, 1, k, PETSC_TRUE));
  PetscCall(MatSolve_SeqDense_TearDown(A, xx, yy, &y, &m, &k));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolve_SeqDense_QR(Mat A, Vec xx, Vec yy)
{
  PetscScalar    *y = NULL;
  PetscBLASInt   m = 0, k = 0;

  PetscFunctionBegin;
  PetscCall(MatSolve_SeqDense_SetUp(A, xx, yy, &y, &m, &k));
  PetscCall(MatSolve_SeqDense_Internal_QR(A, y, PetscMax(m,k), m, 1, k));
  PetscCall(MatSolve_SeqDense_TearDown(A, xx, yy, &y, &m, &k));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolveTranspose_SeqDense_QR(Mat A, Vec xx, Vec yy)
{
  PetscScalar    *y = NULL;
  PetscBLASInt   m = 0, k = 0;

  PetscFunctionBegin;
  PetscCall(MatSolve_SeqDense_SetUp(A, xx, yy, &y, &m, &k));
  PetscCall(MatSolveTranspose_SeqDense_Internal_QR(A, y, PetscMax(m,k), m, 1, k));
  PetscCall(MatSolve_SeqDense_TearDown(A, xx, yy, &y, &m, &k));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatSolve_SeqDense_SetUp(Mat A, Mat B, Mat X, PetscScalar **_y, PetscBLASInt *_ldy, PetscBLASInt *_m, PetscBLASInt *_nrhs, PetscBLASInt *_k)
{
  const PetscScalar *b;
  PetscScalar       *y;
  PetscInt          n, _ldb, _ldx;
  PetscBLASInt      nrhs=0,m=0,k=0,ldb=0,ldx=0,ldy=0;

  PetscFunctionBegin;
  *_ldy=0; *_m=0; *_nrhs=0; *_k=0; *_y = NULL;
  PetscCall(PetscBLASIntCast(A->rmap->n,&m));
  PetscCall(PetscBLASIntCast(A->cmap->n,&k));
  PetscCall(MatGetSize(B,NULL,&n));
  PetscCall(PetscBLASIntCast(n,&nrhs));
  PetscCall(MatDenseGetLDA(B,&_ldb));
  PetscCall(PetscBLASIntCast(_ldb, &ldb));
  PetscCall(MatDenseGetLDA(X,&_ldx));
  PetscCall(PetscBLASIntCast(_ldx, &ldx));
  if (ldx < m) {
    PetscCall(MatDenseGetArrayRead(B,&b));
    PetscCall(PetscMalloc1(nrhs * m, &y));
    if (ldb == m) {
      PetscCall(PetscArraycpy(y,b,ldb*nrhs));
    } else {
      for (PetscInt j = 0; j < nrhs; j++) {
        PetscCall(PetscArraycpy(&y[j*m],&b[j*ldb],m));
      }
    }
    ldy = m;
    PetscCall(MatDenseRestoreArrayRead(B,&b));
  } else {
    if (ldb == ldx) {
      PetscCall(MatCopy(B, X, SAME_NONZERO_PATTERN));
      PetscCall(MatDenseGetArray(X,&y));
    } else {
      PetscCall(MatDenseGetArray(X,&y));
      PetscCall(MatDenseGetArrayRead(B,&b));
      for (PetscInt j = 0; j < nrhs; j++) {
        PetscCall(PetscArraycpy(&y[j*ldx],&b[j*ldb],m));
      }
      PetscCall(MatDenseRestoreArrayRead(B,&b));
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

  PetscFunctionBegin;
  y    = *_y;
  *_y  = NULL;
  k    = *_k;
  ldy = *_ldy;
  nrhs = *_nrhs;
  PetscCall(MatDenseGetLDA(X,&_ldx));
  PetscCall(PetscBLASIntCast(_ldx, &ldx));
  if (ldx != ldy) {
    PetscScalar *xv;
    PetscCall(MatDenseGetArray(X,&xv));
    for (PetscInt j = 0; j < nrhs; j++) {
      PetscCall(PetscArraycpy(&xv[j*ldx],&y[j*ldy],k));
    }
    PetscCall(MatDenseRestoreArray(X,&xv));
    PetscCall(PetscFree(y));
  } else {
    PetscCall(MatDenseRestoreArray(X,&y));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatSolve_SeqDense_LU(Mat A, Mat B, Mat X)
{
  PetscScalar    *y;
  PetscBLASInt   m, k, ldy, nrhs;

  PetscFunctionBegin;
  PetscCall(MatMatSolve_SeqDense_SetUp(A, B, X, &y, &ldy, &m, &nrhs, &k));
  PetscCall(MatSolve_SeqDense_Internal_LU(A, y, ldy, m, nrhs, k, PETSC_FALSE));
  PetscCall(MatMatSolve_SeqDense_TearDown(A, B, X, &y, &ldy, &m, &nrhs, &k));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatSolveTranspose_SeqDense_LU(Mat A, Mat B, Mat X)
{
  PetscScalar    *y;
  PetscBLASInt   m, k, ldy, nrhs;

  PetscFunctionBegin;
  PetscCall(MatMatSolve_SeqDense_SetUp(A, B, X, &y, &ldy, &m, &nrhs, &k));
  PetscCall(MatSolve_SeqDense_Internal_LU(A, y, ldy, m, nrhs, k, PETSC_TRUE));
  PetscCall(MatMatSolve_SeqDense_TearDown(A, B, X, &y, &ldy, &m, &nrhs, &k));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatSolve_SeqDense_Cholesky(Mat A, Mat B, Mat X)
{
  PetscScalar    *y;
  PetscBLASInt   m, k, ldy, nrhs;

  PetscFunctionBegin;
  PetscCall(MatMatSolve_SeqDense_SetUp(A, B, X, &y, &ldy, &m, &nrhs, &k));
  PetscCall(MatSolve_SeqDense_Internal_Cholesky(A, y, ldy, m, nrhs, k, PETSC_FALSE));
  PetscCall(MatMatSolve_SeqDense_TearDown(A, B, X, &y, &ldy, &m, &nrhs, &k));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatSolveTranspose_SeqDense_Cholesky(Mat A, Mat B, Mat X)
{
  PetscScalar    *y;
  PetscBLASInt   m, k, ldy, nrhs;

  PetscFunctionBegin;
  PetscCall(MatMatSolve_SeqDense_SetUp(A, B, X, &y, &ldy, &m, &nrhs, &k));
  PetscCall(MatSolve_SeqDense_Internal_Cholesky(A, y, ldy, m, nrhs, k, PETSC_TRUE));
  PetscCall(MatMatSolve_SeqDense_TearDown(A, B, X, &y, &ldy, &m, &nrhs, &k));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatSolve_SeqDense_QR(Mat A, Mat B, Mat X)
{
  PetscScalar    *y;
  PetscBLASInt   m, k, ldy, nrhs;

  PetscFunctionBegin;
  PetscCall(MatMatSolve_SeqDense_SetUp(A, B, X, &y, &ldy, &m, &nrhs, &k));
  PetscCall(MatSolve_SeqDense_Internal_QR(A, y, ldy, m, nrhs, k));
  PetscCall(MatMatSolve_SeqDense_TearDown(A, B, X, &y, &ldy, &m, &nrhs, &k));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatSolveTranspose_SeqDense_QR(Mat A, Mat B, Mat X)
{
  PetscScalar    *y;
  PetscBLASInt   m, k, ldy, nrhs;

  PetscFunctionBegin;
  PetscCall(MatMatSolve_SeqDense_SetUp(A, B, X, &y, &ldy, &m, &nrhs, &k));
  PetscCall(MatSolveTranspose_SeqDense_Internal_QR(A, y, ldy, m, nrhs, k));
  PetscCall(MatMatSolve_SeqDense_TearDown(A, B, X, &y, &ldy, &m, &nrhs, &k));
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------*/
/* COMMENT: I have chosen to hide row permutation in the pivots,
   rather than put it in the Mat->row slot.*/
PetscErrorCode MatLUFactor_SeqDense(Mat A,IS row,IS col,const MatFactorInfo *minfo)
{
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
  PetscBLASInt   n,m,info;

  PetscFunctionBegin;
  PetscCall(PetscBLASIntCast(A->cmap->n,&n));
  PetscCall(PetscBLASIntCast(A->rmap->n,&m));
  if (!mat->pivots) {
    PetscCall(PetscMalloc1(A->rmap->n,&mat->pivots));
    PetscCall(PetscLogObjectMemory((PetscObject)A,A->rmap->n*sizeof(PetscBLASInt)));
  }
  if (!A->rmap->n || !A->cmap->n) PetscFunctionReturn(0);
  PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
  PetscStackCallBLAS("LAPACKgetrf",LAPACKgetrf_(&m,&n,mat->v,&mat->lda,mat->pivots,&info));
  PetscCall(PetscFPTrapPop());

  PetscCheck(info>=0,PETSC_COMM_SELF,PETSC_ERR_LIB,"Bad argument to LU factorization %d",(int)info);
  PetscCheck(info<=0,PETSC_COMM_SELF,PETSC_ERR_MAT_LU_ZRPVT,"Bad LU factorization %d",(int)info);

  A->ops->solve             = MatSolve_SeqDense_LU;
  A->ops->matsolve          = MatMatSolve_SeqDense_LU;
  A->ops->solvetranspose    = MatSolveTranspose_SeqDense_LU;
  A->ops->matsolvetranspose = MatMatSolveTranspose_SeqDense_LU;
  A->factortype             = MAT_FACTOR_LU;

  PetscCall(PetscFree(A->solvertype));
  PetscCall(PetscStrallocpy(MATSOLVERPETSC,&A->solvertype));

  PetscCall(PetscLogFlops((2.0*A->cmap->n*A->cmap->n*A->cmap->n)/3));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatLUFactorNumeric_SeqDense(Mat fact,Mat A,const MatFactorInfo *info_dummy)
{
  MatFactorInfo  info;

  PetscFunctionBegin;
  PetscCall(MatDuplicateNoCreate_SeqDense(fact,A,MAT_COPY_VALUES));
  PetscCall((*fact->ops->lufactor)(fact,NULL,NULL,&info));
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
  PetscBLASInt   info,n;

  PetscFunctionBegin;
  PetscCall(PetscBLASIntCast(A->cmap->n,&n));
  if (!A->rmap->n || !A->cmap->n) PetscFunctionReturn(0);
  if (A->spd) {
    PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
    PetscStackCallBLAS("LAPACKpotrf",LAPACKpotrf_("L",&n,mat->v,&mat->lda,&info));
    PetscCall(PetscFPTrapPop());
#if defined(PETSC_USE_COMPLEX)
  } else if (A->hermitian) {
    if (!mat->pivots) {
      PetscCall(PetscMalloc1(A->rmap->n,&mat->pivots));
      PetscCall(PetscLogObjectMemory((PetscObject)A,A->rmap->n*sizeof(PetscBLASInt)));
    }
    if (!mat->fwork) {
      PetscScalar dummy;

      mat->lfwork = -1;
      PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
      PetscStackCallBLAS("LAPACKhetrf",LAPACKhetrf_("L",&n,mat->v,&mat->lda,mat->pivots,&dummy,&mat->lfwork,&info));
      PetscCall(PetscFPTrapPop());
      mat->lfwork = (PetscInt)PetscRealPart(dummy);
      PetscCall(PetscMalloc1(mat->lfwork,&mat->fwork));
      PetscCall(PetscLogObjectMemory((PetscObject)A,mat->lfwork*sizeof(PetscBLASInt)));
    }
    PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
    PetscStackCallBLAS("LAPACKhetrf",LAPACKhetrf_("L",&n,mat->v,&mat->lda,mat->pivots,mat->fwork,&mat->lfwork,&info));
    PetscCall(PetscFPTrapPop());
#endif
  } else { /* symmetric case */
    if (!mat->pivots) {
      PetscCall(PetscMalloc1(A->rmap->n,&mat->pivots));
      PetscCall(PetscLogObjectMemory((PetscObject)A,A->rmap->n*sizeof(PetscBLASInt)));
    }
    if (!mat->fwork) {
      PetscScalar dummy;

      mat->lfwork = -1;
      PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
      PetscStackCallBLAS("LAPACKsytrf",LAPACKsytrf_("L",&n,mat->v,&mat->lda,mat->pivots,&dummy,&mat->lfwork,&info));
      PetscCall(PetscFPTrapPop());
      mat->lfwork = (PetscInt)PetscRealPart(dummy);
      PetscCall(PetscMalloc1(mat->lfwork,&mat->fwork));
      PetscCall(PetscLogObjectMemory((PetscObject)A,mat->lfwork*sizeof(PetscBLASInt)));
    }
    PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
    PetscStackCallBLAS("LAPACKsytrf",LAPACKsytrf_("L",&n,mat->v,&mat->lda,mat->pivots,mat->fwork,&mat->lfwork,&info));
    PetscCall(PetscFPTrapPop());
  }
  PetscCheck(!info,PETSC_COMM_SELF,PETSC_ERR_MAT_CH_ZRPVT,"Bad factorization: zero pivot in row %" PetscInt_FMT,(PetscInt)info-1);

  A->ops->solve             = MatSolve_SeqDense_Cholesky;
  A->ops->matsolve          = MatMatSolve_SeqDense_Cholesky;
  A->ops->solvetranspose    = MatSolveTranspose_SeqDense_Cholesky;
  A->ops->matsolvetranspose = MatMatSolveTranspose_SeqDense_Cholesky;
  A->factortype             = MAT_FACTOR_CHOLESKY;

  PetscCall(PetscFree(A->solvertype));
  PetscCall(PetscStrallocpy(MATSOLVERPETSC,&A->solvertype));

  PetscCall(PetscLogFlops((1.0*A->cmap->n*A->cmap->n*A->cmap->n)/3.0));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCholeskyFactorNumeric_SeqDense(Mat fact,Mat A,const MatFactorInfo *info_dummy)
{
  MatFactorInfo  info;

  PetscFunctionBegin;
  info.fill = 1.0;

  PetscCall(MatDuplicateNoCreate_SeqDense(fact,A,MAT_COPY_VALUES));
  PetscCall((*fact->ops->choleskyfactor)(fact,NULL,&info));
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
  PetscBLASInt   n,m,info, min, max;

  PetscFunctionBegin;
  PetscCall(PetscBLASIntCast(A->cmap->n,&n));
  PetscCall(PetscBLASIntCast(A->rmap->n,&m));
  max = PetscMax(m, n);
  min = PetscMin(m, n);
  if (!mat->tau) {
    PetscCall(PetscMalloc1(min,&mat->tau));
    PetscCall(PetscLogObjectMemory((PetscObject)A,min*sizeof(PetscScalar)));
  }
  if (!mat->pivots) {
    PetscCall(PetscMalloc1(n,&mat->pivots));
    PetscCall(PetscLogObjectMemory((PetscObject)A,n*sizeof(PetscScalar)));
  }
  if (!mat->qrrhs) {
    PetscCall(MatCreateVecs(A, NULL, &(mat->qrrhs)));
  }
  if (!A->rmap->n || !A->cmap->n) PetscFunctionReturn(0);
  if (!mat->fwork) {
    PetscScalar dummy;

    mat->lfwork = -1;
    PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
    PetscStackCallBLAS("LAPACKgeqrf",LAPACKgeqrf_(&m,&n,mat->v,&mat->lda,mat->tau,&dummy,&mat->lfwork,&info));
    PetscCall(PetscFPTrapPop());
    mat->lfwork = (PetscInt)PetscRealPart(dummy);
    PetscCall(PetscMalloc1(mat->lfwork,&mat->fwork));
    PetscCall(PetscLogObjectMemory((PetscObject)A,mat->lfwork*sizeof(PetscBLASInt)));
  }
  PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
  PetscStackCallBLAS("LAPACKgeqrf",LAPACKgeqrf_(&m,&n,mat->v,&mat->lda,mat->tau,mat->fwork,&mat->lfwork,&info));
  PetscCall(PetscFPTrapPop());
  PetscCheck(!info,PETSC_COMM_SELF,PETSC_ERR_LIB,"Bad argument to QR factorization %d",(int)info);
  // TODO: try to estimate rank or test for and use geqp3 for rank revealing QR.  For now just say rank is min of m and n
  mat->rank = min;

  A->ops->solve             = MatSolve_SeqDense_QR;
  A->ops->matsolve          = MatMatSolve_SeqDense_QR;
  A->factortype             = MAT_FACTOR_QR;
  if (m == n) {
    A->ops->solvetranspose    = MatSolveTranspose_SeqDense_QR;
    A->ops->matsolvetranspose = MatMatSolveTranspose_SeqDense_QR;
  }

  PetscCall(PetscFree(A->solvertype));
  PetscCall(PetscStrallocpy(MATSOLVERPETSC,&A->solvertype));

  PetscCall(PetscLogFlops(2.0*min*min*(max-min/3.0)));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatQRFactorNumeric_SeqDense(Mat fact,Mat A,const MatFactorInfo *info_dummy)
{
  MatFactorInfo  info;

  PetscFunctionBegin;
  info.fill = 1.0;

  PetscCall(MatDuplicateNoCreate_SeqDense(fact,A,MAT_COPY_VALUES));
  PetscUseMethod(fact,"MatQRFactor_C",(Mat,IS,const MatFactorInfo *),(fact,NULL,&info));
  PetscFunctionReturn(0);
}

PetscErrorCode MatQRFactorSymbolic_SeqDense(Mat fact,Mat A,IS row,const MatFactorInfo *info)
{
  PetscFunctionBegin;
  fact->assembled                  = PETSC_TRUE;
  fact->preallocated               = PETSC_TRUE;
  PetscCall(PetscObjectComposeFunction((PetscObject)fact,"MatQRFactorNumeric_C",MatQRFactorNumeric_SeqDense));
  PetscFunctionReturn(0);
}

/* uses LAPACK */
PETSC_INTERN PetscErrorCode MatGetFactor_seqdense_petsc(Mat A,MatFactorType ftype,Mat *fact)
{
  PetscFunctionBegin;
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A),fact));
  PetscCall(MatSetSizes(*fact,A->rmap->n,A->cmap->n,A->rmap->n,A->cmap->n));
  PetscCall(MatSetType(*fact,MATDENSE));
  (*fact)->trivialsymbolic = PETSC_TRUE;
  if (ftype == MAT_FACTOR_LU || ftype == MAT_FACTOR_ILU) {
    (*fact)->ops->lufactorsymbolic = MatLUFactorSymbolic_SeqDense;
    (*fact)->ops->ilufactorsymbolic = MatLUFactorSymbolic_SeqDense;
  } else if (ftype == MAT_FACTOR_CHOLESKY || ftype == MAT_FACTOR_ICC) {
    (*fact)->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_SeqDense;
  } else if (ftype == MAT_FACTOR_QR) {
    PetscCall(PetscObjectComposeFunction((PetscObject)(*fact),"MatQRFactorSymbolic_C",MatQRFactorSymbolic_SeqDense));
  }
  (*fact)->factortype = ftype;

  PetscCall(PetscFree((*fact)->solvertype));
  PetscCall(PetscStrallocpy(MATSOLVERPETSC,&(*fact)->solvertype));
  PetscCall(PetscStrallocpy(MATORDERINGEXTERNAL,(char**)&(*fact)->preferredordering[MAT_FACTOR_LU]));
  PetscCall(PetscStrallocpy(MATORDERINGEXTERNAL,(char**)&(*fact)->preferredordering[MAT_FACTOR_ILU]));
  PetscCall(PetscStrallocpy(MATORDERINGEXTERNAL,(char**)&(*fact)->preferredordering[MAT_FACTOR_CHOLESKY]));
  PetscCall(PetscStrallocpy(MATORDERINGEXTERNAL,(char**)&(*fact)->preferredordering[MAT_FACTOR_ICC]));
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------*/
static PetscErrorCode MatSOR_SeqDense(Mat A,Vec bb,PetscReal omega,MatSORType flag,PetscReal shift,PetscInt its,PetscInt lits,Vec xx)
{
  Mat_SeqDense      *mat = (Mat_SeqDense*)A->data;
  PetscScalar       *x,*v = mat->v,zero = 0.0,xt;
  const PetscScalar *b;
  PetscInt          m = A->rmap->n,i;
  PetscBLASInt      o = 1,bm = 0;

  PetscFunctionBegin;
#if defined(PETSC_HAVE_CUDA)
  PetscCheck(A->offloadmask != PETSC_OFFLOAD_GPU,PETSC_COMM_SELF,PETSC_ERR_SUP,"Not implemented");
#endif
  if (shift == -1) shift = 0.0; /* negative shift indicates do not error on zero diagonal; this code never zeros on zero diagonal */
  PetscCall(PetscBLASIntCast(m,&bm));
  if (flag & SOR_ZERO_INITIAL_GUESS) {
    /* this is a hack fix, should have another version without the second BLASdotu */
    PetscCall(VecSet(xx,zero));
  }
  PetscCall(VecGetArray(xx,&x));
  PetscCall(VecGetArrayRead(bb,&b));
  its  = its*lits;
  PetscCheck(its > 0,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Relaxation requires global its %" PetscInt_FMT " and local its %" PetscInt_FMT " both positive",its,lits);
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
  PetscCall(VecRestoreArrayRead(bb,&b));
  PetscCall(VecRestoreArray(xx,&x));
  PetscFunctionReturn(0);
}

/* -----------------------------------------------------------------*/
PetscErrorCode MatMultTranspose_SeqDense(Mat A,Vec xx,Vec yy)
{
  Mat_SeqDense      *mat = (Mat_SeqDense*)A->data;
  const PetscScalar *v   = mat->v,*x;
  PetscScalar       *y;
  PetscBLASInt      m, n,_One=1;
  PetscScalar       _DOne=1.0,_DZero=0.0;

  PetscFunctionBegin;
  PetscCall(PetscBLASIntCast(A->rmap->n,&m));
  PetscCall(PetscBLASIntCast(A->cmap->n,&n));
  PetscCall(VecGetArrayRead(xx,&x));
  PetscCall(VecGetArrayWrite(yy,&y));
  if (!A->rmap->n || !A->cmap->n) {
    PetscBLASInt i;
    for (i=0; i<n; i++) y[i] = 0.0;
  } else {
    PetscStackCallBLAS("BLASgemv",BLASgemv_("T",&m,&n,&_DOne,v,&mat->lda,x,&_One,&_DZero,y,&_One));
    PetscCall(PetscLogFlops(2.0*A->rmap->n*A->cmap->n - A->cmap->n));
  }
  PetscCall(VecRestoreArrayRead(xx,&x));
  PetscCall(VecRestoreArrayWrite(yy,&y));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMult_SeqDense(Mat A,Vec xx,Vec yy)
{
  Mat_SeqDense      *mat = (Mat_SeqDense*)A->data;
  PetscScalar       *y,_DOne=1.0,_DZero=0.0;
  PetscBLASInt      m, n, _One=1;
  const PetscScalar *v = mat->v,*x;

  PetscFunctionBegin;
  PetscCall(PetscBLASIntCast(A->rmap->n,&m));
  PetscCall(PetscBLASIntCast(A->cmap->n,&n));
  PetscCall(VecGetArrayRead(xx,&x));
  PetscCall(VecGetArrayWrite(yy,&y));
  if (!A->rmap->n || !A->cmap->n) {
    PetscBLASInt i;
    for (i=0; i<m; i++) y[i] = 0.0;
  } else {
    PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&m,&n,&_DOne,v,&(mat->lda),x,&_One,&_DZero,y,&_One));
    PetscCall(PetscLogFlops(2.0*A->rmap->n*A->cmap->n - A->rmap->n));
  }
  PetscCall(VecRestoreArrayRead(xx,&x));
  PetscCall(VecRestoreArrayWrite(yy,&y));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultAdd_SeqDense(Mat A,Vec xx,Vec zz,Vec yy)
{
  Mat_SeqDense      *mat = (Mat_SeqDense*)A->data;
  const PetscScalar *v = mat->v,*x;
  PetscScalar       *y,_DOne=1.0;
  PetscBLASInt      m, n, _One=1;

  PetscFunctionBegin;
  PetscCall(PetscBLASIntCast(A->rmap->n,&m));
  PetscCall(PetscBLASIntCast(A->cmap->n,&n));
  PetscCall(VecCopy(zz,yy));
  if (!A->rmap->n || !A->cmap->n) PetscFunctionReturn(0);
  PetscCall(VecGetArrayRead(xx,&x));
  PetscCall(VecGetArray(yy,&y));
  PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&m,&n,&_DOne,v,&(mat->lda),x,&_One,&_DOne,y,&_One));
  PetscCall(VecRestoreArrayRead(xx,&x));
  PetscCall(VecRestoreArray(yy,&y));
  PetscCall(PetscLogFlops(2.0*A->rmap->n*A->cmap->n));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultTransposeAdd_SeqDense(Mat A,Vec xx,Vec zz,Vec yy)
{
  Mat_SeqDense      *mat = (Mat_SeqDense*)A->data;
  const PetscScalar *v = mat->v,*x;
  PetscScalar       *y;
  PetscBLASInt      m, n, _One=1;
  PetscScalar       _DOne=1.0;

  PetscFunctionBegin;
  PetscCall(PetscBLASIntCast(A->rmap->n,&m));
  PetscCall(PetscBLASIntCast(A->cmap->n,&n));
  PetscCall(VecCopy(zz,yy));
  if (!A->rmap->n || !A->cmap->n) PetscFunctionReturn(0);
  PetscCall(VecGetArrayRead(xx,&x));
  PetscCall(VecGetArray(yy,&y));
  PetscStackCallBLAS("BLASgemv",BLASgemv_("T",&m,&n,&_DOne,v,&(mat->lda),x,&_One,&_DOne,y,&_One));
  PetscCall(VecRestoreArrayRead(xx,&x));
  PetscCall(VecRestoreArray(yy,&y));
  PetscCall(PetscLogFlops(2.0*A->rmap->n*A->cmap->n));
  PetscFunctionReturn(0);
}

/* -----------------------------------------------------------------*/
static PetscErrorCode MatGetRow_SeqDense(Mat A,PetscInt row,PetscInt *ncols,PetscInt **cols,PetscScalar **vals)
{
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
  PetscInt       i;

  PetscFunctionBegin;
  *ncols = A->cmap->n;
  if (cols) {
    PetscCall(PetscMalloc1(A->cmap->n,cols));
    for (i=0; i<A->cmap->n; i++) (*cols)[i] = i;
  }
  if (vals) {
    const PetscScalar *v;

    PetscCall(MatDenseGetArrayRead(A,&v));
    PetscCall(PetscMalloc1(A->cmap->n,vals));
    v   += row;
    for (i=0; i<A->cmap->n; i++) {(*vals)[i] = *v; v += mat->lda;}
    PetscCall(MatDenseRestoreArrayRead(A,&v));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatRestoreRow_SeqDense(Mat A,PetscInt row,PetscInt *ncols,PetscInt **cols,PetscScalar **vals)
{
  PetscFunctionBegin;
  if (ncols) *ncols = 0;
  if (cols) PetscCall(PetscFree(*cols));
  if (vals) PetscCall(PetscFree(*vals));
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

  PetscFunctionBegin;
  PetscCall(MatDenseGetArray(A,&av));
  if (!mat->roworiented) {
    if (addv == INSERT_VALUES) {
      for (j=0; j<n; j++) {
        if (indexn[j] < 0) {idx += m; continue;}
        PetscCheck(indexn[j] < A->cmap->n,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Column too large: col %" PetscInt_FMT " max %" PetscInt_FMT,indexn[j],A->cmap->n-1);
        for (i=0; i<m; i++) {
          if (indexm[i] < 0) {idx++; continue;}
          PetscCheck(indexm[i] < A->rmap->n,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %" PetscInt_FMT " max %" PetscInt_FMT,indexm[i],A->rmap->n-1);
          av[indexn[j]*mat->lda + indexm[i]] = v[idx++];
        }
      }
    } else {
      for (j=0; j<n; j++) {
        if (indexn[j] < 0) {idx += m; continue;}
        PetscCheck(indexn[j] < A->cmap->n,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Column too large: col %" PetscInt_FMT " max %" PetscInt_FMT,indexn[j],A->cmap->n-1);
        for (i=0; i<m; i++) {
          if (indexm[i] < 0) {idx++; continue;}
          PetscCheck(indexm[i] < A->rmap->n,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %" PetscInt_FMT " max %" PetscInt_FMT,indexm[i],A->rmap->n-1);
          av[indexn[j]*mat->lda + indexm[i]] += v[idx++];
        }
      }
    }
  } else {
    if (addv == INSERT_VALUES) {
      for (i=0; i<m; i++) {
        if (indexm[i] < 0) { idx += n; continue;}
        PetscCheck(indexm[i] < A->rmap->n,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %" PetscInt_FMT " max %" PetscInt_FMT,indexm[i],A->rmap->n-1);
        for (j=0; j<n; j++) {
          if (indexn[j] < 0) { idx++; continue;}
          PetscCheck(indexn[j] < A->cmap->n,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Column too large: col %" PetscInt_FMT " max %" PetscInt_FMT,indexn[j],A->cmap->n-1);
          av[indexn[j]*mat->lda + indexm[i]] = v[idx++];
        }
      }
    } else {
      for (i=0; i<m; i++) {
        if (indexm[i] < 0) { idx += n; continue;}
        PetscCheck(indexm[i] < A->rmap->n,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %" PetscInt_FMT " max %" PetscInt_FMT,indexm[i],A->rmap->n-1);
        for (j=0; j<n; j++) {
          if (indexn[j] < 0) { idx++; continue;}
          PetscCheck(indexn[j] < A->cmap->n,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Column too large: col %" PetscInt_FMT " max %" PetscInt_FMT,indexn[j],A->cmap->n-1);
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
  PetscCall(MatDenseRestoreArray(A,&av));
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

  PetscFunctionBegin;
  PetscCall(MatDenseGetArrayRead(A,&vv));
  /* row-oriented output */
  for (i=0; i<m; i++) {
    if (indexm[i] < 0) {v += n;continue;}
    PetscCheck(indexm[i] < A->rmap->n,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row %" PetscInt_FMT " requested larger than number rows %" PetscInt_FMT,indexm[i],A->rmap->n);
    for (j=0; j<n; j++) {
      if (indexn[j] < 0) {v++; continue;}
      PetscCheck(indexn[j] < A->cmap->n,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Column %" PetscInt_FMT " requested larger than number columns %" PetscInt_FMT,indexn[j],A->cmap->n);
      *v++ = vv[indexn[j]*mat->lda + indexm[i]];
    }
  }
  PetscCall(MatDenseRestoreArrayRead(A,&vv));
  PetscFunctionReturn(0);
}

/* -----------------------------------------------------------------*/

PetscErrorCode MatView_Dense_Binary(Mat mat,PetscViewer viewer)
{
  PetscBool         skipHeader;
  PetscViewerFormat format;
  PetscInt          header[4],M,N,m,lda,i,j,k;
  const PetscScalar *v;
  PetscScalar       *vwork;

  PetscFunctionBegin;
  PetscCall(PetscViewerSetUp(viewer));
  PetscCall(PetscViewerBinaryGetSkipHeader(viewer,&skipHeader));
  PetscCall(PetscViewerGetFormat(viewer,&format));
  if (skipHeader) format = PETSC_VIEWER_NATIVE;

  PetscCall(MatGetSize(mat,&M,&N));

  /* write matrix header */
  header[0] = MAT_FILE_CLASSID; header[1] = M; header[2] = N;
  header[3] = (format == PETSC_VIEWER_NATIVE) ? MATRIX_BINARY_FORMAT_DENSE : M*N;
  if (!skipHeader) PetscCall(PetscViewerBinaryWrite(viewer,header,4,PETSC_INT));

  PetscCall(MatGetLocalSize(mat,&m,NULL));
  if (format != PETSC_VIEWER_NATIVE) {
    PetscInt nnz = m*N, *iwork;
    /* store row lengths for each row */
    PetscCall(PetscMalloc1(nnz,&iwork));
    for (i=0; i<m; i++) iwork[i] = N;
    PetscCall(PetscViewerBinaryWriteAll(viewer,iwork,m,PETSC_DETERMINE,PETSC_DETERMINE,PETSC_INT));
    /* store column indices (zero start index) */
    for (k=0, i=0; i<m; i++)
      for (j=0; j<N; j++, k++)
        iwork[k] = j;
    PetscCall(PetscViewerBinaryWriteAll(viewer,iwork,nnz,PETSC_DETERMINE,PETSC_DETERMINE,PETSC_INT));
    PetscCall(PetscFree(iwork));
  }
  /* store matrix values as a dense matrix in row major order */
  PetscCall(PetscMalloc1(m*N,&vwork));
  PetscCall(MatDenseGetArrayRead(mat,&v));
  PetscCall(MatDenseGetLDA(mat,&lda));
  for (k=0, i=0; i<m; i++)
    for (j=0; j<N; j++, k++)
      vwork[k] = v[i+lda*j];
  PetscCall(MatDenseRestoreArrayRead(mat,&v));
  PetscCall(PetscViewerBinaryWriteAll(viewer,vwork,m*N,PETSC_DETERMINE,PETSC_DETERMINE,PETSC_SCALAR));
  PetscCall(PetscFree(vwork));
  PetscFunctionReturn(0);
}

PetscErrorCode MatLoad_Dense_Binary(Mat mat,PetscViewer viewer)
{
  PetscBool      skipHeader;
  PetscInt       header[4],M,N,m,nz,lda,i,j,k;
  PetscInt       rows,cols;
  PetscScalar    *v,*vwork;

  PetscFunctionBegin;
  PetscCall(PetscViewerSetUp(viewer));
  PetscCall(PetscViewerBinaryGetSkipHeader(viewer,&skipHeader));

  if (!skipHeader) {
    PetscCall(PetscViewerBinaryRead(viewer,header,4,NULL,PETSC_INT));
    PetscCheck(header[0] == MAT_FILE_CLASSID,PetscObjectComm((PetscObject)viewer),PETSC_ERR_FILE_UNEXPECTED,"Not a matrix object in file");
    M = header[1]; N = header[2];
    PetscCheck(M >= 0,PetscObjectComm((PetscObject)viewer),PETSC_ERR_FILE_UNEXPECTED,"Matrix row size (%" PetscInt_FMT ") in file is negative",M);
    PetscCheck(N >= 0,PetscObjectComm((PetscObject)viewer),PETSC_ERR_FILE_UNEXPECTED,"Matrix column size (%" PetscInt_FMT ") in file is negative",N);
    nz = header[3];
    PetscCheck(nz == MATRIX_BINARY_FORMAT_DENSE || nz >= 0,PetscObjectComm((PetscObject)viewer),PETSC_ERR_FILE_UNEXPECTED,"Unknown matrix format %" PetscInt_FMT " in file",nz);
  } else {
    PetscCall(MatGetSize(mat,&M,&N));
    PetscCheck(M >= 0 && N >= 0,PETSC_COMM_SELF,PETSC_ERR_USER,"Matrix binary file header was skipped, thus the user must specify the global sizes of input matrix");
    nz = MATRIX_BINARY_FORMAT_DENSE;
  }

  /* setup global sizes if not set */
  if (mat->rmap->N < 0) mat->rmap->N = M;
  if (mat->cmap->N < 0) mat->cmap->N = N;
  PetscCall(MatSetUp(mat));
  /* check if global sizes are correct */
  PetscCall(MatGetSize(mat,&rows,&cols));
  PetscCheck(M == rows && N == cols,PetscObjectComm((PetscObject)viewer),PETSC_ERR_FILE_UNEXPECTED, "Matrix in file of different sizes (%" PetscInt_FMT ", %" PetscInt_FMT ") than the input matrix (%" PetscInt_FMT ", %" PetscInt_FMT ")",M,N,rows,cols);

  PetscCall(MatGetSize(mat,NULL,&N));
  PetscCall(MatGetLocalSize(mat,&m,NULL));
  PetscCall(MatDenseGetArray(mat,&v));
  PetscCall(MatDenseGetLDA(mat,&lda));
  if (nz == MATRIX_BINARY_FORMAT_DENSE) {  /* matrix in file is dense format */
    PetscInt nnz = m*N;
    /* read in matrix values */
    PetscCall(PetscMalloc1(nnz,&vwork));
    PetscCall(PetscViewerBinaryReadAll(viewer,vwork,nnz,PETSC_DETERMINE,PETSC_DETERMINE,PETSC_SCALAR));
    /* store values in column major order */
    for (j=0; j<N; j++)
      for (i=0; i<m; i++)
        v[i+lda*j] = vwork[i*N+j];
    PetscCall(PetscFree(vwork));
  } else { /* matrix in file is sparse format */
    PetscInt nnz = 0, *rlens, *icols;
    /* read in row lengths */
    PetscCall(PetscMalloc1(m,&rlens));
    PetscCall(PetscViewerBinaryReadAll(viewer,rlens,m,PETSC_DETERMINE,PETSC_DETERMINE,PETSC_INT));
    for (i=0; i<m; i++) nnz += rlens[i];
    /* read in column indices and values */
    PetscCall(PetscMalloc2(nnz,&icols,nnz,&vwork));
    PetscCall(PetscViewerBinaryReadAll(viewer,icols,nnz,PETSC_DETERMINE,PETSC_DETERMINE,PETSC_INT));
    PetscCall(PetscViewerBinaryReadAll(viewer,vwork,nnz,PETSC_DETERMINE,PETSC_DETERMINE,PETSC_SCALAR));
    /* store values in column major order */
    for (k=0, i=0; i<m; i++)
      for (j=0; j<rlens[i]; j++, k++)
        v[i+lda*icols[k]] = vwork[k];
    PetscCall(PetscFree(rlens));
    PetscCall(PetscFree2(icols,vwork));
  }
  PetscCall(MatDenseRestoreArray(mat,&v));
  PetscCall(MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

PetscErrorCode MatLoad_SeqDense(Mat newMat, PetscViewer viewer)
{
  PetscBool      isbinary, ishdf5;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(newMat,MAT_CLASSID,1);
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  /* force binary viewer to load .info file if it has not yet done so */
  PetscCall(PetscViewerSetUp(viewer));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERHDF5,  &ishdf5));
  if (isbinary) {
    PetscCall(MatLoad_Dense_Binary(newMat,viewer));
  } else if (ishdf5) {
#if defined(PETSC_HAVE_HDF5)
    PetscCall(MatLoad_Dense_HDF5(newMat,viewer));
#else
    SETERRQ(PetscObjectComm((PetscObject)newMat),PETSC_ERR_SUP,"HDF5 not supported in this build.\nPlease reconfigure using --download-hdf5");
#endif
  } else {
    SETERRQ(PetscObjectComm((PetscObject)newMat),PETSC_ERR_SUP,"Viewer type %s not yet supported for reading %s matrices",((PetscObject)viewer)->type_name,((PetscObject)newMat)->type_name);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatView_SeqDense_ASCII(Mat A,PetscViewer viewer)
{
  Mat_SeqDense      *a = (Mat_SeqDense*)A->data;
  PetscInt          i,j;
  const char        *name;
  PetscScalar       *v,*av;
  PetscViewerFormat format;
#if defined(PETSC_USE_COMPLEX)
  PetscBool         allreal = PETSC_TRUE;
#endif

  PetscFunctionBegin;
  PetscCall(MatDenseGetArrayRead(A,(const PetscScalar**)&av));
  PetscCall(PetscViewerGetFormat(viewer,&format));
  if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
    PetscFunctionReturn(0);  /* do nothing for now */
  } else if (format == PETSC_VIEWER_ASCII_COMMON) {
    PetscCall(PetscViewerASCIIUseTabs(viewer,PETSC_FALSE));
    for (i=0; i<A->rmap->n; i++) {
      v    = av + i;
      PetscCall(PetscViewerASCIIPrintf(viewer,"row %" PetscInt_FMT ":",i));
      for (j=0; j<A->cmap->n; j++) {
#if defined(PETSC_USE_COMPLEX)
        if (PetscRealPart(*v) != 0.0 && PetscImaginaryPart(*v) != 0.0) {
          PetscCall(PetscViewerASCIIPrintf(viewer," (%" PetscInt_FMT ", %g + %g i) ",j,(double)PetscRealPart(*v),(double)PetscImaginaryPart(*v)));
        } else if (PetscRealPart(*v)) {
          PetscCall(PetscViewerASCIIPrintf(viewer," (%" PetscInt_FMT ", %g) ",j,(double)PetscRealPart(*v)));
        }
#else
        if (*v) {
          PetscCall(PetscViewerASCIIPrintf(viewer," (%" PetscInt_FMT ", %g) ",j,(double)*v));
        }
#endif
        v += a->lda;
      }
      PetscCall(PetscViewerASCIIPrintf(viewer,"\n"));
    }
    PetscCall(PetscViewerASCIIUseTabs(viewer,PETSC_TRUE));
  } else {
    PetscCall(PetscViewerASCIIUseTabs(viewer,PETSC_FALSE));
#if defined(PETSC_USE_COMPLEX)
    /* determine if matrix has all real values */
    for (j=0; j<A->cmap->n; j++) {
      v = av + j*a->lda;
      for (i=0; i<A->rmap->n; i++) {
        if (PetscImaginaryPart(v[i])) { allreal = PETSC_FALSE; break;}
      }
    }
#endif
    if (format == PETSC_VIEWER_ASCII_MATLAB) {
      PetscCall(PetscObjectGetName((PetscObject)A,&name));
      PetscCall(PetscViewerASCIIPrintf(viewer,"%% Size = %" PetscInt_FMT " %" PetscInt_FMT " \n",A->rmap->n,A->cmap->n));
      PetscCall(PetscViewerASCIIPrintf(viewer,"%s = zeros(%" PetscInt_FMT ",%" PetscInt_FMT ");\n",name,A->rmap->n,A->cmap->n));
      PetscCall(PetscViewerASCIIPrintf(viewer,"%s = [\n",name));
    }

    for (i=0; i<A->rmap->n; i++) {
      v = av + i;
      for (j=0; j<A->cmap->n; j++) {
#if defined(PETSC_USE_COMPLEX)
        if (allreal) {
          PetscCall(PetscViewerASCIIPrintf(viewer,"%18.16e ",(double)PetscRealPart(*v)));
        } else {
          PetscCall(PetscViewerASCIIPrintf(viewer,"%18.16e + %18.16ei ",(double)PetscRealPart(*v),(double)PetscImaginaryPart(*v)));
        }
#else
        PetscCall(PetscViewerASCIIPrintf(viewer,"%18.16e ",(double)*v));
#endif
        v += a->lda;
      }
      PetscCall(PetscViewerASCIIPrintf(viewer,"\n"));
    }
    if (format == PETSC_VIEWER_ASCII_MATLAB) {
      PetscCall(PetscViewerASCIIPrintf(viewer,"];\n"));
    }
    PetscCall(PetscViewerASCIIUseTabs(viewer,PETSC_TRUE));
  }
  PetscCall(MatDenseRestoreArrayRead(A,(const PetscScalar**)&av));
  PetscCall(PetscViewerFlush(viewer));
  PetscFunctionReturn(0);
}

#include <petscdraw.h>
static PetscErrorCode MatView_SeqDense_Draw_Zoom(PetscDraw draw,void *Aa)
{
  Mat               A  = (Mat) Aa;
  PetscInt          m  = A->rmap->n,n = A->cmap->n,i,j;
  int               color = PETSC_DRAW_WHITE;
  const PetscScalar *v;
  PetscViewer       viewer;
  PetscReal         xl,yl,xr,yr,x_l,x_r,y_l,y_r;
  PetscViewerFormat format;

  PetscFunctionBegin;
  PetscCall(PetscObjectQuery((PetscObject)A,"Zoomviewer",(PetscObject*)&viewer));
  PetscCall(PetscViewerGetFormat(viewer,&format));
  PetscCall(PetscDrawGetCoordinates(draw,&xl,&yl,&xr,&yr));

  /* Loop over matrix elements drawing boxes */
  PetscCall(MatDenseGetArrayRead(A,&v));
  if (format != PETSC_VIEWER_DRAW_CONTOUR) {
    PetscDrawCollectiveBegin(draw);
    /* Blue for negative and Red for positive */
    for (j = 0; j < n; j++) {
      x_l = j; x_r = x_l + 1.0;
      for (i = 0; i < m; i++) {
        y_l = m - i - 1.0;
        y_r = y_l + 1.0;
        if (PetscRealPart(v[j*m+i]) >  0.) color = PETSC_DRAW_RED;
        else if (PetscRealPart(v[j*m+i]) <  0.) color = PETSC_DRAW_BLUE;
        else continue;
        PetscCall(PetscDrawRectangle(draw,x_l,y_l,x_r,y_r,color,color,color,color));
      }
    }
    PetscDrawCollectiveEnd(draw);
  } else {
    /* use contour shading to indicate magnitude of values */
    /* first determine max of all nonzero values */
    PetscReal minv = 0.0, maxv = 0.0;
    PetscDraw popup;

    for (i=0; i < m*n; i++) {
      if (PetscAbsScalar(v[i]) > maxv) maxv = PetscAbsScalar(v[i]);
    }
    if (minv >= maxv) maxv = minv + PETSC_SMALL;
    PetscCall(PetscDrawGetPopup(draw,&popup));
    PetscCall(PetscDrawScalePopup(popup,minv,maxv));

    PetscDrawCollectiveBegin(draw);
    for (j=0; j<n; j++) {
      x_l = j;
      x_r = x_l + 1.0;
      for (i=0; i<m; i++) {
        y_l   = m - i - 1.0;
        y_r   = y_l + 1.0;
        color = PetscDrawRealToColor(PetscAbsScalar(v[j*m+i]),minv,maxv);
        PetscCall(PetscDrawRectangle(draw,x_l,y_l,x_r,y_r,color,color,color,color));
      }
    }
    PetscDrawCollectiveEnd(draw);
  }
  PetscCall(MatDenseRestoreArrayRead(A,&v));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatView_SeqDense_Draw(Mat A,PetscViewer viewer)
{
  PetscDraw      draw;
  PetscBool      isnull;
  PetscReal      xr,yr,xl,yl,h,w;

  PetscFunctionBegin;
  PetscCall(PetscViewerDrawGetDraw(viewer,0,&draw));
  PetscCall(PetscDrawIsNull(draw,&isnull));
  if (isnull) PetscFunctionReturn(0);

  xr   = A->cmap->n; yr = A->rmap->n; h = yr/10.0; w = xr/10.0;
  xr  += w;          yr += h;        xl = -w;     yl = -h;
  PetscCall(PetscDrawSetCoordinates(draw,xl,yl,xr,yr));
  PetscCall(PetscObjectCompose((PetscObject)A,"Zoomviewer",(PetscObject)viewer));
  PetscCall(PetscDrawZoom(draw,MatView_SeqDense_Draw_Zoom,A));
  PetscCall(PetscObjectCompose((PetscObject)A,"Zoomviewer",NULL));
  PetscCall(PetscDrawSave(draw));
  PetscFunctionReturn(0);
}

PetscErrorCode MatView_SeqDense(Mat A,PetscViewer viewer)
{
  PetscBool      iascii,isbinary,isdraw;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&isdraw));
  if (iascii) PetscCall(MatView_SeqDense_ASCII(A,viewer));
  else if (isbinary) PetscCall(MatView_Dense_Binary(A,viewer));
  else if (isdraw) PetscCall(MatView_SeqDense_Draw(A,viewer));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDensePlaceArray_SeqDense(Mat A,const PetscScalar *array)
{
  Mat_SeqDense *a = (Mat_SeqDense*)A->data;

  PetscFunctionBegin;
  PetscCheck(!a->vecinuse,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  PetscCheck(!a->matinuse,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  PetscCheck(!a->unplacedarray,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreArray() first");
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
  PetscCheck(!a->vecinuse,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  PetscCheck(!a->matinuse,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
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

  PetscFunctionBegin;
  PetscCheck(!a->vecinuse,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  PetscCheck(!a->matinuse,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  if (!a->user_alloc) PetscCall(PetscFree(a->v));
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

  PetscFunctionBegin;
#if defined(PETSC_USE_LOG)
  PetscLogObjectState((PetscObject)mat,"Rows %" PetscInt_FMT " Cols %" PetscInt_FMT,mat->rmap->n,mat->cmap->n);
#endif
  PetscCall(VecDestroy(&(l->qrrhs)));
  PetscCall(PetscFree(l->tau));
  PetscCall(PetscFree(l->pivots));
  PetscCall(PetscFree(l->fwork));
  PetscCall(MatDestroy(&l->ptapwork));
  if (!l->user_alloc) PetscCall(PetscFree(l->v));
  if (!l->unplaced_user_alloc) PetscCall(PetscFree(l->unplacedarray));
  PetscCheck(!l->vecinuse,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  PetscCheck(!l->matinuse,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  PetscCall(VecDestroy(&l->cvec));
  PetscCall(MatDestroy(&l->cmat));
  PetscCall(PetscFree(mat->data));

  PetscCall(PetscObjectChangeTypeName((PetscObject)mat,NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatQRFactor_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatQRFactorSymbolic_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatQRFactorNumeric_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetLDA_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseSetLDA_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetArray_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseRestoreArray_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDensePlaceArray_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseResetArray_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseReplaceArray_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetArrayRead_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseRestoreArrayRead_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetArrayWrite_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseRestoreArrayWrite_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatConvert_seqdense_seqaij_C",NULL));
#if defined(PETSC_HAVE_ELEMENTAL)
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatConvert_seqdense_elemental_C",NULL));
#endif
#if defined(PETSC_HAVE_SCALAPACK)
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatConvert_seqdense_scalapack_C",NULL));
#endif
#if defined(PETSC_HAVE_CUDA)
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatConvert_seqdense_seqdensecuda_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatProductSetFromOptions_seqdensecuda_seqdensecuda_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatProductSetFromOptions_seqdensecuda_seqdense_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatProductSetFromOptions_seqdense_seqdensecuda_C",NULL));
#endif
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatSeqDenseSetPreallocation_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatProductSetFromOptions_seqaij_seqdense_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatProductSetFromOptions_seqdense_seqdense_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatProductSetFromOptions_seqbaij_seqdense_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatProductSetFromOptions_seqsbaij_seqdense_C",NULL));

  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetColumn_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseRestoreColumn_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetColumnVec_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseRestoreColumnVec_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetColumnVecRead_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseRestoreColumnVecRead_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetColumnVecWrite_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseRestoreColumnVecWrite_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetSubMatrix_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat,"MatDenseRestoreSubMatrix_C",NULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatTranspose_SeqDense(Mat A,MatReuse reuse,Mat *matout)
{
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
  PetscInt       k,j,m = A->rmap->n, M = mat->lda, n = A->cmap->n;
  PetscScalar    *v,tmp;

  PetscFunctionBegin;
  if (reuse == MAT_INPLACE_MATRIX) {
    if (m == n) { /* in place transpose */
      PetscCall(MatDenseGetArray(A,&v));
      for (j=0; j<m; j++) {
        for (k=0; k<j; k++) {
          tmp        = v[j + k*M];
          v[j + k*M] = v[k + j*M];
          v[k + j*M] = tmp;
        }
      }
      PetscCall(MatDenseRestoreArray(A,&v));
    } else { /* reuse memory, temporary allocates new memory */
      PetscScalar *v2;
      PetscLayout tmplayout;

      PetscCall(PetscMalloc1((size_t)m*n,&v2));
      PetscCall(MatDenseGetArray(A,&v));
      for (j=0; j<n; j++) {
        for (k=0; k<m; k++) v2[j + (size_t)k*n] = v[k + (size_t)j*M];
      }
      PetscCall(PetscArraycpy(v,v2,(size_t)m*n));
      PetscCall(PetscFree(v2));
      PetscCall(MatDenseRestoreArray(A,&v));
      /* cleanup size dependent quantities */
      PetscCall(VecDestroy(&mat->cvec));
      PetscCall(MatDestroy(&mat->cmat));
      PetscCall(PetscFree(mat->pivots));
      PetscCall(PetscFree(mat->fwork));
      PetscCall(MatDestroy(&mat->ptapwork));
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
      PetscCall(MatCreate(PetscObjectComm((PetscObject)A),&tmat));
      PetscCall(MatSetSizes(tmat,A->cmap->n,A->rmap->n,A->cmap->n,A->rmap->n));
      PetscCall(MatSetType(tmat,((PetscObject)A)->type_name));
      PetscCall(MatSeqDenseSetPreallocation(tmat,NULL));
    } else tmat = *matout;

    PetscCall(MatDenseGetArrayRead(A,(const PetscScalar**)&v));
    PetscCall(MatDenseGetArray(tmat,&v2));
    tmatd = (Mat_SeqDense*)tmat->data;
    M2    = tmatd->lda;
    for (j=0; j<n; j++) {
      for (k=0; k<m; k++) v2[j + k*M2] = v[k + j*M];
    }
    PetscCall(MatDenseRestoreArray(tmat,&v2));
    PetscCall(MatDenseRestoreArrayRead(A,(const PetscScalar**)&v));
    PetscCall(MatAssemblyBegin(tmat,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(tmat,MAT_FINAL_ASSEMBLY));
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

  PetscFunctionBegin;
  if (A1->rmap->n != A2->rmap->n) {*flg = PETSC_FALSE; PetscFunctionReturn(0);}
  if (A1->cmap->n != A2->cmap->n) {*flg = PETSC_FALSE; PetscFunctionReturn(0);}
  PetscCall(MatDenseGetArrayRead(A1,&v1));
  PetscCall(MatDenseGetArrayRead(A2,&v2));
  for (i=0; i<A1->cmap->n; i++) {
    PetscCall(PetscArraycmp(v1,v2,A1->rmap->n,flg));
    if (*flg == PETSC_FALSE) PetscFunctionReturn(0);
    v1 += mat1->lda;
    v2 += mat2->lda;
  }
  PetscCall(MatDenseRestoreArrayRead(A1,&v1));
  PetscCall(MatDenseRestoreArrayRead(A2,&v2));
  *flg = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetDiagonal_SeqDense(Mat A,Vec v)
{
  Mat_SeqDense      *mat = (Mat_SeqDense*)A->data;
  PetscInt          i,n,len;
  PetscScalar       *x;
  const PetscScalar *vv;

  PetscFunctionBegin;
  PetscCall(VecGetSize(v,&n));
  PetscCall(VecGetArray(v,&x));
  len  = PetscMin(A->rmap->n,A->cmap->n);
  PetscCall(MatDenseGetArrayRead(A,&vv));
  PetscCheck(n == A->rmap->n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Nonconforming mat and vec");
  for (i=0; i<len; i++) {
    x[i] = vv[i*mat->lda + i];
  }
  PetscCall(MatDenseRestoreArrayRead(A,&vv));
  PetscCall(VecRestoreArray(v,&x));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDiagonalScale_SeqDense(Mat A,Vec ll,Vec rr)
{
  Mat_SeqDense      *mat = (Mat_SeqDense*)A->data;
  const PetscScalar *l,*r;
  PetscScalar       x,*v,*vv;
  PetscInt          i,j,m = A->rmap->n,n = A->cmap->n;

  PetscFunctionBegin;
  PetscCall(MatDenseGetArray(A,&vv));
  if (ll) {
    PetscCall(VecGetSize(ll,&m));
    PetscCall(VecGetArrayRead(ll,&l));
    PetscCheck(m == A->rmap->n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Left scaling vec wrong size");
    for (i=0; i<m; i++) {
      x = l[i];
      v = vv + i;
      for (j=0; j<n; j++) { (*v) *= x; v+= mat->lda;}
    }
    PetscCall(VecRestoreArrayRead(ll,&l));
    PetscCall(PetscLogFlops(1.0*n*m));
  }
  if (rr) {
    PetscCall(VecGetSize(rr,&n));
    PetscCall(VecGetArrayRead(rr,&r));
    PetscCheck(n == A->cmap->n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Right scaling vec wrong size");
    for (i=0; i<n; i++) {
      x = r[i];
      v = vv + i*mat->lda;
      for (j=0; j<m; j++) (*v++) *= x;
    }
    PetscCall(VecRestoreArrayRead(rr,&r));
    PetscCall(PetscLogFlops(1.0*n*m));
  }
  PetscCall(MatDenseRestoreArray(A,&vv));
  PetscFunctionReturn(0);
}

PetscErrorCode MatNorm_SeqDense(Mat A,NormType type,PetscReal *nrm)
{
  Mat_SeqDense      *mat = (Mat_SeqDense*)A->data;
  PetscScalar       *v,*vv;
  PetscReal         sum = 0.0;
  PetscInt          lda, m=A->rmap->n,i,j;

  PetscFunctionBegin;
  PetscCall(MatDenseGetArrayRead(A,(const PetscScalar**)&vv));
  PetscCall(MatDenseGetLDA(A,&lda));
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
    PetscCall(PetscLogFlops(2.0*A->cmap->n*A->rmap->n));
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
    PetscCall(PetscLogFlops(1.0*A->cmap->n*A->rmap->n));
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
    PetscCall(PetscLogFlops(1.0*A->cmap->n*A->rmap->n));
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No two norm");
  PetscCall(MatDenseRestoreArrayRead(A,(const PetscScalar**)&vv));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetOption_SeqDense(Mat A,MatOption op,PetscBool flg)
{
  Mat_SeqDense   *aij = (Mat_SeqDense*)A->data;

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
    PetscCall(PetscInfo(A,"Option %s ignored\n",MatOptions[op]));
    break;
  case MAT_SPD:
  case MAT_SYMMETRIC:
  case MAT_STRUCTURALLY_SYMMETRIC:
  case MAT_HERMITIAN:
  case MAT_SYMMETRY_ETERNAL:
    /* These options are handled directly by MatSetOption() */
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"unknown option %s",MatOptions[op]);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatZeroEntries_SeqDense(Mat A)
{
  Mat_SeqDense   *l = (Mat_SeqDense*)A->data;
  PetscInt       lda=l->lda,m=A->rmap->n,n=A->cmap->n,j;
  PetscScalar    *v;

  PetscFunctionBegin;
  PetscCall(MatDenseGetArrayWrite(A,&v));
  if (lda>m) {
    for (j=0; j<n; j++) {
      PetscCall(PetscArrayzero(v+j*lda,m));
    }
  } else {
    PetscCall(PetscArrayzero(v,PetscInt64Mult(m,n)));
  }
  PetscCall(MatDenseRestoreArrayWrite(A,&v));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatZeroRows_SeqDense(Mat A,PetscInt N,const PetscInt rows[],PetscScalar diag,Vec x,Vec b)
{
  Mat_SeqDense      *l = (Mat_SeqDense*)A->data;
  PetscInt          m  = l->lda, n = A->cmap->n, i,j;
  PetscScalar       *slot,*bb,*v;
  const PetscScalar *xx;

  PetscFunctionBegin;
  if (PetscDefined(USE_DEBUG)) {
    for (i=0; i<N; i++) {
      PetscCheck(rows[i] >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative row requested to be zeroed");
      PetscCheck(rows[i] < A->rmap->n,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row %" PetscInt_FMT " requested to be zeroed greater than or equal number of rows %" PetscInt_FMT,rows[i],A->rmap->n);
    }
  }
  if (!N) PetscFunctionReturn(0);

  /* fix right hand side if needed */
  if (x && b) {
    PetscCall(VecGetArrayRead(x,&xx));
    PetscCall(VecGetArray(b,&bb));
    for (i=0; i<N; i++) bb[rows[i]] = diag*xx[rows[i]];
    PetscCall(VecRestoreArrayRead(x,&xx));
    PetscCall(VecRestoreArray(b,&bb));
  }

  PetscCall(MatDenseGetArray(A,&v));
  for (i=0; i<N; i++) {
    slot = v + rows[i];
    for (j=0; j<n; j++) { *slot = 0.0; slot += m;}
  }
  if (diag != 0.0) {
    PetscCheck(A->rmap->n == A->cmap->n,PETSC_COMM_SELF,PETSC_ERR_SUP,"Only coded for square matrices");
    for (i=0; i<N; i++) {
      slot  = v + (m+1)*rows[i];
      *slot = diag;
    }
  }
  PetscCall(MatDenseRestoreArray(A,&v));
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
  PetscCheck(!mat->matinuse,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  *array = mat->v;
  PetscFunctionReturn(0);
}

PetscErrorCode MatDenseRestoreArray_SeqDense(Mat A,PetscScalar **array)
{
  PetscFunctionBegin;
  if (array) *array = NULL;
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

.seealso: `MatDenseGetArray()`, `MatDenseRestoreArray()`, `MatDenseGetArrayRead()`, `MatDenseRestoreArrayRead()`, `MatDenseSetLDA()`
@*/
PetscErrorCode  MatDenseGetLDA(Mat A,PetscInt *lda)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidIntPointer(lda,2);
  MatCheckPreallocated(A,1);
  PetscUseMethod(A,"MatDenseGetLDA_C",(Mat,PetscInt*),(A,lda));
  PetscFunctionReturn(0);
}

/*@
   MatDenseSetLDA - Sets the leading dimension of the array used by the dense matrix

   Not collective

   Input Parameters:
+  mat - a MATSEQDENSE or MATMPIDENSE matrix
-  lda - the leading dimension

   Level: intermediate

.seealso: `MatDenseGetArray()`, `MatDenseRestoreArray()`, `MatDenseGetArrayRead()`, `MatDenseRestoreArrayRead()`, `MatDenseGetLDA()`
@*/
PetscErrorCode  MatDenseSetLDA(Mat A,PetscInt lda)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscTryMethod(A,"MatDenseSetLDA_C",(Mat,PetscInt),(A,lda));
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

.seealso: `MatDenseRestoreArray()`, `MatDenseGetArrayRead()`, `MatDenseRestoreArrayRead()`, `MatDenseGetArrayWrite()`, `MatDenseRestoreArrayWrite()`
@*/
PetscErrorCode  MatDenseGetArray(Mat A,PetscScalar **array)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(array,2);
  PetscUseMethod(A,"MatDenseGetArray_C",(Mat,PetscScalar**),(A,array));
  PetscFunctionReturn(0);
}

/*@C
   MatDenseRestoreArray - returns access to the array where the data for a dense matrix is stored obtained by MatDenseGetArray()

   Logically Collective on Mat

   Input Parameters:
+  mat - a dense matrix
-  array - pointer to the data

   Level: intermediate

.seealso: `MatDenseGetArray()`, `MatDenseGetArrayRead()`, `MatDenseRestoreArrayRead()`, `MatDenseGetArrayWrite()`, `MatDenseRestoreArrayWrite()`
@*/
PetscErrorCode  MatDenseRestoreArray(Mat A,PetscScalar **array)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(array,2);
  PetscUseMethod(A,"MatDenseRestoreArray_C",(Mat,PetscScalar**),(A,array));
  PetscCall(PetscObjectStateIncrease((PetscObject)A));
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

.seealso: `MatDenseRestoreArrayRead()`, `MatDenseGetArray()`, `MatDenseRestoreArray()`, `MatDenseGetArrayWrite()`, `MatDenseRestoreArrayWrite()`
@*/
PetscErrorCode  MatDenseGetArrayRead(Mat A,const PetscScalar **array)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(array,2);
  PetscUseMethod(A,"MatDenseGetArrayRead_C",(Mat,const PetscScalar**),(A,array));
  PetscFunctionReturn(0);
}

/*@C
   MatDenseRestoreArrayRead - returns access to the array where the data for a dense matrix is stored obtained by MatDenseGetArrayRead()

   Not Collective

   Input Parameters:
+  mat - a dense matrix
-  array - pointer to the data

   Level: intermediate

.seealso: `MatDenseGetArrayRead()`, `MatDenseGetArray()`, `MatDenseRestoreArray()`, `MatDenseGetArrayWrite()`, `MatDenseRestoreArrayWrite()`
@*/
PetscErrorCode  MatDenseRestoreArrayRead(Mat A,const PetscScalar **array)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(array,2);
  PetscUseMethod(A,"MatDenseRestoreArrayRead_C",(Mat,const PetscScalar**),(A,array));
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

.seealso: `MatDenseRestoreArrayWrite()`, `MatDenseGetArray()`, `MatDenseRestoreArray()`, `MatDenseGetArrayRead()`, `MatDenseRestoreArrayRead()`
@*/
PetscErrorCode  MatDenseGetArrayWrite(Mat A,PetscScalar **array)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(array,2);
  PetscUseMethod(A,"MatDenseGetArrayWrite_C",(Mat,PetscScalar**),(A,array));
  PetscFunctionReturn(0);
}

/*@C
   MatDenseRestoreArrayWrite - returns access to the array where the data for a dense matrix is stored obtained by MatDenseGetArrayWrite()

   Not Collective

   Input Parameters:
+  mat - a dense matrix
-  array - pointer to the data

   Level: intermediate

.seealso: `MatDenseGetArrayWrite()`, `MatDenseGetArray()`, `MatDenseRestoreArray()`, `MatDenseGetArrayRead()`, `MatDenseRestoreArrayRead()`
@*/
PetscErrorCode  MatDenseRestoreArrayWrite(Mat A,PetscScalar **array)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(array,2);
  PetscUseMethod(A,"MatDenseRestoreArrayWrite_C",(Mat,PetscScalar**),(A,array));
  PetscCall(PetscObjectStateIncrease((PetscObject)A));
#if defined(PETSC_HAVE_CUDA)
  A->offloadmask = PETSC_OFFLOAD_CPU;
#endif
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCreateSubMatrix_SeqDense(Mat A,IS isrow,IS iscol,MatReuse scall,Mat *B)
{
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
  PetscInt       i,j,nrows,ncols,ldb;
  const PetscInt *irow,*icol;
  PetscScalar    *av,*bv,*v = mat->v;
  Mat            newmat;

  PetscFunctionBegin;
  PetscCall(ISGetIndices(isrow,&irow));
  PetscCall(ISGetIndices(iscol,&icol));
  PetscCall(ISGetLocalSize(isrow,&nrows));
  PetscCall(ISGetLocalSize(iscol,&ncols));

  /* Check submatrixcall */
  if (scall == MAT_REUSE_MATRIX) {
    PetscInt n_cols,n_rows;
    PetscCall(MatGetSize(*B,&n_rows,&n_cols));
    if (n_rows != nrows || n_cols != ncols) {
      /* resize the result matrix to match number of requested rows/columns */
      PetscCall(MatSetSizes(*B,nrows,ncols,nrows,ncols));
    }
    newmat = *B;
  } else {
    /* Create and fill new matrix */
    PetscCall(MatCreate(PetscObjectComm((PetscObject)A),&newmat));
    PetscCall(MatSetSizes(newmat,nrows,ncols,nrows,ncols));
    PetscCall(MatSetType(newmat,((PetscObject)A)->type_name));
    PetscCall(MatSeqDenseSetPreallocation(newmat,NULL));
  }

  /* Now extract the data pointers and do the copy,column at a time */
  PetscCall(MatDenseGetArray(newmat,&bv));
  PetscCall(MatDenseGetLDA(newmat,&ldb));
  for (i=0; i<ncols; i++) {
    av = v + mat->lda*icol[i];
    for (j=0; j<nrows; j++) bv[j] = av[irow[j]];
    bv += ldb;
  }
  PetscCall(MatDenseRestoreArray(newmat,&bv));

  /* Assemble the matrices so that the correct flags are set */
  PetscCall(MatAssemblyBegin(newmat,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(newmat,MAT_FINAL_ASSEMBLY));

  /* Free work space */
  PetscCall(ISRestoreIndices(isrow,&irow));
  PetscCall(ISRestoreIndices(iscol,&icol));
  *B   = newmat;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCreateSubMatrices_SeqDense(Mat A,PetscInt n,const IS irow[],const IS icol[],MatReuse scall,Mat *B[])
{
  PetscInt       i;

  PetscFunctionBegin;
  if (scall == MAT_INITIAL_MATRIX) {
    PetscCall(PetscCalloc1(n,B));
  }

  for (i=0; i<n; i++) {
    PetscCall(MatCreateSubMatrix_SeqDense(A,irow[i],icol[i],scall,&(*B)[i]));
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
  const PetscScalar *va;
  PetscScalar       *vb;
  PetscInt          lda1=a->lda,lda2=b->lda, m=A->rmap->n,n=A->cmap->n, j;

  PetscFunctionBegin;
  /* If the two matrices don't have the same copy implementation, they aren't compatible for fast copy. */
  if (A->ops->copy != B->ops->copy) {
    PetscCall(MatCopy_Basic(A,B,str));
    PetscFunctionReturn(0);
  }
  PetscCheck(m == B->rmap->n && n == B->cmap->n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"size(B) != size(A)");
  PetscCall(MatDenseGetArrayRead(A,&va));
  PetscCall(MatDenseGetArray(B,&vb));
  if (lda1>m || lda2>m) {
    for (j=0; j<n; j++) {
      PetscCall(PetscArraycpy(vb+j*lda2,va+j*lda1,m));
    }
  } else {
    PetscCall(PetscArraycpy(vb,va,A->rmap->n*A->cmap->n));
  }
  PetscCall(MatDenseRestoreArray(B,&vb));
  PetscCall(MatDenseRestoreArrayRead(A,&va));
  PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetUp_SeqDense(Mat A)
{
  PetscFunctionBegin;
  PetscCall(PetscLayoutSetUp(A->rmap));
  PetscCall(PetscLayoutSetUp(A->cmap));
  if (!A->preallocated) {
    PetscCall(MatSeqDenseSetPreallocation(A,NULL));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatConjugate_SeqDense(Mat A)
{
  Mat_SeqDense   *mat = (Mat_SeqDense *) A->data;
  PetscInt       i,j;
  PetscInt       min = PetscMin(A->rmap->n,A->cmap->n);
  PetscScalar    *aa;

  PetscFunctionBegin;
  PetscCall(MatDenseGetArray(A,&aa));
  for (j=0; j<A->cmap->n; j++) {
    for (i=0; i<A->rmap->n; i++) aa[i+j*mat->lda] = PetscConj(aa[i+j*mat->lda]);
  }
  PetscCall(MatDenseRestoreArray(A,&aa));
  if (mat->tau) for (i = 0; i < min; i++) mat->tau[i] = PetscConj(mat->tau[i]);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatRealPart_SeqDense(Mat A)
{
  Mat_SeqDense   *mat = (Mat_SeqDense *) A->data;
  PetscInt       i,j;
  PetscScalar    *aa;

  PetscFunctionBegin;
  PetscCall(MatDenseGetArray(A,&aa));
  for (j=0; j<A->cmap->n; j++) {
    for (i=0; i<A->rmap->n; i++) aa[i+j*mat->lda] = PetscRealPart(aa[i+j*mat->lda]);
  }
  PetscCall(MatDenseRestoreArray(A,&aa));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatImaginaryPart_SeqDense(Mat A)
{
  Mat_SeqDense   *mat = (Mat_SeqDense *) A->data;
  PetscInt       i,j;
  PetscScalar    *aa;

  PetscFunctionBegin;
  PetscCall(MatDenseGetArray(A,&aa));
  for (j=0; j<A->cmap->n; j++) {
    for (i=0; i<A->rmap->n; i++) aa[i+j*mat->lda] = PetscImaginaryPart(aa[i+j*mat->lda]);
  }
  PetscCall(MatDenseRestoreArray(A,&aa));
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------*/
PetscErrorCode MatMatMultSymbolic_SeqDense_SeqDense(Mat A,Mat B,PetscReal fill,Mat C)
{
  PetscInt       m=A->rmap->n,n=B->cmap->n;
  PetscBool      cisdense;

  PetscFunctionBegin;
  PetscCall(MatSetSizes(C,m,n,m,n));
  PetscCall(PetscObjectTypeCompareAny((PetscObject)C,&cisdense,MATSEQDENSE,MATSEQDENSECUDA,""));
  if (!cisdense) {
    PetscBool flg;

    PetscCall(PetscObjectTypeCompare((PetscObject)B,((PetscObject)A)->type_name,&flg));
    PetscCall(MatSetType(C,flg ? ((PetscObject)A)->type_name : MATDENSE));
  }
  PetscCall(MatSetUp(C));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatMultNumeric_SeqDense_SeqDense(Mat A,Mat B,Mat C)
{
  Mat_SeqDense       *a=(Mat_SeqDense*)A->data,*b=(Mat_SeqDense*)B->data,*c=(Mat_SeqDense*)C->data;
  PetscBLASInt       m,n,k;
  const PetscScalar *av,*bv;
  PetscScalar       *cv;
  PetscScalar       _DOne=1.0,_DZero=0.0;

  PetscFunctionBegin;
  PetscCall(PetscBLASIntCast(C->rmap->n,&m));
  PetscCall(PetscBLASIntCast(C->cmap->n,&n));
  PetscCall(PetscBLASIntCast(A->cmap->n,&k));
  if (!m || !n || !k) PetscFunctionReturn(0);
  PetscCall(MatDenseGetArrayRead(A,&av));
  PetscCall(MatDenseGetArrayRead(B,&bv));
  PetscCall(MatDenseGetArrayWrite(C,&cv));
  PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&m,&n,&k,&_DOne,av,&a->lda,bv,&b->lda,&_DZero,cv,&c->lda));
  PetscCall(PetscLogFlops(1.0*m*n*k + 1.0*m*n*(k-1)));
  PetscCall(MatDenseRestoreArrayRead(A,&av));
  PetscCall(MatDenseRestoreArrayRead(B,&bv));
  PetscCall(MatDenseRestoreArrayWrite(C,&cv));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatTransposeMultSymbolic_SeqDense_SeqDense(Mat A,Mat B,PetscReal fill,Mat C)
{
  PetscInt       m=A->rmap->n,n=B->rmap->n;
  PetscBool      cisdense;

  PetscFunctionBegin;
  PetscCall(MatSetSizes(C,m,n,m,n));
  PetscCall(PetscObjectTypeCompareAny((PetscObject)C,&cisdense,MATSEQDENSE,MATSEQDENSECUDA,""));
  if (!cisdense) {
    PetscBool flg;

    PetscCall(PetscObjectTypeCompare((PetscObject)B,((PetscObject)A)->type_name,&flg));
    PetscCall(MatSetType(C,flg ? ((PetscObject)A)->type_name : MATDENSE));
  }
  PetscCall(MatSetUp(C));
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

  PetscFunctionBegin;
  PetscCall(PetscBLASIntCast(C->rmap->n,&m));
  PetscCall(PetscBLASIntCast(C->cmap->n,&n));
  PetscCall(PetscBLASIntCast(A->cmap->n,&k));
  if (!m || !n || !k) PetscFunctionReturn(0);
  PetscCall(MatDenseGetArrayRead(A,&av));
  PetscCall(MatDenseGetArrayRead(B,&bv));
  PetscCall(MatDenseGetArrayWrite(C,&cv));
  PetscStackCallBLAS("BLASgemm",BLASgemm_("N","T",&m,&n,&k,&_DOne,av,&a->lda,bv,&b->lda,&_DZero,cv,&c->lda));
  PetscCall(MatDenseRestoreArrayRead(A,&av));
  PetscCall(MatDenseRestoreArrayRead(B,&bv));
  PetscCall(MatDenseRestoreArrayWrite(C,&cv));
  PetscCall(PetscLogFlops(1.0*m*n*k + 1.0*m*n*(k-1)));
  PetscFunctionReturn(0);
}

PetscErrorCode MatTransposeMatMultSymbolic_SeqDense_SeqDense(Mat A,Mat B,PetscReal fill,Mat C)
{
  PetscInt       m=A->cmap->n,n=B->cmap->n;
  PetscBool      cisdense;

  PetscFunctionBegin;
  PetscCall(MatSetSizes(C,m,n,m,n));
  PetscCall(PetscObjectTypeCompareAny((PetscObject)C,&cisdense,MATSEQDENSE,MATSEQDENSECUDA,""));
  if (!cisdense) {
    PetscBool flg;

    PetscCall(PetscObjectTypeCompare((PetscObject)B,((PetscObject)A)->type_name,&flg));
    PetscCall(MatSetType(C,flg ? ((PetscObject)A)->type_name : MATDENSE));
  }
  PetscCall(MatSetUp(C));
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

  PetscFunctionBegin;
  PetscCall(PetscBLASIntCast(C->rmap->n,&m));
  PetscCall(PetscBLASIntCast(C->cmap->n,&n));
  PetscCall(PetscBLASIntCast(A->rmap->n,&k));
  if (!m || !n || !k) PetscFunctionReturn(0);
  PetscCall(MatDenseGetArrayRead(A,&av));
  PetscCall(MatDenseGetArrayRead(B,&bv));
  PetscCall(MatDenseGetArrayWrite(C,&cv));
  PetscStackCallBLAS("BLASgemm",BLASgemm_("T","N",&m,&n,&k,&_DOne,av,&a->lda,bv,&b->lda,&_DZero,cv,&c->lda));
  PetscCall(MatDenseRestoreArrayRead(A,&av));
  PetscCall(MatDenseRestoreArrayRead(B,&bv));
  PetscCall(MatDenseRestoreArrayWrite(C,&cv));
  PetscCall(PetscLogFlops(1.0*m*n*k + 1.0*m*n*(k-1)));
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
  Mat_Product    *product = C->product;

  PetscFunctionBegin;
  switch (product->type) {
  case MATPRODUCT_AB:
    PetscCall(MatProductSetFromOptions_SeqDense_AB(C));
    break;
  case MATPRODUCT_AtB:
    PetscCall(MatProductSetFromOptions_SeqDense_AtB(C));
    break;
  case MATPRODUCT_ABt:
    PetscCall(MatProductSetFromOptions_SeqDense_ABt(C));
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
  PetscInt           i,j,m = A->rmap->n,n = A->cmap->n,p;
  PetscScalar        *x;
  const PetscScalar *aa;

  PetscFunctionBegin;
  PetscCheck(!A->factortype,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  PetscCall(VecGetArray(v,&x));
  PetscCall(VecGetLocalSize(v,&p));
  PetscCall(MatDenseGetArrayRead(A,&aa));
  PetscCheck(p == A->rmap->n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Nonconforming matrix and vector");
  for (i=0; i<m; i++) {
    x[i] = aa[i]; if (idx) idx[i] = 0;
    for (j=1; j<n; j++) {
      if (PetscRealPart(x[i]) < PetscRealPart(aa[i+a->lda*j])) {x[i] = aa[i + a->lda*j]; if (idx) idx[i] = j;}
    }
  }
  PetscCall(MatDenseRestoreArrayRead(A,&aa));
  PetscCall(VecRestoreArray(v,&x));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetRowMaxAbs_SeqDense(Mat A,Vec v,PetscInt idx[])
{
  Mat_SeqDense      *a = (Mat_SeqDense*)A->data;
  PetscInt          i,j,m = A->rmap->n,n = A->cmap->n,p;
  PetscScalar       *x;
  PetscReal         atmp;
  const PetscScalar *aa;

  PetscFunctionBegin;
  PetscCheck(!A->factortype,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  PetscCall(VecGetArray(v,&x));
  PetscCall(VecGetLocalSize(v,&p));
  PetscCall(MatDenseGetArrayRead(A,&aa));
  PetscCheck(p == A->rmap->n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Nonconforming matrix and vector");
  for (i=0; i<m; i++) {
    x[i] = PetscAbsScalar(aa[i]);
    for (j=1; j<n; j++) {
      atmp = PetscAbsScalar(aa[i+a->lda*j]);
      if (PetscAbsScalar(x[i]) < atmp) {x[i] = atmp; if (idx) idx[i] = j;}
    }
  }
  PetscCall(MatDenseRestoreArrayRead(A,&aa));
  PetscCall(VecRestoreArray(v,&x));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetRowMin_SeqDense(Mat A,Vec v,PetscInt idx[])
{
  Mat_SeqDense      *a = (Mat_SeqDense*)A->data;
  PetscInt          i,j,m = A->rmap->n,n = A->cmap->n,p;
  PetscScalar       *x;
  const PetscScalar *aa;

  PetscFunctionBegin;
  PetscCheck(!A->factortype,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  PetscCall(MatDenseGetArrayRead(A,&aa));
  PetscCall(VecGetArray(v,&x));
  PetscCall(VecGetLocalSize(v,&p));
  PetscCheck(p == A->rmap->n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Nonconforming matrix and vector");
  for (i=0; i<m; i++) {
    x[i] = aa[i]; if (idx) idx[i] = 0;
    for (j=1; j<n; j++) {
      if (PetscRealPart(x[i]) > PetscRealPart(aa[i+a->lda*j])) {x[i] = aa[i + a->lda*j]; if (idx) idx[i] = j;}
    }
  }
  PetscCall(VecRestoreArray(v,&x));
  PetscCall(MatDenseRestoreArrayRead(A,&aa));
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetColumnVector_SeqDense(Mat A,Vec v,PetscInt col)
{
  Mat_SeqDense      *a = (Mat_SeqDense*)A->data;
  PetscScalar       *x;
  const PetscScalar *aa;

  PetscFunctionBegin;
  PetscCheck(!A->factortype,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  PetscCall(MatDenseGetArrayRead(A,&aa));
  PetscCall(VecGetArray(v,&x));
  PetscCall(PetscArraycpy(x,aa+col*a->lda,A->rmap->n));
  PetscCall(VecRestoreArray(v,&x));
  PetscCall(MatDenseRestoreArrayRead(A,&aa));
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatGetColumnReductions_SeqDense(Mat A,PetscInt type,PetscReal *reductions)
{
  PetscInt          i,j,m,n;
  const PetscScalar *a;

  PetscFunctionBegin;
  PetscCall(MatGetSize(A,&m,&n));
  PetscCall(PetscArrayzero(reductions,n));
  PetscCall(MatDenseGetArrayRead(A,&a));
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
  PetscCall(MatDenseRestoreArrayRead(A,&a));
  if (type == NORM_2) {
    for (i=0; i<n; i++) reductions[i] = PetscSqrtReal(reductions[i]);
  } else if (type == REDUCTION_MEAN_REALPART || type == REDUCTION_MEAN_IMAGINARYPART) {
    for (i=0; i<n; i++) reductions[i] /= m;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode  MatSetRandom_SeqDense(Mat x,PetscRandom rctx)
{
  PetscScalar    *a;
  PetscInt       lda,m,n,i,j;

  PetscFunctionBegin;
  PetscCall(MatGetSize(x,&m,&n));
  PetscCall(MatDenseGetLDA(x,&lda));
  PetscCall(MatDenseGetArray(x,&a));
  for (j=0; j<n; j++) {
    for (i=0; i<m; i++) {
      PetscCall(PetscRandomGetValue(rctx,a+j*lda+i));
    }
  }
  PetscCall(MatDenseRestoreArray(x,&a));
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
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;
  PetscScalar    *v;

  PetscFunctionBegin;
  PetscCheck(!A->factortype,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");
  PetscCall(MatDenseGetArray(A,&v));
  *vals = v+col*a->lda;
  PetscCall(MatDenseRestoreArray(A,&v));
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
                                        MatShift_SeqDense,
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
                                        NULL,
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

.seealso: `MatCreate()`, `MatCreateDense()`, `MatSetValues()`
@*/
PetscErrorCode  MatCreateSeqDense(MPI_Comm comm,PetscInt m,PetscInt n,PetscScalar *data,Mat *A)
{
  PetscFunctionBegin;
  PetscCall(MatCreate(comm,A));
  PetscCall(MatSetSizes(*A,m,n,m,n));
  PetscCall(MatSetType(*A,MATSEQDENSE));
  PetscCall(MatSeqDenseSetPreallocation(*A,data));
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

.seealso: `MatCreate()`, `MatCreateDense()`, `MatSetValues()`, `MatDenseSetLDA()`

@*/
PetscErrorCode  MatSeqDenseSetPreallocation(Mat B,PetscScalar data[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(B,MAT_CLASSID,1);
  PetscTryMethod(B,"MatSeqDenseSetPreallocation_C",(Mat,PetscScalar[]),(B,data));
  PetscFunctionReturn(0);
}

PetscErrorCode  MatSeqDenseSetPreallocation_SeqDense(Mat B,PetscScalar *data)
{
  Mat_SeqDense   *b = (Mat_SeqDense*)B->data;

  PetscFunctionBegin;
  PetscCheck(!b->matinuse,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  B->preallocated = PETSC_TRUE;

  PetscCall(PetscLayoutSetUp(B->rmap));
  PetscCall(PetscLayoutSetUp(B->cmap));

  if (b->lda <= 0) b->lda = B->rmap->n;

  if (!data) { /* petsc-allocated storage */
    if (!b->user_alloc) PetscCall(PetscFree(b->v));
    PetscCall(PetscCalloc1((size_t)b->lda*B->cmap->n,&b->v));
    PetscCall(PetscLogObjectMemory((PetscObject)B,b->lda*B->cmap->n*sizeof(PetscScalar)));

    b->user_alloc = PETSC_FALSE;
  } else { /* user-allocated storage */
    if (!b->user_alloc) PetscCall(PetscFree(b->v));
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
  const PetscScalar *array;
  PetscScalar       *v_colwise;
  PetscInt          M=A->rmap->N,N=A->cmap->N,i,j,k,*rows,*cols;

  PetscFunctionBegin;
  PetscCall(PetscMalloc3(M*N,&v_colwise,M,&rows,N,&cols));
  PetscCall(MatDenseGetArrayRead(A,&array));
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
  PetscCall(MatDenseRestoreArrayRead(A,&array));

  PetscCall(MatCreate(PetscObjectComm((PetscObject)A), &mat_elemental));
  PetscCall(MatSetSizes(mat_elemental,PETSC_DECIDE,PETSC_DECIDE,M,N));
  PetscCall(MatSetType(mat_elemental,MATELEMENTAL));
  PetscCall(MatSetUp(mat_elemental));

  /* PETSc-Elemental interaface uses axpy for setting off-processor entries, only ADD_VALUES is allowed */
  PetscCall(MatSetValues(mat_elemental,M,rows,N,cols,v_colwise,ADD_VALUES));
  PetscCall(MatAssemblyBegin(mat_elemental, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mat_elemental, MAT_FINAL_ASSEMBLY));
  PetscCall(PetscFree3(v_colwise,rows,cols));

  if (reuse == MAT_INPLACE_MATRIX) {
    PetscCall(MatHeaderReplace(A,&mat_elemental));
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
  PetscCheck(b->user_alloc || !data || b->lda == lda,PETSC_COMM_SELF,PETSC_ERR_ORDER,"LDA cannot be changed after allocation of internal storage");
  PetscCheck(lda >= B->rmap->n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"LDA %" PetscInt_FMT " must be at least matrix dimension %" PetscInt_FMT,lda,B->rmap->n);
  b->lda = lda;
  PetscFunctionReturn(0);
}

PetscErrorCode MatCreateMPIMatConcatenateSeqMat_SeqDense(MPI_Comm comm,Mat inmat,PetscInt n,MatReuse scall,Mat *outmat)
{
  PetscFunctionBegin;
  PetscCall(MatCreateMPIMatConcatenateSeqMat_MPIDense(comm,inmat,n,scall,outmat));
  PetscFunctionReturn(0);
}

PetscErrorCode MatDenseGetColumnVec_SeqDense(Mat A,PetscInt col,Vec *v)
{
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;

  PetscFunctionBegin;
  PetscCheck(!a->vecinuse,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  PetscCheck(!a->matinuse,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  if (!a->cvec) {
    PetscCall(VecCreateSeqWithArray(PetscObjectComm((PetscObject)A),A->rmap->bs,A->rmap->n,NULL,&a->cvec));
    PetscCall(PetscLogObjectParent((PetscObject)A,(PetscObject)a->cvec));
  }
  a->vecinuse = col + 1;
  PetscCall(MatDenseGetArray(A,(PetscScalar**)&a->ptrinuse));
  PetscCall(VecPlaceArray(a->cvec,a->ptrinuse + (size_t)col * (size_t)a->lda));
  *v   = a->cvec;
  PetscFunctionReturn(0);
}

PetscErrorCode MatDenseRestoreColumnVec_SeqDense(Mat A,PetscInt col,Vec *v)
{
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;

  PetscFunctionBegin;
  PetscCheck(a->vecinuse,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseGetColumnVec() first");
  PetscCheck(a->cvec,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing internal column vector");
  a->vecinuse = 0;
  PetscCall(MatDenseRestoreArray(A,(PetscScalar**)&a->ptrinuse));
  PetscCall(VecResetArray(a->cvec));
  if (v) *v = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode MatDenseGetColumnVecRead_SeqDense(Mat A,PetscInt col,Vec *v)
{
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;

  PetscFunctionBegin;
  PetscCheck(!a->vecinuse,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  PetscCheck(!a->matinuse,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  if (!a->cvec) {
    PetscCall(VecCreateSeqWithArray(PetscObjectComm((PetscObject)A),A->rmap->bs,A->rmap->n,NULL,&a->cvec));
    PetscCall(PetscLogObjectParent((PetscObject)A,(PetscObject)a->cvec));
  }
  a->vecinuse = col + 1;
  PetscCall(MatDenseGetArrayRead(A,&a->ptrinuse));
  PetscCall(VecPlaceArray(a->cvec,a->ptrinuse + (size_t)col * (size_t)a->lda));
  PetscCall(VecLockReadPush(a->cvec));
  *v   = a->cvec;
  PetscFunctionReturn(0);
}

PetscErrorCode MatDenseRestoreColumnVecRead_SeqDense(Mat A,PetscInt col,Vec *v)
{
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;

  PetscFunctionBegin;
  PetscCheck(a->vecinuse,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseGetColumnVec() first");
  PetscCheck(a->cvec,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing internal column vector");
  a->vecinuse = 0;
  PetscCall(MatDenseRestoreArrayRead(A,&a->ptrinuse));
  PetscCall(VecLockReadPop(a->cvec));
  PetscCall(VecResetArray(a->cvec));
  if (v) *v = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode MatDenseGetColumnVecWrite_SeqDense(Mat A,PetscInt col,Vec *v)
{
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;

  PetscFunctionBegin;
  PetscCheck(!a->vecinuse,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  PetscCheck(!a->matinuse,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  if (!a->cvec) {
    PetscCall(VecCreateSeqWithArray(PetscObjectComm((PetscObject)A),A->rmap->bs,A->rmap->n,NULL,&a->cvec));
    PetscCall(PetscLogObjectParent((PetscObject)A,(PetscObject)a->cvec));
  }
  a->vecinuse = col + 1;
  PetscCall(MatDenseGetArrayWrite(A,(PetscScalar**)&a->ptrinuse));
  PetscCall(VecPlaceArray(a->cvec,a->ptrinuse + (size_t)col * (size_t)a->lda));
  *v   = a->cvec;
  PetscFunctionReturn(0);
}

PetscErrorCode MatDenseRestoreColumnVecWrite_SeqDense(Mat A,PetscInt col,Vec *v)
{
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;

  PetscFunctionBegin;
  PetscCheck(a->vecinuse,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseGetColumnVec() first");
  PetscCheck(a->cvec,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing internal column vector");
  a->vecinuse = 0;
  PetscCall(MatDenseRestoreArrayWrite(A,(PetscScalar**)&a->ptrinuse));
  PetscCall(VecResetArray(a->cvec));
  if (v) *v = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode MatDenseGetSubMatrix_SeqDense(Mat A,PetscInt rbegin,PetscInt rend,PetscInt cbegin,PetscInt cend,Mat *v)
{
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;

  PetscFunctionBegin;
  PetscCheck(!a->vecinuse,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreColumnVec() first");
  PetscCheck(!a->matinuse,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseRestoreSubMatrix() first");
  if (a->cmat && (cend-cbegin != a->cmat->cmap->N || rend-rbegin != a->cmat->rmap->N)) PetscCall(MatDestroy(&a->cmat));
  if (!a->cmat) {
    PetscCall(MatCreateDense(PetscObjectComm((PetscObject)A),rend-rbegin,PETSC_DECIDE,rend-rbegin,cend-cbegin,a->v+rbegin+(size_t)cbegin*a->lda,&a->cmat));
    PetscCall(PetscLogObjectParent((PetscObject)A,(PetscObject)a->cmat));
  } else {
    PetscCall(MatDensePlaceArray(a->cmat,a->v+rbegin+(size_t)cbegin*a->lda));
  }
  PetscCall(MatDenseSetLDA(a->cmat,a->lda));
  a->matinuse = cbegin + 1;
  *v = a->cmat;
#if defined(PETSC_HAVE_CUDA)
  A->offloadmask = PETSC_OFFLOAD_CPU;
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode MatDenseRestoreSubMatrix_SeqDense(Mat A,Mat *v)
{
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;

  PetscFunctionBegin;
  PetscCheck(a->matinuse,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Need to call MatDenseGetSubMatrix() first");
  PetscCheck(a->cmat,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing internal column matrix");
  PetscCheck(*v == a->cmat,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Not the matrix obtained from MatDenseGetSubMatrix()");
  a->matinuse = 0;
  PetscCall(MatDenseResetArray(a->cmat));
  *v   = NULL;
  PetscFunctionReturn(0);
}

/*MC
   MATSEQDENSE - MATSEQDENSE = "seqdense" - A matrix type to be used for sequential dense matrices.

   Options Database Keys:
. -mat_type seqdense - sets the matrix type to "seqdense" during a call to MatSetFromOptions()

  Level: beginner

.seealso: `MatCreateSeqDense()`

M*/
PetscErrorCode MatCreate_SeqDense(Mat B)
{
  Mat_SeqDense   *b;
  PetscMPIInt    size;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)B),&size));
  PetscCheck(size <= 1,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Comm must be of size 1");

  PetscCall(PetscNewLog(B,&b));
  PetscCall(PetscMemcpy(B->ops,&MatOps_Values,sizeof(struct _MatOps)));
  B->data = (void*)b;

  b->roworiented = PETSC_TRUE;

  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatQRFactor_C",MatQRFactor_SeqDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatDenseGetLDA_C",MatDenseGetLDA_SeqDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatDenseSetLDA_C",MatDenseSetLDA_SeqDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatDenseGetArray_C",MatDenseGetArray_SeqDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatDenseRestoreArray_C",MatDenseRestoreArray_SeqDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatDensePlaceArray_C",MatDensePlaceArray_SeqDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatDenseResetArray_C",MatDenseResetArray_SeqDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatDenseReplaceArray_C",MatDenseReplaceArray_SeqDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatDenseGetArrayRead_C",MatDenseGetArray_SeqDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatDenseRestoreArrayRead_C",MatDenseRestoreArray_SeqDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatDenseGetArrayWrite_C",MatDenseGetArray_SeqDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatDenseRestoreArrayWrite_C",MatDenseRestoreArray_SeqDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatConvert_seqdense_seqaij_C",MatConvert_SeqDense_SeqAIJ));
#if defined(PETSC_HAVE_ELEMENTAL)
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatConvert_seqdense_elemental_C",MatConvert_SeqDense_Elemental));
#endif
#if defined(PETSC_HAVE_SCALAPACK)
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatConvert_seqdense_scalapack_C",MatConvert_Dense_ScaLAPACK));
#endif
#if defined(PETSC_HAVE_CUDA)
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatConvert_seqdense_seqdensecuda_C",MatConvert_SeqDense_SeqDenseCUDA));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatProductSetFromOptions_seqdensecuda_seqdensecuda_C",MatProductSetFromOptions_SeqDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatProductSetFromOptions_seqdensecuda_seqdense_C",MatProductSetFromOptions_SeqDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatProductSetFromOptions_seqdense_seqdensecuda_C",MatProductSetFromOptions_SeqDense));
#endif
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatSeqDenseSetPreallocation_C",MatSeqDenseSetPreallocation_SeqDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatProductSetFromOptions_seqaij_seqdense_C",MatProductSetFromOptions_SeqAIJ_SeqDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatProductSetFromOptions_seqdense_seqdense_C",MatProductSetFromOptions_SeqDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatProductSetFromOptions_seqbaij_seqdense_C",MatProductSetFromOptions_SeqXBAIJ_SeqDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatProductSetFromOptions_seqsbaij_seqdense_C",MatProductSetFromOptions_SeqXBAIJ_SeqDense));

  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatDenseGetColumn_C",MatDenseGetColumn_SeqDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatDenseRestoreColumn_C",MatDenseRestoreColumn_SeqDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatDenseGetColumnVec_C",MatDenseGetColumnVec_SeqDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatDenseRestoreColumnVec_C",MatDenseRestoreColumnVec_SeqDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatDenseGetColumnVecRead_C",MatDenseGetColumnVecRead_SeqDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatDenseRestoreColumnVecRead_C",MatDenseRestoreColumnVecRead_SeqDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatDenseGetColumnVecWrite_C",MatDenseGetColumnVecWrite_SeqDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatDenseRestoreColumnVecWrite_C",MatDenseRestoreColumnVecWrite_SeqDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatDenseGetSubMatrix_C",MatDenseGetSubMatrix_SeqDense));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatDenseRestoreSubMatrix_C",MatDenseRestoreSubMatrix_SeqDense));
  PetscCall(PetscObjectChangeTypeName((PetscObject)B,MATSEQDENSE));
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

.seealso: `MatDenseRestoreColumn()`
@*/
PetscErrorCode MatDenseGetColumn(Mat A,PetscInt col,PetscScalar **vals)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidLogicalCollectiveInt(A,col,2);
  PetscValidPointer(vals,3);
  PetscUseMethod(A,"MatDenseGetColumn_C",(Mat,PetscInt,PetscScalar**),(A,col,vals));
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

.seealso: `MatDenseGetColumn()`
@*/
PetscErrorCode MatDenseRestoreColumn(Mat A,PetscScalar **vals)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(vals,2);
  PetscUseMethod(A,"MatDenseRestoreColumn_C",(Mat,PetscScalar**),(A,vals));
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

.seealso: `MATDENSE`, `MATDENSECUDA`, `MatDenseGetColumnVecRead()`, `MatDenseGetColumnVecWrite()`, `MatDenseRestoreColumnVec()`, `MatDenseRestoreColumnVecRead()`, `MatDenseRestoreColumnVecWrite()`
@*/
PetscErrorCode MatDenseGetColumnVec(Mat A,PetscInt col,Vec *v)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  PetscValidLogicalCollectiveInt(A,col,2);
  PetscValidPointer(v,3);
  PetscCheck(A->preallocated,PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Matrix not preallocated");
  PetscCheck(col >= 0 && col < A->cmap->N,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Invalid col %" PetscInt_FMT ", should be in [0,%" PetscInt_FMT ")",col,A->cmap->N);
  PetscUseMethod(A,"MatDenseGetColumnVec_C",(Mat,PetscInt,Vec*),(A,col,v));
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

.seealso: `MATDENSE`, `MATDENSECUDA`, `MatDenseGetColumnVec()`, `MatDenseGetColumnVecRead()`, `MatDenseGetColumnVecWrite()`, `MatDenseRestoreColumnVecRead()`, `MatDenseRestoreColumnVecWrite()`
@*/
PetscErrorCode MatDenseRestoreColumnVec(Mat A,PetscInt col,Vec *v)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  PetscValidLogicalCollectiveInt(A,col,2);
  PetscCheck(A->preallocated,PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Matrix not preallocated");
  PetscCheck(col >= 0 && col < A->cmap->N,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Invalid col %" PetscInt_FMT ", should be in [0,%" PetscInt_FMT ")",col,A->cmap->N);
  PetscUseMethod(A,"MatDenseRestoreColumnVec_C",(Mat,PetscInt,Vec*),(A,col,v));
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

.seealso: `MATDENSE`, `MATDENSECUDA`, `MatDenseGetColumnVec()`, `MatDenseGetColumnVecWrite()`, `MatDenseRestoreColumnVec()`, `MatDenseRestoreColumnVecRead()`, `MatDenseRestoreColumnVecWrite()`
@*/
PetscErrorCode MatDenseGetColumnVecRead(Mat A,PetscInt col,Vec *v)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  PetscValidLogicalCollectiveInt(A,col,2);
  PetscValidPointer(v,3);
  PetscCheck(A->preallocated,PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Matrix not preallocated");
  PetscCheck(col >= 0 && col < A->cmap->N,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Invalid col %" PetscInt_FMT ", should be in [0,%" PetscInt_FMT ")",col,A->cmap->N);
  PetscUseMethod(A,"MatDenseGetColumnVecRead_C",(Mat,PetscInt,Vec*),(A,col,v));
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

.seealso: `MATDENSE`, `MATDENSECUDA`, `MatDenseGetColumnVec()`, `MatDenseGetColumnVecRead()`, `MatDenseGetColumnVecWrite()`, `MatDenseRestoreColumnVec()`, `MatDenseRestoreColumnVecWrite()`
@*/
PetscErrorCode MatDenseRestoreColumnVecRead(Mat A,PetscInt col,Vec *v)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  PetscValidLogicalCollectiveInt(A,col,2);
  PetscCheck(A->preallocated,PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Matrix not preallocated");
  PetscCheck(col >= 0 && col < A->cmap->N,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Invalid col %" PetscInt_FMT ", should be in [0,%" PetscInt_FMT ")",col,A->cmap->N);
  PetscUseMethod(A,"MatDenseRestoreColumnVecRead_C",(Mat,PetscInt,Vec*),(A,col,v));
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

.seealso: `MATDENSE`, `MATDENSECUDA`, `MatDenseGetColumnVec()`, `MatDenseGetColumnVecRead()`, `MatDenseRestoreColumnVec()`, `MatDenseRestoreColumnVecRead()`, `MatDenseRestoreColumnVecWrite()`
@*/
PetscErrorCode MatDenseGetColumnVecWrite(Mat A,PetscInt col,Vec *v)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  PetscValidLogicalCollectiveInt(A,col,2);
  PetscValidPointer(v,3);
  PetscCheck(A->preallocated,PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Matrix not preallocated");
  PetscCheck(col >= 0 && col < A->cmap->N,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Invalid col %" PetscInt_FMT ", should be in [0,%" PetscInt_FMT ")",col,A->cmap->N);
  PetscUseMethod(A,"MatDenseGetColumnVecWrite_C",(Mat,PetscInt,Vec*),(A,col,v));
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

.seealso: `MATDENSE`, `MATDENSECUDA`, `MatDenseGetColumnVec()`, `MatDenseGetColumnVecRead()`, `MatDenseGetColumnVecWrite()`, `MatDenseRestoreColumnVec()`, `MatDenseRestoreColumnVecRead()`
@*/
PetscErrorCode MatDenseRestoreColumnVecWrite(Mat A,PetscInt col,Vec *v)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  PetscValidLogicalCollectiveInt(A,col,2);
  PetscCheck(A->preallocated,PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Matrix not preallocated");
  PetscCheck(col >= 0 && col < A->cmap->N,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Invalid col %" PetscInt_FMT ", should be in [0,%" PetscInt_FMT ")",col,A->cmap->N);
  PetscUseMethod(A,"MatDenseRestoreColumnVecWrite_C",(Mat,PetscInt,Vec*),(A,col,v));
  PetscFunctionReturn(0);
}

/*@
   MatDenseGetSubMatrix - Gives access to a block of rows and columns of a dense matrix, represented as a Mat.

   Collective

   Input Parameters:
+  mat - the Mat object
.  rbegin - the first global row index in the block (if PETSC_DECIDE, is 0)
.  rend - the global row index past the last one in the block (if PETSC_DECIDE, is M)
.  cbegin - the first global column index in the block (if PETSC_DECIDE, is 0)
-  cend - the global column index past the last one in the block (if PETSC_DECIDE, is N)

   Output Parameter:
.  v - the matrix

   Notes:
     The matrix is owned by PETSc. Users need to call MatDenseRestoreSubMatrix() when the matrix is no longer needed.
     The output matrix is not redistributed by PETSc, so depending on the values of rbegin and rend, some processes may have no local rows.

   Level: intermediate

.seealso: `MATDENSE`, `MATDENSECUDA`, `MatDenseGetColumnVec()`, `MatDenseRestoreColumnVec()`, `MatDenseRestoreSubMatrix()`
@*/
PetscErrorCode MatDenseGetSubMatrix(Mat A,PetscInt rbegin,PetscInt rend,PetscInt cbegin,PetscInt cend,Mat *v)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  PetscValidLogicalCollectiveInt(A,rbegin,2);
  PetscValidLogicalCollectiveInt(A,rend,3);
  PetscValidLogicalCollectiveInt(A,cbegin,4);
  PetscValidLogicalCollectiveInt(A,cend,5);
  PetscValidPointer(v,6);
  if (rbegin == PETSC_DECIDE) rbegin = 0;
  if (rend == PETSC_DECIDE) rend = A->rmap->N;
  if (cbegin == PETSC_DECIDE) cbegin = 0;
  if (cend == PETSC_DECIDE) cend = A->cmap->N;
  PetscCheck(A->preallocated,PetscObjectComm((PetscObject)A),PETSC_ERR_ORDER,"Matrix not preallocated");
  PetscCheck(rbegin >= 0 && rbegin <= A->rmap->N,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Invalid rbegin %" PetscInt_FMT ", should be in [0,%" PetscInt_FMT "]",rbegin,A->rmap->N);
  PetscCheck(rend >= rbegin && rend <= A->rmap->N,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Invalid rend %" PetscInt_FMT ", should be in [%" PetscInt_FMT ",%" PetscInt_FMT "]",rend,rbegin,A->rmap->N);
  PetscCheck(cbegin >= 0 && cbegin <= A->cmap->N,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Invalid cbegin %" PetscInt_FMT ", should be in [0,%" PetscInt_FMT "]",cbegin,A->cmap->N);
  PetscCheck(cend >= cbegin && cend <= A->cmap->N,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Invalid cend %" PetscInt_FMT ", should be in [%" PetscInt_FMT ",%" PetscInt_FMT "]",cend,cbegin,A->cmap->N);
  PetscUseMethod(A,"MatDenseGetSubMatrix_C",(Mat,PetscInt,PetscInt,PetscInt,PetscInt,Mat*),(A,rbegin,rend,cbegin,cend,v));
  PetscFunctionReturn(0);
}

/*@
   MatDenseRestoreSubMatrix - Returns access to a block of columns of a dense matrix obtained from MatDenseGetSubMatrix().

   Collective

   Input Parameters:
+  mat - the Mat object
-  v - the Mat object

   Level: intermediate

.seealso: `MATDENSE`, `MATDENSECUDA`, `MatDenseGetColumnVec()`, `MatDenseRestoreColumnVec()`, `MatDenseGetSubMatrix()`
@*/
PetscErrorCode MatDenseRestoreSubMatrix(Mat A,Mat *v)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  PetscValidPointer(v,2);
  PetscUseMethod(A,"MatDenseRestoreSubMatrix_C",(Mat,Mat*),(A,v));
  PetscFunctionReturn(0);
}

#include <petscblaslapack.h>
#include <petsc/private/kernels/blockinvert.h>

PetscErrorCode MatSeqDenseInvert(Mat A)
{
  Mat_SeqDense    *a = (Mat_SeqDense*) A->data;
  PetscInt        bs = A->rmap->n;
  MatScalar       *values = a->v;
  const PetscReal shift = 0.0;
  PetscBool       allowzeropivot = PetscNot(A->erroriffailure),zeropivotdetected=PETSC_FALSE;

  PetscFunctionBegin;
  /* factor and invert each block */
  switch (bs) {
  case 1:
    values[0] = (PetscScalar)1.0 / (values[0] + shift);
    break;
  case 2:
    PetscCall(PetscKernel_A_gets_inverse_A_2(values,shift,allowzeropivot,&zeropivotdetected));
    if (zeropivotdetected) A->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
    break;
  case 3:
    PetscCall(PetscKernel_A_gets_inverse_A_3(values,shift,allowzeropivot,&zeropivotdetected));
    if (zeropivotdetected) A->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
    break;
  case 4:
    PetscCall(PetscKernel_A_gets_inverse_A_4(values,shift,allowzeropivot,&zeropivotdetected));
    if (zeropivotdetected) A->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
    break;
  case 5:
  {
    PetscScalar work[25];
    PetscInt    ipvt[5];

    PetscCall(PetscKernel_A_gets_inverse_A_5(values,ipvt,work,shift,allowzeropivot,&zeropivotdetected));
    if (zeropivotdetected) A->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
  }
    break;
  case 6:
    PetscCall(PetscKernel_A_gets_inverse_A_6(values,shift,allowzeropivot,&zeropivotdetected));
    if (zeropivotdetected) A->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
    break;
  case 7:
    PetscCall(PetscKernel_A_gets_inverse_A_7(values,shift,allowzeropivot,&zeropivotdetected));
    if (zeropivotdetected) A->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
    break;
  default:
  {
    PetscInt    *v_pivots,*IJ,j;
    PetscScalar *v_work;

    PetscCall(PetscMalloc3(bs,&v_work,bs,&v_pivots,bs,&IJ));
    for (j=0; j<bs; j++) {
      IJ[j] = j;
    }
    PetscCall(PetscKernel_A_gets_inverse_A(bs,values,v_pivots,v_work,allowzeropivot,&zeropivotdetected));
    if (zeropivotdetected) A->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
    PetscCall(PetscFree3(v_work,v_pivots,IJ));
  }
  }
  PetscFunctionReturn(0);
}
