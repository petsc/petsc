#include <../src/mat/impls/nest/matnestimpl.h> /*I   "petscmat.h"   I*/
#include <../src/mat/impls/aij/seq/aij.h>
#include <petscsf.h>

static PetscErrorCode MatSetUp_NestIS_Private(Mat,PetscInt,const IS[],PetscInt,const IS[]);
static PetscErrorCode MatCreateVecs_Nest(Mat,Vec*,Vec*);
static PetscErrorCode MatReset_Nest(Mat);

PETSC_INTERN PetscErrorCode MatConvert_Nest_IS(Mat,MatType,MatReuse,Mat*);

/* private functions */
static PetscErrorCode MatNestGetSizes_Private(Mat A,PetscInt *m,PetscInt *n,PetscInt *M,PetscInt *N)
{
  Mat_Nest       *bA = (Mat_Nest*)A->data;
  PetscInt       i,j;

  PetscFunctionBegin;
  *m = *n = *M = *N = 0;
  for (i=0; i<bA->nr; i++) {  /* rows */
    PetscInt sm,sM;
    PetscCall(ISGetLocalSize(bA->isglobal.row[i],&sm));
    PetscCall(ISGetSize(bA->isglobal.row[i],&sM));
    *m  += sm;
    *M  += sM;
  }
  for (j=0; j<bA->nc; j++) {  /* cols */
    PetscInt sn,sN;
    PetscCall(ISGetLocalSize(bA->isglobal.col[j],&sn));
    PetscCall(ISGetSize(bA->isglobal.col[j],&sN));
    *n  += sn;
    *N  += sN;
  }
  PetscFunctionReturn(0);
}

/* operations */
static PetscErrorCode MatMult_Nest(Mat A,Vec x,Vec y)
{
  Mat_Nest       *bA = (Mat_Nest*)A->data;
  Vec            *bx = bA->right,*by = bA->left;
  PetscInt       i,j,nr = bA->nr,nc = bA->nc;

  PetscFunctionBegin;
  for (i=0; i<nr; i++) PetscCall(VecGetSubVector(y,bA->isglobal.row[i],&by[i]));
  for (i=0; i<nc; i++) PetscCall(VecGetSubVector(x,bA->isglobal.col[i],&bx[i]));
  for (i=0; i<nr; i++) {
    PetscCall(VecZeroEntries(by[i]));
    for (j=0; j<nc; j++) {
      if (!bA->m[i][j]) continue;
      /* y[i] <- y[i] + A[i][j] * x[j] */
      PetscCall(MatMultAdd(bA->m[i][j],bx[j],by[i],by[i]));
    }
  }
  for (i=0; i<nr; i++) PetscCall(VecRestoreSubVector(y,bA->isglobal.row[i],&by[i]));
  for (i=0; i<nc; i++) PetscCall(VecRestoreSubVector(x,bA->isglobal.col[i],&bx[i]));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultAdd_Nest(Mat A,Vec x,Vec y,Vec z)
{
  Mat_Nest       *bA = (Mat_Nest*)A->data;
  Vec            *bx = bA->right,*bz = bA->left;
  PetscInt       i,j,nr = bA->nr,nc = bA->nc;

  PetscFunctionBegin;
  for (i=0; i<nr; i++) PetscCall(VecGetSubVector(z,bA->isglobal.row[i],&bz[i]));
  for (i=0; i<nc; i++) PetscCall(VecGetSubVector(x,bA->isglobal.col[i],&bx[i]));
  for (i=0; i<nr; i++) {
    if (y != z) {
      Vec by;
      PetscCall(VecGetSubVector(y,bA->isglobal.row[i],&by));
      PetscCall(VecCopy(by,bz[i]));
      PetscCall(VecRestoreSubVector(y,bA->isglobal.row[i],&by));
    }
    for (j=0; j<nc; j++) {
      if (!bA->m[i][j]) continue;
      /* y[i] <- y[i] + A[i][j] * x[j] */
      PetscCall(MatMultAdd(bA->m[i][j],bx[j],bz[i],bz[i]));
    }
  }
  for (i=0; i<nr; i++) PetscCall(VecRestoreSubVector(z,bA->isglobal.row[i],&bz[i]));
  for (i=0; i<nc; i++) PetscCall(VecRestoreSubVector(x,bA->isglobal.col[i],&bx[i]));
  PetscFunctionReturn(0);
}

typedef struct {
  Mat          *workC;    /* array of Mat with specific containers depending on the underlying MatMatMult implementation */
  PetscScalar  *tarray;   /* buffer for storing all temporary products A[i][j] B[j] */
  PetscInt     *dm,*dn,k; /* displacements and number of submatrices */
} Nest_Dense;

PETSC_INTERN PetscErrorCode MatProductNumeric_Nest_Dense(Mat C)
{
  Mat_Nest          *bA;
  Nest_Dense        *contents;
  Mat               viewB,viewC,productB,workC;
  const PetscScalar *barray;
  PetscScalar       *carray;
  PetscInt          i,j,M,N,nr,nc,ldb,ldc;
  Mat               A,B;

  PetscFunctionBegin;
  MatCheckProduct(C,3);
  A    = C->product->A;
  B    = C->product->B;
  PetscCall(MatGetSize(B,NULL,&N));
  if (!N) {
    PetscCall(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));
    PetscFunctionReturn(0);
  }
  contents = (Nest_Dense*)C->product->data;
  PetscCheck(contents,PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Product data empty");
  bA   = (Mat_Nest*)A->data;
  nr   = bA->nr;
  nc   = bA->nc;
  PetscCall(MatDenseGetLDA(B,&ldb));
  PetscCall(MatDenseGetLDA(C,&ldc));
  PetscCall(MatZeroEntries(C));
  PetscCall(MatDenseGetArrayRead(B,&barray));
  PetscCall(MatDenseGetArray(C,&carray));
  for (i=0; i<nr; i++) {
    PetscCall(ISGetSize(bA->isglobal.row[i],&M));
    PetscCall(MatCreateDense(PetscObjectComm((PetscObject)A),contents->dm[i+1]-contents->dm[i],PETSC_DECIDE,M,N,carray+contents->dm[i],&viewC));
    PetscCall(MatDenseSetLDA(viewC,ldc));
    for (j=0; j<nc; j++) {
      if (!bA->m[i][j]) continue;
      PetscCall(ISGetSize(bA->isglobal.col[j],&M));
      PetscCall(MatCreateDense(PetscObjectComm((PetscObject)A),contents->dn[j+1]-contents->dn[j],PETSC_DECIDE,M,N,(PetscScalar*)(barray+contents->dn[j]),&viewB));
      PetscCall(MatDenseSetLDA(viewB,ldb));

      /* MatMatMultNumeric(bA->m[i][j],viewB,contents->workC[i*nc + j]); */
      workC             = contents->workC[i*nc + j];
      productB          = workC->product->B;
      workC->product->B = viewB; /* use newly created dense matrix viewB */
      PetscCall(MatProductNumeric(workC));
      PetscCall(MatDestroy(&viewB));
      workC->product->B = productB; /* resume original B */

      /* C[i] <- workC + C[i] */
      PetscCall(MatAXPY(viewC,1.0,contents->workC[i*nc + j],SAME_NONZERO_PATTERN));
    }
    PetscCall(MatDestroy(&viewC));
  }
  PetscCall(MatDenseRestoreArray(C,&carray));
  PetscCall(MatDenseRestoreArrayRead(B,&barray));

  PetscCall(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

PetscErrorCode MatNest_DenseDestroy(void *ctx)
{
  Nest_Dense     *contents = (Nest_Dense*)ctx;
  PetscInt       i;

  PetscFunctionBegin;
  PetscCall(PetscFree(contents->tarray));
  for (i=0; i<contents->k; i++) {
    PetscCall(MatDestroy(contents->workC + i));
  }
  PetscCall(PetscFree3(contents->dm,contents->dn,contents->workC));
  PetscCall(PetscFree(contents));
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatProductSymbolic_Nest_Dense(Mat C)
{
  Mat_Nest          *bA;
  Mat               viewB,workC;
  const PetscScalar *barray;
  PetscInt          i,j,M,N,m,n,nr,nc,maxm = 0,ldb;
  Nest_Dense        *contents=NULL;
  PetscBool         cisdense;
  Mat               A,B;
  PetscReal         fill;

  PetscFunctionBegin;
  MatCheckProduct(C,4);
  PetscCheck(!C->product->data,PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Product data not empty");
  A    = C->product->A;
  B    = C->product->B;
  fill = C->product->fill;
  bA   = (Mat_Nest*)A->data;
  nr   = bA->nr;
  nc   = bA->nc;
  PetscCall(MatGetLocalSize(C,&m,&n));
  PetscCall(MatGetSize(C,&M,&N));
  if (m == PETSC_DECIDE || n == PETSC_DECIDE || M == PETSC_DECIDE || N == PETSC_DECIDE) {
    PetscCall(MatGetLocalSize(B,NULL,&n));
    PetscCall(MatGetSize(B,NULL,&N));
    PetscCall(MatGetLocalSize(A,&m,NULL));
    PetscCall(MatGetSize(A,&M,NULL));
    PetscCall(MatSetSizes(C,m,n,M,N));
  }
  PetscCall(PetscObjectTypeCompareAny((PetscObject)C,&cisdense,MATSEQDENSE,MATMPIDENSE,MATSEQDENSECUDA,MATMPIDENSECUDA,""));
  if (!cisdense) {
    PetscCall(MatSetType(C,((PetscObject)B)->type_name));
  }
  PetscCall(MatSetUp(C));
  if (!N) {
    C->ops->productnumeric = MatProductNumeric_Nest_Dense;
    PetscFunctionReturn(0);
  }

  PetscCall(PetscNew(&contents));
  C->product->data = contents;
  C->product->destroy = MatNest_DenseDestroy;
  PetscCall(PetscCalloc3(nr+1,&contents->dm,nc+1,&contents->dn,nr*nc,&contents->workC));
  contents->k = nr*nc;
  for (i=0; i<nr; i++) {
    PetscCall(ISGetLocalSize(bA->isglobal.row[i],contents->dm + i+1));
    maxm = PetscMax(maxm,contents->dm[i+1]);
    contents->dm[i+1] += contents->dm[i];
  }
  for (i=0; i<nc; i++) {
    PetscCall(ISGetLocalSize(bA->isglobal.col[i],contents->dn + i+1));
    contents->dn[i+1] += contents->dn[i];
  }
  PetscCall(PetscMalloc1(maxm*N,&contents->tarray));
  PetscCall(MatDenseGetLDA(B,&ldb));
  PetscCall(MatGetSize(B,NULL,&N));
  PetscCall(MatDenseGetArrayRead(B,&barray));
  /* loops are permuted compared to MatMatMultNumeric so that viewB is created only once per column of A */
  for (j=0; j<nc; j++) {
    PetscCall(ISGetSize(bA->isglobal.col[j],&M));
    PetscCall(MatCreateDense(PetscObjectComm((PetscObject)A),contents->dn[j+1]-contents->dn[j],PETSC_DECIDE,M,N,(PetscScalar*)(barray+contents->dn[j]),&viewB));
    PetscCall(MatDenseSetLDA(viewB,ldb));
    for (i=0; i<nr; i++) {
      if (!bA->m[i][j]) continue;
      /* MatMatMultSymbolic may attach a specific container (depending on MatType of bA->m[i][j]) to workC[i][j] */

      PetscCall(MatProductCreate(bA->m[i][j],viewB,NULL,&contents->workC[i*nc + j]));
      workC = contents->workC[i*nc + j];
      PetscCall(MatProductSetType(workC,MATPRODUCT_AB));
      PetscCall(MatProductSetAlgorithm(workC,"default"));
      PetscCall(MatProductSetFill(workC,fill));
      PetscCall(MatProductSetFromOptions(workC));
      PetscCall(MatProductSymbolic(workC));

      /* since tarray will be shared by all Mat */
      PetscCall(MatSeqDenseSetPreallocation(workC,contents->tarray));
      PetscCall(MatMPIDenseSetPreallocation(workC,contents->tarray));
    }
    PetscCall(MatDestroy(&viewB));
  }
  PetscCall(MatDenseRestoreArrayRead(B,&barray));

  C->ops->productnumeric = MatProductNumeric_Nest_Dense;
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------- */
static PetscErrorCode MatProductSetFromOptions_Nest_Dense_AB(Mat C)
{
  PetscFunctionBegin;
  C->ops->productsymbolic = MatProductSymbolic_Nest_Dense;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatProductSetFromOptions_Nest_Dense(Mat C)
{
  Mat_Product    *product = C->product;

  PetscFunctionBegin;
  if (product->type == MATPRODUCT_AB) {
    PetscCall(MatProductSetFromOptions_Nest_Dense_AB(C));
  }
  PetscFunctionReturn(0);
}
/* --------------------------------------------------------- */

static PetscErrorCode MatMultTranspose_Nest(Mat A,Vec x,Vec y)
{
  Mat_Nest       *bA = (Mat_Nest*)A->data;
  Vec            *bx = bA->left,*by = bA->right;
  PetscInt       i,j,nr = bA->nr,nc = bA->nc;

  PetscFunctionBegin;
  for (i=0; i<nr; i++) PetscCall(VecGetSubVector(x,bA->isglobal.row[i],&bx[i]));
  for (i=0; i<nc; i++) PetscCall(VecGetSubVector(y,bA->isglobal.col[i],&by[i]));
  for (j=0; j<nc; j++) {
    PetscCall(VecZeroEntries(by[j]));
    for (i=0; i<nr; i++) {
      if (!bA->m[i][j]) continue;
      /* y[j] <- y[j] + (A[i][j])^T * x[i] */
      PetscCall(MatMultTransposeAdd(bA->m[i][j],bx[i],by[j],by[j]));
    }
  }
  for (i=0; i<nr; i++) PetscCall(VecRestoreSubVector(x,bA->isglobal.row[i],&bx[i]));
  for (i=0; i<nc; i++) PetscCall(VecRestoreSubVector(y,bA->isglobal.col[i],&by[i]));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTransposeAdd_Nest(Mat A,Vec x,Vec y,Vec z)
{
  Mat_Nest       *bA = (Mat_Nest*)A->data;
  Vec            *bx = bA->left,*bz = bA->right;
  PetscInt       i,j,nr = bA->nr,nc = bA->nc;

  PetscFunctionBegin;
  for (i=0; i<nr; i++) PetscCall(VecGetSubVector(x,bA->isglobal.row[i],&bx[i]));
  for (i=0; i<nc; i++) PetscCall(VecGetSubVector(z,bA->isglobal.col[i],&bz[i]));
  for (j=0; j<nc; j++) {
    if (y != z) {
      Vec by;
      PetscCall(VecGetSubVector(y,bA->isglobal.col[j],&by));
      PetscCall(VecCopy(by,bz[j]));
      PetscCall(VecRestoreSubVector(y,bA->isglobal.col[j],&by));
    }
    for (i=0; i<nr; i++) {
      if (!bA->m[i][j]) continue;
      /* z[j] <- y[j] + (A[i][j])^T * x[i] */
      PetscCall(MatMultTransposeAdd(bA->m[i][j],bx[i],bz[j],bz[j]));
    }
  }
  for (i=0; i<nr; i++) PetscCall(VecRestoreSubVector(x,bA->isglobal.row[i],&bx[i]));
  for (i=0; i<nc; i++) PetscCall(VecRestoreSubVector(z,bA->isglobal.col[i],&bz[i]));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatTranspose_Nest(Mat A,MatReuse reuse,Mat *B)
{
  Mat_Nest       *bA = (Mat_Nest*)A->data, *bC;
  Mat            C;
  PetscInt       i,j,nr = bA->nr,nc = bA->nc;

  PetscFunctionBegin;
  PetscCheck(reuse != MAT_INPLACE_MATRIX || nr == nc,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_SIZ,"Square nested matrix only for in-place");

  if (reuse == MAT_INITIAL_MATRIX || reuse == MAT_INPLACE_MATRIX) {
    Mat *subs;
    IS  *is_row,*is_col;

    PetscCall(PetscCalloc1(nr * nc,&subs));
    PetscCall(PetscMalloc2(nr,&is_row,nc,&is_col));
    PetscCall(MatNestGetISs(A,is_row,is_col));
    if (reuse == MAT_INPLACE_MATRIX) {
      for (i=0; i<nr; i++) {
        for (j=0; j<nc; j++) {
          subs[i + nr * j] = bA->m[i][j];
        }
      }
    }

    PetscCall(MatCreateNest(PetscObjectComm((PetscObject)A),nc,is_col,nr,is_row,subs,&C));
    PetscCall(PetscFree(subs));
    PetscCall(PetscFree2(is_row,is_col));
  } else {
    C = *B;
  }

  bC = (Mat_Nest*)C->data;
  for (i=0; i<nr; i++) {
    for (j=0; j<nc; j++) {
      if (bA->m[i][j]) {
        PetscCall(MatTranspose(bA->m[i][j], reuse, &(bC->m[j][i])));
      } else {
        bC->m[j][i] = NULL;
      }
    }
  }

  if (reuse == MAT_INITIAL_MATRIX || reuse == MAT_REUSE_MATRIX) {
    *B = C;
  } else {
    PetscCall(MatHeaderMerge(A, &C));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatNestDestroyISList(PetscInt n,IS **list)
{
  IS             *lst = *list;
  PetscInt       i;

  PetscFunctionBegin;
  if (!lst) PetscFunctionReturn(0);
  for (i=0; i<n; i++) if (lst[i]) PetscCall(ISDestroy(&lst[i]));
  PetscCall(PetscFree(lst));
  *list = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatReset_Nest(Mat A)
{
  Mat_Nest       *vs = (Mat_Nest*)A->data;
  PetscInt       i,j;

  PetscFunctionBegin;
  /* release the matrices and the place holders */
  PetscCall(MatNestDestroyISList(vs->nr,&vs->isglobal.row));
  PetscCall(MatNestDestroyISList(vs->nc,&vs->isglobal.col));
  PetscCall(MatNestDestroyISList(vs->nr,&vs->islocal.row));
  PetscCall(MatNestDestroyISList(vs->nc,&vs->islocal.col));

  PetscCall(PetscFree(vs->row_len));
  PetscCall(PetscFree(vs->col_len));
  PetscCall(PetscFree(vs->nnzstate));

  PetscCall(PetscFree2(vs->left,vs->right));

  /* release the matrices and the place holders */
  if (vs->m) {
    for (i=0; i<vs->nr; i++) {
      for (j=0; j<vs->nc; j++) {
        PetscCall(MatDestroy(&vs->m[i][j]));
      }
      PetscCall(PetscFree(vs->m[i]));
    }
    PetscCall(PetscFree(vs->m));
  }

  /* restore defaults */
  vs->nr = 0;
  vs->nc = 0;
  vs->splitassembly = PETSC_FALSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_Nest(Mat A)
{
  PetscFunctionBegin;
  PetscCall(MatReset_Nest(A));
  PetscCall(PetscFree(A->data));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatNestGetSubMat_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatNestSetSubMat_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatNestGetSubMats_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatNestGetSize_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatNestGetISs_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatNestGetLocalISs_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatNestSetVecType_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatNestSetSubMats_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatConvert_nest_mpiaij_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatConvert_nest_seqaij_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatConvert_nest_aij_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatConvert_nest_is_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatConvert_nest_mpidense_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatConvert_nest_seqdense_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_nest_seqdense_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_nest_mpidense_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_nest_dense_C",NULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMissingDiagonal_Nest(Mat mat,PetscBool *missing,PetscInt *dd)
{
  Mat_Nest       *vs = (Mat_Nest*)mat->data;
  PetscInt       i;

  PetscFunctionBegin;
  if (dd) *dd = 0;
  if (!vs->nr) {
    *missing = PETSC_TRUE;
    PetscFunctionReturn(0);
  }
  *missing = PETSC_FALSE;
  for (i = 0; i < vs->nr && !(*missing); i++) {
    *missing = PETSC_TRUE;
    if (vs->m[i][i]) {
      PetscCall(MatMissingDiagonal(vs->m[i][i],missing,NULL));
      PetscCheck(!*missing || !dd,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"First missing entry not yet implemented");
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatAssemblyBegin_Nest(Mat A,MatAssemblyType type)
{
  Mat_Nest       *vs = (Mat_Nest*)A->data;
  PetscInt       i,j;
  PetscBool      nnzstate = PETSC_FALSE;

  PetscFunctionBegin;
  for (i=0; i<vs->nr; i++) {
    for (j=0; j<vs->nc; j++) {
      PetscObjectState subnnzstate = 0;
      if (vs->m[i][j]) {
        PetscCall(MatAssemblyBegin(vs->m[i][j],type));
        if (!vs->splitassembly) {
          /* Note: split assembly will fail if the same block appears more than once (even indirectly through a nested
           * sub-block). This could be fixed by adding a flag to Mat so that there was a way to check if a Mat was
           * already performing an assembly, but the result would by more complicated and appears to offer less
           * potential for diagnostics and correctness checking. Split assembly should be fixed once there is an
           * interface for libraries to make asynchronous progress in "user-defined non-blocking collectives".
           */
          PetscCall(MatAssemblyEnd(vs->m[i][j],type));
          PetscCall(MatGetNonzeroState(vs->m[i][j],&subnnzstate));
        }
      }
      nnzstate = (PetscBool)(nnzstate || vs->nnzstate[i*vs->nc+j] != subnnzstate);
      vs->nnzstate[i*vs->nc+j] = subnnzstate;
    }
  }
  if (nnzstate) A->nonzerostate++;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatAssemblyEnd_Nest(Mat A, MatAssemblyType type)
{
  Mat_Nest       *vs = (Mat_Nest*)A->data;
  PetscInt       i,j;

  PetscFunctionBegin;
  for (i=0; i<vs->nr; i++) {
    for (j=0; j<vs->nc; j++) {
      if (vs->m[i][j]) {
        if (vs->splitassembly) {
          PetscCall(MatAssemblyEnd(vs->m[i][j],type));
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatNestFindNonzeroSubMatRow(Mat A,PetscInt row,Mat *B)
{
  Mat_Nest       *vs = (Mat_Nest*)A->data;
  PetscInt       j;
  Mat            sub;

  PetscFunctionBegin;
  sub = (row < vs->nc) ? vs->m[row][row] : (Mat)NULL; /* Prefer to find on the diagonal */
  for (j=0; !sub && j<vs->nc; j++) sub = vs->m[row][j];
  if (sub) PetscCall(MatSetUp(sub));       /* Ensure that the sizes are available */
  *B = sub;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatNestFindNonzeroSubMatCol(Mat A,PetscInt col,Mat *B)
{
  Mat_Nest       *vs = (Mat_Nest*)A->data;
  PetscInt       i;
  Mat            sub;

  PetscFunctionBegin;
  sub = (col < vs->nr) ? vs->m[col][col] : (Mat)NULL; /* Prefer to find on the diagonal */
  for (i=0; !sub && i<vs->nr; i++) sub = vs->m[i][col];
  if (sub) PetscCall(MatSetUp(sub));       /* Ensure that the sizes are available */
  *B = sub;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatNestFindISRange(Mat A,PetscInt n,const IS list[],IS is,PetscInt *begin,PetscInt *end)
{
  PetscInt       i,j,size,m;
  PetscBool      flg;
  IS             out,concatenate[2];

  PetscFunctionBegin;
  PetscValidPointer(list,3);
  PetscValidHeaderSpecific(is,IS_CLASSID,4);
  if (begin) {
    PetscValidIntPointer(begin,5);
    *begin = -1;
  }
  if (end) {
    PetscValidIntPointer(end,6);
    *end = -1;
  }
  for (i=0; i<n; i++) {
    if (!list[i]) continue;
    PetscCall(ISEqualUnsorted(list[i],is,&flg));
    if (flg) {
      if (begin) *begin = i;
      if (end) *end = i+1;
      PetscFunctionReturn(0);
    }
  }
  PetscCall(ISGetSize(is,&size));
  for (i=0; i<n-1; i++) {
    if (!list[i]) continue;
    m = 0;
    PetscCall(ISConcatenate(PetscObjectComm((PetscObject)A),2,list+i,&out));
    PetscCall(ISGetSize(out,&m));
    for (j=i+2; j<n && m<size; j++) {
      if (list[j]) {
        concatenate[0] = out;
        concatenate[1] = list[j];
        PetscCall(ISConcatenate(PetscObjectComm((PetscObject)A),2,concatenate,&out));
        PetscCall(ISDestroy(concatenate));
        PetscCall(ISGetSize(out,&m));
      }
    }
    if (m == size) {
      PetscCall(ISEqualUnsorted(out,is,&flg));
      if (flg) {
        if (begin) *begin = i;
        if (end) *end = j;
        PetscCall(ISDestroy(&out));
        PetscFunctionReturn(0);
      }
    }
    PetscCall(ISDestroy(&out));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatNestFillEmptyMat_Private(Mat A,PetscInt i,PetscInt j,Mat *B)
{
  Mat_Nest       *vs = (Mat_Nest*)A->data;
  PetscInt       lr,lc;

  PetscFunctionBegin;
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A),B));
  PetscCall(ISGetLocalSize(vs->isglobal.row[i],&lr));
  PetscCall(ISGetLocalSize(vs->isglobal.col[j],&lc));
  PetscCall(MatSetSizes(*B,lr,lc,PETSC_DECIDE,PETSC_DECIDE));
  PetscCall(MatSetType(*B,MATAIJ));
  PetscCall(MatSeqAIJSetPreallocation(*B,0,NULL));
  PetscCall(MatMPIAIJSetPreallocation(*B,0,NULL,0,NULL));
  PetscCall(MatSetUp(*B));
  PetscCall(MatSetOption(*B,MAT_NO_OFF_PROC_ENTRIES,PETSC_TRUE));
  PetscCall(MatAssemblyBegin(*B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*B,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatNestGetBlock_Private(Mat A,PetscInt rbegin,PetscInt rend,PetscInt cbegin,PetscInt cend,Mat *B)
{
  Mat_Nest       *vs = (Mat_Nest*)A->data;
  Mat            *a;
  PetscInt       i,j,k,l,nr=rend-rbegin,nc=cend-cbegin;
  char           keyname[256];
  PetscBool      *b;
  PetscBool      flg;

  PetscFunctionBegin;
  *B   = NULL;
  PetscCall(PetscSNPrintf(keyname,sizeof(keyname),"NestBlock_%" PetscInt_FMT "-%" PetscInt_FMT "x%" PetscInt_FMT "-%" PetscInt_FMT,rbegin,rend,cbegin,cend));
  PetscCall(PetscObjectQuery((PetscObject)A,keyname,(PetscObject*)B));
  if (*B) PetscFunctionReturn(0);

  PetscCall(PetscMalloc2(nr*nc,&a,nr*nc,&b));
  for (i=0; i<nr; i++) {
    for (j=0; j<nc; j++) {
      a[i*nc + j] = vs->m[rbegin+i][cbegin+j];
      b[i*nc + j] = PETSC_FALSE;
    }
  }
  if (nc!=vs->nc&&nr!=vs->nr) {
    for (i=0; i<nr; i++) {
      for (j=0; j<nc; j++) {
        flg = PETSC_FALSE;
        for (k=0; (k<nr&&!flg); k++) {
          if (a[j + k*nc]) flg = PETSC_TRUE;
        }
        if (flg) {
          flg = PETSC_FALSE;
          for (l=0; (l<nc&&!flg); l++) {
            if (a[i*nc + l]) flg = PETSC_TRUE;
          }
        }
        if (!flg) {
          b[i*nc + j] = PETSC_TRUE;
          PetscCall(MatNestFillEmptyMat_Private(A,rbegin+i,cbegin+j,a + i*nc + j));
        }
      }
    }
  }
  PetscCall(MatCreateNest(PetscObjectComm((PetscObject)A),nr,nr!=vs->nr?NULL:vs->isglobal.row,nc,nc!=vs->nc?NULL:vs->isglobal.col,a,B));
  for (i=0; i<nr; i++) {
    for (j=0; j<nc; j++) {
      if (b[i*nc + j]) {
        PetscCall(MatDestroy(a + i*nc + j));
      }
    }
  }
  PetscCall(PetscFree2(a,b));
  (*B)->assembled = A->assembled;
  PetscCall(PetscObjectCompose((PetscObject)A,keyname,(PetscObject)*B));
  PetscCall(PetscObjectDereference((PetscObject)*B)); /* Leave the only remaining reference in the composition */
  PetscFunctionReturn(0);
}

static PetscErrorCode MatNestFindSubMat(Mat A,struct MatNestISPair *is,IS isrow,IS iscol,Mat *B)
{
  Mat_Nest       *vs = (Mat_Nest*)A->data;
  PetscInt       rbegin,rend,cbegin,cend;

  PetscFunctionBegin;
  PetscCall(MatNestFindISRange(A,vs->nr,is->row,isrow,&rbegin,&rend));
  PetscCall(MatNestFindISRange(A,vs->nc,is->col,iscol,&cbegin,&cend));
  if (rend == rbegin + 1 && cend == cbegin + 1) {
    if (!vs->m[rbegin][cbegin]) {
      PetscCall(MatNestFillEmptyMat_Private(A,rbegin,cbegin,vs->m[rbegin] + cbegin));
    }
    *B = vs->m[rbegin][cbegin];
  } else if (rbegin != -1 && cbegin != -1) {
    PetscCall(MatNestGetBlock_Private(A,rbegin,rend,cbegin,cend,B));
  } else SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_INCOMP,"Could not find index set");
  PetscFunctionReturn(0);
}

/*
   TODO: This does not actually returns a submatrix we can modify
*/
static PetscErrorCode MatCreateSubMatrix_Nest(Mat A,IS isrow,IS iscol,MatReuse reuse,Mat *B)
{
  Mat_Nest       *vs = (Mat_Nest*)A->data;
  Mat            sub;

  PetscFunctionBegin;
  PetscCall(MatNestFindSubMat(A,&vs->isglobal,isrow,iscol,&sub));
  switch (reuse) {
  case MAT_INITIAL_MATRIX:
    if (sub) PetscCall(PetscObjectReference((PetscObject)sub));
    *B = sub;
    break;
  case MAT_REUSE_MATRIX:
    PetscCheck(sub == *B,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONGSTATE,"Submatrix was not used before in this call");
    break;
  case MAT_IGNORE_MATRIX:       /* Nothing to do */
    break;
  case MAT_INPLACE_MATRIX:       /* Nothing to do */
    SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"MAT_INPLACE_MATRIX is not supported yet");
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetLocalSubMatrix_Nest(Mat A,IS isrow,IS iscol,Mat *B)
{
  Mat_Nest       *vs = (Mat_Nest*)A->data;
  Mat            sub;

  PetscFunctionBegin;
  PetscCall(MatNestFindSubMat(A,&vs->islocal,isrow,iscol,&sub));
  /* We allow the submatrix to be NULL, perhaps it would be better for the user to return an empty matrix instead */
  if (sub) PetscCall(PetscObjectReference((PetscObject)sub));
  *B = sub;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatRestoreLocalSubMatrix_Nest(Mat A,IS isrow,IS iscol,Mat *B)
{
  Mat_Nest       *vs = (Mat_Nest*)A->data;
  Mat            sub;

  PetscFunctionBegin;
  PetscCall(MatNestFindSubMat(A,&vs->islocal,isrow,iscol,&sub));
  PetscCheck(*B == sub,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONGSTATE,"Local submatrix has not been gotten");
  if (sub) {
    PetscCheck(((PetscObject)sub)->refct > 1,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONGSTATE,"Local submatrix has had reference count decremented too many times");
    PetscCall(MatDestroy(B));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetDiagonal_Nest(Mat A,Vec v)
{
  Mat_Nest       *bA = (Mat_Nest*)A->data;
  PetscInt       i;

  PetscFunctionBegin;
  for (i=0; i<bA->nr; i++) {
    Vec bv;
    PetscCall(VecGetSubVector(v,bA->isglobal.row[i],&bv));
    if (bA->m[i][i]) {
      PetscCall(MatGetDiagonal(bA->m[i][i],bv));
    } else {
      PetscCall(VecSet(bv,0.0));
    }
    PetscCall(VecRestoreSubVector(v,bA->isglobal.row[i],&bv));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDiagonalScale_Nest(Mat A,Vec l,Vec r)
{
  Mat_Nest       *bA = (Mat_Nest*)A->data;
  Vec            bl,*br;
  PetscInt       i,j;

  PetscFunctionBegin;
  PetscCall(PetscCalloc1(bA->nc,&br));
  if (r) {
    for (j=0; j<bA->nc; j++) PetscCall(VecGetSubVector(r,bA->isglobal.col[j],&br[j]));
  }
  bl = NULL;
  for (i=0; i<bA->nr; i++) {
    if (l) {
      PetscCall(VecGetSubVector(l,bA->isglobal.row[i],&bl));
    }
    for (j=0; j<bA->nc; j++) {
      if (bA->m[i][j]) {
        PetscCall(MatDiagonalScale(bA->m[i][j],bl,br[j]));
      }
    }
    if (l) {
      PetscCall(VecRestoreSubVector(l,bA->isglobal.row[i],&bl));
    }
  }
  if (r) {
    for (j=0; j<bA->nc; j++) PetscCall(VecRestoreSubVector(r,bA->isglobal.col[j],&br[j]));
  }
  PetscCall(PetscFree(br));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatScale_Nest(Mat A,PetscScalar a)
{
  Mat_Nest       *bA = (Mat_Nest*)A->data;
  PetscInt       i,j;

  PetscFunctionBegin;
  for (i=0; i<bA->nr; i++) {
    for (j=0; j<bA->nc; j++) {
      if (bA->m[i][j]) {
        PetscCall(MatScale(bA->m[i][j],a));
      }
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatShift_Nest(Mat A,PetscScalar a)
{
  Mat_Nest       *bA = (Mat_Nest*)A->data;
  PetscInt       i;
  PetscBool      nnzstate = PETSC_FALSE;

  PetscFunctionBegin;
  for (i=0; i<bA->nr; i++) {
    PetscObjectState subnnzstate = 0;
    PetscCheck(bA->m[i][i],PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"No support for shifting an empty diagonal block, insert a matrix in block (%" PetscInt_FMT ",%" PetscInt_FMT ")",i,i);
    PetscCall(MatShift(bA->m[i][i],a));
    PetscCall(MatGetNonzeroState(bA->m[i][i],&subnnzstate));
    nnzstate = (PetscBool)(nnzstate || bA->nnzstate[i*bA->nc+i] != subnnzstate);
    bA->nnzstate[i*bA->nc+i] = subnnzstate;
  }
  if (nnzstate) A->nonzerostate++;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDiagonalSet_Nest(Mat A,Vec D,InsertMode is)
{
  Mat_Nest       *bA = (Mat_Nest*)A->data;
  PetscInt       i;
  PetscBool      nnzstate = PETSC_FALSE;

  PetscFunctionBegin;
  for (i=0; i<bA->nr; i++) {
    PetscObjectState subnnzstate = 0;
    Vec              bv;
    PetscCall(VecGetSubVector(D,bA->isglobal.row[i],&bv));
    if (bA->m[i][i]) {
      PetscCall(MatDiagonalSet(bA->m[i][i],bv,is));
      PetscCall(MatGetNonzeroState(bA->m[i][i],&subnnzstate));
    }
    PetscCall(VecRestoreSubVector(D,bA->isglobal.row[i],&bv));
    nnzstate = (PetscBool)(nnzstate || bA->nnzstate[i*bA->nc+i] != subnnzstate);
    bA->nnzstate[i*bA->nc+i] = subnnzstate;
  }
  if (nnzstate) A->nonzerostate++;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetRandom_Nest(Mat A,PetscRandom rctx)
{
  Mat_Nest       *bA = (Mat_Nest*)A->data;
  PetscInt       i,j;

  PetscFunctionBegin;
  for (i=0; i<bA->nr; i++) {
    for (j=0; j<bA->nc; j++) {
      if (bA->m[i][j]) {
        PetscCall(MatSetRandom(bA->m[i][j],rctx));
      }
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCreateVecs_Nest(Mat A,Vec *right,Vec *left)
{
  Mat_Nest       *bA = (Mat_Nest*)A->data;
  Vec            *L,*R;
  MPI_Comm       comm;
  PetscInt       i,j;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)A,&comm));
  if (right) {
    /* allocate R */
    PetscCall(PetscMalloc1(bA->nc, &R));
    /* Create the right vectors */
    for (j=0; j<bA->nc; j++) {
      for (i=0; i<bA->nr; i++) {
        if (bA->m[i][j]) {
          PetscCall(MatCreateVecs(bA->m[i][j],&R[j],NULL));
          break;
        }
      }
      PetscCheck(i!=bA->nr,PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_WRONG, "Mat(Nest) contains a null column.");
    }
    PetscCall(VecCreateNest(comm,bA->nc,bA->isglobal.col,R,right));
    /* hand back control to the nest vector */
    for (j=0; j<bA->nc; j++) {
      PetscCall(VecDestroy(&R[j]));
    }
    PetscCall(PetscFree(R));
  }

  if (left) {
    /* allocate L */
    PetscCall(PetscMalloc1(bA->nr, &L));
    /* Create the left vectors */
    for (i=0; i<bA->nr; i++) {
      for (j=0; j<bA->nc; j++) {
        if (bA->m[i][j]) {
          PetscCall(MatCreateVecs(bA->m[i][j],NULL,&L[i]));
          break;
        }
      }
      PetscCheck(j!=bA->nc,PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_WRONG, "Mat(Nest) contains a null row.");
    }

    PetscCall(VecCreateNest(comm,bA->nr,bA->isglobal.row,L,left));
    for (i=0; i<bA->nr; i++) {
      PetscCall(VecDestroy(&L[i]));
    }

    PetscCall(PetscFree(L));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatView_Nest(Mat A,PetscViewer viewer)
{
  Mat_Nest       *bA = (Mat_Nest*)A->data;
  PetscBool      isascii,viewSub = PETSC_FALSE;
  PetscInt       i,j;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  if (isascii) {

    PetscCall(PetscOptionsGetBool(((PetscObject)A)->options,((PetscObject)A)->prefix,"-mat_view_nest_sub",&viewSub,NULL));
    PetscCall(PetscViewerASCIIPrintf(viewer,"Matrix object: \n"));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(PetscViewerASCIIPrintf(viewer, "type=nest, rows=%" PetscInt_FMT ", cols=%" PetscInt_FMT " \n",bA->nr,bA->nc));

    PetscCall(PetscViewerASCIIPrintf(viewer,"MatNest structure: \n"));
    for (i=0; i<bA->nr; i++) {
      for (j=0; j<bA->nc; j++) {
        MatType   type;
        char      name[256] = "",prefix[256] = "";
        PetscInt  NR,NC;
        PetscBool isNest = PETSC_FALSE;

        if (!bA->m[i][j]) {
          PetscCall(PetscViewerASCIIPrintf(viewer, "(%" PetscInt_FMT ",%" PetscInt_FMT ") : NULL \n",i,j));
          continue;
        }
        PetscCall(MatGetSize(bA->m[i][j],&NR,&NC));
        PetscCall(MatGetType(bA->m[i][j], &type));
        if (((PetscObject)bA->m[i][j])->name) PetscCall(PetscSNPrintf(name,sizeof(name),"name=\"%s\", ",((PetscObject)bA->m[i][j])->name));
        if (((PetscObject)bA->m[i][j])->prefix) PetscCall(PetscSNPrintf(prefix,sizeof(prefix),"prefix=\"%s\", ",((PetscObject)bA->m[i][j])->prefix));
        PetscCall(PetscObjectTypeCompare((PetscObject)bA->m[i][j],MATNEST,&isNest));

        PetscCall(PetscViewerASCIIPrintf(viewer,"(%" PetscInt_FMT ",%" PetscInt_FMT ") : %s%stype=%s, rows=%" PetscInt_FMT ", cols=%" PetscInt_FMT " \n",i,j,name,prefix,type,NR,NC));

        if (isNest || viewSub) {
          PetscCall(PetscViewerASCIIPushTab(viewer));  /* push1 */
          PetscCall(MatView(bA->m[i][j],viewer));
          PetscCall(PetscViewerASCIIPopTab(viewer));    /* pop1 */
        }
      }
    }
    PetscCall(PetscViewerASCIIPopTab(viewer));    /* pop0 */
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatZeroEntries_Nest(Mat A)
{
  Mat_Nest       *bA = (Mat_Nest*)A->data;
  PetscInt       i,j;

  PetscFunctionBegin;
  for (i=0; i<bA->nr; i++) {
    for (j=0; j<bA->nc; j++) {
      if (!bA->m[i][j]) continue;
      PetscCall(MatZeroEntries(bA->m[i][j]));
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCopy_Nest(Mat A,Mat B,MatStructure str)
{
  Mat_Nest       *bA = (Mat_Nest*)A->data,*bB = (Mat_Nest*)B->data;
  PetscInt       i,j,nr = bA->nr,nc = bA->nc;
  PetscBool      nnzstate = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCheck(nr == bB->nr && nc == bB->nc,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_INCOMP,"Cannot copy a Mat_Nest of block size (%" PetscInt_FMT ",%" PetscInt_FMT ") to a Mat_Nest of block size (%" PetscInt_FMT ",%" PetscInt_FMT ")",bB->nr,bB->nc,nr,nc);
  for (i=0; i<nr; i++) {
    for (j=0; j<nc; j++) {
      PetscObjectState subnnzstate = 0;
      if (bA->m[i][j] && bB->m[i][j]) {
        PetscCall(MatCopy(bA->m[i][j],bB->m[i][j],str));
      } else PetscCheck(!bA->m[i][j] && !bB->m[i][j],PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_INCOMP,"Matrix block does not exist at %" PetscInt_FMT ",%" PetscInt_FMT,i,j);
      PetscCall(MatGetNonzeroState(bB->m[i][j],&subnnzstate));
      nnzstate = (PetscBool)(nnzstate || bB->nnzstate[i*nc+j] != subnnzstate);
      bB->nnzstate[i*nc+j] = subnnzstate;
    }
  }
  if (nnzstate) B->nonzerostate++;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatAXPY_Nest(Mat Y,PetscScalar a,Mat X,MatStructure str)
{
  Mat_Nest       *bY = (Mat_Nest*)Y->data,*bX = (Mat_Nest*)X->data;
  PetscInt       i,j,nr = bY->nr,nc = bY->nc;
  PetscBool      nnzstate = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCheck(nr == bX->nr && nc == bX->nc,PetscObjectComm((PetscObject)Y),PETSC_ERR_ARG_INCOMP,"Cannot AXPY a MatNest of block size (%" PetscInt_FMT ",%" PetscInt_FMT ") with a MatNest of block size (%" PetscInt_FMT ",%" PetscInt_FMT ")",bX->nr,bX->nc,nr,nc);
  for (i=0; i<nr; i++) {
    for (j=0; j<nc; j++) {
      PetscObjectState subnnzstate = 0;
      if (bY->m[i][j] && bX->m[i][j]) {
        PetscCall(MatAXPY(bY->m[i][j],a,bX->m[i][j],str));
      } else if (bX->m[i][j]) {
        Mat M;

        PetscCheck(str == DIFFERENT_NONZERO_PATTERN,PetscObjectComm((PetscObject)Y),PETSC_ERR_ARG_INCOMP,"Matrix block does not exist at %" PetscInt_FMT ",%" PetscInt_FMT ". Use DIFFERENT_NONZERO_PATTERN",i,j);
        PetscCall(MatDuplicate(bX->m[i][j],MAT_COPY_VALUES,&M));
        PetscCall(MatNestSetSubMat(Y,i,j,M));
        PetscCall(MatDestroy(&M));
      }
      if (bY->m[i][j]) PetscCall(MatGetNonzeroState(bY->m[i][j],&subnnzstate));
      nnzstate = (PetscBool)(nnzstate || bY->nnzstate[i*nc+j] != subnnzstate);
      bY->nnzstate[i*nc+j] = subnnzstate;
    }
  }
  if (nnzstate) Y->nonzerostate++;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDuplicate_Nest(Mat A,MatDuplicateOption op,Mat *B)
{
  Mat_Nest       *bA = (Mat_Nest*)A->data;
  Mat            *b;
  PetscInt       i,j,nr = bA->nr,nc = bA->nc;

  PetscFunctionBegin;
  PetscCall(PetscMalloc1(nr*nc,&b));
  for (i=0; i<nr; i++) {
    for (j=0; j<nc; j++) {
      if (bA->m[i][j]) {
        PetscCall(MatDuplicate(bA->m[i][j],op,&b[i*nc+j]));
      } else {
        b[i*nc+j] = NULL;
      }
    }
  }
  PetscCall(MatCreateNest(PetscObjectComm((PetscObject)A),nr,bA->isglobal.row,nc,bA->isglobal.col,b,B));
  /* Give the new MatNest exclusive ownership */
  for (i=0; i<nr*nc; i++) {
    PetscCall(MatDestroy(&b[i]));
  }
  PetscCall(PetscFree(b));

  PetscCall(MatAssemblyBegin(*B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*B,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

/* nest api */
PetscErrorCode MatNestGetSubMat_Nest(Mat A,PetscInt idxm,PetscInt jdxm,Mat *mat)
{
  Mat_Nest *bA = (Mat_Nest*)A->data;

  PetscFunctionBegin;
  PetscCheck(idxm < bA->nr,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %" PetscInt_FMT " max %" PetscInt_FMT,idxm,bA->nr-1);
  PetscCheck(jdxm < bA->nc,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_OUTOFRANGE,"Col too large: row %" PetscInt_FMT " max %" PetscInt_FMT,jdxm,bA->nc-1);
  *mat = bA->m[idxm][jdxm];
  PetscFunctionReturn(0);
}

/*@
 MatNestGetSubMat - Returns a single, sub-matrix from a nest matrix.

 Not collective

 Input Parameters:
+   A  - nest matrix
.   idxm - index of the matrix within the nest matrix
-   jdxm - index of the matrix within the nest matrix

 Output Parameter:
.   sub - matrix at index idxm,jdxm within the nest matrix

 Level: developer

.seealso: MatNestGetSize(), MatNestGetSubMats(), MatCreateNest(), MATNEST, MatNestSetSubMat(),
          MatNestGetLocalISs(), MatNestGetISs()
@*/
PetscErrorCode  MatNestGetSubMat(Mat A,PetscInt idxm,PetscInt jdxm,Mat *sub)
{
  PetscFunctionBegin;
  PetscUseMethod(A,"MatNestGetSubMat_C",(Mat,PetscInt,PetscInt,Mat*),(A,idxm,jdxm,sub));
  PetscFunctionReturn(0);
}

PetscErrorCode MatNestSetSubMat_Nest(Mat A,PetscInt idxm,PetscInt jdxm,Mat mat)
{
  Mat_Nest       *bA = (Mat_Nest*)A->data;
  PetscInt       m,n,M,N,mi,ni,Mi,Ni;

  PetscFunctionBegin;
  PetscCheck(idxm < bA->nr,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %" PetscInt_FMT " max %" PetscInt_FMT,idxm,bA->nr-1);
  PetscCheck(jdxm < bA->nc,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_OUTOFRANGE,"Col too large: row %" PetscInt_FMT " max %" PetscInt_FMT,jdxm,bA->nc-1);
  PetscCall(MatGetLocalSize(mat,&m,&n));
  PetscCall(MatGetSize(mat,&M,&N));
  PetscCall(ISGetLocalSize(bA->isglobal.row[idxm],&mi));
  PetscCall(ISGetSize(bA->isglobal.row[idxm],&Mi));
  PetscCall(ISGetLocalSize(bA->isglobal.col[jdxm],&ni));
  PetscCall(ISGetSize(bA->isglobal.col[jdxm],&Ni));
  PetscCheck(M == Mi && N == Ni,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_INCOMP,"Submatrix dimension (%" PetscInt_FMT ",%" PetscInt_FMT ") incompatible with nest block (%" PetscInt_FMT ",%" PetscInt_FMT ")",M,N,Mi,Ni);
  PetscCheck(m == mi && n == ni,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_INCOMP,"Submatrix local dimension (%" PetscInt_FMT ",%" PetscInt_FMT ") incompatible with nest block (%" PetscInt_FMT ",%" PetscInt_FMT ")",m,n,mi,ni);

  /* do not increase object state */
  if (mat == bA->m[idxm][jdxm]) PetscFunctionReturn(0);

  PetscCall(PetscObjectReference((PetscObject)mat));
  PetscCall(MatDestroy(&bA->m[idxm][jdxm]));
  bA->m[idxm][jdxm] = mat;
  PetscCall(PetscObjectStateIncrease((PetscObject)A));
  PetscCall(MatGetNonzeroState(mat,&bA->nnzstate[idxm*bA->nc+jdxm]));
  A->nonzerostate++;
  PetscFunctionReturn(0);
}

/*@
 MatNestSetSubMat - Set a single submatrix in the nest matrix.

 Logically collective on the submatrix communicator

 Input Parameters:
+   A  - nest matrix
.   idxm - index of the matrix within the nest matrix
.   jdxm - index of the matrix within the nest matrix
-   sub - matrix at index idxm,jdxm within the nest matrix

 Notes:
 The new submatrix must have the same size and communicator as that block of the nest.

 This increments the reference count of the submatrix.

 Level: developer

.seealso: MatNestSetSubMats(), MatNestGetSubMats(), MatNestGetLocalISs(), MATNEST, MatCreateNest(),
          MatNestGetSubMat(), MatNestGetISs(), MatNestGetSize()
@*/
PetscErrorCode  MatNestSetSubMat(Mat A,PetscInt idxm,PetscInt jdxm,Mat sub)
{
  PetscFunctionBegin;
  PetscUseMethod(A,"MatNestSetSubMat_C",(Mat,PetscInt,PetscInt,Mat),(A,idxm,jdxm,sub));
  PetscFunctionReturn(0);
}

PetscErrorCode MatNestGetSubMats_Nest(Mat A,PetscInt *M,PetscInt *N,Mat ***mat)
{
  Mat_Nest *bA = (Mat_Nest*)A->data;

  PetscFunctionBegin;
  if (M)   *M   = bA->nr;
  if (N)   *N   = bA->nc;
  if (mat) *mat = bA->m;
  PetscFunctionReturn(0);
}

/*@C
 MatNestGetSubMats - Returns the entire two dimensional array of matrices defining a nest matrix.

 Not collective

 Input Parameter:
.   A  - nest matrix

 Output Parameters:
+   M - number of rows in the nest matrix
.   N - number of cols in the nest matrix
-   mat - 2d array of matrices

 Notes:

 The user should not free the array mat.

 In Fortran, this routine has a calling sequence
$   call MatNestGetSubMats(A, M, N, mat, ierr)
 where the space allocated for the optional argument mat is assumed large enough (if provided).

 Level: developer

.seealso: MatNestGetSize(), MatNestGetSubMat(), MatNestGetLocalISs(), MATNEST, MatCreateNest(),
          MatNestSetSubMats(), MatNestGetISs(), MatNestSetSubMat()
@*/
PetscErrorCode  MatNestGetSubMats(Mat A,PetscInt *M,PetscInt *N,Mat ***mat)
{
  PetscFunctionBegin;
  PetscUseMethod(A,"MatNestGetSubMats_C",(Mat,PetscInt*,PetscInt*,Mat***),(A,M,N,mat));
  PetscFunctionReturn(0);
}

PetscErrorCode  MatNestGetSize_Nest(Mat A,PetscInt *M,PetscInt *N)
{
  Mat_Nest *bA = (Mat_Nest*)A->data;

  PetscFunctionBegin;
  if (M) *M = bA->nr;
  if (N) *N = bA->nc;
  PetscFunctionReturn(0);
}

/*@
 MatNestGetSize - Returns the size of the nest matrix.

 Not collective

 Input Parameter:
.   A  - nest matrix

 Output Parameters:
+   M - number of rows in the nested mat
-   N - number of cols in the nested mat

 Notes:

 Level: developer

.seealso: MatNestGetSubMat(), MatNestGetSubMats(), MATNEST, MatCreateNest(), MatNestGetLocalISs(),
          MatNestGetISs()
@*/
PetscErrorCode  MatNestGetSize(Mat A,PetscInt *M,PetscInt *N)
{
  PetscFunctionBegin;
  PetscUseMethod(A,"MatNestGetSize_C",(Mat,PetscInt*,PetscInt*),(A,M,N));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatNestGetISs_Nest(Mat A,IS rows[],IS cols[])
{
  Mat_Nest *vs = (Mat_Nest*)A->data;
  PetscInt i;

  PetscFunctionBegin;
  if (rows) for (i=0; i<vs->nr; i++) rows[i] = vs->isglobal.row[i];
  if (cols) for (i=0; i<vs->nc; i++) cols[i] = vs->isglobal.col[i];
  PetscFunctionReturn(0);
}

/*@C
 MatNestGetISs - Returns the index sets partitioning the row and column spaces

 Not collective

 Input Parameter:
.   A  - nest matrix

 Output Parameters:
+   rows - array of row index sets
-   cols - array of column index sets

 Level: advanced

 Notes:
 The user must have allocated arrays of the correct size. The reference count is not increased on the returned ISs.

.seealso: MatNestGetSubMat(), MatNestGetSubMats(), MatNestGetSize(), MatNestGetLocalISs(), MATNEST,
          MatCreateNest(), MatNestGetSubMats(), MatNestSetSubMats()
@*/
PetscErrorCode  MatNestGetISs(Mat A,IS rows[],IS cols[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscUseMethod(A,"MatNestGetISs_C",(Mat,IS[],IS[]),(A,rows,cols));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatNestGetLocalISs_Nest(Mat A,IS rows[],IS cols[])
{
  Mat_Nest *vs = (Mat_Nest*)A->data;
  PetscInt i;

  PetscFunctionBegin;
  if (rows) for (i=0; i<vs->nr; i++) rows[i] = vs->islocal.row[i];
  if (cols) for (i=0; i<vs->nc; i++) cols[i] = vs->islocal.col[i];
  PetscFunctionReturn(0);
}

/*@C
 MatNestGetLocalISs - Returns the index sets partitioning the row and column spaces

 Not collective

 Input Parameter:
.   A  - nest matrix

 Output Parameters:
+   rows - array of row index sets (or NULL to ignore)
-   cols - array of column index sets (or NULL to ignore)

 Level: advanced

 Notes:
 The user must have allocated arrays of the correct size. The reference count is not increased on the returned ISs.

.seealso: MatNestGetSubMat(), MatNestGetSubMats(), MatNestGetSize(), MatNestGetISs(), MatCreateNest(),
          MATNEST, MatNestSetSubMats(), MatNestSetSubMat()
@*/
PetscErrorCode  MatNestGetLocalISs(Mat A,IS rows[],IS cols[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscUseMethod(A,"MatNestGetLocalISs_C",(Mat,IS[],IS[]),(A,rows,cols));
  PetscFunctionReturn(0);
}

PetscErrorCode  MatNestSetVecType_Nest(Mat A,VecType vtype)
{
  PetscBool      flg;

  PetscFunctionBegin;
  PetscCall(PetscStrcmp(vtype,VECNEST,&flg));
  /* In reality, this only distinguishes VECNEST and "other" */
  if (flg) A->ops->getvecs = MatCreateVecs_Nest;
  else A->ops->getvecs = (PetscErrorCode (*)(Mat,Vec*,Vec*)) 0;
  PetscFunctionReturn(0);
}

/*@C
 MatNestSetVecType - Sets the type of Vec returned by MatCreateVecs()

 Not collective

 Input Parameters:
+  A  - nest matrix
-  vtype - type to use for creating vectors

 Notes:

 Level: developer

.seealso: MatCreateVecs(), MATNEST, MatCreateNest()
@*/
PetscErrorCode  MatNestSetVecType(Mat A,VecType vtype)
{
  PetscFunctionBegin;
  PetscTryMethod(A,"MatNestSetVecType_C",(Mat,VecType),(A,vtype));
  PetscFunctionReturn(0);
}

PetscErrorCode MatNestSetSubMats_Nest(Mat A,PetscInt nr,const IS is_row[],PetscInt nc,const IS is_col[],const Mat a[])
{
  Mat_Nest       *s = (Mat_Nest*)A->data;
  PetscInt       i,j,m,n,M,N;
  PetscBool      cong,isstd,sametype=PETSC_FALSE;
  VecType        vtype,type;

  PetscFunctionBegin;
  PetscCall(MatReset_Nest(A));

  s->nr = nr;
  s->nc = nc;

  /* Create space for submatrices */
  PetscCall(PetscMalloc1(nr,&s->m));
  for (i=0; i<nr; i++) {
    PetscCall(PetscMalloc1(nc,&s->m[i]));
  }
  for (i=0; i<nr; i++) {
    for (j=0; j<nc; j++) {
      s->m[i][j] = a[i*nc+j];
      if (a[i*nc+j]) {
        PetscCall(PetscObjectReference((PetscObject)a[i*nc+j]));
      }
    }
  }
  PetscCall(MatGetVecType(A,&vtype));
  PetscCall(PetscStrcmp(vtype,VECSTANDARD,&isstd));
  if (isstd) {
    /* check if all blocks have the same vectype */
    vtype = NULL;
    for (i=0; i<nr; i++) {
      for (j=0; j<nc; j++) {
        if (a[i*nc+j]) {
          if (!vtype) {  /* first visited block */
            PetscCall(MatGetVecType(a[i*nc+j],&vtype));
            sametype = PETSC_TRUE;
          } else if (sametype) {
            PetscCall(MatGetVecType(a[i*nc+j],&type));
            PetscCall(PetscStrcmp(vtype,type,&sametype));
          }
        }
      }
    }
    if (sametype) {  /* propagate vectype */
      PetscCall(MatSetVecType(A,vtype));
    }
  }

  PetscCall(MatSetUp_NestIS_Private(A,nr,is_row,nc,is_col));

  PetscCall(PetscMalloc1(nr,&s->row_len));
  PetscCall(PetscMalloc1(nc,&s->col_len));
  for (i=0; i<nr; i++) s->row_len[i]=-1;
  for (j=0; j<nc; j++) s->col_len[j]=-1;

  PetscCall(PetscCalloc1(nr*nc,&s->nnzstate));
  for (i=0; i<nr; i++) {
    for (j=0; j<nc; j++) {
      if (s->m[i][j]) {
        PetscCall(MatGetNonzeroState(s->m[i][j],&s->nnzstate[i*nc+j]));
      }
    }
  }

  PetscCall(MatNestGetSizes_Private(A,&m,&n,&M,&N));

  PetscCall(PetscLayoutSetSize(A->rmap,M));
  PetscCall(PetscLayoutSetLocalSize(A->rmap,m));
  PetscCall(PetscLayoutSetSize(A->cmap,N));
  PetscCall(PetscLayoutSetLocalSize(A->cmap,n));

  PetscCall(PetscLayoutSetUp(A->rmap));
  PetscCall(PetscLayoutSetUp(A->cmap));

  /* disable operations that are not supported for non-square matrices,
     or matrices for which is_row != is_col  */
  PetscCall(MatHasCongruentLayouts(A,&cong));
  if (cong && nr != nc) cong = PETSC_FALSE;
  if (cong) {
    for (i = 0; cong && i < nr; i++) {
      PetscCall(ISEqualUnsorted(s->isglobal.row[i],s->isglobal.col[i],&cong));
    }
  }
  if (!cong) {
    A->ops->missingdiagonal = NULL;
    A->ops->getdiagonal     = NULL;
    A->ops->shift           = NULL;
    A->ops->diagonalset     = NULL;
  }

  PetscCall(PetscCalloc2(nr,&s->left,nc,&s->right));
  PetscCall(PetscObjectStateIncrease((PetscObject)A));
  A->nonzerostate++;
  PetscFunctionReturn(0);
}

/*@
   MatNestSetSubMats - Sets the nested submatrices

   Collective on Mat

   Input Parameters:
+  A - nested matrix
.  nr - number of nested row blocks
.  is_row - index sets for each nested row block, or NULL to make contiguous
.  nc - number of nested column blocks
.  is_col - index sets for each nested column block, or NULL to make contiguous
-  a - row-aligned array of nr*nc submatrices, empty submatrices can be passed using NULL

   Notes: this always resets any submatrix information previously set

   Level: advanced

.seealso: MatCreateNest(), MATNEST, MatNestSetSubMat(), MatNestGetSubMat(), MatNestGetSubMats()
@*/
PetscErrorCode MatNestSetSubMats(Mat A,PetscInt nr,const IS is_row[],PetscInt nc,const IS is_col[],const Mat a[])
{
  PetscInt       i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscCheck(nr >= 0,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_OUTOFRANGE,"Number of rows cannot be negative");
  if (nr && is_row) {
    PetscValidPointer(is_row,3);
    for (i=0; i<nr; i++) PetscValidHeaderSpecific(is_row[i],IS_CLASSID,3);
  }
  PetscCheck(nc >= 0,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_OUTOFRANGE,"Number of columns cannot be negative");
  if (nc && is_col) {
    PetscValidPointer(is_col,5);
    for (i=0; i<nc; i++) PetscValidHeaderSpecific(is_col[i],IS_CLASSID,5);
  }
  if (nr*nc > 0) PetscValidPointer(a,6);
  PetscUseMethod(A,"MatNestSetSubMats_C",(Mat,PetscInt,const IS[],PetscInt,const IS[],const Mat[]),(A,nr,is_row,nc,is_col,a));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatNestCreateAggregateL2G_Private(Mat A,PetscInt n,const IS islocal[],const IS isglobal[],PetscBool colflg,ISLocalToGlobalMapping *ltog)
{
  PetscBool      flg;
  PetscInt       i,j,m,mi,*ix;

  PetscFunctionBegin;
  *ltog = NULL;
  for (i=0,m=0,flg=PETSC_FALSE; i<n; i++) {
    if (islocal[i]) {
      PetscCall(ISGetLocalSize(islocal[i],&mi));
      flg  = PETSC_TRUE;      /* We found a non-trivial entry */
    } else {
      PetscCall(ISGetLocalSize(isglobal[i],&mi));
    }
    m += mi;
  }
  if (!flg) PetscFunctionReturn(0);

  PetscCall(PetscMalloc1(m,&ix));
  for (i=0,m=0; i<n; i++) {
    ISLocalToGlobalMapping smap = NULL;
    Mat                    sub = NULL;
    PetscSF                sf;
    PetscLayout            map;
    const PetscInt         *ix2;

    if (!colflg) {
      PetscCall(MatNestFindNonzeroSubMatRow(A,i,&sub));
    } else {
      PetscCall(MatNestFindNonzeroSubMatCol(A,i,&sub));
    }
    if (sub) {
      if (!colflg) {
        PetscCall(MatGetLocalToGlobalMapping(sub,&smap,NULL));
      } else {
        PetscCall(MatGetLocalToGlobalMapping(sub,NULL,&smap));
      }
    }
    /*
       Now we need to extract the monolithic global indices that correspond to the given split global indices.
       In many/most cases, we only want MatGetLocalSubMatrix() to work, in which case we only need to know the size of the local spaces.
    */
    PetscCall(ISGetIndices(isglobal[i],&ix2));
    if (islocal[i]) {
      PetscInt *ilocal,*iremote;
      PetscInt mil,nleaves;

      PetscCall(ISGetLocalSize(islocal[i],&mi));
      PetscCheck(smap,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing local to global map");
      for (j=0; j<mi; j++) ix[m+j] = j;
      PetscCall(ISLocalToGlobalMappingApply(smap,mi,ix+m,ix+m));

      /* PetscSFSetGraphLayout does not like negative indices */
      PetscCall(PetscMalloc2(mi,&ilocal,mi,&iremote));
      for (j=0, nleaves = 0; j<mi; j++) {
        if (ix[m+j] < 0) continue;
        ilocal[nleaves]  = j;
        iremote[nleaves] = ix[m+j];
        nleaves++;
      }
      PetscCall(ISGetLocalSize(isglobal[i],&mil));
      PetscCall(PetscSFCreate(PetscObjectComm((PetscObject)A),&sf));
      PetscCall(PetscLayoutCreate(PetscObjectComm((PetscObject)A),&map));
      PetscCall(PetscLayoutSetLocalSize(map,mil));
      PetscCall(PetscLayoutSetUp(map));
      PetscCall(PetscSFSetGraphLayout(sf,map,nleaves,ilocal,PETSC_USE_POINTER,iremote));
      PetscCall(PetscLayoutDestroy(&map));
      PetscCall(PetscSFBcastBegin(sf,MPIU_INT,ix2,ix + m,MPI_REPLACE));
      PetscCall(PetscSFBcastEnd(sf,MPIU_INT,ix2,ix + m,MPI_REPLACE));
      PetscCall(PetscSFDestroy(&sf));
      PetscCall(PetscFree2(ilocal,iremote));
    } else {
      PetscCall(ISGetLocalSize(isglobal[i],&mi));
      for (j=0; j<mi; j++) ix[m+j] = ix2[i];
    }
    PetscCall(ISRestoreIndices(isglobal[i],&ix2));
    m   += mi;
  }
  PetscCall(ISLocalToGlobalMappingCreate(PetscObjectComm((PetscObject)A),1,m,ix,PETSC_OWN_POINTER,ltog));
  PetscFunctionReturn(0);
}

/* If an IS was provided, there is nothing Nest needs to do, otherwise Nest will build a strided IS */
/*
  nprocessors = NP
  Nest x^T = ((g_0,g_1,...g_nprocs-1), (h_0,h_1,...h_NP-1))
       proc 0: => (g_0,h_0,)
       proc 1: => (g_1,h_1,)
       ...
       proc nprocs-1: => (g_NP-1,h_NP-1,)

            proc 0:                      proc 1:                    proc nprocs-1:
    is[0] = (0,1,2,...,nlocal(g_0)-1)  (0,1,...,nlocal(g_1)-1)  (0,1,...,nlocal(g_NP-1))

            proc 0:
    is[1] = (nlocal(g_0),nlocal(g_0)+1,...,nlocal(g_0)+nlocal(h_0)-1)
            proc 1:
    is[1] = (nlocal(g_1),nlocal(g_1)+1,...,nlocal(g_1)+nlocal(h_1)-1)

            proc NP-1:
    is[1] = (nlocal(g_NP-1),nlocal(g_NP-1)+1,...,nlocal(g_NP-1)+nlocal(h_NP-1)-1)
*/
static PetscErrorCode MatSetUp_NestIS_Private(Mat A,PetscInt nr,const IS is_row[],PetscInt nc,const IS is_col[])
{
  Mat_Nest       *vs = (Mat_Nest*)A->data;
  PetscInt       i,j,offset,n,nsum,bs;
  Mat            sub = NULL;

  PetscFunctionBegin;
  PetscCall(PetscMalloc1(nr,&vs->isglobal.row));
  PetscCall(PetscMalloc1(nc,&vs->isglobal.col));
  if (is_row) { /* valid IS is passed in */
    /* refs on is[] are incremented */
    for (i=0; i<vs->nr; i++) {
      PetscCall(PetscObjectReference((PetscObject)is_row[i]));

      vs->isglobal.row[i] = is_row[i];
    }
  } else {                      /* Create the ISs by inspecting sizes of a submatrix in each row */
    nsum = 0;
    for (i=0; i<vs->nr; i++) {  /* Add up the local sizes to compute the aggregate offset */
      PetscCall(MatNestFindNonzeroSubMatRow(A,i,&sub));
      PetscCheck(sub,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"No nonzero submatrix in row %" PetscInt_FMT,i);
      PetscCall(MatGetLocalSize(sub,&n,NULL));
      PetscCheck(n >= 0,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONGSTATE,"Sizes have not yet been set for submatrix");
      nsum += n;
    }
    PetscCallMPI(MPI_Scan(&nsum,&offset,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)A)));
    offset -= nsum;
    for (i=0; i<vs->nr; i++) {
      PetscCall(MatNestFindNonzeroSubMatRow(A,i,&sub));
      PetscCall(MatGetLocalSize(sub,&n,NULL));
      PetscCall(MatGetBlockSizes(sub,&bs,NULL));
      PetscCall(ISCreateStride(PetscObjectComm((PetscObject)sub),n,offset,1,&vs->isglobal.row[i]));
      PetscCall(ISSetBlockSize(vs->isglobal.row[i],bs));
      offset += n;
    }
  }

  if (is_col) { /* valid IS is passed in */
    /* refs on is[] are incremented */
    for (j=0; j<vs->nc; j++) {
      PetscCall(PetscObjectReference((PetscObject)is_col[j]));

      vs->isglobal.col[j] = is_col[j];
    }
  } else {                      /* Create the ISs by inspecting sizes of a submatrix in each column */
    offset = A->cmap->rstart;
    nsum   = 0;
    for (j=0; j<vs->nc; j++) {
      PetscCall(MatNestFindNonzeroSubMatCol(A,j,&sub));
      PetscCheck(sub,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"No nonzero submatrix in column %" PetscInt_FMT,i);
      PetscCall(MatGetLocalSize(sub,NULL,&n));
      PetscCheck(n >= 0,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONGSTATE,"Sizes have not yet been set for submatrix");
      nsum += n;
    }
    PetscCallMPI(MPI_Scan(&nsum,&offset,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)A)));
    offset -= nsum;
    for (j=0; j<vs->nc; j++) {
      PetscCall(MatNestFindNonzeroSubMatCol(A,j,&sub));
      PetscCall(MatGetLocalSize(sub,NULL,&n));
      PetscCall(MatGetBlockSizes(sub,NULL,&bs));
      PetscCall(ISCreateStride(PetscObjectComm((PetscObject)sub),n,offset,1,&vs->isglobal.col[j]));
      PetscCall(ISSetBlockSize(vs->isglobal.col[j],bs));
      offset += n;
    }
  }

  /* Set up the local ISs */
  PetscCall(PetscMalloc1(vs->nr,&vs->islocal.row));
  PetscCall(PetscMalloc1(vs->nc,&vs->islocal.col));
  for (i=0,offset=0; i<vs->nr; i++) {
    IS                     isloc;
    ISLocalToGlobalMapping rmap = NULL;
    PetscInt               nlocal,bs;
    PetscCall(MatNestFindNonzeroSubMatRow(A,i,&sub));
    if (sub) PetscCall(MatGetLocalToGlobalMapping(sub,&rmap,NULL));
    if (rmap) {
      PetscCall(MatGetBlockSizes(sub,&bs,NULL));
      PetscCall(ISLocalToGlobalMappingGetSize(rmap,&nlocal));
      PetscCall(ISCreateStride(PETSC_COMM_SELF,nlocal,offset,1,&isloc));
      PetscCall(ISSetBlockSize(isloc,bs));
    } else {
      nlocal = 0;
      isloc  = NULL;
    }
    vs->islocal.row[i] = isloc;
    offset            += nlocal;
  }
  for (i=0,offset=0; i<vs->nc; i++) {
    IS                     isloc;
    ISLocalToGlobalMapping cmap = NULL;
    PetscInt               nlocal,bs;
    PetscCall(MatNestFindNonzeroSubMatCol(A,i,&sub));
    if (sub) PetscCall(MatGetLocalToGlobalMapping(sub,NULL,&cmap));
    if (cmap) {
      PetscCall(MatGetBlockSizes(sub,NULL,&bs));
      PetscCall(ISLocalToGlobalMappingGetSize(cmap,&nlocal));
      PetscCall(ISCreateStride(PETSC_COMM_SELF,nlocal,offset,1,&isloc));
      PetscCall(ISSetBlockSize(isloc,bs));
    } else {
      nlocal = 0;
      isloc  = NULL;
    }
    vs->islocal.col[i] = isloc;
    offset            += nlocal;
  }

  /* Set up the aggregate ISLocalToGlobalMapping */
  {
    ISLocalToGlobalMapping rmap,cmap;
    PetscCall(MatNestCreateAggregateL2G_Private(A,vs->nr,vs->islocal.row,vs->isglobal.row,PETSC_FALSE,&rmap));
    PetscCall(MatNestCreateAggregateL2G_Private(A,vs->nc,vs->islocal.col,vs->isglobal.col,PETSC_TRUE,&cmap));
    if (rmap && cmap) PetscCall(MatSetLocalToGlobalMapping(A,rmap,cmap));
    PetscCall(ISLocalToGlobalMappingDestroy(&rmap));
    PetscCall(ISLocalToGlobalMappingDestroy(&cmap));
  }

  if (PetscDefined(USE_DEBUG)) {
    for (i=0; i<vs->nr; i++) {
      for (j=0; j<vs->nc; j++) {
        PetscInt m,n,M,N,mi,ni,Mi,Ni;
        Mat      B = vs->m[i][j];
        if (!B) continue;
        PetscCall(MatGetSize(B,&M,&N));
        PetscCall(MatGetLocalSize(B,&m,&n));
        PetscCall(ISGetSize(vs->isglobal.row[i],&Mi));
        PetscCall(ISGetSize(vs->isglobal.col[j],&Ni));
        PetscCall(ISGetLocalSize(vs->isglobal.row[i],&mi));
        PetscCall(ISGetLocalSize(vs->isglobal.col[j],&ni));
        PetscCheck(M == Mi && N == Ni,PetscObjectComm((PetscObject)sub),PETSC_ERR_ARG_INCOMP,"Global sizes (%" PetscInt_FMT ",%" PetscInt_FMT ") of nested submatrix (%" PetscInt_FMT ",%" PetscInt_FMT ") do not agree with space defined by index sets (%" PetscInt_FMT ",%" PetscInt_FMT ")",M,N,i,j,Mi,Ni);
        PetscCheck(m == mi && n == ni,PetscObjectComm((PetscObject)sub),PETSC_ERR_ARG_INCOMP,"Local sizes (%" PetscInt_FMT ",%" PetscInt_FMT ") of nested submatrix (%" PetscInt_FMT ",%" PetscInt_FMT ") do not agree with space defined by index sets (%" PetscInt_FMT ",%" PetscInt_FMT ")",m,n,i,j,mi,ni);
      }
    }
  }

  /* Set A->assembled if all non-null blocks are currently assembled */
  for (i=0; i<vs->nr; i++) {
    for (j=0; j<vs->nc; j++) {
      if (vs->m[i][j] && !vs->m[i][j]->assembled) PetscFunctionReturn(0);
    }
  }
  A->assembled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@C
   MatCreateNest - Creates a new matrix containing several nested submatrices, each stored separately

   Collective on Mat

   Input Parameters:
+  comm - Communicator for the new Mat
.  nr - number of nested row blocks
.  is_row - index sets for each nested row block, or NULL to make contiguous
.  nc - number of nested column blocks
.  is_col - index sets for each nested column block, or NULL to make contiguous
-  a - row-aligned array of nr*nc submatrices, empty submatrices can be passed using NULL

   Output Parameter:
.  B - new matrix

   Level: advanced

.seealso: MatCreate(), VecCreateNest(), DMCreateMatrix(), MATNEST, MatNestSetSubMat(),
          MatNestGetSubMat(), MatNestGetLocalISs(), MatNestGetSize(),
          MatNestGetISs(), MatNestSetSubMats(), MatNestGetSubMats()
@*/
PetscErrorCode MatCreateNest(MPI_Comm comm,PetscInt nr,const IS is_row[],PetscInt nc,const IS is_col[],const Mat a[],Mat *B)
{
  Mat            A;

  PetscFunctionBegin;
  *B   = NULL;
  PetscCall(MatCreate(comm,&A));
  PetscCall(MatSetType(A,MATNEST));
  A->preallocated = PETSC_TRUE;
  PetscCall(MatNestSetSubMats(A,nr,is_row,nc,is_col,a));
  *B   = A;
  PetscFunctionReturn(0);
}

PetscErrorCode MatConvert_Nest_SeqAIJ_fast(Mat A,MatType newtype,MatReuse reuse,Mat *newmat)
{
  Mat_Nest       *nest = (Mat_Nest*)A->data;
  Mat            *trans;
  PetscScalar    **avv;
  PetscScalar    *vv;
  PetscInt       **aii,**ajj;
  PetscInt       *ii,*jj,*ci;
  PetscInt       nr,nc,nnz,i,j;
  PetscBool      done;

  PetscFunctionBegin;
  PetscCall(MatGetSize(A,&nr,&nc));
  if (reuse == MAT_REUSE_MATRIX) {
    PetscInt rnr;

    PetscCall(MatGetRowIJ(*newmat,0,PETSC_FALSE,PETSC_FALSE,&rnr,(const PetscInt**)&ii,(const PetscInt**)&jj,&done));
    PetscCheck(done,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"MatGetRowIJ");
    PetscCheck(rnr == nr,PetscObjectComm((PetscObject)A),PETSC_ERR_USER,"Cannot reuse matrix, wrong number of rows");
    PetscCall(MatSeqAIJGetArray(*newmat,&vv));
  }
  /* extract CSR for nested SeqAIJ matrices */
  nnz  = 0;
  PetscCall(PetscCalloc4(nest->nr*nest->nc,&aii,nest->nr*nest->nc,&ajj,nest->nr*nest->nc,&avv,nest->nr*nest->nc,&trans));
  for (i=0; i<nest->nr; ++i) {
    for (j=0; j<nest->nc; ++j) {
      Mat B = nest->m[i][j];
      if (B) {
        PetscScalar *naa;
        PetscInt    *nii,*njj,nnr;
        PetscBool   istrans;

        PetscCall(PetscObjectTypeCompare((PetscObject)B,MATTRANSPOSEMAT,&istrans));
        if (istrans) {
          Mat Bt;

          PetscCall(MatTransposeGetMat(B,&Bt));
          PetscCall(MatTranspose(Bt,MAT_INITIAL_MATRIX,&trans[i*nest->nc+j]));
          B    = trans[i*nest->nc+j];
        }
        PetscCall(MatGetRowIJ(B,0,PETSC_FALSE,PETSC_FALSE,&nnr,(const PetscInt**)&nii,(const PetscInt**)&njj,&done));
        PetscCheck(done,PetscObjectComm((PetscObject)B),PETSC_ERR_PLIB,"MatGetRowIJ");
        PetscCall(MatSeqAIJGetArray(B,&naa));
        nnz += nii[nnr];

        aii[i*nest->nc+j] = nii;
        ajj[i*nest->nc+j] = njj;
        avv[i*nest->nc+j] = naa;
      }
    }
  }
  if (reuse != MAT_REUSE_MATRIX) {
    PetscCall(PetscMalloc1(nr+1,&ii));
    PetscCall(PetscMalloc1(nnz,&jj));
    PetscCall(PetscMalloc1(nnz,&vv));
  } else {
    PetscCheck(nnz == ii[nr],PetscObjectComm((PetscObject)A),PETSC_ERR_USER,"Cannot reuse matrix, wrong number of nonzeros");
  }

  /* new row pointer */
  PetscCall(PetscArrayzero(ii,nr+1));
  for (i=0; i<nest->nr; ++i) {
    PetscInt       ncr,rst;

    PetscCall(ISStrideGetInfo(nest->isglobal.row[i],&rst,NULL));
    PetscCall(ISGetLocalSize(nest->isglobal.row[i],&ncr));
    for (j=0; j<nest->nc; ++j) {
      if (aii[i*nest->nc+j]) {
        PetscInt    *nii = aii[i*nest->nc+j];
        PetscInt    ir;

        for (ir=rst; ir<ncr+rst; ++ir) {
          ii[ir+1] += nii[1]-nii[0];
          nii++;
        }
      }
    }
  }
  for (i=0; i<nr; i++) ii[i+1] += ii[i];

  /* construct CSR for the new matrix */
  PetscCall(PetscCalloc1(nr,&ci));
  for (i=0; i<nest->nr; ++i) {
    PetscInt       ncr,rst;

    PetscCall(ISStrideGetInfo(nest->isglobal.row[i],&rst,NULL));
    PetscCall(ISGetLocalSize(nest->isglobal.row[i],&ncr));
    for (j=0; j<nest->nc; ++j) {
      if (aii[i*nest->nc+j]) {
        PetscScalar *nvv = avv[i*nest->nc+j];
        PetscInt    *nii = aii[i*nest->nc+j];
        PetscInt    *njj = ajj[i*nest->nc+j];
        PetscInt    ir,cst;

        PetscCall(ISStrideGetInfo(nest->isglobal.col[j],&cst,NULL));
        for (ir=rst; ir<ncr+rst; ++ir) {
          PetscInt ij,rsize = nii[1]-nii[0],ist = ii[ir]+ci[ir];

          for (ij=0;ij<rsize;ij++) {
            jj[ist+ij] = *njj+cst;
            vv[ist+ij] = *nvv;
            njj++;
            nvv++;
          }
          ci[ir] += rsize;
          nii++;
        }
      }
    }
  }
  PetscCall(PetscFree(ci));

  /* restore info */
  for (i=0; i<nest->nr; ++i) {
    for (j=0; j<nest->nc; ++j) {
      Mat B = nest->m[i][j];
      if (B) {
        PetscInt nnr = 0, k = i*nest->nc+j;

        B    = (trans[k] ? trans[k] : B);
        PetscCall(MatRestoreRowIJ(B,0,PETSC_FALSE,PETSC_FALSE,&nnr,(const PetscInt**)&aii[k],(const PetscInt**)&ajj[k],&done));
        PetscCheck(done,PetscObjectComm((PetscObject)B),PETSC_ERR_PLIB,"MatRestoreRowIJ");
        PetscCall(MatSeqAIJRestoreArray(B,&avv[k]));
        PetscCall(MatDestroy(&trans[k]));
      }
    }
  }
  PetscCall(PetscFree4(aii,ajj,avv,trans));

  /* finalize newmat */
  if (reuse == MAT_INITIAL_MATRIX) {
    PetscCall(MatCreateSeqAIJWithArrays(PetscObjectComm((PetscObject)A),nr,nc,ii,jj,vv,newmat));
  } else if (reuse == MAT_INPLACE_MATRIX) {
    Mat B;

    PetscCall(MatCreateSeqAIJWithArrays(PetscObjectComm((PetscObject)A),nr,nc,ii,jj,vv,&B));
    PetscCall(MatHeaderReplace(A,&B));
  }
  PetscCall(MatAssemblyBegin(*newmat,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*newmat,MAT_FINAL_ASSEMBLY));
  {
    Mat_SeqAIJ *a = (Mat_SeqAIJ*)((*newmat)->data);
    a->free_a     = PETSC_TRUE;
    a->free_ij    = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatAXPY_Dense_Nest(Mat Y,PetscScalar a,Mat X)
{
  Mat_Nest       *nest = (Mat_Nest*)X->data;
  PetscInt       i,j,k,rstart;
  PetscBool      flg;

  PetscFunctionBegin;
  /* Fill by row */
  for (j=0; j<nest->nc; ++j) {
    /* Using global column indices and ISAllGather() is not scalable. */
    IS             bNis;
    PetscInt       bN;
    const PetscInt *bNindices;
    PetscCall(ISAllGather(nest->isglobal.col[j], &bNis));
    PetscCall(ISGetSize(bNis,&bN));
    PetscCall(ISGetIndices(bNis,&bNindices));
    for (i=0; i<nest->nr; ++i) {
      Mat            B,D=NULL;
      PetscInt       bm, br;
      const PetscInt *bmindices;
      B = nest->m[i][j];
      if (!B) continue;
      PetscCall(PetscObjectTypeCompare((PetscObject)B,MATTRANSPOSEMAT,&flg));
      if (flg) {
        PetscTryMethod(B,"MatTransposeGetMat_C",(Mat,Mat*),(B,&D));
        PetscTryMethod(B,"MatHermitianTransposeGetMat_C",(Mat,Mat*),(B,&D));
        PetscCall(MatConvert(B,((PetscObject)D)->type_name,MAT_INITIAL_MATRIX,&D));
        B = D;
      }
      PetscCall(PetscObjectTypeCompareAny((PetscObject)B,&flg,MATSEQSBAIJ,MATMPISBAIJ,""));
      if (flg) {
        if (D) {
          PetscCall(MatConvert(D,MATBAIJ,MAT_INPLACE_MATRIX,&D));
        } else {
          PetscCall(MatConvert(B,MATBAIJ,MAT_INITIAL_MATRIX,&D));
        }
        B = D;
      }
      PetscCall(ISGetLocalSize(nest->isglobal.row[i],&bm));
      PetscCall(ISGetIndices(nest->isglobal.row[i],&bmindices));
      PetscCall(MatGetOwnershipRange(B,&rstart,NULL));
      for (br = 0; br < bm; ++br) {
        PetscInt          row = bmindices[br], brncols, *cols;
        const PetscInt    *brcols;
        const PetscScalar *brcoldata;
        PetscScalar       *vals = NULL;
        PetscCall(MatGetRow(B,br+rstart,&brncols,&brcols,&brcoldata));
        PetscCall(PetscMalloc1(brncols,&cols));
        for (k=0; k<brncols; k++) cols[k] = bNindices[brcols[k]];
        /*
          Nest blocks are required to be nonoverlapping -- otherwise nest and monolithic index layouts wouldn't match.
          Thus, we could use INSERT_VALUES, but I prefer ADD_VALUES.
         */
        if (a != 1.0) {
          PetscCall(PetscMalloc1(brncols,&vals));
          for (k=0; k<brncols; k++) vals[k] = a * brcoldata[k];
          PetscCall(MatSetValues(Y,1,&row,brncols,cols,vals,ADD_VALUES));
          PetscCall(PetscFree(vals));
        } else {
          PetscCall(MatSetValues(Y,1,&row,brncols,cols,brcoldata,ADD_VALUES));
        }
        PetscCall(MatRestoreRow(B,br+rstart,&brncols,&brcols,&brcoldata));
        PetscCall(PetscFree(cols));
      }
      if (D) {
        PetscCall(MatDestroy(&D));
      }
      PetscCall(ISRestoreIndices(nest->isglobal.row[i],&bmindices));
    }
    PetscCall(ISRestoreIndices(bNis,&bNindices));
    PetscCall(ISDestroy(&bNis));
  }
  PetscCall(MatAssemblyBegin(Y,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Y,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

PetscErrorCode MatConvert_Nest_AIJ(Mat A,MatType newtype,MatReuse reuse,Mat *newmat)
{
  Mat_Nest       *nest = (Mat_Nest*)A->data;
  PetscInt       m,n,M,N,i,j,k,*dnnz,*onnz,rstart,cstart,cend;
  PetscMPIInt    size;
  Mat            C;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)A),&size));
  if (size == 1) { /* look for a special case with SeqAIJ matrices and strided-1, contiguous, blocks */
    PetscInt  nf;
    PetscBool fast;

    PetscCall(PetscStrcmp(newtype,MATAIJ,&fast));
    if (!fast) {
      PetscCall(PetscStrcmp(newtype,MATSEQAIJ,&fast));
    }
    for (i=0; i<nest->nr && fast; ++i) {
      for (j=0; j<nest->nc && fast; ++j) {
        Mat B = nest->m[i][j];
        if (B) {
          PetscCall(PetscObjectTypeCompare((PetscObject)B,MATSEQAIJ,&fast));
          if (!fast) {
            PetscBool istrans;

            PetscCall(PetscObjectTypeCompare((PetscObject)B,MATTRANSPOSEMAT,&istrans));
            if (istrans) {
              Mat Bt;

              PetscCall(MatTransposeGetMat(B,&Bt));
              PetscCall(PetscObjectTypeCompare((PetscObject)Bt,MATSEQAIJ,&fast));
            }
          }
        }
      }
    }
    for (i=0, nf=0; i<nest->nr && fast; ++i) {
      PetscCall(PetscObjectTypeCompare((PetscObject)nest->isglobal.row[i],ISSTRIDE,&fast));
      if (fast) {
        PetscInt f,s;

        PetscCall(ISStrideGetInfo(nest->isglobal.row[i],&f,&s));
        if (f != nf || s != 1) { fast = PETSC_FALSE; }
        else {
          PetscCall(ISGetSize(nest->isglobal.row[i],&f));
          nf  += f;
        }
      }
    }
    for (i=0, nf=0; i<nest->nc && fast; ++i) {
      PetscCall(PetscObjectTypeCompare((PetscObject)nest->isglobal.col[i],ISSTRIDE,&fast));
      if (fast) {
        PetscInt f,s;

        PetscCall(ISStrideGetInfo(nest->isglobal.col[i],&f,&s));
        if (f != nf || s != 1) { fast = PETSC_FALSE; }
        else {
          PetscCall(ISGetSize(nest->isglobal.col[i],&f));
          nf  += f;
        }
      }
    }
    if (fast) {
      PetscCall(MatConvert_Nest_SeqAIJ_fast(A,newtype,reuse,newmat));
      PetscFunctionReturn(0);
    }
  }
  PetscCall(MatGetSize(A,&M,&N));
  PetscCall(MatGetLocalSize(A,&m,&n));
  PetscCall(MatGetOwnershipRangeColumn(A,&cstart,&cend));
  if (reuse == MAT_REUSE_MATRIX) C = *newmat;
  else {
    PetscCall(MatCreate(PetscObjectComm((PetscObject)A),&C));
    PetscCall(MatSetType(C,newtype));
    PetscCall(MatSetSizes(C,m,n,M,N));
  }
  PetscCall(PetscMalloc1(2*m,&dnnz));
  onnz = dnnz + m;
  for (k=0; k<m; k++) {
    dnnz[k] = 0;
    onnz[k] = 0;
  }
  for (j=0; j<nest->nc; ++j) {
    IS             bNis;
    PetscInt       bN;
    const PetscInt *bNindices;
    /* Using global column indices and ISAllGather() is not scalable. */
    PetscCall(ISAllGather(nest->isglobal.col[j], &bNis));
    PetscCall(ISGetSize(bNis, &bN));
    PetscCall(ISGetIndices(bNis,&bNindices));
    for (i=0; i<nest->nr; ++i) {
      PetscSF        bmsf;
      PetscSFNode    *iremote;
      Mat            B;
      PetscInt       bm, *sub_dnnz,*sub_onnz, br;
      const PetscInt *bmindices;
      B = nest->m[i][j];
      if (!B) continue;
      PetscCall(ISGetLocalSize(nest->isglobal.row[i],&bm));
      PetscCall(ISGetIndices(nest->isglobal.row[i],&bmindices));
      PetscCall(PetscSFCreate(PetscObjectComm((PetscObject)A), &bmsf));
      PetscCall(PetscMalloc1(bm,&iremote));
      PetscCall(PetscMalloc1(bm,&sub_dnnz));
      PetscCall(PetscMalloc1(bm,&sub_onnz));
      for (k = 0; k < bm; ++k) {
        sub_dnnz[k] = 0;
        sub_onnz[k] = 0;
      }
      /*
       Locate the owners for all of the locally-owned global row indices for this row block.
       These determine the roots of PetscSF used to communicate preallocation data to row owners.
       The roots correspond to the dnnz and onnz entries; thus, there are two roots per row.
       */
      PetscCall(MatGetOwnershipRange(B,&rstart,NULL));
      for (br = 0; br < bm; ++br) {
        PetscInt       row = bmindices[br], brncols, col;
        const PetscInt *brcols;
        PetscInt       rowrel = 0; /* row's relative index on its owner rank */
        PetscMPIInt    rowowner = 0;
        PetscCall(PetscLayoutFindOwnerIndex(A->rmap,row,&rowowner,&rowrel));
        /* how many roots  */
        iremote[br].rank = rowowner; iremote[br].index = rowrel;           /* edge from bmdnnz to dnnz */
        /* get nonzero pattern */
        PetscCall(MatGetRow(B,br+rstart,&brncols,&brcols,NULL));
        for (k=0; k<brncols; k++) {
          col  = bNindices[brcols[k]];
          if (col>=A->cmap->range[rowowner] && col<A->cmap->range[rowowner+1]) {
            sub_dnnz[br]++;
          } else {
            sub_onnz[br]++;
          }
        }
        PetscCall(MatRestoreRow(B,br+rstart,&brncols,&brcols,NULL));
      }
      PetscCall(ISRestoreIndices(nest->isglobal.row[i],&bmindices));
      /* bsf will have to take care of disposing of bedges. */
      PetscCall(PetscSFSetGraph(bmsf,m,bm,NULL,PETSC_OWN_POINTER,iremote,PETSC_OWN_POINTER));
      PetscCall(PetscSFReduceBegin(bmsf,MPIU_INT,sub_dnnz,dnnz,MPI_SUM));
      PetscCall(PetscSFReduceEnd(bmsf,MPIU_INT,sub_dnnz,dnnz,MPI_SUM));
      PetscCall(PetscSFReduceBegin(bmsf,MPIU_INT,sub_onnz,onnz,MPI_SUM));
      PetscCall(PetscSFReduceEnd(bmsf,MPIU_INT,sub_onnz,onnz,MPI_SUM));
      PetscCall(PetscFree(sub_dnnz));
      PetscCall(PetscFree(sub_onnz));
      PetscCall(PetscSFDestroy(&bmsf));
    }
    PetscCall(ISRestoreIndices(bNis,&bNindices));
    PetscCall(ISDestroy(&bNis));
  }
  /* Resize preallocation if overestimated */
  for (i=0;i<m;i++) {
    dnnz[i] = PetscMin(dnnz[i],A->cmap->n);
    onnz[i] = PetscMin(onnz[i],A->cmap->N - A->cmap->n);
  }
  PetscCall(MatSeqAIJSetPreallocation(C,0,dnnz));
  PetscCall(MatMPIAIJSetPreallocation(C,0,dnnz,0,onnz));
  PetscCall(PetscFree(dnnz));
  PetscCall(MatAXPY_Dense_Nest(C,1.0,A));
  if (reuse == MAT_INPLACE_MATRIX) {
    PetscCall(MatHeaderReplace(A,&C));
  } else *newmat = C;
  PetscFunctionReturn(0);
}

PetscErrorCode MatConvert_Nest_Dense(Mat A,MatType newtype,MatReuse reuse,Mat *newmat)
{
  Mat            B;
  PetscInt       m,n,M,N;

  PetscFunctionBegin;
  PetscCall(MatGetSize(A,&M,&N));
  PetscCall(MatGetLocalSize(A,&m,&n));
  if (reuse == MAT_REUSE_MATRIX) {
    B = *newmat;
    PetscCall(MatZeroEntries(B));
  } else {
    PetscCall(MatCreateDense(PetscObjectComm((PetscObject)A),m,PETSC_DECIDE,M,N,NULL,&B));
  }
  PetscCall(MatAXPY_Dense_Nest(B,1.0,A));
  if (reuse == MAT_INPLACE_MATRIX) {
    PetscCall(MatHeaderReplace(A,&B));
  } else if (reuse == MAT_INITIAL_MATRIX) *newmat = B;
  PetscFunctionReturn(0);
}

PetscErrorCode MatHasOperation_Nest(Mat mat,MatOperation op,PetscBool *has)
{
  Mat_Nest       *bA = (Mat_Nest*)mat->data;
  MatOperation   opAdd;
  PetscInt       i,j,nr = bA->nr,nc = bA->nc;
  PetscBool      flg;
  PetscFunctionBegin;

  *has = PETSC_FALSE;
  if (op == MATOP_MULT || op == MATOP_MULT_ADD || op == MATOP_MULT_TRANSPOSE || op == MATOP_MULT_TRANSPOSE_ADD) {
    opAdd = (op == MATOP_MULT || op == MATOP_MULT_ADD ? MATOP_MULT_ADD : MATOP_MULT_TRANSPOSE_ADD);
    for (j=0; j<nc; j++) {
      for (i=0; i<nr; i++) {
        if (!bA->m[i][j]) continue;
        PetscCall(MatHasOperation(bA->m[i][j],opAdd,&flg));
        if (!flg) PetscFunctionReturn(0);
      }
    }
  }
  if (((void**)mat->ops)[op]) *has = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*MC
  MATNEST - MATNEST = "nest" - Matrix type consisting of nested submatrices, each stored separately.

  Level: intermediate

  Notes:
  This matrix type permits scalable use of PCFieldSplit and avoids the large memory costs of extracting submatrices.
  It allows the use of symmetric and block formats for parts of multi-physics simulations.
  It is usually used with DMComposite and DMCreateMatrix()

  Each of the submatrices lives on the same MPI communicator as the original nest matrix (though they can have zero
  rows/columns on some processes.) Thus this is not meant for cases where the submatrices live on far fewer processes
  than the nest matrix.

.seealso: MatCreate(), MatType, MatCreateNest(), MatNestSetSubMat(), MatNestGetSubMat(),
          VecCreateNest(), DMCreateMatrix(), DMCOMPOSITE, MatNestSetVecType(), MatNestGetLocalISs(),
          MatNestGetISs(), MatNestSetSubMats(), MatNestGetSubMats()
M*/
PETSC_EXTERN PetscErrorCode MatCreate_Nest(Mat A)
{
  Mat_Nest       *s;

  PetscFunctionBegin;
  PetscCall(PetscNewLog(A,&s));
  A->data = (void*)s;

  s->nr            = -1;
  s->nc            = -1;
  s->m             = NULL;
  s->splitassembly = PETSC_FALSE;

  PetscCall(PetscMemzero(A->ops,sizeof(*A->ops)));

  A->ops->mult                  = MatMult_Nest;
  A->ops->multadd               = MatMultAdd_Nest;
  A->ops->multtranspose         = MatMultTranspose_Nest;
  A->ops->multtransposeadd      = MatMultTransposeAdd_Nest;
  A->ops->transpose             = MatTranspose_Nest;
  A->ops->assemblybegin         = MatAssemblyBegin_Nest;
  A->ops->assemblyend           = MatAssemblyEnd_Nest;
  A->ops->zeroentries           = MatZeroEntries_Nest;
  A->ops->copy                  = MatCopy_Nest;
  A->ops->axpy                  = MatAXPY_Nest;
  A->ops->duplicate             = MatDuplicate_Nest;
  A->ops->createsubmatrix       = MatCreateSubMatrix_Nest;
  A->ops->destroy               = MatDestroy_Nest;
  A->ops->view                  = MatView_Nest;
  A->ops->getvecs               = NULL; /* Use VECNEST by calling MatNestSetVecType(A,VECNEST) */
  A->ops->getlocalsubmatrix     = MatGetLocalSubMatrix_Nest;
  A->ops->restorelocalsubmatrix = MatRestoreLocalSubMatrix_Nest;
  A->ops->getdiagonal           = MatGetDiagonal_Nest;
  A->ops->diagonalscale         = MatDiagonalScale_Nest;
  A->ops->scale                 = MatScale_Nest;
  A->ops->shift                 = MatShift_Nest;
  A->ops->diagonalset           = MatDiagonalSet_Nest;
  A->ops->setrandom             = MatSetRandom_Nest;
  A->ops->hasoperation          = MatHasOperation_Nest;
  A->ops->missingdiagonal       = MatMissingDiagonal_Nest;

  A->spptr        = NULL;
  A->assembled    = PETSC_FALSE;

  /* expose Nest api's */
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatNestGetSubMat_C",        MatNestGetSubMat_Nest));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatNestSetSubMat_C",        MatNestSetSubMat_Nest));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatNestGetSubMats_C",       MatNestGetSubMats_Nest));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatNestGetSize_C",          MatNestGetSize_Nest));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatNestGetISs_C",           MatNestGetISs_Nest));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatNestGetLocalISs_C",      MatNestGetLocalISs_Nest));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatNestSetVecType_C",       MatNestSetVecType_Nest));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatNestSetSubMats_C",       MatNestSetSubMats_Nest));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatConvert_nest_mpiaij_C",  MatConvert_Nest_AIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatConvert_nest_seqaij_C",  MatConvert_Nest_AIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatConvert_nest_aij_C",     MatConvert_Nest_AIJ));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatConvert_nest_is_C",      MatConvert_Nest_IS));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatConvert_nest_mpidense_C",MatConvert_Nest_Dense));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatConvert_nest_seqdense_C",MatConvert_Nest_Dense));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_nest_seqdense_C",MatProductSetFromOptions_Nest_Dense));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_nest_mpidense_C",MatProductSetFromOptions_Nest_Dense));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_nest_dense_C",MatProductSetFromOptions_Nest_Dense));

  PetscCall(PetscObjectChangeTypeName((PetscObject)A,MATNEST));
  PetscFunctionReturn(0);
}
