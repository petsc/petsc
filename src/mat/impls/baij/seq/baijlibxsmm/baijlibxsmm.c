#include <../src/mat/impls/baij/seq/baij.h> /*I   "petscmat.h"   I*/
#include <libxsmm.h>

typedef struct {
  libxsmm_gemmfunction kernel;
  PetscInt             n, ldb, ldc;
} SeqBAIJLIBXSMM_SeqDense;

PETSC_INTERN PetscErrorCode MatConvert_SeqBAIJ_SeqBAIJLIBXSMM(Mat, MatType, MatReuse, Mat *);

static PetscErrorCode MatDuplicate_SeqBAIJLIBXSMM(Mat A, MatDuplicateOption op, Mat *B)
{
  PetscFunctionBegin;
  PetscCall(MatDuplicate_SeqBAIJ(A, op, B));
  PetscCall(MatConvert_SeqBAIJ_SeqBAIJLIBXSMM(*B, MATSEQBAIJLIBXSMM, MAT_INPLACE_MATRIX, B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatProductDataDestroy_SeqBAIJLIBXSMM(PetscCtxRt ctx)
{
  SeqBAIJLIBXSMM_SeqDense *data = *(SeqBAIJLIBXSMM_SeqDense **)ctx;

  PetscFunctionBegin;
  PetscCall(PetscFree(data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatProductNumeric_SeqBAIJLIBXSMM_SeqDense(Mat C)
{
  Mat                      A = C->product->A, B = C->product->B;
  Mat_SeqBAIJ             *baij = (Mat_SeqBAIJ *)A->data;
  SeqBAIJLIBXSMM_SeqDense *data = (SeqBAIJLIBXSMM_SeqDense *)C->product->data;
  const PetscScalar       *b;
  PetscScalar             *c;
  const MatScalar         *values = baij->a;
  const PetscInt          *cols = baij->j, *rows = baij->i;
  PetscInt                 bs    = A->rmap->bs, ldb, ldc;
  libxsmm_gemm_param       param = {0};

  PetscFunctionBegin;
  MatCheckProduct(C, 1);
  PetscCheck(data && (data->kernel || !data->n), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Missing LIBXSMM product data");
  PetscCall(MatDenseGetLDA(B, &ldb));
  PetscCall(MatDenseGetLDA(C, &ldc));
  PetscCheck(ldb == data->ldb && ldc == data->ldc, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot reuse dense matrices with leading dimensions (%" PetscInt_FMT ",%" PetscInt_FMT ") instead of (%" PetscInt_FMT ",%" PetscInt_FMT ")", ldb, ldc, data->ldb,
             data->ldc);
  PetscCall(MatZeroEntries(C));
  if (!data->n) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(MatDenseGetArrayRead(B, &b));
  PetscCall(MatDenseGetArray(C, &c));
  for (PetscInt i = 0; i < baij->mbs; ++i) {
    for (PetscInt j = rows[i]; j < rows[i + 1]; ++j) {
      param.a.primary = (void *)(values + j * bs * bs);
      param.b.primary = (void *)(b + cols[j] * bs);
      param.c.primary = c + i * bs;
      PetscCallExternalVoid("LIBXSMM JIT kernel", data->kernel(&param));
    }
  }
  PetscCall(MatDenseRestoreArrayRead(B, &b));
  PetscCall(MatDenseRestoreArray(C, &c));
  PetscCall(PetscLogFlops(2.0 * baij->nz * bs * bs * data->n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatProductSymbolic_SeqBAIJLIBXSMM_SeqDense(Mat C)
{
  Mat                      A = C->product->A, B = C->product->B;
  SeqBAIJLIBXSMM_SeqDense *data;
  PetscInt                 m = A->rmap->n, n = B->cmap->n, bs = A->rmap->bs, ldb, ldc;
  PetscBLASInt             bn, bbs, bldb, bldc;
  libxsmm_gemm_shape       shape;
  libxsmm_datatype         datatype = PetscDefined(USE_REAL_SINGLE) ? LIBXSMM_DATATYPE_F32 : LIBXSMM_DATATYPE_F64;

  PetscFunctionBegin;
  MatCheckProduct(C, 1);
  PetscCheck(A->cmap->n == B->rmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Matrix local dimensions are incompatible, %" PetscInt_FMT " != %" PetscInt_FMT, A->cmap->n, B->rmap->n);
  PetscCall(MatSetSizes(C, m, n, m, n));
  PetscCall(MatSetBlockSizesFromMats(C, A, B));
  PetscCall(MatSetType(C, MATSEQDENSE));
  PetscCall(MatSetUp(C));
  PetscCall(MatDenseGetLDA(B, &ldb));
  PetscCall(MatDenseGetLDA(C, &ldc));
  PetscCheck((PetscInt64)PetscMax(ldb, ldc) * ((PetscInt64)n + 1) * sizeof(PetscScalar) <= PETSC_INT32_MAX, PETSC_COMM_SELF, PETSC_ERR_SUP, "LIBXSMM JIT cannot address dense matrices with leading dimensions (%" PetscInt_FMT ",%" PetscInt_FMT ") and %" PetscInt_FMT " columns", ldb, ldc, n);
  PetscCall(PetscBLASIntCast(n, &bn));
  PetscCall(PetscBLASIntCast(bs, &bbs));
  PetscCall(PetscBLASIntCast(ldb, &bldb));
  PetscCall(PetscBLASIntCast(ldc, &bldc));
  PetscCall(PetscNew(&data));
  data->n   = n;
  data->ldb = ldb;
  data->ldc = ldc;
  if (n) {
    PetscCallExternalVoid("libxsmm_create_gemm_shape", shape = libxsmm_create_gemm_shape(bbs, bn, bbs, bbs, bldb, bldc, datatype, datatype, datatype, datatype));
    PetscCallExternalVoid("libxsmm_dispatch_gemm", data->kernel = libxsmm_dispatch_gemm(shape, LIBXSMM_GEMM_FLAG_NONE, LIBXSMM_GEMM_PREFETCH_NONE));
    PetscCheck(data->kernel, PETSC_COMM_SELF, PETSC_ERR_SUP, "LIBXSMM cannot generate a kernel for block size %" PetscInt_FMT " and %" PetscInt_FMT " dense columns", bs, n);
  }
  C->product->data       = data;
  C->product->destroy    = MatProductDataDestroy_SeqBAIJLIBXSMM;
  C->ops->productnumeric = MatProductNumeric_SeqBAIJLIBXSMM_SeqDense;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatProductSetFromOptions_SeqBAIJLIBXSMM_SeqDense(Mat C)
{
  PetscFunctionBegin;
  MatCheckProduct(C, 1);
  if (C->product->type == MATPRODUCT_AB) C->ops->productsymbolic = MatProductSymbolic_SeqBAIJLIBXSMM_SeqDense;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatConvert_SeqBAIJLIBXSMM_SeqBAIJ(Mat A, MatType type, MatReuse reuse, Mat *newmat)
{
  Mat B = *newmat;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &B));
  B->ops->duplicate = MatDuplicate_SeqBAIJ;
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatProductSetFromOptions_seqbaijlibxsmm_seqdense_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatConvert_seqbaijlibxsmm_seqbaij_C", NULL));
  PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATSEQBAIJ));
  *newmat = B;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatConvert_SeqBAIJ_SeqBAIJLIBXSMM(Mat A, MatType type, MatReuse reuse, Mat *newmat)
{
  Mat       B = *newmat;
  PetscBool sametype;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &B));
  PetscCall(PetscObjectTypeCompare((PetscObject)B, MATSEQBAIJLIBXSMM, &sametype));
  if (!sametype) {
    B->ops->duplicate = MatDuplicate_SeqBAIJLIBXSMM;
    PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATSEQBAIJLIBXSMM));
    PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatProductSetFromOptions_seqbaijlibxsmm_seqdense_C", MatProductSetFromOptions_SeqBAIJLIBXSMM_SeqDense));
    PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatConvert_seqbaijlibxsmm_seqbaij_C", MatConvert_SeqBAIJLIBXSMM_SeqBAIJ));
  }
  *newmat = B;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   MATSEQBAIJLIBXSMM - "seqbaijlibxsmm" - A sequential block sparse matrix that uses LIBXSMM kernels for products with `MATSEQDENSE` matrices

   Options Database Key:
. -mat_type seqbaijlibxsmm - sets the matrix type to `MATSEQBAIJLIBXSMM` during a call to `MatSetFromOptions()`

   Level: beginner

   Notes:
   This matrix type is available when PETSc is configured with `--download-libxsmm` or `--with-libxsmm-dir=directory`.

   It has the same storage format and supports the same operations as `MATSEQBAIJ`.

.seealso: [](ch_matrices), `Mat`, `MATBAIJLIBXSMM`, `MATMPIBAIJLIBXSMM`, `MATSEQBAIJ`, `MatMatMult()`
M*/
PETSC_EXTERN PetscErrorCode MatCreate_SeqBAIJLIBXSMM(Mat A)
{
  PetscFunctionBegin;
  PetscCall(MatSetType(A, MATSEQBAIJ));
  PetscCall(MatConvert_SeqBAIJ_SeqBAIJLIBXSMM(A, MATSEQBAIJLIBXSMM, MAT_INPLACE_MATRIX, &A));
  PetscFunctionReturn(PETSC_SUCCESS);
}
