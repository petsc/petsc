
#include <../src/mat/impls/maij/maij.h> /*I "petscmat.h" I*/
#include <../src/mat/utils/freespace.h>

/*@
   MatMAIJGetAIJ - Get the `MATAIJ` matrix describing the blockwise action of the `MATMAIJ` matrix

   Not Collective, but if the `MATMAIJ` matrix is parallel, the `MATAIJ` matrix is also parallel

   Input Parameter:
.  A - the `MATMAIJ` matrix

   Output Parameter:
.  B - the `MATAIJ` matrix

   Level: advanced

   Note:
    The reference count on the `MATAIJ` matrix is not increased so you should not destroy it.

.seealso: [](chapter_matrices), `Mat`, `MATMAIJ`, `MATAIJ`, `MatCreateMAIJ()`
@*/
PetscErrorCode MatMAIJGetAIJ(Mat A, Mat *B)
{
  PetscBool ismpimaij, isseqmaij;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATMPIMAIJ, &ismpimaij));
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATSEQMAIJ, &isseqmaij));
  if (ismpimaij) {
    Mat_MPIMAIJ *b = (Mat_MPIMAIJ *)A->data;

    *B = b->A;
  } else if (isseqmaij) {
    Mat_SeqMAIJ *b = (Mat_SeqMAIJ *)A->data;

    *B = b->AIJ;
  } else {
    *B = A;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   MatMAIJRedimension - Get a new `MATMAIJ` matrix with the same action, but for a different block size

   Logically Collective

   Input Parameters:
+  A - the `MATMAIJ` matrix
-  dof - the block size for the new matrix

   Output Parameter:
.  B - the new `MATMAIJ` matrix

   Level: advanced

.seealso: [](chapter_matrices), `Mat`, `MATMAIJ`, `MatCreateMAIJ()`
@*/
PetscErrorCode MatMAIJRedimension(Mat A, PetscInt dof, Mat *B)
{
  Mat Aij = NULL;

  PetscFunctionBegin;
  PetscValidLogicalCollectiveInt(A, dof, 2);
  PetscCall(MatMAIJGetAIJ(A, &Aij));
  PetscCall(MatCreateMAIJ(Aij, dof, B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDestroy_SeqMAIJ(Mat A)
{
  Mat_SeqMAIJ *b = (Mat_SeqMAIJ *)A->data;

  PetscFunctionBegin;
  PetscCall(MatDestroy(&b->AIJ));
  PetscCall(PetscFree(A->data));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatConvert_seqmaij_seqaijcusparse_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatConvert_seqmaij_seqaij_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatProductSetFromOptions_seqaij_seqmaij_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetUp_MAIJ(Mat A)
{
  PetscFunctionBegin;
  SETERRQ(PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "Must use MatCreateMAIJ() to create MAIJ matrices");
}

static PetscErrorCode MatView_SeqMAIJ(Mat A, PetscViewer viewer)
{
  Mat B;

  PetscFunctionBegin;
  PetscCall(MatConvert(A, MATSEQAIJ, MAT_INITIAL_MATRIX, &B));
  PetscCall(MatView(B, viewer));
  PetscCall(MatDestroy(&B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatView_MPIMAIJ(Mat A, PetscViewer viewer)
{
  Mat B;

  PetscFunctionBegin;
  PetscCall(MatConvert(A, MATMPIAIJ, MAT_INITIAL_MATRIX, &B));
  PetscCall(MatView(B, viewer));
  PetscCall(MatDestroy(&B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDestroy_MPIMAIJ(Mat A)
{
  Mat_MPIMAIJ *b = (Mat_MPIMAIJ *)A->data;

  PetscFunctionBegin;
  PetscCall(MatDestroy(&b->AIJ));
  PetscCall(MatDestroy(&b->OAIJ));
  PetscCall(MatDestroy(&b->A));
  PetscCall(VecScatterDestroy(&b->ctx));
  PetscCall(VecDestroy(&b->w));
  PetscCall(PetscFree(A->data));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatConvert_mpimaij_mpiaijcusparse_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatConvert_mpimaij_mpiaij_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatProductSetFromOptions_mpiaij_mpimaij_C", NULL));
  PetscCall(PetscObjectChangeTypeName((PetscObject)A, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  MATMAIJ - MATMAIJ = "maij" - A matrix type to be used for restriction and interpolation operations for
  multicomponent problems, interpolating or restricting each component the same way independently.
  The matrix type is based on `MATSEQAIJ` for sequential matrices, and `MATMPIAIJ` for distributed matrices.

  Operations provided:
.vb
    MatMult()
    MatMultTranspose()
    MatMultAdd()
    MatMultTransposeAdd()
.ve

  Level: advanced

.seealso: [](chapter_matrices), `Mat`, `MATAIJ`, `MatMAIJGetAIJ()`, `MatMAIJRedimension()`, `MatCreateMAIJ()`
M*/

PETSC_EXTERN PetscErrorCode MatCreate_MAIJ(Mat A)
{
  Mat_MPIMAIJ *b;
  PetscMPIInt  size;

  PetscFunctionBegin;
  PetscCall(PetscNew(&b));
  A->data = (void *)b;

  PetscCall(PetscMemzero(A->ops, sizeof(struct _MatOps)));

  A->ops->setup = MatSetUp_MAIJ;

  b->AIJ  = NULL;
  b->dof  = 0;
  b->OAIJ = NULL;
  b->ctx  = NULL;
  b->w    = NULL;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)A), &size));
  if (size == 1) {
    PetscCall(PetscObjectChangeTypeName((PetscObject)A, MATSEQMAIJ));
  } else {
    PetscCall(PetscObjectChangeTypeName((PetscObject)A, MATMPIMAIJ));
  }
  A->preallocated = PETSC_TRUE;
  A->assembled    = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if PetscHasAttribute(always_inline)
  #define PETSC_FORCE_INLINE __attribute__((always_inline))
#else
  #define PETSC_FORCE_INLINE
#endif

#if defined(__clang__)
  #define PETSC_PRAGMA_UNROLL _Pragma("unroll")
#else
  #define PETSC_PRAGMA_UNROLL
#endif

enum {
  MAT_SEQMAIJ_MAX_TEMPLATE_SIZE = 18
};

// try as hard as possible to get these "template"s inlined, GCC apparently does take 'inline'
// keyword into account for these...
PETSC_FORCE_INLINE static inline PetscErrorCode MatMult_MatMultAdd_SeqMAIJ_Template(Mat A, Vec xx, Vec yy, Vec zz, int N)
{
  const PetscBool    mult_add   = yy == NULL ? PETSC_FALSE : PETSC_TRUE;
  const Mat_SeqMAIJ *b          = (Mat_SeqMAIJ *)A->data;
  const Mat          baij       = b->AIJ;
  const Mat_SeqAIJ  *a          = (Mat_SeqAIJ *)baij->data;
  const PetscInt     m          = baij->rmap->n;
  const PetscInt     nz         = a->nz;
  const PetscInt    *idx        = a->j;
  const PetscInt    *ii         = a->i;
  const PetscScalar *v          = a->a;
  PetscInt           nonzerorow = 0;
  const PetscScalar *x;
  PetscScalar       *z;

  PetscFunctionBegin;
  PetscAssert(N <= MAT_SEQMAIJ_MAX_TEMPLATE_SIZE, PETSC_COMM_SELF, PETSC_ERR_PLIB, "%s() called with N = %d > max size %d", PETSC_FUNCTION_NAME, N, MAT_SEQMAIJ_MAX_TEMPLATE_SIZE);
  if (mult_add && yy != zz) PetscCall(VecCopy(yy, zz));
  PetscCall(VecGetArrayRead(xx, &x));
  if (mult_add) {
    PetscCall(VecGetArray(zz, &z));
  } else {
    PetscCall(VecGetArrayWrite(zz, &z));
  }

  for (PetscInt i = 0; i < m; ++i) {
    PetscInt       jrow = ii[i];
    const PetscInt n    = ii[i + 1] - jrow;
    // leave a line so clang-format does not align these decls
    PetscScalar sum[MAT_SEQMAIJ_MAX_TEMPLATE_SIZE] = {0};

    nonzerorow += n > 0;
    for (PetscInt j = 0; j < n; ++j, ++jrow) {
      const PetscScalar v_jrow     = v[jrow];
      const PetscInt    N_idx_jrow = N * idx[jrow];

      PETSC_PRAGMA_UNROLL
      for (int k = 0; k < N; ++k) sum[k] += v_jrow * x[N_idx_jrow + k];
    }

    PETSC_PRAGMA_UNROLL
    for (int k = 0; k < N; ++k) {
      const PetscInt z_idx = N * i + k;

      if (mult_add) {
        z[z_idx] += sum[k];
      } else {
        z[z_idx] = sum[k];
      }
    }
  }
  PetscCall(PetscLogFlops(2 * N * nz - (mult_add ? 0 : (N * nonzerorow))));
  PetscCall(VecRestoreArrayRead(xx, &x));
  if (mult_add) {
    PetscCall(VecRestoreArray(zz, &z));
  } else {
    PetscCall(VecRestoreArrayWrite(zz, &z));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_FORCE_INLINE static inline PetscErrorCode MatMultTranspose_MatMultTransposeAdd_SeqMAIJ_Template(Mat A, Vec xx, Vec yy, Vec zz, int N)
{
  const PetscBool    mult_add = yy == NULL ? PETSC_FALSE : PETSC_TRUE;
  const Mat_SeqMAIJ *b        = (Mat_SeqMAIJ *)A->data;
  const Mat          baij     = b->AIJ;
  const Mat_SeqAIJ  *a        = (Mat_SeqAIJ *)baij->data;
  const PetscInt     m        = baij->rmap->n;
  const PetscInt     nz       = a->nz;
  const PetscInt    *a_j      = a->j;
  const PetscInt    *a_i      = a->i;
  const PetscScalar *a_a      = a->a;
  const PetscScalar *x;
  PetscScalar       *z;

  PetscFunctionBegin;
  PetscAssert(N <= MAT_SEQMAIJ_MAX_TEMPLATE_SIZE, PETSC_COMM_SELF, PETSC_ERR_PLIB, "%s() called with N = %d > max size %d", PETSC_FUNCTION_NAME, N, MAT_SEQMAIJ_MAX_TEMPLATE_SIZE);
  if (mult_add) {
    if (yy != zz) PetscCall(VecCopy(yy, zz));
  } else {
    PetscCall(VecSet(zz, 0.0));
  }
  PetscCall(VecGetArrayRead(xx, &x));
  PetscCall(VecGetArray(zz, &z));

  for (PetscInt i = 0; i < m; i++) {
    const PetscInt     a_ii = a_i[i];
    const PetscInt    *idx  = a_j + a_ii;
    const PetscScalar *v    = a_a + a_ii;
    const PetscInt     n    = a_i[i + 1] - a_ii;
    PetscScalar        alpha[MAT_SEQMAIJ_MAX_TEMPLATE_SIZE];

    PETSC_PRAGMA_UNROLL
    for (int k = 0; k < N; ++k) alpha[k] = x[N * i + k];
    for (PetscInt j = 0; j < n; ++j) {
      const PetscInt    N_idx_j = N * idx[j];
      const PetscScalar v_j     = v[j];

      PETSC_PRAGMA_UNROLL
      for (int k = 0; k < N; ++k) z[N_idx_j + k] += alpha[k] * v_j;
    }
  }

  PetscCall(PetscLogFlops(2 * N * nz));
  PetscCall(VecRestoreArrayRead(xx, &x));
  PetscCall(VecRestoreArray(zz, &z));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#define MAT_SEQ_MAIJ_INSTANTIATE_MATMULT_MATMULTADD_TEMPLATE(N) \
  static PetscErrorCode PetscConcat(MatMult_SeqMAIJ_, N)(Mat A, Vec xx, Vec yy) \
  { \
    PetscFunctionBegin; \
    PetscCall(MatMult_MatMultAdd_SeqMAIJ_Template(A, xx, NULL, yy, N)); \
    PetscFunctionReturn(PETSC_SUCCESS); \
  } \
  static PetscErrorCode PetscConcat(MatMultTranspose_SeqMAIJ_, N)(Mat A, Vec xx, Vec yy) \
  { \
    PetscFunctionBegin; \
    PetscCall(MatMultTranspose_MatMultTransposeAdd_SeqMAIJ_Template(A, xx, NULL, yy, N)); \
    PetscFunctionReturn(PETSC_SUCCESS); \
  } \
  static PetscErrorCode PetscConcat(MatMultAdd_SeqMAIJ_, N)(Mat A, Vec xx, Vec yy, Vec zz) \
  { \
    PetscFunctionBegin; \
    PetscCall(MatMult_MatMultAdd_SeqMAIJ_Template(A, xx, yy, zz, N)); \
    PetscFunctionReturn(PETSC_SUCCESS); \
  } \
  static PetscErrorCode PetscConcat(MatMultTransposeAdd_SeqMAIJ_, N)(Mat A, Vec xx, Vec yy, Vec zz) \
  { \
    PetscFunctionBegin; \
    PetscCall(MatMultTranspose_MatMultTransposeAdd_SeqMAIJ_Template(A, xx, yy, zz, N)); \
    PetscFunctionReturn(PETSC_SUCCESS); \
  }

// clang-format off
MAT_SEQ_MAIJ_INSTANTIATE_MATMULT_MATMULTADD_TEMPLATE(2)
MAT_SEQ_MAIJ_INSTANTIATE_MATMULT_MATMULTADD_TEMPLATE(3)
MAT_SEQ_MAIJ_INSTANTIATE_MATMULT_MATMULTADD_TEMPLATE(4)
MAT_SEQ_MAIJ_INSTANTIATE_MATMULT_MATMULTADD_TEMPLATE(5)
MAT_SEQ_MAIJ_INSTANTIATE_MATMULT_MATMULTADD_TEMPLATE(6)
MAT_SEQ_MAIJ_INSTANTIATE_MATMULT_MATMULTADD_TEMPLATE(7)
MAT_SEQ_MAIJ_INSTANTIATE_MATMULT_MATMULTADD_TEMPLATE(8)
MAT_SEQ_MAIJ_INSTANTIATE_MATMULT_MATMULTADD_TEMPLATE(9)
MAT_SEQ_MAIJ_INSTANTIATE_MATMULT_MATMULTADD_TEMPLATE(10)
MAT_SEQ_MAIJ_INSTANTIATE_MATMULT_MATMULTADD_TEMPLATE(11)
MAT_SEQ_MAIJ_INSTANTIATE_MATMULT_MATMULTADD_TEMPLATE(16)
MAT_SEQ_MAIJ_INSTANTIATE_MATMULT_MATMULTADD_TEMPLATE(18)
// clang-format on

#undef MAT_SEQ_MAIJ_INSTANTIATE_MATMULT_MATMULTADD_TEMPLATE

static PetscErrorCode MatMult_SeqMAIJ_N(Mat A, Vec xx, Vec yy)
{
  Mat_SeqMAIJ       *b = (Mat_SeqMAIJ *)A->data;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ *)b->AIJ->data;
  const PetscScalar *x, *v;
  PetscScalar       *y, *sums;
  const PetscInt     m = b->AIJ->rmap->n, *idx, *ii;
  PetscInt           n, i, jrow, j, dof = b->dof, k;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(xx, &x));
  PetscCall(VecSet(yy, 0.0));
  PetscCall(VecGetArray(yy, &y));
  idx = a->j;
  v   = a->a;
  ii  = a->i;

  for (i = 0; i < m; i++) {
    jrow = ii[i];
    n    = ii[i + 1] - jrow;
    sums = y + dof * i;
    for (j = 0; j < n; j++) {
      for (k = 0; k < dof; k++) sums[k] += v[jrow] * x[dof * idx[jrow] + k];
      jrow++;
    }
  }

  PetscCall(PetscLogFlops(2.0 * dof * a->nz));
  PetscCall(VecRestoreArrayRead(xx, &x));
  PetscCall(VecRestoreArray(yy, &y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMultAdd_SeqMAIJ_N(Mat A, Vec xx, Vec yy, Vec zz)
{
  Mat_SeqMAIJ       *b = (Mat_SeqMAIJ *)A->data;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ *)b->AIJ->data;
  const PetscScalar *x, *v;
  PetscScalar       *y, *sums;
  const PetscInt     m = b->AIJ->rmap->n, *idx, *ii;
  PetscInt           n, i, jrow, j, dof = b->dof, k;

  PetscFunctionBegin;
  if (yy != zz) PetscCall(VecCopy(yy, zz));
  PetscCall(VecGetArrayRead(xx, &x));
  PetscCall(VecGetArray(zz, &y));
  idx = a->j;
  v   = a->a;
  ii  = a->i;

  for (i = 0; i < m; i++) {
    jrow = ii[i];
    n    = ii[i + 1] - jrow;
    sums = y + dof * i;
    for (j = 0; j < n; j++) {
      for (k = 0; k < dof; k++) sums[k] += v[jrow] * x[dof * idx[jrow] + k];
      jrow++;
    }
  }

  PetscCall(PetscLogFlops(2.0 * dof * a->nz));
  PetscCall(VecRestoreArrayRead(xx, &x));
  PetscCall(VecRestoreArray(zz, &y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMultTranspose_SeqMAIJ_N(Mat A, Vec xx, Vec yy)
{
  Mat_SeqMAIJ       *b = (Mat_SeqMAIJ *)A->data;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ *)b->AIJ->data;
  const PetscScalar *x, *v, *alpha;
  PetscScalar       *y;
  const PetscInt     m = b->AIJ->rmap->n, *idx, dof = b->dof;
  PetscInt           n, i, k;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(xx, &x));
  PetscCall(VecSet(yy, 0.0));
  PetscCall(VecGetArray(yy, &y));
  for (i = 0; i < m; i++) {
    idx   = a->j + a->i[i];
    v     = a->a + a->i[i];
    n     = a->i[i + 1] - a->i[i];
    alpha = x + dof * i;
    while (n-- > 0) {
      for (k = 0; k < dof; k++) y[dof * (*idx) + k] += alpha[k] * (*v);
      idx++;
      v++;
    }
  }
  PetscCall(PetscLogFlops(2.0 * dof * a->nz));
  PetscCall(VecRestoreArrayRead(xx, &x));
  PetscCall(VecRestoreArray(yy, &y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMultTransposeAdd_SeqMAIJ_N(Mat A, Vec xx, Vec yy, Vec zz)
{
  Mat_SeqMAIJ       *b = (Mat_SeqMAIJ *)A->data;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ *)b->AIJ->data;
  const PetscScalar *x, *v, *alpha;
  PetscScalar       *y;
  const PetscInt     m = b->AIJ->rmap->n, *idx, dof = b->dof;
  PetscInt           n, i, k;

  PetscFunctionBegin;
  if (yy != zz) PetscCall(VecCopy(yy, zz));
  PetscCall(VecGetArrayRead(xx, &x));
  PetscCall(VecGetArray(zz, &y));
  for (i = 0; i < m; i++) {
    idx   = a->j + a->i[i];
    v     = a->a + a->i[i];
    n     = a->i[i + 1] - a->i[i];
    alpha = x + dof * i;
    while (n-- > 0) {
      for (k = 0; k < dof; k++) y[dof * (*idx) + k] += alpha[k] * (*v);
      idx++;
      v++;
    }
  }
  PetscCall(PetscLogFlops(2.0 * dof * a->nz));
  PetscCall(VecRestoreArrayRead(xx, &x));
  PetscCall(VecRestoreArray(zz, &y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMult_MPIMAIJ_dof(Mat A, Vec xx, Vec yy)
{
  Mat_MPIMAIJ *b = (Mat_MPIMAIJ *)A->data;

  PetscFunctionBegin;
  /* start the scatter */
  PetscCall(VecScatterBegin(b->ctx, xx, b->w, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall((*b->AIJ->ops->mult)(b->AIJ, xx, yy));
  PetscCall(VecScatterEnd(b->ctx, xx, b->w, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall((*b->OAIJ->ops->multadd)(b->OAIJ, b->w, yy, yy));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMultTranspose_MPIMAIJ_dof(Mat A, Vec xx, Vec yy)
{
  Mat_MPIMAIJ *b = (Mat_MPIMAIJ *)A->data;

  PetscFunctionBegin;
  PetscCall((*b->OAIJ->ops->multtranspose)(b->OAIJ, xx, b->w));
  PetscCall((*b->AIJ->ops->multtranspose)(b->AIJ, xx, yy));
  PetscCall(VecScatterBegin(b->ctx, b->w, yy, ADD_VALUES, SCATTER_REVERSE));
  PetscCall(VecScatterEnd(b->ctx, b->w, yy, ADD_VALUES, SCATTER_REVERSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMultAdd_MPIMAIJ_dof(Mat A, Vec xx, Vec yy, Vec zz)
{
  Mat_MPIMAIJ *b = (Mat_MPIMAIJ *)A->data;

  PetscFunctionBegin;
  /* start the scatter */
  PetscCall(VecScatterBegin(b->ctx, xx, b->w, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall((*b->AIJ->ops->multadd)(b->AIJ, xx, yy, zz));
  PetscCall(VecScatterEnd(b->ctx, xx, b->w, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall((*b->OAIJ->ops->multadd)(b->OAIJ, b->w, zz, zz));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMultTransposeAdd_MPIMAIJ_dof(Mat A, Vec xx, Vec yy, Vec zz)
{
  Mat_MPIMAIJ *b = (Mat_MPIMAIJ *)A->data;

  PetscFunctionBegin;
  PetscCall((*b->OAIJ->ops->multtranspose)(b->OAIJ, xx, b->w));
  PetscCall((*b->AIJ->ops->multtransposeadd)(b->AIJ, xx, yy, zz));
  PetscCall(VecScatterBegin(b->ctx, b->w, zz, ADD_VALUES, SCATTER_REVERSE));
  PetscCall(VecScatterEnd(b->ctx, b->w, zz, ADD_VALUES, SCATTER_REVERSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatProductSetFromOptions_SeqAIJ_SeqMAIJ(Mat C)
{
  Mat_Product *product = C->product;

  PetscFunctionBegin;
  if (product->type == MATPRODUCT_PtAP) {
    C->ops->productsymbolic = MatProductSymbolic_PtAP_SeqAIJ_SeqMAIJ;
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Mat Product type %s is not supported for SeqAIJ and SeqMAIJ matrices", MatProductTypes[product->type]);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatProductSetFromOptions_MPIAIJ_MPIMAIJ(Mat C)
{
  Mat_Product *product = C->product;
  PetscBool    flg     = PETSC_FALSE;
  Mat          A = product->A, P = product->B;
  PetscInt     alg = 1; /* set default algorithm */
#if !defined(PETSC_HAVE_HYPRE)
  const char *algTypes[4] = {"scalable", "nonscalable", "allatonce", "allatonce_merged"};
  PetscInt    nalg        = 4;
#else
  const char *algTypes[5] = {"scalable", "nonscalable", "allatonce", "allatonce_merged", "hypre"};
  PetscInt    nalg        = 5;
#endif

  PetscFunctionBegin;
  PetscCheck(product->type == MATPRODUCT_PtAP, PETSC_COMM_SELF, PETSC_ERR_SUP, "Mat Product type %s is not supported for MPIAIJ and MPIMAIJ matrices", MatProductTypes[product->type]);

  /* PtAP */
  /* Check matrix local sizes */
  PetscCheck(A->rmap->rstart == P->rmap->rstart && A->rmap->rend == P->rmap->rend, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Matrix local dimensions are incompatible, Arow (%" PetscInt_FMT ", %" PetscInt_FMT ") != Prow (%" PetscInt_FMT ",%" PetscInt_FMT ")",
             A->rmap->rstart, A->rmap->rend, P->rmap->rstart, P->rmap->rend);
  PetscCheck(A->cmap->rstart == P->rmap->rstart && A->cmap->rend == P->rmap->rend, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Matrix local dimensions are incompatible, Acol (%" PetscInt_FMT ", %" PetscInt_FMT ") != Prow (%" PetscInt_FMT ",%" PetscInt_FMT ")",
             A->cmap->rstart, A->cmap->rend, P->rmap->rstart, P->rmap->rend);

  /* Set the default algorithm */
  PetscCall(PetscStrcmp(C->product->alg, "default", &flg));
  if (flg) PetscCall(MatProductSetAlgorithm(C, (MatProductAlgorithm)algTypes[alg]));

  /* Get runtime option */
  PetscOptionsBegin(PetscObjectComm((PetscObject)C), ((PetscObject)C)->prefix, "MatProduct_PtAP", "Mat");
  PetscCall(PetscOptionsEList("-mat_product_algorithm", "Algorithmic approach", "MatPtAP", algTypes, nalg, algTypes[alg], &alg, &flg));
  if (flg) PetscCall(MatProductSetAlgorithm(C, (MatProductAlgorithm)algTypes[alg]));
  PetscOptionsEnd();

  PetscCall(PetscStrcmp(C->product->alg, "allatonce", &flg));
  if (flg) {
    C->ops->productsymbolic = MatProductSymbolic_PtAP_MPIAIJ_MPIMAIJ;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(PetscStrcmp(C->product->alg, "allatonce_merged", &flg));
  if (flg) {
    C->ops->productsymbolic = MatProductSymbolic_PtAP_MPIAIJ_MPIMAIJ;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* Convert P from MAIJ to AIJ matrix since implementation not available for MAIJ */
  PetscCall(PetscInfo((PetscObject)A, "Converting from MAIJ to AIJ matrix since implementation not available for MAIJ\n"));
  PetscCall(MatConvert(P, MATMPIAIJ, MAT_INPLACE_MATRIX, &P));
  PetscCall(MatProductSetFromOptions(C));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatPtAPNumeric_SeqAIJ_SeqMAIJ(Mat A, Mat PP, Mat C)
{
  /* This routine requires testing -- first draft only */
  Mat_SeqMAIJ     *pp = (Mat_SeqMAIJ *)PP->data;
  Mat              P  = pp->AIJ;
  Mat_SeqAIJ      *a  = (Mat_SeqAIJ *)A->data;
  Mat_SeqAIJ      *p  = (Mat_SeqAIJ *)P->data;
  Mat_SeqAIJ      *c  = (Mat_SeqAIJ *)C->data;
  const PetscInt  *ai = a->i, *aj = a->j, *pi = p->i, *pj = p->j, *pJ, *pjj;
  const PetscInt  *ci = c->i, *cj = c->j, *cjj;
  const PetscInt   am = A->rmap->N, cn = C->cmap->N, cm = C->rmap->N, ppdof = pp->dof;
  PetscInt         i, j, k, pshift, poffset, anzi, pnzi, apnzj, nextap, pnzj, prow, crow, *apj, *apjdense;
  const MatScalar *aa = a->a, *pa = p->a, *pA, *paj;
  MatScalar       *ca = c->a, *caj, *apa;

  PetscFunctionBegin;
  /* Allocate temporary array for storage of one row of A*P */
  PetscCall(PetscCalloc3(cn, &apa, cn, &apj, cn, &apjdense));

  /* Clear old values in C */
  PetscCall(PetscArrayzero(ca, ci[cm]));

  for (i = 0; i < am; i++) {
    /* Form sparse row of A*P */
    anzi  = ai[i + 1] - ai[i];
    apnzj = 0;
    for (j = 0; j < anzi; j++) {
      /* Get offset within block of P */
      pshift = *aj % ppdof;
      /* Get block row of P */
      prow = *aj++ / ppdof; /* integer division */
      pnzj = pi[prow + 1] - pi[prow];
      pjj  = pj + pi[prow];
      paj  = pa + pi[prow];
      for (k = 0; k < pnzj; k++) {
        poffset = pjj[k] * ppdof + pshift;
        if (!apjdense[poffset]) {
          apjdense[poffset] = -1;
          apj[apnzj++]      = poffset;
        }
        apa[poffset] += (*aa) * paj[k];
      }
      PetscCall(PetscLogFlops(2.0 * pnzj));
      aa++;
    }

    /* Sort the j index array for quick sparse axpy. */
    /* Note: a array does not need sorting as it is in dense storage locations. */
    PetscCall(PetscSortInt(apnzj, apj));

    /* Compute P^T*A*P using outer product (P^T)[:,j]*(A*P)[j,:]. */
    prow    = i / ppdof; /* integer division */
    pshift  = i % ppdof;
    poffset = pi[prow];
    pnzi    = pi[prow + 1] - poffset;
    /* Reset pJ and pA so we can traverse the same row of P 'dof' times. */
    pJ = pj + poffset;
    pA = pa + poffset;
    for (j = 0; j < pnzi; j++) {
      crow = (*pJ) * ppdof + pshift;
      cjj  = cj + ci[crow];
      caj  = ca + ci[crow];
      pJ++;
      /* Perform sparse axpy operation.  Note cjj includes apj. */
      for (k = 0, nextap = 0; nextap < apnzj; k++) {
        if (cjj[k] == apj[nextap]) caj[k] += (*pA) * apa[apj[nextap++]];
      }
      PetscCall(PetscLogFlops(2.0 * apnzj));
      pA++;
    }

    /* Zero the current row info for A*P */
    for (j = 0; j < apnzj; j++) {
      apa[apj[j]]      = 0.;
      apjdense[apj[j]] = 0;
    }
  }

  /* Assemble the final matrix and clean up */
  PetscCall(MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY));
  PetscCall(PetscFree3(apa, apj, apjdense));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatPtAPSymbolic_SeqAIJ_SeqMAIJ(Mat A, Mat PP, PetscReal fill, Mat C)
{
  PetscFreeSpaceList free_space = NULL, current_space = NULL;
  Mat_SeqMAIJ       *pp = (Mat_SeqMAIJ *)PP->data;
  Mat                P  = pp->AIJ;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ *)A->data, *p = (Mat_SeqAIJ *)P->data, *c;
  PetscInt          *pti, *ptj, *ptJ;
  PetscInt          *ci, *cj, *ptadenserow, *ptasparserow, *denserow, *sparserow, *ptaj;
  const PetscInt     an = A->cmap->N, am = A->rmap->N, pn = P->cmap->N, pm = P->rmap->N, ppdof = pp->dof;
  PetscInt           i, j, k, dof, pshift, ptnzi, arow, anzj, ptanzi, prow, pnzj, cnzi, cn;
  MatScalar         *ca;
  const PetscInt    *pi = p->i, *pj = p->j, *pjj, *ai = a->i, *aj = a->j, *ajj;

  PetscFunctionBegin;
  /* Get ij structure of P^T */
  PetscCall(MatGetSymbolicTranspose_SeqAIJ(P, &pti, &ptj));

  cn = pn * ppdof;
  /* Allocate ci array, arrays for fill computation and */
  /* free space for accumulating nonzero column info */
  PetscCall(PetscMalloc1(cn + 1, &ci));
  ci[0] = 0;

  /* Work arrays for rows of P^T*A */
  PetscCall(PetscMalloc4(an, &ptadenserow, an, &ptasparserow, cn, &denserow, cn, &sparserow));
  PetscCall(PetscArrayzero(ptadenserow, an));
  PetscCall(PetscArrayzero(denserow, cn));

  /* Set initial free space to be nnz(A) scaled by aspect ratio of P. */
  /* This should be reasonable if sparsity of PtAP is similar to that of A. */
  /* Note, aspect ratio of P is the same as the aspect ratio of SeqAIJ inside P */
  PetscCall(PetscFreeSpaceGet(PetscIntMultTruncate(ai[am] / pm, pn), &free_space));
  current_space = free_space;

  /* Determine symbolic info for each row of C: */
  for (i = 0; i < pn; i++) {
    ptnzi = pti[i + 1] - pti[i];
    ptJ   = ptj + pti[i];
    for (dof = 0; dof < ppdof; dof++) {
      ptanzi = 0;
      /* Determine symbolic row of PtA: */
      for (j = 0; j < ptnzi; j++) {
        /* Expand ptJ[j] by block size and shift by dof to get the right row of A */
        arow = ptJ[j] * ppdof + dof;
        /* Nonzeros of P^T*A will be in same locations as any element of A in that row */
        anzj = ai[arow + 1] - ai[arow];
        ajj  = aj + ai[arow];
        for (k = 0; k < anzj; k++) {
          if (!ptadenserow[ajj[k]]) {
            ptadenserow[ajj[k]]    = -1;
            ptasparserow[ptanzi++] = ajj[k];
          }
        }
      }
      /* Using symbolic info for row of PtA, determine symbolic info for row of C: */
      ptaj = ptasparserow;
      cnzi = 0;
      for (j = 0; j < ptanzi; j++) {
        /* Get offset within block of P */
        pshift = *ptaj % ppdof;
        /* Get block row of P */
        prow = (*ptaj++) / ppdof; /* integer division */
        /* P has same number of nonzeros per row as the compressed form */
        pnzj = pi[prow + 1] - pi[prow];
        pjj  = pj + pi[prow];
        for (k = 0; k < pnzj; k++) {
          /* Locations in C are shifted by the offset within the block */
          /* Note: we cannot use PetscLLAdd here because of the additional offset for the write location */
          if (!denserow[pjj[k] * ppdof + pshift]) {
            denserow[pjj[k] * ppdof + pshift] = -1;
            sparserow[cnzi++]                 = pjj[k] * ppdof + pshift;
          }
        }
      }

      /* sort sparserow */
      PetscCall(PetscSortInt(cnzi, sparserow));

      /* If free space is not available, make more free space */
      /* Double the amount of total space in the list */
      if (current_space->local_remaining < cnzi) PetscCall(PetscFreeSpaceGet(PetscIntSumTruncate(cnzi, current_space->total_array_size), &current_space));

      /* Copy data into free space, and zero out denserows */
      PetscCall(PetscArraycpy(current_space->array, sparserow, cnzi));

      current_space->array += cnzi;
      current_space->local_used += cnzi;
      current_space->local_remaining -= cnzi;

      for (j = 0; j < ptanzi; j++) ptadenserow[ptasparserow[j]] = 0;
      for (j = 0; j < cnzi; j++) denserow[sparserow[j]] = 0;

      /* Aside: Perhaps we should save the pta info for the numerical factorization. */
      /*        For now, we will recompute what is needed. */
      ci[i * ppdof + 1 + dof] = ci[i * ppdof + dof] + cnzi;
    }
  }
  /* nnz is now stored in ci[ptm], column indices are in the list of free space */
  /* Allocate space for cj, initialize cj, and */
  /* destroy list of free space and other temporary array(s) */
  PetscCall(PetscMalloc1(ci[cn] + 1, &cj));
  PetscCall(PetscFreeSpaceContiguous(&free_space, cj));
  PetscCall(PetscFree4(ptadenserow, ptasparserow, denserow, sparserow));

  /* Allocate space for ca */
  PetscCall(PetscCalloc1(ci[cn] + 1, &ca));

  /* put together the new matrix */
  PetscCall(MatSetSeqAIJWithArrays_private(PetscObjectComm((PetscObject)A), cn, cn, ci, cj, ca, NULL, C));
  PetscCall(MatSetBlockSize(C, pp->dof));

  /* MatCreateSeqAIJWithArrays flags matrix so PETSc doesn't free the user's arrays. */
  /* Since these are PETSc arrays, change flags to free them as necessary. */
  c          = (Mat_SeqAIJ *)(C->data);
  c->free_a  = PETSC_TRUE;
  c->free_ij = PETSC_TRUE;
  c->nonew   = 0;

  C->ops->ptapnumeric    = MatPtAPNumeric_SeqAIJ_SeqMAIJ;
  C->ops->productnumeric = MatProductNumeric_PtAP;

  /* Clean up. */
  PetscCall(MatRestoreSymbolicTranspose_SeqAIJ(P, &pti, &ptj));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatProductSymbolic_PtAP_SeqAIJ_SeqMAIJ(Mat C)
{
  Mat_Product *product = C->product;
  Mat          A = product->A, P = product->B;

  PetscFunctionBegin;
  PetscCall(MatPtAPSymbolic_SeqAIJ_SeqMAIJ(A, P, product->fill, C));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatPtAPNumeric_MPIAIJ_MPIXAIJ_allatonce(Mat, Mat, PetscInt, Mat);

PETSC_INTERN PetscErrorCode MatPtAPNumeric_MPIAIJ_MPIMAIJ_allatonce(Mat A, Mat P, Mat C)
{
  Mat_MPIMAIJ *maij = (Mat_MPIMAIJ *)P->data;

  PetscFunctionBegin;

  PetscCall(MatPtAPNumeric_MPIAIJ_MPIXAIJ_allatonce(A, maij->A, maij->dof, C));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatPtAPSymbolic_MPIAIJ_MPIXAIJ_allatonce(Mat, Mat, PetscInt, PetscReal, Mat);

PETSC_INTERN PetscErrorCode MatPtAPSymbolic_MPIAIJ_MPIMAIJ_allatonce(Mat A, Mat P, PetscReal fill, Mat C)
{
  Mat_MPIMAIJ *maij = (Mat_MPIMAIJ *)P->data;

  PetscFunctionBegin;
  PetscCall(MatPtAPSymbolic_MPIAIJ_MPIXAIJ_allatonce(A, maij->A, maij->dof, fill, C));
  C->ops->ptapnumeric = MatPtAPNumeric_MPIAIJ_MPIMAIJ_allatonce;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatPtAPNumeric_MPIAIJ_MPIXAIJ_allatonce_merged(Mat, Mat, PetscInt, Mat);

PETSC_INTERN PetscErrorCode MatPtAPNumeric_MPIAIJ_MPIMAIJ_allatonce_merged(Mat A, Mat P, Mat C)
{
  Mat_MPIMAIJ *maij = (Mat_MPIMAIJ *)P->data;

  PetscFunctionBegin;

  PetscCall(MatPtAPNumeric_MPIAIJ_MPIXAIJ_allatonce_merged(A, maij->A, maij->dof, C));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatPtAPSymbolic_MPIAIJ_MPIXAIJ_allatonce_merged(Mat, Mat, PetscInt, PetscReal, Mat);

PETSC_INTERN PetscErrorCode MatPtAPSymbolic_MPIAIJ_MPIMAIJ_allatonce_merged(Mat A, Mat P, PetscReal fill, Mat C)
{
  Mat_MPIMAIJ *maij = (Mat_MPIMAIJ *)P->data;

  PetscFunctionBegin;

  PetscCall(MatPtAPSymbolic_MPIAIJ_MPIXAIJ_allatonce_merged(A, maij->A, maij->dof, fill, C));
  C->ops->ptapnumeric = MatPtAPNumeric_MPIAIJ_MPIMAIJ_allatonce_merged;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatProductSymbolic_PtAP_MPIAIJ_MPIMAIJ(Mat C)
{
  Mat_Product *product = C->product;
  Mat          A = product->A, P = product->B;
  PetscBool    flg;

  PetscFunctionBegin;
  PetscCall(PetscStrcmp(product->alg, "allatonce", &flg));
  if (flg) {
    PetscCall(MatPtAPSymbolic_MPIAIJ_MPIMAIJ_allatonce(A, P, product->fill, C));
    C->ops->productnumeric = MatProductNumeric_PtAP;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(PetscStrcmp(product->alg, "allatonce_merged", &flg));
  if (flg) {
    PetscCall(MatPtAPSymbolic_MPIAIJ_MPIMAIJ_allatonce_merged(A, P, product->fill, C));
    C->ops->productnumeric = MatProductNumeric_PtAP;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  SETERRQ(PetscObjectComm((PetscObject)C), PETSC_ERR_SUP, "Mat Product Algorithm is not supported");
}

PETSC_INTERN PetscErrorCode MatConvert_SeqMAIJ_SeqAIJ(Mat A, MatType newtype, MatReuse reuse, Mat *newmat)
{
  Mat_SeqMAIJ *b   = (Mat_SeqMAIJ *)A->data;
  Mat          a   = b->AIJ, B;
  Mat_SeqAIJ  *aij = (Mat_SeqAIJ *)a->data;
  PetscInt     m, n, i, ncols, *ilen, nmax = 0, *icols, j, k, ii, dof = b->dof;
  PetscInt    *cols;
  PetscScalar *vals;

  PetscFunctionBegin;
  PetscCall(MatGetSize(a, &m, &n));
  PetscCall(PetscMalloc1(dof * m, &ilen));
  for (i = 0; i < m; i++) {
    nmax = PetscMax(nmax, aij->ilen[i]);
    for (j = 0; j < dof; j++) ilen[dof * i + j] = aij->ilen[i];
  }
  PetscCall(MatCreate(PETSC_COMM_SELF, &B));
  PetscCall(MatSetSizes(B, dof * m, dof * n, dof * m, dof * n));
  PetscCall(MatSetType(B, newtype));
  PetscCall(MatSeqAIJSetPreallocation(B, 0, ilen));
  PetscCall(PetscFree(ilen));
  PetscCall(PetscMalloc1(nmax, &icols));
  ii = 0;
  for (i = 0; i < m; i++) {
    PetscCall(MatGetRow_SeqAIJ(a, i, &ncols, &cols, &vals));
    for (j = 0; j < dof; j++) {
      for (k = 0; k < ncols; k++) icols[k] = dof * cols[k] + j;
      PetscCall(MatSetValues_SeqAIJ(B, 1, &ii, ncols, icols, vals, INSERT_VALUES));
      ii++;
    }
    PetscCall(MatRestoreRow_SeqAIJ(a, i, &ncols, &cols, &vals));
  }
  PetscCall(PetscFree(icols));
  PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));

  if (reuse == MAT_INPLACE_MATRIX) {
    PetscCall(MatHeaderReplace(A, &B));
  } else {
    *newmat = B;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#include <../src/mat/impls/aij/mpi/mpiaij.h>

PETSC_INTERN PetscErrorCode MatConvert_MPIMAIJ_MPIAIJ(Mat A, MatType newtype, MatReuse reuse, Mat *newmat)
{
  Mat_MPIMAIJ *maij    = (Mat_MPIMAIJ *)A->data;
  Mat          MatAIJ  = ((Mat_SeqMAIJ *)maij->AIJ->data)->AIJ, B;
  Mat          MatOAIJ = ((Mat_SeqMAIJ *)maij->OAIJ->data)->AIJ;
  Mat_SeqAIJ  *AIJ     = (Mat_SeqAIJ *)MatAIJ->data;
  Mat_SeqAIJ  *OAIJ    = (Mat_SeqAIJ *)MatOAIJ->data;
  Mat_MPIAIJ  *mpiaij  = (Mat_MPIAIJ *)maij->A->data;
  PetscInt     dof = maij->dof, i, j, *dnz = NULL, *onz = NULL, nmax = 0, onmax = 0;
  PetscInt    *oicols = NULL, *icols = NULL, ncols, *cols = NULL, oncols, *ocols = NULL;
  PetscInt     rstart, cstart, *garray, ii, k;
  PetscScalar *vals, *ovals;

  PetscFunctionBegin;
  PetscCall(PetscMalloc2(A->rmap->n, &dnz, A->rmap->n, &onz));
  for (i = 0; i < A->rmap->n / dof; i++) {
    nmax  = PetscMax(nmax, AIJ->ilen[i]);
    onmax = PetscMax(onmax, OAIJ->ilen[i]);
    for (j = 0; j < dof; j++) {
      dnz[dof * i + j] = AIJ->ilen[i];
      onz[dof * i + j] = OAIJ->ilen[i];
    }
  }
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A), &B));
  PetscCall(MatSetSizes(B, A->rmap->n, A->cmap->n, A->rmap->N, A->cmap->N));
  PetscCall(MatSetType(B, newtype));
  PetscCall(MatMPIAIJSetPreallocation(B, 0, dnz, 0, onz));
  PetscCall(MatSetBlockSize(B, dof));
  PetscCall(PetscFree2(dnz, onz));

  PetscCall(PetscMalloc2(nmax, &icols, onmax, &oicols));
  rstart = dof * maij->A->rmap->rstart;
  cstart = dof * maij->A->cmap->rstart;
  garray = mpiaij->garray;

  ii = rstart;
  for (i = 0; i < A->rmap->n / dof; i++) {
    PetscCall(MatGetRow_SeqAIJ(MatAIJ, i, &ncols, &cols, &vals));
    PetscCall(MatGetRow_SeqAIJ(MatOAIJ, i, &oncols, &ocols, &ovals));
    for (j = 0; j < dof; j++) {
      for (k = 0; k < ncols; k++) icols[k] = cstart + dof * cols[k] + j;
      for (k = 0; k < oncols; k++) oicols[k] = dof * garray[ocols[k]] + j;
      PetscCall(MatSetValues_MPIAIJ(B, 1, &ii, ncols, icols, vals, INSERT_VALUES));
      PetscCall(MatSetValues_MPIAIJ(B, 1, &ii, oncols, oicols, ovals, INSERT_VALUES));
      ii++;
    }
    PetscCall(MatRestoreRow_SeqAIJ(MatAIJ, i, &ncols, &cols, &vals));
    PetscCall(MatRestoreRow_SeqAIJ(MatOAIJ, i, &oncols, &ocols, &ovals));
  }
  PetscCall(PetscFree2(icols, oicols));

  PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));

  if (reuse == MAT_INPLACE_MATRIX) {
    PetscInt refct          = ((PetscObject)A)->refct; /* save ((PetscObject)A)->refct */
    ((PetscObject)A)->refct = 1;

    PetscCall(MatHeaderReplace(A, &B));

    ((PetscObject)A)->refct = refct; /* restore ((PetscObject)A)->refct */
  } else {
    *newmat = B;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatCreateSubMatrix_MAIJ(Mat mat, IS isrow, IS iscol, MatReuse cll, Mat *newmat)
{
  Mat A;

  PetscFunctionBegin;
  PetscCall(MatConvert(mat, MATAIJ, MAT_INITIAL_MATRIX, &A));
  PetscCall(MatCreateSubMatrix(A, isrow, iscol, cll, newmat));
  PetscCall(MatDestroy(&A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatCreateSubMatrices_MAIJ(Mat mat, PetscInt n, const IS irow[], const IS icol[], MatReuse scall, Mat *submat[])
{
  Mat A;

  PetscFunctionBegin;
  PetscCall(MatConvert(mat, MATAIJ, MAT_INITIAL_MATRIX, &A));
  PetscCall(MatCreateSubMatrices(A, n, irow, icol, scall, submat));
  PetscCall(MatDestroy(&A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatCreateMAIJ - Creates a matrix type providing restriction and interpolation
  operations for multicomponent problems.  It interpolates each component the same
  way independently.  The matrix type is based on `MATSEQAIJ` for sequential matrices,
  and `MATMPIAIJ` for distributed matrices.

  Collective

  Input Parameters:
+ A - the `MATAIJ` matrix describing the action on blocks
- dof - the block size (number of components per node)

  Output Parameter:
. maij - the new `MATMAIJ` matrix

  Level: advanced

  Operations provided:
.vb
    MatMult()
    MatMultTranspose()
    MatMultAdd()
    MatMultTransposeAdd()
    MatView()
.ve

.seealso: [](chapter_matrices), `Mat`, `MATAIJ`, `MATMAIJ`, `MatMAIJGetAIJ()`, `MatMAIJRedimension()`, `MATMAIJ`
@*/
PetscErrorCode MatCreateMAIJ(Mat A, PetscInt dof, Mat *maij)
{
  PetscInt  n;
  Mat       B;
  PetscBool flg;
#if defined(PETSC_HAVE_CUDA)
  /* hack to prevent conversion to AIJ format for CUDA when used inside a parallel MAIJ */
  PetscBool convert = dof < 0 ? PETSC_FALSE : PETSC_TRUE;
#endif

  PetscFunctionBegin;
  dof = PetscAbs(dof);
  PetscCall(PetscObjectReference((PetscObject)A));

  if (dof == 1) *maij = A;
  else {
    PetscCall(MatCreate(PetscObjectComm((PetscObject)A), &B));
    /* propagate vec type */
    PetscCall(MatSetVecType(B, A->defaultvectype));
    PetscCall(MatSetSizes(B, dof * A->rmap->n, dof * A->cmap->n, dof * A->rmap->N, dof * A->cmap->N));
    PetscCall(PetscLayoutSetBlockSize(B->rmap, dof));
    PetscCall(PetscLayoutSetBlockSize(B->cmap, dof));
    PetscCall(PetscLayoutSetUp(B->rmap));
    PetscCall(PetscLayoutSetUp(B->cmap));

    B->assembled = PETSC_TRUE;

    PetscCall(PetscObjectBaseTypeCompare((PetscObject)A, MATSEQAIJ, &flg));
    if (flg) {
      Mat_SeqMAIJ *b;

      PetscCall(MatSetType(B, MATSEQMAIJ));

      B->ops->setup   = NULL;
      B->ops->destroy = MatDestroy_SeqMAIJ;
      B->ops->view    = MatView_SeqMAIJ;

      b      = (Mat_SeqMAIJ *)B->data;
      b->dof = dof;
      b->AIJ = A;

      if (dof == 2) {
        B->ops->mult             = MatMult_SeqMAIJ_2;
        B->ops->multadd          = MatMultAdd_SeqMAIJ_2;
        B->ops->multtranspose    = MatMultTranspose_SeqMAIJ_2;
        B->ops->multtransposeadd = MatMultTransposeAdd_SeqMAIJ_2;
      } else if (dof == 3) {
        B->ops->mult             = MatMult_SeqMAIJ_3;
        B->ops->multadd          = MatMultAdd_SeqMAIJ_3;
        B->ops->multtranspose    = MatMultTranspose_SeqMAIJ_3;
        B->ops->multtransposeadd = MatMultTransposeAdd_SeqMAIJ_3;
      } else if (dof == 4) {
        B->ops->mult             = MatMult_SeqMAIJ_4;
        B->ops->multadd          = MatMultAdd_SeqMAIJ_4;
        B->ops->multtranspose    = MatMultTranspose_SeqMAIJ_4;
        B->ops->multtransposeadd = MatMultTransposeAdd_SeqMAIJ_4;
      } else if (dof == 5) {
        B->ops->mult             = MatMult_SeqMAIJ_5;
        B->ops->multadd          = MatMultAdd_SeqMAIJ_5;
        B->ops->multtranspose    = MatMultTranspose_SeqMAIJ_5;
        B->ops->multtransposeadd = MatMultTransposeAdd_SeqMAIJ_5;
      } else if (dof == 6) {
        B->ops->mult             = MatMult_SeqMAIJ_6;
        B->ops->multadd          = MatMultAdd_SeqMAIJ_6;
        B->ops->multtranspose    = MatMultTranspose_SeqMAIJ_6;
        B->ops->multtransposeadd = MatMultTransposeAdd_SeqMAIJ_6;
      } else if (dof == 7) {
        B->ops->mult             = MatMult_SeqMAIJ_7;
        B->ops->multadd          = MatMultAdd_SeqMAIJ_7;
        B->ops->multtranspose    = MatMultTranspose_SeqMAIJ_7;
        B->ops->multtransposeadd = MatMultTransposeAdd_SeqMAIJ_7;
      } else if (dof == 8) {
        B->ops->mult             = MatMult_SeqMAIJ_8;
        B->ops->multadd          = MatMultAdd_SeqMAIJ_8;
        B->ops->multtranspose    = MatMultTranspose_SeqMAIJ_8;
        B->ops->multtransposeadd = MatMultTransposeAdd_SeqMAIJ_8;
      } else if (dof == 9) {
        B->ops->mult             = MatMult_SeqMAIJ_9;
        B->ops->multadd          = MatMultAdd_SeqMAIJ_9;
        B->ops->multtranspose    = MatMultTranspose_SeqMAIJ_9;
        B->ops->multtransposeadd = MatMultTransposeAdd_SeqMAIJ_9;
      } else if (dof == 10) {
        B->ops->mult             = MatMult_SeqMAIJ_10;
        B->ops->multadd          = MatMultAdd_SeqMAIJ_10;
        B->ops->multtranspose    = MatMultTranspose_SeqMAIJ_10;
        B->ops->multtransposeadd = MatMultTransposeAdd_SeqMAIJ_10;
      } else if (dof == 11) {
        B->ops->mult             = MatMult_SeqMAIJ_11;
        B->ops->multadd          = MatMultAdd_SeqMAIJ_11;
        B->ops->multtranspose    = MatMultTranspose_SeqMAIJ_11;
        B->ops->multtransposeadd = MatMultTransposeAdd_SeqMAIJ_11;
      } else if (dof == 16) {
        B->ops->mult             = MatMult_SeqMAIJ_16;
        B->ops->multadd          = MatMultAdd_SeqMAIJ_16;
        B->ops->multtranspose    = MatMultTranspose_SeqMAIJ_16;
        B->ops->multtransposeadd = MatMultTransposeAdd_SeqMAIJ_16;
      } else if (dof == 18) {
        B->ops->mult             = MatMult_SeqMAIJ_18;
        B->ops->multadd          = MatMultAdd_SeqMAIJ_18;
        B->ops->multtranspose    = MatMultTranspose_SeqMAIJ_18;
        B->ops->multtransposeadd = MatMultTransposeAdd_SeqMAIJ_18;
      } else {
        B->ops->mult             = MatMult_SeqMAIJ_N;
        B->ops->multadd          = MatMultAdd_SeqMAIJ_N;
        B->ops->multtranspose    = MatMultTranspose_SeqMAIJ_N;
        B->ops->multtransposeadd = MatMultTransposeAdd_SeqMAIJ_N;
      }
#if defined(PETSC_HAVE_CUDA)
      PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatConvert_seqmaij_seqaijcusparse_C", MatConvert_SeqMAIJ_SeqAIJ));
#endif
      PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatConvert_seqmaij_seqaij_C", MatConvert_SeqMAIJ_SeqAIJ));
      PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatProductSetFromOptions_seqaij_seqmaij_C", MatProductSetFromOptions_SeqAIJ_SeqMAIJ));
    } else {
      Mat_MPIAIJ  *mpiaij = (Mat_MPIAIJ *)A->data;
      Mat_MPIMAIJ *b;
      IS           from, to;
      Vec          gvec;

      PetscCall(MatSetType(B, MATMPIMAIJ));

      B->ops->setup   = NULL;
      B->ops->destroy = MatDestroy_MPIMAIJ;
      B->ops->view    = MatView_MPIMAIJ;

      b      = (Mat_MPIMAIJ *)B->data;
      b->dof = dof;
      b->A   = A;

      PetscCall(MatCreateMAIJ(mpiaij->A, -dof, &b->AIJ));
      PetscCall(MatCreateMAIJ(mpiaij->B, -dof, &b->OAIJ));

      PetscCall(VecGetSize(mpiaij->lvec, &n));
      PetscCall(VecCreate(PETSC_COMM_SELF, &b->w));
      PetscCall(VecSetSizes(b->w, n * dof, n * dof));
      PetscCall(VecSetBlockSize(b->w, dof));
      PetscCall(VecSetType(b->w, VECSEQ));

      /* create two temporary Index sets for build scatter gather */
      PetscCall(ISCreateBlock(PetscObjectComm((PetscObject)A), dof, n, mpiaij->garray, PETSC_COPY_VALUES, &from));
      PetscCall(ISCreateStride(PETSC_COMM_SELF, n * dof, 0, 1, &to));

      /* create temporary global vector to generate scatter context */
      PetscCall(VecCreateMPIWithArray(PetscObjectComm((PetscObject)A), dof, dof * A->cmap->n, dof * A->cmap->N, NULL, &gvec));

      /* generate the scatter context */
      PetscCall(VecScatterCreate(gvec, from, b->w, to, &b->ctx));

      PetscCall(ISDestroy(&from));
      PetscCall(ISDestroy(&to));
      PetscCall(VecDestroy(&gvec));

      B->ops->mult             = MatMult_MPIMAIJ_dof;
      B->ops->multtranspose    = MatMultTranspose_MPIMAIJ_dof;
      B->ops->multadd          = MatMultAdd_MPIMAIJ_dof;
      B->ops->multtransposeadd = MatMultTransposeAdd_MPIMAIJ_dof;

#if defined(PETSC_HAVE_CUDA)
      PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatConvert_mpimaij_mpiaijcusparse_C", MatConvert_MPIMAIJ_MPIAIJ));
#endif
      PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatConvert_mpimaij_mpiaij_C", MatConvert_MPIMAIJ_MPIAIJ));
      PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatProductSetFromOptions_mpiaij_mpimaij_C", MatProductSetFromOptions_MPIAIJ_MPIMAIJ));
    }
    B->ops->createsubmatrix   = MatCreateSubMatrix_MAIJ;
    B->ops->createsubmatrices = MatCreateSubMatrices_MAIJ;
    PetscCall(MatSetUp(B));
#if defined(PETSC_HAVE_CUDA)
    /* temporary until we have CUDA implementation of MAIJ */
    {
      PetscBool flg;
      if (convert) {
        PetscCall(PetscObjectTypeCompareAny((PetscObject)A, &flg, MATSEQAIJCUSPARSE, MATMPIAIJCUSPARSE, MATAIJCUSPARSE, ""));
        if (flg) PetscCall(MatConvert(B, ((PetscObject)A)->type_name, MAT_INPLACE_MATRIX, &B));
      }
    }
#endif
    *maij = B;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
