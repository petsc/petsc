#define PETSC_SKIP_IMMINTRIN_H_CUDAWORKAROUND 1

#include <petscconf.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h> /*I "petscmat.h" I*/
#include <../src/mat/impls/aij/seq/seqcusparse/cusparsematimpl.h>
#include <../src/mat/impls/aij/mpi/mpicusparse/mpicusparsematimpl.h>
#include <../src/mat/impls/aij/mpi/cupm/mpiaijcupm.hpp>
#include <thrust/advance.h>
#include <thrust/partition.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

struct MatMPIAIJCUSPARSE_Policy {
  typedef Mat_MPIAIJCUSPARSE mat_struct_type;

  static const char *mpi_mat_type;
  static const char *seq_mat_type;
  static const char *vec_seq_type;
  static const char *set_format_c;
  static const char *mpi_convert_hypre_c;

  static PetscErrorCode CopyToGPU(Mat A) { return MatSeqAIJCUSPARSECopyToGPU(A); }
  static PetscErrorCode MergeMats(Mat A, Mat B, MatReuse r, Mat *C) { return MatSeqAIJCUSPARSEMergeMats(A, B, r, C); }

  static PetscErrorCode GetArray(Mat A, PetscScalar **a) { return MatSeqAIJCUSPARSEGetArray(A, a); }
  static PetscErrorCode GetArrayWrite(Mat A, PetscScalar **a) { return MatSeqAIJCUSPARSEGetArrayWrite(A, a); }
  static PetscErrorCode RestoreArray(Mat A, PetscScalar **a) { return MatSeqAIJCUSPARSERestoreArray(A, a); }
  static PetscErrorCode RestoreArrayWrite(Mat A, PetscScalar **a) { return MatSeqAIJCUSPARSERestoreArrayWrite(A, a); }

  static PetscErrorCode SetSubMatFormats(Mat A, Mat B, Mat_MPIAIJCUSPARSE *spptr)
  {
    PetscFunctionBegin;
    PetscCall(MatCUSPARSESetFormat(A, MAT_CUSPARSE_MULT, spptr->diagGPUMatFormat));
    PetscCall(MatCUSPARSESetFormat(B, MAT_CUSPARSE_MULT, spptr->offdiagGPUMatFormat));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
};
const char *MatMPIAIJCUSPARSE_Policy::mpi_mat_type        = MATMPIAIJCUSPARSE;
const char *MatMPIAIJCUSPARSE_Policy::seq_mat_type        = MATSEQAIJCUSPARSE;
const char *MatMPIAIJCUSPARSE_Policy::vec_seq_type        = VECSEQCUDA;
const char *MatMPIAIJCUSPARSE_Policy::set_format_c        = "MatCUSPARSESetFormat_C";
const char *MatMPIAIJCUSPARSE_Policy::mpi_convert_hypre_c = "MatConvert_mpiaijcusparse_hypre_C";

using MatMPIAIJCUSPARSE_CUPM_t = Petsc::mat::aij::cupm::impl::MatMPIAIJCUSPARSE_CUPM<Petsc::device::cupm::DeviceType::CUDA, MatMPIAIJCUSPARSE_Policy>;

static PetscErrorCode MatSetPreallocationCOO_MPIAIJCUSPARSE(Mat mat, PetscCount coo_n, PetscInt coo_i[], PetscInt coo_j[])
{
  return MatMPIAIJCUSPARSE_CUPM_t::SetPreallocationCOO(mat, coo_n, coo_i, coo_j);
}

static PetscErrorCode MatSetValuesCOO_MPIAIJCUSPARSE(Mat mat, const PetscScalar v[], InsertMode imode)
{
  return MatMPIAIJCUSPARSE_CUPM_t::SetValuesCOO(mat, v, imode);
}

static PetscErrorCode MatMPIAIJGetLocalMatMerge_MPIAIJCUSPARSE(Mat A, MatReuse scall, IS *glob, Mat *A_loc)
{
  return MatMPIAIJCUSPARSE_CUPM_t::GetLocalMatMerge(A, scall, glob, A_loc);
}

static PetscErrorCode MatMPIAIJSetPreallocation_MPIAIJCUSPARSE(Mat B, PetscInt d_nz, const PetscInt d_nnz[], PetscInt o_nz, const PetscInt o_nnz[])
{
  return MatMPIAIJCUSPARSE_CUPM_t::SetPreallocation(B, d_nz, d_nnz, o_nz, o_nnz);
}

static PetscErrorCode MatMult_MPIAIJCUSPARSE(Mat A, Vec xx, Vec yy)
{
  return MatMPIAIJCUSPARSE_CUPM_t::Mult(A, xx, yy);
}

static PetscErrorCode MatZeroEntries_MPIAIJCUSPARSE(Mat A)
{
  return MatMPIAIJCUSPARSE_CUPM_t::ZeroEntries(A);
}

static PetscErrorCode MatMultAdd_MPIAIJCUSPARSE(Mat A, Vec xx, Vec yy, Vec zz)
{
  return MatMPIAIJCUSPARSE_CUPM_t::MultAdd(A, xx, yy, zz);
}

static PetscErrorCode MatMultTranspose_MPIAIJCUSPARSE(Mat A, Vec xx, Vec yy)
{
  return MatMPIAIJCUSPARSE_CUPM_t::MultTranspose(A, xx, yy);
}

static PetscErrorCode MatAssemblyEnd_MPIAIJCUSPARSE(Mat A, MatAssemblyType mode)
{
  return MatMPIAIJCUSPARSE_CUPM_t::AssemblyEnd(A, mode);
}

static PetscErrorCode MatCUSPARSESetFormat_MPIAIJCUSPARSE(Mat A, MatCUSPARSEFormatOperation op, MatCUSPARSEStorageFormat format)
{
  Mat_MPIAIJ         *a              = (Mat_MPIAIJ *)A->data;
  Mat_MPIAIJCUSPARSE *cusparseStruct = (Mat_MPIAIJCUSPARSE *)a->spptr;

  PetscFunctionBegin;
  switch (op) {
  case MAT_CUSPARSE_MULT_DIAG:
    cusparseStruct->diagGPUMatFormat = format;
    break;
  case MAT_CUSPARSE_MULT_OFFDIAG:
    cusparseStruct->offdiagGPUMatFormat = format;
    break;
  case MAT_CUSPARSE_ALL:
    cusparseStruct->diagGPUMatFormat    = format;
    cusparseStruct->offdiagGPUMatFormat = format;
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "unsupported operation %d for MatCUSPARSEFormatOperation. Only MAT_CUSPARSE_MULT_DIAG, MAT_CUSPARSE_MULT_DIAG, and MAT_CUSPARSE_MULT_ALL are currently supported.", op);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetFromOptions_MPIAIJCUSPARSE(Mat A, PetscOptionItems PetscOptionsObject)
{
  MatCUSPARSEStorageFormat format;
  PetscBool                flg;
  Mat_MPIAIJ              *a              = (Mat_MPIAIJ *)A->data;
  Mat_MPIAIJCUSPARSE      *cusparseStruct = (Mat_MPIAIJCUSPARSE *)a->spptr;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "MPIAIJCUSPARSE options");
  if (A->factortype == MAT_FACTOR_NONE) {
    PetscCall(PetscOptionsEnum("-mat_cusparse_mult_diag_storage_format", "sets storage format of the diagonal blocks of (mpi)aijcusparse gpu matrices for SpMV", "MatCUSPARSESetFormat", MatCUSPARSEStorageFormats, (PetscEnum)cusparseStruct->diagGPUMatFormat, (PetscEnum *)&format, &flg));
    if (flg) PetscCall(MatCUSPARSESetFormat(A, MAT_CUSPARSE_MULT_DIAG, format));
    PetscCall(PetscOptionsEnum("-mat_cusparse_mult_offdiag_storage_format", "sets storage format of the off-diagonal blocks (mpi)aijcusparse gpu matrices for SpMV", "MatCUSPARSESetFormat", MatCUSPARSEStorageFormats, (PetscEnum)cusparseStruct->offdiagGPUMatFormat, (PetscEnum *)&format, &flg));
    if (flg) PetscCall(MatCUSPARSESetFormat(A, MAT_CUSPARSE_MULT_OFFDIAG, format));
    PetscCall(PetscOptionsEnum("-mat_cusparse_storage_format", "sets storage format of the diagonal and off-diagonal blocks (mpi)aijcusparse gpu matrices for SpMV", "MatCUSPARSESetFormat", MatCUSPARSEStorageFormats, (PetscEnum)cusparseStruct->diagGPUMatFormat, (PetscEnum *)&format, &flg));
    if (flg) PetscCall(MatCUSPARSESetFormat(A, MAT_CUSPARSE_ALL, format));
  }
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDestroy_MPIAIJCUSPARSE(Mat A)
{
  return MatMPIAIJCUSPARSE_CUPM_t::Destroy(A);
}

/* defines MatSetValues_MPICUSPARSE_Hash() */
#define TYPE AIJ
#define TYPE_AIJ
#define SUB_TYPE_CUSPARSE
#include "../src/mat/impls/aij/mpi/mpihashmat.h"
#undef TYPE
#undef TYPE_AIJ
#undef SUB_TYPE_CUSPARSE

static PetscErrorCode MatSetUp_MPI_HASH_CUSPARSE(Mat A)
{
  Mat_MPIAIJ         *b              = (Mat_MPIAIJ *)A->data;
  Mat_MPIAIJCUSPARSE *cusparseStruct = (Mat_MPIAIJCUSPARSE *)b->spptr;

  PetscFunctionBegin;
  PetscCall(MatSetUp_MPI_Hash(A));
  PetscCall(MatCUSPARSESetFormat(b->A, MAT_CUSPARSE_MULT, cusparseStruct->diagGPUMatFormat));
  PetscCall(MatCUSPARSESetFormat(b->B, MAT_CUSPARSE_MULT, cusparseStruct->offdiagGPUMatFormat));
  A->preallocated = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatConvert_MPIAIJ_MPIAIJCUSPARSE(Mat B, MatType, MatReuse reuse, Mat *newmat)
{
  Mat_MPIAIJ *a;
  Mat         A;

  PetscFunctionBegin;
  PetscCall(PetscDeviceInitialize(PETSC_DEVICE_CUDA));
  if (reuse == MAT_INITIAL_MATRIX) PetscCall(MatDuplicate(B, MAT_COPY_VALUES, newmat));
  else if (reuse == MAT_REUSE_MATRIX) PetscCall(MatCopy(B, *newmat, SAME_NONZERO_PATTERN));
  A             = *newmat;
  A->boundtocpu = PETSC_FALSE;
  PetscCall(PetscFree(A->defaultvectype));
  PetscCall(PetscStrallocpy(VECCUDA, &A->defaultvectype));

  a = (Mat_MPIAIJ *)A->data;
  if (a->A) PetscCall(MatSetType(a->A, MATSEQAIJCUSPARSE));
  if (a->B) PetscCall(MatSetType(a->B, MATSEQAIJCUSPARSE));
  if (a->lvec) PetscCall(VecSetType(a->lvec, VECSEQCUDA));

  if (reuse != MAT_REUSE_MATRIX && !a->spptr) PetscCallCXX(a->spptr = new Mat_MPIAIJCUSPARSE);

  A->ops->assemblyend           = MatAssemblyEnd_MPIAIJCUSPARSE;
  A->ops->mult                  = MatMult_MPIAIJCUSPARSE;
  A->ops->multadd               = MatMultAdd_MPIAIJCUSPARSE;
  A->ops->multtranspose         = MatMultTranspose_MPIAIJCUSPARSE;
  A->ops->setfromoptions        = MatSetFromOptions_MPIAIJCUSPARSE;
  A->ops->destroy               = MatDestroy_MPIAIJCUSPARSE;
  A->ops->zeroentries           = MatZeroEntries_MPIAIJCUSPARSE;
  A->ops->productsetfromoptions = MatProductSetFromOptions_MPIAIJBACKEND;
  A->ops->setup                 = MatSetUp_MPI_HASH_CUSPARSE;
  A->ops->getcurrentmemtype     = MatGetCurrentMemType_MPIAIJ;

  PetscCall(PetscObjectChangeTypeName((PetscObject)A, MATMPIAIJCUSPARSE));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatMPIAIJGetLocalMatMerge_C", MatMPIAIJGetLocalMatMerge_MPIAIJCUSPARSE));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatMPIAIJSetPreallocation_C", MatMPIAIJSetPreallocation_MPIAIJCUSPARSE));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatCUSPARSESetFormat_C", MatCUSPARSESetFormat_MPIAIJCUSPARSE));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSetPreallocationCOO_C", MatSetPreallocationCOO_MPIAIJCUSPARSE));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSetValuesCOO_C", MatSetValuesCOO_MPIAIJCUSPARSE));
#if defined(PETSC_HAVE_HYPRE)
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatConvert_mpiaijcusparse_hypre_C", MatConvert_AIJ_HYPRE));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode MatCreate_MPIAIJCUSPARSE(Mat A)
{
  PetscFunctionBegin;
  PetscCall(PetscDeviceInitialize(PETSC_DEVICE_CUDA));
  PetscCall(MatCreate_MPIAIJ(A));
  PetscCall(MatConvert_MPIAIJ_MPIAIJCUSPARSE(A, MATMPIAIJCUSPARSE, MAT_INPLACE_MATRIX, &A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatCreateAIJCUSPARSE - Creates a sparse matrix in `MATAIJCUSPARSE` (compressed row) format. This matrix will ultimately be pushed down
  to NVIDIA GPUs and use the CuSPARSE library for calculations.

  Collective

  Input Parameters:
+ comm  - MPI communicator, set to `PETSC_COMM_SELF`
. m     - number of local rows (or `PETSC_DECIDE` to have calculated if `M` is given)
          This value should be the same as the local size used in creating the
          $y$ vector for the matrix-vector product $y = Ax$.
. n     - This value should be the same as the local size used in creating the
          $x$ vector for the matrix-vector product $y = Ax$. (or `PETSC_DECIDE` to have
          calculated if `N` is given) For square matrices `n` is almost always `m`.
. M     - number of global rows (or `PETSC_DETERMINE` to have calculated if `m` is given)
. N     - number of global columns (or `PETSC_DETERMINE` to have calculated if `n` is given)
. d_nz  - number of nonzeros per row in DIAGONAL portion of local submatrix
          (same value is used for all local rows)
. d_nnz - array containing the number of nonzeros in the various rows of the
          DIAGONAL portion of the local submatrix (possibly different for each row)
          or `NULL`, if `d_nz` is used to specify the nonzero structure.
          The size of this array is equal to the number of local rows, i.e `m`.
          For matrices you plan to factor you must leave room for the diagonal entry and
          put in the entry even if it is zero.
. o_nz  - number of nonzeros per row in the OFF-DIAGONAL portion of local
          submatrix (same value is used for all local rows).
- o_nnz - array containing the number of nonzeros in the various rows of the
          OFF-DIAGONAL portion of the local submatrix (possibly different for
          each row) or `NULL`, if `o_nz` is used to specify the nonzero
          structure. The size of this array is equal to the number
          of local rows, i.e `m`.

  Output Parameter:
. A - the matrix

  Level: intermediate

  Notes:
  It is recommended that one use the `MatCreate()`, `MatSetType()` and/or `MatSetFromOptions()`,
  MatXXXXSetPreallocation() paradigm instead of this routine directly.
  [MatXXXXSetPreallocation() is, for example, `MatSeqAIJSetPreallocation()`]

  The AIJ format, also called the
  compressed row storage), is fully compatible with standard Fortran
  storage.  That is, the stored row and column indices can begin at
  either one (as in Fortran) or zero.

  When working with matrices for GPUs, it is often better to use the `MatSetPreallocationCOO()` and `MatSetValuesCOO()` paradigm rather than using this routine and `MatSetValues()`

.seealso: [](ch_matrices), `Mat`, `MATAIJCUSPARSE`, `MatCreate()`, `MatCreateAIJ()`, `MatSetValues()`, `MatSeqAIJSetColumnIndices()`, `MatCreateSeqAIJWithArrays()`, `MATMPIAIJCUSPARSE`
@*/
PetscErrorCode MatCreateAIJCUSPARSE(MPI_Comm comm, PetscInt m, PetscInt n, PetscInt M, PetscInt N, PetscInt d_nz, const PetscInt d_nnz[], PetscInt o_nz, const PetscInt o_nnz[], Mat *A)
{
  return MatMPIAIJCUSPARSE_CUPM_t::CreateAIJ(comm, m, n, M, N, d_nz, d_nnz, o_nz, o_nnz, A);
}

/*MC
   MATMPIAIJCUSPARSE - A matrix type to be used for sparse matrices on NVIDIA GPUs.

   Options Database Keys:
+  -mat_type mpiaijcusparse                      - sets the matrix type to `MATMPIAIJCUSPARSE`
.  -mat_cusparse_storage_format csr              - sets the storage format of diagonal and off-diagonal matrices. Other options include ell (ellpack) or hyb (hybrid).
.  -mat_cusparse_mult_diag_storage_format csr    - sets the storage format of diagonal matrix. Other options include ell (ellpack) or hyb (hybrid).
-  -mat_cusparse_mult_offdiag_storage_format csr - sets the storage format of off-diagonal matrix. Other options include ell (ellpack) or hyb (hybrid).

  Level: beginner

  Notes:
  These matrices can be in either CSR, ELL, or HYB format. The ELL and HYB formats require CUDA 4.2 or later.

  All matrix calculations are performed on NVIDIA GPUs using the cuSPARSE library.

  Uses 32-bit integers internally. If PETSc is configured `--with-64-bit-indices`, the integer row and column indices are stored on the GPU with `int`. It is unclear what happens
  if some integer values passed in do not fit in `int`.

.seealso: [](ch_matrices), `Mat`, `MatCreateAIJCUSPARSE()`, `MATSEQAIJCUSPARSE`, `MATMPIAIJCUSPARSE`, `MatCreateSeqAIJCUSPARSE()`, `MatCUSPARSESetFormat()`, `MatCUSPARSEStorageFormat`, `MatCUSPARSEFormatOperation`
M*/

/*MC
   MATAIJCUSPARSE - A matrix type to be used for sparse matrices on NVIDIA GPUs; it is as same as `MATSEQAIJCUSPARSE` on one MPI process and `MATMPIAIJCUSPARSE` on multiple processes.

  Level: beginner

.seealso: [](ch_matrices), `Mat`, `MATAIJCUSPARSE`, `MATSEQAIJCUSPARSE`
M*/
