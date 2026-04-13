/* Portions of this code are under:
   Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
*/
#include <../src/mat/impls/aij/mpi/mpiaij.h> /*I "petscmat.h" I*/
#include <../src/mat/impls/aij/seq/seqhipsparse/hipsparsematimpl.h>
#include <../src/mat/impls/aij/mpi/mpihipsparse/mpihipsparsematimpl.h>
#include <../src/mat/impls/aij/mpi/cupm/mpiaijcupm.hpp>
#include <thrust/advance.h>
#include <thrust/partition.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

struct MatMPIAIJHIPSPARSE_Policy {
  typedef Mat_MPIAIJHIPSPARSE mat_struct_type;

  static const char *mpi_mat_type;
  static const char *seq_mat_type;
  static const char *vec_seq_type;
  static const char *set_format_c;
  static const char *mpi_convert_hypre_c;

  static PetscErrorCode CopyToGPU(Mat A) { return MatSeqAIJHIPSPARSECopyToGPU(A); }
  static PetscErrorCode MergeMats(Mat A, Mat B, MatReuse r, Mat *C) { return MatSeqAIJHIPSPARSEMergeMats(A, B, r, C); }

  static PetscErrorCode GetArray(Mat A, PetscScalar **a) { return MatSeqAIJHIPSPARSEGetArray(A, a); }
  static PetscErrorCode GetArrayWrite(Mat A, PetscScalar **a) { return MatSeqAIJHIPSPARSEGetArrayWrite(A, a); }
  static PetscErrorCode RestoreArray(Mat A, PetscScalar **a) { return MatSeqAIJHIPSPARSERestoreArray(A, a); }
  static PetscErrorCode RestoreArrayWrite(Mat A, PetscScalar **a) { return MatSeqAIJHIPSPARSERestoreArrayWrite(A, a); }

  static PetscErrorCode SetSubMatFormats(Mat A, Mat B, Mat_MPIAIJHIPSPARSE *spptr)
  {
    PetscFunctionBegin;
    PetscCall(MatHIPSPARSESetFormat(A, MAT_HIPSPARSE_MULT, spptr->diagGPUMatFormat));
    PetscCall(MatHIPSPARSESetFormat(B, MAT_HIPSPARSE_MULT, spptr->offdiagGPUMatFormat));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
};
const char *MatMPIAIJHIPSPARSE_Policy::mpi_mat_type        = MATMPIAIJHIPSPARSE;
const char *MatMPIAIJHIPSPARSE_Policy::seq_mat_type        = MATSEQAIJHIPSPARSE;
const char *MatMPIAIJHIPSPARSE_Policy::vec_seq_type        = VECSEQHIP;
const char *MatMPIAIJHIPSPARSE_Policy::set_format_c        = "MatHIPSPARSESetFormat_C";
const char *MatMPIAIJHIPSPARSE_Policy::mpi_convert_hypre_c = "MatConvert_mpiaijhipsparse_hypre_C";

using MatMPIAIJHIPSPARSE_CUPM_t = Petsc::mat::aij::cupm::impl::MatMPIAIJCUSPARSE_CUPM<Petsc::device::cupm::DeviceType::HIP, MatMPIAIJHIPSPARSE_Policy>;

static PetscErrorCode MatSetPreallocationCOO_MPIAIJHIPSPARSE(Mat mat, PetscCount coo_n, PetscInt coo_i[], PetscInt coo_j[])
{
  return MatMPIAIJHIPSPARSE_CUPM_t::SetPreallocationCOO(mat, coo_n, coo_i, coo_j);
}

static PetscErrorCode MatSetValuesCOO_MPIAIJHIPSPARSE(Mat mat, const PetscScalar v[], InsertMode imode)
{
  return MatMPIAIJHIPSPARSE_CUPM_t::SetValuesCOO(mat, v, imode);
}

static PetscErrorCode MatMPIAIJGetLocalMatMerge_MPIAIJHIPSPARSE(Mat A, MatReuse scall, IS *glob, Mat *A_loc)
{
  return MatMPIAIJHIPSPARSE_CUPM_t::GetLocalMatMerge(A, scall, glob, A_loc);
}

static PetscErrorCode MatMPIAIJSetPreallocation_MPIAIJHIPSPARSE(Mat B, PetscInt d_nz, const PetscInt d_nnz[], PetscInt o_nz, const PetscInt o_nnz[])
{
  return MatMPIAIJHIPSPARSE_CUPM_t::SetPreallocation(B, d_nz, d_nnz, o_nz, o_nnz);
}

static PetscErrorCode MatMult_MPIAIJHIPSPARSE(Mat A, Vec xx, Vec yy)
{
  return MatMPIAIJHIPSPARSE_CUPM_t::Mult(A, xx, yy);
}

static PetscErrorCode MatZeroEntries_MPIAIJHIPSPARSE(Mat A)
{
  return MatMPIAIJHIPSPARSE_CUPM_t::ZeroEntries(A);
}

static PetscErrorCode MatMultAdd_MPIAIJHIPSPARSE(Mat A, Vec xx, Vec yy, Vec zz)
{
  return MatMPIAIJHIPSPARSE_CUPM_t::MultAdd(A, xx, yy, zz);
}

static PetscErrorCode MatMultTranspose_MPIAIJHIPSPARSE(Mat A, Vec xx, Vec yy)
{
  return MatMPIAIJHIPSPARSE_CUPM_t::MultTranspose(A, xx, yy);
}

static PetscErrorCode MatAssemblyEnd_MPIAIJHIPSPARSE(Mat A, MatAssemblyType mode)
{
  return MatMPIAIJHIPSPARSE_CUPM_t::AssemblyEnd(A, mode);
}

static PetscErrorCode MatHIPSPARSESetFormat_MPIAIJHIPSPARSE(Mat A, MatHIPSPARSEFormatOperation op, MatHIPSPARSEStorageFormat format)
{
  Mat_MPIAIJ          *a               = (Mat_MPIAIJ *)A->data;
  Mat_MPIAIJHIPSPARSE *hipsparseStruct = (Mat_MPIAIJHIPSPARSE *)a->spptr;

  PetscFunctionBegin;
  switch (op) {
  case MAT_HIPSPARSE_MULT_DIAG:
    hipsparseStruct->diagGPUMatFormat = format;
    break;
  case MAT_HIPSPARSE_MULT_OFFDIAG:
    hipsparseStruct->offdiagGPUMatFormat = format;
    break;
  case MAT_HIPSPARSE_ALL:
    hipsparseStruct->diagGPUMatFormat    = format;
    hipsparseStruct->offdiagGPUMatFormat = format;
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "unsupported operation %d for MatHIPSPARSEFormatOperation. Only MAT_HIPSPARSE_MULT_DIAG, MAT_HIPSPARSE_MULT_DIAG, and MAT_HIPSPARSE_MULT_ALL are currently supported.", op);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetFromOptions_MPIAIJHIPSPARSE(Mat A, PetscOptionItems PetscOptionsObject)
{
  MatHIPSPARSEStorageFormat format;
  PetscBool                 flg;
  Mat_MPIAIJ               *a               = (Mat_MPIAIJ *)A->data;
  Mat_MPIAIJHIPSPARSE      *hipsparseStruct = (Mat_MPIAIJHIPSPARSE *)a->spptr;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "MPIAIJHIPSPARSE options");
  if (A->factortype == MAT_FACTOR_NONE) {
    PetscCall(PetscOptionsEnum("-mat_hipsparse_mult_diag_storage_format", "sets storage format of the diagonal blocks of (mpi)aijhipsparse gpu matrices for SpMV", "MatHIPSPARSESetFormat", MatHIPSPARSEStorageFormats, (PetscEnum)hipsparseStruct->diagGPUMatFormat, (PetscEnum *)&format, &flg));
    if (flg) PetscCall(MatHIPSPARSESetFormat(A, MAT_HIPSPARSE_MULT_DIAG, format));
    PetscCall(PetscOptionsEnum("-mat_hipsparse_mult_offdiag_storage_format", "sets storage format of the off-diagonal blocks (mpi)aijhipsparse gpu matrices for SpMV", "MatHIPSPARSESetFormat", MatHIPSPARSEStorageFormats, (PetscEnum)hipsparseStruct->offdiagGPUMatFormat, (PetscEnum *)&format, &flg));
    if (flg) PetscCall(MatHIPSPARSESetFormat(A, MAT_HIPSPARSE_MULT_OFFDIAG, format));
    PetscCall(PetscOptionsEnum("-mat_hipsparse_storage_format", "sets storage format of the diagonal and off-diagonal blocks (mpi)aijhipsparse gpu matrices for SpMV", "MatHIPSPARSESetFormat", MatHIPSPARSEStorageFormats, (PetscEnum)hipsparseStruct->diagGPUMatFormat, (PetscEnum *)&format, &flg));
    if (flg) PetscCall(MatHIPSPARSESetFormat(A, MAT_HIPSPARSE_ALL, format));
  }
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDestroy_MPIAIJHIPSPARSE(Mat A)
{
  return MatMPIAIJHIPSPARSE_CUPM_t::Destroy(A);
}

PETSC_INTERN PetscErrorCode MatConvert_MPIAIJ_MPIAIJHIPSPARSE(Mat B, MatType mtype, MatReuse reuse, Mat *newmat)
{
  Mat_MPIAIJ *a;
  Mat         A;

  PetscFunctionBegin;
  PetscCall(PetscDeviceInitialize(PETSC_DEVICE_HIP));
  if (reuse == MAT_INITIAL_MATRIX) PetscCall(MatDuplicate(B, MAT_COPY_VALUES, newmat));
  else if (reuse == MAT_REUSE_MATRIX) PetscCall(MatCopy(B, *newmat, SAME_NONZERO_PATTERN));
  A             = *newmat;
  A->boundtocpu = PETSC_FALSE;
  PetscCall(PetscFree(A->defaultvectype));
  PetscCall(PetscStrallocpy(VECHIP, &A->defaultvectype));

  a = (Mat_MPIAIJ *)A->data;
  if (a->A) PetscCall(MatSetType(a->A, MATSEQAIJHIPSPARSE));
  if (a->B) PetscCall(MatSetType(a->B, MATSEQAIJHIPSPARSE));
  if (a->lvec) PetscCall(VecSetType(a->lvec, VECSEQHIP));

  if (reuse != MAT_REUSE_MATRIX && !a->spptr) PetscCallCXX(a->spptr = new Mat_MPIAIJHIPSPARSE);

  A->ops->assemblyend           = MatAssemblyEnd_MPIAIJHIPSPARSE;
  A->ops->mult                  = MatMult_MPIAIJHIPSPARSE;
  A->ops->multadd               = MatMultAdd_MPIAIJHIPSPARSE;
  A->ops->multtranspose         = MatMultTranspose_MPIAIJHIPSPARSE;
  A->ops->setfromoptions        = MatSetFromOptions_MPIAIJHIPSPARSE;
  A->ops->destroy               = MatDestroy_MPIAIJHIPSPARSE;
  A->ops->zeroentries           = MatZeroEntries_MPIAIJHIPSPARSE;
  A->ops->productsetfromoptions = MatProductSetFromOptions_MPIAIJBACKEND;
  A->ops->getcurrentmemtype     = MatGetCurrentMemType_MPIAIJ;

  PetscCall(PetscObjectChangeTypeName((PetscObject)A, MATMPIAIJHIPSPARSE));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatMPIAIJGetLocalMatMerge_C", MatMPIAIJGetLocalMatMerge_MPIAIJHIPSPARSE));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatMPIAIJSetPreallocation_C", MatMPIAIJSetPreallocation_MPIAIJHIPSPARSE));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatHIPSPARSESetFormat_C", MatHIPSPARSESetFormat_MPIAIJHIPSPARSE));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSetPreallocationCOO_C", MatSetPreallocationCOO_MPIAIJHIPSPARSE));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSetValuesCOO_C", MatSetValuesCOO_MPIAIJHIPSPARSE));
#if defined(PETSC_HAVE_HYPRE)
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatConvert_mpiaijhipsparse_hypre_C", MatConvert_AIJ_HYPRE));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode MatCreate_MPIAIJHIPSPARSE(Mat A)
{
  PetscFunctionBegin;
  PetscCall(PetscDeviceInitialize(PETSC_DEVICE_HIP));
  PetscCall(MatCreate_MPIAIJ(A));
  PetscCall(MatConvert_MPIAIJ_MPIAIJHIPSPARSE(A, MATMPIAIJHIPSPARSE, MAT_INPLACE_MATRIX, &A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatCreateAIJHIPSPARSE - Creates a sparse matrix in AIJ (compressed row) format
  (the default parallel PETSc format).

  Collective

  Input Parameters:
+ comm  - MPI communicator, set to `PETSC_COMM_SELF`
. m     - number of local rows (or `PETSC_DECIDE` to have calculated if `M` is given)
           This value should be the same as the local size used in creating the
           y vector for the matrix-vector product y = Ax.
. n     - This value should be the same as the local size used in creating the
       x vector for the matrix-vector product y = Ax. (or PETSC_DECIDE to have
       calculated if `N` is given) For square matrices `n` is almost always `m`.
. M     - number of global rows (or `PETSC_DETERMINE` to have calculated if `m` is given)
. N     - number of global columns (or `PETSC_DETERMINE` to have calculated if `n` is given)
. d_nz  - number of nonzeros per row (same for all rows), for the "diagonal" portion of the matrix
. d_nnz - array containing the number of nonzeros in the various rows (possibly different for each row) or `NULL`, for the "diagonal" portion of the matrix
. o_nz  - number of nonzeros per row (same for all rows), for the "off-diagonal" portion of the matrix
- o_nnz - array containing the number of nonzeros in the various rows (possibly different for each row) or `NULL`, for the "off-diagonal" portion of the matrix

  Output Parameter:
. A - the matrix

  Level: intermediate

  Notes:
  This matrix will ultimately pushed down to AMD GPUs and use the HIPSPARSE library for
  calculations. For good matrix assembly performance the user should preallocate the matrix
  storage by setting the parameter `nz` (or the array `nnz`).

  It is recommended that one use the `MatCreate()`, `MatSetType()` and/or `MatSetFromOptions()`,
  MatXXXXSetPreallocation() paradigm instead of this routine directly.
  [MatXXXXSetPreallocation() is, for example, `MatSeqAIJSetPreallocation()`]

  If `d_nnz` (`o_nnz`) is given then `d_nz` (`o_nz`) is ignored

  The `MATAIJ` format (compressed row storage), is fully compatible with standard Fortran
  storage.  That is, the stored row and column indices can begin at
  either one (as in Fortran) or zero.

  Specify the preallocated storage with either `d_nz` (`o_nz`) or `d_nnz` (`o_nnz`) (not both).
  Set `d_nz` (`o_nz`) = `PETSC_DEFAULT` and `d_nnz` (`o_nnz`) = `NULL` for PETSc to control dynamic memory
  allocation.

.seealso: [](ch_matrices), `Mat`, `MatCreate()`, `MatCreateAIJ()`, `MatSetValues()`, `MatSeqAIJSetColumnIndices()`, `MatCreateSeqAIJWithArrays()`, `MATMPIAIJHIPSPARSE`, `MATAIJHIPSPARSE`
@*/
PetscErrorCode MatCreateAIJHIPSPARSE(MPI_Comm comm, PetscInt m, PetscInt n, PetscInt M, PetscInt N, PetscInt d_nz, const PetscInt d_nnz[], PetscInt o_nz, const PetscInt o_nnz[], Mat *A)
{
  return MatMPIAIJHIPSPARSE_CUPM_t::CreateAIJ(comm, m, n, M, N, d_nz, d_nnz, o_nz, o_nnz, A);
}

/*MC
   MATAIJHIPSPARSE - A matrix type to be used for sparse matrices; it is as same as `MATMPIAIJHIPSPARSE`.

   A matrix type whose data resides on GPUs. These matrices can be in either
   CSR, ELL, or Hybrid format. All matrix calculations are performed on AMD GPUs using the HIPSPARSE library.

   This matrix type is identical to `MATSEQAIJHIPSPARSE` when constructed with a single process communicator,
   and `MATMPIAIJHIPSPARSE` otherwise.  As a result, for single process communicators,
   `MatSeqAIJSetPreallocation()` is supported, and similarly `MatMPIAIJSetPreallocation()` is supported
   for communicators controlling multiple processes.  It is recommended that you call both of
   the above preallocation routines for simplicity.

   Options Database Keys:
+  -mat_type mpiaijhipsparse - sets the matrix type to `MATMPIAIJHIPSPARSE`
.  -mat_hipsparse_storage_format csr - sets the storage format of diagonal and off-diagonal matrices. Other options include ell (ellpack) or hyb (hybrid).
.  -mat_hipsparse_mult_diag_storage_format csr - sets the storage format of diagonal matrix. Other options include ell (ellpack) or hyb (hybrid).
-  -mat_hipsparse_mult_offdiag_storage_format csr - sets the storage format of off-diagonal matrix. Other options include ell (ellpack) or hyb (hybrid).

  Level: beginner

.seealso: [](ch_matrices), `Mat`, `MatCreateAIJHIPSPARSE()`, `MATSEQAIJHIPSPARSE`, `MATMPIAIJHIPSPARSE`, `MatCreateSeqAIJHIPSPARSE()`, `MatHIPSPARSESetFormat()`, `MatHIPSPARSEStorageFormat`, `MatHIPSPARSEFormatOperation`
M*/

/*MC
   MATMPIAIJHIPSPARSE - A matrix type to be used for sparse matrices; it is as same as `MATAIJHIPSPARSE`.

  Level: beginner

.seealso: [](ch_matrices), `Mat`, `MATAIJHIPSPARSE`, `MATSEQAIJHIPSPARSE`
M*/
