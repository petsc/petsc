#pragma once
#include <petsc_kokkos.hpp>
#include <petscmat_kokkos.hpp>
#include <petsc/private/kokkosimpl.hpp>
#include <../src/mat/impls/aij/seq/aij.h>
#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_spiluk.hpp>
#include <string>

namespace
{
PETSC_NODISCARD inline decltype(auto) NoInit(std::string label)
{
  return Kokkos::view_alloc(Kokkos::WithoutInitializing, std::move(label));
}
} // namespace

using MatRowMapType = PetscInt;
using MatColIdxType = PetscInt;
using MatScalarType = PetscScalar;

template <class MemorySpace>
using KokkosCsrMatrixType = typename KokkosSparse::CrsMatrix<MatScalarType, MatColIdxType, MemorySpace, void /* MemoryTraits */, MatRowMapType>;
template <class MemorySpace>
using KokkosCsrGraphType = typename KokkosCsrMatrixType<MemorySpace>::staticcrsgraph_type;

using KokkosCsrGraph     = KokkosCsrGraphType<DefaultMemorySpace>;
using KokkosCsrGraphHost = KokkosCsrGraphType<HostMirrorMemorySpace>;

using KokkosCsrMatrix     = KokkosCsrMatrixType<DefaultMemorySpace>;
using KokkosCsrMatrixHost = KokkosCsrMatrixType<HostMirrorMemorySpace>;

using MatRowMapKokkosView = KokkosCsrGraph::row_map_type::non_const_type;
using MatColIdxKokkosView = KokkosCsrGraph::entries_type::non_const_type;
using MatScalarKokkosView = KokkosCsrMatrix::values_type::non_const_type;

using MatRowMapKokkosViewHost = KokkosCsrGraphHost::row_map_type::non_const_type;
using MatColIdxKokkosViewHost = KokkosCsrGraphHost::entries_type::non_const_type;
using MatScalarKokkosViewHost = KokkosCsrMatrixHost::values_type::non_const_type;

using ConstMatRowMapKokkosView = KokkosCsrGraph::row_map_type::const_type;
using ConstMatColIdxKokkosView = KokkosCsrGraph::entries_type::const_type;
using ConstMatScalarKokkosView = KokkosCsrMatrix::values_type::const_type;

using ConstMatRowMapKokkosViewHost = KokkosCsrGraphHost::row_map_type::const_type;
using ConstMatColIdxKokkosViewHost = KokkosCsrGraphHost::entries_type::const_type;
using ConstMatScalarKokkosViewHost = KokkosCsrMatrixHost::values_type::const_type;

using MatRowMapKokkosDualView = Kokkos::DualView<MatRowMapType *>;
using MatColIdxKokkosDualView = Kokkos::DualView<MatColIdxType *>;
using MatScalarKokkosDualView = Kokkos::DualView<MatScalarType *>;

using KernelHandle = KokkosKernels::Experimental::KokkosKernelsHandle<MatRowMapType, MatColIdxType, MatScalarType, DefaultExecutionSpace, DefaultMemorySpace, DefaultMemorySpace>;

using KokkosTeamMemberType = Kokkos::TeamPolicy<DefaultExecutionSpace>::member_type;

/* For mat->spptr of a factorized matrix */
struct Mat_SeqAIJKokkosTriFactors {
  MatRowMapKokkosView   iL_d, iU_d, iLt_d, iUt_d; /* rowmap for L, U, L^t, U^t of A=LU */
  MatColIdxKokkosView   jL_d, jU_d, jLt_d, jUt_d; /* column ids */
  MatScalarKokkosView   aL_d, aU_d, aLt_d, aUt_d; /* matrix values */
  KernelHandle          kh, khL, khU, khLt, khUt; /* Kernel handles for ILU factorization of A, and TRSV of L, U, L^t, U^t */
  PetscScalarKokkosView workVector;
  PetscBool             transpose_updated;         /* Are L^T, U^T updated wrt L, U*/
  PetscBool             sptrsv_symbolic_completed; /* Have we completed the symbolic solve for L and U */

  MatRowMapKokkosViewHost iL_h, iU_h, iLt_h, iUt_h; // temp. buffers when we do factorization with PETSc on host. We copy L, U to these buffers and then copy to device
  MatColIdxKokkosViewHost jL_h, jU_h, jLt_h, jUt_h;
  MatScalarKokkosViewHost aL_h, aU_h, aLt_h, aUt_h, D_h; // D is for LDLT factorization
  MatScalarKokkosView     D_d;
  Mat                     L, U, Lt, Ut; // MATSEQAIJ on host if needed. Their arrays are alias to (iL_h, jL_h, aL_h), (iU_h, jU_h, aU_h) and their transpose.
                                        // MatTranspose() on host might be faster than KK's csr transpose on device.

  PetscIntKokkosView rowperm, colperm; // row permutation and column permutation

  Mat_SeqAIJKokkosTriFactors(PetscInt n) : workVector("workVector", n)
  {
    L = U = Lt = Ut   = nullptr;
    transpose_updated = sptrsv_symbolic_completed = PETSC_FALSE;
  }

  ~Mat_SeqAIJKokkosTriFactors() { Destroy(); }

  void Destroy()
  {
    PetscFunctionBeginUser;
    kh.destroy_spiluk_handle();
    khL.destroy_sptrsv_handle();
    khU.destroy_sptrsv_handle();
    khLt.destroy_sptrsv_handle();
    khUt.destroy_sptrsv_handle();
    PetscCallVoid(MatDestroy(&L));
    PetscCallVoid(MatDestroy(&U));
    PetscCallVoid(MatDestroy(&Lt));
    PetscCallVoid(MatDestroy(&Ut));
    L = U = Lt = Ut   = nullptr;
    transpose_updated = sptrsv_symbolic_completed = PETSC_FALSE;
    PetscFunctionReturnVoid();
  }
};

/* For mat->spptr of a regular matrix */
struct Mat_SeqAIJKokkos {
  MatRowMapKokkosDualView i_dual;
  MatColIdxKokkosDualView j_dual;
  MatScalarKokkosDualView a_dual;
  PetscBool               host_aij_allocated_by_kokkos = PETSC_FALSE; /* Are host views of a, i, j in the duals allocated by Kokkos? */

  MatRowMapKokkosDualView diag_dual; /* Diagonal pointer, built on demand */

  KokkosCsrMatrix  csrmat;       /* The CSR matrix, used to call KK functions */
  PetscObjectState nonzerostate; /* State of the nonzero pattern (graph) on device */

  KokkosCsrMatrix     csrmatT, csrmatH;                     /* Transpose and Hermitian of the matrix (built on demand) */
  PetscBool           transpose_updated, hermitian_updated; /* Are At, Ah updated wrt the matrix? */
  MatRowMapKokkosView transpose_perm;                       // A permutation array making Ta(i) = Aa(perm(i)), where T = A^t

  /* Construct a nrows by ncols matrix with given aseq on host. Caller also specifies a nonzero state */
  Mat_SeqAIJKokkos(PetscInt nrows, PetscInt ncols, Mat_SeqAIJ *aseq, PetscObjectState nzstate, PetscBool copyValues = PETSC_TRUE)
  {
    auto exec = PetscGetKokkosExecutionSpace();

    MatScalarKokkosViewHost a_h(aseq->a, aseq->nz);
    MatRowMapKokkosViewHost i_h(const_cast<MatRowMapType *>(aseq->i), nrows + 1);
    MatColIdxKokkosViewHost j_h(aseq->j, aseq->nz);
    MatRowMapKokkosViewHost diag_h(aseq->diag, nrows);

    auto a_d    = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, exec, a_h);
    auto i_d    = Kokkos::create_mirror_view_and_copy(exec, i_h);
    auto j_d    = Kokkos::create_mirror_view_and_copy(exec, j_h);
    auto diag_d = Kokkos::create_mirror_view_and_copy(exec, diag_h);
    a_dual      = MatScalarKokkosDualView(a_d, a_h);
    i_dual      = MatRowMapKokkosDualView(i_d, i_h);
    j_dual      = MatColIdxKokkosDualView(j_d, j_h);
    diag_dual   = MatColIdxKokkosDualView(diag_d, diag_h);

    a_dual.modify_host(); /* Since caller provided values on host */
    if (copyValues) (void)KokkosDualViewSyncDevice(a_dual, exec);

    csrmat = KokkosCsrMatrix("csrmat", ncols, a_d, KokkosCsrGraph(j_d, i_d));
    Init(nzstate);
  }

  /* Construct with a KokkosCsrMatrix. For performance, only i, j are copied to host, but not the matrix values. */
  Mat_SeqAIJKokkos(const KokkosCsrMatrix &csr) : csrmat(csr) /* Shallow-copy csr's views to csrmat */
  {
    auto a_d = csr.values;
    /* Get a non-const version since I don't want to deal with DualView<const T*>, which is not well defined */
    MatRowMapKokkosView i_d(const_cast<MatRowMapType *>(csr.graph.row_map.data()), csr.graph.row_map.extent(0));
    auto                j_d = csr.graph.entries;
    auto                a_h = Kokkos::create_mirror_view(Kokkos::WithoutInitializing, HostMirrorMemorySpace(), a_d);
    auto                i_h = Kokkos::create_mirror_view_and_copy(HostMirrorMemorySpace(), i_d);
    auto                j_h = Kokkos::create_mirror_view_and_copy(HostMirrorMemorySpace(), j_d);

    // diag_dual is set until MatAssemblyEnd() where we copy diag from host to device
    a_dual = MatScalarKokkosDualView(a_d, a_h);
    a_dual.modify_device(); /* since we did not copy a_d to a_h, we mark device has the latest data */
    i_dual                       = MatRowMapKokkosDualView(i_d, i_h);
    j_dual                       = MatColIdxKokkosDualView(j_d, j_h);
    host_aij_allocated_by_kokkos = PETSC_TRUE; /* That means after deleting aijkok, one shouldn't access aijseq->{a,i,j} anymore! */
    Init();
  }

  // Don't use DualView argument types as we want to be sure that a,i,j on host are allocated by Mat_SeqAIJKokkos itself (vs. by users)
  Mat_SeqAIJKokkos(PetscInt nrows, PetscInt ncols, PetscInt nnz, const MatRowMapKokkosView &i_d, const MatColIdxKokkosView &j_d, const MatScalarKokkosView &a_d) : Mat_SeqAIJKokkos(KokkosCsrMatrix("csrmat", nrows, ncols, nnz, a_d, i_d, j_d)) { }

  MatScalarType *a_host_data() { return a_dual.view_host().data(); }
  MatRowMapType *i_host_data() { return i_dual.view_host().data(); }
  MatColIdxType *j_host_data() { return j_dual.view_host().data(); }

  MatScalarType *a_device_data() { return a_dual.view_device().data(); }
  MatRowMapType *i_device_data() { return i_dual.view_device().data(); }
  MatColIdxType *j_device_data() { return j_dual.view_device().data(); }

  MatColIdxType nrows() { return csrmat.numRows(); }
  MatColIdxType ncols() { return csrmat.numCols(); }
  MatRowMapType nnz() { return csrmat.nnz(); }

  /* Change the csrmat size to n */
  void SetColSize(MatColIdxType n) { csrmat = KokkosCsrMatrix("csrmat", n, a_dual.view_device(), csrmat.graph); }

  void SetDiagonal(const MatRowMapType *diag)
  {
    MatRowMapKokkosViewHost diag_h(const_cast<MatRowMapType *>(diag), nrows());
    auto                    diag_d = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), diag_h);
    diag_dual                      = MatRowMapKokkosDualView(diag_d, diag_h);
  }

  /* Shared init stuff */
  void Init(PetscObjectState nzstate = 0)
  {
    nonzerostate      = nzstate;
    transpose_updated = PETSC_FALSE;
    hermitian_updated = PETSC_FALSE;
  }

  PetscErrorCode DestroyMatTranspose(void)
  {
    PetscFunctionBegin;
    csrmatT = KokkosCsrMatrix(); /* Overwrite with empty matrices */
    csrmatH = KokkosCsrMatrix();
    PetscFunctionReturn(PETSC_SUCCESS);
  }
};

struct MatProductData_SeqAIJKokkos {
  KernelHandle kh;
  PetscBool    reusesym;
  MatProductData_SeqAIJKokkos() : reusesym(PETSC_FALSE) { }
};

PETSC_INTERN PetscErrorCode MatSetSeqAIJKokkosWithCSRMatrix(Mat, Mat_SeqAIJKokkos *);
PETSC_INTERN PetscErrorCode MatCreateSeqAIJKokkosWithCSRMatrix(MPI_Comm, Mat_SeqAIJKokkos *, Mat *);
PETSC_INTERN PetscErrorCode MatSeqAIJKokkosMergeMats(Mat, Mat, MatReuse, Mat *);
PETSC_INTERN PetscErrorCode MatSeqAIJKokkosSyncDevice(Mat);
PETSC_INTERN PetscErrorCode MatSeqAIJKokkosGetKokkosCsrMatrix(Mat, KokkosCsrMatrix *);
PETSC_INTERN PetscErrorCode MatCreateSeqAIJKokkosWithKokkosCsrMatrix(MPI_Comm, KokkosCsrMatrix, Mat *);
PETSC_INTERN PetscErrorCode PrintCsrMatrix(const KokkosCsrMatrix &csrmat);
PETSC_INTERN PetscErrorCode MatConvert_SeqAIJ_SeqAIJKokkos(Mat, MatType, MatReuse, Mat *);
PETSC_INTERN PetscErrorCode MatSeqAIJKokkosModifyDevice(Mat);
PETSC_INTERN PetscErrorCode MatSeqAIJKokkosGenerateTranspose_Private(Mat, KokkosCsrMatrix *);

PETSC_INTERN PetscErrorCode MatSeqAIJGetKokkosView(Mat, MatScalarKokkosView *);
PETSC_INTERN PetscErrorCode MatSeqAIJRestoreKokkosView(Mat, MatScalarKokkosView *);
PETSC_INTERN PetscErrorCode MatSeqAIJGetKokkosViewWrite(Mat, MatScalarKokkosView *);
PETSC_INTERN PetscErrorCode MatSeqAIJRestoreKokkosViewWrite(Mat, MatScalarKokkosView *);
PETSC_INTERN PetscErrorCode MatInvertVariableBlockDiagonal_SeqAIJKokkos(Mat, const PetscIntKokkosView &, const PetscIntKokkosView &, const PetscIntKokkosView &, PetscScalarKokkosView &, PetscScalarKokkosView &);
