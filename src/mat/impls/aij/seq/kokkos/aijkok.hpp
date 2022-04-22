#ifndef __SEQAIJKOKKOSIMPL_HPP
#define __SEQAIJKOKKOSIMPL_HPP

#include <petscaijdevice.h>
#include <petsc/private/vecimpl_kokkos.hpp>
#include <../src/mat/impls/aij/seq/aij.h>
#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_spiluk.hpp>

/*
   Kokkos::View<struct _n_SplitCSRMat,DefaultMemorySpace> is not handled correctly so we define SplitCSRMat
   for the singular purpose of working around this.
*/
typedef struct _n_SplitCSRMat SplitCSRMat;

using MatRowMapType  = PetscInt;
using MatColIdxType  = PetscInt;
using MatScalarType  = PetscScalar;

template<class MemorySpace> using KokkosCsrMatrixType   = typename KokkosSparse::CrsMatrix<MatScalarType,MatColIdxType,MemorySpace,void/* MemoryTraits */,MatRowMapType>;
template<class MemorySpace> using KokkosCsrGraphType    = typename KokkosCsrMatrixType<MemorySpace>::staticcrsgraph_type;

using KokkosCsrGraph                 = KokkosCsrGraphType<DefaultMemorySpace>;
using KokkosCsrGraphHost             = KokkosCsrGraphType<Kokkos::HostSpace>;

using KokkosCsrMatrix                = KokkosCsrMatrixType<DefaultMemorySpace>;
using KokkosCsrMatrixHost            = KokkosCsrMatrixType<Kokkos::HostSpace>;

using MatRowMapKokkosView            = KokkosCsrGraph::row_map_type::non_const_type;
using MatColIdxKokkosView            = KokkosCsrGraph::entries_type::non_const_type;
using MatScalarKokkosView            = KokkosCsrMatrix::values_type::non_const_type;

using MatRowMapKokkosViewHost        = KokkosCsrGraphHost::row_map_type::non_const_type;
using MatColIdxKokkosViewHost        = KokkosCsrGraphHost::entries_type::non_const_type;
using MatScalarKokkosViewHost        = KokkosCsrMatrixHost::values_type::non_const_type;

using ConstMatRowMapKokkosView       = KokkosCsrGraph::row_map_type::const_type;
using ConstMatColIdxKokkosView       = KokkosCsrGraph::entries_type::const_type;
using ConstMatScalarKokkosView       = KokkosCsrMatrix::values_type::const_type;

using ConstMatRowMapKokkosViewHost   = KokkosCsrGraphHost::row_map_type::const_type;
using ConstMatColIdxKokkosViewHost   = KokkosCsrGraphHost::entries_type::const_type;
using ConstMatScalarKokkosViewHost   = KokkosCsrMatrixHost::values_type::const_type;

using MatRowMapKokkosDualView        = Kokkos::DualView<MatRowMapType*>;
using MatColIdxKokkosDualView        = Kokkos::DualView<MatColIdxType*>;
using MatScalarKokkosDualView        = Kokkos::DualView<MatScalarType*>;

using KernelHandle                   = KokkosKernels::Experimental::KokkosKernelsHandle<MatRowMapType,MatColIdxType,MatScalarType,DefaultExecutionSpace,DefaultMemorySpace,DefaultMemorySpace>;

using KokkosTeamMemberType           = Kokkos::TeamPolicy<DefaultExecutionSpace>::member_type;

/* For mat->spptr of a factorized matrix */
struct Mat_SeqAIJKokkosTriFactors {
  MatRowMapKokkosView       iL_d,iU_d,iLt_d,iUt_d; /* rowmap for L, U, L^t, U^t of A=LU */
  MatColIdxKokkosView       jL_d,jU_d,jLt_d,jUt_d; /* column ids */
  MatScalarKokkosView       aL_d,aU_d,aLt_d,aUt_d; /* matrix values */
  KernelHandle              kh,khL,khU,khLt,khUt;  /* Kernel handles for A, L, U, L^t, U^t */
  PetscBool                 transpose_updated;     /* Are L^T, U^T updated wrt L, U*/
  PetscBool                 sptrsv_symbolic_completed; /* Have we completed the symbolic solve for L and U */
  PetscScalarKokkosView     workVector;

  Mat_SeqAIJKokkosTriFactors(PetscInt n)
    : transpose_updated(PETSC_FALSE),sptrsv_symbolic_completed(PETSC_FALSE),workVector("workVector",n) {}

  ~Mat_SeqAIJKokkosTriFactors() {Destroy();}

  void Destroy() {
    kh.destroy_spiluk_handle();
    khL.destroy_sptrsv_handle();
    khU.destroy_sptrsv_handle();
    khLt.destroy_sptrsv_handle();
    khUt.destroy_sptrsv_handle();
    transpose_updated = sptrsv_symbolic_completed = PETSC_FALSE;
  }
};

/* For mat->spptr of a regular matrix */
struct Mat_SeqAIJKokkos {
  MatRowMapKokkosDualView    i_dual;
  MatColIdxKokkosDualView    j_dual;
  MatScalarKokkosDualView    a_dual;

  KokkosCsrMatrix            csrmat; /* The CSR matrix, used to call KK functions */
  PetscObjectState           nonzerostate; /* State of the nonzero pattern (graph) on device */

  KokkosCsrMatrix            csrmatT,csrmatH; /* Transpose and Hermitian of the matrix (built on demand) */
  PetscBool                  transpose_updated,hermitian_updated; /* Are At, Ah updated wrt the matrix? */

  /* COO stuff */
  PetscCountKokkosView       jmap_d; /* perm[disp+jmap[i]..disp+jmap[i+1]) gives indices of entries in v[] associated with i-th nonzero of the matrix */
  PetscCountKokkosView       perm_d; /* The permutation array in sorting (i,j) by row and then by col */

  Kokkos::View<PetscInt*>         i_uncompressed_d;
  Kokkos::View<PetscInt*>         colmap_d; // ugh, this is a parallel construct
  Kokkos::View<SplitCSRMat,DefaultMemorySpace> device_mat_d;
  Kokkos::View<PetscInt*>         diag_d; // factorizations

  /* Construct a nrows by ncols matrix with nnz nonzeros from the given (i,j,a) on host. Caller also specifies a nonzero state */
  Mat_SeqAIJKokkos(PetscInt nrows,PetscInt ncols,PetscInt nnz,const MatRowMapType *i,MatColIdxType *j,MatScalarType *a,PetscObjectState nzstate,PetscBool copyValues=PETSC_TRUE)
  {
    MatScalarKokkosViewHost    a_h(a,nnz);
    MatRowMapKokkosViewHost    i_h(const_cast<MatRowMapType*>(i),nrows+1);
    MatColIdxKokkosViewHost    j_h(j,nnz);

    auto a_d = Kokkos::create_mirror_view(DefaultMemorySpace(),a_h);
    auto i_d = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(),i_h);
    auto j_d = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(),j_h);

    a_dual   = MatScalarKokkosDualView(a_d,a_h);
    i_dual   = MatRowMapKokkosDualView(i_d,i_h);
    j_dual   = MatColIdxKokkosDualView(j_d,j_h);

    a_dual.modify_host(); /* Since caller provided values on host */
    if (copyValues) a_dual.sync_device();

    csrmat       = KokkosCsrMatrix("csrmat",ncols,a_d,KokkosCsrGraph(j_d,i_d));
    nonzerostate = nzstate;
    transpose_updated = hermitian_updated = PETSC_FALSE;
  }

  /* Construct with a KokkosCsrMatrix. For performance, only i, j are copied to host, but not the matrix values. */
  Mat_SeqAIJKokkos(const KokkosCsrMatrix& csr) : csrmat(csr) /* Shallow-copy csr's views to csrmat */
  {
    auto a_d = csr.values;
    /* Get a non-const version since I don't want to deal with DualView<const T*>, which is not well defined */
    MatRowMapKokkosView i_d(const_cast<MatRowMapType*>(csr.graph.row_map.data()),csr.graph.row_map.extent(0));
    auto j_d = csr.graph.entries;
    auto a_h = Kokkos::create_mirror_view(Kokkos::HostSpace(),a_d);
    auto i_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),i_d);
    auto j_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),j_d);

    a_dual = MatScalarKokkosDualView(a_d,a_h);
    a_dual.modify_device(); /* since we did not copy a_d to a_h, we mark device has the latest data */
    i_dual = MatRowMapKokkosDualView(i_d,i_h);
    j_dual = MatColIdxKokkosDualView(j_d,j_h);
    Init();
  }

  Mat_SeqAIJKokkos(PetscInt nrows,PetscInt ncols,PetscInt nnz,
                   MatRowMapKokkosDualView& i,MatColIdxKokkosDualView& j,MatScalarKokkosDualView a)
    :i_dual(i),j_dual(j),a_dual(a)
  {
    csrmat = KokkosCsrMatrix("csrmat",nrows,ncols,nnz,a.view_device(),i.view_device(),j.view_device());
    Init();
  }

  MatScalarType* a_host_data() {return a_dual.view_host().data();}
  MatRowMapType* i_host_data() {return i_dual.view_host().data();}
  MatColIdxType* j_host_data() {return j_dual.view_host().data();}

  MatScalarType* a_device_data() {return a_dual.view_device().data();}
  MatRowMapType* i_device_data() {return i_dual.view_device().data();}
  MatColIdxType* j_device_data() {return j_dual.view_device().data();}

  MatColIdxType  nrows() {return csrmat.numRows();}
  MatColIdxType  ncols() {return csrmat.numCols();}
  MatRowMapType  nnz()   {return csrmat.nnz();}

  /* Change the csrmat size to n */
  void SetColSize(MatColIdxType n) {csrmat = KokkosCsrMatrix("csrmat",n,a_dual.view_device(),csrmat.graph);}

  void SetUpCOO(const Mat_SeqAIJ *aij) {
    jmap_d = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(),PetscCountKokkosViewHost(aij->jmap,aij->nz+1));
    perm_d = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(),PetscCountKokkosViewHost(aij->perm,aij->Atot));
  }

  /* Shared init stuff */
  void Init(void)
  {
    transpose_updated = hermitian_updated = PETSC_FALSE;
    nonzerostate      = 0;
  }

  PetscErrorCode DestroyMatTranspose(void)
  {
    PetscFunctionBegin;
    csrmatT = KokkosCsrMatrix(); /* Overwrite with empty matrices */
    csrmatH = KokkosCsrMatrix();
    PetscFunctionReturn(0);
  }
};

struct MatProductData_SeqAIJKokkos {
  KernelHandle kh;
  PetscBool    reusesym;
  MatProductData_SeqAIJKokkos() : reusesym(PETSC_FALSE){}
};

PETSC_INTERN PetscErrorCode MatSetSeqAIJKokkosWithCSRMatrix(Mat,Mat_SeqAIJKokkos*);
PETSC_INTERN PetscErrorCode MatCreateSeqAIJKokkosWithCSRMatrix(MPI_Comm,Mat_SeqAIJKokkos*,Mat*);
PETSC_INTERN PetscErrorCode MatSeqAIJKokkosMergeMats(Mat,Mat,MatReuse,Mat*);
PETSC_INTERN PetscErrorCode MatSeqAIJKokkosSyncDevice(Mat);
PETSC_INTERN PetscErrorCode PrintCsrMatrix(const KokkosCsrMatrix& csrmat);
PETSC_INTERN PetscErrorCode MatConvert_SeqAIJ_SeqAIJKokkos(Mat,MatType,MatReuse,Mat*);
PETSC_INTERN PetscErrorCode MatSeqAIJKokkosModifyDevice(Mat);

PETSC_INTERN PetscErrorCode MatSeqAIJGetKokkosView(Mat,MatScalarKokkosView*);
PETSC_INTERN PetscErrorCode MatSeqAIJRestoreKokkosView(Mat,MatScalarKokkosView*);
PETSC_INTERN PetscErrorCode MatSeqAIJGetKokkosViewWrite(Mat,MatScalarKokkosView*);
PETSC_INTERN PetscErrorCode MatSeqAIJRestoreKokkosViewWrite(Mat,MatScalarKokkosView*);
#endif
