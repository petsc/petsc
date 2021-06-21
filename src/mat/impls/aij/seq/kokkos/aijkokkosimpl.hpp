#if !defined(SEQAIJKOKKOSIMPL_HPP)
#define SEQAIJKOKKOSIMPL_HPP

#include <petscaijdevice.h>
#include <petsc/private/vecimpl_kokkos.hpp>
#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_spiluk.hpp>

/*
   Kokkos::View<struct _n_SplitCSRMat,DefaultMemorySpace> is not handled correctly so we define SplitCSRMat
   for the singular purpose of working around this.
*/
typedef struct _n_SplitCSRMat SplitCSRMat;

using MatRowOffsetType    = PetscInt;
using MatColumnIndexType  = PetscInt;
using MatValueType        = PetscScalar;

template<class MemorySpace> using KokkosCsrMatrixType   = typename KokkosSparse::CrsMatrix<MatValueType,MatColumnIndexType,MemorySpace,void/* MemoryTraits */,MatRowOffsetType>;
template<class MemorySpace> using KokkosCsrGraphType    = typename KokkosCsrMatrixType<MemorySpace>::staticcrsgraph_type;

using KokkosCsrGraph                      = KokkosCsrGraphType<DefaultMemorySpace>;
using KokkosCsrMatrix                     = KokkosCsrMatrixType<DefaultMemorySpace>;

using KokkosCsrGraphHost                  = KokkosCsrGraphType<DefaultMemorySpace>::HostMirror;

using ConstMatColumnIndexKokkosView       = KokkosCsrGraph::entries_type;
using ConstMatRowOffsetKokkosView         = KokkosCsrGraph::row_map_type;
using ConstMatValueKokkosView             = KokkosCsrMatrix::values_type;

using MatColumnIndexKokkosView            = KokkosCsrGraph::entries_type::non_const_type;
using MatRowOffsetKokkosView              = KokkosCsrGraph::row_map_type::non_const_type;
using MatValueKokkosView                  = KokkosCsrMatrix::values_type::non_const_type;

using MatColumnIndexKokkosViewHost        = MatColumnIndexKokkosView::HostMirror;
using MatRowOffsetKokkosViewHost          = MatRowOffsetKokkosView::HostMirror;
using MatValueKokkosViewHost              = MatValueKokkosView::HostMirror;

using MatValueKokkosDualView              = Kokkos::DualView<MatValueType*>;

using KernelHandle                        = KokkosKernels::Experimental::KokkosKernelsHandle<MatRowOffsetType,MatColumnIndexType,MatValueType,DefaultExecutionSpace,DefaultMemorySpace,DefaultMemorySpace>;

struct Mat_SeqAIJKokkosTriFactors {
  MatRowOffsetKokkosView         iL_d,iU_d,iLt_d,iUt_d; /* rowmap for L, U, L^t, U^t of A=LU */
  MatColumnIndexKokkosView       jL_d,jU_d,jLt_d,jUt_d; /* column ids */
  MatValueKokkosView             aL_d,aU_d,aLt_d,aUt_d; /* matrix values */
  KernelHandle                   kh,khL,khU,khLt,khUt;  /* Kernel handles for A, L, U, L^t, U^t */
  PetscBool                      transpose_updated;     /* Are L^T, U^T updated wrt L, U*/
  PetscBool                      sptrsv_symbolic_completed; /* Have we completed the symbolic solve for L and U */
  PetscScalarKokkosView          workVector;

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

struct Mat_SeqAIJKokkos {
  MatRowOffsetKokkosViewHost     i_h;
  MatRowOffsetKokkosView         i_d;

  MatColumnIndexKokkosViewHost   j_h;
  MatColumnIndexKokkosView       j_d;

  MatValueKokkosViewHost         a_h;
  MatValueKokkosView             a_d;

  MatValueKokkosDualView         a_dual;

  KokkosCsrGraphHost             csrgraph_h;
  KokkosCsrGraph                 csrgraph_d;

  KokkosCsrMatrix                csrmat; /* The CSR matrix */
  PetscObjectState               nonzerostate; /* State of the nonzero pattern (graph) on device */

  Mat                            At,Ah; /* Transpose and Hermitian of the matrix in MATAIJKOKKOS type (built on demand) */
  PetscBool                      transpose_updated,hermitian_updated; /* Are At, Ah updated wrt the matrix? */

  Kokkos::View<PetscInt*>        *i_uncompressed_d;
  Kokkos::View<PetscInt*>        *colmap_d; // ugh, this is a parallel construct
  Kokkos::View<SplitCSRMat,DefaultMemorySpace> device_mat_d;
  Kokkos::View<PetscInt*>        *diag_d; // factorizations

   /* Construct a nrows by ncols matrix of nnz nonzeros with (i,j,a) for the CSR */
  Mat_SeqAIJKokkos(MatColumnIndexType nrows,MatColumnIndexType ncols,MatRowOffsetType nnz,MatRowOffsetType *i,MatColumnIndexType *j,MatValueType *a)
   : i_h(i,nrows+1),j_h(j,nnz),a_h(a,nnz),At(NULL),Ah(NULL),transpose_updated(PETSC_FALSE),hermitian_updated(PETSC_FALSE),
     i_uncompressed_d(NULL),colmap_d(NULL),device_mat_d(NULL),diag_d(NULL)
  {
     i_d        = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(),i_h);
     j_d        = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(),j_h);
     a_d        = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(),a_h);
     csrgraph_d = KokkosCsrGraph(j_d,i_d);
     csrgraph_h = KokkosCsrGraphHost(j_h,i_h);
     a_dual     = MatValueKokkosDualView(a_d,a_h);
     csrmat     = KokkosCsrMatrix("csrmat",ncols,a_d,csrgraph_d);
  }

  ~Mat_SeqAIJKokkos()
  {
    DestroyMatTranspose();
  }

  PetscErrorCode DestroyMatTranspose(void)
  {
    PetscErrorCode ierr;
    PetscFunctionBegin;
    ierr = MatDestroy(&At);CHKERRQ(ierr);
    ierr = MatDestroy(&Ah);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
};

#endif
