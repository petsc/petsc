#if !defined(__SEQAIJKOKKOSIMPL_HPP)
#define __SEQAIJKOKKOSIMPL_HPP

#include <petsc/private/vecimpl_kokkos.hpp>
#include <KokkosSparse_CrsMatrix.hpp>

using MatRowOffsetType    = PetscInt;
using MatColumnIndexType  = PetscInt;
using MatValueType        = PetscScalar;

template<class MemorySpace> using KokkosCsrMatrixType   = typename KokkosSparse::CrsMatrix<MatValueType,MatColumnIndexType,MemorySpace,void/* MemoryTraits */,MatRowOffsetType>;
template<class MemorySpace> using KokkosCsrGraphType    = typename KokkosCsrMatrixType<MemorySpace>::staticcrsgraph_type;

using KokkosCsrGraph                      = KokkosCsrGraphType<DefaultMemorySpace>;
using KokkosCsrMatrix                     = KokkosCsrMatrixType<DefaultMemorySpace>;

using KokkosCsrGraphHost                  = KokkosCsrGraphType<DefaultMemorySpace>::HostMirror;

using MatColumnIndexKokkosView            = KokkosCsrGraph::entries_type;
using MatRowOffsetKokkosView              = KokkosCsrGraph::row_map_type;
using MatValueKokkosView                  = KokkosCsrMatrix::values_type;

using MatColumnIndexKokkosViewHost        = MatColumnIndexKokkosView::HostMirror;
using MatRowOffsetKokkosViewHost          = MatRowOffsetKokkosView::HostMirror;
using MatValueKokkosViewHost              = MatValueKokkosView::HostMirror;

using MatValueKokkosDualView              = Kokkos::DualView<MatValueType*>;

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
  PetscBool                      need_sync_At_values,need_sync_Ah_values; /* A's values are updated but At, Ah are not */

  Kokkos::View<PetscInt*>        *i_uncompressed_d;
  Kokkos::View<PetscInt*>        *colmap_d; // ugh, this is a parallel construct
  Kokkos::View<PetscSplitCSRDataStructure,DefaultMemorySpace> device_mat_d;
  Kokkos::View<PetscInt*>        *diag_d; // factorizations

  /* Construct a nrows by ncols matrix of nnz nonzeros with (i,j,a) for the CSR */
  Mat_SeqAIJKokkos(MatColumnIndexType nrows,MatColumnIndexType ncols,MatRowOffsetType nnz,MatRowOffsetType *i,MatColumnIndexType *j,MatValueType *a)
   : i_h(i,nrows+1),j_h(j,nnz),a_h(a,nnz),At(NULL),Ah(NULL),need_sync_At_values(PETSC_FALSE),need_sync_Ah_values(PETSC_FALSE),
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
