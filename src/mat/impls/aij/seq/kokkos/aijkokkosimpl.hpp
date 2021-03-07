#if !defined(__SEQAIJKOKKOSIMPL_HPP)
#define __SEQAIJKOKKOSIMPL_HPP

#include "Kokkos_Core.hpp"
#include <Kokkos_DualView.hpp>
#include "KokkosSparse_CrsMatrix.hpp"
#include <petscveckokkos.hpp>

using MatRowMap_t                = PetscInt; /* RowMap, not RowOffset, to distinguish from Kokkos OffsetView */
using MatColumnIndex_t           = PetscInt;
using MatValue_t                 = PetscScalar;
using MatDevice_t                = typename Kokkos::Device<DeviceExecutionSpace,DeviceMemorySpace>;

using KokkosCsrMatrix_t          = typename KokkosSparse::CrsMatrix<MatValue_t,MatColumnIndex_t,MatDevice_t,void/* MemoryTraits */,MatRowMap_t>;
using KokkosCsrGraph_t           = typename KokkosCsrMatrix_t::staticcrsgraph_type;

using MatColumnIndexViewDevice_t = typename KokkosCsrGraph_t::entries_type;
using MatRowMapViewDevice_t      = typename KokkosCsrGraph_t::row_map_type;
using MatValueViewDevice_t       = typename KokkosCsrMatrix_t::values_type;

using MatColumnIndexViewHost_t   = MatColumnIndexViewDevice_t::HostMirror;
using MatRowMapViewHost_t        = MatRowMapViewDevice_t::HostMirror;
using MatValueViewHost_t         = MatValueViewDevice_t::HostMirror;

using MatValueDualView_t         = Kokkos::DualView<MatValue_t*>;
//#include "KokkosSparse_spgemm.hpp"
//using MatMatKernelHandle_t       = KokkosKernels::Experimental::KokkosKernelsHandle<MatRowMap_t, MatColumnIndex_t, MatValue_t, MatDevice_t, DeviceMemorySpace, DeviceMemorySpace>;

struct Mat_SeqAIJKokkos {
  MatRowMapViewHost_t        i_h;
  MatColumnIndexViewHost_t   j_h;
  MatValueViewHost_t         a_h;

  MatRowMapViewDevice_t      i_d;
  MatColumnIndexViewDevice_t j_d;
  MatValueViewDevice_t       a_d;

  MatValueDualView_t         a_dual;

  KokkosCsrMatrix_t          csr;
  PetscObjectState           nonzerostate; /* State of the nonzero pattern (graph) on device */

  Kokkos::View<PetscInt*>    *i_uncompressed_d;
  Kokkos::View<PetscInt*>    *colmap_d; // ugh, this is a parallel construct
  Kokkos::View<PetscSplitCSRDataStructure,DeviceMemorySpace> device_mat_d;

  Kokkos::View<PetscInt*>     *diag_d; // factorizations

  Mat_SeqAIJKokkos(MatColumnIndex_t nrows,MatColumnIndex_t ncols,MatRowMap_t nnz,MatRowMap_t *i,MatColumnIndex_t *j,MatValue_t *a)
   : i_h(i,nrows+1),
     j_h(j,nnz),
     a_h(a,nnz),
     i_d(Kokkos::create_mirror_view_and_copy(DeviceMemorySpace(),i_h)),
     j_d(Kokkos::create_mirror_view_and_copy(DeviceMemorySpace(),j_h)),
     a_d(Kokkos::create_mirror_view_and_copy(DeviceMemorySpace(),a_h)),
     a_dual(a_d,a_h),
     csr("AIJKokkos",nrows,ncols,nnz,a_d,i_d,j_d),
     i_uncompressed_d(NULL),
     colmap_d(NULL),
     device_mat_d(NULL),
     diag_d(NULL)
  {};
};

#endif
