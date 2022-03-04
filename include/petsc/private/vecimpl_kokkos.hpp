#if !defined(__VECIMPL_KOKKOS_HPP)
#define __VECIMPL_KOKKOS_HPP

/* types used by all petsc kokkos code */

#include <petscvec_kokkos.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <Kokkos_OffsetView.hpp>

using DefaultExecutionSpace   = Kokkos::DefaultExecutionSpace;
using DefaultMemorySpace      = Kokkos::DefaultExecutionSpace::memory_space;

/* Define a macro if DefaultMemorySpace is HostSpace */
#if defined(KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_SERIAL) || defined(KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_OPENMP) || defined(KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_THREADS) || defined(KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_HPX)
  #define KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_HOST
#endif

/* 1 to 4D PetscScalar Kokkos Views */
template<class MemorySpace> using PetscScalarKokkosViewType                  = Kokkos::View<PetscScalar*,MemorySpace>;
template<class MemorySpace> using PetscScalarKokkosView1DType                = Kokkos::View<PetscScalar*,MemorySpace>;
template<class MemorySpace> using PetscScalarKokkosView2DType                = Kokkos::View<PetscScalar**,Kokkos::LayoutRight,MemorySpace>;
template<class MemorySpace> using PetscScalarKokkosView3DType                = Kokkos::View<PetscScalar***,Kokkos::LayoutRight,MemorySpace>;
template<class MemorySpace> using PetscScalarKokkosView4DType                = Kokkos::View<PetscScalar****,Kokkos::LayoutRight,MemorySpace>;

template<class MemorySpace> using ConstPetscScalarKokkosViewType             = Kokkos::View<const PetscScalar*,MemorySpace>;
template<class MemorySpace> using ConstPetscScalarKokkosView1DType           = Kokkos::View<const PetscScalar*,MemorySpace>;
template<class MemorySpace> using ConstPetscScalarKokkosView2DType           = Kokkos::View<const PetscScalar**,Kokkos::LayoutRight,MemorySpace>;
template<class MemorySpace> using ConstPetscScalarKokkosView3DType           = Kokkos::View<const PetscScalar***,Kokkos::LayoutRight,MemorySpace>;
template<class MemorySpace> using ConstPetscScalarKokkosView4DType           = Kokkos::View<const PetscScalar****,Kokkos::LayoutRight,MemorySpace>;

/* 1 to 4D PetscScalar Kokkos OffsetViews */
template<class MemorySpace> using PetscScalarKokkosOffsetViewType            = Kokkos::Experimental::OffsetView<PetscScalar*,MemorySpace>;
template<class MemorySpace> using PetscScalarKokkosOffsetView1DType          = Kokkos::Experimental::OffsetView<PetscScalar*,MemorySpace>;
template<class MemorySpace> using PetscScalarKokkosOffsetView2DType          = Kokkos::Experimental::OffsetView<PetscScalar**,Kokkos::LayoutRight,MemorySpace>;
template<class MemorySpace> using PetscScalarKokkosOffsetView3DType          = Kokkos::Experimental::OffsetView<PetscScalar***,Kokkos::LayoutRight,MemorySpace>;
template<class MemorySpace> using PetscScalarKokkosOffsetView4DType          = Kokkos::Experimental::OffsetView<PetscScalar****,Kokkos::LayoutRight,MemorySpace>;

template<class MemorySpace> using ConstPetscScalarKokkosOffsetViewType       = Kokkos::Experimental::OffsetView<const PetscScalar*,MemorySpace>;
template<class MemorySpace> using ConstPetscScalarKokkosOffsetView1DType     = Kokkos::Experimental::OffsetView<const PetscScalar*,MemorySpace>;
template<class MemorySpace> using ConstPetscScalarKokkosOffsetView2DType     = Kokkos::Experimental::OffsetView<const PetscScalar**,Kokkos::LayoutRight,MemorySpace>;
template<class MemorySpace> using ConstPetscScalarKokkosOffsetView3DType     = Kokkos::Experimental::OffsetView<const PetscScalar***,Kokkos::LayoutRight,MemorySpace>;
template<class MemorySpace> using ConstPetscScalarKokkosOffsetView4DType     = Kokkos::Experimental::OffsetView<const PetscScalar****,Kokkos::LayoutRight,MemorySpace>;

using PetscScalarKokkosDualView                  = Kokkos::DualView<PetscScalar*>;

/* Shortcut types for Views in the default space and host space */
using PetscScalarKokkosView                      = PetscScalarKokkosViewType<DefaultMemorySpace>;
using PetscScalarKokkosView1D                    = PetscScalarKokkosView1DType<DefaultMemorySpace>;
using PetscScalarKokkosView2D                    = PetscScalarKokkosView2DType<DefaultMemorySpace>;
using PetscScalarKokkosView3D                    = PetscScalarKokkosView3DType<DefaultMemorySpace>;
using PetscScalarKokkosView4D                    = PetscScalarKokkosView4DType<DefaultMemorySpace>;

using PetscScalarKokkosViewHost                  = PetscScalarKokkosViewType<Kokkos::HostSpace>;
using PetscScalarKokkosView1DHost                = PetscScalarKokkosView1DType<Kokkos::HostSpace>;
using PetscScalarKokkosView2DHost                = PetscScalarKokkosView2DType<Kokkos::HostSpace>;
using PetscScalarKokkosView3DHost                = PetscScalarKokkosView3DType<Kokkos::HostSpace>;
using PetscScalarKokkosView4DHost                = PetscScalarKokkosView4DType<Kokkos::HostSpace>;

using ConstPetscScalarKokkosView                 = ConstPetscScalarKokkosViewType<DefaultMemorySpace>;
using ConstPetscScalarKokkosView1D               = ConstPetscScalarKokkosView1DType<DefaultMemorySpace>;
using ConstPetscScalarKokkosView2D               = ConstPetscScalarKokkosView2DType<DefaultMemorySpace>;
using ConstPetscScalarKokkosView3D               = ConstPetscScalarKokkosView3DType<DefaultMemorySpace>;
using ConstPetscScalarKokkosView4D               = ConstPetscScalarKokkosView4DType<DefaultMemorySpace>;

using ConstPetscScalarKokkosViewHost             = ConstPetscScalarKokkosViewType<Kokkos::HostSpace>;
using ConstPetscScalarKokkosView1DHost           = ConstPetscScalarKokkosView1DType<Kokkos::HostSpace>;
using ConstPetscScalarKokkosView2DHost           = ConstPetscScalarKokkosView2DType<Kokkos::HostSpace>;
using ConstPetscScalarKokkosView3DHost           = ConstPetscScalarKokkosView3DType<Kokkos::HostSpace>;
using ConstPetscScalarKokkosView4DHost           = ConstPetscScalarKokkosView4DType<Kokkos::HostSpace>;

/* Shortcut types for OffsetViews in the default space and host space */
using PetscScalarKokkosOffsetView                = PetscScalarKokkosOffsetViewType<DefaultMemorySpace>;
using PetscScalarKokkosOffsetView1D              = PetscScalarKokkosOffsetView1DType<DefaultMemorySpace>;
using PetscScalarKokkosOffsetView2D              = PetscScalarKokkosOffsetView2DType<DefaultMemorySpace>;
using PetscScalarKokkosOffsetView3D              = PetscScalarKokkosOffsetView3DType<DefaultMemorySpace>;
using PetscScalarKokkosOffsetView4D              = PetscScalarKokkosOffsetView4DType<DefaultMemorySpace>;

using PetscScalarKokkosOffsetViewHost            = PetscScalarKokkosOffsetViewType<Kokkos::HostSpace>;
using PetscScalarKokkosOffsetView1DHost          = PetscScalarKokkosOffsetView1DType<Kokkos::HostSpace>;
using PetscScalarKokkosOffsetView2DHost          = PetscScalarKokkosOffsetView2DType<Kokkos::HostSpace>;
using PetscScalarKokkosOffsetView3DHost          = PetscScalarKokkosOffsetView3DType<Kokkos::HostSpace>;
using PetscScalarKokkosOffsetView4DHost          = PetscScalarKokkosOffsetView4DType<Kokkos::HostSpace>;

using ConstPetscScalarKokkosOffsetView           = ConstPetscScalarKokkosOffsetViewType<DefaultMemorySpace>;
using ConstPetscScalarKokkosOffsetView1D         = ConstPetscScalarKokkosOffsetView1DType<DefaultMemorySpace>;
using ConstPetscScalarKokkosOffsetView2D         = ConstPetscScalarKokkosOffsetView2DType<DefaultMemorySpace>;
using ConstPetscScalarKokkosOffsetView3D         = ConstPetscScalarKokkosOffsetView3DType<DefaultMemorySpace>;
using ConstPetscScalarKokkosOffsetView4D         = ConstPetscScalarKokkosOffsetView4DType<DefaultMemorySpace>;

using ConstPetscScalarKokkosOffsetViewHost       = ConstPetscScalarKokkosOffsetViewType<Kokkos::HostSpace>;
using ConstPetscScalarKokkosOffsetView1DHost     = ConstPetscScalarKokkosOffsetView1DType<Kokkos::HostSpace>;
using ConstPetscScalarKokkosOffsetView2DHost     = ConstPetscScalarKokkosOffsetView2DType<Kokkos::HostSpace>;
using ConstPetscScalarKokkosOffsetView3DHost     = ConstPetscScalarKokkosOffsetView3DType<Kokkos::HostSpace>;
using ConstPetscScalarKokkosOffsetView4DHost     = ConstPetscScalarKokkosOffsetView4DType<Kokkos::HostSpace>;

#endif
