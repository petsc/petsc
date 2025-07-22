#pragma once

/* types used by all PETSc Kokkos code */

#include <petscsystypes.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <Kokkos_OffsetView.hpp>

// the pool is defined in veckok.kokkos.cxx as it is currently only used there
PETSC_SINGLE_LIBRARY_INTERN PetscScalar *PetscScalarPool;
PETSC_SINGLE_LIBRARY_INTERN PetscInt     PetscScalarPoolSize;

using DefaultExecutionSpace = Kokkos::DefaultExecutionSpace;
using DefaultMemorySpace    = Kokkos::DefaultExecutionSpace::memory_space;
using HostMirrorMemorySpace = Kokkos::DualView<PetscScalar *>::host_mirror_space::memory_space;

/* Define a macro if DefaultMemorySpace and HostMirrorMemorySpace are the same */
#if defined(KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_SERIAL) || defined(KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_OPENMP) || defined(KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_THREADS) || defined(KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_HPX) || defined(KOKKOS_ENABLE_IMPL_CUDA_UNIFIED_MEMORY) || defined(KOKKOS_IMPL_HIP_UNIFIED_MEMORY)
  #define KOKKOS_ENABLE_UNIFIED_MEMORY
#endif

/* 1 to 4D PetscScalar Kokkos Views */
template <class MemorySpace>
using PetscScalarKokkosViewType = Kokkos::View<PetscScalar *, MemorySpace>;
template <class MemorySpace>
using PetscScalarKokkosView1DType = Kokkos::View<PetscScalar *, MemorySpace>;
template <class MemorySpace>
using PetscScalarKokkosView2DType = Kokkos::View<PetscScalar **, Kokkos::LayoutRight, MemorySpace>;
template <class MemorySpace>
using PetscScalarKokkosView3DType = Kokkos::View<PetscScalar ***, Kokkos::LayoutRight, MemorySpace>;
template <class MemorySpace>
using PetscScalarKokkosView4DType = Kokkos::View<PetscScalar ****, Kokkos::LayoutRight, MemorySpace>;

template <class MemorySpace>
using ConstPetscScalarKokkosViewType = Kokkos::View<const PetscScalar *, MemorySpace>;
template <class MemorySpace>
using ConstPetscScalarKokkosView1DType = Kokkos::View<const PetscScalar *, MemorySpace>;
template <class MemorySpace>
using ConstPetscScalarKokkosView2DType = Kokkos::View<const PetscScalar **, Kokkos::LayoutRight, MemorySpace>;
template <class MemorySpace>
using ConstPetscScalarKokkosView3DType = Kokkos::View<const PetscScalar ***, Kokkos::LayoutRight, MemorySpace>;
template <class MemorySpace>
using ConstPetscScalarKokkosView4DType = Kokkos::View<const PetscScalar ****, Kokkos::LayoutRight, MemorySpace>;

/* 1 to 4D PetscScalar Kokkos OffsetViews */
template <class MemorySpace>
using PetscScalarKokkosOffsetViewType = Kokkos::Experimental::OffsetView<PetscScalar *, MemorySpace>;
template <class MemorySpace>
using PetscScalarKokkosOffsetView1DType = Kokkos::Experimental::OffsetView<PetscScalar *, MemorySpace>;
template <class MemorySpace>
using PetscScalarKokkosOffsetView2DType = Kokkos::Experimental::OffsetView<PetscScalar **, Kokkos::LayoutRight, MemorySpace>;
template <class MemorySpace>
using PetscScalarKokkosOffsetView3DType = Kokkos::Experimental::OffsetView<PetscScalar ***, Kokkos::LayoutRight, MemorySpace>;
template <class MemorySpace>
using PetscScalarKokkosOffsetView4DType = Kokkos::Experimental::OffsetView<PetscScalar ****, Kokkos::LayoutRight, MemorySpace>;

template <class MemorySpace>
using ConstPetscScalarKokkosOffsetViewType = Kokkos::Experimental::OffsetView<const PetscScalar *, MemorySpace>;
template <class MemorySpace>
using ConstPetscScalarKokkosOffsetView1DType = Kokkos::Experimental::OffsetView<const PetscScalar *, MemorySpace>;
template <class MemorySpace>
using ConstPetscScalarKokkosOffsetView2DType = Kokkos::Experimental::OffsetView<const PetscScalar **, Kokkos::LayoutRight, MemorySpace>;
template <class MemorySpace>
using ConstPetscScalarKokkosOffsetView3DType = Kokkos::Experimental::OffsetView<const PetscScalar ***, Kokkos::LayoutRight, MemorySpace>;
template <class MemorySpace>
using ConstPetscScalarKokkosOffsetView4DType = Kokkos::Experimental::OffsetView<const PetscScalar ****, Kokkos::LayoutRight, MemorySpace>;

using PetscScalarKokkosDualView = Kokkos::DualView<PetscScalar *>;

/* Shortcut types for Views in the default space and host space */
using PetscScalarKokkosView   = PetscScalarKokkosViewType<DefaultMemorySpace>;
using PetscScalarKokkosView1D = PetscScalarKokkosView1DType<DefaultMemorySpace>;
using PetscScalarKokkosView2D = PetscScalarKokkosView2DType<DefaultMemorySpace>;
using PetscScalarKokkosView3D = PetscScalarKokkosView3DType<DefaultMemorySpace>;
using PetscScalarKokkosView4D = PetscScalarKokkosView4DType<DefaultMemorySpace>;

using PetscScalarKokkosViewHost   = PetscScalarKokkosViewType<HostMirrorMemorySpace>;
using PetscScalarKokkosView1DHost = PetscScalarKokkosView1DType<HostMirrorMemorySpace>;
using PetscScalarKokkosView2DHost = PetscScalarKokkosView2DType<HostMirrorMemorySpace>;
using PetscScalarKokkosView3DHost = PetscScalarKokkosView3DType<HostMirrorMemorySpace>;
using PetscScalarKokkosView4DHost = PetscScalarKokkosView4DType<HostMirrorMemorySpace>;

using ConstPetscScalarKokkosView   = ConstPetscScalarKokkosViewType<DefaultMemorySpace>;
using ConstPetscScalarKokkosView1D = ConstPetscScalarKokkosView1DType<DefaultMemorySpace>;
using ConstPetscScalarKokkosView2D = ConstPetscScalarKokkosView2DType<DefaultMemorySpace>;
using ConstPetscScalarKokkosView3D = ConstPetscScalarKokkosView3DType<DefaultMemorySpace>;
using ConstPetscScalarKokkosView4D = ConstPetscScalarKokkosView4DType<DefaultMemorySpace>;

using ConstPetscScalarKokkosViewHost   = ConstPetscScalarKokkosViewType<HostMirrorMemorySpace>;
using ConstPetscScalarKokkosView1DHost = ConstPetscScalarKokkosView1DType<HostMirrorMemorySpace>;
using ConstPetscScalarKokkosView2DHost = ConstPetscScalarKokkosView2DType<HostMirrorMemorySpace>;
using ConstPetscScalarKokkosView3DHost = ConstPetscScalarKokkosView3DType<HostMirrorMemorySpace>;
using ConstPetscScalarKokkosView4DHost = ConstPetscScalarKokkosView4DType<HostMirrorMemorySpace>;

/* Shortcut types for OffsetViews in the default space and host space */
using PetscScalarKokkosOffsetView   = PetscScalarKokkosOffsetViewType<DefaultMemorySpace>;
using PetscScalarKokkosOffsetView1D = PetscScalarKokkosOffsetView1DType<DefaultMemorySpace>;
using PetscScalarKokkosOffsetView2D = PetscScalarKokkosOffsetView2DType<DefaultMemorySpace>;
using PetscScalarKokkosOffsetView3D = PetscScalarKokkosOffsetView3DType<DefaultMemorySpace>;
using PetscScalarKokkosOffsetView4D = PetscScalarKokkosOffsetView4DType<DefaultMemorySpace>;

using PetscScalarKokkosOffsetViewHost   = PetscScalarKokkosOffsetViewType<HostMirrorMemorySpace>;
using PetscScalarKokkosOffsetView1DHost = PetscScalarKokkosOffsetView1DType<HostMirrorMemorySpace>;
using PetscScalarKokkosOffsetView2DHost = PetscScalarKokkosOffsetView2DType<HostMirrorMemorySpace>;
using PetscScalarKokkosOffsetView3DHost = PetscScalarKokkosOffsetView3DType<HostMirrorMemorySpace>;
using PetscScalarKokkosOffsetView4DHost = PetscScalarKokkosOffsetView4DType<HostMirrorMemorySpace>;

using ConstPetscScalarKokkosOffsetView   = ConstPetscScalarKokkosOffsetViewType<DefaultMemorySpace>;
using ConstPetscScalarKokkosOffsetView1D = ConstPetscScalarKokkosOffsetView1DType<DefaultMemorySpace>;
using ConstPetscScalarKokkosOffsetView2D = ConstPetscScalarKokkosOffsetView2DType<DefaultMemorySpace>;
using ConstPetscScalarKokkosOffsetView3D = ConstPetscScalarKokkosOffsetView3DType<DefaultMemorySpace>;
using ConstPetscScalarKokkosOffsetView4D = ConstPetscScalarKokkosOffsetView4DType<DefaultMemorySpace>;

using ConstPetscScalarKokkosOffsetViewHost   = ConstPetscScalarKokkosOffsetViewType<HostMirrorMemorySpace>;
using ConstPetscScalarKokkosOffsetView1DHost = ConstPetscScalarKokkosOffsetView1DType<HostMirrorMemorySpace>;
using ConstPetscScalarKokkosOffsetView2DHost = ConstPetscScalarKokkosOffsetView2DType<HostMirrorMemorySpace>;
using ConstPetscScalarKokkosOffsetView3DHost = ConstPetscScalarKokkosOffsetView3DType<HostMirrorMemorySpace>;
using ConstPetscScalarKokkosOffsetView4DHost = ConstPetscScalarKokkosOffsetView4DType<HostMirrorMemorySpace>;

using PetscIntKokkosView       = Kokkos::View<PetscInt *, DefaultMemorySpace>;
using PetscIntKokkosViewHost   = Kokkos::View<PetscInt *, HostMirrorMemorySpace>;
using PetscIntKokkosDualView   = Kokkos::DualView<PetscInt *>;
using PetscCountKokkosView     = Kokkos::View<PetscCount *, DefaultMemorySpace>;
using PetscCountKokkosViewHost = Kokkos::View<PetscCount *, HostMirrorMemorySpace>;

// Sync a Kokkos::DualView<Type *> to HostMirrorMemorySpace in execution space <exec>
// If <MemorySpace> is HostMirrorMemorySpace, fence the exec so that the data on host is immediately available.
template <typename Type>
static PetscErrorCode KokkosDualViewSyncHost(Kokkos::DualView<Type *> &v_dual, const Kokkos::DefaultExecutionSpace &exec)
{
  size_t bytes = v_dual.extent(0) * sizeof(Type);

  PetscFunctionBegin;
  if (v_dual.need_sync_host()) {
    PetscCallCXX(v_dual.sync_host(exec));
    if (!std::is_same_v<DefaultMemorySpace, HostMirrorMemorySpace>) PetscCall(PetscLogGpuToCpu(bytes));
  }
  // even if v_d and v_h share the same memory (as on AMD MI300A) and thus we don't need to sync_host,
  // we still need to fence the execution space as v_d might being populated by some async kernel,
  // and we need to finish it.
  PetscCallCXX(exec.fence());
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename Type>
static PetscErrorCode KokkosDualViewSyncDevice(Kokkos::DualView<Type *> &v_dual, const Kokkos::DefaultExecutionSpace &exec)
{
  size_t bytes = v_dual.extent(0) * sizeof(Type);

  PetscFunctionBegin;
  if (v_dual.need_sync_device()) {
    PetscCallCXX(v_dual.sync_device(exec));
    if (!std::is_same_v<DefaultMemorySpace, HostMirrorMemorySpace>) PetscCall(PetscLogCpuToGpu(bytes));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
