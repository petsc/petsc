#ifndef PETSCVECCUPMIMPL_H
#define PETSCVECCUPMIMPL_H

#include <petsc/private/vecimpl.h>
#include <../src/vec/vec/impls/dvecimpl.h> // for Vec_Seq

#if PetscDefined(HAVE_NVSHMEM)
PETSC_INTERN PetscErrorCode PetscNvshmemInitializeCheck(void);
PETSC_INTERN PetscErrorCode PetscNvshmemMalloc(size_t, void **);
PETSC_INTERN PetscErrorCode PetscNvshmemCalloc(size_t, void **);
PETSC_INTERN PetscErrorCode PetscNvshmemFree_Private(void *);
  #define PetscNvshmemFree(ptr) ((PetscErrorCode)((ptr) && (PetscNvshmemFree_Private(ptr) || ((ptr) = PETSC_NULLPTR, PETSC_SUCCESS))))
PETSC_INTERN PetscErrorCode PetscNvshmemSum(PetscInt, PetscScalar *, const PetscScalar *);
PETSC_INTERN PetscErrorCode PetscNvshmemMax(PetscInt, PetscReal *, const PetscReal *);
PETSC_INTERN PetscErrorCode VecNormAsync_NVSHMEM(Vec, NormType, PetscReal *);
PETSC_INTERN PetscErrorCode VecAllocateNVSHMEM_SeqCUDA(Vec);
#else
  #define PetscNvshmemFree(ptr) PETSC_SUCCESS
#endif

#if defined(__cplusplus) && PetscDefined(HAVE_DEVICE)
  #include <petsc/private/deviceimpl.h>
  #include <petsc/private/cupmobject.hpp>
  #include <petsc/private/cupmblasinterface.hpp>

  #include <petsc/private/cpp/functional.hpp>

  #include <limits> // std::numeric_limits

namespace Petsc
{

namespace vec
{

namespace cupm
{

namespace impl
{

namespace
{

struct no_op {
  template <typename... T>
  constexpr PetscErrorCode operator()(T &&...) const noexcept
  {
    return PETSC_SUCCESS;
  }
};

template <typename T>
struct CooPair {
  using value_type = T;
  using size_type  = PetscCount;

  value_type *&device;
  value_type *&host;
  size_type    size;
};

template <typename U>
static constexpr CooPair<U> make_coo_pair(U *&device, U *&host, PetscCount size) noexcept
{
  return {device, host, size};
}

} // anonymous namespace

// forward declarations
template <device::cupm::DeviceType>
class VecSeq_CUPM;
template <device::cupm::DeviceType>
class VecMPI_CUPM;

// ==========================================================================================
// Vec_CUPMBase
//
// Base class for the VecSeq and VecMPI CUPM implementations. On top of the usual DeviceType
// template parameter it also uses CRTP to be able to use values/calls specific to either
// VecSeq or VecMPI. This is in effect "inside-out" polymorphism.
// ==========================================================================================
template <device::cupm::DeviceType T, typename Derived>
class Vec_CUPMBase : protected device::cupm::impl::CUPMObject<T> {
public:
  PETSC_CUPMOBJECT_HEADER(T);

  // ==========================================================================================
  // Vec_CUPMBase::VectorArray
  //
  // RAII versions of the get/restore array routines. Determines constness of the pointer type,
  // holds the pointer itself provides the implicit conversion operator
  // ==========================================================================================
  template <PetscMemType, PetscMemoryAccessMode>
  class VectorArray;

protected:
  static PetscErrorCode VecView_Debug(Vec v, const char *message = "") noexcept
  {
    const auto   pobj  = PetscObjectCast(v);
    const auto   vimpl = VecIMPLCast(v);
    const auto   vcu   = VecCUPMCast(v);
    PetscMemType mtype;
    MPI_Comm     comm;

    PetscFunctionBegin;
    PetscValidPointer(vimpl, 1);
    PetscValidPointer(vcu, 1);
    PetscCall(PetscObjectGetComm(pobj, &comm));
    PetscCall(PetscPrintf(comm, "---------- %s ----------\n", message));
    PetscCall(PetscObjectPrintClassNamePrefixType(pobj, PETSC_VIEWER_STDOUT_(comm)));
    PetscCall(PetscPrintf(comm, "Address:             %p\n", v));
    PetscCall(PetscPrintf(comm, "Size:                %" PetscInt_FMT "\n", v->map->n));
    PetscCall(PetscPrintf(comm, "Offload mask:        %s\n", PetscOffloadMaskToString(v->offloadmask)));
    PetscCall(PetscPrintf(comm, "Host ptr:            %p\n", vimpl->array));
    PetscCall(PetscPrintf(comm, "Device ptr:          %p\n", vcu->array_d));
    PetscCall(PetscPrintf(comm, "Device alloced ptr:  %p\n", vcu->array_allocated_d));
    PetscCall(PetscCUPMGetMemType(vcu->array_d, &mtype));
    PetscCall(PetscPrintf(comm, "dptr is device mem?  %s\n", PetscBools[static_cast<PetscBool>(PetscMemTypeDevice(mtype))]));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  // Delete the allocated device array if required and replace it with the given array
  static PetscErrorCode ResetAllocatedDevicePtr_(PetscDeviceContext, Vec, PetscScalar * = nullptr) noexcept;
  // Check either the host or device impl pointer is allocated and allocate it if
  // isn't. CastFunctionType casts the Vec to the required type and returns the pointer
  template <typename CastFunctionType>
  static PetscErrorCode VecAllocateCheck_(Vec, void *&, CastFunctionType &&) noexcept;
  // Check the CUPM part (v->spptr) is allocated, otherwise allocate it
  static PetscErrorCode VecCUPMAllocateCheck_(Vec) noexcept;
  // Check the Host part (v->data) is allocated, otherwise allocate it
  static PetscErrorCode VecIMPLAllocateCheck_(Vec) noexcept;
  // Check the Host array is allocated, otherwise allocate it
  static PetscErrorCode HostAllocateCheck_(PetscDeviceContext, Vec) noexcept;
  // Check the CUPM array is allocated, otherwise allocate it
  static PetscErrorCode DeviceAllocateCheck_(PetscDeviceContext, Vec) noexcept;
  // Copy HTOD, allocating device if necessary
  static PetscErrorCode CopyToDevice_(PetscDeviceContext, Vec, bool = false) noexcept;
  // Copy DTOH, allocating host if necessary
  static PetscErrorCode CopyToHost_(PetscDeviceContext, Vec, bool = false) noexcept;

public:
  struct Vec_CUPM {
    PetscScalar *array_d;           // gpu data
    PetscScalar *array_allocated_d; // does PETSc own the array ptr?
    PetscBool    nvshmem;           // is array allocated in nvshmem? It is used to allocate
                                    // Mvctx->lvec in nvshmem

    // COO stuff
    PetscCount *jmap1_d; // [m+1]: i-th entry of the vector has jmap1[i+1]-jmap1[i] repeats
                         // in COO arrays
    PetscCount *perm1_d; // [tot1]: permutation array for local entries
    PetscCount *imap2_d; // [nnz2]: i-th unique entry in recvbuf is imap2[i]-th entry in
                         // the vector
    PetscCount *jmap2_d; // [nnz2+1]
    PetscCount *perm2_d; // [recvlen]
    PetscCount *Cperm_d; // [sendlen]: permutation array to fill sendbuf[]. 'C' for
                         // communication

    // Buffers for remote values in VecSetValuesCOO()
    PetscScalar *sendbuf_d;
    PetscScalar *recvbuf_d;
  };

  // Cast the Vec to its Vec_CUPM struct, i.e. return the result of (Vec_CUPM *)v->spptr
  PETSC_NODISCARD static Vec_CUPM *VecCUPMCast(Vec) noexcept;
  // Cast the Vec to its host struct, i.e. return the result of (Vec_Seq *)v->data
  template <typename U = Derived>
  PETSC_NODISCARD static constexpr auto VecIMPLCast(Vec v) noexcept -> decltype(U::VecIMPLCast_(v));
  // Get the PetscLogEvents for HTOD and DTOH
  PETSC_NODISCARD static constexpr PetscLogEvent VEC_CUPMCopyToGPU() noexcept;
  PETSC_NODISCARD static constexpr PetscLogEvent VEC_CUPMCopyFromGPU() noexcept;
  // Get the VecTypes
  PETSC_NODISCARD static constexpr VecType VECSEQCUPM() noexcept;
  PETSC_NODISCARD static constexpr VecType VECMPICUPM() noexcept;
  PETSC_NODISCARD static constexpr VecType VECCUPM() noexcept;

  // Get the VecType of the calling vector
  template <typename U = Derived>
  PETSC_NODISCARD static constexpr VecType VECIMPLCUPM() noexcept;

  // Call the host destroy function, i.e. VecDestroy_Seq()
  static PetscErrorCode VecDestroy_IMPL(Vec) noexcept;
  // Call the host reset function, i.e. VecResetArray_Seq()
  static PetscErrorCode VecResetArray_IMPL(Vec) noexcept;
  // ... you get the idea
  static PetscErrorCode VecPlaceArray_IMPL(Vec, const PetscScalar *) noexcept;
  // Call the host creation function, i.e. VecCreate_Seq(), and also initialize the CUPM part
  // along with it if needed
  static PetscErrorCode VecCreate_IMPL_Private(Vec, PetscBool *, PetscInt = 0, PetscScalar * = nullptr) noexcept;

  // Shorthand for creating VectorArray's. Need functions to create them, otherwise using them
  // as an unnamed temporary leads to most vexing parse
  PETSC_NODISCARD static auto DeviceArrayRead(PetscDeviceContext dctx, Vec v) noexcept PETSC_DECLTYPE_AUTO_RETURNS(VectorArray<PETSC_MEMTYPE_DEVICE, PETSC_MEMORY_ACCESS_READ>{dctx, v});
  PETSC_NODISCARD static auto DeviceArrayWrite(PetscDeviceContext dctx, Vec v) noexcept PETSC_DECLTYPE_AUTO_RETURNS(VectorArray<PETSC_MEMTYPE_DEVICE, PETSC_MEMORY_ACCESS_WRITE>{dctx, v});
  PETSC_NODISCARD static auto DeviceArrayReadWrite(PetscDeviceContext dctx, Vec v) noexcept PETSC_DECLTYPE_AUTO_RETURNS(VectorArray<PETSC_MEMTYPE_DEVICE, PETSC_MEMORY_ACCESS_READ_WRITE>{dctx, v});
  PETSC_NODISCARD static auto HostArrayRead(PetscDeviceContext dctx, Vec v) noexcept PETSC_DECLTYPE_AUTO_RETURNS(VectorArray<PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ>{dctx, v});
  PETSC_NODISCARD static auto HostArrayWrite(PetscDeviceContext dctx, Vec v) noexcept PETSC_DECLTYPE_AUTO_RETURNS(VectorArray<PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_WRITE>{dctx, v});
  PETSC_NODISCARD static auto HostArrayReadWrite(PetscDeviceContext dctx, Vec v) noexcept PETSC_DECLTYPE_AUTO_RETURNS(VectorArray<PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ_WRITE>{dctx, v});

  // ops-table functions
  static PetscErrorCode Create(Vec) noexcept;
  static PetscErrorCode Destroy(Vec) noexcept;
  template <PetscMemType, PetscMemoryAccessMode, bool = false>
  static PetscErrorCode GetArray(Vec, PetscScalar **, PetscDeviceContext) noexcept;
  template <PetscMemType, PetscMemoryAccessMode, bool = false>
  static PetscErrorCode GetArray(Vec, PetscScalar **) noexcept;
  template <PetscMemType, PetscMemoryAccessMode>
  static PetscErrorCode RestoreArray(Vec, PetscScalar **, PetscDeviceContext) noexcept;
  template <PetscMemType, PetscMemoryAccessMode>
  static PetscErrorCode RestoreArray(Vec, PetscScalar **) noexcept;
  template <PetscMemoryAccessMode>
  static PetscErrorCode GetArrayAndMemtype(Vec, PetscScalar **, PetscMemType *, PetscDeviceContext) noexcept;
  template <PetscMemoryAccessMode>
  static PetscErrorCode GetArrayAndMemtype(Vec, PetscScalar **, PetscMemType *) noexcept;
  template <PetscMemoryAccessMode>
  static PetscErrorCode RestoreArrayAndMemtype(Vec, PetscScalar **, PetscDeviceContext) noexcept;
  template <PetscMemoryAccessMode>
  static PetscErrorCode RestoreArrayAndMemtype(Vec, PetscScalar **) noexcept;
  template <PetscMemType>
  static PetscErrorCode ReplaceArray(Vec, const PetscScalar *) noexcept;
  template <PetscMemType>
  static PetscErrorCode ResetArray(Vec) noexcept;
  template <PetscMemType>
  static PetscErrorCode PlaceArray(Vec, const PetscScalar *) noexcept;

  // common ops shared between Seq and MPI
  static PetscErrorCode Create_CUPM(Vec) noexcept;
  static PetscErrorCode Create_CUPMBase(MPI_Comm, PetscInt, PetscInt, PetscInt, Vec *, PetscBool, PetscLayout /*reference*/ = nullptr) noexcept;
  static PetscErrorCode Initialize_CUPMBase(Vec, PetscBool, PetscScalar *, PetscScalar *, PetscDeviceContext) noexcept;
  template <typename SetupFunctionT = no_op>
  static PetscErrorCode Duplicate_CUPMBase(Vec, Vec *, PetscDeviceContext, SetupFunctionT && = SetupFunctionT{}) noexcept;
  static PetscErrorCode BindToCPU_CUPMBase(Vec, PetscBool, PetscDeviceContext) noexcept;
  static PetscErrorCode GetArrays_CUPMBase(Vec, const PetscScalar **, const PetscScalar **, PetscOffloadMask *, PetscDeviceContext) noexcept;
  static PetscErrorCode ResetPreallocationCOO_CUPMBase(Vec, PetscDeviceContext) noexcept;
  template <std::size_t NCount = 0, std::size_t NScal = 0>
  static PetscErrorCode SetPreallocationCOO_CUPMBase(Vec, PetscCount, const PetscInt[], PetscDeviceContext, const std::array<CooPair<PetscCount>, NCount> & = {}, const std::array<CooPair<PetscScalar>, NScal> & = {}) noexcept;
};

// ==========================================================================================
// Vec_CUPMBase::VectorArray
//
// RAII versions of the get/restore array routines. Determines constness of the pointer type,
// holds the pointer itself and provides the implicit conversion operator.
//
// On construction this calls the moral equivalent of Vec[CUPM]GetArray[Read|Write]()
// (depending on PetscMemoryAccessMode) and on destruction automatically restores the array
// for you
// ==========================================================================================
template <device::cupm::DeviceType T, typename D>
template <PetscMemType MT, PetscMemoryAccessMode MA>
class Vec_CUPMBase<T, D>::VectorArray : public device::cupm::impl::RestoreableArray<T, MT, MA> {
  using base_type = device::cupm::impl::RestoreableArray<T, MT, MA>;

public:
  VectorArray(PetscDeviceContext, Vec) noexcept;
  ~VectorArray() noexcept;

private:
  Vec v_ = nullptr;
};

// ==========================================================================================
// Vec_CUPMBase::VectorArray - Public API
// ==========================================================================================

template <device::cupm::DeviceType T, typename D>
template <PetscMemType MT, PetscMemoryAccessMode MA>
inline Vec_CUPMBase<T, D>::VectorArray<MT, MA>::VectorArray(PetscDeviceContext dctx, Vec v) noexcept : base_type{dctx}, v_{v}
{
  PetscFunctionBegin;
  PetscCallAbort(PETSC_COMM_SELF, Vec_CUPMBase<T, D>::template GetArray<MT, MA, true>(v, &this->ptr_, dctx));
  PetscFunctionReturnVoid();
}

template <device::cupm::DeviceType T, typename D>
template <PetscMemType MT, PetscMemoryAccessMode MA>
inline Vec_CUPMBase<T, D>::VectorArray<MT, MA>::~VectorArray() noexcept
{
  PetscFunctionBegin;
  PetscCallAbort(PETSC_COMM_SELF, Vec_CUPMBase<T, D>::template RestoreArray<MT, MA>(v_, &this->ptr_, this->dctx_));
  PetscFunctionReturnVoid();
}

// ==========================================================================================
// Vec_CUPMBase - Protected API
// ==========================================================================================

template <device::cupm::DeviceType T, typename D>
inline PetscErrorCode Vec_CUPMBase<T, D>::ResetAllocatedDevicePtr_(PetscDeviceContext dctx, Vec v, PetscScalar *new_value) noexcept
{
  auto &device_array = VecCUPMCast(v)->array_allocated_d;

  PetscFunctionBegin;
  if (device_array) {
    if (PetscDefined(HAVE_NVSHMEM) && VecCUPMCast(v)->nvshmem) {
      PetscCall(PetscNvshmemFree(device_array));
    } else {
      cupmStream_t stream;

      PetscCall(GetHandlesFrom_(dctx, &stream));
      PetscCallCUPM(cupmFreeAsync(device_array, stream));
    }
  }
  device_array = new_value;
  PetscFunctionReturn(PETSC_SUCCESS);
}

namespace
{

inline PetscErrorCode VecCUPMCheckMinimumPinnedMemory_Internal(Vec v) noexcept
{
  auto      mem = static_cast<PetscInt>(v->minimum_bytes_pinned_memory);
  PetscBool flg;

  PetscFunctionBegin;
  PetscObjectOptionsBegin(PetscObjectCast(v));
  PetscCall(PetscOptionsRangeInt("-vec_pinned_memory_min", "Minimum size (in bytes) for an allocation to use pinned memory on host", "VecSetPinnedMemoryMin", mem, &mem, &flg, 0, std::numeric_limits<decltype(mem)>::max()));
  if (flg) v->minimum_bytes_pinned_memory = mem;
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

} // anonymous namespace

template <device::cupm::DeviceType T, typename D>
template <typename CastFunctionType>
inline PetscErrorCode Vec_CUPMBase<T, D>::VecAllocateCheck_(Vec v, void *&dest, CastFunctionType &&cast) noexcept
{
  PetscFunctionBegin;
  if (PetscLikely(dest)) PetscFunctionReturn(PETSC_SUCCESS);
  // do the check here so we don't have to do it in every function
  PetscCall(checkCupmBlasIntCast(v->map->n));
  {
    auto impl = cast(v);

    PetscCall(PetscNew(&impl));
    dest = impl;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T, typename D>
inline PetscErrorCode Vec_CUPMBase<T, D>::VecIMPLAllocateCheck_(Vec v) noexcept
{
  PetscFunctionBegin;
  PetscCall(VecAllocateCheck_(v, v->data, VecIMPLCast<D>));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// allocate the Vec_CUPM struct. this is normally done through DeviceAllocateCheck_(), but in
// certain circumstances (such as when the user places the device array) we do not want to do
// the full DeviceAllocateCheck_() as it also allocates the array
template <device::cupm::DeviceType T, typename D>
inline PetscErrorCode Vec_CUPMBase<T, D>::VecCUPMAllocateCheck_(Vec v) noexcept
{
  PetscFunctionBegin;
  PetscCall(VecAllocateCheck_(v, v->spptr, VecCUPMCast));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T, typename D>
inline PetscErrorCode Vec_CUPMBase<T, D>::HostAllocateCheck_(PetscDeviceContext, Vec v) noexcept
{
  PetscFunctionBegin;
  PetscCall(VecIMPLAllocateCheck_(v));
  if (auto &alloc = VecIMPLCast(v)->array_allocated) PetscFunctionReturn(PETSC_SUCCESS);
  else {
    PetscCall(VecCUPMCheckMinimumPinnedMemory_Internal(v));
    {
      const auto n     = v->map->n;
      const auto useit = UseCUPMHostAlloc((n * sizeof(*alloc)) > v->minimum_bytes_pinned_memory);

      v->pinned_memory = static_cast<decltype(v->pinned_memory)>(useit.value());
      PetscCall(PetscMalloc1(n, &alloc));
    }
    if (!VecIMPLCast(v)->array) VecIMPLCast(v)->array = alloc;
    if (v->offloadmask == PETSC_OFFLOAD_UNALLOCATED) v->offloadmask = PETSC_OFFLOAD_CPU;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T, typename D>
inline PetscErrorCode Vec_CUPMBase<T, D>::DeviceAllocateCheck_(PetscDeviceContext dctx, Vec v) noexcept
{
  PetscFunctionBegin;
  PetscCall(VecCUPMAllocateCheck_(v));
  if (auto &alloc = VecCUPMCast(v)->array_d) PetscFunctionReturn(PETSC_SUCCESS);
  else {
    const auto   n                 = v->map->n;
    auto        &array_allocated_d = VecCUPMCast(v)->array_allocated_d;
    cupmStream_t stream;

    PetscCall(GetHandlesFrom_(dctx, &stream));
    PetscCall(PetscCUPMMallocAsync(&array_allocated_d, n, stream));
    alloc = array_allocated_d;
    if (v->offloadmask == PETSC_OFFLOAD_UNALLOCATED) {
      const auto vimp = VecIMPLCast(v);
      v->offloadmask  = (vimp && vimp->array) ? PETSC_OFFLOAD_CPU : PETSC_OFFLOAD_GPU;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T, typename D>
inline PetscErrorCode Vec_CUPMBase<T, D>::CopyToDevice_(PetscDeviceContext dctx, Vec v, bool forceasync) noexcept
{
  PetscFunctionBegin;
  PetscCall(DeviceAllocateCheck_(dctx, v));
  if (v->offloadmask == PETSC_OFFLOAD_CPU) {
    cupmStream_t stream;

    v->offloadmask = PETSC_OFFLOAD_BOTH;
    PetscCall(GetHandlesFrom_(dctx, &stream));
    PetscCall(PetscLogEventBegin(VEC_CUPMCopyToGPU(), v, 0, 0, 0));
    PetscCall(PetscCUPMMemcpyAsync(VecCUPMCast(v)->array_d, VecIMPLCast(v)->array, v->map->n, cupmMemcpyHostToDevice, stream, forceasync));
    PetscCall(PetscLogEventEnd(VEC_CUPMCopyToGPU(), v, 0, 0, 0));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T, typename D>
inline PetscErrorCode Vec_CUPMBase<T, D>::CopyToHost_(PetscDeviceContext dctx, Vec v, bool forceasync) noexcept
{
  PetscFunctionBegin;
  PetscCall(HostAllocateCheck_(dctx, v));
  if (v->offloadmask == PETSC_OFFLOAD_GPU) {
    cupmStream_t stream;

    v->offloadmask = PETSC_OFFLOAD_BOTH;
    PetscCall(GetHandlesFrom_(dctx, &stream));
    PetscCall(PetscLogEventBegin(VEC_CUPMCopyFromGPU(), v, 0, 0, 0));
    PetscCall(PetscCUPMMemcpyAsync(VecIMPLCast(v)->array, VecCUPMCast(v)->array_d, v->map->n, cupmMemcpyDeviceToHost, stream, forceasync));
    PetscCall(PetscLogEventEnd(VEC_CUPMCopyFromGPU(), v, 0, 0, 0));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ==========================================================================================
// Vec_CUPMBase - Public API
// ==========================================================================================

template <device::cupm::DeviceType T, typename D>
inline typename Vec_CUPMBase<T, D>::Vec_CUPM *Vec_CUPMBase<T, D>::VecCUPMCast(Vec v) noexcept
{
  return static_cast<Vec_CUPM *>(v->spptr);
}

// This is a trick to get around the fact that in CRTP the derived class is not yet fully
// defined because Base<Derived> must necessarily be instantiated before Derived is
// complete. By using a dummy template parameter we make the type "dependent" and so will
// only be determined when the derived class is instantiated (and therefore fully defined)
template <device::cupm::DeviceType T, typename D>
template <typename U>
inline constexpr auto Vec_CUPMBase<T, D>::VecIMPLCast(Vec v) noexcept -> decltype(U::VecIMPLCast_(v))
{
  return U::VecIMPLCast_(v);
}

template <device::cupm::DeviceType T, typename D>
inline PetscErrorCode Vec_CUPMBase<T, D>::VecDestroy_IMPL(Vec v) noexcept
{
  return D::VecDestroy_IMPL_(v);
}

template <device::cupm::DeviceType T, typename D>
inline PetscErrorCode Vec_CUPMBase<T, D>::VecResetArray_IMPL(Vec v) noexcept
{
  return D::VecResetArray_IMPL_(v);
}

template <device::cupm::DeviceType T, typename D>
inline PetscErrorCode Vec_CUPMBase<T, D>::VecPlaceArray_IMPL(Vec v, const PetscScalar *a) noexcept
{
  return D::VecPlaceArray_IMPL_(v, a);
}

template <device::cupm::DeviceType T, typename D>
inline PetscErrorCode Vec_CUPMBase<T, D>::VecCreate_IMPL_Private(Vec v, PetscBool *alloc_missing, PetscInt nghost, PetscScalar *host_array) noexcept
{
  return D::VecCreate_IMPL_Private_(v, alloc_missing, nghost, host_array);
}

template <device::cupm::DeviceType T, typename D>
inline constexpr PetscLogEvent Vec_CUPMBase<T, D>::VEC_CUPMCopyToGPU() noexcept
{
  return T == device::cupm::DeviceType::CUDA ? VEC_CUDACopyToGPU : VEC_HIPCopyToGPU;
}

template <device::cupm::DeviceType T, typename D>
inline constexpr PetscLogEvent Vec_CUPMBase<T, D>::VEC_CUPMCopyFromGPU() noexcept
{
  return T == device::cupm::DeviceType::CUDA ? VEC_CUDACopyFromGPU : VEC_HIPCopyFromGPU;
}

template <device::cupm::DeviceType T, typename D>
inline constexpr VecType Vec_CUPMBase<T, D>::VECSEQCUPM() noexcept
{
  return T == device::cupm::DeviceType::CUDA ? VECSEQCUDA : VECSEQHIP;
}

template <device::cupm::DeviceType T, typename D>
inline constexpr VecType Vec_CUPMBase<T, D>::VECMPICUPM() noexcept
{
  return T == device::cupm::DeviceType::CUDA ? VECMPICUDA : VECMPIHIP;
}

template <device::cupm::DeviceType T, typename D>
inline constexpr VecType Vec_CUPMBase<T, D>::VECCUPM() noexcept
{
  return T == device::cupm::DeviceType::CUDA ? VECCUDA : VECHIP;
}

template <device::cupm::DeviceType T, typename D>
template <typename U>
inline constexpr VecType Vec_CUPMBase<T, D>::VECIMPLCUPM() noexcept
{
  return U::VECIMPLCUPM_();
}

// private version that takes a PetscDeviceContext, called by the public variant
template <device::cupm::DeviceType T, typename D>
template <PetscMemType mtype, PetscMemoryAccessMode access, bool force>
inline PetscErrorCode Vec_CUPMBase<T, D>::GetArray(Vec v, PetscScalar **a, PetscDeviceContext dctx) noexcept
{
  constexpr auto hostmem     = PetscMemTypeHost(mtype);
  const auto     oldmask     = v->offloadmask;
  auto          &mask        = v->offloadmask;
  auto           should_sync = false;

  PetscFunctionBegin;
  static_assert((mtype == PETSC_MEMTYPE_HOST) || (mtype == PETSC_MEMTYPE_DEVICE), "");
  PetscCheckTypeNames(v, VECSEQCUPM(), VECMPICUPM());
  if (PetscMemoryAccessRead(access)) {
    // READ or READ_WRITE
    if (((oldmask == PETSC_OFFLOAD_GPU) && hostmem) || ((oldmask == PETSC_OFFLOAD_CPU) && !hostmem)) {
      // if we move the data we should set the flag to synchronize later on
      should_sync = true;
    }
    PetscCall((hostmem ? CopyToHost_ : CopyToDevice_)(dctx, v, force));
  } else {
    // WRITE only
    PetscCall((hostmem ? HostAllocateCheck_ : DeviceAllocateCheck_)(dctx, v));
  }
  *a = hostmem ? VecIMPLCast(v)->array : VecCUPMCast(v)->array_d;
  // if unallocated previously we should zero things out if we intend to read
  if (PetscMemoryAccessRead(access) && (oldmask == PETSC_OFFLOAD_UNALLOCATED)) {
    const auto n = v->map->n;

    if (hostmem) {
      PetscCall(PetscArrayzero(*a, n));
    } else {
      cupmStream_t stream;

      PetscCall(GetHandlesFrom_(dctx, &stream));
      PetscCall(PetscCUPMMemsetAsync(*a, 0, n, stream, force));
      should_sync = true;
    }
  }
  // update the offloadmask if we intend to write, since we assume immediately modified
  if (PetscMemoryAccessWrite(access)) {
    PetscCall(VecSetErrorIfLocked(v, 1));
    // REVIEW ME: this should probably also call PetscObjectStateIncrease() since we assume it
    // is immediately modified
    mask = hostmem ? PETSC_OFFLOAD_CPU : PETSC_OFFLOAD_GPU;
  }
  // if we are a globally blocking stream and we have MOVED data then we should synchronize,
  // since even doing async calls on the NULL stream is not synchronous
  if (!force && should_sync) PetscCall(PetscDeviceContextSynchronize(dctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// v->ops->getarray[read|write] or VecCUPMGetArray[Read|Write]()
template <device::cupm::DeviceType T, typename D>
template <PetscMemType mtype, PetscMemoryAccessMode access, bool force>
inline PetscErrorCode Vec_CUPMBase<T, D>::GetArray(Vec v, PetscScalar **a) noexcept
{
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  PetscCall(GetHandles_(&dctx));
  PetscCall(D::template GetArray<mtype, access, force>(v, a, dctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// private version that takes a PetscDeviceContext, called by the public variant
template <device::cupm::DeviceType T, typename D>
template <PetscMemType mtype, PetscMemoryAccessMode access>
inline PetscErrorCode Vec_CUPMBase<T, D>::RestoreArray(Vec v, PetscScalar **a, PetscDeviceContext) noexcept
{
  PetscFunctionBegin;
  static_assert((mtype == PETSC_MEMTYPE_HOST) || (mtype == PETSC_MEMTYPE_DEVICE), "");
  PetscCheckTypeNames(v, VECSEQCUPM(), VECMPICUPM());
  if (PetscMemoryAccessWrite(access)) {
    // WRITE or READ_WRITE
    PetscCall(PetscObjectStateIncrease(PetscObjectCast(v)));
    v->offloadmask = PetscMemTypeHost(mtype) ? PETSC_OFFLOAD_CPU : PETSC_OFFLOAD_GPU;
  }
  if (a) {
    PetscCall(CheckPointerMatchesMemType_(*a, mtype));
    *a = nullptr;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// v->ops->restorearray[read|write] or VecCUPMRestoreArray[Read|Write]()
template <device::cupm::DeviceType T, typename D>
template <PetscMemType mtype, PetscMemoryAccessMode access>
inline PetscErrorCode Vec_CUPMBase<T, D>::RestoreArray(Vec v, PetscScalar **a) noexcept
{
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  PetscCall(GetHandles_(&dctx));
  PetscCall(D::template RestoreArray<mtype, access>(v, a, dctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T, typename D>
template <PetscMemoryAccessMode access>
inline PetscErrorCode Vec_CUPMBase<T, D>::GetArrayAndMemtype(Vec v, PetscScalar **a, PetscMemType *mtype, PetscDeviceContext dctx) noexcept
{
  PetscFunctionBegin;
  PetscCall(D::template GetArray<PETSC_MEMTYPE_DEVICE, access>(v, a, dctx));
  if (mtype) *mtype = (PetscDefined(HAVE_NVSHMEM) && VecCUPMCast(v)->nvshmem) ? PETSC_MEMTYPE_NVSHMEM : PETSC_MEMTYPE_CUPM();
  PetscFunctionReturn(PETSC_SUCCESS);
}

// v->ops->getarrayandmemtype
template <device::cupm::DeviceType T, typename D>
template <PetscMemoryAccessMode access>
inline PetscErrorCode Vec_CUPMBase<T, D>::GetArrayAndMemtype(Vec v, PetscScalar **a, PetscMemType *mtype) noexcept
{
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  PetscCall(GetHandles_(&dctx));
  PetscCall(D::template GetArrayAndMemtype<access>(v, a, mtype, dctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T, typename D>
template <PetscMemoryAccessMode access>
inline PetscErrorCode Vec_CUPMBase<T, D>::RestoreArrayAndMemtype(Vec v, PetscScalar **a, PetscDeviceContext dctx) noexcept
{
  PetscFunctionBegin;
  PetscCall(D::template RestoreArray<PETSC_MEMTYPE_DEVICE, access>(v, a, dctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// v->ops->restorearrayandmemtype
template <device::cupm::DeviceType T, typename D>
template <PetscMemoryAccessMode access>
inline PetscErrorCode Vec_CUPMBase<T, D>::RestoreArrayAndMemtype(Vec v, PetscScalar **a) noexcept
{
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  PetscCall(GetHandles_(&dctx));
  PetscCall(D::template RestoreArrayAndMemtype<access>(v, a, dctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// v->ops->placearray or VecCUPMPlaceArray()
template <device::cupm::DeviceType T, typename D>
template <PetscMemType mtype>
inline PetscErrorCode Vec_CUPMBase<T, D>::PlaceArray(Vec v, const PetscScalar *a) noexcept
{
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  static_assert((mtype == PETSC_MEMTYPE_HOST) || (mtype == PETSC_MEMTYPE_DEVICE), "");
  PetscCheckTypeNames(v, VECSEQCUPM(), VECMPICUPM());
  PetscCall(CheckPointerMatchesMemType_(a, mtype));
  PetscCall(GetHandles_(&dctx));
  if (PetscMemTypeHost(mtype)) {
    PetscCall(CopyToHost_(dctx, v));
    PetscCall(VecPlaceArray_IMPL(v, a));
    v->offloadmask = PETSC_OFFLOAD_CPU;
  } else {
    PetscCall(VecIMPLAllocateCheck_(v));
    {
      auto &backup_array = VecIMPLCast(v)->unplacedarray;

      PetscCheck(!backup_array, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "VecPlaceArray() was already called on this vector, without a call to VecResetArray()");
      PetscCall(CopyToDevice_(dctx, v));
      PetscCall(PetscObjectStateIncrease(PetscObjectCast(v)));
      backup_array = util::exchange(VecCUPMCast(v)->array_d, const_cast<PetscScalar *>(a));
      // only update the offload mask if we actually assign a pointer
      if (a) v->offloadmask = PETSC_OFFLOAD_GPU;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// v->ops->replacearray or VecCUPMReplaceArray()
template <device::cupm::DeviceType T, typename D>
template <PetscMemType mtype>
inline PetscErrorCode Vec_CUPMBase<T, D>::ReplaceArray(Vec v, const PetscScalar *a) noexcept
{
  const auto         aptr = const_cast<PetscScalar *>(a);
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  static_assert((mtype == PETSC_MEMTYPE_HOST) || (mtype == PETSC_MEMTYPE_DEVICE), "");
  PetscCheckTypeNames(v, VECSEQCUPM(), VECMPICUPM());
  PetscCall(CheckPointerMatchesMemType_(a, mtype));
  PetscCall(GetHandles_(&dctx));
  if (PetscMemTypeHost(mtype)) {
    PetscCall(VecIMPLAllocateCheck_(v));
    {
      const auto vimpl      = VecIMPLCast(v);
      auto      &host_array = vimpl->array_allocated;

      // make sure the users array has the latest values.
      // REVIEW ME: why? we're about to free it
      if (host_array != vimpl->array) PetscCall(CopyToHost_(dctx, v));
      if (host_array) {
        const auto useit = UseCUPMHostAlloc(v->pinned_memory);

        PetscCall(PetscFree(host_array));
      }
      host_array       = aptr;
      vimpl->array     = host_array;
      v->pinned_memory = PETSC_FALSE; // REVIEW ME: we can determine this
      v->offloadmask   = PETSC_OFFLOAD_CPU;
    }
  } else {
    PetscCall(VecCUPMAllocateCheck_(v));
    {
      const auto vcu = VecCUPMCast(v);

      PetscCall(ResetAllocatedDevicePtr_(dctx, v, aptr));
      // don't update the offloadmask if placed pointer is NULL
      vcu->array_d = vcu->array_allocated_d /* = aptr */;
      if (aptr) v->offloadmask = PETSC_OFFLOAD_GPU;
    }
  }
  PetscCall(PetscObjectStateIncrease(PetscObjectCast(v)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// v->ops->resetarray or VecCUPMResetArray()
template <device::cupm::DeviceType T, typename D>
template <PetscMemType mtype>
inline PetscErrorCode Vec_CUPMBase<T, D>::ResetArray(Vec v) noexcept
{
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  static_assert((mtype == PETSC_MEMTYPE_HOST) || (mtype == PETSC_MEMTYPE_DEVICE), "");
  PetscCheckTypeNames(v, VECSEQCUPM(), VECMPICUPM());
  PetscCall(GetHandles_(&dctx));
  // REVIEW ME:
  // this is wildly inefficient but must be done if we assume that the placed array must have
  // correct values
  if (PetscMemTypeHost(mtype)) {
    PetscCall(CopyToHost_(dctx, v));
    PetscCall(VecResetArray_IMPL(v));
    v->offloadmask = PETSC_OFFLOAD_CPU;
  } else {
    PetscCall(VecIMPLAllocateCheck_(v));
    PetscCall(VecCUPMAllocateCheck_(v));
    {
      const auto vcu        = VecCUPMCast(v);
      const auto vimpl      = VecIMPLCast(v);
      auto      &host_array = vimpl->unplacedarray;

      PetscCall(CheckPointerMatchesMemType_(host_array, PETSC_MEMTYPE_DEVICE));
      PetscCall(CopyToDevice_(dctx, v));
      PetscCall(PetscObjectStateIncrease(PetscObjectCast(v)));
      // Need to reset the offloadmask. If we had a stashed pointer we are on the GPU,
      // otherwise check if the host has a valid pointer. If neither, then we are not
      // allocated.
      vcu->array_d = host_array;
      if (host_array) {
        host_array     = nullptr;
        v->offloadmask = PETSC_OFFLOAD_GPU;
      } else if (vimpl->array) {
        v->offloadmask = PETSC_OFFLOAD_CPU;
      } else {
        v->offloadmask = PETSC_OFFLOAD_UNALLOCATED;
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// v->ops->create
template <device::cupm::DeviceType T, typename D>
inline PetscErrorCode Vec_CUPMBase<T, D>::Create(Vec v) noexcept
{
  PetscBool          alloc_missing;
  PetscDeviceContext dctx;

  PetscFunctionBegin;
  PetscCall(VecCreate_IMPL_Private(v, &alloc_missing));
  PetscCall(GetHandles_(&dctx));
  PetscCall(Initialize_CUPMBase(v, alloc_missing, nullptr, nullptr, dctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// v->ops->destroy
template <device::cupm::DeviceType T, typename D>
inline PetscErrorCode Vec_CUPMBase<T, D>::Destroy(Vec v) noexcept
{
  PetscFunctionBegin;
  if (const auto vcu = VecCUPMCast(v)) {
    PetscDeviceContext dctx;

    PetscCall(GetHandles_(&dctx));
    PetscCall(ResetAllocatedDevicePtr_(dctx, v));
    PetscCall(ResetPreallocationCOO_CUPMBase(v, dctx));
    PetscCall(PetscFree(v->spptr));
  }
  PetscCall(PetscObjectSAWsViewOff(PetscObjectCast(v)));
  if (const auto vimpl = VecIMPLCast(v)) {
    if (auto &array_allocated = vimpl->array_allocated) {
      const auto useit = UseCUPMHostAlloc(v->pinned_memory);

      // do this ourselves since we may want to use the cupm functions
      PetscCall(PetscFree(array_allocated));
    }
  }
  v->pinned_memory = PETSC_FALSE;
  PetscCall(VecDestroy_IMPL(v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ================================================================================== //
//                      Common core between Seq and MPI                               //

// VecCreate_CUPM()
template <device::cupm::DeviceType T, typename D>
inline PetscErrorCode Vec_CUPMBase<T, D>::Create_CUPM(Vec v) noexcept
{
  PetscMPIInt size;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm(PetscObjectCast(v)), &size));
  PetscCall(VecSetType(v, size > 1 ? VECMPICUPM() : VECSEQCUPM()));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// VecCreateCUPM()
template <device::cupm::DeviceType T, typename D>
inline PetscErrorCode Vec_CUPMBase<T, D>::Create_CUPMBase(MPI_Comm comm, PetscInt bs, PetscInt n, PetscInt N, Vec *v, PetscBool call_set_type, PetscLayout reference) noexcept
{
  PetscFunctionBegin;
  PetscCall(VecCreate(comm, v));
  if (reference) PetscCall(PetscLayoutReference(reference, &(*v)->map));
  PetscCall(VecSetSizes(*v, n, N));
  if (bs) PetscCall(VecSetBlockSize(*v, bs));
  if (call_set_type) PetscCall(VecSetType(*v, VECIMPLCUPM()));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// VecCreateIMPL_CUPM(), called through v->ops->create
template <device::cupm::DeviceType T, typename D>
inline PetscErrorCode Vec_CUPMBase<T, D>::Initialize_CUPMBase(Vec v, PetscBool allocate_missing, PetscScalar *host_array, PetscScalar *device_array, PetscDeviceContext dctx) noexcept
{
  PetscFunctionBegin;
  // REVIEW ME: perhaps not needed
  PetscCall(PetscDeviceInitialize(PETSC_DEVICE_CUPM()));
  PetscCall(PetscObjectChangeTypeName(PetscObjectCast(v), VECIMPLCUPM()));
  PetscCall(D::BindToCPU(v, PETSC_FALSE));
  if (device_array) {
    PetscCall(CheckPointerMatchesMemType_(device_array, PETSC_MEMTYPE_CUPM()));
    PetscCall(VecCUPMAllocateCheck_(v));
    VecCUPMCast(v)->array_d = device_array;
  }
  if (host_array) {
    PetscCall(CheckPointerMatchesMemType_(host_array, PETSC_MEMTYPE_HOST));
    VecIMPLCast(v)->array = host_array;
  }
  if (allocate_missing) {
    PetscCall(DeviceAllocateCheck_(dctx, v));
    PetscCall(HostAllocateCheck_(dctx, v));
    // REVIEW ME: junchao, is this needed with new calloc() branch? VecSet() will call
    // set() for reference
    // calls device-version
    PetscCall(VecSet(v, 0));
    // zero the host while device is underway
    PetscCall(PetscArrayzero(VecIMPLCast(v)->array, v->map->n));
    v->offloadmask = PETSC_OFFLOAD_BOTH;
  } else {
    if (host_array) {
      v->offloadmask = device_array ? PETSC_OFFLOAD_BOTH : PETSC_OFFLOAD_CPU;
    } else {
      v->offloadmask = device_array ? PETSC_OFFLOAD_GPU : PETSC_OFFLOAD_UNALLOCATED;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// v->ops->duplicate
template <device::cupm::DeviceType T, typename D>
template <typename SetupFunctionT>
inline PetscErrorCode Vec_CUPMBase<T, D>::Duplicate_CUPMBase(Vec v, Vec *y, PetscDeviceContext dctx, SetupFunctionT &&DerivedCreateIMPLCUPM_Async) noexcept
{
  // if the derived setup is the default no_op then we should call VecSetType()
  constexpr auto call_set_type = static_cast<PetscBool>(std::is_same<SetupFunctionT, no_op>::value);
  const auto     vobj          = PetscObjectCast(v);
  const auto     map           = v->map;
  PetscInt       bs;

  PetscFunctionBegin;
  PetscCall(VecGetBlockSize(v, &bs));
  PetscCall(Create_CUPMBase(PetscObjectComm(vobj), bs, map->n, map->N, y, call_set_type, map));
  // Derived class can set up the remainder of the data structures here
  PetscCall(DerivedCreateIMPLCUPM_Async(*y));
  // If the other vector is bound to CPU then the memcpy of the ops struct will give the
  // duplicated vector the host "getarray" function which does not lazily allocate the array
  // (as it is assumed to always exist). So we force allocation here, before we overwrite the
  // ops
  if (v->boundtocpu) PetscCall(HostAllocateCheck_(dctx, *y));
  // in case the user has done some VecSetOps() tomfoolery
  PetscCall(PetscArraycpy((*y)->ops, v->ops, 1));
  {
    const auto yobj = PetscObjectCast(*y);

    PetscCall(PetscObjectListDuplicate(vobj->olist, &yobj->olist));
    PetscCall(PetscFunctionListDuplicate(vobj->qlist, &yobj->qlist));
  }
  (*y)->stash.donotstash   = v->stash.donotstash;
  (*y)->stash.ignorenegidx = v->stash.ignorenegidx;
  (*y)->map->bs            = std::abs(v->map->bs);
  (*y)->bstash.bs          = v->bstash.bs;
  PetscFunctionReturn(PETSC_SUCCESS);
}

  #define VecSetOp_CUPM(op_name, op_host, ...) \
    do { \
      if (usehost) { \
        v->ops->op_name = op_host; \
      } else { \
        v->ops->op_name = __VA_ARGS__; \
      } \
    } while (0)

// v->ops->bindtocpu
template <device::cupm::DeviceType T, typename D>
inline PetscErrorCode Vec_CUPMBase<T, D>::BindToCPU_CUPMBase(Vec v, PetscBool usehost, PetscDeviceContext dctx) noexcept
{
  PetscFunctionBegin;
  v->boundtocpu = usehost;
  if (usehost) PetscCall(CopyToHost_(dctx, v));
  PetscCall(PetscStrFreeAllocpy(usehost ? PETSCRANDER48 : PETSCDEVICERAND(), &v->defaultrandtype));

  // set the base functions that are guaranteed to be the same for both
  v->ops->duplicate = D::Duplicate;
  v->ops->create    = D::Create;
  v->ops->destroy   = D::Destroy;
  v->ops->bindtocpu = D::BindToCPU;
  // Note that setting these to NULL on host breaks convergence in certain areas. I don't know
  // why, and I don't know how, but it is IMPERATIVE these are set as such!
  v->ops->replacearray = D::template ReplaceArray<PETSC_MEMTYPE_HOST>;
  v->ops->restorearray = D::template RestoreArray<PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ_WRITE>;

  // set device-only common functions
  VecSetOp_CUPM(dotnorm2, nullptr, D::DotNorm2);
  VecSetOp_CUPM(getarray, nullptr, D::template GetArray<PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ_WRITE>);
  VecSetOp_CUPM(getarraywrite, nullptr, D::template GetArray<PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_WRITE>);
  VecSetOp_CUPM(restorearraywrite, nullptr, D::template RestoreArray<PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_WRITE>);

  VecSetOp_CUPM(getarrayread, nullptr, [](Vec v, const PetscScalar **a) { return D::template GetArray<PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ>(v, const_cast<PetscScalar **>(a)); });
  VecSetOp_CUPM(restorearrayread, nullptr, [](Vec v, const PetscScalar **a) { return D::template RestoreArray<PETSC_MEMTYPE_HOST, PETSC_MEMORY_ACCESS_READ>(v, const_cast<PetscScalar **>(a)); });

  VecSetOp_CUPM(getarrayandmemtype, nullptr, D::template GetArrayAndMemtype<PETSC_MEMORY_ACCESS_READ_WRITE>);
  VecSetOp_CUPM(restorearrayandmemtype, nullptr, D::template RestoreArrayAndMemtype<PETSC_MEMORY_ACCESS_READ_WRITE>);

  VecSetOp_CUPM(getarraywriteandmemtype, nullptr, D::template GetArrayAndMemtype<PETSC_MEMORY_ACCESS_WRITE>);
  VecSetOp_CUPM(restorearraywriteandmemtype, nullptr, [](Vec v, PetscScalar **a, PetscMemType *) { return D::template RestoreArrayAndMemtype<PETSC_MEMORY_ACCESS_WRITE>(v, a); });

  VecSetOp_CUPM(getarrayreadandmemtype, nullptr, [](Vec v, const PetscScalar **a, PetscMemType *m) { return D::template GetArrayAndMemtype<PETSC_MEMORY_ACCESS_READ>(v, const_cast<PetscScalar **>(a), m); });
  VecSetOp_CUPM(restorearrayreadandmemtype, nullptr, [](Vec v, const PetscScalar **a) { return D::template RestoreArrayAndMemtype<PETSC_MEMORY_ACCESS_READ>(v, const_cast<PetscScalar **>(a)); });

  // set the functions that are always sequential
  using VecSeq_T = VecSeq_CUPM<T>;
  VecSetOp_CUPM(scale, VecScale_Seq, VecSeq_T::Scale);
  VecSetOp_CUPM(copy, VecCopy_Seq, VecSeq_T::Copy);
  VecSetOp_CUPM(set, VecSet_Seq, VecSeq_T::Set);
  VecSetOp_CUPM(swap, VecSwap_Seq, VecSeq_T::Swap);
  VecSetOp_CUPM(axpy, VecAXPY_Seq, VecSeq_T::AXPY);
  VecSetOp_CUPM(axpby, VecAXPBY_Seq, VecSeq_T::AXPBY);
  VecSetOp_CUPM(maxpy, VecMAXPY_Seq, VecSeq_T::MAXPY);
  VecSetOp_CUPM(aypx, VecAYPX_Seq, VecSeq_T::AYPX);
  VecSetOp_CUPM(waxpy, VecWAXPY_Seq, VecSeq_T::WAXPY);
  VecSetOp_CUPM(axpbypcz, VecAXPBYPCZ_Seq, VecSeq_T::AXPBYPCZ);
  VecSetOp_CUPM(pointwisemult, VecPointwiseMult_Seq, VecSeq_T::PointwiseMult);
  VecSetOp_CUPM(pointwisedivide, VecPointwiseDivide_Seq, VecSeq_T::PointwiseDivide);
  VecSetOp_CUPM(setrandom, VecSetRandom_Seq, VecSeq_T::SetRandom);
  VecSetOp_CUPM(dot_local, VecDot_Seq, VecSeq_T::Dot);
  VecSetOp_CUPM(tdot_local, VecTDot_Seq, VecSeq_T::TDot);
  VecSetOp_CUPM(norm_local, VecNorm_Seq, VecSeq_T::Norm);
  VecSetOp_CUPM(mdot_local, VecMDot_Seq, VecSeq_T::MDot);
  VecSetOp_CUPM(reciprocal, VecReciprocal_Default, VecSeq_T::Reciprocal);
  VecSetOp_CUPM(shift, nullptr, VecSeq_T::Shift);
  VecSetOp_CUPM(getlocalvector, nullptr, VecSeq_T::template GetLocalVector<PETSC_MEMORY_ACCESS_READ_WRITE>);
  VecSetOp_CUPM(restorelocalvector, nullptr, VecSeq_T::template RestoreLocalVector<PETSC_MEMORY_ACCESS_READ_WRITE>);
  VecSetOp_CUPM(getlocalvectorread, nullptr, VecSeq_T::template GetLocalVector<PETSC_MEMORY_ACCESS_READ>);
  VecSetOp_CUPM(restorelocalvectorread, nullptr, VecSeq_T::template RestoreLocalVector<PETSC_MEMORY_ACCESS_READ>);
  VecSetOp_CUPM(sum, nullptr, VecSeq_T::Sum);
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Called from VecGetSubVector()
template <device::cupm::DeviceType T, typename D>
inline PetscErrorCode Vec_CUPMBase<T, D>::GetArrays_CUPMBase(Vec v, const PetscScalar **host_array, const PetscScalar **device_array, PetscOffloadMask *mask, PetscDeviceContext dctx) noexcept
{
  PetscFunctionBegin;
  PetscCheckTypeNames(v, VECSEQCUPM(), VECMPICUPM());
  if (host_array) {
    PetscCall(HostAllocateCheck_(dctx, v));
    *host_array = VecIMPLCast(v)->array;
  }
  if (device_array) {
    PetscCall(DeviceAllocateCheck_(dctx, v));
    *device_array = VecCUPMCast(v)->array_d;
  }
  if (mask) *mask = v->offloadmask;
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T, typename D>
inline PetscErrorCode Vec_CUPMBase<T, D>::ResetPreallocationCOO_CUPMBase(Vec v, PetscDeviceContext dctx) noexcept
{
  PetscFunctionBegin;
  if (const auto vcu = VecCUPMCast(v)) {
    cupmStream_t stream;
    // clang-format off
    const auto   cntptrs = util::make_array(
      std::ref(vcu->jmap1_d),
      std::ref(vcu->perm1_d),
      std::ref(vcu->imap2_d),
      std::ref(vcu->jmap2_d),
      std::ref(vcu->perm2_d),
      std::ref(vcu->Cperm_d)
    );
    // clang-format on

    PetscCall(GetHandlesFrom_(dctx, &stream));
    for (auto &&ptr : cntptrs) PetscCallCUPM(cupmFreeAsync(ptr.get(), stream));
    for (auto &&ptr : util::make_array(std::ref(vcu->sendbuf_d), std::ref(vcu->recvbuf_d))) PetscCallCUPM(cupmFreeAsync(ptr.get(), stream));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <device::cupm::DeviceType T, typename D>
template <std::size_t NCount, std::size_t NScal>
inline PetscErrorCode Vec_CUPMBase<T, D>::SetPreallocationCOO_CUPMBase(Vec v, PetscCount, const PetscInt[], PetscDeviceContext dctx, const std::array<CooPair<PetscCount>, NCount> &extra_cntptrs, const std::array<CooPair<PetscScalar>, NScal> &bufptrs) noexcept
{
  PetscFunctionBegin;
  PetscCall(ResetPreallocationCOO_CUPMBase(v, dctx));
  // need to instantiate the private pointer if not already
  PetscCall(VecCUPMAllocateCheck_(v));
  {
    const auto vimpl = VecIMPLCast(v);
    const auto vcu   = VecCUPMCast(v);
    // clang-format off
    const auto cntptrs = util::concat_array(
      util::make_array(
        make_coo_pair(vcu->jmap1_d, vimpl->jmap1, v->map->n + 1),
        make_coo_pair(vcu->perm1_d, vimpl->perm1, vimpl->tot1)
      ),
      extra_cntptrs
    );
    // clang-format on
    cupmStream_t stream;

    PetscCall(GetHandlesFrom_(dctx, &stream));
    // allocate
    for (auto &elem : cntptrs) PetscCall(PetscCUPMMallocAsync(&elem.device, elem.size, stream));
    for (auto &elem : bufptrs) PetscCall(PetscCUPMMallocAsync(&elem.device, elem.size, stream));
    // copy
    for (const auto &elem : cntptrs) PetscCall(PetscCUPMMemcpyAsync(elem.device, elem.host, elem.size, cupmMemcpyHostToDevice, stream, true));
    for (const auto &elem : bufptrs) PetscCall(PetscCUPMMemcpyAsync(elem.device, elem.host, elem.size, cupmMemcpyHostToDevice, stream, true));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

  #define PETSC_VEC_CUPM_BASE_CLASS_HEADER(name, Tp, ...) \
    PETSC_CUPMOBJECT_HEADER(Tp); \
    using name = ::Petsc::vec::cupm::impl::Vec_CUPMBase<Tp, __VA_ARGS__>; \
    friend name; \
    /* introspection */ \
    using name::VecCUPMCast; \
    using name::VecIMPLCast; \
    using name::VECIMPLCUPM; \
    using name::VECSEQCUPM; \
    using name::VECMPICUPM; \
    using name::VECCUPM; \
    using name::VecView_Debug; \
    /* utility */ \
    using typename name::Vec_CUPM; \
    using name::VecCUPMAllocateCheck_; \
    using name::VecIMPLAllocateCheck_; \
    using name::HostAllocateCheck_; \
    using name::DeviceAllocateCheck_; \
    using name::CopyToDevice_; \
    using name::CopyToHost_; \
    using name::Create; \
    using name::Destroy; \
    using name::GetArray; \
    using name::RestoreArray; \
    using name::GetArrayAndMemtype; \
    using name::RestoreArrayAndMemtype; \
    using name::PlaceArray; \
    using name::ReplaceArray; \
    using name::ResetArray; \
    /* base functions */ \
    using name::Create_CUPMBase; \
    using name::Initialize_CUPMBase; \
    using name::Duplicate_CUPMBase; \
    using name::BindToCPU_CUPMBase; \
    using name::Create_CUPM; \
    using name::DeviceArrayRead; \
    using name::DeviceArrayWrite; \
    using name::DeviceArrayReadWrite; \
    using name::HostArrayRead; \
    using name::HostArrayWrite; \
    using name::HostArrayReadWrite; \
    using name::ResetPreallocationCOO_CUPMBase; \
    using name::SetPreallocationCOO_CUPMBase

} // namespace impl

} // namespace cupm

} // namespace vec

} // namespace Petsc

#endif // __cplusplus && PetscDefined(HAVE_DEVICE)

#endif // PETSCVECCUPMIMPL_H
