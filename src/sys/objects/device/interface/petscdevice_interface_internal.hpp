#ifndef PETSCDEVICE_INTERFACE_INTERNAL_HPP
#define PETSCDEVICE_INTERFACE_INTERNAL_HPP

#include <petsc/private/deviceimpl.h>

#include <petsc/private/cpp/utility.hpp>    // std::pair
#include <petsc/private/cpp/functional.hpp> //std::equal_to

#include <unordered_map>
#include <unordered_set>

#if PetscDefined(USE_DEBUG) && PetscDefined(USE_INFO)
  #define PETSC_USE_DEBUG_AND_INFO  1
  #define PetscDebugInfo(dctx, ...) PetscInfo(dctx, __VA_ARGS__)
#else
  #define PetscDebugInfo(dctx, ...) PETSC_SUCCESS
#endif

// this file contains functions needed to bridge the gap between dcontext.cxx and device.cxx
// but are not useful enough to put in the impl header
PETSC_INTERN PetscErrorCode PetscDeviceContextSetDefaultDeviceForType_Internal(PetscDeviceContext, PetscDeviceType);
PETSC_INTERN PetscErrorCode PetscDeviceContextSetRootDeviceType_Internal(PetscDeviceType);
PETSC_INTERN PetscErrorCode PetscDeviceContextSetRootStreamType_Internal(PetscStreamType);
PETSC_INTERN PetscErrorCode PetscDeviceContextSyncClearMap_Internal(PetscDeviceContext);
PETSC_INTERN PetscErrorCode PetscDeviceContextCheckNotOrphaned_Internal(PetscDeviceContext);

// open up namespace std to specialize equal_to for unordered_map
namespace std
{

template <>
struct equal_to<PetscDeviceContext> {
#if PETSC_CPP_VERSION <= 17
  using result_type          = bool;
  using first_argument_type  = PetscDeviceContext;
  using second_argument_type = PetscDeviceContext;
#endif

  constexpr bool operator()(const PetscDeviceContext &x, const PetscDeviceContext &y) const noexcept { return PetscObjectCast(x)->id == PetscObjectCast(y)->id; }
};

} // namespace std

namespace
{

// workaround for bug in:
// clang: https://bugs.llvm.org/show_bug.cgi?id=36684
// gcc: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=96645
//
// see also:
// https://stackoverflow.com/questions/53408962/try-to-understand-compiler-error-message-default-member-initializer-required-be
struct CxxDataParent {
  PetscObjectId    id    = 0;
  PetscObjectState state = 0;

  constexpr CxxDataParent() noexcept = default;

  constexpr explicit CxxDataParent(PetscDeviceContext dctx) noexcept : CxxDataParent(PetscObjectCast(dctx)->id, PetscObjectCast(dctx)->state) { }

private:
  // make this private, we do not want to accept any old id and state pairing
  constexpr CxxDataParent(PetscObjectId id_, PetscObjectState state_) noexcept : id(id_), state(state_) { }
};

struct CxxData {
  using upstream_type = std::unordered_map<PetscDeviceContext, CxxDataParent>;
  using dep_type      = std::unordered_set<PetscObjectId>;

  // double check we didn't specialize for no reason
  static_assert(std::is_same<typename upstream_type::key_equal, std::equal_to<PetscDeviceContext>>::value, "");

  upstream_type upstream{};
  dep_type      deps{};

  PetscErrorCode clear() noexcept;
};

inline PetscErrorCode CxxData::clear() noexcept
{
  PetscFunctionBegin;
  PetscCallCXX(this->upstream.clear());
  PetscCallCXX(this->deps.clear());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_NODISCARD inline CxxData *CxxDataCast(PetscDeviceContext dctx) noexcept
{
  return static_cast<CxxData *>(PetscObjectCast(dctx)->cpp);
}

/*
  needed because PetscInitialize() needs to also query these options to set the defaults. Since
  it does not yet have a PetscDeviceContext to call this with, the actual options queries are
  abstracted out, so you can call this without one.
*/
inline PetscErrorCode PetscDeviceContextQueryOptions_Internal(PetscOptionItems *PetscOptionsObject, std::pair<PetscDeviceType, PetscBool> &deviceType, std::pair<PetscStreamType, PetscBool> &streamType)
{
  auto dtype = static_cast<PetscInt>(deviceType.first);
  auto stype = static_cast<PetscInt>(streamType.first);

  PetscFunctionBegin;
  /* set the device type first */
  PetscCall(PetscOptionsEList("-device_context_device_type", "Underlying PetscDevice", "PetscDeviceContextSetDevice", PetscDeviceTypes, PETSC_DEVICE_MAX, PetscDeviceTypes[dtype], &dtype, &deviceType.second));
  PetscCall(PetscOptionsEList("-device_context_stream_type", "PetscDeviceContext PetscStreamType", "PetscDeviceContextSetStreamType", PetscStreamTypes, PETSC_STREAM_MAX, PetscStreamTypes[stype], &stype, &streamType.second));
  deviceType.first = PetscDeviceTypeCast(dtype);
  streamType.first = PetscStreamTypeCast(stype);
  PetscFunctionReturn(PETSC_SUCCESS);
}

} // anonymous namespace

#endif // PETSCDEVICE_INTERFACE_INTERNAL_HPP
