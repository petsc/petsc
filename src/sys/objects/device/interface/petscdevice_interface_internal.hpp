#pragma once

#include <petsc/private/deviceimpl.h>

#include <petsc/private/cpp/utility.hpp> // std::pair
#include <petsc/private/cpp/memory.hpp>  // std::weak_ptr, std::shared_ptr

#include <unordered_map>
#include <algorithm> // std::lower_bound

// clang's unordered_set implementation outperforms the flat vector implementation in all
// cases. GCC on the other hand only does so for n > 512, before which it is almost twice as
// slow! Even when it does surpass the vector, the speedup is tiny (1.2x). So we use
// unordered_set for clang and hand-rolled flat set for GCC...
//
// https://godbolt.org/z/bb7EWf3s5
//
// This choice is consequential, since adding/checking marks is done for every
// PetscDeviceContextMarkIntentFromID() call
#ifdef __clang__
  #include <unordered_set>
  #define PETSC_USE_UNORDERED_SET_FOR_MARKED 1
#else
  #include <vector>
  #define PETSC_USE_UNORDERED_SET_FOR_MARKED 0
#endif

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

struct _n_WeakContext {
public:
  using weak_ptr_type = std::weak_ptr<_p_PetscDeviceContext>;

  constexpr _n_WeakContext() noexcept = default;

  void swap(_n_WeakContext &other) noexcept
  {
    using std::swap;

    weak_dctx_.swap(other.weak_dctx_);
    swap(state_, other.state_);
  }

  friend void swap(_n_WeakContext &lhs, _n_WeakContext &rhs) noexcept { lhs.swap(rhs); }

  PETSC_NODISCARD const weak_ptr_type &weak_dctx() const noexcept { return weak_dctx_; }

  PETSC_NODISCARD PetscObjectState state() const noexcept { return state_; }

  void set_state(PetscObjectState state) noexcept { state_ = state; }

private:
  weak_ptr_type    weak_dctx_{};
  PetscObjectState state_{};

  friend class CxxData;

  explicit _n_WeakContext(const std::shared_ptr<_p_PetscDeviceContext> &ptr) noexcept : weak_dctx_{ptr}, state_{PetscObjectCast(ptr.get())->state} { }
};

class CxxData {
public:
  struct NoOpDeleter {
    PETSC_CONSTEXPR_14 void operator()(const void *) const noexcept { }
  };

  using upstream_type = std::unordered_map<PetscObjectId, _n_WeakContext>;
#if PETSC_USE_UNORDERED_SET_FOR_MARKED
  using marked_type = std::unordered_set<PetscObjectId>;
#else
  using marked_type = std::vector<PetscObjectId>;
#endif
  using shared_ptr_type = std::shared_ptr<_p_PetscDeviceContext>;

  explicit CxxData(PetscDeviceContext dctx) noexcept : self_{dctx, NoOpDeleter{}} { }

  PETSC_NODISCARD const upstream_type   &upstream() const noexcept { return upstream_; }
  PETSC_NODISCARD upstream_type         &upstream() noexcept { return upstream_; }
  PETSC_NODISCARD const marked_type     &marked_objects() const noexcept { return marked_objects_; }
  PETSC_NODISCARD marked_type           &marked_objects() noexcept { return marked_objects_; }
  PETSC_NODISCARD const shared_ptr_type &self() const noexcept { return self_; }

  PetscErrorCode                 reset_self(PetscDeviceContext) noexcept;
  PetscErrorCode                 clear() noexcept;
  PETSC_NODISCARD _n_WeakContext weak_snapshot() const noexcept;
  PetscErrorCode                 add_mark(PetscObjectId) noexcept;
  PETSC_NODISCARD bool           has_marked(PetscObjectId) const noexcept;

private:
#if !PETSC_USE_UNORDERED_SET_FOR_MARKED
  PETSC_NODISCARD std::pair<bool, typename marked_type::iterator> get_marked_(PetscObjectId id) noexcept
  {
    auto end = this->marked_objects().end();
    auto it  = std::lower_bound(this->marked_objects().begin(), end, id);

    return {it != end && *it == id, it};
  }
#endif

  upstream_type   upstream_{};
  marked_type     marked_objects_{};
  shared_ptr_type self_{};
};

inline PetscErrorCode CxxData::reset_self(PetscDeviceContext dctx) noexcept
{
  PetscFunctionBegin;
  if (dctx) {
    PetscCallCXX(self_.reset(dctx, NoOpDeleter{}));
  } else {
    PetscCallCXX(self_.reset());
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

inline PetscErrorCode CxxData::clear() noexcept
{
  PetscFunctionBegin;
  PetscCallCXX(this->upstream().clear());
  PetscCallCXX(this->marked_objects().clear());
  PetscFunctionReturn(PETSC_SUCCESS);
}

inline _n_WeakContext CxxData::weak_snapshot() const noexcept
{
  return _n_WeakContext{this->self()};
}

inline PetscErrorCode CxxData::add_mark(PetscObjectId id) noexcept
{
  PetscFunctionBegin;
#if PETSC_USE_UNORDERED_SET_FOR_MARKED
  PetscCallCXX(marked_objects_.emplace(id));
#else
  const auto pair = get_marked_(id);

  if (!pair.first) PetscCallCXX(marked_objects_.insert(pair.second, id));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

inline bool CxxData::has_marked(PetscObjectId id) const noexcept
{
#if PETSC_USE_UNORDERED_SET_FOR_MARKED
  return marked_objects().find(id) != marked_objects().end();
#else
  return const_cast<CxxData *>(this)->get_marked_(id).first;
#endif
}

#undef PETSC_USE_UNORDERED_SET_FOR_MARKED

namespace
{

PETSC_NODISCARD inline CxxData *CxxDataCast(PetscDeviceContext dctx) noexcept
{
  return static_cast<CxxData *>(PetscObjectCast(dctx)->cpp);
}

/*
  needed because PetscInitialize() needs to also query these options to set the defaults. Since
  it does not yet have a PetscDeviceContext to call this with, the actual options queries are
  abstracted out, so you can call this without one.
*/
inline PetscErrorCode PetscDeviceContextQueryOptions_Internal(PetscOptionItems PetscOptionsObject, std::pair<PetscDeviceType, PetscBool> &deviceType, std::pair<PetscStreamType, PetscBool> &streamType)
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
