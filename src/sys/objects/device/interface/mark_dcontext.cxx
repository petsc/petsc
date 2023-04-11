#include "petscdevice_interface_internal.hpp" /*I <petscdevice.h> I*/

#include <petsc/private/cpp/object_pool.hpp>
#include <petsc/private/cpp/utility.hpp>
#include <petsc/private/cpp/unordered_map.hpp>

#include <algorithm> // std::remove_if(), std::find_if()
#include <vector>
#include <string>
#include <sstream> // std::ostringstream

#if defined(__clang__)
PETSC_PRAGMA_DIAGNOSTIC_IGNORED_BEGIN("-Wgnu-zero-variadic-macro-arguments")
#endif

// ==========================================================================================
// PetscEvent
// ==========================================================================================

class PetscEventConstructor : public Petsc::ConstructorInterface<_n_PetscEvent, PetscEventConstructor> {
public:
  PetscErrorCode construct_(PetscEvent event) const noexcept
  {
    PetscFunctionBegin;
    PetscCall(PetscMemzero(event, sizeof(*event)));
    PetscCall(underlying().reset(event));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscErrorCode destroy_(PetscEvent event) const noexcept
  {
    PetscFunctionBegin;
    PetscCall(underlying().reset(event));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  static PetscErrorCode reset_(PetscEvent event) noexcept
  {
    PetscFunctionBegin;
    if (auto &destroy = event->destroy) {
      PetscCall((*destroy)(event));
      destroy = nullptr;
    }
    PetscAssert(!event->data, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Event failed to destroy its data member: %p", event->data);
    event->dctx_id    = 0;
    event->dctx_state = 0;
    event->dtype      = PETSC_DEVICE_DEFAULT();
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  static PetscErrorCode invalidate_(PetscEvent) noexcept { return PETSC_SUCCESS; }
};

static Petsc::ObjectPool<_n_PetscEvent, PetscEventConstructor> event_pool;

static PetscErrorCode PetscDeviceContextCreateEvent_Private(PetscDeviceContext dctx, PetscEvent *event)
{
  PetscFunctionBegin;
  PetscValidDeviceContext(dctx, 1);
  PetscValidPointer(event, 2);
  PetscCall(event_pool.allocate(event));
  PetscCall(PetscDeviceContextGetDeviceType(dctx, &(*event)->dtype));
  PetscTryTypeMethod(dctx, createevent, *event);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscEventDestroy_Private(PetscEvent *event)
{
  PetscFunctionBegin;
  PetscValidPointer(event, 1);
  if (*event) PetscCall(event_pool.deallocate(event));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDeviceContextRecordEvent_Private(PetscDeviceContext dctx, PetscEvent event)
{
  PetscObjectId    id;
  PetscObjectState state;

  PetscFunctionBegin;
  PetscValidDeviceContext(dctx, 1);
  PetscValidPointer(event, 2);
  id    = PetscObjectCast(dctx)->id;
  state = PetscObjectCast(dctx)->state;
  // technically state can never be less than event->dctx_state (only equal) but we include
  // it in the check just in case
  if ((id == event->dctx_id) && (state <= event->dctx_state)) PetscFunctionReturn(PETSC_SUCCESS);
  if (dctx->ops->recordevent) {
    // REVIEW ME:
    // TODO maybe move this to impls, as they can determine whether they can interoperate with
    // other device types more readily
    if (PetscDefined(USE_DEBUG) && (event->dtype != PETSC_DEVICE_HOST)) {
      PetscDeviceType dtype;

      PetscCall(PetscDeviceContextGetDeviceType(dctx, &dtype));
      PetscCheck(event->dtype == dtype, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Event type %s does not match device context type %s", PetscDeviceTypes[event->dtype], PetscDeviceTypes[dtype]);
    }
    PetscUseTypeMethod(dctx, recordevent, event);
  }
  event->dctx_id    = id;
  event->dctx_state = state;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDeviceContextWaitForEvent_Private(PetscDeviceContext dctx, PetscEvent event)
{
  PetscFunctionBegin;
  PetscValidDeviceContext(dctx, 1);
  PetscValidPointer(event, 2);
  // empty data implies you cannot wait on this event
  if (!event->data) PetscFunctionReturn(PETSC_SUCCESS);
  if (PetscDefined(USE_DEBUG)) {
    const auto      etype = event->dtype;
    PetscDeviceType dtype;

    PetscCall(PetscDeviceContextGetDeviceType(dctx, &dtype));
    PetscCheck(etype == dtype, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Event type %s does not match device context type %s", PetscDeviceTypes[etype], PetscDeviceTypes[dtype]);
  }
  if (PetscObjectCast(dctx)->id == event->dctx_id) PetscFunctionReturn(PETSC_SUCCESS);
  PetscTryTypeMethod(dctx, waitforevent, event);
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ==========================================================================================
// PetscStackFrame
//
// A helper class that (when debugging is enabled) contains the stack frame from which
// PetscDeviceContextMakrIntentFromID(). It is intended to be derived from, since this enables
// empty-base-class optimization to kick in when debugging is disabled.
// ==========================================================================================

template <bool use_debug>
struct PetscStackFrame;

template <>
struct PetscStackFrame</* use_debug = */ true> {
  std::string file{};
  std::string function{};
  int         line{};

  PetscStackFrame() = default;

  PetscStackFrame(const char *file_, const char *func_, int line_) noexcept : file(split_on_petsc_path_(file_)), function(func_), line(line_) { }

  bool operator==(const PetscStackFrame &other) const noexcept { return line == other.line && file == other.file && function == other.function; }

  PETSC_NODISCARD std::string to_string() const noexcept
  {
    std::string ret;

    ret = '(' + function + "() at " + file + ':' + std::to_string(line) + ')';
    return ret;
  }

private:
  static std::string split_on_petsc_path_(std::string &&in) noexcept
  {
    auto pos = in.find("petsc/src");

    if (pos == std::string::npos) pos = in.find("petsc/include");
    if (pos == std::string::npos) pos = 0;
    return in.substr(pos);
  }

  friend std::ostream &operator<<(std::ostream &os, const PetscStackFrame &frame)
  {
    os << frame.to_string();
    return os;
  }

  friend void swap(PetscStackFrame &lhs, PetscStackFrame &rhs) noexcept
  {
    using std::swap;

    swap(lhs.file, rhs.file);
    swap(lhs.function, rhs.function);
    swap(lhs.line, rhs.line);
  }
};

template <>
struct PetscStackFrame</* use_debug = */ false> {
  template <typename... T>
  constexpr PetscStackFrame(T &&...) noexcept
  {
  }

  constexpr bool operator==(const PetscStackFrame &) const noexcept { return true; }

  PETSC_NODISCARD static std::string to_string() noexcept { return "(unknown)"; }

  friend std::ostream &operator<<(std::ostream &os, const PetscStackFrame &) noexcept
  {
    os << "(unknown)";
    return os;
  }
};

// ==========================================================================================
// MarkedObjectMap
//
// A mapping from a PetscObjectId to a PetscEvent and (if debugging is enabled) a
// PetscStackFrame containing the location where PetscDeviceContextMarkIntentFromID was called
// ==========================================================================================

class MarkedObjectMap : public Petsc::RegisterFinalizeable<MarkedObjectMap> {
public:
  // Note we derive from PetscStackFrame so that the empty base class optimization can kick
  // in. If it were just a member it would still take up storage in optimized builds
  class snapshot_type : private PetscStackFrame<PetscDefined(USE_DEBUG) && !PetscDefined(HAVE_THREADSAFETY)> {
  public:
    using frame_type = PetscStackFrame<PetscDefined(USE_DEBUG) && !PetscDefined(HAVE_THREADSAFETY)>;

    snapshot_type() = default;
    snapshot_type(PetscDeviceContext, frame_type) noexcept;

    ~snapshot_type() noexcept;

    // movable
    snapshot_type(snapshot_type &&) noexcept;
    snapshot_type &operator=(snapshot_type &&) noexcept;

    // not copyable
    snapshot_type(const snapshot_type &) noexcept            = delete;
    snapshot_type &operator=(const snapshot_type &) noexcept = delete;

    PETSC_NODISCARD PetscEvent        event() const noexcept { return event_; }
    PETSC_NODISCARD const frame_type &frame() const noexcept { return *this; }
    PETSC_NODISCARD frame_type       &frame() noexcept { return *this; }

    PETSC_NODISCARD PetscObjectId dctx_id() const noexcept
    {
      PetscFunctionBegin;
      PetscAssertAbort(event(), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Snapshot %s does not contain an event!", frame().to_string().c_str());
      PetscFunctionReturn(event()->dctx_id);
    }

    PetscErrorCode ensure_event(PetscDeviceContext) noexcept;

    friend void swap(snapshot_type &, snapshot_type &) noexcept;

  private:
    PetscEvent event_{}; // the state of device context when this snapshot was recorded

    PETSC_NODISCARD static PetscEvent init_event_(PetscDeviceContext) noexcept;
  };

  // the "value" each key maps to
  struct mapped_type {
    using dependency_type = std::vector<snapshot_type>;

    mapped_type() noexcept;

    PetscMemoryAccessMode mode{PETSC_MEMORY_ACCESS_READ};
    snapshot_type         last_write{};
    dependency_type       dependencies{};
  };

  using map_type = Petsc::UnorderedMap<PetscObjectId, mapped_type>;

  map_type map;

private:
  friend RegisterFinalizeable;

  PetscErrorCode finalize_() noexcept;
};

// ==========================================================================================
// MarkedObjectMap::mapped_type -- Public API
// ==========================================================================================

// workaround for clang bug that produces the following warning
//
// src/sys/objects/device/interface/mark_dcontext.cxx:253:5: error: default member initializer
// for 'mode' needed within definition of enclosing class 'MarkedObjectMap' outside of member
// functions
//     mapped_type() noexcept = default;
//     ^
// https://stackoverflow.com/questions/53408962/try-to-understand-compiler-error-message-default-member-initializer-required-be
MarkedObjectMap::mapped_type::mapped_type() noexcept = default;

// ==========================================================================================
// MarkedObjectMap Private API
// ==========================================================================================

inline PetscErrorCode MarkedObjectMap::finalize_() noexcept
{
  PetscFunctionBegin;
  PetscCall(PetscInfo(nullptr, "Finalizing marked object map\n"));
  PetscCall(map.clear());
  PetscFunctionReturn(PETSC_SUCCESS);
}

// ==========================================================================================
// MarkedObjectMap::snapshot_type Private API
// ==========================================================================================

inline PetscEvent MarkedObjectMap::snapshot_type::init_event_(PetscDeviceContext dctx) noexcept
{
  PetscEvent event = nullptr;

  PetscFunctionBegin;
  PetscCallAbort(PETSC_COMM_SELF, PetscDeviceContextCreateEvent_Private(dctx, &event));
  PetscCallAbort(PETSC_COMM_SELF, PetscDeviceContextRecordEvent_Private(dctx, event));
  PetscFunctionReturn(event);
}

// ==========================================================================================
// MarkedObjectMap::snapshot_type Public API
// ==========================================================================================

MarkedObjectMap::snapshot_type::snapshot_type(PetscDeviceContext dctx, frame_type frame) noexcept : frame_type(std::move(frame)), event_(init_event_(dctx)) { }

MarkedObjectMap::snapshot_type::~snapshot_type() noexcept
{
  PetscFunctionBegin;
  PetscCallAbort(PETSC_COMM_SELF, PetscEventDestroy_Private(&event_));
  PetscFunctionReturnVoid();
}

// movable
MarkedObjectMap::snapshot_type::snapshot_type(snapshot_type &&other) noexcept : frame_type(std::move(other)), event_(Petsc::util::exchange(other.event_, nullptr)) { }

MarkedObjectMap::snapshot_type &MarkedObjectMap::snapshot_type::operator=(snapshot_type &&other) noexcept
{
  PetscFunctionBegin;
  if (this != &other) {
    frame_type::operator=(std::move(other));
    PetscCallAbort(PETSC_COMM_SELF, PetscEventDestroy_Private(&event_));
    event_ = Petsc::util::exchange(other.event_, nullptr);
  }
  PetscFunctionReturn(*this);
}

PetscErrorCode MarkedObjectMap::snapshot_type::ensure_event(PetscDeviceContext dctx) noexcept
{
  PetscFunctionBegin;
  if (PetscUnlikely(!event_)) PetscCall(PetscDeviceContextCreateEvent_Private(dctx, &event_));
  PetscFunctionReturn(PETSC_SUCCESS);
}

void swap(MarkedObjectMap::snapshot_type &lhs, MarkedObjectMap::snapshot_type &rhs) noexcept
{
  using std::swap;

  swap(lhs.frame(), rhs.frame());
  swap(lhs.event_, rhs.event_);
}

// A mapping between PetscObjectId (i.e. some PetscObject) to the list of PetscEvent's encoding
// the last time the PetscObject was accessed
static MarkedObjectMap marked_object_map;

// ==========================================================================================
// Utility Functions
// ==========================================================================================

PetscErrorCode PetscGetMarkedObjectMap_Internal(std::size_t *nkeys, PetscObjectId **keys, PetscMemoryAccessMode **modes, std::size_t **ndeps, PetscEvent ***dependencies)
{
  std::size_t i    = 0;
  const auto &map  = marked_object_map.map;
  const auto  size = *nkeys = map.size();

  PetscFunctionBegin;
  PetscCall(PetscMalloc4(size, keys, size, modes, size, ndeps, size, dependencies));
  for (auto it_ = map.begin(); it_ != map.end(); ++it_) {
    auto       &it = *it_;
    std::size_t j  = 0;

    (*keys)[i]         = it.first;
    (*modes)[i]        = it.second.mode;
    (*ndeps)[i]        = it.second.dependencies.size();
    (*dependencies)[i] = nullptr;
    PetscCall(PetscMalloc1((*ndeps)[i], (*dependencies) + i));
    for (auto &&dep : it.second.dependencies) (*dependencies)[i][j++] = dep.event();
    ++i;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscRestoreMarkedObjectMap_Internal(std::size_t nkeys, PetscObjectId **keys, PetscMemoryAccessMode **modes, std::size_t **ndeps, PetscEvent ***dependencies)
{
  PetscFunctionBegin;
  for (std::size_t i = 0; i < nkeys; ++i) PetscCall(PetscFree((*dependencies)[i]));
  PetscCall(PetscFree4(*keys, *modes, *ndeps, *dependencies));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <typename T>
static PetscErrorCode PetscDeviceContextMapIterVisitor(PetscDeviceContext dctx, T &&callback) noexcept
{
  const auto dctx_id    = PetscObjectCast(dctx)->id;
  auto      &dctx_deps  = CxxDataCast(dctx)->deps;
  auto      &object_map = marked_object_map.map;

  PetscFunctionBegin;
  for (auto &&dep : dctx_deps) {
    const auto mapit = object_map.find(dep);

    // Need this check since the final PetscDeviceContext may run through this *after* the map
    // has been finalized (and cleared), and hence might fail to find its dependencies. This is
    // perfectly valid since the user no longer cares about dangling dependencies after PETSc
    // is finalized
    if (PetscLikely(mapit != object_map.end())) {
      auto      &deps = mapit->second.dependencies;
      const auto end  = deps.end();
      const auto it   = std::remove_if(deps.begin(), end, [&](const MarkedObjectMap::snapshot_type &obj) { return obj.dctx_id() == dctx_id; });

      PetscCall(callback(mapit, deps.cbegin(), static_cast<decltype(deps.cend())>(it)));
      // remove ourselves
      PetscCallCXX(deps.erase(it, end));
      // continue to next object, but erase this one if it has no more dependencies
      if (deps.empty()) PetscCallCXX(object_map.erase(mapit));
    }
  }
  PetscCallCXX(dctx_deps.clear());
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscDeviceContextSyncClearMap_Internal(PetscDeviceContext dctx)
{
  using map_iterator = MarkedObjectMap::map_type::const_iterator;
  using dep_iterator = MarkedObjectMap::mapped_type::dependency_type::const_iterator;

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextMapIterVisitor(dctx, [&](map_iterator mapit, dep_iterator it, dep_iterator end) {
    PetscFunctionBegin;
    if (PetscDefined(USE_DEBUG_AND_INFO)) {
      std::ostringstream oss;
      const auto         mode = PetscMemoryAccessModeToString(mapit->second.mode);

      oss << "synced dctx " << PetscObjectCast(dctx)->id << ", remaining leaves for obj " << mapit->first << ": {";
      while (it != end) {
        oss << "[dctx " << it->dctx_id() << ", " << mode << ' ' << it->frame() << ']';
        if (++it != end) oss << ", ";
      }
      oss << '}';
      PetscCall(PetscInfo(nullptr, "%s\n", oss.str().c_str()));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
  }));
  {
    // the recursive sync clear map call is unbounded in case of a dependenct loop so we make a
    // copy
    // clang-format off
    const std::vector<CxxData::upstream_type::value_type> upstream_copy(
      std::make_move_iterator(CxxDataCast(dctx)->upstream.begin()),
      std::make_move_iterator(CxxDataCast(dctx)->upstream.end())
    );
    // clang-format on

    // aftermath, clear our set of parents (to avoid infinite recursion) and mark ourselves as no
    // longer contained (while the empty graph technically *is* always contained, it is not what
    // we mean by it)
    PetscCall(CxxDataCast(dctx)->clear());
    //dctx->contained = PETSC_FALSE;
    for (auto &&upstrm : upstream_copy) {
      // check that this parent still points to what we originally thought it was
      PetscCheck(upstrm.second.id == PetscObjectCast(upstrm.first)->id, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Upstream dctx %" PetscInt64_FMT " no longer exists, now has id %" PetscInt64_FMT, upstrm.second.id, PetscObjectCast(upstrm.first)->id);
      PetscCall(PetscDeviceContextSyncClearMap_Internal(upstrm.first));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscDeviceContextCheckNotOrphaned_Internal(PetscDeviceContext dctx)
{
  std::ostringstream oss;
  //const auto         allow = dctx->options.allow_orphans, contained = dctx->contained;
  const auto allow = true, contained = true;
  auto       wrote_to_oss = false;
  using map_iterator      = MarkedObjectMap::map_type::const_iterator;
  using dep_iterator      = MarkedObjectMap::mapped_type::dependency_type::const_iterator;

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextMapIterVisitor(dctx, [&](map_iterator mapit, dep_iterator it, dep_iterator end) {
    PetscFunctionBegin;
    if (allow || contained) PetscFunctionReturn(PETSC_SUCCESS);
    wrote_to_oss = true;
    oss << "- PetscObject (id " << mapit->first << "), intent " << PetscMemoryAccessModeToString(mapit->second.mode) << ' ' << it->frame();
    if (std::distance(it, end) == 0) oss << " (orphaned)"; // we were the only dependency
    oss << '\n';
    PetscFunctionReturn(PETSC_SUCCESS);
  }));
  PetscCheck(!wrote_to_oss, PETSC_COMM_SELF, PETSC_ERR_ORDER, "Destroying PetscDeviceContext ('%s', id %" PetscInt64_FMT ") would leave the following dangling (possibly orphaned) dependents:\n%s\nMust synchronize before destroying it, or allow it to be destroyed with orphans",
             PetscObjectCast(dctx)->name ? PetscObjectCast(dctx)->name : "unnamed", PetscObjectCast(dctx)->id, oss.str().c_str());
  PetscCall(CxxDataCast(dctx)->clear());
  PetscFunctionReturn(PETSC_SUCCESS);
}

#define DEBUG_INFO(mess, ...) PetscDebugInfo(dctx, "dctx %" PetscInt64_FMT " (%s) - obj %" PetscInt64_FMT " (%s): " mess, PetscObjectCast(dctx)->id, PetscObjectCast(dctx)->name ? PetscObjectCast(dctx)->name : "unnamed", id, name, ##__VA_ARGS__)

// The current mode is compatible with the previous mode (i.e. read-read) so we need only
// update the existing version and possibly appeand ourselves to the dependency list

template <bool use_debug>
static PetscErrorCode MarkFromID_CompatibleModes(MarkedObjectMap::mapped_type &marked, PetscDeviceContext dctx, PetscObjectId id, PetscMemoryAccessMode mode, PetscStackFrame<use_debug> &frame, const char *PETSC_UNUSED name, bool *update_object_dependencies)
{
  const auto dctx_id             = PetscObjectCast(dctx)->id;
  auto      &object_dependencies = marked.dependencies;
  const auto end                 = object_dependencies.end();
  const auto it                  = std::find_if(object_dependencies.begin(), end, [&](const MarkedObjectMap::snapshot_type &obj) { return obj.dctx_id() == dctx_id; });

  PetscFunctionBegin;
  PetscCall(DEBUG_INFO("new mode (%s) COMPATIBLE with %s mode (%s), no need to serialize\n", PetscMemoryAccessModeToString(mode), object_dependencies.empty() ? "default" : "old", PetscMemoryAccessModeToString(marked.mode)));
  (void)mode;
  if (it != end) {
    using std::swap;

    // we have been here before, all we must do is update our entry then we can bail
    PetscCall(DEBUG_INFO("found old self as dependency, updating\n"));
    PetscAssert(CxxDataCast(dctx)->deps.find(id) != CxxDataCast(dctx)->deps.end(), PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscDeviceContext %" PetscInt64_FMT " listed as dependency for object %" PetscInt64_FMT " (%s), but does not have the object in private dependency list!", dctx_id, id, name);
    swap(it->frame(), frame);
    PetscCall(PetscDeviceContextRecordEvent_Private(dctx, it->event()));
    *update_object_dependencies = false;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  // we have not been here before, need to serialize with the last write event (if it exists)
  // and add ourselves to the dependency list
  if (const auto event = marked.last_write.event()) PetscCall(PetscDeviceContextWaitForEvent_Private(dctx, event));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <bool use_debug>
static PetscErrorCode MarkFromID_IncompatibleModes_UpdateLastWrite(MarkedObjectMap::mapped_type &marked, PetscDeviceContext dctx, PetscObjectId id, PetscMemoryAccessMode mode, PetscStackFrame<use_debug> &frame, const char *PETSC_UNUSED name, bool *update_object_dependencies)
{
  const auto      dctx_id    = PetscObjectCast(dctx)->id;
  auto           &last_write = marked.last_write;
  auto           &last_dep   = marked.dependencies.back();
  PetscDeviceType dtype;

  PetscFunctionBegin;
  PetscAssert(marked.dependencies.size() == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Can only have a single writer as dependency, have %zu!", marked.dependencies.size());
  PetscCall(PetscDeviceContextGetDeviceType(dctx, &dtype));
  if (last_dep.event()->dtype != dtype) {
    PetscCall(DEBUG_INFO("moving last write dependency (intent %s)\n", PetscMemoryAccessModeToString(marked.mode)));
    last_write = std::move(last_dep);
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  // we match the device type of the dependency, we can reuse its event!
  auto      &dctx_upstream_deps     = CxxDataCast(dctx)->deps;
  const auto last_write_was_also_us = last_write.event() && (last_write.dctx_id() == dctx_id);
  using std::swap;

  PetscCall(DEBUG_INFO("we matched the previous write dependency's (intent %s) device type (%s), swapping last dependency with last write\n", PetscMemoryAccessModeToString(marked.mode), PetscDeviceTypes[dtype]));
  if (last_dep.event()->dctx_id != dctx_id) dctx_upstream_deps.emplace(id);
  PetscAssert(dctx_upstream_deps.find(id) != dctx_upstream_deps.end(), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Did not find id %" PetscInt64_FMT "in object dependencies, but we have apparently recorded the last dependency %s!", id,
              last_write.frame().to_string().c_str());
  swap(last_write, last_dep);
  if (last_write_was_also_us) {
    PetscCall(DEBUG_INFO("we were also the last write event (intent %s), updating\n", PetscMemoryAccessModeToString(mode)));
    (void)mode;
    // we are both the last to write *and* the last to leave a write event. This is the
    // fast path, we only need to update the frame and update the recorded event
    swap(last_dep.frame(), frame);
    // last used to be last_write which is not guaranteed to have an event, so must
    // create it now
    PetscCall(last_dep.ensure_event(dctx));
    PetscCall(PetscDeviceContextRecordEvent_Private(dctx, last_dep.event()));
    *update_object_dependencies = false;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// The current mode is NOT compatible with the previous mode. We must serialize with all events
// in the dependency list, possibly clear it, and update the previous write event

template <bool use_debug>
static PetscErrorCode MarkFromID_IncompatibleModes(MarkedObjectMap::mapped_type &marked, PetscDeviceContext dctx, PetscObjectId id, PetscMemoryAccessMode mode, PetscStackFrame<use_debug> &frame, const char *name, bool *update_object_dependencies)
{
  auto &old_mode            = marked.mode;
  auto &object_dependencies = marked.dependencies;

  PetscFunctionBegin;
  // we are NOT compatible with the previous mode
  PetscCall(DEBUG_INFO("new mode (%s) NOT COMPATIBLE with %s mode (%s), serializing then clearing (%zu) %s\n", PetscMemoryAccessModeToString(mode), object_dependencies.empty() ? "default" : "old", PetscMemoryAccessModeToString(old_mode),
                       object_dependencies.size(), object_dependencies.size() == 1 ? "dependency" : "dependencies"));

  for (const auto &dep : object_dependencies) PetscCall(PetscDeviceContextWaitForEvent_Private(dctx, dep.event()));
  // if the previous mode wrote, update the last write node with it
  if (PetscMemoryAccessWrite(old_mode)) PetscCall(MarkFromID_IncompatibleModes_UpdateLastWrite(marked, dctx, id, mode, frame, name, update_object_dependencies));

  old_mode = mode;
  // clear out the old dependencies if are about to append ourselves
  if (*update_object_dependencies) object_dependencies.clear();
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <bool use_debug>
static PetscErrorCode PetscDeviceContextMarkIntentFromID_Private(PetscDeviceContext dctx, PetscObjectId id, PetscMemoryAccessMode mode, PetscStackFrame<use_debug> frame, const char *name)
{
  auto &marked                     = marked_object_map.map[id];
  auto &object_dependencies        = marked.dependencies;
  auto  update_object_dependencies = true;

  PetscFunctionBegin;
  if ((marked.mode == PETSC_MEMORY_ACCESS_READ) && (mode == PETSC_MEMORY_ACCESS_READ)) {
    PetscCall(MarkFromID_CompatibleModes(marked, dctx, id, mode, frame, name, &update_object_dependencies));
  } else {
    PetscCall(MarkFromID_IncompatibleModes(marked, dctx, id, mode, frame, name, &update_object_dependencies));
  }
  if (update_object_dependencies) {
    // become the new leaf by appending ourselves
    PetscCall(DEBUG_INFO("%s with intent %s\n", object_dependencies.empty() ? "dependency list is empty, creating new leaf" : "appending to existing leaves", PetscMemoryAccessModeToString(mode)));
    PetscCallCXX(object_dependencies.emplace_back(dctx, std::move(frame)));
    PetscCallCXX(CxxDataCast(dctx)->deps.emplace(id));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#undef DEBUG_INFO

/*@C
  PetscDeviceContextMarkIntentFromID - Indicate a `PetscDeviceContext`s access intent to the
  auto-dependency system

  Not Collective

  Input Parameters:
+ dctx - The `PetscDeviceContext`
. id   - The `PetscObjectId` to mark
. mode - The desired access intent
- name - The object name (for debug purposes, ignored in optimized builds)

  Notes:
  This routine formally informs the dependency system that `dctx` will access the object
  represented by `id` with `mode` and adds `dctx` to `id`'s list of dependencies (termed
  "leaves").

  If the existing set of leaves have an incompatible `PetscMemoryAccessMode` to `mode`, `dctx`
  will be serialized against them.

  Level: intermediate

.seealso: `PetscDeviceContextWaitForContext()`, `PetscDeviceContextSynchronize()`,
`PetscObjectGetId()`, `PetscMemoryAccessMode`
@*/
PetscErrorCode PetscDeviceContextMarkIntentFromID(PetscDeviceContext dctx, PetscObjectId id, PetscMemoryAccessMode mode, const char name[])
{
#if PetscDefined(USE_DEBUG) && !PetscDefined(HAVE_THREADSAFETY)
  const auto index    = petscstack.currentsize > 2 ? petscstack.currentsize - 2 : 0;
  const auto file     = petscstack.file[index];
  const auto function = petscstack.function[index];
  const auto line     = petscstack.line[index];
#else
  constexpr const char *file     = nullptr;
  constexpr const char *function = nullptr;
  constexpr auto        line     = 0;
#endif

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  if (name) PetscValidCharPointer(name, 4);
  PetscCall(marked_object_map.register_finalize());
  PetscCall(PetscLogEventBegin(DCONTEXT_Mark, dctx, nullptr, nullptr, nullptr));
  PetscCall(PetscDeviceContextMarkIntentFromID_Private(dctx, id, mode, MarkedObjectMap::snapshot_type::frame_type{file, function, line}, name ? name : "unknown object"));
  PetscCall(PetscLogEventEnd(DCONTEXT_Mark, dctx, nullptr, nullptr, nullptr));
  PetscFunctionReturn(PETSC_SUCCESS);
}
