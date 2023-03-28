#include "petscdevice_interface_internal.hpp" /*I <petscdevice.h> I*/
#include <petsc/private/viewerimpl.h>         // _p_PetscViewer for PetscObjectCast()

#include <petsc/private/cpp/object_pool.hpp>
#include <petsc/private/cpp/utility.hpp>
#include <petsc/private/cpp/array.hpp>

#include <vector>
#include <string> // std::to_string among other things

/* Define the allocator */
class PetscDeviceContextConstructor : public Petsc::ConstructorInterface<_p_PetscDeviceContext, PetscDeviceContextConstructor> {
public:
  PetscErrorCode construct_(PetscDeviceContext dctx) const noexcept
  {
    PetscFunctionBegin;
    PetscCall(PetscArrayzero(dctx, 1));
    PetscCall(PetscHeaderInitialize_Private(dctx, PETSC_DEVICE_CONTEXT_CLASSID, "PetscDeviceContext", "PetscDeviceContext", "Sys", PETSC_COMM_SELF, PetscDeviceContextDestroy, PetscDeviceContextView));
    PetscCallCXX(PetscObjectCast(dctx)->cpp = new CxxData());
    PetscCall(underlying().reset(dctx, false));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  static PetscErrorCode destroy_(PetscDeviceContext dctx) noexcept
  {
    PetscFunctionBegin;
    PetscAssert(!dctx->numChildren, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Device context still has %" PetscInt_FMT " un-joined children, must call PetscDeviceContextJoin() with all children before destroying", dctx->numChildren);
    PetscTryTypeMethod(dctx, destroy);
    PetscCall(PetscDeviceDestroy(&dctx->device));
    PetscCall(PetscFree(dctx->childIDs));
    delete CxxDataCast(dctx);
    PetscCall(PetscHeaderDestroy_Private(PetscObjectCast(dctx), PETSC_FALSE));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  static PetscErrorCode reset_(PetscDeviceContext dctx, bool zero = true) noexcept
  {
    PetscFunctionBegin;
    if (zero) {
      // reset the device if the user set it
      if (Petsc::util::exchange(dctx->usersetdevice, PETSC_FALSE)) {
        PetscTryTypeMethod(dctx, destroy);
        PetscCall(PetscDeviceDestroy(&dctx->device));
        PetscCall(PetscArrayzero(dctx->ops, 1));
        dctx->data = nullptr;
      }
      PetscCall(PetscHeaderReset_Internal(PetscObjectCast(dctx)));
      dctx->numChildren = 0;
      dctx->setup       = PETSC_FALSE;
      // don't deallocate the child array, rather just zero it out
      PetscCall(PetscArrayzero(dctx->childIDs, dctx->maxNumChildren));
      PetscCall(CxxDataCast(dctx)->clear());
    }
    dctx->streamType = PETSC_STREAM_DEFAULT_BLOCKING;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  static PetscErrorCode invalidate_(PetscDeviceContext) noexcept { return PETSC_SUCCESS; }
};

static Petsc::ObjectPool<_p_PetscDeviceContext, PetscDeviceContextConstructor> contextPool;

/*@C
  PetscDeviceContextCreate - Creates a `PetscDeviceContext`

  Not Collective

  Output Parameter:
. dctx - The `PetscDeviceContext`

  Level: beginner

  Note:
  Unlike almost every other PETSc class it is advised that most users use
  `PetscDeviceContextDuplicate()` rather than this routine to create new contexts. Contexts of
  different types are incompatible with one another; using `PetscDeviceContextDuplicate()`
  ensures compatible types.

  DAG representation:
.vb
  time ->

  |= CALL =| - dctx ->
.ve

.N ASYNC_API

.seealso: `PetscDeviceContextDuplicate()`, `PetscDeviceContextSetDevice()`,
`PetscDeviceContextSetStreamType()`, `PetscDeviceContextSetUp()`,
`PetscDeviceContextSetFromOptions()`, `PetscDeviceContextView()`, `PetscDeviceContextDestroy()`
@*/
PetscErrorCode PetscDeviceContextCreate(PetscDeviceContext *dctx)
{
  PetscFunctionBegin;
  PetscValidPointer(dctx, 1);
  PetscCall(PetscDeviceInitializePackage());
  PetscCall(PetscLogEventBegin(DCONTEXT_Create, nullptr, nullptr, nullptr, nullptr));
  PetscCall(contextPool.allocate(dctx));
  PetscCall(PetscLogEventEnd(DCONTEXT_Create, nullptr, nullptr, nullptr, nullptr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscDeviceContextDestroy - Frees a `PetscDeviceContext`

  Not Collective

  Input Parameter:
. dctx - The `PetscDeviceContext`

  Level: beginner

  Notes:
  No implicit synchronization occurs due to this routine, all resources are released completely
  asynchronously w.r.t. the host. If one needs to guarantee access to the data produced on
  `dctx`'s stream the user is responsible for calling `PetscDeviceContextSynchronize()` before
  calling this routine.

  DAG representation:
.vb
  time ->

  -> dctx - |= CALL =|
.ve

  Developer Notes:
  `dctx` is never actually "destroyed" in the classical sense. It is returned to an ever
  growing pool of `PetscDeviceContext`s. There are currently no limits on the size of the pool,
  this should perhaps be implemented.

.N ASYNC_API

.seealso: `PetscDeviceContextCreate()`, `PetscDeviceContextSetDevice()`,
`PetscDeviceContextSetUp()`, `PetscDeviceContextSynchronize()`
@*/
PetscErrorCode PetscDeviceContextDestroy(PetscDeviceContext *dctx)
{
  PetscFunctionBegin;
  PetscValidPointer(dctx, 1);
  if (!*dctx) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscLogEventBegin(DCONTEXT_Destroy, nullptr, nullptr, nullptr, nullptr));
  if (--(PetscObjectCast(*dctx)->refct) <= 0) {
    PetscCall(PetscDeviceContextCheckNotOrphaned_Internal(*dctx));
    PetscCall(contextPool.deallocate(dctx));
  }
  PetscCall(PetscLogEventEnd(DCONTEXT_Destroy, nullptr, nullptr, nullptr, nullptr));
  *dctx = nullptr;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscDeviceContextSetStreamType - Set the implementation type of the underlying stream for a
  `PetscDeviceContext`

  Not Collective

  Input Parameters:
+ dctx - The `PetscDeviceContext`
- type - The `PetscStreamType`

  Level: beginner

  Note:
  See `PetscStreamType` in `include/petscdevicetypes.h` for more information on the available
  types and their interactions. If the `PetscDeviceContext` was previously set up and stream
  type was changed, you must call `PetscDeviceContextSetUp()` again after this routine.

.seealso: `PetscStreamType`, `PetscDeviceContextGetStreamType()`, `PetscDeviceContextCreate()`,
`PetscDeviceContextSetUp()`, `PetscDeviceContextSetFromOptions()`
@*/
PetscErrorCode PetscDeviceContextSetStreamType(PetscDeviceContext dctx, PetscStreamType type)
{
  PetscFunctionBegin;
  // do not use getoptionalnullcontext here since we do not want the user to change the stream
  // type
  PetscValidDeviceContext(dctx, 1);
  PetscValidStreamType(type, 2);
  // only need to do complex swapping if the object has already been setup
  if (dctx->setup && (dctx->streamType != type)) {
    dctx->setup = PETSC_FALSE;
    PetscCall(PetscLogEventBegin(DCONTEXT_ChangeStream, dctx, nullptr, nullptr, nullptr));
    PetscUseTypeMethod(dctx, changestreamtype, type);
    PetscCall(PetscLogEventEnd(DCONTEXT_ChangeStream, dctx, nullptr, nullptr, nullptr));
  }
  dctx->streamType = type;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscDeviceContextGetStreamType - Get the implementation type of the underlying stream for a
  `PetscDeviceContext`

  Not Collective

  Input Parameter:
. dctx - The `PetscDeviceContext`

  Output Parameter:
. type - The `PetscStreamType`

  Level: beginner

  Note:
  See `PetscStreamType` in `include/petscdevicetypes.h` for more information on the available
  types and their interactions

.seealso: `PetscDeviceContextSetStreamType()`, `PetscDeviceContextCreate()`,
`PetscDeviceContextSetFromOptions()`
@*/
PetscErrorCode PetscDeviceContextGetStreamType(PetscDeviceContext dctx, PetscStreamType *type)
{
  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscValidIntPointer(type, 2);
  *type = dctx->streamType;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Actual function to set the device.

  1. Repeatedly destroying and recreating internal data structures (like streams and events)
     for recycled PetscDeviceContexts is not free. If done often, it does add up.
  2. The vast majority of PetscDeviceContexts are created by PETSc either as children or
     default contexts. The default contexts *never* change type, and the children are extremely
     unlikely to (chances are if you fork once, you will fork again very soon).
  3. The only time this calculus changes is if the user themselves sets the device type. In
     this case we do not know what the user has changed, so must always wipe the slate clean.

  Thus we need to keep track whether the user explicitly sets the device contexts device.
*/
static PetscErrorCode PetscDeviceContextSetDevice_Private(PetscDeviceContext dctx, PetscDevice device, PetscBool user_set)
{
  PetscFunctionBegin;
  // do not use getoptionalnullcontext here since we do not want the user to change its device
  PetscValidDeviceContext(dctx, 1);
  PetscValidDevice(device, 2);
  if (dctx->device && (dctx->device->id == device->id)) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscLogEventBegin(DCONTEXT_SetDevice, dctx, nullptr, nullptr, nullptr));
  PetscTryTypeMethod(dctx, destroy);
  PetscCall(PetscDeviceDestroy(&dctx->device));
  PetscCall(PetscMemzero(dctx->ops, sizeof(*dctx->ops)));
  PetscCall(PetscDeviceReference_Internal(device));
  // set it before calling the method
  dctx->device = device;
  PetscCall((*device->ops->createcontext)(dctx));
  PetscCall(PetscLogEventEnd(DCONTEXT_SetDevice, dctx, nullptr, nullptr, nullptr));
  dctx->setup         = PETSC_FALSE;
  dctx->usersetdevice = user_set;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscDeviceContextSetDefaultDeviceForType_Internal(PetscDeviceContext dctx, PetscDeviceType type)
{
  PetscDevice device;

  PetscFunctionBegin;
  PetscCall(PetscDeviceGetDefaultForType_Internal(type, &device));
  PetscCall(PetscDeviceContextSetDevice_Private(dctx, device, PETSC_FALSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscDeviceContextSetDevice - Set the underlying `PetscDevice` for a `PetscDeviceContext`

  Not Collective

  Input Parameters:
+ dctx   - The `PetscDeviceContext`
- device - The `PetscDevice`

  Level: intermediate

  Notes:
  This routine is effectively `PetscDeviceContext`'s "set-type" (so every `PetscDeviceContext` must
  also have an attached `PetscDevice`). Unlike the usual set-type semantics, it is not strictly
  necessary to set a contexts device to enable usage, any created `PetscDeviceContext`s will
  always come equipped with the "default" device.

  This routine is a no-op if `device` is already attached to `dctx`.

  This routine may (but is very unlikely to) initialize the backend device and may incur
  synchronization.

.seealso: `PetscDeviceCreate()`, `PetscDeviceConfigure()`, `PetscDeviceContextGetDevice()`,
`PetscDeviceContextGetDeviceType()`
@*/
PetscErrorCode PetscDeviceContextSetDevice(PetscDeviceContext dctx, PetscDevice device)
{
  PetscFunctionBegin;
  PetscCall(PetscDeviceContextSetDevice_Private(dctx, device, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscDeviceContextGetDevice - Get the underlying `PetscDevice` for a `PetscDeviceContext`

  Not Collective

  Input Parameter:
. dctx - the `PetscDeviceContext`

  Output Parameter:
. device - The `PetscDevice`

  Level: intermediate

  Note:
  This is a borrowed reference, the user should not destroy `device`.

.seealso: `PetscDeviceContextSetDevice()`, `PetscDevice`, `PetscDeviceContextGetDeviceType()`
@*/
PetscErrorCode PetscDeviceContextGetDevice(PetscDeviceContext dctx, PetscDevice *device)
{
  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscValidPointer(device, 2);
  PetscAssert(dctx->device, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "PetscDeviceContext %" PetscInt64_FMT " has no attached PetscDevice to get", PetscObjectCast(dctx)->id);
  *device = dctx->device;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscDeviceContextGetDeviceType - Get the `PetscDeviceType` for a `PetscDeviceContext`

  Not Collective

  Input Parameter:
. dctx - The `PetscDeviceContext`

  Output Parameter:
. type - The `PetscDeviceType`

  Level: beginner

  Note:
  This routine is a convenience shorthand for `PetscDeviceContextGetDevice()` ->
  `PetscDeviceGetType()`.

.seealso: `PetscDeviceType`, `PetscDeviceContextGetDevice()`, `PetscDeviceGetType()`, `PetscDevice`
@*/
PetscErrorCode PetscDeviceContextGetDeviceType(PetscDeviceContext dctx, PetscDeviceType *type)
{
  PetscDevice device = nullptr;

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscValidPointer(type, 2);
  PetscCall(PetscDeviceContextGetDevice(dctx, &device));
  PetscCall(PetscDeviceGetType(device, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscDeviceContextSetUp - Prepares a `PetscDeviceContext` for use

  Not Collective

  Input Parameter:
. dctx - The `PetscDeviceContext`

  Level: beginner

  Developer Note:
  This routine is usually the stage where a `PetscDeviceContext` acquires device-side data
  structures such as streams, events, and (possibly) handles.

.seealso: `PetscDeviceContextCreate()`, `PetscDeviceContextSetDevice()`,
`PetscDeviceContextDestroy()`, `PetscDeviceContextSetFromOptions()`
@*/
PetscErrorCode PetscDeviceContextSetUp(PetscDeviceContext dctx)
{
  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  if (dctx->setup) PetscFunctionReturn(PETSC_SUCCESS);
  if (!dctx->device) {
    const auto default_dtype = PETSC_DEVICE_DEFAULT();

    PetscCall(PetscInfo(dctx, "PetscDeviceContext %" PetscInt64_FMT " did not have an explicitly attached PetscDevice, using default with type %s\n", PetscObjectCast(dctx)->id, PetscDeviceTypes[default_dtype]));
    PetscCall(PetscDeviceContextSetDefaultDeviceForType_Internal(dctx, default_dtype));
  }
  PetscCall(PetscLogEventBegin(DCONTEXT_SetUp, dctx, nullptr, nullptr, nullptr));
  PetscUseTypeMethod(dctx, setup);
  PetscCall(PetscLogEventEnd(DCONTEXT_SetUp, dctx, nullptr, nullptr, nullptr));
  dctx->setup = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDeviceContextDuplicate_Private(PetscDeviceContext dctx, PetscStreamType stype, PetscDeviceContext *dctxdup)
{
  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(DCONTEXT_Duplicate, dctx, nullptr, nullptr, nullptr));
  PetscCall(PetscDeviceContextCreate(dctxdup));
  PetscCall(PetscDeviceContextSetStreamType(*dctxdup, stype));
  if (const auto device = dctx->device) PetscCall(PetscDeviceContextSetDevice_Private(*dctxdup, device, dctx->usersetdevice));
  PetscCall(PetscDeviceContextSetUp(*dctxdup));
  PetscCall(PetscLogEventEnd(DCONTEXT_Duplicate, dctx, nullptr, nullptr, nullptr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscDeviceContextDuplicate - Duplicates a `PetscDeviceContext` object

  Not Collective

  Input Parameter:
. dctx - The `PetscDeviceContext` to duplicate

  Output Parameter:
. dctxdup - The duplicated `PetscDeviceContext`

  Level: beginner

  Notes:
  This is a shorthand method for creating a `PetscDeviceContext` with the exact same settings as
  another. Note however that `dctxdup` does not share any of the underlying data with `dctx`,
  (including its current stream-state) they are completely separate objects.

  There is no implied ordering between `dctx` or `dctxdup`.

  DAG representation:
.vb
  time ->

  -> dctx - |= CALL =| - dctx ---->
                       - dctxdup ->
.ve

.seealso: `PetscDeviceContextCreate()`, `PetscDeviceContextSetDevice()`,
`PetscDeviceContextSetStreamType()`
@*/
PetscErrorCode PetscDeviceContextDuplicate(PetscDeviceContext dctx, PetscDeviceContext *dctxdup)
{
  auto stype = PETSC_STREAM_DEFAULT_BLOCKING;

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscValidPointer(dctxdup, 2);
  PetscCall(PetscDeviceContextGetStreamType(dctx, &stype));
  PetscCall(PetscDeviceContextDuplicate_Private(dctx, stype, dctxdup));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscDeviceContextQueryIdle - Returns whether or not a `PetscDeviceContext` is idle

  Not Collective

  Input Parameter:
. dctx - The `PetscDeviceContext`

  Output Parameter:
. idle - `PETSC_TRUE` if `dctx` has NO work, `PETSC_FALSE` if it has work

  Level: intermediate

  Note:
  This routine only refers a singular context and does NOT take any of its children into
  account. That is, if `dctx` is idle but has dependents who do have work this routine still
  returns `PETSC_TRUE`.

.seealso: `PetscDeviceContextCreate()`, `PetscDeviceContextWaitForContext()`, `PetscDeviceContextFork()`
@*/
PetscErrorCode PetscDeviceContextQueryIdle(PetscDeviceContext dctx, PetscBool *idle)
{
  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscValidBoolPointer(idle, 2);
  PetscCall(PetscLogEventBegin(DCONTEXT_QueryIdle, dctx, nullptr, nullptr, nullptr));
  PetscUseTypeMethod(dctx, query, idle);
  PetscCall(PetscLogEventEnd(DCONTEXT_QueryIdle, dctx, nullptr, nullptr, nullptr));
  PetscCall(PetscInfo(dctx, "PetscDeviceContext ('%s', id %" PetscInt64_FMT ") %s idle\n", PetscObjectCast(dctx)->name ? PetscObjectCast(dctx)->name : "unnamed", PetscObjectCast(dctx)->id, *idle ? "was" : "was not"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscDeviceContextWaitForContext - Make one context wait for another context to finish

  Not Collective

  Input Parameters:
+ dctxa - The `PetscDeviceContext` object that is waiting
- dctxb - The `PetscDeviceContext` object that is being waited on

  Level: beginner

  Notes:
  Serializes two `PetscDeviceContext`s. Serialization is performed asynchronously; the host
  does not wait for the serialization to actually occur.

  This routine uses only the state of `dctxb` at the moment this routine was called, so any
  future work queued will not affect `dctxa`. It is safe to pass the same context to both
  arguments (in which case this routine does nothing).

  DAG representation:
.vb
  time ->

  -> dctxa ---/- |= CALL =| - dctxa ->
             /
  -> dctxb -/------------------------>
.ve

.N ASYNC_API

.seealso: `PetscDeviceContextCreate()`, `PetscDeviceContextQueryIdle()`, `PetscDeviceContextJoin()`
@*/
PetscErrorCode PetscDeviceContextWaitForContext(PetscDeviceContext dctxa, PetscDeviceContext dctxb)
{
  PetscObject aobj;

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctxa));
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctxb));
  PetscCheckCompatibleDeviceContexts(dctxa, 1, dctxb, 2);
  if (dctxa == dctxb) PetscFunctionReturn(PETSC_SUCCESS);
  aobj = PetscObjectCast(dctxa);
  PetscCall(PetscLogEventBegin(DCONTEXT_WaitForCtx, dctxa, dctxb, nullptr, nullptr));
  PetscUseTypeMethod(dctxa, waitforcontext, dctxb);
  PetscCallCXX(CxxDataCast(dctxa)->upstream[dctxb] = CxxDataParent(dctxb));
  PetscCall(PetscLogEventEnd(DCONTEXT_WaitForCtx, dctxa, dctxb, nullptr, nullptr));
  PetscCall(PetscInfo(dctxa, "dctx %" PetscInt64_FMT " waiting on dctx %" PetscInt64_FMT "\n", aobj->id, PetscObjectCast(dctxb)->id));
  PetscCall(PetscObjectStateIncrease(aobj));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscDeviceContextForkWithStreamType - Create a set of dependent child contexts from a parent
  context with a prescribed `PetscStreamType`

  Not Collective, Asynchronous

  Input Parameters:
+ dctx  - The parent `PetscDeviceContext`
. stype - The prescribed `PetscStreamType`
- n     - The number of children to create

  Output Parameter:
. dsub - The created child context(s)

  Level: intermediate

  Notes:
  This routine creates `n` edges of a DAG from a source node which are causally dependent on the
  source node. This causal dependency is established as-if by calling
  `PetscDeviceContextWaitForContext()` on every child.

  `dsub` is allocated by this routine and has its lifetime bounded by `dctx`. That is, `dctx`
  expects to free `dsub` (via `PetscDeviceContextJoin()`) before it itself is destroyed.

  This routine only accounts for work queued on `dctx` up until calling this routine, any
  subsequent work enqueued on `dctx` has no effect on `dsub`.

  The `PetscStreamType` of `dctx` does not have to equal `stype`. In fact, it is often the case
  that they are different. This is useful in cases where a routine can locally exploit stream
  parallelism without needing to worry about what stream type the incoming `PetscDeviceContext`
  carries.

  DAG representation:
.vb
  time ->

  -> dctx - |= CALL =| -\----> dctx ------>
                         \---> dsub[0] --->
                          \--> ... ------->
                           \-> dsub[n-1] ->
.ve

.N ASYNC_API

.seealso: `PetscDeviceContextJoin()`, `PetscDeviceContextSynchronize()`,
`PetscDeviceContextQueryIdle()`, `PetscDeviceContextWaitForContext()`
@*/
PetscErrorCode PetscDeviceContextForkWithStreamType(PetscDeviceContext dctx, PetscStreamType stype, PetscInt n, PetscDeviceContext **dsub)
{
  // debugging only
  std::string idList;
  auto        ninput = n;

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscAssert(n >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Number of contexts requested %" PetscInt_FMT " < 0", n);
  PetscValidPointer(dsub, 4);
  *dsub = nullptr;
  /* reserve 4 chars per id, 2 for number and 2 for ', ' separator */
  if (PetscDefined(USE_DEBUG_AND_INFO)) PetscCallCXX(idList.reserve(4 * n));
  PetscCall(PetscLogEventBegin(DCONTEXT_Fork, dctx, nullptr, nullptr, nullptr));
  /* update child totals */
  dctx->numChildren += n;
  /* now to find out if we have room */
  if (dctx->numChildren > dctx->maxNumChildren) {
    const auto numChildren    = dctx->numChildren;
    auto      &maxNumChildren = dctx->maxNumChildren;
    auto       numAllocated   = numChildren;

    /* no room, either from having too many kids or not having any */
    if (auto &childIDs = dctx->childIDs) {
      // the difference is backwards because we have not updated maxNumChildren yet
      numAllocated -= maxNumChildren;
      /* have existing children, must reallocate them */
      PetscCall(PetscRealloc(numChildren * sizeof(*childIDs), &childIDs));
      /* clear the extra memory since realloc doesn't do it for us */
      PetscCall(PetscArrayzero(std::next(childIDs, maxNumChildren), numAllocated));
    } else {
      /* have no children */
      PetscCall(PetscCalloc1(numChildren, &childIDs));
    }
    /* update total number of children */
    maxNumChildren = numChildren;
  }
  PetscCall(PetscMalloc1(n, dsub));
  for (PetscInt i = 0; ninput && (i < dctx->numChildren); ++i) {
    auto &childID = dctx->childIDs[i];
    /* empty child slot */
    if (!childID) {
      auto &childctx = (*dsub)[i];

      /* create the child context in the image of its parent */
      PetscCall(PetscDeviceContextDuplicate_Private(dctx, stype, &childctx));
      PetscCall(PetscDeviceContextWaitForContext(childctx, dctx));
      /* register the child with its parent */
      PetscCall(PetscObjectGetId(PetscObjectCast(childctx), &childID));
      if (PetscDefined(USE_DEBUG_AND_INFO)) {
        PetscCallCXX(idList += std::to_string(childID));
        if (ninput != 1) PetscCallCXX(idList += ", ");
      }
      --ninput;
    }
  }
  PetscCall(PetscLogEventEnd(DCONTEXT_Fork, dctx, nullptr, nullptr, nullptr));
  PetscCall(PetscDebugInfo(dctx, "Forked %" PetscInt_FMT " children from parent %" PetscInt64_FMT " with IDs: %s\n", n, PetscObjectCast(dctx)->id, idList.c_str()));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscDeviceContextFork - Create a set of dependent child contexts from a parent context

  Not Collective, Asynchronous

  Input Parameters:
+ dctx - The parent `PetscDeviceContext`
- n    - The number of children to create

  Output Parameter:
. dsub - The created child context(s)

  Level: beginner

  Notes:
  Behaves identically to `PetscDeviceContextForkWithStreamType()` except that the prescribed
  `PetscStreamType` is taken from `dctx`. In effect this routine is shorthand for\:

.vb
  PetscStreamType stype;

  PetscDeviceContextGetStreamType(dctx, &stype);
  PetscDeviceContextForkWithStreamType(dctx, stype, ...);
.ve

.N ASYNC_API

.seealso: `PetscDeviceContextForkWithStreamType()`, `PetscDeviceContextJoin()`,
`PetscDeviceContextSynchronize()`, `PetscDeviceContextQueryIdle()`
@*/
PetscErrorCode PetscDeviceContextFork(PetscDeviceContext dctx, PetscInt n, PetscDeviceContext **dsub)
{
  auto stype = PETSC_STREAM_DEFAULT_BLOCKING;

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscCall(PetscDeviceContextGetStreamType(dctx, &stype));
  PetscCall(PetscDeviceContextForkWithStreamType(dctx, stype, n, dsub));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscDeviceContextJoin - Converge a set of child contexts

  Not Collective, Asynchronous

  Input Parameters:
+ dctx         - A `PetscDeviceContext` to converge on
. n            - The number of sub contexts to converge
. joinMode     - The type of join to perform
- dsub         - The sub contexts to converge

  Level: beginner

  Notes:
  If `PetscDeviceContextFork()` creates `n` edges from a source node which all depend on the source
  node, then this routine is the exact mirror. That is, it creates a node (represented in `dctx`)
  which receives `n` edges (and optionally destroys them) which is dependent on the completion
  of all incoming edges.

  If `joinMode` is `PETSC_DEVICE_CONTEXT_JOIN_DESTROY`. All contexts in `dsub` will be
  destroyed by this routine. Thus all sub contexts must have been created with the `dctx`
  passed to this routine.

  If `joinMode` is `PETSC_DEVICE_CONTEXT_JOIN_SYNC`. All sub contexts will additionally wait on
  `dctx` after converging. This has the effect of "synchronizing" the outgoing edges. Note the
  sync suffix does NOT refer to the host, i.e. this routine does NOT call
  `PetscDeviceSynchronize()`.

  If `joinMode` is `PETSC_DEVICE_CONTEXT_JOIN_NO_SYNC`. `dctx` waits for all sub contexts but
  the sub contexts do not wait for one another or `dctx` afterwards.

  DAG representations:
  If `joinMode` is `PETSC_DEVICE_CONTEXT_JOIN_DESTROY`
.vb
  time ->

  -> dctx ---------/- |= CALL =| - dctx ->
  -> dsub[0] -----/
  ->  ... -------/
  -> dsub[n-1] -/
.ve
  If `joinMode` is `PETSC_DEVICE_CONTEXT_JOIN_SYNC`
.vb
  time ->

  -> dctx ---------/- |= CALL =| -\----> dctx ------>
  -> dsub[0] -----/                \---> dsub[0] --->
  ->  ... -------/                  \--> ... ------->
  -> dsub[n-1] -/                    \-> dsub[n-1] ->
.ve
  If `joinMode` is `PETSC_DEVICE_CONTEXT_JOIN_NO_SYNC`
.vb
  time ->

  -> dctx ----------/- |= CALL =| - dctx ->
  -> dsub[0] ------/----------------------->
  ->  ... --------/------------------------>
  -> dsub[n-1] --/------------------------->
.ve

.N ASYNC_API

.seealso: `PetscDeviceContextFork()`, `PetscDeviceContextForkWithStreamType()`,
`PetscDeviceContextSynchronize()`, `PetscDeviceContextJoinMode`
@*/
PetscErrorCode PetscDeviceContextJoin(PetscDeviceContext dctx, PetscInt n, PetscDeviceContextJoinMode joinMode, PetscDeviceContext **dsub)
{
  // debugging only
  std::string idList;

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  /* validity of dctx is checked in the wait-for loop */
  PetscValidPointer(dsub, 4);
  PetscAssert(n >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Number of contexts merged %" PetscInt_FMT " < 0", n);
  /* reserve 4 chars per id, 2 for number and 2 for ', ' separator */
  if (PetscDefined(USE_DEBUG_AND_INFO)) PetscCallCXX(idList.reserve(4 * n));
  /* first dctx waits on all the incoming edges */
  PetscCall(PetscLogEventBegin(DCONTEXT_Join, dctx, nullptr, nullptr, nullptr));
  for (PetscInt i = 0; i < n; ++i) {
    PetscCheckCompatibleDeviceContexts(dctx, 1, (*dsub)[i], 4);
    PetscCall(PetscDeviceContextWaitForContext(dctx, (*dsub)[i]));
    if (PetscDefined(USE_DEBUG_AND_INFO)) {
      PetscCallCXX(idList += std::to_string(PetscObjectCast((*dsub)[i])->id));
      if (i + 1 < n) PetscCallCXX(idList += ", ");
    }
  }

  /* now we handle the aftermath */
  switch (joinMode) {
  case PETSC_DEVICE_CONTEXT_JOIN_DESTROY: {
    const auto children = dctx->childIDs;
    const auto maxchild = dctx->maxNumChildren;
    auto      &nchild   = dctx->numChildren;
    PetscInt   j        = 0;

    PetscCheck(n <= nchild, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Trying to destroy %" PetscInt_FMT " children of a parent context that only has %" PetscInt_FMT " children, likely trying to restore to wrong parent", n, nchild);
    /* update child count while it's still fresh in memory */
    nchild -= n;
    for (PetscInt i = 0; i < maxchild; ++i) {
      if (children[i] && (children[i] == PetscObjectCast((*dsub)[j])->id)) {
        /* child is one of ours, can destroy it */
        PetscCall(PetscDeviceContextDestroy((*dsub) + j));
        /* reset the child slot */
        children[i] = 0;
        if (++j == n) break;
      }
    }
    /* gone through the loop but did not find every child */
    PetscCheck(j == n, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "%" PetscInt_FMT " contexts still remain after destroy, this may be because you are trying to restore to the wrong parent context, or the device contexts are not in the same order as they were checked out out in", n - j);
    PetscCall(PetscFree(*dsub));
  } break;
  case PETSC_DEVICE_CONTEXT_JOIN_SYNC:
    for (PetscInt i = 0; i < n; ++i) PetscCall(PetscDeviceContextWaitForContext((*dsub)[i], dctx));
  case PETSC_DEVICE_CONTEXT_JOIN_NO_SYNC:
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unknown PetscDeviceContextJoinMode given");
  }
  PetscCall(PetscLogEventEnd(DCONTEXT_Join, dctx, nullptr, nullptr, nullptr));

  PetscCall(PetscDebugInfo(dctx, "Joined %" PetscInt_FMT " ctxs to ctx %" PetscInt64_FMT ", mode %s with IDs: %s\n", n, PetscObjectCast(dctx)->id, PetscDeviceContextJoinModes[joinMode], idList.c_str()));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscDeviceContextSynchronize - Block the host until all work queued on a
  `PetscDeviceContext` has finished

  Not Collective

  Input Parameter:
. dctx - The `PetscDeviceContext` to synchronize

  Level: beginner

  Notes:
  The host will not return from this routine until `dctx` is idle. Any and all memory
  operations queued on or otherwise associated with (either explicitly or implicitly via
  dependencies) are guaranteed to have finished and be globally visible on return.

  In effect, this routine serves as memory and execution barrier.

  DAG representation:
.vb
  time ->

  -> dctx - |= CALL =| - dctx ->
.ve

.seealso: `PetscDeviceContextFork()`, `PetscDeviceContextJoin()`, `PetscDeviceContextQueryIdle()`
@*/
PetscErrorCode PetscDeviceContextSynchronize(PetscDeviceContext dctx)
{
  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  PetscCall(PetscLogEventBegin(DCONTEXT_Sync, dctx, nullptr, nullptr, nullptr));
  /* if it isn't setup there is nothing to sync on */
  if (dctx->setup) {
    PetscUseTypeMethod(dctx, synchronize);
    PetscCall(PetscDeviceContextSyncClearMap_Internal(dctx));
  }
  PetscCall(PetscLogEventEnd(DCONTEXT_Sync, dctx, nullptr, nullptr, nullptr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* every device type has a vector of null PetscDeviceContexts -- one for each device */
static auto nullContexts          = std::array<std::vector<PetscDeviceContext>, PETSC_DEVICE_MAX>{};
static auto nullContextsFinalizer = false;

static PetscErrorCode PetscDeviceContextGetNullContextForDevice_Private(PetscBool user_set_device, PetscDevice device, PetscDeviceContext *dctx)
{
  PetscInt        devid;
  PetscDeviceType dtype;

  PetscFunctionBegin;
  PetscValidDevice(device, 2);
  PetscValidPointer(dctx, 3);
  if (PetscUnlikely(!nullContextsFinalizer)) {
    const auto finalizer = [] {
      PetscFunctionBegin;
      for (auto &&dvec : nullContexts) {
        for (auto &&dctx : dvec) PetscCall(PetscDeviceContextDestroy(&dctx));
        PetscCallCXX(dvec.clear());
      }
      nullContextsFinalizer = false;
      PetscFunctionReturn(PETSC_SUCCESS);
    };

    nullContextsFinalizer = true;
    PetscCall(PetscRegisterFinalize(std::move(finalizer)));
  }
  PetscCall(PetscDeviceGetDeviceId(device, &devid));
  PetscCall(PetscDeviceGetType(device, &dtype));
  {
    auto &ctxlist = nullContexts[dtype];

    PetscCheck(devid >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Device ID (%" PetscInt_FMT ") must be positive", devid);
    // need to resize the container if not big enough because incrementing the iterator in
    // std::next() (if we haven't initialized that ctx yet) may cause it to fall outside the
    // current size of the container.
    if (static_cast<std::size_t>(devid) >= ctxlist.size()) PetscCallCXX(ctxlist.resize(devid + 1));
    if (PetscUnlikely(!ctxlist[devid])) {
      // we have not seen this device before
      PetscCall(PetscDeviceContextCreate(dctx));
      PetscCall(PetscInfo(*dctx, "Initializing null PetscDeviceContext (of type %s) for device %" PetscInt_FMT "\n", PetscDeviceTypes[dtype], devid));
      {
        const auto pobj   = PetscObjectCast(*dctx);
        const auto name   = "null context " + std::to_string(devid);
        const auto prefix = "null_context_" + std::to_string(devid) + '_';

        PetscCall(PetscObjectSetName(pobj, name.c_str()));
        PetscCall(PetscObjectSetOptionsPrefix(pobj, prefix.c_str()));
      }
      PetscCall(PetscDeviceContextSetStreamType(*dctx, PETSC_STREAM_GLOBAL_BLOCKING));
      PetscCall(PetscDeviceContextSetDevice_Private(*dctx, device, user_set_device));
      PetscCall(PetscDeviceContextSetUp(*dctx));
      // would use ctxlist.cbegin() but GCC 4.8 can't handle const iterator insert!
      PetscCallCXX(ctxlist.insert(std::next(ctxlist.begin(), devid), *dctx));
    } else *dctx = ctxlist[devid];
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Gets the "NULL" context for the current PetscDeviceType and PetscDevice. NULL contexts are
  guaranteed to always be globally blocking.
*/
PetscErrorCode PetscDeviceContextGetNullContext_Internal(PetscDeviceContext *dctx)
{
  PetscDeviceContext gctx;
  PetscDevice        gdev = nullptr;

  PetscFunctionBegin;
  PetscValidPointer(dctx, 1);
  PetscCall(PetscDeviceContextGetCurrentContext(&gctx));
  PetscCall(PetscDeviceContextGetDevice(gctx, &gdev));
  PetscCall(PetscDeviceContextGetNullContextForDevice_Private(gctx->usersetdevice, gdev, dctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscDeviceContextSetFromOptions - Configure a `PetscDeviceContext` from the options database

  Collective on `comm` or `dctx`

  Input Parameters:
+ comm - MPI communicator on which to query the options database (optional)
- dctx - The `PetscDeviceContext` to configure

  Output Parameter:
. dctx - The `PetscDeviceContext`

  Options Database Keys:
+ -device_context_stream_type - type of stream to create inside the `PetscDeviceContext` -
   `PetscDeviceContextSetStreamType()`
- -device_context_device_type - the type of `PetscDevice` to attach by default - `PetscDeviceType`

  Level: beginner

  Note:
  The user may pass `MPI_COMM_NULL` for `comm` in which case the communicator of `dctx` is
  used (which is always `PETSC_COMM_SELF`).

.seealso: `PetscDeviceContextSetStreamType()`, `PetscDeviceContextSetDevice()`,
`PetscDeviceContextView()`
@*/
PetscErrorCode PetscDeviceContextSetFromOptions(MPI_Comm comm, PetscDeviceContext dctx)
{
  const auto pobj     = PetscObjectCast(dctx);
  auto       dtype    = std::make_pair(PETSC_DEVICE_DEFAULT(), PETSC_FALSE);
  auto       stype    = std::make_pair(PETSC_DEVICE_CONTEXT_DEFAULT_STREAM_TYPE, PETSC_FALSE);
  MPI_Comm   old_comm = PETSC_COMM_SELF;

  PetscFunctionBegin;
  // do not user getoptionalnullcontext here, the user is not allowed to set it from options!
  PetscValidDeviceContext(dctx, 2);
  /* set the device type first */
  if (const auto device = dctx->device) PetscCall(PetscDeviceGetType(device, &dtype.first));
  PetscCall(PetscDeviceContextGetStreamType(dctx, &stype.first));

  if (comm == MPI_COMM_NULL) {
    PetscCall(PetscObjectGetComm(pobj, &comm));
  } else {
    // briefly set the communicator for dctx (it is always PETSC_COMM_SELF) so
    // PetscObjectOptionsBegin() behaves as if dctx had comm
    old_comm = Petsc::util::exchange(pobj->comm, comm);
  }

  PetscObjectOptionsBegin(pobj);
  PetscCall(PetscDeviceContextQueryOptions_Internal(PetscOptionsObject, dtype, stype));
  PetscOptionsEnd();
  // reset the comm (should be PETSC_COMM_SELF)
  if (comm != MPI_COMM_NULL) pobj->comm = old_comm;
  if (dtype.second) PetscCall(PetscDeviceContextSetDefaultDeviceForType_Internal(dctx, dtype.first));
  if (stype.second) PetscCall(PetscDeviceContextSetStreamType(dctx, stype.first));
  PetscCall(PetscDeviceContextSetUp(dctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscDeviceContextView - View a `PetscDeviceContext`

  Collective on `viewer`

  Input Parameters:
+ dctx - The `PetscDeviceContext`
- viewer - The `PetscViewer` to view `dctx` with (may be `NULL`)

  Level: beginner

  Note:
  If `viewer` is `NULL`, `PETSC_VIEWER_STDOUT_WORLD` is used instead, in which case this
  routine is collective on `PETSC_COMM_WORLD`.

.seealso: `PetscDeviceContextViewFromOptions()`, `PetscDeviceView()`, `PETSC_VIEWER_STDOUT_WORLD`, `PetscDeviceContextCreate()`
@*/
PetscErrorCode PetscDeviceContextView(PetscDeviceContext dctx, PetscViewer viewer)
{
  PetscBool iascii;

  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PETSC_COMM_WORLD, &viewer));
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCall(PetscObjectTypeCompare(PetscObjectCast(viewer), PETSCVIEWERASCII, &iascii));
  if (iascii) {
    auto        stype = PETSC_STREAM_DEFAULT_BLOCKING;
    PetscViewer sub;

    PetscCall(PetscViewerGetSubViewer(viewer, PETSC_COMM_SELF, &sub));
    PetscCall(PetscObjectPrintClassNamePrefixType(PetscObjectCast(dctx), sub));
    PetscCall(PetscViewerASCIIPushTab(sub));
    PetscCall(PetscDeviceContextGetStreamType(dctx, &stype));
    PetscCall(PetscViewerASCIIPrintf(sub, "stream type: %s\n", PetscStreamTypes[stype]));
    PetscCall(PetscViewerASCIIPrintf(sub, "children: %" PetscInt_FMT "\n", dctx->numChildren));
    if (const auto nchild = dctx->numChildren) {
      PetscCall(PetscViewerASCIIPushTab(sub));
      for (PetscInt i = 0; i < nchild; ++i) {
        if (i == nchild - 1) {
          PetscCall(PetscViewerASCIIPrintf(sub, "%" PetscInt64_FMT, dctx->childIDs[i]));
        } else {
          PetscCall(PetscViewerASCIIPrintf(sub, "%" PetscInt64_FMT ", ", dctx->childIDs[i]));
        }
      }
    }
    PetscCall(PetscViewerASCIIPopTab(sub));
    PetscCall(PetscViewerRestoreSubViewer(viewer, PETSC_COMM_SELF, &sub));
    PetscCall(PetscViewerFlush(viewer));
    PetscCall(PetscViewerASCIIPushTab(viewer));
  }
  if (const auto device = dctx->device) PetscCall(PetscDeviceView(device, viewer));
  if (iascii) PetscCall(PetscViewerASCIIPopTab(viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscDeviceContextViewFromOptions - View a `PetscDeviceContext` from options

  Input Parameters:
+ dctx - The `PetscDeviceContext` to view
. obj  - Optional `PetscObject` to associate (may be `NULL`)
- name - The command line option

  Level: beginner

.seealso: `PetscDeviceContextView()`, `PetscObjectViewFromOptions()`, `PetscDeviceContextCreate()`
@*/
PetscErrorCode PetscDeviceContextViewFromOptions(PetscDeviceContext dctx, PetscObject obj, const char name[])
{
  PetscFunctionBegin;
  PetscCall(PetscDeviceContextGetOptionalNullContext_Internal(&dctx));
  if (obj) PetscValidHeader(obj, 2);
  PetscValidCharPointer(name, 3);
  PetscCall(PetscObjectViewFromOptions(PetscObjectCast(dctx), obj, name));
  PetscFunctionReturn(PETSC_SUCCESS);
}
