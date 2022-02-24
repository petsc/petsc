#include <petsc/private/deviceimpl.h> /*I "petscdevice.h" I*/
#include "objpool.hpp"

const char *const PetscStreamTypes[] = {
  "global_blocking",
  "default_blocking",
  "global_nonblocking",
  "max",
  "PetscStreamType",
  "PETSC_STREAM_",
  nullptr
};

const char *const PetscDeviceContextJoinModes[] = {
  "destroy",
  "sync",
  "no_sync",
  "PetscDeviceContextJoinMode",
  "PETSC_DEVICE_CONTEXT_JOIN_",
  nullptr
};

/* Define the allocator */
struct PetscDeviceContextAllocator : Petsc::AllocatorBase<PetscDeviceContext>
{
  static PetscInt PetscDeviceContextID;

  PETSC_NODISCARD static PetscErrorCode create(PetscDeviceContext *dctx) noexcept
  {
    PetscDeviceContext dc;

    PetscFunctionBegin;
    CHKERRQ(PetscNew(&dc));
    dc->id         = PetscDeviceContextID++;
    dc->streamType = PETSC_STREAM_DEFAULT_BLOCKING;
    *dctx          = dc;
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD static PetscErrorCode destroy(PetscDeviceContext dctx) noexcept
  {
    PetscFunctionBegin;
    PetscAssert(!dctx->numChildren,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Device context still has %" PetscInt_FMT " un-joined children, must call PetscDeviceContextJoin() with all children before destroying",dctx->numChildren);
    if (dctx->ops->destroy) CHKERRQ((*dctx->ops->destroy)(dctx));
    CHKERRQ(PetscDeviceDestroy(&dctx->device));
    CHKERRQ(PetscFree(dctx->childIDs));
    CHKERRQ(PetscFree(dctx));
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD static PetscErrorCode reset(PetscDeviceContext dctx) noexcept
  {
    PetscFunctionBegin;
    /* don't deallocate the child array, rather just zero it out */
    CHKERRQ(PetscArrayzero(dctx->childIDs,dctx->maxNumChildren));
    dctx->setup       = PETSC_FALSE;
    dctx->numChildren = 0;
    dctx->streamType  = PETSC_STREAM_DEFAULT_BLOCKING;
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD static constexpr PetscErrorCode finalize() noexcept { return 0; }
};
/* an ID = 0 is invalid */
PetscInt PetscDeviceContextAllocator::PetscDeviceContextID = 1;

static Petsc::ObjectPool<PetscDeviceContext,PetscDeviceContextAllocator> contextPool;

/*@C
  PetscDeviceContextCreate - Creates a PetscDeviceContext

  Not Collective, Asynchronous

  Output Paramemter:
. dctx - The PetscDeviceContext

  Notes:
  Unlike almost every other PETSc class it is advised that most users use
  PetscDeviceContextDuplicate() rather than this routine to create new contexts. Contexts
  of different types are incompatible with one another; using
  PetscDeviceContextDuplicate() ensures compatible types.

  Level: beginner

.seealso: PetscDeviceContextDuplicate(), PetscDeviceContextSetDevice(),
PetscDeviceContextSetStreamType(), PetscDeviceContextSetUp(),
PetscDeviceContextSetFromOptions(), PetscDeviceContextDestroy()
@*/
PetscErrorCode PetscDeviceContextCreate(PetscDeviceContext *dctx)
{
  PetscFunctionBegin;
  PetscValidPointer(dctx,1);
  CHKERRQ(PetscDeviceInitializePackage());
  CHKERRQ(contextPool.get(*dctx));
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextDestroy - Frees a PetscDeviceContext

  Not Collective, Asynchronous

  Input Parameters:
. dctx - The PetscDeviceContext

  Notes:
  No implicit synchronization occurs due to this routine, all resources are released completely asynchronously
  w.r.t. the host. If one needs to guarantee access to the data produced on this contexts stream one should perform the
  appropriate synchronization before calling this routine.

  Developer Notes:
  The context is never actually "destroyed", only returned to an ever growing pool of
  contexts. There are currently no safeguards on the size of the pool, this should perhaps
  be implemented.

  Level: beginner

.seealso: PetscDeviceContextCreate(), PetscDeviceContextSetDevice(), PetscDeviceContextSetUp(), PetscDeviceContextSynchronize()
@*/
PetscErrorCode PetscDeviceContextDestroy(PetscDeviceContext *dctx)
{
  PetscFunctionBegin;
  if (!*dctx) PetscFunctionReturn(0);
  CHKERRQ(contextPool.reclaim(std::move(*dctx)));
  *dctx = nullptr;
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextSetStreamType - Set the implementation type of the underlying stream for a PetscDeviceContext

  Not Collective, Asynchronous

  Input Parameters:
+ dctx - The PetscDeviceContext
- type - The PetscStreamType

  Notes:
  See PetscStreamType in include/petscdevicetypes.h for more information on the available
  types and their interactions. If the PetscDeviceContext was previously set up and stream
  type was changed, you must call PetscDeviceContextSetUp() again after this routine.

  Level: intermediate

.seealso: PetscStreamType, PetscDeviceContextGetStreamType(), PetscDeviceContextCreate(), PetscDeviceContextSetUp(), PetscDeviceContextSetFromOptions()
@*/
PetscErrorCode PetscDeviceContextSetStreamType(PetscDeviceContext dctx, PetscStreamType type)
{
  PetscFunctionBegin;
  PetscValidDeviceContext(dctx,1);
  PetscValidStreamType(type,2);
  /* only need to do complex swapping if the object has already been setup */
  if (dctx->setup && (dctx->streamType != type)) {
    CHKERRQ((*dctx->ops->changestreamtype)(dctx,type));
    dctx->setup = PETSC_FALSE;
  }
  dctx->streamType = type;
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextGetStreamType - Get the implementation type of the underlying stream for a PetscDeviceContext

  Not Collective, Asynchronous

  Input Parameter:
. dctx - The PetscDeviceContext

  Output Parameter:
. type - The PetscStreamType

  Notes:
  See PetscStreamType in include/petscdevicetypes.h for more information on the available types and their interactions

  Level: intermediate

.seealso: PetscDeviceContextSetStreamType(), PetscDeviceContextCreate(), PetscDeviceContextSetFromOptions()
@*/
PetscErrorCode PetscDeviceContextGetStreamType(PetscDeviceContext dctx, PetscStreamType *type)
{
  PetscFunctionBegin;
  PetscValidDeviceContext(dctx,1);
  PetscValidIntPointer(type,2);
  *type = dctx->streamType;
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextSetDevice - Set the underlying device for the PetscDeviceContext

  Not Collective, Possibly Synchronous

  Input Parameters:
+ dctx   - The PetscDeviceContext
- device - The PetscDevice

  Notes:
  This routine is effectively PetscDeviceContext's "set-type" (so every PetscDeviceContext
  must also have an attached PetscDevice). Unlike the usual set-type semantics, it is
  not stricly necessary to set a contexts device to enable usage, any created device
  contexts will always come equipped with the "default" device.

  This routine is a no-op if dctx is already attached to device.

  This routine may initialize the backend device and incur synchronization.

  Level: intermediate

.seealso: PetscDeviceCreate(), PetscDeviceConfigure(), PetscDeviceContextGetDevice()
@*/
PetscErrorCode PetscDeviceContextSetDevice(PetscDeviceContext dctx, PetscDevice device)
{
  PetscFunctionBegin;
  PetscValidDeviceContext(dctx,1);
  PetscValidDevice(device,2);
  if (dctx->device) {
    /* can't do a strict pointer equality check since PetscDevice's are reused */
    if (dctx->device->ops->createcontext == device->ops->createcontext) PetscFunctionReturn(0);
  }
  CHKERRQ(PetscDeviceDestroy(&dctx->device));
  if (dctx->ops->destroy) CHKERRQ((*dctx->ops->destroy)(dctx));
  CHKERRQ(PetscMemzero(dctx->ops,sizeof(*dctx->ops)));
  CHKERRQ((*device->ops->createcontext)(dctx));
  CHKERRQ(PetscDeviceReference_Internal(device));
  dctx->device = device;
  dctx->setup  = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextGetDevice - Get the underlying PetscDevice for a PetscDeviceContext

  Not Collective, Asynchronous

  Input Parameter:
. dctx - the PetscDeviceContext

  Output Parameter:
. device - The PetscDevice

  Notes:
  This is a borrowed reference, the user should not destroy the device.

  Level: intermediate

.seealso: PetscDeviceContextSetDevice(), PetscDevice
@*/
PetscErrorCode PetscDeviceContextGetDevice(PetscDeviceContext dctx, PetscDevice *device)
{
  PetscFunctionBegin;
  PetscValidDeviceContext(dctx,1);
  PetscValidPointer(device,2);
  PetscAssert(dctx->device,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"PetscDeviceContext %" PetscInt_FMT " has no attached PetscDevice to get",dctx->id);
  *device = dctx->device;
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextSetUp - Prepares a PetscDeviceContext for use

  Not Collective, Asynchronous

  Input Parameter:
. dctx - The PetscDeviceContext

  Developer Notes:
  This routine is usually the stage where a PetscDeviceContext acquires device-side data structures such as streams,
  events, and (possibly) handles.

  Level: beginner

.seealso: PetscDeviceContextCreate(), PetscDeviceContextSetDevice(), PetscDeviceContextDestroy(), PetscDeviceContextSetFromOptions()
@*/
PetscErrorCode PetscDeviceContextSetUp(PetscDeviceContext dctx)
{
  PetscFunctionBegin;
  PetscValidDeviceContext(dctx,1);
  if (!dctx->device) {
    CHKERRQ(PetscInfo(nullptr,"PetscDeviceContext %" PetscInt_FMT " did not have an explicitly attached PetscDevice, using default with type %s\n",dctx->id,PetscDeviceTypes[PETSC_DEVICE_DEFAULT]));
    CHKERRQ(PetscDeviceContextSetDefaultDevice_Internal(dctx));
  }
  if (dctx->setup) PetscFunctionReturn(0);
  CHKERRQ((*dctx->ops->setup)(dctx));
  dctx->setup = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextDuplicate - Duplicates a PetscDeviceContext object

  Not Collective, Asynchronous

  Input Parameter:
. dctx - The PetscDeviceContext to duplicate

  Output Paramter:
. dctxdup - The duplicated PetscDeviceContext

  Notes:
  This is a shorthand method for creating a PetscDeviceContext with the exact same
  settings as another. Note however that the duplicated PetscDeviceContext does not "share"
  any of the underlying data with the original, (including its current stream-state) they
  are completely separate objects.

  Level: beginner

.seealso: PetscDeviceContextCreate(), PetscDeviceContextSetDevice(), PetscDeviceContextSetStreamType()
@*/
PetscErrorCode PetscDeviceContextDuplicate(PetscDeviceContext dctx, PetscDeviceContext *dctxdup)
{
  PetscDeviceContext dup;

  PetscFunctionBegin;
  PetscValidDeviceContext(dctx,1);
  PetscValidPointer(dctxdup,2);
  CHKERRQ(PetscDeviceContextCreate(&dup));
  CHKERRQ(PetscDeviceContextSetStreamType(dup,dctx->streamType));
  if (dctx->device) CHKERRQ(PetscDeviceContextSetDevice(dup,dctx->device));
  CHKERRQ(PetscDeviceContextSetUp(dup));
  *dctxdup = dup;
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextQueryIdle - Returns whether or not a PetscDeviceContext is idle

  Not Collective, Asynchronous

  Input Parameter:
. dctx - The PetscDeviceContext object

  Output Parameter:
. idle - PETSC_TRUE if PetscDeviceContext has NO work, PETSC_FALSE if it has work

  Notes:
  This routine only refers a singular context and does NOT take any of its children into
  account. That is, if dctx is idle but has dependents who do have work, this routine still
  returns PETSC_TRUE.

  Level: intermediate

.seealso: PetscDeviceContextCreate(), PetscDeviceContextWaitForContext(), PetscDeviceContextFork()
@*/
PetscErrorCode PetscDeviceContextQueryIdle(PetscDeviceContext dctx, PetscBool *idle)
{
  PetscFunctionBegin;
  PetscValidDeviceContext(dctx,1);
  PetscValidBoolPointer(idle,2);
  CHKERRQ((*dctx->ops->query)(dctx,idle));
  CHKERRQ(PetscInfo(nullptr,"PetscDeviceContext id %" PetscInt_FMT " %s idle\n",dctx->id,*idle ? "was" : "was not"));
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextWaitForContext - Make one context wait for another context to finish

  Not Collective, Asynchronous

  Input Parameters:
+ dctxa - The PetscDeviceContext object that is waiting
- dctxb - The PetscDeviceContext object that is being waited on

  Notes:
  Serializes two PetscDeviceContexts. This routine uses only the state of dctxb at the moment this routine was
  called, so any future work queued will not affect dctxa. It is safe to pass the same context to both arguments.

  Level: beginner

.seealso: PetscDeviceContextCreate(), PetscDeviceContextQueryIdle(), PetscDeviceContextJoin()
@*/
PetscErrorCode PetscDeviceContextWaitForContext(PetscDeviceContext dctxa, PetscDeviceContext dctxb)
{
  PetscFunctionBegin;
  PetscCheckCompatibleDeviceContexts(dctxa,1,dctxb,2);
  if (dctxa == dctxb) PetscFunctionReturn(0);
  CHKERRQ((*dctxa->ops->waitforcontext)(dctxa,dctxb));
  PetscFunctionReturn(0);
}

#define PETSC_USE_DEBUG_AND_INFO (PetscDefined(USE_DEBUG) && PetscDefined(USE_INFO))
#if PETSC_USE_DEBUG_AND_INFO
#include <string>
#endif
/*@C
  PetscDeviceContextFork - Create a set of dependent child contexts from a parent context

  Not Collective, Asynchronous

  Input Parameters:
+ dctx - The parent PetscDeviceContext
- n    - The number of children to create

  Output Parameter:
. dsub - The created child context(s)

  Notes:
  This routine creates n edges of a DAG from a source node which are causally dependent on the source node, meaning
  that work queued on child contexts will not start until the parent context finishes its work. This accounts for work
  queued on the parent up until calling this function, any subsequent work enqueued on the parent has no effect on the children.

  Any children created with this routine have their lifetimes bounded by the parent. That is, the parent context expects
  to free all of it's children (and ONLY its children) before itself is freed.

  DAG representation:
.vb
  time ->

  -> dctx \----> dctx ------>
           \---> dsub[0] --->
            \--> ... ------->
             \-> dsub[n-1] ->
.ve

  Level: intermediate

.seealso: PetscDeviceContextJoin(), PetscDeviceContextSynchronize(), PetscDeviceContextQueryIdle()
@*/
PetscErrorCode PetscDeviceContextFork(PetscDeviceContext dctx, PetscInt n, PetscDeviceContext **dsub)
{
#if PETSC_USE_DEBUG_AND_INFO
  const PetscInt      nBefore = n;
  static std::string  idList;
#endif
  PetscDeviceContext *dsubTmp = nullptr;
  PetscInt            i = 0;

  PetscFunctionBegin;
  PetscValidDeviceContext(dctx,1);
  PetscValidPointer(dsub,3);
  PetscAssert(n >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Number of contexts requested %" PetscInt_FMT " < 0",n);
#if PETSC_USE_DEBUG_AND_INFO
  /* reserve 4 chars per id, 2 for number and 2 for ', ' separator */
  idList.reserve(4*n);
#endif
  /* update child totals */
  dctx->numChildren += n;
  /* now to find out if we have room */
  if (dctx->numChildren > dctx->maxNumChildren) {
    /* no room, either from having too many kids or not having any */
    if (dctx->childIDs) {
      /* have existing children, must reallocate them */
      CHKERRQ(PetscRealloc(dctx->numChildren*sizeof(*dctx->childIDs),&dctx->childIDs));
      /* clear the extra memory since realloc doesn't do it for us */
      CHKERRQ(PetscArrayzero((dctx->childIDs)+(dctx->maxNumChildren),(dctx->numChildren)-(dctx->maxNumChildren)));
    } else {
      /* have no children */
      CHKERRQ(PetscCalloc1(dctx->numChildren,&dctx->childIDs));
    }
    /* update total number of children */
    dctx->maxNumChildren = dctx->numChildren;
  }
  CHKERRQ(PetscMalloc1(n,&dsubTmp));
  while (n) {
    /* empty child slot */
    if (!(dctx->childIDs[i])) {
      /* create the child context in the image of its parent */
      CHKERRQ(PetscDeviceContextDuplicate(dctx,dsubTmp+i));
      CHKERRQ(PetscDeviceContextWaitForContext(dsubTmp[i],dctx));
      /* register the child with its parent */
      dctx->childIDs[i] = dsubTmp[i]->id;
#if PETSC_USE_DEBUG_AND_INFO
      idList += std::to_string(dsubTmp[i]->id);
      if (n != 1) idList += ", ";
#endif
      --n;
    }
    ++i;
  }
#if PETSC_USE_DEBUG_AND_INFO
  CHKERRQ(PetscInfo(nullptr,"Forked %" PetscInt_FMT " children from parent %" PetscInt_FMT " with IDs: %s\n",nBefore,dctx->id,idList.c_str()));
  /* resets the size but doesn't deallocate the memory */
  idList.clear();
#endif
  /* pass the children back to caller */
  *dsub = dsubTmp;
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextJoin - Converge a set of child contexts

  Not Collective, Asynchronous

  Input Parameters:
+ dctx         - A PetscDeviceContext to converge on
. n            - The number of sub contexts to converge
. joinMode     - The type of join to perform
- dsub         - The sub contexts to converge

  Notes:
  If PetscDeviceContextFork() creates n edges from a source node which all depend on the
  source node, then this routine is the exact mirror. That is, it creates a node
  (represented in dctx) which recieves n edges (and optionally destroys them) which is
  dependent on the completion of all incoming edges.

  If joinMode is PETSC_DEVICE_CONTEXT_JOIN_DESTROY all contexts in dsub will be destroyed
  by this routine. Thus all sub contexts must have been created with the dctx passed to
  this routine.

  if joinMode is PETSC_DEVICE_CONTEXT_JOIN_NO_SYNC dctx waits for all sub contexts but the
  sub contexts do not wait for one another afterwards.

  If joinMode is PETSC_DEVICE_CONTEXT_JOIN_SYNC all sub contexts will additionally
  wait on dctx after converging. This has the effect of "synchronizing" the outgoing
  edges.

  DAG representations:
  If joinMode is PETSC_DEVICE_CONTEXT_JOIN_DESTROY
.vb
  time ->

  -> dctx ---------/- dctx ->
  -> dsub[0] -----/
  ->  ... -------/
  -> dsub[n-1] -/
.ve
  If joinMode is PETSC_DEVICE_CONTEXT_JOIN_NO_SYNC
.vb
  time ->

  -> dctx ---------/- dctx ->
  -> dsub[0] -----/--------->
  ->  ... -------/---------->
  -> dsub[n-1] -/----------->
.ve
  If joinMode is PETSC_DEVICE_CONTEXT_JOIN_SYNC
.vb
  time ->

  -> dctx ---------/- dctx -\----> dctx ------>
  -> dsub[0] -----/          \---> dsub[0] --->
  ->  ... -------/            \--> ... ------->
  -> dsub[n-1] -/              \-> dsub[n-1] ->
.ve

  Level: intermediate

.seealso: PetscDeviceContextFork(), PetscDeviceContextSynchronize(), PetscDeviceContextJoinMode
@*/
PetscErrorCode PetscDeviceContextJoin(PetscDeviceContext dctx, PetscInt n, PetscDeviceContextJoinMode joinMode, PetscDeviceContext **dsub)
{
#if defined(PETSC_USE_DEBUG) && defined(PETSC_USE_INFO)
  static std::string idList;
#endif

  PetscFunctionBegin;
  /* validity of dctx is checked in the wait-for loop */
  PetscValidPointer(dsub,4);
  PetscAssert(n >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Number of contexts merged %" PetscInt_FMT " < 0",n);
#if defined(PETSC_USE_DEBUG) && defined(PETSC_USE_INFO)
  /* reserve 4 chars per id, 2 for number and 2 for ', ' separator */
  idList.reserve(4*n);
#endif
  /* first dctx waits on all the incoming edges */
  for (PetscInt i = 0; i < n; ++i) {
    PetscCheckCompatibleDeviceContexts(dctx,1,(*dsub)[i],4);
    CHKERRQ(PetscDeviceContextWaitForContext(dctx,(*dsub)[i]));
#if defined(PETSC_USE_DEBUG) && defined(PETSC_USE_INFO)
    idList += std::to_string((*dsub)[i]->id);
    if (i+1 < n) idList += ", ";
#endif
  }

  /* now we handle the aftermath */
  switch (joinMode) {
  case PETSC_DEVICE_CONTEXT_JOIN_DESTROY:
    {
      PetscInt j = 0;

      PetscAssert(n <= dctx->numChildren,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Trying to destroy %" PetscInt_FMT " children of a parent context that only has %" PetscInt_FMT " children, likely trying to restore to wrong parent",n,dctx->numChildren);
      /* update child count while it's still fresh in memory */
      dctx->numChildren -= n;
      for (PetscInt i = 0; i < dctx->maxNumChildren; ++i) {
        if (dctx->childIDs[i] && (dctx->childIDs[i] == (*dsub)[j]->id)) {
          /* child is one of ours, can destroy it */
          CHKERRQ(PetscDeviceContextDestroy((*dsub)+j));
          /* reset the child slot */
          dctx->childIDs[i] = 0;
          if (++j == n) break;
        }
      }
      /* gone through the loop but did not find every child, if this triggers (or well, doesn't) on perf-builds we leak the remaining contexts memory */
      PetscAssert(j == n,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"%" PetscInt_FMT " contexts still remain after destroy, this may be because you are trying to restore to the wrong parent context, or the device contexts are not in the same order as they were checked out out in.",n-j);
      CHKERRQ(PetscFree(*dsub));
    }
    break;
  case PETSC_DEVICE_CONTEXT_JOIN_SYNC:
    for (PetscInt i = 0; i < n; ++i) CHKERRQ(PetscDeviceContextWaitForContext((*dsub)[i],dctx));
  case PETSC_DEVICE_CONTEXT_JOIN_NO_SYNC:
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Unknown PetscDeviceContextJoinMode given");
  }

#if defined(PETSC_USE_DEBUG) && defined(PETSC_USE_INFO)
  CHKERRQ(PetscInfo(nullptr,"Joined %" PetscInt_FMT " ctxs to ctx %" PetscInt_FMT ", mode %s with IDs: %s\n",n,dctx->id,PetscDeviceContextJoinModes[joinMode],idList.c_str()));
  idList.clear();
#endif
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextSynchronize - Block the host until all work queued on or associated with a PetscDeviceContext has finished

  Not Collective, Synchronous

  Input Parameters:
. dctx - The PetscDeviceContext to synchronize

  Level: beginner

.seealso: PetscDeviceContextFork(), PetscDeviceContextJoin(), PetscDeviceContextQueryIdle()
@*/
PetscErrorCode PetscDeviceContextSynchronize(PetscDeviceContext dctx)
{
  PetscFunctionBegin;
  PetscValidDeviceContext(dctx,1);
  /* if it isn't setup there is nothing to sync on */
  if (dctx->setup) CHKERRQ((*dctx->ops->synchronize)(dctx));
  PetscFunctionReturn(0);
}

#define PETSC_DEVICE_CONTEXT_DEFAULT_DEVICE PETSC_DEVICE_DEFAULT
// REMOVE ME (change)
#define PETSC_DEVICE_CONTEXT_DEFAULT_STREAM PETSC_STREAM_GLOBAL_BLOCKING

static PetscDeviceType    rootDeviceType = PETSC_DEVICE_CONTEXT_DEFAULT_DEVICE;
static PetscStreamType    rootStreamType = PETSC_DEVICE_CONTEXT_DEFAULT_STREAM;
static PetscDeviceContext globalContext  = nullptr;

/* when PetsDevice initializes PetscDeviceContext eagerly the type of device created should
 * match whatever device is eagerly intialized */
PetscErrorCode PetscDeviceContextSetRootDeviceType_Internal(PetscDeviceType type)
{
  PetscFunctionBegin;
  PetscValidDeviceType(type,1);
  rootDeviceType = type;
  PetscFunctionReturn(0);
}

#if 0
/* currently unused */
PetscErrorCode PetscDeviceContextSetRootStreamType_Internal(PetscStreamType type)
{
  PetscFunctionBegin;
  PetscValidStreamType(type,1);
  rootStreamType = type;
  PetscFunctionReturn(0);
}
#endif

static PetscErrorCode PetscDeviceContextSetupGlobalContext_Private(void)
{
  static const auto PetscDeviceContextFinalizer = []() -> PetscErrorCode {

    PetscFunctionBegin;
    CHKERRQ(PetscDeviceContextDestroy(&globalContext));
    rootDeviceType = PETSC_DEVICE_CONTEXT_DEFAULT_DEVICE;
    rootStreamType = PETSC_DEVICE_CONTEXT_DEFAULT_STREAM;
    PetscFunctionReturn(0);
  };

  PetscFunctionBegin;
  if (globalContext) PetscFunctionReturn(0);
  /* this exists purely as a valid device check. */
  CHKERRQ(PetscDeviceInitializePackage());
  CHKERRQ(PetscRegisterFinalize(PetscDeviceContextFinalizer));
  CHKERRQ(PetscInfo(nullptr,"Initializing global PetscDeviceContext\n"));
  /* we call the allocator directly here since the ObjectPool creates a PetscContainer which
   * eventually tries to call logging functions. However, this routine may be purposefully
   * called __before__ logging is initialized, so the logging function would PETSCABORT */
  CHKERRQ(contextPool.allocator().create(&globalContext));
  CHKERRQ(PetscDeviceContextSetStreamType(globalContext,rootStreamType));
  CHKERRQ(PetscDeviceContextSetDefaultDeviceForType_Internal(globalContext,rootDeviceType));
  CHKERRQ(PetscDeviceContextSetUp(globalContext));
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextGetCurrentContext - Get the current active PetscDeviceContext

  Not Collective, Asynchronous

  Output Parameter:
. dctx - The PetscDeviceContext

  Notes:
  The user generally should not destroy contexts retrieved with this routine unless they
  themselves have created them. There exists no protection against destroying the root
  context.

  Developer Notes:
  Unless the user has set their own, this routine creates the "root" context the first time it
  is called, registering its destructor to PetscFinalize().

  Level: beginner

.seealso: PetscDeviceContextSetCurrentContext(), PetscDeviceContextFork(),
PetscDeviceContextJoin(), PetscDeviceContextCreate()
@*/
PetscErrorCode PetscDeviceContextGetCurrentContext(PetscDeviceContext *dctx)
{
  PetscFunctionBegin;
  PetscValidPointer(dctx,1);
  CHKERRQ(PetscDeviceContextSetupGlobalContext_Private());
  /* while the static analyzer can find global variables, it will throw a warning about not
   * being able to connect this back to the function arguments */
  PetscDisableStaticAnalyzerForExpressionUnderstandingThatThisIsDangerousAndBugprone(PetscValidDeviceContext(globalContext,-1));
  *dctx = globalContext;
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextSetCurrentContext - Set the current active PetscDeviceContext

  Not Collective, Asynchronous

  Input Parameter:
. dctx - The PetscDeviceContext

  Notes:
  This routine can be used to set the defacto "root" PetscDeviceContext to a user-defined
  implementation by calling this routine immediately after PetscInitialize() and ensuring that
  PetscDevice is not greedily intialized. In this case the user is responsible for destroying
  their PetscDeviceContext before PetscFinalize() returns.

  The old context is not stored in any way by this routine; if one is overriding a context that
  they themselves do not control, one should take care to temporarily store it by calling
  PetscDeviceContextGetCurrentContext() before calling this routine.

  Level: beginner

.seealso: PetscDeviceContextGetCurrentContext(), PetscDeviceContextFork(),
PetscDeviceContextJoin(), PetscDeviceContextCreate()
@*/
PetscErrorCode PetscDeviceContextSetCurrentContext(PetscDeviceContext dctx)
{
  PetscFunctionBegin;
  PetscValidDeviceContext(dctx,1);
  PetscAssert(dctx->setup,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"PetscDeviceContext %" PetscInt_FMT " must be set up before being set as global context",dctx->id);
  globalContext = dctx;
  CHKERRQ(PetscInfo(nullptr,"Set global PetscDeviceContext id %" PetscInt_FMT "\n",dctx->id));
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextSetFromOptions - Configure a PetscDeviceContext from the options database

  Collective on comm, Asynchronous

  Input Parameters:
+ comm   - MPI communicator on which to query the options database
. prefix - prefix to prepend to all options database queries, NULL if not needed
- dctx   - The PetscDeviceContext to configure

  Output Parameter:
. dctx - The PetscDeviceContext

  Options Database:
+ -device_context_stream_type - type of stream to create inside the PetscDeviceContext -
   PetscDeviceContextSetStreamType()
- -device_context_device_type - the type of PetscDevice to attach by default - PetscDeviceType

  Level: beginner

.seealso: PetscDeviceContextSetStreamType(), PetscDeviceContextSetDevice()
@*/
PetscErrorCode PetscDeviceContextSetFromOptions(MPI_Comm comm, const char prefix[], PetscDeviceContext dctx)
{
  PetscBool      flag;
  PetscInt       stype,dtype;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (prefix) PetscValidCharPointer(prefix,2);
  PetscValidDeviceContext(dctx,3);
  ierr = PetscOptionsBegin(comm,prefix,"PetscDeviceContext Options","Sys");CHKERRQ(ierr);
  CHKERRQ(PetscOptionsEList("-device_context_stream_type","PetscDeviceContext PetscStreamType","PetscDeviceContextSetStreamType",PetscStreamTypes,PETSC_STREAM_MAX,PetscStreamTypes[dctx->streamType],&stype,&flag));
  if (flag) CHKERRQ(PetscDeviceContextSetStreamType(dctx,static_cast<PetscStreamType>(stype)));
  CHKERRQ(PetscOptionsEList("-device_context_device_type","Underlying PetscDevice","PetscDeviceContextSetDevice",PetscDeviceTypes+1,PETSC_DEVICE_MAX-1,dctx->device ? PetscDeviceTypes[dctx->device->type] : PetscDeviceTypes[PETSC_DEVICE_CONTEXT_DEFAULT_DEVICE],&dtype,&flag));
  if (flag) {
    CHKERRQ(PetscDeviceContextSetDefaultDeviceForType_Internal(dctx,static_cast<PetscDeviceType>(dtype+1)));
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
