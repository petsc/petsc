#include <petsc/private/deviceimpl.h> /*I "petscdevice.h" I*/
#include "objpool.hpp"

/* Define the allocator */
struct PetscDeviceContextAllocator : Petsc::Allocator<PetscDeviceContext>
{
  static PetscInt PetscDeviceContextID;

  PETSC_NODISCARD PetscErrorCode create(PetscDeviceContext *dctx) noexcept
  {
    PetscDeviceContext dc;
    PetscErrorCode     ierr;

    PetscFunctionBegin;
    ierr           = PetscNew(&dc);CHKERRQ(ierr);
    dc->id         = PetscDeviceContextID++;
    dc->idle       = PETSC_TRUE;
    dc->streamType = PETSC_STREAM_DEFAULT_BLOCKING;
    *dctx          = dc;
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD PetscErrorCode destroy(PetscDeviceContext &dctx) const noexcept
  {
    PetscErrorCode ierr;

    PetscFunctionBegin;
    if (PetscUnlikelyDebug(dctx->numChildren)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Device context still has %D un-restored children, must call PetscDeviceContextRestore() on all children before destroying",dctx->numChildren);
    if (dctx->ops->destroy) {ierr = (*dctx->ops->destroy)(dctx);CHKERRQ(ierr);}
    ierr = PetscDeviceDestroy(&dctx->device);CHKERRQ(ierr);
    ierr = PetscFree(dctx->childIDs);CHKERRQ(ierr);
    ierr = PetscFree(dctx);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD PetscErrorCode reset(PetscDeviceContext &dctx) const noexcept
  {
    PetscErrorCode ierr;

    PetscFunctionBegin;
    /* don't deallocate the child array, rather just zero it out */
    ierr = PetscArrayzero(dctx->childIDs,dctx->maxNumChildren);CHKERRQ(ierr);
    dctx->setup       = PETSC_FALSE;
    dctx->numChildren = 0;
    dctx->idle        = PETSC_TRUE;
    dctx->streamType  = PETSC_STREAM_DEFAULT_BLOCKING;
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD PetscErrorCode finalize(void) noexcept
  {
    PetscFunctionBegin;
    PetscDeviceContextID = 0;
    PetscFunctionReturn(0);
  }
};
PetscInt PetscDeviceContextAllocator::PetscDeviceContextID = 0;

static Petsc::ObjectPool<PetscDeviceContext,PetscDeviceContextAllocator> contextPool;

/*@C
  PetscDeviceContextCreate - Creates a PetscDeviceContext

  Not Collective, Asynchronous

  Ouput Paramemters:
. dctx - The PetscDeviceContext

  Notes:
  Unlike almost every other PETSc class it is advised that most users use
  PetscDeviceContextDuplicate() rather than this routine to create new contexts. Contexts
  of different types are incompatible with one another; using
  PetscDeviceContextDuplicate() ensures compatible types.

  Level: beginner

.seealso: PetscDeviceContextDuplicate(), PetscDeviceContextSetDevice(),
PetscDeviceContextSetStreamType(), PetscDeviceContextSetUp(), PetscDeviceContextDestroy(),
PetscDeviceContextSetFromOptions()
@*/
PetscErrorCode PetscDeviceContextCreate(PetscDeviceContext *dctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(dctx,1);
  ierr = PetscDeviceInitializePackage();CHKERRQ(ierr);
  ierr = contextPool.get(*dctx);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*dctx) PetscFunctionReturn(0);
  /* use move assignment whenever possible */
  ierr = contextPool.reclaim(std::move(*dctx));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextSetStreamType - Set the implementation type of the underlying stream for a PetscDeviceContext

  Not Collective, Asynchronous

  Input Paramaters:
+ dctx - The PetscDeviceContext
- type - The PetscStreamType

  Notes:
  See PetscStreamType in include/petscdevicetypes.h for more information on the available
  types and their interactions. If the PetscDeviceContext was previously set up and stream
  type was changed, you must call PetscDeviceContextSetUp() again after this routine.

  Level: intermediate

.seealso: PetscDeviceContextCreate(), PetscDeviceContextSetDevice(),
PetscDeviceContextGetStreamType(), PetscDeviceContextSetUp(), PetscDeviceContextSetFromOptions()
@*/
PetscErrorCode PetscDeviceContextSetStreamType(PetscDeviceContext dctx, PetscStreamType type)
{
  PetscFunctionBegin;
  PetscValidDeviceContext(dctx,1);
  PetscValidStreamType(type,2);
  /* only need to do complex swapping if the object has already been setup */
  if (dctx->setup && (dctx->streamType != type)) {
    PetscErrorCode ierr;

    ierr = (*dctx->ops->changestreamtype)(dctx,type);CHKERRQ(ierr);
    dctx->setup = PETSC_FALSE;
  }
  dctx->streamType = type;
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextGetStreamType - Get the implementation type of the underlying stream for a PetscDeviceContext

  Not Collective, Asynchronous

  Input Paramater:
. dctx - The PetscDeviceContext

  Output Parameter:
. type - The PetscStreamType

  Notes:
  See PetscStreamType in include/petscdevicetypes.h for more information on the available types and their interactions

  Level: intermediate

.seealso: PetscDeviceContextCreate(), PetscDeviceContextSetDevice(),
PetscDeviceContextSetStreamType(), PetscDeviceContextSetFromOptions()
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

  Input Paramaters:
+ dctx   - The PetscDeviceContext
- device - The PetscDevice

  Notes:
  This routine is effectively PetscDeviceContext's "set-type" (so every PetscDeviceContext
  must also have an attached PetscDevice). Unlike the usual set-type semantics, it is
  not stricly necessary to set a contexts device to enable usage, any created device
  contexts will always come equipped with the "default" device.

  Level: intermediate

.seealso: PetscDeviceCreate(), PetscDeviceContextGetDevice()
@*/
PetscErrorCode PetscDeviceContextSetDevice(PetscDeviceContext dctx, PetscDevice device)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidDeviceContext(dctx,1);
  PetscValidDevice(device,2);
  if (dctx->device == device) PetscFunctionReturn(0);
  ierr = PetscDeviceDestroy(&dctx->device);CHKERRQ(ierr);
  ierr = PetscMemzero(dctx->ops,sizeof(*dctx->ops));CHKERRQ(ierr);
  ierr = (*device->ops->createcontext)(dctx);CHKERRQ(ierr);
  dctx->device = PetscDeviceReference(device);
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

.seealso: PetscDeviceContextSetDevice(), PetscDevice
@*/
PetscErrorCode PetscDeviceContextGetDevice(PetscDeviceContext dctx, PetscDevice *device)
{
  PetscFunctionBegin;
  PetscValidDeviceContext(dctx,1);
  PetscValidPointer(device,2);
  *device = dctx->device;
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextSetUp - Prepares a PetscDeviceContext for use

  Not Collective, Asynchronous

  Intput Parameter:
. dctx - The PetscDeviceContext

  Developer Notes:
  This routine is usually the stage where a PetscDeviceContext acquires device-side data structures such as streams,
  events, and (possibly) handles.

  Level: beginner

.seealso: PetscDeviceContextTypes, PetscDeviceContextCreate(),
PetscDeviceContextSetDevice(), PetscDeviceContextDestroy(), PetscDeviceContextSetFromOptions()
@*/
PetscErrorCode PetscDeviceContextSetUp(PetscDeviceContext dctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidDeviceContext(dctx,1);
  if (!dctx->device) {
    ierr = PetscInfo2(NULL,"PetscDeviceContext %d did not have an explicitly attached PetscDevice, using default with type %s\n",dctx->id,PetscDeviceKinds[PETSC_DEVICE_DEFAULT]);CHKERRQ(ierr);
    ierr = PetscDeviceContextSetDevice(dctx,PetscDeviceDefault_Internal());CHKERRQ(ierr);
  }
  if (dctx->setup) PetscFunctionReturn(0);
  ierr = (*dctx->ops->setup)(dctx);CHKERRQ(ierr);
  dctx->setup = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextDuplicate - Duplicates a PetscDeviceContext object

  Not Collective, Asynchronous

  Input Parameter:
. dctx - The PetscDeviceContext to duplicate

  Output Paramter:
. strmdup - The duplicated PetscDeviceContext

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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidDeviceContext(dctx,1);
  PetscValidPointer(dctxdup,2);
  ierr = PetscDeviceContextCreate(dctxdup);CHKERRQ(ierr);
  ierr = PetscDeviceContextSetDevice(*dctxdup,dctx->device);CHKERRQ(ierr);
  ierr = PetscDeviceContextSetStreamType(*dctxdup,dctx->streamType);CHKERRQ(ierr);
  ierr = PetscDeviceContextSetUp(*dctxdup);CHKERRQ(ierr);
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
  This routine only refers a singular context and does NOT take any of its children into account. That is, if dctx is
  idle but has dependents who do have work, this routine still returns PETSC_TRUE.

  Results of PetscDeviceContextQueryIdle() are cached on return, allowing this function to be called repeatedly in an
  efficient manner. When debug mode is enabled this cache is verified on every call to
  this routine, but is blindly believed when debugging is disabled.

  Level: intermediate

.seealso: PetscDeviceContextCreate(), PetscDeviceContextWaitForContext(), PetscDeviceContextFork()
@*/
PetscErrorCode PetscDeviceContextQueryIdle(PetscDeviceContext dctx, PetscBool *idle)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidDeviceContext(dctx,1);
  PetscValidBoolPointer(idle,2);
  if (dctx->idle) {
    *idle = PETSC_TRUE;
    ierr = PetscDeviceContextValidateIdle_Internal(dctx);CHKERRQ(ierr);
  } else {
    ierr = (*dctx->ops->query)(dctx,idle);CHKERRQ(ierr);
    dctx->idle = *idle;
  }
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscCheckCompatibleDeviceContexts(dctxa,1,dctxb,2);
  if (dctxa == dctxb) PetscFunctionReturn(0);
  if (dctxb->idle) {
    /* No need to do the extra function lookup and event record if the stream were waiting on isn't doing anything */
    ierr = PetscDeviceContextValidateIdle_Internal(dctxb);CHKERRQ(ierr);
  } else {
    ierr = (*dctxa->ops->waitforctx)(dctxa,dctxb);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

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
#if defined(PETSC_USE_DEBUG) && defined(PETSC_USE_INFO)
  const PetscInt      nBefore = n;
  static std::string  idList;
#endif
  PetscDeviceContext *dsubTmp = nullptr;
  PetscInt            i = 0;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  PetscValidDeviceContext(dctx,1);
  PetscValidPointer(dsub,3);
  if (PetscUnlikelyDebug(n < 0)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Number of contexts requested %D < 0",n);
#if defined(PETSC_USE_DEBUG) && defined(PETSC_USE_INFO)
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
      ierr = PetscRealloc(dctx->numChildren*sizeof(*dctx->childIDs),&dctx->childIDs);CHKERRQ(ierr);
      /* clear the extra memory since realloc doesn't do it for us */
      ierr = PetscArrayzero((dctx->childIDs)+(dctx->maxNumChildren),(dctx->numChildren)-(dctx->maxNumChildren));CHKERRQ(ierr);
    } else {
      /* have no children */
      ierr = PetscCalloc1(dctx->numChildren,&dctx->childIDs);CHKERRQ(ierr);
    }
    /* update total number of children */
    dctx->maxNumChildren = dctx->numChildren;
  }
  ierr = PetscMalloc1(n,&dsubTmp);CHKERRQ(ierr);
  while (n) {
    /* empty child slot */
    if (!(dctx->childIDs[i])) {
      /* create the child context in the image of its parent */
      ierr = PetscDeviceContextDuplicate(dctx,dsubTmp+i);CHKERRQ(ierr);
      ierr = PetscDeviceContextWaitForContext(dsubTmp[i],dctx);CHKERRQ(ierr);
      /* register the child with its parent */
      dctx->childIDs[i] = dsubTmp[i]->id;
#if defined(PETSC_USE_DEBUG) && defined(PETSC_USE_INFO)
      idList += std::to_string(dsubTmp[i]->id);
      if (n != 1) idList += ", ";
#endif
      --n;
    }
    ++i;
  }
#if defined(PETSC_USE_DEBUG) && defined(PETSC_USE_INFO)
  ierr = PetscInfo3(NULL,"Forked %D children from parent %D with IDs: %s\n",nBefore,dctx->id,idList.c_str());CHKERRQ(ierr);
  /* resets the size but doesn't deallocate the memory */
  idList.clear();
#endif
  /* pass the children back to caller */
  *dsub = dsubTmp;
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextJoin() - Converge a set of child contexts

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
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  /* validity of dctx is checked in the wait-for loop */
  PetscValidPointer(dsub,4);
  if (PetscUnlikelyDebug(n < 0)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Number of contexts merged %D < 0",n);
#if defined(PETSC_USE_DEBUG) && defined(PETSC_USE_INFO)
  /* reserve 4 chars per id, 2 for number and 2 for ', ' separator */
  idList.reserve(4*n);
#endif
  /* first dctx waits on all the incoming edges */
  for (PetscInt i = 0; i < n; ++i) {
    PetscCheckCompatibleDeviceContexts(dctx,1,(*dsub)[i],4);
    ierr = PetscDeviceContextWaitForContext(dctx,(*dsub)[i]);CHKERRQ(ierr);
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

      if (PetscUnlikelyDebug(n > dctx->numChildren)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Trying to destroy %D children of a parent context that only has %D children, likely trying to restore to wrong parent",n,dctx->numChildren);
      /* update child count while it's still fresh in memory */
      dctx->numChildren -= n;
      for (PetscInt i = 0; i < dctx->maxNumChildren; ++i) {
        if (dctx->childIDs[i] && (dctx->childIDs[i] == (*dsub)[j]->id)) {
          /* child is one of ours, can destroy it */
          ierr = PetscDeviceContextDestroy((*dsub)+j);CHKERRQ(ierr);
          /* reset the child slot */
          dctx->childIDs[i] = 0;
          if (++j == n) break;
        }
      }
      /* gone through the loop but did not find every child, if this triggers (or well, doesn't) on perf-builds we leak the remaining contexts memory */
      if (PetscUnlikelyDebug(j != n)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"%D contexts still remain after destroy, this may be because you are trying to restore to the wrong parent context, or the device contexts are not in the same order as they were checked out out in.",n-j);
      ierr = PetscFree(*dsub);CHKERRQ(ierr);
    }
    break;
  case PETSC_DEVICE_CONTEXT_JOIN_SYNC:
    for (PetscInt i = 0; i < n; ++i) {
      ierr = PetscDeviceContextWaitForContext((*dsub)[i],dctx);CHKERRQ(ierr);
    }
  case PETSC_DEVICE_CONTEXT_JOIN_NO_SYNC:
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Unknown PetscDeviceContextJoinMode given");
    break;
  }

#if defined(PETSC_USE_DEBUG) && defined(PETSC_USE_INFO)
  ierr = PetscInfo4(NULL,"Joined %D ctxs to ctx %D, mode %s with IDs: %s\n",n,dctx->id,PetscDeviceContextJoinModes[joinMode],idList.c_str());CHKERRQ(ierr);
  idList.clear();
#endif
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextSynchronize() - Block the host until all work queued on or associated with a PetscDeviceContext has finished

  Not Collective, Synchronous

  Input Parameters:
. dctx - The PetscDeviceContext to synchronize

  Level: beginner

.seealso: PetscDeviceContextFork(), PetscDeviceContextJoin(), PetscDeviceContextQueryIdle()
@*/
PetscErrorCode PetscDeviceContextSynchronize(PetscDeviceContext dctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidDeviceContext(dctx,1);
  /* if it isn't setup there is nothing to sync on */
  if (dctx->setup) {ierr = (*dctx->ops->synchronize)(dctx);CHKERRQ(ierr);}
  dctx->idle = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscDeviceContext globalContext      = nullptr;
static PetscBool          globalContextSetup = PETSC_FALSE;
static PetscStreamType    defaultStreamType  = PETSC_STREAM_DEFAULT_BLOCKING;

/* automatically registered to PetscFinalize() when first context is instantiated, do not
   call */
static PetscErrorCode PetscDeviceContextDestroyGlobalContext_Private(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDeviceContextSynchronize(globalContext);CHKERRQ(ierr);
  ierr = PetscDeviceContextDestroy(&globalContext);CHKERRQ(ierr);
  /* reset everything to defaults */
  defaultStreamType  = PETSC_STREAM_DEFAULT_BLOCKING;
  globalContextSetup = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/* creates and initializes the root context in PetscInitialize() but does not call
   SetUp() as the user may wish to change types after PetscInitialize() */
PetscErrorCode PetscDeviceContextInitializeRootContext_Internal(MPI_Comm comm, const char prefix[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInfo1(NULL,"Initializing root PetscDeviceContext with PetscDeviceKind %s\n",PetscDeviceKinds[PETSC_DEVICE_DEFAULT]);CHKERRQ(ierr);
  ierr = PetscDeviceContextCreate(&globalContext);CHKERRQ(ierr);
  if (PetscUnlikelyDebug(globalContext->id != 0)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"The root current PetscDeviceContext should have id = 0, however it has id = %D",globalContext->id);
  ierr = PetscDeviceContextSetDevice(globalContext,PetscDeviceDefault_Internal());CHKERRQ(ierr);
  ierr = PetscDeviceContextSetStreamType(globalContext,defaultStreamType);CHKERRQ(ierr);
  ierr = PetscDeviceContextSetFromOptions(comm,prefix,globalContext);CHKERRQ(ierr);
  ierr = PetscRegisterFinalize(PetscDeviceContextDestroyGlobalContext_Private);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextGetCurrentContext() - Get the current active PetscDeviceContext

  Not Collective, Asynchronous

  Output Parameter:
. dctx - The PetscDeviceContext

  Notes:
  The user generally should not destroy contexts retrieved with this routine unless they themselves have created
  them. There exists no protection against destroying the root context.

  Developer Notes:
  This routine creates the "root" context the first time it is called, registering its
  destructor to PetscFinalize(). The root context is synchronized before being destroyed.

  Level: beginner

.seealso: PetscDeviceContextSetCurrentContext(), PetscDeviceContextFork(),
PetscDeviceContextJoin(), PetscDeviceContextCreate()
@*/
PetscErrorCode PetscDeviceContextGetCurrentContext(PetscDeviceContext *dctx)
{
  PetscFunctionBegin;
  PetscValidPointer(dctx,1);
  if (PetscUnlikely(!globalContextSetup)) {
    PetscErrorCode ierr;

    /* if there is no available device backend, PetscDeviceInitializePackage() will fire a
       PETSC_ERR_SUP_SYS error. */
    ierr = PetscDeviceInitializePackage();CHKERRQ(ierr);
    ierr = PetscDeviceContextSetUp(globalContext);CHKERRQ(ierr);
    globalContextSetup = PETSC_TRUE;
  }
  *dctx = globalContext;
  PetscFunctionReturn(0);
}

/*@C
  PetscDeviceContextSetCurrentContext() - Set the current active PetscDeviceContext

  Not Collective, Asynchronous

  Input Parameter:
. dctx - The PetscDeviceContext

  Notes:
  The old context is not stored in any way by this routine; if one is overriding a context that they themselves do not
  control, one should take care to temporarily store it by calling PetscDeviceContextGetCurrentContext() before calling
  this routine.

  Level: beginner

.seealso: PetscDeviceContextGetCurrentContext(), PetscDeviceContextFork(),
PetscDeviceContextJoin(), PetscDeviceContextCreate()
@*/
PetscErrorCode PetscDeviceContextSetCurrentContext(PetscDeviceContext dctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidDeviceContext(dctx,1);
  globalContext = dctx;
  ierr = PetscInfo1(NULL,"Set global device context id %D\n",dctx->id);CHKERRQ(ierr);
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
. -device_context_device_kind - the kind of PetscDevice to attach by default
. -device_context_stream_type - type of stream to create inside the PetscDeviceContext -
  PetscDeviceContextSetStreamType()

  Level: beginner

.seealso: PetscDeviceContextSetStreamType()
@*/
PetscErrorCode PetscDeviceContextSetFromOptions(MPI_Comm comm, const char prefix[], PetscDeviceContext dctx)
{
  PetscBool      flag;
  PetscInt       stype,dkind;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (prefix) {PetscValidCharPointer(prefix,2);}
  PetscValidDeviceContext(dctx,3);
  ierr = PetscOptionsBegin(comm,prefix,"PetscDeviceContext Options","Sys");CHKERRQ(ierr);
  ierr = PetscOptionsEList("-device_context_device_kind","Underlying PetscDevice","PetscDeviceContextSetDevice",PetscDeviceKinds+1,PETSC_DEVICE_MAX-1,dctx->device ? PetscDeviceKinds[dctx->device->kind] : PetscDeviceKinds[PETSC_DEVICE_DEFAULT],&dkind,&flag);CHKERRQ(ierr);
  if (flag) {
    ierr = PetscDeviceContextSetDevice(dctx,PetscDeviceDefaultKind_Internal(static_cast<PetscDeviceKind>(dkind+1)));CHKERRQ(ierr);
  }
  ierr = PetscOptionsEList("-device_context_stream_type","PetscDeviceContext PetscStreamType","PetscDeviceContextSetStreamType",PetscStreamTypes,3,PetscStreamTypes[dctx->streamType],&stype,&flag);CHKERRQ(ierr);
  if (flag) {
    ierr = PetscDeviceContextSetStreamType(dctx,static_cast<PetscStreamType>(stype));CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
