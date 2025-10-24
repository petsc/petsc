#include <petscsys.h>
#include <petscviewer.h>
#include <petsc/private/deviceimpl.h>

/* implementations for <include/petscdevice.h> */
PetscErrorCode PetscDeviceCreate(PETSC_UNUSED PetscDeviceType type, PETSC_UNUSED PetscInt devid, PetscDevice *device)
{
  PetscFunctionBegin;
  PetscAssertPointer(device, 3);
  *device = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscDeviceDestroy(PetscDevice *device)
{
  PetscFunctionBegin;
  PetscAssertPointer(device, 1);
  if (*device) *device = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscDeviceConfigure(PETSC_UNUSED PetscDevice device)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscDeviceView(PETSC_UNUSED PetscDevice device, PetscViewer viewer)
{
  PetscFunctionBegin;
  if (viewer) PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscDeviceGetType(PETSC_UNUSED PetscDevice device, PetscDeviceType *type)
{
  PetscFunctionBegin;
  PetscAssertPointer(type, 2);
  *type = PETSC_DEVICE_DEFAULT();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscDeviceGetDeviceId(PETSC_UNUSED PetscDevice device, PetscInt *id)
{
  PetscFunctionBegin;
  PetscAssertPointer(id, 2);
  *id = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscDeviceType PETSC_DEVICE_DEFAULT(void)
{
  return PETSC_DEVICE_HOST;
}

PetscErrorCode PetscDeviceSetDefaultDeviceType(PetscDeviceType type)
{
  PetscFunctionBegin;
  PetscCheck(type == PETSC_DEVICE_HOST, PETSC_COMM_SELF, PETSC_ERR_USER_INPUT, "PetscDevice is disabled but specified PetscDeviceType %s is different than %s", PetscDeviceTypes[type], PetscDeviceTypes[PETSC_DEVICE_HOST]);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscDeviceInitialize(PETSC_UNUSED PetscDeviceType type)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscBool PetscDeviceInitialized(PetscDeviceType type)
{
  return (type == PETSC_DEVICE_HOST) ? PETSC_TRUE : PETSC_FALSE;
}

PetscErrorCode PetscDeviceContextCreate(PetscDeviceContext *dctx)
{
  PetscFunctionBegin;
  PetscAssertPointer(dctx, 1);
  *dctx = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscDeviceContextDestroy(PetscDeviceContext *dctx)
{
  PetscFunctionBegin;
  PetscAssertPointer(dctx, 1);
  if (*dctx) *dctx = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscDeviceContextSetStreamType(PETSC_UNUSED PetscDeviceContext dctx, PETSC_UNUSED PetscStreamType type)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscDeviceContextGetStreamType(PETSC_UNUSED PetscDeviceContext dctx, PetscStreamType *type)
{
  PetscFunctionBegin;
  PetscAssertPointer(type, 2);
  *type = PETSC_STREAM_DEFAULT;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscDeviceContextSetDevice(PETSC_UNUSED PetscDeviceContext dctx, PETSC_UNUSED PetscDevice device)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscDeviceContextGetDevice(PETSC_UNUSED PetscDeviceContext dctx, PetscDevice *device)
{
  PetscFunctionBegin;
  PetscAssertPointer(device, 2);
  *device = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscDeviceContextGetDeviceType(PETSC_UNUSED PetscDeviceContext dctx, PetscDeviceType *type)
{
  PetscFunctionBegin;
  PetscAssertPointer(type, 2);
  *type = PETSC_DEVICE_DEFAULT();
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscDeviceContextSetUp(PETSC_UNUSED PetscDeviceContext dctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscDeviceContextDuplicate(PETSC_UNUSED PetscDeviceContext dctx, PetscDeviceContext *dctxdup)
{
  PetscFunctionBegin;
  PetscAssertPointer(dctxdup, 2);
  *dctxdup = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscDeviceContextQueryIdle(PETSC_UNUSED PetscDeviceContext dctx, PetscBool *idle)
{
  PetscFunctionBegin;
  PetscAssertPointer(idle, 2);
  *idle = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscDeviceContextWaitForContext(PETSC_UNUSED PetscDeviceContext dctxa, PETSC_UNUSED PetscDeviceContext dctxb)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscDeviceContextForkWithStreamType(PETSC_UNUSED PetscDeviceContext dctx, PETSC_UNUSED PetscStreamType stype, PetscInt n, PetscDeviceContext **dsub)
{
  PetscFunctionBegin;
  PetscAssertPointer(dsub, 4);
  PetscAssert(n >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Number of contexts merged %" PetscInt_FMT " < 0", n);
  PetscCall(PetscMalloc1(n, dsub));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscDeviceContextFork(PetscDeviceContext dctx, PetscInt n, PetscDeviceContext **dsub)
{
  PetscFunctionBegin;
  PetscCall(PetscDeviceContextForkWithStreamType(dctx, PETSC_STREAM_DEFAULT, n, dsub));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscDeviceContextJoin(PETSC_UNUSED PetscDeviceContext dctx, PetscInt n, PetscDeviceContextJoinMode joinMode, PetscDeviceContext **dsub)
{
  PetscFunctionBegin;
  PetscAssertPointer(dsub, 4);
  PetscAssert(n >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Number of contexts merged %" PetscInt_FMT " < 0", n);
  if (joinMode == PETSC_DEVICE_CONTEXT_JOIN_DESTROY) PetscCall(PetscFree(*dsub));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscDeviceContextSynchronize(PETSC_UNUSED PetscDeviceContext dctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscDeviceContextSetFromOptions(PETSC_UNUSED MPI_Comm comm, PETSC_UNUSED PetscDeviceContext dctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscDeviceContextView(PETSC_UNUSED PetscDeviceContext dctx, PETSC_UNUSED PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscDeviceContextViewFromOptions(PETSC_UNUSED PetscDeviceContext dctx, PetscObject obj, const char name[])
{
  PetscFunctionBegin;
  if (obj) PetscValidHeader(obj, 2);
  PetscAssertPointer(name, 3);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscDeviceContextGetCurrentContext(PetscDeviceContext *dctx)
{
  PetscFunctionBegin;
  PetscAssertPointer(dctx, 1);
  *dctx = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscDeviceContextSetCurrentContext(PETSC_UNUSED PetscDeviceContext dctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscDeviceContextGetStreamHandle(PETSC_UNUSED PetscDeviceContext dctx, void **handle)
{
  PetscFunctionBegin;
  PetscAssertPointer(handle, 2);
  *handle = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscDeviceAllocate_Private(PETSC_UNUSED PetscDeviceContext dctx, PetscBool clear, PETSC_UNUSED PetscMemType mtype, size_t n, PETSC_UNUSED size_t alignment, void **PETSC_RESTRICT ptr)
{
  PetscFunctionBegin;
  PetscAssertPointer(ptr, 6);
  PetscCall(PetscMallocA(1, clear, __LINE__, PETSC_FUNCTION_NAME, __FILE__, n, ptr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscDeviceDeallocate_Private(PETSC_UNUSED PetscDeviceContext dctx, void *PETSC_RESTRICT ptr)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(ptr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscDeviceMemcpy(PETSC_UNUSED PetscDeviceContext dctx, void *PETSC_RESTRICT dest, const void *PETSC_RESTRICT src, size_t n)
{
  PetscFunctionBegin;
  PetscCall(PetscMemcpy(dest, src, n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscDeviceMemset(PETSC_UNUSED PetscDeviceContext dctx, PETSC_UNUSED void *ptr, PetscInt v, size_t n)
{
  PetscFunctionBegin;
  memset(ptr, (unsigned char)v, n);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* implementations for <include/petsc/private/deviceimpl.h> */
PetscErrorCode PetscDeviceInitializeFromOptions_Internal(PETSC_UNUSED MPI_Comm comm)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscDeviceGetDefaultForType_Internal(PETSC_UNUSED PetscDeviceType type, PetscDevice *device)
{
  PetscFunctionBegin;
  PetscAssertPointer(device, 2);
  *device = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscDeviceContextGetNullContext_Internal(PetscDeviceContext *dctx)
{
  PetscFunctionBegin;
  PetscAssertPointer(dctx, 1);
  *dctx = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscDeviceRegisterMemory(const void *PETSC_RESTRICT ptr, PetscMemType mtype, PETSC_UNUSED size_t size)
{
  PetscFunctionBegin;
  if (PetscMemTypeHost(mtype)) PetscAssertPointer(ptr, 1);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscDeviceGetAttribute(PETSC_UNUSED PetscDevice device, PETSC_UNUSED PetscDeviceAttribute attr, void *value)
{
  PetscFunctionBegin;
  PetscAssertPointer(value, 3);
  *(int *)value = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscGetMarkedObjectMap_Internal(size_t *nkeys, PetscObjectId **keys, PetscMemoryAccessMode **modes, size_t **ndeps, PetscEvent ***dependencies)
{
  PetscFunctionBegin;
  PetscAssertPointer(nkeys, 1);
  PetscAssertPointer(keys, 2);
  PetscAssertPointer(modes, 3);
  PetscAssertPointer(ndeps, 4);
  PetscAssertPointer(dependencies, 5);
  *nkeys        = 0;
  *keys         = NULL;
  *modes        = NULL;
  *ndeps        = NULL;
  *dependencies = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscRestoreMarkedObjectMap_Internal(size_t nkeys, PETSC_UNUSED PetscObjectId **keys, PETSC_UNUSED PetscMemoryAccessMode **modes, PETSC_UNUSED size_t **ndeps, PETSC_UNUSED PetscEvent ***dependencies)
{
  PetscFunctionBegin;
  PetscAssertPointer(keys, 2);
  PetscAssertPointer(modes, 3);
  PetscAssertPointer(ndeps, 4);
  PetscAssertPointer(dependencies, 5);
  if (*keys) *keys = NULL;
  if (*modes) *modes = NULL;
  if (*ndeps) *ndeps = NULL;
  if (*dependencies) *dependencies = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}
