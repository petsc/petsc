#ifndef PETSCDEVICE_H
#define PETSCDEVICE_H

#include <petscdevicetypes.h>
#include <petscviewertypes.h>

/* SUBMANSEC = Sys */

// REVIEW ME: this should probably go somewhere better, configure-time?
#define PETSC_HAVE_HOST 1

/* logging support */
PETSC_EXTERN PetscClassId PETSC_DEVICE_CLASSID;
PETSC_EXTERN PetscClassId PETSC_DEVICE_CONTEXT_CLASSID;

PETSC_EXTERN PetscErrorCode PetscDeviceInitializePackage(void);
PETSC_EXTERN PetscErrorCode PetscDeviceFinalizePackage(void);
PETSC_EXTERN PetscErrorCode PetscGetMemType(const void *, PetscMemType *);

/* PetscDevice */
#if PetscDefined(HAVE_CXX)
PETSC_EXTERN PetscErrorCode  PetscDeviceCreate(PetscDeviceType, PetscInt, PetscDevice *);
PETSC_EXTERN PetscErrorCode  PetscDeviceDestroy(PetscDevice *);
PETSC_EXTERN PetscErrorCode  PetscDeviceConfigure(PetscDevice);
PETSC_EXTERN PetscErrorCode  PetscDeviceView(PetscDevice, PetscViewer);
PETSC_EXTERN PetscErrorCode  PetscDeviceGetType(PetscDevice, PetscDeviceType *);
PETSC_EXTERN PetscErrorCode  PetscDeviceGetDeviceId(PetscDevice, PetscInt *);
PETSC_EXTERN PetscDeviceType PETSC_DEVICE_DEFAULT(void);
PETSC_EXTERN PetscErrorCode  PetscDeviceSetDefaultDeviceType(PetscDeviceType);
PETSC_EXTERN PetscErrorCode  PetscDeviceInitialize(PetscDeviceType);
PETSC_EXTERN PetscBool       PetscDeviceInitialized(PetscDeviceType);
#else
#define PetscDeviceCreate(PetscDeviceType, PetscInt, dev) (*(dev) = PETSC_NULLPTR, 0)
#define PetscDeviceDestroy(dev)                           (*(dev) = PETSC_NULLPTR, 0)
#define PetscDeviceConfigure(PetscDevice)                 0
#define PetscDeviceView(PetscDevice, PetscViewer)         0
#define PetscDeviceGetType(PetscDevice, type)             (*(type) = PETSC_DEVICE_DEFAULT(), 0)
#define PetscDeviceGetDeviceId(PetscDevice, id)           (*(id) = 0)
#define PETSC_DEVICE_DEFAULT()                            PETSC_DEVICE_HOST
#define PetscDeviceSetDefaultDeviceType(PetscDeviceType)  0
#define PetscDeviceInitialize(PetscDeviceType)            0
#define PetscDeviceInitialized(dtype)                     ((dtype) == PETSC_DEVICE_HOST)
#endif /* PetscDefined(HAVE_CXX) */

/* PetscDeviceContext */
#if PetscDefined(HAVE_CXX)
PETSC_EXTERN PetscErrorCode PetscDeviceContextCreate(PetscDeviceContext *);
PETSC_EXTERN PetscErrorCode PetscDeviceContextDestroy(PetscDeviceContext *);
PETSC_EXTERN PetscErrorCode PetscDeviceContextSetStreamType(PetscDeviceContext, PetscStreamType);
PETSC_EXTERN PetscErrorCode PetscDeviceContextGetStreamType(PetscDeviceContext, PetscStreamType *);
PETSC_EXTERN PetscErrorCode PetscDeviceContextSetDevice(PetscDeviceContext, PetscDevice);
PETSC_EXTERN PetscErrorCode PetscDeviceContextGetDevice(PetscDeviceContext, PetscDevice *);
PETSC_EXTERN PetscErrorCode PetscDeviceContextGetDeviceType(PetscDeviceContext, PetscDeviceType *);
PETSC_EXTERN PetscErrorCode PetscDeviceContextSetUp(PetscDeviceContext);
PETSC_EXTERN PetscErrorCode PetscDeviceContextDuplicate(PetscDeviceContext, PetscDeviceContext *);
PETSC_EXTERN PetscErrorCode PetscDeviceContextQueryIdle(PetscDeviceContext, PetscBool *);
PETSC_EXTERN PetscErrorCode PetscDeviceContextWaitForContext(PetscDeviceContext, PetscDeviceContext);
PETSC_EXTERN PetscErrorCode PetscDeviceContextForkWithStreamType(PetscDeviceContext, PetscStreamType, PetscInt, PetscDeviceContext **);
PETSC_EXTERN PetscErrorCode PetscDeviceContextFork(PetscDeviceContext, PetscInt, PetscDeviceContext **);
PETSC_EXTERN PetscErrorCode PetscDeviceContextJoin(PetscDeviceContext, PetscInt, PetscDeviceContextJoinMode, PetscDeviceContext **);
PETSC_EXTERN PetscErrorCode PetscDeviceContextSynchronize(PetscDeviceContext);
PETSC_EXTERN PetscErrorCode PetscDeviceContextSetFromOptions(MPI_Comm, PetscDeviceContext);
PETSC_EXTERN PetscErrorCode PetscDeviceContextView(PetscDeviceContext, PetscViewer);
PETSC_EXTERN PetscErrorCode PetscDeviceContextViewFromOptions(PetscDeviceContext, PetscObject, PetscViewer);
PETSC_EXTERN PetscErrorCode PetscDeviceContextGetCurrentContext(PetscDeviceContext *);
PETSC_EXTERN PetscErrorCode PetscDeviceContextSetCurrentContext(PetscDeviceContext);
#else
#define PetscDeviceContextCreate(dctx)                                                                            (*(dctx) = PETSC_NULLPTR, 0)
#define PetscDeviceContextDestroy(dctx)                                                                           (*(dctx) = PETSC_NULLPTR, 0)
#define PetscDeviceContextSetStreamType(PetscDeviceContext, PetscStreamType)                                      0
#define PetscDeviceContextGetStreamType(PetscDeviceContext, type)                                                 (*(type) = PETSC_STREAM_GLOBAL_BLOCKING, 0)
#define PetscDeviceContextSetDevice(PetscDeviceContext, PetscDevice)                                              0
#define PetscDeviceContextGetDevice(PetscDeviceContext, device)                                                   (*(device) = PETSC_NULLPTR, 0)
#define PetscDeviceContextGetDeviceType(PetscDeviceContext, type)                                                 (*(type) = PETSC_DEVICE_DEFAULT())
#define PetscDeviceContextSetUp(PetscDeviceContext)                                                               0
#define PetscDeviceContextDuplicate(PetscDeviceContextl, PetscDeviceContextr)                                     (*(PetscDeviceContextr) = PETSC_NULLPTR, 0)
#define PetscDeviceContextQueryIdle(PetscDeviceContext, idle)                                                     (*(idle) = PETSC_TRUE, 0)
#define PetscDeviceContextWaitForContext(PetscDeviceContextl, PetscDeviceContextr)                                0
#define PetscDeviceContextForkWithStreamType(PetscDeviceContextp, PetscStreamType, PetscInt, PetscDeviceContextc) (*(PetscDeviceContextc) = PETSC_NULLPTR, 0)
#define PetscDeviceContextFork(PetscDeviceContextp, PetscInt, PetscDeviceContextc)                                (*(PetscDeviceContextc) = PETSC_NULLPTR, 0)
#define PetscDeviceContextJoin(PetscDeviceContextp, PetscInt, PetscDeviceContextJoinMode, PetscDeviceContextc)    (*(PetscDeviceContextc) = PETSC_NULLPTR, 0)
#define PetscDeviceContextSynchronize(PetscDeviceContext)                                                         0
#define PetscDeviceContextSetFromOptions(MPI_Comm, PetscDeviceContext)                                            0
#define PetscDeviceContextView(PetscDeviceContext, PetscViewer)                                                   0
#define PetscDeviceContextViewFromOptions(PetscDeviceContext, PetscObject, PetscViewer)                           0
#define PetscDeviceContextGetCurrentContext(dctx)                                                                 (*(dctx) = PETSC_NULLPTR, 0)
#define PetscDeviceContextSetCurrentContext(PetscDeviceContext)                                                   0
#endif /* PetscDefined(HAVE_CXX) */

/* memory */
#if PetscDefined(HAVE_CXX)
PETSC_EXTERN PetscErrorCode PetscDeviceAllocate(PetscDeviceContext, PetscBool, PetscMemType, size_t, void **PETSC_RESTRICT);
PETSC_EXTERN PetscErrorCode PetscDeviceDeallocate(PetscDeviceContext, void *PETSC_RESTRICT);
PETSC_EXTERN PetscErrorCode PetscDeviceMemcpy(PetscDeviceContext, void *PETSC_RESTRICT, const void *PETSC_RESTRICT, size_t);
PETSC_EXTERN PetscErrorCode PetscDeviceMemset(PetscDeviceContext, void *PETSC_RESTRICT, PetscInt, size_t);
#else
#include <string.h> // memset()
#define PetscDeviceAllocate(PetscDeviceContext, clear, PetscMemType, size, ptr) PetscMallocA(1, (clear), __LINE__, PETSC_FUNCTION_NAME, __FILE__, (size), (ptr))
#define PetscDeviceDeallocate(PetscDeviceContext, ptr)                          PetscFree((ptr))
#define PetscDeviceMemcpy(PetscDeviceContext, dest, src, size)                  PetscMemcpy((dest), (src), (size))
#define PetscDeviceMemset(PetscDeviceContext, ptr, v, size)                     ((void)memset((ptr), (unsigned char)(v), (size)), 0)
#endif /* PetscDefined(HAVE_CXX) */

/*MC
  PetscDeviceMalloc - Allocate device-aware memory

  Synopsis:
  #include <petscdevice.h>
  PetscErrorCode PetscDeviceMalloc(PetscDeviceContext dctx, PetscMemType mtype, size_t n, Type **ptr)

  Not Collective, Asynchronous, Auto-dependency aware

  Input Parameters:
+ dctx  - The `PetscDeviceContext` used to allocate the memory
. mtype - The type of memory to allocate
- n     - The amount (in elements) to allocate

  Output Parameter:
. ptr - The pointer to store the result in

  Notes:
  See `PetscDeviceAllocate()` for more detailed discussion on usage and async semantics.

  This uses the `sizeof()` of the memory type requested to determine the total memory to be
  allocated, therefore you should not multiply the number of elements requested by the
  `sizeof()` the type\:

.vb
  PetscInt *arr;

  // correct
  PetscDeviceMalloc(dctx,PETSC_MEMTYPE_DEVICE,n,&arr);

  // incorrect
  PetscDeviceMalloc(dctx,PETSC_MEMTYPE_DEVICE,n*sizeof(*arr),&arr);
.ve

  This routine falls back to using `PetscMalloc1()` (which is fully synchronous) if PETSc was
  not configured with device support. The user should note that `mtype` is ignored in this
  case, as `PetscMalloc1()` allocates only host memory.

  Level: beginner

.N ASYNC_API

.seealso: `PetscDeviceFree()`, `PetscDeviceCalloc()`, `PetscDeviceArrayCopy()`,
`PetscDeviceArrayZero()`, `PetscDeviceAllocate()`, `PetscDeviceDeallocate()`
M*/
#define PetscDeviceMalloc(dctx, mtype, n, ptr) (PetscDefined(HAVE_DEVICE) ? PetscDeviceAllocate((dctx), PETSC_FALSE, (mtype), (size_t)(n) * sizeof(**(ptr)), (void **)(ptr)) : PetscMalloc1((n), (ptr)))

/*MC
  PetscDeviceCalloc - Allocate zeroed device-aware memory

  Synopsis:
  #include <petscdevice.h>
  PetscErrorCode PetscDeviceCalloc(PetscDeviceContext dctx, PetscMemType mtype, size_t n, Type **ptr)

  Not Collective, Asynchronous, Auto-dependency aware

  Input Parameters:
+ dctx  - The `PetscDeviceContext` used to allocate the memory
. mtype - The type of memory to allocate
- n     - The amount (in elements) to allocate

  Output Parameter:
. ptr - The pointer to store the result in

  Notes:
  Has identical usage to `PetscDeviceMalloc()` except that the memory is zeroed before it is
  returned. See `PetscDeviceMalloc()` for further discussion.

  This routine falls back to using `PetscCalloc1()` if PETSc was not configured with device
  support. The user should note that `mtype` is ignored in this case, as `PetscCalloc1()`
  allocates only host memory.

  Level: beginner

.N ASYNC_API

.seealso: `PetscDeviceFree()`, `PetscDeviceMalloc()`, `PetscDeviceArrayCopy()`,
`PetscDeviceArrayZero()`, `PetscDeviceAllocate()`, `PetscDeviceDeallocate()`
M*/
#define PetscDeviceCalloc(dctx, mtype, n, ptr) (PetscDefined(HAVE_DEVICE) ? PetscDeviceAllocate((dctx), PETSC_TRUE, (mtype), (size_t)(n) * sizeof(**(ptr)), (void **)(ptr)) : PetscCalloc1((n), (ptr)))

/*MC
  PetscDeviceFree - Free device-aware memory

  Synopsis:
  #include <petscdevice.h>
  PetscErrorCode PetscDeviceFree(PetscDeviceContext dctx, void *ptr)

  Not Collective, Asynchronous, Auto-dependency aware

  Input Parameters:
+ dctx - The `PetscDeviceContext` used to free the memory
- ptr  - The pointer to free

  Notes:
  `ptr` must have been allocated using `PetscDeviceMalloc()`, `PetscDeviceCalloc()`, or
  `PetscDeviceAllocate()`, or registered with the system using `PetscRegisterMemory()`.

  Automatically sets `ptr` to `PETSC_NULLPTR` on successful deallocation.

  This routine falls back to using `PetscFree()` if PETSc was not configured with device
  support. The user should note that `PetscFree()` frees only host memory.

  See `PetscDeviceDeallocate()` for more further discussion.

  Level: beginner

.N ASYNC_API

.seealso: `PetscDeviceMalloc()`, `PetscDeviceCalloc()`, `PetscDeviceDeallocate()`
M*/
#define PetscDeviceFree(dctx, ptr) ((ptr) ? (PetscDefined(HAVE_DEVICE) ? (PetscDeviceDeallocate((dctx), (ptr)) || ((ptr) = PETSC_NULLPTR, 0)) : PetscFree(ptr)) : 0)

/*MC
  PetscDeviceArrayCopy - Copy memory in a device-aware manner

  Synopsis:
  #include <petscdevice.h>
  PetscErrorCode PetscDeviceArrayCopy(PetscDeviceContext dctx, void *PETSC_RESTRICT dest, const void *PETSC_RESTRICT src, size_t n, PetscDeviceCopyMode mode)

  Not Collective, Asynchronous, Auto-dependency aware

  Input Parameters:
+ dctx - The `PetscDeviceContext` used to copy the memory
. dest - The pointer to copy to
. src  - The pointer to copy from
- n    - The amount (in elements) to copy

  Notes:
  Both `dest` and `src` must have been allocated using any of `PetscDeviceMalloc()`,
  `PetscDeviceCalloc()` or `PetscDeviceAllocate()`, or registered with the system via
  `PetscDeviceRegisterMemory()`.

  This uses the `sizeof()` of the `src` memory type requested to determine the total memory to
  be copied, therefore you should not multiply the number of elements by the `sizeof()` the
  type\:

.vb
  PetscInt *to,*from;

  // correct
  PetscDeviceArrayCopy(dctx,to,from,n,PETSC_DEVICE_COPY_AUTO);

  // incorrect
  PetscDeviceArrayCopy(dctx,to,from,n*sizeof(*from),PETSC_DEVICE_COPY_AUTO);
.ve

  See `PetscDeviceMemcpy()` for further discussion.

  Level: beginner

.N ASYNC_API

.seealso: `PetscDeviceMalloc()`, `PetscDeviceCalloc()`, `PetscDeviceRegisterMemory()`,
`PetscDeviceFree()`, `PetscDeviceArrayZero()`, `PetscDeviceMemcpy()`
M*/
#define PetscDeviceArrayCopy(dctx, dest, src, n) ((n) ? (PetscDefined(HAVE_DEVICE) ? PetscDeviceMemcpy((dctx), (dest), (src), (size_t)(n) * sizeof(*(src))) : PetscArraycpy((dest), (src), (n))) : 0)

/*MC
  PetscDeviceArrayZero - Zero memory in a device-aware manner

  Synopsis:
  #include <petscdevice.h>
  PetscErrorCode PetscDeviceArrayZero(PetscDeviceContext dctx, PetscMemType mtype, void *ptr, size_t n)

  Not Collective, Asynchronous, Auto-dependency aware

  Input Parameters:
+ dctx  - The `PetscDeviceContext` used to zero the memory
. ptr   - The pointer to the memory
- n     - The amount (in elements) to zero

  Notes:
  `ptr` must have been allocated using any of `PetscDeviceMalloc()`, `PetscDeviceCalloc()` or
  `PetscDeviceAllocate()`, or registered with the system via `PetscDeviceRegisterMemory()`.

  This uses the `sizeof()` of the memory type requested to determine the total memory to be
  zeroed, therefore you should not multiply the number of elements by the `sizeof()` the type\:

.vb
  PetscInt *ptr;

  // correct
  PetscDeviceArrayZero(dctx,PETSC_MEMTYPE_DEVICE,ptr,n);

  // incorrect
  PetscDeviceArrayZero(dctx,PETSC_MEMTYPE_DEVICE,ptr,n*sizeof(*ptr));
.ve

  See `PetscDeviceMemset()` for futher discussion.

  Level: beginner

.N ASYNC_API

.seealso: `PetscDeviceMalloc()`, `PetscDeviceCalloc()`, `PetscDeviceRegisterMemory()`,
`PetscDeviceFree()`, `PetscDeviceArrayCopy()`, `PetscDeviceMemset()`
M*/
#define PetscDeviceArrayZero(dctx, ptr, n) ((n) ? (PetscDefined(HAVE_DEVICE) ? PetscDeviceMemset((dctx), (ptr), 0, (size_t)(n) * sizeof(*(ptr))) : PetscArrayzero((ptr), (n))) : 0)

#endif /* PETSCDEVICE_H */
