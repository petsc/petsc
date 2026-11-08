#pragma once

#include <petscsys.h>
#include <petscdevicetypes.h>

#if PETSC_CPP_VERSION >= 11 // C++11
  #define PETSC_DEVICE_ALIGNOF(...) alignof(decltype(__VA_ARGS__))
#elif PETSC_C_VERSION >= 11 // C11
  #if defined(__GNUC__)
    #define PETSC_DEVICE_ALIGNOF(...) _Alignof(__typeof__(__VA_ARGS__))
  #else
    #include <stddef.h> // max_align_t
    // Note we cannot just do _Alignof(expression) since clang warns that "'_Alignof' applied to an
    // expression is a GNU extension", so we just default to max_align_t which is ultra safe
    #define PETSC_DEVICE_ALIGNOF(...) _Alignof(max_align_t)
  #endif // __GNUC__
#else
  #define PETSC_DEVICE_ALIGNOF(...) PETSC_MEMALIGN
#endif

/* MANSEC = Sys */
/* SUBMANSEC = Device */

// REVIEW ME: this should probably go somewhere better, configure-time?
#define PETSC_HAVE_HOST 1

/* logging support */
PETSC_EXTERN PetscClassId PETSC_DEVICE_CLASSID;
PETSC_EXTERN PetscClassId PETSC_DEVICE_CONTEXT_CLASSID;

PETSC_EXTERN PetscErrorCode PetscDeviceInitializePackage(void);
PETSC_EXTERN PetscErrorCode PetscDeviceFinalizePackage(void);
PETSC_EXTERN PetscErrorCode PetscGetMemType(const void *, PetscMemType *);

/* PetscDevice */
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

/* PetscDeviceContext */
/*MC
  PetscDeviceContextDefault - Placeholder for the default-stream context on the current device

  Level: beginner

  Notes:
  Pass this symbol to device-memory operations or to device-context operations that inspect,
  synchronize, duplicate, or fork a context. It is also accepted by `PetscDeviceContextSetUp()`,
  `PetscDeviceContextSetCurrentContext()`, and as the parent argument of `PetscDeviceContextJoin()`.
  When a call needs a context, it resolves this symbol to a PETSc-owned `PETSC_STREAM_DEFAULT` context
  for the backend and device attached to the current context, creating that context lazily. It does
  not select the current stream. No-work returns do not resolve the placeholder or initialize a device.

  Saving this placeholder does not capture a context. After changing the current device, another call
  resolves it for the new device. Use `PetscDeviceContextGetDefaultContext()` to capture a concrete
  handle. A fork and its destroying join must resolve to the same parent; keep the device unchanged
  between those calls or capture the parent first. Join child arrays must contain concrete handles.

  This symbol cannot be destroyed, reconfigured with `PetscDeviceContextSetDevice()`,
  `PetscDeviceContextSetStreamType()`, or `PetscDeviceContextSetFromOptions()`, or passed to generic
  `PetscObject` routines. It does not represent a `NULL` pointer. Default-stream selection does not
  synchronize unrelated nonblocking streams or establish dependencies on them.

  Developer Note:
  This placeholder is for application use. Resolve it at public API boundaries before ordinary
  object validation, object access, or backend dispatch.

  Some private helpers accept a `NULL` context to request the current context. They resolve it to
  a concrete context before passing it to a public device API or accessing its state.

.seealso: `PetscDeviceContext`, `PetscDeviceContextGetDefaultContext()`, `PetscDeviceContextGetCurrentContext()`, `PetscStreamType`
M*/
PETSC_EXTERN PetscDeviceContext const PetscDeviceContextDefault;

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
PETSC_EXTERN PetscErrorCode PetscDeviceContextDelay(PetscDeviceContext, PetscReal);
PETSC_EXTERN PetscErrorCode PetscDeviceContextSetFromOptions(MPI_Comm, PetscDeviceContext);
PETSC_EXTERN PetscErrorCode PetscDeviceContextView(PetscDeviceContext, PetscViewer);
PETSC_EXTERN PetscErrorCode PetscDeviceContextViewFromOptions(PetscDeviceContext, PetscObject, const char[]);
PETSC_EXTERN PetscErrorCode PetscDeviceContextGetCurrentContext(PetscDeviceContext *);
PETSC_EXTERN PetscErrorCode PetscDeviceContextSetCurrentContext(PetscDeviceContext);
PETSC_EXTERN PetscErrorCode PetscDeviceContextGetStreamHandle(PetscDeviceContext, void **);
PETSC_EXTERN PetscErrorCode PetscDeviceContextGetDefaultContext(PetscDeviceContext *);

/* memory */
PETSC_EXTERN PetscErrorCode PetscDeviceAllocate_Private(PetscDeviceContext, PetscBool, PetscMemType, size_t, size_t, size_t, void **PETSC_RESTRICT);
PETSC_EXTERN PetscErrorCode PetscDeviceDeallocate_Private(PetscDeviceContext, void *PETSC_RESTRICT);
PETSC_EXTERN PetscErrorCode PetscDeviceMemcpy(PetscDeviceContext, void *PETSC_RESTRICT, const void *PETSC_RESTRICT, size_t);
PETSC_EXTERN PetscErrorCode PetscDeviceMemset(PetscDeviceContext, void *PETSC_RESTRICT, PetscInt, size_t);

/*MC
  PetscDeviceMalloc - Allocate device-aware memory

  Synopsis:
  #include <petscdevice.h>
  PetscErrorCode PetscDeviceMalloc(PetscDeviceContext dctx, PetscMemType mtype, size_t n, size_t alignment, Type **ptr)

  Not Collective, Asynchronous, Auto-dependency aware

  Input Parameters:
+ dctx      - The `PetscDeviceContext` used to allocate the memory, or `PetscDeviceContextDefault`
. mtype     - The type of memory to allocate
. n         - The amount (in elements) to allocate
- alignment - The requested alignment in bytes, a positive power of two or `PETSC_DECIDE`

  Output Parameter:
. ptr - The pointer to store the result in

  Level: beginner

  Notes:
  Memory allocated with this function must be freed with `PetscDeviceFree()`.

  If `n` is zero, then `ptr` is set to `PETSC_NULLPTR`.

  CUDA and HIP contexts allocate `PETSC_MEMTYPE_HOST` memory from a pinned host pool and
  `PETSC_MEMTYPE_DEVICE` memory from their device pool. There is no separate pinned host
  memory type. The supplied context selects the backend. Obtain the current context with
  `PetscDeviceContextGetCurrentContext()` to use the same context as high-level PETSc operations.

  If the context has no memory allocation operation, only `PETSC_MEMTYPE_HOST` is supported
  and allocation uses `PetscMalloc1()`. In the C device-interface fallback, `dctx` and `mtype`
  are ignored and allocation always uses `PetscMalloc1()`. Backend allocation failures do
  not trigger a retry with host allocation.

  For CUDA and HIP pools, `PETSC_DECIDE` uses the alignment inferred for the pointed-to type
  in the calling translation unit. When type inference is unavailable, PETSc uses the alignment
  of `max_align_t` or `PETSC_MEMALIGN`. An explicit alignment is honored, or increased to the
  inferred alignment if necessary. This applies to both pinned host and device memory.
  The alignment inferred from a C complex type may be smaller than that required by the
  corresponding CUDA or HIP complex type. If the memory will be accessed using the CUDA
  or HIP type, pass that type's required alignment as `alignment`.

  The `PetscMalloc1()` fallback guarantees `PETSC_MEMALIGN` alignment. An explicit request
  that does not divide `PETSC_MEMALIGN` produces `PETSC_ERR_SUP`; `PETSC_DECIDE` retains the
  ordinary host allocation behavior. A zero element count returns `NULL` without allocating.

  Alignment of an allocation does not guarantee the same alignment for interior pointers.
  For structures shared with device code, the element layout and stride must also agree.

  Allocation, initialization, and pool reuse are ordered on `dctx`, including for pinned host
  memory. Contexts with `PETSC_STREAM_DEFAULT_WITH_BARRIER` or `PETSC_STREAM_NONBLOCKING_WITH_BARRIER`
  synchronize before returning. For other stream types, synchronize `dctx` before accessing the
  contents from the CPU. The alignment argument does not change this synchronization policy.

  This routine uses the `sizeof()` of the memory type requested to determine the total memory
  to be allocated, therefore you should not multiply the number of elements requested by the
  `sizeof()` the type\:

.vb
  PetscInt *arr;

  // correct
  PetscDeviceMalloc(dctx, PETSC_MEMTYPE_DEVICE, n, PETSC_DECIDE, &arr);

  // incorrect
  PetscDeviceMalloc(dctx, PETSC_MEMTYPE_DEVICE, n * sizeof(*arr), PETSC_DECIDE, &arr);
.ve

  Note result stored `ptr` is immediately valid and the user may freely inspect or manipulate
  its value on function return, i.e.\:

.vb
  PetscInt *ptr;

  PetscDeviceMalloc(dctx, PETSC_MEMTYPE_DEVICE, 20, PETSC_DECIDE, &ptr);

  PetscInt *sub_ptr = ptr + 10; // OK, no need to synchronize

  ptr[0] = 10; // ERROR, directly accessing contents of ptr is undefined until synchronization
.ve

  DAG representation:
.vb
  time ->

  -> dctx - |= CALL =| -\- dctx -->
                         \- ptr ->
.ve

.N ASYNC_API

.seealso: `PetscDeviceFree()`, `PetscDeviceCalloc()`, `PetscDeviceArrayCopy()`,
`PetscDeviceArrayZero()`
M*/
#define PetscDeviceMalloc(dctx, mtype, n, alignment, ptr) PetscDeviceAllocate_Private((dctx), PETSC_FALSE, (mtype), (size_t)(n) * sizeof(**(ptr)), (size_t)(alignment), PETSC_DEVICE_ALIGNOF(**(ptr)), (void **)(ptr))

/*MC
  PetscDeviceCalloc - Allocate zeroed device-aware memory

  Synopsis:
  #include <petscdevice.h>
  PetscErrorCode PetscDeviceCalloc(PetscDeviceContext dctx, PetscMemType mtype, size_t n, size_t alignment, Type **ptr)

  Not Collective, Asynchronous, Auto-dependency aware

  Input Parameters:
+ dctx      - The `PetscDeviceContext` used to allocate the memory, or `PetscDeviceContextDefault`
. mtype     - The type of memory to allocate
. n         - The amount (in elements) to allocate
- alignment - The requested alignment in bytes, a positive power of two or `PETSC_DECIDE`

  Output Parameter:
. ptr - The pointer to store the result in

  Level: beginner

  Notes:
  Has the same allocation, alignment, fallback, and synchronization rules as `PetscDeviceMalloc()`,
  and additionally queues zero-initialization on `dctx`. Zero-initialization may still be pending
  when this routine returns. The ordinary host fallback uses `PetscCalloc1()`.

.N ASYNC_API

.seealso: `PetscDeviceFree()`, `PetscDeviceMalloc()`, `PetscDeviceArrayCopy()`,
`PetscDeviceArrayZero()`
M*/
#define PetscDeviceCalloc(dctx, mtype, n, alignment, ptr) PetscDeviceAllocate_Private((dctx), PETSC_TRUE, (mtype), (size_t)(n) * sizeof(**(ptr)), (size_t)(alignment), PETSC_DEVICE_ALIGNOF(**(ptr)), (void **)(ptr))

/*MC
  PetscDeviceFree - Free device-aware memory obtained with  `PetscDeviceMalloc()` or `PetscDeviceCalloc()`

  Synopsis:
  #include <petscdevice.h>
  PetscErrorCode PetscDeviceFree(PetscDeviceContext dctx, void *ptr)

  Not Collective, Asynchronous, Auto-dependency aware

  Input Parameters:
+ dctx - The `PetscDeviceContext` used to free the memory
- ptr  - The pointer to free, may be `NULL`

  Level: beginner

  Notes:
  `ptr` is set to `PETSC_NULLPTR` on successful deallocation.

  `ptr` must have been allocated using `PetscDeviceMalloc()`, `PetscDeviceCalloc()` not `PetscMalloc()` or related routines

  This routine falls back to using `PetscFree()` if PETSc was not configured with device
  support. The user should note that `PetscFree()` frees only host memory.

  DAG representation:
.vb
  time ->

  -> dctx -/- |= CALL =| - dctx ->
  -> ptr -/
.ve

.N ASYNC_API

.seealso: `PetscDeviceMalloc()`, `PetscDeviceCalloc()`
M*/
#define PetscDeviceFree(dctx, ptr) ((PetscErrorCode)(PetscDeviceDeallocate_Private((dctx), (ptr)) || ((ptr) = PETSC_NULLPTR, PETSC_SUCCESS)))

/*MC
  PetscDeviceArrayCopy - Copy memory in a device-aware manner

  Synopsis:
  #include <petscdevice.h>
  PetscErrorCode PetscDeviceArrayCopy(PetscDeviceContext dctx, void *dest, const void *src, size_t n)

  Not Collective, Asynchronous, Auto-dependency aware

  Input Parameters:
+ dctx - The `PetscDeviceContext` used to copy the memory
. dest - The pointer to copy to
. src  - The pointer to copy from
- n    - The amount (in elements) to copy

  Notes:
  Both `dest` and `src` must have been allocated using `PetscDeviceMalloc()` or
  `PetscDeviceCalloc()`.

  This uses the `sizeof()` of the `src` memory type requested to determine the total memory to
  be copied, therefore you should not multiply the number of elements by the `sizeof()` the
  type\:

.vb
  PetscInt *to,*from;

  // correct
  PetscDeviceArrayCopy(dctx,to,from,n);

  // incorrect
  PetscDeviceArrayCopy(dctx,to,from,n*sizeof(*from));
.ve

  See `PetscDeviceMemcpy()` for further discussion.

  Level: beginner

.N ASYNC_API

.seealso: `PetscDeviceMalloc()`, `PetscDeviceCalloc()`, `PetscDeviceFree()`,
`PetscDeviceArrayZero()`, `PetscDeviceMemcpy()`
M*/
#define PetscDeviceArrayCopy(dctx, dest, src, n) PetscDeviceMemcpy((dctx), (dest), (src), (size_t)(n) * sizeof(*(src)))

/*MC
  PetscDeviceArrayZero - Zero memory in a device-aware manner

  Synopsis:
  #include <petscdevice.h>
  PetscErrorCode PetscDeviceArrayZero(PetscDeviceContext dctx, void *ptr, size_t n)

  Not Collective, Asynchronous, Auto-dependency aware

  Input Parameters:
+ dctx  - The `PetscDeviceContext` used to zero the memory
. ptr   - The pointer to the memory
- n     - The amount (in elements) to zero

  Level: beginner

  Notes:
  `ptr` must have been allocated using `PetscDeviceMalloc()` or `PetscDeviceCalloc()`.

  This uses the `sizeof()` of the memory type requested to determine the total memory to be
  zeroed, therefore you should not multiply the number of elements by the `sizeof()` the type\:

.vb
  PetscInt *ptr;

  // correct
  PetscDeviceArrayZero(dctx,ptr,n);

  // incorrect
  PetscDeviceArrayZero(dctx,ptr,n*sizeof(*ptr));
.ve

  See `PetscDeviceMemset()` for further discussion.

.N ASYNC_API

.seealso: `PetscDeviceMalloc()`, `PetscDeviceCalloc()`, `PetscDeviceFree()`,
`PetscDeviceArrayCopy()`, `PetscDeviceMemset()`
M*/
#define PetscDeviceArrayZero(dctx, ptr, n) PetscDeviceMemset((dctx), (ptr), 0, (size_t)(n) * sizeof(*(ptr)))
