#ifndef PETSCDEVICE_H
#define PETSCDEVICE_H

#include <petscdevicetypes.h>
#include <petscviewertypes.h>

#if PETSC_CPP_VERSION >= 11 // C++11
  #define PETSC_DEVICE_ALIGNOF(...) alignof(decltype(__VA_ARGS__))
#elif PETSC_C_VERSION >= 11 // C11
  #ifdef __GNUC__
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
  #define PetscDeviceCreate(PetscDeviceType, PetscInt, dev) (*(dev) = PETSC_NULLPTR, PETSC_SUCCESS)
  #define PetscDeviceDestroy(dev)                           (*(dev) = PETSC_NULLPTR, PETSC_SUCCESS)
  #define PetscDeviceConfigure(PetscDevice)                 PETSC_SUCCESS
  #define PetscDeviceView(PetscDevice, PetscViewer)         PETSC_SUCCESS
  #define PetscDeviceGetType(PetscDevice, type)             (*(type) = PETSC_DEVICE_DEFAULT(), PETSC_SUCCESS)
  #define PetscDeviceGetDeviceId(PetscDevice, id)           (*(id) = 0, PETSC_SUCCESS)
  #define PETSC_DEVICE_DEFAULT()                            PETSC_DEVICE_HOST
  #define PetscDeviceSetDefaultDeviceType(PetscDeviceType)  PETSC_SUCCESS
  #define PetscDeviceInitialize(PetscDeviceType)            PETSC_SUCCESS
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
PETSC_EXTERN PetscErrorCode PetscDeviceContextViewFromOptions(PetscDeviceContext, PetscObject, const char name[]);
PETSC_EXTERN PetscErrorCode PetscDeviceContextGetCurrentContext(PetscDeviceContext *);
PETSC_EXTERN PetscErrorCode PetscDeviceContextSetCurrentContext(PetscDeviceContext);
#else
  #define PetscDeviceContextCreate(dctx)                                                                            (*(dctx) = PETSC_NULLPTR, PETSC_SUCCESS)
  #define PetscDeviceContextDestroy(dctx)                                                                           (*(dctx) = PETSC_NULLPTR, PETSC_SUCCESS)
  #define PetscDeviceContextSetStreamType(PetscDeviceContext, PetscStreamType)                                      PETSC_SUCCESS
  #define PetscDeviceContextGetStreamType(PetscDeviceContext, type)                                                 (*(type) = PETSC_STREAM_GLOBAL_BLOCKING, PETSC_SUCCESS)
  #define PetscDeviceContextSetDevice(PetscDeviceContext, PetscDevice)                                              PETSC_SUCCESS
  #define PetscDeviceContextGetDevice(PetscDeviceContext, device)                                                   (*(device) = PETSC_NULLPTR, PETSC_SUCCESS)
  #define PetscDeviceContextGetDeviceType(PetscDeviceContext, type)                                                 (*(type) = PETSC_DEVICE_DEFAULT(), PETSC_SUCCESS)
  #define PetscDeviceContextSetUp(PetscDeviceContext)                                                               PETSC_SUCCESS
  #define PetscDeviceContextDuplicate(PetscDeviceContextl, PetscDeviceContextr)                                     (*(PetscDeviceContextr) = PETSC_NULLPTR, PETSC_SUCCESS)
  #define PetscDeviceContextQueryIdle(PetscDeviceContext, idle)                                                     (*(idle) = PETSC_TRUE, PETSC_SUCCESS)
  #define PetscDeviceContextWaitForContext(PetscDeviceContextl, PetscDeviceContextr)                                PETSC_SUCCESS
  #define PetscDeviceContextForkWithStreamType(PetscDeviceContextp, PetscStreamType, PetscInt, PetscDeviceContextc) (*(PetscDeviceContextc) = PETSC_NULLPTR, PETSC_SUCCESS)
  #define PetscDeviceContextFork(PetscDeviceContextp, PetscInt, PetscDeviceContextc)                                (*(PetscDeviceContextc) = PETSC_NULLPTR, PETSC_SUCCESS)
  #define PetscDeviceContextJoin(PetscDeviceContextp, PetscInt, PetscDeviceContextJoinMode, PetscDeviceContextc)    (*(PetscDeviceContextc) = PETSC_NULLPTR, PETSC_SUCCESS)
  #define PetscDeviceContextSynchronize(PetscDeviceContext)                                                         PETSC_SUCCESS
  #define PetscDeviceContextSetFromOptions(MPI_Comm, PetscDeviceContext)                                            PETSC_SUCCESS
  #define PetscDeviceContextView(PetscDeviceContext, PetscViewer)                                                   PETSC_SUCCESS
  #define PetscDeviceContextViewFromOptions(PetscDeviceContext, PetscObject, PetscViewer)                           PETSC_SUCCESS
  #define PetscDeviceContextGetCurrentContext(dctx)                                                                 (*(dctx) = PETSC_NULLPTR, PETSC_SUCCESS)
  #define PetscDeviceContextSetCurrentContext(PetscDeviceContext)                                                   PETSC_SUCCESS
#endif /* PetscDefined(HAVE_CXX) */

/* memory */
#if PetscDefined(HAVE_CXX)
PETSC_EXTERN PetscErrorCode PetscDeviceAllocate_Private(PetscDeviceContext, PetscBool, PetscMemType, size_t, size_t, void **PETSC_RESTRICT);
PETSC_EXTERN PetscErrorCode PetscDeviceDeallocate_Private(PetscDeviceContext, void *PETSC_RESTRICT);
PETSC_EXTERN PetscErrorCode PetscDeviceMemcpy(PetscDeviceContext, void *PETSC_RESTRICT, const void *PETSC_RESTRICT, size_t);
PETSC_EXTERN PetscErrorCode PetscDeviceMemset(PetscDeviceContext, void *PETSC_RESTRICT, PetscInt, size_t);
#else
  #include <string.h> // memset()
  #define PetscDeviceAllocate_Private(PetscDeviceContext, clear, PetscMemType, size, alignment, ptr) PetscMallocA(1, (clear), __LINE__, PETSC_FUNCTION_NAME, __FILE__, (size), (ptr))
  #define PetscDeviceDeallocate_Private(PetscDeviceContext, ptr)                                     PetscFree((ptr))
  #define PetscDeviceMemcpy(PetscDeviceContext, dest, src, size)                                     PetscMemcpy((dest), (src), (size))
  #define PetscDeviceMemset(PetscDeviceContext, ptr, v, size)                                        ((void)memset((ptr), (unsigned char)(v), (size)), PETSC_SUCCESS)
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
  Memory allocated with this function must be freed with `PetscDeviceFree()`.

  If `n` is zero, then `ptr` is set to `PETSC_NULLPTR`.

  This routine falls back to using `PetscMalloc1()` if PETSc was not configured with device
  support. The user should note that `mtype` is ignored in this case, as `PetscMalloc1()`
  allocates only host memory.

  This routine uses the `sizeof()` of the memory type requested to determine the total memory
  to be allocated, therefore you should not multiply the number of elements requested by the
  `sizeof()` the type\:

.vb
  PetscInt *arr;

  // correct
  PetscDeviceMalloc(dctx,PETSC_MEMTYPE_DEVICE,n,&arr);

  // incorrect
  PetscDeviceMalloc(dctx,PETSC_MEMTYPE_DEVICE,n*sizeof(*arr),&arr);
.ve

  Note result stored `ptr` is immediately valid and the user may freely inspect or manipulate
  its value on function return, i.e.\:

.vb
  PetscInt *ptr;

  PetscDeviceMalloc(dctx, PETSC_MEMTYPE_DEVICE, 20, &ptr);

  PetscInt *sub_ptr = ptr + 10; // OK, no need to synchronize

  ptr[0] = 10; // ERROR, directly accessing contents of ptr is undefined until synchronization
.ve

  DAG representation:
.vb
  time ->

  -> dctx - |= CALL =| -\- dctx -->
                         \- ptr ->
.ve

  Level: beginner

.N ASYNC_API

.seealso: `PetscDeviceFree()`, `PetscDeviceCalloc()`, `PetscDeviceArrayCopy()`,
`PetscDeviceArrayZero()`
M*/
#define PetscDeviceMalloc(dctx, mtype, n, ptr) PetscDeviceAllocate_Private((dctx), PETSC_FALSE, (mtype), (size_t)(n) * sizeof(**(ptr)), PETSC_DEVICE_ALIGNOF(**(ptr)), (void **)(ptr))

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
`PetscDeviceArrayZero()`
M*/
#define PetscDeviceCalloc(dctx, mtype, n, ptr) PetscDeviceAllocate_Private((dctx), PETSC_TRUE, (mtype), (size_t)(n) * sizeof(**(ptr)), PETSC_DEVICE_ALIGNOF(**(ptr)), (void **)(ptr))

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
  `ptr` may be `NULL`, and is set to `PETSC_NULLPTR` on successful deallocation.

  `ptr` must have been allocated using `PetscDeviceMalloc()`, `PetscDeviceCalloc()`.

  This routine falls back to using `PetscFree()` if PETSc was not configured with device
  support. The user should note that `PetscFree()` frees only host memory.

  DAG representation:
.vb
  time ->

  -> dctx -/- |= CALL =| - dctx ->
  -> ptr -/
.ve

  Level: beginner

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
  Both `dest` and `src` must have been allocated using any of `PetscDeviceMalloc()`,
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

  Level: beginner

.N ASYNC_API

.seealso: `PetscDeviceMalloc()`, `PetscDeviceCalloc()`, `PetscDeviceFree()`,
`PetscDeviceArrayCopy()`, `PetscDeviceMemset()`
M*/
#define PetscDeviceArrayZero(dctx, ptr, n) PetscDeviceMemset((dctx), (ptr), 0, (size_t)(n) * sizeof(*(ptr)))

#endif /* PETSCDEVICE_H */
