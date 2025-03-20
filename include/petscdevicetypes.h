#pragma once

#include <petscsys.h> /*I <petscdevicetypes.h> I*/

// Some overzealous older gcc versions warn that the comparisons below are always true. Neat
// that it can detect this, but the tautology *is* the point of the static_assert()!
#if defined(__GNUC__) && __GNUC__ >= 6 && !PetscDefined(HAVE_WINDOWS_COMPILERS)
  #define PETSC_SHOULD_SILENCE_GCC_TAUTOLOGICAL_COMPARE_WARNING 1
#else
  #define PETSC_SHOULD_SILENCE_GCC_TAUTOLOGICAL_COMPARE_WARNING 0
#endif

/* SUBMANSEC = Sys */

/*E
  PetscMemType - Memory type of a pointer

  Level: intermediate

  Note:
  `PETSC_MEMTYPE_KOKKOS` depends on the Kokkos backend configuration

  Developer Notes:
  This enum uses a function (`PetscMemTypeToString()`) to convert to string representation so
  cannot be used in `PetscOptionsEnum()`.

  Encoding of the bitmask in binary\: xxxxyyyz
.vb
 z = 0                - Host memory
 z = 1                - Device memory
 yyy = 000            - CUDA-related memory
 yyy = 001            - HIP-related memory
 yyy = 010            - SYCL-related memory
 xxxxyyy1 = 0000,0001 - CUDA memory
 xxxxyyy1 = 0001,0001 - CUDA NVSHMEM memory
 xxxxyyy1 = 0000,0011 - HIP memory
 xxxxyyy1 = 0000,0101 - SYCL memory
.ve

  Other types of memory, e.g., CUDA managed memory, can be added when needed.

.seealso: `PetscMemTypeToString()`, `VecGetArrayAndMemType()`,
`PetscSFBcastWithMemTypeBegin()`, `PetscSFReduceWithMemTypeBegin()`
E*/
typedef enum {
  PETSC_MEMTYPE_HOST    = 0,
  PETSC_MEMTYPE_DEVICE  = 1,  /* 0x01 */
  PETSC_MEMTYPE_CUDA    = 1,  /* 0x01 */
  PETSC_MEMTYPE_NVSHMEM = 17, /* 0x11 */
  PETSC_MEMTYPE_HIP     = 3,  /* 0x03 */
  PETSC_MEMTYPE_SYCL    = 5   /* 0x05 */
} PetscMemType;
#if PetscDefined(HAVE_CUDA)
  #define PETSC_MEMTYPE_KOKKOS PETSC_MEMTYPE_CUDA
#elif PetscDefined(HAVE_HIP)
  #define PETSC_MEMTYPE_KOKKOS PETSC_MEMTYPE_HIP
#elif PetscDefined(HAVE_SYCL)
  #define PETSC_MEMTYPE_KOKKOS PETSC_MEMTYPE_SYCL
#else
  #define PETSC_MEMTYPE_KOKKOS PETSC_MEMTYPE_HOST
#endif

#define PetscMemTypeHost(m)    (((m) & 0x1) == PETSC_MEMTYPE_HOST)
#define PetscMemTypeDevice(m)  (((m) & 0x1) == PETSC_MEMTYPE_DEVICE)
#define PetscMemTypeCUDA(m)    (((m) & 0xF) == PETSC_MEMTYPE_CUDA)
#define PetscMemTypeHIP(m)     (((m) & 0xF) == PETSC_MEMTYPE_HIP)
#define PetscMemTypeSYCL(m)    (((m) & 0xF) == PETSC_MEMTYPE_SYCL)
#define PetscMemTypeNVSHMEM(m) ((m) == PETSC_MEMTYPE_NVSHMEM)

#if defined(__cplusplus)
  #if PETSC_SHOULD_SILENCE_GCC_TAUTOLOGICAL_COMPARE_WARNING
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wtautological-compare"
  #endif
static_assert(PetscMemTypeHost(PETSC_MEMTYPE_HOST), "");
static_assert(!PetscMemTypeHost(PETSC_MEMTYPE_DEVICE), "");
static_assert(!PetscMemTypeHost(PETSC_MEMTYPE_CUDA), "");
static_assert(!PetscMemTypeHost(PETSC_MEMTYPE_HIP), "");
static_assert(!PetscMemTypeHost(PETSC_MEMTYPE_SYCL), "");
static_assert(!PetscMemTypeHost(PETSC_MEMTYPE_NVSHMEM), "");

static_assert(!PetscMemTypeDevice(PETSC_MEMTYPE_HOST), "");
static_assert(PetscMemTypeDevice(PETSC_MEMTYPE_DEVICE), "");
static_assert(PetscMemTypeDevice(PETSC_MEMTYPE_CUDA), "");
static_assert(PetscMemTypeDevice(PETSC_MEMTYPE_HIP), "");
static_assert(PetscMemTypeDevice(PETSC_MEMTYPE_SYCL), "");
static_assert(PetscMemTypeDevice(PETSC_MEMTYPE_NVSHMEM), "");

static_assert(PetscMemTypeCUDA(PETSC_MEMTYPE_CUDA), "");
static_assert(PetscMemTypeCUDA(PETSC_MEMTYPE_NVSHMEM), "");
  #if PETSC_SHOULD_SILENCE_GCC_TAUTOLOGICAL_COMPARE_WARNING
    #pragma GCC diagnostic pop
  #endif
#endif // __cplusplus

PETSC_NODISCARD static inline PETSC_CONSTEXPR_14 const char *PetscMemTypeToString(PetscMemType mtype)
{
#ifdef __cplusplus
  static_assert(PETSC_MEMTYPE_CUDA == PETSC_MEMTYPE_DEVICE, "");
#endif
#define PETSC_CASE_NAME(v) \
  case v: \
    return PetscStringize(v)

  switch (mtype) {
    PETSC_CASE_NAME(PETSC_MEMTYPE_HOST);
    /* PETSC_CASE_NAME(PETSC_MEMTYPE_DEVICE); same as PETSC_MEMTYPE_CUDA */
    PETSC_CASE_NAME(PETSC_MEMTYPE_CUDA);
    PETSC_CASE_NAME(PETSC_MEMTYPE_NVSHMEM);
    PETSC_CASE_NAME(PETSC_MEMTYPE_HIP);
    PETSC_CASE_NAME(PETSC_MEMTYPE_SYCL);
  }
  PetscUnreachable();
  return "invalid";
#undef PETSC_CASE_NAME
}

#define PETSC_OFFLOAD_VECKOKKOS_DEPRECATED PETSC_OFFLOAD_VECKOKKOS PETSC_DEPRECATED_ENUM(3, 17, 0, "PETSC_OFFLOAD_KOKKOS", )

/*E
  PetscOffloadMask - indicates which memory (CPU, GPU, or none) contains valid data

  Values:
+ `PETSC_OFFLOAD_UNALLOCATED` - no memory contains valid matrix entries; NEVER used for vectors
. `PETSC_OFFLOAD_GPU`         - GPU has valid vector/matrix entries
. `PETSC_OFFLOAD_CPU`         - CPU has valid vector/matrix entries
. `PETSC_OFFLOAD_BOTH`        - Both GPU and CPU have valid vector/matrix entries and they match
- `PETSC_OFFLOAD_KOKKOS`      - Reserved for Kokkos matrix and vector. It means the offload is managed by Kokkos, thus this flag itself cannot tell you where the valid data is.

  Level: developer

  Developer Note:
  This enum uses a function (`PetscOffloadMaskToString()`) to convert to string representation so
  cannot be used in `PetscOptionsEnum()`.

.seealso: `PetscOffloadMaskToString()`, `PetscOffloadMaskToMemType()`, `PetscOffloadMaskToDeviceCopyMode()`
E*/
typedef enum {
  PETSC_OFFLOAD_UNALLOCATED          = 0,   /* 0x0 */
  PETSC_OFFLOAD_CPU                  = 1,   /* 0x1 */
  PETSC_OFFLOAD_GPU                  = 2,   /* 0x2 */
  PETSC_OFFLOAD_BOTH                 = 3,   /* 0x3 */
  PETSC_OFFLOAD_VECKOKKOS_DEPRECATED = 256, /* 0x100 */
  PETSC_OFFLOAD_KOKKOS               = 256  /* 0x100 */
} PetscOffloadMask;

#define PetscOffloadUnallocated(m) ((m) == PETSC_OFFLOAD_UNALLOCATED)
#define PetscOffloadHost(m)        (((m) & PETSC_OFFLOAD_CPU) == PETSC_OFFLOAD_CPU)
#define PetscOffloadDevice(m)      (((m) & PETSC_OFFLOAD_GPU) == PETSC_OFFLOAD_GPU)
#define PetscOffloadBoth(m)        ((m) == PETSC_OFFLOAD_BOTH)

#if defined(__cplusplus)
  #if PETSC_SHOULD_SILENCE_GCC_TAUTOLOGICAL_COMPARE_WARNING
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wtautological-compare"
  #endif
static_assert(!PetscOffloadHost(PETSC_OFFLOAD_UNALLOCATED), "");
static_assert(PetscOffloadHost(PETSC_OFFLOAD_BOTH), "");
static_assert(!PetscOffloadHost(PETSC_OFFLOAD_GPU), "");
static_assert(PetscOffloadHost(PETSC_OFFLOAD_BOTH), "");
static_assert(!PetscOffloadHost(PETSC_OFFLOAD_KOKKOS), "");

static_assert(!PetscOffloadDevice(PETSC_OFFLOAD_UNALLOCATED), "");
static_assert(!PetscOffloadDevice(PETSC_OFFLOAD_CPU), "");
static_assert(PetscOffloadDevice(PETSC_OFFLOAD_GPU), "");
static_assert(PetscOffloadDevice(PETSC_OFFLOAD_BOTH), "");
static_assert(!PetscOffloadDevice(PETSC_OFFLOAD_KOKKOS), "");

static_assert(PetscOffloadBoth(PETSC_OFFLOAD_BOTH), "");
static_assert(!PetscOffloadBoth(PETSC_OFFLOAD_CPU), "");
static_assert(!PetscOffloadBoth(PETSC_OFFLOAD_GPU), "");
static_assert(!PetscOffloadBoth(PETSC_OFFLOAD_GPU), "");
static_assert(!PetscOffloadBoth(PETSC_OFFLOAD_KOKKOS), "");
  #if PETSC_SHOULD_SILENCE_GCC_TAUTOLOGICAL_COMPARE_WARNING
    #pragma GCC diagnostic pop
  #endif
#endif // __cplusplus

PETSC_NODISCARD static inline PETSC_CONSTEXPR_14 const char *PetscOffloadMaskToString(PetscOffloadMask mask)
{
#define PETSC_CASE_RETURN(v) \
  case v: \
    return PetscStringize(v)

  switch (mask) {
    PETSC_CASE_RETURN(PETSC_OFFLOAD_UNALLOCATED);
    PETSC_CASE_RETURN(PETSC_OFFLOAD_CPU);
    PETSC_CASE_RETURN(PETSC_OFFLOAD_GPU);
    PETSC_CASE_RETURN(PETSC_OFFLOAD_BOTH);
    PETSC_CASE_RETURN(PETSC_OFFLOAD_KOKKOS);
  }
  PetscUnreachable();
  return "invalid";
#undef PETSC_CASE_RETURN
}

PETSC_NODISCARD static inline PETSC_CONSTEXPR_14 PetscMemType PetscOffloadMaskToMemType(PetscOffloadMask mask)
{
  switch (mask) {
  case PETSC_OFFLOAD_UNALLOCATED:
  case PETSC_OFFLOAD_CPU:
    return PETSC_MEMTYPE_HOST;
  case PETSC_OFFLOAD_GPU:
  case PETSC_OFFLOAD_BOTH:
    return PETSC_MEMTYPE_DEVICE;
  case PETSC_OFFLOAD_KOKKOS:
    return PETSC_MEMTYPE_KOKKOS;
  }
  PetscUnreachable();
  return PETSC_MEMTYPE_HOST;
}

/*E
  PetscDeviceInitType - Initialization strategy for `PetscDevice`

  Values:
+ `PETSC_DEVICE_INIT_NONE`  - PetscDevice is never initialized
. `PETSC_DEVICE_INIT_LAZY`  - PetscDevice is initialized on demand
- `PETSC_DEVICE_INIT_EAGER` - PetscDevice is initialized as soon as possible

  Level: beginner

  Note:
  `PETSC_DEVICE_INIT_NONE` implies that any initialization of `PetscDevice` is disallowed and
  doing so results in an error. Useful to ensure that no accelerator is used in a program.

.seealso: `PetscDevice`, `PetscDeviceType`, `PetscDeviceInitialize()`,
`PetscDeviceInitialized()`, `PetscDeviceCreate()`
E*/
typedef enum {
  PETSC_DEVICE_INIT_NONE,
  PETSC_DEVICE_INIT_LAZY,
  PETSC_DEVICE_INIT_EAGER
} PetscDeviceInitType;
PETSC_EXTERN const char *const PetscDeviceInitTypes[];

/*E
  PetscDeviceType - Kind of accelerator device backend

  Values:
+ `PETSC_DEVICE_HOST` - Host, no accelerator backend found
. `PETSC_DEVICE_CUDA` - CUDA enabled GPU
. `PETSC_DEVICE_HIP`  - ROCM/HIP enabled GPU
. `PETSC_DEVICE_SYCL` - SYCL enabled device
- `PETSC_DEVICE_MAX`  - Always 1 greater than the largest valid `PetscDeviceType`, invalid type, do not use

  Level: beginner

  Note:
  One can also use the `PETSC_DEVICE_DEFAULT()` routine to get the current default `PetscDeviceType`.

.seealso: `PetscDevice`, `PetscDeviceInitType`, `PetscDeviceCreate()`, `PETSC_DEVICE_DEFAULT()`
E*/
typedef enum {
  PETSC_DEVICE_HOST,
  PETSC_DEVICE_CUDA,
  PETSC_DEVICE_HIP,
  PETSC_DEVICE_SYCL,
  PETSC_DEVICE_MAX
} PetscDeviceType;
PETSC_EXTERN const char *const PetscDeviceTypes[];

/*E
  PetscDeviceAttribute - Attribute detailing a property or feature of a `PetscDevice`

  Values:
+ `PETSC_DEVICE_ATTR_SIZE_T_SHARED_MEM_PER_BLOCK` - The maximum amount of shared memory per block in a device kernel
- `PETSC_DEVICE_ATTR_MAX`                         - Invalid attribute, do not use

  Level: beginner

.seealso: `PetscDevice`, `PetscDeviceGetAttribute()`
E*/
typedef enum {
  PETSC_DEVICE_ATTR_SIZE_T_SHARED_MEM_PER_BLOCK,
  PETSC_DEVICE_ATTR_MAX
} PetscDeviceAttribute;
PETSC_EXTERN const char *const PetscDeviceAttributes[];

/*S
  PetscDevice - Object to manage an accelerator "device" (usually a GPU)

  Level: beginner

  Note:
  This object is used to house configuration and state of a device, but does not offer any
  ability to interact with or drive device computation. This functionality is facilitated
  instead by the `PetscDeviceContext` object.

.seealso: `PetscDeviceType`, `PetscDeviceInitType`, `PetscDeviceCreate()`,
`PetscDeviceConfigure()`, `PetscDeviceDestroy()`, `PetscDeviceContext`,
`PetscDeviceContextSetDevice()`, `PetscDeviceContextGetDevice()`, `PetscDeviceGetAttribute()`
S*/
typedef struct _n_PetscDevice *PetscDevice;

/*E
  PetscStreamType - indicates how a stream implementation will interact
  with other streams and if it blocks the host.

  Values:
+ `PETSC_STREAM_DEFAULT`                  - Same as the default stream in CUDA or HIP. Streams of this type may or may not synchronize implicitly with other streams. It does not block the host.
. `PETSC_STREAM_NONBLOCKING`              - Same as the nonblocking stream in CUDA or HIP. Streams of this type is truly asynchronous, and is blocked by nothing. It does not block the host. In CUDA, it is created with cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking).
. `PETSC_STREAM_DEFAULT_WITH_BARRIER`     - Same as the default stream in CUDA or HIP. PETSc async functions using this kind of stream will end with a stream synchronization. Stream of this type may or may not synchronize implicitly with other streams.
. `PETSC_STREAM_NONBLOCKING_WITH_BARRIER` - Same as the nonblocking stream in CUDA or HIP. PETSc async functions using this kind of stream will end with a stream synchronization. Streams of this type are truly asynchronous and are blocked by nothing.
- `PETSC_STREAM_MAX`                - Always 1 greater than the largest `PetscStreamType`, do not use

  Level: intermediate

  Note:
  The default stream, also known as the NULL stream or stream 0, can have two different behaviors: legacy behavior and per-thread behavior.
  The behavior is determined at compile time. By default, the legacy default stream is used.
  The legacy default stream implicitly synchronizes with per-thread default streams.
  The per-thread default stream, like nonblocking streams, does not synchronizes with other per-thread streams, but synchronize with the default stream.
  The per-thread default stream may be useful for running kernels launched from different threads concurrently on the same GPU when the Multi-Process Service is not available.
  To use the per-thread default stream, one can enable it by using the nvcc option "--default-stream per-thread" or the hipcc option "-fgpu-default-stream=per-thread", depending on the backend used.

.seealso: `PetscDeviceContextSetStreamType()`, `PetscDeviceContextGetStreamType()`
E*/
typedef enum {
  PETSC_STREAM_DEFAULT,
  PETSC_STREAM_NONBLOCKING,
  PETSC_STREAM_DEFAULT_WITH_BARRIER,
  PETSC_STREAM_NONBLOCKING_WITH_BARRIER,
  PETSC_STREAM_MAX
} PetscStreamType;
PETSC_EXTERN const char *const PetscStreamTypes[];

/*E
  PetscDeviceContextJoinMode - Describes the type of join operation to perform in
  `PetscDeviceContextJoin()`

  Values:
+ `PETSC_DEVICE_CONTEXT_JOIN_DESTROY` - Destroy all incoming sub-contexts after join.
. `PETSC_DEVICE_CONTEXT_JOIN_SYNC`    - Synchronize incoming sub-contexts after join.
- `PETSC_DEVICE_CONTEXT_JOIN_NO_SYNC` - Do not synchronize incoming sub-contexts after join.

  Level: beginner

.seealso: `PetscDeviceContext`, `PetscDeviceContextFork()`, `PetscDeviceContextJoin()`
E*/
typedef enum {
  PETSC_DEVICE_CONTEXT_JOIN_DESTROY,
  PETSC_DEVICE_CONTEXT_JOIN_SYNC,
  PETSC_DEVICE_CONTEXT_JOIN_NO_SYNC
} PetscDeviceContextJoinMode;
PETSC_EXTERN const char *const PetscDeviceContextJoinModes[];

/*S
  PetscDeviceContext - Container to manage stream dependencies and the various solver handles
  for asynchronous device compute.

  Level: beginner

.seealso: `PetscDevice`, `PetscDeviceContextCreate()`, `PetscDeviceContextSetDevice()`,
`PetscDeviceContextDestroy()`, `PetscDeviceContextFork()`, `PetscDeviceContextJoin()`
S*/
typedef struct _p_PetscDeviceContext *PetscDeviceContext;

/*E
  PetscDeviceCopyMode - Describes the copy direction of a device-aware `memcpy`

  Values:
+ `PETSC_DEVICE_COPY_HTOH` - Copy from host memory to host memory
. `PETSC_DEVICE_COPY_DTOH` - Copy from device memory to host memory
. `PETSC_DEVICE_COPY_HTOD` - Copy from host memory to device memory
. `PETSC_DEVICE_COPY_DTOD` - Copy from device memory to device memory
- `PETSC_DEVICE_COPY_AUTO` - Infer the copy direction from the pointers

  Level: beginner

.seealso: `PetscDeviceArrayCopy()`, `PetscDeviceMemcpy()`
E*/
typedef enum {
  PETSC_DEVICE_COPY_HTOH,
  PETSC_DEVICE_COPY_DTOH,
  PETSC_DEVICE_COPY_HTOD,
  PETSC_DEVICE_COPY_DTOD,
  PETSC_DEVICE_COPY_AUTO,
} PetscDeviceCopyMode;
PETSC_EXTERN const char *const PetscDeviceCopyModes[];

PETSC_NODISCARD static inline PetscDeviceCopyMode PetscOffloadMaskToDeviceCopyMode(PetscOffloadMask dest, PetscOffloadMask src)
{
  PetscDeviceCopyMode mode;

  PetscFunctionBegin;
  PetscAssertAbort(dest != PETSC_OFFLOAD_UNALLOCATED, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot copy to unallocated");
  PetscAssertAbort(src != PETSC_OFFLOAD_UNALLOCATED, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot copy from unallocated");

  if (PetscOffloadDevice(dest)) {
    mode = PetscOffloadHost(src) ? PETSC_DEVICE_COPY_HTOD : PETSC_DEVICE_COPY_DTOD;
  } else {
    mode = PetscOffloadHost(src) ? PETSC_DEVICE_COPY_HTOH : PETSC_DEVICE_COPY_DTOH;
  }
  PetscFunctionReturn(mode);
}

PETSC_NODISCARD static inline PETSC_CONSTEXPR_14 PetscDeviceCopyMode PetscMemTypeToDeviceCopyMode(PetscMemType dest, PetscMemType src)
{
  if (PetscMemTypeHost(dest)) {
    return PetscMemTypeHost(src) ? PETSC_DEVICE_COPY_HTOH : PETSC_DEVICE_COPY_DTOH;
  } else {
    return PetscMemTypeDevice(src) ? PETSC_DEVICE_COPY_DTOD : PETSC_DEVICE_COPY_HTOD;
  }
}

/*E
  PetscMemoryAccessMode - Describes the intended usage of a memory region

  Values:
+ `PETSC_MEMORY_ACCESS_READ`       - Read only
. `PETSC_MEMORY_ACCESS_WRITE`      - Write only
- `PETSC_MEMORY_ACCESS_READ_WRITE` - Read and write

  Level: beginner

  Notes:
  This `enum` is a bitmask with the following encoding (assuming 2 bit)\:

.vb
  PETSC_MEMORY_ACCESS_READ       = 0b01
  PETSC_MEMORY_ACCESS_WRITE      = 0b10
  PETSC_MEMORY_ACCESS_READ_WRITE = 0b11

  // consequently
  PETSC_MEMORY_ACCESS_READ | PETSC_MEMORY_ACCESS_WRITE = PETSC_MEMORY_ACCESS_READ_WRITE
.ve

  The following convenience macros are also provided\:

+ `PetscMemoryAccessRead(mode)` - `true` if `mode` is any kind of read, `false` otherwise
- `PetscMemoryAccessWrite(mode)` - `true` if `mode` is any kind of write, `false` otherwise

  Developer Note:
  This enum uses a function (`PetscMemoryAccessModeToString()`) to convert values to string
  representation, so cannot be used in `PetscOptionsEnum()`.

.seealso: `PetscMemoryAccessModeToString()`, `PetscDevice`, `PetscDeviceContext`
E*/
typedef enum {
  PETSC_MEMORY_ACCESS_READ       = 1, /* 01 */
  PETSC_MEMORY_ACCESS_WRITE      = 2, /* 10 */
  PETSC_MEMORY_ACCESS_READ_WRITE = 3  /* 11 */
} PetscMemoryAccessMode;

#define PetscMemoryAccessRead(m)  (((m) & PETSC_MEMORY_ACCESS_READ) == PETSC_MEMORY_ACCESS_READ)
#define PetscMemoryAccessWrite(m) (((m) & PETSC_MEMORY_ACCESS_WRITE) == PETSC_MEMORY_ACCESS_WRITE)

#if defined(__cplusplus)
  #if PETSC_SHOULD_SILENCE_GCC_TAUTOLOGICAL_COMPARE_WARNING
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wtautological-compare"
  #endif
static_assert(PetscMemoryAccessRead(PETSC_MEMORY_ACCESS_READ), "");
static_assert(PetscMemoryAccessRead(PETSC_MEMORY_ACCESS_READ_WRITE), "");
static_assert(!PetscMemoryAccessRead(PETSC_MEMORY_ACCESS_WRITE), "");
static_assert(PetscMemoryAccessWrite(PETSC_MEMORY_ACCESS_WRITE), "");
static_assert(PetscMemoryAccessWrite(PETSC_MEMORY_ACCESS_READ_WRITE), "");
static_assert(!PetscMemoryAccessWrite(PETSC_MEMORY_ACCESS_READ), "");
static_assert((PETSC_MEMORY_ACCESS_READ | PETSC_MEMORY_ACCESS_WRITE) == PETSC_MEMORY_ACCESS_READ_WRITE, "");
  #if PETSC_SHOULD_SILENCE_GCC_TAUTOLOGICAL_COMPARE_WARNING
    #pragma GCC diagnostic pop
  #endif
#endif

PETSC_NODISCARD static inline PETSC_CONSTEXPR_14 const char *PetscMemoryAccessModeToString(PetscMemoryAccessMode mode)
{
#define PETSC_CASE_RETURN(v) \
  case v: \
    return PetscStringize(v)

  switch (mode) {
    PETSC_CASE_RETURN(PETSC_MEMORY_ACCESS_READ);
    PETSC_CASE_RETURN(PETSC_MEMORY_ACCESS_WRITE);
    PETSC_CASE_RETURN(PETSC_MEMORY_ACCESS_READ_WRITE);
  }
  PetscUnreachable();
  return "invalid";
#undef PETSC_CASE_RETURN
}

#undef PETSC_SHOULD_SILENCE_GCC_TAUTOLOGICAL_COMPARE_WARNING
