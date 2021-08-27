#if !defined(PETSCDEVICETYPES_H)
#define PETSCDEVICETYPES_H

#include <petscsys.h> /* for PETSC_HAVE_CUDA/HIP/KOKKOS etc */

/*E
  PetscMemType - Memory type of a pointer

  Level: beginner

  Developer Note:
   Encoding of the bitmask in binary: xxxxyyyz
   z = 0:                Host memory
   z = 1:                Device memory
   yyy = 000:            CUDA-related memory
   yyy = 001:            HIP-related memory
   xxxxyyy1 = 0000,0001: CUDA memory
   xxxxyyy1 = 0001,0001: CUDA NVSHMEM memory
   xxxxyyy1 = 0000,0011: HIP memory

  Other types of memory, e.g., CUDA managed memory, can be added when needed.

.seealso: VecGetArrayAndMemType(), PetscSFBcastWithMemTypeBegin(), PetscSFReduceWithMemTypeBegin()
E*/
typedef enum {PETSC_MEMTYPE_HOST=0, PETSC_MEMTYPE_DEVICE=0x01, PETSC_MEMTYPE_CUDA=0x01, PETSC_MEMTYPE_NVSHMEM=0x11,PETSC_MEMTYPE_HIP=0x03} PetscMemType;

#define PetscMemTypeHost(m)    (((m) & 0x1) == PETSC_MEMTYPE_HOST)
#define PetscMemTypeDevice(m)  (((m) & 0x1) == PETSC_MEMTYPE_DEVICE)
#define PetscMemTypeCUDA(m)    (((m) & 0xF) == PETSC_MEMTYPE_CUDA)
#define PetscMemTypeHIP(m)     (((m) & 0xF) == PETSC_MEMTYPE_HIP)
#define PetscMemTypeNVSHMEM(m) ((m) == PETSC_MEMTYPE_NVSHMEM)

/*E
    PetscOffloadMask - indicates which memory (CPU, GPU, or none) contains valid data

   PETSC_OFFLOAD_UNALLOCATED  - no memory contains valid matrix entries; NEVER used for vectors
   PETSC_OFFLOAD_GPU - GPU has valid vector/matrix entries
   PETSC_OFFLOAD_CPU - CPU has valid vector/matrix entries
   PETSC_OFFLOAD_BOTH - Both GPU and CPU have valid vector/matrix entries and they match
   PETSC_OFFLOAD_VECKOKKOS - Reserved for Vec_Kokkos. The offload is managed by Kokkos, thus this flag is not used in Vec_Kokkos.

   Level: developer
E*/
typedef enum {PETSC_OFFLOAD_UNALLOCATED=0x0,PETSC_OFFLOAD_CPU=0x1,PETSC_OFFLOAD_GPU=0x2,PETSC_OFFLOAD_BOTH=0x3,PETSC_OFFLOAD_VECKOKKOS=0x100} PetscOffloadMask;

/*E
  PetscDeviceKind - Kind of accelerator device backend

$ PETSC_DEVICE_INVALID - Invalid type, do not use
$ PETSC_DEVICE_CUDA    - CUDA enabled GPU
$ PETSC_DEVICE_HIP     - ROCM/HIP enabled GPU
$ PETSC_DEVICE_DEFAULT - Automatically select backend based on availability
$ PETSC_DEVICE_MAX     - Always 1 greater than the largest valid PetscDeviceKInd, invalid type, do not use

  Notes:
  PETSC_DEVICE_DEFAULT is selected in the following order: PETSC_DEVICE_HIP, PETSC_DEVICE_CUDA, PETSC_DEVICE_INVALID.

  Level: beginner

.seealso: PetscDevice, PetscDeviceGetDevice()
E*/
typedef enum {
  PETSC_DEVICE_INVALID = 0,
  PETSC_DEVICE_CUDA    = 1,
  PETSC_DEVICE_HIP     = 2,
#if PetscDefined(HAVE_HIP)
  PETSC_DEVICE_DEFAULT = PETSC_DEVICE_HIP,
#elif PetscDefined(HAVE_CUDA)
  PETSC_DEVICE_DEFAULT = PETSC_DEVICE_CUDA,
#else
  PETSC_DEVICE_DEFAULT = PETSC_DEVICE_INVALID,
#endif
  PETSC_DEVICE_MAX     = 3
} PetscDeviceKind;
PETSC_EXTERN const char *const PetscDeviceKinds[];

/*S
  PetscDevice - Handle to an accelerator "device" (usually a GPU)

  Notes:
  This object is used to house configuration and state of a device, but does not offer any ability to interact with or
  drive device computation. This functionality is facilitated instead by the PetscDeviceContext object.

  Level: beginner

.seealso: PetscDeviceKind, PetscDeviceGetDevice(), PetscDeviceDestroy(), PetscDeviceDestroy(), PetscDeviceContext, PetscDeviceContextSetDevice()
S*/
typedef struct _n_PetscDevice *PetscDevice;

/*E
  PetscStreamType - Stream blocking mode, indicates how a stream implementation will interact with the default "NULL"
  stream, which is usually blocking.

$ PETSC_STREAM_GLOBAL_BLOCKING    - Alias for NULL stream. Any stream of this type will block the host for all other streams to finish work before starting its operations.
$ PETSC_STREAM_DEFAULT_BLOCKING   - Stream will act independent of other streams, but will still be blocked by actions on the NULL stream.
$ PETSC_STREAM_GLOBAL_NONBLOCKING - Stream is truly asynchronous, and is blocked by nothing, not even the NULL stream.
$ PETSC_STREAM_MAX                - Always 1 greater than the largest PetscStreamType, do not use

  Level: intermediate

.seealso: PetscDeviceContextSetStreamType(), PetscDeviceContextGetStreamType()
E*/
typedef enum {
  PETSC_STREAM_GLOBAL_BLOCKING    = 0,
  PETSC_STREAM_DEFAULT_BLOCKING   = 1,
  PETSC_STREAM_GLOBAL_NONBLOCKING = 2,
  PETSC_STREAM_MAX                = 3
} PetscStreamType;
PETSC_EXTERN const char *const PetscStreamTypes[];

/*E
  PetscDeviceContextJoinMode - Describes the type of join operation to perform in PetscDeviceContextJoin()

$ PETSC_DEVICE_CONTEXT_DESTROY - Destroy all incoming sub-contexts after join.
$ PETSC_CONTEXT_JOIN_SYNC      - Synchronize incoming sub-contexts after join.
$ PETSC_CONTEXT_JOIN_NO_SYNC   - Do not synchronize incoming sub-contexts after join.

  Level: beginner

.seealso: PetscDeviceContextFork(), PetscDeviceContextJoin()
E*/
typedef enum {
  PETSC_DEVICE_CONTEXT_JOIN_DESTROY,
  PETSC_DEVICE_CONTEXT_JOIN_SYNC,
  PETSC_DEVICE_CONTEXT_JOIN_NO_SYNC
} PetscDeviceContextJoinMode;
PETSC_EXTERN const char *const PetscDeviceContextJoinModes[];

/*S
  PetscDeviceContext - Container to manage stream dependencies and the various solver handles for asynchronous device compute.

  Level: beginner

.seealso: PetscDevice, PetscDeviceContextCreate(), PetscDeviceContextSetDevice(), PetscDeviceContextDestroy(),
PetscDeviceContextFork(), PetscDeviceContextJoin()
S*/
typedef struct _n_PetscDeviceContext *PetscDeviceContext;
#endif /* PETSCDEVICETYPES_H */
