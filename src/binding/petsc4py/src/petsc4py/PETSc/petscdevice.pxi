# --------------------------------------------------------------------

cdef extern from * nogil:

    ctypedef enum PetscOffloadMask:
        PETSC_OFFLOAD_UNALLOCATED
        PETSC_OFFLOAD_CPU
        PETSC_OFFLOAD_GPU
        PETSC_OFFLOAD_BOTH
        PETSC_OFFLOAD_KOKKOS

    ctypedef enum PetscMemType:
        PETSC_MEMTYPE_HOST
        PETSC_MEMTYPE_CUDA
        PETSC_MEMTYPE_HIP
        PETSC_MEMTYPE_SYCL

    ctypedef enum PetscDeviceType:
        PETSC_DEVICE_HOST
        PETSC_DEVICE_CUDA
        PETSC_DEVICE_HIP
        PETSC_DEVICE_SYCL

    ctypedef enum PetscStreamType:
        PETSC_STREAM_DEFAULT
        PETSC_STREAM_NONBLOCKING
        PETSC_STREAM_DEFAULT_WITH_BARRIER
        PETSC_STREAM_NONBLOCKING_WITH_BARRIER

    ctypedef enum PetscDeviceContextJoinMode:
        PETSC_DEVICE_CONTEXT_JOIN_DESTROY
        PETSC_DEVICE_CONTEXT_JOIN_SYNC
        PETSC_DEVICE_CONTEXT_JOIN_NO_SYNC

    PetscErrorCode PetscDeviceCreate(PetscDeviceType, PetscInt, PetscDevice *)
    PetscErrorCode PetscDeviceDestroy(PetscDevice *)
    PetscErrorCode PetscDeviceConfigure(PetscDevice)
    PetscErrorCode PetscDeviceView(PetscDevice, PetscViewer)
    PetscErrorCode PetscDeviceGetType(PetscDevice, PetscDeviceType *)
    PetscErrorCode PetscDeviceGetDeviceId(PetscDevice, PetscInt *)
    PetscDeviceType PETSC_DEVICE_DEFAULT()
    PetscErrorCode PetscDeviceSetDefaultDeviceType(PetscDeviceType)
    PetscErrorCode PetscDeviceInitialize(PetscDeviceType)
    PetscBool PetscDeviceInitialized(PetscDeviceType)

    PetscErrorCode PetscDeviceContextCreate(PetscDeviceContext *)
    PetscErrorCode PetscDeviceContextDestroy(PetscDeviceContext *)
    PetscErrorCode PetscDeviceContextSetStreamType(PetscDeviceContext, PetscStreamType)
    PetscErrorCode PetscDeviceContextGetStreamType(PetscDeviceContext, PetscStreamType *)
    PetscErrorCode PetscDeviceContextSetDevice(PetscDeviceContext, PetscDevice)
    PetscErrorCode PetscDeviceContextGetDevice(PetscDeviceContext, PetscDevice *)
    PetscErrorCode PetscDeviceContextGetDeviceType(PetscDeviceContext, PetscDeviceType *)
    PetscErrorCode PetscDeviceContextSetUp(PetscDeviceContext)
    PetscErrorCode PetscDeviceContextDuplicate(PetscDeviceContext, PetscDeviceContext *)
    PetscErrorCode PetscDeviceContextQueryIdle(PetscDeviceContext, PetscBool *)
    PetscErrorCode PetscDeviceContextWaitForContext(PetscDeviceContext, PetscDeviceContext)
    PetscErrorCode PetscDeviceContextForkWithStreamType(PetscDeviceContext, PetscStreamType, PetscInt, PetscDeviceContext **)
    PetscErrorCode PetscDeviceContextFork(PetscDeviceContext, PetscInt, PetscDeviceContext **)
    PetscErrorCode PetscDeviceContextJoin(PetscDeviceContext, PetscInt, PetscDeviceContextJoinMode, PetscDeviceContext **)
    PetscErrorCode PetscDeviceContextSynchronize(PetscDeviceContext)
    PetscErrorCode PetscDeviceContextSetFromOptions(MPI_Comm, PetscDeviceContext)
    PetscErrorCode PetscDeviceContextView(PetscDeviceContext, PetscViewer)
    PetscErrorCode PetscDeviceContextViewFromOptions(PetscDeviceContext, PetscObject, const char name[])
    PetscErrorCode PetscDeviceContextGetDefaultContext(PetscDeviceContext *)
    PetscErrorCode PetscDeviceContextGetCurrentContext(PetscDeviceContext *)
    PetscErrorCode PetscDeviceContextSetCurrentContext(PetscDeviceContext)

cdef extern from * nogil: # custom.h
    PetscErrorCode PetscDeviceReference(PetscDevice)
