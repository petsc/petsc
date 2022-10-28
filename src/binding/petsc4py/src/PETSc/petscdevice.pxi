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
        PETSC_STREAM_GLOBAL_BLOCKING
        PETSC_STREAM_DEFAULT_BLOCKING
        PETSC_STREAM_GLOBAL_NONBLOCKING

    ctypedef enum PetscDeviceContextJoinMode:
        PETSC_DEVICE_CONTEXT_JOIN_DESTROY
        PETSC_DEVICE_CONTEXT_JOIN_SYNC
        PETSC_DEVICE_CONTEXT_JOIN_NO_SYNC

    int PetscDeviceCreate(PetscDeviceType, PetscInt, PetscDevice *)
    int PetscDeviceDestroy(PetscDevice *)
    int PetscDeviceConfigure(PetscDevice)
    int PetscDeviceView(PetscDevice, PetscViewer)
    int PetscDeviceGetType(PetscDevice, PetscDeviceType *)
    int PetscDeviceGetDeviceId(PetscDevice, PetscInt *)
    PetscDeviceType PETSC_DEVICE_DEFAULT()
    int PetscDeviceSetDefaultDeviceType(PetscDeviceType)
    int PetscDeviceInitialize(PetscDeviceType)
    PetscBool PetscDeviceInitialized(PetscDeviceType)

    int PetscDeviceContextCreate(PetscDeviceContext *)
    int PetscDeviceContextDestroy(PetscDeviceContext *)
    int PetscDeviceContextSetStreamType(PetscDeviceContext, PetscStreamType)
    int PetscDeviceContextGetStreamType(PetscDeviceContext, PetscStreamType *)
    int PetscDeviceContextSetDevice(PetscDeviceContext, PetscDevice)
    int PetscDeviceContextGetDevice(PetscDeviceContext, PetscDevice *)
    int PetscDeviceContextGetDeviceType(PetscDeviceContext, PetscDeviceType *)
    int PetscDeviceContextSetUp(PetscDeviceContext)
    int PetscDeviceContextDuplicate(PetscDeviceContext, PetscDeviceContext *)
    int PetscDeviceContextQueryIdle(PetscDeviceContext, PetscBool *)
    int PetscDeviceContextWaitForContext(PetscDeviceContext, PetscDeviceContext)
    int PetscDeviceContextForkWithStreamType(PetscDeviceContext, PetscStreamType, PetscInt, PetscDeviceContext **)
    int PetscDeviceContextFork(PetscDeviceContext, PetscInt, PetscDeviceContext **)
    int PetscDeviceContextJoin(PetscDeviceContext, PetscInt, PetscDeviceContextJoinMode, PetscDeviceContext **)
    int PetscDeviceContextSynchronize(PetscDeviceContext)
    int PetscDeviceContextSetFromOptions(MPI_Comm, PetscDeviceContext)
    int PetscDeviceContextView(PetscDeviceContext, PetscViewer)
    int PetscDeviceContextViewFromOptions(PetscDeviceContext, PetscObject, const char name[])
    int PetscDeviceContextGetCurrentContext(PetscDeviceContext *)
    int PetscDeviceContextSetCurrentContext(PetscDeviceContext)

cdef inline PetscDeviceType asDeviceType(object dtype) except <PetscDeviceType>(-1):
  if isinstance(dtype, str):
    dtype = dtype.upper()
    try:
      return getattr(Device.Type, dtype)
    except AttributeError:
      raise ValueError("unknown device type: %s" % dtype)
  return dtype

cdef inline str toDeviceType(PetscDeviceType dtype):
  try:
    return Device.Type.__enum2str[dtype]
  except KeyError:
    raise NotImplementedError("unhandled PetscDeviceType %d" % <int>dtype)

cdef inline PetscStreamType asStreamType(object stype) except <PetscStreamType>(-1):
  if isinstance(stype, str):
    stype = stype.upper()
    try:
      return getattr(DeviceContext.StreamType, stype)
    except AttributeError:
      raise ValueError("unknown stream type: %s" % stype)
  return stype

cdef inline str toStreamType(PetscStreamType stype):
  try:
    return DeviceContext.StreamType.__enum2str[stype]
  except KeyError:
    raise NotImplementedError("unhandled PetscStreamType %d" % <int>stype)

cdef inline PetscDeviceContextJoinMode asJoinMode(object jmode) except <PetscDeviceContextJoinMode>(-1):
  if isinstance(jmode, str):
    jmode = jmode.upper()
    try:
      return getattr(DeviceContext.JoinMode, jmode)
    except AttributeError:
      raise ValueError("unknown join mode: %s" % jmode)
  return jmode

cdef inline str toJoinMode(PetscDeviceContextJoinMode jmode):
  try:
    return DeviceContext.JoinMode.__enum2str[jmode]
  except KeyError:
    raise NotImplementedError("unhandled PetscDeviceContextJoinMode %d" % <int>jmode)
