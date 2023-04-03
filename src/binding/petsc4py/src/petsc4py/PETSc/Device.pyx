# --------------------------------------------------------------------

class staticproperty(property):
  def __get__(self, *args, **kwargs):
    return self.fget.__get__(*args, **kwargs)()

cdef object make_enum_class(str class_name, tuple args):
  cdef dict enum2str = {}
  cdef dict attrs    = {}

  for name, c_enum in args:
    enum2str[c_enum] = name
    attrs[name]      = c_enum

  attrs['__enum2str'] = enum2str
  return type(class_name, (object, ), attrs)

DeviceType = make_enum_class(
  "DeviceType",
  (
    ("HOST"    , PETSC_DEVICE_HOST),
    ("CUDA"    , PETSC_DEVICE_CUDA),
    ("HIP"     , PETSC_DEVICE_HIP),
    ("SYCL"    , PETSC_DEVICE_SYCL),
    ("DEFAULT" , staticproperty(lambda *_,**__: PETSC_DEVICE_DEFAULT()))
  )
)

StreamType = make_enum_class(
  "StreamType",
  (
    ("GLOBAL_BLOCKING"    , PETSC_STREAM_GLOBAL_BLOCKING),
    ("DEFAULT_BLOCKING"   , PETSC_STREAM_DEFAULT_BLOCKING),
    ("GLOBAL_NONBLOCKING" , PETSC_STREAM_GLOBAL_NONBLOCKING),
  )
)

DeviceJoinMode = make_enum_class(
  "DeviceJoinMode",
  (
    ("DESTROY" , PETSC_DEVICE_CONTEXT_JOIN_DESTROY),
    ("SYNC"    , PETSC_DEVICE_CONTEXT_JOIN_SYNC),
    ("NO_SYNC" , PETSC_DEVICE_CONTEXT_JOIN_NO_SYNC),
  )
)

# --------------------------------------------------------------------

cdef class Device:

  Type = DeviceType

  def __cinit__(self):
    self.device = NULL

  def __dealloc__(self):
    self.destroy()

  @classmethod
  def create(cls, dtype = None, device_id = PETSC_DECIDE):
    cdef PetscInt        cdevice_id   = asInt(device_id)
    cdef PetscDeviceType cdevice_type = asDeviceType(dtype if dtype is not None else cls.Type.DEFAULT)
    cdef Device          device       = cls()

    CHKERR(PetscDeviceCreate(cdevice_type, cdevice_id, &device.device))
    return device

  def destroy(self):
    CHKERR(PetscDeviceDestroy(&self.device))

  def configure(self):
    CHKERR(PetscDeviceConfigure(self.device))

  def view(self, Viewer viewer = None):
    cdef PetscViewer vwr = NULL

    if viewer is not None:
      vwr = viewer.vwr
    CHKERR(PetscDeviceView(self.device, vwr))

  def getDeviceType(self):
    cdef PetscDeviceType cdtype

    CHKERR(PetscDeviceGetType(self.device, &cdtype))
    return toDeviceType(cdtype)

  property type:
    def __get__(self):
      return self.getDeviceType()

  def getDeviceId(self):
    cdef PetscInt cdevice_id = 0

    CHKERR(PetscDeviceGetDeviceId(self.device, &cdevice_id))
    return toInt(cdevice_id)

  property device_id:
    def __get__(self):
      return self.getDeviceId()

  @staticmethod
  def setDefaultType(device_type):
    cdef PetscDeviceType cdevice_type = asDeviceType(device_type)

    CHKERR(PetscDeviceSetDefaultDeviceType(cdevice_type))

# --------------------------------------------------------------------

cdef class DeviceContext(Object):

  JoinMode   = DeviceJoinMode
  StreamType = StreamType

  def __cinit__(self):
    self.obj  = <PetscObject*> &self.dctx
    self.dctx = NULL

  def __dealloc__(self):
    self.destroy()

  @classmethod
  def create(cls):
    cdef DeviceContext dctx = cls()

    CHKERR(PetscDeviceContextCreate(&dctx.dctx))
    return dctx

  def getStreamType(self):
    cdef PetscStreamType cstream_type = PETSC_STREAM_DEFAULT_BLOCKING

    CHKERR(PetscDeviceContextGetStreamType(self.dctx, &cstream_type))
    return toStreamType(cstream_type)

  def setStreamType(self, stream_type):
    cdef PetscStreamType cstream_type = asStreamType(stream_type)

    CHKERR(PetscDeviceContextSetStreamType(self.dctx, cstream_type))

  property stream_type:
    def __get__(self):
      return self.getStreamType()

    def __set__(self, stype):
      self.setStreamType(stype)

  def getDevice(self):
    cdef PetscDevice device = NULL

    CHKERR(PetscDeviceContextGetDevice(self.dctx, &device))
    return PyPetscDevice_New(device)

  def setDevice(self, Device device not None):
    cdef PetscDevice cdevice = PyPetscDevice_Get(device)

    CHKERR(PetscDeviceContextSetDevice(self.dctx, cdevice))

  property device:
    def __get__(self):
      return self.getDevice()

    def __set__(self, device):
      self.setDevice(device)

  def setUp(self):
    CHKERR(PetscDeviceContextSetUp(self.dctx))

  def duplicate(self):
    cdef PetscDeviceContext octx = NULL

    CHKERR(PetscDeviceContextDuplicate(self.dctx, &octx))
    return PyPetscDeviceContext_New(octx)

  def idle(self):
    cdef PetscBool is_idle = PETSC_FALSE

    CHKERR(PetscDeviceContextQueryIdle(self.dctx, &is_idle))
    return toBool(is_idle)

  def waitFor(self, other):
    cdef PetscDeviceContext cother = NULL

    if other is not None:
      cother = PyPetscDeviceContext_Get(other)
    CHKERR(PetscDeviceContextWaitForContext(self.dctx, cother))

  def fork(self, PetscInt n, stream_type = None):
    cdef PetscDeviceContext *subctx       = NULL
    cdef PetscStreamType     cstream_type = PETSC_STREAM_DEFAULT_BLOCKING

    try:
      if stream_type is None:
        CHKERR(PetscDeviceContextFork(self.dctx, n, &subctx))
      else:
        cstream_type = asStreamType(stream_type)
        CHKERR(PetscDeviceContextForkWithStreamType(self.dctx, cstream_type, n, &subctx))

      return [PyPetscDeviceContext_New(subctx[i]) for i in range(n)]
    finally:
      CHKERR(PetscFree(subctx))

  def join(self, join_mode, py_sub_ctxs):
    cdef PetscDeviceContext         *np_subctx_copy = NULL
    cdef PetscDeviceContext         *np_subctx      = NULL
    cdef PetscInt                    nsub           = 0
    cdef PetscDeviceContextJoinMode  cjoin_mode     = asJoinMode(join_mode)

    tmp = oarray_p(py_sub_ctxs, &nsub, <void**>&np_subctx)
    try:
      CHKERR(PetscMalloc(<size_t>(nsub) * sizeof(PetscDeviceContext *), &np_subctx_copy))
      CHKERR(PetscMemcpy(np_subctx_copy, np_subctx, <size_t>(nsub) * sizeof(PetscDeviceContext *)))
      CHKERR(PetscDeviceContextJoin(self.dctx, nsub, cjoin_mode, &np_subctx_copy))
    finally:
      CHKERR(PetscFree(np_subctx_copy))

    if cjoin_mode == PETSC_DEVICE_CONTEXT_JOIN_DESTROY:
      for i in range(nsub):
        py_sub_ctxs[i] = None

  def synchronize(self):
    CHKERR(PetscDeviceContextSynchronize(self.dctx))

  def setFromOptions(self, comm = None):
    cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_SELF)

    CHKERR(PetscDeviceContextSetFromOptions(ccomm, self.dctx))

  def viewFromOptions(self, name, Object obj = None):
    cdef const char *cname = NULL
    cdef PetscObject cobj  = NULL

    if obj is not None:
      cobj = obj.obj[0]

    _ = str2bytes(name, &cname)
    CHKERR(PetscDeviceContextViewFromOptions(self.dctx, cobj, cname))

  @staticmethod
  def getCurrent():
    cdef PetscDeviceContext dctx = NULL

    CHKERR(PetscDeviceContextGetCurrentContext(&dctx))
    return PyPetscDeviceContext_New(dctx)

  @staticmethod
  def setCurrent(dctx):
    cdef PetscDeviceContext cdctx = NULL

    if dctx is not None:
      cdctx = PyPetscDeviceContext_Get(dctx)
    CHKERR(PetscDeviceContextSetCurrentContext(cdctx))

  property current:
    def __get__(self):
      return self.getCurrent()

    def __set__(self, dctx):
      self.setCurrent(dctx)

# --------------------------------------------------------------------

del DeviceType
del DeviceJoinMode
del StreamType
del staticproperty
