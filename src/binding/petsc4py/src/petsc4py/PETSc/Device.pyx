# --------------------------------------------------------------------

class staticproperty(property):
    def __get__(self, *args, **kwargs):
        return self.fget.__get__(*args, **kwargs)()


cdef object make_enum_class(str class_name, str class_docstring, tuple args):
    cdef dict enum2str = {}
    cdef dict attrs    = {}

    for name, c_enum in args:
        enum2str[c_enum] = name
        attrs[name]      = c_enum

    attrs['__enum2str'] = enum2str
    attrs['__doc__']    = class_docstring
    return type(class_name, (object,), attrs)

DeviceType = make_enum_class(
  "DeviceType",
  """The type of device.

  See Also
  --------
  Device, Device.create, Device.getDeviceType, Device.type, petsc.PetscDeviceType

  """,
  (
    ("HOST"    , PETSC_DEVICE_HOST),
    ("CUDA"    , PETSC_DEVICE_CUDA),
    ("HIP"     , PETSC_DEVICE_HIP),
    ("SYCL"    , PETSC_DEVICE_SYCL),
    ("DEFAULT" , staticproperty(lambda *_, **__: PETSC_DEVICE_DEFAULT()))
  )
)

StreamType = make_enum_class(
  "StreamType",
  """The type of stream.

  See Also
  --------
  DeviceContext, DeviceContext.getStreamType
  DeviceContext.setStreamType, petsc.PetscStreamType

  """,
  (
    ("DEFAULT"                  , PETSC_STREAM_DEFAULT),
    ("NONBLOCKING"              , PETSC_STREAM_NONBLOCKING),
    ("DEFAULT_WITH_BARRIER"     , PETSC_STREAM_DEFAULT_WITH_BARRIER),
    ("NONBLOCKING_WITH_BARRIER" , PETSC_STREAM_NONBLOCKING_WITH_BARRIER),
  )
)

DeviceJoinMode = make_enum_class(
  "DeviceJoinMode",
  """The type of join to perform.

  See Also
  --------
  DeviceContext, DeviceContext.join, DeviceContext.fork
  petsc.PetscDeviceContextJoinMode

  """,
  (
    ("DESTROY" , PETSC_DEVICE_CONTEXT_JOIN_DESTROY),
    ("SYNC"    , PETSC_DEVICE_CONTEXT_JOIN_SYNC),
    ("NO_SYNC" , PETSC_DEVICE_CONTEXT_JOIN_NO_SYNC),
  )
)

# --------------------------------------------------------------------

cdef class Device:
    """The device object.

    Represents a handle to an accelerator (which may be the host).

    See Also
    --------
    DeviceContext, petsc.PetscDevice

    """

    Type = DeviceType

    def __cinit__(self):
        self.device = NULL

    def __dealloc__(self):
        self.destroy()

    @classmethod
    def create(cls, dtype: Type | None = None, device_id: int = DECIDE) -> Device:
        """Create a device object.

        Not collective.

        Parameters
        ----------
        dtype
            The type of device to create (or `None` for the default).

        device_id
            The numeric id of the device to create.

        See Also
        --------
        destroy, petsc.PetscDeviceCreate

        """
        cdef PetscInt        cdevice_id   = asInt(device_id)
        cdef PetscDeviceType cdevice_type = asDeviceType(dtype if dtype is not None else cls.Type.DEFAULT)
        cdef Device          device       = cls()

        CHKERR(PetscDeviceCreate(cdevice_type, cdevice_id, &device.device))
        return device

    def destroy(self) -> None:
        """Destroy a device object.

        Not collective.

        See Also
        --------
        create, petsc.PetscDeviceDestroy

        """
        CHKERR(PetscDeviceDestroy(&self.device))

    def configure(self) -> None:
        """Configure and setup a device object.

        Not collective.

        See Also
        --------
        create, petsc.PetscDeviceConfigure

        """
        CHKERR(PetscDeviceConfigure(self.device))

    def view(self, Viewer viewer=None) -> None:
        """View a device object.

        Collective.

        Parameters
        ----------
        viewer
            A `Viewer` instance or `None` for the default viewer.

        See Also
        --------
        petsc.PetscDeviceView

        """
        cdef PetscViewer vwr = NULL

        if viewer is not None:
            vwr = viewer.vwr
        CHKERR(PetscDeviceView(self.device, vwr))

    def getDeviceType(self) -> str:
        """Return the type of the device.

        Not collective.

        See Also
        --------
        type, petsc.PetscDeviceGetType

        """
        cdef PetscDeviceType cdtype = PETSC_DEVICE_HOST

        CHKERR(PetscDeviceGetType(self.device, &cdtype))
        return toDeviceType(cdtype)

    def getDeviceId(self) -> int:
        """Return the device id.

        Not collective.

        See Also
        --------
        create, petsc.PetscDeviceGetDeviceId

        """
        cdef PetscInt cdevice_id = 0

        CHKERR(PetscDeviceGetDeviceId(self.device, &cdevice_id))
        return toInt(cdevice_id)

    @staticmethod
    def setDefaultType(device_type: Type | str) -> None:
        """Set the device type to be used as the default in subsequent calls to `create`.

        Not collective.

        See Also
        --------
        create, petsc.PetscDeviceSetDefaultDeviceType

        """
        cdef PetscDeviceType cdevice_type = asDeviceType(device_type)

        CHKERR(PetscDeviceSetDefaultDeviceType(cdevice_type))

    property type:
        """The device type."""
        def __get__(self) -> str:
            return self.getDeviceType()

    property device_id:
        """The device id."""
        def __get__(self) -> int:
            return self.getDeviceId()


# --------------------------------------------------------------------

cdef class DeviceContext(Object):
    """DeviceContext object.

    Represents an abstract handle to a device context.

    See Also
    --------
    Device, petsc.PetscDeviceContext

    """
    JoinMode   = DeviceJoinMode
    StreamType = StreamType

    def __cinit__(self):
        self.obj  = <PetscObject*> &self.dctx
        self.dctx = NULL

    def create(self) -> Self:
        """Create an empty DeviceContext.

        Not collective.

        See Also
        --------
        destroy, Device, petsc.PetscDeviceContextCreate

        """
        cdef PetscDeviceContext dctx = NULL
        CHKERR(PetscDeviceContextCreate(&dctx))
        CHKERR(PetscCLEAR(self.obj)); self.dctx = dctx
        return self

    def destroy(self) -> Self:
        """Destroy a device context.

        Not collective.

        See Also
        --------
        create, petsc.PetscDeviceContextDestroy

        """
        CHKERR(PetscDeviceContextDestroy(&self.dctx))
        return self

    def getStreamType(self) -> str:
        """Return the `StreamType`.

        Not collective.

        See Also
        --------
        stream_type, setStreamType, petsc.PetscDeviceContextGetStreamType

        """
        cdef PetscStreamType cstream_type = PETSC_STREAM_DEFAULT

        CHKERR(PetscDeviceContextGetStreamType(self.dctx, &cstream_type))
        return toStreamType(cstream_type)

    def setStreamType(self, stream_type: StreamType | str) -> None:
        """Set the `StreamType`.

        Not collective.

        Parameters
        ----------
        stream_type
            The type of stream to set

        See Also
        --------
        stream_type, getStreamType, petsc.PetscDeviceContextSetStreamType

        """
        cdef PetscStreamType cstream_type = asStreamType(stream_type)

        CHKERR(PetscDeviceContextSetStreamType(self.dctx, cstream_type))

    def getDevice(self) -> Device:
        """Get the `Device` which this instance is attached to.

        Not collective.

        See Also
        --------
        setDevice, device, Device, petsc.PetscDeviceContextGetDevice

        """
        cdef PetscDevice device = NULL

        CHKERR(PetscDeviceContextGetDevice(self.dctx, &device))
        return PyPetscDevice_New(device)

    def setDevice(self, Device device not None) -> None:
        """Set the `Device` which this `DeviceContext` is attached to.

        Collective.

        Parameters
        ----------
        device
            The `Device` to which this instance is attached to.

        See Also
        --------
        getDevice, device, Device, petsc.PetscDeviceContextSetDevice

        """
        cdef PetscDevice cdevice = PyPetscDevice_Get(device)

        CHKERR(PetscDeviceContextSetDevice(self.dctx, cdevice))

    def setUp(self) -> None:
        """Set up the internal data structures for using the device context.

        Not collective.

        See Also
        --------
        create, petsc.PetscDeviceContextSetUp

        """
        CHKERR(PetscDeviceContextSetUp(self.dctx))

    def duplicate(self) -> DeviceContext:
        """Duplicate a the device context.

        Not collective.

        See Also
        --------
        create, petsc.PetscDeviceContextDuplicate

        """
        cdef DeviceContext octx = type(self)()

        CHKERR(PetscDeviceContextDuplicate(self.dctx, &octx.dctx))
        return octx

    def idle(self) -> bool:
        """Return whether the underlying stream for the device context is idle.

        Not collective.

        See Also
        --------
        synchronize, petsc.PetscDeviceContextQueryIdle

        """
        cdef PetscBool is_idle = PETSC_FALSE

        CHKERR(PetscDeviceContextQueryIdle(self.dctx, &is_idle))
        return toBool(is_idle)

    def waitFor(self, other: DeviceContext | None) -> None:
        """Make this instance wait for ``other``.

        Not collective.

        Parameters
        ----------
        other
            The other `DeviceContext` to wait for

        See Also
        --------
        fork, join, petsc.PetscDeviceContextWaitForContext

        """
        cdef PetscDeviceContext cother = NULL

        if other is not None:
            cother = PyPetscDeviceContext_Get(other)
        CHKERR(PetscDeviceContextWaitForContext(self.dctx, cother))

    def fork(self, n: int, stream_type: DeviceContext.StreamType | str | None = None) -> list[DeviceContext]:
        """Create multiple device contexts which are all logically dependent on this one.

        Not collective.

        Parameters
        ----------
        n
            The number of device contexts to create.
        stream_type
            The type of stream of the forked device context.

        Examples
        --------
        The device contexts created must be destroyed using `join`.

        >>> dctx = PETSc.DeviceContext().getCurrent()
        >>> dctxs = dctx.fork(4)
        >>> ... # perform computations
        >>> # we can mix various join modes
        >>> dctx.join(PETSc.DeviceContext.JoinMode.SYNC, dctxs[0:2])
        >>> dctx.join(PETSc.DeviceContext.JoinMode.SYNC, dctxs[2:])
        >>> ... # some more computations and joins
        >>> # dctxs must be all destroyed with joinMode.DESTROY
        >>> dctx.join(PETSc.DeviceContext.JoinMode.DESTROY, dctxs)

        See Also
        --------
        join, waitFor, petsc.PetscDeviceContextFork

        """
        cdef PetscDeviceContext *csubctxs = NULL
        cdef PetscStreamType cstream_type = PETSC_STREAM_DEFAULT
        cdef PetscInt cn = asInt(n)
        cdef list subctxs = []
        if stream_type is None:
            CHKERR(PetscDeviceContextFork(self.dctx, cn, &csubctxs))
        else:
            cstream_type = asStreamType(stream_type)
            CHKERR(PetscDeviceContextForkWithStreamType(self.dctx, cstream_type, cn, &csubctxs))
        # FIXME: without CXX compiler, csubctxs is NULL
        if csubctxs:
            subctxs = [None] * cn
            for i from 0 <= i < cn:
                subctxs[i] = DeviceContext()
                (<DeviceContext?>subctxs[i]).dctx = csubctxs[i]
            CHKERR(PetscFree(csubctxs))
        return subctxs

    def join(self, join_mode: DeviceJoinMode | str, py_sub_ctxs: list[DeviceContext]) -> None:
        """Join a set of device contexts on this one.

        Not collective.

        Parameters
        ----------
        join_mode
            The type of join to perform.
        py_sub_ctxs
            The list of device contexts to join.

        See Also
        --------
        fork, waitFor, petsc.PetscDeviceContextJoin

        """
        cdef PetscDeviceContext *np_subctx = NULL
        cdef PetscDeviceContextJoinMode cjoin_mode = asJoinMode(join_mode)
        cdef Py_ssize_t nctxs = len(py_sub_ctxs)

        CHKERR(PetscMalloc(<size_t>(nctxs) * sizeof(PetscDeviceContext *), &np_subctx))
        for i from 0 <= i < nctxs:
            dctx = py_sub_ctxs[i]
            np_subctx[i] = (<DeviceContext?>dctx).dctx if dctx is not None else NULL
        CHKERR(PetscDeviceContextJoin(self.dctx, <PetscInt>nctxs, cjoin_mode, &np_subctx))

        if cjoin_mode == PETSC_DEVICE_CONTEXT_JOIN_DESTROY:
            # in this case, PETSc destroys the contexts and frees the array
            for i in range(nctxs):
                (<DeviceContext?>py_sub_ctxs[i]).dctx = NULL
        else:
            # we need to free the temporary array
            CHKERR(PetscFree(np_subctx))

    def synchronize(self) -> None:
        """Synchronize a device context.

        Not collective.

        Notes
        -----
        The underlying stream is considered idle after this routine returns,
        i.e. `idle` will return ``True``.

        See Also
        --------
        idle, petsc.PetscDeviceContextSynchronize

        """
        CHKERR(PetscDeviceContextSynchronize(self.dctx))

    def setFromOptions(self, comm: Comm | None = None) -> None:
        """Configure the `DeviceContext` from the options database.

        Collective.

        Parameters
        ----------
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        Sys.getDefaultComm, petsc.PetscDeviceContextSetFromOptions

        """
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)

        CHKERR(PetscDeviceContextSetFromOptions(ccomm, self.dctx))

    @staticmethod
    def getCurrent() -> DeviceContext:
        """Return the current device context.

        Not collective.

        See Also
        --------
        current, setCurrent, petsc.PetscDeviceContextGetCurrentContext

        """
        cdef PetscDeviceContext dctx = NULL

        CHKERR(PetscDeviceContextGetCurrentContext(&dctx))
        return PyPetscDeviceContext_New(dctx)

    @staticmethod
    def setCurrent(dctx: DeviceContext | None) -> None:
        """Set the current device context.

        Not collective.

        Parameters
        ----------
        dctx
            The `DeviceContext` to set as current (or `None` to use
            the default context).

        See Also
        --------
        current, getCurrent, petsc.PetscDeviceContextSetCurrentContext

        """
        cdef PetscDeviceContext cdctx = NULL

        if dctx is not None:
            cdctx = PyPetscDeviceContext_Get(dctx)
        CHKERR(PetscDeviceContextSetCurrentContext(cdctx))

    property stream_type:
        """The stream type."""
        def __get__(self) -> str:
            return self.getStreamType()

        def __set__(self, stype: StreamType | str) -> None:
            self.setStreamType(stype)

    property device:
        """The device associated to the device context."""
        def __get__(self) -> Device:
            return self.getDevice()

        def __set__(self, Device device) -> None:
            self.setDevice(device)

    property current:
        """The current global device context."""
        def __get__(self) -> DeviceContext:
            return self.getCurrent()

        def __set__(self, dctx: DeviceContext | None) -> None:
            self.setCurrent(dctx)

# --------------------------------------------------------------------

del DeviceType
del DeviceJoinMode
del StreamType
del staticproperty
