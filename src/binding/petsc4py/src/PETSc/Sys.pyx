# --------------------------------------------------------------------

class DeviceType(object):
    INVALID = PETSC_DEVICE_HOST
    CUDA    = PETSC_DEVICE_CUDA
    HIP     = PETSC_DEVICE_HIP
    SYCL    = PETSC_DEVICE_SYCL
    MAX     = PETSC_DEVICE_MAX

# --------------------------------------------------------------------

cdef class Sys:

    @classmethod
    def getVersion(cls, devel=False, date=False, author=False):
        cdef char cversion[256]
        cdef PetscInt major=0, minor=0, micro=0, release=0
        CHKERR( PetscGetVersion(cversion, sizeof(cversion)) )
        CHKERR( PetscGetVersionNumber(&major, &minor, &micro, &release) )
        out = version = (toInt(major), toInt(minor), toInt(micro))
        if devel or date or author:
            out = [version]
            if devel:
                out.append(not <bint>release)
            if date:
                vstr = bytes2str(cversion)
                if release != 0:
                    date = vstr.split(",", 1)[-1].strip()
                else:
                    date = vstr.split("GIT Date:")[-1].strip()
                out.append(date)
            if author:
                author = bytes2str(PETSC_AUTHOR_INFO).split('\n')
                author = tuple([s.strip() for s in author if s])
                out.append(author)
        return tuple(out)

    @classmethod
    def getVersionInfo(cls):
        version, dev, date, author = cls.getVersion(True, True, True)
        return dict(major      = version[0],
                    minor      = version[1],
                    subminor   = version[2],
                    release    = not dev,
                    date       = date,
                    authorinfo = author)

    # --- xxx ---

    @classmethod
    def isInitialized(cls):
        return toBool(PetscInitializeCalled)

    @classmethod
    def isFinalized(cls):
        return toBool(PetscFinalizeCalled)

    # --- xxx ---

    @classmethod
    def getDefaultComm(cls):
        cdef Comm comm = Comm()
        comm.comm = PETSC_COMM_DEFAULT
        return comm

    @classmethod
    def setDefaultComm(cls, comm):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_WORLD)
        if ccomm == MPI_COMM_NULL:
            raise ValueError("null communicator")
        global PETSC_COMM_DEFAULT
        PETSC_COMM_DEFAULT = ccomm

    # --- xxx ---

    @classmethod
    def Print(cls, *args, **kargs):
        cdef object comm = kargs.get('comm', None)
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef object sep = kargs.get('sep', ' ')
        cdef object end = kargs.get('end', '\n')
        if comm_rank(ccomm) == 0:
            if not args: args = ('',)
            format = ['%s', sep] * len(args)
            format[-1] = end
            message = ''.join(format) % args
        else:
            message = ''
        cdef const char *m = NULL
        message = str2bytes(message, &m)
        CHKERR( PetscPrintf(ccomm, '%s', m) )

    @classmethod
    def syncPrint(cls, *args, **kargs):
        cdef object comm = kargs.get('comm', None)
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef object sep = kargs.get('sep', ' ')
        cdef object end = kargs.get('end', '\n')
        cdef object flush = kargs.get('flush', False)
        if not args: args = ('',)
        format = ['%s', sep] * len(args)
        format[-1] = end
        message = ''.join(format) % args
        cdef const char *m = NULL
        message = str2bytes(message, &m)
        CHKERR( PetscSynchronizedPrintf(ccomm, '%s', m) )
        if flush: CHKERR( PetscSynchronizedFlush(ccomm, PETSC_STDOUT) )

    @classmethod
    def syncFlush(cls, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        CHKERR( PetscSynchronizedFlush(ccomm, PETSC_STDOUT) )

    # --- xxx ---

    @classmethod
    def splitOwnership(cls, size, bsize=None, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscInt bs=0, n=0, N=0
        Sys_Sizes(size, bsize, &bs, &n, &N)
        if bs == PETSC_DECIDE: bs = 1
        if n > 0: n = n // bs
        if N > 0: N = N // bs
        CHKERR( PetscSplitOwnership(ccomm, &n, &N) )
        n = n * bs
        N = N * bs
        return (toInt(n), toInt(N))

    @classmethod
    def sleep(cls, seconds=1):
        cdef int s = seconds
        CHKERR( PetscSleep(s) )

    # --- xxx ---

    @classmethod
    def pushErrorHandler(cls, errhandler):
        cdef PetscErrorHandlerFunction handler = NULL
        if errhandler == "python":
            handler = <PetscErrorHandlerFunction> \
                      PetscPythonErrorHandler
        elif errhandler == "debugger":
            handler = PetscAttachDebuggerErrorHandler
        elif errhandler == "emacs":
            handler = PetscEmacsClientErrorHandler
        elif errhandler == "traceback":
            handler = PetscTraceBackErrorHandler
        elif errhandler == "ignore":
            handler = PetscIgnoreErrorHandler
        elif errhandler == "mpiabort":
            handler = PetscMPIAbortErrorHandler
        elif errhandler == "abort":
            handler = PetscAbortErrorHandler
        else:
            raise ValueError(
                "unknown error handler: %s" % errhandler)
        CHKERR( PetscPushErrorHandler(handler, NULL) )


    @classmethod
    def popErrorHandler(cls):
        CHKERR( PetscPopErrorHandler() )

    @classmethod
    def popSignalHandler(cls):
        CHKERR( PetscPopSignalHandler() )

    @classmethod
    def infoAllow(cls, flag, filename=None, mode="w"):
        cdef PetscBool tval = PETSC_FALSE
        cdef const char *cfilename = NULL
        cdef const char *cmode = NULL
        if flag: tval = PETSC_TRUE
        CHKERR( PetscInfoAllow(tval) )
        if filename is not None:
            filename = str2bytes(filename, &cfilename)
            mode = str2bytes(mode, &cmode)
            CHKERR( PetscInfoSetFile(cfilename, cmode) )

    @classmethod
    def registerCitation(cls, citation):
        if not citation: raise ValueError("empty citation")
        cdef const char *cit = NULL
        citation = str2bytes(citation, &cit)
        cdef PetscBool flag = get_citation(citation)
        CHKERR( PetscCitationsRegister(cit, &flag) )
        set_citation(citation, toBool(flag))

    @classmethod
    def hasExternalPackage(cls, package):
        cdef const char *cpackage = NULL
        package = str2bytes(package, &cpackage)
        cdef PetscBool has = PETSC_FALSE
        CHKERR( PetscHasExternalPackage(cpackage, &has) )
        return toBool(has)

    DeviceType = DeviceType

    @staticmethod
    def initializeDevice(device_type):
        """
        Initializes the device *device_type*.

        :arg device_type: A :class:`Sys.DeviceType` enum value.
        """
        CHKERR ( PetscDeviceInitialize(device_type) )

    @staticmethod
    def isDeviceInitialized(device_type):
        """
        Returns *True* if the device *device_type* is initialized,
        otherwise returns *False*.

        :arg device_type: A :class:`Sys.DeviceType` enum value.
        """
        return bool(PetscDeviceInitialized(device_type))


del DeviceType

cdef dict citations_registry = { }

cdef PetscBool get_citation(object citation):
    cdef bint is_set = citations_registry.get(citation)
    return PETSC_TRUE if is_set else PETSC_FALSE

cdef set_citation(object citation, bint is_set):
    citations_registry[citation] = is_set

# --------------------------------------------------------------------
