# --------------------------------------------------------------------

cdef class Sys:

    @classmethod
    def getVersion(cls, patch=False, devel=False,
                   date=False, author=False):
        cdef int cmajor = PETSC_VERSION_MAJOR
        cdef int cminor = PETSC_VERSION_MINOR
        cdef int cmicro = PETSC_VERSION_SUBMINOR
        cdef int cpatch = PETSC_VERSION_PATCH
        cdef int cdevel = not PETSC_VERSION_RELEASE
        cdef const_char *cdate = PETSC_VERSION_DATE
        cdef const_char *cauthorinfo = PETSC_AUTHOR_INFO
        version = (cmajor, cminor, cmicro)
        out = version
        if patch or devel or date or author:
            out = [version]
            if patch:
                out.append(cpatch)
            if devel:
                out.append(<bint>cdevel)
            if date:
                date = bytes2str(cdate)
                out.append(date)
            if author:
                author = bytes2str(cauthorinfo).split('\n')
                author = [s.strip() for s in author if s]
                out.append(author)
        return tuple(out)

    @classmethod
    def getVersionInfo(cls):
        cdef int cmajor = PETSC_VERSION_MAJOR
        cdef int cminor = PETSC_VERSION_MINOR
        cdef int cmicro = PETSC_VERSION_SUBMINOR
        cdef int cpatch = PETSC_VERSION_PATCH
        cdef int crelease = PETSC_VERSION_RELEASE
        cdef const_char *cdate = PETSC_VERSION_DATE
        cdef const_char *cauthorinfo = PETSC_AUTHOR_INFO
        author = bytes2str(cauthorinfo).split('\n')
        author = [s.strip() for s in author if s]
        return dict(major      = cmajor,
                    minor      = cminor,
                    subminor   = cmicro,
                    patch      = cpatch,
                    release    = <bint>crelease,
                    date       = bytes2str(cdate),
                    authorinfo = author)

    # --- xxx ---

    @classmethod
    def isInitialized(cls):
        return <bint>PetscInitializeCalled

    @classmethod
    def isFinalized(cls):
        return <bint>PetscFinalizeCalled

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
        cdef const_char *m = NULL
        message = str2bytes(message, &m)
        CHKERR( PetscPrintf(ccomm, m) )

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
        cdef const_char *m = NULL
        message = str2bytes(message, &m)
        CHKERR( PetscSynchronizedPrintf(ccomm, m) )
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
    def infoAllow(cls, flag):
        cdef PetscBool tval = PETSC_FALSE
        if flag: tval = PETSC_TRUE
        CHKERR( PetscInfoAllow(tval, NULL) )

    @classmethod
    def registerCitation(cls, citation):
        if not citation: raise ValueError("empty citation")
        cdef const_char *cit = NULL
        citation = str2bytes(citation, &cit)
        cdef PetscBool set = get_citation(citation)
        CHKERR( PetscCitationsRegister(cit, &set) )
        set_citation(citation, <bint>set)

cdef dict citations_registry = { }

cdef PetscBool get_citation(object citation):
    cdef bint is_set = citations_registry.get(citation)
    return PETSC_TRUE if is_set else PETSC_FALSE

cdef set_citation(object citation, bint is_set):
    citations_registry[citation] = is_set

# --------------------------------------------------------------------
