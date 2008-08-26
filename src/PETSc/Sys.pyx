# --------------------------------------------------------------------

cdef class Sys:

    @classmethod
    def getVersion(cls, patch=False, date=False, author=False):
        cdef int cmajor = PETSC_VERSION_MAJOR
        cdef int cminor = PETSC_VERSION_MINOR
        cdef int cmicro = PETSC_VERSION_SUBMINOR
        cdef int cpatch = PETSC_VERSION_PATCH
        cdef int crelease = PETSC_VERSION_RELEASE ## XXX unused
        cdef const_char_p cdate       = PETSC_VERSION_DATE
        cdef const_char_p cpatchdate  = PETSC_VERSION_PATCH_DATE
        cdef const_char_p cauthorinfo = PETSC_AUTHOR_INFO
        version = (cmajor, cminor, cmicro)
        out = version
        if patch or date or author:
            out = [version]
            if patch:
                out.append(cpatch)
            if date:
                if patch: date = [cp2str(cdate), cp2str(cpatchdate)]
                else:     date = cp2str(cdate)
                out.append(date)
            if author:
                author = cp2str(cauthorinfo).split('\n')
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
        cdef const_char_p cdate       = PETSC_VERSION_DATE
        cdef const_char_p cpatchdate  = PETSC_VERSION_PATCH_DATE
        cdef const_char_p cauthorinfo = PETSC_AUTHOR_INFO
        author = str(cp2str(cauthorinfo)).split('\n')
        author = [s.strip() for s in author if s]
        return dict(major      = cmajor,
                    minor      = cminor,
                    subminor   = cmicro,
                    patch      = cpatch,
                    release    = <bint>crelease,
                    date       = cp2str(cdate),
                    patchdate  = cp2str(cpatchdate),
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
    def setDefaultComm(cls, Comm comm not None):
        if comm.comm == MPI_COMM_NULL:
            raise ValueError("null communicator")
        global PETSC_COMM_DEFAULT
        PETSC_COMM_DEFAULT = comm.comm

    # --- xxx ---

    @classmethod
    def Print(cls, *args, **kwargs):
        cdef object comm = kwargs.get('comm', None)
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef object sep = kwargs.get('sep', ' ')
        cdef object end = kwargs.get('end', '\n')
        if comm_rank(ccomm) == 0:
            if not args: args = ('',)
            format = ['%s', sep] * len(args)
            format[-1] = end
            message = ''.join(format) % args
        else:
            message = ''
        cdef char *m = str2cp(message)
        CHKERR( PetscPrintf(ccomm, m) )

    @classmethod
    def syncPrint(cls, *args, **kwargs):
        cdef object comm = kwargs.get('comm', None)
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef object sep = kwargs.get('sep', ' ')
        cdef object end = kwargs.get('end', '\n')
        cdef object flush = kwargs.get('flush', False)
        if not args: args = ('',)
        format = ['%s', sep] * len(args)
        format[-1] = end
        message = ''.join(format) % args
        cdef char *m = str2cp(message)
        CHKERR( PetscSynchronizedPrintf(ccomm, m) )
        if flush: CHKERR( PetscSynchronizedFlush(ccomm) )

    @classmethod
    def syncFlush(cls, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        CHKERR( PetscSynchronizedFlush(ccomm) )

    # --- xxx ---

    @classmethod
    def splitOwnership(cls, size, bsize=None, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscInt bs=0, n=0, N=0
        CHKERR( Sys_SplitSizes(ccomm, size, bsize, &bs, &n, &N) )
        return (n, N)

    @classmethod
    def sleep(cls, seconds=1):
        cdef int s = seconds
        CHKERR( PetscSleep(s) )

# --------------------------------------------------------------------

cdef MPI_Comm PETSC_COMM_DEFAULT = MPI_COMM_NULL

# --------------------------------------------------------------------
