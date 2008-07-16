# --------------------------------------------------------------------

cdef class Sys:

    ## @classmethod
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
    getVersion = classmethod(getVersion)

    ## @classmethod
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
    getVersionInfo = classmethod(getVersionInfo)

    # --- xxx ---

    ## @classmethod
    def isInitialized(cls):
        return <bint>PetscInitializeCalled
    isInitialized = classmethod(isInitialized)

    ## @classmethod
    def isFinalized(cls):
        return <bint>PetscFinalizeCalled
    isFinalized = classmethod(isFinalized)

    # --- xxx ---

    ## @classmethod
    def Print(cls, *args, **kwargs):
        cdef object comm = kwargs.get('comm', None)
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_WORLD)
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
    Print = classmethod(Print)

    ## @classmethod
    def syncPrint(cls, *args, **kwargs):
        cdef object comm = kwargs.get('comm', None)
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_WORLD)
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
    syncPrint = classmethod(syncPrint)

    ## @classmethod
    def syncFlush(cls, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_WORLD)
        CHKERR( PetscSynchronizedFlush(ccomm) )
    syncFlush = classmethod(syncFlush)

    # --- xxx ---

    ## @classmethod
    def splitOwnership(cls, size, bsize=None, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_WORLD)
        cdef PetscInt bs=0, n=0, N=0
        CHKERR( Sys_SplitSizes(ccomm, size, bsize, &bs, &n, &N) )
        return (n, N)
    splitOwnership = classmethod(splitOwnership)

    ## @classmethod
    def sleep(cls, seconds=1):
        cdef int s = seconds
        CHKERR( PetscSleep(s) )
    sleep = classmethod(sleep)

# --------------------------------------------------------------------
