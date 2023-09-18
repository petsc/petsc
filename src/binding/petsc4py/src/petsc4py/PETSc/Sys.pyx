# ------------------------------------------------------------------------------

cdef class Sys:

    """System utilities."""

    @classmethod
    def getVersion(
        cls,
        devel: bool = False,
        date: bool = False,
        author: bool = False,
    ) -> tuple[int, int, int]:
        """Return PETSc version information.

        Not collective.

        Parameters
        ----------
        devel
            Additionally, return whether using an in-development version.
        date
            Additionally, return date information.
        author
            Additionally, return author information.

        Returns
        -------
        major : int
            Major version number.
        minor : int
            Minor version number.
        micro : int
            Micro (or patch) version number.

        See Also
        --------
        petsc.PetscGetVersion, petsc.PetscGetVersionNumber

        """
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
    def getVersionInfo(cls) -> dict[str, bool | int | str]:
        """Return PETSc version information.

        Not collective.

        Returns
        -------
        info : dict
            Dictionary with version information.

        See Also
        --------
        petsc.PetscGetVersion, petsc.PetscGetVersionNumber

        """
        version, dev, date, author = cls.getVersion(True, True, True)
        return dict(major      = version[0],
                    minor      = version[1],
                    subminor   = version[2],
                    release    = not dev,
                    date       = date,
                    authorinfo = author)

    # --- xxx ---

    @classmethod
    def isInitialized(cls) -> bool:
        """Return whether PETSc has been initialized.

        Not collective.

        See Also
        --------
        isFinalized

        """
        return toBool(PetscInitializeCalled)

    @classmethod
    def isFinalized(cls) -> bool:
        """Return whether PETSc has been finalized.

        Not collective.

        See Also
        --------
        isInitialized

        """
        return toBool(PetscFinalizeCalled)

    # --- xxx ---

    @classmethod
    def getDefaultComm(cls) -> Comm:
        """Get the default MPI communicator used to create PETSc objects.

        Not collective.

        See Also
        --------
        setDefaultComm

        """
        cdef Comm comm = Comm()
        comm.comm = PETSC_COMM_DEFAULT
        return comm

    @classmethod
    def setDefaultComm(cls, comm: Comm | None) -> None:
        """Set the default MPI communicator used to create PETSc objects.

        Logically collective.

        Parameters
        ----------
        comm
            MPI communicator. If set to `None`, uses `COMM_WORLD`.

        See Also
        --------
        getDefaultComm

        """
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_WORLD)
        if ccomm == MPI_COMM_NULL:
            raise ValueError("null communicator")
        global PETSC_COMM_DEFAULT
        PETSC_COMM_DEFAULT = ccomm

    # --- xxx ---

    @classmethod
    def Print(
        cls,
        *args: Any,
        sep: str = ' ',
        end: str = '\n',
        comm: Comm | None = None,
        **kwargs: Any,
    ) -> None:
        """Print output from the first processor of a communicator.

        Collective.

        Parameters
        ----------
        *args
            Positional arguments.
        sep
            String inserted between values, by default a space.
        end
            String appended after the last value, by default a newline.
        comm
            MPI communicator, defaults to `getDefaultComm`.
        **kwargs
            Keyword arguments.

        See Also
        --------
        petsc.PetscPrintf

        """
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
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
    def syncPrint(
        cls,
        *args: Any,
        sep: str = ' ',
        end: str = '\n',
        flush: bool = False,
        comm: Comm | None = None,
        **kwargs: Any,
    ) -> None:
        """Print synchronized output from several processors of a communicator.

        Not collective.

        Parameters
        ----------
        *args
            Positional arguments.
        sep
            String inserted between values, by default a space.
        end
            String appended after the last value, by default a newline.
        flush
            Whether to flush output with `syncFlush`.
        comm
            MPI communicator, defaults to `getDefaultComm`.
        **kwargs
            Keyword arguments.

        See Also
        --------
        petsc.PetscSynchronizedPrintf, petsc.PetscSynchronizedFlush

        """
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        if not args: args = ('',)
        format = ['%s', sep] * len(args)
        format[-1] = end
        message = ''.join(format) % args
        cdef const char *m = NULL
        message = str2bytes(message, &m)
        CHKERR( PetscSynchronizedPrintf(ccomm, '%s', m) )
        if flush: CHKERR( PetscSynchronizedFlush(ccomm, PETSC_STDOUT) )

    @classmethod
    def syncFlush(cls, comm: Comm | None = None) -> None:
        """Flush output from previous `syncPrint` calls.

        Collective.

        Parameters
        ----------
        comm
            MPI communicator, defaults to `getDefaultComm`.

        See Also
        --------
        petsc.PetscSynchronizedPrintf, petsc.PetscSynchronizedFlush

        """
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        CHKERR( PetscSynchronizedFlush(ccomm, PETSC_STDOUT) )

    # --- xxx ---

    @classmethod
    def splitOwnership(
        cls,
        size: int | tuple[int, int],
        bsize: int | None = None,
        comm: Comm | None = None
    ) -> tuple[int, int]:
        """Given a global (or local) size determines a local (or global) size.

        Collective.

        Parameters
        ----------
        size
            Global size ``N`` or 2-tuple ``(n, N)`` with local and global
            sizes. Either of ``n`` or ``N`` (but not both) can be `None`.
        bsize
            Block size, defaults to ``1``.
        comm
            MPI communicator, defaults to `getDefaultComm`.

        Returns
        -------
        n : int
            The local size.
        N : int
            The global size.

        Notes
        -----
        The ``size`` argument corresponds to the full size of the
        vector. That is, an array with 10 blocks and a block size of 3 will
        have a ``size`` of 30, not 10.

        See Also
        --------
        petsc.PetscSplitOwnership

        """
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
    def sleep(cls, seconds: float = 1.0) -> None:
        """Sleep some number of seconds.

        Not collective.

        Parameters
        ----------
        seconds
            Time to sleep in seconds.

        See Also
        --------
        petsc.PetscSleep

        """
        cdef PetscReal s = asReal(seconds)
        CHKERR( PetscSleep(s) )

    # --- xxx ---

    @classmethod
    def pushErrorHandler(cls, errhandler: str) -> None:
        """Set the current error handler.

        Logically collective.

        Parameters
        ----------
        errhandler
            The name of the error handler.

        See Also
        --------
        petsc.PetscPushErrorHandler

        """
        cdef PetscErrorHandlerFunction handler = NULL
        if errhandler == "python":
            handler = <PetscErrorHandlerFunction> PetscPythonErrorHandler
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
            raise ValueError(f"unknown error handler: {errhandler!r}")
        CHKERR( PetscPushErrorHandler(handler, NULL) )


    @classmethod
    def popErrorHandler(cls) -> None:
        """Remove the current error handler.

        Logically collective.

        See Also
        --------
        petsc.PetscPopErrorHandler

        """
        CHKERR( PetscPopErrorHandler() )

    @classmethod
    def popSignalHandler(cls) -> None:
        """Remove the current signal handler.

        Logically collective.

        See Also
        --------
        petsc.PetscPopSignalHandler

        """
        CHKERR( PetscPopSignalHandler() )

    @classmethod
    def infoAllow(
        cls,
        flag: bool,
        filename: str | None = None,
        mode: str = "w",
    ) -> None:
        """Enables or disables PETSc info messages.

        Not collective.

        Parameters
        ----------
        flag
            Whether to enable info messages.
        filename
            Name of a file where to dump output.
        mode
            Write mode for file, by default ``"w"``.

        See Also
        --------
        petsc.PetscInfoAllow, petsc.PetscInfoSetFile

        """
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
    def registerCitation(cls, citation: str) -> None:
        """Register BibTeX citation.

        Not collective.

        Parameters
        ----------
        citation
            The BibTex citation entry to register.

        See Also
        --------
        petsc.PetscCitationsRegister

        """
        if not citation: raise ValueError("empty citation")
        cdef const char *cit = NULL
        citation = str2bytes(citation, &cit)
        cdef PetscBool flag = get_citation(citation)
        CHKERR( PetscCitationsRegister(cit, &flag) )
        set_citation(citation, toBool(flag))

    @classmethod
    def hasExternalPackage(cls, package: str) -> bool:
        """Return whether PETSc has support for external package.

        Not collective.

        Parameters
        ----------
        package
            The external package name.

        See Also
        --------
        petsc.PetscHasExternalPackage

        """
        cdef const char *cpackage = NULL
        package = str2bytes(package, &cpackage)
        cdef PetscBool has = PETSC_FALSE
        CHKERR( PetscHasExternalPackage(cpackage, &has) )
        return toBool(has)


cdef dict citations_registry = { }

cdef PetscBool get_citation(object citation):
    cdef bint is_set = citations_registry.get(citation)
    return PETSC_TRUE if is_set else PETSC_FALSE

cdef set_citation(object citation, bint is_set):
    citations_registry[citation] = is_set

# ------------------------------------------------------------------------------
