# --------------------------------------------------------------------

class ViewerType(object):
    """Viewer type."""
    SOCKET      = S_(PETSCVIEWERSOCKET)
    ASCII       = S_(PETSCVIEWERASCII)
    BINARY      = S_(PETSCVIEWERBINARY)
    STRING      = S_(PETSCVIEWERSTRING)
    DRAW        = S_(PETSCVIEWERDRAW)
    VU          = S_(PETSCVIEWERVU)
    MATHEMATICA = S_(PETSCVIEWERMATHEMATICA)
    HDF5        = S_(PETSCVIEWERHDF5)
    VTK         = S_(PETSCVIEWERVTK)
    MATLAB      = S_(PETSCVIEWERMATLAB)
    SAWS        = S_(PETSCVIEWERSAWS)
    GLVIS       = S_(PETSCVIEWERGLVIS)
    ADIOS       = S_(PETSCVIEWERADIOS)
    EXODUSII    = S_(PETSCVIEWEREXODUSII)

class ViewerFormat(object):
    """Viewer format."""
    DEFAULT           = PETSC_VIEWER_DEFAULT
    ASCII_MATLAB      = PETSC_VIEWER_ASCII_MATLAB
    ASCII_MATHEMATICA = PETSC_VIEWER_ASCII_MATHEMATICA
    ASCII_IMPL        = PETSC_VIEWER_ASCII_IMPL
    ASCII_INFO        = PETSC_VIEWER_ASCII_INFO
    ASCII_INFO_DETAIL = PETSC_VIEWER_ASCII_INFO_DETAIL
    ASCII_COMMON      = PETSC_VIEWER_ASCII_COMMON
    ASCII_SYMMODU     = PETSC_VIEWER_ASCII_SYMMODU
    ASCII_INDEX       = PETSC_VIEWER_ASCII_INDEX
    ASCII_DENSE       = PETSC_VIEWER_ASCII_DENSE
    ASCII_MATRIXMARKET= PETSC_VIEWER_ASCII_MATRIXMARKET
    ASCII_VTK         = PETSC_VIEWER_ASCII_VTK_DEPRECATED
    ASCII_VTK_CELL    = PETSC_VIEWER_ASCII_VTK_CELL_DEPRECATED
    ASCII_VTK_COORDS  = PETSC_VIEWER_ASCII_VTK_COORDS_DEPRECATED
    ASCII_PCICE       = PETSC_VIEWER_ASCII_PCICE
    ASCII_PYTHON      = PETSC_VIEWER_ASCII_PYTHON
    ASCII_FACTOR_INFO = PETSC_VIEWER_ASCII_FACTOR_INFO
    ASCII_LATEX       = PETSC_VIEWER_ASCII_LATEX
    ASCII_XML         = PETSC_VIEWER_ASCII_XML
    ASCII_GLVIS       = PETSC_VIEWER_ASCII_GLVIS
    ASCII_CSV         = PETSC_VIEWER_ASCII_CSV
    DRAW_BASIC        = PETSC_VIEWER_DRAW_BASIC
    DRAW_LG           = PETSC_VIEWER_DRAW_LG
    DRAW_LG_XRANGE    = PETSC_VIEWER_DRAW_LG_XRANGE
    DRAW_CONTOUR      = PETSC_VIEWER_DRAW_CONTOUR
    DRAW_PORTS        = PETSC_VIEWER_DRAW_PORTS
    VTK_VTS           = PETSC_VIEWER_VTK_VTS
    VTK_VTR           = PETSC_VIEWER_VTK_VTR
    VTK_VTU           = PETSC_VIEWER_VTK_VTU
    BINARY_MATLAB     = PETSC_VIEWER_BINARY_MATLAB
    NATIVE            = PETSC_VIEWER_NATIVE
    HDF5_PETSC        = PETSC_VIEWER_HDF5_PETSC
    HDF5_VIZ          = PETSC_VIEWER_HDF5_VIZ
    HDF5_XDMF         = PETSC_VIEWER_HDF5_XDMF
    HDF5_MAT          = PETSC_VIEWER_HDF5_MAT
    NOFORMAT          = PETSC_VIEWER_NOFORMAT
    LOAD_BALANCE      = PETSC_VIEWER_LOAD_BALANCE
    FAILED            = PETSC_VIEWER_FAILED

class ViewerFileMode(object):
    """Viewer file mode."""
    # native
    READ          = PETSC_FILE_MODE_READ
    WRITE         = PETSC_FILE_MODE_WRITE
    APPEND        = PETSC_FILE_MODE_APPEND
    UPDATE        = PETSC_FILE_MODE_UPDATE
    APPEND_UPDATE = PETSC_FILE_MODE_APPEND_UPDATE
    # aliases
    R, W, A, U = READ, WRITE, APPEND, UPDATE
    AU = UA    = APPEND_UPDATE

class ViewerDrawSize(object):
    """Window size."""
    # native
    FULL_SIZE    = PETSC_DRAW_FULL_SIZE
    HALF_SIZE    = PETSC_DRAW_HALF_SIZE
    THIRD_SIZE   = PETSC_DRAW_THIRD_SIZE
    QUARTER_SIZE = PETSC_DRAW_QUARTER_SIZE
    # aliases
    FULL    = FULL_SIZE
    HALF    = HALF_SIZE
    THIRD   = THIRD_SIZE
    QUARTER = QUARTER_SIZE

# --------------------------------------------------------------------

cdef class Viewer(Object):
    """Viewer object.

    Viewer is described in the `PETSc manual <petsc:sec_viewers>`.

    Viewers can be called as functions where the argument specified is the PETSc object to be viewed. See the example below.

    Examples
    --------
    >>> from petsc4py import PETSc
    >>> u = PETSc.Vec().createWithArray([1,2])
    >>> v = PETSc.Viewer()
    >>> v(u)
    Vec Object: 1 MPI process
      type: seq
    1.
    2.

    See Also
    --------
    petsc.PetscViewer

    """

    Type   = ViewerType
    Format = ViewerFormat
    FileMode = ViewerFileMode
    DrawSize = ViewerDrawSize

    # backward compatibility
    Mode = ViewerFileMode
    Size = ViewerFileMode

    #

    def __cinit__(self):
        self.obj = <PetscObject*> &self.vwr
        self.vwr = NULL

    def __call__(self, Object obj) -> None:
        """View a generic object."""
        assert obj.obj != NULL
        CHKERR( PetscObjectView(obj.obj[0], self.vwr) )

    #

    def view(self, obj: Viewer | Object | None = None) -> None:
        """View the viewer.

        Collective.

        Parameters
        ----------
        obj
            A `Viewer` instance or `None` for the default viewer.
            If none of the above applies, it assumes ``obj`` is an instance of `Object`
            and it calls the generic view for ``obj``.

        Notes
        -----

        See Also
        --------
        petsc.PetscViewerView

        """
        if obj is None:
            CHKERR( PetscViewerView(self.vwr, NULL) )
        elif isinstance(obj, Viewer):
            CHKERR( PetscViewerView(self.vwr, (<Viewer?>obj).vwr) )
        else:
            assert (<Object?>obj).obj != NULL
            CHKERR( PetscObjectView((<Object?>obj).obj[0], self.vwr) )

    def destroy(self) -> Self:
        """Destroy the viewer.

        Collective.

        See Also
        --------
        petsc.PetscViewerDestroy

        """
        CHKERR( PetscViewerDestroy(&self.vwr) )
        return self

    def create(self, comm: Comm | None = None) -> Self:
        """Create a viewer.

        Collective.

        Parameters
        ----------
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        Sys.getDefaultComm, petsc.PetscViewerCreate

        """
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscViewer newvwr = NULL
        CHKERR( PetscViewerCreate(ccomm, &newvwr) )
        PetscCLEAR(self.obj); self.vwr = newvwr
        return self

    def createASCII(
        self,
        name: str,
        mode: FileMode | str | None = None,
        comm: Comm | None = None,
        ) -> Self:
        """Create a viewer of type `Type.ASCII`.

        Collective.

        Parameters
        ----------
        name
            The filename associated with the viewer.
        mode
            The mode type.
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        create, setType, setFileMode, setFileName, Sys.getDefaultComm
        setASCIITab, addASCIITab, subtractASCIITab, getASCIITab

        """
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef const char *cname = NULL
        name = str2bytes(name, &cname)
        cdef PetscFileMode cmode = PETSC_FILE_MODE_WRITE
        if mode is not None: cmode = filemode(mode)
        cdef PetscViewer newvwr = NULL
        CHKERR( PetscViewerCreate(ccomm, &newvwr) )
        PetscCLEAR(self.obj); self.vwr = newvwr
        CHKERR( PetscViewerSetType(self.vwr, PETSCVIEWERASCII) )
        CHKERR( PetscViewerFileSetMode(self.vwr, cmode) )
        CHKERR( PetscViewerFileSetName(self.vwr, cname) )
        return self

    def createBinary(
        self,
        name: str,
        mode: FileMode | str | None = None,
        comm: Comm | None = None,
        ) -> Self:
        """Create a viewer of type `Type.BINARY`.

        Collective.

        Parameters
        ----------
        name
            The filename associated with the viewer.
        mode
            The mode type.
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        create, setType, setFileMode, setFileName, Sys.getDefaultComm

        """
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef const char *cname = NULL
        name = str2bytes(name, &cname)
        cdef PetscFileMode cmode = filemode(mode)
        cdef PetscViewer newvwr = NULL
        CHKERR( PetscViewerBinaryOpen(ccomm, cname, cmode, &newvwr) )
        PetscCLEAR(self.obj); self.vwr = newvwr
        return self

    def createMPIIO(
        self,
        name: str,
        mode: FileMode | str | None = None,
        comm: Comm | None = None,
        ) -> Self:
        """Create a viewer of type `Type.BINARY` supporting MPI-IO.

        Collective.

        Parameters
        ----------
        name
            The filename associated with the viewer.
        mode
            The mode type.
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        create, setType, setFileMode, setFileName, Sys.getDefaultComm

        """
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef const char *cname = NULL
        name = str2bytes(name, &cname)
        cdef PetscFileMode cmode = filemode(mode)
        cdef PetscViewer newvwr = NULL
        CHKERR( PetscViewerCreate(ccomm, &newvwr) )
        PetscCLEAR(self.obj); self.vwr = newvwr
        CHKERR( PetscViewerSetType(self.vwr, PETSCVIEWERBINARY) )
        CHKERR( PetscViewerBinarySetUseMPIIO(self.vwr, PETSC_TRUE) )
        CHKERR( PetscViewerFileSetMode(self.vwr, cmode) )
        CHKERR( PetscViewerFileSetName(self.vwr, cname) )
        return self

    def createVTK(
        self,
        name: str,
        mode: FileMode | str | None = None,
        comm: Comm | None = None,
        ) -> Self:
        """Create a viewer of type `Type.VTK`.

        Collective.

        Parameters
        ----------
        name
            The filename associated with the viewer.
        mode
            The mode type.
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        create, setType, setFileMode, setFileName, Sys.getDefaultComm

        """
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef const char *cname = NULL
        name = str2bytes(name, &cname)
        cdef PetscFileMode cmode = filemode(mode)
        cdef PetscViewer newvwr = NULL
        CHKERR( PetscViewerCreate(ccomm, &newvwr) )
        PetscCLEAR(self.obj); self.vwr = newvwr
        CHKERR( PetscViewerSetType(self.vwr, PETSCVIEWERVTK) )
        CHKERR( PetscViewerFileSetMode(self.vwr, cmode) )
        CHKERR( PetscViewerFileSetName(self.vwr, cname) )
        return self

    def createHDF5(
        self,
        name: str,
        mode: FileMode | str | None = None,
        comm: Comm | None = None,
        ) -> Self:
        """Create a viewer of type `Type.HDF5`.

        Collective.

        Parameters
        ----------
        name
            The filename associated with the viewer.
        mode
            The mode type.
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        create, setType, setFileMode, setFileName, Sys.getDefaultComm

        """
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef const char *cname = NULL
        name = str2bytes(name, &cname)
        cdef PetscFileMode cmode = filemode(mode)
        cdef PetscViewer newvwr = NULL
        CHKERR( PetscViewerCreate(ccomm, &newvwr) )
        PetscCLEAR(self.obj); self.vwr = newvwr
        CHKERR( PetscViewerSetType(self.vwr, PETSCVIEWERHDF5) )
        CHKERR( PetscViewerFileSetMode(self.vwr, cmode) )
        CHKERR( PetscViewerFileSetName(self.vwr, cname) )
        return self

    def createDraw(
        self,
        display: str | None = None,
        title: str | None = None,
        position: tuple[int, int] | None = None,
        size: tuple[int, int] | int | None = None,
        comm: Comm | None = None,
        ) -> Self:
        """Create a `Type.DRAW` viewer.

        Collective.

        Parameters
        ----------
        display
            The X display to use or `None` for the local machine.
        title
            The window title or `None` for no title.
        position
            Screen coordinates of the upper left corner, or `None` for default.
        size
            Window size or `None` for default.
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        Sys.getDefaultComm, petsc.PetscViewerDrawOpen

        """
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef const char *cdisplay = NULL
        cdef const char *ctitle = NULL
        display = str2bytes(display, &cdisplay)
        title = str2bytes(title, &ctitle)
        cdef int x, y, h, w
        x = y = h = w = PETSC_DECIDE
        if position not in (None, PETSC_DECIDE):
            x, y = position
        if size not in (None, PETSC_DECIDE):
            try:
                w, h = size
            except TypeError:
                w = h = size
        cdef PetscViewer newvwr = NULL
        CHKERR( PetscViewerDrawOpen(ccomm, cdisplay, ctitle,
                                    x, y, w, h, &newvwr) )
        PetscCLEAR(self.obj); self.vwr = newvwr
        return self

    def setType(self, vwr_type: Type | str) -> None:
        """Set the type of the viewer.

        Logically collective.

        Parameters
        ----------
        vwr_type
            The type of the viewer.

        See Also
        --------
        getType, petsc.PetscViewerSetType

        """
        cdef PetscViewerType cval = NULL
        vwr_type = str2bytes(vwr_type, &cval)
        CHKERR( PetscViewerSetType(self.vwr, cval) )

    def getType(self) -> str:
        """Return the type of the viewer.

        Not collective.

        See Also
        --------
        setType, petsc.PetscViewerGetType

        """
        cdef PetscViewerType cval = NULL
        CHKERR( PetscViewerGetType(self.vwr, &cval) )
        return bytes2str(cval)

    def getFormat(self) -> Format:
        """Return the format of the viewer.

        Not collective.

        See Also
        --------
        pushFormat, popFormat, petsc.PetscViewerGetFormat

        """
        cdef PetscViewerFormat format = PETSC_VIEWER_DEFAULT
        CHKERR( PetscViewerGetFormat(self.vwr, &format) )
        return format

    def pushFormat(self, format: Format) -> None:
        """Push format to the viewer.

        Collective.

        See Also
        --------
        popFormat, petsc.PetscViewerPushFormat

        """
        CHKERR( PetscViewerPushFormat(self.vwr, format) )

    def popFormat(self) -> None:
        """Pop format from the viewer.

        Collective.

        See Also
        --------
        pushFormat, petsc.PetscViewerPopFormat

        """
        CHKERR( PetscViewerPopFormat(self.vwr) )

    def getSubViewer(self, comm: Comm | None = None) -> Viewer:
        """Return a viewer defined on a subcommunicator.

        Collective.

        Parameters
        ----------
        comm
            The subcommunicator. If `None`, uses `COMM_SELF`.

        Notes
        -----
        Users must call `restoreSubViewer` when done.

        See Also
        --------
        restoreSubViewer, petsc.PetscViewerGetSubViewer

        """
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_SELF)
        cdef Viewer sub = Viewer()
        CHKERR( PetscViewerGetSubViewer(self.vwr, ccomm, &sub.vwr) )
        return sub

    def restoreSubViewer(self, Viewer sub) -> None:
        """Restore a viewer defined on a subcommunicator.

        Collective.

        Parameters
        ----------
        sub
            The subviewer obtained from `getSubViewer`.

        See Also
        --------
        getSubViewer, petsc.PetscViewerRestoreSubViewer

        """
        cdef MPI_Comm ccomm = def_Comm(sub.getComm(), PETSC_COMM_SELF)
        CHKERR( PetscViewerRestoreSubViewer(self.vwr, ccomm, &sub.vwr) )

    @classmethod
    def STDOUT(cls, comm: Comm | None = None) -> Viewer:
        """Return the standard output viewer associated with the communicator.

        Collective.

        Parameters
        ----------
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        """
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef Viewer viewer = Viewer()
        viewer.vwr = PETSC_VIEWER_STDOUT_(ccomm)
        PetscINCREF(viewer.obj)
        return viewer

    @classmethod
    def STDERR(cls, comm: Comm | None = None) -> Viewer:
        """Return the standard error viewer associated with the communicator.

        Collective.

        Parameters
        ----------
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        """
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef Viewer viewer = Viewer()
        viewer.vwr = PETSC_VIEWER_STDERR_(ccomm)
        PetscINCREF(viewer.obj)
        return viewer

    @classmethod
    def ASCII(cls, name : str, comm: Comm | None = None) -> Viewer:
        """Return an ASCII viewer associated with the communicator.

        Collective.

        Parameters
        ----------
        name
            The filename.
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        """
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef const char *cname = NULL
        name = str2bytes(name, &cname)
        cdef Viewer viewer = Viewer()
        CHKERR( PetscViewerASCIIOpen(ccomm, cname, &viewer.vwr) )
        return viewer

    @classmethod
    def BINARY(cls, comm: Comm | None = None) -> Viewer:
        """Return the default `Type.BINARY` viewer associated with the communicator.

        Collective.

        Parameters
        ----------
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        """
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef Viewer viewer = Viewer()
        viewer.vwr = PETSC_VIEWER_BINARY_(ccomm)
        PetscINCREF(viewer.obj)
        return viewer

    @classmethod
    def DRAW(cls, comm: Comm | None = None) -> Viewer:
        """Return the default `Type.DRAW` viewer associated with the communicator.

        Collective.

        Parameters
        ----------
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        """
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef Viewer viewer = Viewer()
        viewer.vwr = PETSC_VIEWER_DRAW_(ccomm)
        PetscINCREF(viewer.obj)
        return viewer

    # --- ASCII viewers ---

    def setASCIITab(self, tabs : int) -> None:
        """Set ASCII tab level.

        Collective.

        See Also
        --------
        getASCIITab, petsc.PetscViewerASCIISetTab

        """
        cdef PetscInt ctabs = asInt(tabs)
        CHKERR( PetscViewerASCIISetTab(self.vwr, ctabs) )

    def getASCIITab(self) -> int:
        """Return the ASCII tab level.

        Not collective.

        See Also
        --------
        setASCIITab, petsc.PetscViewerASCIIGetTab

        """
        cdef PetscInt tabs = 0
        CHKERR( PetscViewerASCIIGetTab(self.vwr, &tabs) )
        return toInt(tabs)

    def addASCIITab(self, tabs: int):
        """Increment the ASCII tab level.

        Collective.

        See Also
        --------
        petsc.PetscViewerASCIIAddTab

        """
        cdef PetscInt ctabs = asInt(tabs)
        CHKERR( PetscViewerASCIIAddTab(self.vwr, ctabs) )

    def subtractASCIITab(self, tabs: int) -> None:
        """Decrement the ASCII tab level.

        Collective.

        See Also
        --------
        petsc.PetscViewerASCIISubtractTab

        """
        cdef PetscInt ctabs = asInt(tabs)
        CHKERR( PetscViewerASCIISubtractTab(self.vwr, ctabs) )

    def pushASCIISynchronized(self) -> None:
        """Allow ASCII synchronized calls.

        Collective.

        See Also
        --------
        printfASCIISynchronized, popASCIISynchronized
        petsc.PetscViewerASCIIPushSynchronized

        """
        CHKERR( PetscViewerASCIIPushSynchronized(self.vwr) )

    def popASCIISynchronized(self) -> None:
        """Disallow ASCII synchronized calls.

        Collective.

        See Also
        --------
        printfASCIISynchronized, pushASCIISynchronized
        petsc.PetscViewerASCIIPopSynchronized

        """
        CHKERR( PetscViewerASCIIPopSynchronized(self.vwr) )

    def pushASCIITab(self) -> None:
        """Push an additional tab level.

        Collective.

        See Also
        --------
        popASCIITab, petsc.PetscViewerASCIIPushTab

        """
        CHKERR( PetscViewerASCIIPushTab(self.vwr) )

    def popASCIITab(self) -> None:
        """Pop an additional tab level pushed via `pushASCIITab`.

        Collective.

        See Also
        --------
        pushASCIITab, petsc.PetscViewerASCIIPopTab

        """
        CHKERR( PetscViewerASCIIPopTab(self.vwr) )

    def useASCIITabs(self, flag: bool) -> None:
        """Enable/disable the use of ASCII tabs.

        Collective.

        See Also
        --------
        petsc.PetscViewerASCIIUseTabs

        """
        cdef PetscBool flg = asBool(flag)
        CHKERR( PetscViewerASCIIUseTabs(self.vwr, flg) )

    def printfASCII(self, msg: str) -> None:
        """Print a message.

        Collective.

        See Also
        --------
        petsc.PetscViewerASCIIPrintf

        """
        cdef const char *cmsg = NULL
        msg = str2bytes(msg, &cmsg)
        CHKERR( PetscViewerASCIIPrintf(self.vwr, '%s', cmsg) )

    def printfASCIISynchronized(self, msg: str) -> None:
        """Print a synchronized message.

        Collective.

        See Also
        --------
        pushASCIISynchronized, petsc.PetscViewerASCIISynchronizedPrintf

        """
        cdef const char *cmsg = NULL
        msg = str2bytes(msg, &cmsg)
        CHKERR( PetscViewerASCIISynchronizedPrintf(self.vwr, '%s', cmsg) )

    # --- methods specific to file viewers ---

    def flush(self) -> None:
        """Flush the viewer.

        Collective.

        See Also
        --------
        petsc.PetscViewerFlush

        """
        CHKERR( PetscViewerFlush(self.vwr) )

    def setFileMode(self, mode: FileMode | str) -> None:
        """Set file mode.

        Collective.

        See Also
        --------
        getFileMode, petsc.PetscViewerFileSetMode

        """
        CHKERR( PetscViewerFileSetMode(self.vwr, filemode(mode)) )

    def getFileMode(self) -> FileMode:
        """Return the file mode.

        Not collective.

        See Also
        --------
        setFileMode, petsc.PetscViewerFileGetMode

        """
        cdef PetscFileMode mode = PETSC_FILE_MODE_READ
        CHKERR( PetscViewerFileGetMode(self.vwr, &mode) )
        return mode

    def setFileName(self, name: str) -> None:
        """Set file name.

        Collective.

        See Also
        --------
        getFileName, petsc.PetscViewerFileSetName

        """
        cdef const char *cval = NULL
        name = str2bytes(name, &cval)
        CHKERR( PetscViewerFileSetName(self.vwr, cval) )

    def getFileName(self) -> str:
        """Return file name.

        Not collective.

        See Also
        --------
        setFileName, petsc.PetscViewerFileGetName

        """
        cdef const char *cval = NULL
        CHKERR( PetscViewerFileGetName(self.vwr, &cval) )
        return bytes2str(cval)

    # --- methods specific to draw viewers ---

    def setDrawInfo(
        self,
        display: str | None = None,
        title: str | None = None,
        position: tuple[int, int] | None = None,
        size: tuple[int, int] | int | None = None,
        ) -> None:
        """Set window information for a `Type.DRAW` viewer.

        Collective.

        Parameters
        ----------
        display
            The X display to use or `None` for the local machine.
        title
            The window title or `None` for no title.
        position
            Screen coordinates of the upper left corner, or `None` for default.
        size
            Window size or `None` for default.

        """
        # FIXME missing manual page
        # See Also
        # --------
        # petsc.PetscViewerDrawSetInfo
        cdef const char *cdisplay = NULL
        cdef const char *ctitle = NULL
        display = str2bytes(display, &cdisplay)
        title = str2bytes(title, &ctitle)
        cdef int x, y, h, w
        x = y = h = w = PETSC_DECIDE
        if position not in (None, PETSC_DECIDE):
            x, y = position
        if size not in (None, PETSC_DECIDE):
            try:
                w, h = size
            except TypeError:
                w = h = size
        CHKERR( PetscViewerDrawSetInfo(self.vwr,
                                       cdisplay, ctitle,
                                       x, y, w, h) )

    def clearDraw(self) -> None:
        """Reset graphics.

        See Also
        --------
        petsc.PetscViewerDrawClear

        """
        CHKERR( PetscViewerDrawClear(self.vwr) )

# --------------------------------------------------------------------

cdef class ViewerHDF5(Viewer):

    def create(self, name, mode=None, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef const char *cname = NULL
        name = str2bytes(name, &cname)
        cdef PetscFileMode cmode = filemode(mode)
        cdef PetscViewer newvwr = NULL
        CHKERR( PetscViewerCreate(ccomm, &newvwr) )
        PetscCLEAR(self.obj); self.vwr = newvwr
        CHKERR( PetscViewerSetType(self.vwr, PETSCVIEWERHDF5) )
        CHKERR( PetscViewerFileSetMode(self.vwr, cmode) )
        CHKERR( PetscViewerFileSetName(self.vwr, cname) )
        return self

    def pushTimestepping(self):
        CHKERR( PetscViewerHDF5PushTimestepping(self.vwr) )

    def popTimestepping(self):
        CHKERR( PetscViewerHDF5PopTimestepping(self.vwr) )

    def getTimestep(self):
        cdef PetscInt ctimestep = 0
        CHKERR( PetscViewerHDF5GetTimestep(self.vwr, &ctimestep) )
        return toInt(ctimestep)

    def setTimestep(self, timestep):
        CHKERR( PetscViewerHDF5SetTimestep(self.vwr, asInt(timestep)) )

    def incrementTimestep(self):
        CHKERR( PetscViewerHDF5IncrementTimestep(self.vwr) )

    def pushGroup(self, group):
        cdef const char *cgroup = NULL
        group = str2bytes(group, &cgroup)
        CHKERR( PetscViewerHDF5PushGroup(self.vwr, cgroup) )

    def popGroup(self):
        CHKERR( PetscViewerHDF5PopGroup(self.vwr) )

    def getGroup(self):
        cdef char *cgroup = NULL
        CHKERR( PetscViewerHDF5GetGroup(self.vwr, NULL, &cgroup) )
        group = bytes2str(cgroup)
        CHKERR( PetscFree(cgroup) )
        return group

# --------------------------------------------------------------------

del ViewerType
del ViewerFormat
del ViewerFileMode
del ViewerDrawSize

# --------------------------------------------------------------------
