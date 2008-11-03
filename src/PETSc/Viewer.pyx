# --------------------------------------------------------------------

class ViewerType(object):
    ASCII  = PETSC_VIEWER_ASCII
    BINARY = PETSC_VIEWER_BINARY
    STRING = PETSC_VIEWER_STRING
    DRAW   = PETSC_VIEWER_DRAW
    SOCKET = PETSC_VIEWER_SOCKET
    ## VU          = PETSC_VIEWER_VU
    ## MATHEMATICA = PETSC_VIEWER_MATHEMATICA
    ## SILO        = PETSC_VIEWER_SILO
    ## NETCDF      = PETSC_VIEWER_NETCDF
    ## HDF4        = PETSC_VIEWER_HDF4
    ## MATLAB      = PETSC_VIEWER_MATLAB

class ViewerFormat(object):
    DEFAULT           = PETSC_VIEWER_DEFAULT
    NATIVE            = PETSC_VIEWER_NATIVE
    ASCII_MATLAB      = PETSC_VIEWER_ASCII_MATLAB
    ASCII_MATHEMATICA = PETSC_VIEWER_ASCII_MATHEMATICA
    ASCII_IMPL        = PETSC_VIEWER_ASCII_IMPL
    ASCII_INFO        = PETSC_VIEWER_ASCII_INFO
    ASCII_INFO_DETAIL = PETSC_VIEWER_ASCII_INFO_DETAIL
    ASCII_COMMON      = PETSC_VIEWER_ASCII_COMMON
    ASCII_SYMMODU     = PETSC_VIEWER_ASCII_SYMMODU
    ASCII_INDEX       = PETSC_VIEWER_ASCII_INDEX
    ASCII_DENSE       = PETSC_VIEWER_ASCII_DENSE
    ASCII_FACTOR_INFO = PETSC_VIEWER_ASCII_FACTOR_INFO
    DRAW_BASIC        = PETSC_VIEWER_DRAW_BASIC
    DRAW_LG           = PETSC_VIEWER_DRAW_LG
    DRAW_CONTOUR      = PETSC_VIEWER_DRAW_CONTOUR
    DRAW_PORTS        = PETSC_VIEWER_DRAW_PORTS
    NOFORMAT          = PETSC_VIEWER_NOFORMAT

class FileMode(object):
    # native
    READ          = PETSC_FILE_MODE_READ
    WRITE         = PETSC_FILE_MODE_WRITE
    APPEND        = PETSC_FILE_MODE_APPEND
    UPDATE        = PETSC_FILE_MODE_UPDATE
    APPEND_UPDATE = PETSC_FILE_MODE_APPEND_UPDATE
    # aliases
    R, W, A, U = READ, WRITE, APPEND, UPDATE
    AU = UA    = APPEND_UPDATE

class DrawSize(object):
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

    Type   = ViewerType
    Format = ViewerFormat
    Mode   = FileMode
    Size   = DrawSize

    #

    def __cinit__(self):
        self.obj = <PetscObject*> &self.vwr
        self.vwr = NULL

    def __call__(self, Object obj):
        assert obj.obj != NULL
        CHKERR( PetscObjectView(obj.obj[0], self.vwr) )

    #

    def view(self, obj=None):
        if obj is None:
            CHKERR( PetscViewerView(self.vwr, NULL) )
        elif isinstance(obj, Viewer):
            CHKERR( PetscViewerView(self.vwr, (<Viewer?>obj).vwr) )
        else:
            assert (<Object?>obj).obj != NULL
            CHKERR( PetscObjectView((<Object?>obj).obj[0], self.vwr) )

    def destroy(self):
        CHKERR( PetscViewerDestroy(self.vwr) )
        self.vwr = NULL
        return self

    def create(self, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscViewer newvwr = NULL
        CHKERR( PetscViewerCreate(ccomm, &newvwr) )
        PetscCLEAR(self.obj); self.vwr = newvwr
        return self

    def createASCII(self, name, mode=None,
                    format=None, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef char *cname = str2cp(name)
        cdef PetscFileMode cmode = PETSC_FILE_MODE_WRITE
        if mode is not None: cmode = mode
        cdef PetscViewerFormat cvfmt = PETSC_VIEWER_DEFAULT
        if format is not None: cvfmt = format
        cdef PetscViewer newvwr = NULL
        CHKERR( PetscViewerASCIIOpen(ccomm, cname, &newvwr) )
        PetscCLEAR(self.obj); self.vwr = newvwr
        CHKERR( PetscViewerFileSetMode(self.vwr, cmode) )
        CHKERR( PetscViewerSetFormat(self.vwr, cvfmt) )
        return self

    def createBinary(self, name, mode=None,
                     format=None, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef char *cname = str2cp(name)
        cdef PetscFileMode cmode = PETSC_FILE_MODE_WRITE
        if mode is not None: cmode = mode
        cdef PetscViewerFormat cvfmt = PETSC_VIEWER_DEFAULT
        if format is not None: cvfmt = format
        cdef PetscViewer newvwr = NULL
        CHKERR( PetscViewerBinaryOpen(ccomm, cname, cmode, &newvwr) )
        PetscCLEAR(self.obj); self.vwr = newvwr
        CHKERR( PetscViewerSetFormat(self.vwr, format) )
        return self

    def createDraw(self, display=None, title=None,
                   position=None, size=None, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef const_char_p cdisplay = ""
        if display is not None: cdisplay = str2cp(display)
        cdef const_char_p ctitle = ""
        if title is not None: ctitle = str2cp(title)
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
        CHKERR( PetscViewerDrawOpen(ccomm, cdisplay, ctitle, x, y, w, h, &newvwr) )
        PetscCLEAR(self.obj); self.vwr = newvwr
        return self

    def setType(self, vwr_type):
        CHKERR( PetscViewerSetType(self.vwr, str2cp(vwr_type)) )

    def getType(self):
        cdef PetscViewerType vwr_type = NULL
        CHKERR( PetscViewerGetType(self.vwr, &vwr_type) )
        return cp2str(vwr_type)

    def setFormat(self, format):
        CHKERR( PetscViewerSetFormat(self.vwr, format) )

    def getFormat(self):
        cdef PetscViewerFormat format = PETSC_VIEWER_DEFAULT
        CHKERR( PetscViewerGetFormat(self.vwr, &format) )
        return format

    def pushFormat(self, format):
        CHKERR( PetscViewerPushFormat(self.vwr, format) )

    def popFormat(self):
        CHKERR( PetscViewerPopFormat(self.vwr) )

    @classmethod
    def STDOUT(cls, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef Viewer viewer = Viewer()
        viewer.vwr = PETSC_VIEWER_STDOUT_(ccomm)
        PetscIncref(<PetscObject>(viewer.vwr))
        return viewer

    @classmethod
    def STDERR(cls, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef Viewer viewer = Viewer()
        viewer.vwr = PETSC_VIEWER_STDERR_(ccomm)
        PetscIncref(<PetscObject>(viewer.vwr))
        return viewer

    @classmethod
    def BINARY(cls, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef Viewer viewer = Viewer()
        viewer.vwr = PETSC_VIEWER_BINARY_(ccomm)
        PetscIncref(<PetscObject>(viewer.vwr))
        return viewer

    @classmethod
    def DRAW(cls, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef Viewer viewer = Viewer()
        viewer.vwr = PETSC_VIEWER_DRAW_(ccomm)
        PetscIncref(<PetscObject>(viewer.vwr))
        return viewer

    # --- methods specific to file viewers ---

    def flush(self):
        CHKERR( PetscViewerFlush(self.vwr) )

    def setFileMode(self, mode):
        CHKERR( PetscViewerFileSetMode(self.vwr, mode) )

    def getFileMode(self):
        cdef PetscFileMode mode
        CHKERR( PetscViewerFileGetMode(self.vwr, &mode) )
        return mode

    def setFileName(self, name):
        CHKERR( PetscViewerFileSetName(self.vwr, str2cp(name)) )

    def getFileName(self):
        cdef char *name = NULL
        CHKERR( PetscViewerFileGetName(self.vwr, &name) )
        return cp2str(name)

    # --- methods specific to draw viewers ---

    def setInfo(self,  display=None, title=None, position=None, size=None):
        cdef const_char_p cdisplay = ""
        if display is not None: cdisplay = str2cp(display)
        cdef const_char_p ctitle = ""
        if title is not None: ctitle = str2cp(title)
        cdef int x, y, h, w
        x = y = h = w = PETSC_DECIDE
        if position not in (None, PETSC_DECIDE):
            x, y = position
        if size not in (None, PETSC_DECIDE):
            try:
                w, h = size
            except TypeError:
                w = h = size
        CHKERR( PetscViewerDrawSetInfo(self.vwr, display, title, x, y, w, h) )

    def clear(self):
        CHKERR( PetscViewerDrawClear(self.vwr) )

# --------------------------------------------------------------------
