# --------------------------------------------------------------------

class ViewerType(object):
    ASCII  = S_(PETSCVIEWERASCII)
    BINARY = S_(PETSCVIEWERBINARY)
    STRING = S_(PETSCVIEWERSTRING)
    DRAW   = S_(PETSCVIEWERDRAW)
    HDF5   = S_(PETSCVIEWERHDF5)
    NETCDF = S_(PETSCVIEWERNETCDF)
    ## SOCKET      = PETSC_VIEWER_SOCKET
    ## VU          = PETSC_VIEWER_VU
    ## MATHEMATICA = PETSC_VIEWER_MATHEMATICA
    ## SILO        = PETSC_VIEWER_SILO
    ## MATLAB      = PETSC_VIEWER_MATLAB

class ViewerFormat(object):
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
    ASCII_VTK         = PETSC_VIEWER_ASCII_VTK
    ASCII_VTK_CELL    = PETSC_VIEWER_ASCII_VTK_CELL
    ASCII_VTK_COORDS  = PETSC_VIEWER_ASCII_VTK_COORDS
    ASCII_PCICE       = PETSC_VIEWER_ASCII_PCICE
    ASCII_PYLITH      = PETSC_VIEWER_ASCII_PYLITH
    ASCII_PYLITH_LOCAL= PETSC_VIEWER_ASCII_PYLITH_LOCAL
    ASCII_PYTHON      = PETSC_VIEWER_ASCII_PYTHON
    ASCII_FACTOR_INFO = PETSC_VIEWER_ASCII_FACTOR_INFO
    DRAW_BASIC        = PETSC_VIEWER_DRAW_BASIC
    DRAW_LG           = PETSC_VIEWER_DRAW_LG
    DRAW_CONTOUR      = PETSC_VIEWER_DRAW_CONTOUR
    DRAW_PORTS        = PETSC_VIEWER_DRAW_PORTS
    NATIVE            = PETSC_VIEWER_NATIVE
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

    def createASCII(self, name, mode=None, format=None, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef const_char *cname = NULL
        name = str2bytes(name, &cname)
        cdef PetscFileMode cmode = PETSC_FILE_MODE_WRITE
        if mode is not None: filemode(mode)
        cdef PetscViewerFormat cvfmt = PETSC_VIEWER_DEFAULT
        if format is not None: cvfmt = format
        cdef PetscViewer newvwr = NULL
        CHKERR( PetscViewerCreate(ccomm, &newvwr) )
        PetscCLEAR(self.obj); self.vwr = newvwr
        CHKERR( PetscViewerSetType(self.vwr, PETSCVIEWERASCII) )
        CHKERR( PetscViewerFileSetMode(self.vwr, cmode) )
        CHKERR( PetscViewerFileSetName(self.vwr, cname) )
        CHKERR( PetscViewerSetFormat(self.vwr, cvfmt) )
        return self

    def createBinary(self, name, mode=None, format=None, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef const_char *cname = NULL
        name = str2bytes(name, &cname)
        cdef PetscFileMode cmode = filemode(mode)
        cdef PetscViewerFormat cvfmt = PETSC_VIEWER_DEFAULT
        if format is not None: cvfmt = format
        cdef PetscViewer newvwr = NULL
        CHKERR( PetscViewerBinaryOpen(ccomm, cname, cmode, &newvwr) )
        PetscCLEAR(self.obj); self.vwr = newvwr
        CHKERR( PetscViewerSetFormat(self.vwr, cvfmt) )
        return self

    def createMPIIO(self, name, mode=None, format=None, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef const_char *cname = NULL
        name = str2bytes(name, &cname)
        cdef PetscFileMode cmode = filemode(mode)
        cdef PetscViewerFormat cvfmt = PETSC_VIEWER_DEFAULT
        if format is not None: cvfmt = format
        cdef PetscViewer newvwr = NULL
        CHKERR( PetscViewerCreate(ccomm, &newvwr) )
        PetscCLEAR(self.obj); self.vwr = newvwr
        CHKERR( PetscViewerSetType(self.vwr, PETSCVIEWERBINARY) )
        CHKERR( PetscViewerBinarySetMPIIO(self.vwr) )
        CHKERR( PetscViewerFileSetMode(self.vwr, cmode) )
        CHKERR( PetscViewerFileSetName(self.vwr, cname) )
        CHKERR( PetscViewerSetFormat(self.vwr, cvfmt) )
        return self

    def createHDF5(self, name, mode=None, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef const_char *cname = NULL
        name = str2bytes(name, &cname)
        cdef PetscFileMode cmode = filemode(mode)
        cdef PetscViewer newvwr = NULL
        CHKERR( PetscViewerCreate(ccomm, &newvwr) )
        PetscCLEAR(self.obj); self.vwr = newvwr
        CHKERR( PetscViewerSetType(self.vwr, PETSCVIEWERHDF5) )
        CHKERR( PetscViewerFileSetMode(self.vwr, cmode) )
        CHKERR( PetscViewerFileSetName(self.vwr, cname) )
        return self

    def createNetCDF(self, name, mode=None, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef const_char *cname = NULL
        name = str2bytes(name, &cname)
        cdef PetscFileMode cmode = filemode(mode)
        cdef PetscViewer newvwr = NULL
        CHKERR( PetscViewerCreate(ccomm, &newvwr) )
        PetscCLEAR(self.obj); self.vwr = newvwr
        CHKERR( PetscViewerSetType(self.vwr, PETSCVIEWERNETCDF) )
        CHKERR( PetscViewerFileSetMode(self.vwr, cmode) )
        CHKERR( PetscViewerFileSetName(self.vwr, cname) )
        return self

    def createDraw(self, display=None, title=None,
                   position=None, size=None, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef const_char *cdisplay = NULL
        cdef const_char *ctitle = NULL
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

    def setType(self, vwr_type):
        cdef PetscViewerType cval = NULL
        vwr_type = str2bytes(vwr_type, &cval)
        CHKERR( PetscViewerSetType(self.vwr, cval) )

    def getType(self):
        cdef PetscViewerType cval = NULL
        CHKERR( PetscViewerGetType(self.vwr, &cval) )
        return bytes2str(cval)

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
    def ASCII(cls, name, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef const_char *cname = NULL
        name = str2bytes(name, &cname)
        cdef Viewer viewer = Viewer()
        CHKERR( PetscViewerASCIIOpen(ccomm, cname, &viewer.vwr) )
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
        cdef const_char *cval = NULL
        name = str2bytes(name, &cval)
        CHKERR( PetscViewerFileSetName(self.vwr, cval) )

    def getFileName(self):
        cdef const_char *cval = NULL
        CHKERR( PetscViewerFileGetName(self.vwr, <char**>&cval) )
        return bytes2str(cval)

    # --- methods specific to draw viewers ---

    def setInfo(self,  display=None, title=None, position=None, size=None):
        cdef const_char *cdisplay = NULL
        cdef const_char *ctitle = NULL
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

    def clear(self):
        CHKERR( PetscViewerDrawClear(self.vwr) )

# --------------------------------------------------------------------

del ViewerType
del ViewerFormat
del FileMode
del DrawSize

# --------------------------------------------------------------------
