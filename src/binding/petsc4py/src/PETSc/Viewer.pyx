# --------------------------------------------------------------------

class ViewerType(object):
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
        CHKERR( PetscViewerDestroy(&self.vwr) )
        return self

    def create(self, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscViewer newvwr = NULL
        CHKERR( PetscViewerCreate(ccomm, &newvwr) )
        PetscCLEAR(self.obj); self.vwr = newvwr
        return self

    def createASCII(self, name, mode=None, comm=None):
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

    def createBinary(self, name, mode=None, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef const char *cname = NULL
        name = str2bytes(name, &cname)
        cdef PetscFileMode cmode = filemode(mode)
        cdef PetscViewer newvwr = NULL
        CHKERR( PetscViewerBinaryOpen(ccomm, cname, cmode, &newvwr) )
        PetscCLEAR(self.obj); self.vwr = newvwr
        return self

    def createMPIIO(self, name, mode=None, comm=None):
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

    def createVTK(self, name, mode=None, comm=None):
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

    def createHDF5(self, name, mode=None, comm=None):
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

    def createDraw(self, display=None, title=None,
                   position=None, size=None, comm=None):
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

    def setType(self, vwr_type):
        cdef PetscViewerType cval = NULL
        vwr_type = str2bytes(vwr_type, &cval)
        CHKERR( PetscViewerSetType(self.vwr, cval) )

    def getType(self):
        cdef PetscViewerType cval = NULL
        CHKERR( PetscViewerGetType(self.vwr, &cval) )
        return bytes2str(cval)

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
        PetscINCREF(viewer.obj)
        return viewer

    @classmethod
    def STDERR(cls, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef Viewer viewer = Viewer()
        viewer.vwr = PETSC_VIEWER_STDERR_(ccomm)
        PetscINCREF(viewer.obj)
        return viewer

    @classmethod
    def ASCII(cls, name, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef const char *cname = NULL
        name = str2bytes(name, &cname)
        cdef Viewer viewer = Viewer()
        CHKERR( PetscViewerASCIIOpen(ccomm, cname, &viewer.vwr) )
        return viewer

    @classmethod
    def BINARY(cls, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef Viewer viewer = Viewer()
        viewer.vwr = PETSC_VIEWER_BINARY_(ccomm)
        PetscINCREF(viewer.obj)
        return viewer

    @classmethod
    def DRAW(cls, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef Viewer viewer = Viewer()
        viewer.vwr = PETSC_VIEWER_DRAW_(ccomm)
        PetscINCREF(viewer.obj)
        return viewer

    # --- ASCII viewers ---

    def setASCIITab(self, tabs):
        cdef PetscInt ctabs = asInt(tabs)
        CHKERR( PetscViewerASCIISetTab(self.vwr, ctabs) )

    def getASCIITab(self):
        cdef PetscInt tabs = 0
        CHKERR( PetscViewerASCIIGetTab(self.vwr, &tabs) )
        return toInt(tabs)

    def addASCIITab(self, tabs):
        cdef PetscInt ctabs = asInt(tabs)
        CHKERR( PetscViewerASCIIAddTab(self.vwr, ctabs) )

    def subtractASCIITab(self, tabs):
        cdef PetscInt ctabs = asInt(tabs)
        CHKERR( PetscViewerASCIISubtractTab(self.vwr, ctabs) )

    def pushASCIISynchronized(self):
        CHKERR( PetscViewerASCIIPushSynchronized(self.vwr) )

    def popASCIISynchronized(self):
        CHKERR( PetscViewerASCIIPopSynchronized(self.vwr) )

    def pushASCIITab(self):
        CHKERR( PetscViewerASCIIPushTab(self.vwr) )

    def popASCIITab(self):
        CHKERR( PetscViewerASCIIPopTab(self.vwr) )

    def useASCIITabs(self, flag):
        cdef PetscBool flg = flag
        CHKERR( PetscViewerASCIIUseTabs(self.vwr, flg) )

    def printfASCII(self, msg):
        cdef const char *cmsg = NULL
        msg = str2bytes(msg, &cmsg)
        CHKERR( PetscViewerASCIIPrintf(self.vwr, '%s', cmsg) )

    def printfASCIISynchronized(self, msg):
        cdef const char *cmsg = NULL
        msg = str2bytes(msg, &cmsg)
        CHKERR( PetscViewerASCIISynchronizedPrintf(self.vwr, '%s', cmsg) )

    # --- methods specific to file viewers ---

    def flush(self):
        CHKERR( PetscViewerFlush(self.vwr) )

    def setFileMode(self, mode):
        CHKERR( PetscViewerFileSetMode(self.vwr, filemode(mode)) )

    def getFileMode(self):
        cdef PetscFileMode mode = PETSC_FILE_MODE_READ
        CHKERR( PetscViewerFileGetMode(self.vwr, &mode) )
        return mode

    def setFileName(self, name):
        cdef const char *cval = NULL
        name = str2bytes(name, &cval)
        CHKERR( PetscViewerFileSetName(self.vwr, cval) )

    def getFileName(self):
        cdef const char *cval = NULL
        CHKERR( PetscViewerFileGetName(self.vwr, &cval) )
        return bytes2str(cval)

    # --- methods specific to draw viewers ---

    def setDrawInfo(self,  display=None, title=None, position=None, size=None):
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

    def clearDraw(self):
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
        cdef const char *cgroup = NULL
        CHKERR( PetscViewerHDF5GetGroup(self.vwr, &cgroup) )
        return bytes2str(cgroup)

# --------------------------------------------------------------------

del ViewerType
del ViewerFormat
del FileMode
del DrawSize

# --------------------------------------------------------------------
