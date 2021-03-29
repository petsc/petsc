cdef extern from * nogil:

    ctypedef const char* PetscViewerType
    PetscViewerType PETSCVIEWERSOCKET
    PetscViewerType PETSCVIEWERASCII
    PetscViewerType PETSCVIEWERBINARY
    PetscViewerType PETSCVIEWERSTRING
    PetscViewerType PETSCVIEWERDRAW
    PetscViewerType PETSCVIEWERVU
    PetscViewerType PETSCVIEWERMATHEMATICA
    PetscViewerType PETSCVIEWERHDF5
    PetscViewerType PETSCVIEWERVTK
    PetscViewerType PETSCVIEWERMATLAB
    PetscViewerType PETSCVIEWERSAWS
    PetscViewerType PETSCVIEWERGLVIS
    PetscViewerType PETSCVIEWERADIOS
    PetscViewerType PETSCVIEWEREXODUSII

    ctypedef enum PetscViewerFormat:
        PETSC_VIEWER_DEFAULT
        PETSC_VIEWER_ASCII_MATLAB
        PETSC_VIEWER_ASCII_MATHEMATICA
        PETSC_VIEWER_ASCII_IMPL
        PETSC_VIEWER_ASCII_INFO
        PETSC_VIEWER_ASCII_INFO_DETAIL
        PETSC_VIEWER_ASCII_COMMON
        PETSC_VIEWER_ASCII_SYMMODU
        PETSC_VIEWER_ASCII_INDEX
        PETSC_VIEWER_ASCII_DENSE
        PETSC_VIEWER_ASCII_MATRIXMARKET
        PETSC_VIEWER_ASCII_VTK_DEPRECATED
        PETSC_VIEWER_ASCII_VTK_CELL_DEPRECATED
        PETSC_VIEWER_ASCII_VTK_COORDS_DEPRECATED
        PETSC_VIEWER_ASCII_PCICE
        PETSC_VIEWER_ASCII_PYTHON
        PETSC_VIEWER_ASCII_FACTOR_INFO
        PETSC_VIEWER_ASCII_LATEX
        PETSC_VIEWER_ASCII_XML
        PETSC_VIEWER_ASCII_GLVIS
        PETSC_VIEWER_ASCII_CSV
        PETSC_VIEWER_DRAW_BASIC
        PETSC_VIEWER_DRAW_LG
        PETSC_VIEWER_DRAW_LG_XRANGE
        PETSC_VIEWER_DRAW_CONTOUR
        PETSC_VIEWER_DRAW_PORTS
        PETSC_VIEWER_VTK_VTS
        PETSC_VIEWER_VTK_VTR
        PETSC_VIEWER_VTK_VTU
        PETSC_VIEWER_BINARY_MATLAB
        PETSC_VIEWER_NATIVE
        PETSC_VIEWER_HDF5_PETSC
        PETSC_VIEWER_HDF5_VIZ
        PETSC_VIEWER_HDF5_XDMF
        PETSC_VIEWER_HDF5_MAT
        PETSC_VIEWER_NOFORMAT
        PETSC_VIEWER_LOAD_BALANCE
        PETSC_VIEWER_FAILED

    ctypedef enum PetscFileMode:
        PETSC_FILE_MODE_READ           "FILE_MODE_READ"
        PETSC_FILE_MODE_WRITE          "FILE_MODE_WRITE"
        PETSC_FILE_MODE_APPEND         "FILE_MODE_APPEND"
        PETSC_FILE_MODE_UPDATE         "FILE_MODE_UPDATE"
        PETSC_FILE_MODE_APPEND_UPDATE  "FILE_MODE_APPEND_UPDATE"

    enum: PETSC_DRAW_FULL_SIZE
    enum: PETSC_DRAW_HALF_SIZE
    enum: PETSC_DRAW_THIRD_SIZE
    enum: PETSC_DRAW_QUARTER_SIZE

    int PetscViewerView(PetscViewer,PetscViewer)
    int PetscViewerDestroy(PetscViewer*)
    int PetscViewerCreate(MPI_Comm,PetscViewer*)
    int PetscViewerSetType(PetscViewer,PetscViewerType)
    int PetscViewerGetType(PetscViewer,PetscViewerType*)

    int PetscViewerSetOptionsPrefix(PetscViewer,char[])
    int PetscViewerAppendOptionsPrefix(PetscViewer,char[])
    int PetscViewerGetOptionsPrefix(PetscViewer,char*[])
    int PetscViewerSetFromOptions(PetscViewer)
    int PetscViewerSetUp(PetscViewer)

    int PetscViewerASCIIOpen(MPI_Comm,char[],PetscViewer*)
    int PetscViewerBinaryCreate(MPI_Comm comm,PetscViewer*)
    int PetscViewerBinaryOpen(MPI_Comm,char[],PetscFileMode,PetscViewer*)
    int PetscViewerDrawOpen(MPI_Comm,char[],char[],int,int,int,int,PetscViewer*)

    int PetscViewerBinarySetUseMPIIO(PetscViewer,PetscBool)

    int PetscViewerSetFormat(PetscViewer,PetscViewerFormat)
    int PetscViewerGetFormat(PetscViewer,PetscViewerFormat*)
    int PetscViewerPushFormat(PetscViewer,PetscViewerFormat)
    int PetscViewerPopFormat(PetscViewer)

    int PetscViewerASCIISetTab(PetscViewer,PetscInt)
    int PetscViewerASCIIGetTab(PetscViewer,PetscInt*)
    int PetscViewerASCIIAddTab(PetscViewer,PetscInt)
    int PetscViewerASCIISubtractTab(PetscViewer,PetscInt)
    int PetscViewerASCIIPushSynchronized(PetscViewer)
    int PetscViewerASCIIPopSynchronized(PetscViewer)
    int PetscViewerASCIIPushTab(PetscViewer)
    int PetscViewerASCIIPopTab(PetscViewer)
    int PetscViewerASCIIUseTabs(PetscViewer,PetscBool)
    int PetscViewerASCIIPrintf(PetscViewer,const char[],...)
    int PetscViewerASCIISynchronizedPrintf(PetscViewer,const char[],...)

    int PetscViewerFileGetName(PetscViewer,char*[])
    int PetscViewerFileSetName(PetscViewer,char[])
    int PetscViewerFileGetMode(PetscViewer,PetscFileMode*)
    int PetscViewerFileSetMode(PetscViewer,PetscFileMode)
    int PetscViewerFlush(PetscViewer)

    int PetscViewerDrawClear(PetscViewer)
    int PetscViewerDrawSetInfo(PetscViewer,char[],char[],int,int,int,int)

    int PetscViewerHDF5GetTimestep(PetscViewer,PetscInt*)
    int PetscViewerHDF5SetTimestep(PetscViewer,PetscInt)
    int PetscViewerHDF5IncrementTimestep(PetscViewer)
    int PetscViewerHDF5PushGroup(PetscViewer,char[])
    int PetscViewerHDF5PopGroup(PetscViewer)
    int PetscViewerHDF5GetGroup(PetscViewer,char*[])

    PetscViewer PETSC_VIEWER_STDOUT_(MPI_Comm) except? NULL
    PetscViewer PETSC_VIEWER_STDOUT_SELF
    PetscViewer PETSC_VIEWER_STDOUT_WORLD

    PetscViewer PETSC_VIEWER_STDERR_(MPI_Comm) except? NULL
    PetscViewer PETSC_VIEWER_STDERR_SELF
    PetscViewer PETSC_VIEWER_STDERR_WORLD

    PetscViewer PETSC_VIEWER_BINARY_(MPI_Comm) except? NULL
    PetscViewer PETSC_VIEWER_BINARY_SELF
    PetscViewer PETSC_VIEWER_BINARY_WORLD

    PetscViewer PETSC_VIEWER_DRAW_(MPI_Comm) except? NULL
    PetscViewer PETSC_VIEWER_DRAW_SELF
    PetscViewer PETSC_VIEWER_DRAW_WORLD

# ---

cdef inline PetscFileMode filemode(object mode) except <PetscFileMode>(-1):
    if mode is None:
        return PETSC_FILE_MODE_READ
    if isinstance(mode, str):
        if   mode == 'r'  : return PETSC_FILE_MODE_READ
        elif mode == 'w'  : return PETSC_FILE_MODE_WRITE
        elif mode == 'a'  : return PETSC_FILE_MODE_APPEND
        elif mode == 'r+' : return PETSC_FILE_MODE_UPDATE
        elif mode == 'w+' : return PETSC_FILE_MODE_UPDATE
        elif mode == 'a+' : return PETSC_FILE_MODE_APPEND_UPDATE
        elif mode == 'u'  : return PETSC_FILE_MODE_UPDATE
        elif mode == 'au' : return PETSC_FILE_MODE_APPEND_UPDATE
        elif mode == 'ua' : return PETSC_FILE_MODE_APPEND_UPDATE
    return mode
