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

    PetscErrorCode PetscViewerView(PetscViewer,PetscViewer)
    PetscErrorCode PetscViewerDestroy(PetscViewer*)
    PetscErrorCode PetscViewerCreate(MPI_Comm,PetscViewer*)
    PetscErrorCode PetscViewerSetType(PetscViewer,PetscViewerType)
    PetscErrorCode PetscViewerGetType(PetscViewer,PetscViewerType*)

    PetscErrorCode PetscViewerSetOptionsPrefix(PetscViewer,char[])
    PetscErrorCode PetscViewerAppendOptionsPrefix(PetscViewer,char[])
    PetscErrorCode PetscViewerGetOptionsPrefix(PetscViewer,char*[])
    PetscErrorCode PetscViewerSetFromOptions(PetscViewer)
    PetscErrorCode PetscViewerSetUp(PetscViewer)

    PetscErrorCode PetscViewerASCIIOpen(MPI_Comm,char[],PetscViewer*)
    PetscErrorCode PetscViewerBinaryCreate(MPI_Comm comm,PetscViewer*)
    PetscErrorCode PetscViewerBinaryOpen(MPI_Comm,char[],PetscFileMode,PetscViewer*)
    PetscErrorCode PetscViewerDrawOpen(MPI_Comm,char[],char[],int,int,int,int,PetscViewer*)

    PetscErrorCode PetscViewerBinarySetUseMPIIO(PetscViewer,PetscBool)

    PetscErrorCode PetscViewerSetFormat(PetscViewer,PetscViewerFormat)
    PetscErrorCode PetscViewerGetFormat(PetscViewer,PetscViewerFormat*)
    PetscErrorCode PetscViewerPushFormat(PetscViewer,PetscViewerFormat)
    PetscErrorCode PetscViewerPopFormat(PetscViewer)

    PetscErrorCode PetscViewerGetSubViewer(PetscViewer,MPI_Comm,PetscViewer*)
    PetscErrorCode PetscViewerRestoreSubViewer(PetscViewer,MPI_Comm,PetscViewer*)

    PetscErrorCode PetscViewerASCIISetTab(PetscViewer,PetscInt)
    PetscErrorCode PetscViewerASCIIGetTab(PetscViewer,PetscInt*)
    PetscErrorCode PetscViewerASCIIAddTab(PetscViewer,PetscInt)
    PetscErrorCode PetscViewerASCIISubtractTab(PetscViewer,PetscInt)
    PetscErrorCode PetscViewerASCIIPushSynchronized(PetscViewer)
    PetscErrorCode PetscViewerASCIIPopSynchronized(PetscViewer)
    PetscErrorCode PetscViewerASCIIPushTab(PetscViewer)
    PetscErrorCode PetscViewerASCIIPopTab(PetscViewer)
    PetscErrorCode PetscViewerASCIIUseTabs(PetscViewer,PetscBool)
    PetscErrorCode PetscViewerASCIIPrintf(PetscViewer,const char[],...)
    PetscErrorCode PetscViewerStringSPrintf(PetscViewer,char[],...)
    PetscErrorCode PetscViewerASCIISynchronizedPrintf(PetscViewer,const char[],...)

    PetscErrorCode PetscViewerFileGetName(PetscViewer,char*[])
    PetscErrorCode PetscViewerFileSetName(PetscViewer,char[])
    PetscErrorCode PetscViewerFileGetMode(PetscViewer,PetscFileMode*)
    PetscErrorCode PetscViewerFileSetMode(PetscViewer,PetscFileMode)
    PetscErrorCode PetscViewerFlush(PetscViewer)

    PetscErrorCode PetscViewerDrawClear(PetscViewer)
    PetscErrorCode PetscViewerDrawSetInfo(PetscViewer,char[],char[],int,int,int,int)

    PetscErrorCode PetscViewerHDF5PushTimestepping(PetscViewer)
    PetscErrorCode PetscViewerHDF5PopTimestepping(PetscViewer)
    PetscErrorCode PetscViewerHDF5GetTimestep(PetscViewer,PetscInt*)
    PetscErrorCode PetscViewerHDF5SetTimestep(PetscViewer,PetscInt)
    PetscErrorCode PetscViewerHDF5IncrementTimestep(PetscViewer)
    PetscErrorCode PetscViewerHDF5PushGroup(PetscViewer,char[])
    PetscErrorCode PetscViewerHDF5PopGroup(PetscViewer)
    PetscErrorCode PetscViewerHDF5GetGroup(PetscViewer,char[],char*[])

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
