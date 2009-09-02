cdef extern  from "petsc.h":

    ctypedef char* PetscViewerType "const char*"
    PetscViewerType PETSC_VIEWER_SOCKET
    PetscViewerType PETSC_VIEWER_ASCII
    PetscViewerType PETSC_VIEWER_BINARY
    PetscViewerType PETSC_VIEWER_STRING
    PetscViewerType PETSC_VIEWER_DRAW
    PetscViewerType PETSC_VIEWER_VU
    PetscViewerType PETSC_VIEWER_MATHEMATICA
    PetscViewerType PETSC_VIEWER_SILO
    PetscViewerType PETSC_VIEWER_NETCDF
    PetscViewerType PETSC_VIEWER_HDF5
    PetscViewerType PETSC_VIEWER_MATLAB


    ctypedef enum PetscViewerFormat:
        PETSC_VIEWER_DEFAULT
        PETSC_VIEWER_NATIVE
        PETSC_VIEWER_ASCII_MATLAB
        PETSC_VIEWER_ASCII_MATHEMATICA
        PETSC_VIEWER_ASCII_IMPL
        PETSC_VIEWER_ASCII_INFO
        PETSC_VIEWER_ASCII_INFO_DETAIL
        PETSC_VIEWER_ASCII_COMMON
        PETSC_VIEWER_ASCII_SYMMODU
        PETSC_VIEWER_ASCII_INDEX
        PETSC_VIEWER_ASCII_DENSE
        PETSC_VIEWER_ASCII_VTK
        PETSC_VIEWER_ASCII_VTK_CELL
        PETSC_VIEWER_ASCII_VTK_COORDS
        PETSC_VIEWER_ASCII_PCICE
        PETSC_VIEWER_ASCII_PYLITH
        PETSC_VIEWER_ASCII_PYLITH_LOCAL
        PETSC_VIEWER_DRAW_BASIC
        PETSC_VIEWER_DRAW_LG
        PETSC_VIEWER_DRAW_CONTOUR
        PETSC_VIEWER_DRAW_PORTS
        PETSC_VIEWER_ASCII_FACTOR_INFO
        PETSC_VIEWER_NOFORMAT

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
    int PetscViewerDestroy(PetscViewer)
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

    int PetscViewerSetFormat(PetscViewer,PetscViewerFormat)
    int PetscViewerGetFormat(PetscViewer,PetscViewerFormat*)
    int PetscViewerPushFormat(PetscViewer,PetscViewerFormat)
    int PetscViewerPopFormat(PetscViewer)

    int PetscViewerFileGetName(PetscViewer,char**)
    int PetscViewerFileSetName(PetscViewer,char[])
    int PetscViewerFileGetMode(PetscViewer,PetscFileMode*)
    int PetscViewerFileSetMode(PetscViewer,PetscFileMode)
    int PetscViewerFlush(PetscViewer)

    int PetscViewerDrawClear(PetscViewer)
    int PetscViewerDrawSetInfo(PetscViewer,char[],char[],int,int,int,int)

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
