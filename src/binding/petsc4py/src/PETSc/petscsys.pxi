cdef extern from * nogil:

    ctypedef enum PetscDataType:
        PETSC_INT
        PETSC_REAL
        PETSC_SCALAR
        PETSC_COMPLEX
        PETSC_DATATYPE_UNKNOWN

    const char PETSC_AUTHOR_INFO[]
    PetscErrorCode PetscGetVersion(char[],size_t)
    PetscErrorCode PetscGetVersionNumber(PetscInt*,PetscInt*,PetscInt*,PetscInt*)

    PetscErrorCode PetscInitialize(int*,char***,char[],char[])
    PetscErrorCode PetscInitializeNoArguments()
    PetscErrorCode PetscFinalize()
    PetscBool PetscInitializeCalled
    PetscBool PetscFinalizeCalled

    ctypedef PetscErrorCode (*PetscErrorHandlerFunction)(
        MPI_Comm,int,char*,char*,int,PetscErrorType,char*,void*)
    PetscErrorHandlerFunction PetscAttachDebuggerErrorHandler
    PetscErrorHandlerFunction PetscEmacsClientErrorHandler
    PetscErrorHandlerFunction PetscTraceBackErrorHandler
    PetscErrorHandlerFunction PetscMPIAbortErrorHandler
    PetscErrorHandlerFunction PetscAbortErrorHandler
    PetscErrorHandlerFunction PetscIgnoreErrorHandler
    PetscErrorCode PetscPushErrorHandler(PetscErrorHandlerFunction,void*)
    PetscErrorCode PetscPopErrorHandler()
    PetscErrorCode PetscPopSignalHandler()
    PetscErrorCode PetscInfoAllow(PetscBool)
    PetscErrorCode PetscInfoSetFile(char*,char*)

    PetscErrorCode PetscErrorMessage(int,char*[],char**)

    PetscErrorCode PetscSplitOwnership(MPI_Comm,PetscInt*,PetscInt*)
    PetscErrorCode PetscSplitOwnershipBlock(MPI_Comm,PetscInt,PetscInt*,PetscInt*)

    FILE *PETSC_STDOUT
    FILE *PETSC_STDERR

    PetscErrorCode PetscPrintf(MPI_Comm,char[],...)
    PetscErrorCode PetscVSNPrintf(char*,size_t,const char[],size_t *,va_list)
    PetscErrorCode PetscVFPrintfDefault(FILE*,const char[],va_list)
    PetscErrorCode PetscSynchronizedPrintf(MPI_Comm,char[],...)
    PetscErrorCode PetscSynchronizedFlush(MPI_Comm,FILE*)

    PetscErrorCode PetscSequentialPhaseBegin(MPI_Comm,int)
    PetscErrorCode PetscSequentialPhaseEnd(MPI_Comm,int)
    PetscErrorCode PetscSleep(int)

    PetscErrorCode PetscCitationsRegister(const char[],PetscBool*)

    PetscErrorCode PetscHasExternalPackage(const char[],PetscBool*)


cdef inline PetscErrorCode Sys_Sizes(
    object size, object bsize,
    PetscInt *_b,
    PetscInt *_n,
    PetscInt *_N,
    ) except PETSC_ERR_PYTHON:
    # get block size
    cdef PetscInt bs=PETSC_DECIDE, b=PETSC_DECIDE
    if bsize is not None: bs = b = asInt(bsize)
    if bs == PETSC_DECIDE: bs = 1
    # unpack and get local and global sizes
    cdef PetscInt n=PETSC_DECIDE, N=PETSC_DECIDE
    cdef object on, oN
    try:
        on, oN = size
    except (TypeError, ValueError):
        on = None; oN = size
    if on is not None: n = asInt(on)
    if oN is not None: N = asInt(oN)
    # check block, local, and and global sizes
    if (bs < 1): raise ValueError(
        "block size %d must be positive" % toInt(bs))
    if n==PETSC_DECIDE and N==PETSC_DECIDE: raise ValueError(
        "local and global sizes cannot be both 'DECIDE'")
    if (n > 0) and (n % bs): raise ValueError(
        "local size %d not divisible by block size %d" %
        (toInt(n), toInt(bs)) )
    if (N > 0) and (N % bs): raise ValueError(
        "global size %d not divisible by block size %d" %
        (toInt(N), toInt(bs)) )
    # return result to the caller
    if _b != NULL: _b[0] = b
    if _n != NULL: _n[0] = n
    if _N != NULL: _N[0] = N
    return PETSC_SUCCESS

cdef inline PetscErrorCode Sys_Layout(
    MPI_Comm comm,
    PetscInt bs,
    PetscInt *_n,
    PetscInt *_N,
    ) except PETSC_ERR_PYTHON:
    cdef PetscInt n = _n[0]
    cdef PetscInt N = _N[0]
    if bs < 0: bs = 1
    if n  > 0: n = n // bs
    if N  > 0: N = N // bs
    CHKERR( PetscSplitOwnership(comm, &n, &N) )
    _n[0] = n * bs
    _N[0] = N * bs
    return PETSC_SUCCESS
