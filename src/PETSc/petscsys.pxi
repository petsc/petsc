cdef extern from * nogil:

    enum: PETSC_VERSION_MAJOR
    enum: PETSC_VERSION_MINOR
    enum: PETSC_VERSION_SUBMINOR
    enum: PETSC_VERSION_PATCH
    enum: PETSC_VERSION_RELEASE
    char* PETSC_VERSION_DATE
    char* PETSC_VERSION_PATCH_DATE
    char* PETSC_AUTHOR_INFO

    int PetscInitialize(int*,char***,char[],char[])
    int PetscInitializeNoArguments()
    int PetscFinalize()
    PetscBool PetscInitializeCalled
    PetscBool PetscFinalizeCalled

    ctypedef enum PetscErrorType:
        PETSC_ERROR_INITIAL
        PETSC_ERROR_REPEAT
    ctypedef int PetscEHF(MPI_Comm,
                          int,char*,char*,char*,
                          int,PetscErrorType,char*,void*)
    PetscEHF PetscAttachDebuggerErrorHandler
    PetscEHF PetscEmacsClientErrorHandler
    PetscEHF PetscTraceBackErrorHandler
    PetscEHF PetscMPIAbortErrorHandler
    PetscEHF PetscAbortErrorHandler
    PetscEHF PetscIgnoreErrorHandler
    int PetscPushErrorHandler(PetscEHF*,void*)
    int PetscPopErrorHandler()

    int PetscErrorMessage(int,char*[],char**)

    int PetscSplitOwnership(MPI_Comm,PetscInt*,PetscInt*)
    int PetscSplitOwnershipBlock(MPI_Comm,PetscInt,PetscInt*,PetscInt*)

    int PetscPrintf(MPI_Comm,char[],...)
    int PetscSynchronizedPrintf(MPI_Comm,char[],...)
    int PetscSynchronizedFlush(MPI_Comm)

    int PetscSequentialPhaseBegin(MPI_Comm,int)
    int PetscSequentialPhaseEnd(MPI_Comm,int)
    int PetscSleep(int)

cdef inline int Sys_SplitSizes(MPI_Comm comm, object size, object bsize,
                               PetscInt *_b, PetscInt *_n, PetscInt *_N) except -1:

    # get block size
    cdef PetscInt bs=PETSC_DECIDE, b=PETSC_DECIDE
    if bsize is not None: bs = asInt(bsize)
    if bs == PETSC_DECIDE: bs = 1
    else: b = bs
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
    # split ownership
    if n > 0: n = n // bs
    if N > 0: N = N // bs
    CHKERR( PetscSplitOwnership(comm, &n, &N) )
    n = n * bs
    N = N * bs
    # set result
    if _b != NULL: _b[0] = b
    if _n != NULL: _n[0] = n
    if _N != NULL: _N[0] = N
    return 0
