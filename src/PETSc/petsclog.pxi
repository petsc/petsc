cdef extern from "petsc.h" nogil:

    ctypedef int PetscCookie


cdef extern from "petsc.h" nogil:

    ctypedef double PetscLogDouble
    int PetscLogFlops(PetscLogDouble)
    int PetscGetFlops(PetscLogDouble*)
    int PetscGetTime(PetscLogDouble*)
    int PetscGetCPUTime(PetscLogDouble*)
    int PetscMallocGetCurrentUsage(PetscLogDouble*)
    int PetscMemoryGetCurrentUsage(PetscLogDouble*)


    ctypedef int PetscLogStage
    int PetscLogStageRegister(char[],PetscLogStage*)
    int PetscLogStagePush(PetscLogStage)
    int PetscLogStagePop()
    int PetscLogStageSetActive(PetscLogStage,PetscTruth)
    int PetscLogStageGetActive(PetscLogStage,PetscTruth*)
    int PetscLogStageSetVisible(PetscLogStage,PetscTruth)
    int PetscLogStageGetVisible(PetscLogStage,PetscTruth*)
    int PetscLogStageGetId(char[],PetscLogStage*)

    ctypedef int PetscLogEvent
    int PetscLogEventRegister(char[],PetscCookie,PetscLogEvent*)
    int PetscLogEventBegin(PetscLogEvent,PetscObject,PetscObject,PetscObject,PetscObject)
    int PetscLogEventEnd(PetscLogEvent,PetscObject,PetscObject,PetscObject,PetscObject)
    int PetscLogEventBarrierBegin(PetscLogEvent,PetscObject,PetscObject,PetscObject,PetscObject,MPI_Comm)
    int PetscLogEventBarrierEnd(PetscLogEvent,PetscObject,PetscObject,PetscObject,PetscObject,MPI_Comm)

    int PetscLogEventActivate(PetscLogEvent)
    int PetscLogEventDeactivate(PetscLogEvent)
    int PetscLogEventSetActiveAll(PetscLogEvent,PetscTruth)
    #int PetscLogEventActivateClass(PetscCookie)
    #int PetscLogEventDeactivateClass(PetscCookie)

cdef inline int event_args2objs(object args, PetscObject o[4]) except -1:
        o[0] = o[1] = o[2] = o[3] = NULL
        cdef Py_ssize_t i=0, n = len(args)
        cdef Object tmp = None
        if n > 4: n = 4
        for 0 <= i < n:
            tmp = args[i]
            if tmp is not None:
                o[i] = tmp.obj[0]
        return 0
