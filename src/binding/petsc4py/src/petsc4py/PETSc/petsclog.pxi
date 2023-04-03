cdef extern from * nogil:

    ctypedef double PetscLogDouble
    ctypedef struct PetscEventPerfInfo:
        int count
        PetscLogDouble flops, time
        PetscLogDouble numMessages
        PetscLogDouble messageLength
        PetscLogDouble numReductions

    PetscErrorCode PetscLogDefaultBegin()
    PetscErrorCode PetscLogAllBegin()
    PetscErrorCode PetscLogView(PetscViewer)
    PetscErrorCode PetscLogIsActive(PetscBool*)

    PetscErrorCode PetscLogFlops(PetscLogDouble)
    PetscErrorCode PetscGetFlops(PetscLogDouble*)
    PetscErrorCode PetscGetCPUTime(PetscLogDouble*)
    PetscErrorCode PetscMallocGetCurrentUsage(PetscLogDouble*)
    PetscErrorCode PetscMemoryGetCurrentUsage(PetscLogDouble*)

    PetscErrorCode PetscTime(PetscLogDouble*)
    PetscErrorCode PetscTimeSubtract(PetscLogDouble*)
    PetscErrorCode PetscTimeAdd(PetscLogDouble*)

    ctypedef int PetscLogStage
    PetscErrorCode PetscLogStageRegister(char[],PetscLogStage*)
    PetscErrorCode PetscLogStagePush(PetscLogStage)
    PetscErrorCode PetscLogStagePop()
    PetscErrorCode PetscLogStageSetActive(PetscLogStage,PetscBool)
    PetscErrorCode PetscLogStageGetActive(PetscLogStage,PetscBool*)
    PetscErrorCode PetscLogStageSetVisible(PetscLogStage,PetscBool)
    PetscErrorCode PetscLogStageGetVisible(PetscLogStage,PetscBool*)
    PetscErrorCode PetscLogStageGetId(char[],PetscLogStage*)

    ctypedef int PetscLogClass "PetscClassId"
    PetscErrorCode PetscLogClassRegister"PetscClassIdRegister"(char[],PetscLogClass*)
    PetscErrorCode PetscLogClassActivate"PetscLogEventActivateClass"(PetscLogClass)
    PetscErrorCode PetscLogClassDeactivate"PetscLogEventDeactivateClass"(PetscLogClass)

    ctypedef int PetscLogEvent
    PetscErrorCode PetscLogEventRegister(char[],PetscLogClass,PetscLogEvent*)
    PetscErrorCode PetscLogEventBegin(PetscLogEvent,PetscObject,PetscObject,PetscObject,PetscObject)
    PetscErrorCode PetscLogEventEnd(PetscLogEvent,PetscObject,PetscObject,PetscObject,PetscObject)

    PetscErrorCode PetscLogEventActivate(PetscLogEvent)
    PetscErrorCode PetscLogEventDeactivate(PetscLogEvent)
    PetscErrorCode PetscLogEventSetActiveAll(PetscLogEvent,PetscBool)
    PetscErrorCode PetscLogEventGetPerfInfo(PetscLogStage,PetscLogEvent,PetscEventPerfInfo*)

cdef extern from * nogil: # custom.h
    PetscErrorCode PetscLogStageFindId(char[],PetscLogStage*)
    PetscErrorCode PetscLogClassFindId(char[],PetscLogClass*)
    PetscErrorCode PetscLogEventFindId(char[],PetscLogEvent*)
    PetscErrorCode PetscLogStageFindName(PetscLogStage,char*[])
    PetscErrorCode PetscLogClassFindName(PetscLogClass,char*[])
    PetscErrorCode PetscLogEventFindName(PetscLogEvent,char*[])


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
