cdef extern from * nogil:

    ctypedef double PetscLogDouble
    ctypedef struct PetscEventPerfInfo:
        int count
        PetscLogDouble flops, time
        PetscLogDouble numMessages
        PetscLogDouble messageLength
        PetscLogDouble numReductions

    int PetscLogDefaultBegin()
    int PetscLogAllBegin()
    int PetscLogView(PetscViewer)
    int PetscLogIsActive(PetscBool*)

    int PetscLogFlops(PetscLogDouble)
    int PetscGetFlops(PetscLogDouble*)
    int PetscGetCPUTime(PetscLogDouble*)
    int PetscMallocGetCurrentUsage(PetscLogDouble*)
    int PetscMemoryGetCurrentUsage(PetscLogDouble*)

    int PetscTime(PetscLogDouble*)
    int PetscTimeSubtract(PetscLogDouble*)
    int PetscTimeAdd(PetscLogDouble*)

    ctypedef int PetscLogStage
    int PetscLogStageRegister(char[],PetscLogStage*)
    int PetscLogStagePush(PetscLogStage)
    int PetscLogStagePop()
    int PetscLogStageSetActive(PetscLogStage,PetscBool)
    int PetscLogStageGetActive(PetscLogStage,PetscBool*)
    int PetscLogStageSetVisible(PetscLogStage,PetscBool)
    int PetscLogStageGetVisible(PetscLogStage,PetscBool*)
    int PetscLogStageGetId(char[],PetscLogStage*)

    ctypedef int PetscLogClass "PetscClassId"
    int PetscLogClassRegister"PetscClassIdRegister"(char[],PetscLogClass*)
    int PetscLogClassActivate"PetscLogEventActivateClass"(PetscLogClass)
    int PetscLogClassDeactivate"PetscLogEventDeactivateClass"(PetscLogClass)

    ctypedef int PetscLogEvent
    int PetscLogEventRegister(char[],PetscLogClass,PetscLogEvent*)
    int PetscLogEventBegin(PetscLogEvent,PetscObject,PetscObject,PetscObject,PetscObject)
    int PetscLogEventEnd(PetscLogEvent,PetscObject,PetscObject,PetscObject,PetscObject)

    int PetscLogEventActivate(PetscLogEvent)
    int PetscLogEventDeactivate(PetscLogEvent)
    int PetscLogEventSetActiveAll(PetscLogEvent,PetscBool)
    int PetscLogEventGetPerfInfo(PetscLogStage,PetscLogEvent,PetscEventPerfInfo*)

cdef extern from "custom.h" nogil:
    int PetscLogStageFindId(char[],PetscLogStage*)
    int PetscLogClassFindId(char[],PetscLogClass*)
    int PetscLogEventFindId(char[],PetscLogEvent*)
    int PetscLogStageFindName(PetscLogStage,char*[])
    int PetscLogClassFindName(PetscLogClass,char*[])
    int PetscLogEventFindName(PetscLogEvent,char*[])


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
