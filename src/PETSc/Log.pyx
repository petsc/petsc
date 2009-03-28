# --------------------------------------------------------------------

cdef class Log:

    @classmethod
    def Stage(cls, name):
        cdef char *cname = str2cp(name)
        cdef PetscLogStage stageid = 0
        if not name: raise ValueError("empty name")
        cdef LogStage stage = get_LogStage(name)
        if stage is not None: return stage
        try:
            CHKERR( PetscLogStageGetId(cname, &stageid) )
        except Error:
            del tracebacklist[:] # XXX this is really ugly
            CHKERR( PetscLogStageRegister(cname, &stageid) )
        stage = reg_LogStage(name, stageid)
        return stage

    @classmethod
    def Class(cls, name):
        cdef char *cname = str2cp(name)
        cdef PetscLogClass classid = 0
        if not name: raise ValueError("empty name")
        cdef LogClass klass = get_LogClass(name)
        if klass is not None: return klass
        CHKERR( PetscLogClassRegister(cname, &classid) )
        klass = reg_LogClass(name, classid)
        return klass

    @classmethod
    def Event(cls, name, klass=None):
        cdef char *cname = str2cp(name)
        cdef PetscLogClass classid = PETSC_OBJECT_COOKIE
        cdef PetscLogEvent eventid = 0
        if not name: raise ValueError("empty name")
        if klass is not None: classid = klass
        cdef LogEvent event = get_LogEvent(name)
        if event is not None: return event
        CHKERR( PetscLogEventRegister(cname, classid, &eventid) )
        event = reg_LogEvent(name, eventid)
        return event

    @classmethod
    def logFlops(cls, flops):
        cdef PetscLogDouble cflops=flops
        CHKERR( PetscLogFlops(cflops) )

    @classmethod
    def addFlops(cls, flops):
        cdef PetscLogDouble cflops=flops
        CHKERR( PetscLogFlops(cflops) )

    @classmethod
    def getFlops(cls):
        cdef PetscLogDouble cflops=0
        CHKERR( PetscGetFlops(&cflops) )
        return cflops

    @classmethod
    def getTime(cls):
        cdef PetscLogDouble wctime=0
        CHKERR( PetscGetTime(&wctime) )
        return wctime

    @classmethod
    def getCPUTime(cls):
        cdef PetscLogDouble cputime=0
        CHKERR( PetscGetCPUTime(&cputime) )
        return cputime

# --------------------------------------------------------------------

cdef class LogStage:

    cdef readonly object        name
    cdef readonly PetscLogStage id

    def __cinit__(self):
        self.name = cp2str("Main Stage")
        self.id   = 0

    def __int__(self):
        return <int> self.id

    def __enter__(self):
        self.push()
        return self

    def __exit__(self, *exc):
        self.pop()

    #

    def push(self):
        CHKERR( PetscLogStagePush(self.id) )

    def pop(self):
        CHKERR( PetscLogStagePop() )

    #

    def activate(self):
        CHKERR( PetscLogStageSetActive(self.id, PETSC_TRUE) )

    def deactivate(self):
        CHKERR( PetscLogStageSetActive(self.id, PETSC_FALSE) )

    def getActive(self):
        cdef PetscTruth tval = PETSC_FALSE
        CHKERR( PetscLogStageGetActive(self.id, &tval) )
        return <bint> tval

    def setActive(self, flag):
        cdef PetscTruth tval = PETSC_FALSE
        if flag: tval = PETSC_TRUE
        CHKERR( PetscLogStageSetActive(self.id, tval) )

    property active:
        def __get__(self):
            return self.getActive()
        def __set__(self, value):
            self.setActive(value)

    #

    def getVisible(self):
        cdef PetscTruth tval = PETSC_FALSE
        CHKERR( PetscLogStageGetVisible(self.id, &tval) )
        return <bint> tval

    def setVisible(self, flag):
        cdef PetscTruth tval = PETSC_FALSE
        if flag: tval = PETSC_TRUE
        CHKERR( PetscLogStageSetVisible(self.id, tval) )

    property visible:
        def __get__(self):
            return self.getVisible()
        def __set__(self, value):
            self.setVisible(value)


cdef dict stage_registry = { }

cdef LogStage get_LogStage(object name):
    return stage_registry.get(name)

cdef LogStage reg_LogStage(object name, PetscLogStage stageid):
    cdef LogStage stage = LogStage()
    stage.name = name
    stage.id = stageid
    stage_registry[name] = stage
    return stage

# --------------------------------------------------------------------

cdef class LogClass:

    cdef readonly object        name
    cdef readonly PetscLogClass id

    def __cinit__(self):
        self.name = cp2str("Object")
        self.id   = PETSC_OBJECT_COOKIE

    def __int__(self):
        return <int> self.id

    #

    def activate(self):
        CHKERR( PetscLogClassActivate(self.id) )

    def deactivate(self):
        CHKERR( PetscLogClassDeactivate(self.id) )

    def getActive(self):
        raise NotImplementedError

    def setActive(self, flag):
        if flag:
            CHKERR( PetscLogClassActivate(self.id) )
        else:
            CHKERR( PetscLogClassDeactivate(self.id) )

    property active:
        def __get__(self):
            return self.getActive()
        def __set__(self, value):
            self.setActive(value)


cdef dict class_registry = { }

cdef LogClass get_LogClass(object name):
    return class_registry.get(name)

cdef LogClass reg_LogClass(object name, PetscLogClass classid):
    cdef LogClass klass = LogClass()
    klass.name = name
    klass.id = classid
    class_registry[name] = klass
    return klass

# --------------------------------------------------------------------

cdef class LogEvent:

    cdef readonly object        name
    cdef readonly PetscLogEvent id

    def __cinit__(self):
        self.name = cp2str("")
        self.id   = 0

    def __int__(self):
        return <int> self.id

    def __enter__(self):
        self.begin()
        return self

    def __exit__(self, *exc):
        self.end()

    #

    def begin(self, *objs):
        cdef PetscObject o[4]
        event_args2objs(objs, o)
        CHKERR( PetscLogEventBegin(self.id, o[0], o[1], o[2], o[3]) )

    def end(self, *objs):
        cdef PetscObject o[4]
        event_args2objs(objs, o)
        CHKERR( PetscLogEventEnd(self.id, o[0], o[1], o[2], o[3]) )

    def barrierBegin(self, Comm comm=None, *objs):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscObject o[4]
        event_args2objs(objs, o)
        CHKERR( PetscLogEventBarrierBegin(self.id, o[0], o[1], o[2], o[3], ccomm) )

    def barrierEnd(self, Comm comm=None, *objs):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscObject o[4]
        event_args2objs(objs, o)
        CHKERR( PetscLogEventBarrierEnd(self.id, o[0], o[1], o[2], o[3], ccomm) )

    #

    def activate(self):
        CHKERR( PetscLogEventActivate(self.id) )

    def deactivate(self):
        CHKERR( PetscLogEventDeactivate(self.id) )

    def getActive(self):
        raise NotImplementedError

    def setActive(self, flag):
        if flag:
            CHKERR( PetscLogEventActivate(self.id) )
        else:
            CHKERR( PetscLogEventDeactivate(self.id) )

    property active:
        def __get__(self):
            return self.getActive()
        def __set__(self, value):
            self.setActive(value)

    def setActiveAll(self, flag):
        cdef PetscTruth tval = PETSC_FALSE
        if flag: tval = PETSC_TRUE
        CHKERR( PetscLogEventSetActiveAll(self.id, tval) )

    property active_all:
        def __get__(self):
            raise NotImplementedError
        def __set__(self, value):
            self.setActiveAll(value)


cdef dict event_registry = { }

cdef LogEvent get_LogEvent(object name):
    return event_registry.get(name)

cdef LogEvent reg_LogEvent(object name, PetscLogEvent eventid):
    cdef LogEvent event = LogEvent()
    event.name = name
    event.id = eventid
    event_registry[name] = event
    return event

# --------------------------------------------------------------------
