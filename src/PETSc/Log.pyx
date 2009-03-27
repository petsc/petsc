# --------------------------------------------------------------------

cdef class Log:

    @classmethod
    def Stage(cls, name):
        cdef char *cname = str2cp(name)
        cdef PetscLogStage stageid = 0
        cdef LogStage stage = None
        if not name: raise ValueError("empty name")
        try:
            stage = stage_registry[name]
        except KeyError:
            try:
                CHKERR( PetscLogStageGetId(cname, &stageid) )
            except Error:
                del tracebacklist[:] # XXX this is really ugly
                CHKERR( PetscLogStageRegister(cname, &stageid) )
            stage = LogStage()
            stage.name = name
            stage.id = stageid
            stage_registry[name] = stage
        return stage

    @classmethod
    def Event(cls, name, klass=None):
        cdef char *cname = str2cp(name)
        cdef PetscCookie cookie = PETSC_OBJECT_COOKIE
        cdef PetscLogEvent eventid = 0
        cdef LogEvent event = None
        if not name: raise ValueError("empty name")
        if klass is not None: cookie = klass
        try:
            event = event_registry[name]
        except KeyError:
            CHKERR( PetscLogEventRegister(cname, cookie, &eventid) )
            event = LogEvent()
            event.name = name
            event.id = eventid
            event_registry[name] = event
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


cdef LogStage MainStage  = LogStage()
cdef dict stage_registry = { MainStage.name : MainStage }

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

# --------------------------------------------------------------------
