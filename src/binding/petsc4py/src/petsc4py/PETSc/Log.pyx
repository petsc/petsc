# --------------------------------------------------------------------

cdef object functools = None
import functools

# --------------------------------------------------------------------

cdef class Log:

    @classmethod
    def Stage(cls, name):
        if not name: raise ValueError("empty name")
        cdef const char *cname = NULL
        name = str2bytes(name, &cname)
        cdef PetscLogStage stageid = -1
        cdef LogStage stage = get_LogStage(name)
        if stage is not None: return stage
        CHKERR( PetscLogStageFindId(cname, &stageid) )
        if stageid == -1:
            CHKERR( PetscLogStageRegister(cname, &stageid) )
        stage = reg_LogStage(name, stageid)
        return stage

    @classmethod
    def Class(cls, name):
        if not name: raise ValueError("empty name")
        cdef const char *cname = NULL
        name = str2bytes(name, &cname)
        cdef PetscLogClass classid = -1
        cdef LogClass klass = get_LogClass(name)
        if klass is not None: return klass
        CHKERR( PetscLogClassFindId(cname, &classid) )
        if classid == -1:
            CHKERR( PetscLogClassRegister(cname, &classid) )
        klass = reg_LogClass(name, classid)
        return klass

    @classmethod
    def Event(cls, name, klass=None):
        if not name: raise ValueError("empty name")
        cdef const char *cname = NULL
        name = str2bytes(name, &cname)
        cdef PetscLogClass classid = PETSC_OBJECT_CLASSID
        cdef PetscLogEvent eventid = -1
        if klass is not None: classid = klass
        cdef LogEvent event = get_LogEvent(name)
        if event is not None: return event
        CHKERR( PetscLogEventFindId(cname, &eventid) )
        if eventid == -1:
            CHKERR( PetscLogEventRegister(cname, classid, &eventid) )
        event = reg_LogEvent(name, eventid)
        return event

    @classmethod
    def begin(cls, all: bool = False):
        """Turn on logging of objects and events.

        Logically collective.

        Parameters
        ----------
        all
            Whether to enable extensive logging.

        Notes
        -----
        If ``all == True`` logging is extensive, which creates large log files and shows the program down.

        If ``all == False``, the default logging functions are used.
        This logs flop rates and object creation and should not slow programs down too much. This routine may be called more than once.

        See Also
        --------
        petsc.PetscLogAllBegin, petsc.PetscLogDefaultBegin

        """
        if all: CHKERR( PetscLogAllBegin() )
        else:   CHKERR( PetscLogDefaultBegin() )

    @classmethod
    def view(cls, Viewer viewer=None) -> None:
        """Print the log.

        Collective.

        Parameters
        ----------
        viewer
            Viewer instance. If `None` then will default to an instance of `Viewer.Type.ASCII`.

        See Also
        --------
        petsc_options, petsc.PetscLogView

        """
        cdef PetscViewer vwr = NULL
        if viewer is not None: vwr = viewer.vwr
        if vwr == NULL: vwr = PETSC_VIEWER_STDOUT_WORLD
        CHKERR( PetscLogView(vwr) )

    @classmethod
    def logFlops(cls, flops: float) -> None:
        """Add floating point operations to the global counter.

        Not collective.

        Parameters
        ----------
        flops
            The number of flops to log.

        See Also
        --------
        petsc.PetscLogFlops

        """
        cdef PetscLogDouble cflops=flops
        CHKERR( PetscLogFlops(cflops) )

    @classmethod
    def addFlops(cls, flops: float) -> None:
        """Add floating point operations to global counter.

        Not collective.

        Parameters
        ----------
        flops
            The number of flops to log.

        Notes
        -----
        This method exists for backward compatibility.

        See Also
        --------
        logFlops, petsc.PetscLogFlops

        """
        cdef PetscLogDouble cflops=flops
        CHKERR( PetscLogFlops(cflops) )

    @classmethod
    def getFlops(cls) -> float:
        """Return the number of flops used on this processor since the program began.

        Not collective.

        Returns
        -------
        float
            Number of floating point operations.

        See Also
        --------
        petsc.PetscGetFlops

        """
        cdef PetscLogDouble cflops=0
        CHKERR( PetscGetFlops(&cflops) )
        return cflops

    @classmethod
    def getTime(cls) -> float:
        """Return the current time of day in seconds.

        Collective.

        Returns
        -------
        wctime : float
            Current time.

        See Also
        --------
        petsc.PetscTime

        """
        cdef PetscLogDouble wctime=0
        CHKERR( PetscTime(&wctime) )
        return wctime

    @classmethod
    def getCPUTime(cls) -> float:
        """Return the CPU time."""
        cdef PetscLogDouble cputime=0
        CHKERR( PetscGetCPUTime(&cputime) )
        return cputime

    @classmethod
    def EventDecorator(cls, name=None, klass=None):
        """Decorate a function with a `PETSc` event."""
        def decorator(func):
            @functools.wraps(func)
            def wrapped_func(*args, **kwargs):
                if name:
                    name_ = name
                else:
                    name_ = ".".join([func.__module__, getattr(func, "__qualname__", func.__name__)])
                with cls.Event(name_, klass):
                    return func(*args, **kwargs)
            return wrapped_func
        return decorator

    @classmethod
    def isActive(cls) -> bool:
        """Return whether logging is currently in progress.

        Not collective.

        See Also
        --------
        petsc.PetscLogIsActive

        """
        cdef PetscBool flag = PETSC_FALSE
        CHKERR( PetscLogIsActive(&flag) )
        return toBool(flag)

# --------------------------------------------------------------------

cdef class LogStage:

    cdef readonly PetscLogStage id

    def __cinit__(self):
        self.id = 0

    def __int__(self):
        return <int> self.id

    def __enter__(self):
        self.push()
        return self

    def __exit__(self, *exc):
        self.pop()

    #

    def push(self) -> None:
        """Push a stage on the logging stack.

        Not collective.

        Notes
        -----
        Events started and stopped until LogStage.pop will be associated with the stage.

        See Also
        --------
        LogStage.pop, petsc.PetscLogStagePush

        """
        CHKERR( PetscLogStagePush(self.id) )

    def pop(self) -> None:
        """Pop a stage on the logging stack that was pushed.

        Not collective.

        See Also
        --------
        LogStage.push, petsc.PetscLogStagePop

        """
        <void>self # unused
        CHKERR( PetscLogStagePop() )

    #

    def getName(self):
        cdef const char *cval = NULL
        CHKERR( PetscLogStageFindName(self.id, &cval) )
        return bytes2str(cval)

    property name:
        def __get__(self):
            return self.getName()
        def __set__(self, value):
            <void>self; <void>value; # unused
            raise TypeError("readonly attribute")

    #

    def activate(self) -> None:
        """Activate the stage.

        Not collective.

        See Also
        --------
        petsc.PetscLogStageSetActive

        """
        CHKERR( PetscLogStageSetActive(self.id, PETSC_TRUE) )

    def deactivate(self) -> None:
        """Deactivate the stage.

        Not collective.

        See Also
        --------
        petsc.PetscLogStageSetActive

        """
        CHKERR( PetscLogStageSetActive(self.id, PETSC_FALSE) )

    def getActive(self) -> bool:
        """Check if the stage is activated.

        Not collective.

        See Also
        --------
        petsc.PetscLogStageGetActive

        """
        cdef PetscBool flag = PETSC_FALSE
        CHKERR( PetscLogStageGetActive(self.id, &flag) )
        return toBool(flag)

    def setActive(self, flag: bool) -> None:
        """Activate or deactivate the current stage.

        Not collective.

        Parameters
        ----------
        flag
            Log if `True`, disable logging if `False`.

        See Also
        --------
        petsc.PetscLogStageSetActive

        """
        cdef PetscBool tval = PETSC_FALSE
        if flag: tval = PETSC_TRUE
        CHKERR( PetscLogStageSetActive(self.id, tval) )

    property active:
        def __get__(self):
            return self.getActive()
        def __set__(self, value):
            self.setActive(value)

    #

    def getVisible(self) -> bool:
        """Return whether the stage is visible.

        Not collective.

        See Also
        --------
        LogStage.setVisible, petsc.PetscLogStageSetVisible

        """
        cdef PetscBool flag = PETSC_FALSE
        CHKERR( PetscLogStageGetVisible(self.id, &flag) )
        return toBool(flag)

    def setVisible(self, flag: bool) -> None:
        """Set the visibility of the stage.

        Not collective.

        Parameters
        ----------
        flag
            `True` to make the stage visible, `False` otherwise.

        See Also
        --------
        LogStage.getVisible, petsc.PetscLogStageSetVisible

        """
        cdef PetscBool tval = PETSC_FALSE
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
    stage.id = stageid
    stage_registry[name] = stage
    return stage

# --------------------------------------------------------------------

cdef class LogClass:

    cdef readonly PetscLogClass id

    def __cinit__(self):
        self.id = PETSC_OBJECT_CLASSID

    def __int__(self):
        return <int> self.id

    #

    def getName(self):
        cdef const char *cval = NULL
        CHKERR( PetscLogClassFindName(self.id, &cval) )
        return bytes2str(cval)

    property name:
        def __get__(self):
            return self.getName()
        def __set__(self, value):
            <void>self; <void>value; # unused
            raise TypeError("readonly attribute")

    #

    def activate(self):
        CHKERR( PetscLogClassActivate(self.id) )

    def deactivate(self):
        CHKERR( PetscLogClassDeactivate(self.id) )

    def getActive(self):
        <void>self # unused
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
    klass.id = classid
    class_registry[name] = klass
    return klass

# --------------------------------------------------------------------

cdef class LogEvent:

    cdef readonly PetscLogEvent id

    def __cinit__(self):
        self.id = 0

    def __int__(self):
        return <int> self.id

    def __enter__(self):
        self.begin()
        return self

    def __exit__(self, *exc):
        self.end()



    def begin(self, *objs) -> None:
        """Log the beginning of a user event.

        Not collective.

        Parameters
        ----------
        *objs
            objects associated with the event

        See Also
        --------
        petsc.PetscLogEventBegin

        """
        cdef PetscObject o[4]
        event_args2objs(objs, o)
        CHKERR( PetscLogEventBegin(self.id, o[0], o[1], o[2], o[3]) )

    def end(self, *objs) -> None:
        """Log the end of a user event.

        Not collective.

        Parameters
        ----------
        *objs
            Objects associated with the event.

        See Also
        --------
        petsc.PetscLogEventEnd

        """
        cdef PetscObject o[4]
        event_args2objs(objs, o)
        CHKERR( PetscLogEventEnd(self.id, o[0], o[1], o[2], o[3]) )

    #
    def getName(self):
        cdef const char *cval = NULL
        CHKERR( PetscLogEventFindName(self.id, &cval) )
        return bytes2str(cval)

    property name:
        def __get__(self):
            return self.getName()
        def __set__(self, value):
            <void>self; <void>value; # unused
            raise TypeError("readonly attribute")

    #

    def activate(self) -> None:
        """Indicate that the event should be logged.

        Not collective.

        See Also
        --------
        petsc.PetscLogEventActivate

        """
        CHKERR( PetscLogEventActivate(self.id) )

    def deactivate(self) -> None:
        """Indicate that the event should not be logged.

        Not collective.

        See Also
        --------
        petsc.PetscLogEventDeactivate

        """
        CHKERR( PetscLogEventDeactivate(self.id) )

    def getActive(self):
        <void>self # unused
        raise NotImplementedError

    def setActive(self, flag: bool) -> None:
        """Indicate whether or not the event should be logged.

        Not collective.

        Parameters
        ----------
        flag
            Activate or deactivate the event.

        See Also
        --------
        petsc.PetscLogEventDeactivate, petsc.PetscLogEventActivate

        """
        if flag:
            CHKERR( PetscLogEventActivate(self.id) )
        else:
            CHKERR( PetscLogEventDeactivate(self.id) )

    property active:
        def __get__(self):
            return self.getActive()
        def __set__(self, value):
            self.setActive(value)

    def getActiveAll(self):
        <void>self # unused
        raise NotImplementedError

    def setActiveAll(self, flag: bool) -> None:
        """Turn on logging of all events.

        Parameters
        ----------
        flag
            Activate (if `True`) or deactivate (if `False`) the logging of all events.

        See Also
        --------
        petsc.PetscLogEventSetActiveAll

        """
        cdef PetscBool tval = PETSC_FALSE
        if flag: tval = PETSC_TRUE
        CHKERR( PetscLogEventSetActiveAll(self.id, tval) )

    property active_all:
        def __get__(self):
            self.getActiveAll()
        def __set__(self, value):
            self.setActiveAll(value)

    #

    def getPerfInfo(self, stage: int | None = None) -> dict:
        """Get the performance information about the given event in the given event.

        Parameters
        ----------
        stage
            The stage number.

        Returns
        -------
        info : dict
            This structure is filled with the performance information.

        See Also
        --------
        petsc.PetscLogEventGetPerfInfo

        """
        cdef PetscEventPerfInfo info
        cdef PetscInt cstage = PETSC_DETERMINE
        if stage is not None: cstage = asInt(stage)
        CHKERR( PetscLogEventGetPerfInfo(cstage, self.id, &info) )
        return info

cdef dict event_registry = { }

cdef LogEvent get_LogEvent(object name):
    return event_registry.get(name)

cdef LogEvent reg_LogEvent(object name, PetscLogEvent eventid):
    cdef LogEvent event = LogEvent()
    event.id = eventid
    event_registry[name] = event
    return event

# --------------------------------------------------------------------
