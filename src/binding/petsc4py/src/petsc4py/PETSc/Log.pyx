# --------------------------------------------------------------------

cdef object functools = None
import functools

# --------------------------------------------------------------------

cdef class Log:
    """Logging support."""

    @classmethod
    def Stage(cls, name: str) -> LogStage:
        """Create a log stage.

        Not collective.

        Parameters
        ----------
        name
            Stage name.

        Returns
        -------
        stage : LogStage
            The log stage. If a stage already exists with name ``name`` then
            it is reused.

        See Also
        --------
        petsc.PetscLogStageRegister

        """
        if not name: raise ValueError("empty name")
        cdef const char *cname = NULL
        name = str2bytes(name, &cname)
        cdef PetscLogStage stageid = -1
        cdef LogStage stage = get_LogStage(name)
        if stage is not None: return stage
        CHKERR(PetscLogStageFindId(cname, &stageid))
        if stageid == -1:
            CHKERR(PetscLogStageRegister(cname, &stageid))
        stage = reg_LogStage(name, stageid)
        return stage

    @classmethod
    def Class(cls, name: str) -> LogClass:
        """Create a log class.

        Not collective.

        Parameters
        ----------
        name
            Class name.

        Returns
        -------
        klass : LogClass
            The log class. If a class already exists with name ``name`` then
            it is reused.

        See Also
        --------
        petsc.PetscClassIdRegister

        """
        if not name: raise ValueError("empty name")
        cdef const char *cname = NULL
        name = str2bytes(name, &cname)
        cdef PetscLogClass classid = -1
        cdef LogClass klass = get_LogClass(name)
        if klass is not None: return klass
        CHKERR(PetscLogClassFindId(cname, &classid))
        if classid == -1:
            CHKERR(PetscLogClassRegister(cname, &classid))
        klass = reg_LogClass(name, classid)
        return klass

    @classmethod
    def Event(cls, name: str, klass: LogClass | None = None) -> LogEvent:
        """Create a log event.

        Not collective.

        Parameters
        ----------
        name
            Event name.
        klass
            Log class. If `None`, defaults to ``PETSC_OBJECT_CLASSID``.

        Returns
        -------
        event : LogEvent
            The log event. If an event already exists with name ``name`` then
            it is reused.

        See Also
        --------
        petsc.PetscLogEventRegister

        """
        if not name: raise ValueError("empty name")
        cdef const char *cname = NULL
        name = str2bytes(name, &cname)
        cdef PetscLogClass classid = PETSC_OBJECT_CLASSID
        cdef PetscLogEvent eventid = -1
        if klass is not None: classid = klass
        cdef LogEvent event = get_LogEvent(name)
        if event is not None: return event
        CHKERR(PetscLogEventFindId(cname, &eventid))
        if eventid == -1:
            CHKERR(PetscLogEventRegister(cname, classid, &eventid))
        event = reg_LogEvent(name, eventid)
        return event

    @classmethod
    def begin(cls) -> None:
        """Turn on logging of objects and events.

        Collective.

        See Also
        --------
        petsc.PetscLogDefaultBegin

        """
        CHKERR(PetscLogDefaultBegin())

    @classmethod
    def view(cls, Viewer viewer=None) -> None:
        """Print the log.

        Collective.

        Parameters
        ----------
        viewer
            A `Viewer` instance or `None` for the default viewer.

        See Also
        --------
        petsc_options, petsc.PetscLogView

        """
        cdef PetscViewer vwr = NULL
        if viewer is not None: vwr = viewer.vwr
        if vwr == NULL: vwr = PETSC_VIEWER_STDOUT_WORLD
        CHKERR(PetscLogView(vwr))

    @classmethod
    def logFlops(cls, flops: float) -> None:
        """Add floating point operations to the current event.

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
        CHKERR(PetscLogFlops(cflops))

    @classmethod
    def addFlops(cls, flops: float) -> None:
        """Add floating point operations to the current event.

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
        CHKERR(PetscLogFlops(cflops))

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
        CHKERR(PetscGetFlops(&cflops))
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
        CHKERR(PetscTime(&wctime))
        return wctime

    @classmethod
    def getCPUTime(cls) -> float:
        """Return the CPU time."""
        cdef PetscLogDouble cputime=0
        CHKERR(PetscGetCPUTime(&cputime))
        return cputime

    @classmethod
    def EventDecorator(cls, name=None, klass=None) -> Any:
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
        CHKERR(PetscLogIsActive(&flag))
        return toBool(flag)

# --------------------------------------------------------------------

cdef class LogStage:
    """Logging support for different stages."""

    cdef PetscLogStage id

    property id:
        """The log stage identifier."""
        def __get__(self) -> int:
            return self.id

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

        Logically collective.

        See Also
        --------
        LogStage.pop, petsc.PetscLogStagePush

        """
        CHKERR(PetscLogStagePush(self.id))

    def pop(self) -> None:
        """Pop a stage from the logging stack.

        Logically collective.

        See Also
        --------
        LogStage.push, petsc.PetscLogStagePop

        """
        <void>self # unused
        CHKERR(PetscLogStagePop())

    #

    def getName(self) -> str:
        """Return the current stage name."""
        cdef const char *cval = NULL
        CHKERR(PetscLogStageFindName(self.id, &cval))
        return bytes2str(cval)

    property name:
        """The current stage name."""
        def __get__(self) -> str:
            return self.getName()

        def __set__(self, value):
            <void>self; <void>value # unused
            raise TypeError("readonly attribute")

    #

    def activate(self) -> None:
        """Activate the stage.

        Logically collective.

        See Also
        --------
        petsc.PetscLogStageSetActive

        """
        CHKERR(PetscLogStageSetActive(self.id, PETSC_TRUE))

    def deactivate(self) -> None:
        """Deactivate the stage.

        Logically collective.

        See Also
        --------
        petsc.PetscLogStageSetActive

        """
        CHKERR(PetscLogStageSetActive(self.id, PETSC_FALSE))

    def getActive(self) -> bool:
        """Check if the stage is activated.

        Not collective.

        See Also
        --------
        petsc.PetscLogStageGetActive

        """
        cdef PetscBool flag = PETSC_FALSE
        CHKERR(PetscLogStageGetActive(self.id, &flag))
        return toBool(flag)

    def setActive(self, flag: bool) -> None:
        """Activate or deactivate the current stage.

        Logically collective.

        See Also
        --------
        petsc.PetscLogStageSetActive

        """
        cdef PetscBool tval = PETSC_FALSE
        if flag: tval = PETSC_TRUE
        CHKERR(PetscLogStageSetActive(self.id, tval))

    property active:
        """Whether the stage is activate."""
        def __get__(self) -> bool:
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
        CHKERR(PetscLogStageGetVisible(self.id, &flag))
        return toBool(flag)

    def setVisible(self, flag: bool) -> None:
        """Set the visibility of the stage.

        Logically collective.

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
        CHKERR(PetscLogStageSetVisible(self.id, tval))

    property visible:
        """Whether the stage is visible."""
        def __get__(self) -> bool:
            return self.getVisible()

        def __set__(self, value):
            self.setVisible(value)


cdef dict stage_registry = {}

cdef LogStage get_LogStage(object name):
    return stage_registry.get(name)

cdef LogStage reg_LogStage(object name, PetscLogStage stageid):
    cdef LogStage stage = LogStage()
    stage.id = stageid
    stage_registry[name] = stage
    return stage

# --------------------------------------------------------------------

cdef class LogClass:
    """Logging support."""

    cdef PetscLogClass id

    property id:
        """The log class identifier."""
        def __get__(self) -> int:
            return self.id

    def __cinit__(self):
        self.id = PETSC_OBJECT_CLASSID

    def __int__(self):
        return <int> self.id

    #

    def getName(self) -> str:
        """Return the log class name."""
        cdef const char *cval = NULL
        CHKERR(PetscLogClassFindName(self.id, &cval))
        return bytes2str(cval)

    property name:
        """The log class name."""
        def __get__(self) -> str:
            return self.getName()

        def __set__(self, value):
            <void>self; <void>value # unused
            raise TypeError("readonly attribute")

    #

    def activate(self) -> None:
        """Activate the log class."""
        CHKERR(PetscLogClassActivate(self.id))

    def deactivate(self) -> None:
        """Deactivate the log class."""
        CHKERR(PetscLogClassDeactivate(self.id))

    def getActive(self) -> bool:
        """Not implemented."""
        <void>self # unused
        raise NotImplementedError

    def setActive(self, flag: bool) -> None:
        """Activate or deactivate the log class."""
        if flag:
            CHKERR(PetscLogClassActivate(self.id))
        else:
            CHKERR(PetscLogClassDeactivate(self.id))

    property active:
        """Log class activation."""
        def __get__(self) -> bool:
            return self.getActive()

        def __set__(self, value):
            self.setActive(value)


cdef dict class_registry = {}

cdef LogClass get_LogClass(object name):
    return class_registry.get(name)

cdef LogClass reg_LogClass(object name, PetscLogClass classid):
    cdef LogClass klass = LogClass()
    klass.id = classid
    class_registry[name] = klass
    return klass

# --------------------------------------------------------------------

cdef class LogEvent:
    """Logging support."""

    cdef PetscLogEvent id

    property id:
        """The log event identifier."""
        def __get__(self) -> int:
            return self.id

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

        Collective.

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
        CHKERR(PetscLogEventBegin(self.id, o[0], o[1], o[2], o[3]))

    def end(self, *objs) -> None:
        """Log the end of a user event.

        Collective.

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
        CHKERR(PetscLogEventEnd(self.id, o[0], o[1], o[2], o[3]))

    #
    def getName(self) -> str:
        """The current event name."""
        cdef const char *cval = NULL
        CHKERR(PetscLogEventFindName(self.id, &cval))
        return bytes2str(cval)

    property name:
        """The current event name."""
        def __get__(self) ->str:
            return self.getName()

        def __set__(self, value):
            <void>self; <void>value # unused
            raise TypeError("readonly attribute")

    #

    def activate(self) -> None:
        """Indicate that the event should be logged.

        Logically collective.

        See Also
        --------
        petsc.PetscLogEventActivate

        """
        CHKERR(PetscLogEventActivate(self.id))

    def deactivate(self) -> None:
        """Indicate that the event should not be logged.

        Logically collective.

        See Also
        --------
        petsc.PetscLogEventDeactivate

        """
        CHKERR(PetscLogEventDeactivate(self.id))

    def getActive(self) -> bool:
        """Not implemented."""
        <void>self # unused
        raise NotImplementedError

    def setActive(self, flag: bool) -> None:
        """Indicate whether or not the event should be logged.

        Logically collective.

        Parameters
        ----------
        flag
            Activate or deactivate the event.

        See Also
        --------
        petsc.PetscLogEventDeactivate, petsc.PetscLogEventActivate

        """
        if flag:
            CHKERR(PetscLogEventActivate(self.id))
        else:
            CHKERR(PetscLogEventDeactivate(self.id))

    property active:
        """Event activation."""
        def __get__(self) -> bool:
            return self.getActive()

        def __set__(self, value):
            self.setActive(value)

    def getActiveAll(self) -> bool:
        """Not implemented."""
        <void>self # unused
        raise NotImplementedError

    def setActiveAll(self, flag: bool) -> None:
        """Turn on logging of all events.

        Logically collective.

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
        CHKERR(PetscLogEventSetActiveAll(self.id, tval))

    property active_all:
        """All events activation."""
        def __get__(self) -> bool:
            self.getActiveAll()

        def __set__(self, value):
            self.setActiveAll(value)

    #

    def getPerfInfo(self, stage: int | None = None) -> dict:
        """Get the performance information about the given event in the given event.

        Not collective.

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
        CHKERR(PetscLogEventGetPerfInfo(<int>cstage, self.id, &info))
        return info

cdef dict event_registry = {}

cdef LogEvent get_LogEvent(object name):
    return event_registry.get(name)

cdef LogEvent reg_LogEvent(object name, PetscLogEvent eventid):
    cdef LogEvent event = LogEvent()
    event.id = eventid
    event_registry[name] = event
    return event

# --------------------------------------------------------------------
