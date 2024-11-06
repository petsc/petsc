# --------------------------------------------------------------------

cdef class Object:
    """Base class wrapping a PETSc object.

    See Also
    --------
    petsc.PetscObject

    """
    # --- special methods ---

    def __cinit__(self):
        self.oval = NULL
        self.obj = &self.oval

    def __dealloc__(self):
        CHKERR(PetscDEALLOC(&self.obj[0]))
        self.obj = NULL

    def __richcmp__(self, other, int op):
        if not isinstance(self,  Object): return NotImplemented
        if not isinstance(other, Object): return NotImplemented
        cdef Object s = self, o = other
        if   op == 2: return (s.obj[0] == o.obj[0])
        elif op == 3: return (s.obj[0] != o.obj[0])
        else: raise TypeError("only '==' and '!='")

    def __bool__(self):
        return self.obj[0] != NULL

    def __copy__(self):
        cdef Object obj = type(self)()
        cdef PetscObject o = self.obj[0]
        if o != NULL:
            CHKERR(PetscObjectReference(o))
        obj.obj[0] = o
        return obj

    def __deepcopy__(self, dict memo):
        cdef object obj_copy = None
        try:
            obj_copy = self.copy
        except AttributeError:
            raise NotImplementedError
        <void>memo # unused
        return obj_copy()

    # --- attribute management ---

    cdef object get_attr(self, char name[]):
        return PetscGetPyObj(self.obj[0], name)

    cdef object set_attr(self, char name[], object attr):
        return PetscSetPyObj(self.obj[0], name, attr)

    cdef object get_dict(self):
        return PetscGetPyDict(self.obj[0], True)

    #

    def view(self, Viewer viewer=None) -> None:
        """Display the object.

        Collective.

        Parameters
        ----------
        viewer
            A `Viewer` instance or `None` for the default viewer.

        See Also
        --------
        petsc.PetscObjectView

        """
        cdef PetscViewer vwr = NULL
        if viewer is not None: vwr = viewer.vwr
        CHKERR(PetscObjectView(self.obj[0], vwr))

    def destroy(self) -> Self:
        """Destroy the object.

        Collective.

        See Also
        --------
        petsc.PetscObjectDestroy

        """
        CHKERR(PetscObjectDestroy(&self.obj[0]))
        return self

    def getType(self) -> str:
        """Return the object type name.

        Not collective.

        See Also
        --------
        petsc.PetscObjectGetType

        """
        cdef const char *cval = NULL
        CHKERR(PetscObjectGetType(self.obj[0], &cval))
        return bytes2str(cval)

    #

    def setOptionsPrefix(self, prefix : str | None) -> None:
        """Set the prefix used for searching for options in the database.

        Logically collective.

        See Also
        --------
        petsc_options, getOptionsPrefix, petsc.PetscObjectSetOptionsPrefix

        """
        cdef const char *cval = NULL
        prefix = str2bytes(prefix, &cval)
        CHKERR(PetscObjectSetOptionsPrefix(self.obj[0], cval))

    def getOptionsPrefix(self) -> str:
        """Return the prefix used for searching for options in the database.

        Not collective.

        See Also
        --------
        petsc_options, setOptionsPrefix, petsc.PetscObjectGetOptionsPrefix

        """
        cdef const char *cval = NULL
        CHKERR(PetscObjectGetOptionsPrefix(self.obj[0], &cval))
        return bytes2str(cval)

    def appendOptionsPrefix(self, prefix: str | None) -> None:
        """Append to the prefix used for searching for options in the database.

        Logically collective.

        See Also
        --------
        petsc_options, setOptionsPrefix, petsc.PetscObjectAppendOptionsPrefix

        """
        cdef const char *cval = NULL
        prefix = str2bytes(prefix, &cval)
        CHKERR(PetscObjectAppendOptionsPrefix(self.obj[0], cval))

    def setFromOptions(self) -> None:
        """Configure the object from the options database.

        Collective.

        Classes that do not implement ``setFromOptions`` use this method
        that, in turn, calls `petsc.PetscObjectSetFromOptions`.

        See Also
        --------
        petsc_options, petsc.PetscObjectSetFromOptions

        """
        CHKERR(PetscObjectSetFromOptions(self.obj[0]))

    def viewFromOptions(self, name : str, Object objpre=None) -> None:
        """View the object via command line options.

        Collective.

        Parameters
        ----------
        name
            The command line option.
        objpre
            Optional object that provides prefix.

        See Also
        --------
        petsc_options, petsc.PetscObjectViewFromOptions

        """
        cdef PetscObject pobj = NULL
        cdef const char *cval = NULL
        pobj = objpre.obj[0] if objpre is not None else NULL
        name = str2bytes(name, &cval)
        CHKERR(PetscObjectViewFromOptions(self.obj[0], pobj, cval))

    def setOptionsHandler(self, handler: PetscOptionsHandlerFunction | None) -> None:
        """Set the callback for processing extra options.

        Logically collective.

        Parameters
        ----------
        handler
            The callback function, called at the end of a ``setFromOptions`` invocation
            for the given class.

        See Also
        --------
        petsc_options, Mat.setFromOptions, KSP.setFromOptions
        petsc.PetscObjectAddOptionsHandler

        """
        if handler is not None:
            CHKERR(PetscObjectAddOptionsHandler(self.obj[0], PetscObjectOptionsHandler_PYTHON, NULL, NULL))
            self.set_attr('__optshandler__', handler)
        else:
            self.set_attr('__optshandler__', None)

    def destroyOptionsHandlers(self) -> None:
        """Clear all the option handlers.

        Collective.

        See Also
        --------
        petsc_options, setOptionsHandler, petsc.PetscObjectDestroyOptionsHandlers

        """
        self.set_attr('__optshandler__', None)
        CHKERR(PetscObjectDestroyOptionsHandlers(self.obj[0]))

    #

    def getComm(self) -> Comm:
        """Return the communicator of the object.

        Not collective.

        See Also
        --------
        petsc.PetscObjectGetComm

        """
        cdef Comm comm = Comm()
        CHKERR(PetscObjectGetComm(self.obj[0], &comm.comm))
        return comm

    def getName(self) -> str:
        """Return the name of the object.

        Not collective.

        See Also
        --------
        petsc.PetscObjectGetName

        """
        cdef const char *cval = NULL
        CHKERR(PetscObjectGetName(self.obj[0], &cval))
        return bytes2str(cval)

    def setName(self, name : str | None) -> None:
        """Associate a name to the object.

        Not collective.

        See Also
        --------
        petsc.PetscObjectSetName

        """
        cdef const char *cval = NULL
        name = str2bytes(name, &cval)
        CHKERR(PetscObjectSetName(self.obj[0], cval))

    def getClassId(self) -> int:
        """Return the class identifier of the object.

        Not collective.

        See Also
        --------
        petsc.PetscObjectGetClassId

        """
        cdef PetscClassId classid = 0
        CHKERR(PetscObjectGetClassId(self.obj[0], &classid))
        return <long>classid

    def getClassName(self) -> str:
        """Return the class name of the object.

        Not collective.

        See Also
        --------
        petsc.PetscObjectGetClassName

        """
        cdef const char *cval = NULL
        CHKERR(PetscObjectGetClassName(self.obj[0], &cval))
        return bytes2str(cval)

    def getRefCount(self) -> int:
        """Return the reference count of the object.

        Not collective.

        See Also
        --------
        petsc.PetscObjectGetReference

        """
        if self.obj[0] == NULL: return 0
        cdef PetscInt refcnt = 0
        CHKERR(PetscObjectGetReference(self.obj[0], &refcnt))
        return toInt(refcnt)

    def getId(self) -> int:
        """Return the unique identifier of the object.

        Not collective.

        See Also
        --------
        petsc.PetscObjectGetId

        """
        cdef PetscObjectId cid = 0
        CHKERR(PetscObjectGetId(self.obj[0], &cid))
        return <long>cid

    # --- general support ---

    def compose(self, name : str | None, Object obj or None) -> None:
        """Associate a PETSc object using a key string.

        Logically collective.

        Parameters
        ----------
        name
            The string identifying the object to be composed.
        obj
            The object to be composed.

        See Also
        --------
        query, petsc.PetscObjectCompose

        """
        cdef const char *cval = NULL
        cdef PetscObject cobj = NULL
        name = str2bytes(name, &cval)
        if obj is not None: cobj = obj.obj[0]
        CHKERR(PetscObjectCompose(self.obj[0], cval, cobj))

    def query(self, name: str) -> Object:
        """Query for the PETSc object associated with a key string.

        Not collective.

        See Also
        --------
        compose, petsc.PetscObjectQuery

        """
        cdef const char *cval = NULL
        cdef PetscObject cobj = NULL
        name = str2bytes(name, &cval)
        CHKERR(PetscObjectQuery(self.obj[0], cval, &cobj))
        if cobj == NULL: return None
        cdef Object obj = subtype_Object(cobj)()
        obj.obj[0] = cobj
        CHKERR(PetscINCREF(obj.obj))
        return obj

    def incRef(self) -> int:
        """Increment the object reference count.

        Logically collective.

        See Also
        --------
        getRefCount, petsc.PetscObjectReference

        """
        cdef PetscObject obj = self.obj[0]
        cdef PetscInt refct = 0
        if obj != NULL:
            CHKERR(PetscObjectReference(obj))
            CHKERR(PetscObjectGetReference(obj, &refct))
        return toInt(refct)

    def decRef(self) -> int:
        """Decrement the object reference count.

        Logically collective.

        See Also
        --------
        getRefCount, petsc.PetscObjectDereference

        """
        cdef PetscObject obj = self.obj[0]
        cdef PetscInt refct = 0
        if obj != NULL:
            CHKERR(PetscObjectGetReference(obj, &refct))
            CHKERR(PetscObjectDereference(obj))
            if refct == 1: self.obj[0] = NULL
            refct -= 1
        return toInt(refct)

    def getAttr(self, name : str) -> object:
        """Return the attribute associated with a given name.

        Not collective.

        See Also
        --------
        setAttr, getDict

        """
        cdef const char *cval = NULL
        name = str2bytes(name, &cval)
        return self.get_attr(<char*>cval)

    def setAttr(self, name : str, attr : object) -> None:
        """Set an the attribute associated with a given name.

        Not collective.

        See Also
        --------
        getAttr, getDict

        """
        cdef const char *cval = NULL
        name = str2bytes(name, &cval)
        self.set_attr(<char*>cval, attr)

    def getDict(self) -> dict:
        """Return the dictionary of attributes.

        Not collective.

        See Also
        --------
        setAttr, getAttr

        """
        return self.get_dict()

    # --- state manipulation ---

    def stateIncrease(self) -> None:
        """Increment the PETSc object state.

        Logically collective.

        See Also
        --------
        stateGet, stateSet, petsc.PetscObjectStateIncrease

        """
        PetscINCSTATE(self.obj)

    def stateGet(self) -> int:
        """Return the PETSc object state.

        Not collective.

        See Also
        --------
        stateSet, stateIncrease, petsc.PetscObjectStateGet

        """
        cdef PetscObjectState state = 0
        CHKERR(PetscObjectStateGet(self.obj[0], &state))
        return <long>state

    def stateSet(self, state : int) -> None:
        """Set the PETSc object state.

        Logically collective.

        See Also
        --------
        stateIncrease, stateGet, petsc.PetscObjectStateSet

        """
        cdef PetscObjectState cstate = asInt(state)
        CHKERR(PetscObjectStateSet(self.obj[0], cstate))

    # --- tab level ---

    def incrementTabLevel(self, tab : int, Object parent=None) -> None:
        """Increment the PETSc object tab level.

        Logically collective.

        See Also
        --------
        setTabLevel, getTabLevel, petsc.PetscObjectIncrementTabLevel

        """
        cdef PetscInt ctab = asInt(tab)
        cdef PetscObject cobj = <PetscObject> NULL if parent is None else parent.obj[0]
        CHKERR(PetscObjectIncrementTabLevel(self.obj[0], cobj, ctab))

    def setTabLevel(self, level : int) -> None:
        """Set the PETSc object tab level.

        Logically collective.

        See Also
        --------
        incrementTabLevel, getTabLevel, petsc.PetscObjectSetTabLevel

        """
        cdef PetscInt clevel = asInt(level)
        CHKERR(PetscObjectSetTabLevel(self.obj[0], clevel))

    def getTabLevel(self) -> None:
        """Return the PETSc object tab level.

        Not collective.

        See Also
        --------
        setTabLevel, incrementTabLevel, petsc.PetscObjectGetTabLevel

        """
        cdef PetscInt clevel = 0
        CHKERR(PetscObjectGetTabLevel(self.obj[0], &clevel))
        return toInt(clevel)

    # --- properties ---

    property type:
        """Object type."""
        def __get__(self) -> str:
            return self.getType()

        def __set__(self, value):
            self.setType(value)

    property prefix:
        """Options prefix."""
        def __get__(self) -> str:
            return self.getOptionsPrefix()

        def __set__(self, value):
            self.setOptionsPrefix(value)

    property comm:
        """The object communicator."""
        def __get__(self) -> Comm:
            return self.getComm()

    property name:
        """The object name."""
        def __get__(self) -> str:
            return self.getName()

        def __set__(self, value):
            self.setName(value)

    property classid:
        """The class identifier."""
        def __get__(self) -> int:
            return self.getClassId()

    property id:
        """The object identifier."""
        def __get__(self) -> int:
            return self.getId()

    property klass:
        """The class name."""
        def __get__(self) -> str:
            return self.getClassName()

    property refcount:
        """Reference count."""
        def __get__(self) -> int:
            return self.getRefCount()

    # --- ctypes support  ---

    property handle:
        """Handle for ctypes support."""
        def __get__(self) -> int:
            cdef PetscObject obj = self.obj[0]
            return PyLong_FromVoidPtr(<void*>obj)

    # --- Fortran support  ---

    property fortran:
        """Fortran handle."""
        def __get__(self) -> int:
            cdef PetscObject obj = self.obj[0]
            return Object_toFortran(obj)

# --------------------------------------------------------------------

include "cyclicgc.pxi"

cdef dict type_registry = {0 : None}
__type_registry__ = type_registry

cdef int PyPetscType_Register(int classid, type cls) except -1:
    global type_registry
    cdef object key = <long>classid
    cdef object value = cls
    cdef const char *dummy = NULL
    if key not in type_registry:
        type_registry[key] = cls
        reg_LogClass(str2bytes(cls.__name__, &dummy),
                     <PetscLogClass>classid)
        TypeEnableGC(<PyTypeObject*>cls)
    else:
        value = type_registry[key]
        if cls is not value:
            raise ValueError(
                "key: %d, cannot register: %s, "
                "already registered: %s" % (key, cls, value))
    return 0

cdef type PyPetscType_Lookup(int classid):
    global type_registry
    cdef object key = <long>classid
    cdef type cls = Object
    try:
        cls = type_registry[key]
    except KeyError:
        cls = Object
    return cls

# --------------------------------------------------------------------
