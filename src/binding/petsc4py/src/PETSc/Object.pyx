# --------------------------------------------------------------------

cdef class Object:

    # --- special methods ---

    def __cinit__(self):
        self.oval = NULL
        self.obj = &self.oval

    def __dealloc__(self):
        CHKERR( PetscDEALLOC(&self.obj[0]) )
        self.obj = NULL

    def __richcmp__(self, other, int op):
        if not isinstance(self,  Object): return NotImplemented
        if not isinstance(other, Object): return NotImplemented
        cdef Object s = self, o = other
        if   op == 2: return (s.obj[0] == o.obj[0])
        elif op == 3: return (s.obj[0] != o.obj[0])
        else: raise TypeError("only '==' and '!='")

    def __nonzero__(self):
        return self.obj[0] != NULL

    def __copy__(self):
        cdef Object obj = type(self)()
        cdef PetscObject o = self.obj[0]
        if o != NULL:
            CHKERR( PetscObjectReference(o) )
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

    def view(self, Viewer viewer=None):
        cdef PetscViewer vwr = NULL
        if viewer is not None: vwr = viewer.vwr
        CHKERR( PetscObjectView(self.obj[0], vwr) )

    def destroy(self):
        CHKERR( PetscObjectDestroy(&self.obj[0]) )
        return self

    def getType(self):
        cdef const char *cval = NULL
        CHKERR( PetscObjectGetType(self.obj[0], &cval) )
        return bytes2str(cval)

    #

    def setOptionsPrefix(self, prefix):
        cdef const char *cval = NULL
        prefix = str2bytes(prefix, &cval)
        CHKERR( PetscObjectSetOptionsPrefix(self.obj[0], cval) )

    def getOptionsPrefix(self):
        cdef const char *cval = NULL
        CHKERR( PetscObjectGetOptionsPrefix(self.obj[0], &cval) )
        return bytes2str(cval)

    def setFromOptions(self):
        CHKERR( PetscObjectSetFromOptions(self.obj[0]) )

    def viewFromOptions(self, name, Object prefix=None):
        cdef PetscObject pobj = NULL
        cdef const char *cval = NULL
        pobj = prefix.obj[0] if prefix is not None else NULL
        name = str2bytes(name, &cval)
        CHKERR( PetscObjectViewFromOptions(self.obj[0], pobj, cval) )

    #

    def getComm(self):
        cdef Comm comm = Comm()
        CHKERR( PetscObjectGetComm(self.obj[0], &comm.comm) )
        return comm

    def getName(self):
        cdef const char *cval = NULL
        CHKERR( PetscObjectGetName(self.obj[0], &cval) )
        return bytes2str(cval)

    def setName(self, name):
        cdef const char *cval = NULL
        name = str2bytes(name, &cval)
        CHKERR( PetscObjectSetName(self.obj[0], cval) )

    def getClassId(self):
        cdef PetscClassId classid = 0
        CHKERR( PetscObjectGetClassId(self.obj[0], &classid) )
        return <long>classid

    def getClassName(self):
        cdef const char *cval = NULL
        CHKERR( PetscObjectGetClassName(self.obj[0], &cval) )
        return bytes2str(cval)

    def getRefCount(self):
        if self.obj[0] == NULL: return 0
        cdef PetscInt refcnt = 0
        CHKERR( PetscObjectGetReference(self.obj[0], &refcnt) )
        return toInt(refcnt)

    # --- general support ---

    def compose(self, name, Object obj or None):
        cdef const char *cval = NULL
        cdef PetscObject cobj = NULL
        name = str2bytes(name, &cval)
        if obj is not None: cobj = obj.obj[0]
        CHKERR( PetscObjectCompose(self.obj[0], cval, cobj) )

    def query(self, name):
        cdef const char *cval = NULL
        cdef PetscObject cobj = NULL
        name = str2bytes(name, &cval)
        CHKERR( PetscObjectQuery(self.obj[0], cval, &cobj) )
        if cobj == NULL: return None
        cdef Object obj = subtype_Object(cobj)()
        obj.obj[0] = cobj
        PetscINCREF(obj.obj)
        return obj

    def incRef(self):
        cdef PetscObject obj = self.obj[0]
        cdef PetscInt refct = 0
        if obj != NULL:
            CHKERR( PetscObjectReference(obj) )
            CHKERR( PetscObjectGetReference(obj, &refct) )
        return (<long>refct)

    def decRef(self):
        cdef PetscObject obj = self.obj[0]
        cdef PetscInt refct = 0
        if obj != NULL:
            CHKERR( PetscObjectGetReference(obj, &refct) )
            CHKERR( PetscObjectDereference(obj) )
            if refct == 1: self.obj[0] = NULL
            refct -= 1
        return (<long>refct)

    def getAttr(self, name):
        cdef const char *cval = NULL
        name = str2bytes(name, &cval)
        return self.get_attr(<char*>cval)

    def setAttr(self, name, attr):
        cdef const char *cval = NULL
        name = str2bytes(name, &cval)
        self.set_attr(<char*>cval, attr)

    def getDict(self):
        return self.get_dict()

    # --- state manipulation ---
    def stateIncrease(self):
        PetscINCSTATE(self.obj)

    def stateGet(self):
        cdef PetscObjectState state = 0
        CHKERR( PetscObjectStateGet(self.obj[0], &state) )
        return toInt(state)

    def stateSet(self, state):
        cdef PetscObjectState cstate = asInt(state)
        CHKERR( PetscObjectStateSet(self.obj[0], cstate) )

    # --- tab level ---

    def incrementTabLevel(self, tab, Object parent=None):
        cdef PetscInt ctab = asInt(tab)
        cdef PetscObject cobj = <PetscObject> NULL if parent is None else parent.obj[0]
        CHKERR( PetscObjectIncrementTabLevel(self.obj[0], cobj, ctab) )

    def setTabLevel(self, level):
        cdef PetscInt clevel = asInt(level)
        CHKERR( PetscObjectSetTabLevel(self.obj[0], clevel) )

    def getTabLevel(self):
        cdef PetscInt clevel = 0
        CHKERR( PetscObjectGetTabLevel(self.obj[0], &clevel) )
        return toInt(clevel)

    # --- properties ---

    property type:
        def __get__(self):
            return self.getType()
        def __set__(self, value):
            self.setType(value)

    property prefix:
        def __get__(self):
            return self.getOptionsPrefix()
        def __set__(self, value):
            self.setOptionsPrefix(value)

    property comm:
        def __get__(self):
            return self.getComm()

    property name:
        def __get__(self):
            return self.getName()
        def __set__(self, value):
            self.setName(value)

    property classid:
        def __get__(self):
            return self.getClassId()

    property klass:
        def __get__(self):
            return self.getClassName()

    property refcount:
        def __get__(self):
            return self.getRefCount()

    # --- ctypes support  ---

    property handle:
        def __get__(self):
            cdef PetscObject obj = self.obj[0]
            return PyLong_FromVoidPtr(<void*>obj)

    # --- Fortran support  ---

    property fortran:
        def __get__(self):
            cdef PetscObject obj = self.obj[0]
            return Object_toFortran(obj)

# --------------------------------------------------------------------

include "cyclicgc.pxi"

cdef dict type_registry = { 0 : None }
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
                "key: %d, cannot register: %s, " \
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
