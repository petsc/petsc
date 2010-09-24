# --------------------------------------------------------------------

cdef class Object:

    # --- special methods ---

    def __cinit__(self):
        self.oval = NULL
        self.obj = &self.oval

    def __dealloc__(self):
        CHKERR( PetscDEALLOC(self.obj) )
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

    def __deepcopy__(self, dict memo not None):
        cdef object obj_copy = None
        try:
            obj_copy = self.copy
        except AttributeError:
            raise NotImplementedError
        return obj_copy()

    # --- reference management ---

    cdef long inc_ref(self) except -1:
        cdef PetscObject obj = self.obj[0]
        cdef PetscInt refct = 0
        if obj != NULL:
            CHKERR( PetscObjectReference(obj) )
            CHKERR( PetscObjectGetReference(obj, &refct) )
        return (<long>refct)


    cdef long dec_ref(self) except -1:
        cdef PetscObject obj = self.obj[0]
        cdef PetscInt refct = 0
        if obj != NULL:
            CHKERR( PetscObjectGetReference(obj, &refct) )
            if refct == 1: self.obj[0] = NULL
            CHKERR( PetscObjectDereference(obj) )
            refct -= 1
        return (<long>refct)

    # --- attribute management ---

    cdef object get_attr(self, char name[]):
        return Object_getAttr(self.obj[0], name)

    cdef object set_attr(self, char name[], object attr):
        Object_setAttr(self.obj[0], name, attr)
        return None

    cdef object get_dict(self):
        return Object_getDict(self.obj[0])

    #

    def view(self, Viewer viewer=None):
        cdef PetscViewer vwr = NULL
        if viewer is not None: vwr = viewer.vwr
        CHKERR( PetscObjectView(self.obj[0], vwr) )

    def destroy(self):
        CHKERR( PetscObjectDestroy(self.obj[0]) )
        self.obj[0] = NULL
        return self

    def getType(self):
        cdef const_char_p tname = NULL
        CHKERR( PetscObjectGetType(self.obj[0], &tname) )
        return cp2str(tname)

    #

    def setOptionsPrefix(self, prefix):
        CHKERR( PetscObjectSetOptionsPrefix(self.obj[0], str2cp(prefix)) )

    def getOptionsPrefix(self):
        cdef const_char_p prefix = NULL
        CHKERR( PetscObjectGetOptionsPrefix(self.obj[0], &prefix) )
        return cp2str(prefix)

    def setFromOptions(self):
        CHKERR( PetscObjectSetFromOptions(self.obj[0]) )

    #

    def getComm(self):
        cdef Comm comm = Comm()
        CHKERR( PetscObjectGetComm(self.obj[0], &comm.comm) )
        return comm

    def getName(self):
        cdef const_char_p name = NULL
        CHKERR( PetscObjectGetName(self.obj[0], &name) )
        return cp2str(name)

    def setName(self, name):
        CHKERR( PetscObjectSetName(self.obj[0], str2cp(name)) )

    def getCookie(self):
        cdef PetscCookie cookie = 0
        CHKERR( PetscObjectGetCookie(self.obj[0], &cookie) )
        return cookie

    def getClassName(self):
        cdef const_char_p cname = NULL
        CHKERR( PetscObjectGetClassName(self.obj[0], &cname) )
        return cp2str(cname)

    def getRefCount(self):
        cdef PetscInt refcnt = 0
        CHKERR( PetscObjectGetReference(self.obj[0], &refcnt) )
        return refcnt

    # --- general support ---

    def compose(self, name, Object obj):
        cdef char *cname = str2cp(name)
        cdef PetscObject cobj = NULL
        if obj is not None: cobj = obj.obj[0]
        CHKERR( PetscObjectCompose(self.obj[0], cname, cobj) )

    def query(self, name):
        cdef char *cname = str2cp(name)
        cdef PetscObject cobj = NULL
        CHKERR( PetscObjectQuery(self.obj[0], cname, &cobj) )
        if cobj == NULL: return None
        cdef PetscCookie cookie = 0
        CHKERR( PetscObjectGetCookie(cobj, &cookie) )
        cdef type Class = TypeRegistryGet(cookie)
        cdef Object newobj = Class()
        PetscIncref(cobj); newobj.obj[0] = cobj
        return newobj

    def incRef(self):
        return self.inc_ref()

    def decRef(self):
        return self.dec_ref()

    def getAttr(self, name):
        cdef char *cname = str2cp(name)
        return self.get_attr(cname)

    def setAttr(self, name, attr):
        cdef char *cname = str2cp(name)
        self.set_attr(cname, attr)

    def getDict(self):
        return self.get_dict()

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

    property cookie:
        def __get__(self):
            return self.getCookie()

    property klass:
        def __get__(self):
            return self.getClassName()

    property refcount:
        def __get__(self):
            return self.getRefCount()

    # --- Fortran support  ---

    property fortran:
        def __get__(self):
            return Object_toFortran(self.obj[0])


# --------------------------------------------------------------------

include "cyclicgc.pxi"

cdef dict type_registry = { 0 : None }
__type_registry__ = type_registry

cdef int TypeRegistryAdd(PetscCookie cookie, type cls) except -1:
    global type_registry
    cdef object key = cookie
    cdef object value = cls
    if key not in type_registry:
        type_registry[key] = cls
        reg_LogClass(cls.__name__, <PetscLogClass>cookie)
        # TypeEnableGC(<PyTypeObject*>cls) # XXX disabled !!!
    else:
        value = type_registry[key]
        if cls is not value:
            raise ValueError(
                "key: %d, cannot register: %s, " \
                "already registered: %s" % (key, cls, value))
    return 0

cdef type TypeRegistryGet(PetscCookie cookie):
    global type_registry
    cdef object key = cookie
    cdef type cls = Object
    try:
        cls = type_registry[key]
    except KeyError:
        cls = Object
    return cls

# --------------------------------------------------------------------
