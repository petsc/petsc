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
        cdef void *attr = NULL
        CHKERR( PetscObjectGetPyObj(self.obj[0], name, &attr) )
        return <object> attr

    cdef object set_attr(self, char name[], object attr):
        CHKERR( PetscObjectSetPyObj(self.obj[0], name, <void*>attr) )
        return None

    cdef object get_dict(self):
        cdef void *dct = NULL
        CHKERR( PetscObjectGetPyDict(self.obj[0], PETSC_TRUE, &dct) )
        return <object> dct

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
        cdef const_char *cval = NULL
        CHKERR( PetscObjectGetType(self.obj[0], &cval) )
        return bytes2str(cval)

    #

    def setOptionsPrefix(self, prefix):
        cdef const_char *cval = NULL
        prefix = str2bytes(prefix, &cval)
        CHKERR( PetscObjectSetOptionsPrefix(self.obj[0], cval) )

    def getOptionsPrefix(self):
        cdef const_char *cval = NULL
        CHKERR( PetscObjectGetOptionsPrefix(self.obj[0], &cval) )
        return bytes2str(cval)

    def setFromOptions(self):
        CHKERR( PetscObjectSetFromOptions(self.obj[0]) )

    #

    def getComm(self):
        cdef Comm comm = Comm()
        CHKERR( PetscObjectGetComm(self.obj[0], &comm.comm) )
        return comm

    def getName(self):
        cdef const_char *cval = NULL
        CHKERR( PetscObjectGetName(self.obj[0], &cval) )
        return bytes2str(cval)

    def setName(self, name):
        cdef const_char *cval = NULL
        name = str2bytes(name, &cval)
        CHKERR( PetscObjectSetName(self.obj[0], cval) )

    def getClassId(self):
        cdef PetscClassId classid = 0
        CHKERR( PetscObjectGetClassId(self.obj[0], &classid) )
        return classid

    def getClassName(self):
        cdef const_char *cval = NULL
        CHKERR( PetscObjectGetClassName(self.obj[0], &cval) )
        return bytes2str(cval)

    def getRefCount(self):
        cdef PetscInt refcnt = 0
        CHKERR( PetscObjectGetReference(self.obj[0], &refcnt) )
        return toInt(refcnt)

    # --- general support ---

    def compose(self, name, Object obj):
        cdef const_char *cval = NULL
        cdef PetscObject cobj = NULL
        name = str2bytes(name, &cval)
        if obj is not None: cobj = obj.obj[0]
        CHKERR( PetscObjectCompose(self.obj[0], cval, cobj) )

    def query(self, name):
        cdef const_char *cval = NULL
        cdef PetscObject cobj = NULL
        name = str2bytes(name, &cval)
        CHKERR( PetscObjectQuery(self.obj[0], cval, &cobj) )
        if cobj == NULL: return None
        cdef PetscClassId classid = 0
        CHKERR( PetscObjectGetClassId(cobj, &classid) )
        cdef type Class = TypeRegistryGet(classid)
        cdef Object newobj = Class()
        PetscIncref(cobj)
        newobj.obj[0] = cobj
        return newobj

    def incRef(self):
        return self.inc_ref()

    def decRef(self):
        return self.dec_ref()

    def getAttr(self, name):
        cdef const_char *cval = NULL
        name = str2bytes(name, &cval)
        return self.get_attr(<char*>cval)

    def setAttr(self, name, attr):
        cdef const_char *cval = NULL
        name = str2bytes(name, &cval)
        self.set_attr(<char*>cval, attr)

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

    property classid:
        def __get__(self):
            return self.getClassId()

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

cdef int TypeRegistryAdd(PetscClassId classid, type cls) except -1:
    global type_registry
    cdef object key = classid
    cdef object value = cls
    cdef const_char *dummy = NULL
    if key not in type_registry:
        type_registry[key] = cls
        reg_LogClass(str2bytes(cls.__name__, &dummy),
                     <PetscLogClass>classid)
        # TypeEnableGC(<PyTypeObject*>cls) # XXX disabled !!!
    else:
        value = type_registry[key]
        if cls is not value:
            raise ValueError(
                "key: %d, cannot register: %s, " \
                "already registered: %s" % (key, cls, value))
    return 0

cdef type TypeRegistryGet(PetscClassId classid):
    global type_registry
    cdef object key = classid
    cdef type cls = Object
    try:
        cls = type_registry[key]
    except KeyError:
        cls = Object
    return cls

# --------------------------------------------------------------------
