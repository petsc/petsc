# --------------------------------------------------------------------

cdef class Object:

    #

    def __cinit__(self):
        self.oval = NULL
        self.obj = &self.oval

    def __dealloc__(self):
        CHKERR( PetscDEALLOC(self.obj) )
        self.obj = NULL

    def __richcmp__(Object self, Object other, int op):
        if op!=2 and op!=3: raise TypeError("only '==' and '!='")
        cdef int eq = (op==2)
        if self.obj == NULL or other.obj == NULL:
            if eq: return self is other
            else:  return self is not other
        else:
            if eq: return (self.obj[0] == other.obj[0])
            else:  return (self.obj[0] != other.obj[0])

    def __nonzero__(self):
        if self.obj == NULL: return False
        return self.obj[0] != NULL

    def __bool__(self):
        if self.obj == NULL: return False
        return self.obj[0] != NULL

    # --- reference management ---

    cdef int incRef(self) except -1:
        cdef PetscObject obj = self.obj[0]
        if obj != NULL: CHKERR( PetscObjectReference(obj) )
        return 0

    cdef int decRef(self) except -1:
        cdef PetscObject *obj = self.obj
        cdef PetscInt refct = 0
        if obj[0] != NULL: CHKERR( PetscObjectGetReference(obj[0], &refct) )
        if refct != 0: CHKERR( PetscObjectDereference(obj[0]) )
        if refct == 1: obj[0] = NULL
        return 0
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
        cdef char* tname = self.obj[0].type_name
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
        cdef char* cname = self.obj[0].class_name
        return cp2str(cname)

    def getRefCount(self):
        cdef PetscInt refcnt = 0
        CHKERR( PetscObjectGetReference(self.obj[0], &refcnt) )
        return refcnt

    # --- general Python support ---

    cpdef object getAttr(self, char name[]):
        return Object_getAttr(self.obj[0], name)

    cpdef object setAttr(self, char name[], object attr):
        Object_setAttr(self.obj[0], name, attr)
        return None

    def getDict(self):
        return Object_getDict(self.obj[0])

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

# --------------------------------------------------------------------
