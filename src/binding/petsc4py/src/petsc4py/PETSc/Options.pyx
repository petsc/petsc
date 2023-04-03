# --------------------------------------------------------------------

cdef class Options:

    cdef PetscOptions opt
    cdef object       _prefix

    def __init__(self, prefix=None):
        self.opt = NULL
        self.prefix  = prefix

    def __dealloc__(self):
        if self.opt == NULL: return
        CHKERR( PetscOptionsDestroy(&self.opt) )

    def __contains__(self, item):
        return self.hasName(item)

    def __getitem__(self, item):
        return self.getString(item)

    def __setitem__(self, item, value):
        self.setValue(item, value)

    def __delitem__(self, item):
        self.delValue(item)

    property prefix:
        def __get__(self):
            return self._prefix
        def __set__(self, prefix):
            self._prefix = getprefix(prefix)
        def __del__(self):
            self._prefix = None
    #

    def create(self):
        if self.opt != NULL: return
        CHKERR( PetscOptionsCreate(&self.opt) )
        return self

    def destroy(self):
        if self.opt == NULL: return
        CHKERR( PetscOptionsDestroy(&self.opt) )
        return self

    def clear(self):
        if self.opt == NULL: return
        CHKERR( PetscOptionsClear(self.opt) )
        return self

    def view(self, Viewer viewer=None):
        cdef PetscViewer vwr = NULL
        if viewer is not None: vwr = viewer.vwr
        CHKERR( PetscOptionsView(self.opt, vwr) )

    def prefixPush(self, prefix):
        prefix = getprefix(prefix)
        cdef const char *cprefix = NULL
        prefix = str2bytes(prefix, &cprefix)
        CHKERR( PetscOptionsPrefixPush(self.opt, cprefix) )

    def prefixPop(self):
        CHKERR( PetscOptionsPrefixPop(self.opt) )
    #

    def hasName(self, name):
        cdef const char *pr = NULL
        cdef const char *nm = NULL
        tmp = getpair(self.prefix, name, &pr, &nm)
        cdef PetscBool flag = PETSC_FALSE
        CHKERR( PetscOptionsHasName(self.opt, pr, nm, &flag) )
        return toBool(flag)

    def setValue(self, name, value):
        cdef const char *pr = NULL
        cdef const char *nm = NULL
        tmp = getpair(self.prefix, name, &pr, &nm)
        if pr == NULL: 
            option = bytes2str(nm)
        else: 
            option = '-%s%s' % (bytes2str(pr), bytes2str(&nm[1]))
        if type(value) is bool: 
            value = str(value).lower()
        elif value is not None : 
            value = str(value)
        cdef const char *key = NULL
        cdef const char *val = NULL
        option = str2bytes(option, &key)
        value  = str2bytes(value,  &val)
        CHKERR( PetscOptionsSetValue(self.opt, key, val) )

    def delValue(self, name):
        cdef const char *pr = NULL
        cdef const char *nm = NULL
        tmp = getpair(self.prefix, name, &pr, &nm)
        if pr == NULL: 
            option = bytes2str(nm)
        else: 
            option = '-%s%s' % (bytes2str(pr), bytes2str(&nm[1]))
        cdef const char *key = NULL
        option = str2bytes(option, &key)
        CHKERR( PetscOptionsClearValue(self.opt, key) )

    #

    def getBool(self, name, default=None):
        return getopt(self.opt, OPT_BOOL, self.prefix, name, default)

    def getInt(self, name, default=None):
        return getopt(self.opt, OPT_INT, self.prefix, name, default)

    def getReal(self, name, default=None):
        return getopt(self.opt, OPT_REAL, self.prefix, name, default)

    def getScalar(self, name, default=None):
        return getopt(self.opt, OPT_SCALAR, self.prefix, name, default)

    def getString(self, name, default=None):
        return getopt(self.opt, OPT_STRING, self.prefix, name, default)

    #

    def insertString(self, string):
        cdef const char *cstring = NULL
        string = str2bytes(string, &cstring)
        CHKERR( PetscOptionsInsertString(self.opt, cstring) )

    def getAll(self):
        cdef char *allopts = NULL
        CHKERR( PetscOptionsGetAll(self.opt, &allopts) )
        options = bytes2str(allopts)
        CHKERR( PetscFree(allopts) )
        return parseopt(options, self.prefix)

# --------------------------------------------------------------------
