# --------------------------------------------------------------------

cdef class Options:

    cdef object _prefix

    def __init__(self, prefix=None):
        self.prefix  = prefix

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

    def setFromOptions(self):
        CHKERR( PetscOptionsSetFromOptions() )
    #

    def hasName(self, name):
        cdef const_char *pr = NULL
        cdef const_char *nm = NULL
        tmp = getpair(self.prefix, name, &pr, &nm)
        cdef PetscBool flag = PETSC_FALSE
        CHKERR( PetscOptionsHasName(pr, nm, &flag) )
        return <bint> flag

    def setValue(self, name, value):
        cdef const_char *pr = NULL
        cdef const_char *nm = NULL
        tmp = getpair(self.prefix, name, &pr, &nm)
        if pr == NULL: 
            option = bytes2str(nm)
        else: 
            option = '-%s%s' % (bytes2str(pr), bytes2str(&nm[1]))
        if type(value) is bool: 
            value = str(value).lower()
        elif value is not None : 
            value = str(value)
        cdef const_char *opt = NULL
        cdef const_char *val = NULL
        option = str2bytes(option, &opt)
        value  = str2bytes(value,  &val)
        CHKERR( PetscOptionsSetValue(opt, val) )

    def delValue(self, name):
        cdef const_char *pr = NULL
        cdef const_char *nm = NULL
        tmp = getpair(self.prefix, name, &pr, &nm)
        if pr == NULL: 
            option = bytes2str(nm)
        else: 
            option = '-%s%s' % (bytes2str(pr), bytes2str(&nm[1]))
        cdef const_char *opt = NULL
        option = str2bytes(option, &opt)
        CHKERR( PetscOptionsClearValue(opt) )

    #

    def getBool(self, name, default=None):
        return getopt(OPT_BOOL, self.prefix, name, default)

    def getInt(self, name, default=None):
        return getopt(OPT_INT, self.prefix, name, default)

    def getReal(self, name, default=None):
        return getopt(OPT_REAL, self.prefix, name, default)

    def getScalar(self, name, default=None):
        return getopt(OPT_SCALAR, self.prefix, name, default)

    def getString(self, name, default=None):
        return getopt(OPT_STRING, self.prefix, name, default)

    def getAll(self):
        cdef char *allopts = NULL
        CHKERR( PetscOptionsGetAll(&allopts) )
        options = bytes2str(allopts)
        CHKERR( PetscFree(allopts) )
        return parseopt(options, self.prefix)

# --------------------------------------------------------------------
