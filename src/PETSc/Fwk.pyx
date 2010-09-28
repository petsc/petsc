# -----------------------------------------------------------------------------

cdef class Fwk(Object):

    cdef PetscFwk fwk

    def __cinit__(self):
        self.obj = <PetscObject*> &self.fwk
        self.fwk = NULL

    def view(self, Viewer viewer=None):
        cdef PetscViewer vwr = NULL
        if viewer is not None: vwr = viewer.vwr
        CHKERR( PetscFwkView(self.fwk, vwr) )

    def destroy(self):
        CHKERR( PetscFwkDestroy(self.fwk) )
        self.fwk = NULL
        return self

    def create(self, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscFwk newfwk = NULL
        CHKERR( PetscFwkCreate(ccomm, &newfwk) )
        PetscCLEAR(self.obj); self.fwk = newfwk
        return self

    def getURL(self):
        cdef const_char *_url = NULL
        CHKERR( PetscFwkGetURL(self.fwk, &_url) )
        if _url == NULL: return None
        return bytes2str(_url)

    def setURL(self, url):
        cdef const_char *_url = NULL
        url = str2bytes(url, &_url)
        CHKERR( PetscFwkSetURL(self.fwk, _url) )
        return 0

    def registerComponent(self, key, url=None):
        cdef const_char *_key = NULL
        cdef const_char *_url = NULL
        key = str2bytes(key, &_key)
        url = str2bytes(url, &_url)
        if _url == NULL:
            CHKERR( PetscFwkRegisterComponent(self.fwk, _key) )
        else:
            CHKERR( PetscFwkRegisterComponentURL(self.fwk, _key, _url) )

    def registerDependence(self, clientkey, serverkey):
        cdef const_char *_clientkey = NULL
        cdef const_char *_serverkey = NULL
        clientkey = str2bytes(clientkey, &_clientkey)
        serverkey = str2bytes(serverkey, &_serverkey)
        CHKERR( PetscFwkRegisterDependence(self.fwk, _clientkey, _serverkey) )
        return self

    def getComponent(self, key):
        cdef const_char *_key = NULL
        cdef PetscFwk component = NULL
        cdef PetscBool found = PETSC_FALSE
        key = str2bytes(key, &_key)
        CHKERR( PetscFwkGetComponent(self.fwk, _key, &component, &found) )
        if found == PETSC_FALSE or component == NULL: return None
        cdef Fwk fwk = Fwk()
        PetscIncref(<PetscObject>component);
        fwk.fwk = component
        return fwk

    def call(self, message):
        cdef const_char *_message = NULL
        message = str2bytes(message, &_message)
        CHKERR( PetscFwkCall(self.fwk, _message) )

    def getParent(self):
        cdef PetscFwk parent = NULL
        CHKERR( PetscFwkGetParent(self.fwk, &parent) )
        if parent == NULL: return None
        cdef Fwk fwk = Fwk()
        PetscIncref(<PetscObject>parent);
        fwk.fwk = parent
        return fwk

    def visit(self, message):
        cdef const_char *_message = NULL
        message = str2bytes(message, &_message)
        CHKERR( PetscFwkVisit(self.fwk, _message) )
        return self

    def __getitem__(self, key):
        return self.getComponent(key)

    def __call__(self, message, *, visit=False):
        if visit:
            return self.visit(message)
        else:
            return self.call(message)

    property url:
        def __get__(self):
            return self.getURL()
        def __set__(self, value):
            self.setURL(value)

    property parent:
        def __get__(self):
            return self.getParent()

    @classmethod
    def DEFAULT(cls, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef Fwk fwk = Fwk()
        fwk.fwk = PETSC_FWK_DEFAULT_(ccomm)
        PetscIncref(<PetscObject>(fwk.fwk))
        return fwk

# -----------------------------------------------------------------------------
