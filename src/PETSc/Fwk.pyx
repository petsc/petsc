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

    def registerComponent(self, key, url):
        cdef const_char *_key = NULL
        cdef const_char *_url = NULL
        key = str2bytes(key, &_key)
        url = str2bytes(url, &_url)
        CHKERR( PetscFwkRegisterComponent(self.fwk, _key, _url) )

    def registerDependence(self, clientkey, serverkey):
        cdef const_char *_clientkey = NULL, 
        cdef const_char *_serverkey = NULL
        clientkey = str2bytes(clientkey, &_clientkey)
        serverkey = str2bytes(serverkey, &_serverkey)
        CHKERR( PetscFwkRegisterDependence(self.fwk, _clientkey, _serverkey) )
        return self

    def configure(self, configuration):
        cdef const_char *_configuration = NULL
        configuration = str2bytes(configuration, &_configuration)
        CHKERR( PetscFwkConfigure(self.fwk, _configuration) )
        return self

    def getComponent(self, key):
        cdef const_char *_key = NULL
        cdef PetscObject cobj = NULL
        cdef PetscTruth found = PETSC_FALSE
        key = str2bytes(key, &_key)
        CHKERR( PetscFwkGetComponent(self.fwk, _key, &cobj, &found) )
        if found == PETSC_FALSE or cobj == NULL: return None
        cdef PetscClassId classid = 0
        CHKERR( PetscObjectGetClassId(cobj, &classid) )
        cdef type klass = TypeRegistryGet(classid)
        cdef Object newobj = klass()
        PetscIncref(cobj); newobj.obj[0] = cobj
        return newobj

    def getURL(self, key):
        cdef const_char *_key = NULL
        cdef const_char *_url = NULL
        cdef PetscTruth found = PETSC_FALSE
        key = str2bytes(key, &_key)
        CHKERR( PetscFwkGetURL(self.fwk, _key, &_url, &found) )
        if found == PETSC_FALSE or _url == NULL: return None
        return bytes2str(_url)

    @classmethod
    def DEFAULT(cls, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef Fwk fwk = Fwk()
        fwk.fwk = PETSC_FWK_DEFAULT_(ccomm)
        PetscIncref(<PetscObject>(fwk.fwk))
        return fwk

# -----------------------------------------------------------------------------
