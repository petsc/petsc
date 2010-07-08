# -----------------------------------------------------------------------------

cdef class Fwk(Object):

    cdef PetscFwk fwk

    def __cinit__(self):
        self.obj = <PetscObject*> &self.fwk
        self.fwk = NULL

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

    def registerComponent(self, url):
        cdef const_char *_url = NULL
        url = str2bytes(url, &_url)
        CHKERR( PetscFwkRegisterComponent(self.fwk, _url) )

    def registerDependence(self, clienturl, serverurl):
        cdef const_char *_clienturl = NULL, 
        cdef const_char *_serverurl = NULL
        clienturl = str2bytes(clienturl, &_clienturl)
        serverurl = str2bytes(serverurl, &_serverurl)
        CHKERR( PetscFwkRegisterDependence(self.fwk, _clienturl, _serverurl) )
        return self

    def viewConfigurationOrder(self, Viewer viewer=None):
        cdef PetscViewer vwr = NULL
        if viewer is not None: vwr = viewer.vwr
        cdef MPI_Comm ccomm = MPI_COMM_NULL
        if vwr == NULL: # XXX
            CHKERR( PetscObjectGetComm(<PetscObject>self.fwk, &ccomm) )
            vwr = PETSC_VIEWER_STDOUT_(ccomm)
        CHKERR( PetscFwkViewConfigurationOrder(self.fwk, vwr) )

    def configure(self, state):
        cdef PetscInt s = asInt(state)
        CHKERR( PetscFwkConfigure(self.fwk, s) )
        return self

    def getComponent(self, url):
        cdef const_char *_url = NULL
        cdef PetscObject cobj = NULL
        cdef PetscTruth found = PETSC_FALSE
        url = str2bytes(url, &_url)
        CHKERR( PetscFwkGetComponent(self.fwk, _url, &cobj, &found) )
        if found == PETSC_FALSE or cobj == NULL: return None
        cdef PetscClassId classid = 0
        CHKERR( PetscObjectGetClassId(cobj, &classid) )
        cdef type klass = TypeRegistryGet(classid)
        cdef Object newobj = klass()
        PetscIncref(cobj); newobj.obj[0] = cobj
        return newobj

    @classmethod
    def DEFAULT(cls, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef Fwk fwk = Fwk()
        fwk.fwk = PETSC_FWK_DEFAULT_(ccomm)
        PetscIncref(<PetscObject>(fwk.fwk))
        return fwk

# -----------------------------------------------------------------------------
