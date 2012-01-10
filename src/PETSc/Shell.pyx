# -----------------------------------------------------------------------------

cdef class Shell(Object):

    def __cinit__(self):
        self.obj = <PetscObject*> &self.shell
        self.shell = NULL

    def view(self, Viewer viewer=None):
        cdef PetscViewer vwr = NULL
        if viewer is not None: vwr = viewer.vwr
        CHKERR( PetscShellView(self.shell, vwr) )

    def destroy(self):
        CHKERR( PetscShellDestroy(&self.shell) )
        return self

    def create(self, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscShell newshell = NULL
        CHKERR( PetscShellCreate(ccomm, &newshell) )
        PetscCLEAR(self.obj); self.shell = newshell
        return self

    def getURL(self):
        cdef const_char *_url = NULL
        CHKERR( PetscShellGetURL(self.shell, &_url) )
        if _url == NULL: return None
        return bytes2str(_url)

    def setURL(self, url):
        cdef const_char *_url = NULL
        url = str2bytes(url, &_url)
        CHKERR( PetscShellSetURL(self.shell, _url) )
        return self

    def registerComponent(self, key, url=None, Shell component=None):
        cdef const_char *_key = NULL
        cdef const_char *_url = NULL
        cdef PetscShell _component = NULL
        key = str2bytes(key, &_key)
        url = str2bytes(url, &_url)
        if component is not None:
            _component = component.shell
        CHKERR( PetscShellRegisterComponentShell(self.shell, _key, _component) )
        if _url != NULL:
            CHKERR( PetscShellGetComponent(self.shell, _key, &_component,NULL) )
            CHKERR( PetscShellSetURL(_component, _url) )
        return self

    def registerDependence(self, serverkey, clientkey):
        cdef const_char *_clientkey = NULL
        cdef const_char *_serverkey = NULL
        clientkey = str2bytes(clientkey, &_clientkey)
        serverkey = str2bytes(serverkey, &_serverkey)
        CHKERR( PetscShellRegisterDependence(self.shell, _serverkey, _clientkey) )
        return self

    def getComponent(self, key):
        cdef const_char *_key = NULL
        cdef PetscShell component = NULL
        cdef PetscBool found = PETSC_FALSE
        key = str2bytes(key, &_key)
        CHKERR( PetscShellGetComponent(self.shell, _key, &component, &found) )
        if found == PETSC_FALSE or component == NULL: return None
        cdef Shell shell = Shell()
        shell.shell = component
        PetscINCREF(shell.obj);
        return shell

    def call(self, message):
        cdef const_char *_message = NULL
        message = str2bytes(message, &_message)
        CHKERR( PetscShellCall(self.shell, _message) )
        return self

    def getVisitor(self):
        cdef PetscShell visitor = NULL
        CHKERR( PetscShellGetVisitor(self.shell, &visitor) )
        if visitor == NULL: 
           return None
        cdef Shell shell = Shell()
        shell.shell = visitor
        PetscINCREF(shell.obj);
        return shell

    def visit(self, message):
        cdef const_char *_message = NULL
        message = str2bytes(message, &_message)
        CHKERR( PetscShellVisit(self.shell, _message) )
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

    property visitor:
        def __get__(self):
            return self.getVisitor()

    @classmethod
    def DEFAULT(cls, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef Shell shell = Shell()
        shell.shell = PETSC_SHELL_DEFAULT_(ccomm)
        PetscINCREF(shell.obj)
        return shell

# -----------------------------------------------------------------------------
