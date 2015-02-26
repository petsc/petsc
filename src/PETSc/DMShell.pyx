cdef class DMShell(DM):

    def create(self, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscDM newdm = NULL
        CHKERR( DMShellCreate(ccomm, &newdm) )
        PetscCLEAR(self.obj); self.dm = newdm
        return self

    def setMatrix(self, Mat mat not None):
        CHKERR( DMShellSetMatrix(self.dm, mat.mat) )

    def setGlobalVector(self, Vec gv not None):
        CHKERR( DMShellSetGlobalVector(self.dm, gv.vec) )

    def setLocalVector(self, Vec lv not None):
        CHKERR( DMShellSetLocalVector(self.dm, lv.vec) )

    def setCreateGlobalVector(self, create_gvec, args=None, kargs=None):
        if create_gvec is not None:
            if args  is None: args = ()
            if kargs is None: kargs = {}
            context = (create_gvec, args, kargs)
            self.set_attr('__create_global_vector__', context)
            CHKERR( DMShellSetCreateGlobalVector(self.dm, DMSHELL_CreateGlobalVector) )
        else:
            CHKERR( DMShellSetCreateGlobalVector(self.dm, NULL) )

    def setCreateLocalVector(self, create_lvec, args=None, kargs=None):
        if create_lvec is not None:
            if args  is None: args = ()
            if kargs is None: kargs = {}
            context = (create_lvec, args, kargs)
            self.set_attr('__create_local_vector__', context)
            CHKERR( DMShellSetCreateLocalVector(self.dm, DMSHELL_CreateLocalVector) )
        else:
            CHKERR( DMShellSetCreateLocalVector(self.dm, NULL) )

    def setGlobalToLocal(self, begin, end, begin_args=None, begin_kargs=None,
                         end_args=None, end_kargs=None):
        cdef PetscDMShellXToYFunction cbegin = NULL, cend = NULL
        if begin is not None:
            if begin_args  is None: args = ()
            if begin_kargs is None: kargs = {}
            context = (begin, args, kargs)
            self.set_attr('__g2l_begin__', context)
            cbegin = &DMSHELL_GlobalToLocalBegin
        if end is not None:
            if end_args  is None: args = ()
            if end_kargs is None: kargs = {}
            context = (end, args, kargs)
            self.set_attr('__g2l_end__', context)
            cend = &DMSHELL_GlobalToLocalEnd
        CHKERR( DMShellSetGlobalToLocal(self.dm, cbegin, cend) )

    def setGlobalToLocalVecScatter(self, Scatter gtol not None):
        CHKERR( DMShellSetGlobalToLocalVecScatter(self.dm, gtol.sct) )

    def setLocalToGlobal(self, begin, end, begin_args=None, begin_kargs=None,
                         end_args=None, end_kargs=None):
        cdef PetscDMShellXToYFunction cbegin = NULL, cend = NULL
        if begin is not None:
            if begin_args  is None: args = ()
            if begin_kargs is None: kargs = {}
            context = (begin, args, kargs)
            self.set_attr('__l2g_begin__', context)
            cbegin = &DMSHELL_LocalToGlobalBegin
        if end is not None:
            if end_args  is None: args = ()
            if end_kargs is None: kargs = {}
            context = (end, args, kargs)
            self.set_attr('__l2g_end__', context)
            cend = &DMSHELL_LocalToGlobalEnd
        CHKERR( DMShellSetLocalToGlobal(self.dm, cbegin, cend) )

    def setLocalToGlobalVecScatter(self, Scatter ltog not None):
        CHKERR( DMShellSetLocalToGlobalVecScatter(self.dm, ltog.sct) )

    def setLocalToLocal(self, begin, end, begin_args=None, begin_kargs=None,
                        end_args=None, end_kargs=None):
        cdef PetscDMShellXToYFunction cbegin = NULL, cend = NULL
        cbegin = NULL
        cend = NULL
        if begin is not None:
            if begin_args  is None: args = ()
            if begin_kargs is None: kargs = {}
            context = (begin, args, kargs)
            self.set_attr('__l2l_begin__', context)
            cbegin = &DMSHELL_LocalToLocalBegin
        if end is not None:
            if end_args  is None: args = ()
            if end_kargs is None: kargs = {}
            context = (end, args, kargs)
            self.set_attr('__l2l_end__', context)
            cend = &DMSHELL_LocalToLocalEnd
        CHKERR( DMShellSetLocalToLocal(self.dm, cbegin, cend) )

    def setLocalToLocalVecScatter(self, Scatter ltol not None):
        CHKERR( DMShellSetLocalToLocalVecScatter(self.dm, ltol.sct) )

    def setCreateMatrix(self, create_matrix, args=None, kargs=None):
        if create_matrix is not None:
            if args  is None: args  = ()
            if kargs is None: kargs = {}
            context = (create_matrix, args, kargs)
            self.set_attr('__create_matrix__', context)
            CHKERR( DMShellSetCreateMatrix(self.dm, DMSHELL_CreateMatrix) )
        else:
            CHKERR( DMShellSetCreateMatrix(self.dm, NULL) )
