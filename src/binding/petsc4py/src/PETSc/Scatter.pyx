# --------------------------------------------------------------------

class ScatterType(object):
   BASIC      = S_(PETSCSFBASIC)
   NEIGHBOR   = S_(PETSCSFNEIGHBOR)
   ALLGATHERV = S_(PETSCSFALLGATHERV)
   ALLGATHER  = S_(PETSCSFALLGATHER)
   GATHERV    = S_(PETSCSFGATHERV)
   GATHER     = S_(PETSCSFGATHER)
   ALLTOALL   = S_(PETSCSFALLTOALL)
   WINDOW     = S_(PETSCSFWINDOW)

# --------------------------------------------------------------------

cdef class Scatter(Object):

    Type = ScatterType
    Mode = ScatterMode

    #

    def __cinit__(self):
        self.obj = <PetscObject*> &self.sct
        self.sct = NULL

    def __call__(self, x, y, addv=None, mode=None):
        self.scatter(x, y, addv, mode)

    #

    def view(self, Viewer viewer=None):
        cdef PetscViewer vwr = NULL
        if viewer is not None: vwr = viewer.vwr
        CHKERR( VecScatterView(self.sct, vwr) )

    def destroy(self):
        CHKERR( VecScatterDestroy(&self.sct) )
        return self

    def create(self, Vec vec_from, IS is_from or None,
               Vec vec_to, IS is_to or None):
        cdef PetscIS cisfrom = NULL, cisto = NULL
        if is_from is not None: cisfrom = is_from.iset
        if is_to   is not None: cisto   = is_to.iset
        cdef PetscScatter newsct = NULL
        CHKERR( VecScatterCreate(
                vec_from.vec, cisfrom, vec_to.vec, cisto, &newsct) )
        PetscCLEAR(self.obj); self.sct = newsct
        return self

    def setType(self, scatter_type):
        cdef PetscScatterType cval = NULL
        vec_type = str2bytes(scatter_type, &cval)
        CHKERR( VecScatterSetType(self.sct, cval) )

    def getType(self):
        cdef PetscScatterType cval = NULL
        CHKERR( VecScatterGetType(self.sct, &cval) )
        return bytes2str(cval)

    def setFromOptions(self):
        CHKERR( VecScatterSetFromOptions(self.sct) )

    def setUp(self):
        CHKERR( VecScatterSetUp(self.sct) )
        return self

    def copy(self):
        cdef Scatter scatter = Scatter()
        CHKERR( VecScatterCopy(self.sct, &scatter.sct) )
        return scatter

    @classmethod
    def toAll(cls, Vec vec):
        cdef Scatter scatter = Scatter()
        cdef Vec ovec = Vec()
        CHKERR( VecScatterCreateToAll(
            vec.vec, &scatter.sct, &ovec.vec) )
        return (scatter, ovec)

    @classmethod
    def toZero(cls, Vec vec):
        cdef Scatter scatter = Scatter()
        cdef Vec ovec = Vec()
        CHKERR( VecScatterCreateToZero(
            vec.vec, &scatter.sct, &ovec.vec) )
        return (scatter, ovec)
    #

    def begin(self, Vec vec_from, Vec vec_to, addv=None, mode=None):
        cdef PetscInsertMode  caddv = insertmode(addv)
        cdef PetscScatterMode csctm = scattermode(mode)
        CHKERR( VecScatterBegin(self.sct, vec_from.vec, vec_to.vec,
                                caddv, csctm) )

    def end(self, Vec vec_from, Vec vec_to, addv=None, mode=None):
        cdef PetscInsertMode  caddv = insertmode(addv)
        cdef PetscScatterMode csctm = scattermode(mode)
        CHKERR( VecScatterEnd(self.sct, vec_from.vec, vec_to.vec,
                              caddv, csctm) )

    #

    def scatterBegin(self, Vec vec_from, Vec vec_to, addv=None, mode=None):
        cdef PetscInsertMode  caddv = insertmode(addv)
        cdef PetscScatterMode csctm = scattermode(mode)
        CHKERR( VecScatterBegin(self.sct, vec_from.vec, vec_to.vec,
                                caddv, csctm) )

    def scatterEnd(self, Vec vec_from, Vec vec_to, addv=None, mode=None):
        cdef PetscInsertMode  caddv = insertmode(addv)
        cdef PetscScatterMode csctm = scattermode(mode)
        CHKERR( VecScatterEnd(self.sct, vec_from.vec, vec_to.vec,
                              caddv, csctm) )

    def scatter(self, Vec vec_from, Vec vec_to, addv=None, mode=None):
        cdef PetscInsertMode  caddv = insertmode(addv)
        cdef PetscScatterMode csctm = scattermode(mode)
        CHKERR( VecScatterBegin(self.sct, vec_from.vec, vec_to.vec,
                                caddv, csctm) )
        CHKERR( VecScatterEnd(self.sct, vec_from.vec, vec_to.vec,
                              caddv, csctm) )

# --------------------------------------------------------------------

del ScatterType

# --------------------------------------------------------------------
