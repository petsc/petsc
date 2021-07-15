# --------------------------------------------------------------------

class MatPartitioningType(object):
    PARTITIONINGCURRENT  = S_(MATPARTITIONINGCURRENT)
    PARTITIONINGAVERAGE  = S_(MATPARTITIONINGAVERAGE)
    PARTITIONINGSQUARE   = S_(MATPARTITIONINGSQUARE)
    PARTITIONINGPARMETIS = S_(MATPARTITIONINGPARMETIS)
    PARTITIONINGCHACO    = S_(MATPARTITIONINGCHACO)
    PARTITIONINGPARTY    = S_(MATPARTITIONINGPARTY)
    PARTITIONINGPTSCOTCH = S_(MATPARTITIONINGPTSCOTCH)
    PARTITIONINGHIERARCH = S_(MATPARTITIONINGHIERARCH)

# --------------------------------------------------------------------

cdef class MatPartitioning(Object):

    Type = MatPartitioningType

    def __cinit__(self):
        self.obj = <PetscObject*> &self.part
        self.part = NULL

    def __call__(self):
        return self.getValue()

    def view(self, Viewer viewer=None):
        assert self.obj != NULL
        cdef PetscViewer vwr = NULL
        if viewer is not None: vwr = viewer.vwr
        CHKERR( MatPartitioningView(self.part, vwr) )

    def destroy(self):
        CHKERR( MatPartitioningDestroy(&self.part) )
        return self

    def create(self, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        CHKERR( MatPartitioningCreate(ccomm, &self.part) )
        return self

    def setType(self, matpartitioning_type):
        cdef PetscMatPartitioningType cval = NULL
        matpartitioning_type = str2bytes(matpartitioning_type, &cval)
        CHKERR( MatPartitioningSetType(self.part, cval) )

    def getType(self):
        cdef PetscMatPartitioningType cval = NULL
        CHKERR( MatPartitioningGetType(self.part, &cval) )
        return bytes2str(cval)

    def setFromOptions(self):
        CHKERR( MatPartitioningSetFromOptions(self.part) )

    def setAdjacency(self, Mat adj):
        CHKERR( MatPartitioningSetAdjacency(self.part, adj.mat) )

    def apply(self, IS partitioning):
        CHKERR( MatPartitioningApply(self.part, &partitioning.iset) )

# --------------------------------------------------------------------

del MatPartitioningType

# --------------------------------------------------------------------
