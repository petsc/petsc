# --------------------------------------------------------------------

class PartitionerType(object):
    PARMETIS        = S_(PETSCPARTITIONERPARMETIS)
    PTSCOTCH        = S_(PETSCPARTITIONERPTSCOTCH)
    CHACO           = S_(PETSCPARTITIONERCHACO)
    SIMPLE          = S_(PETSCPARTITIONERSIMPLE)
    SHELL           = S_(PETSCPARTITIONERSHELL)
    GATHER          = S_(PETSCPARTITIONERGATHER)
    MATPARTITIONING = S_(PETSCPARTITIONERMATPARTITIONING)

# --------------------------------------------------------------------

cdef class Partitioner(Object):

    Type = PartitionerType

    def __cinit__(self):
        self.obj = <PetscObject*> &self.part
        self.part = NULL

    def view(self, Viewer viewer=None):
        cdef PetscViewer vwr = NULL
        if viewer is not None: vwr = viewer.vwr
        CHKERR( PetscPartitionerView(self.part, vwr) )

    def destroy(self):
        CHKERR( PetscPartitionerDestroy(&self.part) )
        return self

    def create(self, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscPartitioner newpart = NULL
        CHKERR( PetscPartitionerCreate(ccomm, &newpart) )
        PetscCLEAR(self.obj); self.part = newpart
        return self

    def setType(self, part_type):
        cdef PetscPartitionerType cval = NULL
        part_type = str2bytes(part_type, &cval)
        CHKERR( PetscPartitionerSetType(self.part, cval) )

    def getType(self):
        cdef PetscPartitionerType cval = NULL
        CHKERR( PetscPartitionerGetType(self.part, &cval) )
        return bytes2str(cval)

    def setFromOptions(self):
        CHKERR( PetscPartitionerSetFromOptions(self.part) )

    def setUp(self):
        CHKERR( PetscPartitionerSetUp(self.part) )

    def reset(self):
        CHKERR( PetscPartitionerReset(self.part) )

    def setShellPartition(self, numProcs, sizes=None, points=None):
        cdef PetscInt cnumProcs = asInt(numProcs)
        cdef PetscInt *csizes = NULL
        cdef PetscInt *cpoints = NULL
        cdef PetscInt nsize = 0
        if sizes is not None:
            sizes = iarray_i(sizes, &nsize, &csizes)
            if nsize != cnumProcs:
                raise ValueError("sizes array should have %d entries (has %d)" %
                                 numProcs, toInt(nsize))
            if points is None:
                raise ValueError("Must provide both sizes and points arrays")
        if points is not None:
            points = iarray_i(points, NULL, &cpoints)
        CHKERR( PetscPartitionerShellSetPartition(self.part, cnumProcs,
                                                  csizes, cpoints) )
