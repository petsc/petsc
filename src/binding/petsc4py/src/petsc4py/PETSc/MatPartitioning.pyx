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
    """Object for managing the partitioning of a matrix or graph."""

    Type = MatPartitioningType

    def __cinit__(self):
        self.obj = <PetscObject*> &self.part
        self.part = NULL

    def __call__(self):
        return self.getValue()

    def view(self, Viewer viewer=None) -> None:
        """View the partitioning data structure.

        Collective.

        Parameters
        ----------
        viewer
            A `Viewer` to display the graph.

        See Also
        --------
        petsc.MatPartitioningView

        """
        assert self.obj != NULL
        cdef PetscViewer vwr = NULL
        if viewer is not None: vwr = viewer.vwr
        CHKERR( MatPartitioningView(self.part, vwr) )

    def destroy(self) -> Self:
        """Destroy the partitioning context.

        Collective.

        See Also
        --------
        create, petsc.MatPartitioningDestroy

        """
        CHKERR( MatPartitioningDestroy(&self.part) )
        return self

    def create(self, comm: Comm | None = None) -> Self:
        """Create a partitioning context.

        Collective.

        Parameters
        ----------
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        destroy, petsc.MatPartitioningCreate

        """
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        CHKERR( MatPartitioningCreate(ccomm, &self.part) )
        return self

    def setType(self, matpartitioning_type: Type | str) -> None:
        """Set the type of the partitioner to use.

        Collective.

        Parameters
        ----------
        matpartitioning_type
            The partitioner type.

        See Also
        --------
        getType, petsc.MatPartitioningSetType

        """
        cdef PetscMatPartitioningType cval = NULL
        matpartitioning_type = str2bytes(matpartitioning_type, &cval)
        CHKERR( MatPartitioningSetType(self.part, cval) )

    def getType(self) -> str:
        """Return the partitioning method.

        Not collective.

        See Also
        --------
        setType, petsc.MatPartitioningGetType

        """
        cdef PetscMatPartitioningType cval = NULL
        CHKERR( MatPartitioningGetType(self.part, &cval) )
        return bytes2str(cval)

    def setFromOptions(self) -> None:
        """Set parameters in the partitioner from the options database.

        Collective.

        See Also
        --------
        petsc_options, petsc.MatPartitioningSetFromOptions

        """
        CHKERR( MatPartitioningSetFromOptions(self.part) )

    def setAdjacency(self, Mat adj) -> None:
        """Set the adjacency graph (matrix) of the thing to be partitioned.

        Collective.

        Parameters
        ----------
        adj
            The adjacency matrix, this can be any `Mat.Type` but the natural
            representation is `Mat.Type.MPIADJ`.

        See Also
        --------
        petsc.MatPartitioningSetAdjacency

        """
        CHKERR( MatPartitioningSetAdjacency(self.part, adj.mat) )

    def apply(self, IS partitioning) -> None:
        """Return a partitioning for the graph represented by a sparse matrix.

        Collective.

        For each local node this tells the processor number that that node is
        assigned to.

        See Also
        --------
        petsc.MatPartitioningApply

        """
        CHKERR( MatPartitioningApply(self.part, &partitioning.iset) )

# --------------------------------------------------------------------

del MatPartitioningType

# --------------------------------------------------------------------
