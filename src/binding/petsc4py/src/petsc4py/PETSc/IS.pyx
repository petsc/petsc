# --------------------------------------------------------------------

class ISType(object):
    GENERAL = S_(ISGENERAL)
    BLOCK   = S_(ISBLOCK)
    STRIDE  = S_(ISSTRIDE)

# --------------------------------------------------------------------

cdef class IS(Object):
    """A collection of indices.

    IS objects are used to index into vectors and matrices and to set up vector
    scatters.

    See Also
    --------
    petsc.IS

    """

    Type = ISType

    #

    def __cinit__(self):
        self.obj = <PetscObject*> &self.iset
        self.iset = NULL

    # buffer interface (PEP 3118)

    def __getbuffer__(self, Py_buffer *view, int flags):
        cdef _IS_buffer buf = _IS_buffer(self)
        buf.acquirebuffer(view, flags)

    def __releasebuffer__(self, Py_buffer *view):
        cdef _IS_buffer buf = <_IS_buffer>(view.obj)
        buf.releasebuffer(view)
        <void>self # unused


    # 'with' statement (PEP 343)

    def __enter__(self):
        cdef _IS_buffer buf = _IS_buffer(self)
        self.set_attr('__buffer__', buf)
        return buf.enter()

    def __exit__(self, *exc):
        cdef _IS_buffer buf = self.get_attr('__buffer__')
        self.set_attr('__buffer__', None)
        return buf.exit()
    #

    def view(self, Viewer viewer=None) -> None:
        """Display the index set.

        Collective.

        Parameters
        ----------
        viewer
            Viewer used to display the IS.

        See Also
        --------
        petsc.ISView

        """
        cdef PetscViewer cviewer = NULL
        if viewer is not None: cviewer = viewer.vwr
        CHKERR( ISView(self.iset, cviewer) )

    def destroy(self) -> Self:
        """Destroy the index set.

        Collective.

        See Also
        --------
        petsc.ISDestroy

        """
        CHKERR( ISDestroy(&self.iset) )
        return self

    def create(self, comm: Comm | None = None) -> Self:
        """Create an IS.

        Collective.

        Parameters
        ----------
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        petsc.ISCreate

        """
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscIS newiset = NULL
        CHKERR( ISCreate(ccomm, &newiset) )
        PetscCLEAR(self.obj); self.iset = newiset
        return self

    def setType(self, is_type: IS.Type | str) -> None:
        """Set the type of the index set.

        Collective.

        Parameters
        ----------
        is_type
            The index set type.

        See Also
        --------
        petsc.ISSetType

        """
        cdef PetscISType cval = NULL
        is_type = str2bytes(is_type, &cval)
        CHKERR( ISSetType(self.iset, cval) )

    def getType(self) -> str:
        """Return the index set type associated with the IS.

        Not collective.

        See Also
        --------
        petsc.ISGetType

        """
        cdef PetscISType cval = NULL
        CHKERR( ISGetType(self.iset, &cval) )
        return bytes2str(cval)

    def createGeneral(
        self,
        indices: Sequence[int],
        comm: Comm | None = None
    ) -> Self:
        """Create an IS with indices.

        Collective.

        Parameters
        ----------
        indices
            Integer array.
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        petsc.ISCreateGeneral

        """
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscInt nidx = 0, *idx = NULL
        cdef PetscCopyMode cm = PETSC_COPY_VALUES
        cdef PetscIS newiset = NULL
        indices = iarray_i(indices, &nidx, &idx)
        CHKERR( ISCreateGeneral(ccomm, nidx, idx, cm, &newiset) )
        PetscCLEAR(self.obj); self.iset = newiset
        return self

    def createBlock(
        self,
        bsize: int,
        indices: Sequence[int],
        comm: Comm | None = None
    ) -> Self:
        """Create a blocked index set.

        Collective.

        Parameters
        ----------
        bsize
            Block size.
        indices
            Integer array of indices.
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        petsc.ISCreateBlock

        """
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscInt bs = asInt(bsize)
        cdef PetscInt nidx = 0, *idx = NULL
        cdef PetscCopyMode cm = PETSC_COPY_VALUES
        cdef PetscIS newiset = NULL
        indices = iarray_i(indices, &nidx, &idx)
        CHKERR( ISCreateBlock(ccomm, bs, nidx, idx, cm, &newiset) )
        PetscCLEAR(self.obj); self.iset = newiset
        return self

    def createStride(
        self,
        size: int,
        first: int=0,
        step: int=0,
        comm: Comm | None = None
    ) -> Self:
        """Create an index set consisting of evenly spaced values.

        Collective.

        Parameters
        ----------
        size
            The length of the locally owned portion of the index set.
        first
            The first element of the index set.
        step
            The difference between adjacent indices.
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        petsc.ISCreateStride

        """
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscInt csize  = asInt(size)
        cdef PetscInt cfirst = asInt(first)
        cdef PetscInt cstep  = asInt(step)
        cdef PetscIS newiset = NULL
        CHKERR( ISCreateStride(ccomm, csize, cfirst, cstep, &newiset) )
        PetscCLEAR(self.obj); self.iset = newiset
        return self

    def duplicate(self) -> IS:
        """Create a copy of the index set.

        Collective.

        See Also
        --------
        IS.copy, petsc.ISDuplicate

        """
        cdef IS iset = type(self)()
        CHKERR( ISDuplicate(self.iset, &iset.iset) )
        return iset

    def copy(self, IS result=None) -> IS:
        """Copy the contents of the index set into another.

        Collective.

        Parameters
        ----------
        result
            The target index set. If `None` then `IS.duplicate` is called first.

        Returns
        -------
        IS
            The copied index set. If ``result`` is not `None` then this is
            returned here.

        See Also
        --------
        IS.duplicate, petsc.ISCopy

        """
        if result is None:
            result = type(self)()
        if result.iset == NULL:
            CHKERR( ISDuplicate(self.iset, &result.iset) )
        CHKERR( ISCopy(self.iset, result.iset) )
        return result

    def load(self, Viewer viewer) -> Self:
        """Load a stored index set.

        Collective.

        Parameters
        ----------
        viewer
            Binary file viewer, either `Viewer.Type.BINARY` or `Viewer.Type.HDF5`.

        See Also
        --------
        petsc.ISLoad

        """
        cdef MPI_Comm comm = MPI_COMM_NULL
        cdef PetscObject obj = <PetscObject>(viewer.vwr)
        if self.iset == NULL:
            CHKERR( PetscObjectGetComm(obj, &comm) )
            CHKERR( ISCreate(comm, &self.iset) )
        CHKERR( ISLoad(self.iset, viewer.vwr) )
        return self

    def allGather(self) -> IS:
        """Concatenate index sets stored across processors.

        Collective.

        The returned index set will be the same on every processor.

        See Also
        --------
        petsc.ISAllGather

        """
        cdef IS iset = IS()
        CHKERR( ISAllGather(self.iset, &iset.iset) )
        return iset

    def toGeneral(self) -> Self:
        """Convert the index set type to `IS.Type.GENERAL`.

        Collective.

        See Also
        --------
        petsc.ISToGeneral, petsc.ISType

        """
        CHKERR( ISToGeneral(self.iset) )
        return self

    def buildTwoSided(self, IS toindx=None) -> IS:
        """Create an index set describing a global mapping.

        Collective.

        This function generates an index set that contains new numbers from
        remote or local on the index set.

        Parameters
        ----------
        toindx
            Index set describing which indices to send, default is to send
            natural numbering.

        Returns
        -------
        IS
            New index set containing the new numbers from remote or local.

        See Also
        --------
        petsc.ISBuildTwoSided

        """
        cdef PetscIS ctoindx = NULL
        if toindx is not None: ctoindx = toindx.iset
        cdef IS result = IS()
        CHKERR( ISBuildTwoSided(self.iset, ctoindx, &result.iset) )
        return result

    def invertPermutation(self, nlocal: int | None = None) -> IS:
        """Invert the index set.

        Collective.

        For this to be correct the index set must be a permutation.

        Parameters
        ----------
        nlocal
            The number of indices on this processor in the resulting index set,
            defaults to ``PETSC_DECIDE``.

        See Also
        --------
        petsc.ISInvertPermutation

        """
        cdef PetscInt cnlocal = PETSC_DECIDE
        if nlocal is not None: cnlocal = asInt(nlocal)
        cdef IS iset = IS()
        CHKERR( ISInvertPermutation(self.iset, cnlocal, &iset.iset) )
        return iset

    def getSize(self) -> int:
        """Return the global length of an index set.

        Not collective.

        See Also
        --------
        petsc.ISGetSize

        """
        cdef PetscInt N = 0
        CHKERR( ISGetSize(self.iset, &N) )
        return toInt(N)

    def getLocalSize(self) -> int:
        """Return the process-local length of the index set.

        Not collective.

        See Also
        --------
        petsc.ISGetLocalSize

        """
        cdef PetscInt n = 0
        CHKERR( ISGetLocalSize(self.iset, &n) )
        return toInt(n)

    def getSizes(self) -> tuple[int, int]:
        """Return the local and global sizes of the index set.

        Not collective.

        Returns
        -------
        local_size : int
            The local size.
        global_size : int
            The global size.

        See Also
        --------
        IS.getLocalSize, IS.getSize

        """
        cdef PetscInt n = 0, N = 0
        CHKERR( ISGetLocalSize(self.iset, &n) )
        CHKERR( ISGetSize(self.iset, &N) )
        return (toInt(n), toInt(N))

    def getBlockSize(self) -> int:
        """Return the number of elements in a block.

        Not collective.

        See Also
        --------
        petsc.ISGetBlockSize

        """
        cdef PetscInt bs = 1
        CHKERR( ISGetBlockSize(self.iset, &bs) )
        return toInt(bs)

    def setBlockSize(self, bs: int) -> None:
        """Set the block size of the index set.

        Logically collective.

        Parameters
        ----------
        bs
            Block size.

        See Also
        --------
        petsc.ISSetBlockSize

        """
        cdef PetscInt cbs = asInt(bs)
        CHKERR( ISSetBlockSize(self.iset, cbs) )

    def sort(self) -> Self:
        """Sort the indices of an index set.

        Collective.

        See Also
        --------
        petsc.ISSort

        """
        CHKERR( ISSort(self.iset) )
        return self

    def isSorted(self) -> bool:
        """Return whether the indices have been sorted.

        Collective.

        See Also
        --------
        petsc.ISSorted

        """
        cdef PetscBool flag = PETSC_FALSE
        CHKERR( ISSorted(self.iset, &flag) )
        return toBool(flag)

    def setPermutation(self) -> Self:
        """Mark the index set as being a permutation.

        Logically collective.

        See Also
        --------
        petsc.ISSetPermutation

        """
        CHKERR( ISSetPermutation(self.iset) )
        return self

    def isPermutation(self) -> bool:
        """Return whether an index set has been declared to be a permutation.

        Logically collective.

        See Also
        --------
        petsc.ISPermutation

        """
        cdef PetscBool flag = PETSC_FALSE
        CHKERR( ISPermutation(self.iset, &flag) )
        return toBool(flag)

    def setIdentity(self) -> Self:
        """Mark the index set as being an identity.

        Logically collective.

        See Also
        --------
        petsc.ISSetIdentity

        """
        CHKERR( ISSetIdentity(self.iset) )
        return self

    def isIdentity(self) -> bool:
        """Return whether the index set has been declared as an identity.

        Collective.

        See Also
        --------
        petsc.ISIdentity

        """
        cdef PetscBool flag = PETSC_FALSE
        CHKERR( ISIdentity(self.iset, &flag) )
        return toBool(flag)

    def equal(self, IS iset) -> bool:
        """Return whether the index sets have the same set of indices or not.

        Collective on ``self``.

        Parameters
        ----------
        iset
            The index set to compare indices with.

        See Also
        --------
        petsc.ISEqual

        """
        cdef PetscBool flag = PETSC_FALSE
        CHKERR( ISEqual(self.iset, iset.iset, &flag) )
        return toBool(flag)

    def sum(self, IS iset) -> IS:
        """Compute the union of two (sorted) index sets.

        Sequential only.

        Parameters
        ----------
        iset
            The index set to compute the union with.

        See Also
        --------
        petsc.ISSum

        """
        cdef IS out = IS()
        CHKERR( ISSum(self.iset, iset.iset, &out.iset) )
        return out

    def expand(self, IS iset) -> IS:
        """Compute the union of two (possibly unsorted) index sets.

        Collective on ``self``.

        To compute the union, `expand` concatenates the two index sets
        and removes any duplicates.

        Parameters
        ----------
        iset
            Index set to compute the union with.

        Returns
        -------
        IS
            The new, combined, index set.

        See Also
        --------
        petsc.ISExpand

        """
        cdef IS out = IS()
        CHKERR( ISExpand(self.iset, iset.iset, &out.iset) )
        return out

    def union(self, IS iset) -> IS: # XXX review this
        """Compute the union of two (possibly unsorted) index sets.

        This function will call either `petsc.ISSum` or `petsc.ISExpand` depending
        on whether or not the input sets are already sorted.

        Sequential only (as `petsc.ISSum` is sequential only).

        Parameters
        ----------
        iset
            Index set to compute the union with.

        Returns
        -------
        IS
            The new, combined, index set.

        See Also
        --------
        IS.expand, IS.sum

        """
        cdef PetscBool flag1=PETSC_FALSE, flag2=PETSC_FALSE
        CHKERR( ISSorted(self.iset, &flag1) )
        CHKERR( ISSorted(iset.iset, &flag2) )
        cdef IS out = IS()
        if flag1==PETSC_TRUE and flag2==PETSC_TRUE:
            CHKERR( ISSum(self.iset, iset.iset, &out.iset) )
        else:
            CHKERR( ISExpand(self.iset, iset.iset, &out.iset) )
        return out

    def difference(self, IS iset: IS) -> IS:
        """Compute the difference between two index sets.

        Collective.

        Parameters
        ----------
        iset
            Index set to compute the difference with.

        Returns
        -------
        IS
            Index set representing the difference between ``self`` and ``iset``.

        See Also
        --------
        petsc.ISDifference

        """
        cdef IS out = IS()
        CHKERR( ISDifference(self.iset, iset.iset, &out.iset) )
        return out

    def complement(self, nmin: int, nmax: int) -> IS:
        """Create a complement index set.

        Collective.

        The complement set of indices is all indices that are not
        in the provided set (and within the provided bounds).

        Parameters
        ----------
        nmin
            Minimum index that can be found in the local part of the complement
            index set.
        nmax
            One greater than the maximum index that can be found in the local
            part of the complement index set.

        Notes
        -----
        For a parallel index set, this will generate the local part of the
        complement on each process.

        To generate the entire complement (on each process) of a parallel
        index set, first call `IS.allGather` and then call this method.

        See Also
        --------
        IS.allGather, petsc.ISComplement

        """
        cdef PetscInt cnmin = asInt(nmin)
        cdef PetscInt cnmax = asInt(nmax)
        cdef IS out = IS()
        CHKERR( ISComplement(self.iset, cnmin, cnmax, &out.iset) )
        return out

    def embed(self, IS iset, drop: bool) -> IS:
        """Embed ``self`` into ``iset``.

        Not collective.

        The embedding is performed by finding the locations in ``iset`` that
        have the same indices as ``self``.

        Parameters
        ----------
        iset
            The index set to embed into.
        drop
            Flag indicating whether to drop indices from ``self`` that are not
            in ``iset``.

        Returns
        -------
        IS
            The embedded index set.

        See Also
        --------
        petsc.ISEmbed

        """
        cdef PetscBool bval = drop
        cdef IS out = IS()
        CHKERR( ISEmbed(self.iset, iset.iset, bval, &out.iset) )
        return out

    def renumber(self, IS mult=None) -> tuple[int, IS]:
        """Renumber the non-negative entries of an index set, starting from 0.

        Collective.

        Parameters
        ----------
        mult
            The multiplicity of each entry in ``self``, default implies a
            multiplicity of 1.

        Returns
        -------
        int
            One past the largest entry of the new index set.
        IS
            The renumbered index set.

        See Also
        --------
        petsc.ISRenumber

        """
        cdef PetscIS mlt = NULL
        if mult is not None: mlt = mult.iset
        cdef IS out = IS()
        cdef PetscInt n = 0
        CHKERR( ISRenumber(self.iset, mlt, &n, &out.iset) )
        return (toInt(n), out)
    #

    def setIndices(self, indices: Sequence[int]) -> None:
        """Set the indices of an index set.

        Logically collective.

        The index set is assumed to be of type `IS.Type.GENERAL`.

        See Also
        --------
        petsc.ISGeneralSetIndices

        """
        cdef PetscInt nidx = 0, *idx = NULL
        cdef PetscCopyMode cm = PETSC_COPY_VALUES
        indices = iarray_i(indices, &nidx, &idx)
        CHKERR( ISGeneralSetIndices(self.iset, nidx, idx, cm) )

    def getIndices(self) -> ArrayInt:
        """Return the indices of the index set.

        Not collective.

        See Also
        --------
        petsc.ISGetIndices

        """
        cdef PetscInt size = 0
        cdef const PetscInt *indices = NULL
        CHKERR( ISGetLocalSize(self.iset, &size) )
        CHKERR( ISGetIndices(self.iset, &indices) )
        cdef object oindices = None
        try:
            oindices = array_i(size, indices)
        finally:
            CHKERR( ISRestoreIndices(self.iset, &indices) )
        return oindices

    def setBlockIndices(self, bsize: int, indices: Sequence[int]) -> None:
        """Set the indices for an index set with type `IS.Type.BLOCK`.

        Collective.

        Parameters
        ----------
        bsize
            Number of elements in each block.
        indices
            List of integers.

        See Also
        --------
        petsc.ISBlockSetIndices

        """
        cdef PetscInt bs = asInt(bsize)
        cdef PetscInt nidx = 0, *idx = NULL
        cdef PetscCopyMode cm = PETSC_COPY_VALUES
        indices = iarray_i(indices, &nidx, &idx)
        CHKERR( ISBlockSetIndices(self.iset, bs, nidx, idx, cm) )

    def getBlockIndices(self) -> ArrayInt:
        """Return the indices of an index set with type `IS.Type.BLOCK`.

        Not collective.

        See Also
        --------
        petsc.ISBlockGetIndices

        """
        cdef PetscInt size = 0, bs = 1
        cdef const PetscInt *indices = NULL
        CHKERR( ISGetLocalSize(self.iset, &size) )
        CHKERR( ISGetBlockSize(self.iset, &bs) )
        CHKERR( ISBlockGetIndices(self.iset, &indices) )
        cdef object oindices = None
        try:
            oindices = array_i(size//bs, indices)
        finally:
            CHKERR( ISBlockRestoreIndices(self.iset, &indices) )
        return oindices

    def setStride(self, size: int, first: int = 0, step: int = 1) -> None:
        """Set the stride information for an index set with type `IS.Type.STRIDE`.

        Logically collective.

        Parameters
        ----------
        size
            Length of the locally owned portion of the index set.
        first
            First element of the index set.
        step
            Difference between adjacent indices.

        See Also
        --------
        petsc.ISStrideSetStride

        """
        cdef PetscInt csize = asInt(size)
        cdef PetscInt cfirst = asInt(first)
        cdef PetscInt cstep = asInt(step)
        CHKERR( ISStrideSetStride(self.iset, csize, cfirst, cstep) )

    def getStride(self) -> tuple[int, int, int]:
        """Return size and stride information.

        Not collective.

        Returns
        -------
        size : int
            Length of the locally owned portion of the index set.
        first : int
            First element of the index set.
        step : int
            Difference between adjacent indices.

        See Also
        --------
        petsc.ISGetLocalSize, petsc.ISStrideGetInfo

        """
        cdef PetscInt size=0, first=0, step=0
        CHKERR( ISGetLocalSize(self.iset, &size) )
        CHKERR( ISStrideGetInfo(self.iset, &first, &step) )
        return (toInt(size), toInt(first), toInt(step))

    def getInfo(self) -> tuple[int, int]:
        """Return stride information for an index set with type `IS.Type.STRIDE`.

        Not collective.

        Returns
        -------
        first : int
            First element of the index set.
        step : int
            Difference between adjacent indices.

        See Also
        --------
        IS.getStride, petsc.ISStrideGetInfo

        """
        cdef PetscInt first = 0, step = 0
        CHKERR( ISStrideGetInfo(self.iset, &first, &step) )
        return (toInt(first), toInt(step))

    #

    property permutation:
        """`True` if index set is a permutation, `False` otherwise.

        Logically collective.

        See Also
        --------
        IS.isPermutation

        """
        def __get__(self) -> bool:
            return self.isPermutation()

    property identity:
        """`True` if index set is an identity, `False` otherwise.

        Collective.

        See Also
        --------
        IS.isIdentity

        """
        def __get__(self) -> bool:
            return self.isIdentity()

    property sorted:
        """`True` if index set is sorted, `False` otherwise.

        Collective.

        See Also
        --------
        IS.isSorted

        """
        def __get__(self) -> bool:
            return self.isSorted()

    #

    property sizes:
        """The local and global sizes of the index set.

        Not collective.

        See Also
        --------
        IS.getSizes

        """
        def __get__(self) -> tuple[int, int]:
            return self.getSizes()

    property size:
        """The global size of the index set.

        Not collective.

        See Also
        --------
        IS.getSize

        """
        def __get__(self) -> int:
            return self.getSize()

    property local_size:
        """The local size of the index set.

        Not collective.

        See Also
        --------
        IS.getLocalSize

        """
        def __get__(self) -> int:
            return self.getLocalSize()

    property block_size:
        """The number of elements in a block.

        Not collective.

        See Also
        --------
        IS.getBlockSize

        """
        def __get__(self) -> int:
            return self.getBlockSize()

    property indices:
        """The indices of the index set.

        Not collective.

        See Also
        --------
        IS.getIndices

        """
        def __get__(self) -> ArrayInt:
            return self.getIndices()

    property array:
        """View of the index set as an array of integers.

        Not collective.

        """
        def __get__(self) -> ArrayInt:
            return asarray(self)

    # --- NumPy array interface (legacy) ---

    property __array_interface__:
        def __get__(self):
            cdef _IS_buffer buf = _IS_buffer(self)
            return buf.__array_interface__

# --------------------------------------------------------------------


class GLMapMode(object):
    """Enum describing mapping behavior for global-to-local maps when global indices are missing.

    MASK
        Give missing global indices a local index of -1.
    DROP
        Drop missing global indices.

    See Also
    --------
    petsc.ISGlobalToLocalMappingMode

    """
    MASK = PETSC_IS_GTOLM_MASK
    DROP = PETSC_IS_GTOLM_DROP


class LGMapType(object):
    BASIC = S_(ISLOCALTOGLOBALMAPPINGBASIC)
    HASH  = S_(ISLOCALTOGLOBALMAPPINGHASH)


# --------------------------------------------------------------------

cdef class LGMap(Object):
    """Mapping from an arbitrary local ordering from ``0`` to ``n-1`` to a global PETSc ordering used by a vector or matrix.

    See Also
    --------
    petsc.ISLocalToGlobalMapping

    """

    MapMode = GLMapMode

    Type = LGMapType
    #

    def __cinit__(self) -> None:
        self.obj = <PetscObject*> &self.lgm
        self.lgm = NULL

    def __call__(
        self,
        indices: Sequence[int],
        result: ArrayInt | None = None
    ) -> None:
        """Convert a locally numbered list of integers to a global numbering.

        Not collective.

        Parameters
        ----------
        indices
            Input indices in local numbering.
        result
            Array to write the global numbering to. If `None` then a
            new array will be allocated.

        See Also
        --------
        IS.apply, petsc.ISLocalToGlobalMappingApply

        """
        self.apply(indices, result)

    #

    def setType(self, lgmap_type: LGMap.Type | str) -> None:
        """Set the type of the local-to-global map.

        Logically collective.

        Parameters
        ----------
        lgmap_type
            The type of the local-to-global mapping.

        Notes
        -----
        Use ``-islocaltoglobalmapping_type`` to set the type in the
        options database.

        See Also
        --------
        petsc_options, petsc.ISLocalToGlobalMappingSetType

        """
        cdef PetscISLocalToGlobalMappingType cval = NULL
        lgmap_type = str2bytes(lgmap_type, &cval)
        CHKERR( ISLocalToGlobalMappingSetType(self.lgm, cval) )

    def setFromOptions(self) -> None:
        """Set mapping options from the options database.

        Not collective.

        See Also
        --------
        petsc_options, petsc.ISLocalToGlobalMappingSetFromOptions

        """
        CHKERR( ISLocalToGlobalMappingSetFromOptions(self.lgm) )

    def view(self, Viewer viewer=None) -> None:
        """View the local-to-global mapping.

        Not collective.

        Parameters
        ----------
        viewer
            Viewer instance, defaults to an instance of `Viewer.Type.ASCII`.

        See Also
        --------
        petsc.ISLocalToGlobalMappingView

        """
        cdef PetscViewer cviewer = NULL
        if viewer is not None: cviewer = viewer.vwr
        CHKERR( ISLocalToGlobalMappingView(self.lgm, cviewer) )

    def destroy(self) -> Self:
        """Destroy the local-to-global mapping.

        Not collective.

        See Also
        --------
        petsc.ISLocalToGlobalMappingDestroy

        """
        CHKERR( ISLocalToGlobalMappingDestroy(&self.lgm) )
        return self

    def create(
        self,
        indices: Sequence[int],
        bsize: int | None = None,
        comm: Comm | None = None
    ) -> Self:
        """Create a local-to-global mapping.

        Not collective.

        Parameters
        ----------
        indices
            Global index for each local element.
        bsize
            Block size, defaults to 1.
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        petsc.ISLocalToGlobalMappingCreate

        """
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscInt bs = 1, nidx = 0, *idx = NULL
        cdef PetscCopyMode cm = PETSC_COPY_VALUES
        cdef PetscLGMap newlgm = NULL
        if bsize is not None: bs = asInt(bsize)
        if bs == PETSC_DECIDE: bs = 1
        indices = iarray_i(indices, &nidx, &idx)
        CHKERR( ISLocalToGlobalMappingCreate(
                ccomm, bs, nidx, idx, cm, &newlgm) )
        PetscCLEAR(self.obj); self.lgm = newlgm
        return self

    def createIS(self, IS iset) -> Self:
        """Create a local-to-global mapping from an index set.

        Not collective.

        Parameters
        ----------
        iset
            Index set containing the global numbers for each local number.

        See Also
        --------
        petsc.ISLocalToGlobalMappingCreateIS

        """
        cdef PetscLGMap newlgm = NULL
        CHKERR( ISLocalToGlobalMappingCreateIS(
            iset.iset, &newlgm) )
        PetscCLEAR(self.obj); self.lgm = newlgm
        return self

    def createSF(self, SF sf, start: int) -> Self:
        """Create a local-to-global mapping from a star forest.

        Collective.

        Parameters
        ----------
        sf
            Star forest mapping contiguous local indices to (rank, offset).
        start
            First global index on this process.

        See Also
        --------
        petsc.ISLocalToGlobalMappingCreateSF

        """
        cdef PetscLGMap newlgm = NULL
        cdef PetscInt cstart = asInt(start)
        CHKERR( ISLocalToGlobalMappingCreateSF(sf.sf, cstart, &newlgm) )
        PetscCLEAR(self.obj); self.lgm = newlgm
        return self

    def getSize(self) -> int:
        """Return the local size of the local-to-global mapping.

        Not collective.

        See Also
        --------
        petsc.ISLocalToGlobalMappingGetSize

        """
        cdef PetscInt n = 0
        CHKERR( ISLocalToGlobalMappingGetSize(self.lgm, &n) )
        return toInt(n)

    def getBlockSize(self) -> int:
        """Return the block size of the local-to-global mapping.

        Not collective.

        See Also
        --------
        petsc.ISLocalToGlobalMappingGetBlockSize

        """
        cdef PetscInt bs = 1
        CHKERR( ISLocalToGlobalMappingGetBlockSize(self.lgm, &bs) )
        return toInt(bs)

    def getIndices(self) -> ArrayInt:
        """Return the global indices for each local point in the mapping.

        Not collective.

        See Also
        --------
        petsc.ISLocalToGlobalMappingGetIndices

        """
        cdef PetscInt size = 0
        cdef const PetscInt *indices = NULL
        CHKERR( ISLocalToGlobalMappingGetSize(
                self.lgm, &size) )
        CHKERR( ISLocalToGlobalMappingGetIndices(
                self.lgm, &indices) )
        cdef object oindices = None
        try:
            oindices = array_i(size, indices)
        finally:
            CHKERR( ISLocalToGlobalMappingRestoreIndices(
                    self.lgm, &indices) )
        return oindices

    def getBlockIndices(self) -> ArrayInt:
        """Return the global indices for each local block.

        Not collective.

        See Also
        --------
        petsc.ISLocalToGlobalMappingGetBlockIndices

        """
        cdef PetscInt size = 0, bs = 1
        cdef const PetscInt *indices = NULL
        CHKERR( ISLocalToGlobalMappingGetSize(
                self.lgm, &size) )
        CHKERR( ISLocalToGlobalMappingGetBlockSize(
                self.lgm, &bs) )
        CHKERR( ISLocalToGlobalMappingGetBlockIndices(
                self.lgm, &indices) )
        cdef object oindices = None
        try:
            oindices = array_i(size//bs, indices)
        finally:
            CHKERR( ISLocalToGlobalMappingRestoreBlockIndices(
                    self.lgm, &indices) )
        return oindices

    def getInfo(self) -> dict[int, ArrayInt]:
        """Determine the indices shared with neighboring processes.

        Collective.

        Returns
        -------
        dict
            Mapping from neighboring processor number to an array of shared
            indices (in local numbering).

        See Also
        --------
        petsc.ISLocalToGlobalMappingGetInfo

        """
        cdef PetscInt i, nproc = 0, *procs = NULL,
        cdef PetscInt *numprocs = NULL, **indices = NULL
        cdef object neighs = { }
        CHKERR( ISLocalToGlobalMappingGetInfo(
                self.lgm, &nproc, &procs, &numprocs, &indices) )
        try:
            for i from 0 <= i < nproc:
                neighs[toInt(procs[i])] = array_i(numprocs[i], indices[i])
        finally:
            ISLocalToGlobalMappingRestoreInfo(
                self.lgm, &nproc, &procs, &numprocs, &indices)
        return neighs

    def getBlockInfo(self) -> dict[int, ArrayInt]:
        """Determine the block indices shared with neighboring processes.

        Collective.

        Returns
        -------
        dict
            Mapping from neighboring processor number to an array of shared
            block indices (in local numbering).

        See Also
        --------
        petsc.ISLocalToGlobalMappingGetBlockInfo

        """
        cdef PetscInt i, nproc = 0, *procs = NULL,
        cdef PetscInt *numprocs = NULL, **indices = NULL
        cdef object neighs = { }
        CHKERR( ISLocalToGlobalMappingGetBlockInfo(
                self.lgm, &nproc, &procs, &numprocs, &indices) )
        try:
            for i from 0 <= i < nproc:
                neighs[toInt(procs[i])] = array_i(numprocs[i], indices[i])
        finally:
            ISLocalToGlobalMappingRestoreBlockInfo(
                self.lgm, &nproc, &procs, &numprocs, &indices)
        return neighs

    #

    def apply(
        self,
        indices: Sequence[int],
        result: ArrayInt | None = None,
    ) -> ArrayInt:
        """Convert a locally numbered list of integers to a global numbering.

        Not collective.

        Parameters
        ----------
        indices
            Input indices in local numbering.
        result
            Array to write the global numbering to. If `None` then a
            new array will be allocated.

        Returns
        -------
        ArrayInt
            Indices in global numbering. If ``result`` is not `None` then this is
            returned here.

        See Also
        --------
        LGMap.applyBlock, petsc.ISLocalToGlobalMappingApply

        """
        cdef PetscInt niidx = 0, *iidx = NULL
        cdef PetscInt noidx = 0, *oidx = NULL
        indices = iarray_i(indices, &niidx, &iidx)
        if result is None: result = empty_i(niidx)
        result  = oarray_i(result,  &noidx, &oidx)
        assert niidx == noidx, "incompatible array sizes"
        CHKERR( ISLocalToGlobalMappingApply(
            self.lgm, niidx, iidx, oidx) )
        return result

    def applyBlock(
        self,
        indices: Sequence[int],
        result: ArrayInt | None = None,
    ) -> ArrayInt:
        """Convert a local block numbering to a global block numbering.

        Not collective.

        Parameters
        ----------
        indices
            Input block indices in local numbering.
        result
            Array to write the global numbering to. If `None` then a
            new array will be allocated.

        Returns
        -------
        ArrayInt
            Block indices in global numbering. If ``result`` is not `None`
            then this is returned here.

        See Also
        --------
        LGMap.apply, petsc.ISLocalToGlobalMappingApplyBlock

        """
        cdef PetscInt niidx = 0, *iidx = NULL
        cdef PetscInt noidx = 0, *oidx = NULL
        indices = iarray_i(indices, &niidx, &iidx)
        if result is None: result = empty_i(niidx)
        result  = oarray_i(result,  &noidx, &oidx)
        assert niidx == noidx, "incompatible array sizes"
        CHKERR( ISLocalToGlobalMappingApplyBlock(
            self.lgm, niidx, iidx, oidx) )
        return result

    def applyIS(self, IS iset) -> IS:
        """Create an index set with global numbering from a local numbering.

        Collective.

        Parameters
        ----------
        iset
            Index set with local numbering.

        Returns
        -------
        IS
            Index set with global numbering.

        See Also
        --------
        petsc.ISLocalToGlobalMappingApplyIS

        """
        cdef IS result = IS()
        CHKERR( ISLocalToGlobalMappingApplyIS(
            self.lgm, iset.iset, &result.iset) )
        return result

    def applyInverse(
        self,
        indices: Sequence[int],
        mode: GLMapMode | str | None = None,
    ) -> ArrayInt:
        """Compute local numbering from global numbering.

        Not collective.

        Parameters
        ----------
        indices
            Indices with a global numbering.
        mode
            Flag indicating what to do with indices that have no local value,
            defaults to ``"mask"``.

        Returns
        -------
        ArrayInt
            Indices with a local numbering.

        See Also
        --------
        petsc.ISGlobalToLocalMappingApply

        """
        cdef PetscGLMapMode cmode = PETSC_IS_GTOLM_MASK
        if mode is not None: cmode = mode
        cdef PetscInt n = 0, *idx = NULL
        indices = iarray_i(indices, &n, &idx)
        cdef PetscInt nout = n, *idxout = NULL
        if cmode != PETSC_IS_GTOLM_MASK:
            CHKERR( ISGlobalToLocalMappingApply(
                    self.lgm, cmode, n, idx, &nout, NULL) )
        result = oarray_i(empty_i(nout), &nout, &idxout)
        CHKERR( ISGlobalToLocalMappingApply(
                self.lgm, cmode, n, idx, &nout, idxout) )
        return result

    def applyBlockInverse(
        self,
        indices: Sequence[int],
        mode: GLMapMode | str | None = None,
    ) -> ArrayInt:
        """Compute blocked local numbering from blocked global numbering.

        Not collective.

        Parameters
        ----------
        indices
            Indices with a global block numbering.
        mode
            Flag indicating what to do with indices that have no local value,
            defaults to ``"mask"``.

        Returns
        -------
        ArrayInt
            Indices with a local block numbering.

        See Also
        --------
        petsc.ISGlobalToLocalMappingApplyBlock

        """
        cdef PetscGLMapMode cmode = PETSC_IS_GTOLM_MASK
        if mode is not None: cmode = mode
        cdef PetscInt n = 0, *idx = NULL
        indices = iarray_i(indices, &n, &idx)
        cdef PetscInt nout = n, *idxout = NULL
        if cmode != PETSC_IS_GTOLM_MASK:
            CHKERR( ISGlobalToLocalMappingApply(
                    self.lgm, cmode, n, idx, &nout, NULL) )
        result = oarray_i(empty_i(nout), &nout, &idxout)
        CHKERR( ISGlobalToLocalMappingApplyBlock(
                self.lgm, cmode, n, idx, &nout, idxout) )
        return result
    #

    property size:
        """The local size.

        Not collective.

        See Also
        --------
        LGMap.getSize

        """
        def __get__(self) -> int:
            return self.getSize()

    property block_size:
        """The block size.

        Not collective.

        See Also
        --------
        LGMap.getBlockSize

        """
        def __get__(self) -> int:
            return self.getBlockSize()

    property indices:
        """The global indices for each local point in the mapping.

        Not collective.

        See Also
        --------
        LGMap.getIndices, petsc.ISLocalToGlobalMappingGetIndices

        """
        def __get__(self) -> ArrayInt:
            return self.getIndices()

    property block_indices:
        """The global indices for each local block in the mapping.

        Not collective.

        See Also
        --------
        LGMap.getBlockIndices, petsc.ISLocalToGlobalMappingGetBlockIndices

        """
        def __get__(self) -> ArrayInt:
            return self.getBlockIndices()

    property info:
        """Mapping describing indices shared with neighboring processes.

        Collective.

        See Also
        --------
        LGMap.getInfo, petsc.ISLocalToGlobalMappingGetInfo

        """
        def __get__(self) -> dict[int, ArrayInt]:
            return self.getInfo()

    property block_info:
        """Mapping describing block indices shared with neighboring processes.

        Collective.

        See Also
        --------
        LGMap.getBlockInfo, petsc.ISLocalToGlobalMappingGetBlockInfo

        """
        def __get__(self) -> dict[int, ArrayInt]:
            return self.getBlockInfo()

# --------------------------------------------------------------------

del ISType
del GLMapMode
del LGMapType
# --------------------------------------------------------------------
