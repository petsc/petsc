# --------------------------------------------------------------------

class SFType(object):
    """The star forest types."""
    BASIC      = S_(PETSCSFBASIC)
    NEIGHBOR   = S_(PETSCSFNEIGHBOR)
    ALLGATHERV = S_(PETSCSFALLGATHERV)
    ALLGATHER  = S_(PETSCSFALLGATHER)
    GATHERV    = S_(PETSCSFGATHERV)
    GATHER     = S_(PETSCSFGATHER)
    ALLTOALL   = S_(PETSCSFALLTOALL)
    WINDOW     = S_(PETSCSFWINDOW)


# --------------------------------------------------------------------

cdef inline int sfcheckbuffer(
    ndarray data,
    object name,
    PetscInt nunits,
    object extent,
    bint writable,
) except -1:
    cdef object nrequired = toInt(nunits) * extent

    if not PyArray_ISCONTIGUOUS(data) and not PyArray_ISFORTRAN(data):
        raise ValueError("%s must be contiguous" % name)
    if writable and not PyArray_ISWRITEABLE(data):
        raise ValueError("%s must be writable" % name)
    if PyArray_NBYTES(data) < nrequired:
        raise ValueError(
            "%s buffer is too small: need at least %d bytes, has %d" %
            (name, nrequired, PyArray_NBYTES(data)))
    return 0


cdef inline int sfcheckbuffers(
    PetscSF sf,
    object unit,
    ndarray rootdata,
    bint rootwritable,
    ndarray leafdata,
    bint leafwritable,
    ndarray leafupdate,
    bint multi,
) except -1:
    cdef PetscSF multisf = NULL
    cdef PetscInt nroots = 0
    cdef PetscInt minleaf = 0
    cdef PetscInt maxleaf = -1
    cdef object rootname = "multirootdata" if multi else "rootdata"
    cdef object extent = unit.extent

    CHKERR(PetscSFGetLeafRange(sf, &minleaf, &maxleaf))
    if minleaf < 0:
        raise ValueError("SF leaf indices must be nonnegative")
    if multi:
        CHKERR(PetscSFGetMultiSF(sf, &multisf))
        CHKERR(PetscSFGetGraph(multisf, &nroots, NULL, NULL, NULL))
    else:
        CHKERR(PetscSFGetGraph(sf, &nroots, NULL, NULL, NULL))
    sfcheckbuffer(rootdata, rootname, nroots, extent, rootwritable)
    sfcheckbuffer(leafdata, "leafdata", maxleaf + 1, extent, leafwritable)
    if leafupdate is not None:
        sfcheckbuffer(leafupdate, "leafupdate", maxleaf + 1,
                      extent, True)
    return 0


# --------------------------------------------------------------------


cdef class SF(Object):
    """Star Forest object for communication.

    SF is used for setting up and managing the communication of certain
    entries of arrays and `Vec` between MPI processes.

    NumPy buffers passed to communication methods must be contiguous. Output
    buffers must be writable. Buffers passed to a ``*Begin`` method must remain
    alive and the same buffers must be passed to the matching ``*End`` method.

    """

    Type = SFType

    def __cinit__(self):
        self.obj = <PetscObject*> &self.sf
        self.sf  = NULL

    def view(self, Viewer viewer=None) -> None:
        """View a star forest.

        Collective.

        Parameters
        ----------
        viewer
            A `Viewer` to display the graph.

        See Also
        --------
        petsc.PetscSFView

        """
        cdef PetscViewer vwr = NULL
        if viewer is not None: vwr = viewer.vwr
        CHKERR(PetscSFView(self.sf, vwr))

    def destroy(self) -> Self:
        """Destroy the star forest.

        Collective.

        See Also
        --------
        petsc.PetscSFDestroy

        """
        CHKERR(PetscSFDestroy(&self.sf))
        return self

    def create(self, comm: Comm | None = None) -> Self:
        """Create a star forest communication context.

        Collective.

        Parameters
        ----------
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        petsc.PetscSFCreate

        """
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscSF newsf = NULL
        CHKERR(PetscSFCreate(ccomm, &newsf))
        CHKERR(PetscCLEAR(self.obj)); self.sf = newsf
        return self

    def setType(self, sf_type: Type | str) -> None:
        """Set the type of the star forest.

        Collective.

        Parameters
        ----------
        sf_type
            The star forest type.

        See Also
        --------
        petsc.PetscSFSetType

        """
        cdef PetscSFType cval = NULL
        sf_type = str2bytes(sf_type, &cval)
        CHKERR(PetscSFSetType(self.sf, cval))

    def getType(self) -> str:
        """Return the type name of the star forest.

        Collective.

        See Also
        --------
        petsc.PetscSFGetType

        """
        cdef PetscSFType cval = NULL
        CHKERR(PetscSFGetType(self.sf, &cval))
        return bytes2str(cval)

    def setFromOptions(self) -> None:
        """Set options using the options database.

        Logically collective.

        See Also
        --------
        petsc_options, petsc.PetscSFSetFromOptions

        """
        CHKERR(PetscSFSetFromOptions(self.sf))

    def setUp(self) -> None:
        """Set up communication structures.

        Collective.

        See Also
        --------
        petsc.PetscSFSetUp

        """
        CHKERR(PetscSFSetUp(self.sf))

    def reset(self) -> None:
        """Reset a star forest so that different sizes or neighbors can be used.

        Collective.

        See Also
        --------
        petsc.PetscSFReset

        """
        CHKERR(PetscSFReset(self.sf))

    #

    def getGraph(self) -> tuple[int, ArrayInt, ArrayInt]:
        """Return star forest graph.

        Not collective.

        The number of leaves can be determined from the size of ``ilocal``.

        Returns
        -------
        nroots : int
            Number of root vertices on the current process (these are possible
            targets for other process to attach leaves).
        ilocal : ArrayInt
            Locations of leaves in leafdata buffers.
        iremote : ArrayInt
            Remote locations of root vertices for each leaf on the current
            process.

        See Also
        --------
        petsc.PetscSFGetGraph

        """
        cdef PetscInt nroots = 0, nleaves = 0
        cdef const PetscInt *ilocal = NULL
        cdef const PetscSFNode *iremote = NULL
        CHKERR(PetscSFGetGraph(self.sf, &nroots, &nleaves, &ilocal, &iremote))
        if ilocal == NULL:
            local = arange(0, nleaves, 1)
        else:
            local = array_i(nleaves, ilocal)
        remote = array_i(nleaves*2, <const PetscInt*>iremote)
        remote = remote.reshape(nleaves, 2)
        return toInt(nroots), local, remote

    def setGraph(self, nroots: int, local: Sequence[int], remote: Sequence[int]) -> None:
        """Set star forest graph.

        Collective.

        The number of leaves argument can be determined from the size of
        ``local`` and/or ``remote``.

        Parameters
        ----------
        nroots
            Number of root vertices on the current process (these are possible
            targets for other process to attach leaves).
        local
            Locations of leaves in leafdata buffers, pass `None` for contiguous
            storage.
        remote
            Remote locations of root vertices for each leaf on the current
            process. Should be ``2*nleaves`` long as (rank, index) pairs.

        See Also
        --------
        petsc.PetscSFSetGraph

        """
        cdef PetscInt cnroots = asInt(nroots)
        cdef PetscInt nleaves = 0
        cdef PetscInt nremote = 0
        cdef PetscInt *ilocal = NULL
        cdef PetscSFNode* iremote = NULL
        remote = iarray_i(remote, &nremote, <PetscInt**>&iremote)
        if local is not None:
            local = iarray_i(local, &nleaves, &ilocal)
            if 2*nleaves != nremote:
                raise ValueError("remote array must contain one rank-index pair per local leaf")
        else:
            if nremote % 2 != 0:
                raise ValueError("remote array length must be even")
            nleaves = nremote // 2
        CHKERR(PetscSFSetGraph(self.sf, cnroots, nleaves, ilocal, PETSC_COPY_VALUES, iremote, PETSC_COPY_VALUES))

    def setRankOrder(self, flag: bool = True) -> None:
        """Sort multi-points for gathers and scatters by rank order.

        Logically collective.

        Parameters
        ----------
        flag
            `True` to sort, `False` to skip sorting.

        See Also
        --------
        petsc.PetscSFSetRankOrder

        """
        cdef PetscBool bval = asBool(flag)
        CHKERR(PetscSFSetRankOrder(self.sf, bval))

    def getMulti(self) -> SF:
        """Return the inner SF implementing gathers and scatters.

        Collective.

        See Also
        --------
        petsc.PetscSFGetMultiSF

        """
        cdef SF sf = SF()
        CHKERR(PetscSFGetMultiSF(self.sf, &sf.sf))
        CHKERR(PetscINCREF(sf.obj))
        return sf

    def createInverse(self) -> SF:
        """Create the inverse map.

        Collective.

        Create the inverse map given a PetscSF in which all vertices have
        degree 1.

        See Also
        --------
        petsc.PetscSFCreateInverseSF

        """
        cdef SF sf = SF()
        CHKERR(PetscSFCreateInverseSF(self.sf, &sf.sf))
        return sf

    def computeDegree(self) -> ArrayInt:
        """Compute and return the degree of each root vertex.

        Collective.

        See Also
        --------
        petsc.PetscSFComputeDegreeBegin, petsc.PetscSFComputeDegreeEnd

        """
        cdef const PetscInt *cdegree = NULL
        cdef PetscInt nroots = 0
        CHKERR(PetscSFComputeDegreeBegin(self.sf, &cdegree))
        CHKERR(PetscSFComputeDegreeEnd(self.sf, &cdegree))
        CHKERR(PetscSFGetGraph(self.sf, &nroots, NULL, NULL, NULL))
        degree = array_i(nroots, cdegree)
        return degree

    def createEmbeddedRootSF(self, selected: Sequence[int]) -> SF:
        """Remove edges from all but the selected roots.

        Collective.

        Does not remap indices.

        Parameters
        ----------
        selected
            Indices of the selected roots on this process.

        See Also
        --------
        petsc.PetscSFCreateEmbeddedRootSF

        """
        cdef PetscInt nroots = asInt(len(selected))
        cdef PetscInt *cselected = NULL
        selected = iarray_i(selected, &nroots, &cselected)
        cdef SF sf = SF()
        CHKERR(PetscSFCreateEmbeddedRootSF(self.sf, nroots, cselected, &sf.sf))
        return sf

    def createEmbeddedLeafSF(self, selected: Sequence[int]) -> SF:
        """Remove edges from all but the selected leaves.

        Collective.

        Does not remap indices.

        Parameters
        ----------
        selected
            Indices of the selected roots on this process.

        See Also
        --------
        petsc.PetscSFCreateEmbeddedLeafSF

        """
        cdef PetscInt nleaves = asInt(len(selected))
        cdef PetscInt *cselected = NULL
        selected = iarray_i(selected, &nleaves, &cselected)
        cdef SF sf = SF()
        CHKERR(PetscSFCreateEmbeddedLeafSF(self.sf, nleaves, cselected, &sf.sf))
        return sf

    def createSectionSF(self, Section rootSection, remoteOffsets: Sequence[int] | None, Section leafSection) -> SF:
        """Create an expanded `SF` of DOFs.

        Collective.

        Assumes the input `SF` relates points.

        Parameters
        ----------
        rootSection
            Data layout of remote points for outgoing data (this is usually
            the serial section).
        remoteOffsets
            Offsets for point data on remote processes (these are offsets from
            the root section), or `None`.
        leafSection
            Data layout of local points for incoming data (this is the
            distributed section).

        See Also
        --------
        petsc.PetscSFCreateSectionSF

        """
        cdef SF sectionSF = SF()
        cdef PetscInt noffsets = 0
        cdef PetscInt *cremoteOffsets = NULL
        if remoteOffsets is not None:
            remoteOffsets = iarray_i(remoteOffsets, &noffsets, &cremoteOffsets)
        CHKERR(PetscSFCreateSectionSF(self.sf, rootSection.sec, cremoteOffsets,
                                      leafSection.sec, &sectionSF.sf))
        return sectionSF

    def distributeSection(self, Section rootSection, Section leafSection=None) -> tuple[ArrayInt, Section]:
        """Create a new, reorganized `Section`.

        Collective.

        Moves from the root to the leaves of the `SF`.

        Parameters
        ----------
        rootSection
            Section defined on root space.
        leafSection
            Section defined on the leaf space.

        See Also
        --------
        petsc.PetscSFDistributeSection

        """
        cdef PetscInt lpStart = 0
        cdef PetscInt lpEnd = 0
        cdef PetscInt *cremoteOffsets = NULL
        cdef ndarray remoteOffsets
        cdef MPI_Comm ccomm = def_Comm(self.comm, PETSC_COMM_DEFAULT)
        if leafSection is None:
            leafSection = Section()
        if leafSection.sec == NULL:
            CHKERR(PetscSectionCreate(ccomm, &leafSection.sec))
        CHKERR(PetscSFDistributeSection(self.sf, rootSection.sec,
                                        &cremoteOffsets, leafSection.sec))
        CHKERR(PetscSectionGetChart(leafSection.sec, &lpStart, &lpEnd))
        remoteOffsets = array_i(lpEnd-lpStart, cremoteOffsets)
        CHKERR(PetscFree(cremoteOffsets))
        return (remoteOffsets, leafSection)

    def compose(self, SF sf) -> SF:
        """Compose a new `SF`.

        Collective.

        Puts the ``sf`` under this object in a top (roots) down (leaves) view.

        Parameters
        ----------
        sf
            `SF` to put under this object.

        See Also
        --------
        petsc.PetscSFCompose

        """
        cdef SF csf = SF()
        CHKERR(PetscSFCompose(self.sf, sf.sf, &csf.sf))
        return csf

    def bcastBegin(self, unit: Datatype, ndarray rootdata, ndarray leafdata, op: Op) -> None:
        """Begin pointwise broadcast.

        Collective.

        Root values are reduced to leaf values. This call has to be concluded
        with a call to `bcastEnd`.

        Parameters
        ----------
        unit
            MPI datatype.
        rootdata
            Buffer to broadcast.
        leafdata
            Buffer to be reduced with values from each leaf's respective root.
        op
            MPI reduction operation.

        See Also
        --------
        bcastEnd, petsc.PetscSFBcastBegin

        """
        cdef MPI_Datatype dtype = mpi4py_Datatype_Get(unit)
        cdef MPI_Op cop = mpi4py_Op_Get(op)
        sfcheckbuffers(self.sf, unit, rootdata, False, leafdata, True,
                       None, False)
        CHKERR(PetscSFBcastBegin(self.sf, dtype, <const void*>PyArray_DATA(rootdata),
                                 <void*>PyArray_DATA(leafdata), cop))

    def bcastEnd(self, unit: Datatype, ndarray rootdata, ndarray leafdata, op: Op) -> None:
        """End a broadcast & reduce operation started with `bcastBegin`.

        Collective.

        Parameters
        ----------
        unit
            MPI datatype.
        rootdata
            Buffer to broadcast.
        leafdata
            Buffer to be reduced with values from each leaf's respective root.
        op
            MPI reduction operation.

        See Also
        --------
        bcastBegin, petsc.PetscSFBcastEnd

        """
        cdef MPI_Datatype dtype = mpi4py_Datatype_Get(unit)
        cdef MPI_Op cop = mpi4py_Op_Get(op)
        sfcheckbuffers(self.sf, unit, rootdata, False, leafdata, True,
                       None, False)
        CHKERR(PetscSFBcastEnd(self.sf, dtype, <const void*>PyArray_DATA(rootdata),
                               <void*>PyArray_DATA(leafdata), cop))

    def reduceBegin(self, unit: Datatype, ndarray leafdata, ndarray rootdata, op: Op) -> None:
        """Begin reduction of leafdata into rootdata.

        Collective.

        This call has to be completed with call to `reduceEnd`.

        Parameters
        ----------
        unit
            MPI datatype.
        leafdata
            Values to reduce.
        rootdata
            Result of reduction of values from all leaves of each root.
        op
            MPI reduction operation.

        See Also
        --------
        reduceEnd, petsc.PetscSFReduceBegin

        """
        cdef MPI_Datatype dtype = mpi4py_Datatype_Get(unit)
        cdef MPI_Op cop = mpi4py_Op_Get(op)
        sfcheckbuffers(self.sf, unit, rootdata, True, leafdata, False,
                       None, False)
        CHKERR(PetscSFReduceBegin(self.sf, dtype, <const void*>PyArray_DATA(leafdata),
                                  <void*>PyArray_DATA(rootdata), cop))

    def reduceEnd(self, unit: Datatype, ndarray leafdata, ndarray rootdata, op: Op) -> None:
        """End a reduction operation started with `reduceBegin`.

        Collective.

        Parameters
        ----------
        unit
            MPI datatype.
        leafdata
            Values to reduce.
        rootdata
            Result of reduction of values from all leaves of each root.
        op
            MPI reduction operation.

        See Also
        --------
        reduceBegin, petsc.PetscSFReduceEnd

        """
        cdef MPI_Datatype dtype = mpi4py_Datatype_Get(unit)
        cdef MPI_Op cop = mpi4py_Op_Get(op)
        sfcheckbuffers(self.sf, unit, rootdata, True, leafdata, False,
                       None, False)
        CHKERR(PetscSFReduceEnd(self.sf, dtype, <const void*>PyArray_DATA(leafdata),
                                <void*>PyArray_DATA(rootdata), cop))

    def scatterBegin(self, unit: Datatype, ndarray multirootdata, ndarray leafdata) -> None:
        """Begin pointwise scatter operation.

        Collective.

        Operation is from multi-roots to leaves.
        This call has to be completed with `scatterEnd`.

        Parameters
        ----------
        unit
            MPI datatype.
        multirootdata
            Root buffer to send to each leaf, one unit of data per leaf.
        leafdata
            Leaf data to be updated with personal data from each respective root.

        See Also
        --------
        scatterEnd, petsc.PetscSFScatterBegin

        """
        cdef MPI_Datatype dtype = mpi4py_Datatype_Get(unit)
        sfcheckbuffers(self.sf, unit, multirootdata, False, leafdata, True,
                       None, True)
        CHKERR(PetscSFScatterBegin(self.sf, dtype, <const void*>PyArray_DATA(multirootdata),
                                   <void*>PyArray_DATA(leafdata)))

    def scatterEnd(self, unit: Datatype, ndarray multirootdata, ndarray leafdata) -> None:
        """End scatter operation that was started with `scatterBegin`.

        Collective.

        Parameters
        ----------
        unit
            MPI datatype.
        multirootdata
            Root buffer to send to each leaf, one unit of data per leaf.
        leafdata
            Leaf data to be updated with personal data from each respective root.

        See Also
        --------
        scatterBegin, petsc.PetscSFScatterEnd

        """
        cdef MPI_Datatype dtype = mpi4py_Datatype_Get(unit)
        sfcheckbuffers(self.sf, unit, multirootdata, False, leafdata, True,
                       None, True)
        CHKERR(PetscSFScatterEnd(self.sf, dtype, <const void*>PyArray_DATA(multirootdata),
                                 <void*>PyArray_DATA(leafdata)))

    def gatherBegin(self, unit: Datatype, ndarray leafdata, ndarray multirootdata) -> None:
        """Begin pointwise gather of all leaves into multi-roots.

        Collective.

        This call has to be completed with `gatherEnd`.

        Parameters
        ----------
        unit
            MPI datatype.
        leafdata
            Leaf data to gather to roots.
        multirootdata
            Root buffer to gather into, amount of space per root is
            equal to its degree.

        See Also
        --------
        gatherEnd, petsc.PetscSFGatherBegin

        """
        cdef MPI_Datatype dtype = mpi4py_Datatype_Get(unit)
        sfcheckbuffers(self.sf, unit, multirootdata, True, leafdata, False,
                       None, True)
        CHKERR(PetscSFGatherBegin(self.sf, dtype, <const void*>PyArray_DATA(leafdata),
                                  <void*>PyArray_DATA(multirootdata)))

    def gatherEnd(self, unit: Datatype, ndarray leafdata, ndarray multirootdata) -> None:
        """End gather operation that was started with `gatherBegin`.

        Collective.

        Parameters
        ----------
        unit
            MPI datatype.
        leafdata
            Leaf data to gather to roots.
        multirootdata
            Root buffer to gather into, amount of space per root is
            equal to its degree.

        See Also
        --------
        gatherBegin, petsc.PetscSFGatherEnd

        """
        cdef MPI_Datatype dtype = mpi4py_Datatype_Get(unit)
        sfcheckbuffers(self.sf, unit, multirootdata, True, leafdata, False,
                       None, True)
        CHKERR(PetscSFGatherEnd(self.sf, dtype, <const void*>PyArray_DATA(leafdata),
                                <void*>PyArray_DATA(multirootdata)))

    def fetchAndOpBegin(self, unit: Datatype, ndarray rootdata, ndarray leafdata, ndarray leafupdate, op: Op) -> None:
        """Begin fetch and update operation.

        Collective.

        This operation fetches values from root and updates atomically
        by applying an operation using the leaf value.

        This call has to be completed with `fetchAndOpEnd`.

        Parameters
        ----------
        unit
            MPI datatype.
        rootdata
            Root values to be updated, input state is seen by first process
            to perform an update.
        leafdata
            Leaf values to use in reduction.
        leafupdate
            State at each leaf's respective root immediately prior to my atomic
            update.
        op
            MPI reduction operation.

        See Also
        --------
        fetchAndOpEnd, petsc.PetscSFFetchAndOpBegin

        """
        cdef MPI_Datatype dtype = mpi4py_Datatype_Get(unit)
        cdef MPI_Op cop = mpi4py_Op_Get(op)
        sfcheckbuffers(self.sf, unit, rootdata, True, leafdata, False,
                       leafupdate, False)
        CHKERR(PetscSFFetchAndOpBegin(self.sf, dtype, <void*>PyArray_DATA(rootdata),
                                      <const void*>PyArray_DATA(leafdata),
                                      <void*>PyArray_DATA(leafupdate), cop))

    def fetchAndOpEnd(self, unit: Datatype, ndarray rootdata, ndarray leafdata, ndarray leafupdate, op: Op) -> None:
        """End operation started in a matching call to `fetchAndOpBegin`.

        Collective.

        Parameters
        ----------
        unit
            MPI datatype.
        rootdata
            Root values to be updated, input state is seen by first process
            to perform an update.
        leafdata
            Leaf values to use in reduction.
        leafupdate
            State at each leaf's respective root immediately prior to my atomic
            update.
        op
            MPI reduction operation.

        See Also
        --------
        fetchAndOpBegin, petsc.PetscSFFetchAndOpEnd

        """
        cdef MPI_Datatype dtype = mpi4py_Datatype_Get(unit)
        cdef MPI_Op cop = mpi4py_Op_Get(op)
        sfcheckbuffers(self.sf, unit, rootdata, True, leafdata, False,
                       leafupdate, False)
        CHKERR(PetscSFFetchAndOpEnd(self.sf, dtype, <void*>PyArray_DATA(rootdata),
                                    <const void*>PyArray_DATA(leafdata),
                                    <void*>PyArray_DATA(leafupdate), cop))

# --------------------------------------------------------------------

del SFType

# --------------------------------------------------------------------
