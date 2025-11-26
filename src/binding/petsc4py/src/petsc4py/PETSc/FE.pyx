# --------------------------------------------------------------------

class FEType(object):
    """The finite element types."""
    BASIC     = S_(PETSCFEBASIC)
    OPENCL    = S_(PETSCFEOPENCL)
    COMPOSITE = S_(PETSCFECOMPOSITE)

# --------------------------------------------------------------------


cdef class FE(Object):
    """A PETSc object that manages a finite element space."""

    Type = FEType

    def __cinit__(self):
        self.obj = <PetscObject*> &self.fe
        self.fe = NULL

    def view(self, Viewer viewer=None) -> None:
        """View a `FE` object.

        Collective.

        Parameters
        ----------
        viewer
            A `Viewer` to display the graph.

        See Also
        --------
        petsc.PetscFEView

        """
        cdef PetscViewer vwr = NULL
        if viewer is not None: vwr = viewer.vwr
        CHKERR(PetscFEView(self.fe, vwr))

    def destroy(self) -> Self:
        """Destroy the `FE` object.

        Collective.

        See Also
        --------
        petsc.PetscFEDestroy

        """
        CHKERR(PetscFEDestroy(&self.fe))
        return self

    def create(self, comm: Comm | None = None) -> Self:
        """Create an empty `FE` object.

        Collective.

        The type can then be set with `setType`.

        Parameters
        ----------
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        setType, petsc.PetscFECreate

        """
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscFE newfe = NULL
        CHKERR(PetscFECreate(ccomm, &newfe))
        CHKERR(PetscCLEAR(self.obj)); self.fe = newfe
        return self

    def createDefault(
        self,
        dim: int,
        nc: int,
        isSimplex: bool,
        qorder: int = DETERMINE,
        prefix: str | None = None,
        comm: Comm | None = None) -> Self:
        """Create a `FE` for basic FEM computation.

        Collective.

        Parameters
        ----------
        dim
            The spatial dimension.
        nc
            The number of components.
        isSimplex
            Flag for simplex reference cell, otherwise it's a tensor product.
        qorder
            The quadrature order or `DETERMINE` to use `Space` polynomial
            degree.
        prefix
            The options prefix, or `None`.
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        petsc.PetscFECreateDefault

        """
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscFE newfe = NULL
        cdef PetscInt cdim = asInt(dim)
        cdef PetscInt cnc = asInt(nc)
        cdef PetscInt cqorder = asInt(qorder)
        cdef PetscBool cisSimplex = asBool(isSimplex)
        cdef const char *cprefix = NULL
        if prefix:
            prefix = str2bytes(prefix, &cprefix)
        CHKERR(PetscFECreateDefault(ccomm, cdim, cnc, cisSimplex, cprefix, cqorder, &newfe))
        CHKERR(PetscCLEAR(self.obj)); self.fe = newfe
        return self

    def createByCell(
        self,
        dim: int,
        nc: int,
        ctype: DM.PolytopeType,
        qorder: int = DETERMINE,
        prefix: str | None = None,
        comm: Comm | None = None) -> Self:
        """Create a `FE` for basic FEM computation.

        Collective.

        Parameters
        ----------
        dim
            The spatial dimension.
        nc
            The number of components.
        ctype
            The cell type.
        qorder
            The quadrature order or `DETERMINE` to use `Space` polynomial
            degree.
        prefix
            The options prefix, or `None`.
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        petsc.PetscFECreateByCell

        """
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_SELF)
        cdef PetscFE newfe = NULL
        cdef PetscInt cdim = asInt(dim)
        cdef PetscInt cnc = asInt(nc)
        cdef PetscInt cqorder = asInt(qorder)
        cdef PetscDMPolytopeType cCellType = ctype
        cdef const char *cprefix = NULL
        if prefix:
            prefix = str2bytes(prefix, &cprefix)
        CHKERR(PetscFECreateDefault(ccomm, cdim, cnc, cCellType, cprefix, cqorder, &newfe))
        CHKERR(PetscCLEAR(self.obj)); self.fe = newfe
        return self

    def createLagrange(
        self,
        dim: int,
        nc: int,
        isSimplex: bool,
        k: int,
        qorder: int = DETERMINE,
        comm: Comm | None = None) -> Self:
        """Create a `FE` for the basic Lagrange space of degree k.

        Collective.

        Parameters
        ----------
        dim
            The spatial dimension.
        nc
            The number of components.
        isSimplex
            Flag for simplex reference cell, otherwise it's a tensor product.
        k
            The degree of the space.
        qorder
            The quadrature order or `DETERMINE` to use `Space` polynomial
            degree.
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        petsc.PetscFECreateLagrange

        """
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscFE newfe = NULL
        cdef PetscInt cdim = asInt(dim)
        cdef PetscInt cnc = asInt(nc)
        cdef PetscInt ck = asInt(k)
        cdef PetscInt cqorder = asInt(qorder)
        cdef PetscBool cisSimplex = asBool(isSimplex)
        CHKERR(PetscFECreateLagrange(ccomm, cdim, cnc, cisSimplex, ck, cqorder, &newfe))
        CHKERR(PetscCLEAR(self.obj)); self.fe = newfe
        return self

    def getQuadrature(self) -> Quad:
        """Return the `Quad` used to calculate inner products.

        Not collective.

        See Also
        --------
        setQuadrature, petsc.PetscFEGetQuadrature

        """
        cdef Quad quad = Quad()
        CHKERR(PetscFEGetQuadrature(self.fe, &quad.quad))
        CHKERR(PetscINCREF(quad.obj))
        return quad

    def getDimension(self) -> int:
        """Return the dimension of the finite element space on a cell.

        Not collective.

        See Also
        --------
        petsc.PetscFEGetDimension

        """
        cdef PetscInt cdim = 0
        CHKERR(PetscFEGetDimension(self.fe, &cdim))
        return toInt(cdim)

    def getSpatialDimension(self) -> int:
        """Return the spatial dimension of the element.

        Not collective.

        See Also
        --------
        petsc.PetscFEGetSpatialDimension

        """
        cdef PetscInt csdim = 0
        CHKERR(PetscFEGetSpatialDimension(self.fe, &csdim))
        return toInt(csdim)

    def getNumComponents(self) -> int:
        """Return the number of components in the element.

        Not collective.

        See Also
        --------
        setNumComponents, petsc.PetscFEGetNumComponents

        """
        cdef PetscInt comp = 0
        CHKERR(PetscFEGetNumComponents(self.fe, &comp))
        return toInt(comp)

    def setNumComponents(self, comp: int) -> None:
        """Set the number of field components in the element.

        Not collective.

        Parameters
        ----------
        comp
            The number of field components.

        See Also
        --------
        getNumComponents, petsc.PetscFESetNumComponents

        """
        cdef PetscInt ccomp = asInt(comp)
        CHKERR(PetscFESetNumComponents(self.fe, ccomp))

    def getNumDof(self) -> ndarray:
        """Return the number of DOFs.

        Not collective.

        Return the number of DOFs (dual basis vectors) associated with mesh
        points on the reference cell of a given dimension.

        See Also
        --------
        petsc.PetscFEGetNumDof

        """
        cdef const PetscInt *numDof = NULL
        cdef PetscInt cdim = 0
        CHKERR(PetscFEGetDimension(self.fe, &cdim))
        CHKERR(PetscFEGetNumDof(self.fe, &numDof))
        return array_i(cdim, numDof)

    def getTileSizes(self) -> tuple[int, int, int, int]:
        """Return the tile sizes for evaluation.

        Not collective.

        Returns
        -------
        blockSize : int
            The number of elements in a block.
        numBlocks : int
            The number of blocks in a batch.
        batchSize : int
            The number of elements in a batch.
        numBatches : int
            The number of batches in a chunk.

        See Also
        --------
        setTileSizes, petsc.PetscFEGetTileSizes

        """
        cdef PetscInt blockSize = 0, numBlocks = 0
        cdef PetscInt batchSize = 0, numBatches = 0
        CHKERR(PetscFEGetTileSizes(self.fe, &blockSize, &numBlocks, &batchSize, &numBatches))
        return toInt(blockSize), toInt(numBlocks), toInt(batchSize), toInt(numBatches)

    def setTileSizes(
        self,
        blockSize: int,
        numBlocks: int,
        batchSize: int,
        numBatches: int) -> None:
        """Set the tile sizes for evaluation.

        Not collective.

        Parameters
        ----------
        blockSize
            The number of elements in a block.
        numBlocks
            The number of blocks in a batch.
        batchSize
            The number of elements in a batch.
        numBatches
            The number of batches in a chunk.

        See Also
        --------
        getTileSizes, petsc.PetscFESetTileSizes

        """
        cdef PetscInt cblockSize = asInt(blockSize), cnumBlocks = asInt(numBlocks)
        cdef PetscInt cbatchSize = asInt(batchSize), cnumBatches = asInt(numBatches)
        CHKERR(PetscFESetTileSizes(self.fe, cblockSize, cnumBlocks, cbatchSize, cnumBatches))

    def getFaceQuadrature(self) -> Quad:
        """Return the `Quad` used to calculate inner products on faces.

        Not collective.

        See Also
        --------
        setFaceQuadrature, petsc.PetscFEGetFaceQuadrature

        """
        cdef Quad quad = Quad()
        CHKERR(PetscFEGetFaceQuadrature(self.fe, &quad.quad))
        CHKERR(PetscINCREF(quad.obj))
        return quad

    def setQuadrature(self, Quad quad) -> Self:
        """Set the `Quad` used to calculate inner products.

        Not collective.

        Parameters
        ----------
        quad
            The `Quad` object.

        See Also
        --------
        getQuadrature, petsc.PetscFESetQuadrature

        """
        CHKERR(PetscFESetQuadrature(self.fe, quad.quad))
        return self

    def setFaceQuadrature(self, Quad quad) -> Quad:
        """Set the `Quad` used to calculate inner products on faces.

        Not collective.

        Parameters
        ----------
        quad
            The `Quad` object.

        See Also
        --------
        getFaceQuadrature, petsc.PetscFESetFaceQuadrature

        """
        CHKERR(PetscFESetFaceQuadrature(self.fe, quad.quad))
        return self

    def setType(self, fe_type: Type | str) -> Self:
        """Build a particular `FE`.

        Collective.

        Parameters
        ----------
        fe_type
            The kind of FEM space.

        See Also
        --------
        petsc.PetscFESetType

        """
        cdef PetscFEType cval = NULL
        fe_type = str2bytes(fe_type, &cval)
        CHKERR(PetscFESetType(self.fe, cval))
        return self

    def getBasisSpace(self) -> Space:
        """Return the `Space` used for the approximation of the `FE` solution.

        Not collective.

        See Also
        --------
        setBasisSpace, petsc.PetscFEGetBasisSpace

        """
        cdef Space sp = Space()
        CHKERR(PetscFEGetBasisSpace(self.fe, &sp.space))
        CHKERR(PetscINCREF(sp.obj))
        return sp

    def setBasisSpace(self, Space sp) -> None:
        """Set the `Space` used for the approximation of the solution.

        Not collective.

        Parameters
        ----------
        sp
            The `Space` object.

        See Also
        --------
        getBasisSpace, petsc.PetscFESetBasisSpace

        """
        CHKERR(PetscFESetBasisSpace(self.fe, sp.space))

    def setFromOptions(self) -> None:
        """Set parameters in a `FE` from the options database.

        Collective.

        See Also
        --------
        petsc_options, petsc.PetscFESetFromOptions

        """
        CHKERR(PetscFESetFromOptions(self.fe))

    def setUp(self) -> None:
        """Construct data structures for the `FE` after the `Type` has been set.

        Collective.

        See Also
        --------
        petsc.PetscFESetUp

        """
        CHKERR(PetscFESetUp(self.fe))

    def getDualSpace(self) -> DualSpace:
        """Return the `DualSpace` used to define the inner product for the `FE`.

        Not collective.

        See Also
        --------
        setDualSpace, DualSpace, petsc.PetscFEGetDualSpace

        """
        cdef DualSpace dspace = DualSpace()
        CHKERR(PetscFEGetDualSpace(self.fe, &dspace.dualspace))
        CHKERR(PetscINCREF(dspace.obj))
        return dspace

    def setDualSpace(self, DualSpace dspace) -> None:
        """Set the `DualSpace` used to define the inner product.

        Not collective.

        Parameters
        ----------
        dspace
            The `DualSpace` object.

        See Also
        --------
        getDualSpace, DualSpace, petsc.PetscFESetDualSpace

        """
        CHKERR(PetscFESetDualSpace(self.fe, dspace.dualspace))

# --------------------------------------------------------------------

del FEType

# --------------------------------------------------------------------
