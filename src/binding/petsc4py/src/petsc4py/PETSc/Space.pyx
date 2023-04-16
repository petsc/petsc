# --------------------------------------------------------------------

class SpaceType(object):
    POLYNOMIAL = S_(PETSCSPACEPOLYNOMIAL)
    PTRIMMED   = S_(PETSCSPACEPTRIMMED)
    TENSOR     = S_(PETSCSPACETENSOR)
    SUM        = S_(PETSCSPACESUM)
    POINT      = S_(PETSCSPACEPOINT)
    SUBSPACE   = S_(PETSCSPACESUBSPACE)
    WXY        = S_(PETSCSPACEWXY)

# --------------------------------------------------------------------

cdef class Space(Object):
    """Linear space object."""
    Type = SpaceType

    def __cinit__(self):
        self.obj = <PetscObject*> &self.space
        self.space  = NULL

    def setUp(self) -> None:
        """Construct data structures for the `Space`.

        Collective on `Space`.

        See Also
        --------
        petsc.PetscSpaceSetUp

        """
        CHKERR( PetscSpaceSetUp(self.space) )

    def create(self, comm: Comm | None = None) -> Self:
        """Create an empty `Space` object.

        Collective.

        The type can then be set with `setType`.

        Parameters
        ----------
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        petsc.PetscSpaceCreate

        """
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscSpace newsp = NULL
        CHKERR( PetscSpaceCreate(ccomm, &newsp) )
        PetscCLEAR(self.obj); self.space = newsp
        return self

    def destroy(self) -> Self:
        """Destroy the `Space` object.

        Collective.

        See Also
        --------
        petsc.PetscSpaceDestroy

        """
        CHKERR( PetscSpaceDestroy(&self.space) )
        return self

    def view(self, Viewer viewer=None) -> None:
        """View a `Space`.

        Collective.

        Parameters
        ----------
        viewer
            A `Viewer` to display the `Space`.

        See Also
        --------
        petsc.PetscSpaceView

        """
        cdef PetscViewer vwr = NULL
        if viewer is not None: vwr = viewer.vwr
        CHKERR( PetscSpaceView(self.space, vwr) )

    def setFromOptions(self) -> None:
        """Set parameters in `Space` from the options database.

        Collective.

        See Also
        --------
        petsc_options, petsc.PetscSpaceSetFromOptions

        """
        CHKERR( PetscSpaceSetFromOptions(self.space) )

    def getDimension(self) -> int:
        """Return the number of basis vectors.

        See Also
        --------
        petsc.PetscSpaceGetDimension

        """
        cdef PetscInt cdim
        CHKERR( PetscSpaceGetDimension(self.space, &cdim))
        return toInt(cdim)

    def getDegree(self) -> tuple[int, int]:
        """Return the polynomial degrees that characterize this space.

        Returns
        -------
        minDegree : int
            The degree of the largest polynomial space contained in the space.
        maxDegree : int
            The degree of the smallest polynomial space containing the space.

        See Also
        --------
        setDegree, petsc.PetscSpaceGetDegree

        """
        cdef PetscInt cdegmax, cdegmin
        CHKERR( PetscSpaceGetDegree(self.space, &cdegmin, &cdegmax))
        return toInt(cdegmin), toInt(cdegmax)

    def setDegree(self, degree: int | None, maxDegree: int | None) -> None:
        """Set the degree of approximation for this space.

        One of ``degree`` and ``maxDegree`` can be `None`.

        Parameters
        ----------
        degree
            The degree of the largest polynomial space contained in the space.
        maxDegree
            The degree of the largest polynomial space containing the space.

        See Also
        --------
        getDegree, petsc.PetscSpaceSetDegree

        """
        assert( (degree != None) & (maxDegree != None))
        cdef PetscInt cdegree = PETSC_DETERMINE
        if degree is not None: cdegree = asInt(degree)
        cdef PetscInt cmaxdegree = PETSC_DETERMINE
        if maxDegree is not None: cmaxdegree = asInt(maxDegree)
        CHKERR( PetscSpaceSetDegree(self.space, cdegree, cmaxdegree) )

    def getNumVariables(self) -> int:
        """Return the number of variables for this space.

        See Also
        --------
        setNumVariables, petsc.PetscSpaceGetNumVariables

        """
        cdef PetscInt cnvars
        CHKERR( PetscSpaceGetNumVariables(self.space, &cnvars))
        return toInt(cnvars)

    def setNumVariables(self, n: int) -> None:
        """Set the number of variables for this space.

        Parameters
        ----------
        n
            The number of variables (``x``, ``y``, ``z`` etc.).

        See Also
        --------
        getNumVariables, petsc.PetscSpaceSetNumVariables

        """
        cdef PetscInt cn = asInt(n)
        CHKERR( PetscSpaceSetNumVariables(self.space, cn) )

    def getNumComponents(self) -> int:
        """Return the number of components for this space.

        See Also
        --------
        setNumComponents, petsc.PetscSpaceGetNumComponents

        """
        cdef PetscInt cncomps
        CHKERR( PetscSpaceGetNumComponents(self.space, &cncomps))
        return toInt(cncomps)

    def setNumComponents(self, nc: int) -> None:
        """Set the number of components for this space.

        Parameters
        ----------
        nc
            The number of components.

        See Also
        --------
        getNumComponents, petsc.PetscSpaceSetNumComponents

        """
        cdef PetscInt cnc = asInt(nc)
        CHKERR( PetscSpaceSetNumComponents(self.space, cnc) )

    #def evaluate(self, points):
    #    cdef PetscInt  cnpoints = 0, cdim=0, cnfuncs=0
    #    cdef PetscReal *cpoints = NULL
    #    cdef PetscReal *B = NULL, *D = NULL, *H = NULL
    #    points = iarray_r(points, &cnpoints,  &cpoints)
    #    # Get the dimension of the space
    #    CHKERR( PetscSpaceGetDimension( self.space, &cnfuncs) )
    #    CHKERR( PetscSpace)
    #    CHKERR( PetscSpaceEvaluate(self.space, cnpoints, &cpoints, &B, &D, &H) )
    #    return array_r(cnpoints*cdim, B), array_r(cnpoints*cnc, D), array_r(, H)

    def getType(self) -> str:
        """Return the type of the space object.

        Not collective.

        See Also
        --------
        setType, petsc.PetscSpaceGetType

        """
        cdef PetscSpaceType cval = NULL
        CHKERR( PetscSpaceGetType(self.space, &cval) )
        return bytes2str(cval)

    def setType(self, space_type: Type | str) -> Self:
        """Build a particular type of space.

        Collective.

        Parameters
        ----------
        space_type
            The kind of space.

        See Also
        --------
        getType, petsc.PetscSpaceSetType

        """
        cdef PetscSpaceType cval = NULL
        space_type = str2bytes(space_type, &cval)
        CHKERR( PetscSpaceSetType(self.space, cval) )
        return self

    def getSumConcatenate(self) -> bool:
        """Return the concatenate flag for this space.

        A concatenated sum space will have the number of components equal to
        the sum of the number of components of all subspaces.
        A non-concatenated, or direct sum space will have the same number of
        components as its subspaces.

        See Also
        --------
        setSumConcatenate, petsc.PetscSpaceSumGetConcatenate

        """
        cdef PetscBool concatenate
        CHKERR( PetscSpaceSumGetConcatenate(self.space, &concatenate))
        return toBool(concatenate)

    def setSumConcatenate(self, concatenate: bool) -> None:
        """Set the concatenate flag for this space.

        A concatenated sum space will have the number of components equal to
        the sum of the number of components of all subspaces.
        A non-concatenated, or direct sum space will have the same number of
        components as its subspaces.

        Parameters
        ----------
        concatenate
            `True` if subspaces are concatenated components,
            `False` if direct summands.

        See Also
        --------
        getSumConcatenate, petsc.PetscSpaceSumSetConcatenate

        """
        cdef PetscBool cconcatenate = asBool(concatenate)
        CHKERR( PetscSpaceSumSetConcatenate(self.space, concatenate))

    def getSumNumSubspaces(self) -> int:
        """Return the number of spaces in the sum.

        See Also
        --------
        setSumNumSubspaces, petsc.PetscSpaceSumGetNumSubspaces

        """
        cdef PetscInt numSumSpaces
        CHKERR( PetscSpaceSumGetNumSubspaces(self.space, &numSumSpaces))
        return toInt(numSumSpaces)

    def getSumSubspace(self, s: int) -> Space:
        """Return a space in the sum.

        Parameters
        ----------
        s
            The space number.

        See Also
        --------
        setSumSubspace, petsc.PetscSpaceSumGetSubspace

        """
        cdef Space subsp = Space()
        cdef PetscInt cs = asInt(s)
        CHKERR( PetscSpaceSumGetSubspace(self.space, s, &subsp.space) )
        return subsp

    def setSumSubspace(self, s: int, Space subsp) -> None:
        """Set a space in the sum.

        Parameters
        ----------
        s
            The space number.
        subsp
            The number of spaces.

        See Also
        --------
        getSumSubspace, petsc.PetscSpaceSumSetSubspace

        """
        cdef PetscInt cs = asInt(s)
        CHKERR( PetscSpaceSumSetSubspace(self.space, cs, subsp.space) )

    def setSumNumSubspaces(self, numSumSpaces: int) -> None:
        """Set the number of spaces in the sum.

        Parameters
        ----------
        numSumSpaces
            The number of spaces.

        See Also
        --------
        getSumNumSubspaces, petsc.PetscSpaceSumSetNumSubspaces

        """
        cdef PetscInt cnumSumSpaces = asInt(numSumSpaces)
        CHKERR( PetscSpaceSumSetNumSubspaces(self.space, cnumSumSpaces) )

    def getTensorNumSubspaces(self) -> int:
        """Return the number of spaces in the tensor product.

        See Also
        --------
        setTensorNumSubspaces, petsc.PetscSpaceTensorGetNumSubspaces

        """
        cdef PetscInt cnumTensSpaces = 0
        CHKERR( PetscSpaceTensorGetNumSubspaces(self.space, &cnumTensSpaces) )
        return toInt(cnumTensSpaces)

    def setTensorSubspace(self, s: int, Space subsp) -> None:
        """Set a space in the tensor product.

        Parameters
        ----------
        s
            The space number.
        subsp
            The number of spaces.

        See Also
        --------
        getTensorSubspace, petsc.PetscSpaceTensorSetSubspace

        """
        cdef PetscInt cs = asInt(s)
        CHKERR( PetscSpaceTensorSetSubspace(self.space, cs, subsp.space) )

    def getTensorSubspace(self, s: int) -> Space:
        """Return a space in the tensor product.

        Parameters
        ----------
        s
            The space number.

        See Also
        --------
        setTensorSubspace, petsc.PetscSpaceTensorGetSubspace

        """
        cdef PetscInt cs = asInt(s)
        cdef Space subsp = Space()
        CHKERR( PetscSpaceTensorGetSubspace(self.space, cs, &subsp.space) )
        return subsp

    def setTensorNumSubspaces(self, numTensSpaces: int) -> None:
        """Set the number of spaces in the tensor product.

        Parameters
        ----------
        numTensSpaces
            The number of spaces.

        See Also
        --------
        getTensorNumSubspaces, petsc.PetscSpaceTensorSetNumSubspaces

        """
        cdef PetscInt cnumTensSpaces = asInt(numTensSpaces)
        CHKERR( PetscSpaceTensorSetNumSubspaces(self.space, cnumTensSpaces) )

    def getPolynomialTensor(self) -> bool:
        """Return whether a function space is a space of tensor polynomials.

        Return `True` if a function space is a space of tensor polynomials
        (the space is spanned by polynomials whose degree in each variable is
        bounded by the given order), as opposed to polynomials (the space is
        spanned by polynomials whose total degree—summing over all variables
        is bounded by the given order).

        See Also
        --------
        setPolynomialTensor, petsc.PetscSpacePolynomialGetTensor

        """
        cdef PetscBool ctensor
        CHKERR( PetscSpacePolynomialGetTensor(self.space, &ctensor) )
        return toBool(ctensor)

    def setPolynomialTensor(self, tensor: bool) -> None:
        """Set whether a function space is a space of tensor polynomials.

        Set to `True` for a function space which is a space of tensor
        polynomials (the space is spanned by polynomials whose degree in each
        variable is bounded by the given order), as opposed to polynomials
        (the space is spanned by polynomials whose total degree—summing over
        all variables is bounded by the given order).

        Parameters
        ----------
        tensor
            `True` for a tensor polynomial space, `False` for a polynomial
            space.

        See Also
        --------
        getPolynomialTensor, petsc.PetscSpacePolynomialSetTensor

        """
        cdef PetscBool ctensor = asBool(tensor)
        CHKERR( PetscSpacePolynomialSetTensor(self.space, ctensor) )

    def setPointPoints(self, Quad quad) -> None:
        """Set the evaluation points for the space to be based on a quad.

        Logically collective.

        Sets the evaluation points for the space to coincide with the points
        of a quadrature rule.

        Parameters
        ----------
        quad
            The `Quad` defining the points.

        See Also
        --------
        getPointPoints, petsc.PetscSpacePointSetPoints

        """
        CHKERR( PetscSpacePointSetPoints(self.space, quad.quad))

    def getPointPoints(self) -> Quad:
        """Return the evaluation points for the space as the points of a quad.

        Logically collective.

        See Also
        --------
        setPointPoints, petsc.PetscSpacePointGetPoints

        """
        cdef Quad quad = Quad()
        CHKERR( PetscSpacePointGetPoints(self.space, &quad.quad))
        return quad

    def setPTrimmedFormDegree(self, formDegree: int) -> None:
        """Set the form degree of the trimmed polynomials.

        Parameters
        ----------
        formDegree
            The form degree.

        See Also
        --------
        getPTrimmedFormDegree, petsc.PetscSpacePTrimmedSetFormDegree

        """
        cdef PetscInt cformDegree = asInt(formDegree)
        CHKERR( PetscSpacePTrimmedSetFormDegree(self.space, cformDegree) )

    def getPTrimmedFormDegree(self) -> int:
        """Return the form degree of the trimmed polynomials.

        See Also
        --------
        setPTrimmedFormDegree, petsc.PetscSpacePTrimmedGetFormDegree

        """
        cdef PetscInt cformDegree = 0
        CHKERR( PetscSpacePTrimmedGetFormDegree(self.space, &cformDegree) )
        return toInt(cformDegree)

    def viewFromOptions(self, name: str, Object obj=None) -> None:
        """View a `Space` based on values in the options database.

        Collective.

        Parameters
        ----------
        name
            Command line option name.
        obj
            Optional object that provides the options prefix.

        See Also
        --------
        petsc_options, petsc.PetscSpaceViewFromOptions

        """
        cdef const char *cname = NULL
        _ = str2bytes(name, &cname)
        cdef PetscObject  cobj = NULL
        if obj is not None: cobj = obj.obj[0]
        CHKERR( PetscSpaceViewFromOptions(self.space, cobj, cname) )

# --------------------------------------------------------------------

class DualSpaceType(object):
    LAGRANGE = S_(PETSCDUALSPACELAGRANGE)
    SIMPLE   = S_(PETSCDUALSPACESIMPLE)
    REFINED  = S_(PETSCDUALSPACEREFINED)
    BDM      = S_(PETSCDUALSPACEBDM)

# --------------------------------------------------------------------

cdef class DualSpace(Object):
    """Dual space to a linear space."""

    Type = DualSpaceType

    def __cinit__(self):
        self.obj = <PetscObject*> &self.dualspace
        self.dualspace  = NULL

    def setUp(self) -> None:
        """Construct a basis for a `DualSpace`.

        Collective.

        See Also
        --------
        petsc.PetscDualSpaceSetUp

        """
        CHKERR( PetscDualSpaceSetUp(self.dualspace) )

    def create(self, comm: Comm | None = None) -> Self:
        """Create an empty `DualSpace` object.

        Collective.

        The type can then be set with `setType`.

        Parameters
        ----------
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        petsc.PetscDualSpaceCreate

        """
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscDualSpace newdsp = NULL
        CHKERR( PetscDualSpaceCreate(ccomm, &newdsp) )
        PetscCLEAR(self.obj); self.dualspace = newdsp
        return self

    def view(self, Viewer viewer=None) -> None:
        """View a `DualSpace`.

        Collective.

        Parameters
        ----------
        viewer
            A `Viewer` to display the `DualSpace`.

        See Also
        --------
        petsc.PetscDualSpaceView

        """
        cdef PetscViewer vwr = NULL
        if viewer is not None: vwr = viewer.vwr
        CHKERR( PetscDualSpaceView(self.dualspace, vwr) )

    def destroy(self) -> Self:
        """Destroy the `DualSpace` object.

        Collective.

        See Also
        --------
        petsc.PetscDualSpaceDestroy

        """
        CHKERR( PetscDualSpaceDestroy(&self.dualspace) )
        return self

    def duplicate(self) -> DualSpace:
        """Create a duplicate `DualSpace` object that is not set up.

        Collective.

        See Also
        --------
        petsc.PetscDualSpaceDuplicate

        """
        cdef DualSpace spNew = DualSpace()
        CHKERR( PetscDualSpaceDuplicate(self.dualspace, &spNew.dualspace) )

    def getDM(self) -> DM:
        """Return the `DM` representing the reference cell of a `DualSpace`.

        Not collective.

        See Also
        --------
        setDM, petsc.PetscDualSpaceGetDM

        """
        cdef DM dm = DM()
        CHKERR( PetscDualSpaceGetDM(self.dualspace, &dm.dm) )
        return dm

    def setDM(self, DM dm) -> None:
        """Set the `DM` representing the reference cell.

        Not collective.

        Parameters
        ----------
        dm
            The reference cell.

        See Also
        --------
        getDM, petsc.PetscDualSpaceSetDM

        """
        CHKERR( PetscDualSpaceSetDM(self.dualspace, dm.dm) )

    def getDimension(self) -> int:
        """Return the dimension of the dual space.

        Not collective.

        The dimension of the dual space, i.e. the number of basis functionals.

        See Also
        --------
        petsc.PetscDualSpaceGetDimension

        """
        cdef PetscInt cdim
        CHKERR( PetscDualSpaceGetDimension(self.dualspace, &cdim))
        return toInt(cdim)

    def getNumComponents(self) -> int:
        """Return the number of components for this space.

        See Also
        --------
        setNumComponents, petsc.PetscDualSpaceGetNumComponents

        """
        cdef PetscInt cncomps
        CHKERR( PetscDualSpaceGetNumComponents(self.dualspace, &cncomps))
        return toInt(cncomps)

    def setNumComponents(self, nc: int) -> None:
        """Set the number of components for this space.

        Parameters
        ----------
        nc
            The number of components

        See Also
        --------
        getNumComponents, petsc.PetscDualSpaceSetNumComponents

        """
        cdef PetscInt cnc = asInt(nc)
        CHKERR( PetscDualSpaceSetNumComponents(self.dualspace, cnc) )

    def getType(self) -> str:
        """Return the type of the dual space object.

        Not collective.

        See Also
        --------
        setType, petsc.PetscDualSpaceGetType

        """
        cdef PetscDualSpaceType cval = NULL
        CHKERR( PetscDualSpaceGetType(self.dualspace, &cval) )
        return bytes2str(cval)

    def setType(self, dualspace_type: Type | str) -> Self:
        """Build a particular type of dual space.

        Collective.

        Parameters
        ----------
        dualspace_type
            The kind of space.

        See Also
        --------
        getType, petsc.PetscDualSpaceSetType

        """
        cdef PetscDualSpaceType cval = NULL
        space_type = str2bytes(dualspace_type, &cval)
        CHKERR( PetscDualSpaceSetType(self.dualspace, cval) )
        return self

    def getOrder(self) -> int:
        """Return the order of the dual space.

        Not collective.

        See Also
        --------
        setOrder, petsc.PetscDualSpaceGetOrder

        """
        cdef PetscInt corder
        CHKERR( PetscDualSpaceGetOrder(self.dualspace, &corder))
        return toInt(corder)

    def setOrder(self, order: int) -> None:
        """Set the order of the dual space.

        Not collective.

        Parameters
        ----------
        order
            The order.

        See Also
        --------
        getOrder, petsc.PetscDualSpaceSetOrder

        """
        cdef PetscInt corder = asInt(order)
        CHKERR( PetscDualSpaceSetOrder(self.dualspace, corder) )

    def getNumDof(self) -> ArrayInt:
        """Return the number of degrees of freedom for each spatial dimension.

        Not collective.

        See Also
        --------
        petsc.PetscDualSpaceGetNumDof

        """
        cdef const PetscInt *cndof = NULL
        cdef PetscInt cdim = 0
        CHKERR( PetscDualSpaceGetDimension(self.dualspace, &cdim) )
        CHKERR( PetscDualSpaceGetNumDof(self.dualspace, &cndof) )
        return array_i(cdim + 1, cndof)

    def getFunctional(self, i: int) -> Quad:
        """Return the i-th basis functional in the dual space.

        Not collective.

        Parameters
        ----------
        i
            The basis number.

        See Also
        --------
        petsc.PetscDualSpaceGetFunctional

        """
        cdef PetscInt ci = asInt(i)
        cdef Quad functional = Quad()
        CHKERR( PetscDualSpaceGetFunctional( self.dualspace, ci, &functional.quad) )
        return functional

    def getInteriorDimension(self) -> int:
        """Return the interior dimension of the dual space.

        Not collective.

        The interior dimension of the dual space, i.e. the number of basis
        functionals assigned to the interior of the reference domain.

        See Also
        --------
        petsc.PetscDualSpaceGetInteriorDimension

        """
        cdef PetscInt cintdim = 0
        CHKERR( PetscDualSpaceGetInteriorDimension(self.dualspace, &cintdim) )
        return toInt(cintdim)

    def getLagrangeContinuity(self) -> bool:
        """Return whether the element is continuous.

        Not collective.

        See Also
        --------
        setLagrangeContinuity, petsc.PetscDualSpaceLagrangeGetContinuity

        """
        cdef PetscBool ccontinuous = PETSC_FALSE
        CHKERR( PetscDualSpaceLagrangeGetContinuity(self.dualspace, &ccontinuous))
        return toBool(ccontinuous)

    def setLagrangeContinuity(self, continuous: bool) -> None:
        """Indicate whether the element is continuous.

        Not collective.

        Parameters
        ----------
        continuous
            The flag for element continuity.

        See Also
        --------
        getLagrangeContinuity, petsc.PetscDualSpaceLagrangeSetContinuity

        """
        cdef PetscBool ccontinuous = asBool(continuous)
        CHKERR( PetscDualSpaceLagrangeSetContinuity(self.dualspace, ccontinuous))

    def getLagrangeTensor(self) -> bool:
        """Return the tensor nature of the dual space.

        Not collective.

        See Also
        --------
        setLagrangeTensor, petsc.PetscDualSpaceLagrangeGetTensor

        """
        cdef PetscBool ctensor = PETSC_FALSE
        CHKERR( PetscDualSpaceLagrangeGetTensor(self.dualspace, &ctensor))
        return toBool(ctensor)

    def setLagrangeTensor(self, tensor: bool) -> None:
        """Set the tensor nature of the dual space.

        Not collective.

        Parameters
        ----------
        tensor
            Whether the dual space has tensor layout (vs. simplicial).

        See Also
        --------
        getLagrangeTensor, petsc.PetscDualSpaceLagrangeSetTensor

        """
        cdef PetscBool ctensor = asBool(tensor)
        CHKERR( PetscDualSpaceLagrangeSetTensor(self.dualspace, ctensor))

    def getLagrangeTrimmed(self) -> bool:
        """Return the trimmed nature of the dual space.

        Not collective.

        See Also
        --------
        setLagrangeTrimmed, petsc.PetscDualSpaceLagrangeGetTrimmed

        """
        cdef PetscBool ctrimmed = PETSC_FALSE
        CHKERR( PetscDualSpaceLagrangeGetTrimmed(self.dualspace, &ctrimmed))
        return toBool(ctrimmed)

    def setLagrangeTrimmed(self, trimmed: bool) -> None:
        """Set the trimmed nature of the dual space.

        Not collective.

        Parameters
        ----------
        trimmed
            Whether the dual space represents to dual basis of a trimmed
            polynomial space (e.g. Raviart-Thomas and higher order /
            other form degree variants).

        See Also
        --------
        getLagrangeTrimmed, petsc.PetscDualSpaceLagrangeSetTrimmed

        """
        cdef PetscBool ctrimmed = asBool(trimmed)
        CHKERR( PetscDualSpaceLagrangeSetTrimmed(self.dualspace, ctrimmed))

    def viewFromOptions(self, name: str, Object obj=None) -> None:
        """View a `DualSpace` based on values in the options database.

        Collective.

        Parameters
        ----------
        name
            Command line option name.
        obj
            Optional object that provides the options prefix.

        See Also
        --------
        petsc_options, petsc.PetscSpaceViewFromOptions

        """
        cdef const char *cname = NULL
        _ = str2bytes(name, &cname)
        cdef PetscObject  cobj = NULL
        if obj is not None: cobj = obj.obj[0]
        CHKERR( PetscDualSpaceViewFromOptions(self.dualspace, cobj, cname) )

    def setSimpleDimension(self, dim: int) -> None:
        """Set the number of functionals in the dual space basis.

        Logically collective.

        Parameters
        ----------
        dim
            The basis dimension.

        See Also
        --------
        petsc.PetscDualSpaceSimpleSetDimension

        """
        cdef PetscInt cdim = asInt(dim)
        CHKERR( PetscDualSpaceSimpleSetDimension(self.dualspace, cdim) )

    def setSimpleFunctional(self, func: int, Quad functional) -> None:
        """Set the given basis element for this dual space.

        Not collective.

        Parameters
        ----------
        func
            The basis index.
        functional
            The basis functional.

        See Also
        --------
        petsc.PetscDualSpaceSimpleSetFunctional

        """
        cdef PetscInt cfunc = asInt(func)
        CHKERR( PetscDualSpaceSimpleSetFunctional(self.dualspace, cfunc, functional.quad) )

del SpaceType
del DualSpaceType
