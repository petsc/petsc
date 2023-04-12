# --------------------------------------------------------------------

class RandomType(object):
    """The random number generator type."""
    RAND      = S_(PETSCRAND)
    RAND48    = S_(PETSCRAND48)
    SPRNG     = S_(PETSCSPRNG)
    RANDER48  = S_(PETSCRANDER48)
    RANDOM123 = S_(PETSCRANDOM123)

# --------------------------------------------------------------------

cdef class Random(Object):
    """The random number generator object.

    See Also
    --------
    petsc.PetscRandom

    """

    Type = RandomType

    def __cinit__(self) -> None:
        self.obj = <PetscObject*> &self.rnd
        self.rnd = NULL

    def __call__(self) -> Scalar:
        """Generate a scalar random number.

        Not collective.

        See Also
        --------
        petsc.PetscRandomGetValue

        """
        return self.getValue()

    def view(self, Viewer viewer=None) -> None:
        """View a random number generator object.

        Collective.

        Parameters
        ----------
        viewer
            A `Viewer` instance or `None` for the default viewer.

        See Also
        --------
        petsc.PetscRandomView

        """
        assert self.obj != NULL
        cdef PetscViewer vwr = NULL
        if viewer is not None: vwr = viewer.vwr
        CHKERR( PetscRandomView(self.rnd, vwr) )

    def destroy(self) -> Self:
        """Destroy the random number generator object.

        Collective.

        See Also
        --------
        petsc.PetscRandomDestroy

        """
        CHKERR( PetscRandomDestroy(&self.rnd) )
        return self

    def create(self, comm: Comm | None = None) -> Self:
        """Create a random number generator object.

        Collective.

        Parameters
        ----------
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        Sys.getDefaultComm, petsc.PetscRandomCreate

        """
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        CHKERR( PetscRandomCreate(ccomm, &self.rnd) )
        return self

    def setType(self, rnd_type: Random.Type | str) -> None:
        """Set the type of the random number generator object.

        Collective.

        Parameters
        ----------
        rnd_type
            The type of the generator.

        See Also
        --------
        getType, petsc.PetscRandomSetType

        """
        cdef PetscRandomType cval = NULL
        rnd_type = str2bytes(rnd_type, &cval)
        CHKERR( PetscRandomSetType(self.rnd, cval) )

    def getType(self) -> str:
        """Return the type of the random number generator object.

        Not collective.

        See Also
        --------
        setType, petsc.PetscRandomGetType

        """
        cdef PetscRandomType cval = NULL
        CHKERR( PetscRandomGetType(self.rnd, &cval) )
        return bytes2str(cval)

    def setFromOptions(self) -> None:
        """Configure the random number generator from the options database.

        Collective.

        See Also
        --------
        petsc_options, petsc.PetscRandomSetFromOptions

        """
        CHKERR( PetscRandomSetFromOptions(self.rnd) )

    def getValue(self) -> Scalar:
        """Generate a scalar random number.

        Not collective.

        See Also
        --------
        petsc.PetscRandomGetValue

        """
        cdef PetscScalar sval = 0
        CHKERR( PetscRandomGetValue(self.rnd, &sval) )
        return toScalar(sval)

    def getValueReal(self) -> float:
        """Generate a real random number.

        Not collective.

        See Also
        --------
        petsc.PetscRandomGetValueReal

        """
        cdef PetscReal rval = 0
        CHKERR( PetscRandomGetValueReal(self.rnd, &rval) )
        return toReal(rval)

    def getSeed(self) -> int:
        """Return the random number generator seed.

        Not collective.

        See Also
        --------
        setSeed, petsc.PetscRandomGetSeed

        """
        cdef unsigned long seed = 0
        CHKERR( PetscRandomGetSeed(self.rnd, &seed) )
        return seed

    def setSeed(self, seed: int | None = None) -> None:
        """Set the seed of random number generator.

        Not collective.

        Parameters
        ----------
        seed
            The value for the seed. If `None`, it only seeds the generator.

        See Also
        --------
        getSeed, petsc.PetscRandomSetSeed, petsc.PetscRandomSeed

        """
        if seed is not None:
            CHKERR( PetscRandomSetSeed(self.rnd, seed) )
        CHKERR( PetscRandomSeed(self.rnd) )

    def getInterval(self) -> tuple[Scalar, Scalar]:
        """Return the interval containing the random numbers generated.

        Not collective.

        See Also
        --------
        setInterval, petsc.PetscRandomGetInterval

        """
        cdef PetscScalar sval1 = 0
        cdef PetscScalar sval2 = 1
        CHKERR( PetscRandomGetInterval(self.rnd, &sval1, &sval2) )
        return (toScalar(sval1), toScalar(sval2))

    def setInterval(self, interval: tuple[Scalar, Scalar]) -> None:
        """Set the interval of the random number generator.

        Not collective.

        See Also
        --------
        getInterval, petsc.PetscRandomSetInterval

        """
        cdef PetscScalar sval1 = 0
        cdef PetscScalar sval2 = 1
        low, high = interval
        sval1 = asScalar(low)
        sval2 = asScalar(high)
        CHKERR( PetscRandomSetInterval(self.rnd, sval1, sval2) )

    #

    property seed:
        """The seed of the random number generator."""
        def __get__(self) -> int:
            return self.getSeed()
        def __set__(self, value: int | None) -> None:
            self.setSeed(value)

    property interval:
        """The interval of the generated random numbers."""
        def __get__(self) -> tuple[Scalar, Scalar]:
            return self.getInterval()
        def __set__(self, value: tuple[Scalar, Scalar]):
            self.setInterval(value)

# --------------------------------------------------------------------

del RandomType

# --------------------------------------------------------------------
