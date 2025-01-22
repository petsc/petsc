# --------------------------------------------------------------------

cdef class Options:
    """The options database object.

    A dictionary-like object to store and operate with
    command line options.

    Parameters
    ----------
    prefix : str, optional
        Optional string to prepend to all the options.

    Examples
    --------
    Create an option database and operate with it.

    >>> from petsc4py import PETSc
    >>> opts = PETSc.Options()
    >>> opts['a'] = 1 # insert the command-line option '-a 1'
    >>> if 'a' in opts: # if the option is present
    >>>     val = opts['a'] # return the option value as 'str'
    >>> a_int = opts.getInt('a') # return the option value as 'int'
    >>> a_bool = opts.getBool('a') # return the option value as 'bool'

    Read command line and use default values.

    >>> from petsc4py import PETSc
    >>> opts = PETSc.Options()
    >>> b_float = opts.getReal('b', 1) # return the value or 1.0 if not present

    Read command line options prepended with a prefix.

    >>> from petsc4py import PETSc
    >>> opts = PETSc.Options('prefix_')
    >>> opts.getString('b', 'some_default_string') # read -prefix_b xxx

    See Also
    --------
    petsc_options

    """

    cdef PetscOptions opt
    cdef object       _prefix

    def __init__(self, prefix = None):
        self.opt = NULL
        self.prefix = prefix

    def __dealloc__(self):
        if self.opt == NULL: return
        CHKERR(PetscOptionsDestroy(&self.opt))

    def __contains__(self, item):
        return self.hasName(item)

    def __getitem__(self, item):
        return self.getString(item)

    def __setitem__(self, item, value):
        self.setValue(item, value)

    def __delitem__(self, item):
        self.delValue(item)

    property prefix:
        """Prefix for options."""
        def __get__(self) -> str:
            return self._prefix

        def __set__(self, prefix):
            self._prefix = getprefix(prefix)

        def __del__(self):
            self._prefix = None
    #

    def create(self) -> Self:
        """Create an options database."""
        if self.opt != NULL: return
        CHKERR(PetscOptionsCreate(&self.opt))
        return self

    def destroy(self) -> Self:
        """Destroy an options database."""
        if self.opt == NULL: return
        CHKERR(PetscOptionsDestroy(&self.opt))
        return self

    def clear(self) -> Self:
        """Clear an options database."""
        if self.opt == NULL: return
        CHKERR(PetscOptionsClear(self.opt))
        return self

    def view(self, Viewer viewer=None) -> None:
        """View the options database.

        Collective.

        Parameters
        ----------
        viewer
            A `Viewer` instance or `None` for the default viewer.

        See Also
        --------
        Viewer, petsc.PetscOptionsView

        """
        cdef PetscViewer vwr = NULL
        if viewer is not None: vwr = viewer.vwr
        CHKERR(PetscOptionsView(self.opt, vwr))

    def prefixPush(self, prefix: str | Options | Object | None) -> None:
        """Push a prefix for the options database.

        Logically collective.

        See Also
        --------
        prefixPop, petsc.PetscOptionsPrefixPush

        """
        prefix = getprefix(prefix)
        cdef const char *cprefix = NULL
        prefix = str2bytes(prefix, &cprefix)
        CHKERR(PetscOptionsPrefixPush(self.opt, cprefix))

    def prefixPop(self) -> None:
        """Pop a prefix for the options database.

        Logically collective.

        See Also
        --------
        prefixPush, petsc.PetscOptionsPrefixPop

        """
        CHKERR(PetscOptionsPrefixPop(self.opt))
    #

    def hasName(self, name: str) -> bool:
        """Return the boolean indicating if the option is in the database."""
        cdef const char *pr = NULL
        cdef const char *nm = NULL
        cdef object unused = getpair(self.prefix, name, &pr, &nm)
        cdef PetscBool flag = PETSC_FALSE
        CHKERR(PetscOptionsHasName(self.opt, pr, nm, &flag))
        return toBool(flag)

    def used(self, name: str) -> bool:
        """Return the boolean indicating if the option was queried from the database."""
        cdef const char *key = NULL
        cdef PetscBool flag = PETSC_FALSE
        name = str2bytes(name, &key)
        CHKERR(PetscOptionsUsed(self.opt, key, &flag))
        return toBool(flag)

    def setValue(self, name: str,
                 value: bool | int | float | Scalar | Sequence[bool] | Sequence[int] | Sequence[float] | Sequence[Scalar] | str) -> None:
        """Set a value for an option.

        Logically collective.

        Parameters
        ----------
        name
            The string identifying the option.
        value
            The option value.

        See Also
        --------
        delValue, petsc.PetscOptionsSetValue

        """
        cdef const char *pr = NULL
        cdef const char *nm = NULL
        cdef object unused = getpair(self.prefix, name, &pr, &nm)
        if pr == NULL:
            option = bytes2str(nm)
        else:
            option = '-%s%s' % (bytes2str(pr), bytes2str(&nm[1]))

        if isinstance(value, ndarray):
            value = value.tolist()
        if isinstance(value, (tuple, list)):
            value = str(value).replace(' ', '').\
                    replace('(', '').replace(')', '').\
                    replace('[', '').replace(']', '')
        elif isinstance(value, bool):
            value = str(value).lower()
        elif value is not None:
            value = str(value)
        cdef const char *key = NULL
        cdef const char *val = NULL
        option = str2bytes(option, &key)
        value  = str2bytes(value,  &val)
        CHKERR(PetscOptionsSetValue(self.opt, key, val))

    def delValue(self, name: str) -> None:
        """Delete an option from the database.

        Logically collective.

        See Also
        --------
        setValue, petsc.PetscOptionsClearValue

        """
        cdef const char *pr = NULL
        cdef const char *nm = NULL
        cdef object unused = getpair(self.prefix, name, &pr, &nm)
        if pr == NULL:
            option = bytes2str(nm)
        else:
            option = '-%s%s' % (bytes2str(pr), bytes2str(&nm[1]))
        cdef const char *key = NULL
        option = str2bytes(option, &key)
        CHKERR(PetscOptionsClearValue(self.opt, key))

    #

    def getBool(self, name: str, default=None) -> bool:
        """Return the boolean value associated with the option.

        Not collective.

        Parameters
        ----------
        name
            The option name.
        default
            The default value.
            If `None`, it raises a `KeyError` if the option is not found.

        See Also
        --------
        getBoolArray, petsc.PetscOptionsGetBool

        """
        return getopt(self.opt, OPT_BOOL, self.prefix, name, default)

    def getBoolArray(self, name: str, default=None) -> ArrayBool:
        """Return the boolean values associated with the option.

        Not collective.

        Parameters
        ----------
        name
            The option name.
        default
            The default value.
            If `None`, it raises a `KeyError` if the option is not found.

        See Also
        --------
        getBool, petsc.PetscOptionsGetBoolArray

        """
        return getopt(self.opt, OPT_BOOLARRAY, self.prefix, name, default)

    def getInt(self, name: str, default=None) -> int:
        """Return the integer value associated with the option.

        Not collective.

        Parameters
        ----------
        name
            The option name.
        default
            The default value.
            If `None`, it raises a `KeyError` if the option is not found.

        See Also
        --------
        getIntArray, petsc.PetscOptionsGetInt

        """
        return getopt(self.opt, OPT_INT, self.prefix, name, default)

    def getIntArray(self, name: str, default=None) -> ArrayInt:
        """Return the integer array associated with the option.

        Not collective.

        Parameters
        ----------
        name
            The option name.
        default
            The default value.
            If `None`, it raises a `KeyError` if the option is not found.

        See Also
        --------
        getInt, petsc.PetscOptionsGetIntArray

        """
        return getopt(self.opt, OPT_INTARRAY, self.prefix, name, default)

    def getReal(self, name: str, default=None) -> float:
        """Return the real value associated with the option.

        Not collective.

        Parameters
        ----------
        name
            The option name.
        default
            The default value.
            If `None`, it raises a `KeyError` if the option is not found.

        See Also
        --------
        getRealArray, petsc.PetscOptionsGetReal

        """
        return getopt(self.opt, OPT_REAL, self.prefix, name, default)

    def getRealArray(self, name: str, default=None) -> ArrayReal:
        """Return the real array associated with the option.

        Not collective.

        Parameters
        ----------
        name
            The option name.
        default
            The default value.
            If `None`, it raises a `KeyError` if the option is not found.

        See Also
        --------
        getReal, petsc.PetscOptionsGetRealArray

        """
        return getopt(self.opt, OPT_REALARRAY, self.prefix, name, default)

    def getScalar(self, name: str, default=None) -> Scalar:
        """Return the scalar value associated with the option.

        Not collective.

        Parameters
        ----------
        name
            The option name.
        default
            The default value.
            If `None`, it raises a `KeyError` if the option is not found.

        See Also
        --------
        getScalarArray, petsc.PetscOptionsGetScalar

        """
        return getopt(self.opt, OPT_SCALAR, self.prefix, name, default)

    def getScalarArray(self, name: str, default=None) -> ArrayScalar:
        """Return the scalar array associated with the option.

        Not collective.

        Parameters
        ----------
        name
            The option name.
        default
            The default value.
            If `None`, it raises a `KeyError` if the option is not found.

        See Also
        --------
        getScalar, petsc.PetscOptionsGetScalarArray

        """
        return getopt(self.opt, OPT_SCALARARRAY, self.prefix, name, default)

    def getString(self, name: str, default=None) -> str:
        """Return the string associated with the option.

        Not collective.

        Parameters
        ----------
        name
            The option name.
        default
            The default value.
            If `None`, it raises a `KeyError` if the option is not found.

        See Also
        --------
        petsc.PetscOptionsGetString

        """
        return getopt(self.opt, OPT_STRING, self.prefix, name, default)

    #

    def insertString(self, string: str) -> None:
        """Insert a string in the options database.

        Logically collective.

        See Also
        --------
        petsc.PetscOptionsInsertString

        """
        cdef const char *cstring = NULL
        string = str2bytes(string, &cstring)
        CHKERR(PetscOptionsInsertString(self.opt, cstring))

    def getAll(self) -> dict[str, str]:
        """Return all the options and their values.

        Not collective.

        See Also
        --------
        petsc.PetscOptionsGetAll

        """
        cdef char *allopts = NULL
        CHKERR(PetscOptionsGetAll(self.opt, &allopts))
        options = bytes2str(allopts)
        CHKERR(PetscFree(allopts))
        return parseopt(options, self.prefix)

# --------------------------------------------------------------------
