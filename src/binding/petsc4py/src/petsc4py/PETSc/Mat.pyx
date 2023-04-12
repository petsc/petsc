# --------------------------------------------------------------------

class MatType(object):
    """Matrix type.

    See Also
    --------
    petsc.MatType

    """
    SAME            = S_(MATSAME)
    MAIJ            = S_(MATMAIJ)
    SEQMAIJ         = S_(MATSEQMAIJ)
    MPIMAIJ         = S_(MATMPIMAIJ)
    KAIJ            = S_(MATKAIJ)
    SEQKAIJ         = S_(MATSEQKAIJ)
    MPIKAIJ         = S_(MATMPIKAIJ)
    IS              = S_(MATIS)
    AIJ             = S_(MATAIJ)
    SEQAIJ          = S_(MATSEQAIJ)
    MPIAIJ          = S_(MATMPIAIJ)
    AIJCRL          = S_(MATAIJCRL)
    SEQAIJCRL       = S_(MATSEQAIJCRL)
    MPIAIJCRL       = S_(MATMPIAIJCRL)
    AIJCUSPARSE     = S_(MATAIJCUSPARSE)
    SEQAIJCUSPARSE  = S_(MATSEQAIJCUSPARSE)
    MPIAIJCUSPARSE  = S_(MATMPIAIJCUSPARSE)
    AIJVIENNACL     = S_(MATAIJVIENNACL)
    SEQAIJVIENNACL  = S_(MATSEQAIJVIENNACL)
    MPIAIJVIENNACL  = S_(MATMPIAIJVIENNACL)
    AIJPERM         = S_(MATAIJPERM)
    SEQAIJPERM      = S_(MATSEQAIJPERM)
    MPIAIJPERM      = S_(MATMPIAIJPERM)
    AIJSELL         = S_(MATAIJSELL)
    SEQAIJSELL      = S_(MATSEQAIJSELL)
    MPIAIJSELL      = S_(MATMPIAIJSELL)
    AIJMKL          = S_(MATAIJMKL)
    SEQAIJMKL       = S_(MATSEQAIJMKL)
    MPIAIJMKL       = S_(MATMPIAIJMKL)
    BAIJMKL         = S_(MATBAIJMKL)
    SEQBAIJMKL      = S_(MATSEQBAIJMKL)
    MPIBAIJMKL      = S_(MATMPIBAIJMKL)
    SHELL           = S_(MATSHELL)
    DENSE           = S_(MATDENSE)
    DENSECUDA       = S_(MATDENSECUDA)
    SEQDENSE        = S_(MATSEQDENSE)
    SEQDENSECUDA    = S_(MATSEQDENSECUDA)
    MPIDENSE        = S_(MATMPIDENSE)
    MPIDENSECUDA    = S_(MATMPIDENSECUDA)
    ELEMENTAL       = S_(MATELEMENTAL)
    BAIJ            = S_(MATBAIJ)
    SEQBAIJ         = S_(MATSEQBAIJ)
    MPIBAIJ         = S_(MATMPIBAIJ)
    MPIADJ          = S_(MATMPIADJ)
    SBAIJ           = S_(MATSBAIJ)
    SEQSBAIJ        = S_(MATSEQSBAIJ)
    MPISBAIJ        = S_(MATMPISBAIJ)
    MFFD            = S_(MATMFFD)
    NORMAL          = S_(MATNORMAL)
    NORMALHERMITIAN = S_(MATNORMALHERMITIAN)
    LRC             = S_(MATLRC)
    SCATTER         = S_(MATSCATTER)
    BLOCKMAT        = S_(MATBLOCKMAT)
    COMPOSITE       = S_(MATCOMPOSITE)
    FFT             = S_(MATFFT)
    FFTW            = S_(MATFFTW)
    SEQCUFFT        = S_(MATSEQCUFFT)
    TRANSPOSE       = S_(MATTRANSPOSEVIRTUAL)
    HERMITIANTRANSPOSE = S_(MATHERMITIANTRANSPOSEVIRTUAL)
    SCHURCOMPLEMENT = S_(MATSCHURCOMPLEMENT)
    PYTHON          = S_(MATPYTHON)
    HYPRE           = S_(MATHYPRE)
    HYPRESTRUCT     = S_(MATHYPRESTRUCT)
    HYPRESSTRUCT    = S_(MATHYPRESSTRUCT)
    SUBMATRIX       = S_(MATSUBMATRIX)
    LOCALREF        = S_(MATLOCALREF)
    NEST            = S_(MATNEST)
    PREALLOCATOR    = S_(MATPREALLOCATOR)
    SELL            = S_(MATSELL)
    SEQSELL         = S_(MATSEQSELL)
    MPISELL         = S_(MATMPISELL)
    DUMMY           = S_(MATDUMMY)
    LMVM            = S_(MATLMVM)
    LMVMDFP         = S_(MATLMVMDFP)
    LMVMBFGS        = S_(MATLMVMBFGS)
    LMVMSR1         = S_(MATLMVMSR1)
    LMVMBROYDEN     = S_(MATLMVMBROYDEN)
    LMVMBADBROYDEN  = S_(MATLMVMBADBROYDEN)
    LMVMSYMBROYDEN  = S_(MATLMVMSYMBROYDEN)
    LMVMSYMBADBROYDEN = S_(MATLMVMSYMBADBROYDEN)
    LMVMDIAGBBROYDEN = S_(MATLMVMDIAGBROYDEN)
    CONSTANTDIAGONAL = S_(MATCONSTANTDIAGONAL)
    H2OPUS           = S_(MATH2OPUS)

class MatOption(object):
    """Matrix option.

    See Also
    --------
    petsc.MatOption

    """
    OPTION_MIN                  = MAT_OPTION_MIN
    UNUSED_NONZERO_LOCATION_ERR = MAT_UNUSED_NONZERO_LOCATION_ERR
    ROW_ORIENTED                = MAT_ROW_ORIENTED
    SYMMETRIC                   = MAT_SYMMETRIC
    STRUCTURALLY_SYMMETRIC      = MAT_STRUCTURALLY_SYMMETRIC
    FORCE_DIAGONAL_ENTRIES      = MAT_FORCE_DIAGONAL_ENTRIES
    IGNORE_OFF_PROC_ENTRIES     = MAT_IGNORE_OFF_PROC_ENTRIES
    USE_HASH_TABLE              = MAT_USE_HASH_TABLE
    KEEP_NONZERO_PATTERN        = MAT_KEEP_NONZERO_PATTERN
    IGNORE_ZERO_ENTRIES         = MAT_IGNORE_ZERO_ENTRIES
    USE_INODES                  = MAT_USE_INODES
    HERMITIAN                   = MAT_HERMITIAN
    SYMMETRY_ETERNAL            = MAT_SYMMETRY_ETERNAL
    NEW_NONZERO_LOCATION_ERR    = MAT_NEW_NONZERO_LOCATION_ERR
    IGNORE_LOWER_TRIANGULAR     = MAT_IGNORE_LOWER_TRIANGULAR
    ERROR_LOWER_TRIANGULAR      = MAT_ERROR_LOWER_TRIANGULAR
    GETROW_UPPERTRIANGULAR      = MAT_GETROW_UPPERTRIANGULAR
    SPD                         = MAT_SPD
    NO_OFF_PROC_ZERO_ROWS       = MAT_NO_OFF_PROC_ZERO_ROWS
    NO_OFF_PROC_ENTRIES         = MAT_NO_OFF_PROC_ENTRIES
    NEW_NONZERO_LOCATIONS       = MAT_NEW_NONZERO_LOCATIONS
    NEW_NONZERO_ALLOCATION_ERR  = MAT_NEW_NONZERO_ALLOCATION_ERR
    SUBSET_OFF_PROC_ENTRIES     = MAT_SUBSET_OFF_PROC_ENTRIES
    SUBMAT_SINGLEIS             = MAT_SUBMAT_SINGLEIS
    STRUCTURE_ONLY              = MAT_STRUCTURE_ONLY
    SORTED_FULL                 = MAT_SORTED_FULL
    OPTION_MAX                  = MAT_OPTION_MAX

class MatAssemblyType(object):
    """Matrix assembly type.

    See Also
    --------
    petsc.MatAssemblyType

    """
    # native
    FINAL_ASSEMBLY = MAT_FINAL_ASSEMBLY
    FLUSH_ASSEMBLY = MAT_FLUSH_ASSEMBLY
    # aliases
    FINAL = FINAL_ASSEMBLY
    FLUSH = FLUSH_ASSEMBLY

class MatInfoType(object):
    """Matrix info type."""
    LOCAL = MAT_LOCAL
    GLOBAL_MAX = MAT_GLOBAL_MAX
    GLOBAL_SUM = MAT_GLOBAL_SUM

class MatStructure(object):
    """Matrix modification structure.

    See Also
    --------
    petsc.MatStructure

    """
    # native
    SAME_NONZERO_PATTERN      = MAT_SAME_NONZERO_PATTERN
    DIFFERENT_NONZERO_PATTERN = MAT_DIFFERENT_NONZERO_PATTERN
    SUBSET_NONZERO_PATTERN    = MAT_SUBSET_NONZERO_PATTERN
    UNKNOWN_NONZERO_PATTERN   = MAT_UNKNOWN_NONZERO_PATTERN
    # aliases
    SAME      = SAME_NZ      = SAME_NONZERO_PATTERN
    SUBSET    = SUBSET_NZ    = SUBSET_NONZERO_PATTERN
    DIFFERENT = DIFFERENT_NZ = DIFFERENT_NONZERO_PATTERN
    UNKNOWN   = UNKNOWN_NZ   = UNKNOWN_NONZERO_PATTERN

class MatDuplicateOption(object):
    """Matrix duplicate option.

    See Also
    --------
    petsc.MatDuplicateOption

    """
    DO_NOT_COPY_VALUES    = MAT_DO_NOT_COPY_VALUES
    COPY_VALUES           = MAT_COPY_VALUES
    SHARE_NONZERO_PATTERN = MAT_SHARE_NONZERO_PATTERN

class MatOrderingType(object):
    """Factored matrix ordering type.

    See Also
    --------
    petsc.MatOrderingType

    """
    NATURAL     = S_(MATORDERINGNATURAL)
    ND          = S_(MATORDERINGND)
    OWD         = S_(MATORDERING1WD)
    RCM         = S_(MATORDERINGRCM)
    QMD         = S_(MATORDERINGQMD)
    ROWLENGTH   = S_(MATORDERINGROWLENGTH)
    WBM         = S_(MATORDERINGWBM)
    SPECTRAL    = S_(MATORDERINGSPECTRAL)
    AMD         = S_(MATORDERINGAMD)
    METISND     = S_(MATORDERINGMETISND)

class MatSolverType(object):
    """Factored matrix solver type.

    See Also
    --------
    petsc.MatSolverType

    """
    SUPERLU         = S_(MATSOLVERSUPERLU)
    SUPERLU_DIST    = S_(MATSOLVERSUPERLU_DIST)
    STRUMPACK       = S_(MATSOLVERSTRUMPACK)
    UMFPACK         = S_(MATSOLVERUMFPACK)
    CHOLMOD         = S_(MATSOLVERCHOLMOD)
    KLU             = S_(MATSOLVERKLU)
    ELEMENTAL       = S_(MATSOLVERELEMENTAL)
    SCALAPACK       = S_(MATSOLVERSCALAPACK)
    ESSL            = S_(MATSOLVERESSL)
    LUSOL           = S_(MATSOLVERLUSOL)
    MUMPS           = S_(MATSOLVERMUMPS)
    MKL_PARDISO     = S_(MATSOLVERMKL_PARDISO)
    MKL_CPARDISO    = S_(MATSOLVERMKL_CPARDISO)
    PASTIX          = S_(MATSOLVERPASTIX)
    MATLAB          = S_(MATSOLVERMATLAB)
    PETSC           = S_(MATSOLVERPETSC)
    BAS             = S_(MATSOLVERBAS)
    CUSPARSE        = S_(MATSOLVERCUSPARSE)
    CUDA            = S_(MATSOLVERCUDA)
    SPQR            = S_(MATSOLVERSPQR)

class MatFactorShiftType(object):
    """Factored matrix shift type.

    See Also
    --------
    petsc.MatFactorShiftType

    """
    # native
    NONE              = MAT_SHIFT_NONE
    NONZERO           = MAT_SHIFT_NONZERO
    POSITIVE_DEFINITE = MAT_SHIFT_POSITIVE_DEFINITE
    INBLOCKS          = MAT_SHIFT_INBLOCKS
    # aliases
    NZ = MAT_SHIFT_NONZERO
    PD = MAT_SHIFT_POSITIVE_DEFINITE

class MatSORType(object):
    """Matrix SOR type.

    See Also
    --------
    petsc.MatSORType

    """
    FORWARD_SWEEP         = SOR_FORWARD_SWEEP
    BACKWARD_SWEEP        = SOR_BACKWARD_SWEEP
    SYMMETRY_SWEEP        = SOR_SYMMETRIC_SWEEP
    LOCAL_FORWARD_SWEEP   = SOR_LOCAL_FORWARD_SWEEP
    LOCAL_BACKWARD_SWEEP  = SOR_LOCAL_BACKWARD_SWEEP
    LOCAL_SYMMETRIC_SWEEP = SOR_LOCAL_SYMMETRIC_SWEEP
    ZERO_INITIAL_GUESS    = SOR_ZERO_INITIAL_GUESS
    EISENSTAT             = SOR_EISENSTAT
    APPLY_UPPER           = SOR_APPLY_UPPER
    APPLY_LOWER           = SOR_APPLY_LOWER

@cython.internal
cdef class MatStencil:
    """Associate structured grid coordinates with matrix indices.

    See Also
    --------
    petsc.MatStencil

    """

    cdef PetscMatStencil stencil

    property i:
        "First logical grid coordinate."
        def __get__(self) -> int:
            return toInt(self.stencil.i)
        def __set__(self, value: int) -> None:
            self.stencil.i = asInt(value)

    property j:
        "Second logical grid coordinate."
        def __get__(self) -> int:
            return toInt(self.stencil.j)
        def __set__(self, value: int) -> None:
            self.stencil.j = asInt(value)

    property k:
        "Third logical grid coordinate."
        def __get__(self) -> int:
            return toInt(self.stencil.k)
        def __set__(self, value: int) -> None:
            self.stencil.k = asInt(value)

    property c:
        "Field component."
        def __get__(self) -> int:
            return toInt(self.stencil.c)
        def __set__(self, value: int) -> None:
            self.stencil.c = asInt(value)

    property index:
        "Logical grid coordinates ``(i, j, k)``."
        def __get__(self) -> tuple[int, int, int]:
            cdef PetscMatStencil *s = &self.stencil
            return toInt(s.i), toInt(s.j), toInt(s.k)
        def __set__(self, value: Sequence[int]) -> None:
            cdef PetscMatStencil *s = &self.stencil
            s.i = s.j = s.k = 0
            asDims(value, &s.i, &s.j, &s.k)

    property field:
        "Field component."
        def __get__(self) -> int:
            cdef PetscMatStencil *s = &self.stencil
            return toInt(s.c)
        def __set__(self, value: int) -> None:
            cdef PetscMatStencil *s = &self.stencil
            s.c = asInt(value)

# --------------------------------------------------------------------

cdef class Mat(Object):
    """Matrix object.

    Mat is described in the `PETSc manual <petsc:manual/mat>`.

    See Also
    --------
    petsc.Mat

    """

    Type            = MatType
    Option          = MatOption
    AssemblyType    = MatAssemblyType
    InfoType        = MatInfoType
    Structure       = MatStructure
    DuplicateOption = MatDuplicateOption
    OrderingType    = MatOrderingType
    SolverType      = MatSolverType
    FactorShiftType = MatFactorShiftType
    SORType         = MatSORType
    #

    def __cinit__(self):
        self.obj = <PetscObject*> &self.mat
        self.mat = NULL

    # unary operations

    def __pos__(self):
        return mat_pos(self)

    def __neg__(self):
        return mat_neg(self)

    # inplace binary operations

    def __iadd__(self, other):
        return mat_iadd(self, other)

    def __isub__(self, other):
        return mat_isub(self, other)

    def __imul__(self, other):
        return mat_imul(self, other)

    def __idiv__(self, other):
        return mat_idiv(self, other)

    def __itruediv__(self, other):
        return mat_idiv(self, other)

    # binary operations

    def __add__(self, other):
        if isinstance(self, Mat):
            return mat_add(self, other)
        else:
            return mat_radd(other, self)

    def __sub__(self, other):
        if isinstance(self, Mat):
            return mat_sub(self, other)
        else:
            return mat_rsub(other, self)

    def __mul__(self, other):
        if isinstance(self, Mat):
            if isinstance(other, Vec):
                return mat_mul_vec(self, other)
            else:
                return mat_mul(self, other)
        else:
            return mat_rmul(other, self)

    def __div__(self, other):
        if isinstance(self, Mat):
            return mat_div(self, other)
        else:
            return mat_rdiv(other, self)

    def __truediv__(self, other):
        if isinstance(self, Mat):
            return mat_div(self, other)
        else:
            return mat_rdiv(other, self)

    #

    def __getitem__(self, ij):
        return mat_getitem(self, ij)

    def __setitem__(self, ij, v):
        mat_setitem(self, ij, v)

    def __call__(self, x, y=None):
        if y is None:
            y = self.createVecLeft()
        self.mult(x, y)
        return y
    #

    def view(self, Viewer viewer=None) -> None:
        """View the matrix.

        Collective.

        Parameters
        ----------
        viewer
            A `Viewer` instance or `None` for the default viewer.

        Notes
        -----
        Viewers with type `Viewer.Type.ASCII` are only recommended for small
        matrices on small numbers of processes. Larger matrices should use a
        binary format like `Viewer.Type.BINARY`.

        See Also
        --------
        load, Viewer, petsc.MatView

        """
        cdef PetscViewer vwr = NULL
        if viewer is not None: vwr = viewer.vwr
        CHKERR( MatView(self.mat, vwr) )

    def destroy(self) -> Self:
        """Destroy the matrix.

        Collective.

        See Also
        --------
        create, petsc.MatDestroy

        """
        CHKERR( MatDestroy(&self.mat) )
        return self

    def create(self, comm: Comm | None = None) -> Self:
        """Create the matrix.

        Collective.

        Once created, the user should call `setType` or
        `setFromOptions` before using the matrix. Alternatively, specific
        creation routines can be used such as `createAIJ` or
        `createBAIJ` can be used.

        Parameters
        ----------
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        destroy, petsc.MatCreate

        """
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscMat newmat = NULL
        CHKERR( MatCreate(ccomm, &newmat) )
        PetscCLEAR(self.obj); self.mat = newmat
        return self

    def setType(self, mat_type: Type | str) -> None:
        """Set the matrix type.

        Collective.

        Parameters
        ----------
        mat_type
            The matrix type.

        See Also
        --------
        create, getType, petsc.MatSetType

        """
        cdef PetscMatType cval = NULL
        mat_type = str2bytes(mat_type, &cval)
        CHKERR( MatSetType(self.mat, cval) )

    def setSizes(
        self,
        size: MatSizeSpec,
        bsize: MatBlockSizeSpec | None = None,
    ) -> None:
        """Set the local, global and block sizes.

        Collective.

        Parameters
        ----------
        size
            Matrix size.
        bsize
            Matrix block size. If `None`, a block size of ``1`` is set.

        Examples
        --------
        Create a `Mat` with ``n`` rows and columns and the same local and
        global sizes.

        >>> mat = PETSc.Mat().create()
        >>> mat.setFromOptions()
        >>> mat.setSizes(n)

        Create a `Mat` with ``nr`` rows, ``nc`` columns and the same local and
        global sizes.

        >>> mat = PETSc.Mat().create()
        >>> mat.setFromOptions()
        >>> mat.setSizes([nr, nc])

        Create a `Mat` with ``nrl`` local rows, ``nrg`` global rows, ``ncl``
        local columns and ``ncg`` global columns.

        >>> mat = PETSc.Mat().create()
        >>> mat.setFromOptions()
        >>> mat.setSizes([[nrl, nrg], [ncl, ncg]])

        See Also
        --------
        setBlockSize, setBlockSizes, petsc.MatSetSizes, petsc.MatSetBlockSize
        petsc.MatSetBlockSizes

        """
        cdef PetscInt rbs = 0, cbs = 0, m = 0, n = 0, M = 0, N = 0
        Mat_Sizes(size, bsize, &rbs, &cbs, &m, &n, &M, &N)
        CHKERR( MatSetSizes(self.mat, m, n, M, N) )
        if rbs != PETSC_DECIDE:
            if cbs != PETSC_DECIDE:
                CHKERR( MatSetBlockSizes(self.mat, rbs, cbs) )
            else:
                CHKERR( MatSetBlockSize(self.mat, rbs) )

    def setBlockSize(self, bsize: int) -> None:
        """Set the matrix block size (same for rows and columns).

        Logically collective.

        Parameters
        ----------
        bsize
            Block size.

        See Also
        --------
        setBlockSizes, setSizes, petsc.MatSetBlockSize

        """
        cdef PetscInt bs = asInt(bsize)
        CHKERR( MatSetBlockSize(self.mat, bs) )

    def setBlockSizes(self, row_bsize: int, col_bsize: int) -> None:
        """Set the row and column block sizes.

        Logically collective.

        Parameters
        ----------
        row_bsize
            Row block size.
        col_bsize
            Column block size.

        See Also
        --------
        setBlockSize, setSizes, petsc.MatSetBlockSizes

        """
        cdef PetscInt rbs = asInt(row_bsize)
        cdef PetscInt cbs = asInt(col_bsize)
        CHKERR( MatSetBlockSizes(self.mat, rbs, cbs) )

    def setVecType(self, vec_type: Vec.Type | str) -> None:
        """Set the vector type.

        Collective.

        Parameters
        ----------
        vec_type
            Vector type used when creating vectors with `createVecs`.

        See Also
        --------
        getVecType, petsc.MatSetVecType

        """
        cdef PetscVecType cval = NULL
        vec_type = str2bytes(vec_type, &cval)
        CHKERR( MatSetVecType(self.mat, cval) )

    def getVecType(self) -> str:
        """Return the vector type used by the matrix.

        Not collective.

        See Also
        --------
        setVecType, petsc.MatGetVecType

        """
        cdef PetscVecType cval = NULL
        CHKERR( MatGetVecType(self.mat, &cval) )
        return bytes2str(cval)

    #

    def createAIJ(
        self,
        size: MatSizeSpec,
        bsize: MatBlockSizeSpec | None = None,
        nnz: NNZSpec | None = None,
        csr: CSRIndicesSpec | None = None,
        comm: Comm | None = None,
    ) -> Self:
        """Create a sparse `Type.AIJ` matrix, optionally preallocating.

        Collective.

        To preallocate the matrix the user can either pass ``nnz`` or ``csr``
        describing the sparsity. If neither is set then preallocation will not
        occur. Consult the `PETSc manual <petsc:sec_matsparse>` for
        more information.

        Parameters
        ----------
        size
            Matrix size.
        bsize
            Matrix block size. If `None`, a block size of ``1`` is set.
        nnz
            Optional non-zeros preallocation pattern.
        csr
            Optional compressed sparse row layout information.
            If provided, it takes precedence on ``nnz``.
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        setSizes, createBAIJ, petsc.MATAIJ, petsc.MATSEQAIJ, petsc.MATMPIAIJ
        petsc.MatCreateAIJ, petsc.MatSeqAIJSetPreallocation
        petsc.MatSeqAIJSetPreallocationCSR

        """
        # create matrix
        cdef PetscMat newmat = NULL
        Mat_Create(MATAIJ, comm, size, bsize, &newmat)
        PetscCLEAR(self.obj); self.mat = newmat
        # preallocate matrix
        Mat_AllocAIJ(self.mat, nnz, csr)
        return self

    def createBAIJ(
        self,
        size: MatSizeSpec,
        bsize: MatBlockSizeSpec,
        nnz: NNZSpec | None = None,
        csr: CSRIndicesSpec | None = None,
        comm: Comm | None = None,
    ) -> Self:
        """Create a sparse blocked `Type.BAIJ` matrix, optionally preallocating.

        Collective.

        To preallocate the matrix the user can either pass ``nnz`` or ``csr``
        describing the sparsity. If neither is set then preallocation will not
        occur. Consult the `PETSc manual <petsc:sec_matsparse>` for
        more information.

        Parameters
        ----------
        size
            Matrix size.
        bsize
            Matrix block size.
        nnz
            Optional non-zeros preallocation pattern for block rows.
        csr
            Optional block-compressed sparse row layout information.
            If provided, it takes precedence on ``nnz``.
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        setSizes, createAIJ, petsc.MATBAIJ, petsc.MATSEQBAIJ
        petsc.MATMPIBAIJ, petsc.MatCreateBAIJ

        """
        # create matrix
        cdef PetscMat newmat = NULL
        Mat_Create(MATBAIJ, comm, size, bsize, &newmat)
        PetscCLEAR(self.obj); self.mat = newmat
        # preallocate matrix
        Mat_AllocAIJ(self.mat, nnz, csr)
        return self

    def createSBAIJ(
        self,
        size: MatSizeSpec,
        bsize: int,
        nnz: NNZSpec | None = None,
        csr: CSRIndicesSpec | None = None,
        comm: Comm | None = None,
    ) -> Self:
        """Create a sparse `Type.SBAIJ` matrix in symmetric block format.

        Collective.

        To preallocate the matrix the user can either pass ``nnz`` or ``csr``
        describing the sparsity. If neither is set then preallocation will not
        occur. Consult the `PETSc manual <petsc:sec_matsparse>` for
        more information.

        Parameters
        ----------
        size
            Matrix size.
        bsize
            Matrix block size.
        nnz
            Optional upper-triangular (including diagonal)
            non-zeros preallocation pattern for block rows.
        csr
            Optional block-compressed sparse row layout information.
            If provided, it takes precedence on ``nnz``.
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        createAIJ, createBAIJ, petsc.MatCreateSBAIJ

        """
        # create matrix
        cdef PetscMat newmat = NULL
        Mat_Create(MATSBAIJ, comm, size, bsize, &newmat)
        PetscCLEAR(self.obj); self.mat = newmat
        # preallocate matrix
        Mat_AllocAIJ(self.mat, nnz, csr)
        return self

    def createAIJCRL(
        self,
        size: MatSizeSpec,
        bsize: MatBlockSizeSpec | None = None,
        nnz: NNZSpec | None = None,
        csr: CSRIndicesSpec | None = None,
        comm: Comm | None = None,
    ) -> Self:
        """Create a sparse `Type.AIJCRL` matrix.

        Collective.

        This is similar to `Type.AIJ` matrices but stores some additional
        information that improves vectorization for the matrix-vector product.

        To preallocate the matrix the user can either pass ``nnz`` or ``csr``
        describing the sparsity. If neither is set then preallocation will not
        occur. Consult the `PETSc manual <petsc:sec_matsparse>` for
        more information.

        Parameters
        ----------
        size
            Matrix size.
        bsize
            Matrix block size. If `None`, a block size of ``1`` is set.
        nnz
            Optional non-zeros preallocation pattern.
        csr
            Optional compressed sparse row layout information.
            If provided, it takes precedence on ``nnz``.
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        createAIJ, createBAIJ, petsc.MatCreateSeqAIJCRL
        petsc.MatCreateMPIAIJCRL

        """
        # create matrix
        cdef PetscMat newmat = NULL
        Mat_Create(MATAIJCRL, comm, size, bsize, &newmat)
        PetscCLEAR(self.obj); self.mat = newmat
        # preallocate matrix
        Mat_AllocAIJ(self.mat, nnz, csr)
        return self

    def setPreallocationNNZ(self, nnz: NNZSpec) -> Self:
        """Preallocate memory for the matrix with a non-zero pattern.

        Collective.

        Correct preallocation can result in a dramatic reduction in matrix
        assembly time.

        Parameters
        ----------
        nnz
            The number of non-zeros per row for the local portion of the matrix,
            or a 2-tuple for the on-process and off-process part of the matrix.

        See Also
        --------
        setPreallocationCSR, createAIJ, petsc.MatSeqAIJSetPreallocation
        petsc.MatMPIAIJSetPreallocation

        """
        cdef PetscBool done = PETSC_FALSE
        CHKERR( MatIsPreallocated(self.mat, &done) )
        # if done: raise Error(PETSC_ERR_ORDER)
        Mat_AllocAIJ_NNZ(self.mat, nnz)
        return self

    def setPreallocationCSR(self, csr: CSRIndicesSpec) -> Self:
        """Preallocate memory for the matrix with a CSR layout.

        Collective.

        Correct preallocation can result in a dramatic reduction in matrix
        assembly time.

        Parameters
        ----------
        csr
            Local matrix data in compressed sparse row layout format.

        Notes
        -----
        Must use the block-compressed form with `Type.BAIJ` and `Type.SBAIJ`.

        See Also
        --------
        setPreallocationNNZ, createAIJ, createBAIJ, createSBAIJ
        petsc.MatSeqAIJSetPreallocationCSR
        petsc.MatMPIAIJSetPreallocationCSR
        petsc.MatSeqBAIJSetPreallocationCSR
        petsc.MatMPIBAIJSetPreallocationCSR
        petsc.MatSeqSBAIJSetPreallocationCSR
        petsc.MatMPISBAIJSetPreallocationCSR

        """
        cdef PetscBool done = PETSC_FALSE
        CHKERR( MatIsPreallocated(self.mat, &done) )
        # if done: raise Error(PETSC_ERR_ORDER)
        Mat_AllocAIJ_CSR(self.mat, csr)
        return self

    def createAIJWithArrays(
        self,
        size: MatSizeSpec,
        csr: CSRSpec | tuple[CSRSpec, CSRSpec],
        bsize: MatBlockSizeSpec | None = None,
        comm: Comm | None = None,
    ) -> Self:
        """Create a sparse `Type.AIJ` matrix with data in CSR format.

        Collective.

        Parameters
        ----------
        size
            Matrix size.
        csr
            Local matrix data in compressed sparse row format.
        bsize
            Matrix block size. If `None`, a block size of ``1`` is set.
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        Notes
        -----
        For `Type.SEQAIJ` matrices, the ``csr`` data is not copied.
        For `Type.MPIAIJ` matrices, the ``csr`` data is not copied only
        in the case it represents on-process and off-process information.

        See Also
        --------
        createAIJ, petsc.MatCreateSeqAIJWithArrays
        petsc.MatCreateMPIAIJWithArrays, petsc.MatCreateMPIAIJWithSplitArrays

        """
        # communicator
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        # sizes and block sizes
        cdef PetscInt rbs = 0, cbs = 0, m = 0, n = 0, M = 0, N = 0
        Mat_Sizes(size, bsize, &rbs, &cbs, &m, &n, &M, &N)
        if rbs == PETSC_DECIDE: rbs = 1
        if cbs == PETSC_DECIDE: cbs = rbs
        Sys_Layout(ccomm, rbs, &m, &M)
        Sys_Layout(ccomm, cbs, &n, &N)
        # unpack CSR argument
        cdef object pi, pj, pv, poi, poj, pov
        try:
            (pi, pj, pv), (poi, poj, pov) = csr
        except (TypeError, ValueError):
            pi, pj, pv = csr
            poi = poj = pov = None
        # rows, cols, and values
        cdef PetscInt ni=0, noi=0, *i=NULL, *oi=NULL
        cdef PetscInt nj=0, noj=0, *j=NULL, *oj=NULL
        pi = iarray_i(pi, &ni, &i) # Row pointers (diagonal)
        pj = iarray_i(pj, &nj, &j) # Column indices (diagonal)
        if ni != m+1:  raise ValueError(
            "A matrix with %d rows requires a row pointer of length %d (given: %d)" %
            (toInt(m), toInt(m+1), toInt(ni)))
        if poi is not None and poj is not None:
            poi = iarray_i(poi, &noi, &oi) # Row pointers (off-diagonal)
            poj = iarray_i(poj, &noj, &oj) # Column indices (off-diagonal)
        cdef PetscInt nv=0, nov=0
        cdef PetscScalar *v=NULL, *ov=NULL
        pv = iarray_s(pv, &nv, &v) # Non-zero values (diagonal)
        if nj != nv:  raise ValueError(
            "Given %d column indices but %d non-zero values" %
            (toInt(nj), toInt(nv)))
        if pov is not None:
            pov = iarray_s(pov, &nov, &ov) # Non-zero values (off-diagonal)
        # create matrix
        cdef PetscMat newmat = NULL
        if comm_size(ccomm) == 1:
            CHKERR( MatCreateSeqAIJWithArrays(
                ccomm, m, n, i, j, v, &newmat) )
            csr = (pi, pj, pv)
        else:
            # if off-diagonal components are provided then SplitArrays can be
            # used (and not cause a copy).
            if oi != NULL and oj != NULL and ov != NULL:
                CHKERR( MatCreateMPIAIJWithSplitArrays(
                    ccomm, m, n, M, N, i, j, v, oi, oj, ov, &newmat) )
                csr = ((pi, pj, pv), (poi, poj, pov))
            else:
                CHKERR( MatCreateMPIAIJWithArrays(
                    ccomm, m, n, M, N, i, j, v, &newmat) )
                csr = None
        PetscCLEAR(self.obj); self.mat = newmat
        self.set_attr('__csr__', csr)
        return self

    #

    def createDense(
        self,
        size: MatSizeSpec,
        bsize: MatBlockSizeSpec | None = None,
        array: Sequence[Scalar] | None = None,
        comm: Comm | None = None
    ) -> Self:
        """Create a `Type.DENSE` matrix.

        Collective.

        Parameters
        ----------
        size
            Matrix size.
        bsize
            Matrix block size. If `None`, a block size of ``1`` is set.
        array
            Optional matrix data. If `None`, memory is internally allocated.
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        createDenseCUDA, petsc.MATDENSE, petsc.MatCreateDense

        """
        # create matrix
        cdef PetscMat newmat = NULL
        Mat_Create(MATDENSE, comm, size, bsize, &newmat)
        PetscCLEAR(self.obj); self.mat = newmat
        # preallocate matrix
        if array is not None:
            array = Mat_AllocDense(self.mat, array)
            self.set_attr('__array__', array)
        return self

    def createDenseCUDA(
        self,
        size: MatSizeSpec,
        bsize: MatBlockSizeSpec | None = None,
        array: Sequence[Scalar] | None = None,
        cudahandle: int | None = None,
        comm: Comm | None = None,
    ) -> Self:
        """Create a `Type.DENSECUDA` matrix with optional host and device data.

        Collective.

        Parameters
        ----------
        size
            Matrix size.
        bsize
            Matrix block size. If `None`, a block size of ``1`` is set.
        array
            Host data. Will be lazily allocated if `None`.
        cudahandle
            Address of the array on the GPU. Will be lazily allocated if
            `None`. If ``cudahandle`` is provided, ``array`` will be
            ignored.
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        createDense, petsc.MatCreateDenseCUDA

        """
        # create matrix
        cdef PetscMat newmat = NULL
        # communicator
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        # sizes and block sizes
        cdef PetscInt rbs = 0, cbs = 0, m = 0, n = 0, M = 0, N = 0

        # FIXME handle the case of array not None?
        if cudahandle is not None:
            Mat_Sizes(size, None, &rbs, &cbs, &m, &n, &M, &N)
            if rbs == PETSC_DECIDE: rbs = 1
            if cbs == PETSC_DECIDE: cbs = rbs
            Sys_Layout(ccomm, rbs, &m, &M)
            Sys_Layout(ccomm, cbs, &n, &N)
            # create matrix and set sizes
            CHKERR( MatCreateDenseCUDA(ccomm, m, n, M, N, <PetscScalar*>(<Py_uintptr_t>cudahandle), &newmat) )
            # Does block size make sense for MATDENSE?
            CHKERR( MatSetBlockSizes(newmat, rbs, cbs) )
        else:
            Mat_Create(MATDENSECUDA, comm, size, bsize, &newmat)
            if array is not None:
                array = Mat_AllocDense(self.mat, array)
                self.set_attr('__array__', array)
        PetscCLEAR(self.obj); self.mat = newmat
        return self

    def setPreallocationDense(self, array: Sequence[Scalar]) -> Self:
        """Set the array used for storing matrix elements for a dense matrix.

        Collective.

        Parameters
        ----------
        array
            Array that will be used to store matrix data.

        See Also
        --------
        petsc.MatSeqDenseSetPreallocation, petsc.MatMPIDenseSetPreallocation

        """
        cdef PetscBool done = PETSC_FALSE
        CHKERR( MatIsPreallocated(self.mat, &done) )
        # if done: raise Error(PETSC_ERR_ORDER)
        array = Mat_AllocDense(self.mat, array)
        self.set_attr('__array__', array)
        return self

    #

    def createScatter(self, Scatter scatter, comm: Comm | None = None) -> Self:
        """Create a `Type.SCATTER` matrix from a vector scatter.

        Collective.

        Parameters
        ----------
        scatter
            Vector scatter.
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        petsc.MATSCATTER, petsc.MatCreateScatter

        """
        if comm is None: comm = scatter.getComm()
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscMat newmat = NULL
        CHKERR( MatCreateScatter(ccomm, scatter.sct, &newmat) )
        PetscCLEAR(self.obj); self.mat = newmat
        return self

    def createNormal(self, Mat mat) -> Self:
        """Create a `Type.NORMAL` matrix representing AᵀA.

        Collective.

        Parameters
        ----------
        mat
            The (possibly rectangular) matrix A.

        Notes
        -----
        The product AᵀA is never actually formed. Instead A and Aᵀ are used
        during `mult` and various other matrix operations.

        See Also
        --------
        petsc.MATNORMAL, petsc.MatCreateNormal

        """
        cdef PetscMat newmat = NULL
        CHKERR( MatCreateNormal(mat.mat, &newmat) )
        PetscCLEAR(self.obj); self.mat = newmat
        return self

    def createTranspose(self, Mat mat) -> Self:
        """Create a `Type.TRANSPOSE` matrix that behaves like Aᵀ.

        Collective.

        Parameters
        ----------
        mat
            Matrix A to represent the transpose of.

        Notes
        -----
        The transpose is never actually formed. Instead `multTranspose` is
        called whenever the matrix-vector product is computed.

        See Also
        --------
        createNormal, petsc.MatCreateTranspose

        """
        cdef PetscMat newmat = NULL
        CHKERR( MatCreateTranspose(mat.mat, &newmat) )
        PetscCLEAR(self.obj); self.mat = newmat
        return self

    def createNormalHermitian(self, Mat mat) -> Self:
        """Create a `Type.NORMALHERMITIAN` matrix representing (A*)ᵀA.

        Collective.

        Parameters
        ----------
        mat
            The (possibly rectangular) matrix A.

        Notes
        -----
        The product (A*)ᵀA is never actually formed.

        See Also
        --------
        createHermitianTranspose, petsc.MATNORMAL, petsc.MATNORMALHERMITIAN
        petsc.MatCreateNormalHermitian

        """
        cdef PetscMat newmat = NULL
        CHKERR( MatCreateNormalHermitian(mat.mat, &newmat) )
        PetscCLEAR(self.obj); self.mat = newmat
        return self

    def createHermitianTranspose(self, Mat mat) -> Self:
        """Create a `Type.HERMITIANTRANSPOSE` matrix that behaves like (A*)ᵀ.

        Collective.

        Parameters
        ----------
        mat
            Matrix A to represent the hermitian transpose of.

        Notes
        -----
        The Hermitian transpose is never actually formed.

        See Also
        --------
        createNormal, createNormalHermitian
        petsc.MATHERMITIANTRANSPOSEVIRTUAL, petsc.MatCreateHermitianTranspose

        """
        cdef PetscMat newmat = NULL
        CHKERR( MatCreateHermitianTranspose(mat.mat, &newmat) )
        PetscCLEAR(self.obj); self.mat = newmat
        return self

    def createLRC(self, Mat A, Mat U, Vec c, Mat V) -> Self:
        """Create a low-rank correction `Type.LRC` matrix representing A + UCVᵀ.

        Collective.

        Parameters
        ----------
        A
            Sparse matrix, can be `None`.
        U, V
            Dense rectangular matrices.
        c
            Vector containing the diagonal of C, can be `None`.

        Notes
        -----
        The matrix A + UCVᵀ is never actually formed.

        C is a diagonal matrix (represented as a vector) of order k, where k
        is the number of columns of both U and V.

        If A is `None` then the new object behaves like a low-rank matrix UCVᵀ.

        Use the same matrix for ``V`` and ``U`` (or ``V=None``) for a symmetric
        low-rank correction, A + UCUᵀ.

        If ``c`` is `None` then the low-rank correction is just U*Vᵀ. If a
        sequential ``c`` vector is used for a parallel matrix, PETSc assumes
        that the values of the vector are consistently set across processors.

        See Also
        --------
        petsc.MATLRC, petsc.MatCreateLRC

        """
        cdef PetscMat Amat = NULL
        cdef PetscMat Umat = U.mat
        cdef PetscVec cvec = NULL
        cdef PetscMat Vmat = NULL
        cdef PetscMat newmat = NULL
        if A is not None: Amat = A.mat
        if c is not None: cvec = c.vec
        if V is not None: Vmat = V.mat
        CHKERR( MatCreateLRC(Amat, Umat, cvec, Vmat, &newmat) )
        PetscCLEAR(self.obj); self.mat = newmat
        return self

    def createSubMatrixVirtual(self, Mat A, IS isrow, IS iscol=None) -> Self:
        """Create a `Type.SUBMATRIX` matrix that acts as a submatrix.

        Collective.

        Parameters
        ----------
        A
            Matrix to extract submatrix from.
        isrow
            Rows present in the submatrix.
        iscol
            Columns present in the submatrix, defaults to ``isrow``.

        See Also
        --------
        petsc.MatCreateSubMatrixVirtual

        """
        if iscol is None: iscol = isrow
        cdef PetscMat newmat = NULL
        CHKERR( MatCreateSubMatrixVirtual(A.mat, isrow.iset, iscol.iset, &newmat) )
        PetscCLEAR(self.obj); self.mat = newmat
        return self

    def createNest(
        self,
        mats: Sequence[Mat],
        isrows: Sequence[IS] | None = None,
        iscols: Sequence[IS] | None = None,
        comm: Comm | None = None,
    ) -> Self:
        """Create a `Type.NEST` matrix containing multiple submatrices.

        Collective.

        Parameters
        ----------
        mats
            Row-aligned iterable of matrices with size
            ``len(isrows)*len(iscols)``. Empty submatrices can be set with
            `None`.
        isrows
            Index set for each nested row block, defaults to contiguous
            ordering.
        iscols
            Index set for each nested column block, defaults to contiguous
            ordering.
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        petsc.MatCreateNest, petsc.MATNEST

        """
        cdef object mat
        mats = [list(mat) for mat in mats]
        if isrows:
            isrows = list(isrows)
            assert len(isrows) == len(mats)
        else:
            isrows = None
        if iscols:
            iscols = list(iscols)
            assert len(iscols) == len(mats[0])
        else:
            iscols = None
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef Py_ssize_t i, mr = len(mats)
        cdef Py_ssize_t j, mc = len(mats[0])
        cdef PetscInt nr = <PetscInt>mr
        cdef PetscInt nc = <PetscInt>mc
        cdef PetscMat *cmats   = NULL
        cdef PetscIS  *cisrows = NULL
        cdef PetscIS  *ciscols = NULL
        cdef object tmp1, tmp2, tmp3
        tmp1 = oarray_p(empty_p(nr*nc), NULL, <void**>&cmats)
        for i from 0 <= i < mr:
            for j from 0 <= j < mc:
                mat = mats[i][j]
                cmats[i*mc+j] = (<Mat?>mat).mat if mat is not None else NULL
        if isrows is not None:
            tmp2 = oarray_p(empty_p(nr), NULL, <void**>&cisrows)
            for i from 0 <= i < mr: cisrows[i] = (<IS?>isrows[i]).iset
        if iscols is not None:
            tmp3 = oarray_p(empty_p(nc), NULL, <void**>&ciscols)
            for j from 0 <= j < mc: ciscols[j] = (<IS?>iscols[j]).iset
        cdef PetscMat newmat = NULL
        CHKERR( MatCreateNest(ccomm, nr, cisrows, nc, ciscols, cmats, &newmat) )
        PetscCLEAR(self.obj); self.mat = newmat
        return self

    def createH2OpusFromMat(
        self,
        Mat A,
        coordinates: Sequence[Scalar] | None = None,
        dist: bool | None = None,
        eta: float | None = None,
        leafsize: int | None = None,
        maxrank: int | None = None,
        bs: int | None = None,
        rtol: float | None = None,
    ) -> Self:
        """Create a hierarchical `Type.H2OPUS` matrix sampling from a provided operator.

        Parameters
        ----------
        A
            Matrix to be sampled.
        coordinates
            Coordinates of the points.
        dist
            Whether or not coordinates are distributed, defaults to `False`.
        eta
            Admissibility condition tolerance, defaults to `DECIDE`.
        leafsize
            Leaf size in cluster tree, defaults to `DECIDE`.
        maxrank
            Maximum rank permitted, defaults to `DECIDE`.
        bs
            Maximum number of samples to take concurrently, defaults to
            `DECIDE`.
        rtol
            Relative tolerance for construction, defaults to `DECIDE`.

        Notes
        -----
        See `petsc.MatCreateH2OpusFromMat` for the appropriate database
        options.

        See Also
        --------
        petsc_options, petsc.MatCreateH2OpusFromMat

        """
        cdef PetscInt cdim = 1
        cdef PetscReal *coords = NULL
        cdef PetscBool cdist = PETSC_FALSE
        cdef PetscReal peta = PETSC_DECIDE
        cdef PetscInt lsize = PETSC_DECIDE
        cdef PetscInt maxr = PETSC_DECIDE
        cdef PetscInt pbs = PETSC_DECIDE
        cdef PetscReal tol = PETSC_DECIDE
        cdef ndarray xyz
        cdef PetscInt nvtx
        cdef PetscInt rl = 0, cl = 0
        if dist is not None: cdist = asBool(dist)
        if eta is not None: peta = asReal(eta)
        if leafsize is not None: lsize = asInt(leafsize)
        if maxrank is not None: maxr = asInt(maxrank)
        if bs is not None: pbs = asInt(bs)
        if rtol is not None: tol = asReal(rtol)

        if coordinates is not None:
            xyz = iarray(coordinates, NPY_PETSC_REAL)
            if PyArray_ISFORTRAN(xyz): xyz = PyArray_Copy(xyz)
            if PyArray_NDIM(xyz) != 2: raise ValueError(
                ("coordinates must have two dimensions: "
                 "coordinates.ndim=%d") % (PyArray_NDIM(xyz)) )
            nvtx = <PetscInt> PyArray_DIM(xyz, 0)
            CHKERR( MatGetLocalSize(A.mat, &rl, &cl) )
            if cl != rl: raise ValueError("Not for rectangular matrices")
            if nvtx < rl: raise ValueError(
                ("coordinates size must be at least %d" % rl ))
            cdim = <PetscInt> PyArray_DIM(xyz, 1)
            coords = <PetscReal*> PyArray_DATA(xyz)

        cdef PetscMat newmat = NULL
        CHKERR( MatCreateH2OpusFromMat(A.mat, cdim, coords, cdist, peta, lsize, maxr, pbs, tol, &newmat) )
        PetscCLEAR(self.obj); self.mat = newmat
        return self

    def createIS(
        self,
        size: MatSizeSpec,
        LGMap lgmapr = None,
        LGMap lgmapc = None,
        comm: Comm | None = None,
        ) -> Self:
        """Create a `Type.IS` matrix representing globally unassembled operators.

        Collective.

        Parameters
        ----------
        size
            Matrix size.
        lgmapr
            Optional local-to-global mapping for the rows.
            If `None`, the local row space matches the global row space.
        lgmapc
            Optional local-to-global mapping for the columns.
            If `None`, the local column space matches the global column space.
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        petsc.MATIS

        """
        # communicator and sizes
        if comm is None and lgmapr is not None: comm = lgmapr.getComm()
        if comm is None and lgmapc is not None: comm = lgmapc.getComm()
        cdef PetscLGMap lgmr = NULL
        cdef PetscLGMap lgmc = NULL
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscInt rbs = 0, cbs = 0, m = 0, n = 0, M = 0, N = 0
        Mat_Sizes(size, None, &rbs, &cbs, &m, &n, &M, &N)
        Sys_Layout(ccomm, rbs, &m, &M)
        Sys_Layout(ccomm, cbs, &n, &N)
        # create matrix
        cdef PetscMat newmat = NULL
        cdef PetscInt bs = 1
        if rbs == cbs: bs = rbs
        if lgmapr is not None:
           lgmr = lgmapr.lgm
        if lgmapc is not None:
           lgmc = lgmapc.lgm
        CHKERR( MatCreateIS(ccomm, bs, m, n, M, N, lgmr, lgmc, &newmat) )
        PetscCLEAR(self.obj); self.mat = newmat
        return self

    def createPython(self, size: MatSizeSpec, context: Any = None, comm: Comm | None = None) -> Self:
        """Create a `Type.PYTHON` matrix.

        Collective.

        Parameters
        ----------
        size
            Matrix size.
        context
            An instance of the Python class implementing the required methods.
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        petsc_python_mat, setType, setPythonContext, Type.PYTHON

        """
        # communicator and sizes
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscInt rbs = 0, cbs = 0, m = 0, n = 0, M = 0, N = 0
        Mat_Sizes(size, None, &rbs, &cbs, &m, &n, &M, &N)
        Sys_Layout(ccomm, rbs, &m, &M)
        Sys_Layout(ccomm, cbs, &n, &N)
        # create matrix
        # FIXME: propagate block sizes?
        cdef PetscMat newmat = NULL
        CHKERR( MatCreate(ccomm, &newmat) )
        PetscCLEAR(self.obj); self.mat = newmat
        CHKERR( MatSetSizes(self.mat, m, n, M, N) )
        CHKERR( MatSetType(self.mat, MATPYTHON) )
        CHKERR( MatPythonSetContext(self.mat, <void*>context) )
        return self

    def setPythonContext(self, context: Any) -> None:
        """Set the instance of the Python class implementing the required Python methods.

        Not collective.

        See Also
        --------
        petsc_python_mat, getPythonContext

        """
        CHKERR( MatPythonSetContext(self.mat, <void*>context) )

    def getPythonContext(self) -> Any:
        """Return the instance of the Python class implementing the required Python methods.

        Not collective.

        See Also
        --------
        petsc_python_mat, setPythonContext

        """
        cdef void *context = NULL
        CHKERR( MatPythonGetContext(self.mat, &context) )
        if context == NULL: return None
        else: return <object> context

    def setPythonType(self, py_type: str) -> None:
        """Set the fully qualified Python name of the class to be used.

        Collective.

        See Also
        --------
        petsc_python_mat, setPythonContext, getPythonType
        petsc.MatPythonSetType

        """
        cdef const char *cval = NULL
        py_type = str2bytes(py_type, &cval)
        CHKERR( MatPythonSetType(self.mat, cval) )

    def getPythonType(self) -> str:
        """Return the fully qualified Python name of the class used by the matrix.

        Not collective.

        See Also
        --------
        petsc_python_mat, setPythonContext, setPythonType
        petsc.MatPythonGetType

        """
        cdef const char *cval = NULL
        CHKERR( MatPythonGetType(self.mat, &cval) )
        return bytes2str(cval)

    #

    def setOptionsPrefix(self, prefix: str) -> None:
        """Set the prefix used for searching for options in the database.

        Logically collective.

        See Also
        --------
        petsc_options, getOptionsPrefix, petsc.MatSetOptionsPrefix

        """
        cdef const char *cval = NULL
        prefix = str2bytes(prefix, &cval)
        CHKERR( MatSetOptionsPrefix(self.mat, cval) )

    def getOptionsPrefix(self) -> str:
        """Return the prefix used for searching for options in the database.

        Not collective.

        See Also
        --------
        petsc_options, setOptionsPrefix, petsc.MatGetOptionsPrefix

        """
        cdef const char *cval = NULL
        CHKERR( MatGetOptionsPrefix(self.mat, &cval) )
        return bytes2str(cval)

    def appendOptionsPrefix(self, prefix: str) -> None:
        """Append to the prefix used for searching for options in the database.

        Logically collective.

        See Also
        --------
        petsc_options, setOptionsPrefix, petsc.MatAppendOptionsPrefix

        """
        cdef const char *cval = NULL
        prefix = str2bytes(prefix, &cval)
        CHKERR( MatAppendOptionsPrefix(self.mat, cval) )

    def setFromOptions(self) -> None:
        """Configure the matrix from the options database.

        Collective.

        See Also
        --------
        petsc_options, petsc.MatSetFromOptions

        """
        CHKERR( MatSetFromOptions(self.mat) )

    def setUp(self) -> None:
        """Set up the internal data structures for using the matrix.

        Collective.

        See Also
        --------
        petsc.MatSetUp

        """
        CHKERR( MatSetUp(self.mat) )
        return self

    def setOption(self, option: Option, flag: bool) -> None:
        """Set option.

        Collective.

        See Also
        --------
        getOption, petsc.MatSetOption

        """
        CHKERR( MatSetOption(self.mat, option, flag) )

    def getOption(self, option: Option) -> bool:
        """Return the option value.

        Not collective.

        See Also
        --------
        setOption, petsc.MatGetOption

        """
        cdef PetscBool flag = PETSC_FALSE
        CHKERR( MatGetOption(self.mat, option, &flag) )
        return toBool(flag)

    def getType(self) -> str:
        """Return the type of the matrix.

        Not collective.

        See Also
        --------
        setType, Type, petsc.MatGetType

        """
        cdef PetscMatType cval = NULL
        CHKERR( MatGetType(self.mat, &cval) )
        return bytes2str(cval)

    def getSize(self) -> tuple[int, int]:
        """Return the global number of rows and columns.

        Not collective.

        See Also
        --------
        getLocalSize, getSizes, petsc.MatGetSize

        """
        cdef PetscInt M = 0, N = 0
        CHKERR( MatGetSize(self.mat, &M, &N) )
        return (toInt(M), toInt(N))

    def getLocalSize(self) -> tuple[int, int]:
        """Return the local number of rows and columns.

        Not collective.

        See Also
        --------
        getSize, petsc.MatGetLocalSize

        """
        cdef PetscInt m = 0, n = 0
        CHKERR( MatGetLocalSize(self.mat, &m, &n) )
        return (toInt(m), toInt(n))

    def getSizes(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """Return the tuple of 2-tuples of the type ``(local, global)`` for rows and columns.

        Not collective.

        See Also
        --------
        getLocalSize, getSize

        """
        cdef PetscInt m = 0, n = 0
        cdef PetscInt M = 0, N = 0
        CHKERR( MatGetLocalSize(self.mat, &m, &n) )
        CHKERR( MatGetSize(self.mat, &M, &N) )
        return ((toInt(m), toInt(M)), (toInt(n), toInt(N)))

    def getBlockSize(self) -> int:
        """Return the matrix block size.

        Not collective.

        See Also
        --------
        getBlockSize, petsc.MatGetBlockSize

        """
        cdef PetscInt bs = 0
        CHKERR( MatGetBlockSize(self.mat, &bs) )
        return toInt(bs)

    def getBlockSizes(self) -> tuple[int, int]:
        """Return the row and column block sizes.

        Not collective.

        See Also
        --------
        getBlockSize, petsc.MatGetBlockSizes

        """
        cdef PetscInt rbs = 0, cbs = 0
        CHKERR( MatGetBlockSizes(self.mat, &rbs, &cbs) )
        return (toInt(rbs), toInt(cbs))

    def getOwnershipRange(self) -> tuple[int, int]:
        """Return the locally owned range of rows.

        Not collective.

        See Also
        --------
        getOwnershipRanges, getOwnershipRangeColumn, petsc.MatGetOwnershipRange

        """
        cdef PetscInt ival1 = 0, ival2 = 0
        CHKERR( MatGetOwnershipRange(self.mat, &ival1, &ival2) )
        return (toInt(ival1), toInt(ival2))

    def getOwnershipRanges(self) -> ArrayInt:
        """Return the range of rows owned by each process.

        Not collective.

        The returned array is the result of exclusive scan of the local sizes.

        See Also
        --------
        getOwnershipRange, petsc.MatGetOwnershipRanges

        """
        cdef const PetscInt *rowrng = NULL
        CHKERR( MatGetOwnershipRanges(self.mat, &rowrng) )
        cdef MPI_Comm comm = MPI_COMM_NULL
        CHKERR( PetscObjectGetComm(<PetscObject>self.mat, &comm) )
        cdef int size = -1
        CHKERR( <PetscErrorCode>MPI_Comm_size(comm, &size) )
        return array_i(size+1, rowrng)

    def getOwnershipRangeColumn(self) -> tuple[int, int]:
        """Return the locally owned range of columns.

        Not collective.

        See Also
        --------
        getOwnershipRangesColumn, getOwnershipRange
        petsc.MatGetOwnershipRangeColumn

        """
        cdef PetscInt ival1 = 0, ival2 = 0
        CHKERR( MatGetOwnershipRangeColumn(self.mat, &ival1, &ival2) )
        return (toInt(ival1), toInt(ival2))

    def getOwnershipRangesColumn(self) -> ArrayInt:
        """Return the range of columns owned by each process.

        Not collective.

        See Also
        --------
        getOwnershipRangeColumn, petsc.MatGetOwnershipRangesColumn

        """
        cdef const PetscInt *colrng = NULL
        CHKERR( MatGetOwnershipRangesColumn(self.mat, &colrng) )
        cdef MPI_Comm comm = MPI_COMM_NULL
        CHKERR( PetscObjectGetComm(<PetscObject>self.mat, &comm) )
        cdef int size = -1
        CHKERR( <PetscErrorCode>MPI_Comm_size(comm, &size) )
        return array_i(size+1, colrng)

    def getOwnershipIS(self) -> tuple[IS, IS]:
        """Return the ranges of rows and columns owned by each process as index sets.

        Not collective.

        See Also
        --------
        getOwnershipRanges, getOwnershipRangesColumn, petsc.MatGetOwnershipIS

        """
        cdef IS rows = IS()
        cdef IS cols = IS()
        CHKERR( MatGetOwnershipIS(self.mat, &rows.iset, &cols.iset) )
        return (rows, cols)

    def getInfo(self, info: InfoType = None) -> dict[str, float]:
        """Return summary information.

        Collective.

        Parameters
        ----------
        info
            If `None`, it uses `InfoType.GLOBAL_SUM`.

        See Also
        --------
        petsc.MatInfo, petsc.MatGetInfo

        """
        cdef PetscMatInfoType itype = infotype(info)
        cdef PetscMatInfo cinfo
        CHKERR( MatGetInfo(self.mat, itype, &cinfo) )
        return cinfo

    def duplicate(self, copy: bool = False) -> Mat:
        """Return a clone of the matrix.

        Collective.

        Parameters
        ----------
        copy
            If `True`, it also copies the values.

        See Also
        --------
        petsc.MatDuplicate

        """
        cdef PetscMatDuplicateOption flag = MAT_DO_NOT_COPY_VALUES
        if copy: flag = MAT_COPY_VALUES
        if copy > MAT_COPY_VALUES: flag = MAT_SHARE_NONZERO_PATTERN
        cdef Mat mat = type(self)()
        CHKERR( MatDuplicate(self.mat, flag, &mat.mat) )
        return mat

    def copy(self, Mat result=None, structure: Structure | None = None) -> Mat:
        """Return a copy of the matrix.

        Collective.

        Parameters
        ----------
        result
            Optional return matrix. If `None`, it is internally created.
        structure
            The copy structure. Only relevant if ``result`` is not `None`.

        See Also
        --------
        petsc.MatCopy, petsc.MatDuplicate

        """
        cdef PetscMatDuplicateOption copy = MAT_COPY_VALUES
        cdef PetscMatStructure mstr = matstructure(structure)
        if result is None:
            result = type(self)()
        if result.mat == NULL:
            CHKERR( MatDuplicate(self.mat, copy, &result.mat) )
        else:
            CHKERR( MatCopy(self.mat, result.mat, mstr) )
        return result

    def load(self, Viewer viewer) -> Self:
        """Load a matrix.

        Collective.

        See Also
        --------
        petsc.MatLoad

        """
        cdef MPI_Comm comm = MPI_COMM_NULL
        cdef PetscObject obj = <PetscObject>(viewer.vwr)
        if self.mat == NULL:
            CHKERR( PetscObjectGetComm(obj, &comm) )
            CHKERR( MatCreate(comm, &self.mat) )
        CHKERR( MatLoad(self.mat, viewer.vwr) )
        return self

    def convert(self, mat_type: Type | str = None, Mat out=None) -> Mat:
        """Convert the matrix type.

        Collective.

        Parameters
        ----------
        mat_type
            The type of the new matrix. If `None` uses `Type.SAME`.
        out
            Optional return matrix. If `None`, inplace conversion is performed.
            Otherwise, the matrix is reused.

        See Also
        --------
        petsc.MatConvert

        """
        cdef PetscMatType mtype = MATSAME
        cdef PetscMatReuse reuse = MAT_INITIAL_MATRIX
        mat_type = str2bytes(mat_type, &mtype)
        if mtype == NULL: mtype = MATSAME
        if out is None: out = self
        if out.mat == self.mat:
            reuse = MAT_INPLACE_MATRIX
        elif out.mat == NULL:
            reuse = MAT_INITIAL_MATRIX
        else:
            reuse = MAT_REUSE_MATRIX
        CHKERR( MatConvert(self.mat, mtype, reuse, &out.mat) )
        return out

    def transpose(self, Mat out=None) -> Mat:
        """Return the transposed matrix.

        Collective.

        Parameters
        ----------
        out
            Optional return matrix. If `None`, inplace transposition is performed.
            Otherwise, the matrix is reused.

        See Also
        --------
        petsc.MatTranspose

        """
        cdef PetscMatReuse reuse = MAT_INITIAL_MATRIX
        if out is None: out = self
        if out.mat == self.mat:
            reuse = MAT_INPLACE_MATRIX
        elif out.mat == NULL:
            reuse = MAT_INITIAL_MATRIX
        else:
            reuse = MAT_REUSE_MATRIX
        CHKERR( MatTranspose(self.mat, reuse, &out.mat) )
        return out

    def setTransposePrecursor(self, Mat out) -> None:
        """Set transpose precursor.

        See Also
        --------
        petsc.MatTransposeSetPrecursor

        """
        CHKERR( MatTransposeSetPrecursor(self.mat, out.mat) )

    def hermitianTranspose(self, Mat out=None) -> Mat:
        """Return the transposed Hermitian matrix.

        Collective.

        Parameters
        ----------
        out
            Optional return matrix. If `None`, inplace transposition is performed.
            Otherwise, the matrix is reused.

        See Also
        --------
        petsc.MatHermitianTranspose

        """
        cdef PetscMatReuse reuse = MAT_INITIAL_MATRIX
        if out is None: out = self
        if out.mat == self.mat:
            reuse = MAT_INPLACE_MATRIX
        elif out.mat == NULL:
            reuse = MAT_INITIAL_MATRIX
        else:
            reuse = MAT_REUSE_MATRIX
        CHKERR( MatHermitianTranspose(self.mat, reuse, &out.mat) )
        return out

    def realPart(self, Mat out=None) -> Mat:
        """Return the real part of the matrix.

        Collective.

        Parameters
        ----------
        out
            Optional return matrix. If `None`, the operation is performed in-place.
            Otherwise, the operation is performed on ``out``.

        See Also
        --------
        imagPart, conjugate, petsc.MatRealPart

        """
        if out is None:
            out = self
        elif out.mat == NULL:
            CHKERR( MatDuplicate(self.mat, MAT_COPY_VALUES, &out.mat) )
        CHKERR( MatRealPart(out.mat) )
        return out

    def imagPart(self, Mat out=None) -> Mat:
        """Return the imaginary part of the matrix.

        Collective.

        Parameters
        ----------
        out
            Optional return matrix. If `None`, the operation is performed in-place.
            Otherwise, the operation is performed on ``out``.

        See Also
        --------
        realPart, conjugate, petsc.MatImaginaryPart

        """
        if out is None:
            out = self
        elif out.mat == NULL:
            CHKERR( MatDuplicate(self.mat, MAT_COPY_VALUES, &out.mat) )
        CHKERR( MatImaginaryPart(out.mat) )
        return out

    def conjugate(self, Mat out=None) -> Mat:
        """Return the conjugate matrix.

        Collective.

        Parameters
        ----------
        out
            Optional return matrix. If `None`, the operation is performed in-place.
            Otherwise, the operation is performed on ``out``.

        See Also
        --------
        realPart, imagPart, petsc.MatConjugate

        """
        if out is None:
            out = self
        elif out.mat == NULL:
            CHKERR( MatDuplicate(self.mat, MAT_COPY_VALUES, &out.mat) )
        CHKERR( MatConjugate(out.mat) )
        return out

    def permute(self, IS row, IS col) -> Mat:
        """Return the permuted matrix.

        Collective.

        Parameters
        ----------
        row
            Row permutation.
        col
            Column permutation.

        See Also
        --------
        petsc.MatPermute

        """
        cdef Mat mat = Mat()
        CHKERR( MatPermute(self.mat, row.iset, col.iset, &mat.mat) )
        return mat

    def equal(self, Mat mat) -> bool:
        """Return the result of matrix comparison.

        Collective.

        See Also
        --------
        petsc.MatEqual

        """
        cdef PetscBool flag = PETSC_FALSE
        CHKERR( MatEqual(self.mat, mat.mat, &flag) )
        return toBool(flag)

    def isTranspose(self, Mat mat=None, tol: float = 0) -> bool:
        """Return the result of matrix comparison with transposition.

        Collective.

        Parameters
        ----------
        mat
            Matrix to compare against. Uses ``self`` if `None`.
        tol
            Tolerance for comparison.

        See Also
        --------
        petsc.MatIsTranspose

        """
        if mat is None: mat = self
        cdef PetscReal rval = asReal(tol)
        cdef PetscBool flag = PETSC_FALSE
        CHKERR( MatIsTranspose(self.mat, mat.mat, rval, &flag) )
        return toBool(flag)

    def isSymmetric(self, tol: float = 0) -> bool:
        """Return the boolean indicating if the matrix is symmetric.

        Collective.

        Parameters
        ----------
        tol
            Tolerance for comparison.

        See Also
        --------
        petsc.MatIsSymmetric

        """
        cdef PetscReal rval = asReal(tol)
        cdef PetscBool flag = PETSC_FALSE
        CHKERR( MatIsSymmetric(self.mat, rval, &flag) )
        return toBool(flag)

    def isSymmetricKnown(self) -> tuple[bool, bool]:
        """Return the 2-tuple indicating if the matrix is known to be symmetric.

        Not collective.

        See Also
        --------
        petsc.MatIsSymmetricKnown

        """
        cdef PetscBool flag1 = PETSC_FALSE
        cdef PetscBool flag2 = PETSC_FALSE
        CHKERR( MatIsSymmetricKnown(self.mat, &flag1, &flag2) )
        return (toBool(flag1), toBool(flag2))

    def isHermitian(self, tol: float = 0) -> bool:
        """Return the boolean indicating if the matrix is Hermitian.

        Collective.

        Parameters
        ----------
        tol
            Tolerance for comparison.

        See Also
        --------
        petsc.MatIsHermitian

        """
        cdef PetscReal rval = asReal(tol)
        cdef PetscBool flag = PETSC_FALSE
        CHKERR( MatIsHermitian(self.mat, rval, &flag) )
        return toBool(flag)

    def isHermitianKnown(self) -> tuple[bool, bool]:
        """Return the 2-tuple indicating if the matrix is known to be Hermitian.

        Not collective.

        See Also
        --------
        petsc.MatIsHermitianKnown

        """
        cdef PetscBool flag1 = PETSC_FALSE
        cdef PetscBool flag2 = PETSC_FALSE
        CHKERR( MatIsHermitianKnown(self.mat, &flag1, &flag2) )
        return (toBool(flag1), toBool(flag2))

    def isStructurallySymmetric(self) -> bool:
        """Return the boolean indicating if the matrix is structurally symmetric."""
        cdef PetscBool flag = PETSC_FALSE
        CHKERR( MatIsStructurallySymmetric(self.mat, &flag) )
        return toBool(flag)

    def zeroEntries(self) -> None:
        """Zero the entries of the matrix.

        Collective.

        See Also
        --------
        petsc.MatZeroEntries

        """
        CHKERR( MatZeroEntries(self.mat) )

    def getValue(self, row, col) -> Scalar:
        """Return the value in the (row,col) position.

        Not collective.

        See Also
        --------
        petsc.MatGetValues

        """
        cdef PetscInt    ival1 = asInt(row)
        cdef PetscInt    ival2 = asInt(col)
        cdef PetscScalar sval  = 0
        CHKERR( MatGetValues(self.mat, 1, &ival1, 1, &ival2, &sval) )
        return toScalar(sval)

    def getValues(self, rows: Sequence[int], cols: Sequence[int], values: ArrayScalar = None) -> ArrayScalar:
        """Return the values in the ``zip(rows,cols)`` positions.

        Not collective.

        Parameters
        ----------
        rows
            Row indices.
        cols
            Column indices.
        values
            Optional array where to store the values.

        See Also
        --------
        petsc.MatGetValues

        """
        return matgetvalues(self.mat, rows, cols, values)

    def getValuesCSR(self) -> tuple[ArrayInt, ArrayInt, ArrayScalar]:
        """Return the CSR representation of the local part of the matrix.

        Not collective.

        See Also
        --------
        petsc.MatGetRow

        """
        # row ownership
        cdef PetscInt rstart=0, rend=0, nrows=0
        CHKERR( MatGetOwnershipRange(self.mat, &rstart, &rend) )
        nrows = rend - rstart
        # first pass: row pointer array
        cdef PetscInt *AI = NULL
        cdef ndarray ai = oarray_i(empty_i(nrows+1), NULL, &AI)
        cdef PetscInt irow=0, ncols=0
        AI[0] = 0
        for irow from 0 <= irow < nrows:
            CHKERR( MatGetRow(self.mat, irow+rstart, &ncols, NULL, NULL) )
            AI[irow+1] = AI[irow] + ncols
            CHKERR( MatRestoreRow(self.mat, irow+rstart, &ncols, NULL, NULL) )
        # second pass: column indices and values
        cdef PetscInt *AJ = NULL
        cdef ndarray aj = oarray_i(empty_i(AI[nrows]), NULL, &AJ)
        cdef PetscScalar *AV = NULL
        cdef ndarray av = oarray_s(empty_s(AI[nrows]), NULL, &AV)
        cdef const PetscInt *cols = NULL
        cdef const PetscScalar *vals = NULL
        for irow from 0 <= irow < nrows:
            CHKERR( MatGetRow(self.mat, irow+rstart, &ncols, &cols, &vals) )
            CHKERR( PetscMemcpy(AJ+AI[irow], cols, <size_t>ncols*sizeof(PetscInt)) )
            CHKERR( PetscMemcpy(AV+AI[irow], vals, <size_t>ncols*sizeof(PetscScalar)) )
            CHKERR( MatRestoreRow(self.mat, irow+rstart, &ncols, &cols, &vals) )
        #
        return (ai, aj, av)

    def getRow(self, row: int) -> tuple[ArrayInt, ArrayScalar]:
        """Return the column indices and values for the requested row.

        Not collective.

        See Also
        --------
        petsc.MatGetRow

        """
        cdef PetscInt irow = asInt(row)
        cdef PetscInt ncols = 0
        cdef const PetscInt *icols=NULL
        cdef const PetscScalar *svals=NULL
        CHKERR( MatGetRow(self.mat, irow, &ncols, &icols, &svals) )
        cdef object cols = array_i(ncols, icols)
        cdef object vals = array_s(ncols, svals)
        CHKERR( MatRestoreRow(self.mat, irow, &ncols, &icols, &svals) )
        return (cols, vals)

    def getRowIJ(self, symmetric: bool = False, compressed: bool = False) -> tuple[ArrayInt, ArrayInt]:
        """Return the CSR representation of the local sparsity pattern.

        Collective.

        Parameters
        ----------
        symmetric
            If `True`, return the symmetrized graph.
        compressed
            If `True`, return the compressed graph.

        See Also
        --------
        petsc.MatGetRowIJ

        """
        cdef PetscInt shift=0
        cdef PetscBool symm=symmetric
        cdef PetscBool bcmp=compressed
        cdef PetscInt n=0
        cdef const PetscInt *ia=NULL
        cdef const PetscInt *ja=NULL
        cdef PetscBool done=PETSC_FALSE
        CHKERR( MatGetRowIJ(self.mat, shift, symm, bcmp, &n, &ia, &ja, &done) )
        cdef object ai=None, aj=None
        if done != PETSC_FALSE: ai = array_i(  n+1, ia)
        if done != PETSC_FALSE: aj = array_i(ia[n], ja)
        CHKERR( MatRestoreRowIJ(self.mat, shift, symm, bcmp, &n, &ia, &ja, &done) )
        return (ai, aj)

    def getColumnIJ(self, symmetric: bool = False, compressed: bool = False) -> tuple[ArrayInt, ArrayInt]:
        """Return the CSC representation of the local sparsity pattern.

        Collective.

        Parameters
        ----------
        symmetric
            If `True`, return the symmetrized graph.
        compressed
            If `True`, return the compressed graph.

        See Also
        --------
        petsc.MatGetRowIJ

        """
        cdef PetscInt shift=0
        cdef PetscBool symm=symmetric, bcmp=compressed
        cdef PetscInt n=0
        cdef const PetscInt *ia=NULL
        cdef const PetscInt *ja=NULL
        cdef PetscBool done=PETSC_FALSE
        CHKERR( MatGetColumnIJ(self.mat, shift, symm, bcmp, &n, &ia, &ja, &done) )
        cdef object ai=None, aj=None
        if done != PETSC_FALSE: ai = array_i(  n+1, ia)
        if done != PETSC_FALSE: aj = array_i(ia[n], ja)
        CHKERR( MatRestoreColumnIJ(self.mat, shift, symm, bcmp, &n, &ia, &ja, &done) )
        return (ai, aj)

    def setValue(
        self,
        row: int,
        col: int,
        value: Scalar,
        addv: InsertModeSpec = None,
        ) -> None:
        """Set or add a value to the ``(row, col)`` entry of the matrix.

        Not collective.

        Parameters
        ----------
        row
            Row index.
        col
            Column index.
        value
            The scalar value.
        addv
            Insertion mode.

        See Also
        --------
        petsc.MatSetValues

        """
        cdef PetscInt    ival1 = asInt(row)
        cdef PetscInt    ival2 = asInt(col)
        cdef PetscScalar sval  = asScalar(value)
        cdef PetscInsertMode caddv = insertmode(addv)
        CHKERR( MatSetValues(self.mat, 1, &ival1, 1, &ival2, &sval, caddv) )

    def setValues(
        self,
        rows: Sequence[int],
        cols: Sequence[int],
        values: Sequence[Scalar],
        addv: InsertModeSpec = None,
        ) -> None:
        """Set or add values to the rows ⊗ col entries of the matrix.

        Not collective.

        Parameters
        ----------
        rows
            Row indices.
        cols
            Column indices.
        values
            The scalar values. A sequence of length at least ``len(rows) * len(cols)``.
        addv
            Insertion mode.

        See Also
        --------
        petsc.MatSetValues

        """
        matsetvalues(self.mat, rows, cols, values, addv, 0, 0)

    def setValuesRCV(self, R, C, V, addv=None) -> None:
        """Undocumented."""
        matsetvalues_rcv(self.mat, R, C, V, addv, 0, 0)

    def setValuesIJV(
        self,
        I: Sequence[int],
        J: Sequence[int],
        V: Sequence[Scalar],
        addv: InsertModeSpec = None,
        rowmap: Sequence[int] = None,
        ) -> None:
        """Set or add a subset of values stored in CSR format.

        Not collective.

        Parameters
        ----------
        I
            Row pointers.
        J
            Column indices.
        V
            The scalar values.
        addv
            Insertion mode.
        rowmap
            Optional iterable indicating which row to insert.

        See Also
        --------
        petsc.MatSetValues

        """
        matsetvalues_ijv(self.mat, I, J, V, addv, rowmap, 0, 0)

    def setValuesCSR(
        self,
        I: Sequence[int],
        J: Sequence[int],
        V: Sequence[Scalar],
        addv: InsertModeSpec = None,
        ) -> None:
        """Set or add values stored in CSR format.

        Not collective.

        Parameters
        ----------
        I
            Row pointers.
        J
            Column indices.
        V
            The scalar values.
        addv
            Insertion mode.

        See Also
        --------
        petsc.MatSetValues

        """
        matsetvalues_csr(self.mat, I, J, V, addv, 0, 0)

    def setValuesBlocked(
        self,
        rows: Sequence[int],
        cols: Sequence[int],
        values: Sequence[Scalar],
        addv: InsertModeSpec = None,
        ) -> None:
        """Set or add values to the rows ⊗ col block entries of the matrix.

        Not collective.

        Parameters
        ----------
        rows
            Block row indices.
        cols
            Block column indices.
        values
            The scalar values. A sequence of length at least ``len(rows) * len(cols) * bs * bs``,
            where ``bs`` is the block size of the matrix.
        addv
            Insertion mode.

        See Also
        --------
        petsc.MatSetValuesBlocked

        """
        matsetvalues(self.mat, rows, cols, values, addv, 1, 0)

    def setValuesBlockedRCV(self, R, C, V, addv=None) -> None:
        """Undocumented."""
        matsetvalues_rcv(self.mat, R, C, V, addv, 1, 0)

    def setValuesBlockedIJV(
        self,
        I: Sequence[int],
        J: Sequence[int],
        V: Sequence[Scalar],
        addv: InsertModeSpec = None,
        rowmap: Sequence[int] = None,
        ) -> None:
        """Set or add a subset of values stored in block CSR format.

        Not collective.

        Parameters
        ----------
        I
            Block row pointers.
        J
            Block column indices.
        V
            The scalar values.
        addv
            Insertion mode.
        rowmap
            Optional iterable indicating which block row to insert.

        See Also
        --------
        petsc.MatSetValuesBlocked

        """
        matsetvalues_ijv(self.mat, I, J, V, addv, rowmap, 1, 0)

    def setValuesBlockedCSR(
        self,
        I: Sequence[int],
        J: Sequence[int],
        V: Sequence[Scalar],
        addv: InsertModeSpec = None,
        ) -> None:
        """Set or add values stored in block CSR format.

        Not collective.

        Parameters
        ----------
        I
            Block row pointers.
        J
            Block column indices.
        V
            The scalar values.
        addv
            Insertion mode.

        See Also
        --------
        petsc.MatSetValuesBlocked

        """
        matsetvalues_csr(self.mat, I, J, V, addv, 1, 0)

    def setLGMap(self, LGMap rmap, LGMap cmap=None) -> None:
        """Set the local-to-global mappings.

        Collective.

        Parameters
        ----------
        rmap
            Row mapping.
        cmap
            Column mapping. If `None`, ``cmap = rmap``.

        See Also
        --------
        getLGMap, petsc.MatSetLocalToGlobalMapping

        """
        if cmap is None: cmap = rmap
        CHKERR( MatSetLocalToGlobalMapping(self.mat, rmap.lgm, cmap.lgm) )

    def getLGMap(self) -> tuple[LGMap, LGMap]:
        """Return the local-to-global mappings.

        Not collective.

        See Also
        --------
        setLGMap, petsc.MatGetLocalToGlobalMapping

        """
        cdef LGMap cmap = LGMap()
        cdef LGMap rmap = LGMap()
        CHKERR( MatGetLocalToGlobalMapping(self.mat, &rmap.lgm, &cmap.lgm) )
        PetscINCREF(cmap.obj)
        PetscINCREF(rmap.obj)
        return (rmap, cmap)

    def setValueLocal(
        self,
        row: int,
        col: int,
        value: Scalar,
        addv: InsertModeSpec = None,
        ) -> None:
        """Set or add a value to the ``(row, col)`` entry of the matrix in local ordering.

        Not collective.

        Parameters
        ----------
        row
            Local row index.
        col
            Local column index.
        value
            The scalar value.
        addv
            Insertion mode.

        See Also
        --------
        petsc.MatSetValuesLocal

        """
        cdef PetscInt    ival1 = asInt(row)
        cdef PetscInt    ival2 = asInt(col)
        cdef PetscScalar sval  = asScalar(value)
        cdef PetscInsertMode caddv = insertmode(addv)
        CHKERR( MatSetValuesLocal(
                self.mat, 1, &ival1, 1, &ival2, &sval, caddv) )

    def setValuesLocal(
        self,
        rows: Sequence[int],
        cols: Sequence[int],
        values: Sequence[Scalar],
        addv: InsertModeSpec = None,
        ) -> None:
        """Set or add values to the rows ⊗ col entries of the matrix in local ordering.

        Not collective.

        Parameters
        ----------
        rows
            Local row indices.
        cols
            Local column indices.
        values
            The scalar values. A sequence of length at least ``len(rows) * len(cols)``.
        addv
            Insertion mode.

        See Also
        --------
        petsc.MatSetValuesLocal

        """
        matsetvalues(self.mat, rows, cols, values, addv, 0, 1)

    def setValuesLocalRCV(self, R, C, V, addv=None) -> None:
        """Undocumented."""
        matsetvalues_rcv(self.mat, R, C, V, addv, 0, 1)

    def setValuesLocalIJV(
        self,
        I: Sequence[int],
        J: Sequence[int],
        V: Sequence[Scalar],
        addv: InsertModeSpec = None,
        rowmap: Sequence[int] = None,
        ) -> None:
        """Set or add a subset of values stored in CSR format.

        Not collective.

        Parameters
        ----------
        I
            Row pointers.
        J
            Local column indices.
        V
            The scalar values.
        addv
            Insertion mode.
        rowmap
            Optional iterable indicating which row to insert.

        See Also
        --------
        petsc.MatSetValuesLocal

        """
        matsetvalues_ijv(self.mat, I, J, V, addv, rowmap, 0, 1)

    def setValuesLocalCSR(
        self,
        I: Sequence[int],
        J: Sequence[int],
        V: Sequence[Scalar],
        addv: InsertModeSpec = None,
        ) -> None:
        """Set or add values stored in CSR format.

        Not collective.

        Parameters
        ----------
        I
            Row pointers.
        J
            Local column indices.
        V
            The scalar values.
        addv
            Insertion mode.

        See Also
        --------
        petsc.MatSetValuesLocal

        """
        matsetvalues_csr(self.mat, I, J, V, addv, 0, 1)

    def setValuesBlockedLocal(
        self,
        rows: Sequence[int],
        cols: Sequence[int],
        values: Sequence[Scalar],
        addv: InsertModeSpec = None,
        ) -> None:
        """Set or add values to the rows ⊗ col block entries of the matrix in local ordering.

        Not collective.

        Parameters
        ----------
        rows
            Local block row indices.
        cols
            Local block column indices.
        values
            The scalar values. A sequence of length at least ``len(rows) * len(cols) * bs * bs``,
            where ``bs`` is the block size of the matrix.
        addv
            Insertion mode.

        See Also
        --------
        petsc.MatSetValuesBlockedLocal

        """
        matsetvalues(self.mat, rows, cols, values, addv, 1, 1)

    def setValuesBlockedLocalRCV(self, R, C, V, addv=None) -> None:
        """Undocumented."""
        matsetvalues_rcv(self.mat, R, C, V, addv, 1, 1)

    def setValuesBlockedLocalIJV(
        self,
        I: Sequence[int],
        J: Sequence[int],
        V: Sequence[Scalar],
        addv: InsertModeSpec = None,
        rowmap: Sequence[int] = None,
        ) -> None:
        """Set or add a subset of values stored in block CSR format.

        Not collective.

        Parameters
        ----------
        I
            Block row pointers.
        J
            Local block column indices.
        V
            The scalar values.
        addv
            Insertion mode.
        rowmap
            Optional iterable indicating which block row to insert.

        See Also
        --------
        petsc.MatSetValuesBlockedLocal

        """
        matsetvalues_ijv(self.mat, I, J, V, addv, rowmap, 1, 1)

    def setValuesBlockedLocalCSR(
        self,
        I: Sequence[int],
        J: Sequence[int],
        V: Sequence[Scalar],
        addv: InsertModeSpec = None,
        ) -> None:
        """Set or add values stored in block CSR format.

        Not collective.

        Parameters
        ----------
        I
            Block row pointers.
        J
            Local block column indices.
        V
            The scalar values.
        addv
            Insertion mode.

        See Also
        --------
        petsc.MatSetValuesBlockedLocal

        """
        matsetvalues_csr(self.mat, I, J, V, addv, 1, 1)

    #

    Stencil = MatStencil

    def setStencil(self, dims: DimsSpec, starts: DimsSpec | None = None, dof: int = 1) -> None:
        """Set matrix stencil.

        Not collective.

        See Also
        --------
        petsc.MatSetStencil

        """
        cdef PetscInt ndim, ndof
        cdef PetscInt cdims[3], cstarts[3]
        cdims[0] = cdims[1] = cdims[2] = 1
        cstarts[0] = cstarts[1] = cstarts[2] = 0
        ndim = asDims(dims, &cdims[0], &cdims[1], &cdims[2])
        ndof = asInt(dof)
        if starts is not None:
            asDims(dims, &cstarts[0], &cstarts[1], &cstarts[2])
        CHKERR( MatSetStencil(self.mat, ndim, cdims, cstarts, ndof) )

    def setValueStencil(
        self,
        MatStencil row: Stencil,
        MatStencil col: Stencil,
        value: Sequence[Scalar],
        addv: InsertModeSpec = None,
        ) -> None:
        """Set or add a value to row and col stencil.

        Not collective.

        Parameters
        ----------
        row
            Row stencil.
        col
            Column stencil.
        value
            The scalar values.
        addv
            Insertion mode.

        See Also
        --------
        petsc.MatSetValuesStencil

        """
        cdef MatStencil r = row, c = col
        cdef PetscInsertMode im = insertmode(addv)
        matsetvaluestencil(self.mat, r, c, value, im, 0)

    def setValueStagStencil(self, row, col, value, addv=None) -> None:
        """Not implemented."""
        raise NotImplementedError

    def setValueBlockedStencil(
        self,
        row: Stencil,
        col: Stencil,
        value: Sequence[Scalar],
        addv: InsertModeSpec = None,
        ) -> None:
        """Set or add a block values to row and col stencil.

        Not collective.

        Parameters
        ----------
        row
            Row stencil.
        col
            Column stencil.
        value
            The scalar values.
        addv
            Insertion mode.

        See Also
        --------
        petsc.MatSetValuesBlockedStencil

        """
        cdef MatStencil r = row, c = col
        cdef PetscInsertMode im = insertmode(addv)
        matsetvaluestencil(self.mat, r, c, value, im, 1)

    def setValueBlockedStagStencil(self, row, col, value, addv=None) -> None:
        """Not implemented."""
        raise NotImplementedError

    def zeroRows(self, rows: IS | Sequence[int], diag: Scalar = 1, Vec x=None, Vec b=None) -> None:
        """Zero selected rows of the matrix.

        Collective.

        Parameters
        ----------
        rows
            Row indices to be zeroed.
        diag
            Scalar value to be inserted into the diagonal.
        x
            Optional solution vector to be modified for zeroed rows.
        b
            Optional right-hand side vector to be modified.
            It will be adjusted with provided solution entries.

        See Also
        --------
        zeroRowsLocal, petsc.MatZeroRows, petsc.MatZeroRowsIS

        """
        cdef PetscInt ni=0, *i=NULL
        cdef PetscScalar sval = asScalar(diag)
        cdef PetscVec xvec=NULL, bvec=NULL
        if x is not None: xvec = x.vec
        if b is not None: bvec = b.vec
        if isinstance(rows, IS):
            CHKERR( MatZeroRowsIS(self.mat, (<IS>rows).iset, sval, xvec, bvec) )
        else:
            rows = iarray_i(rows, &ni, &i)
            CHKERR( MatZeroRows(self.mat, ni, i, sval, xvec, bvec) )

    def zeroRowsLocal(self, rows: IS | Sequence[int], diag: Scalar = 1, Vec x=None, Vec b=None) -> None:
        """Zero selected rows of the matrix in local ordering.

        Collective.

        Parameters
        ----------
        rows
            Local row indices to be zeroed.
        diag
            Scalar value to be inserted into the diagonal.
        x
            Optional solution vector to be modified for zeroed rows.
        b
            Optional right-hand side vector to be modified.
            It will be adjusted with provided solution entries.

        See Also
        --------
        zeroRows, petsc.MatZeroRowsLocal, petsc.MatZeroRowsLocalIS

        """
        cdef PetscInt ni=0, *i=NULL
        cdef PetscScalar sval = asScalar(diag)
        cdef PetscVec xvec=NULL, bvec=NULL
        if x is not None: xvec = x.vec
        if b is not None: bvec = b.vec
        if isinstance(rows, IS):
            CHKERR( MatZeroRowsLocalIS(self.mat, (<IS>rows).iset, sval, xvec, bvec) )
        else:
            rows = iarray_i(rows, &ni, &i)
            CHKERR( MatZeroRowsLocal(self.mat, ni, i, sval, xvec, bvec) )

    def zeroRowsColumns(self, rows: IS | Sequence[int], diag: Scalar = 1, Vec x=None, Vec b=None) -> None:
        """Zero selected rows and columns of the matrix.

        Collective.

        Parameters
        ----------
        rows
            Row/column indices to be zeroed.
        diag
            Scalar value to be inserted into the diagonal.
        x
            Optional solution vector to be modified for zeroed rows.
        b
            Optional right-hand side vector to be modified.
            It will be adjusted with provided solution entries.

        See Also
        --------
        zeroRowsColumnsLocal, zeroRows, petsc.MatZeroRowsColumns
        petsc.MatZeroRowsColumnsIS

        """
        cdef PetscInt ni=0, *i=NULL
        cdef PetscScalar sval = asScalar(diag)
        cdef PetscVec xvec=NULL, bvec=NULL
        if x is not None: xvec = x.vec
        if b is not None: bvec = b.vec
        if isinstance(rows, IS):
            CHKERR( MatZeroRowsColumnsIS(self.mat, (<IS>rows).iset, sval, xvec, bvec) )
        else:
            rows = iarray_i(rows, &ni, &i)
            CHKERR( MatZeroRowsColumns(self.mat, ni, i, sval, xvec, bvec) )

    def zeroRowsColumnsLocal(self, rows: IS | Sequence[int], diag: Scalar = 1, Vec x=None, Vec b=None) -> None:
        """Zero selected rows and columns of the matrix in local ordering.

        Collective.

        Parameters
        ----------
        rows
            Local row/column indices to be zeroed.
        diag
            Scalar value to be inserted into the diagonal.
        x
            Optional solution vector to be modified for zeroed rows.
        b
            Optional right-hand side vector to be modified.
            It will be adjusted with provided solution entries.

        See Also
        --------
        zeroRowsLocal, zeroRowsColumns, petsc.MatZeroRowsColumnsLocal
        petsc.MatZeroRowsColumnsLocalIS

        """
        cdef PetscInt ni=0, *i=NULL
        cdef PetscScalar sval = asScalar(diag)
        cdef PetscVec xvec=NULL, bvec=NULL
        if x is not None: xvec = x.vec
        if b is not None: bvec = b.vec
        if isinstance(rows, IS):
            CHKERR( MatZeroRowsColumnsLocalIS(self.mat, (<IS>rows).iset, sval, xvec, bvec) )
        else:
            rows = iarray_i(rows, &ni, &i)
            CHKERR( MatZeroRowsColumnsLocal(self.mat, ni, i, sval, xvec, bvec) )

    def zeroRowsColumnsStencil(self, rows: Sequence[Stencil], diag: Scalar = 1, Vec x=None, Vec b=None) -> None:
        """Zero selected rows and columns of the matrix.

        Collective.

        Parameters
        ----------
        rows
            Iterable of stencil rows and columns.
        diag
            Scalar value to be inserted into the diagonal.
        x
            Optional solution vector to be modified for zeroed rows.
        b
            Optional right-hand side vector to be modified.
            It will be adjusted with provided solution entries.

        See Also
        --------
        zeroRowsLocal, zeroRowsColumns, petsc.MatZeroRowsColumnsStencil

        """
        cdef PetscScalar sval = asScalar(diag)
        cdef PetscInt nrows = asInt(len(rows))
        cdef PetscMatStencil st
        cdef MatStencil r
        cdef PetscMatStencil *crows = NULL
        CHKERR( PetscMalloc(<size_t>(nrows+1)*sizeof(st), &crows) )
        for i in range(nrows):
            r = rows[i]
            crows[i] = r.stencil
        cdef PetscVec xvec = NULL, bvec = NULL
        if x is not None: xvec = x.vec
        if b is not None: bvec = b.vec
        CHKERR( MatZeroRowsColumnsStencil(self.mat, nrows, crows, sval, xvec, bvec) )
        CHKERR( PetscFree( crows ) )

    def storeValues(self) -> None:
        """Stash a copy of the matrix values.

        Collective.

        See Also
        --------
        retrieveValues, petsc.MatStoreValues

        """
        CHKERR( MatStoreValues(self.mat) )

    def retrieveValues(self) -> None:
        """Retrieve a copy of the matrix values previously stored with `storeValues`.

        Collective.

        See Also
        --------
        storeValues, petsc.MatRetrieveValues

        """
        CHKERR( MatRetrieveValues(self.mat) )

    def assemblyBegin(self, assembly: MatAssemblySpec = None) -> None:
        """Begin an assembling stage of the matrix.

        Collective.

        Parameters
        ----------
        assembly
            The assembly type.

        See Also
        --------
        assemblyEnd, assemble, petsc.MatAssemblyBegin

        """
        cdef PetscMatAssemblyType flag = assemblytype(assembly)
        CHKERR( MatAssemblyBegin(self.mat, flag) )

    def assemblyEnd(self, assembly: MatAssemblySpec = None) -> None:
        """Complete an assembling stage of the matrix initiated with `assemblyBegin`.

        Collective.

        Parameters
        ----------
        assembly
            The assembly type.

        See Also
        --------
        assemblyBegin, assemble, petsc.MatAssemblyEnd

        """
        cdef PetscMatAssemblyType flag = assemblytype(assembly)
        CHKERR( MatAssemblyEnd(self.mat, flag) )

    def assemble(self, assembly: MatAssemblySpec = None) -> None:
        """Assemble the matrix.

        Collective.

        Parameters
        ----------
        assembly
            The assembly type.

        See Also
        --------
        assemblyBegin, assemblyEnd

        """
        cdef PetscMatAssemblyType flag = assemblytype(assembly)
        CHKERR( MatAssemblyBegin(self.mat, flag) )
        CHKERR( MatAssemblyEnd(self.mat, flag) )

    def isAssembled(self) -> bool:
        """The boolean flag indicating if the matrix is assembled.

        Not collective.

        See Also
        --------
        assemble, petsc.MatAssembled

        """
        cdef PetscBool flag = PETSC_FALSE
        CHKERR( MatAssembled(self.mat, &flag) )
        return toBool(flag)

    def findZeroRows(self) -> IS:
        """Return the index set of empty rows.

        Collective.

        See Also
        --------
        petsc.MatFindZeroRows

        """
        cdef IS zerorows = IS()
        CHKERR( MatFindZeroRows(self.mat, &zerorows.iset) )
        return zerorows

    def createVecs(
        self,
        side: Literal['r', 'R', 'right', 'Right', 'RIGHT', 'l', 'L', 'left', 'Left', 'LEFT'] | None = None,
        ) -> Vec | tuple[Vec, Vec]:
        """Return vectors that can be used in matrix vector products.

        Collective.

        Parameters
        ----------
        side
            If `None` returns a 2-tuple of vectors ``(right, left)``.
            Otherwise it just return a left or right vector.

        Notes
        -----
        ``right`` vectors are vectors in the column space of the matrix.
        ``left`` vectors are vectors in the row space of the matrix.

        See Also
        --------
        createVecLeft, createVecRight, petsc.MatCreateVecs

        """
        cdef Vec vecr, vecl
        if side is None:
            vecr = Vec(); vecl = Vec();
            CHKERR( MatCreateVecs(self.mat, &vecr.vec, &vecl.vec) )
            return (vecr, vecl)
        elif side in ('r', 'R', 'right', 'Right', 'RIGHT'):
            vecr = Vec()
            CHKERR( MatCreateVecs(self.mat, &vecr.vec, NULL) )
            return vecr
        elif side in ('l', 'L', 'left',  'Left', 'LEFT'):
            vecl = Vec()
            CHKERR( MatCreateVecs(self.mat, NULL, &vecl.vec) )
            return vecl
        else:
            raise ValueError("side '%r' not understood" % side)

    def createVecRight(self) -> Vec:
        """Return a right vector, a vector that the matrix can be multiplied against.

        Collective.

        See Also
        --------
        createVecs, createVecLeft, petsc.MatCreateVecs

        """
        cdef Vec vecr = Vec()
        CHKERR( MatCreateVecs(self.mat, &vecr.vec, NULL) )
        return vecr

    def createVecLeft(self) -> Vec:
        """Return a left vector, a vector that the matrix vector product can be stored in.

        Collective.

        See Also
        --------
        createVecs, createVecRight, petsc.MatCreateVecs

        """
        cdef Vec vecl = Vec()
        CHKERR( MatCreateVecs(self.mat, NULL, &vecl.vec) )
        return vecl

    getVecs = createVecs
    getVecRight = createVecRight
    getVecLeft = createVecLeft

    #

    def getColumnVector(self, column: int, Vec result=None) -> Vec:
        """Return the columnᵗʰ column vector of the matrix.

        Collective.

        Parameters
        ----------
        column
            Column index.
        result
            Optional vector to store the result.

        See Also
        --------
        petsc.MatGetColumnVector

        """
        cdef PetscInt ival = asInt(column)
        if result is None:
            result = Vec()
        if result.vec == NULL:
            CHKERR( MatCreateVecs(self.mat, NULL, &result.vec) )
        CHKERR( MatGetColumnVector(self.mat, result.vec, ival) )
        return result

    def getRedundantMatrix(self, nsubcomm: int, subcomm: Comm | None = None, Mat out=None) -> Mat:
        """Return redundant matrices on subcommunicators.

        Parameters
        ----------
        nsubcomm
            The number of subcommunicators.
        subcomm
            Communicator split or `None` for the null communicator.
        out
            Optional resultant matrix.
            When `None`, a new matrix is created, and ``MAT_INITIAL_MATRIX`` is used.
            When not `None`, the matrix is reused with ``MAT_REUSE_MATRIX``.

        See Also
        --------
        petsc.MatCreateRedundantMatrix

        """
        cdef PetscInt _nsubcomm   = asInt(nsubcomm)
        cdef MPI_Comm _subcomm    = MPI_COMM_NULL
        if subcomm:   _subcomm    = def_Comm(subcomm, PETSC_COMM_DEFAULT)
        cdef PetscMatReuse reuse  = MAT_INITIAL_MATRIX
        if out is None: out       = Mat()
        if out.mat != NULL: reuse = MAT_REUSE_MATRIX
        CHKERR( MatCreateRedundantMatrix(self.mat, _nsubcomm, _subcomm, reuse, &out.mat))
        return out

    def getDiagonal(self, Vec result=None) -> Vec:
        """Return the diagonal of the matrix.

        Collective.

        Parameters
        ----------
        result
            Optional vector to store the result.

        See Also
        --------
        setDiagonal, petsc.MatGetDiagonal

        """
        if result is None:
            result = Vec()
        if result.vec == NULL:
            CHKERR( MatCreateVecs(self.mat, NULL, &result.vec) )
        CHKERR( MatGetDiagonal(self.mat, result.vec) )
        return result

    def getRowSum(self, Vec result=None) -> Vec:
        """Return the row-sum vector.

        Collective.

        Parameters
        ----------
        result
            Optional vector to store the result.

        See Also
        --------
        petsc.MatGetRowSum

        """
        if result is None:
            result = Vec()
        if result.vec == NULL:
            CHKERR( MatCreateVecs(self.mat, NULL, &result.vec) )
        CHKERR( MatGetRowSum(self.mat, result.vec) )
        return result

    def setDiagonal(self, Vec diag, addv: InsertModeSpec = None) -> None:
        """Set the diagonal values of the matrix.

        Collective.

        Parameters
        ----------
        diag
            Vector storing diagonal values.
        addv
            Insertion mode.

        See Also
        --------
        getDiagonal, petsc.MatDiagonalSet

        """
        cdef PetscInsertMode caddv = insertmode(addv)
        CHKERR( MatDiagonalSet(self.mat, diag.vec, caddv) )

    def diagonalScale(self, Vec L=None, Vec R=None) -> None:
        """Perform left and/or right diagonal scaling of the matrix.

        Collective.

        Parameters
        ----------
        L
            Optional left scaling vector.
        R
            Optional right scaling vector.

        See Also
        --------
        petsc.MatDiagonalScale

        """
        cdef PetscVec vecl=NULL, vecr=NULL
        if L is not None: vecl = L.vec
        if R is not None: vecr = R.vec
        CHKERR( MatDiagonalScale(self.mat, vecl, vecr) )

    def invertBlockDiagonal(self) -> ArrayScalar:
        """Return the inverse of the block-diagonal entries.

        Collective.

        See Also
        --------
        petsc.MatInvertBlockDiagonal

        """
        cdef PetscInt bs = 0, m = 0
        cdef const PetscScalar *cibdiag = NULL
        CHKERR( MatGetBlockSize(self.mat, &bs) )
        CHKERR( MatGetLocalSize(self.mat, &m, NULL) )
        CHKERR( MatInvertBlockDiagonal(self.mat, &cibdiag) )
        cdef ndarray ibdiag = array_s(m*bs, cibdiag)
        ibdiag.shape = (toInt(m//bs), toInt(bs), toInt(bs))
        return ibdiag.transpose(0, 2, 1)

    # null space

    def setNullSpace(self, NullSpace nsp) -> None:
        """Set the nullspace.

        Collective.

        See Also
        --------
        getNullSpace, petsc.MatSetNullSpace

        """
        CHKERR( MatSetNullSpace(self.mat, nsp.nsp) )

    def getNullSpace(self) -> NullSpace:
        """Return the nullspace.

        Not collective.

        See Also
        --------
        setNullSpace, petsc.MatGetNullSpace

        """
        cdef NullSpace nsp = NullSpace()
        CHKERR( MatGetNullSpace(self.mat, &nsp.nsp) )
        PetscINCREF(nsp.obj)
        return nsp

    def setTransposeNullSpace(self, NullSpace nsp) -> None:
        """Set the transpose nullspace.

        Collective.

        See Also
        --------
        setNullSpace, getTransposeNullSpace, petsc.MatSetTransposeNullSpace

        """
        CHKERR( MatSetTransposeNullSpace(self.mat, nsp.nsp) )

    def getTransposeNullSpace(self) -> NullSpace:
        """Return the transpose nullspace.

        Not collective.

        See Also
        --------
        getNullSpace, setTransposeNullSpace, petsc.MatGetTransposeNullSpace

        """
        cdef NullSpace nsp = NullSpace()
        CHKERR( MatGetTransposeNullSpace(self.mat, &nsp.nsp) )
        PetscINCREF(nsp.obj)
        return nsp

    def setNearNullSpace(self, NullSpace nsp) -> None:
        """Set the near-nullspace.

        Collective.

        See Also
        --------
        setNullSpace, getNearNullSpace, petsc.MatSetNearNullSpace

        """
        CHKERR( MatSetNearNullSpace(self.mat, nsp.nsp) )

    def getNearNullSpace(self) -> NullSpace:
        """Return the near-nullspace.

        Not collective.

        See Also
        --------
        getNullSpace, setNearNullSpace, petsc.MatSetNearNullSpace

        """
        cdef NullSpace nsp = NullSpace()
        CHKERR( MatGetNearNullSpace(self.mat, &nsp.nsp) )
        PetscINCREF(nsp.obj)
        return nsp

    # matrix-vector product

    def mult(self, Vec x, Vec y) -> None:
        """Perform the matrix vector product y = A @ x.

        Collective.

        Parameters
        ----------
        x
            The input vector.
        y
            The output vector.

        See Also
        --------
        petsc.MatMult

        """
        CHKERR( MatMult(self.mat, x.vec, y.vec) )

    def multAdd(self, Vec x, Vec v, Vec y) -> None:
        """Perform the matrix vector product with addition y = A @ x + v.

        Collective.

        Parameters
        ----------
        x
            The input vector for the matrix-vector product.
        v
            The input vector to be added to.
        y
            The output vector.

        See Also
        --------
        petsc.MatMultAdd

        """
        CHKERR( MatMultAdd(self.mat, x.vec, v.vec, y.vec) )

    def multTranspose(self, Vec x, Vec y) -> None:
        """Perform the transposed matrix vector product y = A^T @ x.

        Collective.

        Parameters
        ----------
        x
            The input vector.
        y
            The output vector.

        See Also
        --------
        petsc.MatMultTranspose

        """
        CHKERR( MatMultTranspose(self.mat, x.vec, y.vec) )

    def multTransposeAdd(self, Vec x, Vec v, Vec y) -> None:
        """Perform the transposed matrix vector product with addition y = A^T @ x + v.

        Collective.

        Parameters
        ----------
        x
            The input vector for the transposed matrix-vector product.
        v
            The input vector to be added to.
        y
            The output vector.

        See Also
        --------
        petsc.MatMultTransposeAdd

        """
        CHKERR( MatMultTransposeAdd(self.mat, x.vec, v.vec, y.vec) )

    def multHermitian(self, Vec x, Vec y) -> None:
        """Perform the Hermitian matrix vector product y = A^H @ x.

        Collective.

        Parameters
        ----------
        x
            The input vector for the Hermitian matrix-vector product.
        y
            The output vector.

        See Also
        --------
        petsc.MatMultHermitianTranspose

        """
        CHKERR( MatMultHermitian(self.mat, x.vec, y.vec) )

    def multHermitianAdd(self, Vec x, Vec v, Vec y) -> None:
        """Perform the Hermitian matrix vector product with addition y = A^H @ x + v.

        Collective.

        Parameters
        ----------
        x
            The input vector for the Hermitian matrix-vector product.
        v
            The input vector to be added to.
        y
            The output vector.

        See Also
        --------
        petsc.MatMultHermitianTransposeAdd

        """
        CHKERR( MatMultHermitianAdd(self.mat, x.vec, v.vec, y.vec) )

    # SOR

    def SOR(
        self,
        Vec b,
        Vec x,
        omega:float = 1.0,
        sortype:SORType | None = None,
        shift:float = 0.0,
        its:int = 1,
        lits:int = 1,
        ) -> None:
        """Compute relaxation (SOR, Gauss-Seidel) sweeps.

        Neighborwise collective.

        See Also
        --------
        petsc.MatSOR

        """
        cdef PetscReal comega = asReal(omega)
        cdef PetscMatSORType csortype = SOR_LOCAL_SYMMETRIC_SWEEP
        if sortype is not None:
            csortype = <PetscMatSORType> asInt(sortype)
        cdef PetscReal cshift = asReal(shift)
        cdef PetscInt cits = asInt(its)
        cdef PetscInt clits = asInt(lits)
        CHKERR( MatSOR(self.mat, b.vec, comega, csortype, cshift, cits, clits, x.vec) )

    #

    def getDiagonalBlock(self) -> Mat:
        """Return the part of the matrix associated with the on-process coupling.

        Not collective.

        See Also
        --------
        petsc.MatGetDiagonalBlock

        """
        cdef Mat submat = Mat()
        CHKERR( MatGetDiagonalBlock(self.mat, &submat.mat) )
        PetscINCREF(submat.obj)
        return submat

    def increaseOverlap(self, IS iset, overlap: int = 1) -> None:
        """Increase the overlap of a index set.

        Collective.

        See Also
        --------
        petsc.MatIncreaseOverlap

        """
        cdef PetscInt ival = asInt(overlap)
        CHKERR( MatIncreaseOverlap(self.mat, 1, &iset.iset, ival) )

    def createSubMatrix(self, IS isrow, IS iscol=None, Mat submat=None) -> Mat:
        """Return a submatrix.

        Collective.

        Parameters
        ----------
        isrow
            Row index set.
        iscol
            Column index set. If `None`, ``iscol = isrow``.
        submat
            Optional resultant matrix.
            When `None`, a new matrix is created, and ``MAT_INITIAL_MATRIX`` is used.
            When not `None`, the matrix is reused with ``MAT_REUSE_MATRIX``.

        See Also
        --------
        petsc.MatCreateSubMatrix

        """
        cdef PetscMatReuse reuse = MAT_INITIAL_MATRIX
        cdef PetscIS ciscol = NULL
        if iscol is not None: ciscol = iscol.iset
        if submat is None: submat = Mat()
        if submat.mat != NULL: reuse = MAT_REUSE_MATRIX
        CHKERR( MatCreateSubMatrix(self.mat, isrow.iset, ciscol,
                                reuse, &submat.mat) )
        return submat

    def createSubMatrices(
        self,
        isrows: IS | Sequence[IS],
        iscols: IS | Sequence[IS] = None,
        submats: Mat | Sequence[Mat] = None,
        ) -> Sequence[Mat]:
        """Return several sequential submatrices.

        Collective.

        Parameters
        ----------
        isrows
            Row index sets.
        iscols
            Column index sets. If `None`, ``iscols = isrows``.
        submats
            Optional resultant matrices.
            When `None`, new matrices are created, and ``MAT_INITIAL_MATRIX`` is used.
            When not `None`, the matrices are reused with ``MAT_REUSE_MATRIX``.

        See Also
        --------
        petsc.MatCreateSubMatrices

        """
        if iscols is None: iscols = isrows
        isrows = [isrows] if isinstance(isrows, IS) else list(isrows)
        iscols = [iscols] if isinstance(iscols, IS) else list(iscols)
        assert len(isrows) == len(iscols)
        cdef Py_ssize_t i, n = len(isrows)
        cdef PetscMatReuse reuse = MAT_INITIAL_MATRIX
        cdef PetscIS  *cisrows = NULL
        cdef PetscIS  *ciscols = NULL
        cdef PetscMat *cmats   = NULL
        cdef object tmp1, tmp2
        cdef Mat mat
        tmp1 = oarray_p(empty_p(n), NULL, <void**>&cisrows)
        for i from 0 <= i < n: cisrows[i] = (<IS?>isrows[i]).iset
        tmp2 = oarray_p(empty_p(n), NULL, <void**>&ciscols)
        for i from 0 <= i < n: ciscols[i] = (<IS?>iscols[i]).iset
        if submats is not None:
            reuse = MAT_REUSE_MATRIX
            submats = list(submats)
            assert len(submats) == len(isrows)
            CHKERR( PetscMalloc(<size_t>(n+1)*sizeof(PetscMat), &cmats) )
            for i from 0 <= i < n: cmats[i] = (<Mat?>submats[i]).mat
        CHKERR( MatCreateSubMatrices(self.mat, <PetscInt>n, cisrows, ciscols, reuse, &cmats) )
        for i from 0 <= i < n: PetscINCREF(<PetscObject*>&cmats[i])
        if reuse == MAT_INITIAL_MATRIX:
            submats = [None] * n
            for i from 0 <= i < n:
                submats[i] = mat = Mat()
                mat.mat = cmats[i]
        CHKERR( MatDestroyMatrices(<PetscInt>n, &cmats) )
        return submats

    #

    def getLocalSubMatrix(self, IS isrow, IS iscol, Mat submat=None) -> Mat:
        """Return a reference to a submatrix specified in local numbering.

        Collective.

        Parameters
        ----------
        isrow
            Row index set.
        iscol
            Column index set.
        submat
            Optional resultant matrix.
            When `None`, a new matrix is created.
            When not `None`, the matrix is first destroyed and then recreated.

        See Also
        --------
        restoreLocalSubMatrix, petsc.MatGetLocalSubMatrix

        """
        if submat is None: submat = Mat()
        else: CHKERR( MatDestroy(&submat.mat) )
        CHKERR( MatGetLocalSubMatrix(self.mat, isrow.iset, iscol.iset, &submat.mat) )
        return submat

    def restoreLocalSubMatrix(self, IS isrow, IS iscol, Mat submat):
        """Restore a reference to a submatrix obtained with `getLocalSubMatrix`.

        Collective.

        Parameters
        ----------
        isrow
            Row index set.
        iscol
            Column index set.
        submat
            The submatrix.

        See Also
        --------
        getLocalSubMatrix, petsc.MatRestoreLocalSubMatrix

        """
        CHKERR( MatRestoreLocalSubMatrix(self.mat, isrow.iset, iscol.iset, &submat.mat) )

    #

    def norm(
        self,
        norm_type: NormTypeSpec = None,
    ) -> float | tuple[float, float]:
        """Compute the requested matrix norm.

        Collective.

        A 2-tuple is returned if `NormType.NORM_1_AND_2` is specified.

        See Also
        --------
        petsc.MatNorm, petsc.NormType

        """
        cdef PetscNormType norm_1_2 = PETSC_NORM_1_AND_2
        cdef PetscNormType ntype = PETSC_NORM_FROBENIUS
        if norm_type is not None: ntype = norm_type
        cdef PetscReal rval[2]
        CHKERR( MatNorm(self.mat, ntype, rval) )
        if ntype != norm_1_2: return toReal(rval[0])
        else: return (toReal(rval[0]), toReal(rval[1]))

    def scale(self, alpha: Scalar) -> None:
        """Scale the matrix.

        Collective.

        See Also
        --------
        petsc.MatScale

        """
        cdef PetscScalar sval = asScalar(alpha)
        CHKERR( MatScale(self.mat, sval) )

    def shift(self, alpha: Scalar) -> None:
        """Shift the matrix.

        Collective.

        See Also
        --------
        petsc.MatShift

        """
        cdef PetscScalar sval = asScalar(alpha)
        CHKERR( MatShift(self.mat, sval) )

    def chop(self, tol: float) -> None:
        """Set entries smallest of tol (in absolute values) to zero.

        Collective.

        See Also
        --------
        petsc.MatChop

        """
        cdef PetscReal rval = asReal(tol)
        CHKERR( MatChop(self.mat, rval) )

    def setRandom(self, Random random=None) -> None:
        """Set random values in the matrix.

        Collective.

        Parameters
        ----------
        random
            The random number generator object or `None` for the default.

        See Also
        --------
        petsc.MatSetRandom

        """
        cdef PetscRandom rnd = NULL
        if random is not None: rnd = random.rnd
        CHKERR( MatSetRandom(self.mat, rnd) )

    def axpy(self, alpha: Scalar, Mat X, structure: Structure = None) -> None:
        """Perform the matrix summation ``self`` + = ɑ·X.

        Collective.

        Parameters
        ----------
        alpha
            The scalar.
        X
            The matrix to be added.
        structure
            The structure of the operation.

        See Also
        --------
        petsc.MatAXPY

        """
        cdef PetscScalar sval = asScalar(alpha)
        cdef PetscMatStructure flag = matstructure(structure)
        CHKERR( MatAXPY(self.mat, sval, X.mat, flag) )

    def aypx(self, alpha: Scalar, Mat X, structure: Structure = None) -> None:
        """Perform the matrix summation ``self`` = ɑ·``self`` + X.

        Collective.

        Parameters
        ----------
        alpha
            The scalar.
        X
            The matrix to be added.
        structure
            The structure of the operation.

        See Also
        --------
        petsc.MatAYPX

        """
        cdef PetscScalar sval = asScalar(alpha)
        cdef PetscMatStructure flag = matstructure(structure)
        CHKERR( MatAYPX(self.mat, sval, X.mat, flag) )

    # matrix-matrix product

    def matMult(
        self,
        Mat mat,
        Mat result=None,
        fill: float | None = None
    ) -> Mat:
        """Perform matrix-matrix multiplication C=AB.

        Neighborwise collective.

        Parameters
        ----------
        mat
            The right hand matrix B.
        result
            The optional resultant matrix C. When `None`, a new matrix
            is created, and ``MAT_INITIAL_MATRIX`` is used. When C is
            not `None`, the matrix is reused with ``MAT_REUSE_MATRIX``.
        fill
            Expected fill as ratio of nnz(C)/(nnz(A) + nnz(B)), use
            `None` if you do not have a good estimate. If the
            result is a dense matrix this is irrelevant.

        Returns
        -------
        result : Mat
            The resultant product matrix C.

        Notes
        -----
        To determine the correct fill value, run with -info and search
        for the string "Fill ratio" to see the value actually needed.

        See also
        --------
        petsc.MatMatMult, petsc.MatReuse

        """
        cdef PetscMatReuse reuse = MAT_INITIAL_MATRIX
        cdef PetscReal rval = 2
        if result is None:
            result = Mat()
        elif result.mat != NULL:
            reuse = MAT_REUSE_MATRIX
        if fill is not None: rval = asReal(fill)
        CHKERR( MatMatMult(self.mat, mat.mat, reuse, rval, &result.mat) )
        return result

    def matTransposeMult(
        self,
        Mat mat,
        Mat result=None,
        fill: float | None = None
    ):
        """Perform matrix-matrix multiplication C=ABᵀ.

        Neighborwise collective.

        Parameters
        ----------
        mat
            The right hand matrix B.
        result
            The optional resultant matrix C. When `None`, a new matrix
            is created, and ``MAT_INITIAL_MATRIX`` is used. When C is
            not `None`, the matrix is reused with ``MAT_REUSE_MATRIX``.
        fill
            Expected fill as ratio of nnz(C)/(nnz(A) + nnz(B)), use
            `None` if you do not have a good estimate. If the
            result is a dense matrix this is irrelevant.

        Returns
        -------
        result : Mat
            The resultant product matrix C.

        Notes
        -----
        To determine the correct fill value, run with -info and search
        for the string "Fill ratio" to see the value actually needed.

        See also
        --------
        petsc.MatMatTransposeMult, petsc.MatReuse

        """
        cdef PetscMatReuse reuse = MAT_INITIAL_MATRIX
        cdef PetscReal rval = 2
        if result is None:
            result = Mat()
        elif result.mat != NULL:
            reuse = MAT_REUSE_MATRIX
        if fill is not None: rval = asReal(fill)
        CHKERR( MatMatTransposeMult(self.mat, mat.mat, reuse, rval, &result.mat) )
        return result

    def transposeMatMult(
        self,
        Mat mat,
        Mat result=None,
        fill: float | None = None
    ):
        """Perform matrix-matrix multiplication C=AᵀB.

        Neighborwise collective.

        Parameters
        ----------
        mat
            The right hand matrix B.
        result
            The optional resultant matrix C. When `None`, a new matrix
            is created, and ``MAT_INITIAL_MATRIX`` is used. When C is
            not `None`, the matrix is reused with ``MAT_REUSE_MATRIX``.
        fill
            Expected fill as ratio of nnz(C)/(nnz(A) + nnz(B)), use
            `None` if you do not have a good estimate. If the
            result is a dense matrix this is irrelevant.

        Returns
        -------
        result : Mat
            The resultant product matrix C.

        Notes
        -----
        To determine the correct fill value, run with -info and search
        for the string "Fill ratio" to see the value actually needed.

        See also
        --------
        petsc.MatTransposeMatMult, petsc.MatReuse

        """
        cdef PetscMatReuse reuse = MAT_INITIAL_MATRIX
        cdef PetscReal rval = 2
        if result is None:
            result = Mat()
        elif result.mat != NULL:
            reuse = MAT_REUSE_MATRIX
        if fill is not None: rval = asReal(fill)
        CHKERR( MatTransposeMatMult(self.mat, mat.mat, reuse, rval, &result.mat) )
        return result

    def ptap(
        self,
        Mat P,
        Mat result=None,
        fill: float | None = None
    ) -> Mat:
        """Creates the matrix product C = PᵀAP.

        Neighborwise collective.

        Parameters
        ----------
        P
            The matrix P.
        result
            The optional resultant matrix C. When `None`, a new matrix
            is created, and ``MAT_INITIAL_MATRIX`` is used. When C is
            not `None`, the matrix is reused with ``MAT_REUSE_MATRIX``.
        fill
            Expected fill as ratio of nnz(C)/(nnz(A) + nnz(P)), use
            `None` if you do not have a good estimate. If the
            result is a dense matrix this is irrelevant.

        Returns
        -------
        result : Mat
            The resultant product matrix C.

        Notes
        -----
        To determine the correct fill value, run with -info and search
        for the string "Fill ratio" to see the value actually needed.

        An alternative approach to this function is to use
        `petsc.MatProductCreate` and set the desired options before the
        computation is done.

        See also
        --------
        petsc.MatPtAP, petsc.MatReuse

        """
        cdef PetscMatReuse reuse = MAT_INITIAL_MATRIX
        cdef PetscReal cfill = PETSC_DEFAULT
        if result is None:
            result = Mat()
        elif result.mat != NULL:
            reuse = MAT_REUSE_MATRIX
        if fill is not None: cfill = asReal(fill)
        CHKERR( MatPtAP(self.mat, P.mat, reuse, cfill, &result.mat) )
        return result

    def rart(
        self,
        Mat R,
        Mat result=None,
        fill: float | None = None
    ) -> Mat:
        """Create the matrix product C = RARᵀ.

        Neighborwise collective.

        Parameters
        ----------
        R
            The projection matrix.
        result
            The optional resultant matrix C. When `None`, a new matrix
            is created, and ``MAT_INITIAL_MATRIX`` is used. When C is
            not `None`, the matrix is reused with ``MAT_REUSE_MATRIX``.
        fill
            Expected fill as ratio of nnz(C)/nnz(A), use `None` if
            you do not have a good estimate. If the result is a dense
            matrix this is irrelevant.

        Returns
        -------
        result : Mat
            The resultant product matrix C.

        Notes
        -----
        To determine the correct fill value, run with -info and search
        for the string "Fill ratio" to see the value actually needed.

        See also
        --------
        petsc.MatRARt, petsc.MatReuse

        """
        cdef PetscMatReuse reuse = MAT_INITIAL_MATRIX
        cdef PetscReal cfill = PETSC_DEFAULT
        if result is None:
            result = Mat()
        elif result.mat != NULL:
            reuse = MAT_REUSE_MATRIX
        if fill is not None: cfill = asReal(fill)
        CHKERR( MatRARt(self.mat, R.mat, reuse, cfill, &result.mat) )
        return result

    def matMatMult(
        self,
        Mat B,
        Mat C,
        Mat result=None,
        fill: float | None = None
    ) -> Mat:
        """Perform matrix-matrix-matrix multiplication D=ABC.

        Neighborwise collective.

        Parameters
        ----------
        B
            The middle matrix B.
        C
            The right hand matrix C.
        result
            The optional resultant matrix D. When `None`, a new matrix
            is created, and ``MAT_INITIAL_MATRIX`` is used. When D is
            not `None`, the matrix is reused with ``MAT_REUSE_MATRIX``.
        fill
            Expected fill as ratio of nnz(C)/nnz(A), use `None` if
            you do not have a good estimate. If the result is a dense
            matrix this is irrelevant.

        Returns
        -------
        result : Mat
            The resultant product matrix D.

        See also
        --------
        petsc.MatMatMatMult, petsc.MatReuse

        """
        cdef PetscMatReuse reuse = MAT_INITIAL_MATRIX
        cdef PetscReal cfill = PETSC_DEFAULT
        if result is None:
            result = Mat()
        elif result.mat != NULL:
            reuse = MAT_REUSE_MATRIX
        if fill is not None: cfill = asReal(fill)
        CHKERR( MatMatMatMult(self.mat, B.mat, C.mat, reuse, cfill, &result.mat) )
        return result

    def kron(
        self,
        Mat mat,
        Mat result=None
    ) -> Mat:
        """Compute C, the Kronecker product of A and B.

        Parameters
        ----------
        mat
            The right hand matrix B.
        result
            The optional resultant matrix. When `None`, a new matrix
            is created, and ``MAT_INITIAL_MATRIX`` is used. When it is
            not `None`, the matrix is reused with ``MAT_REUSE_MATRIX``.

        Returns
        -------
        result : Mat
            The resultant matrix C, the Kronecker product of A and B.

        See also
        --------
        petsc.MatSeqAIJKron, petsc.MatReuse

        """
        cdef PetscMatReuse reuse = MAT_INITIAL_MATRIX
        if result is None:
            result = Mat()
        elif result.mat != NULL:
            reuse = MAT_REUSE_MATRIX
        CHKERR( MatSeqAIJKron(self.mat, mat.mat, reuse, &result.mat) )
        return result

    def bindToCPU(self, flg: bool) -> None:
        """Mark a matrix to temporarily stay on the CPU.

        Once marked, perform computations on the CPU.

        Parameters
        ----------
        flg
            Bind to the CPU if `True`.

        See also
        --------
        petsc.MatBindToCPU

        """
        cdef PetscBool bindFlg = asBool(flg)
        CHKERR( MatBindToCPU(self.mat, bindFlg) )

    def boundToCPU(self) -> bool:
        """Query if a matrix is bound to the CPU.

        See also
        --------
        petsc.MatBoundToCPU

        """
        cdef PetscBool flg = PETSC_TRUE
        CHKERR( MatBoundToCPU(self.mat, &flg) )
        return toBool(flg)

    # XXX factorization

    def getOrdering(self, ord_type: OrderingType) -> tuple[IS, IS]:
        """Return a reordering for a matrix to improve a LU factorization.

        Collective.

        Parameters
        ----------
        ord_type
            The type of reordering.

        Returns
        -------
        rp : IS
            The row permutation indices.
        cp : IS
            The column permutation indices.

        See Also
        --------
        petsc.MatGetOrdering

        """
        cdef PetscMatOrderingType cval = NULL
        ord_type = str2bytes(ord_type, &cval)
        cdef IS rp = IS(), cp = IS()
        CHKERR( MatGetOrdering(self.mat, cval, &rp.iset, &cp.iset) )
        return (rp, cp)

    def reorderForNonzeroDiagonal(
        self,
        IS isrow,
        IS iscol,
        atol: float = 0
        ) -> None:
        """Change a matrix ordering to remove zeros from the diagonal.

        Collective.

        Parameters
        ----------
        isrow
            The row reordering.
        iscol
            The column reordering.
        atol
            The absolute tolerance. Values along the diagonal whose absolute value
            are smaller than this tolerance are moved off the diagonal.

        See Also
        --------
        getOrdering, petsc.MatReorderForNonzeroDiagonal

        """
        cdef PetscReal rval = asReal(atol)
        cdef PetscIS rp = isrow.iset, cp = iscol.iset
        CHKERR( MatReorderForNonzeroDiagonal(self.mat, rval, rp, cp) )

    def factorLU(
        self,
        IS isrow,
        IS iscol,
        options: dict[str, Any] | None = None,
        ) -> None:
        """Perform an in-place LU factorization.

        Collective.

        Parameters
        ----------
        isrow
            The row permutation.
        iscol
            The column permutation.
        options
            An optional dictionary of options for the factorization. These include
            ``fill``, the expected fill as a ratio of the original fill and
            ``dtcol``, the pivot tolerance where ``0`` indicates no pivot and ``1``
            indicates full column pivoting.

        See Also
        --------
        petsc.MatLUFactor

        """
        cdef PetscMatFactorInfo info
        matfactorinfo(PETSC_FALSE, PETSC_FALSE, options, &info)
        CHKERR( MatLUFactor(self.mat, isrow.iset, iscol.iset, &info) )

    def factorSymbolicLU(self, Mat mat, IS isrow, IS iscol, options=None) -> None:
        """Not implemented."""
        raise NotImplementedError

    def factorNumericLU(self, Mat mat, options=None) -> None:
        """Not implemented."""
        raise NotImplementedError

    def factorILU(
        self,
        IS isrow,
        IS iscol,
        options: dict[str, Any] | None = None,
        ) -> None:
        """Perform an in-place ILU factorization.

        Collective.

        Parameters
        ----------
        isrow
            The row permutation.
        iscol
            The column permutation.
        options
            An optional dictionary of options for the factorization. These include
            ``levels``, the number of levels of fill, ``fill``, the expected fill
            as a ratio of the original fill, and ``dtcol``, the pivot tolerance
            where ``0`` indicates no pivot and ``1`` indicates full column pivoting.

        See Also
        --------
        petsc.MatILUFactor

        """
        cdef PetscMatFactorInfo info
        matfactorinfo(PETSC_TRUE, PETSC_FALSE, options, &info)
        CHKERR( MatILUFactor(self.mat, isrow.iset, iscol.iset, &info) )

    def factorSymbolicILU(self, IS isrow, IS iscol, options=None) -> None:
        """Not implemented."""
        raise NotImplementedError

    def factorCholesky(
        self,
        IS isperm,
        options: dict[str, Any] | None = None,
        ) -> None:
        """Perform an in-place Cholesky factorization.

        Collective.

        Parameters
        ----------
        isperm
            The row and column permutations.
        options
            An optional dictionary of options for the factorization. These include
            ``fill``, the expected fill as a ratio of the original fill.

        See Also
        --------
        factorLU, petsc.MatCholeskyFactor

        """
        cdef PetscMatFactorInfo info
        matfactorinfo(PETSC_FALSE, PETSC_TRUE, options, &info)
        CHKERR( MatCholeskyFactor(self.mat, isperm.iset, &info) )

    def factorSymbolicCholesky(self, IS isperm, options=None) -> None:
        """Not implemented."""
        raise NotImplementedError

    def factorNumericCholesky(self, Mat mat, options=None) -> None:
        """Not implemented."""
        raise NotImplementedError

    def factorICC(
        self,
        IS isperm,
        options: dict[str, Any] | None = None,
        ) -> None:
        """Perform an in-place an incomplete Cholesky factorization.

        Collective.

        Parameters
        ----------
        isperm
            The row and column permutations
        options
            An optional dictionary of options for the factorization. These include
            ``fill``, the expected fill as a ratio of the original fill.

        See Also
        --------
        factorILU, petsc.MatICCFactor

        """
        cdef PetscMatFactorInfo info
        matfactorinfo(PETSC_TRUE, PETSC_TRUE, options, &info)
        CHKERR( MatICCFactor(self.mat, isperm.iset, &info) )

    def factorSymbolicICC(self, IS isperm, options=None) -> None:
        """Not implemented."""
        raise NotImplementedError

    def getInertia(self) -> tuple[int, int, int]:
        """Return the inertia from a factored matrix.

        Collective. The matrix must have been factored by calling `factorCholesky`.

        Returns
        -------
        n : int
            The number of negative eigenvalues.
        z : int
            The number of zero eigenvalues.
        p : int
            The number of positive eigenvalues.

        See Also
        --------
        petsc.MatGetInertia

        """
        cdef PetscInt ival1 = 0, ival2 = 0, ival3 = 0
        CHKERR( MatGetInertia(self.mat, &ival1, &ival2, &ival3) )
        return (toInt(ival1), toInt(ival2), toInt(ival3))

    def setUnfactored(self) -> None:
        """Set a factored matrix to be treated as unfactored.

        Logically collective.

        See Also
        --------
        petsc.MatSetUnfactored

        """
        CHKERR( MatSetUnfactored(self.mat) )

    # IS

    def fixISLocalEmpty(self, fix: bool) -> None:
        """Compress out zero local rows from the local matrices.

        Collective.

        Parameters
        ----------
        fix
            When `True`, new local matrices and local to global maps are generated
            during the final assembly process.

        See Also
        --------
        petsc.MatISFixLocalEmpty

        """
        cdef PetscBool cfix = asBool(fix)
        CHKERR( MatISFixLocalEmpty(self.mat, cfix) )

    def getISLocalMat(self) -> Mat:
        """Return the local matrix stored inside a `Type.IS` matrix.

        See Also
        --------
        petsc.MatISGetLocalMat

        """
        cdef Mat local = Mat()
        CHKERR( MatISGetLocalMat(self.mat, &local.mat) )
        PetscINCREF(local.obj)
        return local

    def restoreISLocalMat(self, Mat local not None) -> None:
        """Restore the local matrix obtained with `getISLocalMat`.

        Parameters
        ----------
        local
            The local matrix.

        See Also
        --------
        petsc.MatISRestoreLocalMat

        """
        CHKERR( MatISRestoreLocalMat(self.mat, &local.mat) )

    def setISLocalMat(self, Mat local not None) -> None:
        """Set the local matrix stored inside a `Type.IS`.

        Parameters
        ----------
        local
            The local matrix.

        See Also
        --------
        petsc.MatISSetLocalMat

        """
        CHKERR( MatISSetLocalMat(self.mat, local.mat) )

    def setISPreallocation(
        self,
        nnz: Sequence[int],
        onnz: Sequence[int],
        ) -> Self:
        """Preallocate memory for a `Type.IS` parallel matrix.

        Parameters
        ----------
        nnz
            The sequence whose length corresponds to the number of local rows
            and values which represent the number of nonzeros in the various
            rows of the *diagonal* of the local submatrix.
        onnz:
            The sequence whose length corresponds to the number of local rows
            and values which represent the number of nonzeros in the various
            rows of the *off diagonal* of the local submatrix.

        See Also
        --------
        petsc.MatISSetPreallocation

        """
        cdef PetscInt *cnnz = NULL
        cdef PetscInt *connz = NULL
        nnz = iarray_i(nnz, NULL, &cnnz)
        onnz = iarray_i(onnz, NULL, &connz)
        CHKERR( MatISSetPreallocation(self.mat, 0, cnnz, 0, connz) )
        return self

    # LRC

    def getLRCMats(self) -> tuple[Mat, Mat, Vec, Mat]:
        """Return the constituents of a `Type.LRC` matrix.

        Collective.

        Returns
        -------
        A : Mat
            The ``A`` matrix.
        U : Mat
            The first dense rectangular matrix.
        c : Vec
            The sequential vector containing the diagonal of ``C``.
        V : Mat
            The second dense rectangular matrix.

        See Also
        --------
        petsc.MatLRCGetMats

        """
        cdef Mat A = Mat()
        cdef Mat U = Mat()
        cdef Vec c = Vec()
        cdef Mat V = Mat()
        CHKERR( MatLRCGetMats(self.mat, &A.mat, &U.mat, &c.vec, &V.mat) )
        PetscINCREF(A.obj)
        PetscINCREF(U.obj)
        PetscINCREF(c.obj)
        PetscINCREF(V.obj)
        return (A, U, c, V)

    # H2Opus

    def H2OpusOrthogonalize(self) -> Self:
        """Orthogonalize the basis tree of a hierarchical matrix.

        See Also
        --------
        petsc.MatH2OpusOrthogonalize

        """
        CHKERR( MatH2OpusOrthogonalize(self.mat) )
        return self

    def H2OpusCompress(self, tol: float):
        """Compress a hierarchical matrix.

        Parameters
        ----------
        tol
            The absolute truncation threshold.

        See Also
        --------
        petsc.MatH2OpusCompress

        """
        cdef PetscReal _tol = asReal(tol)
        CHKERR( MatH2OpusCompress(self.mat, _tol) )
        return self

    def H2OpusLowRankUpdate(self, Mat U, Mat V=None, s: float = 1.0):
        """Perform a low-rank update of the form ``self`` += sUVᵀ.

        Parameters
        ----------
        U
            The dense low-rank update matrix.
        V
            The dense low-rank update matrix. If `None`, ``V = U``.
        s
            The scaling factor.

        See Also
        --------
        petsc.MatH2OpusLowRankUpdate

        """
        cdef PetscScalar _s = asScalar(s)
        cdef PetscMat vmat = NULL
        if V is not None:
            vmat = V.mat
        CHKERR( MatH2OpusLowRankUpdate(self.mat, U.mat, vmat, _s) )
        return self

    # MUMPS

    def setMumpsIcntl(self, icntl: int, ival: int) -> None:
        """Set a MUMPS parameter, ``ICNTL[icntl] = ival``.

        Logically collective.

        Parameters
        ----------
        icntl
            The index of the MUMPS parameter array.
        ival
            The value to set.

        See Also
        --------
        petsc_options, petsc.MatMumpsSetIcntl

        """
        cdef PetscInt _icntl = asInt(icntl)
        cdef PetscInt _ival = asInt(ival)
        CHKERR( MatMumpsSetIcntl(self.mat, _icntl, _ival) );

    def getMumpsIcntl(self, icntl: int) -> int:
        """Return the MUMPS parameter, ``ICNTL[icntl]``.

        Logically collective.

        See Also
        --------
        petsc_options, petsc.MatMumpsGetIcntl

        """
        cdef PetscInt _icntl = asInt(icntl)
        cdef PetscInt ival = 0
        CHKERR( MatMumpsGetIcntl(self.mat, _icntl, &ival) );
        return toInt(ival)

    def setMumpsCntl(self, icntl: int, val: float):
        """Set a MUMPS parameter, ``CNTL[icntl] = val``.

        Logically collective.

        Parameters
        ----------
        icntl
            The index of the MUMPS parameter array.
        val
            The value to set.

        See Also
        --------
        petsc_options, petsc.MatMumpsSetCntl

        """
        cdef PetscInt _icntl = asInt(icntl)
        cdef PetscReal _val = asReal(val)
        CHKERR( MatMumpsSetCntl(self.mat, _icntl, _val) );

    def getMumpsCntl(self, icntl: int) -> float:
        """Return the MUMPS parameter, ``CNTL[icntl]``.

        Logically collective.

        See Also
        --------
        petsc_options, petsc.MatMumpsGetCntl

        """
        cdef PetscInt _icntl = asInt(icntl)
        cdef PetscReal val = 0
        CHKERR( MatMumpsGetCntl(self.mat, _icntl, &val) );
        return toReal(val)

    def getMumpsInfo(self, icntl: int) -> int:
        """Return the MUMPS parameter, ``INFO[icntl]``.

        Logically collective.

        Parameters
        ----------
        icntl
            The index of the MUMPS INFO array.

        See Also
        --------
        petsc.MatMumpsGetInfo

        """
        cdef PetscInt _icntl = asInt(icntl)
        cdef PetscInt ival = 0
        CHKERR( MatMumpsGetInfo(self.mat, _icntl, &ival) );
        return toInt(ival)

    def getMumpsInfog(self, icntl: int) -> int:
        """Return the MUMPS parameter, ``INFOG[icntl]``.

        Logically collective.

        Parameters
        ----------
        icntl
            The index of the MUMPS INFOG array.

        See Also
        --------
        petsc.MatMumpsGetInfog

        """
        cdef PetscInt _icntl = asInt(icntl)
        cdef PetscInt ival = 0
        CHKERR( MatMumpsGetInfog(self.mat, _icntl, &ival) );
        return toInt(ival)

    def getMumpsRinfo(self, icntl: int) -> float:
        """Return the MUMPS parameter, ``RINFO[icntl]``.

        Logically collective.

        Parameters
        ----------
        icntl
            The index of the MUMPS RINFO array.

        See Also
        --------
        petsc.MatMumpsGetRinfo

        """
        cdef PetscInt _icntl = asInt(icntl)
        cdef PetscReal val = 0
        CHKERR( MatMumpsGetRinfo(self.mat, _icntl, &val) );
        return toReal(val)

    def getMumpsRinfog(self, icntl: int) -> float:
        """Return the MUMPS parameter, ``RINFOG[icntl]``.

        Logically collective.

        Parameters
        ----------
        icntl
            The index of the MUMPS RINFOG array.

        See Also
        --------
        petsc.MatMumpsGetRinfog

        """
        cdef PetscInt _icntl = asInt(icntl)
        cdef PetscReal val = 0
        CHKERR( MatMumpsGetRinfog(self.mat, _icntl, &val) );
        return toReal(val)

    # solve

    def solveForward(self, Vec b, Vec x) -> None:
        """Solve Lx = b, given a factored matrix A = LU.

        Neighborwise collective.

        Parameters
        ----------
        b
            The right-hand side vector.
        x
            The output solution vector.

        See Also
        --------
        petsc.MatForwardSolve

        """
        CHKERR( MatForwardSolve(self.mat, b.vec, x.vec) )

    def solveBackward(self, Vec b, Vec x) -> None:
        """Solve Ux=b, given a factored matrix A=LU.

        Neighborwise collective.

        Parameters
        ----------
        b
            The right-hand side vector.
        x
            The output solution vector.

        See Also
        --------
        petsc.MatBackwardSolve

        """
        CHKERR( MatBackwardSolve(self.mat, b.vec, x.vec) )

    def solve(self, Vec b, Vec x) -> None:
        """Solve Ax=b, given a factored matrix.

        Neighborwise collective. The vectors ``b`` and ``x`` cannot be the same.
        Most users should employ the `KSP` interface for linear solvers instead
        of working directly with matrix algebra routines.

        Parameters
        ----------
        b
            The right-hand side vector.
        x
            The output solution vector, must be different than ``b``.

        See Also
        --------
        KSP.create, solveTranspose, petsc.MatSolve

        """
        CHKERR(MatSolve(self.mat, b.vec, x.vec) )

    def solveTranspose(self, Vec b, Vec x) -> None:
        """Solve Aᵀx=b, given a factored matrix.

        Neighborwise collective. The vectors ``b`` and ``x`` cannot be the same.

        Parameters
        ----------
        b
            The right-hand side vector.
        x
            The output solution vector, must be different than ``b``.

        See Also
        --------
        KSP.create, petsc.MatSolve, petsc.MatSolveTranspose

        """
        CHKERR( MatSolveTranspose(self.mat, b.vec, x.vec) )

    def solveAdd(self, Vec b, Vec y, Vec x) -> None:
        """Solve x=y+A⁻¹b, given a factored matrix.

        Neighborwise collective. The vectors ``b`` and ``x`` cannot be the same.

        Parameters
        ----------
        b
            The right-hand side vector.
        y
            The vector to be added
        x
            The output solution vector, must be different than ``b``.

        See Also
        --------
        KSP.create, petsc.MatSolve, petsc.MatSolveAdd

        """
        CHKERR( MatSolveAdd(self.mat, b.vec, y.vec, x.vec) )

    def solveTransposeAdd(self, Vec b, Vec y, Vec x) -> None:
        """Solve x=y+A⁻ᵀb, given a factored matrix.

        Neighborwise collective. The vectors ``b`` and ``x`` cannot be the same.

        Parameters
        ----------
        b
            The right-hand side vector.
        y
            The vector to be added
        x
            The output solution vector, must be different than ``b``.

        See Also
        --------
        KSP.create, petsc.MatSolve, petsc.MatSolveTransposeAdd

        """
        CHKERR( MatSolveTransposeAdd(self.mat, b.vec, y.vec, x.vec) )

    def matSolve(self, Mat B, Mat X) -> None:
        """Solve AX=B, given a factored matrix A.

        Neighborwise collective.

        Parameters
        ----------
        B
            The right-hand side matrix of type `Type.DENSE`. Can be of type
            `Type.AIJ` if using MUMPS.
        X
            The output solution matrix, must be different than ``B``.

        See Also
        --------
        KSP.create, petsc.MatMatSolve

        """
        CHKERR( MatMatSolve(self.mat, B.mat, X.mat) )

    # dense matrices

    def setDenseLDA(self, lda: int) -> None:
        """Set the leading dimension of the array used by the dense matrix.

        Not collective.

        Parameters
        ----------
        lda
            The leading dimension.

        See Also
        --------
        petsc.MatDenseSetLDA

        """
        cdef PetscInt _ilda = asInt(lda)
        CHKERR( MatDenseSetLDA(self.mat, _ilda) )

    def getDenseLDA(self) -> int:
        """Return the leading dimension of the array used by the dense matrix.

        Not collective.

        See Also
        --------
        petsc.MatDenseGetLDA

        """
        cdef PetscInt lda=0
        CHKERR( MatDenseGetLDA(self.mat, &lda) )
        return toInt(lda)

    def getDenseArray(self, readonly: bool = False) -> ArrayScalar:
        """Return the array where the data is stored.

        Not collective.

        Parameters
        ----------
        readonly
            Enable to obtain a read only array.

        See Also
        --------
        petsc.MatDenseGetArrayRead, petsc.MatDenseGetArray

        """
        cdef PetscInt m=0, N=0, lda=0
        cdef PetscScalar *data = NULL
        CHKERR( MatGetLocalSize(self.mat, &m, NULL) )
        CHKERR( MatGetSize(self.mat, NULL, &N) )
        CHKERR( MatDenseGetLDA(self.mat, &lda) )
        if readonly:
            CHKERR( MatDenseGetArrayRead(self.mat, <const PetscScalar**>&data) )
        else:
            CHKERR( MatDenseGetArray(self.mat, &data) )
        cdef int typenum = NPY_PETSC_SCALAR
        cdef int itemsize = <int>sizeof(PetscScalar)
        cdef int flags = NPY_ARRAY_FARRAY
        cdef npy_intp dims[2], strides[2]
        dims[0] = <npy_intp>m; strides[0] = <npy_intp>sizeof(PetscScalar);
        dims[1] = <npy_intp>N; strides[1] = <npy_intp>(lda*sizeof(PetscScalar));
        array = <object>PyArray_New(<PyTypeObject*>ndarray, 2, dims, typenum,
                                    strides, data, itemsize, flags, NULL)
        if readonly:
            CHKERR( MatDenseRestoreArrayRead(self.mat, <const PetscScalar**>&data) )
        else:
            CHKERR( MatDenseRestoreArray(self.mat, &data) )
        return array

    def getDenseLocalMatrix(self) -> Mat:
        """Return the local part of the dense matrix.

        Not collective.

        See Also
        --------
        petsc.MatDenseGetLocalMatrix

        """
        cdef Mat mat = type(self)()
        CHKERR( MatDenseGetLocalMatrix(self.mat, &mat.mat) )
        PetscINCREF(mat.obj)
        return mat

    def getDenseColumnVec(self, i: int, mode: AccessModeSpec = 'rw') -> Vec:
        """Return the iᵗʰ column vector of the dense matrix.

        Collective.

        Parameters
        ----------
        i
            The column index to access.
        mode
            The access type of the returned array

        See Also
        --------
        restoreDenseColumnVec, petsc.MatDenseGetColumnVec
        petsc.MatDenseGetColumnVecRead, petsc.MatDenseGetColumnVecWrite

        """
        if mode is None: mode = 'rw'
        if mode not in ['rw', 'r', 'w']:
            raise ValueError("Invalid mode: expected 'rw', 'r', or 'w'")
        cdef Vec v = Vec()
        cdef PetscInt _i = asInt(i)
        if mode == 'rw':
            CHKERR( MatDenseGetColumnVec(self.mat, _i, &v.vec) )
        elif mode == 'r':
            CHKERR( MatDenseGetColumnVecRead(self.mat, _i, &v.vec) )
        else:
            CHKERR( MatDenseGetColumnVecWrite(self.mat, _i, &v.vec) )
        PetscINCREF(v.obj)
        return v

    def restoreDenseColumnVec(self, i: int, mode: AccessModeSpec = 'rw') -> None:
        """Restore the iᵗʰ column vector of the dense matrix.

        Collective.

        Parameters
        ----------
        i
            The column index to be restored.
        mode
            The access type of the restored array

        See Also
        --------
        getDenseColumnVec, petsc.MatDenseRestoreColumnVec
        petsc.MatDenseRestoreColumnVecRead, petsc.MatDenseRestoreColumnVecWrite

        """
        cdef PetscInt _i = asInt(i)
        if mode == 'rw':
            CHKERR( MatDenseRestoreColumnVec(self.mat, _i, NULL) )
        elif mode == 'r':
            CHKERR( MatDenseRestoreColumnVecRead(self.mat, _i, NULL) )
        else:
            CHKERR( MatDenseRestoreColumnVecWrite(self.mat, _i, NULL) )

    # Nest

    def getNestSize(self) -> tuple[int, int]:
        """Return the number of rows and columns of the matrix.

        Not collective.

        See Also
        --------
        petsc.MatNestGetSize

        """
        cdef PetscInt nrows, ncols
        CHKERR( MatNestGetSize(self.mat, &nrows, &ncols) )
        return toInt(nrows), toInt(ncols)

    def getNestISs(self) -> tuple[list[IS], list[IS]]:
        """Return the index sets representing the row and column spaces.

        Not collective.

        See Also
        --------
        petsc.MatNestGetISs

        """
        cdef PetscInt i, nrows =0, ncols = 0
        cdef PetscIS *cisrows = NULL
        cdef PetscIS *ciscols = NULL
        CHKERR( MatNestGetSize(self.mat, &nrows, &ncols) )
        cdef object tmpr = oarray_p(empty_p(nrows), NULL, <void**>&cisrows)
        cdef object tmpc = oarray_p(empty_p(ncols), NULL, <void**>&ciscols)
        CHKERR( MatNestGetISs(self.mat, cisrows, ciscols) )
        cdef object isetsrows = [ref_IS(cisrows[i]) for i from 0 <= i < nrows]
        cdef object isetscols = [ref_IS(ciscols[i]) for i from 0 <= i < ncols]
        return isetsrows, isetscols

    def getNestLocalISs(self) -> tuple[list[IS], list[IS]]:
        """Return the local index sets representing the row and column spaces.

        Not collective.

        See Also
        --------
        petsc.MatNestGetLocalISs

        """
        cdef PetscInt i, nrows =0, ncols = 0
        cdef PetscIS *cisrows = NULL
        cdef PetscIS *ciscols = NULL
        CHKERR( MatNestGetSize(self.mat, &nrows, &ncols) )
        cdef object tmpr = oarray_p(empty_p(nrows), NULL, <void**>&cisrows)
        cdef object tmpc = oarray_p(empty_p(ncols), NULL, <void**>&ciscols)
        CHKERR( MatNestGetLocalISs(self.mat, cisrows, ciscols) )
        cdef object isetsrows = [ref_IS(cisrows[i]) for i from 0 <= i < nrows]
        cdef object isetscols = [ref_IS(ciscols[i]) for i from 0 <= i < ncols]
        return isetsrows, isetscols

    def getNestSubMatrix(self, i: int, j: int) -> Mat:
        """Return a single submatrix.

        Not collective.

        Parameters
        ----------
        i
            The first index of the matrix within the nesting.
        j
            The second index of the matrix within the nesting.

        See Also
        --------
        petsc.MatNestGetSubMat

        """
        cdef Mat submat = Mat()
        cdef PetscInt idxm = asInt(i)
        cdef PetscInt jdxm = asInt(j)
        CHKERR( MatNestGetSubMat(self.mat, idxm, jdxm, &submat.mat) )
        PetscINCREF(submat.obj)
        return submat

    # DM

    def getDM(self) -> DM:
        """Return the DM defining the data layout of the matrix.

        Not collective.

        See Also
        --------
        petsc.MatGetDM

        """
        cdef PetscDM newdm = NULL
        CHKERR( MatGetDM(self.mat, &newdm) )
        cdef DM dm = subtype_DM(newdm)()
        dm.dm = newdm
        PetscINCREF(dm.obj)
        return dm

    def setDM(self, DM dm) -> None:
        """Set the DM defining the data layout of the matrix.

        Not collective.

        Parameters
        ----------
        dm
            The `DM`.

        See Also
        --------
        petsc.MatSetDM

        """
        CHKERR( MatSetDM(self.mat, dm.dm) )

    # backward compatibility

    PtAP = ptap

    #

    property sizes:
        """Matrix local and global sizes."""
        def __get__(self) -> tuple[tuple[int, int], tuple[int, int]]:
            return self.getSizes()
        def __set__(self, value):
            self.setSizes(value)

    property size:
        """Matrix global size."""
        def __get__(self) -> tuple[int, int]:
            return self.getSize()

    property local_size:
        """Matrix local size."""
        def __get__(self) -> int:
            return self.getLocalSize()

    property block_size:
        """Matrix block size."""
        def __get__(self) -> int:
            return self.getBlockSize()

    property block_sizes:
        """Matrix row and column block sizes."""
        def __get__(self) -> tuple[int, int]:
            return self.getBlockSizes()

    property owner_range:
        """Matrix local row range."""
        def __get__(self) -> tuple[int, int]:
            return self.getOwnershipRange()

    property owner_ranges:
        """Matrix row ranges."""
        def __get__(self) -> ArrayInt:
            return self.getOwnershipRanges()

    #

    property assembled:
        """The boolean flag indicating if the matrix is assembled."""
        def __get__(self) -> bool:
            return self.isAssembled()
    property symmetric:
        """The boolean flag indicating if the matrix is symmetric."""
        def __get__(self) -> bool:
            return self.isSymmetric()
    property hermitian:
        """The boolean flag indicating if the matrix is Hermitian."""
        def __get__(self) -> bool:
            return self.isHermitian()
    property structsymm:
        """The boolean flag indicating if the matrix is structurally symmetric."""
        def __get__(self) -> bool:
            return self.isStructurallySymmetric()

    # TODO Stream
    def __dlpack__(self, stream=-1):
        return self.toDLPack('rw')

    def __dlpack_device__(self):
        (dltype, devId, _, _, _) = mat_get_dlpack_ctx(self)
        return (dltype, devId)

    def toDLPack(self, mode: AccessModeSpec = 'rw') -> Any:
        """Return a DLPack `PyCapsule` wrapping the vector data."""
        if mode is None: mode = 'rw'
        if mode is None: mode = 'rw'
        if mode not in ['rw', 'r', 'w']:
            raise ValueError("Invalid mode: expected 'rw', 'r', or 'w'")

        cdef int64_t ndim = 0
        (device_type, device_id, ndim, shape, strides) = mat_get_dlpack_ctx(self)
        hostmem = (device_type == kDLCPU)

        cdef DLManagedTensor* dlm_tensor = <DLManagedTensor*>malloc(sizeof(DLManagedTensor))
        cdef DLTensor* dl_tensor = &dlm_tensor.dl_tensor
        cdef PetscScalar *a = NULL
        cdef int64_t* shape_strides = NULL
        dl_tensor.byte_offset = 0

        # DLPack does not currently play well with our get/restore model
        # Call restore right-away and hope that the consumer will do the right thing
        # and not modify memory requested with read access
        # By restoring now, we guarantee the sanity of the ObjectState
        if mode == 'w':
            if hostmem:
                CHKERR( MatDenseGetArrayWrite(self.mat, <PetscScalar**>&a) )
                CHKERR( MatDenseRestoreArrayWrite(self.mat, NULL) )
            else:
                CHKERR( MatDenseCUDAGetArrayWrite(self.mat, <PetscScalar**>&a) )
                CHKERR( MatDenseCUDARestoreArrayWrite(self.mat, NULL) )
        elif mode == 'r':
            if hostmem:
                CHKERR( MatDenseGetArrayRead(self.mat, <const PetscScalar**>&a) )
                CHKERR( MatDenseRestoreArrayRead(self.mat, NULL) )
            else:
                CHKERR( MatDenseCUDAGetArrayRead(self.mat, <const PetscScalar**>&a) )
                CHKERR( MatDenseCUDARestoreArrayRead(self.mat, NULL) )
        else:
            if hostmem:
                CHKERR( MatDenseGetArray(self.mat, <PetscScalar**>&a) )
                CHKERR( MatDenseRestoreArray(self.mat, NULL) )
            else:
                CHKERR( MatDenseCUDAGetArray(self.mat, <PetscScalar**>&a) )
                CHKERR( MatDenseCUDARestoreArray(self.mat, NULL) )
        dl_tensor.data = <void *>a

        cdef DLContext* ctx = &dl_tensor.ctx
        ctx.device_type = device_type
        ctx.device_id = device_id
        shape_strides = <int64_t*>malloc(sizeof(int64_t)*2*ndim)
        for i in range(ndim):
            shape_strides[i] = shape[i]
        for i in range(ndim):
            shape_strides[i+ndim] = strides[i]
        dl_tensor.ndim = ndim
        dl_tensor.shape = shape_strides
        dl_tensor.strides = shape_strides + ndim

        cdef DLDataType* dtype = &dl_tensor.dtype
        dtype.code = <uint8_t>DLDataTypeCode.kDLFloat
        if sizeof(PetscScalar) == 8:
            dtype.bits = <uint8_t>64
        elif sizeof(PetscScalar) == 4:
            dtype.bits = <uint8_t>32
        else:
            raise ValueError('Unsupported PetscScalar type')
        dtype.lanes = <uint16_t>1
        dlm_tensor.manager_ctx = <void *>self.mat
        CHKERR( PetscObjectReference(<PetscObject>self.mat) )
        dlm_tensor.manager_deleter = manager_deleter
        dlm_tensor.del_obj = <dlpack_manager_del_obj>PetscDEALLOC
        return PyCapsule_New(dlm_tensor, 'dltensor', pycapsule_deleter)

# --------------------------------------------------------------------

cdef class NullSpace(Object):
    """Nullspace object.

    See Also
    --------
    petsc.MatNullSpace

    """
    #

    def __cinit__(self):
        self.obj  = <PetscObject*> &self.nsp
        self.nsp = NULL

    def __call__(self, vec):
        self.remove(vec)

    #

    def view(self, Viewer viewer=None) -> None:
        """View the null space.

        Collective.

        Parameters
        ----------
        viewer
            A `Viewer` instance or `None` for the default viewer.

        See Also
        --------
        Viewer, petsc.MatNullSpaceView

        """
        cdef PetscViewer vwr = NULL
        if viewer is not None: vwr = viewer.vwr
        CHKERR( MatNullSpaceView(self.nsp, vwr) )

    def destroy(self) -> Self:
        """Destroy the null space.

        Collective.

        See Also
        --------
        create, petsc.MatNullSpaceDestroy

        """
        CHKERR( MatNullSpaceDestroy(&self.nsp) )
        return self

    def create(
        self,
        constant: bool = False,
        vectors: Sequence[Vec] = (),
        comm=None
        ) -> Self:
        """Create the null space.

        Collective.

        Parameters
        ----------
        constant
            A flag to indicate the null space contains the constant vector.
        vectors
            The sequence of vectors that span the null space, excluding the constant vector.
        comm
            MPI communicator, defaults to `Sys.getDefaultComm`.

        See Also
        --------
        destroy, petsc.MatNullSpaceCreate

        """
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscBool has_const = PETSC_FALSE
        if constant: has_const = PETSC_TRUE
        cdef PetscInt i = 0, nv = <PetscInt>len(vectors)
        cdef PetscVec *v = NULL
        cdef object tmp2 = oarray_p(empty_p(nv), NULL, <void**>&v)
        for i from 0 <= i < nv:
            v[i] = (<Vec?>(vectors[<Py_ssize_t>i])).vec
        cdef PetscNullSpace newnsp = NULL
        CHKERR( MatNullSpaceCreate(ccomm, has_const, nv, v, &newnsp) )
        PetscCLEAR(self.obj); self.nsp = newnsp
        return self

    def createRigidBody(self, Vec coords) -> Self:
        """Create rigid body modes from coordinates.

        Parameters
        ----------
        coords
            The block coordinates of each node. This requires the block size to have been set.

        See Also
        --------
        petsc.MatNullSpaceCreateRigidBody

        """
        cdef PetscNullSpace newnsp = NULL
        CHKERR( MatNullSpaceCreateRigidBody(coords.vec, &newnsp) )
        PetscCLEAR(self.obj); self.nsp = newnsp
        return self

    def setFunction(
        self,
        function: MatNullFunction,
        args: tuple[Any, ...] | None = None,
        kargs: dict[str, Any] | None = None,
        ) -> None:
        """Set the callback to remove the nullspace.

        Logically collective.

        Parameters
        ----------
        function
            The callback.
        args
            Positional arguments for the callback.
        kargs
            Keyword arguments for the callback.

        See Also
        --------
        getFunction, petsc.MatNullSpaceSetFunction

        """
        if function is not None:
            CHKERR( MatNullSpaceSetFunction(
                    self.nsp, NullSpace_Function, NULL) )
            if args is None: args = ()
            if kargs is None: kargs = {}
            self.set_attr('__function__', (function, args, kargs))
        else:
            CHKERR( MatNullSpaceSetFunction(self.nsp, NULL, NULL) )
            self.set_attr('__function__', None)
    #

    def hasConstant(self) -> bool:
        """Return whether the null space contains the constant.

        See Also
        --------
        petsc.MatNullSpaceGetVecs

        """
        cdef PetscBool flag = PETSC_FALSE
        CHKERR( MatNullSpaceGetVecs(self.nsp, &flag, NULL, NULL) )
        return toBool(flag)

    def getVecs(self) -> list[Vec]:
        """Return the vectors defining the null space.

        Not collective.

        See Also
        --------
        petsc.MatNullSpaceGetVecs

        """
        cdef PetscInt i = 0, nv = 0
        cdef const PetscVec *v = NULL
        CHKERR( MatNullSpaceGetVecs(self.nsp, NULL, &nv, &v) )
        cdef Vec vec = None
        cdef list vectors = []
        for i from 0 <= i < nv:
            vec = Vec()
            vec.vec = v[i]
            PetscINCREF(vec.obj)
            vectors.append(vec)
        return vectors

    def getFunction(self) -> MatNullFunction:
        """Return the callback to remove the nullspace.

        Not collective.

        See Also
        --------
        setFunction

        """
        return self.get_attr('__function__')

    #

    def remove(self, Vec vec) -> None:
        """Remove all components of a null space from a vector.

        Collective.

        Parameters
        ----------
        vec
            The vector from which the null space is removed.

        See Also
        --------
        petsc.MatNullSpaceRemove

        """
        CHKERR( MatNullSpaceRemove(self.nsp, vec.vec) )

    def test(self, Mat mat) -> bool:
        """Return if the claimed null space is valid for a matrix.

        Collective.

        Parameters
        ----------
        mat
            The matrix to check.

        See Also
        --------
        petsc.MatNullSpaceTest

        """
        cdef PetscBool flag = PETSC_FALSE
        CHKERR( MatNullSpaceTest(self.nsp, mat.mat, &flag) )
        return toBool(flag)

# --------------------------------------------------------------------

del MatType
del MatOption
del MatAssemblyType
del MatInfoType
del MatStructure
del MatDuplicateOption
del MatOrderingType
del MatSolverType
del MatFactorShiftType
del MatSORType

# --------------------------------------------------------------------
