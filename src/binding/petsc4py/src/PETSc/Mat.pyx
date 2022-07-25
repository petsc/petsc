# --------------------------------------------------------------------

class MatType(object):
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
    TRANSPOSEMAT    = S_(MATTRANSPOSEMAT)
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
    # native
    FINAL_ASSEMBLY = MAT_FINAL_ASSEMBLY
    FLUSH_ASSEMBLY = MAT_FLUSH_ASSEMBLY
    # aliases
    FINAL = FINAL_ASSEMBLY
    FLUSH = FLUSH_ASSEMBLY

class MatInfoType(object):
    LOCAL = MAT_LOCAL
    GLOBAL_MAX = MAT_GLOBAL_MAX
    GLOBAL_SUM = MAT_GLOBAL_SUM

class MatStructure(object):
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
    DO_NOT_COPY_VALUES    = MAT_DO_NOT_COPY_VALUES
    COPY_VALUES           = MAT_COPY_VALUES
    SHARE_NONZERO_PATTERN = MAT_SHARE_NONZERO_PATTERN

class MatOrderingType(object):
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
    SUPERLU         = S_(MATSOLVERSUPERLU)
    SUPERLU_DIST    = S_(MATSOLVERSUPERLU_DIST)
    STRUMPACK       = S_(MATSOLVERSTRUMPACK)
    UMFPACK         = S_(MATSOLVERUMFPACK)
    CHOLMOD         = S_(MATSOLVERCHOLMOD)
    KLU             = S_(MATSOLVERKLU)
    SPARSEELEMENTAL = S_(MATSOLVERSPARSEELEMENTAL)
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
    # native
    NONE              = MAT_SHIFT_NONE
    NONZERO           = MAT_SHIFT_NONZERO
    POSITIVE_DEFINITE = MAT_SHIFT_POSITIVE_DEFINITE
    INBLOCKS          = MAT_SHIFT_INBLOCKS
    # aliases
    NZ = MAT_SHIFT_NONZERO
    PD = MAT_SHIFT_POSITIVE_DEFINITE

class MatSORType(object):
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

# --------------------------------------------------------------------

cdef class Mat(Object):

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

    def view(self, Viewer viewer=None):
        cdef PetscViewer vwr = NULL
        if viewer is not None: vwr = viewer.vwr
        CHKERR( MatView(self.mat, vwr) )

    def destroy(self):
        CHKERR( MatDestroy(&self.mat) )
        return self

    def create(self, comm=None):
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscMat newmat = NULL
        CHKERR( MatCreate(ccomm, &newmat) )
        PetscCLEAR(self.obj); self.mat = newmat
        return self

    def setType(self, mat_type):
        cdef PetscMatType cval = NULL
        mat_type = str2bytes(mat_type, &cval)
        CHKERR( MatSetType(self.mat, cval) )

    def setSizes(self, size, bsize=None):
        cdef PetscInt rbs = 0, cbs = 0, m = 0, n = 0, M = 0, N = 0
        Mat_Sizes(size, bsize, &rbs, &cbs, &m, &n, &M, &N)
        CHKERR( MatSetSizes(self.mat, m, n, M, N) )
        if rbs != PETSC_DECIDE:
            if cbs != PETSC_DECIDE:
                CHKERR( MatSetBlockSizes(self.mat, rbs, cbs) )
            else:
                CHKERR( MatSetBlockSize(self.mat, rbs) )

    def setBlockSize(self, bsize):
        cdef PetscInt bs = asInt(bsize)
        CHKERR( MatSetBlockSize(self.mat, bs) )

    def setBlockSizes(self, row_bsize, col_bsize):
        cdef PetscInt rbs = asInt(row_bsize)
        cdef PetscInt cbs = asInt(col_bsize)
        CHKERR( MatSetBlockSizes(self.mat, rbs, cbs) )

    def setVecType(self, vec_type):
        cdef PetscVecType cval = NULL
        vec_type = str2bytes(vec_type, &cval)
        CHKERR( MatSetVecType(self.mat, cval) )

    def getVecType(self):
        cdef PetscVecType cval = NULL
        CHKERR( MatGetVecType(self.mat, &cval) )
        return bytes2str(cval)

    #

    def createAIJ(self, size, bsize=None, nnz=None, csr=None, comm=None):
        # create matrix
        cdef PetscMat newmat = NULL
        Mat_Create(MATAIJ, comm, size, bsize, &newmat)
        PetscCLEAR(self.obj); self.mat = newmat
        # preallocate matrix
        Mat_AllocAIJ(self.mat, nnz, csr)
        return self

    def createBAIJ(self, size, bsize, nnz=None, csr=None, comm=None):
        # create matrix
        cdef PetscMat newmat = NULL
        Mat_Create(MATBAIJ, comm, size, bsize, &newmat)
        PetscCLEAR(self.obj); self.mat = newmat
        # preallocate matrix
        Mat_AllocAIJ(self.mat, nnz, csr)
        return self

    def createSBAIJ(self, size, bsize, nnz=None, csr=None, comm=None):
        # create matrix
        cdef PetscMat newmat = NULL
        Mat_Create(MATSBAIJ, comm, size, bsize, &newmat)
        PetscCLEAR(self.obj); self.mat = newmat
        # preallocate matrix
        Mat_AllocAIJ(self.mat, nnz, csr)
        return self

    def createAIJCRL(self, size, bsize=None, nnz=None, csr=None, comm=None):
        # create matrix
        cdef PetscMat newmat = NULL
        Mat_Create(MATAIJCRL, comm, size, bsize, &newmat)
        PetscCLEAR(self.obj); self.mat = newmat
        # preallocate matrix
        Mat_AllocAIJ(self.mat, nnz, csr)
        return self

    def setPreallocationNNZ(self, nnz):
        cdef PetscBool done = PETSC_FALSE
        CHKERR( MatIsPreallocated(self.mat, &done) )
        # if done: raise Error(PETSC_ERR_ORDER)
        Mat_AllocAIJ_NNZ(self.mat, nnz)
        return self

    def setPreallocationCSR(self, csr):
        cdef PetscBool done = PETSC_FALSE
        CHKERR( MatIsPreallocated(self.mat, &done) )
        # if done: raise Error(PETSC_ERR_ORDER)
        Mat_AllocAIJ_CSR(self.mat, csr)
        return self

    def createAIJWithArrays(self, size, csr, bsize=None, comm=None):
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

    def createDense(self, size, bsize=None, array=None, comm=None):
        # create matrix
        cdef PetscMat newmat = NULL
        Mat_Create(MATDENSE, comm, size, bsize, &newmat)
        PetscCLEAR(self.obj); self.mat = newmat
        # preallocate matrix
        if array is not None:
            array = Mat_AllocDense(self.mat, array)
            self.set_attr('__array__', array)
        return self

    def createDenseCUDA(self, size, bsize=None, array=None, comm=None):
        # create matrix
        cdef PetscMat newmat = NULL
        Mat_Create(MATDENSECUDA, comm, size, bsize, &newmat)
        PetscCLEAR(self.obj); self.mat = newmat
        # preallocate matrix
        if array is not None:
            raise RuntimeError('passing GPU data not yet supported. Report it if you need this feature')
        return self

    def setPreallocationDense(self, array):
        cdef PetscBool done = PETSC_FALSE
        CHKERR( MatIsPreallocated(self.mat, &done) )
        # if done: raise Error(PETSC_ERR_ORDER)
        array = Mat_AllocDense(self.mat, array)
        self.set_attr('__array__', array)
        return self

    #

    def createScatter(self, Scatter scatter, comm=None):
        if comm is None: comm = scatter.getComm()
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscMat newmat = NULL
        CHKERR( MatCreateScatter(ccomm, scatter.sct, &newmat) )
        PetscCLEAR(self.obj); self.mat = newmat
        return self

    def createNormal(self, Mat mat):
        cdef PetscMat newmat = NULL
        CHKERR( MatCreateNormal(mat.mat, &newmat) )
        PetscCLEAR(self.obj); self.mat = newmat
        return self

    def createTranspose(self, Mat mat):
        cdef PetscMat newmat = NULL
        CHKERR( MatCreateTranspose(mat.mat, &newmat) )
        PetscCLEAR(self.obj); self.mat = newmat
        return self

    def createNormalHermitian(self, Mat mat):
        cdef PetscMat newmat = NULL
        CHKERR( MatCreateNormalHermitian(mat.mat, &newmat) )
        PetscCLEAR(self.obj); self.mat = newmat
        return self

    def createHermitianTranspose(self, Mat mat):
        cdef PetscMat newmat = NULL
        CHKERR( MatCreateHermitianTranspose(mat.mat, &newmat) )
        PetscCLEAR(self.obj); self.mat = newmat
        return self

    def createLRC(self, Mat A or None, Mat U, Vec c or None, Mat V or None):
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

    def createSubMatrixVirtual(self, Mat A, IS isrow, IS iscol=None):
        if iscol is None: iscol = isrow
        cdef PetscMat newmat = NULL
        CHKERR( MatCreateSubMatrixVirtual(A.mat, isrow.iset, iscol.iset, &newmat) )
        PetscCLEAR(self.obj); self.mat = newmat
        return self

    def createNest(self, mats, isrows=None, iscols=None, comm=None):
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

    def createH2OpusFromMat(self, Mat A, coordinates=None, dist=None, eta=None, leafsize=None, maxrank=None, bs=None, rtol=None):
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

    def createIS(self, size, LGMap lgmapr=None, LGMap lgmapc=None, comm=None):
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

    def createPython(self, size, context=None, comm=None):
        # communicator and sizes
        cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
        cdef PetscInt rbs = 0, cbs = 0, m = 0, n = 0, M = 0, N = 0
        Mat_Sizes(size, None, &rbs, &cbs, &m, &n, &M, &N)
        Sys_Layout(ccomm, rbs, &m, &M)
        Sys_Layout(ccomm, cbs, &n, &N)
        # create matrix
        cdef PetscMat newmat = NULL
        CHKERR( MatCreate(ccomm, &newmat) )
        PetscCLEAR(self.obj); self.mat = newmat
        CHKERR( MatSetSizes(self.mat, m, n, M, N) )
        CHKERR( MatSetType(self.mat, MATPYTHON) )
        CHKERR( MatPythonSetContext(self.mat, <void*>context) )
        return self

    def setPythonContext(self, context):
        CHKERR( MatPythonSetContext(self.mat, <void*>context) )

    def getPythonContext(self):
        cdef void *context = NULL
        CHKERR( MatPythonGetContext(self.mat, &context) )
        if context == NULL: return None
        else: return <object> context

    def setPythonType(self, py_type):
        cdef const char *cval = NULL
        py_type = str2bytes(py_type, &cval)
        CHKERR( MatPythonSetType(self.mat, cval) )

    #

    def setOptionsPrefix(self, prefix):
        cdef const char *cval = NULL
        prefix = str2bytes(prefix, &cval)
        CHKERR( MatSetOptionsPrefix(self.mat, cval) )

    def getOptionsPrefix(self):
        cdef const char *cval = NULL
        CHKERR( MatGetOptionsPrefix(self.mat, &cval) )
        return bytes2str(cval)

    def appendOptionsPrefix(self, prefix):
        cdef const char *cval = NULL
        prefix = str2bytes(prefix, &cval)
        CHKERR( MatAppendOptionsPrefix(self.mat, cval) )

    def setFromOptions(self):
        CHKERR( MatSetFromOptions(self.mat) )

    def setUp(self):
        CHKERR( MatSetUp(self.mat) )
        return self

    def setOption(self, option, flag):
        CHKERR( MatSetOption(self.mat, option, flag) )

    def getOption(self, option):
        cdef PetscBool flag = PETSC_FALSE
        CHKERR( MatGetOption(self.mat, option, &flag) )
        return toBool(flag)

    def getType(self):
        cdef PetscMatType cval = NULL
        CHKERR( MatGetType(self.mat, &cval) )
        return bytes2str(cval)

    def getSize(self):
        cdef PetscInt M = 0, N = 0
        CHKERR( MatGetSize(self.mat, &M, &N) )
        return (toInt(M), toInt(N))

    def getLocalSize(self):
        cdef PetscInt m = 0, n = 0
        CHKERR( MatGetLocalSize(self.mat, &m, &n) )
        return (toInt(m), toInt(n))

    def getSizes(self):
        cdef PetscInt m = 0, n = 0
        cdef PetscInt M = 0, N = 0
        CHKERR( MatGetLocalSize(self.mat, &m, &n) )
        CHKERR( MatGetSize(self.mat, &M, &N) )
        return ((toInt(m), toInt(M)), (toInt(n), toInt(N)))

    def getBlockSize(self):
        cdef PetscInt bs = 0
        CHKERR( MatGetBlockSize(self.mat, &bs) )
        return toInt(bs)

    def getBlockSizes(self):
        cdef PetscInt rbs = 0, cbs = 0
        CHKERR( MatGetBlockSizes(self.mat, &rbs, &cbs) )
        return (toInt(rbs), toInt(cbs))

    def getOwnershipRange(self):
        cdef PetscInt ival1 = 0, ival2 = 0
        CHKERR( MatGetOwnershipRange(self.mat, &ival1, &ival2) )
        return (toInt(ival1), toInt(ival2))

    def getOwnershipRanges(self):
        cdef const PetscInt *rowrng = NULL
        CHKERR( MatGetOwnershipRanges(self.mat, &rowrng) )
        cdef MPI_Comm comm = MPI_COMM_NULL
        CHKERR( PetscObjectGetComm(<PetscObject>self.mat, &comm) )
        cdef int size = -1
        CHKERR( MPI_Comm_size(comm, &size) )
        return array_i(size+1, rowrng)

    def getOwnershipRangeColumn(self):
        cdef PetscInt ival1 = 0, ival2 = 0
        CHKERR( MatGetOwnershipRangeColumn(self.mat, &ival1, &ival2) )
        return (toInt(ival1), toInt(ival2))

    def getOwnershipRangesColumn(self):
        cdef const PetscInt *colrng = NULL
        CHKERR( MatGetOwnershipRangesColumn(self.mat, &colrng) )
        cdef MPI_Comm comm = MPI_COMM_NULL
        CHKERR( PetscObjectGetComm(<PetscObject>self.mat, &comm) )
        cdef int size = -1
        CHKERR( MPI_Comm_size(comm, &size) )
        return array_i(size+1, colrng)

    def getOwnershipIS(self):
        cdef IS rows = IS()
        cdef IS cols = IS()
        CHKERR( MatGetOwnershipIS(self.mat, &rows.iset, &cols.iset) )
        return (rows, cols)

    def getInfo(self, info=None):
        cdef PetscMatInfoType itype = infotype(info)
        cdef PetscMatInfo cinfo
        CHKERR( MatGetInfo(self.mat, itype, &cinfo) )
        return cinfo

    def duplicate(self, copy=False):
        cdef PetscMatDuplicateOption flag = MAT_DO_NOT_COPY_VALUES
        if copy: flag = MAT_COPY_VALUES
        if copy > MAT_COPY_VALUES: flag = MAT_SHARE_NONZERO_PATTERN
        cdef Mat mat = type(self)()
        CHKERR( MatDuplicate(self.mat, flag, &mat.mat) )
        return mat

    def copy(self, Mat result=None, structure=None):
        cdef PetscMatDuplicateOption copy = MAT_COPY_VALUES
        cdef PetscMatStructure mstr = matstructure(structure)
        if result is None:
            result = type(self)()
        if result.mat == NULL:
            CHKERR( MatDuplicate(self.mat, copy, &result.mat) )
        else:
            CHKERR( MatCopy(self.mat, result.mat, mstr) )
        return result

    def load(self, Viewer viewer):
        cdef MPI_Comm comm = MPI_COMM_NULL
        cdef PetscObject obj = <PetscObject>(viewer.vwr)
        if self.mat == NULL:
            CHKERR( PetscObjectGetComm(obj, &comm) )
            CHKERR( MatCreate(comm, &self.mat) )
        CHKERR( MatLoad(self.mat, viewer.vwr) )
        return self

    def convert(self, mat_type=None, Mat out=None):
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

    def transpose(self, Mat out=None):
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

    def setTransposePrecursor(self, Mat out):
        CHKERR( MatTransposeSetPrecursor(self.mat, out.mat) )
        return out

    def hermitianTranspose(self, Mat out=None):
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

    def realPart(self, Mat out=None):
        if out is None:
            out = self
        elif out.mat == NULL:
            CHKERR( MatDuplicate(self.mat, MAT_COPY_VALUES, &out.mat) )
        CHKERR( MatRealPart(out.mat) )
        return out

    def imagPart(self, Mat out=None):
        if out is None:
            out = self
        elif out.mat == NULL:
            CHKERR( MatDuplicate(self.mat, MAT_COPY_VALUES, &out.mat) )
        CHKERR( MatImaginaryPart(out.mat) )
        return out

    def conjugate(self, Mat out=None):
        if out is None:
            out = self
        elif out.mat == NULL:
            CHKERR( MatDuplicate(self.mat, MAT_COPY_VALUES, &out.mat) )
        CHKERR( MatConjugate(out.mat) )
        return out

    def permute(self, IS row, IS col):
        cdef Mat mat = Mat()
        CHKERR( MatPermute(self.mat, row.iset, col.iset, &mat.mat) )
        return mat

    def equal(self, Mat mat):
        cdef PetscBool flag = PETSC_FALSE
        CHKERR( MatEqual(self.mat, mat.mat, &flag) )
        return toBool(flag)

    def isTranspose(self, Mat mat=None, tol=0):
        if mat is None: mat = self
        cdef PetscReal rval = asReal(tol)
        cdef PetscBool flag = PETSC_FALSE
        CHKERR( MatIsTranspose(self.mat, mat.mat, rval, &flag) )
        return toBool(flag)

    def isSymmetric(self, tol=0):
        cdef PetscReal rval = asReal(tol)
        cdef PetscBool flag = PETSC_FALSE
        CHKERR( MatIsSymmetric(self.mat, rval, &flag) )
        return toBool(flag)

    def isSymmetricKnown(self):
        cdef PetscBool flag1 = PETSC_FALSE
        cdef PetscBool flag2 = PETSC_FALSE
        CHKERR( MatIsSymmetricKnown(self.mat, &flag1, &flag2) )
        return (toBool(flag1), toBool(flag2))

    def isHermitian(self, tol=0):
        cdef PetscReal rval = asReal(tol)
        cdef PetscBool flag = PETSC_FALSE
        CHKERR( MatIsHermitian(self.mat, rval, &flag) )
        return toBool(flag)

    def isHermitianKnown(self):
        cdef PetscBool flag1 = PETSC_FALSE
        cdef PetscBool flag2 = PETSC_FALSE
        CHKERR( MatIsHermitianKnown(self.mat, &flag1, &flag2) )
        return (toBool(flag1), toBool(flag2))

    def isStructurallySymmetric(self):
        cdef PetscBool flag = PETSC_FALSE
        CHKERR( MatIsStructurallySymmetric(self.mat, &flag) )
        return toBool(flag)

    def zeroEntries(self):
        CHKERR( MatZeroEntries(self.mat) )

    def getValue(self, row, col):
        cdef PetscInt    ival1 = asInt(row)
        cdef PetscInt    ival2 = asInt(col)
        cdef PetscScalar sval  = 0
        CHKERR( MatGetValues(self.mat, 1, &ival1, 1, &ival2, &sval) )
        return toScalar(sval)

    def getValues(self, rows, cols, values=None):
        return matgetvalues(self.mat, rows, cols, values)

    def getValuesCSR(self):
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

    def getRow(self, row):
        cdef PetscInt irow = asInt(row)
        cdef PetscInt ncols = 0
        cdef const PetscInt *icols=NULL
        cdef const PetscScalar *svals=NULL
        CHKERR( MatGetRow(self.mat, irow, &ncols, &icols, &svals) )
        cdef object cols = array_i(ncols, icols)
        cdef object vals = array_s(ncols, svals)
        CHKERR( MatRestoreRow(self.mat, irow, &ncols, &icols, &svals) )
        return (cols, vals)

    def getRowIJ(self, symmetric=False, compressed=False):
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

    def getColumnIJ(self, symmetric=False, compressed=False):
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

    def setValue(self, row, col, value, addv=None):
        cdef PetscInt    ival1 = asInt(row)
        cdef PetscInt    ival2 = asInt(col)
        cdef PetscScalar sval  = asScalar(value)
        cdef PetscInsertMode caddv = insertmode(addv)
        CHKERR( MatSetValues(self.mat, 1, &ival1, 1, &ival2, &sval, caddv) )

    def setValues(self, rows, cols, values, addv=None):
        matsetvalues(self.mat, rows, cols, values, addv, 0, 0)

    def setValuesRCV(self, R, C, V, addv=None):
        matsetvalues_rcv(self.mat, R, C, V, addv, 0, 0)

    def setValuesIJV(self, I, J, V, addv=None, rowmap=None):
        matsetvalues_ijv(self.mat, I, J, V, addv, rowmap, 0, 0)

    def setValuesCSR(self, I, J, V, addv=None):
        matsetvalues_csr(self.mat, I, J, V, addv, 0, 0)

    def setValuesBlocked(self, rows, cols, values, addv=None):
        matsetvalues(self.mat, rows, cols, values, addv, 1, 0)

    def setValuesBlockedRCV(self, R, C, V, addv=None):
        matsetvalues_rcv(self.mat, R, C, V, addv, 1, 0)

    def setValuesBlockedIJV(self, I, J, V, addv=None, rowmap=None):
        matsetvalues_ijv(self.mat, I, J, V, addv, rowmap, 1, 0)

    def setValuesBlockedCSR(self, I, J, V, addv=None):
        matsetvalues_csr(self.mat, I, J, V, addv, 1, 0)

    def setLGMap(self, LGMap rmap, LGMap cmap=None):
        if cmap is None: cmap = rmap
        CHKERR( MatSetLocalToGlobalMapping(self.mat, rmap.lgm, cmap.lgm) )

    def getLGMap(self):
        cdef LGMap cmap = LGMap()
        cdef LGMap rmap = LGMap()
        CHKERR( MatGetLocalToGlobalMapping(self.mat, &rmap.lgm, &cmap.lgm) )
        PetscINCREF(cmap.obj)
        PetscINCREF(rmap.obj)
        return (rmap, cmap)

    def setValueLocal(self, row, col, value, addv=None):
        cdef PetscInt    ival1 = asInt(row)
        cdef PetscInt    ival2 = asInt(col)
        cdef PetscScalar sval  = asScalar(value)
        cdef PetscInsertMode caddv = insertmode(addv)
        CHKERR( MatSetValuesLocal(
                self.mat, 1, &ival1, 1, &ival2, &sval, caddv) )

    def setValuesLocal(self, rows, cols, values, addv=None):
        matsetvalues(self.mat, rows, cols, values, addv, 0, 1)

    def setValuesLocalRCV(self, R, C, V, addv=None):
        matsetvalues_rcv(self.mat, R, C, V, addv, 0, 1)

    def setValuesLocalIJV(self, I, J, V, addv=None, rowmap=None):
        matsetvalues_ijv(self.mat, I, J, V, addv, rowmap, 0, 1)

    def setValuesLocalCSR(self, I, J, V, addv=None):
        matsetvalues_csr(self.mat, I, J, V, addv, 0, 1)

    def setValuesBlockedLocal(self, rows, cols, values, addv=None):
        matsetvalues(self.mat, rows, cols, values, addv, 1, 1)

    def setValuesBlockedLocalRCV(self, R, C, V, addv=None):
        matsetvalues_rcv(self.mat, R, C, V, addv, 1, 1)

    def setValuesBlockedLocalIJV(self, I, J, V, addv=None, rowmap=None):
        matsetvalues_ijv(self.mat, I, J, V, addv, rowmap, 1, 1)

    def setValuesBlockedLocalCSR(self, I, J, V, addv=None):
        matsetvalues_csr(self.mat, I, J, V, addv, 1, 1)

    #

    Stencil = _Mat_Stencil

    def setStencil(self, dims, starts=None, dof=1):
        cdef PetscInt ndim, ndof
        cdef PetscInt cdims[3], cstarts[3]
        cdims[0] = cdims[1] = cdims[2] = 1
        cstarts[0] = cstarts[1] = cstarts[2] = 0
        ndim = asDims(dims, &cdims[0], &cdims[1], &cdims[2])
        ndof = asInt(dof)
        if starts is not None:
            asDims(dims, &cstarts[0], &cstarts[1], &cstarts[2])
        CHKERR( MatSetStencil(self.mat, ndim, cdims, cstarts, ndof) )

    def setValueStencil(self, row, col, value, addv=None):
        cdef _Mat_Stencil r = row, c = col
        cdef PetscInsertMode im = insertmode(addv)
        matsetvaluestencil(self.mat, r, c, value, im, 0)

    def setValueStagStencil(self, row, col, value, addv=None):
        raise NotImplementedError('setValueStagStencil not yet implemented in petsc4py')

    def setValueBlockedStencil(self, row, col, value, addv=None):
        cdef _Mat_Stencil r = row, c = col
        cdef PetscInsertMode im = insertmode(addv)
        matsetvaluestencil(self.mat, r, c, value, im, 1)

    def setValueBlockedStagStencil(self, row, col, value, addv=None):
        raise NotImplementedError('setValueBlockedStagStencil not yet implemented in petsc4py')

    def zeroRows(self, rows, diag=1, Vec x=None, Vec b=None):
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

    def zeroRowsLocal(self, rows, diag=1, Vec x=None, Vec b=None):
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

    def zeroRowsColumns(self, rows, diag=1, Vec x=None, Vec b=None):
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

    def zeroRowsColumnsLocal(self, rows, diag=1, Vec x=None, Vec b=None):
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

    def zeroRowsColumnsStencil(self, rows, diag=1, Vec x=None, Vec b=None):
        cdef PetscScalar sval = asScalar(diag)
        cdef PetscInt nrows = asInt(len(rows))
        cdef PetscMatStencil st
        cdef _Mat_Stencil r
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

    def storeValues(self):
        CHKERR( MatStoreValues(self.mat) )

    def retrieveValues(self):
        CHKERR( MatRetrieveValues(self.mat) )

    def assemblyBegin(self, assembly=None):
        cdef PetscMatAssemblyType flag = assemblytype(assembly)
        CHKERR( MatAssemblyBegin(self.mat, flag) )

    def assemblyEnd(self, assembly=None):
        cdef PetscMatAssemblyType flag = assemblytype(assembly)
        CHKERR( MatAssemblyEnd(self.mat, flag) )

    def assemble(self, assembly=None):
        cdef PetscMatAssemblyType flag = assemblytype(assembly)
        CHKERR( MatAssemblyBegin(self.mat, flag) )
        CHKERR( MatAssemblyEnd(self.mat, flag) )

    def isAssembled(self):
        cdef PetscBool flag = PETSC_FALSE
        CHKERR( MatAssembled(self.mat, &flag) )
        return toBool(flag)

    def findZeroRows(self):
        cdef IS zerorows = IS()
        CHKERR( MatFindZeroRows(self.mat, &zerorows.iset) )
        return zerorows

    def createVecs(self, side=None):
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

    def createVecRight(self):
        cdef Vec vecr = Vec()
        CHKERR( MatCreateVecs(self.mat, &vecr.vec, NULL) )
        return vecr

    def createVecLeft(self):
        cdef Vec vecl = Vec()
        CHKERR( MatCreateVecs(self.mat, NULL, &vecl.vec) )
        return vecl

    getVecs = createVecs
    getVecRight = createVecRight
    getVecLeft = createVecLeft

    #

    def getColumnVector(self, column, Vec result=None):
        cdef PetscInt ival = asInt(column)
        if result is None:
            result = Vec()
        if result.vec == NULL:
            CHKERR( MatCreateVecs(self.mat, NULL, &result.vec) )
        CHKERR( MatGetColumnVector(self.mat, result.vec, ival) )
        return result

    def getRedundantMatrix(self, nsubcomm, subcomm=None, Mat out=None):
        cdef PetscInt _nsubcomm   = asInt(nsubcomm)
        cdef MPI_Comm _subcomm    = MPI_COMM_NULL
        if subcomm:   _subcomm    = def_Comm(subcomm, PETSC_COMM_DEFAULT)
        cdef PetscMatReuse reuse  = MAT_INITIAL_MATRIX
        if out is None: out       = Mat()
        if out.mat != NULL: reuse = MAT_REUSE_MATRIX
        CHKERR( MatCreateRedundantMatrix(self.mat, _nsubcomm, _subcomm, reuse, &out.mat))
        return out

    def getDiagonal(self, Vec result=None):
        if result is None:
            result = Vec()
        if result.vec == NULL:
            CHKERR( MatCreateVecs(self.mat, NULL, &result.vec) )
        CHKERR( MatGetDiagonal(self.mat, result.vec) )
        return result

    def getRowSum(self, Vec result=None):
        if result is None:
            result = Vec()
        if result.vec == NULL:
            CHKERR( MatCreateVecs(self.mat, NULL, &result.vec) )
        CHKERR( MatGetRowSum(self.mat, result.vec) )
        return result

    def setDiagonal(self, Vec diag, addv=None):
        cdef PetscInsertMode caddv = insertmode(addv)
        CHKERR( MatDiagonalSet(self.mat, diag.vec, caddv) )

    def diagonalScale(self, Vec L=None, Vec R=None):
        cdef PetscVec vecl=NULL, vecr=NULL
        if L is not None: vecl = L.vec
        if R is not None: vecr = R.vec
        CHKERR( MatDiagonalScale(self.mat, vecl, vecr) )

    def invertBlockDiagonal(self):
        cdef PetscInt bs = 0, m = 0
        cdef const PetscScalar *cibdiag = NULL
        CHKERR( MatGetBlockSize(self.mat, &bs) )
        CHKERR( MatGetLocalSize(self.mat, &m, NULL) )
        CHKERR( MatInvertBlockDiagonal(self.mat, &cibdiag) )
        cdef ndarray ibdiag = array_s(m*bs, cibdiag)
        ibdiag.shape = (toInt(m//bs), toInt(bs), toInt(bs))
        return ibdiag.transpose(0, 2, 1)

    # null space

    def setNullSpace(self, NullSpace nsp):
        CHKERR( MatSetNullSpace(self.mat, nsp.nsp) )

    def getNullSpace(self):
        cdef NullSpace nsp = NullSpace()
        CHKERR( MatGetNullSpace(self.mat, &nsp.nsp) )
        PetscINCREF(nsp.obj)
        return nsp

    def setTransposeNullSpace(self, NullSpace nsp):
        CHKERR( MatSetTransposeNullSpace(self.mat, nsp.nsp) )

    def getTransposeNullSpace(self):
        cdef NullSpace nsp = NullSpace()
        CHKERR( MatGetTransposeNullSpace(self.mat, &nsp.nsp) )
        PetscINCREF(nsp.obj)
        return nsp

    def setNearNullSpace(self, NullSpace nsp):
        CHKERR( MatSetNearNullSpace(self.mat, nsp.nsp) )

    def getNearNullSpace(self):
        cdef NullSpace nsp = NullSpace()
        CHKERR( MatGetNearNullSpace(self.mat, &nsp.nsp) )
        PetscINCREF(nsp.obj)
        return nsp

    # matrix-vector product

    def mult(self, Vec x, Vec y):
        CHKERR( MatMult(self.mat, x.vec, y.vec) )

    def multAdd(self, Vec x, Vec v, Vec y):
        CHKERR( MatMultAdd(self.mat, x.vec, v.vec, y.vec) )

    def multTranspose(self, Vec x, Vec y):
        CHKERR( MatMultTranspose(self.mat, x.vec, y.vec) )

    def multTransposeAdd(self, Vec x, Vec v, Vec y):
        CHKERR( MatMultTransposeAdd(self.mat, x.vec, v.vec, y.vec) )

    def multHermitian(self, Vec x, Vec y):
        CHKERR( MatMultHermitian(self.mat, x.vec, y.vec) )

    def multHermitianAdd(self, Vec x, Vec v, Vec y):
        CHKERR( MatMultHermitianAdd(self.mat, x.vec, v.vec, y.vec) )

    # SOR

    def SOR(self, Vec b, Vec x, omega=1.0, sortype=None, shift=0.0, its=1, lits=1):
        cdef PetscReal comega = asReal(omega)
        cdef PetscMatSORType csortype = SOR_LOCAL_SYMMETRIC_SWEEP
        if sortype is not None:
            csortype = <PetscMatSORType> asInt(sortype)
        cdef PetscInt cshift = asInt(shift)
        cdef PetscInt cits = asInt(its)
        cdef PetscInt clits = asInt(lits)
        CHKERR( MatSOR(self.mat, b.vec, comega, csortype, cshift, cits, clits, x.vec) )

    #

    def getDiagonalBlock(self):
        cdef Mat submat = Mat()
        CHKERR( MatGetDiagonalBlock(self.mat, &submat.mat) )
        PetscINCREF(submat.obj)
        return submat

    def increaseOverlap(self, IS iset, overlap=1):
        cdef PetscInt ival = asInt(overlap)
        CHKERR( MatIncreaseOverlap(self.mat, 1, &iset.iset, ival) )

    def createSubMatrix(self, IS isrow, IS iscol=None, Mat submat=None):
        cdef PetscMatReuse reuse = MAT_INITIAL_MATRIX
        cdef PetscIS ciscol = NULL
        if iscol is not None: ciscol = iscol.iset
        if submat is None: submat = Mat()
        if submat.mat != NULL: reuse = MAT_REUSE_MATRIX
        CHKERR( MatCreateSubMatrix(self.mat, isrow.iset, ciscol,
                                reuse, &submat.mat) )
        return submat

    def createSubMatrices(self, isrows, iscols=None, submats=None):
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

    def getLocalSubMatrix(self, IS isrow, IS iscol, Mat submat=None):
        if submat is None: submat = Mat()
        else: CHKERR( MatDestroy(&submat.mat) )
        CHKERR( MatGetLocalSubMatrix(self.mat, isrow.iset, iscol.iset, &submat.mat) )
        return submat

    def restoreLocalSubMatrix(self, IS isrow, IS iscol, Mat submat):
        CHKERR( MatRestoreLocalSubMatrix(self.mat, isrow.iset, iscol.iset, &submat.mat) )

    #

    def norm(self, norm_type=None):
        cdef PetscNormType norm_1_2 = PETSC_NORM_1_AND_2
        cdef PetscNormType ntype = PETSC_NORM_FROBENIUS
        if norm_type is not None: ntype = norm_type
        cdef PetscReal rval[2]
        CHKERR( MatNorm(self.mat, ntype, rval) )
        if ntype != norm_1_2: return toReal(rval[0])
        else: return (toReal(rval[0]), toReal(rval[1]))

    def scale(self, alpha):
        cdef PetscScalar sval = asScalar(alpha)
        CHKERR( MatScale(self.mat, sval) )

    def shift(self, alpha):
        cdef PetscScalar sval = asScalar(alpha)
        CHKERR( MatShift(self.mat, sval) )

    def chop(self, tol):
        cdef PetscReal rval = asReal(tol)
        CHKERR( MatChop(self.mat, rval) )

    def setRandom(self, Random random=None):
        cdef PetscRandom rnd = NULL
        if random is not None: rnd = random.rnd
        CHKERR( MatSetRandom(self.mat, rnd) )

    def axpy(self, alpha, Mat X, structure=None):
        cdef PetscScalar sval = asScalar(alpha)
        cdef PetscMatStructure flag = matstructure(structure)
        CHKERR( MatAXPY(self.mat, sval, X.mat, flag) )

    def aypx(self, alpha, Mat X, structure=None):
        cdef PetscScalar sval = asScalar(alpha)
        cdef PetscMatStructure flag = matstructure(structure)
        CHKERR( MatAYPX(self.mat, sval, X.mat, flag) )

    # matrix-matrix product

    def matMult(self, Mat mat, Mat result=None, fill=None):
        cdef PetscMatReuse reuse = MAT_INITIAL_MATRIX
        cdef PetscReal rval = 2
        if result is None:
            result = Mat()
        elif result.mat != NULL:
            reuse = MAT_REUSE_MATRIX
        if fill is not None: rval = asReal(fill)
        CHKERR( MatMatMult(self.mat, mat.mat, reuse, rval, &result.mat) )
        return result

    def matTransposeMult(self, Mat mat, Mat result=None, fill=None):
        cdef PetscMatReuse reuse = MAT_INITIAL_MATRIX
        cdef PetscReal rval = 2
        if result is None:
            result = Mat()
        elif result.mat != NULL:
            reuse = MAT_REUSE_MATRIX
        if fill is not None: rval = asReal(fill)
        CHKERR( MatMatTransposeMult(self.mat, mat.mat, reuse, rval, &result.mat) )
        return result

    def transposeMatMult(self, Mat mat, Mat result=None, fill=None):
        cdef PetscMatReuse reuse = MAT_INITIAL_MATRIX
        cdef PetscReal rval = 2
        if result is None:
            result = Mat()
        elif result.mat != NULL:
            reuse = MAT_REUSE_MATRIX
        if fill is not None: rval = asReal(fill)
        CHKERR( MatTransposeMatMult(self.mat, mat.mat, reuse, rval, &result.mat) )
        return result

    def ptap(self, Mat P, Mat result=None, fill=None):
        cdef PetscMatReuse reuse = MAT_INITIAL_MATRIX
        cdef PetscReal cfill = PETSC_DEFAULT
        if result is None:
            result = Mat()
        elif result.mat != NULL:
            reuse = MAT_REUSE_MATRIX
        if fill is not None: cfill = asReal(fill)
        CHKERR( MatPtAP(self.mat, P.mat, reuse, cfill, &result.mat) )
        return result

    def rart(self, Mat R, Mat result=None, fill=None):
        cdef PetscMatReuse reuse = MAT_INITIAL_MATRIX
        cdef PetscReal cfill = PETSC_DEFAULT
        if result is None:
            result = Mat()
        elif result.mat != NULL:
            reuse = MAT_REUSE_MATRIX
        if fill is not None: cfill = asReal(fill)
        CHKERR( MatRARt(self.mat, R.mat, reuse, cfill, &result.mat) )
        return result

    def matMatMult(self, Mat B, Mat C, Mat result=None, fill=None):
        cdef PetscMatReuse reuse = MAT_INITIAL_MATRIX
        cdef PetscReal cfill = PETSC_DEFAULT
        if result is None:
            result = Mat()
        elif result.mat != NULL:
            reuse = MAT_REUSE_MATRIX
        if fill is not None: cfill = asReal(fill)
        CHKERR( MatMatMatMult(self.mat, B.mat, C.mat, reuse, cfill, &result.mat) )
        return result

    def kron(self, Mat mat, Mat result=None):
        cdef PetscMatReuse reuse = MAT_INITIAL_MATRIX
        if result is None:
            result = Mat()
        elif result.mat != NULL:
            reuse = MAT_REUSE_MATRIX
        CHKERR( MatSeqAIJKron(self.mat, mat.mat, reuse, &result.mat) )
        return result

    def bindToCPU(self, flg):
        cdef PetscBool bindFlg = asBool(flg)
        CHKERR( MatBindToCPU(self.mat, bindFlg) )

    def boundToCPU(self):
        cdef PetscBool flg = PETSC_TRUE
        CHKERR( MatBoundToCPU(self.mat, &flg) )
        return toBool(flg)

    # XXX factorization

    def getOrdering(self, ord_type):
        cdef PetscMatOrderingType cval = NULL
        ord_type = str2bytes(ord_type, &cval)
        cdef IS rp = IS(), cp = IS()
        CHKERR( MatGetOrdering(self.mat, cval, &rp.iset, &cp.iset) )
        return (rp, cp)

    def reorderForNonzeroDiagonal(self, IS isrow, IS iscol, atol=0):
        cdef PetscReal rval = asReal(atol)
        cdef PetscIS rp = isrow.iset, cp = iscol.iset
        CHKERR( MatReorderForNonzeroDiagonal(self.mat, rval, rp, cp) )

    def factorLU(self, IS isrow, IS iscol, options=None):
        cdef PetscMatFactorInfo info
        matfactorinfo(PETSC_FALSE, PETSC_FALSE, options, &info)
        CHKERR( MatLUFactor(self.mat, isrow.iset, iscol.iset, &info) )
    def factorSymbolicLU(self, Mat mat, IS isrow, IS iscol, options=None):
        <void>self; <void>mat; <void>isrow; <void>iscol; <void>options; # unused
        raise NotImplementedError
    def factorNumericLU(self, Mat mat, options=None):
        <void>self; <void>mat; <void>options; # unused
        raise NotImplementedError
    def factorILU(self, IS isrow, IS iscol, options=None):
        cdef PetscMatFactorInfo info
        matfactorinfo(PETSC_TRUE, PETSC_FALSE, options, &info)
        CHKERR( MatILUFactor(self.mat, isrow.iset, iscol.iset, &info) )
    def factorSymbolicILU(self, IS isrow, IS iscol, options=None):
        <void>self; <void>isrow; <void>iscol; <void>options; # unused
        raise NotImplementedError

    def factorCholesky(self, IS isperm, options=None):
        cdef PetscMatFactorInfo info
        matfactorinfo(PETSC_FALSE, PETSC_TRUE, options, &info)
        CHKERR( MatCholeskyFactor(self.mat, isperm.iset, &info) )
    def factorSymbolicCholesky(self, IS isperm, options=None):
        <void>self; <void>isperm; <void>options; # unused
        raise NotImplementedError
    def factorNumericCholesky(self, Mat mat, options=None):
        <void>self; <void>mat; <void>options; # unused
        raise NotImplementedError
    def factorICC(self, IS isperm, options=None):
        cdef PetscMatFactorInfo info
        matfactorinfo(PETSC_TRUE, PETSC_TRUE, options, &info)
        CHKERR( MatICCFactor(self.mat, isperm.iset, &info) )
    def factorSymbolicICC(self, IS isperm, options=None):
        <void>self; <void>isperm; <void>options; # unused
        raise NotImplementedError

    def getInertia(self):
        cdef PetscInt ival1 = 0, ival2 = 0, ival3 = 0
        CHKERR( MatGetInertia(self.mat, &ival1, &ival2, &ival3) )
        return (toInt(ival1), toInt(ival2), toInt(ival3))

    def setUnfactored(self):
        CHKERR( MatSetUnfactored(self.mat) )

    # IS

    def fixISLocalEmpty(self, fix):
        cdef PetscBool cfix = asBool(fix)
        CHKERR( MatISFixLocalEmpty(self.mat, cfix) )

    def getISLocalMat(self):
        cdef Mat local = Mat()
        CHKERR( MatISGetLocalMat(self.mat, &local.mat) )
        PetscINCREF(local.obj)
        return local

    def restoreISLocalMat(self, Mat local not None):
        CHKERR( MatISRestoreLocalMat(self.mat, &local.mat) )

    def setISLocalMat(self, Mat local not None):
        CHKERR( MatISSetLocalMat(self.mat, local.mat) )

    def setISPreallocation(self, nnz, onnz):
        cdef PetscInt *cnnz = NULL
        cdef PetscInt *connz = NULL
        nnz = iarray_i(nnz, NULL, &cnnz)
        onnz = iarray_i(onnz, NULL, &connz)
        CHKERR( MatISSetPreallocation(self.mat, 0, cnnz, 0, connz) )
        return self

    # LRC

    def getLRCMats(self):
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

    def H2OpusOrthogonalize(self):
        CHKERR( MatH2OpusOrthogonalize(self.mat) )
        return self

    def H2OpusCompress(self, tol):
        cdef PetscReal _tol = asReal(tol)
        CHKERR( MatH2OpusCompress(self.mat, _tol) )
        return self

    def H2OpusLowRankUpdate(self, Mat U, Mat V=None, s = 1.0):
        cdef PetscScalar _s = asScalar(s)
        cdef PetscMat vmat = NULL
        if V is not None:
            vmat = V.mat
        CHKERR( MatH2OpusLowRankUpdate(self.mat, U.mat, vmat, _s) )
        return self

    # MUMPS

    def setMumpsIcntl(self, icntl, ival):
        cdef PetscInt _icntl = asInt(icntl)
        cdef PetscInt _ival = asInt(ival)
        CHKERR( MatMumpsSetIcntl(self.mat, _icntl, _ival) );

    def getMumpsIcntl(self, icntl):
        cdef PetscInt _icntl = asInt(icntl)
        cdef PetscInt ival = 0
        CHKERR( MatMumpsGetIcntl(self.mat, _icntl, &ival) );
        return toInt(ival)

    def setMumpsCntl(self, icntl, val):
        cdef PetscInt _icntl = asInt(icntl)
        cdef PetscReal _val = asReal(val)
        CHKERR( MatMumpsSetCntl(self.mat, _icntl, _val) );

    def getMumpsCntl(self, icntl):
        cdef PetscInt _icntl = asInt(icntl)
        cdef PetscReal val = 0
        CHKERR( MatMumpsGetCntl(self.mat, _icntl, &val) );
        return toReal(val)

    def getMumpsInfo(self, icntl):
        cdef PetscInt _icntl = asInt(icntl)
        cdef PetscInt ival = 0
        CHKERR( MatMumpsGetInfo(self.mat, _icntl, &ival) );
        return toInt(ival)

    def getMumpsInfog(self, icntl):
        cdef PetscInt _icntl = asInt(icntl)
        cdef PetscInt ival = 0
        CHKERR( MatMumpsGetInfog(self.mat, _icntl, &ival) );
        return toInt(ival)

    def getMumpsRinfo(self, icntl):
        cdef PetscInt _icntl = asInt(icntl)
        cdef PetscReal val = 0
        CHKERR( MatMumpsGetRinfo(self.mat, _icntl, &val) );
        return toReal(val)

    def getMumpsRinfog(self, icntl):
        cdef PetscInt _icntl = asInt(icntl)
        cdef PetscReal val = 0
        CHKERR( MatMumpsGetRinfog(self.mat, _icntl, &val) );
        return toReal(val)

    # solve

    def solveForward(self, Vec b, Vec x):
        CHKERR( MatForwardSolve(self.mat, b.vec, x.vec) )

    def solveBackward(self, Vec b, Vec x):
        CHKERR( MatBackwardSolve(self.mat, b.vec, x.vec) )

    def solve(self, Vec b, Vec x):
        CHKERR( MatSolve(self.mat, b.vec, x.vec) )

    def solveTranspose(self, Vec b, Vec x):
        CHKERR( MatSolveTranspose(self.mat, b.vec, x.vec) )

    def solveAdd(self, Vec b, Vec y, Vec x):
        CHKERR( MatSolveAdd(self.mat, b.vec, y.vec, x.vec) )

    def solveTransposeAdd(self, Vec b, Vec y, Vec x):
        CHKERR( MatSolveTransposeAdd(self.mat, b.vec, y.vec, x.vec) )

    def matSolve(self, Mat B, Mat X):
        CHKERR( MatMatSolve(self.mat, B.mat, X.mat) )

    # dense matrices

    def setDenseLDA(self, lda):
        cdef PetscInt _ilda = asInt(lda)
        CHKERR( MatDenseSetLDA(self.mat, _ilda) )

    def getDenseLDA(self):
        cdef PetscInt lda=0
        CHKERR( MatDenseGetLDA(self.mat, &lda) )
        return toInt(lda)

    def getDenseArray(self,readonly=False):
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

    def getDenseLocalMatrix(self):
        cdef Mat mat = type(self)()
        CHKERR( MatDenseGetLocalMatrix(self.mat, &mat.mat) )
        PetscINCREF(mat.obj)
        return mat

    def getDenseColumnVec(self, i, mode='rw'):
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

    def restoreDenseColumnVec(self, i, mode='rw'):
        cdef PetscInt _i = asInt(i)
        if mode == 'rw':
            CHKERR( MatDenseRestoreColumnVec(self.mat, _i, NULL) )
        elif mode == 'r':
            CHKERR( MatDenseRestoreColumnVecRead(self.mat, _i, NULL) )
        else:
            CHKERR( MatDenseRestoreColumnVecWrite(self.mat, _i, NULL) )

    # Nest

    def getNestSize(self):
        cdef PetscInt nrows, ncols
        CHKERR( MatNestGetSize(self.mat, &nrows, &ncols) )
        return toInt(nrows), toInt(ncols)

    def getNestISs(self):
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

    def getNestLocalISs(self):
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

    def getNestSubMatrix(self, i, j):
        cdef Mat submat = Mat()
        cdef PetscInt idxm = asInt(i)
        cdef PetscInt jdxm = asInt(j)
        CHKERR( MatNestGetSubMat(self.mat, idxm, jdxm, &submat.mat) )
        PetscINCREF(submat.obj)
        return submat

    # MatIS

    def getISLocalMat(self):
        cdef Mat localmat = type(self)()
        CHKERR( MatISGetLocalMat(self.mat, &localmat.mat) )
        PetscINCREF(localmat.obj)
        return localmat

    # DM

    def getDM(self):
        cdef PetscDM newdm = NULL
        CHKERR( MatGetDM(self.mat, &newdm) )
        cdef DM dm = subtype_DM(newdm)()
        dm.dm = newdm
        PetscINCREF(dm.obj)
        return dm

    def setDM(self, DM dm):
        CHKERR( MatSetDM(self.mat, dm.dm) )

    # backward compatibility

    PtAP = ptap

    #

    property sizes:
        def __get__(self):
            return self.getSizes()
        def __set__(self, value):
            self.setSizes(value)

    property size:
        def __get__(self):
            return self.getSize()

    property local_size:
        def __get__(self):
            return self.getLocalSize()

    property block_size:
        def __get__(self):
            return self.getBlockSize()

    property block_sizes:
        def __get__(self):
            return self.getBlockSizes()

    property owner_range:
        def __get__(self):
            return self.getOwnershipRange()

    property owner_ranges:
        def __get__(self):
            return self.getOwnershipRanges()

    #

    property assembled:
        def __get__(self):
            return self.isAssembled()
    property symmetric:
        def __get__(self):
            return self.isSymmetric()
    property hermitian:
        def __get__(self):
            return self.isHermitian()
    property structsymm:
        def __get__(self):
            return self.isStructurallySymmetric()

    # TODO Stream
    def __dlpack__(self, stream=-1):
        return self.toDLPack('rw')

    def __dlpack_device__(self):
        (dltype, devId, _, _, _) = mat_get_dlpack_ctx(self)
        return (dltype, devId)

    def toDLPack(self, mode='rw'):
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

    #

    def __cinit__(self):
        self.obj  = <PetscObject*> &self.nsp
        self.nsp = NULL

    def __call__(self, vec):
        self.remove(vec)

    #

    def view(self, Viewer viewer=None):
        cdef PetscViewer vwr = NULL
        if viewer is not None: vwr = viewer.vwr
        CHKERR( MatNullSpaceView(self.nsp, vwr) )

    def destroy(self):
        CHKERR( MatNullSpaceDestroy(&self.nsp) )
        return self

    def create(self, constant=False, vectors=(),  comm=None):
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

    def createRigidBody(self, Vec coords):
        cdef PetscNullSpace newnsp = NULL
        CHKERR( MatNullSpaceCreateRigidBody(coords.vec, &newnsp) )
        PetscCLEAR(self.obj); self.nsp = newnsp
        return self

    def setFunction(self, function, args=None, kargs=None):
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

    def hasConstant(self):
        cdef PetscBool flag = PETSC_FALSE
        CHKERR( MatNullSpaceGetVecs(self.nsp, &flag, NULL, NULL) )
        return toBool(flag)

    def getVecs(self):
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

    def getFunction(self):
        return self.get_attr('__function__')

    #

    def remove(self, Vec vec):
        CHKERR( MatNullSpaceRemove(self.nsp, vec.vec) )

    def test(self, Mat mat):
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
