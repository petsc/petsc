cdef extern from * nogil:

    ctypedef char* PetscMatType "const char*"
    PetscMatType MATSAME
    PetscMatType MATMAIJ
    PetscMatType   MATSEQMAIJ
    PetscMatType   MATMPIMAIJ
    PetscMatType MATIS
    PetscMatType MATAIJ
    PetscMatType   MATSEQAIJ
    PetscMatType   MATMPIAIJ
    PetscMatType     MATAIJCRL
    PetscMatType       MATSEQAIJCRL
    PetscMatType       MATMPIAIJCRL
    PetscMatType     MATAIJCUSP
    PetscMatType       MATSEQAIJCUSP
    PetscMatType       MATMPIAIJCUSP
    PetscMatType     MATAIJPERM
    PetscMatType       MATSEQAIJPERM
    PetscMatType       MATMPIAIJPERM
    PetscMatType MATSHELL
    PetscMatType MATDENSE
    PetscMatType   MATSEQDENSE
    PetscMatType   MATMPIDENSE
    PetscMatType MATBAIJ
    PetscMatType   MATSEQBAIJ
    PetscMatType   MATMPIBAIJ
    PetscMatType MATMPIADJ
    PetscMatType MATSBAIJ
    PetscMatType   MATSEQSBAIJ
    PetscMatType   MATMPISBAIJ
    PetscMatType MATDAAD
    PetscMatType MATMFFD
    PetscMatType MATNORMAL
    PetscMatType MATLRC
    PetscMatType MATSCATTER
    PetscMatType MATBLOCKMAT
    PetscMatType MATCOMPOSITE
    PetscMatType MATFFT
    PetscMatType   MATFFTW
    PetscMatType   MATSEQCUFFT
    PetscMatType MATTRANSPOSEMAT
    PetscMatType MATSCHURCOMPLEMENT
    #PetscMatType MATPYTHON
    PetscMatType MATHYPRESTRUCT
    PetscMatType MATHYPRESSTRUCT
    PetscMatType MATSUBMATRIX
    PetscMatType MATLOCALREF
    PetscMatType MATNEST

    ctypedef char* PetscMatOrderingType "const char*"
    PetscMatOrderingType MATORDERINGNATURAL
    PetscMatOrderingType MATORDERINGND
    PetscMatOrderingType MATORDERING1WD
    PetscMatOrderingType MATORDERINGRCM
    PetscMatOrderingType MATORDERINGQMD
    PetscMatOrderingType MATORDERINGROWLENGTH
    PetscMatOrderingType MATORDERINGAMD

    ctypedef enum PetscMatReuse "MatReuse":
        MAT_INITIAL_MATRIX
        MAT_REUSE_MATRIX
        MAT_IGNORE_MATRIX

    ctypedef enum PetscMatDuplicateOption "MatDuplicateOption":
        MAT_DO_NOT_COPY_VALUES
        MAT_COPY_VALUES
        MAT_SHARE_NONZERO_PATTERN

    ctypedef enum PetscMatAssemblyType "MatAssemblyType":
        MAT_FLUSH_ASSEMBLY
        MAT_FINAL_ASSEMBLY

    ctypedef enum  PetscMatStructure "MatStructure":
        MAT_SAME_NONZERO_PATTERN      "SAME_NONZERO_PATTERN"
        MAT_DIFFERENT_NONZERO_PATTERN "DIFFERENT_NONZERO_PATTERN"
        MAT_SUBSET_NONZERO_PATTERN    "SUBSET_NONZERO_PATTERN"
        MAT_SAME_PRECONDITIONER       "SAME_PRECONDITIONER"

    ctypedef enum PetscMatReuse "MatReuse":
        MAT_INITIAL_MATRIX
        MAT_REUSE_MATRIX

    ctypedef enum PetscMatOption "MatOption":
        MAT_ROW_ORIENTED
        MAT_NEW_NONZERO_LOCATIONS
        MAT_SYMMETRIC
        MAT_STRUCTURALLY_SYMMETRIC
        MAT_NEW_DIAGONALS
        MAT_IGNORE_OFF_PROC_ENTRIES
        MAT_NEW_NONZERO_LOCATION_ERR
        MAT_NEW_NONZERO_ALLOCATION_ERR
        MAT_USE_HASH_TABLE
        MAT_KEEP_NONZERO_PATTERN
        MAT_IGNORE_ZERO_ENTRIES
        MAT_USE_INODES
        MAT_HERMITIAN
        MAT_SYMMETRY_ETERNAL
        MAT_IGNORE_LOWER_TRIANGULAR
        MAT_ERROR_LOWER_TRIANGULAR
        MAT_GETROW_UPPERTRIANGULAR

    int MatView(PetscMat,PetscViewer)
    int MatDestroy(PetscMat*)
    int MatCreate(MPI_Comm,PetscMat*)

    int MatCreateIS(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,PetscLGMap,PetscMat*)
    int MatISGetLocalMat(PetscMat,PetscMat*)

    int MatCreateScatter(MPI_Comm,PetscScatter,PetscMat*)
    int MatScatterSetVecScatter(PetscMat,PetscScatter)
    int MatScatterGetVecScatter(PetscMat,PetscScatter*)

    int MatCreateNormal(PetscMat,PetscMat*)
    int MatCreateTranspose(PetscMat,PetscMat*)
    int MatCreateLRC(PetscMat,PetscMat,PetscMat,PetscMat*)
    int MatCreateSubMatrix(PetscMat,PetscIS,PetscIS,PetscMat*)
    int MatCreateShell(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,void*,PetscMat*)

    int MatSetSizes(PetscMat,PetscInt,PetscInt,PetscInt,PetscInt)
    int MatSetBlockSize(PetscMat,PetscInt)
    int MatSetType(PetscMat,PetscMatType)
    int MatSetOption(PetscMat,PetscMatOption,PetscBool)

    int MatSetOptionsPrefix(PetscMat,char[])
    int MatGetOptionsPrefix(PetscMat,char*[])
    int MatSetFromOptions(PetscMat)
    int MatSetUp(PetscMat)

    int MatGetType(PetscMat,PetscMatType*)
    int MatGetSize(PetscMat,PetscInt*,PetscInt*)
    int MatGetLocalSize(PetscMat,PetscInt*,PetscInt*)
    int MatGetBlockSize(PetscMat,PetscInt*)
    int MatGetOwnershipRange(PetscMat,PetscInt*,PetscInt*)
    int MatGetOwnershipRanges(PetscMat,const_PetscInt*[])
    int MatGetOwnershipRangeColumn(PetscMat,PetscInt*,PetscInt*)
    int MatGetOwnershipRangesColumn(PetscMat,const_PetscInt*[])

    int MatEqual(PetscMat,PetscMat,PetscBool*)
    int MatLoad(PetscMat,PetscViewer)
    int MatDuplicate(PetscMat,PetscMatDuplicateOption,PetscMat*)
    int MatCopy(PetscMat,PetscMat,PetscMatStructure)
    int MatTranspose(PetscMat,PetscMatReuse,PetscMat*)
    int MatConvert(PetscMat,PetscMatType,PetscMatReuse,PetscMat*)

    int MatIsSymmetric(PetscMat,PetscReal,PetscBool*)
    int MatIsStructurallySymmetric(PetscMat,PetscBool*)
    int MatIsHermitian(PetscMat,PetscReal,PetscBool*)
    int MatIsSymmetricKnown(PetscMat,PetscBool*,PetscBool*)
    int MatIsHermitianKnown(PetscMat,PetscBool*,PetscBool*)
    int MatIsTranspose(PetscMat A,PetscMat B,PetscReal tol,PetscBool *flg)

    int MatGetVecs(PetscMat,PetscVec*,PetscVec*)

    int MatSetValue(PetscMat,PetscInt,PetscInt,PetscScalar,PetscInsertMode)
    int MatSetValues(PetscMat,PetscInt,PetscInt[],PetscInt,PetscInt[],PetscScalar[],PetscInsertMode)
    int MatSetValuesBlocked(PetscMat,PetscInt,PetscInt[],PetscInt,PetscInt[],PetscScalar[],PetscInsertMode)

    int MatSetLocalToGlobalMapping(PetscMat,PetscLGMap,PetscLGMap)
    int MatSetLocalToGlobalMappingBlock(PetscMat,PetscLGMap,PetscLGMap)

    int MatSetValueLocal(PetscMat,PetscInt,PetscInt,PetscScalar,PetscInsertMode)
    int MatSetValuesLocal(PetscMat,PetscInt,PetscInt[],PetscInt,PetscInt[],PetscScalar[],PetscInsertMode)
    int MatSetValuesBlockedLocal(PetscMat,PetscInt,PetscInt[],PetscInt,PetscInt[],PetscScalar[],PetscInsertMode)

    int MatSetStencil(PetscMat,PetscInt,PetscInt[],PetscInt[],PetscInt)
    ctypedef struct PetscMatStencil "MatStencil":
        PetscInt k,j,i,c
    int MatSetValuesStencil(PetscMat,PetscInt,PetscMatStencil[],PetscInt,PetscMatStencil[],PetscScalar[],PetscInsertMode)
    int MatSetValuesBlockedStencil(PetscMat,PetscInt,PetscMatStencil[],PetscInt,PetscMatStencil[],PetscScalar[],PetscInsertMode)

    int MatGetValues(PetscMat,PetscInt,PetscInt[],PetscInt,PetscInt[],PetscScalar[])
    int MatGetRow(PetscMat,PetscInt,PetscInt*,const_PetscInt*[],const_PetscScalar*[])
    int MatRestoreRow(PetscMat,PetscInt,PetscInt*,const_PetscInt*[],const_PetscScalar*[])
    int MatGetRowIJ(PetscMat,PetscInt,PetscBool,PetscBool,PetscInt*,PetscInt*[],PetscInt*[],PetscBool*)
    int MatRestoreRowIJ(PetscMat,PetscInt,PetscBool,PetscBool,PetscInt*,PetscInt*[],PetscInt*[],PetscBool*)
    int MatGetColumnIJ(PetscMat,PetscInt,PetscBool,PetscBool,PetscInt*,PetscInt*[],PetscInt*[],PetscBool*)
    int MatRestoreColumnIJ(PetscMat,PetscInt,PetscBool,PetscBool,PetscInt*,PetscInt*[],PetscInt*[],PetscBool*)

    int MatZeroEntries(PetscMat)
    int MatStoreValues(PetscMat)
    int MatRetrieveValues(PetscMat)
    int MatAssemblyBegin(PetscMat,PetscMatAssemblyType)
    int MatAssemblyEnd(PetscMat,PetscMatAssemblyType)
    int MatAssembled(PetscMat,PetscBool*)

    int MatDiagonalSet(PetscMat,PetscVec,PetscInsertMode)
    int MatDiagonalScale(PetscMat, PetscVec OPTIONAL, PetscVec OPTIONAL)
    int MatScale(PetscMat,PetscScalar)
    int MatShift(PetscMat,PetscScalar)
    int MatAXPY(PetscMat,PetscScalar,PetscMat,PetscMatStructure)
    int MatAYPX(PetscMat,PetscScalar,PetscMat,PetscMatStructure)
    int MatMatMult(PetscMat,PetscMat,PetscMatReuse,PetscReal,PetscMat*)
    int MatMatMultTranspose(PetscMat,PetscMat,PetscMatReuse,PetscReal,PetscMat*)
    int MatMatMultSymbolic(PetscMat,PetscMat,PetscReal,PetscMat*)
    int MatMatMultNumeric(PetscMat,PetscMat,PetscMat)

    int MatInterpolate(PetscMat,PetscVec,PetscVec)
    int MatInterpolateAdd(PetscMat,PetscVec,PetscVec,PetscVec)
    int MatRestrict(PetscMat,PetscVec,PetscVec)

    int MatPermute(PetscMat,PetscIS,PetscIS,PetscMat*)
    int MatPermuteSparsify(PetscMat,PetscInt,PetscReal,PetscReal,PetscIS,PetscIS,PetscMat*)

    int MatMerge(MPI_Comm,PetscMat,PetscInt,PetscMatReuse,PetscMat*)
    int MatGetSubMatrix(PetscMat,PetscIS,PetscIS,PetscMatReuse,PetscMat*)
    int MatGetSubMatrices(PetscMat,PetscInt,PetscIS[],PetscIS[],PetscMatReuse,PetscMat*[])
    int MatIncreaseOverlap(PetscMat,PetscInt,PetscIS[],PetscInt)
    int MatGetDiagonalBlock(PetscMat,PetscMat*)

    int MatConjugate(PetscMat)
    int MatRealPart(PetscMat)
    int MatImaginaryPart(PetscMat)

    int MatZeroRows(PetscMat,PetscInt,PetscInt[],PetscScalar,PetscVec,PetscVec)
    int MatZeroRowsLocal(PetscMat,PetscInt,PetscInt[],PetscScalar,PetscVec,PetscVec)
    int MatZeroRowsIS(PetscMat,PetscIS,PetscScalar,PetscVec,PetscVec)
    int MatZeroRowsLocalIS(PetscMat,PetscIS,PetscScalar,PetscVec,PetscVec)

    int MatGetDiagonal(PetscMat,PetscVec)
    int MatGetRowMax(PetscMat,PetscVec,PetscInt[])
    int MatGetRowMaxAbs(PetscMat,PetscVec,PetscInt[])
    int MatGetColumnVector(PetscMat,PetscVec,PetscInt)

    int MatNorm(PetscMat,PetscNormType,PetscReal*)

    int MatMult(PetscMat,PetscVec,PetscVec)
    int MatMultAdd(PetscMat,PetscVec,PetscVec,PetscVec)
    int MatMultTranspose(PetscMat,PetscVec,PetscVec)
    int MatMultTransposeAdd(PetscMat,PetscVec,PetscVec,PetscVec)
    int MatMultHermitian"MatMultHermitianTranspose"(PetscMat,PetscVec,PetscVec)
    int MatMultHermitianAdd"MatMultHermitianTransposeAdd"(PetscMat,PetscVec,PetscVec,PetscVec)
    int MatMultConstrained(PetscMat,PetscVec,PetscVec)
    int MatMultTransposeConstrained(PetscMat,PetscVec,PetscVec)

    int MatGetOrdering(PetscMat,PetscMatOrderingType,PetscIS*,PetscIS*)
    int MatReorderForNonzeroDiagonal(PetscMat,PetscReal,PetscIS,PetscIS)

    ctypedef char* PetscMatSolverPackage "const char*"
    ctypedef enum PetscMatFactorShiftType "MatFactorShiftType":
        MAT_SHIFT_NONE
        MAT_SHIFT_NONZERO
        MAT_SHIFT_POSITIVE_DEFINITE
        MAT_SHIFT_INBLOCKS

    ctypedef struct PetscMatFactorInfo "MatFactorInfo":
        PetscReal fill
        PetscReal levels, diagonal_fill
        PetscReal usedt, dt, dtcol, dtcount
        PetscReal zeropivot, pivotinblocks
        PetscReal shifttype, shiftamount
    int MatFactorInfoInitialize(PetscMatFactorInfo*)

    int MatCholeskyFactor(PetscMat,PetscIS,PetscMatFactorInfo*)
    int MatCholeskyFactorSymbolic(PetscMat,PetscIS,PetscMatFactorInfo*,PetscMat*)
    int MatCholeskyFactorNumeric(PetscMat,PetscMatFactorInfo*,PetscMat*)
    int MatLUFactor(PetscMat,PetscIS,PetscIS,PetscMatFactorInfo*)
    int MatILUFactor(PetscMat,PetscIS,PetscIS,PetscMatFactorInfo*)
    int MatICCFactor(PetscMat,PetscIS,PetscMatFactorInfo*)
    int MatLUFactorSymbolic(PetscMat,PetscIS,PetscIS,PetscMatFactorInfo*,PetscMat*)
    int MatILUFactorSymbolic(PetscMat,PetscIS,PetscIS,PetscMatFactorInfo*,PetscMat*)
    int MatICCFactorSymbolic(PetscMat,PetscIS,PetscMatFactorInfo*,PetscMat*)
    int MatLUFactorNumeric(PetscMat,PetscMatFactorInfo*,PetscMat*)
    int MatILUDTFactor(PetscMat,PetscIS,PetscIS,PetscMatFactorInfo*,PetscMat*)
    int MatGetInertia(PetscMat,PetscInt*,PetscInt*,PetscInt*)
    int MatSetUnfactored(PetscMat)

    int MatForwardSolve(PetscMat,PetscVec,PetscVec)
    int MatBackwardSolve(PetscMat,PetscVec,PetscVec)
    int MatSolve(PetscMat,PetscVec,PetscVec)
    int MatSolveTranspose(PetscMat,PetscVec,PetscVec)
    int MatSolveAdd(PetscMat,PetscVec,PetscVec,PetscVec)
    int MatSolveTransposeAdd(PetscMat,PetscVec,PetscVec,PetscVec)
    int MatMatSolve(PetscMat,PetscMat,PetscMat)

    int MatComputeExplicitOperator(PetscMat,PetscMat*)
    int MatUseScaledForm(PetscMat,PetscBool)
    int MatScaleSystem(PetscMat,PetscVec,PetscVec)
    int MatUnScaleSystem(PetscMat,PetscVec,PetscVec)


cdef extern from "custom.h" nogil:
    enum: MAT_SKIP_ALLOCATION
    int MatCreateAnyAIJ(MPI_Comm,PetscInt,
                        PetscInt,PetscInt,
                        PetscInt,PetscInt,
                        PetscMat*)
    int MatCreateAnyAIJCRL(MPI_Comm,PetscInt,
                           PetscInt,PetscInt,
                           PetscInt,PetscInt,
                           PetscMat*)
    int MatAnyAIJSetPreallocation(PetscMat,PetscInt,
                                  PetscInt,PetscInt[],
                                  PetscInt,PetscInt[])
    int MatAnyAIJSetPreallocationCSR(PetscMat,PetscInt,PetscInt[],
                                     PetscInt[],PetscScalar[])
    int MatCreateAnyDense(MPI_Comm,PetscInt,
                          PetscInt,PetscInt,
                          PetscInt,PetscInt,
                          PetscMat*)
    int MatAnyDenseSetPreallocation(PetscMat,PetscInt,PetscScalar[])

cdef extern from "libpetsc4py.h":
    PetscMatType MATPYTHON
    int MatPythonSetContext(PetscMat,void*)
    int MatPythonGetContext(PetscMat,void**)
    int MatPythonSetType(PetscMat,char[])

# -----------------------------------------------------------------------------

cdef extern from * nogil:
    int MatNullSpaceDestroy(PetscNullSpace*)
    int MatNullSpaceView(PetscNullSpace,PetscViewer)
    int MatNullSpaceCreate(MPI_Comm,PetscBool,PetscInt,PetscVec[],
                           PetscNullSpace*)
    int MatNullSpaceRemove(PetscNullSpace,PetscVec,PetscVec*)
    int MatNullSpaceAttach(PetscMat,PetscNullSpace)
    int MatNullSpaceTest(PetscNullSpace,PetscMat)

    ctypedef int MatNullSpaceFunction(PetscNullSpace,
                                      PetscVec,
                                      void*) except PETSC_ERR_PYTHON
    int MatNullSpaceSetFunction(PetscNullSpace,MatNullSpaceFunction*,void*)

cdef inline NullSpace ref_NullSpace(PetscNullSpace nsp):
    cdef NullSpace ob = <NullSpace> NullSpace()
    ob.nsp = nsp
    PetscINCREF(ob.obj)
    return ob

cdef int NullSpace_Function(
    PetscNullSpace n,
    PetscVec       v,
    void *         ctx,
    ) except PETSC_ERR_PYTHON with gil:
    cdef NullSpace nsp = ref_NullSpace(n)
    cdef Vec vec = ref_Vec(v)
    (function, args, kargs) = nsp.get_attr('__function__')
    function(nsp, vec, *args, **kargs)
    return 0

# -----------------------------------------------------------------------------

cdef inline Mat ref_Mat(PetscMat mat):
    cdef Mat ob = <Mat> Mat()
    ob.mat = mat
    PetscINCREF(ob.obj)
    return ob

# -----------------------------------------------------------------------------

# unary operations

cdef Mat mat_pos(Mat self):
    cdef Mat mat = type(self)()
    CHKERR( MatDuplicate(self.mat, MAT_COPY_VALUES, &mat.mat) )
    return mat

cdef Mat mat_neg(Mat self):
    cdef Mat mat = <Mat> mat_pos(self)
    CHKERR( MatScale(mat.mat, -1) )
    return mat

# inplace binary operations

cdef Mat mat_iadd(Mat self, other):
    if isinstance(other, Mat):
        self.axpy(1, other)
    elif isinstance(other, (tuple, list)):
        alpha, mat = other
        self.axpy(alpha, other)
    elif isinstance(other, Vec):
        self.setDiagonal(other, PETSC_ADD_VALUES)
    else:
        self.shift(other)
    return self

cdef Mat mat_isub(Mat self, other):
    if isinstance(other, Mat):
        self.axpy(-1, other)
    elif isinstance(other, (tuple, list)):
        alpha, mat = other
        self.axpy(-alpha, other)
    elif isinstance(other, Vec):
        diag = other.copy()
        diag.scale(-1)
        self.setDiagonal(diag, PETSC_ADD_VALUES)
        diag.destroy()
    else:
        self.shift(other)
    return self

cdef Mat mat_imul(Mat self, other):
    if (isinstance(other, tuple) or
        isinstance(other, list)):
        L, R = other
        self.diagonalScale(L, R)
    else:
        self.scale(other)
    return self

cdef Mat mat_idiv(Mat self, other):
    if isinstance(other, (tuple, list)):
        L, R = other
        if isinstance(L, Vec):
            L = L.copy()
            L.reciprocal()
        if isinstance(R, Vec):
            R = R.copy()
            R.reciprocal()
        self.diagonalScale(L, R)
    else:
        self.scale(other)
    return self

# binary operations

cdef Mat mat_add(Mat self, other):
    return mat_iadd(mat_pos(self), other)

cdef Mat mat_sub(Mat self, other):
    return mat_isub(mat_pos(self), other)

cdef Mat mat_mul(Mat self, other):
    if isinstance(other, Mat):
        return self.matMult(other)
    else:
        return mat_imul(mat_pos(self), other)

cdef Vec mat_mul_vec(Mat self, Vec other):
    cdef Vec result = self.getVecLeft()
    self.mult(other, result)
    return result

cdef Mat mat_div(Mat self, other):
    return mat_idiv(mat_pos(self), other)

# reflected binary operations

cdef Mat mat_radd(Mat self, other):
    return mat_add(self, other)

cdef Mat mat_rsub(Mat self, other):
    cdef Mat mat = <Mat> mat_sub(self, other)
    mat.scale(-1)
    return mat

cdef Mat mat_rmul(Mat self, other):
    return mat_mul(self, other)

cdef Mat mat_rdiv(Mat self, other):
    raise NotImplementedError

# -----------------------------------------------------------------------------

cdef inline PetscMatStructure matstructure(object structure) \
    except <PetscMatStructure>(-1):
    if   structure is None:  return MAT_DIFFERENT_NONZERO_PATTERN
    elif structure is False: return MAT_DIFFERENT_NONZERO_PATTERN
    elif structure is True:  return MAT_SAME_NONZERO_PATTERN
    else:                    return structure

cdef inline PetscMatAssemblyType assemblytype(object assembly) \
    except <PetscMatAssemblyType>(-1):
    if   assembly is None:  return MAT_FINAL_ASSEMBLY
    elif assembly is False: return MAT_FINAL_ASSEMBLY
    elif assembly is True:  return MAT_FLUSH_ASSEMBLY
    else:                   return assembly

# -----------------------------------------------------------------------------

cdef inline int Mat_BlockSize(object bsize, PetscInt *_bs) except -1:
    cdef PetscInt bs = PETSC_DECIDE
    if bsize is not None: bs = asInt(bsize)
    if bs != PETSC_DECIDE and bs < 1: raise ValueError(
        "block size %d must be positive" % toInt(bs) )
    _bs[0] = bs
    return 0

cdef inline int Mat_SplitSizes(MPI_Comm comm,
                               object size, object bsize,
                               PetscInt *b,
                               PetscInt *m, PetscInt *n,
                               PetscInt *M, PetscInt *N) except -1:
    # unpack row and column sizes
    cdef object rsize, csize
    try:
        rsize , csize = size
    except (TypeError, ValueError):
        rsize = csize = size
    # split row and column sizes
    CHKERR( Sys_SplitSizes(comm, rsize, bsize, b, m, M) )
    CHKERR( Sys_SplitSizes(comm, csize, bsize, b, n, N) )
    return 0


cdef inline int Mat_AllocAIJ_SKIP(PetscMat A,
                                  PetscInt bs) except -1:
    cdef PetscInt d_nz=MAT_SKIP_ALLOCATION, *d_nnz=NULL
    cdef PetscInt o_nz=MAT_SKIP_ALLOCATION, *o_nnz=NULL
    CHKERR( MatAnyAIJSetPreallocation(A, bs, d_nz, d_nnz, o_nz, o_nnz) )
    return 0

cdef inline int Mat_AllocAIJ_DEFAULT(PetscMat A,
                                     PetscInt bs) except -1:
    cdef PetscInt d_nz=PETSC_DECIDE, *d_nnz=NULL
    cdef PetscInt o_nz=PETSC_DECIDE, *o_nnz=NULL
    CHKERR( MatAnyAIJSetPreallocation(A, bs, d_nz, d_nnz, o_nz, o_nnz) )
    return 0

cdef inline int Mat_AllocAIJ_NNZ(PetscMat A, PetscInt bs, object NNZ) \
    except -1:
    # unpack NNZ argument
    cdef object od_nnz, oo_nnz
    try:
        od_nnz, oo_nnz = NNZ
    except (TypeError, ValueError):
        od_nnz, oo_nnz = NNZ, None
    # diagonal and off-diagonal number of nonzeros
    cdef PetscInt d_nz=PETSC_DECIDE, d_n=0, *d_nnz=NULL
    if od_nnz is not None:
        od_nnz = iarray_i(od_nnz, &d_n, &d_nnz)
        if   d_n == 0: d_nnz = NULL # just in case
        elif d_n == 1: d_nz = d_nnz[0]; d_n=0; d_nnz = NULL
    cdef PetscInt o_nz=PETSC_DECIDE, o_n=0, *o_nnz=NULL
    if oo_nnz is not None:
        oo_nnz = iarray_i(oo_nnz, &o_n, &o_nnz)
        if   o_n == 0: o_nnz = NULL # just in case
        elif o_n == 1: o_nz = o_nnz[0]; o_n=0; o_nnz = NULL
    # check sizes
    cdef PetscInt m=0, b=bs
    CHKERR( MatGetLocalSize(A, &m, NULL) )
    if bs == PETSC_DECIDE: b = 1
    if m != PETSC_DECIDE and (d_n > 1 or o_n > 1):
        if d_n > 1 and d_n*b != m: raise ValueError(
            "size(d_nnz) is %d, expected %d" %
            (toInt(d_n), toInt(m)) )
        if o_n > 1 and o_n*b != m: raise ValueError(
            "size(o_nnz) is %d, expected %d" %
            (toInt(o_n), toInt(m)) )
    # preallocate
    CHKERR( MatAnyAIJSetPreallocation(A, bs, d_nz, d_nnz, o_nz, o_nnz) )

cdef inline int Mat_AllocAIJ_CSR(PetscMat A, PetscInt bs, object CSR) \
    except -1:
    # unpack CSR argument
    cdef object oi, oj, ov
    try:
        oi, oj, ov = CSR
    except (TypeError, ValueError):
        oi, oj = CSR; ov = None
    # rows, cols, and values
    cdef PetscInt ni=0, *i=NULL
    cdef PetscInt nj=0, *j=NULL
    oi = iarray_i(oi, &ni, &i)
    oj = iarray_i(oj, &nj, &j)
    cdef PetscInt nv=0
    cdef PetscScalar *v=NULL
    if ov is not None:
        ov = iarray_s(ov, &nv, &v)
    # check sizes
    cdef PetscInt m=0, b=bs
    CHKERR( MatGetLocalSize(A, &m, NULL) )
    if bs == PETSC_DECIDE: b = 1
    if (m != PETSC_DECIDE) and (((ni-1)*b) != m):
        raise ValueError("size(I) is %d, expected %d" %
                         (toInt(ni), toInt(m//b+1)) )
    if (i[0] != 0):
        raise ValueError("I[0] is %d, expected %d" %
                         (toInt(i[0]), toInt(0)) )
    if (i[ni-1] != nj):
        raise ValueError("size(J) is %d, expected %d" %
                         (toInt(nj), toInt(i[ni-1])) )
    if ov is not None and (nj*b*b != nv):
        raise ValueError("size(V) is %d, expected %d" %
                         (toInt(nv), toInt(nj*b*b)) )
    # preallocate
    CHKERR( MatAnyAIJSetPreallocationCSR(A, bs, i, j, v) )


cdef inline int Mat_AllocDense_DEFAULT(PetscMat A,
                                       PetscInt bs) except -1:
    cdef PetscScalar *data=NULL
    CHKERR( MatAnyDenseSetPreallocation(A, bs, data) )
    return 0

cdef inline int Mat_AllocDense_ARRAY(PetscMat A, PetscInt bs,
                                     object array) except -1:
    cdef PetscInt size=0
    cdef PetscScalar *data=NULL
    cdef PetscInt m=0, n=0, b=bs
    CHKERR( MatGetLocalSize(A, &m, &n) )
    if bs == PETSC_DECIDE: b = 1
    array = ofarray_s(array, &size, &data)
    if m*n != size: raise ValueError(
        "size(array) is %d, expected %dx%d=%d" %
        (toInt(size), toInt(m), toInt(n), toInt(m*n)) )
    CHKERR( MatAnyDenseSetPreallocation(A, b, data) )
    return 0

# -----------------------------------------------------------------------------

ctypedef int MatSetValuesFcn(PetscMat,PetscInt,const_PetscInt[],
                             PetscInt,const_PetscInt[],
                             const_PetscScalar[],PetscInsertMode)

cdef inline MatSetValuesFcn* matsetvalues_fcn(int blocked, int local):
    cdef MatSetValuesFcn *setvalues = NULL
    if blocked and local: setvalues = MatSetValuesBlockedLocal
    elif blocked:         setvalues = MatSetValuesBlocked
    elif local:           setvalues = MatSetValuesLocal
    else:                 setvalues = MatSetValues
    return setvalues

cdef inline int matsetvalues(PetscMat A,
                             object oi, object oj, object ov,
                             object oaddv, int blocked, int local) except -1:
    # block size
    cdef PetscInt bs=1
    if blocked: CHKERR( MatGetBlockSize(A, &bs) )
    if bs < 1: bs = 1
    # rows, cols, and values
    cdef PetscInt ni=0, *i=NULL
    cdef PetscInt nj=0, *j=NULL
    cdef PetscInt nv=0
    cdef PetscScalar *v=NULL
    cdef object ai = iarray_i(oi, &ni, &i)
    cdef object aj = iarray_i(oj, &nj, &j)
    cdef object av = iarray_s(ov, &nv, &v)
    if ni*nj*bs*bs != nv: raise ValueError(
        "incompatible array sizes: ni=%d, nj=%d, nv=%d" %
        (toInt(ni), toInt(nj), toInt(nv)) )
    # MatSetValuesXXX function and insert mode
    cdef MatSetValuesFcn *setvalues = \
         matsetvalues_fcn(blocked, local)
    cdef PetscInsertMode addv = insertmode(oaddv)
    # actual call
    CHKERR( setvalues(A, ni, i, nj, j, v, addv) )
    return 0

cdef inline int matsetvalues_rcv(PetscMat A,
                                 object oi, object oj, object ov,
                                 object oaddv,
                                 int blocked, int local) except -1:
    # block size
    cdef PetscInt bs=1
    if blocked: CHKERR( MatGetBlockSize(A, &bs) )
    if bs < 1: bs = 1
    # rows, cols, and values
    cdef PetscInt ni=0, *i=NULL
    cdef PetscInt nj=0, *j=NULL
    cdef PetscInt nv=0
    cdef PetscScalar *v=NULL
    cdef ndarray ai = iarray_i(oi, &ni, &i)
    cdef ndarray aj = iarray_i(oj, &nj, &j)
    cdef ndarray av = iarray_s(ov, &nv, &v)
    # check various dimensions
    if PyArray_NDIM(ai) != 2: raise ValueError(
        ("row indices must have two dimensions: "
         "rows.ndim=%d") % (PyArray_NDIM(ai)) )
    elif not PyArray_ISCONTIGUOUS(ai): raise ValueError(
        "expecting a C-contiguous array")
    if PyArray_NDIM(aj) != 2: raise ValueError(
        ("column indices must have two dimensions: "
         "cols.ndim=%d") % (PyArray_NDIM(aj)) )
    elif not PyArray_ISCONTIGUOUS(aj): raise ValueError(
        "expecting a C-contiguous array")
    if PyArray_NDIM(av) < 2: raise ValueError(
        ("values must have two or more dimensions: "
         "vals.ndim=%d") % (PyArray_NDIM(av)) )
    elif not PyArray_ISCONTIGUOUS(av): raise ValueError(
        "expecting a C-contiguous array")
    # check various shapes
    cdef Py_ssize_t nm = PyArray_DIM(ai, 0)
    cdef Py_ssize_t si = PyArray_DIM(ai, 1)
    cdef Py_ssize_t sj = PyArray_DIM(aj, 1)
    cdef Py_ssize_t sv = PyArray_SIZE(av) // PyArray_DIM(av, 0)
    if ((nm != PyArray_DIM(aj, 0)) or
        (nm != PyArray_DIM(av, 0)) or
        (si*bs * sj*bs != sv)): raise ValueError(
        ("input arrays have incompatible shapes: "
         "rows.shape=%s, cols.shape=%s, vals.shape=%s") %
        (ai.shape, aj.shape, av.shape))
    # MatSetValuesXXX function and insert mode
    cdef MatSetValuesFcn *setvalues = \
         matsetvalues_fcn(blocked, local)
    cdef PetscInsertMode addv = insertmode(oaddv)
    # actual calls
    cdef Py_ssize_t k=0
    for k from 0 <= k < nm:
        CHKERR( setvalues(A,
                          <PetscInt>si, &i[k*si], 
                          <PetscInt>sj, &j[k*sj], 
                          &v[k*sv], addv) )
    return 0

cdef inline int matsetvalues_ijv(PetscMat A,
                                 object oi, object oj, object ov,
                                 object oaddv,
                                 object om,
                                 int blocked, int local) except -1:
    # block size
    cdef PetscInt bs=1
    if blocked: CHKERR( MatGetBlockSize(A, &bs) )
    if bs < 1: bs = 1
    cdef PetscInt bs2 = bs*bs
    # column pointers, column indices, and values
    cdef PetscInt ni=0, *i=NULL
    cdef PetscInt nj=0, *j=NULL
    cdef PetscInt nv=0
    cdef PetscScalar *v=NULL
    cdef object ai = iarray_i(oi, &ni, &i)
    cdef object aj = iarray_i(oj, &nj, &j)
    cdef object av = iarray_s(ov, &nv, &v)
    # row indices
    cdef object am = None
    cdef PetscInt nm=0, *m=NULL
    cdef PetscInt rs=0, re=ni-1
    if om is not None:
        am = iarray_i(om, &nm, &m)
    else:
        if not local:
            CHKERR( MatGetOwnershipRange(A, &rs, &re) )
            rs //= bs; re //= bs
        nm = re - rs
    # check various sizes
    if (ni-1 != nm): raise ValueError(
        "size(I) is %d, expected %d" %
        (toInt(ni), toInt(nm+1)) )
    if (i[0] != 0):raise ValueError(
        "I[0] is %d, expected %d" %
        (toInt(i[0]), 0) )
    if (i[ni-1] != nj): raise ValueError(
        "size(J) is %d, expected %d" %
        (toInt(nj), toInt(i[ni-1])) )
    if (nj*bs2  != nv): raise ValueError(
        "size(V) is %d, expected %d" %
        (toInt(nv), toInt(nj*bs2)) )
    # MatSetValuesXXX function and insert mode
    cdef MatSetValuesFcn *setvalues = \
         matsetvalues_fcn(blocked, local)
    cdef PetscInsertMode addv = insertmode(oaddv)
    # actual call
    cdef PetscInt k=0, l=0
    cdef PetscInt irow=0, ncol=0, *icol=NULL
    cdef PetscScalar *sval=NULL
    for k from 0 <= k < nm:
        irow = m[k] if m!=NULL else rs+k
        ncol = i[k+1] - i[k]
        icol = j + i[k]
        if blocked:
            sval = v + i[k]*bs2
            for l from 0 <= l < ncol:
                CHKERR( setvalues(A, 1, &irow, 1, &icol[l],
                                  &sval[l*bs2], addv) )
        else:
            sval = v + i[k]
            CHKERR( setvalues(A, 1, &irow, ncol, icol, sval, addv) )
    return 0

cdef inline int matsetvalues_csr(PetscMat A,
                                 object oi, object oj, object ov,
                                 object oaddv,
                                 int blocked, int local) except -1:
    matsetvalues_ijv(A, oi, oj, ov, oaddv, None, blocked, local)
    return 0

cdef inline matgetvalues(PetscMat mat,
                         object orows, object ocols, object values):
    cdef PetscInt ni=0, nj=0, nv=0
    cdef PetscInt *i=NULL, *j=NULL
    cdef PetscScalar *v=NULL
    cdef ndarray rows = iarray_i(orows, &ni, &i)
    cdef ndarray cols = iarray_i(ocols, &nj, &j)
    if values is None:
        values = empty_s(ni*nj)
        values.shape = rows.shape + cols.shape
    values = oarray_s(values, &nv, &v)
    if (ni*nj != nv): raise ValueError(
        "incompatible array sizes: ni=%d, nj=%d, nv=%d" %
        (toInt(ni), toInt(nj), toInt(nv)))
    CHKERR( MatGetValues(mat, ni, i, nj, j, v) )
    return values

# -----------------------------------------------------------------------------

cdef extern from "custom.h":
    int MatFactorInfoDefaults(PetscBool,PetscBool,PetscMatFactorInfo*)

cdef inline PetscMatFactorShiftType matfactorshifttype(object st) \
    except <PetscMatFactorShiftType>(-1):
    if isinstance(st, str):
        if st == "none": return MAT_SHIFT_NONE
        if st == "nonzero": return MAT_SHIFT_NONZERO
        if st == "positive_definite": return MAT_SHIFT_POSITIVE_DEFINITE
        if st == "inblocks": return MAT_SHIFT_INBLOCKS
        if st == "na": return MAT_SHIFT_NONZERO
        if st == "pd": return MAT_SHIFT_POSITIVE_DEFINITE
        else: raise ValueError("unknown shift type: %s" % st)
    return st

cdef int matfactorinfo(PetscBool inc, PetscBool chol, object opts,
                       PetscMatFactorInfo *info) except -1:
    CHKERR( MatFactorInfoDefaults(inc,chol,info) )
    if opts is None: return 0
    cdef dict options = dict(opts)
    #
    cdef fill = options.pop('fill', None)
    if fill is not None:
        info.fill = asReal(fill)
    #
    cdef zeropivot = options.pop('zeropivot', None)
    if zeropivot is not None:
        info.zeropivot = asReal(zeropivot)
    #
    cdef levels = options.pop('levels', None)
    if levels is not None:
        info.levels  = <PetscReal>asInt(levels)
    cdef diagonal_fill = options.pop('diagonal_fill', None)
    if diagonal_fill is not None:
        info.diagonal_fill = <PetscReal>(<bint>diagonal_fill)
    #
    cdef dt = options.pop('dt', None)
    if dt is not None:
        info.dt = asReal(dt)
    cdef dtcol = options.pop('dtcol', None)
    if dtcol is not None:
        info.dtcol = asReal(dtcol)
    cdef dtcount = options.pop('dtcount', None)
    if dtcount is not None:
        info.dtcount = <PetscReal>asInt(dtcount)
    if ((dt is not None) or 
        (dtcol is not None) or
        (dtcount is not None)):
        info.usedt = <PetscReal>PETSC_TRUE
    #
    cdef shifttype = options.pop('shifttype', None)
    if shifttype is not None:
        info.shifttype = <PetscReal>matfactorshifttype(shifttype)
    cdef shiftamount = options.pop('shiftamount', None)
    if shiftamount is not None:
        info.shiftamount = asReal(shiftamount)
    #
    if options:
        raise ValueError("unknown options: %s"
                         % list(options.keys()))
    return 0

# -----------------------------------------------------------------------------

cdef object mat_getitem(Mat self, object ij):
    cdef PetscInt M=0, N=0
    rows, cols = ij
    if isinstance(rows, slice):
        CHKERR( MatGetSize(self.mat, &M, NULL) )
        start, stop, stride = rows.indices(toInt(M))
        rows = arange(start, stop, stride)
    if isinstance(cols, slice):
        CHKERR( MatGetSize(self.mat, NULL, &N) )
        start, stop, stride = cols.indices(toInt(N))
        cols = arange(start, stop, stride)
    return matgetvalues(self.mat, rows, cols, None)


cdef int mat_setitem(Mat self, object ij, object v) except -1:
    cdef PetscInt M=0, N=0
    rows, cols = ij
    if isinstance(rows, slice):
        CHKERR( MatGetSize(self.mat, &M, NULL) )
        start, stop, stride = rows.indices(toInt(M))
        rows = arange(start, stop, stride)
    if isinstance(cols, slice):
        CHKERR( MatGetSize(self.mat, NULL, &N) )
        start, stop, stride = cols.indices(toInt(N))
        cols = arange(start, stop, stride)
    matsetvalues(self.mat, rows, cols, v, None, 0, 0)
    return 0

# -----------------------------------------------------------------------------

#@cython.internal
cdef class _Mat_Stencil:
   cdef PetscMatStencil stencil
   property i:
       def __set__(self, value):
           self.stencil.i = asInt(value)
   property j:
       def __set__(self, value):
           self.stencil.j = asInt(value)
   property k:
       def __set__(self, value):
           self.stencil.k = asInt(value)
   property c:
       def __set__(self, value):
           self.stencil.c = asInt(value)
   property index:
       def __set__(self, value):
           cdef PetscMatStencil *s = &self.stencil
           s.k = s.j = s.i = 0
           asDims(value, &s.i, &s.j, &s.k)
   property field:
       def __set__(self, value):
           cdef PetscMatStencil *s = &self.stencil
           s.c = asInt(value)

cdef matsetvaluestencil(PetscMat A,
                        _Mat_Stencil r, _Mat_Stencil c, object value,
                        PetscInsertMode im, int blocked):
    # block size
    cdef PetscInt bs=1
    if blocked: CHKERR( MatGetBlockSize(A, &bs) )
    if bs < 1: bs = 1
    # values
    cdef PetscInt    nv = 1
    cdef PetscScalar *v = NULL
    cdef object av = iarray_s(value, &nv, &v)
    if bs*bs != nv: raise ValueError(
        "incompatible array sizes: nv=%d" % toInt(nv) )
    if blocked:
        CHKERR( MatSetValuesBlockedStencil(A,
                                           1, &r.stencil,
                                           1, &c.stencil,
                                           v, im) )
    else:
        CHKERR( MatSetValuesStencil(A,
                                    1, &r.stencil,
                                    1, &c.stencil,
                                    v, im) )
    return 0

# -----------------------------------------------------------------------------
