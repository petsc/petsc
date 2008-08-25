# --------------------------------------------------------------------

cdef extern from "petscmat.h":

    ctypedef char* PetscMatType "const char*"
    PetscMatType MATSAME
    PetscMatType MATSEQMAIJ
    PetscMatType MATMPIMAIJ
    PetscMatType MATMAIJ
    PetscMatType MATIS
    PetscMatType MATMPIROWBS
    PetscMatType MATSEQAIJ
    PetscMatType MATMPIAIJ
    PetscMatType MATAIJ
    PetscMatType MATSHELL
    PetscMatType MATSEQDENSE
    PetscMatType MATMPIDENSE
    PetscMatType MATDENSE
    PetscMatType MATSEQBAIJ
    PetscMatType MATMPIBAIJ
    PetscMatType MATBAIJ
    PetscMatType MATMPIADJ
    PetscMatType MATSEQSBAIJ
    PetscMatType MATMPISBAIJ
    PetscMatType MATSBAIJ
    PetscMatType MATDAAD
    PetscMatType MATMFFD
    PetscMatType MATNORMAL
    PetscMatType MATLRC
    PetscMatType MATSEQCSRPERM
    PetscMatType MATMPICSRPERM
    PetscMatType MATCSRPERM
    PetscMatType MATSEQCRL
    PetscMatType MATMPICRL
    PetscMatType MATCRL
    PetscMatType MATSCATTER
    PetscMatType MATBLOCKMAT
    PetscMatType MATCOMPOSITE
    PetscMatType MATSEQFFTW

    ctypedef char* PetscMatOrderingType "const char*"
    PetscMatOrderingType MAT_ORDERING_NATURAL     "MATORDERING_NATURAL"
    PetscMatOrderingType MAT_ORDERING_ND          "MATORDERING_ND"
    PetscMatOrderingType MAT_ORDERING_1WD         "MATORDERING_1WD"
    PetscMatOrderingType MAT_ORDERING_RCM         "MATORDERING_RCM"
    PetscMatOrderingType MAT_ORDERING_QMD         "MATORDERING_QMD"
    PetscMatOrderingType MAT_ORDERING_ROWLENGTH   "MATORDERING_ROWLENGTH"
    PetscMatOrderingType MAT_ORDERING_DSC_ND      "MATORDERING_DSC_ND"
    PetscMatOrderingType MAT_ORDERING_DSC_MMD     "MATORDERING_DSC_MMD"
    PetscMatOrderingType MAT_ORDERING_DSC_MDF     "MATORDERING_DSC_MDF"
    PetscMatOrderingType MAT_ORDERING_CONSTRAINED "MATORDERING_CONSTRAINED "
    PetscMatOrderingType MAT_ORDERING_IDENTITY    "MATORDERING_IDENTITY"
    PetscMatOrderingType MAT_ORDERING_REVERSE     "MATORDERING_REVERSE"

    ctypedef enum PetscMatReuse "MatReuse":
        MAT_INITIAL_MATRIX
        MAT_REUSE_MATRIX

    ctypedef enum PetscMatDuplicateOption "MatDuplicateOption":
        MAT_DO_NOT_COPY_VALUES
        MAT_COPY_VALUES

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
        MAT_KEEP_ZEROED_ROWS
        MAT_IGNORE_ZERO_ENTRIES
        MAT_USE_INODES
        MAT_HERMITIAN
        MAT_SYMMETRY_ETERNAL
        MAT_USE_COMPRESSEDROW
        MAT_IGNORE_LOWER_TRIANGULAR
        MAT_ERROR_LOWER_TRIANGULAR
        MAT_GETROW_UPPERTRIANGULAR

    int MatView(PetscMat,PetscViewer)
    int MatDestroy(PetscMat)
    int MatCreate(MPI_Comm,PetscMat*)

    int MatCreateIS(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,PetscLGMap,PetscMat*)
    int MatISGetLocalMat(PetscMat,PetscMat*)

    int MatCreateScatter(MPI_Comm,PetscScatter,PetscMat*)
    int MatScatterSetVecScatter(PetscMat,PetscScatter)
    int MatScatterGetVecScatter(PetscMat,PetscScatter*)

    int MatCreateNormal(PetscMat,PetscMat*)
    int MatCreateLRC(PetscMat,PetscMat,PetscMat,PetscMat*)
    int MatCreateShell(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,void*,PetscMat*)

    int MatSetSizes(PetscMat,PetscInt,PetscInt,PetscInt,PetscInt)
    int MatSetBlockSize(PetscMat,PetscInt)
    int MatSetType(PetscMat,PetscMatType)
    int MatSetOption(PetscMat,PetscMatOption,PetscTruth)

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
    int MatGetOwnershipRangeRow"MatGetOwnershipRange"(PetscMat,PetscInt*,PetscInt*)
    int MatGetOwnershipRangesRow"MatGetOwnershipRanges"(PetscMat,const_PetscInt*[])
    int MatGetOwnershipRangeColumn(PetscMat,PetscInt*,PetscInt*)
    int MatGetOwnershipRangesColumn(PetscMat,const_PetscInt*[])

    int MatLoad(PetscViewer,PetscMatType,PetscMat*)
    int MatDuplicate(PetscMat,PetscMatDuplicateOption,PetscMat*)
    int MatCopy(PetscMat,PetscMat,PetscMatStructure)
    int MatCompress(PetscMat)
    int MatTranspose(PetscMat,PetscMatReuse,PetscMat*)
    int MatConvert(PetscMat,PetscMatType,PetscMatReuse,PetscMat*)

    int MatIsSymmetric(PetscMat,PetscReal,PetscTruth*)
    int MatIsStructurallySymmetric(PetscMat,PetscTruth*)
    int MatIsHermitian(PetscMat,PetscReal,PetscTruth*)
    int MatIsSymmetricKnown(PetscMat,PetscTruth*,PetscTruth*)
    int MatIsHermitianKnown(PetscMat,PetscTruth*,PetscTruth*)
    int MatIsTranspose(PetscMat A,PetscMat B,PetscReal tol,PetscTruth *flg)

    int MatGetVecs(PetscMat,PetscVec*,PetscVec*)

    int MatSetValue(PetscMat,PetscInt,PetscInt,PetscScalar,PetscInsertMode)
    int MatSetValues(PetscMat,PetscInt,PetscInt[],PetscInt,PetscInt[],PetscScalar[],PetscInsertMode)
    int MatSetValuesBlocked(PetscMat,PetscInt,PetscInt[],PetscInt,PetscInt[],PetscScalar[],PetscInsertMode)

    int MatSetLocalToGlobalMapping(PetscMat,PetscLGMap)
    int MatSetLocalToGlobalMappingBlock(PetscMat,PetscLGMap)

    int MatSetValueLocal(PetscMat,PetscInt,PetscInt,PetscScalar,PetscInsertMode)
    int MatSetValuesLocal(PetscMat,PetscInt,PetscInt[],PetscInt,PetscInt[],PetscScalar[],PetscInsertMode)
    int MatSetValuesBlockedLocal(PetscMat,PetscInt,PetscInt[],PetscInt,PetscInt[],PetscScalar[],PetscInsertMode)

    int MatGetValues(PetscMat,PetscInt,PetscInt[],PetscInt,PetscInt[],PetscScalar[])
    int MatGetRow(PetscMat,PetscInt,PetscInt*,const_PetscInt*[],const_PetscScalar*[])
    int MatRestoreRow(PetscMat,PetscInt,PetscInt*,const_PetscInt*[],const_PetscScalar*[])
    int MatGetRowIJ(PetscMat,PetscInt,PetscTruth,PetscTruth,PetscInt*,PetscInt*[],PetscInt*[],PetscTruth*)
    int MatRestoreRowIJ(PetscMat,PetscInt,PetscTruth,PetscTruth,PetscInt*,PetscInt*[],PetscInt*[],PetscTruth*)
    int MatGetColumnIJ(PetscMat,PetscInt,PetscTruth,PetscTruth,PetscInt*,PetscInt*[],PetscInt*[],PetscTruth*)
    int MatRestoreColumnIJ(PetscMat,PetscInt,PetscTruth,PetscTruth,PetscInt*,PetscInt*[],PetscInt*[],PetscTruth*)

    int MatZeroEntries(PetscMat)
    int MatStoreValues(PetscMat)
    int MatRetrieveValues(PetscMat)
    int MatAssemblyBegin(PetscMat,PetscMatAssemblyType)
    int MatAssemblyEnd(PetscMat,PetscMatAssemblyType)
    int MatAssembled(PetscMat,PetscTruth*)

    int MatDiagonalSet(PetscMat,PetscVec,PetscInsertMode)
    int MatDiagonalScale(PetscMat, PetscVec OPTIONAL, PetscVec OPTIONAL)
    int MatScale(PetscMat,PetscScalar)
    int MatShift(PetscMat,PetscScalar)
    int MatAXPY(PetscMat,PetscScalar,PetscMat,PetscMatStructure)
    int MatAYPX(PetscMat,PetscScalar,PetscMat,PetscMatStructure)
    int MatInterpolate(PetscMat,PetscVec,PetscVec)
    int MatInterpolateAdd(PetscMat,PetscVec,PetscVec,PetscVec)
    int MatRestrict(PetscMat,PetscVec,PetscVec)

    int MatPermute(PetscMat,PetscIS,PetscIS,PetscMat*)
    int MatPermuteSparsify(PetscMat,PetscInt,PetscReal,PetscReal,PetscIS,PetscIS,PetscMat*)

    int MatMerge(MPI_Comm,PetscMat,PetscInt,PetscMatReuse,PetscMat*)
    int MatGetSubMatrix(PetscMat,PetscIS,PetscIS,PetscInt,PetscMatReuse,PetscMat*)
    int MatGetSubMatrices(PetscMat,PetscInt,PetscIS[],PetscIS[],PetscMatReuse,PetscMat*[])
    int MatIncreaseOverlap(PetscMat,PetscInt,PetscIS[],PetscInt)

    int MatConjugate(PetscMat)
    int MatRealPart(PetscMat)
    int MatImaginaryPart(PetscMat)

    int MatZeroRows(PetscMat,PetscInt,PetscInt[],PetscScalar)
    int MatZeroRowsLocal(PetscMat,PetscInt,PetscInt[],PetscScalar)
    int MatZeroRowsIS(PetscMat,PetscIS,PetscScalar)
    int MatZeroRowsLocalIS(PetscMat,PetscIS,PetscScalar)

    int MatGetDiagonal(PetscMat,PetscVec)
    int MatGetRowMax(PetscMat,PetscVec,PetscInt[])
    int MatGetRowMaxAbs(PetscMat,PetscVec,PetscInt[])
    int MatGetColumnVector(PetscMat,PetscVec,PetscInt)

    int MatNorm(PetscMat,PetscNormType,PetscReal*)

    int MatMult(PetscMat,PetscVec,PetscVec)
    int MatMultAdd(PetscMat,PetscVec,PetscVec,PetscVec)
    int MatMultTranspose(PetscMat,PetscVec,PetscVec)
    int MatMultTransposeAdd(PetscMat,PetscVec,PetscVec,PetscVec)
    int MatMultConstrained(PetscMat,PetscVec,PetscVec)
    int MatMultTransposeConstrained(PetscMat,PetscVec,PetscVec)

    int MatGetOrdering(PetscMat,PetscMatOrderingType,PetscIS*,PetscIS*)
    int MatReorderForNonzeroDiagonal(PetscMat,PetscReal,PetscIS,PetscIS)

    ctypedef struct PetscMatFactorInfo "MatFactorInfo":
        PetscReal shiftnz, shiftpd, shift_fraction, shiftinblocks
        PetscReal fill, diagonal_fill
        PetscReal dt, dtcol, dtcount, levels
        PetscReal zeropivot, pivotinblocks
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

    int MatComputeExplicitOperator(PetscMat,PetscMat*)
    int MatUseScaledForm(PetscMat,PetscTruth)
    int MatScaleSystem(PetscMat,PetscVec,PetscVec)
    int MatUnScaleSystem(PetscMat,PetscVec,PetscVec)


cdef extern from "custom.h":
    enum: MAT_SKIP_ALLOCATION
    int MatCreateAnyAIJ(MPI_Comm,PetscInt,
                        PetscInt,PetscInt,
                        PetscInt,PetscInt,
                        PetscMat*)
    int MatAnyAIJSetPreallocation(PetscMat,PetscInt,
                                  PetscInt,PetscInt[],
                                  PetscInt,PetscInt[])
    int MatAnyAIJSetPreallocationCSR(PetscMat,PetscInt,PetscInt[],
                                     PetscInt[],PetscScalar[])

# --------------------------------------------------------------------

cdef extern from "petscmat.h":

    ctypedef int (PetscNullSpaceFunction)(PetscVec,void*) except PETSC_ERR_PYTHON

    int MatNullSpaceDestroy(PetscNullSpace)
    int MatNullSpaceCreate(MPI_Comm,PetscTruth,PetscInt,PetscVec[],PetscNullSpace*)
    int MatNullSpaceRemove(PetscNullSpace,PetscVec,PetscVec*)
    int MatNullSpaceSetFunction(PetscNullSpace,PetscNullSpaceFunction*,void*)
    int MatNullSpaceAttach(PetscMat,PetscNullSpace)
    int MatNullSpaceTest(PetscNullSpace,PetscMat)

cdef inline Mat ref_NullSpace(PetscNullSpace nsp):
    cdef NullSpace ob = <NullSpace> NullSpace()
    PetscIncref(<PetscObject>nsp)
    ob.nsp = nsp
    return ob

cdef inline object NullSpace_getFun(PetscNullSpace nsp):
    return Object_getAttr(<PetscObject>nsp, "__function__")

cdef int NullSpace_Function(PetscVec v,
                            void*    ctx) except PETSC_ERR_PYTHON:
    cdef PetscNullSpace nsp = <PetscNullSpace> ctx
    cdef Vec vec = ref_Vec(v)
    (function, args, kargs) = NullSpace_getFun(nsp)
    function(vec, *args, **kargs)
    return 0

cdef inline int NullSpace_setFun(PetscNullSpace nsp, object fun) except -1:
    if fun is None: CHKERR( MatNullSpaceSetFunction(nsp, NULL, NULL) )
    else: CHKERR( MatNullSpaceSetFunction(nsp, NullSpace_Function, nsp) )
    Object_setAttr(<PetscObject>nsp, "__function__", fun)
    return 0


# --------------------------------------------------------------------

cdef inline Mat ref_Mat(PetscMat mat):
    cdef Mat ob = <Mat> Mat()
    PetscIncref(<PetscObject>mat)
    ob.mat = mat
    return ob

# --------------------------------------------------------------------

cdef inline PetscMatStructure matstructure(object structure) except *:
    if   structure is None:  return MAT_DIFFERENT_NONZERO_PATTERN
    elif structure is False: return MAT_DIFFERENT_NONZERO_PATTERN
    elif structure is True:  return MAT_SAME_NONZERO_PATTERN
    else:                    return structure

cdef inline PetscMatAssemblyType assemblytype(object assembly) except *:
    if   assembly is None:  return MAT_FINAL_ASSEMBLY
    elif assembly is False: return MAT_FINAL_ASSEMBLY
    elif assembly is True:  return MAT_FLUSH_ASSEMBLY
    else:                   return assembly


# --------------------------------------------------------------------

cdef inline int Mat_BlockSize(object bsize, PetscInt *_bs) except -1:
    cdef PetscInt bs = PETSC_DECIDE
    if bsize is not None: bs = bsize
    if bs != PETSC_DECIDE and bs < 1: raise ValueError(
        "block size %d must be positive" % bs)
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


cdef inline int Mat_AllocAIJ_NNZ(PetscMat A, PetscInt bs, object NNZ) except -1:
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
        if d_n > 1 and d_n*b != m:
            raise ValueError("size(d_nnz) is %d, expected %d" % (d_n, m))
        if o_n > 1 and o_n*b != m:
            raise ValueError("size(o_nnz) is %d, expected %d" % (o_n, m))
    # preallocate
    CHKERR( MatAnyAIJSetPreallocation(A, bs, d_nz, d_nnz, o_nz, o_nnz) )


cdef inline int Mat_AllocAIJ_CSR(PetscMat A, PetscInt bs, object CSR) except -1:
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
        raise ValueError("size(I) is %d, expected %d" % (ni, (m/b+1)))
    if (i[0] != 0):
        raise ValueError("I[0] is %d, expected %d"    % (i[0], 0))
    if (i[ni-1] != nj):
        raise ValueError("size(J) is %d, expected %d" % (nj, i[ni-1]))
    if ov is not None and (nj*b*b != nv):
        raise ValueError("size(V) is %d, expected %d" % (nv, nj*b*b))
    # preallocate
    CHKERR( MatAnyAIJSetPreallocationCSR(A, bs, i, j, v) )


# --------------------------------------------------------------------

cdef extern from "petsc.h":
    ctypedef PetscInt    const_PetscInt    "const PetscInt"
    ctypedef PetscReal   const_PetscReal   "const PetscReal"
    ctypedef PetscScalar const_PetscScalar "const PetscScalar"

ctypedef int (MatSetValuesFcn)(PetscMat,PetscInt,const_PetscInt[],
                               PetscInt,const_PetscInt[],
                               const_PetscScalar[],PetscInsertMode)

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
        "incompatible array sizes: " \
        "ni=%d, nj=%d, nv=%d" % (ni, nj, nv) )
    # insert mode
    cdef PetscInsertMode addv = insertmode(oaddv)
    # MatSetValuesXXX function
    cdef MatSetValuesFcn *setvalues = NULL
    if blocked and local: setvalues = MatSetValuesBlockedLocal
    elif blocked:         setvalues = MatSetValuesBlocked
    elif local:           setvalues = MatSetValuesLocal
    else:                 setvalues = MatSetValues
    # actual call
    CHKERR( setvalues(A, ni, i, nj, j, v, addv) )
    return 0

cdef inline int matsetvaluescsr(PetscMat A, object om,
                                object oi, object oj, object ov,
                                object oaddv, int blocked, int local) except -1:
    # block size
    cdef PetscInt bs=1, bs2=1
    if blocked: CHKERR( MatGetBlockSize(A, &bs) )
    if bs < 1: bs = 1
    bs2 = bs*bs
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
            rs /= bs; re /= bs
        nm = re - rs
    # check various sizes
    if (ni-1 != nm):
        raise ValueError("size(I) is %d, expected %d" % (ni, nm+1))
    if (i[0] != 0):
        raise ValueError("I[0] is %d, expected %d"    % (i[0], 0))
    if (i[ni-1] != nj):
        raise ValueError("size(J) is %d, expected %d" % ( nj, i[ni-1]))
    if (nj*bs2  != nv):
        raise ValueError("size(V) is %d, expected %d" % ( nv, nj*bs2))
    # insert mode
    cdef PetscInsertMode addv = insertmode(oaddv)
    # MatSetValuesXXX function
    cdef MatSetValuesFcn *setvalues = NULL
    if blocked and local: setvalues = MatSetValuesBlockedLocal
    elif blocked:         setvalues = MatSetValuesBlocked
    elif local:           setvalues = MatSetValuesLocal
    else:                 setvalues = MatSetValues
    # actual call
    cdef PetscInt k=0, l=0
    cdef PetscInt irow=0, ncol=0, *icol=NULL
    cdef PetscScalar *sval=NULL
    for k in range(nm):
        irow = m[k] if m!=NULL else rs+k
        ncol = i[k+1] - i[k]
        icol = j + i[k]
        if blocked:
            sval = v + i[k]*bs2
            for l in range(ncol):
                CHKERR( setvalues(A, 1, &irow, 1, &icol[l], &sval[l*bs2], addv) )
        else:
            sval = v + i[k]
            CHKERR( setvalues(A, 1, &irow, ncol, icol, sval, addv) )
    return 0

# --------------------------------------------------------------------

cdef object mat_getitem(Mat self, object ij):
    cdef PetscInt M=0, N=0
    i, j = ij
    if typecheck(i, slice):
        CHKERR( MatGetSize(self.mat, &M, NULL) )
        start, stop, stride = i.indices(M)
        i = arange(start, stop, stride)
    if typecheck(j, slice):
        CHKERR( MatGetSize(self.mat, NULL, &N) )
        start, stop, stride = j.indices(N)
        j = arange(start, stop, stride)
    return self.getValues(i, j)


cdef object mat_setitem(Mat self, object ij, object v):
    cdef PetscInt M=0, N=0
    oi, oj = ij
    if typecheck(oi, slice):
        CHKERR( MatGetSize(self.mat, &M, NULL) )
        start, stop, stride = oi.indices(M)
        oi = arange(start, stop, stride)
    if typecheck(oj, slice):
        CHKERR( MatGetSize(self.mat, NULL, &N) )
        start, stop, stride = oj.indices(N)
        oj = arange(start, stop, stride)
    return self.setValues(oi, oj, v)

# --------------------------------------------------------------------

cdef extern from "ctorreg.h":
    PetscMatType MATPYTHON
    int MatPythonSetContext(PetscMat,void*)
    int MatPythonGetContext(PetscMat,void**)

# --------------------------------------------------------------------
