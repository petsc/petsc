cdef extern from * nogil:

    ctypedef const char* PetscMatType "MatType"
    PetscMatType MATSAME
    PetscMatType MATMAIJ
    PetscMatType   MATSEQMAIJ
    PetscMatType   MATMPIMAIJ
    PetscMatType MATKAIJ
    PetscMatType   MATSEQKAIJ
    PetscMatType   MATMPIKAIJ
    PetscMatType MATIS
    PetscMatType MATAIJ
    PetscMatType   MATSEQAIJ
    PetscMatType   MATMPIAIJ
    PetscMatType     MATAIJCRL
    PetscMatType       MATSEQAIJCRL
    PetscMatType       MATMPIAIJCRL
    PetscMatType     MATAIJCUSPARSE
    PetscMatType       MATSEQAIJCUSPARSE
    PetscMatType       MATMPIAIJCUSPARSE
    PetscMatType     MATAIJVIENNACL
    PetscMatType       MATSEQAIJVIENNACL
    PetscMatType       MATMPIAIJVIENNACL
    PetscMatType     MATAIJPERM
    PetscMatType       MATSEQAIJPERM
    PetscMatType       MATMPIAIJPERM
    PetscMatType MATAIJSELL
    PetscMatType   MATSEQAIJSELL
    PetscMatType   MATMPIAIJSELL
    PetscMatType MATAIJMKL
    PetscMatType    MATSEQAIJMKL
    PetscMatType    MATMPIAIJMKL
    PetscMatType MATBAIJMKL
    PetscMatType    MATSEQBAIJMKL
    PetscMatType    MATMPIBAIJMKL
    PetscMatType MATSHELL
    PetscMatType MATDENSE
    PetscMatType   MATSEQDENSE
    PetscMatType   MATMPIDENSE
    PetscMatType MATDENSECUDA
    PetscMatType   MATSEQDENSECUDA
    PetscMatType   MATMPIDENSECUDA
    PetscMatType MATELEMENTAL
    PetscMatType MATBAIJ
    PetscMatType   MATSEQBAIJ
    PetscMatType   MATMPIBAIJ
    PetscMatType MATMPIADJ
    PetscMatType MATSBAIJ
    PetscMatType   MATSEQSBAIJ
    PetscMatType   MATMPISBAIJ
    PetscMatType MATMFFD
    PetscMatType MATNORMAL
    PetscMatType MATNORMALHERMITIAN
    PetscMatType MATLRC
    PetscMatType MATSCATTER
    PetscMatType MATBLOCKMAT
    PetscMatType MATCOMPOSITE
    PetscMatType MATFFT
    PetscMatType   MATFFTW
    PetscMatType   MATSEQCUFFT
    PetscMatType MATTRANSPOSEVIRTUAL
    PetscMatType MATHERMITIANTRANSPOSEVIRTUAL
    PetscMatType MATSCHURCOMPLEMENT
    PetscMatType MATPYTHON
    PetscMatType MATHYPRE
    PetscMatType MATHYPRESTRUCT
    PetscMatType MATHYPRESSTRUCT
    PetscMatType MATSUBMATRIX
    PetscMatType MATLOCALREF
    PetscMatType MATNEST
    PetscMatType MATPREALLOCATOR
    PetscMatType MATSELL
    PetscMatType MATSEQSELL
    PetscMatType MATMPISELL
    PetscMatType MATDUMMY
    PetscMatType MATLMVM
    PetscMatType MATLMVMDFP
    PetscMatType MATLMVMBFGS
    PetscMatType MATLMVMSR1
    PetscMatType MATLMVMBROYDEN
    PetscMatType MATLMVMBADBROYDEN
    PetscMatType MATLMVMSYMBROYDEN
    PetscMatType MATLMVMSYMBADBROYDEN
    PetscMatType MATLMVMDIAGBROYDEN
    PetscMatType MATCONSTANTDIAGONAL
    PetscMatType MATH2OPUS

    ctypedef const char* PetscMatOrderingType "MatOrderingType"
    PetscMatOrderingType MATORDERINGNATURAL
    PetscMatOrderingType MATORDERINGND
    PetscMatOrderingType MATORDERING1WD
    PetscMatOrderingType MATORDERINGRCM
    PetscMatOrderingType MATORDERINGQMD
    PetscMatOrderingType MATORDERINGROWLENGTH
    PetscMatOrderingType MATORDERINGWBM
    PetscMatOrderingType MATORDERINGSPECTRAL
    PetscMatOrderingType MATORDERINGAMD
    PetscMatOrderingType MATORDERINGMETISND

    ctypedef const char* PetscMatSolverType "MatSolverType"
    PetscMatSolverType MATSOLVERSUPERLU
    PetscMatSolverType MATSOLVERSUPERLU_DIST
    PetscMatSolverType MATSOLVERSTRUMPACK
    PetscMatSolverType MATSOLVERUMFPACK
    PetscMatSolverType MATSOLVERCHOLMOD
    PetscMatSolverType MATSOLVERKLU
    PetscMatSolverType MATSOLVERSPARSEELEMENTAL
    PetscMatSolverType MATSOLVERELEMENTAL
    PetscMatSolverType MATSOLVERSCALAPACK
    PetscMatSolverType MATSOLVERESSL
    PetscMatSolverType MATSOLVERLUSOL
    PetscMatSolverType MATSOLVERMUMPS
    PetscMatSolverType MATSOLVERMKL_PARDISO
    PetscMatSolverType MATSOLVERMKL_CPARDISO
    PetscMatSolverType MATSOLVERPASTIX
    PetscMatSolverType MATSOLVERMATLAB
    PetscMatSolverType MATSOLVERPETSC
    PetscMatSolverType MATSOLVERBAS
    PetscMatSolverType MATSOLVERCUSPARSE
    PetscMatSolverType MATSOLVERCUDA
    PetscMatSolverType MATSOLVERSPQR

    ctypedef enum PetscMatReuse "MatReuse":
        MAT_INITIAL_MATRIX
        MAT_REUSE_MATRIX
        MAT_IGNORE_MATRIX
        MAT_INPLACE_MATRIX

    ctypedef enum PetscMatDuplicateOption "MatDuplicateOption":
        MAT_DO_NOT_COPY_VALUES
        MAT_COPY_VALUES
        MAT_SHARE_NONZERO_PATTERN

    ctypedef enum PetscMatSORType "MatSORType":
        SOR_FORWARD_SWEEP
        SOR_BACKWARD_SWEEP
        SOR_SYMMETRIC_SWEEP
        SOR_LOCAL_FORWARD_SWEEP
        SOR_LOCAL_BACKWARD_SWEEP
        SOR_LOCAL_SYMMETRIC_SWEEP
        SOR_ZERO_INITIAL_GUESS
        SOR_EISENSTAT
        SOR_APPLY_UPPER
        SOR_APPLY_LOWER

    ctypedef enum PetscMatProductType "MatProductType":
        MATPRODUCT_UNSPECIFIED
        MATPRODUCT_AB
        MATPRODUCT_AtB
        MATPRODUCT_ABt
        MATPRODUCT_PtAP
        MATPRODUCT_RARt
        MATPRODUCT_ABC

    ctypedef enum PetscMatAssemblyType "MatAssemblyType":
        MAT_FLUSH_ASSEMBLY
        MAT_FINAL_ASSEMBLY

    ctypedef enum PetscMatInfoType "MatInfoType":
        MAT_LOCAL
        MAT_GLOBAL_MAX
        MAT_GLOBAL_SUM

    ctypedef enum  PetscMatStructure "MatStructure":
        MAT_SAME_NONZERO_PATTERN      "SAME_NONZERO_PATTERN"
        MAT_DIFFERENT_NONZERO_PATTERN "DIFFERENT_NONZERO_PATTERN"
        MAT_SUBSET_NONZERO_PATTERN    "SUBSET_NONZERO_PATTERN"
        MAT_UNKNOWN_NONZERO_PATTERN   "UNKNOWN_NONZERO_PATTERN"

    ctypedef enum PetscMatOption "MatOption":
        MAT_OPTION_MIN
        MAT_UNUSED_NONZERO_LOCATION_ERR
        MAT_ROW_ORIENTED
        MAT_SYMMETRIC
        MAT_STRUCTURALLY_SYMMETRIC
        MAT_FORCE_DIAGONAL_ENTRIES
        MAT_IGNORE_OFF_PROC_ENTRIES
        MAT_USE_HASH_TABLE
        MAT_KEEP_NONZERO_PATTERN
        MAT_IGNORE_ZERO_ENTRIES
        MAT_USE_INODES
        MAT_HERMITIAN
        MAT_SYMMETRY_ETERNAL
        MAT_NEW_NONZERO_LOCATION_ERR
        MAT_IGNORE_LOWER_TRIANGULAR
        MAT_ERROR_LOWER_TRIANGULAR
        MAT_GETROW_UPPERTRIANGULAR
        MAT_SPD
        MAT_NO_OFF_PROC_ZERO_ROWS
        MAT_NO_OFF_PROC_ENTRIES
        MAT_NEW_NONZERO_LOCATIONS
        MAT_NEW_NONZERO_ALLOCATION_ERR
        MAT_SUBSET_OFF_PROC_ENTRIES
        MAT_SUBMAT_SINGLEIS
        MAT_STRUCTURE_ONLY
        MAT_SORTED_FULL
        MAT_OPTION_MAX

    ctypedef enum PetscMatOperation "MatOperation":
        pass

    PetscErrorCode MatView(PetscMat,PetscViewer)
    PetscErrorCode MatDestroy(PetscMat*)
    PetscErrorCode MatCreate(MPI_Comm,PetscMat*)
    PetscErrorCode MatCreateDenseCUDA(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,PetscScalar[],PetscMat*)

    PetscErrorCode MatCreateIS(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,PetscLGMap,PetscLGMap,PetscMat*)
    PetscErrorCode MatISGetLocalMat(PetscMat,PetscMat*)
    PetscErrorCode MatISGetMPIXAIJ(PetscMat,PetscMatReuse,PetscMat*)

    PetscErrorCode MatCreateScatter(MPI_Comm,PetscScatter,PetscMat*)
    PetscErrorCode MatScatterSetVecScatter(PetscMat,PetscScatter)
    PetscErrorCode MatScatterGetVecScatter(PetscMat,PetscScatter*)

    PetscErrorCode MatCreateNormal(PetscMat,PetscMat*)
    PetscErrorCode MatCreateTranspose(PetscMat,PetscMat*)
    PetscErrorCode MatCreateNormalHermitian(PetscMat,PetscMat*)
    PetscErrorCode MatCreateHermitianTranspose(PetscMat,PetscMat*)
    PetscErrorCode MatCreateLRC(PetscMat,PetscMat,PetscVec,PetscMat,PetscMat*)
    PetscErrorCode MatCreateSubMatrixVirtual(PetscMat,PetscIS,PetscIS,PetscMat*)
    PetscErrorCode MatCreateRedundantMatrix(PetscMat,PetscInt,MPI_Comm,PetscMatReuse,PetscMat*)
    PetscErrorCode MatCreateNest(MPI_Comm,PetscInt,PetscIS[],PetscInt,PetscIS[],PetscMat[],PetscMat*)
    PetscErrorCode MatCreateShell(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,void*,PetscMat*)
    PetscErrorCode MatCreateH2OpusFromMat(PetscMat,PetscInt,const PetscReal[],PetscBool,PetscReal,PetscInt,PetscInt,PetscInt,PetscReal,PetscMat*)
    PetscErrorCode MatCreateSeqAIJWithArrays(MPI_Comm,PetscInt,PetscInt,PetscInt[],PetscInt[],PetscScalar[],PetscMat*)
    PetscErrorCode MatCreateMPIAIJWithArrays(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt[],PetscInt[],PetscScalar[],PetscMat*)
    PetscErrorCode MatCreateMPIAIJWithSplitArrays(MPI_Comm,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt[],PetscInt[],PetscScalar[],PetscInt[],PetscInt[],PetscScalar[],PetscMat*)

    PetscErrorCode MatSetSizes(PetscMat,PetscInt,PetscInt,PetscInt,PetscInt)
    PetscErrorCode MatSetBlockSize(PetscMat,PetscInt)
    PetscErrorCode MatSetBlockSizes(PetscMat,PetscInt,PetscInt)
    PetscErrorCode MatSetType(PetscMat,PetscMatType)
    PetscErrorCode MatSetVecType(PetscMat,PetscVecType)
    PetscErrorCode MatGetVecType(PetscMat,PetscVecType*)
    PetscErrorCode MatSetOption(PetscMat,PetscMatOption,PetscBool)
    PetscErrorCode MatGetOption(PetscMat,PetscMatOption,PetscBool*)

    enum: MAT_SKIP_ALLOCATION
    PetscErrorCode MatSeqAIJSetPreallocation  (PetscMat,PetscInt,PetscInt[])
    PetscErrorCode MatMPIAIJSetPreallocation  (PetscMat,PetscInt,PetscInt[],PetscInt,PetscInt[])
    PetscErrorCode MatSeqBAIJSetPreallocation (PetscMat,PetscInt,PetscInt,PetscInt[])
    PetscErrorCode MatMPIBAIJSetPreallocation (PetscMat,PetscInt,PetscInt,PetscInt[],PetscInt,PetscInt[])
    PetscErrorCode MatSeqSBAIJSetPreallocation(PetscMat,PetscInt,PetscInt,PetscInt[])
    PetscErrorCode MatMPISBAIJSetPreallocation(PetscMat,PetscInt,PetscInt,PetscInt[],PetscInt,PetscInt[])
    PetscErrorCode MatSeqAIJSetPreallocationCSR  (PetscMat,         PetscInt[],PetscInt[],PetscScalar[])
    PetscErrorCode MatMPIAIJSetPreallocationCSR  (PetscMat,         PetscInt[],PetscInt[],PetscScalar[])
    PetscErrorCode MatSeqBAIJSetPreallocationCSR (PetscMat,PetscInt,PetscInt[],PetscInt[],PetscScalar[])
    PetscErrorCode MatMPIBAIJSetPreallocationCSR (PetscMat,PetscInt,PetscInt[],PetscInt[],PetscScalar[])
    PetscErrorCode MatSeqSBAIJSetPreallocationCSR(PetscMat,PetscInt,PetscInt[],PetscInt[],PetscScalar[])
    PetscErrorCode MatMPISBAIJSetPreallocationCSR(PetscMat,PetscInt,PetscInt[],PetscInt[],PetscScalar[])
    PetscErrorCode MatSeqDenseSetPreallocation(PetscMat,PetscScalar[])
    PetscErrorCode MatMPIDenseSetPreallocation(PetscMat,PetscScalar[])
    PetscErrorCode MatISSetPreallocation(PetscMat,PetscInt,PetscInt[],PetscInt,PetscInt[])

    PetscErrorCode MatSetOptionsPrefix(PetscMat,char[])
    PetscErrorCode MatAppendOptionsPrefix(PetscMat,char[])
    PetscErrorCode MatGetOptionsPrefix(PetscMat,char*[])
    PetscErrorCode MatSetFromOptions(PetscMat)
    PetscErrorCode MatSetUp(PetscMat)

    PetscErrorCode MatGetType(PetscMat,PetscMatType*)
    PetscErrorCode MatGetSize(PetscMat,PetscInt*,PetscInt*)
    PetscErrorCode MatGetLocalSize(PetscMat,PetscInt*,PetscInt*)
    PetscErrorCode MatGetBlockSize(PetscMat,PetscInt*)
    PetscErrorCode MatGetBlockSizes(PetscMat,PetscInt*,PetscInt*)
    PetscErrorCode MatGetOwnershipRange(PetscMat,PetscInt*,PetscInt*)
    PetscErrorCode MatGetOwnershipRanges(PetscMat,const PetscInt*[])
    PetscErrorCode MatGetOwnershipRangeColumn(PetscMat,PetscInt*,PetscInt*)
    PetscErrorCode MatGetOwnershipRangesColumn(PetscMat,const PetscInt*[])
    PetscErrorCode MatGetOwnershipIS(PetscMat,PetscIS*,PetscIS*)
    PetscErrorCode MatNestGetISs(PetscMat,PetscIS*,PetscIS*)
    PetscErrorCode MatNestGetLocalISs(PetscMat,PetscIS*,PetscIS*)
    PetscErrorCode MatNestGetSize(PetscMat,PetscInt*,PetscInt*)
    PetscErrorCode MatNestGetSubMat(PetscMat,PetscInt,PetscInt,PetscMat*)

    PetscErrorCode MatEqual(PetscMat,PetscMat,PetscBool*)
    PetscErrorCode MatLoad(PetscMat,PetscViewer)
    PetscErrorCode MatDuplicate(PetscMat,PetscMatDuplicateOption,PetscMat*)
    PetscErrorCode MatCopy(PetscMat,PetscMat,PetscMatStructure)
    PetscErrorCode MatTranspose(PetscMat,PetscMatReuse,PetscMat*)
    PetscErrorCode MatTransposeSetPrecursor(PetscMat,PetscMat)
    PetscErrorCode MatHermitianTranspose(PetscMat,PetscMatReuse,PetscMat*)
    PetscErrorCode MatConvert(PetscMat,PetscMatType,PetscMatReuse,PetscMat*)

    PetscErrorCode MatIsSymmetric(PetscMat,PetscReal,PetscBool*)
    PetscErrorCode MatIsStructurallySymmetric(PetscMat,PetscBool*)
    PetscErrorCode MatIsHermitian(PetscMat,PetscReal,PetscBool*)
    PetscErrorCode MatIsSymmetricKnown(PetscMat,PetscBool*,PetscBool*)
    PetscErrorCode MatIsHermitianKnown(PetscMat,PetscBool*,PetscBool*)
    PetscErrorCode MatIsTranspose(PetscMat,PetscMat,PetscReal,PetscBool*)

    PetscErrorCode MatCreateVecs(PetscMat,PetscVec*,PetscVec*)

    PetscErrorCode MatSetValue(PetscMat,PetscInt,PetscInt,PetscScalar,PetscInsertMode)
    PetscErrorCode MatSetValues(PetscMat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],const PetscScalar[],PetscInsertMode)
    PetscErrorCode MatSetValuesBlocked(PetscMat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],const PetscScalar[],PetscInsertMode)

    PetscErrorCode MatSetLocalToGlobalMapping(PetscMat,PetscLGMap,PetscLGMap)
    PetscErrorCode MatGetLocalToGlobalMapping(PetscMat,PetscLGMap*,PetscLGMap*)
    PetscErrorCode MatSetValueLocal(PetscMat,PetscInt,PetscInt,PetscScalar,PetscInsertMode)
    PetscErrorCode MatSetValuesLocal(PetscMat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],const PetscScalar[],PetscInsertMode)
    PetscErrorCode MatSetValuesBlockedLocal(PetscMat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],const PetscScalar[],PetscInsertMode)

    PetscErrorCode MatSetStencil(PetscMat,PetscInt,const PetscInt[],const PetscInt[],PetscInt)
    ctypedef struct PetscMatStencil "MatStencil":
        PetscInt k,j,i,c
    PetscErrorCode MatSetValuesStencil(PetscMat,PetscInt,const PetscMatStencil[],PetscInt,const PetscMatStencil[],const PetscScalar[],PetscInsertMode)
    PetscErrorCode MatSetValuesBlockedStencil(PetscMat,PetscInt,const PetscMatStencil[],PetscInt,const PetscMatStencil[],const PetscScalar[],PetscInsertMode)

    PetscErrorCode MatGetValues(PetscMat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],PetscScalar[])
    PetscErrorCode MatGetRow(PetscMat,PetscInt,PetscInt*,const PetscInt*[],const PetscScalar*[])
    PetscErrorCode MatRestoreRow(PetscMat,PetscInt,PetscInt*,const PetscInt*[],const PetscScalar*[])
    PetscErrorCode MatGetRowIJ(PetscMat,PetscInt,PetscBool,PetscBool,PetscInt*,const PetscInt*[],const PetscInt*[],PetscBool*)
    PetscErrorCode MatRestoreRowIJ(PetscMat,PetscInt,PetscBool,PetscBool,PetscInt*,const PetscInt*[],const PetscInt*[],PetscBool*)
    PetscErrorCode MatGetColumnIJ(PetscMat,PetscInt,PetscBool,PetscBool,PetscInt*,const PetscInt*[],const PetscInt*[],PetscBool*)
    PetscErrorCode MatRestoreColumnIJ(PetscMat,PetscInt,PetscBool,PetscBool,PetscInt*,const PetscInt*[],const PetscInt*[],PetscBool*)

    PetscErrorCode MatZeroEntries(PetscMat)
    PetscErrorCode MatStoreValues(PetscMat)
    PetscErrorCode MatRetrieveValues(PetscMat)
    PetscErrorCode MatAssemblyBegin(PetscMat,PetscMatAssemblyType)
    PetscErrorCode MatAssemblyEnd(PetscMat,PetscMatAssemblyType)
    PetscErrorCode MatAssembled(PetscMat,PetscBool*)

    PetscErrorCode MatDiagonalSet(PetscMat,PetscVec,PetscInsertMode)
    PetscErrorCode MatDiagonalScale(PetscMat,PetscVec,PetscVec)
    PetscErrorCode MatScale(PetscMat,PetscScalar)
    PetscErrorCode MatShift(PetscMat,PetscScalar)
    PetscErrorCode MatChop(PetscMat,PetscReal)
    PetscErrorCode MatSetRandom(PetscMat,PetscRandom)
    PetscErrorCode MatAXPY(PetscMat,PetscScalar,PetscMat,PetscMatStructure)
    PetscErrorCode MatAYPX(PetscMat,PetscScalar,PetscMat,PetscMatStructure)
    PetscErrorCode MatMatMult(PetscMat,PetscMat,PetscMatReuse,PetscReal,PetscMat*)
    PetscErrorCode MatMatTransposeMult(PetscMat,PetscMat,PetscMatReuse,PetscReal,PetscMat*)
    PetscErrorCode MatTransposeMatMult(PetscMat,PetscMat,PetscMatReuse,PetscReal,PetscMat*)

    PetscErrorCode MatPtAP(PetscMat,PetscMat,PetscMatReuse,PetscReal,PetscMat*)
    PetscErrorCode MatRARt(PetscMat,PetscMat,PetscMatReuse,PetscReal,PetscMat*)
    PetscErrorCode MatMatMatMult(PetscMat,PetscMat,PetscMat,PetscMatReuse,PetscReal,PetscMat*)
    PetscErrorCode MatSeqAIJKron(PetscMat,PetscMat,PetscMatReuse,PetscMat*)

    PetscErrorCode MatInterpolate(PetscMat,PetscVec,PetscVec)
    PetscErrorCode MatInterpolateAdd(PetscMat,PetscVec,PetscVec,PetscVec)
    PetscErrorCode MatRestrict(PetscMat,PetscVec,PetscVec)

    PetscErrorCode MatPermute(PetscMat,PetscIS,PetscIS,PetscMat*)
    PetscErrorCode MatPermuteSparsify(PetscMat,PetscInt,PetscReal,PetscReal,PetscIS,PetscIS,PetscMat*)

    PetscErrorCode MatMerge(MPI_Comm,PetscMat,PetscInt,PetscMatReuse,PetscMat*)
    PetscErrorCode MatCreateSubMatrix(PetscMat,PetscIS,PetscIS,PetscMatReuse,PetscMat*)
    PetscErrorCode MatCreateSubMatrices(PetscMat,PetscInt,PetscIS[],PetscIS[],PetscMatReuse,PetscMat*[])
    PetscErrorCode MatIncreaseOverlap(PetscMat,PetscInt,PetscIS[],PetscInt)
    PetscErrorCode MatGetDiagonalBlock(PetscMat,PetscMat*)
    PetscErrorCode MatGetLocalSubMatrix(PetscMat,PetscIS,PetscIS,PetscMat*)
    PetscErrorCode MatRestoreLocalSubMatrix(PetscMat,PetscIS,PetscIS,PetscMat*)
    PetscErrorCode MatDestroyMatrices(PetscInt,PetscMat*[])

    PetscErrorCode MatConjugate(PetscMat)
    PetscErrorCode MatRealPart(PetscMat)
    PetscErrorCode MatImaginaryPart(PetscMat)

    PetscErrorCode MatZeroRows(PetscMat,PetscInt,PetscInt[],PetscScalar,PetscVec,PetscVec)
    PetscErrorCode MatZeroRowsLocal(PetscMat,PetscInt,PetscInt[],PetscScalar,PetscVec,PetscVec)
    PetscErrorCode MatZeroRowsIS(PetscMat,PetscIS,PetscScalar,PetscVec,PetscVec)
    PetscErrorCode MatZeroRowsLocalIS(PetscMat,PetscIS,PetscScalar,PetscVec,PetscVec)
    PetscErrorCode MatFindZeroRows(PetscMat,PetscIS*)

    PetscErrorCode MatZeroRowsColumns(PetscMat,PetscInt,PetscInt[],PetscScalar,PetscVec,PetscVec)
    PetscErrorCode MatZeroRowsColumnsLocal(PetscMat,PetscInt,PetscInt[],PetscScalar,PetscVec,PetscVec)
    PetscErrorCode MatZeroRowsColumnsIS(PetscMat,PetscIS,PetscScalar,PetscVec,PetscVec)
    PetscErrorCode MatZeroRowsColumnsLocalIS(PetscMat,PetscIS,PetscScalar,PetscVec,PetscVec)
    PetscErrorCode MatZeroRowsColumnsStencil(PetscMat,PetscInt,const PetscMatStencil[],PetscScalar,PetscVec,PetscVec)

    PetscErrorCode MatGetDiagonal(PetscMat,PetscVec)
    PetscErrorCode MatGetRowSum(PetscMat,PetscVec)
    PetscErrorCode MatInvertBlockDiagonal(PetscMat,const PetscScalar**)
    PetscErrorCode MatGetRowMax(PetscMat,PetscVec,PetscInt[])
    PetscErrorCode MatGetRowMaxAbs(PetscMat,PetscVec,PetscInt[])
    PetscErrorCode MatGetColumnVector(PetscMat,PetscVec,PetscInt)

    PetscErrorCode MatNorm(PetscMat,PetscNormType,PetscReal*)

    PetscErrorCode MatMult(PetscMat,PetscVec,PetscVec)
    PetscErrorCode MatMultAdd(PetscMat,PetscVec,PetscVec,PetscVec)
    PetscErrorCode MatMultTranspose(PetscMat,PetscVec,PetscVec)
    PetscErrorCode MatMultTransposeAdd(PetscMat,PetscVec,PetscVec,PetscVec)
    PetscErrorCode MatMultHermitian"MatMultHermitianTranspose"(PetscMat,PetscVec,PetscVec)
    PetscErrorCode MatMultHermitianAdd"MatMultHermitianTransposeAdd"(PetscMat,PetscVec,PetscVec,PetscVec)

    PetscErrorCode MatBindToCPU(PetscMat,PetscBool)
    PetscErrorCode MatBoundToCPU(PetscMat,PetscBool*)

    PetscErrorCode MatSOR(PetscMat,PetscVec,PetscReal,PetscMatSORType,PetscReal,PetscInt,PetscInt,PetscVec)

    PetscErrorCode MatGetOrdering(PetscMat,PetscMatOrderingType,PetscIS*,PetscIS*)
    PetscErrorCode MatReorderForNonzeroDiagonal(PetscMat,PetscReal,PetscIS,PetscIS)

    PetscErrorCode MatISFixLocalEmpty(PetscMat,PetscBool)
    PetscErrorCode MatISGetLocalMat(PetscMat,PetscMat*)
    PetscErrorCode MatISRestoreLocalMat(PetscMat,PetscMat*)
    PetscErrorCode MatISSetLocalMat(PetscMat,PetscMat)

    PetscErrorCode MatH2OpusOrthogonalize(PetscMat)
    PetscErrorCode MatH2OpusCompress(PetscMat,PetscReal)
    PetscErrorCode MatH2OpusLowRankUpdate(PetscMat,PetscMat,PetscMat,PetscScalar)

    PetscErrorCode MatMissingDiagonal(Mat,PetscBool*,PetscInt*)

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

    ctypedef struct PetscMatInfo "MatInfo":
        PetscLogDouble block_size
        PetscLogDouble nz_allocated, nz_used, nz_unneeded
        PetscLogDouble memory
        PetscLogDouble assemblies
        PetscLogDouble mallocs
        PetscLogDouble fill_ratio_given, fill_ratio_needed
        PetscLogDouble factor_mallocs

    PetscErrorCode MatGetInfo(PetscMat,PetscMatInfoType,PetscMatInfo*)

    PetscErrorCode MatFactorInfoInitialize(PetscMatFactorInfo*)

    PetscErrorCode MatCholeskyFactor(PetscMat,PetscIS,PetscMatFactorInfo*)
    PetscErrorCode MatCholeskyFactorSymbolic(PetscMat,PetscIS,PetscMatFactorInfo*,PetscMat*)
    PetscErrorCode MatCholeskyFactorNumeric(PetscMat,PetscMatFactorInfo*,PetscMat*)
    PetscErrorCode MatLUFactor(PetscMat,PetscIS,PetscIS,PetscMatFactorInfo*)
    PetscErrorCode MatILUFactor(PetscMat,PetscIS,PetscIS,PetscMatFactorInfo*)
    PetscErrorCode MatICCFactor(PetscMat,PetscIS,PetscMatFactorInfo*)
    PetscErrorCode MatLUFactorSymbolic(PetscMat,PetscIS,PetscIS,PetscMatFactorInfo*,PetscMat*)
    PetscErrorCode MatILUFactorSymbolic(PetscMat,PetscIS,PetscIS,PetscMatFactorInfo*,PetscMat*)
    PetscErrorCode MatICCFactorSymbolic(PetscMat,PetscIS,PetscMatFactorInfo*,PetscMat*)
    PetscErrorCode MatLUFactorNumeric(PetscMat,PetscMatFactorInfo*,PetscMat*)
    PetscErrorCode MatILUDTFactor(PetscMat,PetscIS,PetscIS,PetscMatFactorInfo*,PetscMat*)
    PetscErrorCode MatGetInertia(PetscMat,PetscInt*,PetscInt*,PetscInt*)
    PetscErrorCode MatSetUnfactored(PetscMat)

    PetscErrorCode MatLRCGetMats(PetscMat,PetscMat*,PetscMat*,PetscVec*,PetscMat*)

    PetscErrorCode MatMumpsSetIcntl(PetscMat,PetscInt,PetscInt)
    PetscErrorCode MatMumpsGetIcntl(PetscMat,PetscInt,PetscInt*)
    PetscErrorCode MatMumpsSetCntl(PetscMat,PetscInt,PetscReal)
    PetscErrorCode MatMumpsGetCntl(PetscMat,PetscInt,PetscReal*)
    PetscErrorCode MatMumpsGetInfo(PetscMat,PetscInt,PetscInt*)
    PetscErrorCode MatMumpsGetInfog(PetscMat,PetscInt,PetscInt*)
    PetscErrorCode MatMumpsGetRinfo(PetscMat,PetscInt,PetscReal*)
    PetscErrorCode MatMumpsGetRinfog(PetscMat,PetscInt,PetscReal*)

    PetscErrorCode MatForwardSolve(PetscMat,PetscVec,PetscVec)
    PetscErrorCode MatBackwardSolve(PetscMat,PetscVec,PetscVec)
    PetscErrorCode MatSolve(PetscMat,PetscVec,PetscVec)
    PetscErrorCode MatSolveTranspose(PetscMat,PetscVec,PetscVec)
    PetscErrorCode MatSolveAdd(PetscMat,PetscVec,PetscVec,PetscVec)
    PetscErrorCode MatSolveTransposeAdd(PetscMat,PetscVec,PetscVec,PetscVec)
    PetscErrorCode MatMatSolve(PetscMat,PetscMat,PetscMat)

    PetscErrorCode MatComputeExplicitOperator(PetscMat,PetscMat*)
    PetscErrorCode MatUseScaledForm(PetscMat,PetscBool)
    PetscErrorCode MatScaleSystem(PetscMat,PetscVec,PetscVec)
    PetscErrorCode MatUnScaleSystem(PetscMat,PetscVec,PetscVec)

    PetscErrorCode MatDenseSetLDA(PetscMat,PetscInt)
    PetscErrorCode MatDenseGetLDA(PetscMat,PetscInt*)
    PetscErrorCode MatDenseGetLocalMatrix(PetscMat,PetscMat*)
    PetscErrorCode MatDenseGetArray(PetscMat,PetscScalar*[])
    PetscErrorCode MatDenseRestoreArray(PetscMat,PetscScalar*[])
    PetscErrorCode MatDenseGetArrayWrite(PetscMat,PetscScalar*[])
    PetscErrorCode MatDenseRestoreArrayWrite(PetscMat,PetscScalar*[])
    PetscErrorCode MatDenseGetArrayRead(PetscMat,const PetscScalar*[])
    PetscErrorCode MatDenseRestoreArrayRead(PetscMat,const PetscScalar*[])
    PetscErrorCode MatDenseGetColumnVec(PetscMat,PetscInt,PetscVec*)
    PetscErrorCode MatDenseRestoreColumnVec(PetscMat,PetscInt,PetscVec*)
    PetscErrorCode MatDenseGetColumnVecRead(PetscMat,PetscInt,PetscVec*)
    PetscErrorCode MatDenseRestoreColumnVecRead(PetscMat,PetscInt,PetscVec*)
    PetscErrorCode MatDenseGetColumnVecWrite(PetscMat,PetscInt,PetscVec*)
    PetscErrorCode MatDenseRestoreColumnVecWrite(PetscMat,PetscInt,PetscVec*)
    PetscErrorCode MatDenseCUDAGetArray(PetscMat,PetscScalar*[])
    PetscErrorCode MatDenseCUDARestoreArray(PetscMat,PetscScalar*[])
    PetscErrorCode MatDenseCUDAGetArrayWrite(PetscMat,PetscScalar*[])
    PetscErrorCode MatDenseCUDARestoreArrayWrite(PetscMat,PetscScalar*[])
    PetscErrorCode MatDenseCUDAGetArrayRead(PetscMat,const PetscScalar*[])
    PetscErrorCode MatDenseCUDARestoreArrayRead(PetscMat,const PetscScalar*[])

    PetscErrorCode MatProductGetType(PetscMat,PetscMatProductType*)
    PetscErrorCode MatProductGetMats(PetscMat,PetscMat*,PetscMat*,PetscMat*)

    PetscErrorCode MatPythonSetType(PetscMat,char[])
    PetscErrorCode MatPythonGetType(PetscMat,char*[])

cdef extern from * nogil: # custom.h
    PetscErrorCode MatGetCurrentMemType(PetscMat,PetscMemType*)
    PetscErrorCode MatIsPreallocated(PetscMat,PetscBool*)
    PetscErrorCode MatHasPreallocationAIJ(PetscMat,PetscBool*,PetscBool*,PetscBool*,PetscBool*)

# -----------------------------------------------------------------------------

cdef extern from * nogil:
    PetscErrorCode MatNullSpaceDestroy(PetscNullSpace*)
    PetscErrorCode MatNullSpaceView(PetscNullSpace,PetscViewer)
    PetscErrorCode MatNullSpaceCreate(MPI_Comm,PetscBool,PetscInt,PetscVec[],
                           PetscNullSpace*)
    PetscErrorCode MatNullSpaceCreateRigidBody(PetscVec,PetscNullSpace*)
    PetscErrorCode MatNullSpaceGetVecs(PetscNullSpace,PetscBool*,PetscInt*,const PetscVec*[])
    PetscErrorCode MatNullSpaceRemove(PetscNullSpace,PetscVec)
    PetscErrorCode MatNullSpaceTest(PetscNullSpace,PetscMat,PetscBool*)

    ctypedef PetscErrorCode MatNullSpaceFunction(PetscNullSpace,
                                      PetscVec,
                                      void*) except PETSC_ERR_PYTHON
    PetscErrorCode MatNullSpaceSetFunction(PetscNullSpace,MatNullSpaceFunction*,void*)

    PetscErrorCode MatSetNullSpace(PetscMat,PetscNullSpace)
    PetscErrorCode MatGetNullSpace(PetscMat,PetscNullSpace*)
    PetscErrorCode MatSetTransposeNullSpace(PetscMat,PetscNullSpace)
    PetscErrorCode MatGetTransposeNullSpace(PetscMat,PetscNullSpace*)
    PetscErrorCode MatSetNearNullSpace(PetscMat,PetscNullSpace)
    PetscErrorCode MatGetNearNullSpace(PetscMat,PetscNullSpace*)

cdef inline NullSpace ref_NullSpace(PetscNullSpace nsp):
    cdef NullSpace ob = <NullSpace> NullSpace()
    ob.nsp = nsp
    PetscINCREF(ob.obj)
    return ob

cdef PetscErrorCode NullSpace_Function(
    PetscNullSpace n,
    PetscVec       v,
    void *         ctx,
    ) except PETSC_ERR_PYTHON with gil:
    cdef NullSpace nsp = ref_NullSpace(n)
    cdef Vec vec = ref_Vec(v)
    (function, args, kargs) = nsp.get_attr('__function__')
    function(nsp, vec, *args, **kargs)
    return PETSC_SUCCESS

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
        self.axpy(alpha, mat)
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
        self.axpy(-alpha, mat)
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
        other = 1/other
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
    cdef Vec result = self.createVecLeft()
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
    <void>self; <void>other; # unused
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

cdef inline PetscMatInfoType infotype(object info) \
    except <PetscMatInfoType>(-1):
    if   info is None: return MAT_GLOBAL_SUM
    else:              return info

# -----------------------------------------------------------------------------

cdef inline PetscErrorCode Mat_Sizes(
    object size, object bsize,
    PetscInt *r, PetscInt *c,
    PetscInt *m, PetscInt *n,
    PetscInt *M, PetscInt *N,
    ) except PETSC_ERR_PYTHON:
    # unpack row and column sizes
    cdef object rsize, csize
    try:
        rsize , csize = size
    except (TypeError, ValueError):
        rsize = csize = size
    # unpack row and column block sizes
    cdef object rbsize, cbsize
    try:
        rbsize , cbsize = bsize
    except (TypeError, ValueError):
        rbsize = cbsize = bsize
    # split row and column sizes
    Sys_Sizes(rsize, rbsize, r, m, M)
    Sys_Sizes(csize, cbsize, c, n, N)
    return PETSC_SUCCESS

cdef inline PetscErrorCode Mat_Create(
    PetscMatType mtype,
    object comm,
    object size,
    object bsize,
    PetscMat *A,
    ) except PETSC_ERR_PYTHON:
    # communicator
    cdef MPI_Comm ccomm = def_Comm(comm, PETSC_COMM_DEFAULT)
    # sizes and block sizes
    cdef PetscInt rbs = 0, cbs = 0, m = 0, n = 0, M = 0, N = 0
    Mat_Sizes(size, bsize, &rbs, &cbs, &m, &n, &M, &N)
    if rbs == PETSC_DECIDE: rbs = 1
    if cbs == PETSC_DECIDE: cbs = rbs
    Sys_Layout(ccomm, rbs, &m, &M)
    Sys_Layout(ccomm, cbs, &n, &N)
    # create matrix and set sizes
    cdef PetscMat mat = NULL
    CHKERR( MatCreate(ccomm, &mat) )
    CHKERR( MatSetSizes(mat, m, n, M, N) )
    CHKERR( MatSetBlockSizes(mat, rbs, cbs) )
    CHKERR( MatSetType(mat, mtype) )
    A[0] = mat
    return PETSC_SUCCESS

cdef inline PetscErrorCode Mat_AllocAIJ_NNZ( PetscMat A, object NNZ) except PETSC_ERR_PYTHON:
    #
    cdef PetscBool aij=PETSC_FALSE, baij=PETSC_FALSE, sbaij=PETSC_FALSE, aijis=PETSC_FALSE
    CHKERR( MatHasPreallocationAIJ(A, &aij, &baij, &sbaij, &aijis))
    # local row size and block size
    cdef PetscInt m=0, bs=1
    CHKERR( MatGetLocalSize(A, &m, NULL) )
    if baij == PETSC_TRUE or sbaij == PETSC_TRUE:
        CHKERR( MatGetBlockSize(A, &bs) )
        assert bs > 0, "block size not set"
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
    if m == PETSC_DECIDE:
        if d_n > 1 and d_n*bs > m: m = d_n*bs
        if o_n > 1 and o_n*bs > m: m = o_n*bs
    # check array sizes
    if d_n > 1 and d_n*bs != m: raise ValueError(
        "size(d_nnz) is %d, expected %d" %
            (toInt(d_n), toInt(m//bs)) )
    if o_n > 1 and o_n*bs != m: raise ValueError(
        "size(o_nnz) is %d, expected %d" %
            (toInt(o_n), toInt(m//bs)) )
    # preallocate
    if aij == PETSC_TRUE:
        CHKERR( MatSeqAIJSetPreallocation(A, d_nz, d_nnz) )
        CHKERR( MatMPIAIJSetPreallocation(A, d_nz, d_nnz, o_nz, o_nnz) )
    if baij == PETSC_TRUE:
        CHKERR( MatSeqBAIJSetPreallocation(A, bs, d_nz, d_nnz) )
        CHKERR( MatMPIBAIJSetPreallocation(A, bs, d_nz, d_nnz, o_nz, o_nnz) )
    if sbaij == PETSC_TRUE:
        CHKERR( MatSeqSBAIJSetPreallocation(A, bs, d_nz, d_nnz) )
        CHKERR( MatMPISBAIJSetPreallocation(A, bs, d_nz, d_nnz, o_nz, o_nnz) )
    if aijis == PETSC_TRUE:
        CHKERR( MatISSetPreallocation(A, d_nz, d_nnz, o_nz, o_nnz) )
    return PETSC_SUCCESS

cdef inline PetscErrorCode Mat_AllocAIJ_CSR(PetscMat A, object CSR) except PETSC_ERR_PYTHON:
    #
    cdef PetscBool aij=PETSC_FALSE, baij=PETSC_FALSE, sbaij=PETSC_FALSE, aijis=PETSC_FALSE
    CHKERR( MatHasPreallocationAIJ(A, &aij, &baij, &sbaij, &aijis))
    # local row size and block size
    cdef PetscInt m=0, bs = 1
    CHKERR( MatGetLocalSize(A, &m, NULL) )
    if baij == PETSC_TRUE or sbaij == PETSC_TRUE:
        CHKERR( MatGetBlockSize(A, &bs) )
        assert bs > 0, "block size not set"
    # unpack CSR argument
    cdef object oi, oj, ov
    try:
        oi, oj, ov = CSR
    except (TypeError, ValueError):
        oi, oj = CSR; ov = None
    # rows, cols, and values
    cdef PetscInt ni=0, *i=NULL
    cdef PetscInt nj=0, *j=NULL
    cdef PetscInt nv=0
    cdef PetscScalar *v=NULL
    oi = iarray_i(oi, &ni, &i)
    oj = iarray_i(oj, &nj, &j)
    if ov is not None:
        ov = iarray_s(ov, &nv, &v)
    if m == PETSC_DECIDE: m = (ni-1)*bs
    # check array sizes
    if ((ni-1)*bs != m):
        raise ValueError("size(I) is %d, expected %d" %
                         (toInt(ni), toInt(m//bs+1)) )
    if (i[0] != 0):
        raise ValueError("I[0] is %d, expected %d" %
                         (toInt(i[0]), toInt(0)) )
    if (i[ni-1] != nj):
        raise ValueError("size(J) is %d, expected %d" %
                         (toInt(nj), toInt(i[ni-1])) )
    if v != NULL and (nj*bs*bs != nv):
        raise ValueError("size(V) is %d, expected %d" %
                         (toInt(nv), toInt(nj*bs*bs)) )
    # preallocate
    if aij == PETSC_TRUE:
        CHKERR( MatSeqAIJSetPreallocationCSR(A, i, j, v) )
        CHKERR( MatMPIAIJSetPreallocationCSR(A, i, j, v) )
    if baij == PETSC_TRUE:
        CHKERR( MatSeqBAIJSetPreallocationCSR(A, bs, i, j, v) )
        CHKERR( MatMPIBAIJSetPreallocationCSR(A, bs, i, j, v) )
    if sbaij == PETSC_TRUE:
        CHKERR( MatSeqSBAIJSetPreallocationCSR(A, bs, i, j, v) )
        CHKERR( MatMPISBAIJSetPreallocationCSR(A, bs, i, j, v) )
    return PETSC_SUCCESS

cdef inline PetscErrorCode Mat_AllocAIJ(PetscMat A,object NNZ, object CSR) except PETSC_ERR_PYTHON:
    if CSR is not None:
        return Mat_AllocAIJ_CSR(A, CSR)
    if NNZ is not None:
        return Mat_AllocAIJ_NNZ(A, NNZ)
    return PETSC_SUCCESS

cdef inline object Mat_AllocDense(PetscMat A, object array):
    cdef PetscInt m=0, N=0
    CHKERR( MatGetLocalSize(A, &m, NULL) )
    CHKERR( MatGetSize(A, NULL, &N) )
    cdef PetscInt size=0
    cdef PetscScalar *data=NULL
    if array is not None:
        array = ofarray_s(array, &size, &data)
        if m*N != size: raise ValueError(
            "size(array) is %d, expected %dx%d=%d" %
            (toInt(size), toInt(m), toInt(N), toInt(m*N)) )
    CHKERR( MatSeqDenseSetPreallocation(A, data) )
    CHKERR( MatMPIDenseSetPreallocation(A, data) )
    return array

# -----------------------------------------------------------------------------

ctypedef PetscErrorCode MatSetValuesFcn(PetscMat,
                             PetscInt,const PetscInt*,
                             PetscInt,const PetscInt*,
                             const PetscScalar*,PetscInsertMode)

cdef inline MatSetValuesFcn* matsetvalues_fcn(int blocked, int local):
    cdef MatSetValuesFcn *setvalues = NULL
    if blocked and local: setvalues = MatSetValuesBlockedLocal
    elif blocked:         setvalues = MatSetValuesBlocked
    elif local:           setvalues = MatSetValuesLocal
    else:                 setvalues = MatSetValues
    return setvalues

cdef inline PetscErrorCode matsetvalues(PetscMat A,
                             object oi, object oj, object ov,
                             object oaddv, int blocked, int local) except PETSC_ERR_PYTHON:
    # block size
    cdef PetscInt rbs=1, cbs=1
    if blocked: CHKERR( MatGetBlockSizes(A, &rbs, &cbs) )
    if rbs < 1: rbs = 1
    if cbs < 1: cbs = 1
    # rows, cols, and values
    cdef PetscInt ni=0, *i=NULL
    cdef PetscInt nj=0, *j=NULL
    cdef PetscInt nv=0
    cdef PetscScalar *v=NULL
    oi = iarray_i(oi, &ni, &i)
    oj = iarray_i(oj, &nj, &j)
    ov = iarray_s(ov, &nv, &v)
    if ni*nj*rbs*cbs != nv: raise ValueError(
        "incompatible array sizes: ni=%d, nj=%d, nv=%d" %
        (toInt(ni), toInt(nj), toInt(nv)) )
    # MatSetValuesXXX function and insert mode
    cdef MatSetValuesFcn *setvalues = matsetvalues_fcn(blocked, local)
    cdef PetscInsertMode addv = insertmode(oaddv)
    # actual call
    CHKERR( setvalues(A, ni, i, nj, j, v, addv) )
    return PETSC_SUCCESS

cdef inline PetscErrorCode matsetvalues_rcv(PetscMat A,
                                 object oi, object oj, object ov,
                                 object oaddv,
                                 int blocked, int local) except PETSC_ERR_PYTHON:
    # block size
    cdef PetscInt rbs=1, cbs=1
    if blocked: CHKERR( MatGetBlockSizes(A, &rbs, &cbs) )
    if rbs < 1: rbs = 1
    if cbs < 1: cbs = 1
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
        (si*rbs * sj*cbs != sv)): raise ValueError(
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
    return PETSC_SUCCESS

cdef inline PetscErrorCode matsetvalues_ijv(PetscMat A,
                                 object oi, object oj, object ov,
                                 object oaddv,
                                 object om,
                                 int blocked, int local) except PETSC_ERR_PYTHON:
    # block size
    cdef PetscInt rbs=1, cbs=1
    if blocked: CHKERR( MatGetBlockSizes(A, &rbs, &cbs) )
    if rbs < 1: rbs = 1
    if cbs < 1: cbs = 1
    # column pointers, column indices, and values
    cdef PetscInt ni=0, *i=NULL
    cdef PetscInt nj=0, *j=NULL
    cdef PetscInt nv=0
    cdef PetscScalar *v=NULL
    oi = iarray_i(oi, &ni, &i)
    oj = iarray_i(oj, &nj, &j)
    ov = iarray_s(ov, &nv, &v)
    # row indices
    cdef PetscInt nm=0, *m=NULL
    cdef PetscInt rs=0, re=ni-1
    if om is not None:
        om = iarray_i(om, &nm, &m)
    else:
        if not local:
            CHKERR( MatGetOwnershipRange(A, &rs, &re) )
            rs //= rbs; re //= rbs
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
    if (nj*rbs*cbs != nv): raise ValueError(
        "size(V) is %d, expected %d" %
        (toInt(nv), toInt(nj*rbs*cbs)) )
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
            sval = v + i[k]*rbs*cbs
            for l from 0 <= l < ncol:
                CHKERR( setvalues(A, 1, &irow, 1, &icol[l],
                                  &sval[l*rbs*cbs], addv) )
        else:
            sval = v + i[k]
            CHKERR( setvalues(A, 1, &irow, ncol, icol, sval, addv) )
    return PETSC_SUCCESS

cdef inline PetscErrorCode matsetvalues_csr(PetscMat A,
                                 object oi, object oj, object ov,
                                 object oaddv,
                                 int blocked, int local) except PETSC_ERR_PYTHON:
    matsetvalues_ijv(A, oi, oj, ov, oaddv, None, blocked, local)
    return PETSC_SUCCESS

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

cdef extern from * nogil: # custom.h
    PetscErrorCode MatFactorInfoDefaults(PetscBool,PetscBool,PetscMatFactorInfo*)

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

cdef PetscErrorCode matfactorinfo(PetscBool inc, PetscBool chol, object opts,
                       PetscMatFactorInfo *info) except PETSC_ERR_PYTHON:
    CHKERR( MatFactorInfoDefaults(inc,chol,info) )
    if opts is None: return PETSC_SUCCESS
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
    return PETSC_SUCCESS

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


cdef PetscErrorCode mat_setitem(Mat self, object ij, object v) except PETSC_ERR_PYTHON:
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
    return PETSC_SUCCESS

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
    cdef PetscInt rbs=1, cbs=1
    if blocked: CHKERR( MatGetBlockSizes(A, &rbs, &cbs) )
    if rbs < 1: rbs = 1
    if cbs < 1: cbs = 1
    # values
    cdef PetscInt    nv = 1
    cdef PetscScalar *v = NULL
    value = iarray_s(value, &nv, &v)
    if rbs*cbs != nv: raise ValueError(
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

cdef mat_get_dlpack_ctx(Mat self):
    if 'dense' not in self.getType():
        raise NotImplementedError("Not for type {}".format(self.getType()))
    cdef object ctx0 = self.get_attr('__dltensor_ctx__')
    cdef PetscInt n = 0, m = 0, lda = 0
    cdef int64_t ndim = 2
    cdef int64_t* shape_arr = NULL
    cdef int64_t* strides_arr = NULL
    cdef object s1 = None
    cdef object s2 = None
    cdef PetscInt devId = 0
    cdef PetscMemType mtype = PETSC_MEMTYPE_HOST
    if ctx0 is None: # First time in, create a linear memory view
        s1 = oarray_p(empty_p(ndim), NULL, <void**>&shape_arr)
        s2 = oarray_p(empty_p(ndim), NULL, <void**>&strides_arr)
        CHKERR( MatGetSize(self.mat, NULL, &n) )
        CHKERR( MatGetLocalSize(self.mat, &m, NULL) )
        CHKERR( MatDenseGetLDA(self.mat, &lda) )
        shape_arr[0] = <int64_t>m
        shape_arr[1] = <int64_t>n
        strides_arr[0] = 1
        strides_arr[1] = <int64_t>lda
    else:
        (_, _, ndim, s1, s2) = ctx0

    devType_ = { PETSC_MEMTYPE_HOST : kDLCPU, PETSC_MEMTYPE_CUDA : kDLCUDA }
    CHKERR( MatGetCurrentMemType(self.mat, &mtype) )
    dtype = devType_.get(mtype, kDLCPU)
    if dtype != kDLCPU:
        CHKERR( PetscObjectGetDeviceId(<PetscObject>self.mat, &devId) )
    ctx0 = (dtype, devId, ndim, s1, s2)
    self.set_attr('__dltensor_ctx__', ctx0)
    return ctx0

# -----------------------------------------------------------------------------
