/* Portions of this code are under:
   Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
*/
#include <petsc/private/matimpl.h>

const char *MatOptions_Shifted[] = {"UNUSED_NONZERO_LOCATION_ERR", "ROW_ORIENTED", "NOT_A_VALID_OPTION", "SYMMETRIC", "STRUCTURALLY_SYMMETRIC", "FORCE_DIAGONAL_ENTRIES", "IGNORE_OFF_PROC_ENTRIES", "USE_HASH_TABLE", "KEEP_NONZERO_PATTERN", "IGNORE_ZERO_ENTRIES", "USE_INODES", "HERMITIAN", "SYMMETRY_ETERNAL", "NEW_NONZERO_LOCATION_ERR", "IGNORE_LOWER_TRIANGULAR", "ERROR_LOWER_TRIANGULAR", "GETROW_UPPERTRIANGULAR", "SPD", "NO_OFF_PROC_ZERO_ROWS", "NO_OFF_PROC_ENTRIES", "NEW_NONZERO_LOCATIONS", "NEW_NONZERO_ALLOCATION_ERR", "SUBSET_OFF_PROC_ENTRIES", "SUBMAT_SINGLEIS", "STRUCTURE_ONLY", "SORTED_FULL", "FORM_EXPLICIT_TRANSPOSE", "STRUCTURAL_SYMMETRY_ETERNAL", "SPD_ETERNAL", "MatOption", "MAT_", NULL};
const char *const *MatOptions                  = MatOptions_Shifted + 2;
const char *const  MatFactorShiftTypes[]       = {"NONE", "NONZERO", "POSITIVE_DEFINITE", "INBLOCKS", "MatFactorShiftType", "PC_FACTOR_", NULL};
const char *const  MatStructures[]             = {"DIFFERENT", "SUBSET", "SAME", "UNKNOWN", "MatStructure", "MAT_STRUCTURE_", NULL};
const char *const  MatFactorShiftTypesDetail[] = {NULL, "diagonal shift to prevent zero pivot", "Manteuffel shift", "diagonal shift on blocks to prevent zero pivot"};
const char *const  MPPTScotchStrategyTypes[]   = {"DEFAULT", "QUALITY", "SPEED", "BALANCE", "SAFETY", "SCALABILITY", "MPPTScotchStrategyType", "MP_PTSCOTCH_", NULL};
const char *const  MPChacoGlobalTypes[]        = {"", "MULTILEVEL", "SPECTRAL", "", "LINEAR", "RANDOM", "SCATTERED", "MPChacoGlobalType", "MP_CHACO_", NULL};
const char *const  MPChacoLocalTypes[]         = {"", "KERNIGHAN", "NONE", "MPChacoLocalType", "MP_CHACO_", NULL};
const char *const  MPChacoEigenTypes[]         = {"LANCZOS", "RQI", "MPChacoEigenType", "MP_CHACO_", NULL};

extern PetscErrorCode MatMFFDInitializePackage(void);
extern PetscErrorCode MatSolverTypeDestroy(void);
static PetscBool      MatPackageInitialized = PETSC_FALSE;
/*@C
  MatFinalizePackage - This function destroys everything in the Petsc interface to the `Mat` package. It is
  called from `PetscFinalize()`.

  Level: developer

.seealso: `Mat`, `PetscFinalize()`, `MatInitializePackage()`
@*/
PetscErrorCode MatFinalizePackage(void)
{
  MatRootName nnames, names = MatRootNameList;

  PetscFunctionBegin;
  PetscCall(MatSolverTypeDestroy());
  while (names) {
    nnames = names->next;
    PetscCall(PetscFree(names->rname));
    PetscCall(PetscFree(names->sname));
    PetscCall(PetscFree(names->mname));
    PetscCall(PetscFree(names));
    names = nnames;
  }
  PetscCall(PetscFunctionListDestroy(&MatList));
  PetscCall(PetscFunctionListDestroy(&MatOrderingList));
  PetscCall(PetscFunctionListDestroy(&MatColoringList));
  PetscCall(PetscFunctionListDestroy(&MatPartitioningList));
  PetscCall(PetscFunctionListDestroy(&MatCoarsenList));
  MatRootNameList                  = NULL;
  MatPackageInitialized            = PETSC_FALSE;
  MatRegisterAllCalled             = PETSC_FALSE;
  MatOrderingRegisterAllCalled     = PETSC_FALSE;
  MatColoringRegisterAllCalled     = PETSC_FALSE;
  MatPartitioningRegisterAllCalled = PETSC_FALSE;
  MatCoarsenRegisterAllCalled      = PETSC_FALSE;
  /* this is not ideal because it exposes SeqAIJ implementation details directly into the base Mat code */
  PetscCall(PetscFunctionListDestroy(&MatSeqAIJList));
  MatSeqAIJRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_MUMPS)
PETSC_EXTERN PetscErrorCode MatSolverTypeRegister_MUMPS(void);
#endif
#if defined(PETSC_HAVE_CUDA)
PETSC_EXTERN PetscErrorCode MatSolverTypeRegister_CUSPARSE(void);
#endif
#if defined(PETSC_HAVE_HIP)
PETSC_EXTERN PetscErrorCode MatSolverTypeRegister_HIPSPARSE(void);
#endif
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
PETSC_EXTERN PetscErrorCode MatSolverTypeRegister_KOKKOS(void);
#endif
#if defined(PETSC_HAVE_VIENNACL)
PETSC_EXTERN PetscErrorCode MatSolverTypeRegister_ViennaCL(void);
#endif
#if defined(PETSC_HAVE_ELEMENTAL)
PETSC_EXTERN PetscErrorCode MatSolverTypeRegister_Elemental(void);
#endif
#if defined(PETSC_HAVE_SCALAPACK)
PETSC_EXTERN PetscErrorCode MatSolverTypeRegister_ScaLAPACK(void);
#endif
#if defined(PETSC_HAVE_MATLAB)
PETSC_EXTERN PetscErrorCode MatSolverTypeRegister_Matlab(void);
#endif
#if defined(PETSC_HAVE_ESSL)
PETSC_EXTERN PetscErrorCode MatSolverTypeRegister_Essl(void);
#endif
#if defined(PETSC_HAVE_SUPERLU)
PETSC_EXTERN PetscErrorCode MatSolverTypeRegister_SuperLU(void);
#endif
#if defined(PETSC_HAVE_STRUMPACK)
PETSC_EXTERN PetscErrorCode MatSolverTypeRegister_STRUMPACK(void);
#endif
#if defined(PETSC_HAVE_PASTIX)
PETSC_EXTERN PetscErrorCode MatSolverTypeRegister_Pastix(void);
#endif
#if defined(PETSC_HAVE_SUPERLU_DIST)
PETSC_EXTERN PetscErrorCode MatSolverTypeRegister_SuperLU_DIST(void);
#endif
#if defined(PETSC_HAVE_ELEMENTAL)
PETSC_EXTERN PetscErrorCode MatSolverTypeRegister_SparseElemental(void);
#endif
#if defined(PETSC_HAVE_MKL_PARDISO)
PETSC_EXTERN PetscErrorCode MatSolverTypeRegister_MKL_Pardiso(void);
#endif
#if defined(PETSC_HAVE_MKL_CPARDISO)
PETSC_EXTERN PetscErrorCode MatSolverTypeRegister_MKL_CPardiso(void);
#endif
#if defined(PETSC_HAVE_SUITESPARSE)
PETSC_EXTERN PetscErrorCode MatSolverTypeRegister_SuiteSparse(void);
#endif
#if defined(PETSC_HAVE_LUSOL)
PETSC_EXTERN PetscErrorCode MatSolverTypeRegister_Lusol(void);
#endif

PETSC_INTERN PetscErrorCode MatGetFactor_seqaij_petsc(Mat, MatFactorType, Mat *);
PETSC_INTERN PetscErrorCode MatGetFactor_seqbaij_petsc(Mat, MatFactorType, Mat *);
PETSC_INTERN PetscErrorCode MatGetFactor_seqsbaij_petsc(Mat, MatFactorType, Mat *);
PETSC_INTERN PetscErrorCode MatGetFactor_seqdense_petsc(Mat, MatFactorType, Mat *);
#if defined(PETSC_HAVE_CUDA)
PETSC_INTERN PetscErrorCode MatGetFactor_seqdense_cuda(Mat, MatFactorType, Mat *);
#endif
#if defined(PETSC_HAVE_HIP)
PETSC_INTERN PetscErrorCode MatGetFactor_seqdense_hip(Mat, MatFactorType, Mat *);
#endif
PETSC_INTERN PetscErrorCode MatGetFactor_constantdiagonal_petsc(Mat, MatFactorType, Mat *);
PETSC_INTERN PetscErrorCode MatGetFactor_seqaij_bas(Mat, MatFactorType, Mat *);

/*@C
  MatInitializePackage - This function initializes everything in the `Mat` package. It is called
  from `PetscDLLibraryRegister_petscmat()` when using dynamic libraries, and on the first call to `MatCreate()`
  when using shared or static libraries.

  Level: developer

.seealso: `Mat`, `PetscInitialize()`, `MatFinalizePackage()`
@*/
PetscErrorCode MatInitializePackage(void)
{
  char      logList[256];
  PetscBool opt, pkg;

  PetscFunctionBegin;
  if (MatPackageInitialized) PetscFunctionReturn(0);
  MatPackageInitialized = PETSC_TRUE;
  /* Initialize subpackage */
  PetscCall(MatMFFDInitializePackage());
  /* Register Classes */
  PetscCall(PetscClassIdRegister("Matrix", &MAT_CLASSID));
  PetscCall(PetscClassIdRegister("Matrix FD Coloring", &MAT_FDCOLORING_CLASSID));
  PetscCall(PetscClassIdRegister("Matrix Coloring", &MAT_COLORING_CLASSID));
  PetscCall(PetscClassIdRegister("Matrix MatTranspose Coloring", &MAT_TRANSPOSECOLORING_CLASSID));
  PetscCall(PetscClassIdRegister("Matrix Partitioning", &MAT_PARTITIONING_CLASSID));
  PetscCall(PetscClassIdRegister("Matrix Coarsen", &MAT_COARSEN_CLASSID));
  PetscCall(PetscClassIdRegister("Matrix Null Space", &MAT_NULLSPACE_CLASSID));
  /* Register Constructors */
  PetscCall(MatRegisterAll());
  PetscCall(MatOrderingRegisterAll());
  PetscCall(MatColoringRegisterAll());
  PetscCall(MatPartitioningRegisterAll());
  PetscCall(MatCoarsenRegisterAll());
  PetscCall(MatSeqAIJRegisterAll());
  /* Register Events */
  PetscCall(PetscLogEventRegister("MatMult", MAT_CLASSID, &MAT_Mult));
  PetscCall(PetscLogEventRegister("MatMults", MAT_CLASSID, &MAT_Mults));
  PetscCall(PetscLogEventRegister("MatMultAdd", MAT_CLASSID, &MAT_MultAdd));
  PetscCall(PetscLogEventRegister("MatMultTranspose", MAT_CLASSID, &MAT_MultTranspose));
  PetscCall(PetscLogEventRegister("MatMultTrAdd", MAT_CLASSID, &MAT_MultTransposeAdd));
  PetscCall(PetscLogEventRegister("MatSolve", MAT_CLASSID, &MAT_Solve));
  PetscCall(PetscLogEventRegister("MatSolves", MAT_CLASSID, &MAT_Solves));
  PetscCall(PetscLogEventRegister("MatSolveAdd", MAT_CLASSID, &MAT_SolveAdd));
  PetscCall(PetscLogEventRegister("MatSolveTranspos", MAT_CLASSID, &MAT_SolveTranspose));
  PetscCall(PetscLogEventRegister("MatSolveTrAdd", MAT_CLASSID, &MAT_SolveTransposeAdd));
  PetscCall(PetscLogEventRegister("MatSOR", MAT_CLASSID, &MAT_SOR));
  PetscCall(PetscLogEventRegister("MatForwardSolve", MAT_CLASSID, &MAT_ForwardSolve));
  PetscCall(PetscLogEventRegister("MatBackwardSolve", MAT_CLASSID, &MAT_BackwardSolve));
  PetscCall(PetscLogEventRegister("MatLUFactor", MAT_CLASSID, &MAT_LUFactor));
  PetscCall(PetscLogEventRegister("MatLUFactorSym", MAT_CLASSID, &MAT_LUFactorSymbolic));
  PetscCall(PetscLogEventRegister("MatLUFactorNum", MAT_CLASSID, &MAT_LUFactorNumeric));
  PetscCall(PetscLogEventRegister("MatQRFactor", MAT_CLASSID, &MAT_QRFactor));
  PetscCall(PetscLogEventRegister("MatQRFactorSym", MAT_CLASSID, &MAT_QRFactorSymbolic));
  PetscCall(PetscLogEventRegister("MatQRFactorNum", MAT_CLASSID, &MAT_QRFactorNumeric));
  PetscCall(PetscLogEventRegister("MatCholeskyFctr", MAT_CLASSID, &MAT_CholeskyFactor));
  PetscCall(PetscLogEventRegister("MatCholFctrSym", MAT_CLASSID, &MAT_CholeskyFactorSymbolic));
  PetscCall(PetscLogEventRegister("MatCholFctrNum", MAT_CLASSID, &MAT_CholeskyFactorNumeric));
  PetscCall(PetscLogEventRegister("MatFctrFactSchur", MAT_CLASSID, &MAT_FactorFactS));
  PetscCall(PetscLogEventRegister("MatFctrInvSchur", MAT_CLASSID, &MAT_FactorInvS));
  PetscCall(PetscLogEventRegister("MatILUFactor", MAT_CLASSID, &MAT_ILUFactor));
  PetscCall(PetscLogEventRegister("MatILUFactorSym", MAT_CLASSID, &MAT_ILUFactorSymbolic));
  PetscCall(PetscLogEventRegister("MatICCFactorSym", MAT_CLASSID, &MAT_ICCFactorSymbolic));
  PetscCall(PetscLogEventRegister("MatCopy", MAT_CLASSID, &MAT_Copy));
  PetscCall(PetscLogEventRegister("MatConvert", MAT_CLASSID, &MAT_Convert));
  PetscCall(PetscLogEventRegister("MatScale", MAT_CLASSID, &MAT_Scale));
  PetscCall(PetscLogEventRegister("MatResidual", MAT_CLASSID, &MAT_Residual));
  PetscCall(PetscLogEventRegister("MatAssemblyBegin", MAT_CLASSID, &MAT_AssemblyBegin));
  PetscCall(PetscLogEventRegister("MatAssemblyEnd", MAT_CLASSID, &MAT_AssemblyEnd));
  PetscCall(PetscLogEventRegister("MatSetValues", MAT_CLASSID, &MAT_SetValues));
  PetscCall(PetscLogEventRegister("MatGetValues", MAT_CLASSID, &MAT_GetValues));
  PetscCall(PetscLogEventRegister("MatGetRow", MAT_CLASSID, &MAT_GetRow));
  PetscCall(PetscLogEventRegister("MatGetRowIJ", MAT_CLASSID, &MAT_GetRowIJ));
  PetscCall(PetscLogEventRegister("MatCreateSubMats", MAT_CLASSID, &MAT_CreateSubMats));
  PetscCall(PetscLogEventRegister("MatCreateSubMat", MAT_CLASSID, &MAT_CreateSubMat));
  PetscCall(PetscLogEventRegister("MatGetOrdering", MAT_CLASSID, &MAT_GetOrdering));
  PetscCall(PetscLogEventRegister("MatIncreaseOvrlp", MAT_CLASSID, &MAT_IncreaseOverlap));
  PetscCall(PetscLogEventRegister("MatPartitioning", MAT_PARTITIONING_CLASSID, &MAT_Partitioning));
  PetscCall(PetscLogEventRegister("MatPartitioningND", MAT_PARTITIONING_CLASSID, &MAT_PartitioningND));
  PetscCall(PetscLogEventRegister("MatCoarsen", MAT_COARSEN_CLASSID, &MAT_Coarsen));
  PetscCall(PetscLogEventRegister("MatZeroEntries", MAT_CLASSID, &MAT_ZeroEntries));
  PetscCall(PetscLogEventRegister("MatLoad", MAT_CLASSID, &MAT_Load));
  PetscCall(PetscLogEventRegister("MatView", MAT_CLASSID, &MAT_View));
  PetscCall(PetscLogEventRegister("MatAXPY", MAT_CLASSID, &MAT_AXPY));
  PetscCall(PetscLogEventRegister("MatFDColorCreate", MAT_FDCOLORING_CLASSID, &MAT_FDColoringCreate));
  PetscCall(PetscLogEventRegister("MatFDColorSetUp", MAT_FDCOLORING_CLASSID, &MAT_FDColoringSetUp));
  PetscCall(PetscLogEventRegister("MatFDColorApply", MAT_FDCOLORING_CLASSID, &MAT_FDColoringApply));
  PetscCall(PetscLogEventRegister("MatFDColorFunc", MAT_FDCOLORING_CLASSID, &MAT_FDColoringFunction));
  PetscCall(PetscLogEventRegister("MatTranspose", MAT_CLASSID, &MAT_Transpose));
  PetscCall(PetscLogEventRegister("MatMatSolve", MAT_CLASSID, &MAT_MatSolve));
  PetscCall(PetscLogEventRegister("MatMatTrSolve", MAT_CLASSID, &MAT_MatTrSolve));
  PetscCall(PetscLogEventRegister("MatMatMultSym", MAT_CLASSID, &MAT_MatMultSymbolic));
  PetscCall(PetscLogEventRegister("MatMatMultNum", MAT_CLASSID, &MAT_MatMultNumeric));
  PetscCall(PetscLogEventRegister("MatMatMatMultSym", MAT_CLASSID, &MAT_MatMatMultSymbolic));
  PetscCall(PetscLogEventRegister("MatMatMatMultNum", MAT_CLASSID, &MAT_MatMatMultNumeric));
  PetscCall(PetscLogEventRegister("MatPtAPSymbolic", MAT_CLASSID, &MAT_PtAPSymbolic));
  PetscCall(PetscLogEventRegister("MatPtAPNumeric", MAT_CLASSID, &MAT_PtAPNumeric));
  PetscCall(PetscLogEventRegister("MatRARtSym", MAT_CLASSID, &MAT_RARtSymbolic));
  PetscCall(PetscLogEventRegister("MatRARtNum", MAT_CLASSID, &MAT_RARtNumeric));
  PetscCall(PetscLogEventRegister("MatMatTrnMultSym", MAT_CLASSID, &MAT_MatTransposeMultSymbolic));
  PetscCall(PetscLogEventRegister("MatMatTrnMultNum", MAT_CLASSID, &MAT_MatTransposeMultNumeric));
  PetscCall(PetscLogEventRegister("MatTrnMatMultSym", MAT_CLASSID, &MAT_TransposeMatMultSymbolic));
  PetscCall(PetscLogEventRegister("MatTrnMatMultNum", MAT_CLASSID, &MAT_TransposeMatMultNumeric));
  PetscCall(PetscLogEventRegister("MatTrnColorCreate", MAT_CLASSID, &MAT_TransposeColoringCreate));
  PetscCall(PetscLogEventRegister("MatRedundantMat", MAT_CLASSID, &MAT_RedundantMat));
  PetscCall(PetscLogEventRegister("MatGetSeqNZStrct", MAT_CLASSID, &MAT_GetSequentialNonzeroStructure));
  PetscCall(PetscLogEventRegister("MatGetMultiProcB", MAT_CLASSID, &MAT_GetMultiProcBlock));
  PetscCall(PetscLogEventRegister("MatSetRandom", MAT_CLASSID, &MAT_SetRandom));

  /* these may be specific to MPIAIJ matrices */
  PetscCall(PetscLogEventRegister("MatMPISumSeqNumeric", MAT_CLASSID, &MAT_Seqstompinum));
  PetscCall(PetscLogEventRegister("MatMPISumSeqSymbolic", MAT_CLASSID, &MAT_Seqstompisym));
  PetscCall(PetscLogEventRegister("MatMPISumSeq", MAT_CLASSID, &MAT_Seqstompi));
  PetscCall(PetscLogEventRegister("MatMPIConcateSeq", MAT_CLASSID, &MAT_Merge));
  PetscCall(PetscLogEventRegister("MatGetLocalMat", MAT_CLASSID, &MAT_Getlocalmat));
  PetscCall(PetscLogEventRegister("MatGetLocalMatCondensed", MAT_CLASSID, &MAT_Getlocalmatcondensed));
  PetscCall(PetscLogEventRegister("MatGetBrowsOfAcols", MAT_CLASSID, &MAT_GetBrowsOfAcols));
  PetscCall(PetscLogEventRegister("MatGetBrAoCol", MAT_CLASSID, &MAT_GetBrowsOfAocols));

  PetscCall(PetscLogEventRegister("MatApplyPAPt_Symbolic", MAT_CLASSID, &MAT_Applypapt_symbolic));
  PetscCall(PetscLogEventRegister("MatApplyPAPt_Numeric", MAT_CLASSID, &MAT_Applypapt_numeric));
  PetscCall(PetscLogEventRegister("MatApplyPAPt", MAT_CLASSID, &MAT_Applypapt));

  PetscCall(PetscLogEventRegister("MatGetSymTrans", MAT_CLASSID, &MAT_Getsymtranspose));
  PetscCall(PetscLogEventRegister("MatGetSymTransR", MAT_CLASSID, &MAT_Getsymtransreduced));
  PetscCall(PetscLogEventRegister("MatCUSPARSCopyTo", MAT_CLASSID, &MAT_CUSPARSECopyToGPU));
  PetscCall(PetscLogEventRegister("MatCUSPARSCopyFr", MAT_CLASSID, &MAT_CUSPARSECopyFromGPU));
  PetscCall(PetscLogEventRegister("MatCUSPARSSolAnl", MAT_CLASSID, &MAT_CUSPARSESolveAnalysis));
  PetscCall(PetscLogEventRegister("MatCUSPARSGenT", MAT_CLASSID, &MAT_CUSPARSEGenerateTranspose));
  PetscCall(PetscLogEventRegister("MatHIPSPARSCopyTo", MAT_CLASSID, &MAT_HIPSPARSECopyToGPU));
  PetscCall(PetscLogEventRegister("MatHIPSPARSCopyFr", MAT_CLASSID, &MAT_HIPSPARSECopyFromGPU));
  PetscCall(PetscLogEventRegister("MatHIPSPARSSolAnl", MAT_CLASSID, &MAT_HIPSPARSESolveAnalysis));
  PetscCall(PetscLogEventRegister("MatHIPSPARSGenT", MAT_CLASSID, &MAT_HIPSPARSEGenerateTranspose));
  PetscCall(PetscLogEventRegister("MatVCLCopyTo", MAT_CLASSID, &MAT_ViennaCLCopyToGPU));
  PetscCall(PetscLogEventRegister("MatDenseCopyTo", MAT_CLASSID, &MAT_DenseCopyToGPU));
  PetscCall(PetscLogEventRegister("MatDenseCopyFrom", MAT_CLASSID, &MAT_DenseCopyFromGPU));
  PetscCall(PetscLogEventRegister("MatSetValBatch", MAT_CLASSID, &MAT_SetValuesBatch));

  PetscCall(PetscLogEventRegister("MatColoringApply", MAT_COLORING_CLASSID, &MATCOLORING_Apply));
  PetscCall(PetscLogEventRegister("MatColoringComm", MAT_COLORING_CLASSID, &MATCOLORING_Comm));
  PetscCall(PetscLogEventRegister("MatColoringLocal", MAT_COLORING_CLASSID, &MATCOLORING_Local));
  PetscCall(PetscLogEventRegister("MatColoringIS", MAT_COLORING_CLASSID, &MATCOLORING_ISCreate));
  PetscCall(PetscLogEventRegister("MatColoringSetUp", MAT_COLORING_CLASSID, &MATCOLORING_SetUp));
  PetscCall(PetscLogEventRegister("MatColoringWeights", MAT_COLORING_CLASSID, &MATCOLORING_Weights));

  PetscCall(PetscLogEventRegister("MatSetPreallCOO", MAT_CLASSID, &MAT_PreallCOO));
  PetscCall(PetscLogEventRegister("MatSetValuesCOO", MAT_CLASSID, &MAT_SetVCOO));

  PetscCall(PetscLogEventRegister("MatH2OpusBuild", MAT_CLASSID, &MAT_H2Opus_Build));
  PetscCall(PetscLogEventRegister("MatH2OpusComp", MAT_CLASSID, &MAT_H2Opus_Compress));
  PetscCall(PetscLogEventRegister("MatH2OpusOrth", MAT_CLASSID, &MAT_H2Opus_Orthog));
  PetscCall(PetscLogEventRegister("MatH2OpusLR", MAT_CLASSID, &MAT_H2Opus_LR));

  /* Mark non-collective events */
  PetscCall(PetscLogEventSetCollective(MAT_SetValues, PETSC_FALSE));
  PetscCall(PetscLogEventSetCollective(MAT_SetValuesBatch, PETSC_FALSE));
  PetscCall(PetscLogEventSetCollective(MAT_GetRow, PETSC_FALSE));
  /* Turn off high traffic events by default */
  PetscCall(PetscLogEventSetActiveAll(MAT_SetValues, PETSC_FALSE));
  PetscCall(PetscLogEventSetActiveAll(MAT_GetValues, PETSC_FALSE));
  PetscCall(PetscLogEventSetActiveAll(MAT_GetRow, PETSC_FALSE));
  /* Process Info */
  {
    PetscClassId classids[7];

    classids[0] = MAT_CLASSID;
    classids[1] = MAT_FDCOLORING_CLASSID;
    classids[2] = MAT_COLORING_CLASSID;
    classids[3] = MAT_TRANSPOSECOLORING_CLASSID;
    classids[4] = MAT_PARTITIONING_CLASSID;
    classids[5] = MAT_COARSEN_CLASSID;
    classids[6] = MAT_NULLSPACE_CLASSID;
    PetscCall(PetscInfoProcessClass("mat", 7, classids));
  }

  /* Process summary exclusions */
  PetscCall(PetscOptionsGetString(NULL, NULL, "-log_exclude", logList, sizeof(logList), &opt));
  if (opt) {
    PetscCall(PetscStrInList("mat", logList, ',', &pkg));
    if (pkg) PetscCall(PetscLogEventExcludeClass(MAT_CLASSID));
    if (pkg) PetscCall(PetscLogEventExcludeClass(MAT_FDCOLORING_CLASSID));
    if (pkg) PetscCall(PetscLogEventExcludeClass(MAT_COLORING_CLASSID));
    if (pkg) PetscCall(PetscLogEventExcludeClass(MAT_TRANSPOSECOLORING_CLASSID));
    if (pkg) PetscCall(PetscLogEventExcludeClass(MAT_PARTITIONING_CLASSID));
    if (pkg) PetscCall(PetscLogEventExcludeClass(MAT_COARSEN_CLASSID));
    if (pkg) PetscCall(PetscLogEventExcludeClass(MAT_NULLSPACE_CLASSID));
  }

  /* Register the PETSc built in factorization based solvers */
  PetscCall(MatSolverTypeRegister(MATSOLVERPETSC, MATSEQAIJ, MAT_FACTOR_LU, MatGetFactor_seqaij_petsc));
  PetscCall(MatSolverTypeRegister(MATSOLVERPETSC, MATSEQAIJ, MAT_FACTOR_CHOLESKY, MatGetFactor_seqaij_petsc));
  PetscCall(MatSolverTypeRegister(MATSOLVERPETSC, MATSEQAIJ, MAT_FACTOR_ILU, MatGetFactor_seqaij_petsc));
  PetscCall(MatSolverTypeRegister(MATSOLVERPETSC, MATSEQAIJ, MAT_FACTOR_ICC, MatGetFactor_seqaij_petsc));

  PetscCall(MatSolverTypeRegister(MATSOLVERPETSC, MATSEQAIJPERM, MAT_FACTOR_LU, MatGetFactor_seqaij_petsc));
  PetscCall(MatSolverTypeRegister(MATSOLVERPETSC, MATSEQAIJPERM, MAT_FACTOR_CHOLESKY, MatGetFactor_seqaij_petsc));
  PetscCall(MatSolverTypeRegister(MATSOLVERPETSC, MATSEQAIJPERM, MAT_FACTOR_ILU, MatGetFactor_seqaij_petsc));
  PetscCall(MatSolverTypeRegister(MATSOLVERPETSC, MATSEQAIJPERM, MAT_FACTOR_ICC, MatGetFactor_seqaij_petsc));

  PetscCall(MatSolverTypeRegister(MATSOLVERPETSC, MATCONSTANTDIAGONAL, MAT_FACTOR_LU, MatGetFactor_constantdiagonal_petsc));
  PetscCall(MatSolverTypeRegister(MATSOLVERPETSC, MATCONSTANTDIAGONAL, MAT_FACTOR_CHOLESKY, MatGetFactor_constantdiagonal_petsc));
  PetscCall(MatSolverTypeRegister(MATSOLVERPETSC, MATCONSTANTDIAGONAL, MAT_FACTOR_ILU, MatGetFactor_constantdiagonal_petsc));
  PetscCall(MatSolverTypeRegister(MATSOLVERPETSC, MATCONSTANTDIAGONAL, MAT_FACTOR_ICC, MatGetFactor_constantdiagonal_petsc));

#if defined(PETSC_HAVE_MKL_SPARSE)
  PetscCall(MatSolverTypeRegister(MATSOLVERPETSC, MATSEQAIJMKL, MAT_FACTOR_LU, MatGetFactor_seqaij_petsc));
  PetscCall(MatSolverTypeRegister(MATSOLVERPETSC, MATSEQAIJMKL, MAT_FACTOR_CHOLESKY, MatGetFactor_seqaij_petsc));
  PetscCall(MatSolverTypeRegister(MATSOLVERPETSC, MATSEQAIJMKL, MAT_FACTOR_ILU, MatGetFactor_seqaij_petsc));
  PetscCall(MatSolverTypeRegister(MATSOLVERPETSC, MATSEQAIJMKL, MAT_FACTOR_ICC, MatGetFactor_seqaij_petsc));

  PetscCall(MatSolverTypeRegister(MATSOLVERPETSC, MATSEQBAIJMKL, MAT_FACTOR_LU, MatGetFactor_seqbaij_petsc));
  PetscCall(MatSolverTypeRegister(MATSOLVERPETSC, MATSEQBAIJMKL, MAT_FACTOR_CHOLESKY, MatGetFactor_seqbaij_petsc));
  PetscCall(MatSolverTypeRegister(MATSOLVERPETSC, MATSEQBAIJMKL, MAT_FACTOR_ILU, MatGetFactor_seqbaij_petsc));
  PetscCall(MatSolverTypeRegister(MATSOLVERPETSC, MATSEQBAIJMKL, MAT_FACTOR_ICC, MatGetFactor_seqbaij_petsc));
#endif
  /* Above, we register the PETSc built-in factorization solvers for MATSEQAIJMKL.  In the future, we may want to use
     * some of the MKL-provided ones instead. */

  PetscCall(MatSolverTypeRegister(MATSOLVERPETSC, MATSEQAIJCRL, MAT_FACTOR_LU, MatGetFactor_seqaij_petsc));
  PetscCall(MatSolverTypeRegister(MATSOLVERPETSC, MATSEQAIJCRL, MAT_FACTOR_CHOLESKY, MatGetFactor_seqaij_petsc));
  PetscCall(MatSolverTypeRegister(MATSOLVERPETSC, MATSEQAIJCRL, MAT_FACTOR_ILU, MatGetFactor_seqaij_petsc));
  PetscCall(MatSolverTypeRegister(MATSOLVERPETSC, MATSEQAIJCRL, MAT_FACTOR_ICC, MatGetFactor_seqaij_petsc));

  PetscCall(MatSolverTypeRegister(MATSOLVERPETSC, MATSEQBAIJ, MAT_FACTOR_LU, MatGetFactor_seqbaij_petsc));
  PetscCall(MatSolverTypeRegister(MATSOLVERPETSC, MATSEQBAIJ, MAT_FACTOR_CHOLESKY, MatGetFactor_seqbaij_petsc));
  PetscCall(MatSolverTypeRegister(MATSOLVERPETSC, MATSEQBAIJ, MAT_FACTOR_ILU, MatGetFactor_seqbaij_petsc));
  PetscCall(MatSolverTypeRegister(MATSOLVERPETSC, MATSEQBAIJ, MAT_FACTOR_ICC, MatGetFactor_seqbaij_petsc));

  PetscCall(MatSolverTypeRegister(MATSOLVERPETSC, MATSEQSBAIJ, MAT_FACTOR_CHOLESKY, MatGetFactor_seqsbaij_petsc));
  PetscCall(MatSolverTypeRegister(MATSOLVERPETSC, MATSEQSBAIJ, MAT_FACTOR_ICC, MatGetFactor_seqsbaij_petsc));

  PetscCall(MatSolverTypeRegister(MATSOLVERPETSC, MATSEQDENSE, MAT_FACTOR_LU, MatGetFactor_seqdense_petsc));
  PetscCall(MatSolverTypeRegister(MATSOLVERPETSC, MATSEQDENSE, MAT_FACTOR_ILU, MatGetFactor_seqdense_petsc));
  PetscCall(MatSolverTypeRegister(MATSOLVERPETSC, MATSEQDENSE, MAT_FACTOR_CHOLESKY, MatGetFactor_seqdense_petsc));
  PetscCall(MatSolverTypeRegister(MATSOLVERPETSC, MATSEQDENSE, MAT_FACTOR_QR, MatGetFactor_seqdense_petsc));
#if defined(PETSC_HAVE_CUDA)
  PetscCall(MatSolverTypeRegister(MATSOLVERCUDA, MATSEQDENSE, MAT_FACTOR_LU, MatGetFactor_seqdense_cuda));
  PetscCall(MatSolverTypeRegister(MATSOLVERCUDA, MATSEQDENSE, MAT_FACTOR_CHOLESKY, MatGetFactor_seqdense_cuda));
  PetscCall(MatSolverTypeRegister(MATSOLVERCUDA, MATSEQDENSE, MAT_FACTOR_QR, MatGetFactor_seqdense_cuda));
  PetscCall(MatSolverTypeRegister(MATSOLVERCUDA, MATSEQDENSECUDA, MAT_FACTOR_LU, MatGetFactor_seqdense_cuda));
  PetscCall(MatSolverTypeRegister(MATSOLVERCUDA, MATSEQDENSECUDA, MAT_FACTOR_CHOLESKY, MatGetFactor_seqdense_cuda));
  PetscCall(MatSolverTypeRegister(MATSOLVERCUDA, MATSEQDENSECUDA, MAT_FACTOR_QR, MatGetFactor_seqdense_cuda));
#endif
#if defined(PETSC_HAVE_HIP)
  PetscCall(MatSolverTypeRegister(MATSOLVERHIP, MATSEQDENSE, MAT_FACTOR_LU, MatGetFactor_seqdense_hip));
  PetscCall(MatSolverTypeRegister(MATSOLVERHIP, MATSEQDENSE, MAT_FACTOR_CHOLESKY, MatGetFactor_seqdense_hip));
  PetscCall(MatSolverTypeRegister(MATSOLVERHIP, MATSEQDENSE, MAT_FACTOR_QR, MatGetFactor_seqdense_hip));
  PetscCall(MatSolverTypeRegister(MATSOLVERHIP, MATSEQDENSEHIP, MAT_FACTOR_LU, MatGetFactor_seqdense_hip));
  PetscCall(MatSolverTypeRegister(MATSOLVERHIP, MATSEQDENSEHIP, MAT_FACTOR_CHOLESKY, MatGetFactor_seqdense_hip));
  PetscCall(MatSolverTypeRegister(MATSOLVERHIP, MATSEQDENSEHIP, MAT_FACTOR_QR, MatGetFactor_seqdense_hip));
#endif

  PetscCall(MatSolverTypeRegister(MATSOLVERBAS, MATSEQAIJ, MAT_FACTOR_ICC, MatGetFactor_seqaij_bas));

  /*
     Register the external package factorization based solvers
        Eventually we don't want to have these hardwired here at compile time of PETSc
  */
#if defined(PETSC_HAVE_MUMPS)
  PetscCall(MatSolverTypeRegister_MUMPS());
#endif
#if defined(PETSC_HAVE_CUDA)
  PetscCall(MatSolverTypeRegister_CUSPARSE());
#endif
#if defined(PETSC_HAVE_HIP)
  PetscCall(MatSolverTypeRegister_HIPSPARSE());
#endif
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
  PetscCall(MatSolverTypeRegister_KOKKOS());
#endif
#if defined(PETSC_HAVE_VIENNACL)
  PetscCall(MatSolverTypeRegister_ViennaCL());
#endif
#if defined(PETSC_HAVE_ELEMENTAL)
  PetscCall(MatSolverTypeRegister_Elemental());
#endif
#if defined(PETSC_HAVE_SCALAPACK)
  PetscCall(MatSolverTypeRegister_ScaLAPACK());
#endif
#if defined(PETSC_HAVE_MATLAB)
  PetscCall(MatSolverTypeRegister_Matlab());
#endif
#if defined(PETSC_HAVE_ESSL)
  PetscCall(MatSolverTypeRegister_Essl());
#endif
#if defined(PETSC_HAVE_SUPERLU)
  PetscCall(MatSolverTypeRegister_SuperLU());
#endif
#if defined(PETSC_HAVE_STRUMPACK)
  PetscCall(MatSolverTypeRegister_STRUMPACK());
#endif
#if defined(PETSC_HAVE_PASTIX)
  PetscCall(MatSolverTypeRegister_Pastix());
#endif
#if defined(PETSC_HAVE_SUPERLU_DIST)
  PetscCall(MatSolverTypeRegister_SuperLU_DIST());
#endif
#if defined(PETSC_HAVE_ELEMENTAL)
  PetscCall(MatSolverTypeRegister_SparseElemental());
#endif
#if defined(PETSC_HAVE_MKL_PARDISO)
  PetscCall(MatSolverTypeRegister_MKL_Pardiso());
#endif
#if defined(PETSC_HAVE_MKL_CPARDISO)
  PetscCall(MatSolverTypeRegister_MKL_CPardiso());
#endif
#if defined(PETSC_HAVE_SUITESPARSE)
  PetscCall(MatSolverTypeRegister_SuiteSparse());
#endif
#if defined(PETSC_HAVE_LUSOL)
  PetscCall(MatSolverTypeRegister_Lusol());
#endif
  /* Register package finalizer */
  PetscCall(PetscRegisterFinalize(MatFinalizePackage));
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_DYNAMIC_LIBRARIES)
/*
  PetscDLLibraryRegister - This function is called when the dynamic library it is in is opened.

  This one registers all the matrix methods that are in the basic PETSc Matrix library.

 */
PETSC_EXTERN PetscErrorCode PetscDLLibraryRegister_petscmat(void)
{
  PetscFunctionBegin;
  PetscCall(MatInitializePackage());
  PetscFunctionReturn(0);
}

#endif /* PETSC_HAVE_DYNAMIC_LIBRARIES */
