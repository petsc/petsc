
#include <petsc/private/matimpl.h>

const char       *MatOptions_Shifted[] = {"UNUSED_NONZERO_LOCATION_ERR",
                                  "ROW_ORIENTED",
                                  "NOT_A_VALID_OPTION",
                                  "SYMMETRIC",
                                  "STRUCTURALLY_SYMMETRIC",
                                  "FORCE_DIAGONAL_ENTRIES",
                                  "IGNORE_OFF_PROC_ENTRIES",
                                  "USE_HASH_TABLE",
                                  "KEEP_NONZERO_PATTERN",
                                  "IGNORE_ZERO_ENTRIES",
                                  "USE_INODES",
                                  "HERMITIAN",
                                  "SYMMETRY_ETERNAL",
                                  "NEW_NONZERO_LOCATION_ERR",
                                  "IGNORE_LOWER_TRIANGULAR",
                                  "ERROR_LOWER_TRIANGULAR",
                                  "GETROW_UPPERTRIANGULAR",
                                  "SPD",
                                  "NO_OFF_PROC_ZERO_ROWS",
                                  "NO_OFF_PROC_ENTRIES",
                                  "NEW_NONZERO_LOCATIONS",
                                  "NEW_NONZERO_ALLOCATION_ERR",
                                  "SUBSET_OFF_PROC_ENTRIES",
                                  "SUBMAT_SINGLEIS",
                                  "STRUCTURE_ONLY",
                                  "SORTED_FULL",
                                  "FORM_EXPLICIT_TRANSPOSE",
                                  "MatOption","MAT_",NULL};
const char *const* MatOptions = MatOptions_Shifted+2;
const char *const MatFactorShiftTypes[] = {"NONE","NONZERO","POSITIVE_DEFINITE","INBLOCKS","MatFactorShiftType","PC_FACTOR_",NULL};
const char *const MatStructures[] = {"DIFFERENT","SUBSET","SAME","UNKNOWN","MatStructure","MAT_STRUCTURE_",NULL};
const char *const MatFactorShiftTypesDetail[] = {NULL,"diagonal shift to prevent zero pivot","Manteuffel shift","diagonal shift on blocks to prevent zero pivot"};
const char *const MPPTScotchStrategyTypes[] = {"DEFAULT","QUALITY","SPEED","BALANCE","SAFETY","SCALABILITY","MPPTScotchStrategyType","MP_PTSCOTCH_",NULL};
const char *const MPChacoGlobalTypes[] = {"","MULTILEVEL","SPECTRAL","","LINEAR","RANDOM","SCATTERED","MPChacoGlobalType","MP_CHACO_",NULL};
const char *const MPChacoLocalTypes[] = {"","KERNIGHAN","NONE","MPChacoLocalType","MP_CHACO_",NULL};
const char *const MPChacoEigenTypes[] = {"LANCZOS","RQI","MPChacoEigenType","MP_CHACO_",NULL};

extern PetscErrorCode  MatMFFDInitializePackage(void);
extern PetscErrorCode  MatSolverTypeDestroy(void);
static PetscBool MatPackageInitialized = PETSC_FALSE;
/*@C
  MatFinalizePackage - This function destroys everything in the Petsc interface to the Mat package. It is
  called from PetscFinalize().

  Level: developer

.seealso: PetscFinalize(), MatInitializePackage()
@*/
PetscErrorCode  MatFinalizePackage(void)
{
  MatRootName    nnames,names = MatRootNameList;

  PetscFunctionBegin;
  CHKERRQ(MatSolverTypeDestroy());
  while (names) {
    nnames = names->next;
    CHKERRQ(PetscFree(names->rname));
    CHKERRQ(PetscFree(names->sname));
    CHKERRQ(PetscFree(names->mname));
    CHKERRQ(PetscFree(names));
    names  = nnames;
  }
  CHKERRQ(PetscFunctionListDestroy(&MatList));
  CHKERRQ(PetscFunctionListDestroy(&MatOrderingList));
  CHKERRQ(PetscFunctionListDestroy(&MatColoringList));
  CHKERRQ(PetscFunctionListDestroy(&MatPartitioningList));
  CHKERRQ(PetscFunctionListDestroy(&MatCoarsenList));
  MatRootNameList                  = NULL;
  MatPackageInitialized            = PETSC_FALSE;
  MatRegisterAllCalled             = PETSC_FALSE;
  MatOrderingRegisterAllCalled     = PETSC_FALSE;
  MatColoringRegisterAllCalled     = PETSC_FALSE;
  MatPartitioningRegisterAllCalled = PETSC_FALSE;
  MatCoarsenRegisterAllCalled      = PETSC_FALSE;
  /* this is not ideal because it exposes SeqAIJ implementation details directly into the base Mat code */
  CHKERRQ(PetscFunctionListDestroy(&MatSeqAIJList));
  MatSeqAIJRegisterAllCalled       = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_MUMPS)
PETSC_EXTERN PetscErrorCode MatSolverTypeRegister_MUMPS(void);
#endif
#if defined(PETSC_HAVE_CUDA)
PETSC_EXTERN PetscErrorCode MatSolverTypeRegister_CUSPARSE(void);
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
#if defined(PETSC_HAVE_MATLAB_ENGINE)
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

PETSC_INTERN PetscErrorCode MatGetFactor_seqaij_petsc(Mat,MatFactorType,Mat*);
PETSC_INTERN PetscErrorCode MatGetFactor_seqbaij_petsc(Mat,MatFactorType,Mat*);
PETSC_INTERN PetscErrorCode MatGetFactor_seqsbaij_petsc(Mat,MatFactorType,Mat*);
PETSC_INTERN PetscErrorCode MatGetFactor_seqdense_petsc(Mat,MatFactorType,Mat*);
#if defined(PETSC_HAVE_CUDA)
PETSC_INTERN PetscErrorCode MatGetFactor_seqdense_cuda(Mat,MatFactorType,Mat*);
#endif
PETSC_INTERN PetscErrorCode MatGetFactor_constantdiagonal_petsc(Mat,MatFactorType,Mat*);
PETSC_INTERN PetscErrorCode MatGetFactor_seqaij_bas(Mat,MatFactorType,Mat*);

/*@C
  MatInitializePackage - This function initializes everything in the Mat package. It is called
  from PetscDLLibraryRegister_petscmat() when using dynamic libraries, and on the first call to MatCreate()
  when using shared or static libraries.

  Level: developer

.seealso: PetscInitialize(), MatFinalizePackage()
@*/
PetscErrorCode  MatInitializePackage(void)
{
  char           logList[256];
  PetscBool      opt,pkg;

  PetscFunctionBegin;
  if (MatPackageInitialized) PetscFunctionReturn(0);
  MatPackageInitialized = PETSC_TRUE;
  /* Initialize subpackage */
  CHKERRQ(MatMFFDInitializePackage());
  /* Register Classes */
  CHKERRQ(PetscClassIdRegister("Matrix",&MAT_CLASSID));
  CHKERRQ(PetscClassIdRegister("Matrix FD Coloring",&MAT_FDCOLORING_CLASSID));
  CHKERRQ(PetscClassIdRegister("Matrix Coloring",&MAT_COLORING_CLASSID));
  CHKERRQ(PetscClassIdRegister("Matrix MatTranspose Coloring",&MAT_TRANSPOSECOLORING_CLASSID));
  CHKERRQ(PetscClassIdRegister("Matrix Partitioning",&MAT_PARTITIONING_CLASSID));
  CHKERRQ(PetscClassIdRegister("Matrix Coarsen",&MAT_COARSEN_CLASSID));
  CHKERRQ(PetscClassIdRegister("Matrix Null Space",&MAT_NULLSPACE_CLASSID));
  /* Register Constructors */
  CHKERRQ(MatRegisterAll());
  CHKERRQ(MatOrderingRegisterAll());
  CHKERRQ(MatColoringRegisterAll());
  CHKERRQ(MatPartitioningRegisterAll());
  CHKERRQ(MatCoarsenRegisterAll());
  CHKERRQ(MatSeqAIJRegisterAll());
  /* Register Events */
  CHKERRQ(PetscLogEventRegister("MatMult",          MAT_CLASSID,&MAT_Mult));
  CHKERRQ(PetscLogEventRegister("MatMults",         MAT_CLASSID,&MAT_Mults));
  CHKERRQ(PetscLogEventRegister("MatMultAdd",       MAT_CLASSID,&MAT_MultAdd));
  CHKERRQ(PetscLogEventRegister("MatMultTranspose", MAT_CLASSID,&MAT_MultTranspose));
  CHKERRQ(PetscLogEventRegister("MatMultTrAdd",     MAT_CLASSID,&MAT_MultTransposeAdd));
  CHKERRQ(PetscLogEventRegister("MatSolve",         MAT_CLASSID,&MAT_Solve));
  CHKERRQ(PetscLogEventRegister("MatSolves",        MAT_CLASSID,&MAT_Solves));
  CHKERRQ(PetscLogEventRegister("MatSolveAdd",      MAT_CLASSID,&MAT_SolveAdd));
  CHKERRQ(PetscLogEventRegister("MatSolveTranspos", MAT_CLASSID,&MAT_SolveTranspose));
  CHKERRQ(PetscLogEventRegister("MatSolveTrAdd",    MAT_CLASSID,&MAT_SolveTransposeAdd));
  CHKERRQ(PetscLogEventRegister("MatSOR",           MAT_CLASSID,&MAT_SOR));
  CHKERRQ(PetscLogEventRegister("MatForwardSolve",  MAT_CLASSID,&MAT_ForwardSolve));
  CHKERRQ(PetscLogEventRegister("MatBackwardSolve", MAT_CLASSID,&MAT_BackwardSolve));
  CHKERRQ(PetscLogEventRegister("MatLUFactor",      MAT_CLASSID,&MAT_LUFactor));
  CHKERRQ(PetscLogEventRegister("MatLUFactorSym",   MAT_CLASSID,&MAT_LUFactorSymbolic));
  CHKERRQ(PetscLogEventRegister("MatLUFactorNum",   MAT_CLASSID,&MAT_LUFactorNumeric));
  CHKERRQ(PetscLogEventRegister("MatQRFactor",      MAT_CLASSID,&MAT_QRFactor));
  CHKERRQ(PetscLogEventRegister("MatQRFactorSym",   MAT_CLASSID,&MAT_QRFactorSymbolic));
  CHKERRQ(PetscLogEventRegister("MatQRFactorNum",   MAT_CLASSID,&MAT_QRFactorNumeric));
  CHKERRQ(PetscLogEventRegister("MatCholeskyFctr",  MAT_CLASSID,&MAT_CholeskyFactor));
  CHKERRQ(PetscLogEventRegister("MatCholFctrSym",   MAT_CLASSID,&MAT_CholeskyFactorSymbolic));
  CHKERRQ(PetscLogEventRegister("MatCholFctrNum",   MAT_CLASSID,&MAT_CholeskyFactorNumeric));
  CHKERRQ(PetscLogEventRegister("MatFctrFactSchur", MAT_CLASSID,&MAT_FactorFactS));
  CHKERRQ(PetscLogEventRegister("MatFctrInvSchur",  MAT_CLASSID,&MAT_FactorInvS));
  CHKERRQ(PetscLogEventRegister("MatILUFactor",     MAT_CLASSID,&MAT_ILUFactor));
  CHKERRQ(PetscLogEventRegister("MatILUFactorSym",  MAT_CLASSID,&MAT_ILUFactorSymbolic));
  CHKERRQ(PetscLogEventRegister("MatICCFactorSym",  MAT_CLASSID,&MAT_ICCFactorSymbolic));
  CHKERRQ(PetscLogEventRegister("MatCopy",          MAT_CLASSID,&MAT_Copy));
  CHKERRQ(PetscLogEventRegister("MatConvert",       MAT_CLASSID,&MAT_Convert));
  CHKERRQ(PetscLogEventRegister("MatScale",         MAT_CLASSID,&MAT_Scale));
  CHKERRQ(PetscLogEventRegister("MatResidual",      MAT_CLASSID,&MAT_Residual));
  CHKERRQ(PetscLogEventRegister("MatAssemblyBegin", MAT_CLASSID,&MAT_AssemblyBegin));
  CHKERRQ(PetscLogEventRegister("MatAssemblyEnd",   MAT_CLASSID,&MAT_AssemblyEnd));
  CHKERRQ(PetscLogEventRegister("MatSetValues",     MAT_CLASSID,&MAT_SetValues));
  CHKERRQ(PetscLogEventRegister("MatGetValues",     MAT_CLASSID,&MAT_GetValues));
  CHKERRQ(PetscLogEventRegister("MatGetRow",        MAT_CLASSID,&MAT_GetRow));
  CHKERRQ(PetscLogEventRegister("MatGetRowIJ",      MAT_CLASSID,&MAT_GetRowIJ));
  CHKERRQ(PetscLogEventRegister("MatCreateSubMats", MAT_CLASSID,&MAT_CreateSubMats));
  CHKERRQ(PetscLogEventRegister("MatCreateSubMat",  MAT_CLASSID,&MAT_CreateSubMat));
  CHKERRQ(PetscLogEventRegister("MatGetOrdering",   MAT_CLASSID,&MAT_GetOrdering));
  CHKERRQ(PetscLogEventRegister("MatIncreaseOvrlp", MAT_CLASSID,&MAT_IncreaseOverlap));
  CHKERRQ(PetscLogEventRegister("MatPartitioning",  MAT_PARTITIONING_CLASSID,&MAT_Partitioning));
  CHKERRQ(PetscLogEventRegister("MatPartitioningND",MAT_PARTITIONING_CLASSID,&MAT_PartitioningND));
  CHKERRQ(PetscLogEventRegister("MatCoarsen",       MAT_COARSEN_CLASSID,&MAT_Coarsen));
  CHKERRQ(PetscLogEventRegister("MatZeroEntries",   MAT_CLASSID,&MAT_ZeroEntries));
  CHKERRQ(PetscLogEventRegister("MatLoad",          MAT_CLASSID,&MAT_Load));
  CHKERRQ(PetscLogEventRegister("MatView",          MAT_CLASSID,&MAT_View));
  CHKERRQ(PetscLogEventRegister("MatAXPY",          MAT_CLASSID,&MAT_AXPY));
  CHKERRQ(PetscLogEventRegister("MatFDColorCreate", MAT_FDCOLORING_CLASSID,&MAT_FDColoringCreate));
  CHKERRQ(PetscLogEventRegister("MatFDColorSetUp",  MAT_FDCOLORING_CLASSID,&MAT_FDColoringSetUp));
  CHKERRQ(PetscLogEventRegister("MatFDColorApply",  MAT_FDCOLORING_CLASSID,&MAT_FDColoringApply));
  CHKERRQ(PetscLogEventRegister("MatFDColorFunc",   MAT_FDCOLORING_CLASSID,&MAT_FDColoringFunction));
  CHKERRQ(PetscLogEventRegister("MatTranspose",     MAT_CLASSID,&MAT_Transpose));
  CHKERRQ(PetscLogEventRegister("MatMatSolve",      MAT_CLASSID,&MAT_MatSolve));
  CHKERRQ(PetscLogEventRegister("MatMatTrSolve",    MAT_CLASSID,&MAT_MatTrSolve));
  CHKERRQ(PetscLogEventRegister("MatMatMultSym",    MAT_CLASSID,&MAT_MatMultSymbolic));
  CHKERRQ(PetscLogEventRegister("MatMatMultNum",    MAT_CLASSID,&MAT_MatMultNumeric));
  CHKERRQ(PetscLogEventRegister("MatMatMatMultSym", MAT_CLASSID,&MAT_MatMatMultSymbolic));
  CHKERRQ(PetscLogEventRegister("MatMatMatMultNum", MAT_CLASSID,&MAT_MatMatMultNumeric));
  CHKERRQ(PetscLogEventRegister("MatPtAPSymbolic",  MAT_CLASSID,&MAT_PtAPSymbolic));
  CHKERRQ(PetscLogEventRegister("MatPtAPNumeric",   MAT_CLASSID,&MAT_PtAPNumeric));
  CHKERRQ(PetscLogEventRegister("MatRARtSym",       MAT_CLASSID,&MAT_RARtSymbolic));
  CHKERRQ(PetscLogEventRegister("MatRARtNum",       MAT_CLASSID,&MAT_RARtNumeric));
  CHKERRQ(PetscLogEventRegister("MatMatTrnMultSym", MAT_CLASSID,&MAT_MatTransposeMultSymbolic));
  CHKERRQ(PetscLogEventRegister("MatMatTrnMultNum", MAT_CLASSID,&MAT_MatTransposeMultNumeric));
  CHKERRQ(PetscLogEventRegister("MatTrnMatMultSym", MAT_CLASSID,&MAT_TransposeMatMultSymbolic));
  CHKERRQ(PetscLogEventRegister("MatTrnMatMultNum", MAT_CLASSID,&MAT_TransposeMatMultNumeric));
  CHKERRQ(PetscLogEventRegister("MatTrnColorCreate",MAT_CLASSID,&MAT_TransposeColoringCreate));
  CHKERRQ(PetscLogEventRegister("MatRedundantMat",  MAT_CLASSID,&MAT_RedundantMat));
  CHKERRQ(PetscLogEventRegister("MatGetSeqNZStrct", MAT_CLASSID,&MAT_GetSequentialNonzeroStructure));
  CHKERRQ(PetscLogEventRegister("MatGetMultiProcB", MAT_CLASSID,&MAT_GetMultiProcBlock));
  CHKERRQ(PetscLogEventRegister("MatSetRandom",     MAT_CLASSID,&MAT_SetRandom));

  /* these may be specific to MPIAIJ matrices */
  CHKERRQ(PetscLogEventRegister("MatMPISumSeqNumeric",MAT_CLASSID,&MAT_Seqstompinum));
  CHKERRQ(PetscLogEventRegister("MatMPISumSeqSymbolic",MAT_CLASSID,&MAT_Seqstompisym));
  CHKERRQ(PetscLogEventRegister("MatMPISumSeq",MAT_CLASSID,&MAT_Seqstompi));
  CHKERRQ(PetscLogEventRegister("MatMPIConcateSeq",MAT_CLASSID,&MAT_Merge));
  CHKERRQ(PetscLogEventRegister("MatGetLocalMat",MAT_CLASSID,&MAT_Getlocalmat));
  CHKERRQ(PetscLogEventRegister("MatGetLocalMatCondensed",MAT_CLASSID,&MAT_Getlocalmatcondensed));
  CHKERRQ(PetscLogEventRegister("MatGetBrowsOfAcols",MAT_CLASSID,&MAT_GetBrowsOfAcols));
  CHKERRQ(PetscLogEventRegister("MatGetBrAoCol",MAT_CLASSID,&MAT_GetBrowsOfAocols));

  CHKERRQ(PetscLogEventRegister("MatApplyPAPt_Symbolic",MAT_CLASSID,&MAT_Applypapt_symbolic));
  CHKERRQ(PetscLogEventRegister("MatApplyPAPt_Numeric",MAT_CLASSID,&MAT_Applypapt_numeric));
  CHKERRQ(PetscLogEventRegister("MatApplyPAPt",MAT_CLASSID,&MAT_Applypapt));

  CHKERRQ(PetscLogEventRegister("MatGetSymTrans",MAT_CLASSID,&MAT_Getsymtranspose));
  CHKERRQ(PetscLogEventRegister("MatGetSymTransR",MAT_CLASSID,&MAT_Getsymtransreduced));
  CHKERRQ(PetscLogEventRegister("MatCUSPARSCopyTo",MAT_CLASSID,&MAT_CUSPARSECopyToGPU));
  CHKERRQ(PetscLogEventRegister("MatCUSPARSCopyFr",MAT_CLASSID,&MAT_CUSPARSECopyFromGPU));
  CHKERRQ(PetscLogEventRegister("MatCUSPARSSolAnl",MAT_CLASSID,&MAT_CUSPARSESolveAnalysis));
  CHKERRQ(PetscLogEventRegister("MatCUSPARSGenT",MAT_CLASSID,&MAT_CUSPARSEGenerateTranspose));
  CHKERRQ(PetscLogEventRegister("MatVCLCopyTo",  MAT_CLASSID,&MAT_ViennaCLCopyToGPU));
  CHKERRQ(PetscLogEventRegister("MatDenseCopyTo",MAT_CLASSID,&MAT_DenseCopyToGPU));
  CHKERRQ(PetscLogEventRegister("MatDenseCopyFrom",MAT_CLASSID,&MAT_DenseCopyFromGPU));
  CHKERRQ(PetscLogEventRegister("MatSetValBatch",MAT_CLASSID,&MAT_SetValuesBatch));

  CHKERRQ(PetscLogEventRegister("MatColoringApply",MAT_COLORING_CLASSID,&MATCOLORING_Apply));
  CHKERRQ(PetscLogEventRegister("MatColoringComm",MAT_COLORING_CLASSID,&MATCOLORING_Comm));
  CHKERRQ(PetscLogEventRegister("MatColoringLocal",MAT_COLORING_CLASSID,&MATCOLORING_Local));
  CHKERRQ(PetscLogEventRegister("MatColoringIS",MAT_COLORING_CLASSID,&MATCOLORING_ISCreate));
  CHKERRQ(PetscLogEventRegister("MatColoringSetUp",MAT_COLORING_CLASSID,&MATCOLORING_SetUp));
  CHKERRQ(PetscLogEventRegister("MatColoringWeights",MAT_COLORING_CLASSID,&MATCOLORING_Weights));

  CHKERRQ(PetscLogEventRegister("MatSetPreallCOO",MAT_CLASSID,&MAT_PreallCOO));
  CHKERRQ(PetscLogEventRegister("MatSetValuesCOO",MAT_CLASSID,&MAT_SetVCOO));

  CHKERRQ(PetscLogEventRegister("MatH2OpusBuild",MAT_CLASSID,&MAT_H2Opus_Build));
  CHKERRQ(PetscLogEventRegister("MatH2OpusComp", MAT_CLASSID,&MAT_H2Opus_Compress));
  CHKERRQ(PetscLogEventRegister("MatH2OpusOrth", MAT_CLASSID,&MAT_H2Opus_Orthog));
  CHKERRQ(PetscLogEventRegister("MatH2OpusLR",   MAT_CLASSID,&MAT_H2Opus_LR));

  /* Mark non-collective events */
  CHKERRQ(PetscLogEventSetCollective(MAT_SetValues,      PETSC_FALSE));
  CHKERRQ(PetscLogEventSetCollective(MAT_SetValuesBatch, PETSC_FALSE));
  CHKERRQ(PetscLogEventSetCollective(MAT_GetRow,         PETSC_FALSE));
  /* Turn off high traffic events by default */
  CHKERRQ(PetscLogEventSetActiveAll(MAT_SetValues, PETSC_FALSE));
  CHKERRQ(PetscLogEventSetActiveAll(MAT_GetValues, PETSC_FALSE));
  CHKERRQ(PetscLogEventSetActiveAll(MAT_GetRow,    PETSC_FALSE));
  /* Process Info */
  {
    PetscClassId  classids[7];

    classids[0] = MAT_CLASSID;
    classids[1] = MAT_FDCOLORING_CLASSID;
    classids[2] = MAT_COLORING_CLASSID;
    classids[3] = MAT_TRANSPOSECOLORING_CLASSID;
    classids[4] = MAT_PARTITIONING_CLASSID;
    classids[5] = MAT_COARSEN_CLASSID;
    classids[6] = MAT_NULLSPACE_CLASSID;
    CHKERRQ(PetscInfoProcessClass("mat", 7, classids));
  }

  /* Process summary exclusions */
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,sizeof(logList),&opt));
  if (opt) {
    CHKERRQ(PetscStrInList("mat",logList,',',&pkg));
    if (pkg) CHKERRQ(PetscLogEventExcludeClass(MAT_CLASSID));
    if (pkg) CHKERRQ(PetscLogEventExcludeClass(MAT_FDCOLORING_CLASSID));
    if (pkg) CHKERRQ(PetscLogEventExcludeClass(MAT_COLORING_CLASSID));
    if (pkg) CHKERRQ(PetscLogEventExcludeClass(MAT_TRANSPOSECOLORING_CLASSID));
    if (pkg) CHKERRQ(PetscLogEventExcludeClass(MAT_PARTITIONING_CLASSID));
    if (pkg) CHKERRQ(PetscLogEventExcludeClass(MAT_COARSEN_CLASSID));
    if (pkg) CHKERRQ(PetscLogEventExcludeClass(MAT_NULLSPACE_CLASSID));
  }

  /* Register the PETSc built in factorization based solvers */
  CHKERRQ(MatSolverTypeRegister(MATSOLVERPETSC, MATSEQAIJ,        MAT_FACTOR_LU,MatGetFactor_seqaij_petsc));
  CHKERRQ(MatSolverTypeRegister(MATSOLVERPETSC, MATSEQAIJ,        MAT_FACTOR_CHOLESKY,MatGetFactor_seqaij_petsc));
  CHKERRQ(MatSolverTypeRegister(MATSOLVERPETSC, MATSEQAIJ,        MAT_FACTOR_ILU,MatGetFactor_seqaij_petsc));
  CHKERRQ(MatSolverTypeRegister(MATSOLVERPETSC, MATSEQAIJ,        MAT_FACTOR_ICC,MatGetFactor_seqaij_petsc));

  CHKERRQ(MatSolverTypeRegister(MATSOLVERPETSC, MATSEQAIJPERM,    MAT_FACTOR_LU,MatGetFactor_seqaij_petsc));
  CHKERRQ(MatSolverTypeRegister(MATSOLVERPETSC, MATSEQAIJPERM,    MAT_FACTOR_CHOLESKY,MatGetFactor_seqaij_petsc));
  CHKERRQ(MatSolverTypeRegister(MATSOLVERPETSC, MATSEQAIJPERM,    MAT_FACTOR_ILU,MatGetFactor_seqaij_petsc));
  CHKERRQ(MatSolverTypeRegister(MATSOLVERPETSC, MATSEQAIJPERM,    MAT_FACTOR_ICC,MatGetFactor_seqaij_petsc));

  CHKERRQ(MatSolverTypeRegister(MATSOLVERPETSC, MATCONSTANTDIAGONAL,MAT_FACTOR_LU,MatGetFactor_constantdiagonal_petsc));
  CHKERRQ(MatSolverTypeRegister(MATSOLVERPETSC, MATCONSTANTDIAGONAL,MAT_FACTOR_CHOLESKY,MatGetFactor_constantdiagonal_petsc));
  CHKERRQ(MatSolverTypeRegister(MATSOLVERPETSC, MATCONSTANTDIAGONAL,MAT_FACTOR_ILU,MatGetFactor_constantdiagonal_petsc));
  CHKERRQ(MatSolverTypeRegister(MATSOLVERPETSC, MATCONSTANTDIAGONAL,MAT_FACTOR_ICC,MatGetFactor_constantdiagonal_petsc));

#if defined(PETSC_HAVE_MKL_SPARSE)
  CHKERRQ(MatSolverTypeRegister(MATSOLVERPETSC, MATSEQAIJMKL,     MAT_FACTOR_LU,MatGetFactor_seqaij_petsc));
  CHKERRQ(MatSolverTypeRegister(MATSOLVERPETSC, MATSEQAIJMKL,     MAT_FACTOR_CHOLESKY,MatGetFactor_seqaij_petsc));
  CHKERRQ(MatSolverTypeRegister(MATSOLVERPETSC, MATSEQAIJMKL,     MAT_FACTOR_ILU,MatGetFactor_seqaij_petsc));
  CHKERRQ(MatSolverTypeRegister(MATSOLVERPETSC, MATSEQAIJMKL,     MAT_FACTOR_ICC,MatGetFactor_seqaij_petsc));

  CHKERRQ(MatSolverTypeRegister(MATSOLVERPETSC, MATSEQBAIJMKL,    MAT_FACTOR_LU,MatGetFactor_seqbaij_petsc));
  CHKERRQ(MatSolverTypeRegister(MATSOLVERPETSC, MATSEQBAIJMKL,    MAT_FACTOR_CHOLESKY,MatGetFactor_seqbaij_petsc));
  CHKERRQ(MatSolverTypeRegister(MATSOLVERPETSC, MATSEQBAIJMKL,    MAT_FACTOR_ILU,MatGetFactor_seqbaij_petsc));
  CHKERRQ(MatSolverTypeRegister(MATSOLVERPETSC, MATSEQBAIJMKL,    MAT_FACTOR_ICC,MatGetFactor_seqbaij_petsc));
#endif
    /* Above, we register the PETSc built-in factorization solvers for MATSEQAIJMKL.  In the future, we may want to use
     * some of the MKL-provided ones instead. */

  CHKERRQ(MatSolverTypeRegister(MATSOLVERPETSC, MATSEQAIJCRL,     MAT_FACTOR_LU,MatGetFactor_seqaij_petsc));
  CHKERRQ(MatSolverTypeRegister(MATSOLVERPETSC, MATSEQAIJCRL,     MAT_FACTOR_CHOLESKY,MatGetFactor_seqaij_petsc));
  CHKERRQ(MatSolverTypeRegister(MATSOLVERPETSC, MATSEQAIJCRL,     MAT_FACTOR_ILU,MatGetFactor_seqaij_petsc));
  CHKERRQ(MatSolverTypeRegister(MATSOLVERPETSC, MATSEQAIJCRL,     MAT_FACTOR_ICC,MatGetFactor_seqaij_petsc));

  CHKERRQ(MatSolverTypeRegister(MATSOLVERPETSC, MATSEQBAIJ,       MAT_FACTOR_LU,MatGetFactor_seqbaij_petsc));
  CHKERRQ(MatSolverTypeRegister(MATSOLVERPETSC, MATSEQBAIJ,       MAT_FACTOR_CHOLESKY,MatGetFactor_seqbaij_petsc));
  CHKERRQ(MatSolverTypeRegister(MATSOLVERPETSC, MATSEQBAIJ,       MAT_FACTOR_ILU,MatGetFactor_seqbaij_petsc));
  CHKERRQ(MatSolverTypeRegister(MATSOLVERPETSC, MATSEQBAIJ,       MAT_FACTOR_ICC,MatGetFactor_seqbaij_petsc));

  CHKERRQ(MatSolverTypeRegister(MATSOLVERPETSC, MATSEQSBAIJ,      MAT_FACTOR_CHOLESKY,MatGetFactor_seqsbaij_petsc));
  CHKERRQ(MatSolverTypeRegister(MATSOLVERPETSC, MATSEQSBAIJ,      MAT_FACTOR_ICC,MatGetFactor_seqsbaij_petsc));

  CHKERRQ(MatSolverTypeRegister(MATSOLVERPETSC, MATSEQDENSE,      MAT_FACTOR_LU,MatGetFactor_seqdense_petsc));
  CHKERRQ(MatSolverTypeRegister(MATSOLVERPETSC, MATSEQDENSE,      MAT_FACTOR_ILU,MatGetFactor_seqdense_petsc));
  CHKERRQ(MatSolverTypeRegister(MATSOLVERPETSC, MATSEQDENSE,      MAT_FACTOR_CHOLESKY,MatGetFactor_seqdense_petsc));
  CHKERRQ(MatSolverTypeRegister(MATSOLVERPETSC, MATSEQDENSE,      MAT_FACTOR_QR,MatGetFactor_seqdense_petsc));
#if defined(PETSC_HAVE_CUDA)
  CHKERRQ(MatSolverTypeRegister(MATSOLVERCUDA, MATSEQDENSE,       MAT_FACTOR_LU,MatGetFactor_seqdense_cuda));
  CHKERRQ(MatSolverTypeRegister(MATSOLVERCUDA, MATSEQDENSE,       MAT_FACTOR_CHOLESKY,MatGetFactor_seqdense_cuda));
  CHKERRQ(MatSolverTypeRegister(MATSOLVERCUDA, MATSEQDENSE,       MAT_FACTOR_QR,MatGetFactor_seqdense_cuda));
  CHKERRQ(MatSolverTypeRegister(MATSOLVERCUDA, MATSEQDENSECUDA,   MAT_FACTOR_LU,MatGetFactor_seqdense_cuda));
  CHKERRQ(MatSolverTypeRegister(MATSOLVERCUDA, MATSEQDENSECUDA,   MAT_FACTOR_CHOLESKY,MatGetFactor_seqdense_cuda));
  CHKERRQ(MatSolverTypeRegister(MATSOLVERCUDA, MATSEQDENSECUDA,   MAT_FACTOR_QR,MatGetFactor_seqdense_cuda));
#endif

  CHKERRQ(MatSolverTypeRegister(MATSOLVERBAS,   MATSEQAIJ,        MAT_FACTOR_ICC,MatGetFactor_seqaij_bas));

  /*
     Register the external package factorization based solvers
        Eventually we don't want to have these hardwired here at compile time of PETSc
  */
#if defined(PETSC_HAVE_MUMPS)
  CHKERRQ(MatSolverTypeRegister_MUMPS());
#endif
#if defined(PETSC_HAVE_CUDA)
  CHKERRQ(MatSolverTypeRegister_CUSPARSE());
#endif
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
  CHKERRQ(MatSolverTypeRegister_KOKKOS());
#endif
#if defined(PETSC_HAVE_VIENNACL)
  CHKERRQ(MatSolverTypeRegister_ViennaCL());
#endif
#if defined(PETSC_HAVE_ELEMENTAL)
  CHKERRQ(MatSolverTypeRegister_Elemental());
#endif
#if defined(PETSC_HAVE_SCALAPACK)
  CHKERRQ(MatSolverTypeRegister_ScaLAPACK());
#endif
#if defined(PETSC_HAVE_MATLAB_ENGINE)
  CHKERRQ(MatSolverTypeRegister_Matlab());
#endif
#if defined(PETSC_HAVE_ESSL)
  CHKERRQ(MatSolverTypeRegister_Essl());
#endif
#if defined(PETSC_HAVE_SUPERLU)
  CHKERRQ(MatSolverTypeRegister_SuperLU());
#endif
#if defined(PETSC_HAVE_STRUMPACK)
  CHKERRQ(MatSolverTypeRegister_STRUMPACK());
#endif
#if defined(PETSC_HAVE_PASTIX)
  CHKERRQ(MatSolverTypeRegister_Pastix());
#endif
#if defined(PETSC_HAVE_SUPERLU_DIST)
  CHKERRQ(MatSolverTypeRegister_SuperLU_DIST());
#endif
#if defined(PETSC_HAVE_ELEMENTAL)
  CHKERRQ(MatSolverTypeRegister_SparseElemental());
#endif
#if defined(PETSC_HAVE_MKL_PARDISO)
  CHKERRQ(MatSolverTypeRegister_MKL_Pardiso());
#endif
#if defined(PETSC_HAVE_MKL_CPARDISO)
  CHKERRQ(MatSolverTypeRegister_MKL_CPardiso());
#endif
#if defined(PETSC_HAVE_SUITESPARSE)
  CHKERRQ(MatSolverTypeRegister_SuiteSparse());
#endif
#if defined(PETSC_HAVE_LUSOL)
  CHKERRQ(MatSolverTypeRegister_Lusol());
#endif
  /* Register package finalizer */
  CHKERRQ(PetscRegisterFinalize(MatFinalizePackage));
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
  CHKERRQ(MatInitializePackage());
  PetscFunctionReturn(0);
}

#endif /* PETSC_HAVE_DYNAMIC_LIBRARIES */
