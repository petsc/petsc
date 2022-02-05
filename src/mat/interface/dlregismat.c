
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSolverTypeDestroy();CHKERRQ(ierr);
  while (names) {
    nnames = names->next;
    ierr   = PetscFree(names->rname);CHKERRQ(ierr);
    ierr   = PetscFree(names->sname);CHKERRQ(ierr);
    ierr   = PetscFree(names->mname);CHKERRQ(ierr);
    ierr   = PetscFree(names);CHKERRQ(ierr);
    names  = nnames;
  }
  ierr = PetscFunctionListDestroy(&MatList);CHKERRQ(ierr);
  ierr = PetscFunctionListDestroy(&MatOrderingList);CHKERRQ(ierr);
  ierr = PetscFunctionListDestroy(&MatColoringList);CHKERRQ(ierr);
  ierr = PetscFunctionListDestroy(&MatPartitioningList);CHKERRQ(ierr);
  ierr = PetscFunctionListDestroy(&MatCoarsenList);CHKERRQ(ierr);
  MatRootNameList                  = NULL;
  MatPackageInitialized            = PETSC_FALSE;
  MatRegisterAllCalled             = PETSC_FALSE;
  MatOrderingRegisterAllCalled     = PETSC_FALSE;
  MatColoringRegisterAllCalled     = PETSC_FALSE;
  MatPartitioningRegisterAllCalled = PETSC_FALSE;
  MatCoarsenRegisterAllCalled      = PETSC_FALSE;
  /* this is not ideal because it exposes SeqAIJ implementation details directly into the base Mat code */
  ierr = PetscFunctionListDestroy(&MatSeqAIJList);CHKERRQ(ierr);
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

  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (MatPackageInitialized) PetscFunctionReturn(0);
  MatPackageInitialized = PETSC_TRUE;
  /* Initialize subpackage */
  ierr = MatMFFDInitializePackage();CHKERRQ(ierr);
  /* Register Classes */
  ierr = PetscClassIdRegister("Matrix",&MAT_CLASSID);CHKERRQ(ierr);
  ierr = PetscClassIdRegister("Matrix FD Coloring",&MAT_FDCOLORING_CLASSID);CHKERRQ(ierr);
  ierr = PetscClassIdRegister("Matrix Coloring",&MAT_COLORING_CLASSID);CHKERRQ(ierr);
  ierr = PetscClassIdRegister("Matrix MatTranspose Coloring",&MAT_TRANSPOSECOLORING_CLASSID);CHKERRQ(ierr);
  ierr = PetscClassIdRegister("Matrix Partitioning",&MAT_PARTITIONING_CLASSID);CHKERRQ(ierr);
  ierr = PetscClassIdRegister("Matrix Coarsen",&MAT_COARSEN_CLASSID);CHKERRQ(ierr);
  ierr = PetscClassIdRegister("Matrix Null Space",&MAT_NULLSPACE_CLASSID);CHKERRQ(ierr);
  /* Register Constructors */
  ierr = MatRegisterAll();CHKERRQ(ierr);
  ierr = MatOrderingRegisterAll();CHKERRQ(ierr);
  ierr = MatColoringRegisterAll();CHKERRQ(ierr);
  ierr = MatPartitioningRegisterAll();CHKERRQ(ierr);
  ierr = MatCoarsenRegisterAll();CHKERRQ(ierr);
  ierr = MatSeqAIJRegisterAll();CHKERRQ(ierr);
  /* Register Events */
  ierr = PetscLogEventRegister("MatMult",          MAT_CLASSID,&MAT_Mult);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatMults",         MAT_CLASSID,&MAT_Mults);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatMultConstr",    MAT_CLASSID,&MAT_MultConstrained);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatMultAdd",       MAT_CLASSID,&MAT_MultAdd);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatMultTranspose", MAT_CLASSID,&MAT_MultTranspose);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatMultTrConstr",  MAT_CLASSID,&MAT_MultTransposeConstrained);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatMultTrAdd",     MAT_CLASSID,&MAT_MultTransposeAdd);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatSolve",         MAT_CLASSID,&MAT_Solve);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatSolves",        MAT_CLASSID,&MAT_Solves);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatSolveAdd",      MAT_CLASSID,&MAT_SolveAdd);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatSolveTranspos", MAT_CLASSID,&MAT_SolveTranspose);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatSolveTrAdd",    MAT_CLASSID,&MAT_SolveTransposeAdd);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatSOR",           MAT_CLASSID,&MAT_SOR);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatForwardSolve",  MAT_CLASSID,&MAT_ForwardSolve);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatBackwardSolve", MAT_CLASSID,&MAT_BackwardSolve);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatLUFactor",      MAT_CLASSID,&MAT_LUFactor);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatLUFactorSym",   MAT_CLASSID,&MAT_LUFactorSymbolic);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatLUFactorNum",   MAT_CLASSID,&MAT_LUFactorNumeric);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatQRFactor",      MAT_CLASSID,&MAT_QRFactor);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatQRFactorSym",   MAT_CLASSID,&MAT_QRFactorSymbolic);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatQRFactorNum",   MAT_CLASSID,&MAT_QRFactorNumeric);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatCholeskyFctr",  MAT_CLASSID,&MAT_CholeskyFactor);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatCholFctrSym",   MAT_CLASSID,&MAT_CholeskyFactorSymbolic);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatCholFctrNum",   MAT_CLASSID,&MAT_CholeskyFactorNumeric);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatFctrFactSchur", MAT_CLASSID,&MAT_FactorFactS);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatFctrInvSchur",  MAT_CLASSID,&MAT_FactorInvS);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatILUFactor",     MAT_CLASSID,&MAT_ILUFactor);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatILUFactorSym",  MAT_CLASSID,&MAT_ILUFactorSymbolic);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatICCFactorSym",  MAT_CLASSID,&MAT_ICCFactorSymbolic);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatCopy",          MAT_CLASSID,&MAT_Copy);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatConvert",       MAT_CLASSID,&MAT_Convert);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatScale",         MAT_CLASSID,&MAT_Scale);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatResidual",      MAT_CLASSID,&MAT_Residual);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatAssemblyBegin", MAT_CLASSID,&MAT_AssemblyBegin);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatAssemblyEnd",   MAT_CLASSID,&MAT_AssemblyEnd);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatSetValues",     MAT_CLASSID,&MAT_SetValues);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatGetValues",     MAT_CLASSID,&MAT_GetValues);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatGetRow",        MAT_CLASSID,&MAT_GetRow);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatGetRowIJ",      MAT_CLASSID,&MAT_GetRowIJ);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatCreateSubMats", MAT_CLASSID,&MAT_CreateSubMats);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatCreateSubMat",  MAT_CLASSID,&MAT_CreateSubMat);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatGetOrdering",   MAT_CLASSID,&MAT_GetOrdering);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatIncreaseOvrlp", MAT_CLASSID,&MAT_IncreaseOverlap);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatPartitioning",  MAT_PARTITIONING_CLASSID,&MAT_Partitioning);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatPartitioningND",MAT_PARTITIONING_CLASSID,&MAT_PartitioningND);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatCoarsen",       MAT_COARSEN_CLASSID,&MAT_Coarsen);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatZeroEntries",   MAT_CLASSID,&MAT_ZeroEntries);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatLoad",          MAT_CLASSID,&MAT_Load);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatView",          MAT_CLASSID,&MAT_View);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatAXPY",          MAT_CLASSID,&MAT_AXPY);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatFDColorCreate", MAT_FDCOLORING_CLASSID,&MAT_FDColoringCreate);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatFDColorSetUp",  MAT_FDCOLORING_CLASSID,&MAT_FDColoringSetUp);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatFDColorApply",  MAT_FDCOLORING_CLASSID,&MAT_FDColoringApply);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatFDColorFunc",   MAT_FDCOLORING_CLASSID,&MAT_FDColoringFunction);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatTranspose",     MAT_CLASSID,&MAT_Transpose);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatMatSolve",      MAT_CLASSID,&MAT_MatSolve);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatMatTrSolve",    MAT_CLASSID,&MAT_MatTrSolve);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatMatMultSym",    MAT_CLASSID,&MAT_MatMultSymbolic);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatMatMultNum",    MAT_CLASSID,&MAT_MatMultNumeric);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatMatMatMultSym", MAT_CLASSID,&MAT_MatMatMultSymbolic);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatMatMatMultNum", MAT_CLASSID,&MAT_MatMatMultNumeric);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatPtAPSymbolic",  MAT_CLASSID,&MAT_PtAPSymbolic);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatPtAPNumeric",   MAT_CLASSID,&MAT_PtAPNumeric);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatRARtSym",       MAT_CLASSID,&MAT_RARtSymbolic);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatRARtNum",       MAT_CLASSID,&MAT_RARtNumeric);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatMatTrnMultSym", MAT_CLASSID,&MAT_MatTransposeMultSymbolic);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatMatTrnMultNum", MAT_CLASSID,&MAT_MatTransposeMultNumeric);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatTrnMatMultSym", MAT_CLASSID,&MAT_TransposeMatMultSymbolic);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatTrnMatMultNum", MAT_CLASSID,&MAT_TransposeMatMultNumeric);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatTrnColorCreate",MAT_CLASSID,&MAT_TransposeColoringCreate);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatRedundantMat",  MAT_CLASSID,&MAT_RedundantMat);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatGetSeqNZStrct", MAT_CLASSID,&MAT_GetSequentialNonzeroStructure);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatGetMultiProcB", MAT_CLASSID,&MAT_GetMultiProcBlock);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatSetRandom",     MAT_CLASSID,&MAT_SetRandom);CHKERRQ(ierr);

  /* these may be specific to MPIAIJ matrices */
  ierr = PetscLogEventRegister("MatMPISumSeqNumeric",MAT_CLASSID,&MAT_Seqstompinum);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatMPISumSeqSymbolic",MAT_CLASSID,&MAT_Seqstompisym);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatMPISumSeq",MAT_CLASSID,&MAT_Seqstompi);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatMPIConcateSeq",MAT_CLASSID,&MAT_Merge);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatGetLocalMat",MAT_CLASSID,&MAT_Getlocalmat);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatGetLocalMatCondensed",MAT_CLASSID,&MAT_Getlocalmatcondensed);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatGetBrowsOfAcols",MAT_CLASSID,&MAT_GetBrowsOfAcols);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatGetBrAoCol",MAT_CLASSID,&MAT_GetBrowsOfAocols);CHKERRQ(ierr);

  ierr = PetscLogEventRegister("MatApplyPAPt_Symbolic",MAT_CLASSID,&MAT_Applypapt_symbolic);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatApplyPAPt_Numeric",MAT_CLASSID,&MAT_Applypapt_numeric);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatApplyPAPt",MAT_CLASSID,&MAT_Applypapt);CHKERRQ(ierr);

  ierr = PetscLogEventRegister("MatGetSymTrans",MAT_CLASSID,&MAT_Getsymtranspose);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatGetSymTransR",MAT_CLASSID,&MAT_Getsymtransreduced);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatCUSPARSCopyTo",MAT_CLASSID,&MAT_CUSPARSECopyToGPU);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatCUSPARSCopyFr",MAT_CLASSID,&MAT_CUSPARSECopyFromGPU);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatCUSPARSSolAnl",MAT_CLASSID,&MAT_CUSPARSESolveAnalysis);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatCUSPARSGenT",MAT_CLASSID,&MAT_CUSPARSEGenerateTranspose);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatVCLCopyTo",  MAT_CLASSID,&MAT_ViennaCLCopyToGPU);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatDenseCopyTo",MAT_CLASSID,&MAT_DenseCopyToGPU);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatDenseCopyFrom",MAT_CLASSID,&MAT_DenseCopyFromGPU);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatSetValBatch",MAT_CLASSID,&MAT_SetValuesBatch);CHKERRQ(ierr);

  ierr = PetscLogEventRegister("MatColoringApply",MAT_COLORING_CLASSID,&MATCOLORING_Apply);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatColoringComm",MAT_COLORING_CLASSID,&MATCOLORING_Comm);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatColoringLocal",MAT_COLORING_CLASSID,&MATCOLORING_Local);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatColoringIS",MAT_COLORING_CLASSID,&MATCOLORING_ISCreate);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatColoringSetUp",MAT_COLORING_CLASSID,&MATCOLORING_SetUp);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatColoringWeights",MAT_COLORING_CLASSID,&MATCOLORING_Weights);CHKERRQ(ierr);

  ierr = PetscLogEventRegister("MatSetPreallCOO",MAT_CLASSID,&MAT_PreallCOO);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatSetValuesCOO",MAT_CLASSID,&MAT_SetVCOO);CHKERRQ(ierr);

  ierr = PetscLogEventRegister("MatH2OpusBuild",MAT_CLASSID,&MAT_H2Opus_Build);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatH2OpusComp", MAT_CLASSID,&MAT_H2Opus_Compress);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MatH2OpusOrth", MAT_CLASSID,&MAT_H2Opus_Orthog);CHKERRQ(ierr);

  /* Mark non-collective events */
  ierr = PetscLogEventSetCollective(MAT_SetValues,      PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscLogEventSetCollective(MAT_SetValuesBatch, PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscLogEventSetCollective(MAT_GetRow,         PETSC_FALSE);CHKERRQ(ierr);
  /* Turn off high traffic events by default */
  ierr = PetscLogEventSetActiveAll(MAT_SetValues, PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscLogEventSetActiveAll(MAT_GetValues, PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscLogEventSetActiveAll(MAT_GetRow,    PETSC_FALSE);CHKERRQ(ierr);
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
    ierr = PetscInfoProcessClass("mat", 7, classids);CHKERRQ(ierr);
  }

  /* Process summary exclusions */
  ierr = PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,sizeof(logList),&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrInList("mat",logList,',',&pkg);CHKERRQ(ierr);
    if (pkg) {ierr = PetscLogEventExcludeClass(MAT_CLASSID);CHKERRQ(ierr);}
    if (pkg) {ierr = PetscLogEventExcludeClass(MAT_FDCOLORING_CLASSID);CHKERRQ(ierr);}
    if (pkg) {ierr = PetscLogEventExcludeClass(MAT_COLORING_CLASSID);CHKERRQ(ierr);}
    if (pkg) {ierr = PetscLogEventExcludeClass(MAT_TRANSPOSECOLORING_CLASSID);CHKERRQ(ierr);}
    if (pkg) {ierr = PetscLogEventExcludeClass(MAT_PARTITIONING_CLASSID);CHKERRQ(ierr);}
    if (pkg) {ierr = PetscLogEventExcludeClass(MAT_COARSEN_CLASSID);CHKERRQ(ierr);}
    if (pkg) {ierr = PetscLogEventExcludeClass(MAT_NULLSPACE_CLASSID);CHKERRQ(ierr);}
  }

  /* Register the PETSc built in factorization based solvers */
  ierr = MatSolverTypeRegister(MATSOLVERPETSC, MATSEQAIJ,        MAT_FACTOR_LU,MatGetFactor_seqaij_petsc);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERPETSC, MATSEQAIJ,        MAT_FACTOR_CHOLESKY,MatGetFactor_seqaij_petsc);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERPETSC, MATSEQAIJ,        MAT_FACTOR_ILU,MatGetFactor_seqaij_petsc);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERPETSC, MATSEQAIJ,        MAT_FACTOR_ICC,MatGetFactor_seqaij_petsc);CHKERRQ(ierr);

  ierr = MatSolverTypeRegister(MATSOLVERPETSC, MATSEQAIJPERM,    MAT_FACTOR_LU,MatGetFactor_seqaij_petsc);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERPETSC, MATSEQAIJPERM,    MAT_FACTOR_CHOLESKY,MatGetFactor_seqaij_petsc);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERPETSC, MATSEQAIJPERM,    MAT_FACTOR_ILU,MatGetFactor_seqaij_petsc);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERPETSC, MATSEQAIJPERM,    MAT_FACTOR_ICC,MatGetFactor_seqaij_petsc);CHKERRQ(ierr);

  ierr = MatSolverTypeRegister(MATSOLVERPETSC, MATCONSTANTDIAGONAL,MAT_FACTOR_LU,MatGetFactor_constantdiagonal_petsc);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERPETSC, MATCONSTANTDIAGONAL,MAT_FACTOR_CHOLESKY,MatGetFactor_constantdiagonal_petsc);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERPETSC, MATCONSTANTDIAGONAL,MAT_FACTOR_ILU,MatGetFactor_constantdiagonal_petsc);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERPETSC, MATCONSTANTDIAGONAL,MAT_FACTOR_ICC,MatGetFactor_constantdiagonal_petsc);CHKERRQ(ierr);

#if defined(PETSC_HAVE_MKL_SPARSE)
  ierr = MatSolverTypeRegister(MATSOLVERPETSC, MATSEQAIJMKL,     MAT_FACTOR_LU,MatGetFactor_seqaij_petsc);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERPETSC, MATSEQAIJMKL,     MAT_FACTOR_CHOLESKY,MatGetFactor_seqaij_petsc);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERPETSC, MATSEQAIJMKL,     MAT_FACTOR_ILU,MatGetFactor_seqaij_petsc);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERPETSC, MATSEQAIJMKL,     MAT_FACTOR_ICC,MatGetFactor_seqaij_petsc);CHKERRQ(ierr);

  ierr = MatSolverTypeRegister(MATSOLVERPETSC, MATSEQBAIJMKL,    MAT_FACTOR_LU,MatGetFactor_seqbaij_petsc);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERPETSC, MATSEQBAIJMKL,    MAT_FACTOR_CHOLESKY,MatGetFactor_seqbaij_petsc);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERPETSC, MATSEQBAIJMKL,    MAT_FACTOR_ILU,MatGetFactor_seqbaij_petsc);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERPETSC, MATSEQBAIJMKL,    MAT_FACTOR_ICC,MatGetFactor_seqbaij_petsc);CHKERRQ(ierr);
#endif
    /* Above, we register the PETSc built-in factorization solvers for MATSEQAIJMKL.  In the future, we may want to use
     * some of the MKL-provided ones instead. */

  ierr = MatSolverTypeRegister(MATSOLVERPETSC, MATSEQAIJCRL,     MAT_FACTOR_LU,MatGetFactor_seqaij_petsc);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERPETSC, MATSEQAIJCRL,     MAT_FACTOR_CHOLESKY,MatGetFactor_seqaij_petsc);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERPETSC, MATSEQAIJCRL,     MAT_FACTOR_ILU,MatGetFactor_seqaij_petsc);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERPETSC, MATSEQAIJCRL,     MAT_FACTOR_ICC,MatGetFactor_seqaij_petsc);CHKERRQ(ierr);

  ierr = MatSolverTypeRegister(MATSOLVERPETSC, MATSEQBAIJ,       MAT_FACTOR_LU,MatGetFactor_seqbaij_petsc);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERPETSC, MATSEQBAIJ,       MAT_FACTOR_CHOLESKY,MatGetFactor_seqbaij_petsc);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERPETSC, MATSEQBAIJ,       MAT_FACTOR_ILU,MatGetFactor_seqbaij_petsc);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERPETSC, MATSEQBAIJ,       MAT_FACTOR_ICC,MatGetFactor_seqbaij_petsc);CHKERRQ(ierr);

  ierr = MatSolverTypeRegister(MATSOLVERPETSC, MATSEQSBAIJ,      MAT_FACTOR_CHOLESKY,MatGetFactor_seqsbaij_petsc);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERPETSC, MATSEQSBAIJ,      MAT_FACTOR_ICC,MatGetFactor_seqsbaij_petsc);CHKERRQ(ierr);

  ierr = MatSolverTypeRegister(MATSOLVERPETSC, MATSEQDENSE,      MAT_FACTOR_LU,MatGetFactor_seqdense_petsc);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERPETSC, MATSEQDENSE,      MAT_FACTOR_ILU,MatGetFactor_seqdense_petsc);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERPETSC, MATSEQDENSE,      MAT_FACTOR_CHOLESKY,MatGetFactor_seqdense_petsc);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERPETSC, MATSEQDENSE,      MAT_FACTOR_QR,MatGetFactor_seqdense_petsc);CHKERRQ(ierr);
#if defined(PETSC_HAVE_CUDA)
  ierr = MatSolverTypeRegister(MATSOLVERCUDA, MATSEQDENSE,       MAT_FACTOR_LU,MatGetFactor_seqdense_cuda);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERCUDA, MATSEQDENSE,       MAT_FACTOR_CHOLESKY,MatGetFactor_seqdense_cuda);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERCUDA, MATSEQDENSE,       MAT_FACTOR_QR,MatGetFactor_seqdense_cuda);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERCUDA, MATSEQDENSECUDA,   MAT_FACTOR_LU,MatGetFactor_seqdense_cuda);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERCUDA, MATSEQDENSECUDA,   MAT_FACTOR_CHOLESKY,MatGetFactor_seqdense_cuda);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERCUDA, MATSEQDENSECUDA,   MAT_FACTOR_QR,MatGetFactor_seqdense_cuda);CHKERRQ(ierr);
#endif

  ierr = MatSolverTypeRegister(MATSOLVERBAS,   MATSEQAIJ,        MAT_FACTOR_ICC,MatGetFactor_seqaij_bas);CHKERRQ(ierr);

  /*
     Register the external package factorization based solvers
        Eventually we don't want to have these hardwired here at compile time of PETSc
  */
#if defined(PETSC_HAVE_MUMPS)
  ierr = MatSolverTypeRegister_MUMPS();CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_CUDA)
  ierr = MatSolverTypeRegister_CUSPARSE();CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_KOKKOS_KERNELS)
  ierr = MatSolverTypeRegister_KOKKOS();CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_VIENNACL)
  ierr = MatSolverTypeRegister_ViennaCL();CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_ELEMENTAL)
  ierr = MatSolverTypeRegister_Elemental();CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_SCALAPACK)
  ierr = MatSolverTypeRegister_ScaLAPACK();CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_MATLAB_ENGINE)
  ierr = MatSolverTypeRegister_Matlab();CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_ESSL)
  ierr = MatSolverTypeRegister_Essl();CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_SUPERLU)
  ierr = MatSolverTypeRegister_SuperLU();CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_STRUMPACK)
  ierr = MatSolverTypeRegister_STRUMPACK();CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_PASTIX)
  ierr = MatSolverTypeRegister_Pastix();CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_SUPERLU_DIST)
  ierr = MatSolverTypeRegister_SuperLU_DIST();CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_ELEMENTAL)
  ierr = MatSolverTypeRegister_SparseElemental();CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_MKL_PARDISO)
  ierr = MatSolverTypeRegister_MKL_Pardiso();CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_MKL_CPARDISO)
  ierr = MatSolverTypeRegister_MKL_CPardiso();CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_SUITESPARSE)
  ierr = MatSolverTypeRegister_SuiteSparse();CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_LUSOL)
  ierr = MatSolverTypeRegister_Lusol();CHKERRQ(ierr);
#endif
  /* Register package finalizer */
  ierr = PetscRegisterFinalize(MatFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_DYNAMIC_LIBRARIES)
/*
  PetscDLLibraryRegister - This function is called when the dynamic library it is in is opened.

  This one registers all the matrix methods that are in the basic PETSc Matrix library.

 */
PETSC_EXTERN PetscErrorCode PetscDLLibraryRegister_petscmat(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatInitializePackage();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#endif /* PETSC_HAVE_DYNAMIC_LIBRARIES */
